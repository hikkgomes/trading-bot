from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.binance_user_stream import normalise_user_event
from src.data.database import PlatformDatabase
from src.domain._codec import to_primitive
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderType, ProtectionKind
from src.execution.broker import (
    BrokerFill,
    BrokerIncome,
    BrokerOrderState,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.broker import OrderSide as BrokerSide
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.reconciler import reconcile_account
from src.execution.recovery import RecoveryAction, RecoveryActionType, SqlRecoveryStore
from src.execution.stops import SqlStopStore, StopManager, StopStatus
from src.observability.decision_trace import SqlDecisionTraceStore
from src.services.financial_application import FinancialApplication, FinancialScope
from src.services.income_recovery import reconcile_income
from src.services.live_execution import ApprovedLiveExecution
from src.services.order_execution import DatabaseUserStreamWorker
from src.services.order_recovery import DatabaseLiveRecoveryWorker
from src.services.platform_testnet_connected import _verify_native_stop
from src.services.protective_stops import LiveProtectiveStopService
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-09-09T00:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


@pytest.mark.parametrize(
    "side,observed_at,accepted",
    [
        (OrderSide.BUY, NOW, False),
        (OrderSide.SELL, NOW, True),
        (OrderSide.SELL, "2026-09-08T23:59:00+00:00", False),
        (OrderSide.SELL, "2026-09-09T00:00:01+00:00", False),
        (OrderSide.SELL, None, False),
    ],
)
def test_final_entry_guard_uses_exchange_time_and_execution_side(
    tmp_path, side, observed_at, accepted
):
    from src.risk.engine import SqlRiskSnapshotStore

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'entry-quote.sqlite3'}")
    database.create_schema()
    SqlRiskSnapshotStore(database.engine).save(
        {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": INSTRUMENT,
            "source_event_time": observed_at,
            "values": {
                "close": 100,
                "spread_bps": 1,
                "bid_depth_quote": 2000,
                "ask_depth_quote": 50,
            },
        },
        created_at=NOW,
    )
    guard = ApprovedLiveExecution.__new__(ApprovedLiveExecution)
    guard.engine = database.engine
    guard.products = {"active_income": {"account_id": "account"}}
    guard.risk_configuration = {
        "global": {"maximum_market_data_staleness_seconds": 5},
        "instrument": {"maximum_visible_depth_fraction": 0.1, "maximum_slippage_bps": 25},
    }
    intent = OrderIntent(
        "entry",
        "portfolio",
        INSTRUMENT,
        side,
        1,
        OrderType.MARKET,
        NOW,
        valid_until="2026-09-09T00:01:00+00:00",
    )
    if accepted:
        guard._assert_market_admission({"product_id": "active_income"}, intent)
    else:
        with pytest.raises(PermissionError):
            guard._assert_market_admission({"product_id": "active_income"}, intent)
    guard._assert_market_admission(
        {"product_id": "active_income"}, replace(intent, reduce_only=True)
    )
    database.dispose()


def test_recovery_replay_is_idempotent_but_later_equal_exposure_gets_a_new_order(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'flatten-identity.sqlite3'}")
    database.create_schema()
    orders = OrderManager(SqlOrderStore(database.engine))
    recovery = ApprovedLiveExecution.__new__(ApprovedLiveExecution)
    recovery.engine = database.engine
    recovery.order_manager = orders
    recovery.positions = PositionManager(SqlPositionStore(database.engine))
    recovery.product_portfolios = {"active_income": "portfolio"}
    recovery.products = {
        "active_income": {"account_id": "account", "base_accounting_asset": "USDT"}
    }
    recovery.accounts = {
        "account": {"account_id": "account", "environment": "testnet", "venue": "binance"}
    }
    recovery.venues = {
        "active_income": SimpleNamespace(
            instruments={INSTRUMENT: SimpleNamespace(exchange_symbol="BTCUSDT")},
            broker=SimpleNamespace(get_price=lambda _: 100),
            submit=orders.create,
        )
    }
    first = recovery.recover_action(
        "active_income",
        RecoveryAction(
            RecoveryActionType.EMERGENCY_FLATTEN, INSTRUMENT, 1, operation_id="incident-1"
        ),
    )
    identity = first["order_id"]
    orders.persist_for_submission(identity)
    orders.submitted(identity)
    orders.apply_fill(Fill("recovery-fill", identity, INSTRUMENT, OrderSide.SELL, 1, 100, 0, NOW))
    replay = recovery.recover_action(
        "active_income",
        RecoveryAction(
            RecoveryActionType.EMERGENCY_FLATTEN, INSTRUMENT, 1, operation_id="incident-1"
        ),
    )
    assert replay["order_id"] == identity and len(orders.all()) == 1
    later = recovery.recover_action(
        "active_income",
        RecoveryAction(
            RecoveryActionType.EMERGENCY_FLATTEN, INSTRUMENT, 1, operation_id="incident-2"
        ),
    )
    assert later["order_id"] != identity and len(orders.all()) == 2
    assert all(order.reduce_only for order in orders.all())
    database.dispose()


class NativeBroker:
    def __init__(self):
        self.trades = {}
        self.trade_queries = []
        self.external_positions = []
        self.parent_cancellations = 0
        self.config = SimpleNamespace(market_type="futures")

    def list_account_futures_positions(self):
        return self.external_positions

    @staticmethod
    def _symbols_match(left, right):
        return left == right

    def supports_native_protective_stops(self):
        return True

    def place_protective_stop(
        self, *, symbol, side, qty, trigger_price, client_id, kind=ProtectionKind.STOP
    ):
        self.native = ProtectiveOrder(
            symbol,
            BrokerSide(side.value),
            qty,
            trigger_price,
            ProtectiveOrderStatus.OPEN,
            "algo-1",
            client_id,
        )
        return self.native

    def get_protective_stop(self, **kwargs):
        return self.native

    def query_order(self, **kwargs):
        if kwargs["exchange_order_id"] == "parent-1":
            return BrokerOrderState("parent-1", "parent-client", "cancelled", 1, 100)
        assert kwargs["exchange_order_id"] == "child-1"
        quantity = sum(trade.quantity for trade in self.trades.values())
        return BrokerOrderState(
            "child-1",
            "child-client",
            "closed" if quantity == 1 else "partially_filled",
            quantity,
            95,
        )

    def query_order_fills(self, **kwargs):
        if kwargs["exchange_order_id"] == "parent-1":
            return (
                BrokerFill(
                    "entry-trade",
                    "parent-1",
                    "parent-client",
                    "BTCUSDT",
                    BrokerSide.BUY,
                    1,
                    100,
                    0.1,
                    1788911999,
                    "USDT",
                ),
            )
        cursor = int(kwargs.get("from_trade_id", "0"))
        self.trade_queries.append(cursor)
        return tuple(trade for trade in self.trades.values() if int(trade.trade_id) >= cursor)

    def cancel_order(self, **kwargs):
        assert kwargs["exchange_order_id"] == "parent-1"
        self.parent_cancellations += 1
        return self.query_order(**kwargs)


@pytest.mark.parametrize("transport", ["user_stream", "rest_recovery"])
def test_native_trigger_child_fills_reconcile_once_through_shared_ledger(tmp_path, transport):
    from test_platform_execution_recovery import _allocation_books

    from src.portfolio.books import book_balances

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'protection.sqlite3'}")
    database.create_schema()
    book = _allocation_books(database, "account", "native-child")[0]
    orders = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="user-stream-fixture",
        node_id="fixture",
        role="execution",
        capabilities=("user_stream_event",),
        observed_at=NOW,
    )
    broker = NativeBroker()
    entry_at = "2026-09-08T23:59:59+00:00"
    service = LiveProtectiveStopService(
        stop_manager=StopManager(SqlStopStore(database.engine)),
        venues={
            "active_income": SimpleNamespace(
                broker=broker,
                instruments={INSTRUMENT: SimpleNamespace(exchange_symbol="BTCUSDT")},
            )
        },
        products={"active_income": {"portfolio_id": "portfolio", "account_id": "account"}},
        accounts={"account": {"market": "usdt_futures"}},
        order_manager=orders,
        positions=positions,
        queue=queue,
    )
    entry = OrderIntent(
        "entry",
        "portfolio",
        INSTRUMENT,
        OrderSide.BUY,
        2,
        OrderType.MARKET,
        entry_at,
        strategy_contributions={book: 1},
        metadata={
            "reference_price": 100,
            "protective_stop_price": 95,
            "target_metadata": {
                "account_id": "account",
                "execution_environment": "testnet",
                "book_contributions": [{"book_id": book, "signed_quantity": 2}],
            },
        },
    )
    orders.create(entry)
    service.prepare_entry("active_income", entry, entry_at)
    orders.persist_for_submission(entry.order_id)
    orders.bind_client_order_id(entry.order_id, client_order_id="parent-client")
    orders.submitted(entry.order_id)
    orders.bind_exchange_acknowledgement(
        entry.order_id, exchange_order_id="parent-1", client_order_id="parent-client"
    )
    FinancialApplication(
        database.engine,
        FinancialScope(
            "active_income",
            "account",
            "testnet",
            "binance",
            "USDT",
        ),
    ).apply_fill(
        Fill(
            "entry-fill",
            "entry",
            INSTRUMENT,
            OrderSide.BUY,
            1,
            100,
            0.1,
            entry_at,
            fee_asset="USDT",
            metadata={"trade_id": "entry-trade"},
        )
    )
    orders.reload()
    entry = orders.get(entry.order_id)
    service.on_fill("active_income", entry, 1, NOW)
    evidence = {
        "native_stop_created": False,
        "native_stop_readback_matched": False,
        "partial_fill_quantity_handled": False,
        "algo_update_processed": False,
    }
    _verify_native_stop(
        protective_service=service,
        broker=broker,
        instrument=SimpleNamespace(exchange_symbol="BTCUSDT"),
        order_id=entry.order_id,
        product_id="active_income",
        position_quantity=1,
        stop_evidence=evidence,
    )
    assert evidence == {
        "native_stop_created": True,
        "native_stop_readback_matched": True,
        "partial_fill_quantity_handled": False,
        "algo_update_processed": False,
    }
    assert broker.native.qty == 1
    assert orders.get(entry.order_id).filled_quantity == 1 < entry.quantity
    broker.native = replace(
        broker.native,
        status=ProtectiveOrderStatus.TRIGGERED,
        child_order_id="child-1",
        filled_qty=0.4,
        average_price=95,
    )
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(
            database.engine,
            product_id="active_income",
        ),
    )
    worker = DatabaseUserStreamWorker(
        engine=database.engine,
        queue=queue,
        worker_id="user-stream-fixture",
        order_manager=orders,
        positions=positions,
        ledgers={"active_income": ledger},
        trace_store=SqlDecisionTraceStore(database.engine),
        account_products={"account": "active_income"},
        account_environments={"account": "testnet"},
        on_algo_update=service.on_algo_update,
    )
    _deliver(
        worker,
        queue,
        "trigger",
        {
            "e": "ALGO_UPDATE",
            "o": {
                "s": "BTCUSDT",
                "aid": "algo-1",
                "caid": broker.native.client_id,
                "X": "TRIGGERED",
                "ai": "child-1",
            },
        },
    )
    children = [order for order in orders.all() if order.reduce_only]
    assert len(children) == 1
    assert children[0].filled_quantity == 0
    positions.reload()
    assert positions.get("portfolio", INSTRUMENT).quantity == 1
    recovery = _recovery_worker(database, orders, positions, ledger, broker, queue, service)
    recovered = recovery.run_once(now=NOW)
    assert recovered["reason_code"] == "live_recovery_verified", recovered
    assert broker.parent_cancellations == 1
    assert orders.get(entry.order_id).status.value == "reconciled"
    assert orders.get(entry.order_id).filled_quantity == 1
    for event_id, trade_id, quantity, cumulative in (
        ("partial", 2, 0.4, 0.4),
        ("partial-replay", 2, 0.4, 0.4),
        ("final", 3, 0.6, 1),
    ):
        if transport == "rest_recovery":
            broker.trades[trade_id] = BrokerFill(
                str(trade_id),
                "child-1",
                "child-client",
                "BTCUSDT",
                BrokerSide.SELL,
                quantity,
                95,
                quantity / 10,
                1788912000,
                "USDT",
            )
            queue.enqueue_if_absent(
                job_id=event_id,
                name="live_order_recovery",
                available_at=NOW,
                payload={
                    "product_id": "active_income",
                    "recovery_kind": "protective_child",
                    "protective_child_order_id": children[0].order_id,
                },
            )
            result = recovery.run_once(now=NOW)
            assert result["reason_code"] == "live_recovery_verified", result
            continue
        _deliver(
            worker,
            queue,
            event_id,
            {
                "e": "ORDER_TRADE_UPDATE",
                "o": {
                    "s": "BTCUSDT",
                    "c": "child-client",
                    "i": "child-1",
                    "x": "TRADE",
                    "X": "FILLED" if cumulative == 1 else "PARTIALLY_FILLED",
                    "S": "SELL",
                    "l": str(quantity),
                    "z": str(cumulative),
                    "L": "95",
                    "n": str(quantity / 10),
                    "N": "USDT",
                    "t": trade_id,
                },
            },
        )
    assert positions.get("portfolio", INSTRUMENT).quantity == 0
    assert positions.get(book, INSTRUMENT).quantity == 0
    assert book_balances(database.engine, book, at=NOW)["USDT"] == pytest.approx(-5.20)
    assert len(orders.fills_for(children[0].order_id)) == 2
    assert orders.get(children[0].order_id).filled_quantity == 1
    if transport == "rest_recovery":
        restarted_orders = OrderManager(SqlOrderStore(database.engine))
        assert restarted_orders.get(children[0].order_id).metadata["rest_checkpoint"] == {
            "next_trade_id": "4",
            "verified_quantity": 1,
        }
        assert broker.trade_queries[:4] == [0, 0, 3, 3]
    assert ledger.balances()["assets:cash"] == Decimal("-5.20")
    assert ledger.balances()["expense:fees"] == Decimal("0.20")
    assert ledger.balances()["income:realised_pnl"] == Decimal("5.00")
    service.stop_manager = StopManager(SqlStopStore(database.engine))
    service.reconcile("active_income", NOW)
    assert service.stop_manager.all()[0].status is StopStatus.RECONCILED
    broker.external_positions = [SimpleNamespace(symbol="BTCUSDT", qty=0.2, avg_price=95)]
    with pytest.raises(RuntimeError, match="fill-history reconciliation"):
        recovery.execute_action(
            "active_income",
            RecoveryAction(
                RecoveryActionType.RECONCILE_POSITION,
                INSTRUMENT,
                0.2,
            ),
        )
    assert positions.get("portfolio", INSTRUMENT).quantity == 0
    assert ledger.balances()["assets:cash"] == Decimal("-5.20")
    broker.external_positions = []
    verified = recovery.execute_action(
        "active_income",
        RecoveryAction(
            RecoveryActionType.RECONCILE_POSITION,
            INSTRUMENT,
            0.2,
        ),
    )
    assert verified["quantity"] == 0
    scope = FinancialScope("active_income", "account", "testnet", "binance", "USDT")
    incomes = (
        BrokerIncome("shared-id", "BTCUSDT", "COMMISSION", -0.04, "USDT", 1788912000, trade_id="2"),
        BrokerIncome("shared-id", "BTCUSDT", "REALIZED_PNL", -2, "USDT", 1788912000, trade_id="2"),
        BrokerIncome("shared-id", "BTCUSDT", "FUNDING_FEE", 0.30, "USDT", 1788911999.5),
    )
    for income in incomes:
        assert reconcile_income(database.engine, ledger, scope, income) is True
        assert reconcile_income(database.engine, ledger, scope, income) is False
    assert ledger.balances()["assets:cash"] == Decimal("-4.90")
    assert book_balances(database.engine, book, at=NOW)["USDT"] == pytest.approx(-4.90)
    assert ledger.balances()["expense:fees"] == Decimal("0.20")
    assert ledger.balances()["income:realised_pnl"] == Decimal("5.00")
    with pytest.raises(ValueError, match="changed its financial values"):
        reconcile_income(database.engine, ledger, scope, replace(incomes[0], amount=-0.4))
    with pytest.raises(ValueError, match="does not reconcile"):
        reconcile_income(
            database.engine, ledger, scope, replace(incomes[0], income_id="other", amount=-0.4)
        )
    database.dispose()


def _recovery_worker(database, orders, positions, ledger, broker, queue, service):
    queue.register_worker(
        worker_id="recovery-fixture",
        node_id="fixture",
        role="execution",
        capabilities=("live_order_recovery",),
        observed_at=NOW,
    )
    approved = ApprovedLiveExecution(
        engine=database.engine,
        order_manager=orders,
        positions=positions,
        ledgers={"active_income": ledger},
        configuration={
            "products": {
                "products": [
                    {
                        "product_id": "active_income",
                        "portfolio_id": "portfolio",
                        "account_id": "account",
                        "execution_mode": "paper",
                    }
                ]
            },
            "accounts": {"accounts": [{"account_id": "account", "environment": "testnet"}]},
            "promotion": {"policies": []},
            "risk": {},
        },
    )
    approved.venues = service.venues
    return DatabaseLiveRecoveryWorker(
        queue=queue,
        worker_id="recovery-fixture",
        store=SqlRecoveryStore(database.engine),
        account_products={"account": "active_income"},
        execute_action=approved.recover_action,
        reconcile_product=lambda _: reconcile_account(
            local_positions=positions.current_quantities("portfolio"),
            exchange_positions={INSTRUMENT: 1 - sum(t.quantity for t in broker.trades.values())},
            local_open_order_ids=set(),
            exchange_open_order_ids=set(),
        ),
    )


def _deliver(worker, queue, identity, payload):
    event = normalise_user_event(
        account_id="account", market="futures", payload=payload, receive_timestamp=NOW
    )
    queue.enqueue_if_absent(
        job_id=identity,
        name="user_stream_event",
        available_at=NOW,
        payload={"account_id": "account", "market": "futures", "event": to_primitive(event)},
    )
    result = worker.run_once(now=NOW)
    assert result["reason_code"] == "user_stream_event_recorded", result


def test_trade_history_paginates_exact_large_ids_and_deduplicates_boundaries(monkeypatch):
    from src.execution import history

    monkeypatch.setattr(history, "PAGE_SIZE", 2)
    start = 900719925474099312345
    calls = []

    def fetch(symbol, since, limit, params):
        calls.append(params["fromId"])
        if len(calls) == 1:
            return [{"id": str(start)}, {"id": str(start + 1)}]
        if len(calls) == 2:
            return [{"id": str(start + 1)}, {"id": str(start + 2)}]
        return []

    rows = list(history.trade_history(fetch, "BTCUSDT", {"orderId": "order"}))
    assert [row["id"] for row in rows] == [str(start + offset) for offset in range(3)]
    assert calls == [0, start + 2, start + 3]


def test_income_pagination_preserves_same_time_and_distinct_income_types(monkeypatch):
    from src.execution import history

    monkeypatch.setattr(history, "PAGE_SIZE", 2)
    rows = [
        {"tranId": "1", "incomeType": kind, "time": 1000}
        for kind in (
            "FUNDING_FEE",
            "COMMISSION",
            "REALIZED_PNL",
        )
    ]
    calls = []

    def fetch(params):
        calls.append(params)
        return rows[:2] if params["page"] == 1 else rows[2:]

    assert list(history.income_history(fetch, {"startTime": 0, "endTime": 2000})) == rows
    assert [call["page"] for call in calls] == [1, 2]
    assert {call["endTime"] for call in calls} == {2000}


def test_repeated_full_history_page_cannot_certify_completion(monkeypatch):
    from src.execution import history

    monkeypatch.setattr(history, "PAGE_SIZE", 1)
    with pytest.raises(RuntimeError, match="did not advance"):
        list(history.trade_history(lambda *args: [{"id": "4"}], "BTCUSDT", {}))
    with pytest.raises(RuntimeError, match="did not advance"):
        list(history.income_history(lambda *args: [{"tranId": "4"}], {}))
