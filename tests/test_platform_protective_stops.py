from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from src.domain.market_events import MarketEvent
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType, ProtectionKind
from src.execution.broker import Position, ProtectiveOrder, ProtectiveOrderStatus
from src.execution.stops import ProtectiveStop, StopManager, StopStatus
from src.services.protective_stops import LiveProtectiveStopService, ProtectiveStopError

NOW = "2026-08-31T10:00:00+00:00"
INSTRUMENT = "binance:futures:BTCUSDT:USDT"


class _StopStore:
    def __init__(self) -> None:
        self.events: list[ProtectiveStop] = []

    def append(self, stop: ProtectiveStop) -> None:
        self.events.append(stop)

    def read(self) -> tuple[ProtectiveStop, ...]:
        return tuple(self.events)


@dataclass
class _Instrument:
    exchange_symbol: str = "BTCUSDT"


class _Broker:
    def __init__(self) -> None:
        self.placed: list[ProtectiveOrder] = []
        self.cancelled: list[str] = []
        self.actions: list[str] = []
        self.position_quantity = 1.0

    def supports_native_protective_stops(self) -> bool:
        return True

    def get_protective_stop(self, *, symbol, order_id, client_id):
        return next(item for item in self.placed if item.client_id == client_id)

    def get_position(self, symbol):
        return Position(symbol=symbol, qty=self.position_quantity)

    def place_protective_stop(
        self, *, symbol, side, qty, trigger_price, client_id, kind=ProtectionKind.STOP
    ):
        self.actions.append("place")
        item = ProtectiveOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            trigger_price=trigger_price,
            status=ProtectiveOrderStatus.OPEN,
            order_id=f"native-{len(self.placed) + 1}",
            client_id=client_id,
            kind=kind,
        )
        self.placed.append(item)
        return item

    def cancel_protective_stop(self, *, symbol, order_id, client_id):
        self.actions.append("cancel")
        self.cancelled.append(order_id)
        return ProtectiveOrder(
            symbol=symbol,
            side=OrderSide.SELL,
            qty=1.0,
            trigger_price=95.0,
            status=ProtectiveOrderStatus.CANCELED,
            order_id=order_id,
            client_id=client_id,
        )


class _Venue:
    def __init__(self, broker: _Broker) -> None:
        self.broker = broker
        self.instruments = {INSTRUMENT: _Instrument()}


def _order(*, quantity: float = 1.0, reduce_only: bool = False) -> OrderIntent:
    return OrderIntent(
        order_id="entry-1" if not reduce_only else "close-1",
        portfolio_id="active-income-portfolio",
        instrument_id=INSTRUMENT,
        side=OrderSide.BUY if not reduce_only else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.MARKET,
        created_at=NOW,
        status=OrderStatus.PERSISTED,
        reduce_only=reduce_only,
        metadata={
            "reference_price": 100.0,
            "protective_stop_price": 95.0,
        },
    )


def _service(*, broker: _Broker | None = None, manager: StopManager | None = None):
    broker = broker or _Broker()
    manager = manager or StopManager(_StopStore())
    service = LiveProtectiveStopService(
        stop_manager=manager,
        venues={"active_income": _Venue(broker)},
        products={
            "active_income": {
                "portfolio_id": "active-income-portfolio",
                "account_id": "futures-account",
            }
        },
        accounts={"futures-account": {"market": "usdt_futures"}},
    )
    return service, broker, manager


def test_entry_stop_is_durable_before_native_side_effect_and_confirmed_after_fill() -> None:
    service, broker, manager = _service()
    order = _order()
    stop = service.prepare_entry("active_income", order, NOW)
    assert stop is not None
    assert stop.status is StopStatus.ACTIVE
    assert broker.placed == []

    protected = service.on_fill("active_income", order, 1.0, NOW)
    assert protected is not None
    assert protected.status is StopStatus.PROTECTED
    assert protected.protected_quantity == pytest.approx(1.0)
    assert len(broker.placed) == 1
    assert manager.for_entry_order(order.order_id)[0].native_order_id == "native-1"


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
@pytest.mark.parametrize("quote_delay", [0.0, 601.0])
def test_live_bracket_uses_entry_quote_and_rechecks_expiry(
    tmp_path, monkeypatch, side, quote_delay
):
    from types import SimpleNamespace

    from src.data.database import PlatformDatabase
    from src.domain.instruments import Instrument, MarketType
    from src.domain.orders import Fill
    from src.execution.broker import BrokerOrderAcknowledgement
    from src.execution.live_exchange import BrokerExecutionVenue
    from src.execution.order_manager import OrderManager, SqlOrderStore
    from src.execution.position_manager import PositionManager, SqlPositionStore
    from src.execution.stops import SqlStopStore
    from src.services.financial_application import FinancialApplication, FinancialScope
    from src.services.live_execution import ApprovedLiveExecution
    from src.services.order_execution import DatabaseLiveExecutionWorker
    from src.services.scheduler import DatabaseJobQueue

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote.db'}")
    database.create_schema()
    orders = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    broker = _Broker()
    broker.get_price = lambda symbol: 120.0
    broker.normalize_order_qty = lambda symbol, quantity, **_kwargs: quantity
    sent = []

    def submit(order):
        persisted = OrderManager(SqlOrderStore(database.engine)).get("entry-1")
        assert persisted.metadata["entry_reference"]["price"] == 120
        sent.append(order)
        return BrokerOrderAcknowledgement("exchange-entry", order.client_id, "acknowledged")

    broker.submit_order = submit
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=3,
        minimum_quantity=0.001,
        minimum_notional=5,
    )
    venue = BrokerExecutionVenue(
        order_manager=orders,
        position_manager=positions,
        broker=broker,
        instruments={INSTRUMENT: instrument},
    )
    stops = StopManager(SqlStopStore(database.engine))
    service = LiveProtectiveStopService(
        stop_manager=stops,
        venues={"active_income": venue},
        products={
            "active_income": {
                "portfolio_id": "active-income-portfolio",
                "account_id": "futures-account",
            }
        },
        accounts={"futures-account": {"market": "usdt_futures"}},
        order_manager=orders,
        positions=positions,
    )
    order = replace(
        _order(quantity=1.2),
        status=OrderStatus.CREATED,
        side=side,
        metadata={
            "reference_price": 100,
            "target_metadata": {
                "position_rules": {
                    "stop_loss": 0.03,
                    "take_profit": 0.06,
                    "horizon_bars": 8,
                    "execution_policy": "resting_bracket/v1",
                }
            },
        },
    )
    orders.create(order)
    orders.persist_for_submission(order.order_id)
    service.prepare_entry("active_income", orders.get(order.order_id), NOW)
    context = SimpleNamespace(products={"active_income": {"account_id": "futures-account"}})
    worker = DatabaseLiveExecutionWorker(
        queue=DatabaseJobQueue(database.engine),
        worker_id="quote-fixture",
        order_manager=orders,
        positions=positions,
        ledgers={},
        trace_store=None,
        venues={"active_income": venue},
        authorise=lambda payload, intent: ApprovedLiveExecution._order_context(
            context, payload, intent
        ),
        prepare_protective_stop=service.prepare_entry,
    )
    clock = iter((0.0, quote_delay))
    monkeypatch.setattr("src.services.order_execution.monotonic", lambda: next(clock))
    if quote_delay:
        with pytest.raises(PermissionError, match="expired"):
            worker._submit_order({"product_id": "active_income", "order_id": order.order_id}, NOW)
        assert not sent and not broker.placed
        database.dispose()
        return
    worker._submit_order({"product_id": "active_income", "order_id": order.order_id}, NOW)
    assert orders.get(order.order_id).quantity == 1
    assert orders.get(order.order_id).metadata["entry_reference"]["notional"] == 120
    FinancialApplication(
        database.engine,
        FinancialScope("active_income", "futures-account", "testnet", "binance", "USDT"),
    ).apply_fill(
        Fill(
            "quote-fill",
            order.order_id,
            INSTRUMENT,
            side,
            1,
            121,
            0,
            NOW,
            fee_asset="USDT",
            metadata={"trade_id": "1"},
        )
    )
    orders.reload()
    positions.reload()
    sign = 1 if side is OrderSide.BUY else -1
    broker.position_quantity = sign
    service.on_fill("active_income", orders.get(order.order_id), sign, NOW)
    assert {row.kind: row.trigger_price for row in broker.placed} == pytest.approx(
        {
            ProtectionKind.STOP: 120 * (1 - sign * 0.03),
            ProtectionKind.TAKE_PROFIT: 120 * (1 + sign * 0.06),
        }
    )
    assert positions.get(order.portfolio_id, INSTRUMENT).metadata["reference_entry_price"] == 120
    assert orders.get(order.order_id).metadata["reference_price"] == 100
    with pytest.raises(ValueError, match="before submission"):
        orders.bind_entry_reference(order.order_id, price=130, observed_at=NOW)
    with pytest.raises(ValueError, match="cannot be repriced"):
        stops.reprice_unsubmitted(stops.all()[0].stop_id, 130)
    database.dispose()


def test_native_reconciliation_does_not_submit_or_count_paper_protection():
    service, broker, manager = _service()
    manager.create(
        ProtectiveStop(
            stop_id="paper-only",
            portfolio_id="active-income-portfolio",
            instrument_id=INSTRUMENT,
            exit_side=OrderSide.SELL,
            quantity=10,
            trigger_price=95,
            created_at=NOW,
            entry_order_id="paper-entry",
            entry_quantity=10,
            execution_mode="paper",
        )
    )
    assert service.reconcile("active_income", NOW) == ()
    assert broker.actions == []
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    protected = service.on_fill("active_income", order, 1, NOW)
    assert protected.protected_quantity == 1
    assert broker.placed[0].qty == 1
    assert len(broker.placed) == 1
    assert manager.for_entry_order(order.order_id)[0].native_order_id == "native-1"


def test_partial_fill_resizes_and_replaces_native_stop() -> None:
    service, broker, _manager = _service()
    broker.position_quantity = 0.4
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)

    resized = service.on_fill("active_income", order, 0.4, NOW)
    assert resized is not None
    assert resized.status is StopStatus.PROTECTED
    assert resized.quantity == pytest.approx(0.4)
    assert resized.protected_quantity == pytest.approx(0.4)
    assert broker.cancelled == ["native-1"]
    assert [item.qty for item in broker.placed] == [1.0, 0.4]
    assert broker.actions == ["place", "place", "cancel"]


def test_scale_in_protects_only_uncovered_position_quantity() -> None:
    service, broker, manager = _service()
    first = _order()
    second = replace(first, order_id="entry-2")
    service.prepare_entry("active_income", first, NOW)
    service.on_fill("active_income", first, 1.0, NOW)
    service.prepare_entry("active_income", second, NOW)
    service.on_fill("active_income", second, 1.4, NOW)
    assert [item.qty for item in broker.placed] == pytest.approx([1.0, 0.4])
    assert sum(stop.protected_quantity for stop in manager.active()) == pytest.approx(1.4)


def test_partial_close_preserves_old_stop_until_replacement_is_confirmed() -> None:
    service, broker, manager = _service()
    broker.position_quantity = 0.4
    entry = _order()
    service.prepare_entry("active_income", entry, NOW)
    service.on_fill("active_income", entry, 1.0, NOW)
    service.on_fill("active_income", _order(reduce_only=True), 0.4, NOW)
    assert broker.actions == ["place", "place", "cancel"]
    assert broker.placed[0].client_id != broker.placed[1].client_id
    assert sum(stop.protected_quantity for stop in manager.active()) == pytest.approx(0.4)


def test_accepted_stop_timeout_is_recovered_after_restart_without_resubmission() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    place = broker.place_protective_stop

    def accepted_timeout(**kwargs):
        place(**kwargs)
        raise TimeoutError("response lost after exchange acceptance")

    broker.place_protective_stop = accepted_timeout
    with pytest.raises(ProtectiveStopError):
        service.on_fill("active_income", order, 1.0, NOW)
    restarted, _, _ = _service(broker=broker, manager=StopManager(manager.store))
    recovered = restarted.on_fill("active_income", order, 1.0, NOW)
    assert recovered.status is StopStatus.PROTECTED
    assert len(broker.placed) == 1


def test_failed_replacement_retains_previous_exchange_identity() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)

    def unavailable(**kwargs):
        raise TimeoutError("exchange unavailable")

    broker.place_protective_stop = unavailable
    with pytest.raises(ProtectiveStopError):
        service.on_fill("active_income", order, 0.4, NOW)
    pending = StopManager(manager.store).for_entry_order(order.order_id)[0]
    assert pending.previous_native_order_id == "native-1"
    assert pending.previous_native_client_id == broker.placed[0].client_id
    assert pending.submission_attempted is True
    assert broker.cancelled == []
    with pytest.raises(ProtectiveStopError, match="blocks new entry risk"):
        service.prepare_entry("active_income", replace(order, order_id="entry-2"), NOW)


def test_trigger_during_replacement_does_not_certify_protection() -> None:
    service, broker, manager = _service()
    broker.position_quantity = 0.4
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)
    cancel = broker.cancel_protective_stop

    def triggered(**kwargs):
        return replace(cancel(**kwargs), status=ProtectiveOrderStatus.TRIGGERED)

    broker.cancel_protective_stop = triggered
    with pytest.raises(ProtectiveStopError, match="trigger race"):
        service.on_fill("active_income", order, 0.4, NOW)
    unresolved = manager.for_entry_order(order.order_id)[0]
    assert unresolved.status is StopStatus.CONFIRMATION_FAILED
    assert unresolved.previous_native_order_id == "native-1"
    assert unresolved.native_order_id == "native-2"
    assert broker.actions == ["place", "place", "cancel"]


def test_changed_exchange_position_keeps_previous_protection_and_blocks_entry() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)
    broker.position_quantity = 0.2
    with pytest.raises(ProtectiveStopError, match="position changed"):
        service.on_fill("active_income", order, 0.4, NOW)
    assert broker.cancelled == []
    assert manager.for_entry_order(order.order_id)[0].previous_native_order_id == "native-1"


@pytest.mark.parametrize("field", ["quantity", "trigger_price", "protected_quantity"])
def test_nonfinite_stop_values_cannot_certify_coverage(field) -> None:
    service, _, _ = _service()
    stop = service.prepare_entry("active_income", _order(), NOW)
    with pytest.raises(ValueError, match="finite"):
        replace(stop, **{field: float("nan")})


def test_flat_recovery_cancels_both_orders_after_interrupted_replacement() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)
    with pytest.raises(ProtectiveStopError, match="position changed"):
        service.on_fill("active_income", order, 0.4, NOW)
    restarted, _, refreshed = _service(broker=broker, manager=StopManager(manager.store))
    restarted.on_fill("active_income", _order(reduce_only=True), 0.0, NOW)
    assert set(broker.cancelled) == {"native-1", "native-2"}
    assert refreshed.active() == ()


def test_futures_entry_without_trigger_fails_before_exchange_side_effect() -> None:
    service, broker, _manager = _service()
    order = replace(_order(), metadata={"reference_price": 100.0})
    with pytest.raises(ProtectiveStopError, match="protective_stop_price"):
        service.prepare_entry("active_income", order, NOW)
    assert broker.placed == []


def test_stop_failure_is_durable_and_latches_unprotected_state() -> None:
    broker = _Broker()

    def fail(**_kwargs):
        raise RuntimeError("exchange unavailable")

    broker.place_protective_stop = fail
    service, _broker, manager = _service(broker=broker)
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    with pytest.raises(ProtectiveStopError, match="exchange unavailable"):
        service.on_fill("active_income", order, 1.0, NOW)
    assert (
        manager.get(manager.for_entry_order(order.order_id)[0].stop_id).status
        is StopStatus.CONFIRMATION_FAILED
    )


def test_rejected_entry_cancels_its_unsubmitted_protective_stop() -> None:
    service, broker, manager = _service()
    order = _order()
    service.prepare_entry("active_income", order, NOW)

    cancelled = service.on_order_status(
        "active_income",
        replace(order, status=OrderStatus.REJECTED),
        OrderStatus.REJECTED.value,
        NOW,
    )

    assert cancelled is not None
    assert cancelled.status is StopStatus.CANCELLED
    assert manager.active() == ()
    assert broker.placed == []


def test_unknown_algo_update_is_not_silently_ignored() -> None:
    service, _broker, _manager = _service()
    event = MarketEvent(
        instrument_id=INSTRUMENT,
        event_type="algo_update",
        exchange_timestamp=NOW,
        receive_timestamp=NOW,
        sequence=1,
        payload={
            "event": "ALGO_UPDATE",
            "data": {"a": {"s": "BTCUSDT", "aid": "unknown", "algoStatus": "TRIGGERED"}},
        },
    )
    result = service.on_algo_update("active_income", event)
    assert result["reason_code"] == "unknown_protective_algo_update"


def test_reconcile_cancels_orphaned_native_stop_after_position_is_flat() -> None:
    class Positions:
        @staticmethod
        def reload():
            pass

        @staticmethod
        def get(_portfolio_id, _instrument_id):
            return type("Position", (), {"quantity": 0.0})()

    broker = _Broker()
    manager = StopManager(_StopStore())
    service = LiveProtectiveStopService(
        stop_manager=manager,
        venues={"active_income": _Venue(broker)},
        products={"active_income": {"portfolio_id": "active-income-portfolio"}},
        accounts={"futures-account": {"market": "usdt_futures"}},
        positions=Positions(),
    )
    order = _order()
    service.prepare_entry("active_income", order, NOW)
    service.on_fill("active_income", order, 1.0, NOW)
    stop_id = manager.for_entry_order(order.order_id)[0].stop_id

    result = service.reconcile("active_income", NOW)

    assert result == ({"stop_id": stop_id, "status": "cancelled_flat"},)
    assert broker.cancelled == ["native-1"]
    assert manager.active() == ()


def test_native_bracket_keeps_stop_and_target_coverage_separate_after_partial_fills():
    service, broker, manager = _service()
    order = _order()
    order = replace(
        order,
        metadata={
            **order.metadata,
            "position_rules": {
                "stop_loss": 0.05,
                "take_profit": 0.06,
                "horizon_bars": 3,
                "execution_policy": "resting_bracket/v1",
            },
        },
    )
    service.prepare_entry("active_income", order, NOW)
    assert len(manager.for_entry_order(order.order_id)) == 2
    assert not broker.placed
    for quantity in (0.4, 1.0):
        broker.position_quantity = quantity
        filled = replace(
            order,
            filled_quantity=quantity,
            status=OrderStatus.PARTIALLY_FILLED if quantity < 1 else OrderStatus.FILLED,
        )
        service.on_fill("active_income", filled, quantity, NOW)
        coverage = {stop.kind: stop.protected_quantity for stop in manager.active()}
        assert coverage == {ProtectionKind.STOP: quantity, ProtectionKind.TAKE_PROFIT: quantity}
    assert [item.kind for item in broker.placed[:2]] == [
        ProtectionKind.STOP,
        ProtectionKind.TAKE_PROFIT,
    ]
    assert broker.placed[1].trigger_price == pytest.approx(106)
    broker.position_quantity = 0.2
    service.on_fill("active_income", _order(reduce_only=True, quantity=0.8), 0.2, NOW)
    assert {stop.kind: stop.protected_quantity for stop in manager.active()} == {
        ProtectionKind.STOP: 0.2,
        ProtectionKind.TAKE_PROFIT: 0.2,
    }
    broker.position_quantity = 0
    service.on_fill("active_income", _order(reduce_only=True, quantity=0.2), 0, NOW)
    assert not manager.active()
