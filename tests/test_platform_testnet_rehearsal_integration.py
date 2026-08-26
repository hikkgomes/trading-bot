from __future__ import annotations

import datetime as dt
import os
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.binance_market import normalise_public_event
from src.data.binance_user_stream import normalise_user_event
from src.data.database import PlatformDatabase
from src.data.feature_store import SqlFeatureStore
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import to_primitive
from src.domain.instruments import Instrument, MarketType
from src.execution.broker import (
    Broker,
    BrokerOrderAcknowledgement,
)
from src.execution.broker import (
    Fill as BrokerFill,
)
from src.execution.broker import (
    Order as BrokerOrder,
)
from src.execution.broker import (
    Position as BrokerPosition,
)
from src.execution.live_exchange import BrokerExecutionVenue
from src.execution.order_groups import OrderGroupManager, SqlOrderGroupStore
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.reconciler import reconcile_account
from src.execution.recovery import SqlRecoveryStore
from src.observability.decision_trace import SqlDecisionTraceStore
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.risk.engine import (
    SqlRiskDecisionStore,
    SqlRiskSnapshotStore,
)
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.health import DatabaseHeartbeatStore
from src.services.order_execution import (
    DatabaseExecutionWorker,
    DatabaseLiveExecutionWorker,
    DatabaseUserStreamWorker,
)
from src.services.order_recovery import DatabaseLiveRecoveryWorker
from src.services.platform_smoke import (
    _install_risk_policy,
    _seed_strategy,
    _smoke_state_policy,
)
from src.services.platform_testnet_rehearsal import PlatformTestnetRehearsal
from src.services.portfolio_engine import (
    DatabasePortfolioTargetBuilder,
    DatabasePortfolioTargetWorker,
)
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.portfolio_state import (
    DatabasePortfolioSourceService,
    DatabasePortfolioStateWorker,
)
from src.services.risk_service import DatabaseRiskWorker
from src.services.scheduler import DatabaseJobQueue
from src.services.strategy_evaluator import DatabaseStrategyEvaluator


class _TestnetBroker(Broker):
    """Explicit no-network broker used only by the PostgreSQL rehearsal."""

    name = "platform-testnet-adapter"

    def __init__(self) -> None:
        self.submissions: list[BrokerOrder] = []

    def get_price(self, symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return 102_000.0

    def get_balance(self) -> float:
        return 10_000.0

    def get_position(self, symbol: str) -> BrokerPosition:
        return BrokerPosition(symbol=symbol)

    def place_order(self, order: BrokerOrder) -> BrokerFill:
        return BrokerFill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=102_000.0,
            fee=0.0,
            exchange_order_id="testnet-order-1",
            client_order_id=order.client_id,
        )

    def submit_order(self, order: BrokerOrder) -> BrokerOrderAcknowledgement:
        self.submissions.append(order)
        return BrokerOrderAcknowledgement(
            exchange_order_id="testnet-order-1",
            client_order_id=str(order.client_id),
            status="acknowledged",
        )


class _MappedWorker:
    def __init__(self, worker: Any, success_reason: str, success_key: str) -> None:
        self.worker = worker
        self.success_reason = success_reason
        self.success_key = success_key

    def run_once(self, *, now: str) -> dict[str, Any]:
        result = dict(self.worker.run_once(now=now))
        if not result.get(self.success_key):
            return {"reason_code": str(result.get("reason_code") or "stage_failed"), **result}
        return {
            "reason_code": self.success_reason,
            "worker_result": result,
        }


class _LivePipeline:
    def __init__(self, planner: DatabaseExecutionWorker, submitter: DatabaseLiveExecutionWorker):
        self.planner = planner
        self.submitter = submitter

    def run_once(self, *, now: str) -> dict[str, Any]:
        planned = dict(self.planner.run_once(now=now))
        if planned.get("orders", 0) != 1:
            return {"reason_code": "live_order_submission_failed", "planning": planned}
        submitted = dict(self.submitter.run_once(now=now))
        return {**submitted, "planning": planned}


class _UserStreamCoordinator:
    def __init__(self, worker: DatabaseUserStreamWorker, flags: dict[str, bool]):
        self.worker = worker
        self.flags = flags

    def run_once(self, *, now: str) -> dict[str, Any]:
        result = dict(self.worker.run_once(now=now))
        if result.get("accounting_job_id"):
            self.flags["accounting"] = True
        order_result = result.get("order_result")
        if isinstance(order_result, dict) and order_result.get("recovery_job_id"):
            self.flags["recovery"] = True
        return result


class _ClearingWorker:
    def __init__(self, worker: Any, flag: dict[str, bool], name: str):
        self.worker = worker
        self.flag = flag
        self.name = name

    def run_once(self, *, now: str) -> dict[str, Any]:
        result = dict(self.worker.run_once(now=now))
        if str(result.get("reason_code") or "").endswith("recorded") or str(
            result.get("reason_code") or ""
        ).endswith("created"):
            self.flag[self.name] = False
        return result


def test_postgresql_platform_testnet_rehearsal_uses_real_service_chain(tmp_path: Path) -> None:
    database_url = os.environ.get("TRADING_PLATFORM_TESTNET_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("set TRADING_PLATFORM_TESTNET_DATABASE_URL for the PostgreSQL rehearsal")
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        pytest.fail("TRADING_PLATFORM_TESTNET_DATABASE_URL must be a PostgreSQL URL")

    database = PlatformDatabase(database_url)
    try:
        assert database.is_postgresql
        database.assert_migrated()
        _run_rehearsal(database, tmp_path)
    finally:
        database.dispose()


def _run_rehearsal(database: PlatformDatabase, root: Path) -> None:
    suffix = uuid4().hex[:10]
    now = dt.datetime(2026, 8, 24, tzinfo=dt.UTC).isoformat()
    prefix = f"platform-testnet:{suffix}"
    product_id = f"active_income_testnet_{suffix}"
    account_id = f"testnet-account-{suffix}"
    portfolio_id = f"testnet-portfolio-{suffix}"
    risk_policy_id = f"testnet-risk-{suffix}"
    product = {
        "product_id": product_id,
        "account_id": account_id,
        "portfolio_id": portfolio_id,
        "base_accounting_asset": "USDT",
        "risk_policy_id": risk_policy_id,
        "execution_costs": {"fee_bps": 5.0, "slippage_bps": 2.0},
        "maximum_positions": 12,
        "maximum_gross": 1.5,
        "maximum_net": 0.5,
        "maximum_margin": 0.5,
        "execution_mode": "live",
    }
    accounts = {
        account_id: {
            "account_id": account_id,
            "market": "usdt_futures",
            "margin_mode": "isolated",
            "maximum_leverage": 3.0,
            "quote_assets": ["USDT"],
            "settlement_assets": ["USDT"],
        }
    }
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    queue = DatabaseJobQueue(database.engine)
    roles = (
        "data",
        "feature",
        "strategy",
        "state",
        "portfolio",
        "risk",
        "execution",
        "live",
        "user",
        "accounting",
        "recovery",
    )
    worker_ids = {role: f"{prefix}:{role}" for role in roles}
    for role, worker_id in worker_ids.items():
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex",
            role=role,
            capabilities=(role,),
            observed_at=now,
        )
    universe_snapshot_id = SqlUniverseStore(database.engine).record_snapshot(
        universe_id=f"{prefix}:universe",
        observed_at=now,
        observations=(
            InstrumentObservation(
                instrument, 1_000.0, 1e9, 1_000_000.0, 1.0, 1e9, 0.0, 0.2, 1e7, 1.0
            ),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    _install_risk_policy(database, risk_policy_id)
    assignment_id = _seed_strategy(
        database,
        product,
        instrument,
        universe_snapshot_id,
        now,
        prefix,
        execution_mode="live",
    )
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    snapshots = SqlRiskSnapshotStore(database.engine)
    repository = SqlPortfolioRepository(database.engine, require_pipeline_identity=True)
    feature_store = SqlFeatureStore(database.engine)
    positions = PositionManager(SqlPositionStore(database.engine))
    order_manager = OrderManager(SqlOrderStore(database.engine))
    traces = SqlDecisionTraceStore(database.engine)
    order_groups = OrderGroupManager(SqlOrderGroupStore(database.engine))
    ledger = Ledger(
        product_id=product_id,
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id=product_id),
    )
    accounting_service = AccountingService(
        engine=database.engine,
        ledgers={product_id: ledger},
        snapshot_store=snapshots,
    )
    queue.enqueue(
        job_id=f"{prefix}:initial-balance",
        name="accounting_event",
        payload={
            "kind": "balance",
            "product_id": product_id,
            "account_id": account_id,
            "observed_at": now,
            "balances": {"USDT": 10_000.0},
        },
        available_at=now,
    )
    initial_accounting = DatabaseAccountingWorker(
        queue=queue,
        worker_id=worker_ids["accounting"],
        service=accounting_service,
    ).run_once(now=now)
    assert initial_accounting["reason_code"] == "accounting_event_recorded"

    close_ms = int(dt.datetime.fromisoformat(now).timestamp() * 1_000) - 1
    market_event = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=now,
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": "100000",
                "h": "102500",
                "l": "99750",
                "c": "102000",
                "v": "25",
                "spread_bps": "1",
                "visible_depth": "10000000",
                "volatility": "0.2",
                "funding": "0.0",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id=f"{prefix}:market-event",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(market_event),
        },
        available_at=now,
    )
    data_worker = DatabaseMarketDataWriter(
        queue=queue,
        worker_id=worker_ids["data"],
        root=root / "parquet",
        snapshot_store=snapshots,
        product_ids_by_market={"futures": (product_id,)},
    )
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id=worker_ids["feature"],
        store=feature_store,
        job_names=("live_feature_calculation",),
        parquet_root=root / "parquet",
        active_assignments=lambda instrument_id: tuple(
            item
            for item in assignments.active_assignments(product_id)
            if item.get("instrument_id") == instrument_id
        ),
        snapshot_store=snapshots,
        feature_graph_for_assignment=lambda _assignment: {"required_nodes": ["bar_return"]},
    )
    source_service = DatabasePortfolioSourceService(
        engine=database.engine,
        store=snapshots,
        products={product_id: product},
        accounts=accounts,
    )
    state_worker = DatabasePortfolioStateWorker(
        queue=queue,
        worker_id=worker_ids["state"],
        store=snapshots,
        refresh_sources=source_service.refresh,
    )
    strategy_worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id=worker_ids["strategy"],
        feature_store=feature_store,
        portfolio=repository,
        assignments=assignments,
    )
    portfolio_worker = DatabasePortfolioTargetWorker(
        queue=queue,
        worker_id=worker_ids["portfolio"],
        build_target=DatabasePortfolioTargetBuilder(
            repository=repository,
            snapshot_store=snapshots,
            positions=positions,
            product_configuration={product_id: product},
            account_configuration=accounts,
        ),
    )
    risk_worker = DatabaseRiskWorker(
        queue=queue,
        worker_id=worker_ids["risk"],
        store=SqlRiskDecisionStore(database.engine),
        snapshot_store=snapshots,
        execution_modes={product_id: "live"},
    )
    execution_worker = DatabaseExecutionWorker(
        queue=queue,
        worker_id=worker_ids["execution"],
        order_manager=order_manager,
        positions=positions,
        risk_store=SqlRiskDecisionStore(database.engine),
        trace_store=traces,
        order_groups=order_groups,
        snapshot_store=snapshots,
        product_execution={
            product_id: {
                "execution_mode": "live",
                "execution_costs": product["execution_costs"],
                "base_accounting_asset": "USDT",
            }
        },
    )
    broker = _TestnetBroker()
    venue = BrokerExecutionVenue(
        order_manager=order_manager,
        position_manager=positions,
        broker=broker,
        instruments={instrument.instrument_id: instrument},
    )
    live_worker = DatabaseLiveExecutionWorker(
        queue=queue,
        worker_id=worker_ids["live"],
        order_manager=order_manager,
        positions=positions,
        ledgers={product_id: ledger},
        trace_store=traces,
        venues={product_id: venue},
        authorise=lambda payload, _order: _assert_live_assignment(
            assignments, product_id, str(payload["product_id"])
        ),
        order_groups=order_groups,
    )
    user_worker = DatabaseUserStreamWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_ids["user"],
        order_manager=order_manager,
        positions=positions,
        ledgers={product_id: ledger},
        trace_store=traces,
        account_products={account_id: product_id},
        order_groups=order_groups,
    )
    recovery_worker = DatabaseLiveRecoveryWorker(
        queue=queue,
        worker_id=worker_ids["recovery"],
        store=SqlRecoveryStore(database.engine),
        reconcile_product=lambda _product: reconcile_account(
            local_positions={},
            exchange_positions={},
            local_open_order_ids=set(),
            exchange_open_order_ids={"unexpected-testnet-order"},
        ),
        account_products={account_id: product_id},
    )

    assert data_worker.run_once(now=now)["reason_code"] == "market_event_written"
    DatabaseHeartbeatStore(database.engine).record(
        service_name="data-writer",
        node_id="linux-optiplex",
        observed_at=now,
        healthy=True,
        payload={"reason_code": "market_event_written"},
    )
    assert feature_worker.run_once(now=now)["features"] > 0
    assert (
        state_worker.schedule_from_latest(
            products={product_id: product},
            state_policies={product_id: _smoke_state_policy(product, instrument)},
            now=now,
        )
        == 1
    )
    assert state_worker.run_once(now=now)["reason_code"] == "canonical_portfolio_state_published"

    assignment = assignments.by_id(assignment_id)
    assert assignment is not None and assignment["execution_mode"] == "live"
    flags = {"accounting": False, "recovery": False}
    user_events: deque[str] = deque(("fill", "balance", "unknown"))

    def enqueue_user_event() -> bool:
        if not user_events:
            return False
        kind = user_events.popleft()
        order_manager.reload()
        order = next(
            (
                item
                for item in order_manager.all()
                if item.portfolio_id == portfolio_id
            ),
            None,
        )
        if kind == "fill":
            if order is None:
                raise AssertionError("live execution did not create an order before fill event")
            client_id = str(order.metadata["client_order_id"])
            payload = {
                "e": "ORDER_TRADE_UPDATE",
                "E": close_ms + 2,
                "o": {
                    "s": "BTCUSDT",
                    "c": client_id,
                    "S": order.side.value.upper(),
                    "x": "TRADE",
                    "X": "FILLED",
                    "l": str(order.quantity),
                    "L": "102000",
                    "n": "0",
                    "N": "USDT",
                    "t": f"testnet-trade-{suffix}",
                },
            }
        elif kind == "balance":
            payload = {
                "e": "ACCOUNT_UPDATE",
                "E": close_ms + 3,
                "a": {"B": [{"a": "USDT", "f": "10000", "l": "0"}]},
            }
        else:
            payload = {
                "e": "ORDER_TRADE_UPDATE",
                "E": close_ms + 4,
                "o": {
                    "s": "BTCUSDT",
                    "c": f"unknown-testnet-client-{suffix}",
                    "S": "BUY",
                    "x": "NEW",
                    "X": "NEW",
                },
            }
        event = normalise_user_event(
            account_id=account_id,
            market="futures",
            payload=payload,
            receive_timestamp=now,
        )
        queue.enqueue(
            job_id=f"{prefix}:user:{kind}",
            name="user_stream_event",
            payload={
                "account_id": account_id,
                "market": "futures",
                "event": to_primitive(event),
            },
            available_at=now,
        )
        return True

    rehearsal = PlatformTestnetRehearsal(
        active_assignment=_AssignmentCheck(assignments, product_id),
        strategy_evaluator=_MappedWorker(
            strategy_worker, "strategy_evaluation_recorded", "forecast_id"
        ),
        portfolio=_MappedWorker(portfolio_worker, "portfolio_target_created", "risk_job_id"),
        risk=_RiskStage(risk_worker),
        live_execution=_LivePipeline(execution_worker, live_worker),
        user_stream=_UserStreamCoordinator(user_worker, flags),
        accounting=_ClearingWorker(
            DatabaseAccountingWorker(
                queue=queue,
                worker_id=worker_ids["accounting"],
                service=accounting_service,
            ),
            flags,
            "accounting",
        ),
        recovery=_ClearingWorker(recovery_worker, flags, "recovery"),
        has_pending_user_stream=enqueue_user_event,
        has_pending_accounting=lambda: flags["accounting"],
        has_pending_recovery=lambda: flags["recovery"],
    )
    report = rehearsal.run(now=now)
    assert report.ok, report.to_dict()
    assert len(broker.submissions) == 1
    order_manager.reload()
    assert order_manager.all()[0].status.value == "filled"
    positions.reload()
    assert positions.get(portfolio_id, instrument.instrument_id).quantity > 0
    assert len(SqlRecoveryStore(database.engine).read()) == 1


class _AssignmentCheck:
    def __init__(self, assignments: SqlActiveStrategyAssignmentRepository, product_id: str):
        self.assignments = assignments
        self.product_id = product_id

    def run_once(self, *, now: str) -> dict[str, Any]:
        assignment = self.assignments.active(self.product_id)
        if assignment is None or assignment["execution_mode"] != "live":
            return {"reason_code": "active_assignment_missing"}
        return {"reason_code": "active_assignment_loaded", "assignment_id": assignment["id"]}


class _RiskStage:
    def __init__(self, worker: DatabaseRiskWorker):
        self.worker = worker

    def run_once(self, *, now: str) -> dict[str, Any]:
        result = dict(self.worker.run_once(now=now))
        if result.get("accepted") is not True:
            return {
                "reason_code": str(result.get("reason_code") or "risk_decision_failed"),
                **result,
            }
        return {"reason_code": "risk_decision_recorded", "worker_result": result}


def _assert_live_assignment(
    assignments: SqlActiveStrategyAssignmentRepository,
    product_id: str,
    payload_product_id: str,
) -> None:
    if product_id != payload_product_id:
        raise AssertionError("live order product binding changed")
    assignment = assignments.active(product_id)
    if assignment is None or assignment["execution_mode"] != "live":
        raise AssertionError("live order has no approved live assignment")
