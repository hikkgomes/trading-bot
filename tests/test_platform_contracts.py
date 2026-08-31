from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.accounting.btc_performance import build_btc_performance_report
from src.accounting.ledger import JsonlLedgerStore, Ledger, SqlLedgerStore
from src.accounting.nav import NavSnapshot, btc_nav, usdt_nav
from src.accounting.reconciliation import reconcile_accounting
from src.agents.context import AgentContext
from src.agents.openclaw_bridge import OpenClawAgentBridge
from src.agents.proposals import AgentAction, AgentProposal, AgentRole
from src.agents.reviewer import AgentCodeReviewer
from src.agents.sandbox import CommandResult
from src.agents.store import SqlAgentStore
from src.data.binance_market import normalise_public_event
from src.data.binance_user_stream import normalise_user_event
from src.data.database import CORE_TABLE_NAMES, PlatformDatabase
from src.data.feature_store import DeterministicFeatureCalculator, FeatureValue, SqlFeatureStore
from src.data.historical_query import DuckDBHistoricalQuery
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash, to_primitive
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.instruments import Instrument, MarketType
from src.domain.market_events import MarketEvent, MarketEventType
from src.domain.orders import OrderSide, OrderStatus
from src.domain.portfolios import TargetPosition
from src.domain.risk import RiskDecision
from src.domain.strategies import MechanismCategory, ResearchThesis, StrategySourceType
from src.execution.live_exchange import BrokerExecutionVenue
from src.execution.order_groups import (
    JsonlOrderGroupStore,
    OrderGroupManager,
    OrderGroupStatus,
    SqlOrderGroupStore,
    plan_order_group,
)
from src.execution.order_manager import JsonlOrderStore, OrderManager, SqlOrderStore
from src.execution.order_planner import plan_orders
from src.execution.paper import PaperBroker
from src.execution.paper_exchange import PaperExchange
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.reconciler import reconcile_account
from src.execution.recovery import (
    JsonlRecoveryStore,
    RecoveryActionType,
    SqlRecoveryStore,
    plan_recovery,
)
from src.execution.stops import JsonlStopStore, ProtectiveStop, SqlStopStore, StopManager
from src.observability.decision_trace import (
    DecisionTrace,
    DecisionTraceStage,
    JsonlDecisionTraceStore,
    SqlDecisionTraceStore,
)
from src.observability.health import assess_platform_health
from src.observability.metrics import MetricsRegistry
from src.observability.reports import DatabasePlatformReport
from src.portfolio.optimiser import PortfolioConstraints
from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import BtcAllocationPolicy, target_btc_allocation
from src.products.execution_diagnostic import ExecutionDiagnostic
from src.research.artefacts import StrategyArtefact, StrategyArtefactStore
from src.research.backtest.bar_engine import BarPortfolioEngine, BarStep
from src.research.backtest.event_engine import (
    EventReplayEngine,
    ReplayEvent,
    SimulatedLimitOrder,
    SimulatedOrderSide,
    SimulatedOrderStatus,
)
from src.research.catalogue import registered_strategy_candidates, registered_strategy_theses
from src.research.coordinator import ResearchCoordinator
from src.research.ml import MlExperimentRunner, ModelArtefactStore, SqlModelArtefactStore
from src.research.providers import provider_candidate
from src.research.store import SqlResearchStore
from src.research.theses import REQUIRED_NEGATIVE_CONTROLS, SqlThesisRegistry
from src.risk.account import AccountRiskLimits, assess_account_risk
from src.risk.engine import JsonlRiskDecisionStore, SqlRiskDecisionStore, combine_risk_decisions
from src.risk.global_risk import GlobalRiskLimits, assess_global_risk
from src.risk.instrument import InstrumentRiskLimits, assess_instrument_risk
from src.risk.product import ProductRiskLimits, assess_product_risk
from src.risk.sleeve import SleeveRiskLimits, assess_sleeve_risk
from src.risk.strategy import StrategyRiskLimits, assess_strategy_risk
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.backups import (
    BackupStore,
    create_directory_archive,
    postgresql_environment,
    verify_directory_archive,
)
from src.services.config import PlatformConfig, load_platform_config, load_split_configuration
from src.services.control_api import DatabaseControlPlane
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.execution_service import ExecutionService
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.health import DatabaseHeartbeatStore
from src.services.market_gap_recovery import BinanceMarketGapRepair
from src.services.market_gateway import funding_event_from_mark_price
from src.services.order_execution import (
    DatabaseExecutionWorker,
    DatabaseLiveExecutionWorker,
    DatabasePaperExecutionWorker,
    DatabaseUserStreamWorker,
)
from src.services.order_recovery import DatabaseLiveRecoveryWorker
from src.services.portfolio_engine import DatabasePortfolioWorker, DatabaseProductCoordinator
from src.services.portfolio_service import (
    DatabaseProductCycleWorker,
    DatabaseProductSupervisor,
    SqlPortfolioRepository,
)
from src.services.product_supervisor import (
    ActiveIncomeProductSupervisor,
    BtcAccumulationProductSupervisor,
)
from src.services.promotion import (
    DatabasePromotionWorker,
    LifecycleState,
    PromotionEvidence,
    PromotionPolicy,
    SqlPromotionStore,
    decide_promotion,
)
from src.services.report_worker import DatabaseReportWorker
from src.services.risk_service import DatabaseRiskWorker
from src.services.runtime import ServiceRuntime
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-13T12:00:00+00:00"


def _test_thesis(*, budget: int = 20) -> ResearchThesis:
    return ResearchThesis(
        mechanism_category=MechanismCategory.BEHAVIOURAL,
        market_rationale="A predeclared test mechanism.",
        expected_causal_chain=("state", "signal", "return"),
        expected_direction="declared signal direction",
        expected_horizon="one day",
        required_data=("closed bars",),
        permitted_features=("returns",),
        instrument_universe=(BTC,),
        generalisation_scope={"product": "active_income"},
        failure_regimes=("structural break",),
        falsification_tests=("chronological holdout",),
        negative_controls=REQUIRED_NEGATIVE_CONTROLS,
        execution_capacity_assumptions={"maximum_participation": 0.01},
        parent_thesis_ids=(),
        cumulative_trial_budget=budget,
        created_at=NOW,
        creator_identity="test-suite",
    )


LATER = "2026-08-13T13:00:00+00:00"
BTC = "binance:futures:BTCUSDT:USDT"
ETH = "binance:futures:ETHUSDT:USDT"


def forecast(
    *,
    strategy_id: str,
    instrument_id: str = BTC,
    direction: ForecastDirection = ForecastDirection.LONG,
    score: float = 0.8,
    confidence: float = 0.75,
    maximum_position: float = 0.2,
    metadata: dict[str, object] | None = None,
) -> AlphaForecast:
    return AlphaForecast(
        strategy_version_id=strategy_id,
        product_id="active_income",
        instrument_id=instrument_id,
        direction=direction,
        score=score,
        expected_return=0.01,
        confidence=confidence,
        horizon_seconds=3600,
        valid_from=NOW,
        valid_until=LATER,
        target_volatility=0.1,
        maximum_position=maximum_position,
        metadata=metadata or {},
    )


def accepted_risk_assessment(*, product_id: str = "active_income"):
    decisions = (
        assess_strategy_risk(
            decision_id="strategy-ok",
            position_fraction=0.1,
            turnover_fraction=0.1,
            trades_today=1,
            expected_slippage_bps=2,
            expected_funding_cost_fraction=0.001,
            limits=StrategyRiskLimits(0.2, 0.5, 10, 5, 0.01),
        ),
        assess_instrument_risk(
            decision_id="instrument-ok",
            position_notional=1_000,
            order_notional=500,
            visible_depth_fraction=0.01,
            spread_bps=1,
            volatility=0.2,
            concentration_fraction=0.1,
            limits=InstrumentRiskLimits(2_000, 1_000, 0.05, 5, 1, 0.2),
        ),
        assess_sleeve_risk(
            decision_id="sleeve-ok",
            capital_fraction=0.2,
            drawdown_fraction=0.01,
            maximum_correlation=0.4,
            beta=0.1,
            turnover_fraction=0.2,
            limits=SleeveRiskLimits(0.3, 0.1, 0.8, 0.5, 0.5),
        ),
        assess_product_risk(
            decision_id="product-ok",
            gross_fraction=0.3,
            net_fraction=0.1,
            drawdown_fraction=0.01,
            margin_fraction=0.2,
            daily_pnl_fraction=0.01,
            limits=ProductRiskLimits(0.6, 0.4, 0.1, 0.5, 0.03),
        ),
        assess_account_risk(
            decision_id="account-ok",
            used_margin_fraction=0.2,
            liquidation_buffer_fraction=0.8,
            unknown_positions={},
            limits=AccountRiskLimits(0.5, 0.3),
        ),
        assess_global_risk(
            decision_id="global-ok",
            drawdown_fraction=0.01,
            exchange_connected=True,
            data_age_seconds=1,
            clock_skew_seconds=0.1,
            database_healthy=True,
            execution_drift=False,
            model_drift=False,
            limits=GlobalRiskLimits(0.2, 60, 2),
        ),
    )
    return combine_risk_decisions(
        decisions,
        assessment_id=f"portfolio-ok:{product_id}",
        product_id=product_id,
    )


def btc_forecast(*, strategy_id: str, direction: ForecastDirection) -> AlphaForecast:
    return AlphaForecast(
        strategy_version_id=strategy_id,
        product_id="btc_accumulation",
        instrument_id="binance:spot:BTCUSDT",
        direction=direction,
        score=0.8,
        expected_return=0.01,
        confidence=0.75,
        horizon_seconds=3600,
        valid_from=NOW,
        valid_until=LATER,
        target_volatility=0.1,
        maximum_position=0.2,
    )


def test_active_income_allocates_simultaneous_multi_symbol_targets():
    portfolio = ActiveIncomePortfolio(
        PortfolioConstraints(
            portfolio_id="active_income",
            equity=10_000,
            max_positions=2,
            max_gross_fraction=0.4,
            max_net_fraction=0.4,
            max_symbol_fraction=0.2,
        )
    )

    targets = portfolio.target_positions(
        [
            forecast(strategy_id="trend", instrument_id=BTC),
            forecast(strategy_id="mean_reversion", instrument_id=ETH),
        ],
        prices={BTC: 100_000, ETH: 5_000},
        valid_until=LATER,
        correlations={BTC: {ETH: 0.4}},
        beta_by_instrument={BTC: 1.0, ETH: 0.1},
    )

    assert {target.instrument_id for target in targets} == {BTC, ETH}
    assert sum(abs(target.target_fraction) for target in targets) == pytest.approx(0.24)


def test_active_income_allocation_uses_liquidity_funding_margin_and_existing_positions():
    portfolio = ActiveIncomePortfolio(
        PortfolioConstraints(
            portfolio_id="active_income",
            equity=10_000,
            max_positions=3,
            max_gross_fraction=0.8,
            max_net_fraction=0.8,
            max_symbol_fraction=0.5,
        )
    )

    targets = portfolio.target_positions(
        [forecast(strategy_id="trend", instrument_id=BTC, maximum_position=0.5)],
        prices={BTC: 100_000, ETH: 5_000},
        valid_until=LATER,
        observed_volatility={BTC: 0.2},
        liquidity_fraction_caps={BTC: 0.1},
        funding_rates={BTC: 0.002},
        current_quantities={ETH: 1.0},
        available_margin_fraction=0.05,
    )

    by_symbol = {target.instrument_id: target for target in targets}
    assert by_symbol[BTC].target_fraction == pytest.approx(0.05)
    assert by_symbol[BTC].metadata["funding_rate"] == pytest.approx(0.002)
    assert by_symbol[ETH].target_quantity == 0
    assert by_symbol[ETH].metadata["reason_code"] == "no_valid_forecast"


def test_active_income_allocation_enforces_cluster_and_drawdown_limits():
    portfolio = ActiveIncomePortfolio(
        PortfolioConstraints(
            portfolio_id="active_income",
            equity=10_000,
            max_positions=3,
            max_gross_fraction=0.8,
            max_net_fraction=0.8,
            max_symbol_fraction=0.5,
            max_cluster_fraction=0.1,
            max_drawdown_fraction=0.1,
        )
    )
    clustered = portfolio.target_positions(
        [
            forecast(strategy_id="btc", instrument_id=BTC, maximum_position=0.5),
            forecast(strategy_id="eth", instrument_id=ETH, maximum_position=0.5),
        ],
        prices={BTC: 100_000, ETH: 5_000},
        valid_until=LATER,
        cluster_by_instrument={BTC: "large-cap", ETH: "large-cap"},
    )
    risk_exit = portfolio.target_positions(
        [forecast(strategy_id="btc", instrument_id=BTC)],
        prices={BTC: 100_000, ETH: 5_000},
        valid_until=LATER,
        current_quantities={BTC: 0.01, ETH: 0.2},
        product_drawdown_fraction=0.11,
    )

    assert sum(abs(item.target_fraction) for item in clustered) == pytest.approx(0.1)
    assert {item.metadata["cluster"] for item in clustered} == {"large-cap"}
    assert {item.instrument_id for item in risk_exit} == {BTC, ETH}
    assert all(item.target_quantity == 0 for item in risk_exit)
    assert all(item.metadata["reason_code"] == "product_drawdown_limit" for item in risk_exit)


def test_order_manager_represents_partial_fill_before_recovery(tmp_path: Path):
    portfolio = ActiveIncomePortfolio(
        PortfolioConstraints(portfolio_id="active_income", equity=10_000)
    )
    target = portfolio.target_positions(
        [forecast(strategy_id="trend")], prices={BTC: 100_000}, valid_until=LATER
    )[0]
    manager = OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl"))
    positions = PositionManager()
    exchange = PaperExchange(
        order_manager=manager,
        position_manager=positions,
        price_source=lambda _: 100_000,
        fill_fraction=0.5,
    )
    service = ExecutionService(paper_exchange=exchange, positions=positions)
    decision = RiskDecision(
        decision_id="risk-1",
        scope="account",
        accepted=True,
        reason_code=None,
        evaluated_at=NOW,
        input_snapshot={},
    )

    orders, fills, traces = service.execute_targets(
        portfolio_id="active_income", targets=[target], risk_decision=decision
    )

    assert len(fills) == len(orders) == len(traces) == 1
    assert manager.get(orders[0].order_id).status is OrderStatus.PARTIALLY_FILLED
    assert positions.get("active_income", BTC).quantity == pytest.approx(target.target_quantity / 2)
    assert traces[0].first_blocked_stage == DecisionTraceStage.ORDER_FILLED.value
    assert (
        traces[0].stages[DecisionTraceStage.ORDER_FILLED.value]["reason_code"]
        == "partial_fill_pending"
    )

    exchange.fill_remaining(orders[0].order_id)

    assert manager.get(orders[0].order_id).status is OrderStatus.FILLED
    assert positions.get("active_income", BTC).quantity == pytest.approx(target.target_quantity)


def test_order_and_position_state_recover_from_journal(tmp_path: Path):
    journal = tmp_path / "orders.jsonl"
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=0.01,
        target_notional=1_000,
        target_fraction=0.1,
        strategy_contributions={"trend": 0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )
    manager = OrderManager(JsonlOrderStore(journal))
    positions = PositionManager()
    exchange = PaperExchange(
        order_manager=manager,
        position_manager=positions,
        price_source=lambda _: 100_000,
        fill_fraction=0.5,
    )
    order = plan_orders((target,), current_quantities={}, decided_at=NOW)[0]
    exchange.submit(order)

    recovered_manager = OrderManager(JsonlOrderStore(journal))
    recovered_positions = PositionManager()
    recovered_positions.recover_from_orders(recovered_manager)

    assert recovered_manager.get(order.order_id).status is OrderStatus.PARTIALLY_FILLED
    assert len(recovered_manager.fills_for(order.order_id)) == 1
    assert recovered_positions.get("active_income", BTC).quantity == pytest.approx(0.005)


def test_position_reversal_is_close_then_open_not_one_reduce_only_order():
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=-1.0,
        target_notional=-100_000,
        target_fraction=-0.1,
        strategy_contributions={"reversal": -0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )

    orders = plan_orders((target,), current_quantities={BTC: 1.0}, decided_at=NOW)

    assert len(orders) == 2
    assert orders[0].quantity == 1.0
    assert orders[0].reduce_only is True
    assert orders[0].metadata["phase"] == "close_for_reversal"
    assert orders[1].quantity == 1.0
    assert orders[1].reduce_only is False
    assert orders[1].metadata["phase"] == "open_after_reversal"


def test_multi_leg_group_has_durable_deterministic_unwind_path(tmp_path: Path):
    targets = (
        TargetPosition(
            portfolio_id="active_income",
            instrument_id=BTC,
            target_quantity=0.01,
            target_notional=1_000,
            target_fraction=0.1,
            strategy_contributions={"pair": 0.1},
            risk_budget=0.1,
            valid_until=LATER,
        ),
        TargetPosition(
            portfolio_id="active_income",
            instrument_id=ETH,
            target_quantity=-0.2,
            target_notional=-1_000,
            target_fraction=-0.1,
            strategy_contributions={"pair": -0.1},
            risk_budget=0.1,
            valid_until=LATER,
        ),
    )
    planned = plan_order_group(targets, current_quantities={}, decided_at=NOW)
    store = JsonlOrderGroupStore(tmp_path / "groups.jsonl")
    manager = OrderGroupManager(store)
    manager.create(planned.group)
    manager.transition(planned.group.group_id, OrderGroupStatus.PRIMARY_SUBMITTED)

    recovered = OrderGroupManager(store)
    recovery = recovered.recovery_plan(planned.group.group_id)

    assert all(order.group_id == planned.group.group_id for order in planned.orders)
    assert recovered.get(planned.group.group_id).status is OrderGroupStatus.PRIMARY_SUBMITTED
    assert recovery.action == "unwind_to_flat"
    assert recovery.target_quantities == {BTC: 0.0, ETH: 0.0}


def test_protective_stops_survive_restart_and_trigger_once(tmp_path: Path):
    store = JsonlStopStore(tmp_path / "stops.jsonl")
    manager = StopManager(store)
    manager.create(
        ProtectiveStop(
            stop_id="stop-1",
            portfolio_id="active_income",
            instrument_id=BTC,
            exit_side=OrderSide.SELL,
            quantity=0.01,
            trigger_price=95_000,
            created_at=NOW,
        )
    )

    recovered = StopManager(store)
    triggered = recovered.evaluate({BTC: 94_000}, triggered_at=LATER)

    assert len(triggered) == 1
    assert StopManager(store).active() == ()


def test_reconciliation_creates_durable_fail_closed_recovery_plan(tmp_path: Path):
    reconciliation = reconcile_account(
        local_positions={BTC: 0.1},
        exchange_positions={BTC: 0.2, ETH: -1.0},
        local_open_order_ids={"known"},
        exchange_open_order_ids={"known", "unknown"},
    )
    store = JsonlRecoveryStore(tmp_path / "recovery.jsonl")

    plan = plan_recovery(reconciliation, created_at=NOW, store=store)

    assert plan is not None
    assert plan.requires_operator_review is True
    assert {action.action_type for action in plan.actions} == {
        RecoveryActionType.CANCEL_UNKNOWN_ORDER,
        RecoveryActionType.EMERGENCY_FLATTEN,
        RecoveryActionType.RECONCILE_POSITION,
    }
    assert store.read() == (plan,)


def test_live_recovery_worker_reconciles_missing_exchange_order(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-optiplex:execution-engine",
        node_id="linux-optiplex",
        role="execution-engine",
        capabilities=("live_order_recovery",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="live-recovery-1",
        name="live_order_recovery",
        payload={"product_id": "active_income", "order_id": "missing-order"},
        available_at=NOW,
    )
    worker = DatabaseLiveRecoveryWorker(
        queue=queue,
        worker_id="linux-optiplex:execution-engine",
        store=SqlRecoveryStore(database.engine),
        reconcile_product=lambda _product_id: reconcile_account(
            local_positions={},
            exchange_positions={},
            local_open_order_ids={"missing-order"},
            exchange_open_order_ids=set(),
        ),
        account_products={"binance-futures-main": "active_income"},
    )

    result = worker.run_once(now=NOW)
    plans = SqlRecoveryStore(database.engine).read()

    assert result["reason_code"] == "live_recovery_plan_created"
    assert result["operator_review_required"] is True
    assert plans[0].actions[0].action_type is RecoveryActionType.RECONCILE_ORDER
    assert plans[0].actions[0].target == "missing-order"


def test_live_recovery_worker_verifies_a_user_stream_reconnect_without_difference(
    tmp_path: Path,
):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'stream-recovery.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-optiplex:execution-engine",
        node_id="linux-optiplex",
        role="execution-engine",
        capabilities=("live_order_recovery",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="stream-recovery-1",
        name="live_order_recovery",
        payload={
            "account_id": "binance-futures-main",
            "market": "futures",
            "recovery_kind": "user_stream_reconnect",
            "reason_code": "user_stream_disconnect",
            "observed_at": NOW,
        },
        available_at=NOW,
    )
    worker = DatabaseLiveRecoveryWorker(
        queue=queue,
        worker_id="linux-optiplex:execution-engine",
        store=SqlRecoveryStore(database.engine),
        reconcile_product=lambda _product_id: reconcile_account(
            local_positions={},
            exchange_positions={},
            local_open_order_ids=set(),
            exchange_open_order_ids=set(),
        ),
        account_products={"binance-futures-main": "active_income"},
    )

    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "live_recovery_verified"
    database.dispose()


def test_rejected_risk_stops_before_order_planning(tmp_path: Path):
    diagnostic = ExecutionDiagnostic()
    manager = OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl"))
    positions = PositionManager()
    service = ExecutionService(
        paper_exchange=PaperExchange(
            order_manager=manager,
            position_manager=positions,
            price_source=lambda _: diagnostic.price,
        ),
        positions=positions,
    )
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=0.01,
        target_notional=1_000,
        target_fraction=0.1,
        strategy_contributions={"test": 0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )
    risk = RiskDecision(
        decision_id="risk-2",
        scope="account",
        accepted=False,
        reason_code="account_margin_limit",
        evaluated_at=NOW,
        input_snapshot={},
    )

    orders, fills, traces = service.execute_targets(
        portfolio_id="active_income", targets=[target], risk_decision=risk
    )

    assert orders == fills == ()
    assert traces[0].first_blocked_stage == DecisionTraceStage.RISK_ACCEPTED.value
    assert not manager.all()


def test_btc_allocation_is_fractional_not_a_synthetic_short_signal():
    allocation = target_btc_allocation(
        [
            forecast(
                strategy_id="risk-off",
                direction=ForecastDirection.SHORT,
                score=1.0,
                confidence=1.0,
                maximum_position=0.2,
            )
        ],
        policy=BtcAllocationPolicy(core_btc_fraction=0.7, max_tactical_fraction=0.3),
    )

    assert allocation.target_btc_fraction == pytest.approx(0.5)
    assert allocation.stablecoin_fraction == pytest.approx(0.5)


def test_diagnostic_completes_paper_open_and_close_without_live_eligibility(tmp_path: Path):
    journal = tmp_path / "diagnostic_orders.jsonl"
    report = ExecutionDiagnostic().run(journal_path=journal)

    assert report["ok"] is True
    assert report["paper_trade_allowed"] is True
    assert report["live_allowed"] is False
    assert report["promotion_eligible"] is False
    assert report["final_quantity"] == 0.0
    assert report["accounting_entries_added"] == 6
    assert report["decision_traces_added"] == 2
    assert ExecutionDiagnostic().run(journal_path=journal)["ok"] is True


def test_registered_catalogue_enters_the_common_research_contract():
    candidates = registered_strategy_candidates(
        product="active_income", dataset_snapshot_hashes=("sha256:" + "a" * 64,)
    )

    assert len(candidates) >= 22
    assert {candidate.provider for candidate in candidates} == {"registered_strategy_catalogue"}
    assert all(
        candidate.definition.source_type.value == "registered_python" for candidate in candidates
    )


def test_all_strategy_sources_share_one_persistent_research_queue(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = SqlResearchStore(database.engine)
    coordinator = ResearchCoordinator(store)
    thesis = _test_thesis()
    SqlThesisRegistry(database.engine).register(thesis)
    dataset_hashes = ("sha256:" + "a" * 64,)
    source_types = (
        StrategySourceType.PARAMETER_SEARCH,
        StrategySourceType.GENERATED_DSL,
        StrategySourceType.MUTATION,
        StrategySourceType.CROSSOVER,
        StrategySourceType.MACHINE_LEARNING,
        StrategySourceType.CROSS_SECTIONAL,
        StrategySourceType.RELATIVE_VALUE,
        StrategySourceType.MICROSTRUCTURE,
        StrategySourceType.ENSEMBLE,
        StrategySourceType.AGENT_GENERATED_PYTHON,
    )
    candidates = [
        provider_candidate(
            identity=f"candidate-{source_type.value}",
            version="v1",
            family=source_type.value,
            product="active_income",
            thesis_id=thesis.thesis_id,
            lineage_id=canonical_hash({"source": source_type.value}),
            provider=f"provider-{source_type.value}",
            source_type=source_type,
            source_payload={"kind": source_type.value},
            dataset_snapshot_hashes=dataset_hashes,
            submitted_at=NOW,
        )
        for source_type in source_types
    ]
    catalogue_theses = registered_strategy_theses(
        product="active_income", instrument_universe=(BTC,)
    )
    registry = SqlThesisRegistry(database.engine)
    for catalogue_thesis in catalogue_theses.values():
        registry.register(catalogue_thesis)
    candidates.append(
        registered_strategy_candidates(
            product="active_income",
            dataset_snapshot_hashes=dataset_hashes,
            instrument_universe=(BTC,),
        )[0]
    )

    coordinator.register(candidates)
    recovered = ResearchCoordinator(store)

    assert len(recovered.pending()) == len(source_types) + 1
    assert {item.definition.source_type for item in recovered.pending()} == {
        *source_types,
        StrategySourceType.REGISTERED_PYTHON,
    }
    assert {
        "instrument",
        "strategy_definition",
        "experiment",
        "target_position",
        "order_intent",
        "accounting_entry",
        "job",
        "worker_lease",
        "service_heartbeat",
    } <= CORE_TABLE_NAMES


def test_ml_experiment_is_chronological_bounded_and_content_addressed(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    runner = MlExperimentRunner(
        artefact_store=ModelArtefactStore(tmp_path / "models"),
        metadata_store=SqlModelArtefactStore(database.engine),
        maximum_rows=100,
    )
    start = dt.datetime.fromisoformat(NOW)
    rows = tuple(
        {
            "available_at": (start + dt.timedelta(minutes=index)).isoformat(),
            "momentum": float(index % 5),
            "volatility": float((index % 3) + 1),
            "label": float(index % 2),
        }
        for index in range(40)
    )

    first = runner.run(
        candidate_id="ml-candidate-1",
        model_name="logistic_regression",
        feature_names=("momentum", "volatility"),
        target_name="label",
        rows=rows,
        created_at=NOW,
        train_fraction=0.7,
        embargo_rows=2,
    )
    second = runner.run(
        candidate_id="ml-candidate-1",
        model_name="logistic_regression",
        feature_names=("momentum", "volatility"),
        target_name="label",
        rows=rows,
        created_at=NOW,
        train_fraction=0.7,
        embargo_rows=2,
    )

    assert first == second
    assert first.train_rows == 26
    assert first.validation_rows == 12
    assert first.content_hash.startswith("sha256:")
    assert (tmp_path / "models" / first.relative_path).is_file()


def test_protected_results_persist_but_never_enter_adaptive_feedback(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    coordinator = ResearchCoordinator(SqlResearchStore(database.engine))
    thesis = _test_thesis()
    SqlThesisRegistry(database.engine).register(thesis)
    candidate = provider_candidate(
        identity="protected-candidate",
        version="v1",
        family="generated",
        product="active_income",
        thesis_id=thesis.thesis_id,
        lineage_id=canonical_hash({"lineage": "protected"}),
        provider="generated",
        source_type=StrategySourceType.GENERATED_DSL,
        source_payload={"rule": "close > open"},
        dataset_snapshot_hashes=("sha256:" + "b" * 64,),
        submitted_at=NOW,
    )
    candidate_id = coordinator.submit(candidate)

    coordinator.evaluate(
        candidate_id,
        screening=lambda _: (True, None, {"signals": 100}),
        development=lambda _: (True, None, {"return": 0.1}),
        robustness=lambda _: (True, None, {"stable": True}),
        protected=lambda _: (True, None, {"protected_metric": 99}),
    )
    recovered = ResearchCoordinator(SqlResearchStore(database.engine))

    assert recovered.result(candidate_id) is not None
    assert "protected_metric" in str(recovered.result(candidate_id).evidence)
    assert "protected_metric" not in str(recovered.development_feedback())
    assert "protected_rejected" not in recovered.development_feedback()[0]["stages"]


def test_strategy_artefact_is_reproducible_and_content_addressed(tmp_path: Path):
    thesis = _test_thesis()
    candidate = provider_candidate(
        identity="artifact-candidate",
        version="v1",
        family="ml",
        product="active_income",
        thesis_id=thesis.thesis_id,
        lineage_id=canonical_hash({"lineage": "artifact"}),
        provider="ml",
        source_type=StrategySourceType.MACHINE_LEARNING,
        source_payload={"model": "lightgbm"},
        dataset_snapshot_hashes=("sha256:" + "c" * 64,),
        submitted_at=NOW,
    )
    artefact = StrategyArtefact(
        definition=candidate.definition,
        dependency_hash="sha256:" + "d" * 64,
        dataset_snapshot_hashes=candidate.dataset_snapshot_hashes,
        feature_set_version="features-v1",
        cost_model_version="costs-v1",
        validation_evidence={"accepted": True},
        holdout_claim={"claim_id": "claim-1"},
        promotion_policy={"paper": True, "live_canary": False},
        position_limits={"maximum_position": 0.1},
        risk_limits={"maximum_drawdown": 0.05},
        model_hashes=("sha256:" + "e" * 64,),
        supported_products=("active_income",),
        supported_instruments=(BTC,),
        created_at=NOW,
    )
    store = StrategyArtefactStore(tmp_path / "artefacts")

    first = store.put(artefact)
    second = store.put(artefact)

    assert first == second
    assert first.name == f"{artefact.artefact_hash.removeprefix('sha256:')}.json"


def test_database_workers_claim_by_priority_and_resume_after_lease_expiry(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="mac-research",
        node_id="macbook",
        role="research-worker",
        capabilities=("research", "ml"),
        observed_at=NOW,
    )
    queue.register_worker(
        worker_id="mac-recovery",
        node_id="macbook",
        role="research-worker",
        capabilities=("research",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="low",
        name="research",
        payload={"candidate": "low"},
        available_at=NOW,
        priority=1,
    )
    queue.enqueue(
        job_id="high",
        name="research",
        payload={"candidate": "high"},
        available_at=NOW,
        priority=10,
    )

    high = queue.claim(worker_id="mac-research", now=NOW, lease_seconds=10)
    low = queue.claim(worker_id="mac-recovery", now=NOW, lease_seconds=10)

    assert high is not None and high.job_id == "high"
    assert low is not None and low.job_id == "low"
    queue.complete(low, completed_at="2026-08-13T12:00:05+00:00")
    assert queue.recover_expired(now="2026-08-13T12:00:11+00:00") == 1
    recovered = queue.claim(
        worker_id="mac-recovery",
        now="2026-08-13T12:00:11+00:00",
        lease_seconds=10,
    )
    assert recovered is not None
    assert recovered.job_id == "high"
    assert recovered.attempt == 2


def test_database_risk_worker_persists_all_six_scopes_and_aggregate(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-risk",
        node_id="linux-optiplex",
        role="risk-engine",
        capabilities=("risk_assessment",),
        observed_at=NOW,
    )
    blueprint = accepted_risk_assessment()
    queue.enqueue(
        job_id="risk-job-1",
        name="risk_assessment",
        payload={
            "product_id": "active_income",
            "assessment_id": "risk-assessment-1",
            **{
                item.scope: {
                    "decision_id": f"risk-job-1:{item.scope}",
                    "inputs": dict(item.input_snapshot),
                    "limits": dict(item.limits),
                }
                for item in blueprint.decisions
            },
        },
        available_at=NOW,
    )
    store = SqlRiskDecisionStore(database.engine)
    worker = DatabaseRiskWorker(
        queue=queue,
        worker_id="linux-risk",
        store=store,
    )

    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "risk_assessment_accepted"
    assert store.assessment("risk-assessment-1").accepted is True
    assert len(store.read()) == 7


def test_decision_trace_requires_reason_for_first_blocked_stage():
    trace = DecisionTrace.start(event_id="event-1", instrument_id=BTC).pass_stage(
        DecisionTraceStage.DATA_AVAILABLE
    )

    with pytest.raises(ValueError, match="reason_code"):
        trace.block(DecisionTraceStage.FEATURE_AVAILABLE, reason_code="")


def test_product_ledgers_cannot_mix_btc_and_usdt_accounting():
    btc_ledger = Ledger(product_id="btc_accumulation", accounting_asset="BTC")
    usdt_ledger = Ledger(product_id="active_income", accounting_asset="USDT")

    btc_entry = btc_ledger.append(
        entry_id="btc-fee",
        postings={"expense:fees": Decimal("0.001"), "assets:btc": Decimal("-0.001")},
        occurred_at=NOW,
    )
    usdt_entry = usdt_ledger.append(
        entry_id="usdt-fee",
        postings={"expense:fees": Decimal("1.25"), "assets:usdt": Decimal("-1.25")},
        occurred_at=NOW,
    )

    assert btc_entry.accounting_asset == "BTC"
    assert usdt_entry.accounting_asset == "USDT"
    assert btc_entry.entry_hash != usdt_entry.entry_hash


def test_btc_performance_reports_hold_benchmark_exposure_and_regimes():
    ledger = Ledger(product_id="btc_accumulation", accounting_asset="BTC")
    ledger.record_capital(entry_id="capital", amount=Decimal("1"), occurred_at=NOW)
    ledger.record_fee(entry_id="fee", amount=Decimal("0.001"), occurred_at=NOW)
    snapshots = (
        NavSnapshot(
            product_id="btc_accumulation",
            accounting_asset="BTC",
            nav=1.0,
            observed_at=NOW,
            components={
                "btc_balance": 0.7,
                "stablecoin_balance": 30_000,
                "stablecoin_per_btc": 100_000,
                "regime": "trend",
            },
            passive_benchmark_nav=1.0,
        ),
        NavSnapshot(
            product_id="btc_accumulation",
            accounting_asset="BTC",
            nav=0.95,
            observed_at=LATER,
            components={
                "btc_balance": 0.7,
                "stablecoin_balance": 30_000,
                "stablecoin_per_btc": 120_000,
                "regime": "risk_off",
            },
            passive_benchmark_nav=1.0,
        ),
    )

    report = build_btc_performance_report(snapshots, ledger=ledger)

    assert report.btc_balance == pytest.approx(0.7)
    assert report.btc_nav == pytest.approx(0.95)
    assert report.btc_vs_passive_hold == pytest.approx(-0.05)
    assert report.time_outside_btc_fraction == pytest.approx(1.0)
    assert report.missed_btc_appreciation == pytest.approx(0.05)
    assert report.fees_paid_btc == Decimal("0.001")
    assert report.performance_by_regime == {"trend": pytest.approx(-0.05)}


def test_ledger_reconstructs_nav_costs_and_attribution_after_restart(tmp_path: Path):
    store = JsonlLedgerStore(tmp_path / "active_income_ledger.jsonl")
    ledger = Ledger(product_id="active_income", accounting_asset="USDT", store=store)
    ledger.record_capital(entry_id="capital", amount=Decimal("1000"), occurred_at=NOW)
    ledger.record_realised_pnl(
        entry_id="pnl",
        amount=Decimal("25"),
        occurred_at=NOW,
        attribution={"strategy": "trend", "symbol": BTC, "sleeve": "directional"},
    )
    ledger.record_fee(
        entry_id="fee",
        amount=Decimal("2"),
        occurred_at=NOW,
        attribution={"strategy": "trend", "symbol": BTC, "sleeve": "directional"},
    )
    ledger.record_funding(
        entry_id="funding",
        amount=Decimal("-1"),
        occurred_at=NOW,
        attribution={"strategy": "trend", "symbol": BTC, "sleeve": "directional"},
    )

    recovered = Ledger(product_id="active_income", accounting_asset="USDT", store=store)

    assert recovered.nav() == Decimal("1022")
    assert recovered.attribution("strategy")["trend"] == Decimal("22")
    assert len(recovered.entries) == 4


def test_six_level_risk_decisions_are_persisted_with_input_snapshots(tmp_path: Path):
    decisions = (
        assess_strategy_risk(
            decision_id="strategy-1",
            position_fraction=0.1,
            turnover_fraction=0.1,
            trades_today=1,
            expected_slippage_bps=2,
            expected_funding_cost_fraction=0.001,
            limits=StrategyRiskLimits(0.2, 0.5, 10, 5, 0.01),
        ),
        assess_instrument_risk(
            decision_id="instrument-1",
            position_notional=1_000,
            order_notional=500,
            visible_depth_fraction=0.01,
            spread_bps=1,
            volatility=0.2,
            concentration_fraction=0.1,
            limits=InstrumentRiskLimits(2_000, 1_000, 0.05, 5, 1, 0.2),
        ),
        assess_sleeve_risk(
            decision_id="sleeve-1",
            capital_fraction=0.2,
            drawdown_fraction=0.01,
            maximum_correlation=0.4,
            beta=0.1,
            turnover_fraction=0.2,
            limits=SleeveRiskLimits(0.3, 0.1, 0.8, 0.5, 0.5),
        ),
        assess_product_risk(
            decision_id="product-1",
            gross_fraction=0.3,
            net_fraction=0.1,
            drawdown_fraction=0.01,
            margin_fraction=0.2,
            daily_pnl_fraction=0.01,
            limits=ProductRiskLimits(0.6, 0.4, 0.1, 0.5, 0.03),
        ),
        assess_account_risk(
            decision_id="account-1",
            used_margin_fraction=0.2,
            liquidation_buffer_fraction=0.8,
            unknown_positions={},
            limits=AccountRiskLimits(0.5, 0.3),
        ),
        assess_global_risk(
            decision_id="global-1",
            drawdown_fraction=0.01,
            exchange_connected=True,
            data_age_seconds=1,
            clock_skew_seconds=0.1,
            database_healthy=True,
            execution_drift=False,
            model_drift=False,
            limits=GlobalRiskLimits(0.2, 60, 2),
        ),
    )
    store = JsonlRiskDecisionStore(tmp_path / "risk.jsonl")

    assessment = combine_risk_decisions(
        decisions,
        assessment_id="portfolio-1",
        product_id="active_income",
        store=store,
    )

    assert assessment.accepted is True
    assert len(store.read()) == 7
    assert all(item.input_hash.startswith("sha256:") for item in assessment.decisions)


def test_active_income_supervisor_runs_durable_end_to_end_paper_cycle(tmp_path: Path):
    order_store = JsonlOrderStore(tmp_path / "orders.jsonl")
    trace_store = JsonlDecisionTraceStore(tmp_path / "traces.jsonl")
    ledger_store = JsonlLedgerStore(tmp_path / "ledger.jsonl")
    ledger = Ledger(product_id="active_income", accounting_asset="USDT", store=ledger_store)
    ledger.record_capital(entry_id="capital", amount=Decimal("10000"), occurred_at=NOW)
    manager = OrderManager(order_store)
    positions = PositionManager()
    exchange = PaperExchange(
        order_manager=manager,
        position_manager=positions,
        price_source=lambda _: 100_000,
        fee_bps=5,
        fee_asset="USDT",
    )
    execution = ExecutionService(
        paper_exchange=exchange,
        positions=positions,
        ledger=ledger,
        trace_store=trace_store,
    )
    supervisor = ActiveIncomeProductSupervisor(
        portfolio=ActiveIncomePortfolio(
            PortfolioConstraints(portfolio_id="active_income", equity=10_000)
        ),
        execution_service=execution,
    )
    with pytest.raises(ValueError, match="another product"):
        supervisor.process_forecasts(
            event_id="wrong-product-risk",
            event_instrument_id=BTC,
            forecasts=(forecast(strategy_id="trend"),),
            prices={BTC: 100_000},
            valid_until=LATER,
            risk_assessment=accepted_risk_assessment(product_id="btc_accumulation"),
        )

    result = supervisor.process_forecasts(
        event_id="market-event-1",
        event_instrument_id=BTC,
        forecasts=(forecast(strategy_id="trend"),),
        prices={BTC: 100_000},
        valid_until=LATER,
        risk_assessment=accepted_risk_assessment(),
    )

    assert result.accepted is True
    assert len(result.targets) == len(result.orders) == len(result.fills) == 1
    assert positions.get("active_income", BTC).quantity == pytest.approx(0.012)
    assert ledger.nav() == Decimal("9999.4")
    assert len(JsonlDecisionTraceStore(tmp_path / "traces.jsonl").read()) == 1
    assert len(OrderManager(order_store).all_fills()) == 1
    assert (
        Ledger(product_id="active_income", accounting_asset="USDT", store=ledger_store).nav()
        == ledger.nav()
    )


def test_realised_pnl_and_slippage_reconcile_without_double_counting(tmp_path: Path):
    prices = {BTC: 100.0}
    positions = PositionManager()
    manager = OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl"))
    ledger = Ledger(product_id="active_income", accounting_asset="USDT")
    ledger.record_capital(entry_id="capital", amount=Decimal("10000"), occurred_at=NOW)
    execution = ExecutionService(
        paper_exchange=PaperExchange(
            order_manager=manager,
            position_manager=positions,
            price_source=prices.__getitem__,
            fee_bps=0,
            slippage_bps=100,
            fee_asset="USDT",
        ),
        positions=positions,
        ledger=ledger,
    )
    opened = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=1,
        target_notional=100,
        target_fraction=0.01,
        strategy_contributions={"trend": 0.01},
        risk_budget=0.1,
        valid_until=LATER,
    )
    execution.execute_targets(
        portfolio_id="active_income",
        targets=(opened,),
        risk_decision=accepted_risk_assessment().aggregate,
    )
    prices[BTC] = 110
    execution.execute_targets(
        portfolio_id="active_income",
        targets=(
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0,
                target_notional=0,
                target_fraction=0,
                strategy_contributions={"trend": 0},
                risk_budget=0.1,
                valid_until=LATER,
            ),
        ),
        risk_decision=accepted_risk_assessment().aggregate,
    )

    effects = {entry.metadata["kind"]: entry for entry in ledger.entries[1:]}
    assert effects["realised_pnl"].metadata["pnl_effect"] == "10.00"
    assert ledger.nav() == Decimal("10007.9")


def test_active_income_supervisor_records_exact_no_forecast_cause(tmp_path: Path):
    manager = OrderManager(JsonlOrderStore(tmp_path / "orders.jsonl"))
    positions = PositionManager()
    trace_store = JsonlDecisionTraceStore(tmp_path / "traces.jsonl")
    supervisor = ActiveIncomeProductSupervisor(
        portfolio=ActiveIncomePortfolio(
            PortfolioConstraints(portfolio_id="active_income", equity=10_000)
        ),
        execution_service=ExecutionService(
            paper_exchange=PaperExchange(
                order_manager=manager,
                position_manager=positions,
                price_source=lambda _: 100_000,
            ),
            positions=positions,
            trace_store=trace_store,
        ),
    )

    result = supervisor.process_forecasts(
        event_id="market-event-no-signal",
        event_instrument_id=BTC,
        forecasts=(),
        prices={BTC: 100_000},
        valid_until=LATER,
        risk_assessment=accepted_risk_assessment(),
    )

    assert result.first_blocked_stage == DecisionTraceStage.SIGNAL_PRODUCED.value
    assert (
        result.traces[0].stages[DecisionTraceStage.SIGNAL_PRODUCED.value]["reason_code"]
        == "no_actionable_forecast"
    )
    assert JsonlDecisionTraceStore(tmp_path / "traces.jsonl").read()[0][1] == result.traces[0]


def test_btc_supervisor_rebuys_from_actual_stablecoin_balance_and_accounts_in_btc(
    tmp_path: Path,
):
    instrument_id = "binance:spot:BTCUSDT"
    ledger = Ledger(
        product_id="btc_accumulation",
        accounting_asset="BTC",
        store=JsonlLedgerStore(tmp_path / "btc_ledger.jsonl"),
    )
    ledger.record_capital(entry_id="capital", amount=Decimal("1"), occurred_at=NOW)
    manager = OrderManager(JsonlOrderStore(tmp_path / "btc_orders.jsonl"))
    positions = PositionManager()
    supervisor = BtcAccumulationProductSupervisor(
        execution_service=ExecutionService(
            paper_exchange=PaperExchange(
                order_manager=manager,
                position_manager=positions,
                price_source=lambda _: 100_000,
                fee_bps=5,
                fee_asset="BTC",
                fee_in_base=True,
            ),
            positions=positions,
            ledger=ledger,
        ),
        policy=BtcAllocationPolicy(core_btc_fraction=0.70, max_tactical_fraction=0.30),
    )

    result = supervisor.process_forecasts(
        event_id="btc-market-event-1",
        instrument_id=instrument_id,
        forecasts=(btc_forecast(strategy_id="btc-momentum", direction=ForecastDirection.LONG),),
        btc_balance=0.5,
        stablecoin_balance=50_000,
        stablecoin_per_btc=100_000,
        valid_until=LATER,
        risk_assessment=accepted_risk_assessment(product_id="btc_accumulation"),
    )

    assert result.btc_nav_before_costs == pytest.approx(1.0)
    assert result.allocation.target_btc_fraction == pytest.approx(0.82)
    assert result.orders[0].quantity == pytest.approx(0.32)
    assert positions.get("btc_accumulation", instrument_id).quantity == pytest.approx(0.81984)
    assert ledger.nav() == Decimal("0.99984")


def test_bar_engine_rebalances_multi_symbol_targets_with_costs_and_funding():
    result = BarPortfolioEngine(initial_equity=1_000, fee_bps=10).simulate(
        (
            BarStep(
                timestamp=NOW,
                prices={BTC: 100_000, ETH: 5_000},
                target_fractions={BTC: 0.2, ETH: -0.1},
                funding_rates={ETH: 0.001},
            ),
            BarStep(
                timestamp=LATER,
                prices={BTC: 105_000, ETH: 4_500},
                target_fractions={BTC: 0.0, ETH: 0.0},
            ),
        )
    )

    assert result.quantities == {BTC: 0.0, ETH: 0.0}
    assert result.fees_paid > 0
    assert result.funding_paid < 0
    assert len(result.equity_curve) == 2


def test_split_configuration_assigns_every_service_and_blocks_mac_execution():
    config = load_platform_config()

    assert config.node("linux-optiplex").production_authority is True
    assert set(load_split_configuration()) == {
        "accounts",
        "products",
        "portfolios",
        "promotion",
        "research",
        "risk",
    }
    with pytest.raises(PermissionError, match="not assigned"):
        config.assert_service_assignment(node_id="macbook-research", service="execution-engine")
    with pytest.raises(ValueError, match="PostgreSQL"):
        config.database_url({config.database_url_env: "sqlite:///platform.db"})
    with pytest.raises(ValueError, match="TLS"):
        config.database_url(
            {config.database_url_env: "postgresql+psycopg://platform@database/platform"}
        )
    secure_url = config.database_url(
        {
            config.database_url_env: (
                "postgresql+psycopg://platform@database/platform?sslmode=verify-full"
            )
        }
    )
    assert "connect_timeout=5" in secure_url


def test_platform_configuration_rejects_execution_on_non_linux_authority():
    payload = {
        "schema": "platform/v1",
        "nodes": [
            {
                "node_id": "unsafe",
                "operating_system": "macos",
                "production_authority": True,
                "services": sorted(load_platform_config().node("linux-optiplex").services),
            },
            {
                "node_id": "linux",
                "operating_system": "linux",
                "production_authority": False,
                "services": sorted(load_platform_config().node("macbook-research").services),
            },
        ],
        "postgresql": {"url_env": "DATABASE_URL"},
        "paths": {
            "parquet": "data",
            "artefacts": "data/artefacts",
            "reports": "data/reports",
            "backups": "backups",
        },
        "worker_limits": {"research-worker": 1},
        "network": {},
        "logging": {},
        "metrics": {},
        "alerting": {},
        "backup": {},
    }

    with pytest.raises(ValueError, match="Linux"):
        PlatformConfig.from_dict(payload)


def test_service_runtime_persists_health_and_enforces_order_authority(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = DatabaseHeartbeatStore(database.engine)
    execution = ServiceRuntime(
        config=load_platform_config(),
        node_id="linux-optiplex",
        service_name="execution-engine",
        heartbeat_store=store,
    )
    research = ServiceRuntime(
        config=load_platform_config(),
        node_id="macbook-research",
        service_name="research-worker",
        heartbeat_store=store,
    )

    execution.assert_order_submission_authority()
    with pytest.raises(PermissionError, match="cannot submit"):
        research.assert_order_submission_authority()
    cycle = research.run_once(lambda: {"jobs": 0}, observed_at=NOW)

    assert cycle.healthy is True
    assert store.latest()[0].payload["reason_code"] == "cycle_completed"
    assert store.stale(now=LATER, maximum_age_seconds=60)[0].service_name == "research-worker"


def test_control_events_and_heartbeats_are_authoritative_database_state(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    heartbeat_store.record(
        service_name="portfolio-engine",
        node_id="linux-optiplex",
        observed_at=NOW,
        healthy=True,
        payload={"reason_code": "service_alive"},
    )
    control = DatabaseControlPlane(
        database.engine,
        heartbeat_store,
        configuration={
            "accounts": {
                "api_secret_env": "BINANCE_API_SECRET",
                "account_id": "binance-main",
            }
        },
    )

    control.set_paused(
        target="active_income",
        paused=True,
        reason_code="operator_risk_review",
        requested_by="operator",
        changed_at=NOW,
    )
    assert control.is_paused("active_income") is True
    control.set_paused(
        target="active_income",
        paused=False,
        reason_code="review_completed",
        requested_by="operator",
        changed_at=LATER,
    )

    assert control.is_paused("active_income") is False
    assert control.status()["operations"]["heartbeats"][0]["service_name"] == "portfolio-engine"
    assert control.configuration_view()["configuration"] == {
        "accounts": {
            "account_id": "binance-main",
            "api_secret_env": "<redacted>",
        }
    }
    assert control.report()["schema"] == "platform.report/v1"


def test_verified_backup_restores_operational_and_research_files(tmp_path: Path):
    operational = tmp_path / "operational.sql"
    research = tmp_path / "research.parquet"
    operational.write_bytes(b"database dump")
    research.write_bytes(b"parquet bytes")
    store = BackupStore(tmp_path / "backups")

    bundle = store.create(
        backup_id="backup-1",
        created_at=NOW,
        files={"postgresql/platform.sql": operational, "parquet/research.parquet": research},
    )
    restored = store.restore(backup_id=bundle.backup_id, destination=tmp_path / "restore")

    assert len(restored) == 2
    assert (tmp_path / "restore/postgresql/platform.sql").read_bytes() == b"database dump"
    assert (tmp_path / "restore/parquet/research.parquet").read_bytes() == b"parquet bytes"


def test_postgresql_backup_environment_and_parquet_archive_are_verifiable(tmp_path: Path):
    environment = postgresql_environment(
        "postgresql+psycopg://platform:secret@database.internal:5433/trading"
        "?sslmode=verify-full&connect_timeout=7"
    )
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    (parquet / "events.parquet").write_bytes(b"verified parquet")
    archive = create_directory_archive(parquet, tmp_path / "parquet.tar.gz")

    assert environment == {
        "PGHOST": "database.internal",
        "PGPORT": "5433",
        "PGDATABASE": "trading",
        "PGCONNECT_TIMEOUT": "7",
        "PGUSER": "platform",
        "PGPASSWORD": "secret",
        "PGSSLMODE": "verify-full",
    }
    assert verify_directory_archive(archive) == 2


def test_sql_stores_recover_orders_accounting_risk_and_blocked_causes(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    order_store = SqlOrderStore(database.engine)
    manager = OrderManager(order_store)
    positions = PositionManager()
    exchange = PaperExchange(
        order_manager=manager,
        position_manager=positions,
        price_source=lambda _: 100_000,
    )
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=0.01,
        target_notional=1_000,
        target_fraction=0.1,
        strategy_contributions={"trend": 0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )
    order = plan_orders((target,), current_quantities={}, decided_at=NOW)[0]
    exchange.submit(order)

    recovered_orders = OrderManager(SqlOrderStore(database.engine))
    recovered_positions = PositionManager()
    recovered_positions.recover_from_orders(recovered_orders)
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    ledger.record_capital(entry_id="capital", amount=Decimal("1000"), occurred_at=NOW)
    risk_store = SqlRiskDecisionStore(database.engine)
    assessment = accepted_risk_assessment()
    for decision in (*assessment.decisions, assessment.aggregate):
        risk_store.append(decision)
    risk_store.append(assessment.aggregate)
    trace_store = SqlDecisionTraceStore(database.engine)
    trace = DecisionTrace.start(event_id="no-data", instrument_id=ETH, evaluated_at=NOW).block(
        DecisionTraceStage.DATA_AVAILABLE,
        reason_code="market_data_stale",
    )
    trace_store.append(trace)
    trace_store.append(trace)

    assert recovered_positions.get("active_income", BTC).quantity == pytest.approx(0.01)
    assert len(recovered_orders.all_fills()) == 1
    assert Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    ).nav() == Decimal("1000")
    assert len(risk_store.read()) == 7
    assert trace_store.read()[0][1].first_blocked_stage == "data_available"


def test_paper_and_broker_venues_use_the_same_durable_order_contract(tmp_path: Path):
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
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=instrument.instrument_id,
        target_quantity=0.01,
        target_notional=1_000,
        target_fraction=0.1,
        strategy_contributions={"trend": 0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )
    intent = plan_orders((target,), current_quantities={}, decided_at=NOW)[0]
    manager = OrderManager(JsonlOrderStore(tmp_path / "broker-orders.jsonl"))
    positions = PositionManager()
    venue = BrokerExecutionVenue(
        order_manager=manager,
        position_manager=positions,
        broker=PaperBroker(price_source=lambda _: 100_000),
        instruments={instrument.instrument_id: instrument},
    )

    acknowledgement = venue.submit(intent)

    assert acknowledgement.order_id == intent.order_id
    assert manager.get(intent.order_id).status is OrderStatus.ACKNOWLEDGED
    assert positions.get("active_income", instrument.instrument_id).quantity == 0


def test_live_queue_uses_a_mandatory_authoriser_and_the_canonical_broker_contract(
    tmp_path: Path,
):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    worker_id = "linux-optiplex:execution-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="execution-engine",
        capabilities=("execute_targets", "live_order_submit"),
        observed_at=NOW,
    )
    risk_store = SqlRiskDecisionStore(database.engine)
    assessment = accepted_risk_assessment()
    for decision in (*assessment.decisions, assessment.aggregate):
        risk_store.append(decision)
    target = TargetPosition(
        portfolio_id="active_income",
        instrument_id=BTC,
        target_quantity=0.01,
        target_notional=1_000,
        target_fraction=0.1,
        strategy_contributions={"approved-strategy": 0.1},
        risk_budget=0.1,
        valid_until=LATER,
    )
    queue.enqueue(
        job_id="live-target-1",
        name="execute_targets",
        payload={
            "product_id": "active_income",
            "event_id": "live-event-1",
            "evaluated_at": NOW,
            "risk_assessment_id": assessment.aggregate.decision_id,
            "execution_mode": "live",
            "prices": {BTC: 100_000},
            "targets": [to_primitive(target)],
        },
        available_at=NOW,
    )
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    traces = SqlDecisionTraceStore(database.engine)
    planner = DatabaseExecutionWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        risk_store=risk_store,
        trace_store=traces,
        product_execution={
            "active_income": {
                "execution_mode": "live",
                "execution_costs": {"fee_bps": 5, "slippage_bps": 2},
                "base_accounting_asset": "USDT",
            }
        },
    )
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
        order_manager=order_manager,
        position_manager=positions,
        broker=PaperBroker(price_source=lambda _: 100_000),
        instruments={instrument.instrument_id: instrument},
    )
    authorised: list[tuple[str, str]] = []
    submitter = DatabaseLiveExecutionWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        ledgers={
            "active_income": Ledger(
                product_id="active_income",
                accounting_asset="USDT",
                store=SqlLedgerStore(database.engine, product_id="active_income"),
            )
        },
        trace_store=traces,
        venues={"active_income": venue},
        authorise=lambda payload, order: authorised.append(
            (str(payload["authorisation_at"]), order.order_id)
        ),
    )

    planned = planner.run_once(now=NOW)
    submitted = submitter.run_once(now=LATER)

    assert planned["reason_code"] == "live_orders_enqueued"
    assert submitted["reason_code"] == "live_order_acknowledged"
    assert authorised == [(LATER, submitted["order_id"])]
    assert positions.get("active_income", BTC).quantity == 0


def test_database_product_supervisor_executes_dynamic_multi_symbol_forecasts(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    repository = SqlPortfolioRepository(database.engine)
    repository.save_forecast(forecast(strategy_id="btc-trend", instrument_id=BTC))
    repository.save_forecast(forecast(strategy_id="eth-trend", instrument_id=ETH))
    manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    active_execution = ExecutionService(
        paper_exchange=PaperExchange(
            order_manager=manager,
            position_manager=positions,
            price_source=lambda instrument_id: {BTC: 100_000, ETH: 5_000}[instrument_id],
            fee_asset="USDT",
        ),
        positions=positions,
        trace_store=SqlDecisionTraceStore(database.engine),
    )
    supervisor = DatabaseProductSupervisor(
        repository=repository,
        active_income=ActiveIncomeProductSupervisor(
            portfolio=ActiveIncomePortfolio(
                PortfolioConstraints(
                    portfolio_id="active_income",
                    equity=10_000,
                    max_positions=5,
                    max_gross_fraction=0.5,
                )
            ),
            execution_service=active_execution,
        ),
        btc_accumulation=BtcAccumulationProductSupervisor(execution_service=active_execution),
    )

    result = supervisor.run_active_income(
        event_id="portfolio-cycle-1",
        evaluated_at=NOW,
        prices={BTC: 100_000, ETH: 5_000},
        valid_until=LATER,
        risk_assessment=accepted_risk_assessment(),
        correlations={BTC: {ETH: 0.2}},
        beta_by_instrument={BTC: 1.0, ETH: 0.1},
        liquidity_fraction_caps={BTC: 0.2, ETH: 0.2},
        funding_rates={BTC: 0.0, ETH: 0.0},
        available_margin_fraction=0.5,
    )

    assert result.accepted is True
    assert {target.instrument_id for target in result.targets} == {BTC, ETH}
    assert {position.instrument_id for position in positions.all()} == {BTC, ETH}
    assert {
        target.instrument_id for target in repository.latest_targets(portfolio_id="active_income")
    } == {BTC, ETH}


def test_database_product_cycle_claims_risk_approved_work_and_persists_fills(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    repository = SqlPortfolioRepository(database.engine)
    repository.save_forecast(forecast(strategy_id="btc-trend", instrument_id=BTC))
    repository.save_forecast(forecast(strategy_id="eth-trend", instrument_id=ETH))
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    prices: dict[str, float] = {}
    execution = ExecutionService(
        paper_exchange=PaperExchange(
            order_manager=order_manager,
            position_manager=positions,
            price_source=prices.__getitem__,
            fee_asset="USDT",
        ),
        positions=positions,
        trace_store=SqlDecisionTraceStore(database.engine),
    )
    supervisor = DatabaseProductSupervisor(
        repository=repository,
        active_income=ActiveIncomeProductSupervisor(
            portfolio=ActiveIncomePortfolio(
                PortfolioConstraints(
                    portfolio_id="active-income-portfolio",
                    equity=1,
                    max_positions=5,
                    max_gross_fraction=0.5,
                )
            ),
            execution_service=execution,
        ),
        btc_accumulation=BtcAccumulationProductSupervisor(execution_service=execution),
    )
    risk_store = SqlRiskDecisionStore(database.engine)
    assessment = accepted_risk_assessment()
    for decision in (*assessment.decisions, assessment.aggregate):
        risk_store.append(decision)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-optiplex:product-supervisor",
        node_id="linux-optiplex",
        role="product-supervisor",
        capabilities=("active_income_cycle",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="active-cycle-1",
        name="active_income_cycle",
        payload={
            "event_id": "portfolio-cycle-queued",
            "evaluated_at": NOW,
            "valid_until": LATER,
            "risk_assessment_id": assessment.aggregate.decision_id,
            "equity": 10_000,
            "prices": {BTC: 100_000, ETH: 5_000},
            "correlations": {BTC: {ETH: 0.2}},
            "beta_by_instrument": {BTC: 0.2, ETH: 0.1},
        },
        available_at=NOW,
    )
    paused = {"value": True}
    worker = DatabaseProductCycleWorker(
        queue=queue,
        worker_id="linux-optiplex:product-supervisor",
        supervisor=supervisor,
        risk_store=risk_store,
        update_prices=prices.update,
        is_paused=lambda target: target == "global" and paused["value"],
    )

    assert worker.run_once(now=NOW)["reason_code"] == "product_supervisor_paused"
    paused["value"] = False
    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "product_cycle_completed"
    assert result["fills"] == 2
    assert {item.instrument_id for item in positions.all()} == {BTC, ETH}
    assert worker.run_once(now=NOW)["reason_code"] == "product_cycle_queue_empty"
    assert len(OrderManager(SqlOrderStore(database.engine)).all_fills()) == 2
    assert {
        position.instrument_id
        for position in PositionManager(SqlPositionStore(database.engine)).all()
    } == {BTC, ETH}


def test_split_product_portfolio_execution_and_paper_workers_run_end_to_end(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    repository = SqlPortfolioRepository(database.engine)
    group_metadata = {"order_group_key": "btc-eth-pair", "recovery_policy": "unwind"}
    repository.save_forecast(
        forecast(strategy_id="btc-trend", instrument_id=BTC, metadata=group_metadata)
    )
    repository.save_forecast(
        forecast(strategy_id="eth-trend", instrument_id=ETH, metadata=group_metadata)
    )
    risk_store = SqlRiskDecisionStore(database.engine)
    assessment = accepted_risk_assessment()
    for decision in (*assessment.decisions, assessment.aggregate):
        risk_store.append(decision)
    for worker_id, role, capabilities in (
        (
            "linux-optiplex:product-supervisor",
            "product-supervisor",
            ("active_income_cycle", "btc_accumulation_cycle"),
        ),
        (
            "linux-optiplex:portfolio-engine",
            "portfolio-engine",
            ("active_income_portfolio", "btc_accumulation_portfolio"),
        ),
        ("linux-optiplex:execution-engine", "execution-engine", ("execute_targets",)),
        ("linux-optiplex:paper-engine", "paper-engine", ("paper_order_submit",)),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex",
            role=role,
            capabilities=capabilities,
            observed_at=NOW,
        )
    queue.enqueue(
        job_id="active-cycle-split",
        name="active_income_cycle",
        payload={
            "event_id": "portfolio-cycle-split",
            "evaluated_at": NOW,
            "valid_until": LATER,
            "risk_assessment_id": assessment.aggregate.decision_id,
            "equity": 10_000,
            "prices": {BTC: 100_000, ETH: 5_000},
            "correlations": {BTC: {ETH: 0.2}},
            "beta_by_instrument": {BTC: 0.2, ETH: 0.1},
        },
        available_at=NOW,
    )
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    traces = SqlDecisionTraceStore(database.engine)
    order_groups = OrderGroupManager(SqlOrderGroupStore(database.engine))
    coordinator = DatabaseProductCoordinator(
        queue=queue, worker_id="linux-optiplex:product-supervisor"
    )
    portfolio_worker = DatabasePortfolioWorker(
        queue=queue,
        worker_id="linux-optiplex:portfolio-engine",
        repository=repository,
        positions=positions,
        active_income=ActiveIncomePortfolio(
            PortfolioConstraints(
                portfolio_id="active-income-portfolio",
                equity=1,
                max_positions=5,
                max_gross_fraction=0.5,
            )
        ),
        risk_store=risk_store,
        trace_store=traces,
        execution_modes={"active_income": "paper", "btc_accumulation": "paper"},
    )
    execution_worker = DatabaseExecutionWorker(
        queue=queue,
        worker_id="linux-optiplex:execution-engine",
        order_manager=order_manager,
        positions=positions,
        risk_store=risk_store,
        trace_store=traces,
        order_groups=order_groups,
        product_execution={
            "active_income": {
                "execution_mode": "paper",
                "execution_costs": {"fee_bps": 5, "slippage_bps": 2},
                "base_accounting_asset": "USDT",
            }
        },
    )
    paper_worker = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id="linux-optiplex:paper-engine",
        order_manager=order_manager,
        positions=positions,
        ledgers={
            "active_income": Ledger(
                product_id="active_income",
                accounting_asset="USDT",
                store=SqlLedgerStore(database.engine, product_id="active_income"),
            )
        },
        trace_store=traces,
        order_groups=order_groups,
    )

    assert coordinator.run_once(now=NOW)["reason_code"] == "portfolio_cycle_enqueued"
    assert portfolio_worker.run_once(now=NOW)["reason_code"] == "execution_cycle_enqueued"
    assert execution_worker.run_once(now=NOW)["reason_code"] == "paper_orders_enqueued"
    assert paper_worker.run_once(now=NOW)["reason_code"] == "paper_order_filled"
    assert paper_worker.run_once(now=NOW)["reason_code"] == "paper_order_filled"

    assert {item.instrument_id for item in positions.all()} == {BTC, ETH}
    assert len(OrderManager(SqlOrderStore(database.engine)).all_fills()) == 2
    grouped_orders = tuple(item for item in order_manager.all() if item.group_id is not None)
    assert len(grouped_orders) == 2
    order_groups.reload()
    assert order_groups.get(grouped_orders[0].group_id).status is OrderGroupStatus.ACTIVE
    assert (
        len(
            Ledger(
                product_id="active_income",
                accounting_asset="USDT",
                store=SqlLedgerStore(database.engine, product_id="active_income"),
            ).entries
        )
        == 4
    )


def test_database_paper_worker_completes_a_partial_fill_with_a_leased_follow_up(
    tmp_path: Path,
):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-optiplex:paper-engine",
        node_id="linux-optiplex",
        role="paper-engine",
        capabilities=("paper_order_submit", "paper_order_continue"),
        observed_at=NOW,
    )
    manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    order = plan_orders(
        (
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0.01,
                target_notional=1_000,
                target_fraction=0.1,
                strategy_contributions={"trend": 0.1},
                risk_budget=0.1,
                valid_until=LATER,
            ),
        ),
        current_quantities={},
        decided_at=NOW,
        prices={BTC: 100_000},
    )[0]
    manager.create(order)
    manager.persist_for_submission(order.order_id)
    queue.enqueue(
        job_id="paper-partial-1",
        name="paper_order_submit",
        payload={
            "order_id": order.order_id,
            "product_id": "active_income",
            "event_id": "event-partial-1",
            "price": 100_000,
            "execution_costs": {"fee_bps": 5, "slippage_bps": 2},
            "accounting_asset": "USDT",
            "fee_in_base": False,
            "fill_fraction": 0.5,
        },
        available_at=NOW,
    )
    worker = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id="linux-optiplex:paper-engine",
        order_manager=manager,
        positions=positions,
        ledgers={
            "active_income": Ledger(
                product_id="active_income",
                accounting_asset="USDT",
                store=SqlLedgerStore(database.engine, product_id="active_income"),
            )
        },
        trace_store=SqlDecisionTraceStore(database.engine),
    )

    first = worker.run_once(now=NOW)
    second = worker.run_once(now=NOW)

    assert first["reason_code"] == "paper_order_partially_filled"
    assert first["continuation_job_id"]
    assert second["reason_code"] == "paper_order_filled"
    assert manager.get(order.order_id).status is OrderStatus.FILLED
    assert positions.get("active_income", BTC).quantity == pytest.approx(0.01)
    assert len(manager.fills_for(order.order_id)) == 2


def test_user_stream_recovers_live_partial_fills_into_orders_positions_and_accounting(
    tmp_path: Path,
):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    worker_id = "linux-optiplex:execution-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="execution-engine",
        capabilities=("user_stream_event",),
        observed_at=NOW,
    )
    manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    order = plan_orders(
        (
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0.01,
                target_notional=1_000,
                target_fraction=0.1,
                strategy_contributions={"trend": 0.1},
                risk_budget=0.1,
                valid_until=LATER,
            ),
        ),
        current_quantities={},
        decided_at=NOW,
        prices={BTC: 100_000},
    )[0]
    manager.create(order)
    manager.persist_for_submission(order.order_id)

    def enqueue_order_event(
        *,
        job_id: str,
        trade_id: int,
        quantity: float,
        status: str,
        at: str,
        target_order=order,
        execution_type: str = "TRADE",
        fee_asset: str = "USDT",
    ):
        event = normalise_user_event(
            account_id="binance-futures-main",
            market="futures",
            payload={
                "e": "ORDER_TRADE_UPDATE",
                "E": int(dt.datetime.fromisoformat(at).timestamp() * 1_000),
                "o": {
                    "s": "BTCUSDT",
                    "c": target_order.order_id[:36],
                    "S": "BUY",
                    "x": execution_type,
                    "X": status,
                    "l": str(quantity),
                    "L": "100000",
                    "n": "0.5",
                    "N": fee_asset,
                    "t": trade_id,
                },
            },
            receive_timestamp=at,
        )
        queue.enqueue(
            job_id=job_id,
            name="user_stream_event",
            payload={
                "account_id": "binance-futures-main",
                "market": "futures",
                "event": to_primitive(event),
            },
            available_at=at,
        )

    enqueue_order_event(
        job_id="user-fill-1",
        trade_id=101,
        quantity=0.004,
        status="PARTIALLY_FILLED",
        at=NOW,
    )
    enqueue_order_event(
        job_id="user-fill-2",
        trade_id=102,
        quantity=0.006,
        status="FILLED",
        at="2026-08-13T12:00:01+00:00",
    )
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    worker = DatabaseUserStreamWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_id,
        order_manager=manager,
        positions=positions,
        ledgers={"active_income": ledger},
        trace_store=SqlDecisionTraceStore(database.engine),
        account_products={"binance-futures-main": "active_income"},
    )

    first = worker.run_once(now=NOW)
    second = worker.run_once(now="2026-08-13T12:00:01+00:00")

    assert first["order_result"]["reason_code"] == "exchange_order_partially_filled"
    assert second["order_result"]["reason_code"] == "exchange_order_filled"
    assert manager.get(order.order_id).status is OrderStatus.FILLED
    assert positions.get("active_income", BTC).quantity == pytest.approx(0.01)
    assert len(manager.fills_for(order.order_id)) == 2
    assert len(ledger.entries) == 2

    partial_order = plan_orders(
        (
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0.02,
                target_notional=2_000,
                target_fraction=0.2,
                strategy_contributions={"trend": 0.2},
                risk_budget=0.2,
                valid_until=LATER,
            ),
        ),
        current_quantities={BTC: 0.01},
        decided_at="2026-08-13T12:00:02+00:00",
        prices={BTC: 100_000},
    )[0]
    manager.create(partial_order)
    manager.persist_for_submission(partial_order.order_id)
    enqueue_order_event(
        job_id="user-fill-before-cancel",
        trade_id=103,
        quantity=0.004,
        status="PARTIALLY_FILLED",
        at="2026-08-13T12:00:02+00:00",
        target_order=partial_order,
    )
    worker.run_once(now="2026-08-13T12:00:02+00:00")
    enqueue_order_event(
        job_id="user-cancel-after-partial",
        trade_id=104,
        quantity=0,
        status="CANCELED",
        at="2026-08-13T12:00:03+00:00",
        target_order=partial_order,
        execution_type="CANCELED",
    )

    cancellation = worker.run_once(now="2026-08-13T12:00:03+00:00")

    assert cancellation["order_result"]["reason_code"] == "exchange_order_cancelled"
    assert manager.get(partial_order.order_id).status is OrderStatus.CANCELLED
    assert positions.get("active_income", BTC).quantity == pytest.approx(0.014)

    foreign_fee_order = plan_orders(
        (
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0.02,
                target_notional=2_000,
                target_fraction=0.2,
                strategy_contributions={"trend": 0.2},
                risk_budget=0.2,
                valid_until=LATER,
            ),
        ),
        current_quantities={BTC: 0.014},
        decided_at="2026-08-13T12:00:04+00:00",
        prices={BTC: 100_000},
    )[0]
    manager.create(foreign_fee_order)
    manager.persist_for_submission(foreign_fee_order.order_id)
    enqueue_order_event(
        job_id="user-foreign-fee-fill",
        trade_id=105,
        quantity=0.006,
        status="FILLED",
        at="2026-08-13T12:00:04+00:00",
        target_order=foreign_fee_order,
        fee_asset="BNB",
    )

    foreign_fee = worker.run_once(now="2026-08-13T12:00:04+00:00")

    assert foreign_fee["order_result"]["reason_code"] == "fee_conversion_required"
    assert foreign_fee["order_result"]["recovery_job_id"]
    assert manager.get(foreign_fee_order.order_id).status is OrderStatus.RECOVERY_REQUIRED
    assert manager.fills_for(foreign_fee_order.order_id) == ()
    assert positions.get("active_income", BTC).quantity == pytest.approx(0.014)
    assert len(ledger.entries) == 3


def test_universe_is_point_in_time_dynamic_and_has_exact_exclusion_reasons(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = SqlUniverseStore(database.engine)
    policy = UniverseEligibilityPolicy()

    def observation(index: int, *, status: str = "trading", volume: float = 20_000_000):
        instrument = Instrument(
            venue="binance",
            market_type=MarketType.FUTURES,
            base_asset=f"COIN{index}",
            quote_asset="USDT",
            settlement_asset="USDT",
            exchange_symbol=f"COIN{index}USDT",
            price_precision=4,
            quantity_precision=2,
            minimum_quantity=0.01,
            minimum_notional=5,
            status=status,
        )
        return InstrumentObservation(
            instrument=instrument,
            listing_age_days=365,
            quote_volume=volume,
            trade_count=50_000,
            spread_bps=2,
            open_interest=5_000_000,
            funding_rate=0.0001,
            realised_volatility=0.5,
            depth_notional=500_000,
            data_completeness=1.0,
            strategy_eligibility=("directional", "cross_sectional"),
        )

    first = tuple(observation(index) for index in range(30)) + (observation(99, volume=100),)
    first_snapshot = store.record_snapshot(
        universe_id="binance-usdt-perpetuals",
        observed_at=NOW,
        observations=first,
        policy=policy,
    )
    second = (observation(0, status="settled"), *first[1:])
    store.record_snapshot(
        universe_id="binance-usdt-perpetuals",
        observed_at=LATER,
        observations=second,
        policy=policy,
    )

    historical = store.members_at(
        universe_id="binance-usdt-perpetuals",
        observed_at="2026-08-13T12:30:00+00:00",
        eligible_only=False,
    )
    current = store.members_at(
        universe_id="binance-usdt-perpetuals",
        observed_at=LATER,
        eligible_only=False,
    )

    assert historical[0].snapshot_id == first_snapshot
    assert sum(item.eligible for item in historical) == 30
    assert (
        next(item for item in historical if item.instrument.base_asset == "COIN99").reason_code
        == "quote_volume_too_low"
    )
    assert next(item for item in historical if item.instrument.base_asset == "COIN0").eligible
    assert (
        next(item for item in current if item.instrument.base_asset == "COIN0").reason_code
        == "listing_not_trading"
    )


def test_binance_events_enter_the_canonical_immutable_parquet_path(tmp_path: Path):
    public = normalise_public_event(
        market="futures",
        stream="btcusdt@aggTrade",
        payload={"e": "aggTrade", "E": 1_786_622_400_000, "s": "BTCUSDT", "a": 42},
        receive_timestamp=NOW,
    )
    user = normalise_user_event(
        account_id="binance-futures-main",
        market="futures",
        payload={
            "e": "ORDER_TRADE_UPDATE",
            "E": 1_786_622_400_000,
            "o": {"s": "BTCUSDT", "x": "TRADE"},
        },
        receive_timestamp=NOW,
    )
    assert public.event_type is MarketEventType.AGGREGATE_TRADE
    assert user.event_type is MarketEventType.FILL_UPDATE
    mark = normalise_public_event(
        market="futures",
        stream="btcusdt@markPrice@1s",
        payload={
            "e": "markPriceUpdate",
            "E": 1_786_622_400_000,
            "s": "BTCUSDT",
            "p": "100000",
            "r": "0.0001",
            "T": 1_786_651_200_000,
        },
        receive_timestamp=NOW,
    )
    funding = funding_event_from_mark_price(mark)
    assert funding is not None
    assert funding.event_type is MarketEventType.FUNDING_RATE
    assert funding.payload["funding_rate"] == pytest.approx(0.0001)

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-data",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="market-event-42",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(public),
        },
        available_at=NOW,
    )
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
    )

    result = writer.run_once(now=NOW)
    path = Path(result["path"])

    assert result["reason_code"] == "market_event_written"
    assert "raw/binance/futures/aggregate_trade/BTCUSDT/date=2026-08-13" in str(path)
    assert pq.read_table(path).to_pylist()[0]["event_id"].startswith("sha256:")


def test_closed_candle_persistence_enqueues_and_builds_live_features(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-data",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
    )
    queue.register_worker(
        worker_id="linux-feature",
        node_id="linux-optiplex",
        role="feature-service",
        capabilities=("live_feature_calculation",),
        observed_at=NOW,
    )
    close_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": "100",
                "h": "103",
                "l": "99",
                "c": "102",
                "v": "25",
                "x": True,
            },
        },
        receive_timestamp=NOW,
    )
    queue.enqueue(
        job_id="closed-candle-1",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=NOW,
    )
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
    )
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=SqlFeatureStore(database.engine),
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
    )

    written = writer.run_once(now=NOW)
    featured = feature_worker.run_once(now=NOW)

    assert written["feature_job_id"].startswith("live-feature:")
    assert "/bars/binance/futures/BTCUSDT/1m/" in written["bar_path"]
    assert pq.read_table(written["bar_path"]).to_pylist()[0]["close"] == pytest.approx(102)
    assert featured["reason_code"] == "features_persisted"
    assert featured["features"] == 3
    assert (
        len(
            SqlFeatureStore(database.engine).available(
                instrument_id=BTC,
                at=NOW,
                feature_set_version="core-bars-v1",
            )
        )
        == 3
    )


def test_market_gap_repair_uses_bounded_rest_identity() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, object]]:
            return [
                {
                    "a": 2,
                    "p": "101",
                    "q": "0.1",
                    "f": 10,
                    "l": 10,
                    "T": 1_800_000_001_000,
                    "m": False,
                    "M": True,
                },
                {
                    "a": 3,
                    "p": "102",
                    "q": "0.2",
                    "f": 11,
                    "l": 11,
                    "T": 1_800_000_002_000,
                    "m": True,
                    "M": True,
                },
            ]

    class Session:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get(self, url: str, *, params: dict[str, object], timeout: float) -> Response:
            self.calls.append({"url": url, "params": params, "timeout": timeout})
            return Response()

    session = Session()
    repair = BinanceMarketGapRepair(
        session=session,
        base_urls={"spot": "https://spot.test", "futures": "https://futures.test"},
        clock_ms=lambda: 1_800_000_010_000,
    )

    events = repair(
        {
            "market": "spot",
            "symbol": "BTCUSDT",
            "event_type": "aggregate_trade",
            "previous_sequence": 1,
            "sequence": 4,
            "observed_at": NOW,
        }
    )

    assert [event.sequence for event in events] == [2, 3]
    assert session.calls == [
        {
            "url": "https://spot.test/api/v3/aggTrades",
            "params": {"symbol": "BTCUSDT", "limit": 1_000, "fromId": 2, "toId": 3},
            "timeout": 10.0,
        }
    ]


def test_market_data_writer_persists_a_sequence_gap_repair(tmp_path: Path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'gap.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-data",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("market_event_write", "market_data_gap_recovery"),
        observed_at=NOW,
    )
    def depth(sequence: int) -> MarketEvent:
        return normalise_public_event(
            market="futures",
            stream="btcusdt@depth20@100ms",
            payload={
                "e": "depthUpdate",
                "E": 1_800_000_000_000 + sequence,
                "s": "BTCUSDT",
                "U": sequence,
                "u": sequence,
                "b": [["100", "1"]],
                "a": [["101", "1"]],
            },
            receive_timestamp=NOW,
        )

    repaired = depth(2)
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
        gap_repair=lambda _payload: (repaired,),
    )
    for sequence in (1, 3):
        event = depth(sequence)
        queue.enqueue(
            job_id=f"depth-{sequence}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": "futures",
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=NOW,
        )

        result = writer.run_once(now=NOW)
        if sequence == 1:
            assert result["sequence_status"] == "ok"
        else:
            assert result["sequence_status"] == "gap"
            assert result["gap_recovery_job_id"].startswith("market-gap:")

    recovery = writer.run_once(now=NOW)

    assert recovery["reason_code"] == "market_gap_recovered"
    assert recovery["repaired_events"] == 1
    database.dispose()


def test_live_and_historical_feature_workers_produce_the_same_values(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    for worker_id, role, capability in (
        ("linux-data", "data-writer", "market_event_write"),
        ("linux-feature", "feature-worker", "live_feature_calculation"),
        ("mac-feature", "feature-worker", "historical_feature_calculation"),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex" if worker_id.startswith("linux") else "macbook-research",
            role=role,
            capabilities=(capability,),
            observed_at=NOW,
        )
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
            "s": "BTCUSDT",
            "k": {
                "t": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 59_999,
                "T": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1,
                "i": "1m",
                "o": "100",
                "h": "105",
                "l": "99",
                "c": "104",
                "v": "50",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id="feature:live",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=NOW,
    )
    queue.enqueue(
        job_id="feature:historical",
        name="historical_feature_calculation",
        payload={
            "feature_set_version": "core-bars-v1",
            "instrument_id": BTC,
            "source_event_time": "2026-08-13T11:59:00+00:00",
            "source_close_time": "2026-08-13T11:59:59+00:00",
            "availability_time": NOW,
            "inputs": {"open": 100, "high": 105, "low": 99, "close": 104, "volume": 50},
        },
        available_at=NOW,
    )
    assert (
        DatabaseMarketDataWriter(
            queue=queue,
            worker_id="linux-data",
            root=tmp_path / "data",
        ).run_once(now=NOW)["reason_code"]
        == "market_event_written"
    )
    store = SqlFeatureStore(database.engine)
    live = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=store,
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
    ).run_once(now=NOW)
    historical = DatabaseFeatureWorker(
        queue=queue,
        worker_id="mac-feature",
        store=store,
        job_names=("historical_feature_calculation",),
    ).run_once(now=NOW)

    assert live["feature_ids"] == historical["feature_ids"]
    assert (
        len(store.available(instrument_id=BTC, at=LATER, feature_set_version="core-bars-v1")) == 3
    )


def test_live_feature_jobs_reject_scalar_inputs(tmp_path: Path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-feature",
        node_id="linux-optiplex",
        role="feature-worker",
        capabilities=("live_feature_calculation",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="feature:scalar-live",
        name="live_feature_calculation",
        payload={
            "feature_set_version": "core-bars-v1",
            "instrument_id": BTC,
            "source_event_time": NOW,
            "source_close_time": NOW,
            "availability_time": NOW,
            "inputs": {"open": 100, "high": 105, "low": 99, "close": 104, "volume": 50},
        },
        available_at=NOW,
    )
    result = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=SqlFeatureStore(database.engine),
        job_names=("live_feature_calculation",),
    ).run_once(now=NOW)
    assert result["reason_code"] == "feature_calculation_failed"
    assert result["error"] == "live feature jobs require immutable input references"


def test_live_feature_reference_must_select_the_declared_source_candle(tmp_path: Path) -> None:
    from src.data.parquet_store import PartitionedBarStore

    first = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
            "s": "BTCUSDT",
            "k": {
                "t": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 59_999,
                "T": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
                "i": "1m",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100",
                "v": "10",
                "x": True,
            },
        },
    )
    later = "2026-08-24T00:01:00+00:00"
    later_ms = int(dt.datetime.fromisoformat(later).timestamp() * 1_000)
    second = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=later,
        payload={
            "e": "kline",
            "E": later_ms,
            "s": "BTCUSDT",
            "k": {
                "t": later_ms - 59_999,
                "T": later_ms,
                "i": "1m",
                "o": "100",
                "h": "103",
                "l": "99",
                "c": "102",
                "v": "12",
                "x": True,
            },
        },
    )
    root = tmp_path / "data"
    bars = PartitionedBarStore(root)
    bars.put(first, venue="binance", market="futures", symbol="BTCUSDT")
    bars.put(second, venue="binance", market="futures", symbol="BTCUSDT")
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'feature.sqlite3'}")
    database.create_schema()
    worker = DatabaseFeatureWorker(
        queue=DatabaseJobQueue(database.engine),
        worker_id="feature-worker",
        store=SqlFeatureStore(database.engine),
        job_names=("live_feature_calculation",),
        parquet_root=root,
    )
    payload = {
        "instrument_id": first.instrument_id,
        "source_event_time": dt.datetime.fromtimestamp(
            float(first.payload["data"]["k"]["t"]) / 1_000, dt.UTC
        ).isoformat(),
        "source_close_time": first.close_timestamp,
        "availability_time": second.availability_timestamp,
        "input_references": {
            "bar_window": {
                "kind": "partitioned_bar_window",
                "relative_pattern": "bars/binance/futures/BTCUSDT/1m/**/*.parquet",
                "through_close_time": second.close_timestamp,
                "minimum_history": 1,
                "source_event_ids": [second.event_id],
            }
        },
    }
    reference = payload["input_references"]["bar_window"]
    reference["content_hash"] = canonical_hash(
        {key: value for key, value in reference.items() if key != "content_hash"}
    )
    with pytest.raises(ValueError, match="does not match the source event timestamps"):
        worker._resolve_input_references(payload)


def test_duckdb_historical_query_materialises_safe_parquet_results(tmp_path: Path):
    pytest.importorskip("duckdb")
    import pandas as pd

    root = tmp_path / "data"
    partition = root / "bars" / "symbol=BTCUSDT"
    partition.mkdir(parents=True)
    pd.DataFrame({"close": [100.0, 101.0], "volume": [10.0, 20.0]}).to_parquet(
        partition / "part.parquet"
    )
    query = DuckDBHistoricalQuery(root)

    frame = query.query_frame(
        relative_pattern="bars/**/*.parquet",
        columns=("close",),
        where_sql="close > ?",
        parameters=(100,),
    )

    assert frame.to_dict("records") == [{"close": 101.0}]
    with pytest.raises(ValueError, match="escapes"):
        query.query_arrow(relative_pattern="../outside/*.parquet")


def test_agent_can_propose_python_strategy_and_required_tests_without_execution_access(
    tmp_path: Path,
):
    proposal = AgentProposal(
        proposal_id="agent-strategy-1",
        role=AgentRole.IMPLEMENTER,
        action=AgentAction.CREATE_PYTHON_STRATEGY,
        product_id="active_income",
        created_at=NOW,
        thesis="A bounded deterministic trend strategy.",
        files={
            "src/strategies/library/agent_trend.py": (
                "from src.strategies.base import Strategy\n"
                "class AgentTrend(Strategy):\n"
                "    name = 'agent_trend'\n"
                "    def generate_signals(self, df):\n"
                "        return self._empty_signals(df)\n"
            ),
            "tests/test_agent_trend.py": (
                "def test_deterministic(): pass\n"
                "def test_no_lookahead(): pass\n"
                "def test_signal_domain(): pass\n"
                "def test_synthetic_signal(): pass\n"
                "def test_cost_adjusted_backtest(): pass\n"
            ),
        },
        research_jobs=({"name": "bounded_backtest", "maximum_seconds": 60},),
    )

    class PassingRunner:
        def run(self, argv):
            return CommandResult(tuple(argv), 0, "passed", "")

    outcome = AgentCodeReviewer().review(
        proposal=proposal,
        workspace=tmp_path,
        runner=PassingRunner(),
    )
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = SqlAgentStore(database.engine)
    store.save_proposal(proposal)
    store.save_review(
        proposal_id=proposal.proposal_id,
        created_at=NOW,
        payload={"accepted": outcome.accepted},
    )

    assert outcome.accepted is True
    assert len(store.records("proposal")) == 1
    assert store.records("review")[0]["accepted"] is True
    with pytest.raises(ValueError, match="forbidden execution marker"):
        AgentProposal(
            proposal_id="unsafe-agent-strategy",
            role=AgentRole.IMPLEMENTER,
            action=AgentAction.CREATE_PYTHON_STRATEGY,
            product_id="active_income",
            created_at=NOW,
            thesis="Try to bypass the execution boundary.",
            files={"src/strategies/library/unsafe.py": "from src.execution import Broker\n"},
        )


def test_openclaw_context_excludes_holdout_and_agent_jobs_are_bounded(tmp_path: Path):
    context = AgentContext(
        created_at=NOW,
        values={
            "strategy_catalogue": ["sma_cross"],
            "development_results": {"sma_cross": {"accepted": True}},
            "resource_budget": {"maximum_seconds": 60},
            "public_market_summaries": {"BTCUSDT": {"daily_return": 0.01}},
        },
    )
    assert context.content_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="forbidden key"):
        AgentContext(created_at=NOW, values={"development_results": {"holdout_sharpe": 3}})

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    bridge = OpenClawAgentBridge(store=SqlAgentStore(database.engine), queue=queue)
    proposal = bridge.ingest(
        {
            "schema": "openclaw.agent_proposal/v1",
            "source": "openclaw",
            "proposal_id": "openclaw-thesis-1",
            "role": "researcher",
            "action": "create_thesis",
            "product_id": "active_income",
            "created_at": NOW,
            "thesis": "Test cross-sectional residual momentum.",
        }
    )
    queue.register_worker(
        worker_id="mac-agent",
        node_id="macbook-research",
        role="agent-sandbox",
        capabilities=("agent_research",),
        observed_at=NOW,
    )
    claimed = queue.claim(
        worker_id="mac-agent",
        now=NOW,
        lease_seconds=60,
        names=("agent_research",),
    )

    assert proposal.action is AgentAction.CREATE_THESIS
    assert claimed is not None
    assert claimed.payload["agent_may_submit_orders"] is False


def test_sql_stops_groups_and_recovery_survive_service_restart(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    stop_store = SqlStopStore(database.engine)
    stop_manager = StopManager(stop_store)
    stop_manager.create(
        ProtectiveStop(
            stop_id="sql-stop-1",
            portfolio_id="active_income",
            instrument_id=BTC,
            exit_side=OrderSide.SELL,
            quantity=0.01,
            trigger_price=95_000,
            created_at=NOW,
        )
    )
    group_plan = plan_order_group(
        (
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=BTC,
                target_quantity=0.01,
                target_notional=1_000,
                target_fraction=0.1,
                strategy_contributions={"pair": 0.1},
                risk_budget=0.1,
                valid_until=LATER,
            ),
            TargetPosition(
                portfolio_id="active_income",
                instrument_id=ETH,
                target_quantity=-0.2,
                target_notional=-1_000,
                target_fraction=-0.1,
                strategy_contributions={"pair": -0.1},
                risk_budget=0.1,
                valid_until=LATER,
            ),
        ),
        current_quantities={},
        decided_at=NOW,
    )
    group_manager = OrderGroupManager(SqlOrderGroupStore(database.engine))
    group_manager.create(group_plan.group)
    group_manager.transition(group_plan.group.group_id, OrderGroupStatus.PRIMARY_SUBMITTED)
    recovery = reconcile_account(
        local_positions={BTC: 0.01},
        exchange_positions={BTC: 0.02},
        local_open_order_ids=set(),
        exchange_open_order_ids=set(),
    )
    plan_recovery(
        recovery,
        created_at=NOW,
        store=SqlRecoveryStore(database.engine),
    )

    assert StopManager(SqlStopStore(database.engine)).active()[0].stop_id == "sql-stop-1"
    assert (
        OrderGroupManager(SqlOrderGroupStore(database.engine)).get(group_plan.group.group_id).status
        is OrderGroupStatus.PRIMARY_SUBMITTED
    )
    assert SqlRecoveryStore(database.engine).read()[0].reason_code == "exchange_state_mismatch"


def test_event_replay_models_queue_partial_fill_cancel_latency_and_connection_gaps():
    events = (
        ReplayEvent(
            event_time=NOW,
            receive_time=NOW,
            instrument_id=BTC,
            best_bid=99,
            best_ask=100,
            bid_depth=5,
            ask_depth=0.5,
            traded_at_ask=0.2,
            mark_price=100,
        ),
        ReplayEvent(
            event_time="2026-08-13T12:00:01+00:00",
            receive_time="2026-08-13T12:00:01+00:00",
            instrument_id=BTC,
            best_bid=100,
            best_ask=101,
            bid_depth=5,
            ask_depth=0.3,
            traded_at_ask=0.4,
            mark_price=101,
            connected=False,
        ),
        ReplayEvent(
            event_time="2026-08-13T12:00:02+00:00",
            receive_time="2026-08-13T12:00:02+00:00",
            instrument_id=BTC,
            best_bid=99,
            best_ask=100,
            bid_depth=5,
            ask_depth=0.3,
            traded_at_ask=0.5,
            mark_price=99,
        ),
    )
    order = SimulatedLimitOrder(
        order_id="event-order-1",
        instrument_id=BTC,
        side=SimulatedOrderSide.BUY,
        quantity=1.0,
        limit_price=100.5,
        submitted_at=NOW,
        expires_at="2026-08-13T12:01:00+00:00",
        cancel_requested_at="2026-08-13T12:00:02+00:00",
        queue_ahead_quantity=0.1,
    )

    result = EventReplayEngine(cancel_latency_seconds=0.25).simulate(
        events=events,
        orders=(order,),
    )

    assert result.orders[0].status is SimulatedOrderStatus.PARTIAL
    assert result.orders[0].remaining_quantity == pytest.approx(0.2)
    assert len(result.orders[0].fills) == 2
    assert result.connection_gaps == ("2026-08-13T12:00:01+00:00",)
    assert result.orders[0].fills[0].adverse_selection < 0


def test_event_replay_applies_funding_after_the_position_fill():
    events = (
        ReplayEvent(
            event_time=NOW,
            receive_time=NOW,
            instrument_id=BTC,
            best_bid=99,
            best_ask=100,
            bid_depth=1,
            ask_depth=1,
            mark_price=100,
        ),
        ReplayEvent(
            event_time=LATER,
            receive_time=LATER,
            instrument_id=BTC,
            best_bid=109,
            best_ask=110,
            bid_depth=1,
            ask_depth=1,
            mark_price=110,
            funding_rate=0.001,
        ),
    )
    order = SimulatedLimitOrder(
        order_id="funding-order-1",
        instrument_id=BTC,
        side=SimulatedOrderSide.BUY,
        quantity=1,
        limit_price=101,
        submitted_at=NOW,
        expires_at=LATER,
    )

    result = EventReplayEngine().simulate(events=events, orders=(order,))

    assert result.positions[BTC] == pytest.approx(1)
    assert result.funding_paid == pytest.approx(0.11)


def test_live_canary_promotion_requires_approval_preflight_capacity_and_evidence(tmp_path: Path):
    policy = PromotionPolicy(
        automatic_paper_promotion=True,
        automatic_live_canary_promotion=True,
        canary_capital_limit=0.01,
        required_forward_evidence_days=60,
        maximum_drawdown=0.1,
        maximum_execution_drift=0.1,
        maximum_model_drift=0.1,
    )
    evidence = PromotionEvidence(
        strategy_artefact_hash="sha256:" + "a" * 64,
        source_commit_hash="sha256:" + "b" * 64,
        validation_accepted=True,
        protected_holdout_accepted=True,
        forward_evidence_days=90,
        forward_evidence_accepted=True,
        drawdown=0.02,
        execution_drift=0.01,
        model_drift=0.01,
        portfolio_capacity=0.05,
        requested_capital=0.02,
        risk_budget_available=0.03,
        live_approval=True,
        fresh_preflight=True,
        forward_summary_id="summary-v1",
        forward_decision_id="decision-v1",
        forward_independent_decisions=60,
        forward_net_pnl=1.0,
        forward_data_uptime=1.0,
        forward_effective_trades=60,
        forward_fill_rate=1.0,
    )

    decision = decide_promotion(
        strategy_version_id="trend:v1",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=evidence,
        policy=policy,
        evaluated_at=NOW,
    )
    canary_decision = decide_promotion(
        strategy_version_id="trend:v1-canary",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=evidence,
        policy=policy,
        evaluated_at=NOW,
        requested_transition="live_canary",
    )
    blocked = decide_promotion(
        strategy_version_id="trend:v2",
        current_state=LifecycleState.FORWARD_PAPER,
        evidence=PromotionEvidence(**{**evidence.__dict__, "live_approval": False}),
        policy=policy,
        evaluated_at=NOW,
        requested_transition="live_canary",
    )
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = SqlPromotionStore(database.engine)
    store.append(decision)
    store.append(canary_decision)
    store.append(blocked)

    assert decision.next_state is LifecycleState.LIVE_READY
    assert canary_decision.next_state is LifecycleState.LIVE_CANARY
    assert canary_decision.capital_limit == pytest.approx(0.01)
    assert blocked.accepted is False
    assert blocked.reason_code == "live_approval_missing"
    assert store.latest("trend:v1") == decision

    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-promotion",
        node_id="linux-optiplex",
        role="promotion-engine",
        capabilities=("promotion_evaluation",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="promotion-job-1",
        name="promotion_evaluation",
        payload={
            "strategy_version_id": "trend:v3",
            "current_state": "registered",
            "evaluated_at": NOW,
            "policy": policy.__dict__,
            "evidence": evidence.__dict__,
        },
        available_at=NOW,
    )
    queued = DatabasePromotionWorker(
        queue=queue,
        worker_id="linux-promotion",
        store=store,
    ).run_once(now=NOW)

    assert queued["reason_code"] == "forward_paper_promoted"
    assert store.latest("trend:v3").next_state is LifecycleState.FORWARD_PAPER


def test_features_are_deterministic_availability_safe_and_identical_live_vs_history(
    tmp_path: Path,
):
    calculator = DeterministicFeatureCalculator(
        version="features-v1",
        function=lambda inputs: {
            "range": inputs["high"] - inputs["low"],
            "return": inputs["close"] / inputs["open"] - 1,
        },
    )
    historical = calculator.calculate(
        instrument_id=BTC,
        source_event_time=NOW,
        source_close_time="2026-08-13T12:01:00+00:00",
        availability_time="2026-08-13T12:01:01+00:00",
        inputs={"open": 100, "high": 105, "low": 99, "close": 104},
    )
    live = calculator.calculate(
        instrument_id=BTC,
        source_event_time=NOW,
        source_close_time="2026-08-13T12:01:00+00:00",
        availability_time="2026-08-13T12:01:01+00:00",
        inputs={"open": 100, "high": 105, "low": 99, "close": 104},
    )
    calculator.assert_live_historical_match(historical, live)
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    store = SqlFeatureStore(database.engine)
    store.save(historical)

    assert (
        len(
            store.available(
                instrument_id=BTC,
                at="2026-08-13T12:01:01+00:00",
                feature_set_version="features-v1",
            )
        )
        == 2
    )
    with pytest.raises(ValueError, match="before the source candle closes"):
        FeatureValue(
            feature_set_version="features-v1",
            feature_name="unsafe",
            instrument_id=BTC,
            source_event_time=NOW,
            source_close_time="2026-08-13T12:01:00+00:00",
            availability_time=NOW,
            value=1,
        )


def test_accounting_service_persists_nav_balances_funding_and_reconciles_costs(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    ledger.record_capital(entry_id="capital", amount=Decimal("1000"), occurred_at=NOW)
    service = AccountingService(engine=database.engine, ledgers={"active_income": ledger})
    service.record_balances(
        account_id="binance-futures-main",
        observed_at=NOW,
        balances={"USDT": 1000},
    )
    service.record_nav(
        NavSnapshot(
            product_id="active_income",
            accounting_asset="USDT",
            nav=1000,
            observed_at=NOW,
            components={"cash": 1000},
        )
    )
    service.record_funding(
        product_id="active_income",
        entry_id="funding-1",
        amount=Decimal("2"),
        occurred_at=NOW,
        attribution={
            "strategy": "carry",
            "symbol": BTC,
            "sleeve": "funding",
            "product": "active_income",
        },
    )

    assert ledger.nav() == Decimal("1002")
    assert usdt_nav(cash_balance=1000, positions={BTC: (0.1, 100_000, 101_000)}) == 1100
    assert btc_nav(btc_balance=0.5, stablecoin_balance=50_000, stablecoin_per_btc=100_000) == 1
    assert reconcile_accounting(
        ledger=ledger,
        fills=(),
        expected_funding=Decimal("2"),
    ).matched

    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-accounting",
        node_id="linux-optiplex",
        role="accounting-service",
        capabilities=("accounting_event",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="accounting-nav-2",
        name="accounting_event",
        payload={
            "kind": "nav",
            "snapshot": {
                "product_id": "active_income",
                "accounting_asset": "USDT",
                "nav": 1002,
                "observed_at": LATER,
                "components": {"cash": 1002},
            },
        },
        available_at=NOW,
    )
    recorded = DatabaseAccountingWorker(
        queue=queue,
        worker_id="linux-accounting",
        service=service,
    ).run_once(now=NOW)

    assert recorded["reason_code"] == "accounting_event_recorded"
    assert recorded["kind"] == "nav"


def test_observability_reports_all_domains_metrics_and_exact_health_causes(tmp_path: Path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    heartbeat_store.record(
        service_name="product-supervisor",
        node_id="linux-optiplex",
        observed_at=NOW,
        healthy=True,
        payload={"reason_code": "cycle_completed"},
    )
    SqlDecisionTraceStore(database.engine).append(
        DecisionTrace.start(event_id="blocked", instrument_id=BTC, evaluated_at=NOW).block(
            DecisionTraceStage.DATA_AVAILABLE,
            reason_code="market_data_stale",
        )
    )
    btc_ledger = Ledger(
        product_id="btc_accumulation",
        accounting_asset="BTC",
        store=SqlLedgerStore(database.engine, product_id="btc_accumulation"),
    )
    btc_ledger.record_capital(entry_id="btc-capital", amount=Decimal("1"), occurred_at=NOW)
    AccountingService(engine=database.engine, ledgers={"btc_accumulation": btc_ledger}).record_nav(
        NavSnapshot(
            product_id="btc_accumulation",
            accounting_asset="BTC",
            nav=1,
            observed_at=NOW,
            components={
                "btc_balance": 0.8,
                "stablecoin_balance": 20_000,
                "stablecoin_per_btc": 100_000,
                "regime": "trend",
            },
            passive_benchmark_nav=1,
        )
    )
    report = DatabasePlatformReport(database.engine).build()
    health = assess_platform_health(
        config=load_platform_config(),
        store=heartbeat_store,
        now=NOW,
        maximum_age_seconds=60,
    )
    metrics = MetricsRegistry()
    metrics.set_gauge("platform_healthy", float(health.healthy))
    metrics.increment("decision_blocked_total", stage="data_available")

    assert set(report) == {"schema", "trading", "research", "products", "operations"}
    assert report["operations"]["decision_funnel_blocked"] == {
        "data_available:market_data_stale": 1
    }
    assert report["products"]["btc_accumulation"]["btc_performance"]["btc_balance"] == 0.8
    assert health.healthy is False
    assert health.reason_code == "service_heartbeats_missing"
    assert 'decision_blocked_total{stage="data_available"} 1' in metrics.render()
    report_worker = DatabaseReportWorker(engine=database.engine, root=tmp_path / "reports")
    written = report_worker.run_once(now=NOW)
    repeated = report_worker.run_once(now=NOW)
    assert written == repeated
    assert Path(written["path"]).is_file()
