"""Executable lifecycle for every assigned platform service."""

from __future__ import annotations

import argparse
import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.agents.code_worker import AgentCodeWorkflow
from src.agents.sandbox import SandboxPolicy
from src.agents.store import SqlAgentStore
from src.autopilot.event_capture import load_event_capture_config
from src.data.database import PlatformDatabase, active_strategy_assignment, strategy_artefact
from src.data.feature_store import SqlFeatureStore
from src.data.parquet_store import PartitionedBacktestStore
from src.data.universe import SqlUniverseStore
from src.domain._codec import timestamp
from src.execution.order_groups import OrderGroupManager, SqlOrderGroupStore
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.recovery import SqlRecoveryStore
from src.execution.stops import SqlStopStore, StopManager
from src.observability.decision_trace import SqlDecisionTraceStore
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.research.datasets import CanonicalDatasetResolver, SqlCanonicalDatasetRepository
from src.research.evaluation import EvidencePolicy
from src.research.ml import MlExperimentRunner, ModelArtefactStore, SqlModelArtefactStore
from src.research.store import SqlResearchStore
from src.risk.engine import SqlRiskDecisionStore, SqlRiskPolicyStore, SqlRiskSnapshotStore
from src.risk.policies import install_product_risk_policies
from src.services.account_reconciliation import AccountReconciliationService
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.agent_worker import DatabaseAgentJobHandlers
from src.services.alerting import SqlAlertService, configured_alert_service
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.services.config import load_platform_config, load_split_configuration
from src.services.control_api import DatabaseControlPlane, build_control_server
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.emergency import DatabaseEmergencyFlattenWorker
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.forward_observation import (
    DatabaseForwardObservationWorker,
    DatabaseForwardSummaryWorker,
)
from src.services.health import DatabaseHeartbeatStore
from src.services.heavy_compute import HeavyComputeLeaseStore
from src.services.live_execution import ApprovedLiveExecution
from src.services.market_gateway import DatabaseMarketGateway, UserStreamAccount
from src.services.order_execution import (
    DatabaseExecutionWorker,
    DatabaseLiveExecutionWorker,
    DatabasePaperExecutionWorker,
    DatabaseUserStreamWorker,
)
from src.services.order_recovery import DatabaseLiveRecoveryWorker
from src.services.paper_diagnostic import DatabaseDiagnosticPaperWorker
from src.services.platform_bootstrap import PlatformBootstrap
from src.services.portfolio_engine import (
    DatabasePortfolioTargetBuilder,
    DatabasePortfolioTargetWorker,
    DatabaseProductCoordinator,
)
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.portfolio_state import (
    DatabasePortfolioSourceService,
    DatabasePortfolioStateWorker,
    portfolio_state_policies,
)
from src.services.promotion import (
    DatabasePromotionWorker,
    PromotionPolicy,
    SqlPromotionPolicyStore,
    SqlPromotionStore,
)
from src.services.protective_stops import LiveProtectiveStopService
from src.services.report_worker import DatabaseReportWorker
from src.services.research_jobs import DatabaseResearchJobHandlers
from src.services.research_worker import ResearchWorker
from src.services.risk_service import DatabaseRiskWorker
from src.services.runtime import ServiceRuntime, utc_now
from src.services.scheduler import DatabaseJobQueue, PlatformScheduler
from src.services.strategy_evaluator import DatabaseStrategyEvaluator
from src.services.universe_service import DatabaseUniverseService


def _parse_bind(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise ValueError("control API bind must use host:port")
    port = int(port_text)
    if not 1 <= port <= 65_535:
        raise ValueError("control API port must be in [1, 65535]")
    return host, port


def _idle_cycle(service_name: str) -> dict[str, Any]:
    return {"reason_code": "service_waiting_for_input", "service": service_name}


def _by_id(
    payload: Mapping[str, Any], *, collection: str, identity: str
) -> dict[str, dict[str, Any]]:
    records = payload.get(collection)
    if not isinstance(records, list):
        raise ValueError(f"{collection} must be a list")
    return {str(record[identity]): dict(record) for record in records}


def _product_coordination_cycle(
    *, database: PlatformDatabase, node_id: str
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:product-supervisor"
    capabilities = ("active_income_cycle", "btc_accumulation_cycle")
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="product-supervisor",
        capabilities=capabilities,
        observed_at=utc_now(),
    )
    worker = DatabaseProductCoordinator(queue=queue, worker_id=worker_id)
    return lambda: worker.run_once(now=utc_now())


def _execution_components(database: PlatformDatabase):
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    traces = SqlDecisionTraceStore(database.engine)
    return order_manager, positions, traces


def _portfolio_cycle(
    *,
    database: PlatformDatabase,
    configuration: Mapping[str, Mapping[str, Any]],
    node_id: str,
) -> Callable[[], dict[str, Any]]:
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    risk = configuration["risk"]
    install_product_risk_policies(
        SqlRiskPolicyStore(database.engine),
        risk_configuration=risk,
        products=products,
    )
    _, positions, _ = _execution_components(database)
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:portfolio-engine"
    capabilities = (
        "active_income_portfolio",
        "btc_accumulation_portfolio",
        "portfolio_target_build",
    )
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="portfolio-engine",
        capabilities=capabilities,
        observed_at=utc_now(),
    )
    repository = SqlPortfolioRepository(database.engine)
    snapshot_store = SqlRiskSnapshotStore(database.engine)
    target_worker = DatabasePortfolioTargetWorker(
        queue=queue,
        worker_id=worker_id,
        build_target=DatabasePortfolioTargetBuilder(
            repository=repository,
            snapshot_store=snapshot_store,
            positions=positions,
            product_configuration=products,
            account_configuration={
                str(account["account_id"]): dict(account)
                for account in configuration["accounts"]["accounts"]
            },
        ),
    )

    def run_once() -> dict[str, Any]:
        return target_worker.run_once(now=utc_now())

    return run_once


def _portfolio_state_cycle(
    *,
    database: PlatformDatabase,
    configuration: Mapping[str, Mapping[str, Any]],
    node_id: str,
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:portfolio-state-service"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="portfolio-state-service",
        capabilities=("portfolio_state_publish",),
        observed_at=utc_now(),
    )
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    accounts = _by_id(configuration["accounts"], collection="accounts", identity="account_id")
    source_service = DatabasePortfolioSourceService(
        engine=database.engine,
        store=SqlRiskSnapshotStore(database.engine),
        products=products,
        accounts=accounts,
    )
    worker = DatabasePortfolioStateWorker(
        queue=queue,
        worker_id=worker_id,
        store=SqlRiskSnapshotStore(database.engine),
        refresh_sources=source_service.refresh,
    )
    policies = portfolio_state_policies(configuration, products)

    def run_once() -> dict[str, Any]:
        now = utc_now()
        result = worker.run_once(now=now)
        if result.get("reason_code") != "portfolio_state_queue_empty":
            return result
        scheduled = worker.schedule_from_latest(
            products=products,
            state_policies=policies,
            now=now,
        )
        if not scheduled:
            return {
                "reason_code": worker.last_schedule_reason,
                "products": len(products),
            }
        return worker.run_once(now=now)

    return run_once


def _execution_cycle(
    *,
    database: PlatformDatabase,
    configuration: Mapping[str, Mapping[str, Any]],
    node_id: str,
    alerts: SqlAlertService | None = None,
    control_plane: DatabaseControlPlane | None = None,
) -> Callable[[], dict[str, Any]]:
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    order_manager, positions, traces = _execution_components(database)
    order_groups = OrderGroupManager(SqlOrderGroupStore(database.engine))
    queue = DatabaseJobQueue(database.engine)
    control_plane = control_plane or DatabaseControlPlane(
        database.engine,
        DatabaseHeartbeatStore(database.engine),
        configuration=dict(configuration),
        alerts=alerts,
    )
    worker_id = f"{node_id}:execution-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="execution-engine",
        capabilities=(
            "execute_targets",
            "live_order_submit",
            "live_order_recovery",
            "user_stream_event",
        ),
        observed_at=utc_now(),
    )
    worker = DatabaseExecutionWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        risk_store=SqlRiskDecisionStore(database.engine),
        trace_store=traces,
        order_groups=order_groups,
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_execution={
            product_id: {
                "execution_mode": product["execution_mode"],
                "execution_costs": product["execution_costs"],
                "base_accounting_asset": product["base_accounting_asset"],
            }
            for product_id, product in products.items()
        },
    )
    execution_ledgers = {
        product_id: Ledger(
            product_id=product_id,
            accounting_asset=str(product["base_accounting_asset"]),
            store=SqlLedgerStore(database.engine, product_id=product_id),
        )
        for product_id, product in products.items()
    }
    account_products = {
        str(account["account_id"]): str(account["products"][0])
        for account in configuration["accounts"]["accounts"]
    }
    live_worker = None
    recovery_worker = None
    emergency_worker = None
    protective_service = None
    live_product_ids: frozenset[str] = frozenset()
    approved_live = None
    if any(product["execution_mode"] == "live" for product in products.values()):
        approved_live = ApprovedLiveExecution(
            engine=database.engine,
            configuration=configuration,
            order_manager=order_manager,
            positions=positions,
        )
        live_product_ids = frozenset(approved_live.venues)
        protective_service = LiveProtectiveStopService(
            stop_manager=StopManager(SqlStopStore(database.engine)),
            venues=approved_live.venues,
            products=products,
            accounts=approved_live.accounts,
            queue=queue,
            order_manager=order_manager,
            positions=positions,
            alerts=alerts,
        )
        live_worker = DatabaseLiveExecutionWorker(
            queue=queue,
            worker_id=worker_id,
            order_manager=order_manager,
            positions=positions,
            ledgers={
                product_id: execution_ledgers[product_id]
                for product_id, product in products.items()
                if product["execution_mode"] == "live"
            },
            trace_store=traces,
            venues=approved_live.venues,
            authorise=approved_live.authorise,
            order_groups=order_groups,
            prepare_protective_stop=protective_service.prepare_entry,
            control_plane=control_plane,
        )
        emergency_worker = DatabaseEmergencyFlattenWorker(
            queue=queue,
            worker_id=worker_id,
            order_manager=order_manager,
            positions=positions,
            venues=approved_live.venues,
            products=products,
            alerts=alerts,
        )
        recovery_worker = DatabaseLiveRecoveryWorker(
            queue=queue,
            worker_id=worker_id,
            store=SqlRecoveryStore(database.engine),
            reconcile_product=approved_live.reconcile,
            account_products=account_products,
        )

    def on_live_fill(product_id: str, order, quantity: float, at: str) -> object:
        if protective_service is None or product_id not in live_product_ids:
            return None
        return protective_service.on_fill(product_id, order, quantity, at)

    def on_algo_update(product_id: str, event) -> object:
        if protective_service is None or product_id not in live_product_ids:
            return {"reason_code": "protective_algo_handler_not_configured"}
        return protective_service.on_algo_update(product_id, event)

    user_stream_worker = DatabaseUserStreamWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        ledgers=execution_ledgers,
        trace_store=traces,
        account_products=account_products,
        order_groups=order_groups,
        on_live_fill=on_live_fill if protective_service is not None else None,
        on_algo_update=on_algo_update if protective_service is not None else None,
    )

    def run_once() -> dict[str, Any]:
        now = utc_now()
        if emergency_worker is not None:
            result = emergency_worker.run_once(now=now)
            if result["reason_code"] != "emergency_queue_empty":
                return result
        result = worker.run_once(now=now)
        if result["reason_code"] != "execution_queue_empty":
            return result
        if live_worker is not None:
            result = live_worker.run_once(now=now)
            if result["reason_code"] != "live_order_queue_empty":
                return result
        result = user_stream_worker.run_once(now=now)
        if result["reason_code"] != "user_stream_queue_empty":
            return result
        if recovery_worker is not None:
            result = recovery_worker.run_once(now=now)
            if result["reason_code"] != "live_recovery_queue_empty":
                return result
        if protective_service is not None and approved_live is not None:
            reconciled = {
                product_id: protective_service.reconcile(product_id, now)
                for product_id in sorted(approved_live.venues)
            }
            if any(reconciled.values()):
                return {
                    "reason_code": "protective_stops_reconciled",
                    "stops": {
                        product_id: list(results) for product_id, results in reconciled.items()
                    },
                }
        return result

    return run_once


def _paper_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    configuration: Mapping[str, Mapping[str, Any]],
) -> Callable[[], dict[str, Any]]:
    order_manager, positions, traces = _execution_components(database)
    order_groups = OrderGroupManager(SqlOrderGroupStore(database.engine))
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:paper-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="paper-engine",
        capabilities=(
            "paper_order_submit",
            "paper_order_continue",
            "diagnostic_paper_open",
            "diagnostic_paper_close",
        ),
        observed_at=utc_now(),
    )
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    ledgers = {
        product_id: Ledger(
            product_id=product_id,
            accounting_asset=str(product["base_accounting_asset"]),
            store=SqlLedgerStore(database.engine, product_id=product_id),
        )
        for product_id, product in products.items()
    }
    diagnostic_worker = DatabaseDiagnosticPaperWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        products=products,
    )
    worker = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id=worker_id,
        order_manager=order_manager,
        positions=positions,
        ledgers=ledgers,
        trace_store=traces,
        order_groups=order_groups,
    )

    def run_once() -> dict[str, Any]:
        diagnostic_result = diagnostic_worker.run_once(now=utc_now())
        paper_result = worker.run_once(now=utc_now())
        if paper_result["reason_code"] != "paper_order_queue_empty":
            return {**paper_result, "diagnostic": diagnostic_result}
        return diagnostic_result

    return run_once


def _risk_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    configuration: Mapping[str, Mapping[str, Any]],
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:risk-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="risk-engine",
        capabilities=("risk_assessment",),
        observed_at=utc_now(),
    )
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    worker = DatabaseRiskWorker(
        queue=queue,
        worker_id=worker_id,
        store=SqlRiskDecisionStore(database.engine),
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        execution_modes={
            product_id: str(product["execution_mode"]) for product_id, product in products.items()
        },
    )
    return lambda: worker.run_once(now=utc_now())


def _data_writer_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    root: Path,
    configuration: Mapping[str, Mapping[str, Any]],
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:data-writer"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=utc_now(),
    )
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    accounts = _by_id(configuration["accounts"], collection="accounts", identity="account_id")
    products_by_market: dict[str, list[str]] = {}
    for product_id, product in products.items():
        account = accounts[str(product["account_id"])]
        market = {"spot": "spot", "usdt_futures": "futures"}.get(str(account["market"]))
        if market is None:
            raise ValueError(f"unsupported account market for {product_id}")
        products_by_market.setdefault(market, []).append(product_id)
    worker = DatabaseMarketDataWriter(
        queue=queue,
        worker_id=worker_id,
        root=root,
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market=products_by_market,
    )
    return lambda: worker.run_once(now=utc_now())


def _market_gateway_cycle(
    *,
    database: PlatformDatabase,
    configuration: Mapping[str, Mapping[str, Any]],
    config_root: Path,
    maximum_seconds: float,
) -> Callable[[], dict[str, Any]]:
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    live_account_ids = {
        str(product["account_id"])
        for product in products.values()
        if product.get("execution_mode") == "live"
    }
    accounts = tuple(
        account
        for payload in configuration["accounts"]["accounts"]
        if str(payload["account_id"]) in live_account_ids
        if (account := UserStreamAccount.from_config(payload)) is not None
    )
    universe = DatabaseUniverseService(store=SqlUniverseStore(database.engine))
    gateway = DatabaseMarketGateway(
        queue=DatabaseJobQueue(database.engine),
        capture_config=load_event_capture_config(config_root / "event_capture.json"),
        accounts=accounts,
        universe_symbols=lambda: universe.eligible_symbols_for_capture(),
    )
    if maximum_seconds <= 0:
        raise ValueError("market gateway heartbeat interval must be positive")
    return gateway


def _feature_cycle(
    *, database: PlatformDatabase, node_id: str, service_name: str, parquet_root: Path
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:{service_name}"
    job_names = (
        ("live_feature_calculation",)
        if service_name == "feature-service"
        else ("historical_feature_calculation",)
    )
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role=service_name,
        capabilities=job_names,
        observed_at=utc_now(),
    )
    worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id=worker_id,
        store=SqlFeatureStore(database.engine),
        job_names=job_names,
        parquet_root=parquet_root,
        active_assignments=_active_assignments(database.engine),
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        feature_graph_for_assignment=_artefact_feature_graph(database.engine),
    )
    return lambda: worker.run_once(now=utc_now())


def _active_assignments(engine) -> Callable[[str], tuple[Mapping[str, Any], ...]]:
    def load(instrument_id: str) -> tuple[Mapping[str, Any], ...]:
        with engine.connect() as connection:
            product_ids = tuple(
                str(value)
                for value in connection.execute(
                    select(active_strategy_assignment.c.product_id).distinct()
                )
                .scalars()
                .all()
                if value
            )
        repository = SqlActiveStrategyAssignmentRepository(engine)
        assignments: list[Mapping[str, Any]] = []
        for product_id in product_ids:
            for row in repository.active_assignments(product_id, at=utc_now()):
                payload = row["payload"]
                if isinstance(payload, Mapping):
                    if payload.get("diagnostic") is True:
                        continue
                    configured = payload.get("instrument_ids") or payload.get("instruments")
                    if isinstance(configured, list | tuple) and instrument_id not in {
                        str(value) for value in configured
                    }:
                        continue
                assignments.append(dict(row))
        return tuple(assignments)

    return load


def _artefact_feature_graph(engine) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def load(assignment: Mapping[str, Any]) -> Mapping[str, Any]:
        with engine.connect() as connection:
            payload = connection.execute(
                select(strategy_artefact.c.payload).where(
                    strategy_artefact.c.id == str(assignment["artefact_hash"])
                )
            ).scalar_one_or_none()
        if not isinstance(payload, Mapping):
            raise ValueError("active assignment artefact is missing")
        definition = payload.get("definition")
        graph = definition.get("feature_graph") if isinstance(definition, Mapping) else None
        if not isinstance(graph, Mapping):
            raise ValueError("active artefact has no feature graph")
        return graph

    return load


def _strategy_evaluator_cycle(
    *, database: PlatformDatabase, node_id: str
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:strategy-evaluator"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="strategy-evaluator",
        capabilities=("strategy_evaluation",),
        observed_at=utc_now(),
    )
    worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id=worker_id,
        feature_store=SqlFeatureStore(database.engine),
        portfolio=SqlPortfolioRepository(database.engine, require_pipeline_identity=True),
        assignments=SqlActiveStrategyAssignmentRepository(database.engine),
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        artefact_dispatcher=ArtefactDispatcher.default(),
    )
    return lambda: worker.run_once(now=utc_now())


def _universe_cycle(*, database: PlatformDatabase, node_id: str) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:universe-service"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="universe-service",
        capabilities=("universe_refresh",),
        observed_at=utc_now(),
    )
    service = DatabaseUniverseService(
        store=SqlUniverseStore(database.engine), queue=queue, worker_id=worker_id
    )
    return lambda: service.run_once(now=utc_now())


def _platform_scheduler_cycle(
    *, database: PlatformDatabase, node_id: str, configuration: Mapping[str, Mapping[str, Any]]
) -> Callable[[], dict[str, Any]]:
    scheduler = PlatformScheduler(
        engine=database.engine,
        products=_by_id(configuration["products"], collection="products", identity="product_id"),
        node_id=node_id,
        bootstrap=PlatformBootstrap(
            engine=database.engine,
            configuration=configuration,
            node_id=node_id,
        ),
    )
    return lambda: scheduler.run_once(now=utc_now())


def _promotion_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    configuration: Mapping[str, Mapping[str, Any]],
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:promotion-engine"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="promotion-engine",
        capabilities=(
            "forward_paper_observation",
            "forward_paper_summary",
            "promotion_evaluation",
        ),
        observed_at=utc_now(),
    )
    policy_store = SqlPromotionPolicyStore(database.engine)
    for raw_policy in configuration["promotion"]["policies"]:
        policy_fields = {
            key: raw_policy[key]
            for key in PromotionPolicy.__dataclass_fields__
            if key in raw_policy
        }
        policy_store.put(
            str(raw_policy["policy_id"]),
            PromotionPolicy(**policy_fields),
            created_at=utc_now(),
        )
    forward_worker = DatabaseForwardObservationWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_id,
    )
    policies_by_id = {
        str(policy["policy_id"]): dict(policy) for policy in configuration["promotion"]["policies"]
    }
    policies_by_product = {
        str(product["product_id"]): policies_by_id[str(product["promotion_policy_id"])]
        for product in configuration["products"]["products"]
    }
    minimum_forward_days = {
        product_id: int(policy["required_forward_evidence_days"])
        for product_id, policy in policies_by_product.items()
    }
    summary_worker = DatabaseForwardSummaryWorker(
        engine=database.engine,
        queue=queue,
        worker_id=worker_id,
        minimum_days_by_product=minimum_forward_days,
        policies_by_product=policies_by_product,
    )
    worker = DatabasePromotionWorker(
        queue=queue,
        worker_id=worker_id,
        store=SqlPromotionStore(database.engine),
        policy_store=policy_store,
    )

    def run_once() -> dict[str, Any]:
        now = utc_now()
        forward = forward_worker.run_once(now=now)
        if forward.get("reason_code") != "forward_observation_queue_empty":
            return forward
        summary = summary_worker.run_once(now=now)
        if summary.get("reason_code") != "forward_summary_queue_empty":
            return summary
        return worker.run_once(now=now)

    return run_once


def _accounting_cycle(*, database: PlatformDatabase, node_id: str) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:accounting-service"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="accounting-service",
        capabilities=("accounting_event",),
        observed_at=utc_now(),
    )
    service = AccountingService(
        engine=database.engine,
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        ledgers={
            product_id: Ledger(
                product_id=product_id,
                accounting_asset=asset,
                store=SqlLedgerStore(database.engine, product_id=product_id),
            )
            for product_id, asset in (
                ("btc_accumulation", "BTC"),
                ("active_income", "USDT"),
            )
        },
    )
    worker = DatabaseAccountingWorker(
        queue=queue,
        worker_id=worker_id,
        service=service,
    )
    return lambda: worker.run_once(now=utc_now())


def _account_reconciliation_cycle(
    *,
    database: PlatformDatabase,
    configuration: Mapping[str, Mapping[str, Any]],
    alerts: SqlAlertService | None = None,
) -> Callable[[], dict[str, Any]]:
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    accounts = _by_id(configuration["accounts"], collection="accounts", identity="account_id")
    service = AccountReconciliationService(
        engine=database.engine,
        products=products,
        accounts=accounts,
    )

    def run_once() -> dict[str, Any]:
        try:
            return service.reconcile_once(now=utc_now())
        except Exception as exc:
            if alerts is not None:
                try:
                    alerts.emit(
                        event_type="account_reconciliation_failed",
                        severity="critical",
                        dedupe_key="account-reconciliation:failed",
                        target="account-reconciliation",
                        message="account reconciliation failed",
                        emitted_at=utc_now(),
                        payload={"error_type": type(exc).__name__},
                        cooldown_seconds=60,
                    )
                except Exception:
                    pass
            raise

    return run_once


def _report_cycle(
    *,
    database: PlatformDatabase,
    root: Path,
    node_id: str,
    risk_configuration: Mapping[str, Any] | None = None,
    alerts: SqlAlertService | None = None,
    alerting_configuration: Mapping[str, Any] | None = None,
) -> Callable[[], dict[str, Any]]:
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:report-worker"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="report-worker",
        capabilities=("reporting",),
        observed_at=utc_now(),
    )
    worker = DatabaseReportWorker(
        engine=database.engine,
        root=root,
        queue=queue,
        worker_id=worker_id,
        account_stale_after_seconds=int(
            (risk_configuration or {}).get("maximum_database_staleness_seconds", 60)
        ),
        market_data_stale_after_seconds=int(
            (risk_configuration or {}).get("maximum_market_data_staleness_seconds", 5)
        ),
        alerts=alerts,
        minimum_valid_screenings_before_progress=int(
            (alerting_configuration or {}).get("minimum_valid_screenings_before_progress", 10)
        ),
    )
    return lambda: worker.run_once(now=utc_now())


def _research_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    service_name: str,
    runtime: ServiceRuntime,
    maximum_runtime_seconds: int,
    parquet_root: Path,
    artefact_root: Path,
    research_configuration: Mapping[str, Any],
    configuration: Mapping[str, Mapping[str, Any]] | None = None,
) -> Callable[[], dict[str, Any]]:
    job_names_by_service = {
        "research-worker": (
            "dataset_snapshot_validate",
            "register_strategy_catalogue",
            "generate_hypotheses",
            "register_candidate",
            "evaluate_candidate",
            "bounded_backtest",
        ),
        "ml-worker": (
            "register_ml_candidate",
            "train_ml_experiment",
            "evaluate_candidate",
            "bounded_backtest",
        ),
        "event-replay-worker": ("event_replay",),
    }
    job_names = job_names_by_service[service_name]
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:{service_name}"
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role=service_name,
        capabilities=job_names,
        observed_at=utc_now(),
    )
    dataset_resolver = CanonicalDatasetResolver(SqlCanonicalDatasetRepository(database.engine))
    raw_policy = research_configuration.get("evidence_policy", {})
    if not isinstance(raw_policy, Mapping):
        raise ValueError("research.evidence_policy must be an object")
    evidence_policy = EvidencePolicy.from_configuration(raw_policy)

    def load_research_dataset(kind: str, payload: Mapping[str, Any]) -> Any:
        snapshot_ids = tuple(str(item) for item in payload["dataset_snapshot_ids"])
        is_forward = str(payload.get("requested_stage") or "") == "forward"
        if is_forward:
            artefact_hash = str(payload.get("artefact_hash") or "")
            artefact_created_at = str(payload.get("artefact_created_at") or "")
            with database.engine.connect() as connection:
                stored_created_at = connection.execute(
                    select(strategy_artefact.c.created_at).where(
                        strategy_artefact.c.id == artefact_hash
                    )
                ).scalar_one_or_none()
            if stored_created_at is None or timestamp(
                str(stored_created_at), field="artefact.created_at"
            ) != timestamp(artefact_created_at, field="artefact_created_at"):
                raise ValueError(
                    "forward research must bind the exact immutable artefact timestamp"
                )
        raw_roles = payload.get("dataset_roles")
        allowed_roles = None
        if isinstance(raw_roles, Mapping):
            stage = str(payload.get("requested_stage") or "")
            role = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
                "protected": "protected_holdout",
                "forward": "forward_observation",
            }.get(stage)
            if role is not None:
                snapshot_ids = tuple(
                    snapshot_id
                    for snapshot_id in snapshot_ids
                    if str(raw_roles.get(snapshot_id)) == role
                )
                allowed_roles = frozenset({role})
        context = dataset_resolver.resolve_context(
            snapshot_ids=snapshot_ids,
            feature_manifest_id=str(payload["feature_manifest_id"]),
            cost_model_id=str(payload["cost_model_id"]),
            parameter_set_id=str(payload["parameter_set_id"]),
            allowed_roles=allowed_roles,
            minimum_availability_timestamp=(
                str(payload["artefact_created_at"]) if is_forward else None
            ),
            maximum_availability_timestamp=(str(payload["evaluated_at"]) if is_forward else None),
        )
        return context.get(kind, context)

    all_handlers = DatabaseResearchJobHandlers(
        SqlResearchStore(database.engine),
        artefact_store=PartitionedBacktestStore(parquet_root / "research"),
        ml_runner=MlExperimentRunner(
            artefact_store=ModelArtefactStore(artefact_root / "models"),
            metadata_store=SqlModelArtefactStore(database.engine),
        ),
        dataset_resolver=dataset_resolver,
        dataset_loader=load_research_dataset,
        evidence_policy=evidence_policy,
        configuration=configuration,
    ).handlers()
    worker = ResearchWorker(
        runtime=runtime,
        queue=queue,
        worker_id=worker_id,
        handlers={name: all_handlers[name] for name in job_names},
        lease_seconds=maximum_runtime_seconds,
        heavy_compute=HeavyComputeLeaseStore(database.engine),
    )
    return lambda: worker.run_once(now=utc_now())


def _agent_cycle(
    *,
    database: PlatformDatabase,
    node_id: str,
    runtime: ServiceRuntime,
    repository: Path,
    worktree_root: Path,
    research_configuration: Mapping[str, Any],
) -> Callable[[], dict[str, Any]]:
    limits = dict(research_configuration["resource_limits"])
    budgets = dict(research_configuration["search_budgets"])
    maximum_runtime_seconds = int(limits["maximum_runtime_seconds"])
    queue = DatabaseJobQueue(database.engine)
    worker_id = f"{node_id}:agent-sandbox"
    job_names = ("agent_research", "agent_code_workflow")
    queue.register_worker(
        worker_id=worker_id,
        node_id=node_id,
        role="agent-sandbox",
        capabilities=job_names,
        observed_at=utc_now(),
    )
    store = SqlAgentStore(database.engine)
    workflow = AgentCodeWorkflow(
        repository=repository,
        worktree_root=worktree_root,
        store=store,
        sandbox_policy=SandboxPolicy(
            timeout_seconds=maximum_runtime_seconds,
            maximum_memory_mb=int(budgets["maximum_memory_mb"]),
            maximum_file_bytes=int(limits["maximum_patch_bytes"]),
        ),
    )
    worker = ResearchWorker(
        runtime=runtime,
        queue=queue,
        worker_id=worker_id,
        handlers=DatabaseAgentJobHandlers(
            queue=queue,
            store=store,
            code_workflow=workflow,
            maximum_runtime_seconds=maximum_runtime_seconds,
        ).handlers(),
        lease_seconds=maximum_runtime_seconds,
        heavy_compute=HeavyComputeLeaseStore(database.engine),
    )
    return lambda: worker.run_once(now=utc_now())


def run(args: argparse.Namespace) -> int:
    config = load_platform_config(args.config)
    split_configuration = load_split_configuration(args.config.parent)
    config.assert_service_assignment(node_id=args.node, service=args.service)
    if args.validate:
        return 0
    database = PlatformDatabase(config.database_url())
    if not database.is_postgresql:
        raise ValueError("platform services require PostgreSQL")
    if args.service == "migration-service" or args.initialise_schema:
        database.migrate()
        PlatformBootstrap(
            engine=database.engine,
            configuration=split_configuration,
            node_id=args.node,
        ).ensure(now=utc_now())
    else:
        database.assert_migrated()
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    alerts = configured_alert_service(database.engine, configuration=config.alerting)
    control_plane = DatabaseControlPlane(
        database.engine,
        heartbeat_store,
        configuration=split_configuration,
        alerts=alerts,
    )
    runtime = ServiceRuntime(
        config=config,
        node_id=args.node,
        service_name=args.service,
        heartbeat_store=heartbeat_store,
    )
    if args.service == "market-gateway":
        work = _market_gateway_cycle(
            database=database,
            configuration=split_configuration,
            config_root=args.config.parent,
            maximum_seconds=float(args.interval_seconds),
        )
    elif args.service == "product-supervisor":
        work = _product_coordination_cycle(database=database, node_id=args.node)
    elif args.service == "portfolio-engine":
        work = _portfolio_cycle(
            database=database, configuration=split_configuration, node_id=args.node
        )
    elif args.service == "portfolio-state-service":
        work = _portfolio_state_cycle(
            database=database,
            configuration=split_configuration,
            node_id=args.node,
        )
    elif args.service == "execution-engine":
        work = _execution_cycle(
            database=database,
            configuration=split_configuration,
            node_id=args.node,
            alerts=alerts,
            control_plane=control_plane,
        )
    elif args.service == "paper-engine":
        work = _paper_cycle(
            database=database,
            node_id=args.node,
            configuration=split_configuration,
        )
    elif args.service == "risk-engine":
        work = _risk_cycle(database=database, node_id=args.node, configuration=split_configuration)
    elif args.service == "strategy-evaluator":
        work = _strategy_evaluator_cycle(database=database, node_id=args.node)
    elif args.service == "universe-service":
        work = _universe_cycle(database=database, node_id=args.node)
    elif args.service == "platform-scheduler":
        work = _platform_scheduler_cycle(
            database=database, node_id=args.node, configuration=split_configuration
        )
    elif args.service == "data-writer":
        work = _data_writer_cycle(
            database=database,
            node_id=args.node,
            root=Path(config.paths["parquet"]),
            configuration=split_configuration,
        )
    elif args.service in {"feature-service", "feature-build-worker"}:
        work = _feature_cycle(
            database=database,
            node_id=args.node,
            service_name=args.service,
            parquet_root=Path(config.paths["parquet"]),
        )
    elif args.service == "promotion-engine":
        work = _promotion_cycle(
            database=database,
            node_id=args.node,
            configuration=split_configuration,
        )
    elif args.service == "accounting-service":
        work = _accounting_cycle(database=database, node_id=args.node)
    elif args.service == "account-reconciliation":
        work = _account_reconciliation_cycle(
            database=database, configuration=split_configuration, alerts=alerts
        )
    elif args.service == "report-worker":
        work = _report_cycle(
            database=database,
            root=Path(config.paths["reports"]),
            node_id=args.node,
            risk_configuration=split_configuration["risk"],
            alerts=alerts,
            alerting_configuration=config.alerting,
        )
    elif args.service in {"research-worker", "ml-worker", "event-replay-worker"}:
        work = _research_cycle(
            database=database,
            node_id=args.node,
            service_name=args.service,
            runtime=runtime,
            maximum_runtime_seconds=int(
                split_configuration["research"]["resource_limits"]["maximum_runtime_seconds"]
            ),
            parquet_root=Path(config.paths["parquet"]),
            artefact_root=Path(config.paths["artefacts"]),
            research_configuration=split_configuration["research"],
            configuration=split_configuration,
        )
    elif args.service == "agent-sandbox":
        configured_worktree_root = os.environ.get("TRADING_PLATFORM_AGENT_WORKTREE_ROOT")
        if not configured_worktree_root:
            raise RuntimeError(
                "TRADING_PLATFORM_AGENT_WORKTREE_ROOT must be configured for agent-sandbox"
            )
        work = _agent_cycle(
            database=database,
            node_id=args.node,
            runtime=runtime,
            repository=Path.cwd(),
            worktree_root=Path(configured_worktree_root),
            research_configuration=split_configuration["research"],
        )
    else:

        def work() -> dict[str, Any]:
            return _idle_cycle(args.service)

    market_gateway = work if isinstance(work, DatabaseMarketGateway) else None
    if args.service != "control-api":
        unpaused_work = work

        def work() -> dict[str, Any]:
            if control_plane.service_is_paused(args.service):
                return {"reason_code": "service_paused", "service": args.service}
            return unpaused_work()

    if args.service == "control-api" and not args.once:
        token = os.environ.get(args.control_token_env, "")
        server = build_control_server(
            bind=_parse_bind(args.control_bind),
            control_plane=DatabaseControlPlane(
                database.engine,
                heartbeat_store,
                configuration=split_configuration,
                alerts=alerts,
            ),
            bearer_token=token,
        )
        runtime.heartbeat(payload={"reason_code": "control_api_started"})

        def stop_server(_signum: int, _frame: object) -> None:
            server.shutdown()

        signal.signal(signal.SIGTERM, stop_server)
        signal.signal(signal.SIGINT, stop_server)
        server.serve_forever(poll_interval=0.5)
        database.dispose()
        return 0
    if args.once:
        cycle = runtime.run_once(work)
        database.dispose()
        return 0 if cycle.healthy else 1

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stopping:
        cycle = runtime.run_once(work)
        reason_code = str(cycle.detail.get("reason_code") or "")
        if (
            args.service == "report-worker"
            or reason_code == "market_gateway_running"
            or reason_code.endswith("queue_empty")
            or reason_code.endswith("paused")
            or reason_code
            in {
                "service_waiting_for_input",
                "service_cycle_failed",
                "portfolio_state_waiting_for_source_snapshots",
                "portfolio_state_idle",
                "canonical_portfolio_state_published",
                "platform_scheduler_completed",
                "account_reconciliation_completed",
            }
        ):
            time.sleep(args.interval_seconds)
    if market_gateway is not None:
        market_gateway.stop()
    runtime.heartbeat(payload={"reason_code": "service_stopped"})
    database.dispose()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one assigned platform service.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--node", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--control-bind", default="127.0.0.1:8088")
    parser.add_argument("--control-token-env", default="TRADING_CONTROL_TOKEN")
    parser.add_argument("--initialise-schema", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
