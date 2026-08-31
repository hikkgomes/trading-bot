"""Privilege-aligned process runners for the PostgreSQL platform."""

from __future__ import annotations

import argparse
import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.data.database import PlatformDatabase
from src.services.alerting import configured_alert_service
from src.services.config import (
    PLATFORM_PROCESS_SERVICES,
    load_platform_config,
    load_split_configuration,
)
from src.services.control_api import DatabaseControlPlane
from src.services.health import DatabaseHeartbeatStore
from src.services.runtime import ServiceRuntime
from src.services.supervisor import (
    _account_reconciliation_cycle,
    _accounting_cycle,
    _agent_cycle,
    _by_id,
    _data_writer_cycle,
    _execution_cycle,
    _feature_cycle,
    _market_gateway_cycle,
    _paper_cycle,
    _platform_scheduler_cycle,
    _portfolio_cycle,
    _portfolio_state_cycle,
    _product_coordination_cycle,
    _promotion_cycle,
    _report_cycle,
    _research_cycle,
    _risk_cycle,
    _strategy_evaluator_cycle,
    _universe_cycle,
)

PROCESS_SERVICES = {
    process: services
    for process, services in PLATFORM_PROCESS_SERVICES.items()
    if process in {"trading-runtime", "research-runtime", "agent-runtime"}
}


def _configuration_by_id(configuration: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    products = _by_id(configuration["products"], collection="products", identity="product_id")
    accounts = _by_id(configuration["accounts"], collection="accounts", identity="account_id")
    return products, accounts


def _build_trading_work(
    *,
    database: PlatformDatabase,
    config: Any,
    node_id: str,
    configuration: Mapping[str, Any],
    alerts: Any,
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    products, _accounts = _configuration_by_id(configuration)
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    control_plane = DatabaseControlPlane(
        database.engine,
        heartbeat_store,
        configuration=configuration,
    )
    return {
        "market-gateway": _market_gateway_cycle(
            database=database,
            configuration=configuration,
            config_root=Path("config"),
            maximum_seconds=15.0,
        ),
        "data-writer": _data_writer_cycle(
            database=database,
            node_id=node_id,
            root=Path(config.paths["parquet"]),
            configuration=configuration,
        ),
        "feature-service": _feature_cycle(
            database=database,
            node_id=node_id,
            service_name="feature-service",
            parquet_root=Path(config.paths["parquet"]),
        ),
        "universe-service": _universe_cycle(database=database, node_id=node_id),
        "platform-scheduler": _platform_scheduler_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "strategy-evaluator": _strategy_evaluator_cycle(database=database, node_id=node_id),
        "product-supervisor": _product_coordination_cycle(database=database, node_id=node_id),
        "portfolio-engine": _portfolio_cycle(
            database=database, configuration=configuration, node_id=node_id
        ),
        "portfolio-state-service": _portfolio_state_cycle(
            database=database, configuration=configuration, node_id=node_id
        ),
        "risk-engine": _risk_cycle(database=database, node_id=node_id, configuration=configuration),
        "execution-engine": _execution_cycle(
            database=database,
            configuration=configuration,
            node_id=node_id,
            alerts=alerts,
            control_plane=control_plane,
        ),
        "paper-engine": _paper_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "accounting-service": _accounting_cycle(database=database, node_id=node_id),
        "account-reconciliation": _account_reconciliation_cycle(
            database=database, configuration=configuration, alerts=alerts
        ),
    }


def _build_research_work(
    *, database: PlatformDatabase, config: Any, configuration: Mapping[str, Any], node_id: str
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    runtime_by_service = {
        service: ServiceRuntime(
            config=config,
            node_id=node_id,
            service_name=service,
            heartbeat_store=heartbeat_store,
        )
        for service in PROCESS_SERVICES["research-runtime"]
    }
    work: dict[str, Callable[[], Mapping[str, Any] | None]] = {
        "promotion-engine": _promotion_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "research-worker": _research_cycle(
            database=database,
            node_id=node_id,
            service_name="research-worker",
            runtime=runtime_by_service["research-worker"],
            maximum_runtime_seconds=int(
                configuration["research"]["resource_limits"]["maximum_runtime_seconds"]
            ),
            parquet_root=Path(config.paths["parquet"]),
            artefact_root=Path(config.paths["artefacts"]),
            research_configuration=configuration["research"],
            configuration=configuration,
        ),
        "ml-worker": _research_cycle(
            database=database,
            node_id=node_id,
            service_name="ml-worker",
            runtime=runtime_by_service["ml-worker"],
            maximum_runtime_seconds=int(
                configuration["research"]["resource_limits"]["maximum_runtime_seconds"]
            ),
            parquet_root=Path(config.paths["parquet"]),
            artefact_root=Path(config.paths["artefacts"]),
            research_configuration=configuration["research"],
            configuration=configuration,
        ),
        "event-replay-worker": _research_cycle(
            database=database,
            node_id=node_id,
            service_name="event-replay-worker",
            runtime=runtime_by_service["event-replay-worker"],
            maximum_runtime_seconds=int(
                configuration["research"]["resource_limits"]["maximum_runtime_seconds"]
            ),
            parquet_root=Path(config.paths["parquet"]),
            artefact_root=Path(config.paths["artefacts"]),
            research_configuration=configuration["research"],
            configuration=configuration,
        ),
        "feature-build-worker": _feature_cycle(
            database=database,
            node_id=node_id,
            service_name="feature-build-worker",
            parquet_root=Path(config.paths["parquet"]),
        ),
        "report-worker": _report_cycle(
            database=database,
            root=Path(config.paths["reports"]),
            node_id=node_id,
            risk_configuration=configuration["risk"],
            alerting_configuration=config.alerting,
        ),
    }
    return work


def _build_agent_work(
    *, database: PlatformDatabase, config: Any, configuration: Mapping[str, Any], node_id: str
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    runtime = ServiceRuntime(
        config=config,
        node_id=node_id,
        service_name="agent-sandbox",
        heartbeat_store=DatabaseHeartbeatStore(database.engine),
    )
    root = os.environ.get("TRADING_PLATFORM_AGENT_WORKTREE_ROOT")
    if not root:
        raise RuntimeError("TRADING_PLATFORM_AGENT_WORKTREE_ROOT must be configured")
    return {
        "agent-sandbox": _agent_cycle(
            database=database,
            node_id=node_id,
            runtime=runtime,
            repository=Path.cwd(),
            worktree_root=Path(root),
            research_configuration=configuration["research"],
        )
    }


def _build_work(
    process_name: str,
    *,
    database: PlatformDatabase,
    config: Any,
    configuration: Mapping[str, Any],
    node_id: str,
    alerts: Any,
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    if process_name == "trading-runtime":
        return _build_trading_work(
            database=database,
            config=config,
            node_id=node_id,
            configuration=configuration,
            alerts=alerts,
        )
    if process_name == "research-runtime":
        return _build_research_work(
            database=database, config=config, configuration=configuration, node_id=node_id
        )
    if process_name == "agent-runtime":
        return _build_agent_work(
            database=database, config=config, configuration=configuration, node_id=node_id
        )
    raise ValueError(f"unsupported platform process: {process_name}")


def run(args: argparse.Namespace) -> int:
    config = load_platform_config(args.config)
    configuration = load_split_configuration(args.config.parent)
    services = PROCESS_SERVICES.get(args.process)
    if services is None:
        raise ValueError(f"unsupported platform process: {args.process}")
    for service in services:
        config.assert_service_assignment(node_id=args.node, service=service)
    if args.validate:
        return 0
    database = PlatformDatabase(config.database_url())
    if not database.is_postgresql:
        raise ValueError("platform processes require PostgreSQL")
    database.assert_migrated()
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    alerts = configured_alert_service(database.engine, configuration=config.alerting)
    work = _build_work(
        args.process,
        database=database,
        config=config,
        configuration=configuration,
        node_id=args.node,
        alerts=alerts,
    )
    runtimes = {
        service: ServiceRuntime(
            config=config,
            node_id=args.node,
            service_name=service,
            heartbeat_store=heartbeat_store,
        )
        for service in work
    }
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stopping:
        cycles = {service: runtimes[service].run_once(work[service]) for service in work}
        if args.once:
            database.dispose()
            return 0 if all(cycle.healthy for cycle in cycles.values()) else 1
        time.sleep(args.interval_seconds)
    gateway = work.get("market-gateway")
    stop = getattr(gateway, "stop", None)
    if callable(stop):
        stop()
    database.dispose()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a privilege-aligned platform process.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--node", required=True)
    parser.add_argument("--process", required=True, choices=tuple(PROCESS_SERVICES))
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
