"""Privilege-aligned process runners for the PostgreSQL platform."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
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
    _research_cycle,
    _risk_cycle,
    _strategy_evaluator_cycle,
    _universe_cycle,
)

PROCESS_SERVICES = {
    process: services
    for process, services in PLATFORM_PROCESS_SERVICES.items()
    if process
    in {
        "market-gateway",
        "data-writer",
        "trading-runtime",
        "scheduler-operations",
        "research-runtime",
        "evaluation-runtime",
        "agent-runtime",
    }
}


def _build_trading_work(
    *,
    database: PlatformDatabase,
    config: Any,
    node_id: str,
    configuration: Mapping[str, Any],
    alerts: Any,
    selected_services: tuple[str, ...] | None = None,
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    selected = set(selected_services or PLATFORM_PROCESS_SERVICES["trading-runtime"])
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    control_plane = DatabaseControlPlane(
        database.engine,
        heartbeat_store,
        configuration={str(key): dict(value) for key, value in configuration.items()},
    )
    builders: dict[str, Callable[[], Callable[[], Mapping[str, Any] | None]]] = {
        "market-gateway": lambda: _market_gateway_cycle(
            database=database,
            configuration=configuration,
            config_root=Path("config"),
            maximum_seconds=15.0,
        ),
        "data-writer": lambda: _data_writer_cycle(
            database=database,
            node_id=node_id,
            root=Path(config.paths["parquet"]),
            configuration=configuration,
        ),
        "feature-service": lambda: _feature_cycle(
            database=database,
            node_id=node_id,
            service_name="feature-service",
            parquet_root=Path(config.paths["parquet"]),
        ),
        "universe-service": lambda: _universe_cycle(database=database, node_id=node_id),
        "platform-scheduler": lambda: _platform_scheduler_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "strategy-evaluator": lambda: _strategy_evaluator_cycle(database=database, node_id=node_id),
        "product-supervisor": lambda: _product_coordination_cycle(
            database=database, node_id=node_id
        ),
        "portfolio-engine": lambda: _portfolio_cycle(
            database=database, configuration=configuration, node_id=node_id
        ),
        "portfolio-state-service": lambda: _portfolio_state_cycle(
            database=database, configuration=configuration, node_id=node_id
        ),
        "risk-engine": lambda: _risk_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "execution-engine": lambda: _execution_cycle(
            database=database,
            configuration=configuration,
            node_id=node_id,
            alerts=alerts,
            control_plane=control_plane,
        ),
        "paper-engine": lambda: _paper_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
        "accounting-service": lambda: _accounting_cycle(database=database, node_id=node_id),
        "account-reconciliation": lambda: _account_reconciliation_cycle(
            database=database, configuration=configuration, alerts=alerts
        ),
        "promotion-engine": lambda: _promotion_cycle(
            database=database, node_id=node_id, configuration=configuration
        ),
    }
    return {name: builders[name]() for name in selected if name in builders}


def _build_research_work(
    *,
    database: PlatformDatabase,
    config: Any,
    configuration: Mapping[str, Any],
    node_id: str,
    selected_services: tuple[str, ...] | None = None,
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    selected = set(selected_services or PLATFORM_PROCESS_SERVICES["research-runtime"])
    heartbeat_store = DatabaseHeartbeatStore(database.engine)
    runtime_by_service = {
        service: ServiceRuntime(
            config=config,
            node_id=node_id,
            service_name=service,
            heartbeat_store=heartbeat_store,
        )
        for service in selected
    }
    work: dict[str, Callable[[], Mapping[str, Any] | None]] = {}
    if "promotion-engine" in selected:
        work["promotion-engine"] = _promotion_cycle(
            database=database, node_id=node_id, configuration=configuration
        )
    for service_name in (
        "research-worker",
        "evaluation-worker",
        "ml-worker",
        "event-replay-worker",
    ):
        if service_name in selected:
            work[service_name] = _research_cycle(
                database=database,
                node_id=node_id,
                service_name=service_name,
                runtime=runtime_by_service[service_name],
                maximum_runtime_seconds=int(
                    configuration["research"]["resource_limits"]["maximum_runtime_seconds"]
                ),
                parquet_root=Path(config.paths["parquet"]),
                artefact_root=Path(config.paths["artefacts"]),
                research_configuration=configuration["research"],
                configuration=configuration,
            )
    if "feature-build-worker" in selected:
        work["feature-build-worker"] = _feature_cycle(
            database=database,
            node_id=node_id,
            service_name="feature-build-worker",
            parquet_root=Path(config.paths["parquet"]),
        )
    if "agent-coordinator" in selected:
        work.update(
            _build_agent_coordinator_work(
                database=database, config=config, configuration=configuration, node_id=node_id
            )
        )
    return work


def _build_agent_coordinator_work(
    *, database: PlatformDatabase, config: Any, configuration: Mapping[str, Any], node_id: str
) -> dict[str, Callable[[], Mapping[str, Any] | None]]:
    runtime = ServiceRuntime(
        config=config,
        node_id=node_id,
        service_name="agent-coordinator",
        heartbeat_store=DatabaseHeartbeatStore(database.engine),
    )
    root = os.environ.get("TRADING_PLATFORM_AGENT_WORKTREE_ROOT")
    if not root:
        raise RuntimeError("TRADING_PLATFORM_AGENT_WORKTREE_ROOT must be configured")
    return {
        "agent-coordinator": _agent_cycle(
            database=database,
            node_id=node_id,
            runtime=runtime,
            repository=Path.cwd(),
            worktree_root=Path(root),
            research_configuration=configuration["research"],
            forbidden_secret_names=tuple(
                str(account[field])
                for account in configuration["accounts"]["accounts"]
                for field in ("api_key_env", "api_secret_env")
                if account.get(field)
            ),
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
    if process_name in {
        "market-gateway",
        "data-writer",
        "trading-runtime",
        "scheduler-operations",
    }:
        return _build_trading_work(
            database=database,
            config=config,
            node_id=node_id,
            configuration=configuration,
            alerts=alerts,
            selected_services=PROCESS_SERVICES[process_name],
        )
    if process_name in {"research-runtime", "evaluation-runtime"}:
        return _build_research_work(
            database=database,
            config=config,
            configuration=configuration,
            node_id=node_id,
            selected_services=PROCESS_SERVICES[process_name],
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
    if args.process == "agent-runtime":
        return _run_agent_transport(args, configuration=configuration)
    database = PlatformDatabase(config.database_url())
    if not database.is_postgresql:
        raise ValueError("platform processes require PostgreSQL")
    database.assert_migrated()
    from src.services.database_roles import check_service_roles

    check_service_roles(database.engine, config, services)
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
    return _run_process_loop(
        args,
        database=database,
        work=work,
        config=config,
        heartbeat_store=heartbeat_store,
    )


def _run_agent_transport(args: argparse.Namespace, *, configuration: Mapping[str, Any]) -> int:
    from src.agents.consumer import consume_configured_review

    forbidden = {
        "TRADING_PLATFORM_DATABASE_URL",
        *(
            str(account[field])
            for account in configuration["accounts"]["accounts"]
            for field in ("api_key_env", "api_secret_env")
            if account.get(field)
        ),
    }
    if any(os.environ.get(name) for name in forbidden):
        raise PermissionError("agent transport must not receive database or exchange credentials")
    while True:
        result = consume_configured_review()
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


def _run_process_loop(
    args: argparse.Namespace,
    *,
    database: PlatformDatabase,
    work: Mapping[str, Callable[[], Mapping[str, Any] | None]],
    config: Any,
    heartbeat_store: DatabaseHeartbeatStore,
) -> int:
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
    with ThreadPoolExecutor(
        max_workers=max(1, len(work)), thread_name_prefix="platform-service"
    ) as pool:
        while not stopping:
            futures = {
                service: pool.submit(runtimes[service].run_once, work[service]) for service in work
            }
            cycles = {service: future.result() for service, future in futures.items()}
            if args.once:
                _shutdown_work(work)
                database.dispose()
                return 0 if all(cycle.healthy for cycle in cycles.values()) else 1
            time.sleep(args.interval_seconds)
    _shutdown_work(work)
    database.dispose()
    return 0


def _shutdown_work(work: Mapping[str, Callable[[], Mapping[str, Any] | None]]) -> None:
    for cycle in work.values():
        stop = getattr(cycle, "stop", None)
        if callable(stop):
            stop()
        close = getattr(cycle, "close", None)
        if callable(close):
            close()


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
