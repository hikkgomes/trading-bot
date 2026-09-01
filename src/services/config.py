"""Validated configuration for the single-node Linux platform."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

from src.research.evidence import EvidenceProfile

LINUX_SERVICES = frozenset(
    {
        "market-gateway",
        "data-writer",
        "feature-service",
        "portfolio-engine",
        "portfolio-state-service",
        "risk-engine",
        "execution-engine",
        "paper-engine",
        "product-supervisor",
        "accounting-service",
        "account-reconciliation",
        "promotion-engine",
        "control-api",
        "strategy-evaluator",
        "universe-service",
        "platform-scheduler",
        "research-worker",
        "ml-worker",
        "event-replay-worker",
        "agent-sandbox",
        "feature-build-worker",
        "migration-service",
    }
)
MAC_SERVICES: frozenset[str] = frozenset()
ORDER_SUBMISSION_SERVICES = frozenset({"execution-engine"})

PLATFORM_PROCESS_SERVICES: dict[str, tuple[str, ...]] = {
    "trading-runtime": (
        "market-gateway",
        "data-writer",
        "feature-service",
        "universe-service",
        "platform-scheduler",
        "strategy-evaluator",
        "product-supervisor",
        "portfolio-engine",
        "portfolio-state-service",
        "risk-engine",
        "execution-engine",
        "paper-engine",
        "accounting-service",
        "account-reconciliation",
    ),
    "research-runtime": (
        "promotion-engine",
        "research-worker",
        "ml-worker",
        "event-replay-worker",
        "feature-build-worker",
    ),
    "agent-runtime": ("agent-sandbox",),
    "control-api": ("control-api",),
    "migration-service": ("migration-service",),
}


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _platform_nodes(data: Mapping[str, Any]) -> tuple[NodeConfig, ...]:
    raw_nodes = data.get("nodes")
    if data.get("schema") != "platform/v2":
        if isinstance(raw_nodes, list) and any(
            isinstance(node, dict) and node.get("operating_system") != "linux" for node in raw_nodes
        ):
            raise ValueError("all platform services must run on Linux")
        raise ValueError("platform schema must be platform/v2")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("platform.nodes must define at least one node")
    nodes = tuple(NodeConfig.from_dict(item) for item in raw_nodes)
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("platform node IDs must be unique")
    if sum(node.production_authority for node in nodes) != 1:
        raise ValueError("platform must define one production authority")
    assigned = [service for node in nodes for service in node.services]
    if set(assigned) != LINUX_SERVICES or len(assigned) != len(LINUX_SERVICES):
        raise ValueError("each required service must be assigned exactly once")
    return nodes


def _platform_database(data: Mapping[str, Any]) -> tuple[dict[str, Any], int, bool]:
    database = _mapping(data.get("postgresql"), field="platform.postgresql")
    connect_timeout = database.get("connect_timeout_seconds", 5)
    if not isinstance(connect_timeout, int) or isinstance(connect_timeout, bool):
        raise ValueError("postgresql.connect_timeout_seconds must be an integer")
    if connect_timeout <= 0:
        raise ValueError("postgresql.connect_timeout_seconds must be positive")
    require_tls = database.get("require_tls", True)
    if not isinstance(require_tls, bool):
        raise ValueError("postgresql.require_tls must be a boolean")
    return database, connect_timeout, require_tls


def _platform_paths(data: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        key: _string(item, field=f"platform.paths.{key}")
        for key, item in _mapping(data.get("paths"), field="platform.paths").items()
    }
    required = {"parquet", "artefacts", "backups"}
    if set(paths) != required:
        raise ValueError(f"platform.paths must contain {sorted(required)}")
    return paths


def _platform_worker_limits(data: Mapping[str, Any]) -> dict[str, int]:
    raw_limits = _mapping(data.get("worker_limits"), field="platform.worker_limits")
    limits: dict[str, int] = {}
    for name, limit in raw_limits.items():
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError(f"platform.worker_limits.{name} must be a positive integer")
        limits[name] = limit
    return limits


def _platform_security_domains(data: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw_domains = _mapping(data.get("security_domains", {}), field="platform.security_domains")
    domains: dict[str, tuple[str, ...]] = {}
    for domain, services_value in raw_domains.items():
        if not isinstance(services_value, list | tuple) or not services_value:
            raise ValueError(f"platform.security_domains.{domain} must be a non-empty list")
        domains[str(domain)] = tuple(
            _string(item, field=f"security_domains.{domain}[]") for item in services_value
        )
    expected_domains = {
        "trading-runtime",
        "trading-research",
        "trading-agent",
        "trading-migration",
    }
    if set(domains) != expected_domains:
        raise ValueError("platform.security_domains must define the four Linux security domains")
    services = [service for values in domains.values() for service in values]
    if set(services) != LINUX_SERVICES or len(services) != len(LINUX_SERVICES):
        raise ValueError("each service must belong to exactly one security domain")
    return domains


def _platform_processes(data: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    default = {key: list(value) for key, value in PLATFORM_PROCESS_SERVICES.items()}
    raw_processes = _mapping(data.get("processes", default), field="platform.processes")
    processes: dict[str, tuple[str, ...]] = {}
    for process, services in raw_processes.items():
        if not isinstance(services, list | tuple) or not services:
            raise ValueError(f"platform.processes.{process} must be a non-empty list")
        processes[str(process)] = tuple(
            _string(item, field=f"platform.processes.{process}[]") for item in services
        )
    if set(processes) != set(PLATFORM_PROCESS_SERVICES):
        raise ValueError("platform.processes must define every platform process")
    services = [service for values in processes.values() for service in values]
    if set(services) != LINUX_SERVICES or len(services) != len(LINUX_SERVICES):
        raise ValueError("each platform service must belong to exactly one process")
    return processes


def _platform_alerting(data: Mapping[str, Any]) -> dict[str, Any]:
    alerting = _mapping(data.get("alerting"), field="platform.alerting")
    minimum = alerting.get("minimum_valid_screenings_before_progress", 10)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise ValueError(
            "platform.alerting.minimum_valid_screenings_before_progress must be positive"
        )
    return alerting


def _platform_backup(data: Mapping[str, Any]) -> dict[str, Any]:
    backup = _mapping(data.get("backup"), field="platform.backup")
    for field_name in ("maximum_age_seconds", "retention_days"):
        value = backup.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"platform.backup.{field_name} must be a positive integer")
    minimum_free_bytes = backup.get("minimum_free_bytes", 0)
    if (
        not isinstance(minimum_free_bytes, int)
        or isinstance(minimum_free_bytes, bool)
        or minimum_free_bytes < 0
    ):
        raise ValueError("platform.backup.minimum_free_bytes must be a non-negative integer")
    return backup


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), field=str(path))


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    operating_system: str
    production_authority: bool
    services: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> NodeConfig:
        data = _mapping(value, field="node")
        services_value = data.get("services")
        if not isinstance(services_value, list) or not services_value:
            raise ValueError("node.services must be a non-empty list")
        services = tuple(_string(item, field="node.services[]") for item in services_value)
        if len(services) != len(set(services)):
            raise ValueError("node.services must not contain duplicates")
        operating_system = _string(data.get("operating_system"), field="node.operating_system")
        production_authority = data.get("production_authority") is True
        if production_authority and operating_system != "linux":
            raise ValueError("only a Linux node can be the production authority")
        if operating_system != "linux":
            raise ValueError("all platform services must run on Linux")
        if not set(services).issubset(LINUX_SERVICES):
            raise ValueError("a Linux node contains an unsupported service assignment")
        return cls(
            node_id=_string(data.get("node_id"), field="node.node_id"),
            operating_system=operating_system,
            production_authority=production_authority,
            services=services,
        )


@dataclass(frozen=True)
class PlatformConfig:
    schema: str
    nodes: tuple[NodeConfig, ...]
    database_url_env: str
    database_connect_timeout_seconds: int
    database_require_tls: bool
    paths: dict[str, str]
    worker_limits: dict[str, int]
    network: dict[str, Any]
    logging: dict[str, Any]
    metrics: dict[str, Any]
    alerting: dict[str, Any]
    backup: dict[str, Any]
    resource_limits: dict[str, Any]
    security_domains: dict[str, tuple[str, ...]]
    processes: dict[str, tuple[str, ...]]

    @classmethod
    def from_dict(cls, value: object) -> PlatformConfig:
        data = _mapping(value, field="platform")
        nodes = _platform_nodes(data)
        database, connect_timeout, require_tls = _platform_database(data)
        paths = _platform_paths(data)
        worker_limits = _platform_worker_limits(data)
        security_domains = _platform_security_domains(data)
        processes = _platform_processes(data)
        alerting = _platform_alerting(data)
        backup = _platform_backup(data)
        return cls(
            schema=_string(data.get("schema"), field="platform.schema"),
            nodes=nodes,
            database_url_env=_string(database.get("url_env"), field="postgresql.url_env"),
            database_connect_timeout_seconds=connect_timeout,
            database_require_tls=require_tls,
            paths=paths,
            worker_limits=worker_limits,
            network=_mapping(data.get("network"), field="platform.network"),
            logging=_mapping(data.get("logging"), field="platform.logging"),
            metrics=_mapping(data.get("metrics"), field="platform.metrics"),
            alerting=alerting,
            backup=backup,
            resource_limits=_mapping(
                data.get("resource_limits", {}), field="platform.resource_limits"
            ),
            security_domains=security_domains,
            processes=processes,
        )

    def node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"unknown node_id: {node_id}")

    def assert_service_assignment(self, *, node_id: str, service: str) -> NodeConfig:
        node = self.node(node_id)
        if service not in node.services:
            raise PermissionError(f"service {service} is not assigned to node {node_id}")
        if service in ORDER_SUBMISSION_SERVICES and not node.production_authority:
            raise PermissionError("only the production authority can submit exchange orders")
        return node

    def database_url(self, environment: dict[str, str] | None = None) -> str:
        source = os.environ if environment is None else environment
        value = source.get(self.database_url_env, "").strip()
        if not value:
            raise ValueError(
                f"database URL environment variable is missing: {self.database_url_env}"
            )
        if not value.startswith(("postgresql+psycopg://", "postgresql://")):
            raise ValueError("the shared database URL must use PostgreSQL")
        parts = urlsplit(value)
        is_local_socket = not parts.hostname and (
            "host=" in parts.query or parts.path.startswith("/")
        )
        if self.database_require_tls and not is_local_socket:
            sslmode = parse_qs(parts.query).get("sslmode", [])
            if not sslmode or sslmode[-1] not in {"require", "verify-ca", "verify-full"}:
                raise ValueError("the shared PostgreSQL URL must require TLS")
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("connect_timeout", str(self.database_connect_timeout_seconds))
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )


def load_platform_config(path: Path = Path("config/platform.json")) -> PlatformConfig:
    return PlatformConfig.from_dict(load_json_object(path))


def load_split_configuration(root: Path = Path("config")) -> dict[str, dict[str, Any]]:
    required = (
        "accounts",
        "products",
        "portfolios",
        "research",
        "risk",
        "promotion",
    )
    configuration = {name: load_json_object(root / f"{name}.json") for name in required}
    validate_split_configuration(configuration)
    return configuration


def _unique_records(
    payload: dict[str, Any], *, collection: str, key: str
) -> dict[str, dict[str, Any]]:
    records = payload.get(collection)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{collection} must be a non-empty list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in records:
        record = _mapping(item, field=f"{collection}[]")
        identity = _string(record.get(key), field=f"{collection}[].{key}")
        if identity in indexed:
            raise ValueError(f"duplicate {key}: {identity}")
        indexed[identity] = record
    return indexed


def validate_split_configuration(configuration: dict[str, dict[str, Any]]) -> None:
    accounts, products, portfolios, policies, risk_products = _split_records(configuration)
    _validate_promotion_policies(policies)
    _validate_products(products, accounts, portfolios, policies, risk_products)
    _validate_research_configuration(configuration["research"])


def _split_records(
    configuration: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    required = {"accounts", "products", "portfolios", "research", "risk", "promotion"}
    if set(configuration) != required:
        raise ValueError(f"split configuration must contain {sorted(required)}")
    for name, payload in configuration.items():
        if payload.get("schema") not in {f"{name}/v1", f"{name}/v2"}:
            raise ValueError(f"{name} configuration has an unsupported schema")
    accounts = _unique_records(configuration["accounts"], collection="accounts", key="account_id")
    products = _unique_records(configuration["products"], collection="products", key="product_id")
    portfolios = _unique_records(
        configuration["portfolios"], collection="portfolios", key="portfolio_id"
    )
    policies = _unique_records(configuration["promotion"], collection="policies", key="policy_id")
    risk_products = _mapping(configuration["risk"].get("products"), field="risk.products")
    return accounts, products, portfolios, policies, risk_products


def _validate_promotion_policies(policies: dict[str, dict[str, Any]]) -> None:
    for policy_id, policy in policies.items():
        _validate_promotion_switches(policy_id, policy)
        _validate_promotion_fractions(policy_id, policy)
        _validate_promotion_counts(policy_id, policy)


def _validate_promotion_switches(policy_id: str, policy: Mapping[str, Any]) -> None:
    for field in (
        "automatic_paper_promotion",
        "automatic_live_canary_promotion",
        "automatic_live_ready_promotion",
    ):
        if not isinstance(policy.get(field), bool):
            raise ValueError(f"promotion policy {policy_id}.{field} must be a boolean")
    required_days = policy.get("required_forward_evidence_days")
    if not isinstance(required_days, int) or isinstance(required_days, bool) or required_days <= 0:
        raise ValueError(
            f"promotion policy {policy_id}.required_forward_evidence_days must be positive"
        )


def _validate_promotion_fractions(policy_id: str, policy: Mapping[str, Any]) -> None:
    for field in (
        "canary_capital_limit",
        "maximum_drawdown",
        "maximum_execution_drift",
        "maximum_model_drift",
        "paper_capital_limit",
        "maximum_forward_slippage",
    ):
        _require_nonnegative_number(
            policy.get(field),
            field=f"promotion policy {policy_id}.{field}",
        )
    for field in ("minimum_forward_fill_rate", "minimum_forward_data_uptime"):
        value = policy.get(field, 0.0)
        if not _is_number(value) or not 0 <= float(value) <= 1:
            raise ValueError(
                f"promotion policy {policy_id}.{field} must be a number between zero and one"
            )
    for field in (
        "maximum_forward_tail_loss",
        "minimum_forward_net_pnl",
        "minimum_forward_objective_excess_fraction",
    ):
        _require_nonnegative_number(
            policy.get(field, 0.0),
            field=f"promotion policy {policy_id}.{field}",
        )


def _validate_promotion_counts(policy_id: str, policy: Mapping[str, Any]) -> None:
    for field in (
        "minimum_forward_effective_trades",
        "maximum_forward_rejected_orders",
        "minimum_forward_trading_days",
        "minimum_forward_cycles",
        "minimum_forward_effective_episodes",
        "maximum_forward_data_gaps",
    ):
        value = policy.get(field, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"promotion policy {policy_id}.{field} must be a non-negative integer")
    minimum_decisions = policy.get("minimum_forward_independent_decisions", 1)
    if (
        not isinstance(minimum_decisions, int)
        or isinstance(minimum_decisions, bool)
        or minimum_decisions < 1
    ):
        raise ValueError(
            f"promotion policy {policy_id}.minimum_forward_independent_decisions must be a positive integer"
        )


def _is_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_nonnegative_number(value: object, *, field: str) -> None:
    if not _is_number(value) or float(value) < 0:
        raise ValueError(f"{field} must be finite and non-negative")


def _validate_products(
    products: dict[str, dict[str, Any]],
    accounts: dict[str, dict[str, Any]],
    portfolios: dict[str, dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    risk_products: Mapping[str, Any],
) -> None:
    if set(products) != {"btc_accumulation", "active_income"}:
        raise ValueError("products must define btc_accumulation and active_income")
    for product_id, product in products.items():
        _validate_product_bindings(
            product_id, product, accounts, portfolios, policies, risk_products
        )
        _validate_product_mode(product_id, product, accounts)
        _validate_product_timeouts(product_id, product)
        _validate_product_costs(product_id, product)
        if product_id == "btc_accumulation":
            _validate_btc_product(product_id, product, accounts)
        if product_id == "active_income":
            _validate_active_income_product(product_id, product)


def _validate_product_bindings(
    product_id: str,
    product: Mapping[str, Any],
    accounts: Mapping[str, Mapping[str, Any]],
    portfolios: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
    risk_products: Mapping[str, Any],
) -> None:
    account_id = _string(product.get("account_id"), field=f"{product_id}.account_id")
    portfolio_id = _string(product.get("portfolio_id"), field=f"{product_id}.portfolio_id")
    promotion_id = _string(
        product.get("promotion_policy_id"), field=f"{product_id}.promotion_policy_id"
    )
    risk_id = _string(product.get("risk_policy_id"), field=f"{product_id}.risk_policy_id")
    if account_id not in accounts:
        raise ValueError(f"product {product_id} refers to an unknown account")
    if product_id not in accounts[account_id].get("products", []):
        raise ValueError(f"account {account_id} does not grant product access to {product_id}")
    if portfolio_id not in portfolios:
        raise ValueError(f"product {product_id} refers to an unknown portfolio")
    if portfolios[portfolio_id].get("product_id") != product_id:
        raise ValueError(f"portfolio {portfolio_id} belongs to another product")
    if promotion_id not in policies:
        raise ValueError(f"product {product_id} refers to an unknown promotion policy")
    if risk_id not in risk_products:
        raise ValueError(f"product {product_id} refers to an unknown risk policy")


def _validate_product_mode(
    product_id: str,
    product: Mapping[str, Any],
    accounts: Mapping[str, Mapping[str, Any]],
) -> None:
    mode = product.get("execution_mode")
    if mode not in {"paper", "live"}:
        raise ValueError(f"product {product_id} has an invalid execution mode")
    account = accounts[str(product["account_id"])]
    if mode == "paper":
        if (
            not isinstance(account.get("paper_starting_balances"), dict)
            or not account["paper_starting_balances"]
        ):
            raise ValueError(f"paper account {account['account_id']} needs paper_starting_balances")
        if not isinstance(account.get("paper_starting_positions"), dict):
            raise ValueError(
                f"paper account {account['account_id']} needs paper_starting_positions"
            )


def _validate_product_timeouts(product_id: str, product: Mapping[str, Any]) -> None:
    _require_positive_integer(
        product.get("preflight_max_age_seconds", 3_600), f"product {product_id} preflight age"
    )
    defaults = {
        "account_snapshot_max_age_seconds": 60,
        "maximum_funding_age_seconds": 28_800,
        "connected_testnet_max_age_seconds": 86_400,
    }
    for field, default in defaults.items():
        _require_positive_integer(product.get(field, default), f"product {product_id}.{field}")


def _validate_product_costs(product_id: str, product: Mapping[str, Any]) -> None:
    costs = _mapping(product.get("execution_costs"), field=f"{product_id}.execution_costs")
    for name in ("fee_bps", "slippage_bps"):
        value = costs.get(name)
        if not _is_number(value) or float(value) < 0:
            raise ValueError(f"product {product_id} has an invalid {name}")


def _validate_btc_product(
    product_id: str,
    product: Mapping[str, Any],
    accounts: Mapping[str, Mapping[str, Any]],
) -> None:
    account = accounts[str(product["account_id"])]
    if account.get("market") != "spot":
        raise ValueError("BTC accumulation must use a spot account")
    _validate_btc_symbol_scope(product_id, product)
    for field in ("btc_core_fraction", "btc_max_tactical_fraction", "btc_minimum_fraction"):
        value = product.get(field)
        if not _is_number(value) or not 0 <= float(value) <= 1:
            raise ValueError(f"product {product_id}.{field} must be between zero and one")
    if float(product["btc_core_fraction"]) == 0 and float(product["btc_max_tactical_fraction"]) > 0:
        raise ValueError(
            "BTC accumulation tactical allocation needs a positive neutral BTC fraction"
        )
    if float(product["btc_minimum_fraction"]) > float(product["btc_core_fraction"]):
        raise ValueError("BTC minimum fraction cannot exceed the neutral BTC fraction")


def _validate_btc_symbol_scope(product_id: str, product: Mapping[str, Any]) -> None:
    if str(product.get("universe_id") or "") != "btc-spot":
        raise ValueError("BTC accumulation must use the btc-spot universe")
    for field in ("exchange_symbol", "live_exchange_symbol"):
        declared = product.get(field)
        if declared is not None and str(declared).upper() != "BTCUSDT":
            raise ValueError(f"product {product_id}.{field} must be BTCUSDT")
    for field in ("exchange_symbols", "live_exchange_symbols"):
        declared = product.get(field)
        if declared is not None and (
            not isinstance(declared, list | tuple)
            or tuple(str(value).upper() for value in declared) != ("BTCUSDT",)
        ):
            raise ValueError(f"product {product_id}.{field} must contain BTCUSDT only")


def _validate_active_income_product(product_id: str, product: Mapping[str, Any]) -> None:
    symbols = product.get("live_exchange_symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)
        or len({str(symbol).upper() for symbol in symbols}) != len(symbols)
    ):
        raise ValueError(
            f"product {product_id}.live_exchange_symbols must be a unique non-empty list"
        )


def _require_positive_integer(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _validate_research_configuration(research: Mapping[str, Any]) -> None:
    permissions = _mapping(research.get("agent_permissions"), field="agent_permissions")
    if permissions.get("submit_exchange_orders") is not False:
        raise ValueError("research agents must be denied exchange order submission")
    worker_node = research.get("worker_node")
    if worker_node is not None and (not isinstance(worker_node, str) or not worker_node.strip()):
        raise ValueError("research.worker_node must identify a Linux worker")
    validation = _mapping(research.get("validation"), field="research.validation")
    if validation.get("chronological_only") is not True:
        raise ValueError("research validation must be chronological")
    holdout = _mapping(research.get("holdout"), field="research.holdout")
    if holdout.get("protected") is not True or holdout.get("adaptive_feedback") is not False:
        raise ValueError("protected holdout data must be excluded from adaptive feedback")
    _validate_evidence_policy(
        _mapping(research.get("evidence_policy"), field="research.evidence_policy")
    )


def _validate_evidence_policy(policy: Mapping[str, Any]) -> None:
    version = policy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("research.evidence_policy.version must be non-empty")
    for field in (
        "minimum_cost_adjusted_return",
        "minimum_deflated_sharpe",
        "minimum_walk_forward_pass_fraction",
        "maximum_backtest_overfitting_probability",
        "maximum_portfolio_correlation",
    ):
        value = policy.get(field)
        if not _is_number(value) or not 0 <= float(value) <= 1:
            raise ValueError(f"research.evidence_policy.{field} must be finite in [0, 1]")
    windows = policy.get("minimum_walk_forward_windows")
    if not isinstance(windows, int) or isinstance(windows, bool) or windows < 3:
        raise ValueError("research.evidence_policy.minimum_walk_forward_windows must be >= 3")
    observations = policy.get("minimum_bootstrap_observations")
    if not isinstance(observations, int) or isinstance(observations, bool) or observations < 30:
        raise ValueError("research.evidence_policy.minimum_bootstrap_observations must be >= 30")
    procedures = {
        field: policy.get(field)
        for field in ("bootstrap_method", "multiple_testing_method", "pbo_method")
    }
    if any(not isinstance(value, str) or not value.strip() for value in procedures.values()):
        raise ValueError("research.evidence_policy statistical procedures must be named")
    confidence_method = policy.get("confidence_method", "bootstrap")
    if confidence_method not in {"bootstrap", "deflated_sharpe"}:
        raise ValueError(
            "research.evidence_policy.confidence_method must be bootstrap or deflated_sharpe"
        )
    profiles = policy.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("research.evidence_policy.profiles must be a list")
    try:
        tuple(EvidenceProfile.from_mapping(profile) for profile in profiles)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"research.evidence_policy.profiles is invalid: {exc}") from exc
