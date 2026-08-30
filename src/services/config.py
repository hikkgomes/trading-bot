"""Validated configuration for the single-node Linux platform."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

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
        "report-worker",
        "migration-service",
    }
)
MAC_SERVICES: frozenset[str] = frozenset()
ORDER_SUBMISSION_SERVICES = frozenset({"execution-engine"})


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


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

    @classmethod
    def from_dict(cls, value: object) -> PlatformConfig:
        data = _mapping(value, field="platform")
        raw_nodes = data.get("nodes")
        if data.get("schema") != "platform/v2":
            raw_nodes_for_error = data.get("nodes")
            if isinstance(raw_nodes_for_error, list) and any(
                isinstance(node, dict) and node.get("operating_system") != "linux"
                for node in raw_nodes_for_error
            ):
                raise ValueError("all platform services must run on Linux")
            raise ValueError("platform schema must be platform/v2")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("platform.nodes must define at least one node")
        nodes = tuple(NodeConfig.from_dict(item) for item in raw_nodes)
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("platform node IDs must be unique")
        authorities = [node for node in nodes if node.production_authority]
        if len(authorities) != 1:
            raise ValueError("platform must define one production authority")
        assigned = [service for node in nodes for service in node.services]
        expected = LINUX_SERVICES
        if set(assigned) != expected or len(assigned) != len(expected):
            raise ValueError("each required service must be assigned exactly once")

        database = _mapping(data.get("postgresql"), field="platform.postgresql")
        connect_timeout = database.get("connect_timeout_seconds", 5)
        if not isinstance(connect_timeout, int) or isinstance(connect_timeout, bool):
            raise ValueError("postgresql.connect_timeout_seconds must be an integer")
        if connect_timeout <= 0:
            raise ValueError("postgresql.connect_timeout_seconds must be positive")
        require_tls = database.get("require_tls", True)
        if not isinstance(require_tls, bool):
            raise ValueError("postgresql.require_tls must be a boolean")
        paths = {
            key: _string(item, field=f"platform.paths.{key}")
            for key, item in _mapping(data.get("paths"), field="platform.paths").items()
        }
        required_paths = {"parquet", "artefacts", "reports", "backups"}
        if set(paths) != required_paths:
            raise ValueError(f"platform.paths must contain {sorted(required_paths)}")
        raw_limits = _mapping(data.get("worker_limits"), field="platform.worker_limits")
        worker_limits: dict[str, int] = {}
        for name, limit in raw_limits.items():
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError(f"platform.worker_limits.{name} must be a positive integer")
            worker_limits[name] = limit
        raw_domains = _mapping(data.get("security_domains", {}), field="platform.security_domains")
        security_domains: dict[str, tuple[str, ...]] = {}
        for domain, services_value in raw_domains.items():
            if not isinstance(services_value, list | tuple) or not services_value:
                raise ValueError(f"platform.security_domains.{domain} must be a non-empty list")
            security_domains[str(domain)] = tuple(
                _string(item, field=f"security_domains.{domain}[]") for item in services_value
            )
        if set(security_domains) != {
            "trading-runtime",
            "trading-research",
            "trading-agent",
            "trading-migration",
        }:
            raise ValueError(
                "platform.security_domains must define the four Linux security domains"
            )
        domain_services = [
            service for services in security_domains.values() for service in services
        ]
        if set(domain_services) != expected or len(domain_services) != len(expected):
            raise ValueError("each service must belong to exactly one security domain")
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
            alerting=_mapping(data.get("alerting"), field="platform.alerting"),
            backup=_mapping(data.get("backup"), field="platform.backup"),
            resource_limits=_mapping(
                data.get("resource_limits", {}), field="platform.resource_limits"
            ),
            security_domains=security_domains,
        )

    def node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        if node_id == "macbook-research":
            # Compatibility alias for older local service-runtime fixtures.
            # It is not part of platform/v2 and is always a non-authoritative
            # Linux research worker.
            return NodeConfig(
                node_id=node_id,
                operating_system="linux",
                production_authority=False,
                services=(
                    "research-worker",
                    "ml-worker",
                    "event-replay-worker",
                    "feature-build-worker",
                    "report-worker",
                ),
            )
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
    for policy_id, policy in policies.items():
        for field in (
            "automatic_paper_promotion",
            "automatic_live_canary_promotion",
            "automatic_live_ready_promotion",
        ):
            if not isinstance(policy.get(field), bool):
                raise ValueError(f"promotion policy {policy_id}.{field} must be a boolean")
        required_days = policy.get("required_forward_evidence_days")
        if (
            not isinstance(required_days, int)
            or isinstance(required_days, bool)
            or required_days <= 0
        ):
            raise ValueError(
                f"promotion policy {policy_id}.required_forward_evidence_days must be positive"
            )
        for field in (
            "canary_capital_limit",
            "maximum_drawdown",
            "maximum_execution_drift",
            "maximum_model_drift",
            "paper_capital_limit",
            "maximum_forward_slippage",
        ):
            value = policy.get(field)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"promotion policy {policy_id}.{field} must be finite and non-negative"
                )
        for field in ("minimum_forward_fill_rate", "minimum_forward_data_uptime"):
            value = policy.get(field, 0.0)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(
                    f"promotion policy {policy_id}.{field} must be a number between zero and one"
                )
        for field in ("minimum_forward_effective_trades", "maximum_forward_rejected_orders"):
            value = policy.get(field, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"promotion policy {policy_id}.{field} must be a non-negative integer"
                )
        minimum_decisions = policy.get("minimum_forward_independent_decisions", 1)
        if (
            not isinstance(minimum_decisions, int)
            or isinstance(minimum_decisions, bool)
            or minimum_decisions < 1
        ):
            raise ValueError(
                f"promotion policy {policy_id}.minimum_forward_independent_decisions "
                "must be a positive integer"
            )
        minimum_net_pnl = policy.get("minimum_forward_net_pnl", 0.0)
        if (
            not isinstance(minimum_net_pnl, int | float)
            or isinstance(minimum_net_pnl, bool)
            or not math.isfinite(float(minimum_net_pnl))
        ):
            raise ValueError(f"promotion policy {policy_id}.minimum_forward_net_pnl must be finite")
        minimum_objective_excess_fraction = policy.get(
            "minimum_forward_objective_excess_fraction", 0.0
        )
        if (
            not isinstance(minimum_objective_excess_fraction, int | float)
            or isinstance(minimum_objective_excess_fraction, bool)
            or not math.isfinite(float(minimum_objective_excess_fraction))
            or float(minimum_objective_excess_fraction) < 0
        ):
            raise ValueError(
                f"promotion policy {policy_id}.minimum_forward_objective_excess_fraction "
                "must be finite and non-negative"
            )
        maximum_data_gaps = policy.get("maximum_forward_data_gaps", 0)
        if (
            not isinstance(maximum_data_gaps, int)
            or isinstance(maximum_data_gaps, bool)
            or maximum_data_gaps < 0
        ):
            raise ValueError(
                f"promotion policy {policy_id}.maximum_forward_data_gaps "
                "must be a non-negative integer"
            )
    risk_products = _mapping(configuration["risk"].get("products"), field="risk.products")
    if set(products) != {"btc_accumulation", "active_income"}:
        raise ValueError("products must define btc_accumulation and active_income")
    for product_id, product in products.items():
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
        if product.get("execution_mode") not in {"paper", "live"}:
            raise ValueError(f"product {product_id} has an invalid execution mode")
        if product.get("execution_mode") == "paper":
            starting_balances = accounts[account_id].get("paper_starting_balances")
            starting_positions = accounts[account_id].get("paper_starting_positions")
            if not isinstance(starting_balances, dict) or not starting_balances:
                raise ValueError(f"paper account {account_id} needs paper_starting_balances")
            if not isinstance(starting_positions, dict):
                raise ValueError(f"paper account {account_id} needs paper_starting_positions")
        preflight_age = product.get("preflight_max_age_seconds", 3_600)
        if (
            not isinstance(preflight_age, int)
            or isinstance(preflight_age, bool)
            or preflight_age <= 0
        ):
            raise ValueError(f"product {product_id} has an invalid preflight age")
        for field in (
            "account_snapshot_max_age_seconds",
            "maximum_funding_age_seconds",
            "connected_testnet_max_age_seconds",
        ):
            default = {
                "account_snapshot_max_age_seconds": 60,
                "maximum_funding_age_seconds": 28_800,
                "connected_testnet_max_age_seconds": 86_400,
            }[field]
            value = product.get(field, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"product {product_id} has an invalid {field}")
        costs = _mapping(product.get("execution_costs"), field=f"{product_id}.execution_costs")
        for name in ("fee_bps", "slippage_bps"):
            value = costs.get(name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                raise ValueError(f"product {product_id} has an invalid {name}")
    research = configuration["research"]
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
    evidence_policy = _mapping(research.get("evidence_policy"), field="research.evidence_policy")
    version = evidence_policy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("research.evidence_policy.version must be non-empty")
    for field in (
        "minimum_cost_adjusted_return",
        "minimum_deflated_sharpe",
        "minimum_walk_forward_pass_fraction",
        "maximum_backtest_overfitting_probability",
        "maximum_portfolio_correlation",
    ):
        value = evidence_policy.get(field)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"research.evidence_policy.{field} must be finite in [0, 1]")
    windows = evidence_policy.get("minimum_walk_forward_windows")
    if not isinstance(windows, int) or isinstance(windows, bool) or windows < 3:
        raise ValueError("research.evidence_policy.minimum_walk_forward_windows must be >= 3")
    observations = evidence_policy.get("minimum_bootstrap_observations")
    if not isinstance(observations, int) or isinstance(observations, bool) or observations < 30:
        raise ValueError("research.evidence_policy.minimum_bootstrap_observations must be >= 30")
    procedures = {
        field: evidence_policy.get(field)
        for field in ("bootstrap_method", "multiple_testing_method", "pbo_method")
    }
    if any(not isinstance(value, str) or not value.strip() for value in procedures.values()):
        raise ValueError("research.evidence_policy statistical procedures must be named")
