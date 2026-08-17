"""Validated configuration for the two-node platform."""

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
        "risk-engine",
        "execution-engine",
        "paper-engine",
        "product-supervisor",
        "accounting-service",
        "promotion-engine",
        "control-api",
    }
)
MAC_SERVICES = frozenset(
    {
        "research-worker",
        "ml-worker",
        "event-replay-worker",
        "agent-sandbox",
        "feature-build-worker",
        "report-worker",
    }
)
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
        if operating_system == "linux" and not set(services).issubset(LINUX_SERVICES):
            raise ValueError("a Linux node contains an unsupported service assignment")
        if operating_system == "macos" and not set(services).issubset(MAC_SERVICES):
            raise ValueError("a macOS node contains an execution service assignment")
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

    @classmethod
    def from_dict(cls, value: object) -> PlatformConfig:
        data = _mapping(value, field="platform")
        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, list) or len(raw_nodes) != 2:
            raise ValueError("platform.nodes must define exactly two nodes")
        nodes = tuple(NodeConfig.from_dict(item) for item in raw_nodes)
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("platform node IDs must be unique")
        authorities = [node for node in nodes if node.production_authority]
        if len(authorities) != 1:
            raise ValueError("platform must define one production authority")
        assigned = [service for node in nodes for service in node.services]
        expected = LINUX_SERVICES | MAC_SERVICES
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
        if self.database_require_tls:
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
        if payload.get("schema") != f"{name}/v1":
            raise ValueError(f"{name} configuration has an unsupported schema")

    accounts = _unique_records(configuration["accounts"], collection="accounts", key="account_id")
    products = _unique_records(configuration["products"], collection="products", key="product_id")
    portfolios = _unique_records(
        configuration["portfolios"], collection="portfolios", key="portfolio_id"
    )
    policies = _unique_records(configuration["promotion"], collection="policies", key="policy_id")
    for policy_id, policy in policies.items():
        for field in ("automatic_paper_promotion", "automatic_live_canary_promotion"):
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
        preflight_age = product.get("preflight_max_age_seconds", 3_600)
        if (
            not isinstance(preflight_age, int)
            or isinstance(preflight_age, bool)
            or preflight_age <= 0
        ):
            raise ValueError(f"product {product_id} has an invalid preflight age")
        costs = _mapping(product.get("execution_costs"), field=f"{product_id}.execution_costs")
        for name in ("fee_bps", "slippage_bps"):
            value = costs.get(name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                raise ValueError(f"product {product_id} has an invalid {name}")
    research = configuration["research"]
    permissions = _mapping(research.get("agent_permissions"), field="agent_permissions")
    if permissions.get("submit_exchange_orders") is not False:
        raise ValueError("research agents must be denied exchange order submission")
    if research.get("worker_node") != "macbook-research":
        raise ValueError("research work must be assigned to the macOS research node")
    validation = _mapping(research.get("validation"), field="research.validation")
    if validation.get("chronological_only") is not True:
        raise ValueError("research validation must be chronological")
    holdout = _mapping(research.get("holdout"), field="research.holdout")
    if holdout.get("protected") is not True or holdout.get("adaptive_feedback") is not False:
        raise ValueError("protected holdout data must be excluded from adaptive feedback")
