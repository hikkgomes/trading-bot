"""SQLAlchemy schema for the shared PostgreSQL operational control plane."""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine

metadata = MetaData()


def _id_payload_table(name: str) -> Table:
    return Table(
        name,
        metadata,
        Column("id", String(160), primary_key=True),
        Column("created_at", String(40), nullable=False),
        Column("payload", JSON, nullable=False, default=dict),
    )


instrument = Table(
    "instrument",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("venue", String(40), nullable=False),
    Column("market_type", String(40), nullable=False),
    Column("exchange_symbol", String(80), nullable=False),
    Column("base_asset", String(40), nullable=False),
    Column("quote_asset", String(40), nullable=False),
    Column("settlement_asset", String(40)),
    Column("payload", JSON, nullable=False),
)
instrument_status = Table(
    "instrument_status",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("instrument_id", ForeignKey("instrument.id"), nullable=False, index=True),
    Column("observed_at", String(40), nullable=False, index=True),
    Column("status", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
universe = _id_payload_table("universe")
universe_snapshot = Table(
    "universe_snapshot",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("universe_id", ForeignKey("universe.id"), nullable=False, index=True),
    Column("observed_at", String(40), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
universe_member = Table(
    "universe_member",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("snapshot_id", ForeignKey("universe_snapshot.id"), nullable=False, index=True),
    Column("instrument_id", ForeignKey("instrument.id"), nullable=False, index=True),
    Column("eligible", Boolean, nullable=False),
    Column("reason_code", String(120)),
    Column("payload", JSON, nullable=False),
)
dataset_snapshot = _id_payload_table("dataset_snapshot")
feature_set = _id_payload_table("feature_set")
feature_manifest = _id_payload_table("feature_manifest")

strategy_definition = Table(
    "strategy_definition",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("identity", String(160), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("source_type", String(80), nullable=False),
    Column("source_hash", String(80), nullable=False),
    Column("definition", JSON, nullable=False),
)
strategy_version = Table(
    "strategy_version",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("definition_id", ForeignKey("strategy_definition.id"), nullable=False, index=True),
    Column("version", String(80), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
strategy_lineage = _id_payload_table("strategy_lineage")
experiment = Table(
    "experiment",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("provider", String(120), nullable=False, index=True),
    Column("state", String(80), nullable=False, index=True),
    Column("submitted_at", String(40), nullable=False),
    Column("dataset_snapshot_hashes", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
)
experiment_run = _id_payload_table("experiment_run")
experiment_metric = _id_payload_table("experiment_metric")
validation_result = Table(
    "validation_result",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("experiment_id", ForeignKey("experiment.id"), nullable=False, index=True),
    Column("state", String(80), nullable=False),
    Column("accepted", Boolean, nullable=False),
    Column("reason_code", String(160)),
    Column("evidence", JSON, nullable=False),
)
holdout_claim = _id_payload_table("holdout_claim")
model_artifact = _id_payload_table("model_artifact")
strategy_artefact = _id_payload_table("strategy_artefact")
forward_evidence = _id_payload_table("forward_evidence")

agent_action = _id_payload_table("agent_action")
agent_proposal = _id_payload_table("agent_proposal")
agent_patch = _id_payload_table("agent_patch")
agent_review = _id_payload_table("agent_review")
agent_disposition = _id_payload_table("agent_disposition")

portfolio = _id_payload_table("portfolio")
portfolio_sleeve = _id_payload_table("portfolio_sleeve")
portfolio_strategy = _id_payload_table("portfolio_strategy")
alpha_forecast = _id_payload_table("alpha_forecast")
target_position = _id_payload_table("target_position")
risk_snapshot = _id_payload_table("risk_snapshot")
risk_decision = Table(
    "risk_decision",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("scope", String(80), nullable=False, index=True),
    Column("evaluated_at", String(40), nullable=False, index=True),
    Column("accepted", Boolean, nullable=False),
    Column("reason_code", String(160)),
    Column("payload", JSON, nullable=False),
)

account = _id_payload_table("account")
balance_snapshot = _id_payload_table("balance_snapshot")
order_intent = _id_payload_table("order_intent")
exchange_order = Table(
    "exchange_order",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("order_id", ForeignKey("order_intent.id"), nullable=False, index=True),
    Column("sequence", Integer, nullable=False),
    Column("created_at", String(40), nullable=False, index=True),
    Column("status", String(80), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("order_id", "sequence", name="uq_exchange_order_sequence"),
)
order_group = Table(
    "order_group",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("group_id", String(160), nullable=False, index=True),
    Column("sequence", Integer, nullable=False),
    Column("created_at", String(40), nullable=False, index=True),
    Column("status", String(80), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("group_id", "sequence", name="uq_order_group_sequence"),
)
protective_stop = Table(
    "protective_stop",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("stop_id", String(160), nullable=False, index=True),
    Column("sequence", Integer, nullable=False),
    Column("created_at", String(40), nullable=False, index=True),
    Column("status", String(80), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("stop_id", "sequence", name="uq_protective_stop_sequence"),
)
fill = Table(
    "fill",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("order_id", ForeignKey("order_intent.id"), nullable=False, index=True),
    Column("created_at", String(40), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
)
position = _id_payload_table("position")
position_event = _id_payload_table("position_event")
reconciliation_event = _id_payload_table("reconciliation_event")

nav_snapshot = _id_payload_table("nav_snapshot")
accounting_entry = Table(
    "accounting_entry",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("sequence", Integer, nullable=False),
    Column("created_at", String(40), nullable=False, index=True),
    Column("entry_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("product_id", "sequence", name="uq_accounting_entry_sequence"),
)
trade_attribution = _id_payload_table("trade_attribution")
funding_entry = _id_payload_table("funding_entry")
fee_entry = _id_payload_table("fee_entry")

job = Table(
    "job",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("name", String(160), nullable=False, index=True),
    Column("state", String(40), nullable=False, index=True),
    Column("priority", Integer, nullable=False, default=0),
    Column("available_at", String(40), nullable=False, index=True),
    Column("lease_owner", String(160), index=True),
    Column("lease_expires_at", String(40), index=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("payload", JSON, nullable=False),
)
job_attempt = Table(
    "job_attempt",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("job_id", ForeignKey("job.id"), nullable=False, index=True),
    Column("worker_id", String(160), nullable=False, index=True),
    Column("started_at", String(40), nullable=False),
    Column("completed_at", String(40)),
    Column("status", String(40), nullable=False),
    Column("error", Text),
    Column("payload", JSON, nullable=False),
)
worker = Table(
    "worker",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("node_id", String(160), nullable=False, index=True),
    Column("role", String(120), nullable=False, index=True),
    Column("last_heartbeat", String(40), nullable=False),
    Column("status", String(40), nullable=False),
    Column("capabilities", JSON, nullable=False),
    Column("payload", JSON, nullable=False),
)
worker_lease = Table(
    "worker_lease",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("job_id", ForeignKey("job.id"), nullable=False, index=True),
    Column("worker_id", ForeignKey("worker.id"), nullable=False, index=True),
    Column("expires_at", String(40), nullable=False, index=True),
    Column("status", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
service_heartbeat = Table(
    "service_heartbeat",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("service_name", String(160), nullable=False, index=True),
    Column("node_id", String(160), nullable=False, index=True),
    Column("observed_at", String(40), nullable=False, index=True),
    Column("healthy", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)
alert = _id_payload_table("alert")
control_event = _id_payload_table("control_event")
promotion_event = _id_payload_table("promotion_event")
decision_trace = Table(
    "decision_trace",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("event_id", String(240), nullable=False, index=True),
    Column("instrument_id", String(200), nullable=False, index=True),
    Column("evaluated_at", String(40), nullable=False, index=True),
    Column("first_blocked_stage", String(120), index=True),
    Column("payload", JSON, nullable=False),
)


class PlatformDatabase:
    """Explicit database lifecycle; production URLs must use PostgreSQL."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = (
            url.replace("postgresql://", "postgresql+psycopg://", 1)
            if url.startswith("postgresql://")
            else url
        )
        self.engine: Engine = create_engine(self.url, echo=echo, future=True, pool_pre_ping=True)

    @property
    def is_postgresql(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def create_schema(self) -> None:
        metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()


CORE_TABLE_NAMES = frozenset(metadata.tables)
