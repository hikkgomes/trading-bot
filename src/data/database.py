"""SQLAlchemy schema for the shared PostgreSQL operational control plane."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeDecorator


class UtcTimestamp(TypeDecorator[str]):
    """TIMESTAMPTZ storage with the canonical ISO-string application contract."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: str | dt.datetime | None, dialect):
        if value is None or isinstance(value, dt.datetime):
            parsed = value
        else:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed is not None and parsed.tzinfo is None:
            raise ValueError("database timestamps must include a timezone")
        return parsed

    def process_result_value(self, value: dt.datetime | str | None, dialect) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).replace(microsecond=0).isoformat()


metadata = MetaData()


schema_migration = Table(
    "schema_migration",
    metadata,
    Column("version", String(80), primary_key=True),
    Column("applied_at", UtcTimestamp(), nullable=False),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("revision_hash", String(80), nullable=False),
)


def _id_payload_table(name: str) -> Table:
    return Table(
        name,
        metadata,
        Column("id", String(160), primary_key=True),
        Column("created_at", UtcTimestamp(), nullable=False),
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
    Column("observed_at", UtcTimestamp(), nullable=False, index=True),
    Column("status", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
universe = _id_payload_table("universe")
universe_snapshot = Table(
    "universe_snapshot",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("universe_id", ForeignKey("universe.id"), nullable=False, index=True),
    Column("observed_at", UtcTimestamp(), nullable=False, index=True),
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
dataset_bundle = Table(
    "dataset_bundle",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
feature_set = _id_payload_table("feature_set")
feature_manifest = _id_payload_table("feature_manifest")
cost_model_manifest = _id_payload_table("cost_model_manifest")

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
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
)
strategy_identity = Table(
    "strategy_identity",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("behavior_hash", ForeignKey("strategy_definition.id"), nullable=False, index=True),
    Column("submitted_spec", JSON, nullable=False),
    Column("generation_method", String(120), nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("parent_hashes", JSON, nullable=False),
    Column("is_duplicate", Boolean, nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    CheckConstraint("is_duplicate IS NOT NULL", name="ck_strategy_identity_duplicate"),
)
strategy_lineage = _id_payload_table("strategy_lineage")
generation_feedback = _id_payload_table("generation_feedback")
research_thesis = Table(
    "research_thesis",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
    Column("creator_identity", String(160), nullable=False),
    Column("cumulative_trial_budget", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    CheckConstraint("cumulative_trial_budget > 0", name="ck_thesis_trial_budget_positive"),
)
thesis_trial = Table(
    "thesis_trial",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("thesis_id", ForeignKey("research_thesis.id"), nullable=False, index=True),
    Column("candidate_id", String(80), nullable=False, unique=True, index=True),
    Column("lineage_id", String(80), nullable=False, index=True),
    Column("ordinal", Integer, nullable=False),
    Column("claimed_at", UtcTimestamp(), nullable=False),
    UniqueConstraint("thesis_id", "ordinal", name="uq_thesis_trial_ordinal"),
    CheckConstraint("ordinal > 0", name="ck_thesis_trial_ordinal_positive"),
)
experiment = Table(
    "experiment",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("provider", String(120), nullable=False, index=True),
    Column("state", String(80), nullable=False, index=True),
    Column("submitted_at", UtcTimestamp(), nullable=False),
    Column("dataset_snapshot_hashes", JSON, nullable=False),
    Column("dataset_bundle_id", ForeignKey("dataset_bundle.id"), index=True),
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

validation_stage = Table(
    "validation_stage",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("experiment_id", ForeignKey("experiment.id"), nullable=False, index=True),
    Column("stage", String(80), nullable=False, index=True),
    Column("source_run_id", ForeignKey("experiment_run.id"), nullable=False, index=True),
    Column("evaluated_at", UtcTimestamp(), nullable=False, index=True),
    Column("state", String(40), nullable=False),
    Column("accepted", Boolean, nullable=False),
    Column("reason_code", String(160)),
    Column("evidence_hash", String(80), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("experiment_id", "stage", name="uq_validation_stage_experiment_stage"),
    CheckConstraint(
        "accepted IS TRUE OR reason_code IS NOT NULL",
        name="ck_validation_stage_rejection_reason",
    ),
)

holdout_outcome = Table(
    "holdout_outcome",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("holdout_claim_id", ForeignKey("holdout_claim.id"), nullable=False, index=True),
    Column("evaluated_at", UtcTimestamp(), nullable=False, index=True),
    Column("accepted", Boolean, nullable=False),
    Column("outcome_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)

forward_paper_observation = Table(
    "forward_paper_observation",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("instrument_id", String(200), nullable=False, index=True),
    Column("observed_at", UtcTimestamp(), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False),
    Column("observation_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
forward_paper_summary = Table(
    "forward_paper_summary",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False, index=True),
    Column("observed_from", UtcTimestamp(), nullable=False),
    Column("observed_until", UtcTimestamp(), nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
forward_paper_decision = Table(
    "forward_paper_decision",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("summary_id", ForeignKey("forward_paper_summary.id"), nullable=False, index=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False, index=True),
    Column("decided_at", UtcTimestamp(), nullable=False, index=True),
    Column("accepted", Boolean, nullable=False),
    Column("reason_code", String(160)),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
    CheckConstraint(
        "accepted IS TRUE OR reason_code IS NOT NULL",
        name="ck_forward_paper_decision_rejection_reason",
    ),
)

strategy_approval = Table(
    "strategy_approval",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("account_id", String(160), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False),
    Column("source_commit_hash", String(80), nullable=False),
    Column("engine_version", String(160), nullable=False),
    Column("capital_cap", Numeric(24, 12), nullable=False),
    Column("approved_by", String(160), nullable=False),
    Column("approved_at", UtcTimestamp(), nullable=False, index=True),
    Column("status", String(40), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    CheckConstraint("capital_cap >= 0", name="ck_strategy_approval_capital_nonnegative"),
    CheckConstraint(
        "status IN ('approved', 'revoked', 'expired')",
        name="ck_strategy_approval_status",
    ),
    UniqueConstraint(
        "strategy_version_id",
        "product_id",
        "account_id",
        "artefact_hash",
        "approved_at",
        name="uq_strategy_approval_identity",
    ),
)

production_preflight = Table(
    "production_preflight",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("account_id", String(160), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False),
    Column("source_commit_hash", String(80), nullable=False),
    Column("engine_version", String(160), nullable=False),
    Column("capital_cap", Numeric(24, 12), nullable=False),
    Column("checked_at", UtcTimestamp(), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("accepted", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
    CheckConstraint("capital_cap >= 0", name="ck_preflight_capital_nonnegative"),
)

active_strategy_assignment = Table(
    "active_strategy_assignment",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("portfolio_id", String(160), nullable=False, index=True),
    Column("sleeve_id", String(160), nullable=False, default="default", index=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("instrument_id", String(160), index=True),
    Column("universe_id", String(160), index=True),
    Column("assignment_scope_id", String(180), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False),
    Column("lifecycle_state", String(80), nullable=False),
    Column("execution_mode", String(40), nullable=False),
    Column("capital_limit", Numeric(24, 12), nullable=False),
    Column("risk_budget", Numeric(24, 12), nullable=False, default=0),
    Column("assigned_at", UtcTimestamp(), nullable=False, index=True),
    Column("active_until", UtcTimestamp(), index=True),
    Column("assigned_by", String(160), nullable=False),
    Column("assignment_reason", Text, nullable=False, default="unspecified"),
    Column("active", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
    CheckConstraint("execution_mode IN ('paper', 'live')", name="ck_assignment_execution_mode"),
    CheckConstraint("capital_limit >= 0", name="ck_assignment_capital_nonnegative"),
    CheckConstraint("risk_budget >= 0", name="ck_assignment_risk_nonnegative"),
    CheckConstraint(
        "(instrument_id IS NOT NULL AND universe_id IS NULL) OR "
        "(instrument_id IS NULL AND universe_id IS NOT NULL)",
        name="ck_assignment_instrument_xor_universe",
    ),
    CheckConstraint(
        "lifecycle_state IN "
        "('registered', 'development', 'forward_paper', 'live_ready', 'live_canary', 'live', 'suspended', 'retired')",
        name="ck_assignment_lifecycle_state",
    ),
)
Index(
    "ix_active_strategy_assignment_authority_event",
    active_strategy_assignment.c.product_id,
    active_strategy_assignment.c.portfolio_id,
    active_strategy_assignment.c.sleeve_id,
    active_strategy_assignment.c.strategy_version_id,
    active_strategy_assignment.c.assignment_scope_id,
    active_strategy_assignment.c.execution_mode,
    unique=False,
)

import_provenance = Table(
    "import_provenance",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("source_path", Text, nullable=False),
    Column("source_hash", String(80), nullable=False, unique=True),
    Column("destination_hash", String(80), nullable=False),
    Column("imported_at", UtcTimestamp(), nullable=False),
    Column("archived_path", Text),
    Column("payload", JSON, nullable=False),
)

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
risk_policy = _id_payload_table("risk_policy")
risk_decision = Table(
    "risk_decision",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("scope", String(80), nullable=False, index=True),
    Column("evaluated_at", UtcTimestamp(), nullable=False, index=True),
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
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
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
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
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
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
    Column("status", String(80), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("stop_id", "sequence", name="uq_protective_stop_sequence"),
)
fill = Table(
    "fill",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("order_id", ForeignKey("order_intent.id"), nullable=False, index=True),
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
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
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
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
    Column("available_at", UtcTimestamp(), nullable=False, index=True),
    Column("lease_owner", String(160), index=True),
    Column("lease_expires_at", UtcTimestamp(), index=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("max_attempts", Integer, nullable=False, default=3),
    Column("terminal_reason", Text),
    Column("producer_identity", String(200), nullable=False, default="platform"),
    Column("content_hash", String(80), nullable=False),
    Column("payload", JSON, nullable=False),
)
job_attempt = Table(
    "job_attempt",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("job_id", ForeignKey("job.id"), nullable=False, index=True),
    Column("worker_id", String(160), nullable=False, index=True),
    Column("started_at", UtcTimestamp(), nullable=False),
    Column("completed_at", UtcTimestamp()),
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
    Column("last_heartbeat", UtcTimestamp(), nullable=False),
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
    Column("expires_at", UtcTimestamp(), nullable=False, index=True),
    Column("status", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)
heavy_compute_lease = Table(
    "heavy_compute_lease",
    metadata,
    Column("slot_id", String(80), primary_key=True),
    Column("owner", String(160)),
    Column("job_id", ForeignKey("job.id")),
    Column("acquired_at", UtcTimestamp()),
    Column("expires_at", UtcTimestamp(), index=True),
    Column("status", String(40), nullable=False),
)
service_heartbeat = Table(
    "service_heartbeat",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("service_name", String(160), nullable=False, index=True),
    Column("node_id", String(160), nullable=False, index=True),
    Column("observed_at", UtcTimestamp(), nullable=False, index=True),
    Column("healthy", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)
platform_schedule = Table(
    "platform_schedule",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("job_name", String(160), nullable=False),
    Column("interval_seconds", Integer, nullable=False),
    Column("next_run_at", UtcTimestamp(), nullable=False, index=True),
    Column("last_run_at", UtcTimestamp(), index=True),
    Column("last_job_id", String(200)),
    Column("state", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("updated_at", UtcTimestamp(), nullable=False),
    UniqueConstraint("job_name", name="uq_platform_schedule_job_name"),
)
platform_bootstrap = Table(
    "platform_bootstrap",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
account_snapshot = Table(
    "account_snapshot",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("account_id", String(160), nullable=False, index=True),
    Column("observed_at", UtcTimestamp(), nullable=False, index=True),
    Column("source", String(80), nullable=False),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
platform_rehearsal_report = Table(
    "platform_rehearsal_report",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("account_id", String(160), nullable=False, index=True),
    Column("created_at", UtcTimestamp(), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("accepted", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)
alert = _id_payload_table("alert")
control_event = _id_payload_table("control_event")
promotion_event = _id_payload_table("promotion_event")
promotion_policy = _id_payload_table("promotion_policy")
decision_trace = Table(
    "decision_trace",
    metadata,
    Column("id", String(200), primary_key=True),
    Column("event_id", String(240), nullable=False, index=True),
    Column("instrument_id", String(200), nullable=False, index=True),
    Column("evaluated_at", UtcTimestamp(), nullable=False, index=True),
    Column("first_blocked_stage", String(120), index=True),
    Column("payload", JSON, nullable=False),
)


class PlatformDatabase:
    """Explicit database lifecycle; production URLs must use PostgreSQL.

    ``create_schema`` is intentionally limited to SQLite fixtures used by unit
    tests. Production services must call ``migrate`` so schema changes are
    versioned and recorded in the database.
    """

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
        if self.is_postgresql:
            raise RuntimeError(
                "PostgreSQL schemas are migration-managed; call PlatformDatabase.migrate()"
            )
        metadata.create_all(self.engine)

    def migrate(self, *, target: str | None = None) -> tuple[str, ...]:
        """Apply Alembic in production; initialise isolated SQLite tests locally."""

        if not self.is_postgresql:
            import datetime as dt
            import hashlib
            from pathlib import Path

            from sqlalchemy import insert, select

            metadata.create_all(self.engine)
            revisions = (
                "001_platform_v2_baseline.py",
                "002_platform_v2_authority.py",
                "003_platform_expansion_authority.py",
                "004_research_thesis_authority.py",
                "005_platform_bootstrap_authority.py",
                "006_research_dataset_bundles.py",
                "007_bounded_job_retries.py",
                "008_forward_paper_summaries.py",
                "009_generation_feedback.py",
                "010_live_ready_lifecycle.py",
            )
            applied: list[str] = []
            with self.engine.begin() as connection:
                for filename in revisions:
                    revision = filename.split("_", 1)[0]
                    if target is not None and target not in {revision, filename}:
                        continue
                    exists = connection.execute(
                        select(schema_migration.c.version).where(
                            schema_migration.c.version == revision
                        )
                    ).first()
                    if exists:
                        continue
                    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / filename
                    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    connection.execute(
                        insert(schema_migration).values(
                            version=revision,
                            applied_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                            content_hash=digest,
                            revision_hash=digest,
                        )
                    )
                    applied.append(revision)
            return tuple(applied)

        from pathlib import Path

        from alembic import command
        from alembic.config import Config

        config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        command.upgrade(config, target or "head")
        return (target or "head",)

    def assert_migrated(self) -> None:
        """Fail closed when a production database has not run migrations."""

        from sqlalchemy import inspect, text

        tables = set(inspect(self.engine).get_table_names())
        missing = sorted(CORE_TABLE_NAMES - tables)
        if missing:
            raise RuntimeError(f"database migrations are incomplete; missing tables: {missing}")
        if self.is_postgresql:
            with self.engine.connect() as connection:
                revision = connection.execute(text("SELECT version_num FROM alembic_version"))
                if revision.scalar_one_or_none() != "platform_v2_0010":
                    raise RuntimeError("database is not at the current Alembic revision")

    def dispose(self) -> None:
        self.engine.dispose()


CORE_TABLE_NAMES = frozenset(metadata.tables)
