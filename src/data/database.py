"""SQLAlchemy schema for the shared PostgreSQL operational control plane."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
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
    literal_column,
    select,
    union_all,
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
        return value.astimezone(dt.UTC).isoformat()


class BookAmount(TypeDecorator[Decimal]):
    """Native PostgreSQL decimals; lossless text in the SQLite development adapter."""

    impl = Numeric()
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(Text() if dialect.name == "sqlite" else self.impl)

    def process_bind_param(self, value, dialect):
        amount = Decimal(str(value))
        if not amount.is_finite():
            raise ValueError("book amount must be finite")
        return str(amount) if dialect.name == "sqlite" else amount

    def process_result_value(self, value, dialect):
        return Decimal(str(value))


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
    Column("behavior_hash", String(80), nullable=False, index=True),
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
research_epoch = Table(
    "research_epoch",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("dataset_bundle_id", ForeignKey("dataset_bundle.id"), nullable=False, unique=True),
    Column("state", String(40), nullable=False),
    Column("round_cursor", Integer, nullable=False),
    Column("budget", Integer, nullable=False),
    Column("consumed", Integer, nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    CheckConstraint(
        "round_cursor >= 0 AND consumed >= 0 AND budget >= consumed",
        name="ck_research_epoch_budget",
    ),
)
research_round = Table(
    "research_round",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("epoch_id", ForeignKey("research_epoch.id"), nullable=False),
    Column("round_number", Integer, nullable=False),
    Column("state", String(40), nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("epoch_id", "round_number", name="uq_research_round_cursor"),
)
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
capital_baseline = _id_payload_table("capital_baseline")
evidence_invalidation = _id_payload_table("evidence_invalidation")
financial_application = Table(
    "financial_application",
    metadata,
    Column("id", ForeignKey("fill.id"), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("account_id", String(160), nullable=False),
    Column("environment", String(40), nullable=False),
    Column("venue", String(40), nullable=False),
    Column("instrument_id", String(160), nullable=False),
    Column("exchange_trade_id", String(240), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint(
        "account_id",
        "environment",
        "venue",
        "instrument_id",
        "exchange_trade_id",
        name="uq_financial_scoped_fill",
    ),
)
financial_outbox = Table(
    "financial_outbox",
    metadata,
    Column("id", ForeignKey("financial_application.id"), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("processed_at", UtcTimestamp()),
    Column("payload", JSON, nullable=False),
)

paper_eligibility_event = Table(
    "paper_eligibility_event",
    metadata,
    Column("id", ForeignKey("strategy_artefact.id"), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("candidate_id", ForeignKey("experiment.id"), nullable=False),
    Column("engine_version", String(160), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint(
        "candidate_id", "engine_version", name="uq_paper_eligibility_candidate_engine"
    ),
)
selection_cohort = Table(
    "selection_cohort",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("scope_id", String(160), nullable=False, unique=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
)
selection_cohort_result = Table(
    "selection_cohort_result",
    metadata,
    Column("id", ForeignKey("selection_cohort.id"), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
)
selection_cohort_open = Table(
    "selection_cohort_open",
    metadata,
    Column("id", ForeignKey("selection_cohort.id"), primary_key=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
)

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
    Column("evaluation_identity", String(80), unique=True, index=True),
    Column("engine_version", String(160)),
    Column("outcome_category", String(40)),
    Column("payload", JSON, nullable=False),
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

forward_qualification_window = Table(
    "forward_qualification_window",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("artefact_hash", String(80), nullable=False, index=True),
    Column("opened_at", UtcTimestamp(), nullable=False, index=True),
    Column("previous_window_id", ForeignKey("forward_qualification_window.id"), unique=True),
    Column("source_event_id", String(160), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
)
Index(
    "uq_forward_window_initial_scope",
    forward_qualification_window.c.strategy_version_id,
    forward_qualification_window.c.product_id,
    forward_qualification_window.c.artefact_hash,
    unique=True,
    postgresql_where=forward_qualification_window.c.previous_window_id.is_(None),
    sqlite_where=forward_qualification_window.c.previous_window_id.is_(None),
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

research_calibration = _id_payload_table("research_calibration")
research_policy_approval = _id_payload_table("research_policy_approval")

legacy_research_archive = Table(
    "legacy_research_archive",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("source_hash", String(80), nullable=False, unique=True),
    Column("content_hash", String(80), nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
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
Index("ix_risk_snapshot_created", risk_snapshot.c.created_at.desc(), risk_snapshot.c.id.desc())
Index(
    "ix_risk_snapshot_market_scope",
    risk_snapshot.c.payload["kind"].as_string(),
    risk_snapshot.c.payload["product_id"].as_string(),
    risk_snapshot.c.payload["instrument_id"].as_string(),
    risk_snapshot.c.created_at.desc(),
)
Index(
    "ix_risk_snapshot_product_time",
    risk_snapshot.c.payload["kind"].as_string(),
    risk_snapshot.c.payload["product_id"].as_string(),
    risk_snapshot.c.created_at.desc(),
)
Index(
    "ix_risk_snapshot_candle_scope",
    risk_snapshot.c.payload["kind"].as_string(),
    risk_snapshot.c.payload["product_id"].as_string(),
    risk_snapshot.c.payload["source_event_type"].as_string(),
    risk_snapshot.c.payload["source_timeframe"].as_string(),
    risk_snapshot.c.payload["instrument_id"].as_string(),
    risk_snapshot.c.created_at,
)
market_input_buffer = Table(
    "market_input_buffer",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("product_id", String(80), nullable=False),
    Column("instrument_id", String(160), nullable=False),
    Column("event_type", String(80), nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    CheckConstraint(
        "event_type IN ('best_bid_ask', 'depth_update', 'mark_price', 'funding_rate', "
        "'trade', 'aggregate_trade', 'candle')",
        name="ck_market_input_buffer_event_type",
    ),
    CheckConstraint(
        "json_extract(payload, '$.kind') IS 'market_data_input'",
        name="ck_market_input_buffer_kind",
    ),
    CheckConstraint(
        "json_extract(payload, '$.source_event_type') IS event_type",
        name="ck_market_input_buffer_source_type",
    ),
    CheckConstraint(
        "event_type <> 'candle' OR "
        "COALESCE(json_extract(payload, '$.source_candle_closed'), 0) = 0",
        name="ck_market_input_buffer_unclosed_candle",
    ),
    CheckConstraint(
        "json_extract(payload, '$.product_id') IS product_id AND "
        "json_extract(payload, '$.instrument_id') IS instrument_id",
        name="ck_market_input_buffer_scope",
    ),
)
Index(
    "ix_market_input_buffer_scope_time",
    market_input_buffer.c.product_id,
    market_input_buffer.c.instrument_id,
    market_input_buffer.c.created_at.desc(),
    market_input_buffer.c.id.desc(),
)
Index("ix_market_input_buffer_time", market_input_buffer.c.created_at)
market_input_sources = union_all(
    select(risk_snapshot.c.id, risk_snapshot.c.created_at, risk_snapshot.c.payload).where(
        risk_snapshot.c.payload["kind"].as_string() == "market_data_input"
    ),
    select(
        market_input_buffer.c.id,
        market_input_buffer.c.created_at,
        market_input_buffer.c.payload,
    ).where(market_input_buffer.c.payload["kind"].as_string() == "market_data_input"),
).subquery("market_input_sources")
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
strategy_book = Table(
    "strategy_book",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("artefact_hash", ForeignKey("strategy_artefact.id"), nullable=False),
    Column("strategy_version_id", ForeignKey("strategy_version.id"), nullable=False),
    Column("execution_mode", String(40), nullable=False),
    Column("sleeve_id", String(160), nullable=False),
    Column("account_id", String(160), nullable=False, index=True),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint(
        "artefact_hash", "execution_mode", "sleeve_id", name="uq_strategy_book_artefact_mode"
    ),
    CheckConstraint("execution_mode IN ('paper', 'live')", name="ck_strategy_book_mode"),
)
book_event = _id_payload_table("book_event")
book_fill_allocation = Table(
    "book_fill_allocation",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("fill_id", ForeignKey("fill.id"), nullable=False, index=True),
    Column("book_id", ForeignKey("strategy_book.id"), nullable=False, index=True),
    Column("signed_quantity", BookAmount(), nullable=False),
    Column("fee", BookAmount(), nullable=False),
    Column("fee_asset", String(40), nullable=False),
    Column("created_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("fill_id", "book_id", name="uq_book_fill_allocation"),
    CheckConstraint(
        "CAST(signed_quantity AS NUMERIC) <> 0 AND CAST(fee AS NUMERIC) >= 0",
        name="ck_book_fill_amounts",
    ),
    CheckConstraint(
        "CAST(signed_quantity AS TEXT) NOT IN ('NaN','Infinity','-Infinity') "
        "AND CAST(fee AS TEXT) NOT IN ('NaN','Infinity','-Infinity')",
        name="ck_book_fill_finite",
    ),
)
account_portfolio_plan = Table(
    "account_portfolio_plan",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("product_id", String(80), nullable=False),
    Column("account_id", String(160), nullable=False),
    Column("decision_at", UtcTimestamp(), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("product_id", "account_id", "decision_at", name="uq_account_decision_cutoff"),
)
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
Index("ix_job_state_name_available", job.c.state, job.c.name, job.c.available_at)
Index("ix_job_state_available_id", job.c.state, job.c.available_at, job.c.id)
Index(
    "ix_job_scheduled_prefix",
    job.c.id,
    postgresql_where=literal_column("id LIKE 'scheduled:%'"),
    sqlite_where=literal_column("id LIKE 'scheduled:%'"),
)
job_quarantine = Table(
    "job_quarantine",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("job_id", ForeignKey("job.id"), nullable=False, index=True),
    Column("quarantined_at", UtcTimestamp(), nullable=False, index=True),
    Column("original_state", String(40), nullable=False),
    Column("reason", String(160), nullable=False),
    Column("content_hash", String(80), nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("job_id", name="uq_job_quarantine_job"),
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
    Column(
        "outcome_category",
        Text,
        Computed(
            "COALESCE(payload->>'outcome_category', "
            "CASE WHEN status IN ('failed','expired') THEN 'blocked_infrastructure' "
            "ELSE 'unclassified' END)",
            persisted=True,
        ),
        nullable=False,
    ),
    Index("ix_job_attempt_status_started", "status", "started_at"),
)
Index("ix_job_attempt_outcome_category", job_attempt.c.outcome_category)
Index(
    "ix_job_attempt_scheduled_prefix",
    job_attempt.c.job_id,
    postgresql_where=literal_column("job_id LIKE 'scheduled:%'"),
    sqlite_where=literal_column("job_id LIKE 'scheduled:%'"),
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
platform_upgrade_marker = _id_payload_table("platform_upgrade_marker")
emergency_action = Table(
    "emergency_action",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("job_id", ForeignKey("job.id"), nullable=False, unique=True, index=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("state", String(80), nullable=False, index=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("updated_at", UtcTimestamp(), nullable=False, index=True),
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
historical_bootstrap_progress = Table(
    "historical_bootstrap_progress",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("product_id", String(80), nullable=False, index=True),
    Column("market", String(40), nullable=False),
    Column("symbol", String(80), nullable=False),
    Column("timeframe", String(40), nullable=False),
    Column("range_start", UtcTimestamp(), nullable=False),
    Column("range_end", UtcTimestamp(), nullable=False),
    Column("next_start", UtcTimestamp(), nullable=False),
    Column("status", String(40), nullable=False, index=True),
    Column("updated_at", UtcTimestamp(), nullable=False, index=True),
    Column("content_hash", String(80), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
    UniqueConstraint(
        "product_id",
        "market",
        "symbol",
        "timeframe",
        "range_start",
        "range_end",
        name="uq_historical_bootstrap_range",
    ),
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
alert_delivery_attempt = Table(
    "alert_delivery_attempt",
    metadata,
    Column("id", String(240), primary_key=True),
    Column("alert_id", ForeignKey("alert.id"), nullable=False, index=True),
    Column("sink", String(120), nullable=False),
    Column("attempt_no", Integer, nullable=False),
    Column("attempted_at", UtcTimestamp(), nullable=False, index=True),
    Column("status", String(40), nullable=False, index=True),
    Column("error", Text),
    Column("next_attempt_at", UtcTimestamp(), index=True),
    Column("acknowledged_at", UtcTimestamp()),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("alert_id", "sink", "attempt_no", name="uq_alert_delivery_attempt"),
)
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
        self.engine: Engine = create_engine(
            self.url, echo=echo, future=True, pool_pre_ping=True, hide_parameters=True
        )

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
                "011_behavior_hash_identity.py",
                "012_quarantine_invalid_candidates.py",
                "013_quarantine_invalid_btc_scope.py",
                "014_operational_durability.py",
                "015_legacy_job_quarantine.py",
                "016_emergency_action_state.py",
                "017_evaluation_identity.py",
                "018_capital_baseline.py",
                "019_economic_evidence_invalidation.py",
                "020_strategy_state_index.py",
                "021_paper_eligibility.py",
                "022_financial_application.py",
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
        config.attributes["database_url"] = self.url
        config.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        command.upgrade(config, target or "head")
        return (target or "head",)

    def assert_migrated(self) -> None:
        """Fail closed when a production database has not run migrations."""

        from pathlib import Path

        from sqlalchemy import inspect, text

        tables = set(inspect(self.engine).get_table_names())
        missing = sorted(CORE_TABLE_NAMES - tables)
        if missing:
            raise RuntimeError(f"database migrations are incomplete; missing tables: {missing}")
        if self.is_postgresql:
            from alembic.config import Config
            from alembic.script import ScriptDirectory

            config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
            expected_revision = ScriptDirectory.from_config(config).get_current_head()
            with self.engine.connect() as connection:
                revision = connection.execute(text("SELECT version_num FROM alembic_version"))
                if revision.scalar_one_or_none() != expected_revision:
                    raise RuntimeError("database is not at the current Alembic revision")

    def dispose(self) -> None:
        self.engine.dispose()


CORE_TABLE_NAMES = frozenset(metadata.tables)
