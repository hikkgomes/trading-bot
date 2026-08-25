"""Complete PostgreSQL platform schema owned by Alembic."""

from alembic import op

revision = "platform_v2_0001"
down_revision = None
branch_labels = None
depends_on = None


SCHEMA_SQL = r"""

CREATE TABLE account (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE accounting_entry (
	id VARCHAR(200) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	sequence INTEGER NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	entry_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_accounting_entry_sequence UNIQUE (product_id, sequence), 
	UNIQUE (entry_hash)
);


CREATE TABLE agent_action (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE agent_disposition (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE agent_patch (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE agent_proposal (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE agent_review (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE alert (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE alpha_forecast (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE balance_snapshot (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE control_event (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE dataset_snapshot (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE decision_trace (
	id VARCHAR(200) NOT NULL, 
	event_id VARCHAR(240) NOT NULL, 
	instrument_id VARCHAR(200) NOT NULL, 
	evaluated_at TIMESTAMPTZ NOT NULL, 
	first_blocked_stage VARCHAR(120), 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE experiment_metric (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE experiment_run (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE feature_manifest (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE feature_set (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE fee_entry (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE forward_evidence (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE funding_entry (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE holdout_claim (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE import_provenance (
	id VARCHAR(200) NOT NULL, 
	source_path TEXT NOT NULL, 
	source_hash VARCHAR(80) NOT NULL, 
	destination_hash VARCHAR(80) NOT NULL, 
	imported_at TIMESTAMPTZ NOT NULL, 
	archived_path TEXT, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (source_hash)
);


CREATE TABLE instrument (
	id VARCHAR(160) NOT NULL, 
	venue VARCHAR(40) NOT NULL, 
	market_type VARCHAR(40) NOT NULL, 
	exchange_symbol VARCHAR(80) NOT NULL, 
	base_asset VARCHAR(40) NOT NULL, 
	quote_asset VARCHAR(40) NOT NULL, 
	settlement_asset VARCHAR(40), 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE job (
	id VARCHAR(160) NOT NULL, 
	name VARCHAR(160) NOT NULL, 
	state VARCHAR(40) NOT NULL, 
	priority INTEGER NOT NULL, 
	available_at TIMESTAMPTZ NOT NULL, 
	lease_owner VARCHAR(160), 
	lease_expires_at TIMESTAMPTZ, 
	attempts INTEGER NOT NULL, 
	producer_identity VARCHAR(200) NOT NULL, 
	content_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE model_artifact (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE nav_snapshot (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE order_group (
	id VARCHAR(240) NOT NULL, 
	group_id VARCHAR(160) NOT NULL, 
	sequence INTEGER NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_order_group_sequence UNIQUE (group_id, sequence)
);


CREATE TABLE order_intent (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE portfolio (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE portfolio_sleeve (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE portfolio_strategy (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE position (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE position_event (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE promotion_event (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE promotion_policy (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE protective_stop (
	id VARCHAR(240) NOT NULL, 
	stop_id VARCHAR(160) NOT NULL, 
	sequence INTEGER NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_protective_stop_sequence UNIQUE (stop_id, sequence)
);


CREATE TABLE reconciliation_event (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE risk_decision (
	id VARCHAR(160) NOT NULL, 
	scope VARCHAR(80) NOT NULL, 
	evaluated_at TIMESTAMPTZ NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	reason_code VARCHAR(160), 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE risk_policy (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE risk_snapshot (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE schema_migration (
	version VARCHAR(80) NOT NULL, 
	applied_at TIMESTAMPTZ NOT NULL, 
	content_hash VARCHAR(80) NOT NULL, 
	revision_hash VARCHAR(80) NOT NULL, 
	PRIMARY KEY (version), 
	UNIQUE (content_hash)
);


CREATE TABLE service_heartbeat (
	id VARCHAR(160) NOT NULL, 
	service_name VARCHAR(160) NOT NULL, 
	node_id VARCHAR(160) NOT NULL, 
	observed_at TIMESTAMPTZ NOT NULL, 
	healthy BOOLEAN NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE strategy_artefact (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE strategy_definition (
	id VARCHAR(80) NOT NULL, 
	identity VARCHAR(160) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	source_type VARCHAR(80) NOT NULL, 
	source_hash VARCHAR(80) NOT NULL, 
	definition JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE strategy_lineage (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE target_position (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE trade_attribution (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE universe (
	id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE worker (
	id VARCHAR(160) NOT NULL, 
	node_id VARCHAR(160) NOT NULL, 
	role VARCHAR(120) NOT NULL, 
	last_heartbeat TIMESTAMPTZ NOT NULL, 
	status VARCHAR(40) NOT NULL, 
	capabilities JSONB NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE exchange_order (
	id VARCHAR(240) NOT NULL, 
	order_id VARCHAR(160) NOT NULL, 
	sequence INTEGER NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_exchange_order_sequence UNIQUE (order_id, sequence), 
	FOREIGN KEY(order_id) REFERENCES order_intent (id)
);


CREATE TABLE fill (
	id VARCHAR(240) NOT NULL, 
	order_id VARCHAR(160) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(order_id) REFERENCES order_intent (id)
);


CREATE TABLE heavy_compute_lease (
	slot_id VARCHAR(80) NOT NULL, 
	owner VARCHAR(160), 
	job_id VARCHAR(160), 
	acquired_at TIMESTAMPTZ, 
	expires_at TIMESTAMPTZ, 
	status VARCHAR(40) NOT NULL, 
	PRIMARY KEY (slot_id), 
	FOREIGN KEY(job_id) REFERENCES job (id)
);


CREATE TABLE holdout_outcome (
	id VARCHAR(200) NOT NULL, 
	holdout_claim_id VARCHAR(160) NOT NULL, 
	evaluated_at TIMESTAMPTZ NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	outcome_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(holdout_claim_id) REFERENCES holdout_claim (id), 
	UNIQUE (outcome_hash)
);


CREATE TABLE instrument_status (
	id VARCHAR(160) NOT NULL, 
	instrument_id VARCHAR(160) NOT NULL, 
	observed_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(40) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(instrument_id) REFERENCES instrument (id)
);


CREATE TABLE job_attempt (
	id VARCHAR(200) NOT NULL, 
	job_id VARCHAR(160) NOT NULL, 
	worker_id VARCHAR(160) NOT NULL, 
	started_at TIMESTAMPTZ NOT NULL, 
	completed_at TIMESTAMPTZ, 
	status VARCHAR(40) NOT NULL, 
	error TEXT, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(job_id) REFERENCES job (id)
);


CREATE TABLE strategy_identity (
	id VARCHAR(160) NOT NULL, 
	behavior_hash VARCHAR(80) NOT NULL, 
	submitted_spec JSONB NOT NULL, 
	generation_method VARCHAR(120) NOT NULL, 
	metadata JSONB NOT NULL, 
	parent_hashes JSONB NOT NULL, 
	is_duplicate BOOLEAN NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_strategy_identity_duplicate CHECK (is_duplicate IS NOT NULL), 
	FOREIGN KEY(behavior_hash) REFERENCES strategy_definition (id)
);


CREATE TABLE strategy_version (
	id VARCHAR(160) NOT NULL, 
	definition_id VARCHAR(80) NOT NULL, 
	version VARCHAR(80) NOT NULL, 
	created_at TIMESTAMPTZ NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(definition_id) REFERENCES strategy_definition (id)
);


CREATE TABLE universe_snapshot (
	id VARCHAR(160) NOT NULL, 
	universe_id VARCHAR(160) NOT NULL, 
	observed_at TIMESTAMPTZ NOT NULL, 
	content_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(universe_id) REFERENCES universe (id), 
	UNIQUE (content_hash)
);


CREATE TABLE worker_lease (
	id VARCHAR(200) NOT NULL, 
	job_id VARCHAR(160) NOT NULL, 
	worker_id VARCHAR(160) NOT NULL, 
	expires_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(40) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(job_id) REFERENCES job (id), 
	FOREIGN KEY(worker_id) REFERENCES worker (id)
);


CREATE TABLE active_strategy_assignment (
	id VARCHAR(200) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	portfolio_id VARCHAR(160) NOT NULL, 
	sleeve_id VARCHAR(160) NOT NULL, 
	strategy_version_id VARCHAR(160) NOT NULL, 
	instrument_id VARCHAR(160), 
	universe_id VARCHAR(160), 
	assignment_scope_id VARCHAR(180) NOT NULL, 
	artefact_hash VARCHAR(80) NOT NULL, 
	lifecycle_state VARCHAR(80) NOT NULL, 
	execution_mode VARCHAR(40) NOT NULL, 
	capital_limit NUMERIC(24, 12) NOT NULL, 
	risk_budget NUMERIC(24, 12) NOT NULL, 
	assigned_at TIMESTAMPTZ NOT NULL, 
	active_until TIMESTAMPTZ, 
	assigned_by VARCHAR(160) NOT NULL, 
	assignment_reason TEXT NOT NULL, 
	active BOOLEAN NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_assignment_execution_mode CHECK (execution_mode IN ('paper', 'live')), 
	CONSTRAINT ck_assignment_capital_nonnegative CHECK (capital_limit >= 0), 
	CONSTRAINT ck_assignment_risk_nonnegative CHECK (risk_budget >= 0), 
	CONSTRAINT ck_assignment_instrument_xor_universe CHECK ((instrument_id IS NOT NULL AND universe_id IS NULL) OR (instrument_id IS NULL AND universe_id IS NOT NULL)), 
	CONSTRAINT ck_assignment_lifecycle_state CHECK (lifecycle_state IN ('registered', 'development', 'forward_paper', 'live_canary', 'live', 'suspended', 'retired')), 
	FOREIGN KEY(strategy_version_id) REFERENCES strategy_version (id)
);


CREATE TABLE experiment (
	id VARCHAR(80) NOT NULL, 
	strategy_version_id VARCHAR(160) NOT NULL, 
	provider VARCHAR(120) NOT NULL, 
	state VARCHAR(80) NOT NULL, 
	submitted_at TIMESTAMPTZ NOT NULL, 
	dataset_snapshot_hashes JSONB NOT NULL, 
	metadata JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(strategy_version_id) REFERENCES strategy_version (id)
);


CREATE TABLE forward_paper_observation (
	id VARCHAR(200) NOT NULL, 
	strategy_version_id VARCHAR(160) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	instrument_id VARCHAR(200) NOT NULL, 
	observed_at TIMESTAMPTZ NOT NULL, 
	artefact_hash VARCHAR(80) NOT NULL, 
	observation_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(strategy_version_id) REFERENCES strategy_version (id), 
	UNIQUE (observation_hash)
);


CREATE TABLE production_preflight (
	id VARCHAR(200) NOT NULL, 
	strategy_version_id VARCHAR(160) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	account_id VARCHAR(160) NOT NULL, 
	artefact_hash VARCHAR(80) NOT NULL, 
	source_commit_hash VARCHAR(80) NOT NULL, 
	engine_version VARCHAR(160) NOT NULL, 
	capital_cap NUMERIC(24, 12) NOT NULL, 
	checked_at TIMESTAMPTZ NOT NULL, 
	content_hash VARCHAR(80) NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_preflight_capital_nonnegative CHECK (capital_cap >= 0), 
	FOREIGN KEY(strategy_version_id) REFERENCES strategy_version (id), 
	UNIQUE (content_hash)
);


CREATE TABLE strategy_approval (
	id VARCHAR(200) NOT NULL, 
	strategy_version_id VARCHAR(160) NOT NULL, 
	product_id VARCHAR(80) NOT NULL, 
	account_id VARCHAR(160) NOT NULL, 
	artefact_hash VARCHAR(80) NOT NULL, 
	source_commit_hash VARCHAR(80) NOT NULL, 
	engine_version VARCHAR(160) NOT NULL, 
	capital_cap NUMERIC(24, 12) NOT NULL, 
	approved_by VARCHAR(160) NOT NULL, 
	approved_at TIMESTAMPTZ NOT NULL, 
	status VARCHAR(40) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_strategy_approval_capital_nonnegative CHECK (capital_cap >= 0), 
	CONSTRAINT ck_strategy_approval_status CHECK (status IN ('approved', 'revoked', 'expired')), 
	CONSTRAINT uq_strategy_approval_identity UNIQUE (strategy_version_id, product_id, account_id, artefact_hash, approved_at), 
	FOREIGN KEY(strategy_version_id) REFERENCES strategy_version (id)
);


CREATE TABLE universe_member (
	id VARCHAR(240) NOT NULL, 
	snapshot_id VARCHAR(160) NOT NULL, 
	instrument_id VARCHAR(160) NOT NULL, 
	eligible BOOLEAN NOT NULL, 
	reason_code VARCHAR(120), 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(snapshot_id) REFERENCES universe_snapshot (id), 
	FOREIGN KEY(instrument_id) REFERENCES instrument (id)
);


CREATE TABLE validation_result (
	id VARCHAR(160) NOT NULL, 
	experiment_id VARCHAR(80) NOT NULL, 
	state VARCHAR(80) NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	reason_code VARCHAR(160), 
	evidence JSONB NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(experiment_id) REFERENCES experiment (id)
);


CREATE TABLE validation_stage (
	id VARCHAR(200) NOT NULL, 
	experiment_id VARCHAR(80) NOT NULL, 
	stage VARCHAR(80) NOT NULL, 
	source_run_id VARCHAR(160) NOT NULL, 
	evaluated_at TIMESTAMPTZ NOT NULL, 
	state VARCHAR(40) NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	reason_code VARCHAR(160), 
	evidence_hash VARCHAR(80) NOT NULL, 
	payload JSONB NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_validation_stage_experiment_stage UNIQUE (experiment_id, stage), 
	CONSTRAINT ck_validation_stage_rejection_reason CHECK (accepted IS TRUE OR reason_code IS NOT NULL), 
	FOREIGN KEY(experiment_id) REFERENCES experiment (id), 
	FOREIGN KEY(source_run_id) REFERENCES experiment_run (id)
);

CREATE INDEX ix_accounting_entry_created_at ON accounting_entry (created_at);

CREATE INDEX ix_accounting_entry_product_id ON accounting_entry (product_id);

CREATE INDEX ix_decision_trace_evaluated_at ON decision_trace (evaluated_at);

CREATE INDEX ix_decision_trace_event_id ON decision_trace (event_id);

CREATE INDEX ix_decision_trace_first_blocked_stage ON decision_trace (first_blocked_stage);

CREATE INDEX ix_decision_trace_instrument_id ON decision_trace (instrument_id);

CREATE INDEX ix_job_available_at ON job (available_at);

CREATE INDEX ix_job_lease_expires_at ON job (lease_expires_at);

CREATE INDEX ix_job_lease_owner ON job (lease_owner);

CREATE INDEX ix_job_name ON job (name);

CREATE INDEX ix_job_state ON job (state);

CREATE INDEX ix_order_group_created_at ON order_group (created_at);

CREATE INDEX ix_order_group_group_id ON order_group (group_id);

CREATE INDEX ix_order_group_status ON order_group (status);

CREATE INDEX ix_protective_stop_created_at ON protective_stop (created_at);

CREATE INDEX ix_protective_stop_status ON protective_stop (status);

CREATE INDEX ix_protective_stop_stop_id ON protective_stop (stop_id);

CREATE INDEX ix_risk_decision_evaluated_at ON risk_decision (evaluated_at);

CREATE INDEX ix_risk_decision_scope ON risk_decision (scope);

CREATE INDEX ix_service_heartbeat_node_id ON service_heartbeat (node_id);

CREATE INDEX ix_service_heartbeat_observed_at ON service_heartbeat (observed_at);

CREATE INDEX ix_service_heartbeat_service_name ON service_heartbeat (service_name);

CREATE INDEX ix_strategy_definition_identity ON strategy_definition (identity);

CREATE INDEX ix_strategy_definition_product_id ON strategy_definition (product_id);

CREATE INDEX ix_worker_node_id ON worker (node_id);

CREATE INDEX ix_worker_role ON worker (role);

CREATE INDEX ix_exchange_order_created_at ON exchange_order (created_at);

CREATE INDEX ix_exchange_order_order_id ON exchange_order (order_id);

CREATE INDEX ix_exchange_order_status ON exchange_order (status);

CREATE INDEX ix_fill_created_at ON fill (created_at);

CREATE INDEX ix_fill_order_id ON fill (order_id);

CREATE INDEX ix_heavy_compute_lease_expires_at ON heavy_compute_lease (expires_at);

CREATE INDEX ix_holdout_outcome_evaluated_at ON holdout_outcome (evaluated_at);

CREATE INDEX ix_holdout_outcome_holdout_claim_id ON holdout_outcome (holdout_claim_id);

CREATE INDEX ix_instrument_status_instrument_id ON instrument_status (instrument_id);

CREATE INDEX ix_instrument_status_observed_at ON instrument_status (observed_at);

CREATE INDEX ix_job_attempt_job_id ON job_attempt (job_id);

CREATE INDEX ix_job_attempt_worker_id ON job_attempt (worker_id);

CREATE INDEX ix_strategy_identity_behavior_hash ON strategy_identity (behavior_hash);

CREATE INDEX ix_strategy_version_definition_id ON strategy_version (definition_id);

CREATE INDEX ix_universe_snapshot_observed_at ON universe_snapshot (observed_at);

CREATE INDEX ix_universe_snapshot_universe_id ON universe_snapshot (universe_id);

CREATE INDEX ix_worker_lease_expires_at ON worker_lease (expires_at);

CREATE INDEX ix_worker_lease_job_id ON worker_lease (job_id);

CREATE INDEX ix_worker_lease_worker_id ON worker_lease (worker_id);

CREATE INDEX ix_active_strategy_assignment_active_until ON active_strategy_assignment (active_until);

CREATE INDEX ix_active_strategy_assignment_assigned_at ON active_strategy_assignment (assigned_at);

CREATE INDEX ix_active_strategy_assignment_assignment_scope_id ON active_strategy_assignment (assignment_scope_id);

CREATE INDEX ix_active_strategy_assignment_authority_event ON active_strategy_assignment (product_id, portfolio_id, sleeve_id, strategy_version_id, assignment_scope_id, execution_mode);

CREATE INDEX ix_active_strategy_assignment_instrument_id ON active_strategy_assignment (instrument_id);

CREATE INDEX ix_active_strategy_assignment_portfolio_id ON active_strategy_assignment (portfolio_id);

CREATE INDEX ix_active_strategy_assignment_product_id ON active_strategy_assignment (product_id);

CREATE INDEX ix_active_strategy_assignment_sleeve_id ON active_strategy_assignment (sleeve_id);

CREATE INDEX ix_active_strategy_assignment_strategy_version_id ON active_strategy_assignment (strategy_version_id);

CREATE INDEX ix_active_strategy_assignment_universe_id ON active_strategy_assignment (universe_id);

CREATE INDEX ix_experiment_provider ON experiment (provider);

CREATE INDEX ix_experiment_state ON experiment (state);

CREATE INDEX ix_experiment_strategy_version_id ON experiment (strategy_version_id);

CREATE INDEX ix_forward_paper_observation_instrument_id ON forward_paper_observation (instrument_id);

CREATE INDEX ix_forward_paper_observation_observed_at ON forward_paper_observation (observed_at);

CREATE INDEX ix_forward_paper_observation_product_id ON forward_paper_observation (product_id);

CREATE INDEX ix_forward_paper_observation_strategy_version_id ON forward_paper_observation (strategy_version_id);

CREATE INDEX ix_production_preflight_account_id ON production_preflight (account_id);

CREATE INDEX ix_production_preflight_checked_at ON production_preflight (checked_at);

CREATE INDEX ix_production_preflight_product_id ON production_preflight (product_id);

CREATE INDEX ix_production_preflight_strategy_version_id ON production_preflight (strategy_version_id);

CREATE INDEX ix_strategy_approval_account_id ON strategy_approval (account_id);

CREATE INDEX ix_strategy_approval_approved_at ON strategy_approval (approved_at);

CREATE INDEX ix_strategy_approval_product_id ON strategy_approval (product_id);

CREATE INDEX ix_strategy_approval_status ON strategy_approval (status);

CREATE INDEX ix_strategy_approval_strategy_version_id ON strategy_approval (strategy_version_id);

CREATE INDEX ix_universe_member_instrument_id ON universe_member (instrument_id);

CREATE INDEX ix_universe_member_snapshot_id ON universe_member (snapshot_id);

CREATE INDEX ix_validation_result_experiment_id ON validation_result (experiment_id);

CREATE INDEX ix_validation_stage_evaluated_at ON validation_stage (evaluated_at);

CREATE INDEX ix_validation_stage_experiment_id ON validation_stage (experiment_id);

CREATE INDEX ix_validation_stage_source_run_id ON validation_stage (source_run_id);

CREATE INDEX ix_validation_stage_stage ON validation_stage (stage);
"""


def upgrade() -> None:
    op.execute(SCHEMA_SQL)


def downgrade() -> None:
    raise RuntimeError("platform baseline downgrade is forbidden")
