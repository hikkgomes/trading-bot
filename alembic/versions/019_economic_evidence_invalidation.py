"""Retain old economic evidence but remove its qualification authority."""

from alembic import op

revision = "platform_v2_0019"
down_revision = "platform_v2_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE evidence_invalidation (
            id VARCHAR(160) PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        INSERT INTO evidence_invalidation
        SELECT 'stage:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'validation_stage',
                'reason', 'economic_engine_replaced', 'replacement_engine', 'economic-engine/v4')
        FROM validation_stage WHERE engine_version IS DISTINCT FROM 'economic-engine/v4';
        INSERT INTO evidence_invalidation
        SELECT 'artefact:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'strategy_artefact',
                'reason', 'economic_engine_replaced', 'replacement_engine', 'economic-engine/v4')
        FROM strategy_artefact WHERE payload->>'engine_version' IS DISTINCT FROM 'economic-engine/v4';
        UPDATE experiment SET metadata = metadata || jsonb_build_object(
                'prior_engine_state', state, 'engine_invalidation', 'economic-engine/v4'),
            state = 'queued'
        WHERE state <> 'legacy_import' AND id IN (
            SELECT experiment_id FROM validation_stage
            WHERE engine_version IS DISTINCT FROM 'economic-engine/v4'
        );
        INSERT INTO active_strategy_assignment
        SELECT (jsonb_populate_record(NULL::active_strategy_assignment,
            to_jsonb(a) || jsonb_build_object(
                'id', 'retired-engine-v4:' || md5(a.id),
                'active', false, 'lifecycle_state', 'retired',
                'assigned_at', CURRENT_TIMESTAMP, 'assigned_by', 'migration:019',
                'assignment_reason', 'economic_engine_replaced',
                'payload', a.payload || jsonb_build_object('supersedes_assignment_id', a.id)
            ))).*
        FROM active_strategy_assignment a
        WHERE execution_mode = 'paper' AND artefact_hash IN (
            SELECT payload->>'source_id' FROM evidence_invalidation
            WHERE payload->>'kind' = 'strategy_artefact'
        );
        ALTER TABLE validation_stage DROP CONSTRAINT uq_validation_stage_experiment_stage;
        CREATE TRIGGER evidence_invalidation_append_only BEFORE UPDATE OR DELETE ON evidence_invalidation
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON evidence_invalidation TO trading_runtime;
        GRANT SELECT ON evidence_invalidation TO trading_research;
        REVOKE ALL ON evidence_invalidation FROM trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Economic evidence invalidation cannot be reversed")
