"""Retain previous economics and protected exposure without qualification authority."""

from alembic import op

revision = "platform_v2_0033"
down_revision = "platform_v2_0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO evidence_invalidation
        SELECT 'stage:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'validation_stage',
                'reason', 'executable_economics_and_scope_replaced',
                'replacement_engine', 'economic-engine/v5')
        FROM validation_stage WHERE engine_version IS DISTINCT FROM 'economic-engine/v5'
        ON CONFLICT DO NOTHING;
        INSERT INTO evidence_invalidation
        SELECT 'artefact:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'strategy_artefact',
                'reason', 'executable_economics_and_scope_replaced',
                'replacement_engine', 'economic-engine/v5')
        FROM strategy_artefact WHERE payload->>'engine_version' IS DISTINCT FROM 'economic-engine/v5'
        ON CONFLICT DO NOTHING;
        INSERT INTO evidence_invalidation
        SELECT 'approval:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'strategy_approval',
                'reason', 'dependent_artefact_invalidated', 'artefact_hash', artefact_hash)
        FROM strategy_approval WHERE 'artefact:' || artefact_hash IN (SELECT id FROM evidence_invalidation)
        ON CONFLICT DO NOTHING;
        INSERT INTO evidence_invalidation
        SELECT 'preflight:' || id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', id, 'kind', 'production_preflight',
                'reason', 'dependent_artefact_invalidated', 'artefact_hash', artefact_hash)
        FROM production_preflight WHERE 'artefact:' || artefact_hash IN (SELECT id FROM evidence_invalidation)
        ON CONFLICT DO NOTHING;
        UPDATE experiment SET metadata = metadata || jsonb_build_object(
                'prior_engine_state', state, 'engine_invalidation', 'economic-engine/v5'),
            state = 'queued'
        WHERE state <> 'legacy_import' AND id IN (
            SELECT experiment_id FROM validation_stage
            WHERE engine_version IS DISTINCT FROM 'economic-engine/v5'
        );
        INSERT INTO active_strategy_assignment
        SELECT (jsonb_populate_record(NULL::active_strategy_assignment,
            to_jsonb(a) || jsonb_build_object(
                'id', 'retired-engine-v5:' || md5(a.id),
                'active', false, 'lifecycle_state', 'retired',
                'assigned_at', CURRENT_TIMESTAMP, 'assigned_by', 'migration:033',
                'assignment_reason', 'economic_engine_replaced',
                'payload', a.payload || jsonb_build_object('supersedes_assignment_id', a.id)
            ))).*
        FROM active_strategy_assignment a
        WHERE execution_mode = 'paper' AND active AND 'artefact:' || artefact_hash IN (
            SELECT id FROM evidence_invalidation
        );
    """)


def downgrade() -> None:
    raise RuntimeError("Invalid economic evidence must not regain qualification authority")
