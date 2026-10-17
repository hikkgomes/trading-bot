"""Invalidate unqualified futures evidence without releasing consumed trials."""

from alembic import op

revision = "platform_v2_0051"
down_revision = "platform_v2_0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        LOCK TABLE experiment, validation_stage, job IN SHARE ROW EXCLUSIVE MODE;
        INSERT INTO evidence_invalidation
        SELECT 'stage:' || s.id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', s.id, 'kind', 'validation_stage',
                'reason', 'futures_financial_feasibility_required',
                'replacement_contract', 'futures-financial-feasibility/v1')
        FROM validation_stage s
        JOIN experiment e ON e.id=s.experiment_id
        JOIN strategy_version v ON v.id=e.strategy_version_id
        JOIN strategy_definition d ON d.id=v.definition_id
        JOIN experiment_run r ON r.id=s.source_run_id
        WHERE d.product_id='active_income' AND (
            s.payload #>> '{evidence,evaluation_context,financial_risk_contract}'
                IS DISTINCT FROM 'futures-financial-feasibility/v1'
            OR r.payload #>> '{receipt,financial_risk_contract}'
                IS DISTINCT FROM 'futures-financial-feasibility/v1')
        ON CONFLICT (id) DO NOTHING;

        INSERT INTO evidence_invalidation
        SELECT 'artefact:' || a.id, CURRENT_TIMESTAMP,
            jsonb_build_object('source_id', a.id, 'kind', 'strategy_artefact',
                'reason', 'futures_financial_feasibility_required',
                'replacement_contract', 'futures-financial-feasibility/v1')
        FROM strategy_artefact a
        WHERE a.payload->>'product_id'='active_income'
        ON CONFLICT (id) DO NOTHING;

        CREATE TEMP TABLE futures_revalidation ON COMMIT DROP AS
        SELECT e.id, (
            EXISTS (SELECT 1 FROM research_epoch ep
                WHERE ep.dataset_bundle_id=e.dataset_bundle_id
                AND ep.state IN ('active', 'waiting_for_feedback', 'exhausted'))
            AND NOT EXISTS (SELECT 1 FROM holdout_claim h
                WHERE h.payload->>'strategy_version_id'=e.strategy_version_id)
            AND NOT EXISTS (SELECT 1 FROM selection_cohort c
                WHERE c.payload->'all_trials' ? e.id)
        ) AS can_revalidate
        FROM experiment e
        WHERE EXISTS (SELECT 1 FROM validation_stage s
            JOIN evidence_invalidation i ON i.id='stage:' || s.id
            WHERE s.experiment_id=e.id
                AND i.payload->>'reason'='futures_financial_feasibility_required');

        UPDATE experiment e SET
            metadata=(metadata - 'evaluation_deferred' - 'dataset_waiting') ||
                jsonb_build_object('financial_revalidation', jsonb_build_object(
                    'prior_state', state, 'migration', 'platform_v2_0051',
                    'contract', 'futures-financial-feasibility/v1',
                    'can_revalidate', r.can_revalidate)),
            state=CASE WHEN r.can_revalidate THEN 'queued' ELSE 'blocked_data' END
        FROM futures_revalidation r WHERE r.id=e.id AND e.state <> 'legacy_import';

        INSERT INTO job_quarantine
            (id, job_id, quarantined_at, original_state, reason, content_hash, payload)
        SELECT 'financial-v1:' || md5(j.id), j.id, CURRENT_TIMESTAMP, j.state,
            'futures_financial_feasibility_required', j.content_hash,
            jsonb_build_object('job_id', j.id, 'original_payload', j.payload,
                'original_attempts', j.attempts, 'migration', 'platform_v2_0051')
        FROM job j
        JOIN experiment e ON e.id=j.payload->>'candidate_id'
        JOIN strategy_version v ON v.id=e.strategy_version_id
        JOIN strategy_definition d ON d.id=v.definition_id
        WHERE j.name='evaluate_candidate' AND d.product_id='active_income'
            AND j.state IN ('pending', 'dead_letter')
        ON CONFLICT (job_id) DO NOTHING;
        UPDATE job j SET state='quarantined', lease_owner=NULL, lease_expires_at=NULL,
            terminal_reason='futures_financial_feasibility_required'
        FROM job_quarantine q WHERE q.job_id=j.id
            AND q.reason='futures_financial_feasibility_required'
            AND j.state IN ('pending', 'dead_letter');

        INSERT INTO active_strategy_assignment
        SELECT (jsonb_populate_record(NULL::active_strategy_assignment,
            to_jsonb(a) || jsonb_build_object(
                'id', 'retired-financial-v1:' || md5(a.id),
                'active', false, 'lifecycle_state', 'retired',
                'assigned_at', CURRENT_TIMESTAMP, 'assigned_by', 'migration:051',
                'assignment_reason', 'futures_financial_feasibility_required',
                'payload', a.payload || jsonb_build_object('supersedes_assignment_id', a.id)
            ))).*
        FROM active_strategy_assignment a
        WHERE execution_mode='paper' AND active AND artefact_hash IN (
            SELECT payload->>'source_id' FROM evidence_invalidation
            WHERE payload->>'kind'='strategy_artefact'
                AND payload->>'reason'='futures_financial_feasibility_required');
    """)


def downgrade() -> None:
    raise RuntimeError("Consumed trials and protected evidence cannot be reopened")
