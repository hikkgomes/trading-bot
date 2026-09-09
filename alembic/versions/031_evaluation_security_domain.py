"""Separate frozen evaluation from adaptive research and restrict queue ownership."""

from alembic import op

revision = "platform_v2_0031"
down_revision = "platform_v2_0030"
branch_labels = None
depends_on = None

PUBLIC_JOBS = "'generate_hypotheses','register_candidate','register_ml_candidate','bounded_backtest','train_ml_experiment','event_replay','historical_feature_calculation','agent_review','agent_research','agent_code_workflow','agent_delivery'"
EVALUATION_JOBS = "'evaluate_candidate','dataset_snapshot_validate','register_strategy_catalogue'"
RESEARCH_WORKERS = (
    "'research-worker','ml-worker','event-replay-worker','feature-build-worker','agent-coordinator'"
)


def upgrade() -> None:
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='trading_evaluator') THEN
                CREATE ROLE trading_evaluator NOLOGIN;
            END IF;
        END $$;
        GRANT USAGE ON SCHEMA public TO trading_evaluator;
        GRANT SELECT ON alembic_version, schema_migration TO trading_research, trading_evaluator;
        REVOKE ALL ON platform_schedule, account_snapshot, platform_rehearsal_report, holdout_claim,
            forward_evidence, forward_paper_observation, forward_paper_summary,
            forward_paper_decision, strategy_artefact FROM trading_research, trading_agent;
        REVOKE INSERT ON dataset_bundle FROM trading_research;
        GRANT SELECT, INSERT ON strategy_definition, strategy_version, strategy_identity,
            strategy_lineage, experiment, model_artifact, experiment_run, experiment_metric,
            agent_proposal, agent_patch, agent_review, agent_action, agent_disposition
            TO trading_research;
        GRANT SELECT ON evidence_invalidation, dataset_bundle, cost_model_manifest TO trading_research;
        GRANT SELECT ON instrument, instrument_status, universe, universe_snapshot, universe_member,
            dataset_snapshot, dataset_bundle, feature_set, feature_manifest, cost_model_manifest,
            strategy_definition, strategy_version, strategy_identity, strategy_lineage,
            research_thesis, thesis_trial, experiment, experiment_run, experiment_metric,
            validation_result, validation_stage, holdout_claim, holdout_outcome,
            model_artifact, strategy_artefact, paper_eligibility_event,
            selection_cohort, selection_cohort_result, selection_cohort_open,
            evidence_invalidation, legacy_research_archive, import_provenance,
            research_calibration, research_policy_approval, generation_feedback,
            historical_bootstrap_progress, platform_bootstrap TO trading_evaluator;
        GRANT INSERT ON dataset_snapshot, dataset_bundle, feature_manifest, cost_model_manifest,
            strategy_definition, strategy_version, strategy_identity, strategy_lineage,
            research_thesis, thesis_trial, experiment, experiment_run, experiment_metric,
            validation_result, validation_stage, holdout_claim, holdout_outcome,
            strategy_artefact, paper_eligibility_event, selection_cohort,
            selection_cohort_result, selection_cohort_open, generation_feedback TO trading_evaluator;
        GRANT UPDATE (state, metadata) ON experiment TO trading_evaluator;
        CREATE VIEW adaptive_candidate WITH (security_barrier=true) AS
            SELECT id, strategy_version_id, provider,
                CASE WHEN state='legacy_import' THEN 'legacy_import' ELSE 'queued' END AS state,
                submitted_at, dataset_snapshot_hashes, dataset_bundle_id,
                metadata - ARRAY['evaluation_deferred','dataset_waiting','blocked_reason'] AS metadata
            FROM experiment;
        GRANT SELECT ON adaptive_candidate TO trading_runtime,trading_research,trading_evaluator;
        REVOKE SELECT ON experiment FROM trading_research;
        GRANT SELECT (id) ON experiment TO trading_research;
        ALTER POLICY dataset_snapshot_adaptive_read ON dataset_snapshot
            USING (payload->>'role' IN ('screening','development','robustness'));
        ALTER POLICY validation_stage_adaptive_read ON validation_stage
            USING (stage IN ('screening','development','robustness'));
        ALTER POLICY experiment_run_adaptive_read ON experiment_run
            USING (COALESCE(payload->>'phase','') NOT IN
                ('holdout','protected','final_holdout','final','forward','forward_observation')
                AND COALESCE(payload->>'run_name','') NOT IN
                ('canonical:protected','canonical:forward'));
        GRANT SELECT, INSERT, UPDATE ON worker, worker_lease, job_attempt,
            heavy_compute_lease, service_heartbeat TO trading_research, trading_evaluator;
        GRANT DELETE ON worker_lease TO trading_research, trading_evaluator;
        GRANT DELETE ON service_heartbeat TO trading_research, trading_evaluator;
        GRANT SELECT ON job TO trading_research, trading_evaluator;
        GRANT SELECT ON control_event TO trading_research, trading_evaluator;
        GRANT UPDATE (state, priority, available_at, lease_owner, lease_expires_at,
            attempts, terminal_reason) ON job TO trading_research, trading_evaluator;
    """)
    for table in (
        "dataset_snapshot",
        "experiment_run",
        "experiment_metric",
        "validation_result",
        "validation_stage",
    ):
        op.execute(
            f"CREATE POLICY {table}_evaluation ON {table} TO trading_evaluator USING (true) WITH CHECK (true)"
        )
    op.execute("""
        CREATE POLICY experiment_run_diagnostic_insert ON experiment_run FOR INSERT TO trading_research
            WITH CHECK (COALESCE(payload->>'phase','') NOT IN ('protected','holdout','final','final_holdout')
                AND COALESCE(payload->>'run_name','') NOT LIKE 'canonical:%');
        CREATE POLICY experiment_metric_diagnostic_insert ON experiment_metric FOR INSERT TO trading_research
            WITH CHECK (EXISTS (SELECT 1 FROM experiment_run r
                WHERE r.id=experiment_metric.payload->>'run_id'
                AND COALESCE(r.payload->>'run_name','') NOT LIKE 'canonical:%'));
    """)
    conditions = {
        "control_event": (
            f"regexp_replace(payload->>'target','^service:','') IN ({RESEARCH_WORKERS})",
            "regexp_replace(payload->>'target','^service:','')='evaluation-worker'",
        ),
        "job": (f"name IN ({PUBLIC_JOBS})", f"name IN ({EVALUATION_JOBS})"),
        "worker": (f"role IN ({RESEARCH_WORKERS})", "role='evaluation-worker'"),
        "service_heartbeat": (
            f"service_name IN ({RESEARCH_WORKERS})",
            "service_name='evaluation-worker'",
        ),
        "job_attempt": ("EXISTS (SELECT FROM job j WHERE j.id=job_attempt.job_id)",) * 2,
        "worker_lease": ("EXISTS (SELECT FROM job j WHERE j.id=worker_lease.job_id)",) * 2,
    }
    for table, (research, evaluation) in conditions.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_runtime_owner ON {table} TO trading_runtime USING (true) WITH CHECK (true)"
        )
        op.execute(
            f"CREATE POLICY {table}_research_owner ON {table} TO trading_research USING ({research}) WITH CHECK ({research})"
        )
        op.execute(
            f"CREATE POLICY {table}_evaluation_owner ON {table} TO trading_evaluator USING ({evaluation}) WITH CHECK ({evaluation})"
        )
    _typed_submission()


def _typed_submission() -> None:
    op.execute(
        "DROP FUNCTION submit_typed_research_job(VARCHAR,VARCHAR,JSONB,TIMESTAMPTZ,INTEGER,VARCHAR,VARCHAR)"
    )
    op.execute(f"""
        CREATE FUNCTION submit_typed_research_job(request JSONB) RETURNS BOOLEAN
        LANGUAGE plpgsql SECURITY DEFINER SET search_path=public,pg_temp AS $$
        DECLARE previous job%ROWTYPE; inserted_id VARCHAR;
        BEGIN
            IF request->>'name' NOT IN ({PUBLIC_JOBS},{EVALUATION_JOBS})
                OR request->>'id' IS NULL OR request->>'id'=''
                OR request->>'producer_identity' IS NULL OR request->>'producer_identity'=''
                OR request->>'content_hash' !~ '^sha256:[0-9a-f]{{64}}$'
                OR jsonb_typeof(request->'payload') IS DISTINCT FROM 'object'
                OR request->>'state' IS DISTINCT FROM 'pending'
                OR (request->>'attempts')::int IS DISTINCT FROM 0
                OR COALESCE((request->>'max_attempts')::int,0)<1
            THEN RAISE EXCEPTION 'typed research job identity or type is invalid'; END IF;
            INSERT INTO job(id,name,state,priority,available_at,attempts,max_attempts,producer_identity,content_hash,payload)
            VALUES(request->>'id',request->>'name','pending',(request->>'priority')::int,
                (request->>'available_at')::timestamptz,0,(request->>'max_attempts')::int,
                request->>'producer_identity',request->>'content_hash',request->'payload')
            ON CONFLICT (id) DO NOTHING RETURNING id INTO inserted_id;
            IF inserted_id IS NOT NULL THEN RETURN true; END IF;
            SELECT * INTO STRICT previous FROM job WHERE id=request->>'id';
            IF previous.name IS DISTINCT FROM request->>'name'
                OR previous.priority IS DISTINCT FROM (request->>'priority')::int
                OR previous.max_attempts IS DISTINCT FROM (request->>'max_attempts')::int
                OR previous.producer_identity IS DISTINCT FROM request->>'producer_identity'
                OR previous.content_hash IS DISTINCT FROM request->>'content_hash'
                OR previous.payload IS DISTINCT FROM request->'payload'
                OR (previous.state='pending' AND previous.attempts=0 AND previous.id NOT LIKE 'work:%'
                    AND previous.available_at IS DISTINCT FROM (request->>'available_at')::timestamptz)
            THEN RAISE EXCEPTION 'job identity collision'; END IF;
            RETURN false;
        END $$;
        REVOKE ALL ON FUNCTION submit_typed_research_job(JSONB) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION submit_typed_research_job(JSONB) TO trading_research,trading_evaluator;
    """)


def downgrade() -> None:
    raise RuntimeError("Evaluation and adaptive research security boundaries cannot be removed")
