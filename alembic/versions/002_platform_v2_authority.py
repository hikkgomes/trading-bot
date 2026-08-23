"""Platform v2 append-only and privilege boundary."""

from alembic import op

revision = "platform_v2_0002"
down_revision = "platform_v2_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION trading_platform_reject_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'append-only platform record cannot be mutated: %', TG_TABLE_NAME;
        END;
        $$;
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_platform_owner') THEN
                CREATE ROLE trading_platform_owner NOLOGIN;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_runtime') THEN
                CREATE ROLE trading_runtime NOLOGIN;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_research') THEN
                CREATE ROLE trading_research NOLOGIN;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_agent') THEN
                CREATE ROLE trading_agent NOLOGIN;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_platform_owner') THEN
                GRANT trading_platform_owner TO CURRENT_USER;
                GRANT USAGE ON SCHEMA public TO trading_platform_owner;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_runtime') THEN
                    GRANT USAGE ON SCHEMA public TO trading_runtime;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_research') THEN
                    GRANT USAGE ON SCHEMA public TO trading_research;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_agent') THEN
                    GRANT USAGE ON SCHEMA public TO trading_agent;
                END IF;
            END IF;
        END $$;
        DO $$
        DECLARE table_name TEXT;
        BEGIN
            FOREACH table_name IN ARRAY ARRAY[
                'job', 'order_intent', 'exchange_order', 'fill', 'position',
                'balance_snapshot', 'accounting_entry', 'strategy_approval',
                'production_preflight', 'active_strategy_assignment', 'control_event',
                'promotion_event'
            ] LOOP
                IF to_regclass('public.' || table_name) IS NOT NULL THEN
                    EXECUTE format('REVOKE ALL ON TABLE %I FROM trading_research, trading_agent', table_name);
                END IF;
            END LOOP;
            FOREACH table_name IN ARRAY ARRAY[
                'instrument', 'market_event', 'universe', 'universe_snapshot',
                'universe_member', 'dataset_snapshot', 'feature_set', 'feature_manifest',
                'feature_value', 'strategy_definition', 'strategy_version',
                'strategy_identity', 'strategy_lineage', 'experiment_run',
                'experiment_metric', 'validation_result', 'validation_stage',
                'holdout_claim', 'holdout_outcome', 'forward_evidence',
                'forward_paper_observation', 'model_artifact'
            ] LOOP
                IF to_regclass('public.' || table_name) IS NOT NULL THEN
                    EXECUTE format('GRANT SELECT ON TABLE %I TO trading_research', table_name);
                END IF;
            END LOOP;
            FOREACH table_name IN ARRAY ARRAY[
                'agent_proposal', 'agent_patch', 'agent_action'
            ] LOOP
                IF to_regclass('public.' || table_name) IS NOT NULL THEN
                    EXECUTE format('GRANT SELECT, INSERT ON TABLE %I TO trading_agent', table_name);
                END IF;
            END LOOP;
        END $$;
        DO $$
        DECLARE table_name TEXT;
        BEGIN
            FOREACH table_name IN ARRAY ARRAY[
                'universe', 'universe_snapshot', 'universe_member', 'dataset_snapshot',
                'feature_set', 'feature_manifest', 'strategy_definition',
                'strategy_version', 'strategy_identity', 'strategy_lineage',
                'experiment_run', 'experiment_metric', 'validation_result',
                'validation_stage', 'holdout_claim', 'model_artifact',
                'holdout_outcome', 'forward_evidence', 'forward_paper_observation',
                'strategy_artefact', 'strategy_approval', 'production_preflight',
                'import_provenance', 'agent_action', 'agent_proposal', 'agent_patch',
                'agent_review', 'agent_disposition', 'alpha_forecast',
                'target_position', 'risk_snapshot', 'risk_policy', 'risk_decision',
                'promotion_event', 'promotion_policy'
            ] LOOP
                IF to_regclass('public.' || table_name) IS NOT NULL THEN
                    EXECUTE format(
                        'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I '
                        'FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation()',
                        table_name || '_append_only', table_name
                    );
                END IF;
            END LOOP;
        END $$;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public
            TO trading_runtime;
        GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public
            TO trading_runtime;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_runtime;
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO trading_runtime;

        CREATE OR REPLACE FUNCTION submit_typed_research_job(
            requested_id VARCHAR,
            requested_name VARCHAR,
            requested_payload JSONB,
            requested_available_at TIMESTAMPTZ,
            requested_priority INTEGER,
            requested_producer VARCHAR,
            requested_content_hash VARCHAR
        ) RETURNS VARCHAR
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public, pg_temp
        AS $$
        BEGIN
            IF requested_name NOT IN (
                'evaluate_candidate', 'research_screening', 'research_development',
                'research_robustness', 'research_forward', 'agent_proposal_review'
            ) THEN
                RAISE EXCEPTION 'job type is not permitted: %', requested_name;
            END IF;
            IF requested_id IS NULL OR requested_id = ''
                OR requested_producer IS NULL OR requested_producer = ''
                OR requested_content_hash !~ '^sha256:[0-9a-f]{64}$'
                OR requested_payload IS NULL
            THEN
                RAISE EXCEPTION 'typed research job identity is invalid';
            END IF;
            INSERT INTO job (
                id, name, state, priority, available_at, lease_owner,
                lease_expires_at, attempts, producer_identity, content_hash, payload
            ) VALUES (
                requested_id, requested_name, 'queued', requested_priority,
                requested_available_at, NULL, NULL, 0, requested_producer,
                requested_content_hash, requested_payload
            );
            RETURN requested_id;
        END;
        $$;
        REVOKE ALL ON FUNCTION submit_typed_research_job(
            VARCHAR, VARCHAR, JSONB, TIMESTAMPTZ, INTEGER, VARCHAR, VARCHAR
        ) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION submit_typed_research_job(
            VARCHAR, VARCHAR, JSONB, TIMESTAMPTZ, INTEGER, VARCHAR, VARCHAR
        ) TO trading_research, trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("platform authority downgrade is forbidden")
