"""Platform v2 append-only and privilege boundary."""

from alembic import op

revision = "platform-v2-0002"
down_revision = "platform-v2-0001"
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
                    IF table_name = 'job' THEN
                        EXECUTE 'GRANT SELECT, INSERT ON TABLE job TO trading_research, trading_agent';
                    END IF;
                END IF;
            END LOOP;
        END $$;
        """
    )


def downgrade() -> None:
    raise RuntimeError("platform authority downgrade is forbidden")
