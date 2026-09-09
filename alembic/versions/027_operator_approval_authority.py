"""Separate human live approval from the runtime database role."""

from alembic import op

revision = "platform_v2_0027"
down_revision = "platform_v2_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_operator') THEN
                CREATE ROLE trading_operator NOLOGIN;
            END IF;
        END $$;
        GRANT USAGE ON SCHEMA public TO trading_operator;
        REVOKE ALL ON strategy_approval FROM PUBLIC, trading_research, trading_agent;
        REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
            ON strategy_approval FROM trading_runtime;
        GRANT SELECT ON strategy_approval TO trading_runtime;
        GRANT SELECT, INSERT ON strategy_approval TO trading_operator;
    """)


def downgrade() -> None:
    raise RuntimeError("Runtime self-approval must not be restored")
