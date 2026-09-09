"""Preserve initial capital resources and the fixed BTC reserve."""

from alembic import op

revision = "platform_v2_0018"
down_revision = "platform_v2_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE capital_baseline (
            id VARCHAR(160) PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TRIGGER capital_baseline_append_only BEFORE UPDATE OR DELETE ON capital_baseline
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON capital_baseline TO trading_runtime;
        REVOKE ALL ON capital_baseline FROM trading_research, trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Capital history downgrade is forbidden")
