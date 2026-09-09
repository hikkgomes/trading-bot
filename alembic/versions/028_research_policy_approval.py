"""Retain calibration evidence and separate operator policy approval."""

from alembic import op

revision = "platform_v2_0028"
down_revision = "platform_v2_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE research_calibration (
            id VARCHAR(160) PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TABLE research_policy_approval (
            id VARCHAR(160) PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TRIGGER research_calibration_append_only BEFORE UPDATE OR DELETE ON research_calibration
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER research_policy_approval_append_only BEFORE UPDATE OR DELETE ON research_policy_approval
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        REVOKE ALL ON research_calibration, research_policy_approval
            FROM PUBLIC, trading_runtime, trading_research, trading_agent;
        GRANT SELECT ON research_calibration, research_policy_approval TO trading_runtime, trading_research;
        GRANT SELECT, INSERT ON research_calibration, research_policy_approval TO trading_operator;
        CREATE INDEX research_policy_approval_scope ON research_policy_approval
            ((payload->>'policy_hash'), created_at);
    """)


def downgrade() -> None:
    raise RuntimeError("Calibration and operator approval history cannot be removed")
