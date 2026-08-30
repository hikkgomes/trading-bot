"""Persist adaptive generation feedback and duplicate outcomes."""

from alembic import op


revision = "platform_v2_0009"
down_revision = "platform_v2_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE generation_feedback (
            id VARCHAR(80) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_generation_feedback_created_at
            ON generation_feedback (created_at);
        CREATE TRIGGER generation_feedback_append_only
        BEFORE UPDATE OR DELETE ON generation_feedback
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON generation_feedback TO trading_runtime;
        GRANT SELECT, INSERT ON generation_feedback TO trading_research;
        GRANT SELECT ON generation_feedback TO trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("generation feedback downgrade is forbidden")
