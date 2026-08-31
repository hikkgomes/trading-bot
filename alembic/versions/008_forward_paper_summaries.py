"""Separate forward observations from aggregate promotion decisions."""

from alembic import op

revision = "platform_v2_0008"
down_revision = "platform_v2_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE forward_paper_summary (
            id VARCHAR(200) PRIMARY KEY,
            strategy_version_id VARCHAR(160) NOT NULL REFERENCES strategy_version(id),
            product_id VARCHAR(80) NOT NULL,
            artefact_hash VARCHAR(80) NOT NULL,
            observed_from TIMESTAMPTZ NOT NULL,
            observed_until TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_forward_paper_summary_strategy_version_id
            ON forward_paper_summary (strategy_version_id);
        CREATE INDEX ix_forward_paper_summary_product_id
            ON forward_paper_summary (product_id);
        CREATE INDEX ix_forward_paper_summary_created_at
            ON forward_paper_summary (created_at);
        CREATE TABLE forward_paper_decision (
            id VARCHAR(200) PRIMARY KEY,
            summary_id VARCHAR(200) NOT NULL REFERENCES forward_paper_summary(id),
            strategy_version_id VARCHAR(160) NOT NULL REFERENCES strategy_version(id),
            product_id VARCHAR(80) NOT NULL,
            artefact_hash VARCHAR(80) NOT NULL,
            decided_at TIMESTAMPTZ NOT NULL,
            accepted BOOLEAN NOT NULL,
            reason_code VARCHAR(160),
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL,
            CONSTRAINT ck_forward_paper_decision_rejection_reason
                CHECK (accepted IS TRUE OR reason_code IS NOT NULL)
        );
        CREATE INDEX ix_forward_paper_decision_summary_id
            ON forward_paper_decision (summary_id);
        CREATE INDEX ix_forward_paper_decision_strategy_version_id
            ON forward_paper_decision (strategy_version_id);
        CREATE INDEX ix_forward_paper_decision_product_id
            ON forward_paper_decision (product_id);
        CREATE INDEX ix_forward_paper_decision_decided_at
            ON forward_paper_decision (decided_at);
        CREATE TRIGGER forward_paper_summary_append_only
        BEFORE UPDATE OR DELETE ON forward_paper_summary
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER forward_paper_decision_append_only
        BEFORE UPDATE OR DELETE ON forward_paper_decision
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON forward_paper_summary, forward_paper_decision TO trading_runtime;
        GRANT SELECT, INSERT ON forward_paper_summary, forward_paper_decision TO trading_research;
        GRANT SELECT ON forward_paper_summary, forward_paper_decision TO trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("forward paper summary downgrade is forbidden")
