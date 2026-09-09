"""Add the immutable artefact-to-paper eligibility outbox."""

from alembic import op

revision = "platform_v2_0021"
down_revision = "platform_v2_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE paper_eligibility_event (
            id VARCHAR(160) PRIMARY KEY REFERENCES strategy_artefact(id),
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            candidate_id VARCHAR(160) NOT NULL REFERENCES experiment(id),
            engine_version VARCHAR(160) NOT NULL,
            CONSTRAINT uq_paper_eligibility_candidate_engine UNIQUE(candidate_id, engine_version)
        );
        CREATE TRIGGER paper_eligibility_event_append_only BEFORE UPDATE OR DELETE ON paper_eligibility_event
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON paper_eligibility_event TO trading_runtime;
        REVOKE ALL ON paper_eligibility_event FROM trading_research, trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Paper eligibility history downgrade is forbidden")
