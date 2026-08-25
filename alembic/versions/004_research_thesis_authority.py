"""Immutable research theses and lineage-wide trial accounting."""

from alembic import op

revision = "platform_v2_0004"
down_revision = "platform_v2_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE research_thesis (
            id VARCHAR(80) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            creator_identity VARCHAR(160) NOT NULL,
            cumulative_trial_budget INTEGER NOT NULL,
            payload JSONB NOT NULL,
            CONSTRAINT ck_thesis_trial_budget_positive CHECK (cumulative_trial_budget > 0)
        );
        CREATE INDEX ix_research_thesis_created_at ON research_thesis (created_at);

        CREATE TABLE thesis_trial (
            id VARCHAR(80) PRIMARY KEY,
            thesis_id VARCHAR(80) NOT NULL REFERENCES research_thesis(id),
            candidate_id VARCHAR(80) NOT NULL UNIQUE,
            lineage_id VARCHAR(80) NOT NULL,
            ordinal INTEGER NOT NULL,
            claimed_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT uq_thesis_trial_ordinal UNIQUE (thesis_id, ordinal),
            CONSTRAINT ck_thesis_trial_ordinal_positive CHECK (ordinal > 0)
        );
        CREATE INDEX ix_thesis_trial_thesis_id ON thesis_trial (thesis_id);
        CREATE INDEX ix_thesis_trial_candidate_id ON thesis_trial (candidate_id);
        CREATE INDEX ix_thesis_trial_lineage_id ON thesis_trial (lineage_id);

        CREATE TRIGGER research_thesis_append_only
        BEFORE UPDATE OR DELETE ON research_thesis
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER thesis_trial_append_only
        BEFORE UPDATE OR DELETE ON thesis_trial
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();

        GRANT SELECT ON research_thesis, thesis_trial TO trading_runtime;
        GRANT SELECT, INSERT ON research_thesis, thesis_trial TO trading_research;
        GRANT SELECT ON research_thesis, thesis_trial TO trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("research authority downgrade is forbidden")
