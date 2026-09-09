"""Persist complete frozen finalist cohorts and sealed joint inference."""

from alembic import op

revision = "platform_v2_0026"
down_revision = "platform_v2_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE selection_cohort (
            id VARCHAR(160) PRIMARY KEY, scope_id VARCHAR(160) NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TABLE selection_cohort_result (
            id VARCHAR(160) PRIMARY KEY REFERENCES selection_cohort(id),
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TABLE selection_cohort_open (
            id VARCHAR(160) PRIMARY KEY REFERENCES selection_cohort(id),
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL
        );
        CREATE TRIGGER selection_cohort_append_only BEFORE UPDATE OR DELETE ON selection_cohort
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER selection_cohort_result_append_only BEFORE UPDATE OR DELETE ON selection_cohort_result
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER selection_cohort_open_append_only BEFORE UPDATE OR DELETE ON selection_cohort_open
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        REVOKE ALL ON selection_cohort, selection_cohort_result, selection_cohort_open FROM trading_research, trading_agent;
        GRANT SELECT ON selection_cohort TO trading_research;
        GRANT SELECT, INSERT ON selection_cohort, selection_cohort_result, selection_cohort_open TO trading_runtime;
    """)


def downgrade() -> None:
    raise RuntimeError("Frozen cohort and exposure history cannot be removed")
