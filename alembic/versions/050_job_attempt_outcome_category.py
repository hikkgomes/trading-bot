"""Store the queue-attempt outcome projection for all-history reports."""

from alembic import op

revision = "platform_v2_0050"
down_revision = "platform_v2_0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE public.job_attempt
        ADD COLUMN outcome_category TEXT
        GENERATED ALWAYS AS (
            COALESCE(
                payload->>'outcome_category',
                CASE
                    WHEN status IN ('failed', 'expired') THEN 'blocked_infrastructure'
                    ELSE 'unclassified'
                END
            )
        ) STORED NOT NULL;
        CREATE INDEX ix_job_attempt_outcome_category
            ON public.job_attempt (outcome_category);
        ANALYZE public.job_attempt;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX public.ix_job_attempt_outcome_category")
    op.execute("ALTER TABLE public.job_attempt DROP COLUMN outcome_category")
