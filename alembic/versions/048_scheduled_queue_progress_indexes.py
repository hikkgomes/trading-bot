"""Index the scheduled-job families used by the operator progress report."""

from alembic import op

revision = "platform_v2_0048"
down_revision = "platform_v2_0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_job_scheduled_prefix ON public.job (id) WHERE id LIKE 'scheduled:%'"
    )
    op.execute(
        "CREATE INDEX ix_job_attempt_scheduled_prefix ON public.job_attempt (job_id) "
        "WHERE job_id LIKE 'scheduled:%'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX public.ix_job_attempt_scheduled_prefix")
    op.execute("DROP INDEX public.ix_job_scheduled_prefix")
