"""Bound oldest-due and latest-terminal queue lookups."""

from alembic import op

revision = "platform_v2_0047"
down_revision = "platform_v2_0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX ix_job_state_available_id ON public.job (state, available_at, id)")


def downgrade() -> None:
    op.execute("DROP INDEX public.ix_job_state_available_id")
