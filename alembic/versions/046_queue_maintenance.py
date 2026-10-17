"""Keep queue index-only scans cheap as attempt history grows."""

from alembic import op

revision = "platform_v2_0046"
down_revision = "platform_v2_0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE public.job SET (
            autovacuum_vacuum_scale_factor = 0.02,
            autovacuum_analyze_scale_factor = 0.02
        );
        ALTER TABLE public.job_attempt SET (
            autovacuum_vacuum_scale_factor = 0.02,
            autovacuum_analyze_scale_factor = 0.02
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE public.job RESET (
            autovacuum_vacuum_scale_factor, autovacuum_analyze_scale_factor
        );
        ALTER TABLE public.job_attempt RESET (
            autovacuum_vacuum_scale_factor, autovacuum_analyze_scale_factor
        );
        """
    )
