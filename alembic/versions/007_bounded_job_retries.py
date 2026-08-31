"""Bound queue retries and retain terminal failure state."""

from alembic import op

revision = "platform_v2_0007"
down_revision = "platform_v2_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE job
            ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 3,
            ADD COLUMN IF NOT EXISTS terminal_reason TEXT;
        ALTER TABLE job
            ADD CONSTRAINT ck_job_max_attempts_positive CHECK (max_attempts > 0);
        CREATE INDEX IF NOT EXISTS ix_job_terminal_reason ON job (terminal_reason);
        """
    )


def downgrade() -> None:
    raise RuntimeError("bounded job retry downgrade is forbidden")
