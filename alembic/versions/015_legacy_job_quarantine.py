"""Add one-time quarantine records for legacy research queue work."""

from alembic import op

revision = "platform_v2_0015"
down_revision = "platform_v2_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_upgrade_marker (
            id VARCHAR(160) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS job_quarantine (
            id VARCHAR(240) PRIMARY KEY,
            job_id VARCHAR(160) NOT NULL REFERENCES job(id),
            quarantined_at TIMESTAMPTZ NOT NULL,
            original_state VARCHAR(40) NOT NULL,
            reason VARCHAR(160) NOT NULL,
            content_hash VARCHAR(80) NOT NULL,
            payload JSONB NOT NULL,
            CONSTRAINT uq_job_quarantine_job UNIQUE (job_id)
        );
        CREATE INDEX IF NOT EXISTS ix_job_quarantine_job ON job_quarantine (job_id);
        CREATE INDEX IF NOT EXISTS ix_job_quarantine_time
            ON job_quarantine (quarantined_at);
        """
    )


def downgrade() -> None:
    raise RuntimeError("legacy job quarantine downgrade is forbidden")
