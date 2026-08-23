"""Platform v2 owner baseline with explicit authority tables."""

from alembic import op

revision = "platform-v2-0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
            version VARCHAR(80) PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            revision_hash VARCHAR(80) NOT NULL
        );
        CREATE TABLE IF NOT EXISTS heavy_compute_lease (
            slot_id VARCHAR(80) PRIMARY KEY,
            owner VARCHAR(160),
            job_id VARCHAR(160),
            acquired_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            status VARCHAR(40) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_heavy_compute_lease_expires_at
            ON heavy_compute_lease (expires_at);
        """
    )


def downgrade() -> None:
    raise RuntimeError("platform baseline downgrade is forbidden")
