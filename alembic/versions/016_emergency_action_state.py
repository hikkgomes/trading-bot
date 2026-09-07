"""Add durable emergency action state-machine records."""

from alembic import op

revision = "platform_v2_0016"
down_revision = "platform_v2_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS emergency_action (
            id VARCHAR(240) PRIMARY KEY,
            job_id VARCHAR(160) NOT NULL UNIQUE REFERENCES job(id),
            product_id VARCHAR(80) NOT NULL,
            state VARCHAR(80) NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_emergency_action_product
            ON emergency_action (product_id);
        CREATE INDEX IF NOT EXISTS ix_emergency_action_state
            ON emergency_action (state);
        CREATE INDEX IF NOT EXISTS ix_emergency_action_updated
            ON emergency_action (updated_at);
        """
    )


def downgrade() -> None:
    raise RuntimeError("emergency action state downgrade is forbidden")
