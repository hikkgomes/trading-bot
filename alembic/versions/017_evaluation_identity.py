"""Index versioned research results without changing historical provenance."""

from alembic import op

revision = "platform_v2_0017"
down_revision = "platform_v2_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE validation_stage ADD COLUMN evaluation_identity VARCHAR(80);
        ALTER TABLE validation_stage ADD COLUMN engine_version VARCHAR(160);
        ALTER TABLE validation_stage ADD COLUMN outcome_category VARCHAR(40);
        CREATE UNIQUE INDEX ix_validation_stage_evaluation_identity
            ON validation_stage (evaluation_identity);
        CREATE INDEX ix_job_attempt_status_started ON job_attempt (status, started_at);
    """)


def downgrade() -> None:
    raise RuntimeError("Research evidence identity downgrade is forbidden")
