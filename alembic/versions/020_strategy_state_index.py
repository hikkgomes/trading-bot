"""Index immutable strategy state carried by assignment-bound decisions."""

from alembic import op

revision = "platform_v2_0020"
down_revision = "platform_v2_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX ix_alpha_forecast_assignment_state ON alpha_forecast
            ((payload->'metadata'->>'assignment_id'),
             (payload->>'strategy_version_id'), created_at DESC);
    """)


def downgrade() -> None:
    raise RuntimeError("Strategy state index downgrade is forbidden")
