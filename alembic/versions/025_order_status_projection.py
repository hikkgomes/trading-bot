"""Correct order status projections while retaining original journal events."""

from alembic import op

revision = "platform_v2_0025"
down_revision = "platform_v2_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE exchange_order
        SET status = payload->'intent'->>'status'
        WHERE payload->'intent'->>'status' IS NOT NULL
          AND status IS DISTINCT FROM payload->'intent'->>'status';
    """)


def downgrade() -> None:
    raise RuntimeError("Corrected financial status projections cannot be reverted")
