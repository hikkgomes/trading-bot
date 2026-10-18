"""Keep disposable quotes off the durable financial write path."""

from alembic import op

revision = "platform_v2_0052"
down_revision = "platform_v2_0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE market_input_buffer SET UNLOGGED")


def downgrade() -> None:
    op.execute("ALTER TABLE market_input_buffer SET LOGGED")
