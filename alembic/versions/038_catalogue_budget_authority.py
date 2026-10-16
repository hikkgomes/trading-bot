"""Let the catalogue evaluator reserve the same atomic budget as other providers."""

from alembic import op

revision = "platform_v2_0038"
down_revision = "platform_v2_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("GRANT INSERT, UPDATE ON research_epoch, research_round TO trading_evaluator")


def downgrade() -> None:
    op.execute("REVOKE INSERT, UPDATE ON research_epoch, research_round FROM trading_evaluator")
