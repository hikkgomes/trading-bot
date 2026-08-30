"""Add the evidence-complete, non-authoritative live-ready state."""

from alembic import op


revision = "platform_v2_0010"
down_revision = "platform_v2_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE active_strategy_assignment
            DROP CONSTRAINT IF EXISTS ck_assignment_lifecycle_state;
        ALTER TABLE active_strategy_assignment
            ADD CONSTRAINT ck_assignment_lifecycle_state CHECK (
                lifecycle_state IN (
                    'registered', 'development', 'forward_paper', 'live_ready',
                    'live_canary', 'live', 'suspended', 'retired'
                )
            );
        """
    )


def downgrade() -> None:
    raise RuntimeError("live-ready lifecycle downgrade is forbidden")
