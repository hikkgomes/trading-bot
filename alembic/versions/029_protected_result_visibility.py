"""Restrict adaptive result reads to explicitly pre-protected coordinator states."""

from alembic import op

revision = "platform_v2_0029"
down_revision = "platform_v2_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER POLICY validation_result_adaptive_read ON validation_result
        USING (
            state IN ('screen_rejected', 'development_rejected', 'robustness_rejected')
            AND COALESCE(evidence->>'phase', '') NOT IN
                ('holdout', 'protected', 'final_holdout', 'final')
            AND CASE WHEN jsonb_typeof(evidence->'stages') = 'object' THEN
                (evidence->'stages') - ARRAY[
                    'screen_rejected', 'development_rejected', 'robustness_rejected'
                ] = '{}'::jsonb
            ELSE false END
        )
        """
    )


def downgrade() -> None:
    raise RuntimeError("Protected result visibility cannot be weakened by downgrade")
