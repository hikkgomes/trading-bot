"""Index the bounded operator-report lookups."""

from alembic import op

revision = "platform_v2_0043"
down_revision = "platform_v2_0042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX ix_balance_snapshot_created
            ON public.balance_snapshot (created_at DESC, id DESC);
        CREATE INDEX ix_evidence_invalidation_kind_source
            ON public.evidence_invalidation
                ((CAST(payload->>'kind' AS VARCHAR)), (CAST(payload->>'source_id' AS VARCHAR)));
        ANALYZE public.balance_snapshot;
        ANALYZE public.evidence_invalidation;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX public.ix_evidence_invalidation_kind_source;
        DROP INDEX public.ix_balance_snapshot_created;
        """
    )
