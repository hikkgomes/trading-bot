"""Index bounded portfolio state lookups."""

from alembic import op

revision = "platform_v2_0045"
down_revision = "platform_v2_0044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX ix_account_snapshot_account_observed_id
            ON public.account_snapshot (account_id, observed_at DESC, id DESC);
        CREATE INDEX ix_balance_snapshot_account_created_id
            ON public.balance_snapshot
                ((CAST(payload->>'account_id' AS VARCHAR)), created_at DESC, id DESC);
        CREATE INDEX ix_position_portfolio_instrument_created_id
            ON public.position
                ((CAST(payload->>'portfolio_id' AS VARCHAR)),
                 (CAST(payload->>'instrument_id' AS VARCHAR)), created_at DESC, id DESC);
        CREATE INDEX ix_order_intent_portfolio_id
            ON public.order_intent
                ((CAST(payload->>'portfolio_id' AS VARCHAR)), id);
        ANALYZE public.account_snapshot;
        ANALYZE public.balance_snapshot;
        ANALYZE public.position;
        ANALYZE public.order_intent;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX public.ix_order_intent_portfolio_id;
        DROP INDEX public.ix_position_portfolio_instrument_created_id;
        DROP INDEX public.ix_balance_snapshot_account_created_id;
        DROP INDEX public.ix_account_snapshot_account_observed_id;
        """
    )
