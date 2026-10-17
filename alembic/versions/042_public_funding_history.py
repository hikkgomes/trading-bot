"""Expose public settlement history without account funding ledger access."""

from alembic import op

revision = "platform_v2_0042"
down_revision = "platform_v2_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE VIEW public.public_funding_history WITH (security_barrier=true) AS
        SELECT f.id, f.created_at,
            (SELECT jsonb_object_agg(item.key, item.value)
             FROM jsonb_each(f.payload) AS item
             WHERE item.key IN (
                'kind','product_id','instrument_id','symbol','funding_time','funding_rate',
                'mark_price','source','source_event_id','supersedes_record_id','availability_time'
             )) AS payload
        FROM public.funding_entry AS f
        WHERE f.payload->>'kind'='historical_funding'
            AND f.payload->>'source'='binance_public_funding_rate';
        REVOKE ALL ON public.public_funding_history FROM PUBLIC;
        GRANT SELECT ON public.public_funding_history
            TO trading_runtime, trading_research, trading_evaluator;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW public.public_funding_history")
