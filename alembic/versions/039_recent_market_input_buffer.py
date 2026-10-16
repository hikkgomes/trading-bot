"""Store unpinned high-rate market inputs in a short-lived buffer."""

from alembic import op

revision = "platform_v2_0039"
down_revision = "platform_v2_0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE market_input_buffer (
            id VARCHAR(160) PRIMARY KEY,
            product_id VARCHAR(80) NOT NULL,
            instrument_id VARCHAR(160) NOT NULL,
            event_type VARCHAR(80) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL,
            CONSTRAINT ck_market_input_buffer_event_type CHECK (
                event_type IN (
                    'best_bid_ask', 'depth_update', 'mark_price', 'funding_rate',
                    'trade', 'aggregate_trade', 'candle'
                )
            ),
            CONSTRAINT ck_market_input_buffer_kind CHECK (
                payload->>'kind' IS NOT DISTINCT FROM 'market_data_input'
            ),
            CONSTRAINT ck_market_input_buffer_source_type CHECK (
                payload->>'source_event_type' IS NOT DISTINCT FROM event_type
            ),
            CONSTRAINT ck_market_input_buffer_unclosed_candle CHECK (
                event_type <> 'candle'
                OR payload->>'source_candle_closed' IS DISTINCT FROM 'true'
            ),
            CONSTRAINT ck_market_input_buffer_scope CHECK (
                payload->>'product_id' IS NOT DISTINCT FROM product_id
                AND payload->>'instrument_id' IS NOT DISTINCT FROM instrument_id
            )
        );
        CREATE INDEX ix_market_input_buffer_scope_time
            ON market_input_buffer(product_id, instrument_id, created_at DESC, id DESC);
        CREATE INDEX ix_market_input_buffer_time ON market_input_buffer(created_at);
        REVOKE ALL ON TABLE market_input_buffer FROM PUBLIC;
        GRANT SELECT, INSERT, DELETE ON TABLE market_input_buffer TO trading_runtime;
        REVOKE UPDATE ON TABLE market_input_buffer FROM trading_runtime;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX ix_market_input_buffer_time;
        DROP INDEX ix_market_input_buffer_scope_time;
        DROP TABLE market_input_buffer;
        """
    )
