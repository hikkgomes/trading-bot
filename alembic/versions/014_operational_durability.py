"""Add resumable history bootstrap and durable alert delivery state."""

from alembic import op

revision = "platform_v2_0014"
down_revision = "platform_v2_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_bootstrap_progress (
            id VARCHAR(240) PRIMARY KEY,
            product_id VARCHAR(80) NOT NULL,
            market VARCHAR(40) NOT NULL,
            symbol VARCHAR(80) NOT NULL,
            timeframe VARCHAR(40) NOT NULL,
            range_start TIMESTAMPTZ NOT NULL,
            range_end TIMESTAMPTZ NOT NULL,
            next_start TIMESTAMPTZ NOT NULL,
            status VARCHAR(40) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL,
            CONSTRAINT uq_historical_bootstrap_range
                UNIQUE (product_id, market, symbol, timeframe, range_start, range_end)
        );
        CREATE INDEX IF NOT EXISTS ix_historical_bootstrap_progress_product
            ON historical_bootstrap_progress (product_id);
        CREATE INDEX IF NOT EXISTS ix_historical_bootstrap_progress_status
            ON historical_bootstrap_progress (status);
        CREATE INDEX IF NOT EXISTS ix_historical_bootstrap_progress_updated
            ON historical_bootstrap_progress (updated_at);

        CREATE TABLE IF NOT EXISTS alert_delivery_attempt (
            id VARCHAR(240) PRIMARY KEY,
            alert_id VARCHAR(160) NOT NULL REFERENCES alert(id),
            sink VARCHAR(120) NOT NULL,
            attempt_no INTEGER NOT NULL,
            attempted_at TIMESTAMPTZ NOT NULL,
            status VARCHAR(40) NOT NULL,
            error TEXT,
            next_attempt_at TIMESTAMPTZ,
            acknowledged_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            CONSTRAINT uq_alert_delivery_attempt UNIQUE (alert_id, sink, attempt_no)
        );
        CREATE INDEX IF NOT EXISTS ix_alert_delivery_attempt_alert
            ON alert_delivery_attempt (alert_id);
        CREATE INDEX IF NOT EXISTS ix_alert_delivery_attempt_pending
            ON alert_delivery_attempt (status, next_attempt_at);
        """
    )


def downgrade() -> None:
    raise RuntimeError("operational durability downgrade is forbidden")
