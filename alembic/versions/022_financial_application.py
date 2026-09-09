"""Persist scoped financial application receipts and recoverable follow-up work."""

from alembic import op

revision = "platform_v2_0022"
down_revision = "platform_v2_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE financial_application (
            id VARCHAR(160) PRIMARY KEY REFERENCES fill(id),
            created_at TIMESTAMPTZ NOT NULL, account_id VARCHAR(160) NOT NULL,
            environment VARCHAR(40) NOT NULL, venue VARCHAR(40) NOT NULL,
            instrument_id VARCHAR(160) NOT NULL, exchange_trade_id VARCHAR(240) NOT NULL,
            payload JSONB NOT NULL,
            CONSTRAINT uq_financial_scoped_fill UNIQUE(account_id, environment, venue, instrument_id, exchange_trade_id)
        );
        CREATE TABLE financial_outbox (
            id VARCHAR(160) PRIMARY KEY REFERENCES financial_application(id),
            created_at TIMESTAMPTZ NOT NULL, processed_at TIMESTAMPTZ, payload JSONB NOT NULL
        );
        CREATE TRIGGER financial_application_append_only BEFORE UPDATE OR DELETE ON financial_application
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON financial_application TO trading_runtime;
        GRANT SELECT, INSERT, UPDATE ON financial_outbox TO trading_runtime;
        REVOKE ALL ON financial_application, financial_outbox FROM trading_research, trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Financial application history downgrade is forbidden")
