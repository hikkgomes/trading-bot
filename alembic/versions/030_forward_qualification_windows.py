"""Retain prospective operational paper windows independently of financial positions."""

from alembic import op

revision = "platform_v2_0030"
down_revision = "platform_v2_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE forward_qualification_window (
            id VARCHAR(80) PRIMARY KEY,
            strategy_version_id VARCHAR(160) NOT NULL REFERENCES strategy_version(id),
            product_id VARCHAR(80) NOT NULL,
            artefact_hash VARCHAR(80) NOT NULL,
            opened_at TIMESTAMPTZ NOT NULL,
            previous_window_id VARCHAR(80) UNIQUE REFERENCES forward_qualification_window(id),
            source_event_id VARCHAR(160) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_forward_qualification_window_strategy_version_id ON forward_qualification_window(strategy_version_id);
        CREATE INDEX ix_forward_qualification_window_product_id ON forward_qualification_window(product_id);
        CREATE INDEX ix_forward_qualification_window_artefact_hash ON forward_qualification_window(artefact_hash);
        CREATE INDEX ix_forward_qualification_window_opened_at ON forward_qualification_window(opened_at);
        CREATE UNIQUE INDEX uq_forward_window_initial_scope ON forward_qualification_window(strategy_version_id, product_id, artefact_hash) WHERE previous_window_id IS NULL;
        CREATE TRIGGER forward_qualification_window_append_only
            BEFORE UPDATE OR DELETE ON forward_qualification_window
            FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON forward_qualification_window TO trading_runtime;
        REVOKE ALL ON forward_qualification_window FROM trading_research, trading_agent;
    """)


def downgrade() -> None:
    raise RuntimeError("Forward qualification window history cannot be removed")
