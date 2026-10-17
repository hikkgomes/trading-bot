"""Index ordered market-input scopes for report latest-row walks."""

from alembic import op

revision = "platform_v2_0049"
down_revision = "platform_v2_0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX ix_risk_snapshot_market_scope_latest
        ON risk_snapshot (
            (COALESCE(CAST(payload->>'product_id' AS VARCHAR), '')),
            (CAST(payload->>'product_id' AS VARCHAR) IS NULL),
            (COALESCE(CAST(payload->>'instrument_id' AS VARCHAR), '')),
            (CAST(payload->>'instrument_id' AS VARCHAR) IS NULL),
            created_at DESC,
            id DESC
        )
        WHERE CAST(payload->>'kind' AS VARCHAR) = 'market_data_input'
        """
    )


def downgrade() -> None:
    op.drop_index("ix_risk_snapshot_market_scope_latest", table_name="risk_snapshot")
