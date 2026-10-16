"""Bound operational lookup work as immutable event history grows."""

from alembic import op

revision = "platform_v2_0037"
down_revision = "platform_v2_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX ix_risk_snapshot_created ON risk_snapshot(created_at DESC, id DESC);
        CREATE INDEX ix_risk_snapshot_market_scope ON risk_snapshot
            ((CAST(payload->>'kind' AS VARCHAR)), (CAST(payload->>'product_id' AS VARCHAR)),
             (CAST(payload->>'instrument_id' AS VARCHAR)), created_at DESC);
        CREATE INDEX ix_risk_snapshot_product_time ON risk_snapshot
            ((CAST(payload->>'kind' AS VARCHAR)), (CAST(payload->>'product_id' AS VARCHAR)),
             created_at DESC);
        CREATE INDEX ix_job_state_name_available ON job(state, name, available_at);
        CREATE INDEX ix_risk_snapshot_candle_scope ON risk_snapshot
            ((CAST(payload->>'kind' AS VARCHAR)), (CAST(payload->>'product_id' AS VARCHAR)),
             (CAST(payload->>'source_event_type' AS VARCHAR)), (CAST(payload->>'source_timeframe' AS VARCHAR)),
             (CAST(payload->>'instrument_id' AS VARCHAR)), created_at);
        ANALYZE risk_snapshot;
        ANALYZE job;
    """)


def downgrade() -> None:
    for name in (
        "ix_risk_snapshot_candle_scope",
        "ix_job_state_name_available",
        "ix_risk_snapshot_product_time",
        "ix_risk_snapshot_market_scope",
        "ix_risk_snapshot_created",
    ):
        op.drop_index(name)
