"""Hide protected samples and outcomes from adaptive database roles."""

from alembic import op

revision = "platform_v2_0024"
down_revision = "platform_v2_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("REVOKE ALL ON holdout_outcome FROM trading_research, trading_agent")
    filters = {
        "dataset_snapshot": "payload->>'role' IN ('screening', 'development', 'robustness', 'forward_observation')",
        "experiment_run": "COALESCE(payload->>'phase', '') NOT IN ('holdout', 'protected', 'final_holdout', 'final') AND COALESCE(payload->>'run_name', '') <> 'canonical:protected'",
        "experiment_metric": "EXISTS (SELECT 1 FROM experiment_run r WHERE r.id = experiment_metric.payload->>'run_id')",
        "validation_result": "COALESCE(evidence->>'phase', '') NOT IN ('holdout', 'protected', 'final_holdout', 'final')",
        "validation_stage": "stage IN ('screening', 'development', 'robustness', 'forward')",
    }
    for table, condition in filters.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_runtime ON {table} TO trading_runtime USING (true) WITH CHECK (true)"
        )
        op.execute(
            f"CREATE POLICY {table}_adaptive_read ON {table} FOR SELECT TO trading_research USING ({condition})"
        )


def downgrade() -> None:
    raise RuntimeError("Protected research visibility cannot be weakened by downgrade")
