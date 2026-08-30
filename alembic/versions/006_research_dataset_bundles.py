"""Add immutable, complete research dataset bundles."""

from alembic import op


revision = "platform_v2_0006"
down_revision = "platform_v2_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE dataset_bundle (
            id VARCHAR(160) PRIMARY KEY,
            product_id VARCHAR(80) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_dataset_bundle_product_id ON dataset_bundle (product_id);
        CREATE INDEX ix_dataset_bundle_created_at ON dataset_bundle (created_at);
        ALTER TABLE experiment
            ADD COLUMN IF NOT EXISTS dataset_bundle_id VARCHAR(160)
            REFERENCES dataset_bundle(id);
        CREATE INDEX IF NOT EXISTS ix_experiment_dataset_bundle_id
            ON experiment (dataset_bundle_id);
        CREATE TRIGGER dataset_bundle_append_only
        BEFORE UPDATE OR DELETE ON dataset_bundle
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        GRANT SELECT, INSERT ON dataset_bundle TO trading_research;
        GRANT SELECT ON dataset_bundle TO trading_runtime, trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("research dataset bundle downgrade is forbidden")
