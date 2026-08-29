"""Persist autonomous scheduler, bootstrap, account, and rehearsal authority."""

from alembic import op

revision = "platform_v2_0005"
down_revision = "platform_v2_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE platform_schedule (
            id VARCHAR(200) PRIMARY KEY,
            job_name VARCHAR(160) NOT NULL UNIQUE,
            interval_seconds INTEGER NOT NULL,
            next_run_at TIMESTAMPTZ NOT NULL,
            last_run_at TIMESTAMPTZ,
            last_job_id VARCHAR(200),
            state VARCHAR(40) NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT ck_platform_schedule_interval_positive CHECK (interval_seconds > 0)
        );
        CREATE INDEX ix_platform_schedule_next_run_at ON platform_schedule (next_run_at);
        CREATE INDEX ix_platform_schedule_last_run_at ON platform_schedule (last_run_at);

        CREATE TABLE platform_bootstrap (
            id VARCHAR(160) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );

        CREATE TABLE cost_model_manifest (
            id VARCHAR(160) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        );

        CREATE TABLE account_snapshot (
            id VARCHAR(200) PRIMARY KEY,
            account_id VARCHAR(160) NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            source VARCHAR(80) NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_account_snapshot_account_id ON account_snapshot (account_id);
        CREATE INDEX ix_account_snapshot_observed_at ON account_snapshot (observed_at);

        CREATE TABLE platform_rehearsal_report (
            id VARCHAR(200) PRIMARY KEY,
            product_id VARCHAR(80) NOT NULL,
            account_id VARCHAR(160) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            content_hash VARCHAR(80) NOT NULL UNIQUE,
            accepted BOOLEAN NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_platform_rehearsal_report_product_id
            ON platform_rehearsal_report (product_id);
        CREATE INDEX ix_platform_rehearsal_report_created_at
            ON platform_rehearsal_report (created_at);

        CREATE TRIGGER platform_bootstrap_append_only
        BEFORE UPDATE OR DELETE ON platform_bootstrap
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER cost_model_manifest_append_only
        BEFORE UPDATE OR DELETE ON cost_model_manifest
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER account_snapshot_append_only
        BEFORE UPDATE OR DELETE ON account_snapshot
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        CREATE TRIGGER platform_rehearsal_report_append_only
        BEFORE UPDATE OR DELETE ON platform_rehearsal_report
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();

        GRANT SELECT, INSERT, UPDATE ON platform_schedule TO trading_runtime;
        GRANT SELECT ON platform_bootstrap TO trading_runtime;
        GRANT SELECT, INSERT ON cost_model_manifest TO trading_runtime;
        GRANT SELECT, INSERT ON account_snapshot, platform_rehearsal_report TO trading_runtime;
        GRANT SELECT, INSERT, UPDATE ON platform_schedule TO trading_research;
        GRANT SELECT ON platform_bootstrap, account_snapshot, platform_rehearsal_report,
            cost_model_manifest
            TO trading_research;
        GRANT SELECT ON platform_schedule, platform_bootstrap, account_snapshot,
            platform_rehearsal_report, cost_model_manifest TO trading_agent;
        """
    )


def downgrade() -> None:
    raise RuntimeError("platform bootstrap authority downgrade is forbidden")
