"""Multi-strategy assignment and immutable risk-policy authority."""

from alembic import op

revision = "platform_v2_0003"
down_revision = "platform_v2_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
        IF to_regclass('public.active_strategy_assignment') IS NOT NULL THEN
        ALTER TABLE active_strategy_assignment
            ADD COLUMN IF NOT EXISTS sleeve_id VARCHAR(160) NOT NULL DEFAULT 'default',
            ADD COLUMN IF NOT EXISTS instrument_id VARCHAR(160),
            ADD COLUMN IF NOT EXISTS universe_id VARCHAR(160),
            ADD COLUMN IF NOT EXISTS assignment_scope_id VARCHAR(180),
            ADD COLUMN IF NOT EXISTS risk_budget NUMERIC(24,12) NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS active_until TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS assignment_reason TEXT NOT NULL DEFAULT 'unspecified';
        UPDATE active_strategy_assignment
        SET universe_id = COALESCE(universe_id, 'product:' || product_id),
            assignment_scope_id = COALESCE(
                assignment_scope_id,
                CASE
                    WHEN instrument_id IS NOT NULL THEN 'instrument:' || instrument_id
                    ELSE 'universe:' || COALESCE(universe_id, 'product:' || product_id)
                END
            );
        ALTER TABLE active_strategy_assignment
            ALTER COLUMN assignment_scope_id SET NOT NULL;
        ALTER TABLE active_strategy_assignment
            DROP CONSTRAINT IF EXISTS ck_assignment_instrument_xor_universe,
            DROP CONSTRAINT IF EXISTS ck_assignment_risk_nonnegative;
        ALTER TABLE active_strategy_assignment
            ADD CONSTRAINT ck_assignment_instrument_xor_universe CHECK (
                (instrument_id IS NOT NULL AND universe_id IS NULL)
                OR (instrument_id IS NULL AND universe_id IS NOT NULL)
            ),
            ADD CONSTRAINT ck_assignment_risk_nonnegative CHECK (risk_budget >= 0);
        DROP INDEX IF EXISTS uq_active_strategy_assignment_product_active;
        CREATE INDEX IF NOT EXISTS ix_active_strategy_assignment_authority_event
        ON active_strategy_assignment (
            product_id, portfolio_id, sleeve_id, strategy_version_id,
            assignment_scope_id, execution_mode
        );

        EXECUTE 'CREATE OR REPLACE VIEW current_strategy_assignment AS
        SELECT latest.* FROM (
            SELECT DISTINCT ON (
                product_id, portfolio_id, sleeve_id, strategy_version_id,
                assignment_scope_id, execution_mode
            ) assignment.*
            FROM active_strategy_assignment AS assignment
            WHERE assignment.assigned_at::timestamptz <= CURRENT_TIMESTAMP
            ORDER BY product_id, portfolio_id, sleeve_id, strategy_version_id,
                assignment_scope_id, execution_mode,
                assignment.assigned_at::timestamptz DESC, assignment.id DESC
        ) AS latest
        WHERE latest.active
          AND (latest.active_until IS NULL OR latest.active_until > CURRENT_TIMESTAMP)';
        END IF;
        END $$;

        CREATE TABLE IF NOT EXISTS risk_policy (
            id VARCHAR(200) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE TRIGGER active_strategy_assignment_append_only
        BEFORE UPDATE OR DELETE ON active_strategy_assignment
        FOR EACH ROW EXECUTE FUNCTION trading_platform_reject_mutation();
        """
    )


def downgrade() -> None:
    raise RuntimeError("platform authority downgrade is forbidden")
