"""Keep behavioural identity independent from strategy-definition identity."""

from alembic import op

revision = "platform_v2_0011"
down_revision = "platform_v2_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            constraint_name text;
        BEGIN
            SELECT pg_constraint.conname
              INTO constraint_name
              FROM pg_constraint
              JOIN pg_attribute
                ON pg_attribute.attrelid = pg_constraint.conrelid
               AND pg_attribute.attnum = ANY(pg_constraint.conkey)
             WHERE pg_constraint.conrelid = 'strategy_identity'::regclass
               AND pg_constraint.contype = 'f'
               AND pg_attribute.attname = 'behavior_hash';
            IF constraint_name IS NOT NULL THEN
                EXECUTE format(
                    'ALTER TABLE strategy_identity DROP CONSTRAINT %I',
                    constraint_name
                );
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    raise RuntimeError("behaviour identity foreign-key downgrade is forbidden")
