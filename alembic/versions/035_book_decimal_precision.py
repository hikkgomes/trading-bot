"""Preserve finite allocation amounts without rounding valid fractional fills."""

from alembic import op

revision = "platform_v2_0035"
down_revision = "platform_v2_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE book_fill_allocation ALTER COLUMN signed_quantity TYPE NUMERIC;
        ALTER TABLE book_fill_allocation ALTER COLUMN fee TYPE NUMERIC;
        ALTER TABLE book_fill_allocation ADD CONSTRAINT ck_book_fill_finite CHECK (
            signed_quantity NOT IN ('NaN'::NUMERIC,'Infinity'::NUMERIC,'-Infinity'::NUMERIC)
            AND fee NOT IN ('NaN'::NUMERIC,'Infinity'::NUMERIC,'-Infinity'::NUMERIC)
        );
    """)


def downgrade() -> None:
    raise RuntimeError("Financial allocations must not lose decimal precision")
