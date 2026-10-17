"""Keep recovered retry budgets out of immutable job identity checks."""

from alembic import op

revision = "platform_v2_0041"
down_revision = "platform_v2_0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Preserve the function's role checks, grants and immutable payload contract.
    op.execute(
        """
        DO $$
        DECLARE definition TEXT;
            original TEXT := 'OR previous.max_attempts IS DISTINCT FROM (request->>''max_attempts'')::int';
            replacement TEXT := 'OR (previous.attempts=0 AND previous.max_attempts IS DISTINCT FROM (request->>''max_attempts'')::int)';
        BEGIN
            SELECT pg_get_functiondef('public.submit_typed_research_job(jsonb)'::regprocedure)
                INTO definition;
            IF strpos(definition, original)=0 THEN
                RAISE EXCEPTION 'unexpected typed job submission function';
            END IF;
            EXECUTE replace(definition, original, replacement);
        END $$;
        """
    )


def downgrade() -> None:
    raise RuntimeError("recovered job replay must remain idempotent")
