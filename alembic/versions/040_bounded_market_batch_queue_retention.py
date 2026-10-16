"""Retain only a bounded recent history of completed public market batches."""

from alembic import op

revision = "platform_v2_0040"
down_revision = "platform_v2_0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_job_completed_public_market_batch_retention
            ON public.job (available_at, id)
            WHERE name = 'market_event_batch_write'
                AND state = 'completed'
                AND payload->>'queue_retention_policy' = 'public_market_batch_completed_7d/v1';
        REVOKE DELETE ON TABLE public.job, public.job_attempt FROM PUBLIC, trading_runtime;
        CREATE OR REPLACE FUNCTION public.prune_completed_public_market_batch_jobs()
        RETURNS INTEGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
        DECLARE
            expired_ids VARCHAR[];
            deleted_count INTEGER := 0;
            cutoff TIMESTAMPTZ := pg_catalog.clock_timestamp() - INTERVAL '7 days';
        BEGIN
            SELECT array_agg(eligible.id) INTO expired_ids
            FROM (
                SELECT j.id
                FROM public.job AS j
                WHERE j.name = 'market_event_batch_write'
                    AND j.state = 'completed'
                    AND j.available_at < cutoff
                    AND j.payload->>'queue_retention_policy' =
                        'public_market_batch_completed_7d/v1'
                    AND EXISTS (
                        SELECT 1 FROM public.job_attempt AS a
                        WHERE a.job_id = j.id AND a.status = 'completed'
                            AND a.completed_at < cutoff
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM public.job_attempt AS a
                        WHERE a.job_id = j.id AND a.status = 'completed'
                            AND (a.completed_at IS NULL OR a.completed_at >= cutoff)
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM public.worker_lease AS lease
                        WHERE lease.job_id = j.id AND lease.status = 'active'
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM public.job_quarantine AS audit
                        WHERE audit.job_id = j.id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM public.emergency_action AS action
                        WHERE action.job_id = j.id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM public.heavy_compute_lease AS lease
                        WHERE lease.job_id = j.id
                    )
                ORDER BY j.available_at, j.id
                LIMIT 1000
                FOR UPDATE OF j SKIP LOCKED
            ) AS eligible;

            IF COALESCE(array_length(expired_ids, 1), 0) = 0 THEN
                RETURN 0;
            END IF;

            DELETE FROM public.worker_lease WHERE job_id = ANY(expired_ids);
            DELETE FROM public.job_attempt WHERE job_id = ANY(expired_ids);
            DELETE FROM public.job WHERE id = ANY(expired_ids);
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END
        $$;
        REVOKE ALL ON FUNCTION public.prune_completed_public_market_batch_jobs() FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION public.prune_completed_public_market_batch_jobs()
            TO trading_runtime;
        """
    )


def downgrade() -> None:
    raise RuntimeError("expired public market queue history cannot be restored")
