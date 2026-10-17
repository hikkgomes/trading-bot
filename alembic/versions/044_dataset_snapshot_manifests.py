"""Expose bounded immutable snapshot metadata to service roles."""

from alembic import op

revision = "platform_v2_0044"
down_revision = "platform_v2_0043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION public.dataset_snapshot_manifest(snapshot_payload JSONB)
        RETURNS JSONB
        LANGUAGE SQL
        IMMUTABLE
        PARALLEL SAFE
        SET search_path = pg_catalog
        AS $$
            SELECT jsonb_strip_nulls(jsonb_build_object(
                'snapshot_id', snapshot_payload->'snapshot_id',
                'product_id', snapshot_payload->'product_id',
                'role', snapshot_payload->'role',
                'content_hash', snapshot_payload->'content_hash',
                'interval', snapshot_payload->'interval',
                'universe_snapshot_id', snapshot_payload->'universe_snapshot_id',
                'availability_timestamp', snapshot_payload->'availability_timestamp',
                'feature_manifest_id', snapshot_payload->'feature_manifest_id',
                'cost_model_id', snapshot_payload->'cost_model_id',
                'parameter_set_id', snapshot_payload->'parameter_set_id',
                'engine_version', snapshot_payload->'engine_version',
                'instrument_scope', snapshot_payload->'instrument_scope',
                'model_artefact_id', snapshot_payload->'model_artefact_id',
                'event_data_segment_ids', snapshot_payload->'event_data_segment_ids',
                'non_promotable', CASE
                    WHEN jsonb_typeof(snapshot_payload) IS DISTINCT FROM 'object' THEN TRUE
                    ELSE COALESCE(
                        snapshot_payload->'synthetic' = 'true'::jsonb
                        OR snapshot_payload->'diagnostic' = 'true'::jsonb
                        OR snapshot_payload->'promotable' = 'false'::jsonb
                        OR snapshot_payload->'payload'->'synthetic' = 'true'::jsonb
                        OR snapshot_payload->'payload'->'diagnostic' = 'true'::jsonb
                        OR snapshot_payload->'payload'->'promotable' = 'false'::jsonb,
                        FALSE
                    )
                END
            ))
        $$;
        REVOKE ALL ON FUNCTION public.dataset_snapshot_manifest(JSONB) FROM PUBLIC;
        GRANT EXECUTE ON FUNCTION public.dataset_snapshot_manifest(JSONB)
            TO trading_runtime, trading_evaluator, trading_platform_owner;

        ALTER TABLE public.dataset_snapshot
            ADD COLUMN manifest JSONB GENERATED ALWAYS AS
                (public.dataset_snapshot_manifest(payload)) STORED;

        CREATE VIEW public.dataset_snapshot_manifest WITH (security_barrier=true) AS
        SELECT
            s.id AS snapshot_id,
            s.manifest->>'product_id' AS product_id,
            s.manifest->>'role' AS role,
            s.manifest->>'content_hash' AS content_hash,
            s.manifest->'interval' AS "interval",
            s.manifest->>'universe_snapshot_id' AS universe_snapshot_id,
            s.manifest->>'availability_timestamp' AS availability_timestamp,
            s.manifest->>'feature_manifest_id' AS feature_manifest_id,
            s.manifest->>'cost_model_id' AS cost_model_id,
            s.manifest->>'parameter_set_id' AS parameter_set_id,
            s.manifest->>'engine_version' AS engine_version,
            s.manifest->'instrument_scope' AS instrument_scope,
            s.manifest->>'model_artefact_id' AS model_artefact_id,
            s.manifest->'event_data_segment_ids' AS event_data_segment_ids,
            s.manifest->'non_promotable' = 'true'::jsonb AS non_promotable
        FROM public.dataset_snapshot AS s
        WHERE s.id = s.manifest->>'snapshot_id';
        REVOKE ALL ON public.dataset_snapshot_manifest FROM PUBLIC;
        GRANT SELECT ON public.dataset_snapshot_manifest
            TO trading_runtime, trading_research, trading_evaluator;
        """
    )


def downgrade() -> None:
    raise RuntimeError("dataset snapshot metadata access cannot be rolled back")
