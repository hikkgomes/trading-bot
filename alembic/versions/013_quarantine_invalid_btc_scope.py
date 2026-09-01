"""Quarantine historical BTC candidates outside the BTCUSDT spot scope."""

from alembic import op

revision = "platform_v2_0013"
down_revision = "platform_v2_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE experiment AS e
           SET state = 'legacy_import',
               metadata = COALESCE(e.metadata, '{}'::jsonb) || jsonb_build_object(
                   'legacy_quarantine', jsonb_build_object(
                       'field', 'universe.symbols',
                       'original_state', e.state,
                       'reason_code', 'btc_universe_outside_btcusdt_spot'
                   )
               )
          FROM strategy_version AS sv
          JOIN strategy_definition AS sd
            ON sd.id = sv.definition_id
         WHERE sv.id = e.strategy_version_id
           AND sd.product_id = 'btc_accumulation'
           AND e.state <> 'legacy_import'
           AND NOT (
               CASE
                   WHEN jsonb_typeof(sd.definition->'universe'->'symbols') = 'array'
                   THEN jsonb_array_length(sd.definition->'universe'->'symbols') = 1
                        AND sd.definition->'universe'->'symbols' @> '["BTCUSDT"]'::jsonb
                   ELSE FALSE
               END
           );
        """
    )


def downgrade() -> None:
    raise RuntimeError("BTC universe quarantine downgrade is forbidden")
