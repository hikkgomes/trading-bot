"""Quarantine historical candidates with the obsolete universe contract."""

from alembic import op

revision = "platform_v2_0012"
down_revision = "platform_v2_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE experiment AS e
           SET state = 'legacy_import',
               metadata = COALESCE(e.metadata, '{}'::jsonb) || jsonb_build_object(
                   'legacy_quarantine', jsonb_build_object(
                       'field', 'universe.predeclared',
                       'original_state', e.state,
                       'reason_code', 'obsolete_universe_contract'
                   )
               )
          FROM strategy_version AS sv
          JOIN strategy_definition AS sd
            ON sd.id = sv.definition_id
         WHERE sv.id = e.strategy_version_id
           AND sd.definition->'universe' ? 'predeclared'
           AND e.state <> 'legacy_import';
        """
    )


def downgrade() -> None:
    raise RuntimeError("legacy candidate quarantine downgrade is forbidden")
