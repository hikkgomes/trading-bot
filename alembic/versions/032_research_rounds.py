"""Persist bounded research epochs and convert terminal trial dispositions once."""

import hashlib
import json

from sqlalchemy import text

from alembic import op

revision = "platform_v2_0032"
down_revision = "platform_v2_0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE research_epoch (
            id VARCHAR(80) PRIMARY KEY, product_id VARCHAR(80) NOT NULL,
            dataset_bundle_id VARCHAR(80) NOT NULL UNIQUE REFERENCES dataset_bundle(id),
            state VARCHAR(40) NOT NULL, round_cursor INTEGER NOT NULL,
            budget INTEGER NOT NULL, consumed INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            CONSTRAINT ck_research_epoch_budget CHECK (round_cursor >= 0 AND consumed >= 0 AND budget >= consumed)
        );
        CREATE INDEX ix_research_epoch_product_id ON research_epoch(product_id);
        CREATE TABLE research_round (
            id VARCHAR(80) PRIMARY KEY, epoch_id VARCHAR(80) NOT NULL REFERENCES research_epoch(id),
            round_number INTEGER NOT NULL, state VARCHAR(40) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
            CONSTRAINT uq_research_round_cursor UNIQUE(epoch_id, round_number)
        );
        UPDATE experiment SET metadata = metadata || jsonb_build_object('prior_terminal_state', state),
            state = 'infrastructure_exhausted'
        WHERE state LIKE 'blocked_dead_letter:%';
        GRANT SELECT, INSERT, UPDATE ON research_epoch, research_round TO trading_runtime, trading_research;
        GRANT SELECT ON research_epoch, research_round TO trading_evaluator;
        REVOKE ALL ON research_epoch, research_round FROM trading_agent;
        CREATE FUNCTION trading_platform_guard_research_epoch() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN
            IF NEW.round_cursor < OLD.round_cursor OR NEW.consumed < OLD.consumed
                OR NEW.budget <> OLD.budget OR NEW.dataset_bundle_id <> OLD.dataset_bundle_id
                OR (OLD.state IN ('exhausted','closed_for_selection') AND NEW.state <> OLD.state) THEN
                RAISE EXCEPTION 'research epoch cannot reopen, lose trials or change its budget';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER research_epoch_monotonic BEFORE UPDATE ON research_epoch
            FOR EACH ROW EXECUTE FUNCTION trading_platform_guard_research_epoch();
        CREATE FUNCTION trading_platform_guard_research_round() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN
            IF OLD.state = 'completed' OR NEW.payload->'request' <> OLD.payload->'request' THEN
                RAISE EXCEPTION 'research round request and completed result are immutable';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER research_round_immutable BEFORE UPDATE ON research_round
            FOR EACH ROW EXECUTE FUNCTION trading_platform_guard_research_round();
        CREATE FUNCTION trading_platform_guard_candidate_epoch() RETURNS trigger
            LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
        DECLARE product TEXT;
        BEGIN
            SELECT d.product_id INTO product FROM public.strategy_definition d
                JOIN public.strategy_version v ON v.definition_id=d.id WHERE v.id=NEW.strategy_version_id;
            PERFORM pg_advisory_xact_lock(hashtext('selection-cohort:' || product));
            IF EXISTS (SELECT 1 FROM public.research_epoch e WHERE e.dataset_bundle_id=NEW.dataset_bundle_id
                        AND e.state IN ('closed_for_selection','exhausted'))
                OR EXISTS (SELECT 1 FROM public.selection_cohort c WHERE c.payload->>'protected_snapshot_id' =
                           NEW.metadata->'_dataset_plan'->>'protected_holdout_snapshot_id') THEN
                RAISE EXCEPTION 'research candidate set is already closed';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER candidate_epoch_closed BEFORE INSERT ON experiment
            FOR EACH ROW EXECUTE FUNCTION trading_platform_guard_candidate_epoch();
    """)
    _adopt_existing_trials()


def _adopt_existing_trials() -> None:
    connection = op.get_bind()
    rows = (
        connection.execute(
            text("""
        SELECT b.id, b.product_id, array_agg(e.id ORDER BY e.id) AS candidates,
            EXISTS (SELECT 1 FROM selection_cohort c
                WHERE c.payload->>'protected_snapshot_id' = b.payload->'stage_snapshot_ids'->>'protected_holdout') AS frozen
        FROM dataset_bundle b JOIN experiment e ON e.dataset_bundle_id=b.id
        WHERE e.provider='bounded_hypothesis_generator'
        GROUP BY b.id, b.product_id
    """)
        )
        .mappings()
        .all()
    )
    for row in rows:
        identity = {
            "product_id": row["product_id"],
            "bundle_id": row["id"],
            "engine_version": "economic-engine/v4",
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {
            "engine_version": "economic-engine/v4",
            "adopted_candidate_ids": row["candidates"],
            "migration": revision,
            "last_completion": None,
        }
        connection.execute(
            text("""
            INSERT INTO research_epoch VALUES (:id,:product,:bundle,:state,0,:budget,:used,CURRENT_TIMESTAMP,CAST(:payload AS jsonb))
        """),
            {
                "id": "sha256:" + digest,
                "product": row["product_id"],
                "bundle": row["id"],
                "state": "closed_for_selection" if row["frozen"] else "waiting_for_feedback",
                "budget": max(100, len(row["candidates"])),
                "used": len(row["candidates"]),
                "payload": json.dumps(payload),
            },
        )


def downgrade() -> None:
    raise RuntimeError("Research trials, cursors and exclusions must be retained")
