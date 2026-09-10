"""Retain every provider's historical trial in the existing epoch budget."""

import hashlib
import json

from sqlalchemy import text

from alembic import op

revision = "platform_v2_0036"
down_revision = "platform_v2_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    # PostgreSQL transactional DDL restores the trigger on rollback as well.
    op.execute("LOCK TABLE research_epoch, experiment IN SHARE ROW EXCLUSIVE MODE")
    op.execute("ALTER TABLE research_epoch DISABLE TRIGGER research_epoch_monotonic")
    rows = (
        connection.execute(
            text("""
        SELECT b.id, b.product_id, array_agg(e.id ORDER BY e.id) AS candidates,
            EXISTS (SELECT 1 FROM selection_cohort c
                WHERE c.payload->>'protected_snapshot_id' =
                    b.payload->'stage_snapshot_ids'->>'protected_holdout') AS frozen
        FROM dataset_bundle b JOIN experiment e ON e.dataset_bundle_id=b.id
        GROUP BY b.id, b.product_id
    """)
        )
        .mappings()
        .all()
    )
    for row in rows:
        _retain_trials(connection, row)
    op.execute("ALTER TABLE research_epoch ENABLE TRIGGER research_epoch_monotonic")


def _retain_trials(connection, row):
    previous = (
        connection.execute(
            text("SELECT * FROM research_epoch WHERE dataset_bundle_id=:bundle"),
            {"bundle": row["id"]},
        )
        .mappings()
        .first()
    )
    identity = {
        "product_id": row["product_id"],
        "bundle_id": row["id"],
        "engine_version": "economic-engine/v4",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    consumed = max(int(previous["consumed"]) if previous else 0, len(row["candidates"]))
    budget = max(int(previous["budget"]) if previous else 100, consumed)
    state = previous["state"] if previous else "waiting_for_feedback"
    if state not in {"exhausted", "closed_for_selection"}:
        state = (
            "closed_for_selection"
            if row["frozen"]
            else "exhausted"
            if consumed >= budget
            else state
        )
    payload = (
        dict(previous["payload"])
        if previous
        else {"engine_version": "economic-engine/v4", "last_completion": None}
    )
    payload["all_provider_count_reconciliation"] = {
        "migration": revision,
        "candidate_ids": row["candidates"],
        "previous_consumed": previous["consumed"] if previous else 0,
        "previous_budget": previous["budget"] if previous else None,
    }
    connection.execute(
        text("""
        INSERT INTO research_epoch
            (id,product_id,dataset_bundle_id,state,round_cursor,budget,consumed,created_at,payload)
        VALUES (:id,:product,:bundle,:state,0,:budget,:consumed,CURRENT_TIMESTAMP,CAST(:payload AS jsonb))
        ON CONFLICT (dataset_bundle_id) DO UPDATE
        SET state=EXCLUDED.state, budget=EXCLUDED.budget, consumed=EXCLUDED.consumed,
            payload=EXCLUDED.payload
    """),
        {
            "id": previous["id"] if previous else "sha256:" + digest,
            "product": row["product_id"],
            "bundle": row["id"],
            "state": state,
            "budget": budget,
            "consumed": consumed,
            "payload": json.dumps(payload),
        },
    )


def downgrade() -> None:
    raise RuntimeError("Historical trials and protected exposure must remain consumed")
