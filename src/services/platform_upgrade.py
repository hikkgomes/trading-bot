"""Backup-first upgrade actions for the PostgreSQL platform authority."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select, update

from src.data.database import (
    PlatformDatabase,
    job,
    job_attempt,
    job_quarantine,
    platform_upgrade_marker,
    worker_lease,
)
from src.domain._codec import canonical_hash, json_value, timestamp

LEGACY_RESEARCH_JOB_NAMES = frozenset(
    {
        "dataset_snapshot_validate",
        "register_strategy_catalogue",
        "generate_hypotheses",
        "candidate_generation",
        "register_candidate",
        "candidate_evaluation",
        "evaluate_candidate",
        "bounded_backtest",
        "event_replay",
        "register_ml_candidate",
        "train_ml_experiment",
        "historical_feature_calculation",
        "historical_bootstrap",
        "promotion_evaluation",
        "forward_paper_observation",
        "forward_paper_summary",
        "agent_review",
        "ml_research",
        "maintenance",
    }
)
MARKER_ID = "legacy-research-queue-quarantined-v1"


def quarantine_legacy_research_jobs(
    engine,
    *,
    now: str | None = None,
    marker_id: str = MARKER_ID,
) -> dict[str, Any]:
    """Quarantine legacy pending/dead work exactly once before scheduling."""

    observed_at = timestamp(now or dt.datetime.now(dt.UTC), field="upgrade timestamp")
    with engine.begin() as connection:
        marker = connection.execute(
            select(platform_upgrade_marker.c.payload).where(
                platform_upgrade_marker.c.id == marker_id
            )
        ).scalar_one_or_none()
        if marker is not None:
            payload = dict(marker) if isinstance(marker, Mapping) else {}
            return {"schema": "platform.upgrade/v1", "already_applied": True, **payload}
        statement = (
            select(job)
            .where(
                job.c.state.in_(("pending", "running", "dead_letter")),
                job.c.name.in_(LEGACY_RESEARCH_JOB_NAMES),
            )
            .order_by(job.c.id)
            .execution_options(yield_per=200)
        )
        quarantined: list[str] = []
        rows = connection.execute(statement).mappings()
        for batch in rows.partitions(200):
            quarantined.extend(_quarantine_batch(connection, batch, observed_at))
        marker_payload = {
            "schema": "platform.upgrade/v1",
            "marker_id": marker_id,
            "quarantined_jobs": sorted(quarantined),
            "count": len(quarantined),
            "observed_at": observed_at,
        }
        connection.execute(
            insert(platform_upgrade_marker).values(
                id=marker_id,
                created_at=observed_at,
                payload=json_value(marker_payload, field="upgrade marker"),
            )
        )
    return {"schema": "platform.upgrade/v1", "already_applied": False, **marker_payload}


def _quarantine_batch(connection, rows, observed_at: str) -> list[str]:
    ids = [str(row["id"]) for row in rows]
    reason = "legacy_research_queue_excluded_from_platform_v2"
    existing = set(
        connection.execute(
            select(job_quarantine.c.job_id).where(job_quarantine.c.job_id.in_(ids))
        ).scalars()
    )
    values = []
    for row in rows:
        if row["id"] in existing:
            continue
        payload = {
            "job_id": str(row["id"]),
            "name": str(row["name"]),
            "state": str(row["state"]),
            "content_hash": str(row["content_hash"]),
            "payload": row["payload"],
            "reason": reason,
            "quarantined_at": observed_at,
        }
        values.append(
            {
                "id": canonical_hash(payload),
                "job_id": payload["job_id"],
                "quarantined_at": observed_at,
                "original_state": payload["state"],
                "reason": reason,
                "content_hash": payload["content_hash"],
                "payload": json_value(payload, field="quarantined job"),
            }
        )
    if values:
        connection.execute(insert(job_quarantine), values)
    connection.execute(
        update(job)
        .where(job.c.id.in_(ids))
        .values(
            state="quarantined",
            lease_owner=None,
            lease_expires_at=None,
            terminal_reason=reason,
        )
    )
    connection.execute(
        update(job_attempt)
        .where(
            job_attempt.c.job_id.in_(ids),
            job_attempt.c.status == "running",
        )
        .values(status="quarantined", completed_at=observed_at, error=reason)
    )
    connection.execute(
        update(worker_lease)
        .where(
            worker_lease.c.job_id.in_(ids),
            worker_lease.c.status == "active",
        )
        .values(status="quarantined")
    )
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quarantine legacy research queue work.")
    parser.add_argument(
        "--database-url", default=os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or TRADING_PLATFORM_DATABASE_URL is required")
    database = PlatformDatabase(args.database_url)
    if args.check:
        database.assert_migrated()
        result = {"schema": "platform.upgrade/v1", "check": True}
    else:
        database.migrate()
        result = quarantine_legacy_research_jobs(database.engine)
    result.pop("quarantined_jobs", None)
    print(json.dumps(result, sort_keys=True))
    database.dispose()


if __name__ == "__main__":
    main()
