"""Reconcile execution attempts with immutable measured research results."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.domain._codec import canonical_hash
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.outcomes import OutcomeCategory


def current_stages(stages, invalidations):
    excluded = {
        row["payload"]["source_id"]
        for row in invalidations
        if row["payload"].get("kind") == "validation_stage"
    }
    return [
        row
        for row in stages
        if row.get("engine_version") == ECONOMIC_ENGINE_VERSION and row["id"] not in excluded
    ]


def evaluation_identity(payload: Mapping[str, Any]) -> str | None:
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    required = ("candidate_id", "dataset_snapshot_ids", "executor_version", "input_hash")
    if not all(receipt.get(key) for key in required):
        return None
    return canonical_hash(
        {
            "stage": payload.get("run_name"),
            "candidate_id": receipt["candidate_id"],
            "datasets": sorted(receipt["dataset_snapshot_ids"]),
            "engine_version": receipt["executor_version"],
        }
    )


def measured_progress(
    runs: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identities = [evaluation_identity(row["payload"]) for row in runs]
    measured = [identity for identity in identities if identity is not None]
    current = [
        identity
        for row, identity in zip(runs, identities, strict=True)
        if identity is not None
        and row["payload"]["receipt"]["executor_version"] == ECONOMIC_ENGINE_VERSION
    ]
    deferrals = Counter(
        str(row.get("error") or "unspecified")
        for row in attempts
        if row.get("status") == "deferred"
        and _attempt_outcome(row) == OutcomeCategory.BLOCKED_INFRASTRUCTURE.value
    )
    return {
        "economic_engine_version": ECONOMIC_ENGINE_VERSION,
        "unique_measured_evaluations": len(set(current)),
        "duplicate_measured_evaluations": len(current) - len(set(current)),
        "historical_measured_evaluations": len(set(measured) - set(current)),
        "runs_without_execution_receipts": len(identities) - len(measured),
        "attempt_count": len(attempts),
        "resource_deferrals": dict(deferrals),
        "attempt_outcomes": dict(Counter(_attempt_outcome(row) for row in attempts)),
    }


def _attempt_outcome(row: Mapping[str, Any]) -> str:
    payload = row.get("payload") or {}
    if isinstance(payload, Mapping) and payload.get("outcome_category"):
        return OutcomeCategory(payload["outcome_category"]).value
    if row.get("status") in {"failed", "expired"}:
        return OutcomeCategory.BLOCKED_INFRASTRUCTURE.value
    # A completed transport job is not an accepted research evaluation.
    return "unclassified"


def eligible_work(jobs: Sequence[Mapping[str, Any]], *, now: str) -> dict[str, Any]:
    pending = [row for row in jobs if row["state"] == "pending" and row["available_at"] <= now]
    oldest = min(pending, key=lambda row: row["available_at"], default=None)
    return {
        "eligible_work_count": len(pending),
        "oldest_eligible_work": (
            {"job_id": oldest["id"], "available_at": oldest["available_at"]}
            if oldest is not None
            else None
        ),
    }
