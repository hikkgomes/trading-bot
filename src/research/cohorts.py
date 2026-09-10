"""Freeze all finalists before protected access and seal their joint statistic."""

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from sqlalchemy import exists, insert, select, text

from src.data.database import (
    dataset_snapshot,
    evidence_invalidation,
    experiment,
    holdout_claim,
    holdout_outcome,
    selection_cohort,
    selection_cohort_open,
    selection_cohort_result,
    validation_stage,
)
from src.domain._codec import canonical_hash
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.executors import DataUnavailableError
from src.research.outcomes import terminal_trial_disposition
from src.research.protected_exposure import legacy_exposure_conflicts
from src.research.rounds import assert_epoch_closed
from src.research.selection import FrozenCohort, selection_adjusted_block_bootstrap


class CohortPending(RuntimeError):
    """Other trials or the durable freeze boundary have not completed."""


def selection_stream(evidence: Mapping[str, Any]) -> dict[str, Any]:
    account = evidence.get("product_accounting")
    if not isinstance(account, Mapping) or not account.get("nav_series"):
        raise DataUnavailableError("cohort inference requires the canonical account path")
    marks = account["nav_series"]
    returns = list(evidence["return_ledger"]["net_returns"])
    if len(returns) != len(marks):
        raise DataUnavailableError("cohort returns and account marks are not aligned")
    benchmark = []
    previous = float(account["initial_value"])
    for mark in marks:
        if account["objective_unit"] == "BTC":
            current = float(mark["benchmark_nav"])
            benchmark.append((current - float(mark.get("external_flow", 0))) / previous - 1)
            previous = current
        else:
            benchmark.append(0.0)
    return {
        "timestamps": [mark["timestamp"] for mark in marks],
        "net_returns": returns,
        "benchmark_returns": benchmark,
        "sample_unit": "calendar_portfolio_return",
    }


def prepare_cohort(
    engine, candidate, *, now: str, policy_hash: str, configuration_hash: str | None = None
) -> dict[str, Any]:
    if candidate.dataset_plan is None:
        raise DataUnavailableError("protected evaluation requires a complete frozen dataset plan")
    snapshot_id = candidate.dataset_plan.protected_holdout_snapshot_id
    scope = canonical_hash(
        {"product_id": candidate.definition.product, "protected_snapshot": snapshot_id}
    )
    with engine.begin() as connection:
        _lock(connection, candidate.definition.product)
        existing = connection.execute(
            select(selection_cohort.c.payload).where(selection_cohort.c.scope_id == scope)
        ).scalar_one_or_none()
        if existing is None:
            existing = _freeze(connection, candidate, snapshot_id, scope, now, policy_hash)
            connection.execute(
                insert(selection_cohort).values(
                    id=existing["cohort_id"], scope_id=scope, created_at=now, payload=existing
                )
            )
        _open_cohort(connection, existing, candidate, now, policy_hash, configuration_hash)
    if now <= existing["frozen_at"]:
        raise CohortPending("cohort is frozen; protected access starts on the next clock tick")
    return dict(existing)


def _open_cohort(connection, cohort, candidate, now, policy_hash, configuration_hash):
    from src.research.policy_approval import current_approval

    if cohort["engine_version"] != ECONOMIC_ENGINE_VERSION or cohort["policy_hash"] != policy_hash:
        raise DataUnavailableError(
            "protected cohort belongs to a different engine or selection policy"
        )
    if candidate.candidate_id not in cohort["members"]:
        raise DataUnavailableError("candidate is outside the already frozen protected cohort")
    if now <= cohort["frozen_at"]:
        return
    if connection.execute(
        select(selection_cohort_open.c.id).where(selection_cohort_open.c.id == cohort["cohort_id"])
    ).first():
        return
    _assert_unopened(connection, cohort["protected_snapshot_id"], cohort, cohort["product_id"])
    approval = (
        current_approval(
            connection, policy_hash=policy_hash, configuration_hash=configuration_hash, at=now
        )
        if configuration_hash
        else None
    )
    connection.execute(
        insert(selection_cohort_open).values(
            id=cohort["cohort_id"],
            created_at=now,
            payload={
                "cohort_id": cohort["cohort_id"],
                "opened_at": now,
                "protected_snapshot_id": cohort["protected_snapshot_id"],
                "research_configuration_hash": configuration_hash,
                "research_policy_approval_id": approval["approval_id"] if approval else None,
            },
        )
    )


def _lock(connection, scope: str) -> None:
    if connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
            {"scope": f"selection-cohort:{scope}"},
        )


def _freeze(connection, candidate, snapshot_id, scope, now, policy_hash):
    assert_epoch_closed(connection, candidate.dataset_bundle_id)
    metadata = (
        connection.execute(
            select(
                dataset_snapshot.c.payload["interval"].label("interval"),
                dataset_snapshot.c.payload["instrument_scope"].label("instruments"),
            ).where(dataset_snapshot.c.id == snapshot_id)
        )
        .mappings()
        .one()
    )
    _assert_unopened(connection, snapshot_id, metadata, candidate.definition.product)
    trials = (
        connection.execute(
            select(experiment)
            .where(
                experiment.c.metadata["_dataset_plan"]["protected_holdout_snapshot_id"].as_string()
                == snapshot_id,
            )
            .order_by(experiment.c.id)
        )
        .mappings()
        .all()
    )
    stage_rows = (
        connection.execute(
            select(validation_stage).where(
                validation_stage.c.experiment_id.in_([row["id"] for row in trials]),
                validation_stage.c.engine_version == ECONOMIC_ENGINE_VERSION,
            )
        )
        .mappings()
        .all()
    )
    invalid = set(connection.execute(select(evidence_invalidation.c.id)).scalars())
    stages: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in stage_rows:
        if f"stage:{row['id']}" not in invalid:
            stages[row["experiment_id"]][row["stage"]] = row
    members, exclusions, streams = _finalists(trials, stages)
    timestamps, development, _benchmark = _aligned_streams(members, streams)
    frozen = FrozenCohort(
        tuple(members), tuple(metadata["instruments"]), now, canonical_hash(development.tolist())
    )
    return {
        "schema": "research.selection_cohort/v1",
        "cohort_id": frozen.manifest_id,
        "scope_id": scope,
        "product_id": candidate.definition.product,
        "protected_snapshot_id": snapshot_id,
        "interval": metadata["interval"],
        "members": members,
        "all_trials": [row["id"] for row in trials],
        "excluded_trials": exclusions,
        "instruments": list(frozen.instruments),
        "frozen_at": now,
        "development_hash": frozen.development_hash,
        "development_timestamps": timestamps,
        "development_returns": development.tolist(),
        "engine_version": ECONOMIC_ENGINE_VERSION,
        "policy_hash": policy_hash,
        "selection_rule": "all current-engine robustness finalists; maximum studentised mean",
    }


def _finalists(trials, stages):
    members, excluded, streams = [], {}, {}
    for trial in trials:
        identity = trial["id"]
        current = stages.get(identity, {})
        disposition = _disposition(trial, current)
        if disposition == "pending":
            raise CohortPending(f"cohort waits for trial {identity}")
        if disposition != "finalist":
            excluded[identity] = disposition
            continue
        members.append(identity)
        streams[identity] = selection_stream(current["development"]["payload"]["evidence"])
    if not members:
        raise DataUnavailableError("protected cohort has no measured finalists")
    return members, excluded, streams


def _disposition(trial, stages) -> str:
    if stages.get("robustness", {}).get("accepted") and stages.get("development", {}).get(
        "accepted"
    ):
        return "finalist"
    if any(not stage["accepted"] for stage in stages.values()):
        return "rejected"
    terminal = terminal_trial_disposition(str(trial["state"]), trial["metadata"])
    if terminal is not None:
        return terminal.value
    return "pending"


def _assert_unopened(connection, snapshot_id, metadata, product_id):
    if legacy_exposure_conflicts(
        connection,
        {
            "interval": metadata["interval"],
            "instrument_scope": metadata["instruments"],
            "product_id": product_id,
        },
    ):
        raise DataUnavailableError("protected interval overlaps legacy exposure")
    prior = connection.execute(
        select(holdout_claim.c.id)
        .where(
            holdout_claim.c.payload["data_snapshot_id"].as_string() == snapshot_id,
        )
        .limit(1)
    ).first()
    if prior is not None:
        raise DataUnavailableError("protected data was opened before this cohort was frozen")
    cohorts = connection.execute(
        select(selection_cohort.c.payload).where(
            exists(
                select(selection_cohort_open.c.id).where(
                    selection_cohort_open.c.id == selection_cohort.c.id
                )
            )
        )
    ).scalars()
    for cohort in cohorts:
        if cohort["product_id"] != product_id or not set(cohort["instruments"]) & set(
            metadata["instruments"]
        ):
            continue
        start, end = (
            dt.datetime.fromisoformat(metadata["interval"][key]) for key in ("start", "end")
        )
        prior_start, prior_end = (
            dt.datetime.fromisoformat(cohort["interval"][key]) for key in ("start", "end")
        )
        if start < prior_end and prior_start < end:
            raise DataUnavailableError("protected interval overlaps an already opened cohort")


def _aligned_streams(members, streams):
    reference = streams[members[0]]
    for identity in members:
        stream = streams[identity]
        if (
            stream["timestamps"] != reference["timestamps"]
            or stream["benchmark_returns"] != reference["benchmark_returns"]
        ):
            raise DataUnavailableError("cohort streams must share a calendar and benchmark")
    values = np.asarray([streams[identity]["net_returns"] for identity in members], dtype=float).T
    return reference["timestamps"], values, reference["benchmark_returns"]


def complete_cohort(engine, cohort: Mapping[str, Any], *, now: str) -> None:
    with engine.connect() as connection:
        if connection.execute(
            select(selection_cohort_result.c.id).where(
                selection_cohort_result.c.id == cohort["cohort_id"]
            )
        ).first():
            return
        rows = (
            connection.execute(
                select(holdout_claim.c.created_at, holdout_outcome.c.payload)
                .join(
                    holdout_outcome,
                    holdout_outcome.c.holdout_claim_id == holdout_claim.c.id,
                )
                .where(holdout_claim.c.payload["cohort_id"].as_string() == cohort["cohort_id"])
            )
            .mappings()
            .all()
        )
    streams = {
        row["payload"]["execution_receipt"]["candidate_id"]: row["payload"]["selection_stream"]
        for row in rows
    }
    if len(streams) != len(rows) or not set(streams).issubset(cohort["members"]):
        raise DataUnavailableError("sealed outcomes do not match the frozen cohort membership")
    if any(
        row["payload"]["execution_receipt"]["dataset_snapshot_ids"]
        != [cohort["protected_snapshot_id"]]
        or row["payload"]["execution_receipt"]["executor_version"] != cohort["engine_version"]
        for row in rows
    ):
        raise DataUnavailableError("sealed cohort outcome has different data or engine bindings")
    if set(streams) != set(cohort["members"]):
        return
    times, returns, benchmark = _aligned_streams(cohort["members"], streams)
    frozen = FrozenCohort(
        tuple(cohort["members"]),
        tuple(cohort["instruments"]),
        cohort["frozen_at"],
        cohort["development_hash"],
    )
    result = selection_adjusted_block_bootstrap(
        frozen,
        timestamps=times,
        net_returns=returns,
        benchmark_returns=benchmark,
        development_returns=np.asarray(cohort["development_returns"]),
        opened_at=min(row["created_at"] for row in rows),
        seed=int(cohort["cohort_id"][7:15], 16),
    )
    with engine.begin() as connection:
        _lock(connection, cohort["scope_id"])
        if connection.execute(
            select(selection_cohort_result.c.id).where(
                selection_cohort_result.c.id == cohort["cohort_id"]
            )
        ).first():
            return
        connection.execute(
            insert(selection_cohort_result).values(
                id=cohort["cohort_id"], created_at=now, payload=result
            )
        )
