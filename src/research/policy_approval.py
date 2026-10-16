"""Operator approval of reviewed calibration evidence, separate from live capital."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from sqlalchemy import exists, insert, or_, select

from src.data.database import PlatformDatabase, research_calibration, research_policy_approval
from src.domain._codec import canonical_hash, non_empty, timestamp
from src.research.calibration import CASES, GENERATOR_VERSION, _case_summary
from src.research.policy_identity import research_policy_engine_digest
from src.research.selection import METHOD


def research_qualification(
    connection, *, artefact_hash, strategy_version_id, product_id, configuration_hash, at
):
    from src.data.database import (
        dataset_snapshot,
        experiment,
        selection_cohort,
        selection_cohort_open,
        selection_cohort_result,
    )
    from src.research.canonical import _assert_artefact_binding, _assert_canonical_artifact
    from src.research.protected_exposure import legacy_exposure_conflicts

    if configuration_hash is None:
        return _unqualified("research_policy_context_unavailable")
    artefact = _assert_canonical_artifact(connection, artefact_hash)
    _assert_artefact_binding(
        artefact, strategy_version_id=strategy_version_id, product_id=product_id
    )
    metadata = artefact.get("metadata", {})
    candidate_id, cohort_id = metadata.get("candidate_id"), metadata.get("selection_cohort_id")
    if not candidate_id or not cohort_id:
        return _unqualified("research_cohort_unavailable")
    version = connection.execute(
        select(experiment.c.strategy_version_id).where(experiment.c.id == candidate_id)
    ).scalar_one_or_none()
    if version != strategy_version_id:
        return _unqualified("research_candidate_binding_invalid")
    records = [
        connection.execute(
            select(table.c.payload).where(table.c.id == cohort_id)
        ).scalar_one_or_none()
        for table in (selection_cohort, selection_cohort_open, selection_cohort_result)
    ]
    cohort, opening, result = records
    if cohort is None or opening is None or result is None:
        return _unqualified("research_cohort_unavailable")
    if cohort["product_id"] != product_id or candidate_id not in cohort["members"]:
        return _unqualified("research_candidate_binding_invalid")
    snapshot = connection.execute(
        select(dataset_snapshot.c.payload).where(
            dataset_snapshot.c.id == cohort["protected_snapshot_id"]
        )
    ).scalar_one_or_none()
    if snapshot is None:
        return _unqualified("research_cohort_unavailable")
    if legacy_exposure_conflicts(connection, snapshot):
        return _unqualified("research_interval_exposed")
    approval_id = opening.get("research_policy_approval_id")
    if not approval_id or opening.get("research_configuration_hash") != configuration_hash:
        return _unqualified("research_policy_unapproved")
    approval = current_approval(
        connection,
        policy_hash=cohort["policy_hash"],
        configuration_hash=configuration_hash,
        at=at,
        approval_id=approval_id,
    )
    if approval is None or approval["approved_at"] > opening["opened_at"]:
        return _unqualified("research_policy_unapproved")
    return {
        **_qualified_result(cohort, result, candidate_id, approval),
        "configuration_hash": configuration_hash,
    }


def _unqualified(reason):
    return {"qualified": False, "reason_code": reason, "method": METHOD}


def _qualified_result(cohort, result, candidate_id, approval):
    from src.research.selection import FrozenCohort

    frozen = FrozenCohort(
        tuple(cohort["members"]),
        tuple(cohort["instruments"]),
        cohort["frozen_at"],
        cohort["development_hash"],
    )
    draws = result.get("draws")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2000:
        return _unqualified("research_inference_invalid")
    if (
        result.get("cohort_id") != frozen.manifest_id
        or result.get("members") != list(frozen.members)
        or result.get("instruments") != list(frozen.instruments)
        or result.get("method") != METHOD
    ):
        return _unqualified("research_inference_invalid")
    if result.get("status") != "measured":
        return _unqualified("research_inference_inconclusive")
    pvalues = result.get("adjusted_pvalues", {})
    if set(pvalues) != set(frozen.members) or any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 1 / (draws + 1) <= value <= 1
        for value in pvalues.values()
    ):
        return _unqualified("research_inference_invalid")
    qualified = pvalues[candidate_id] <= approval["nominal_test_alpha"]
    return {
        "qualified": qualified,
        "reason_code": None if qualified else "research_selection_threshold",
        "method": METHOD,
        "cohort_id": cohort["cohort_id"],
        "candidate_id": candidate_id,
        "approval_id": approval["approval_id"],
        "calibration_id": approval["calibration_id"],
        "inference_hash": canonical_hash(result),
    }


def calibration_source_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("selection.py", "calibration.py")
    }


def validate_calibration(receipt, configuration):
    expected = {
        "schema": "research.selection_calibration/v1",
        "method": METHOD,
        "generator_version": GENERATOR_VERSION,
        "policy_hash": canonical_hash(configuration),
        "source_hashes": calibration_source_hashes(),
        "qualification_authority": False,
        "policy_approval_required": True,
        "calibration_scope": "selection_inference_kernel",
        "candidates": int(configuration["search_budgets"]["maximum_candidates_per_epoch"]),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("calibration source, method or configuration binding is invalid")
    alpha = float(receipt["nominal_test_alpha"])
    budget = float(receipt["proposed_false_qualification_budget"])
    configured_budget = float(
        configuration.get("selection_inference", {}).get("false_qualification_budget", 0.1)
    )
    if not 0 < alpha < 1 or not 0 < budget < 1 or budget != configured_budget:
        raise ValueError("calibration cutoff or configured risk budget is invalid")
    summary = _validated_trials(receipt, alpha)
    if any(
        summary[case]["qualification_rate_interval_95"][1] > budget
        for case in ("null", "cost_trap", "regime_limited")
    ):
        raise ValueError(
            "calibration negative-control upper confidence bound exceeds the configured budget"
        )
    if any(summary[case]["qualification_rate"] != 0 for case in ("leakage_trap", "flat")):
        raise ValueError("calibration admitted a negative control")
    if summary["flat"]["deferral_rate"] != 1:
        raise ValueError("flat calibration samples must remain inconclusive")
    return {"nominal_test_alpha": alpha, "false_qualification_budget": budget, "summary": summary}


def _validated_trials(receipt, alpha):
    replicates = receipt["replicates"]
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 1 <= replicates <= 500
    ):
        raise ValueError("calibration replicate count is invalid")
    trials = receipt["trials"]
    observations = receipt["observations"]
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 128:
        raise ValueError("calibration observation count is invalid")
    if len(trials) != len(CASES) * replicates:
        raise ValueError("calibration trial inventory is incomplete")
    summary = {}
    for index, case in enumerate(CASES):
        selected = [trial for trial in trials if trial["case"] == case]
        expected_seeds = set(
            range(receipt["seed"] + index * 10000, receipt["seed"] + index * 10000 + replicates)
        )
        if len(selected) != replicates or {trial["seed"] for trial in selected} != expected_seeds:
            raise ValueError("calibration trial seeds are incomplete or duplicated")
        for trial in selected:
            _validate_trial(trial, alpha, observations, receipt["candidates"])
        summary[case] = _case_summary(selected)
    if summary != receipt["summary"]:
        raise ValueError("calibration summary does not match its trials")
    return summary


def _validate_trial(trial, alpha, observations, candidates):
    inference = trial["inference"]
    _validate_trial_design(inference, observations, candidates)
    if (
        inference.get("method") != METHOD
        or inference.get("draws", 0) < 2000
        or inference.get("seed") != trial["seed"]
    ):
        raise ValueError("calibration trial inference binding is invalid")
    if (
        inference.get("status") not in {"measured", "inconclusive"}
        or type(trial["causal"]) is not bool
    ):
        raise ValueError("calibration trial outcome is invalid")
    pvalues = tuple(inference.get("adjusted_pvalues", {}).values())
    if inference["status"] == "measured" and (
        not pvalues or any(not math.isfinite(value) or not 0 <= value <= 1 for value in pvalues)
    ):
        raise ValueError("calibration trial adjusted p-values are invalid")
    if any(
        not math.isclose(
            value * (inference["draws"] + 1), round(value * (inference["draws"] + 1)), abs_tol=1e-9
        )
        or value < 1 / (inference["draws"] + 1)
        for value in pvalues
    ):
        raise ValueError("calibration p-values do not match the seeded bootstrap resolution")
    qualified = trial["causal"] and inference["status"] == "measured" and min(pvalues) <= alpha
    if trial["qualified_in_simulation"] is not qualified:
        raise ValueError("calibration trial qualification does not match its inference")


def _validate_trial_design(inference, observations, candidates):
    members = [f"configuration-{index}" for index in range(candidates)]
    times = inference.get("timestamps", ())
    if inference.get("members") != members or inference.get("sample_unit") != "calendar_interval":
        raise ValueError("calibration must cover the declared configuration cohort and calendar")
    if (
        len(times) != observations
        or len(set(times)) != observations
        or list(times) != sorted(times)
    ):
        raise ValueError("calibration calendar is incomplete or unordered")
    if inference.get("qualification_authority") is not False:
        raise ValueError("calibration inference cannot grant itself authority")
    if inference.get("status") == "measured" and set(inference.get("adjusted_pvalues", {})) != set(
        members
    ):
        raise ValueError("calibration p-values omit a cohort member")


def _append(connection, table, identity, at, payload):
    existing = connection.execute(
        select(table.c.payload).where(table.c.id == identity)
    ).scalar_one_or_none()
    if existing is not None:
        if existing != payload:
            raise ValueError("immutable research policy record conflicts with existing content")
        return
    connection.execute(insert(table).values(id=identity, created_at=at, payload=payload))


def approve_policy(
    engine,
    *,
    receipt_path: Path,
    expected_hash: str,
    configuration,
    approved_by: str,
    approved_at: str,
):
    from src.research.evaluation import EvidencePolicy

    raw = receipt_path.read_bytes()
    receipt_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    if expected_hash != receipt_id:
        raise ValueError("calibration file differs from the operator-reviewed hash")
    receipt = json.loads(raw)
    validated = validate_calibration(receipt, configuration)
    at = timestamp(approved_at, field="approved_at")
    payload = {
        "schema": "research.policy_approval/v1",
        "status": "approved",
        "approved_by": non_empty(approved_by, field="approved_by"),
        "approved_at": at,
        "method": METHOD,
        "calibration_id": receipt_id,
        "calibration_content_hash": canonical_hash(receipt),
        "configuration_hash": canonical_hash(configuration),
        "policy_hash": EvidencePolicy.from_configuration(
            configuration["evidence_policy"]
        ).policy_hash,
        "engine_hash": research_policy_engine_digest(),
        **validated,
    }
    identity = canonical_hash(payload)
    with engine.begin() as connection:
        _append(connection, research_calibration, receipt_id, at, receipt)
        _append(connection, research_policy_approval, identity, at, payload)
    return identity


def revoke_policy(engine, *, approval_id: str, revoked_by: str, revoked_at: str):
    at = timestamp(revoked_at, field="revoked_at")
    with engine.begin() as connection:
        original = connection.execute(
            select(research_policy_approval.c.payload).where(
                research_policy_approval.c.id == approval_id
            )
        ).scalar_one_or_none()
        if original is None or original.get("status") != "approved" or at < original["approved_at"]:
            raise ValueError("research policy approval is missing or revocation predates approval")
        payload = {
            "schema": "research.policy_approval/v1",
            "status": "revoked",
            "approval_id": approval_id,
            "policy_hash": original["policy_hash"],
            "configuration_hash": original["configuration_hash"],
            "revoked_by": non_empty(revoked_by, field="revoked_by"),
            "revoked_at": at,
        }
        identity = canonical_hash(payload)
        _append(connection, research_policy_approval, identity, at, payload)
    return identity


def current_approval(
    connection,
    *,
    policy_hash: str,
    configuration_hash: str,
    at: str,
    approval_id: str | None = None,
):
    at = timestamp(at, field="policy decision time")
    rows = tuple(
        connection.execute(
            select(research_policy_approval.c.id, research_policy_approval.c.payload)
            .where(
                research_policy_approval.c.created_at <= at,
                research_policy_approval.c.payload["policy_hash"].as_string() == policy_hash,
                research_policy_approval.c.payload["configuration_hash"].as_string()
                == configuration_hash,
                or_(
                    research_policy_approval.c.payload["status"].as_string() == "revoked",
                    exists(
                        select(research_calibration.c.id).where(
                            research_calibration.c.id
                            == research_policy_approval.c.payload["calibration_id"].as_string()
                        )
                    ),
                ),
            )
            .order_by(
                research_policy_approval.c.created_at.desc(), research_policy_approval.c.id.desc()
            )
        ).mappings()
    )
    revoked = {
        row["payload"]["approval_id"] for row in rows if row["payload"].get("status") == "revoked"
    }
    if not rows:
        return None
    engine_hash = research_policy_engine_digest()
    return next(
        (
            {"approval_id": row["id"], **row["payload"]}
            for row in rows
            if _approval_matches(row, engine_hash, at)
            and row["id"] not in revoked
            and (approval_id is None or row["id"] == approval_id)
        ),
        None,
    )


def _approval_matches(row, engine_hash, at):
    payload = row["payload"]
    alpha = payload.get("nominal_test_alpha")
    return (
        payload.get("status") == "approved"
        and payload.get("schema") == "research.policy_approval/v1"
        and row["id"] == canonical_hash(payload)
        and payload.get("method") == METHOD
        and payload.get("engine_hash") == engine_hash
        and payload.get("approved_at", at) <= at
        and isinstance(alpha, int | float)
        and not isinstance(alpha, bool)
        and math.isfinite(alpha)
        and 0 < alpha < 1
    )


def main():
    parser = argparse.ArgumentParser(
        description="Approve or revoke reviewed research policy; grants no live capital"
    )
    parser.add_argument("--database-url", default=os.environ.get("TRADING_OPERATOR_DATABASE_URL"))
    commands = parser.add_subparsers(dest="action", required=True)
    approve = commands.add_parser("approve")
    approve.add_argument("--receipt", type=Path, required=True)
    approve.add_argument("--receipt-hash", required=True)
    approve.add_argument("--configuration", type=Path, default=Path("config/research.json"))
    revoke = commands.add_parser("revoke")
    revoke.add_argument("--approval-id", required=True)
    for command in (approve, revoke):
        command.add_argument("--operator", required=True)
        command.add_argument("--at", required=True)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("an operator database URL is required")
    database = PlatformDatabase(args.database_url)
    if not database.is_postgresql:
        parser.error("operator policy commands require PostgreSQL")
    if args.action == "approve":
        identity = approve_policy(
            database.engine,
            receipt_path=args.receipt,
            expected_hash=args.receipt_hash,
            configuration=json.loads(args.configuration.read_text()),
            approved_by=args.operator,
            approved_at=args.at,
        )
    else:
        identity = revoke_policy(
            database.engine,
            approval_id=args.approval_id,
            revoked_by=args.operator,
            revoked_at=args.at,
        )
    print(
        json.dumps({"action": args.action, "record_id": identity, "live_capital_authority": False})
    )


if __name__ == "__main__":
    main()
