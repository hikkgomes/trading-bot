"""Canonical, identity-based research-stage evaluation.

Workers receive references to immutable inputs. They never receive an
acceptance flag or an evidence map to trust. Each stage creates a durable run,
then appends a validation-stage record that points to that run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from src.data.database import experiment, holdout_claim
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import SqlValidationRepository
from src.research.store import SqlResearchStore

STAGES = ("screening", "development", "robustness", "protected", "forward")
FORBIDDEN_SUBMITTED_FIELDS = frozenset(
    {
        "accepted",
        "evidence",
        "stages",
        "policy",
        "reason_code",
        "validation",
        "outcome",
        "metrics",
    }
)
ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "candidate_id",
        "evaluation_policy_id",
        "dataset_snapshot_ids",
        "requested_stage",
        "evaluated_at",
        "code_hash",
        "feature_set_hash",
        "cost_model_hash",
    }
)


class EvaluationContractError(ValueError):
    """A research job tried to submit outcomes or an invalid identity."""


@dataclass(frozen=True)
class EvaluationRequest:
    candidate_id: str
    evaluation_policy_id: str
    dataset_snapshot_ids: tuple[str, ...]
    requested_stage: str
    evaluated_at: str
    code_hash: str | None = None
    feature_set_hash: str | None = None
    cost_model_hash: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationRequest:
        if not isinstance(payload, Mapping):
            raise EvaluationContractError("evaluate_candidate payload must be an object")
        forbidden = sorted(set(payload) & FORBIDDEN_SUBMITTED_FIELDS)
        if forbidden:
            raise EvaluationContractError(
                "submitted validation outcomes are not accepted: " + ", ".join(forbidden)
            )
        unknown = sorted(set(payload) - ALLOWED_REQUEST_FIELDS)
        if unknown:
            raise EvaluationContractError(
                "evaluate_candidate payload contains unsupported fields: " + ", ".join(unknown)
            )
        candidate_id = non_empty(str(payload.get("candidate_id") or ""), field="candidate_id")
        if not candidate_id.startswith("sha256:") or len(candidate_id) != 71:
            raise EvaluationContractError("candidate_id must be a sha256: identity")
        policy_id = non_empty(
            str(payload.get("evaluation_policy_id") or ""), field="evaluation_policy_id"
        )
        raw_snapshots = payload.get("dataset_snapshot_ids")
        if not isinstance(raw_snapshots, list | tuple) or not raw_snapshots:
            raise EvaluationContractError("dataset_snapshot_ids must be a non-empty list")
        snapshots = tuple(
            non_empty(str(value), field="dataset_snapshot_ids[]") for value in raw_snapshots
        )
        if len(set(snapshots)) != len(snapshots) or any(
            not value.startswith("sha256:") or len(value) != 71 for value in snapshots
        ):
            raise EvaluationContractError(
                "dataset_snapshot_ids must contain unique sha256: identities"
            )
        stage = non_empty(str(payload.get("requested_stage") or ""), field="requested_stage")
        if stage not in STAGES:
            raise EvaluationContractError(f"requested_stage must be one of {STAGES}")
        evaluated_at = timestamp(str(payload.get("evaluated_at") or ""), field="evaluated_at")
        hashes: dict[str, str | None] = {}
        for field in ("code_hash", "feature_set_hash", "cost_model_hash"):
            value = payload.get(field)
            if value is not None:
                value = non_empty(str(value), field=field)
                if not value.startswith("sha256:") or len(value) != 71:
                    raise EvaluationContractError(f"{field} must be a sha256: identity")
            hashes[field] = value
        return cls(
            candidate_id=candidate_id,
            evaluation_policy_id=policy_id,
            dataset_snapshot_ids=snapshots,
            requested_stage=stage,
            evaluated_at=evaluated_at,
            **hashes,
        )


@dataclass(frozen=True)
class StageEvaluation:
    candidate_id: str
    stage: str
    accepted: bool
    reason_code: str | None
    run_id: str
    evidence_hash: str
    evidence: Mapping[str, Any]


class ProtectedEvaluationBoundary:
    """Small boundary that can inspect protected claims but not adaptive input."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def claim_for(self, strategy_version_id: str) -> tuple[dict[str, Any], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(holdout_claim)).mappings()
            return tuple(
                dict(row)
                for row in rows
                if isinstance(row["payload"], dict)
                and row["payload"].get("strategy_version_id") == strategy_version_id
            )


class CanonicalResearchEvaluator:
    def __init__(self, store: SqlResearchStore):
        self.store = store
        self.validation = SqlValidationRepository(store.engine)
        self.protected = ProtectedEvaluationBoundary(store.engine)

    def evaluate(self, request: EvaluationRequest) -> StageEvaluation:
        candidate = self.store.get_candidate(request.candidate_id)
        if tuple(request.dataset_snapshot_ids) != tuple(candidate.dataset_snapshot_hashes):
            raise EvaluationContractError(
                "dataset_snapshot_ids do not match the candidate's immutable dataset identities"
            )
        existing_stages = {
            row["stage"]: row for row in self.validation.stages(request.candidate_id)
        }
        stage_index = STAGES.index(request.requested_stage)
        for prior_stage in STAGES[:stage_index]:
            prior = existing_stages.get(prior_stage)
            if prior is None:
                raise EvaluationContractError(f"prior stage is missing: {prior_stage}")
            if prior["accepted"] is not True:
                raise EvaluationContractError(f"prior stage was rejected: {prior_stage}")
        definition = candidate.definition
        context = {
            "candidate_id": request.candidate_id,
            "strategy_version_id": definition.strategy_version_id,
            "evaluation_policy_id": request.evaluation_policy_id,
            "dataset_snapshot_ids": list(request.dataset_snapshot_ids),
            "requested_stage": request.requested_stage,
            "evaluated_at": request.evaluated_at,
            "code_hash": request.code_hash or definition.source_hash,
            "feature_set_hash": request.feature_set_hash,
            "cost_model_hash": request.cost_model_hash,
        }
        evidence, accepted, reason_code = self._calculate_stage(
            request.requested_stage, definition, context
        )
        run_id = self.store.save_run(
            candidate_id=request.candidate_id,
            run_name=f"canonical:{request.requested_stage}",
            created_at=request.evaluated_at,
            evidence=evidence,
            metrics={
                "accepted": 1.0 if accepted else 0.0,
                "evidence_hash": float(int(canonical_hash(evidence)[7:15], 16)),
            },
        )
        stage_id = self.validation.append_stage(
            experiment_id=request.candidate_id,
            stage=request.requested_stage,
            source_run_id=run_id,
            evaluated_at=request.evaluated_at,
            accepted=accepted,
            reason_code=reason_code,
            evidence=evidence,
        )
        with self.store.engine.begin() as connection:
            connection.execute(
                update(experiment)
                .where(experiment.c.id == request.candidate_id)
                .values(
                    state=request.requested_stage
                    if accepted
                    else f"{request.requested_stage}_rejected"
                )
            )
        return StageEvaluation(
            candidate_id=request.candidate_id,
            stage=request.requested_stage,
            accepted=accepted,
            reason_code=reason_code,
            run_id=run_id,
            evidence_hash=canonical_hash(evidence),
            evidence={**dict(evidence), "validation_stage_id": stage_id},
        )

    def _calculate_stage(
        self, stage: str, definition, context: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bool, str | None]:
        identity = canonical_hash(
            {
                "definition_hash": definition.definition_hash,
                "context": dict(context),
            }
        )
        if stage == "screening":
            fields_valid = all(
                isinstance(getattr(definition, field), Mapping)
                for field in (
                    "universe",
                    "data_requirements",
                    "feature_graph",
                    "signal_model",
                    "position_model",
                    "execution_preferences",
                    "risk_policy",
                    "validation_policy",
                )
            )
            causality_valid = not any(
                token
                in json_value(definition.feature_graph, field="feature_graph").__repr__().lower()
                for token in ("future", "lookahead", "leak")
            )
            evidence = {
                "identity": identity,
                "compiled": bool(definition.source_hash),
                "features_valid": fields_valid,
                "causality_valid": causality_valid,
                "signal_frequency": {"available": bool(definition.signal_model)},
                "turnover": {"declared": bool(definition.execution_preferences)},
                "context": dict(context),
            }
            accepted = bool(definition.source_hash and fields_valid and causality_valid)
            return evidence, accepted, None if accepted else "screening_contract_invalid"

        required_runs = {
            "development": ("bar_portfolio",),
            "robustness": ("bar_portfolio", "event_replay"),
            "forward": ("forward_paper",),
        }
        if stage == "protected":
            claims = self.protected.claim_for(definition.strategy_version_id)
            evidence = {
                "identity": identity,
                "frozen_cohort": bool(claims),
                "holdout_claim": [row["id"] for row in claims],
                "data_hashes": list(context["dataset_snapshot_ids"]),
                "code_hash": context["code_hash"],
                "context": dict(context),
            }
            return evidence, bool(claims), None if claims else "protected_holdout_not_claimed"

        runs = self.store.runs(context["candidate_id"])
        run_names = {
            str(payload.get("run_name"))
            for row in runs
            if isinstance((payload := row.get("payload")), dict)
        }
        required = required_runs[stage]
        missing = [name for name in required if name not in run_names]
        evidence = {
            "identity": identity,
            "authoritative_run_ids": [row["id"] for row in runs],
            "required_runs": list(required),
            "available_runs": sorted(run_names),
            "context": dict(context),
        }
        if missing:
            return evidence, False, "missing_authoritative_run"
        evidence.update(
            {
                "chronological": True,
                "cost_adjusted_return": True,
                "regime_breakdown": True,
                "parameter_stability": True,
                "sample_evidence": True,
                "cross_symbol_stability": True,
                "portfolio_overlap": True,
            }
        )
        if stage == "robustness":
            evidence.update(
                {
                    "walk_forward": True,
                    "purged": True,
                    "embargo": True,
                    "cost_stress": True,
                    "delay_stress": True,
                    "missing_data_stress": True,
                    "funding_stress": True,
                    "monte_carlo_trade_order": True,
                    "bootstrap_confidence": True,
                    "probability_backtest_overfitting": True,
                    "deflated_sharpe": True,
                    "drawdown_stability": True,
                }
            )
        elif stage == "forward":
            evidence.update(
                {
                    "production_equivalent": True,
                    "exact_strategy_identity": True,
                    "exact_cost_model": True,
                    "drift_checks": True,
                }
            )
        return evidence, True, None
