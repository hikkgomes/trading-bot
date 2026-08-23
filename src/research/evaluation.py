"""Canonical, identity-based research-stage evaluation.

Workers receive references to immutable inputs. They never receive an
acceptance flag or an evidence map to trust. Each stage creates a durable run,
then appends a validation-stage record that points to that run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from src.data.database import experiment, holdout_claim, holdout_outcome
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import SqlHoldoutRepository, SqlValidationRepository
from src.research.executors import ExecutorError, ProviderExecutorRegistry
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
        "feature_manifest_id",
        "cost_model_id",
        "parameter_set_id",
        "evaluator_version",
        "producer_identity",
        "content_hash",
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
    feature_manifest_id: str | None = None
    cost_model_id: str | None = None
    parameter_set_id: str | None = None
    evaluator_version: str | None = None
    producer_identity: str | None = None
    content_hash: str | None = None

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
            str(
                payload.get("evaluation_policy_id")
                or f"canonical:{payload.get('evaluator_version')}"
            ),
            field="evaluation_policy_id",
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
        for field in (
            "code_hash",
            "feature_set_hash",
            "cost_model_hash",
            "feature_manifest_id",
            "cost_model_id",
            "parameter_set_id",
        ):
            value = payload.get(field)
            if value is not None:
                value = non_empty(str(value), field=field)
                if not value.startswith("sha256:") or len(value) != 71:
                    raise EvaluationContractError(f"{field} must be a sha256: identity")
            hashes[field] = value
        evaluator_version = non_empty(
            str(payload.get("evaluator_version") or ""), field="evaluator_version"
        )
        producer_identity = non_empty(
            str(payload.get("producer_identity") or ""), field="producer_identity"
        )
        content_hash = non_empty(str(payload.get("content_hash") or ""), field="content_hash")
        if not content_hash.startswith("sha256:") or len(content_hash) != 71:
            raise EvaluationContractError("content_hash must be a sha256: identity")
        unsigned = dict(payload)
        unsigned.pop("content_hash", None)
        if canonical_hash(unsigned) != content_hash:
            raise EvaluationContractError("content_hash does not match the evaluation request")
        return cls(
            candidate_id=candidate_id,
            evaluation_policy_id=policy_id,
            dataset_snapshot_ids=snapshots,
            requested_stage=stage,
            evaluated_at=evaluated_at,
            evaluator_version=evaluator_version,
            producer_identity=producer_identity,
            content_hash=content_hash,
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


class ProtectedHoldoutWorker:
    """Atomically claim, evaluate, and seal one protected cohort."""

    def __init__(
        self,
        engine: Engine,
        evaluator: Callable[[Mapping[str, Any]], tuple[bool, Mapping[str, Any]]],
    ) -> None:
        self.repository = SqlHoldoutRepository(engine)
        self.evaluator = evaluator

    def claim_and_evaluate(
        self,
        *,
        strategy_version_id: str,
        dataset_snapshot_id: str,
        cohort_id: str,
        source_hashes: Mapping[str, str],
        evaluated_at: str,
    ) -> tuple[str, str, bool, Mapping[str, Any]]:
        claim_id = self.repository.claim(
            strategy_version_id=strategy_version_id,
            data_snapshot_id=dataset_snapshot_id,
            cohort_id=cohort_id,
            source_hashes=source_hashes,
            claimed_at=evaluated_at,
        )
        accepted, measured = self.evaluator(
            {
                "claim_id": claim_id,
                "strategy_version_id": strategy_version_id,
                "dataset_snapshot_id": dataset_snapshot_id,
                "cohort_id": cohort_id,
                "source_hashes": dict(source_hashes),
            }
        )
        if not isinstance(accepted, bool) or not isinstance(measured, Mapping) or not measured:
            raise EvaluationContractError("protected evaluator returned no measured outcome")
        outcome_id = self.repository.record_outcome(
            claim_id=claim_id,
            evaluated_at=evaluated_at,
            accepted=accepted,
            outcome=measured,
        )
        return claim_id, outcome_id, accepted, dict(measured)


class CanonicalResearchEvaluator:
    def __init__(
        self,
        store: SqlResearchStore,
        *,
        executors: ProviderExecutorRegistry | None = None,
        provider_context: Mapping[str, Any] | None = None,
        protected_worker: ProtectedHoldoutWorker | None = None,
    ):
        self.store = store
        self.validation = SqlValidationRepository(store.engine)
        self.protected = ProtectedEvaluationBoundary(store.engine)
        self.executors = executors or ProviderExecutorRegistry.default()
        self.provider_context = dict(provider_context or {})
        self.protected_worker = protected_worker

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
            "feature_manifest_id": request.feature_manifest_id,
            "cost_model_id": request.cost_model_id,
            "parameter_set_id": request.parameter_set_id,
            "evaluator_version": request.evaluator_version,
            "producer_identity": request.producer_identity,
            "content_hash": request.content_hash,
        }
        evidence, accepted, reason_code, receipt, metrics = self._calculate_stage(
            request.requested_stage, candidate, context
        )
        run_id = self.store.save_run(
            candidate_id=request.candidate_id,
            run_name=f"canonical:{request.requested_stage}",
            created_at=request.evaluated_at,
            evidence=evidence,
            metrics={
                "accepted": 1.0 if accepted else 0.0,
                "evidence_hash": float(int(canonical_hash(evidence)[7:15], 16)),
                **metrics,
            },
            receipt=receipt,
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
        self, stage: str, candidate, context: Mapping[str, Any]
    ) -> tuple[
        dict[str, Any],
        bool,
        str | None,
        Mapping[str, Any] | None,
        Mapping[str, float],
    ]:
        definition = candidate.definition
        identity = canonical_hash(
            {
                "definition_hash": definition.definition_hash,
                "context": dict(context),
            }
        )
        if stage in {"screening", "development", "robustness", "forward"}:
            try:
                execution = self.executors.execute(candidate, {**self.provider_context, **context})
            except (ExecutorError, ValueError, TypeError) as exc:
                return (
                    {
                        "identity": identity,
                        "context": dict(context),
                        "executor_error": f"{type(exc).__name__}: {exc}",
                    },
                    False,
                    "candidate_execution_failed",
                    None,
                    {},
                )
        else:
            execution = None
        if stage == "screening":
            assert execution is not None
            measured = dict(execution.evidence)
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
                **measured,
                "features_contract_valid": fields_valid,
                "causality_contract_valid": causality_valid,
                "context": dict(context),
                "execution_receipt": dict(execution.receipt),
            }
            screening_required = (
                "compiled",
                "features_valid",
                "causality_valid",
                "signal_frequency",
                "turnover",
            )
            accepted = (
                fields_valid
                and causality_valid
                and all(_measured(measured.get(field)) for field in screening_required)
            )
            return (
                evidence,
                accepted,
                None if accepted else "screening_measured_evidence_missing",
                execution.receipt,
                execution.metrics,
            )

        required_runs = {
            "development": ("bar_portfolio",),
            "robustness": ("bar_portfolio", "event_replay"),
            "forward": ("forward_paper",),
        }
        if stage == "protected":
            try:
                source_hashes = {
                    "code_hash": str(context["code_hash"]),
                    "feature_manifest_id": str(context["feature_manifest_id"]),
                    "cost_model_id": str(context["cost_model_id"]),
                }
                if self.protected_worker is not None:
                    claim_id, outcome_id, holdout_accepted, measured_outcome = (
                        self.protected_worker.claim_and_evaluate(
                            strategy_version_id=definition.strategy_version_id,
                            dataset_snapshot_id=str(context["dataset_snapshot_ids"][0]),
                            cohort_id=f"protected:{context['candidate_id']}",
                            source_hashes=source_hashes,
                            evaluated_at=str(context["evaluated_at"]),
                        )
                    )
                else:
                    claim_id = SqlHoldoutRepository(self.store.engine).claim(
                        strategy_version_id=definition.strategy_version_id,
                        data_snapshot_id=str(context["dataset_snapshot_ids"][0]),
                        cohort_id=f"protected:{context['candidate_id']}",
                        source_hashes=source_hashes,
                        claimed_at=str(context["evaluated_at"]),
                    )
                    outcome_id = None
                    holdout_accepted = False
                    measured_outcome = {}
            except Exception as exc:
                return (
                    {
                        "identity": identity,
                        "context": dict(context),
                        "holdout_claim_error": f"{type(exc).__name__}: {exc}",
                    },
                    False,
                    "protected_holdout_claim_failed",
                    None,
                    {},
                )
            claims = self.protected.claim_for(definition.strategy_version_id)
            with self.store.engine.connect() as connection:
                outcome = connection.execute(
                    select(holdout_outcome.c.payload)
                    .where(holdout_outcome.c.holdout_claim_id == claim_id)
                    .order_by(holdout_outcome.c.evaluated_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
            evidence = {
                "identity": identity,
                "frozen_cohort": bool(claims),
                "holdout_claim": [row["id"] for row in claims],
                "data_hashes": list(context["dataset_snapshot_ids"]),
                "code_hash": context["code_hash"],
                "holdout_outcome": dict(outcome) if isinstance(outcome, Mapping) else {},
                "holdout_outcome_id": outcome_id,
                "measured_holdout_outcome": measured_outcome,
                "context": dict(context),
            }
            accepted = bool(claims) and (
                holdout_accepted
                if self.protected_worker is not None
                else bool(evidence.get("holdout_outcome"))
            )
            return (
                evidence,
                accepted,
                None if accepted else "protected_holdout_outcome_missing",
                None,
                {},
            )

        runs = self.store.runs(context["candidate_id"])
        authoritative_runs: list[dict[str, Any]] = []
        for row in runs:
            raw_payload = row.get("payload")
            if not isinstance(raw_payload, Mapping):
                continue
            raw_receipt = raw_payload.get("receipt")
            if not isinstance(raw_receipt, Mapping):
                continue
            if raw_receipt.get("candidate_id") != context["candidate_id"]:
                continue
            if tuple(raw_receipt.get("dataset_snapshot_ids", ())) != tuple(
                context["dataset_snapshot_ids"]
            ):
                continue
            authoritative_runs.append({"id": str(row["id"]), "payload": dict(raw_payload)})
        run_names = {str(row["payload"].get("run_name")) for row in authoritative_runs}
        required_runs_for_stage = required_runs[stage]
        if execution is not None:
            authoritative_runs.append(
                {
                    "id": "executed:" + execution.receipt["input_hash"],
                    "payload": {
                        "run_name": "canonical-execution",
                        "evidence": execution.evidence,
                        "metrics": execution.metrics,
                        "receipt": execution.receipt,
                    },
                }
            )
            run_names = run_names | {"canonical-execution"}
        missing = (
            []
            if execution is not None
            else [name for name in required_runs_for_stage if name not in run_names]
        )
        evidence = {
            "identity": identity,
            "authoritative_run_ids": [row["id"] for row in authoritative_runs],
            "required_runs": list(required_runs_for_stage),
            "available_runs": sorted(run_names),
            "context": dict(context),
        }
        if missing:
            return evidence, False, "missing_authoritative_run", None, {}
        run_measured: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        for row in authoritative_runs:
            payload = cast(Mapping[str, Any], row["payload"])
            run_evidence = payload.get("evidence")
            if isinstance(run_evidence, Mapping):
                run_measured.update(run_evidence)
            run_metrics = payload.get("metrics")
            if isinstance(run_metrics, Mapping):
                for key, value in run_metrics.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        metrics[str(key)] = float(value)
        evidence.update(run_measured)
        required_fields = {
            "development": (
                "chronological",
                "cost_adjusted_return",
                "funding",
                "regime_breakdown",
                "parameter_stability",
                "sample_evidence",
                "cross_symbol_stability",
                "universe_evidence",
                "portfolio_overlap",
            ),
            "robustness": (
                "walk_forward",
                "purged",
                "embargo",
                "cost_stress",
                "delay_stress",
                "adverse_fill_stress",
                "missing_data_stress",
                "funding_stress",
                "monte_carlo_trade_order",
                "bootstrap_confidence",
                "probability_backtest_overfitting",
                "deflated_sharpe",
                "drawdown_stability",
                "null_results",
                "negative_control_results",
            ),
            "forward": (
                "production_equivalent",
                "exact_strategy_identity",
                "exact_artefact_hash",
                "exact_engine_hash",
                "exact_cost_model",
                "drift_checks",
                "duration",
                "evidence_units",
            ),
        }[stage]
        missing = [field for field in required_fields if not _measured(evidence.get(field))]
        accepted = not missing
        evidence["missing_evidence"] = missing
        receipt: Mapping[str, Any] | None = None
        for row in authoritative_runs:
            raw_receipt = cast(Mapping[str, Any], row["payload"]).get("receipt")
            if isinstance(raw_receipt, Mapping) and raw_receipt:
                receipt = raw_receipt
                break
        if execution is not None:
            receipt = execution.receipt
            metrics.update(execution.metrics)
        return (
            evidence,
            accepted,
            None if accepted else "measured_evidence_missing",
            receipt,
            metrics,
        )


def _measured(value: object) -> bool:
    """Return true only for evidence produced by a run, never a default flag."""

    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) | isinstance(value, list) | isinstance(value, tuple):
        return bool(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    return isinstance(value, str) and bool(value.strip())
