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
from src.research.theses import REQUIRED_NEGATIVE_CONTROLS, SqlThesisRegistry, ThesisError

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
        "dataset_roles",
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
    dataset_roles: Mapping[str, str] | None = None

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
        raw_roles = payload.get("dataset_roles")
        roles: dict[str, str] | None = None
        if raw_roles is not None:
            if not isinstance(raw_roles, Mapping) or set(raw_roles) != set(snapshots):
                raise EvaluationContractError(
                    "dataset_roles must map every dataset snapshot identity to one role"
                )
            allowed_roles = {
                "screening",
                "development",
                "robustness",
                "protected_holdout",
                "forward_observation",
                "unspecified",
            }
            roles = {str(key): str(value) for key, value in raw_roles.items()}
            if any(value not in allowed_roles for value in roles.values()):
                raise EvaluationContractError("dataset_roles contains an unsupported role")
            expected_role = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
                "protected": "protected_holdout",
                "forward": "forward_observation",
            }[stage]
            if sum(value == expected_role for value in roles.values()) != 1:
                raise EvaluationContractError(
                    f"dataset_roles must contain exactly one {expected_role} snapshot"
                )
            if stage != "protected" and "protected_holdout" in roles.values():
                raise EvaluationContractError(
                    "adaptive evaluation cannot contain a protected_holdout snapshot"
                )
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
            dataset_roles=roles,
        )

    def snapshot_ids_for_stage(self, stage: str) -> tuple[str, ...]:
        if not self.dataset_roles:
            return (
                (self.dataset_snapshot_ids[0],)
                if stage == "protected"
                else self.dataset_snapshot_ids
            )
        role = {
            "screening": "screening",
            "development": "development",
            "robustness": "robustness",
            "protected": "protected_holdout",
            "forward": "forward_observation",
        }[stage]
        return tuple(
            snapshot_id
            for snapshot_id in self.dataset_snapshot_ids
            if self.dataset_roles.get(snapshot_id) == role
        )

    def protected_snapshot_id(self) -> str:
        values = self.snapshot_ids_for_stage("protected")
        if len(values) != 1:
            raise EvaluationContractError("exactly one protected_holdout snapshot is required")
        return values[0]


@dataclass(frozen=True)
class StageEvaluation:
    candidate_id: str
    stage: str
    accepted: bool
    reason_code: str | None
    run_id: str
    evidence_hash: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class EvidencePolicy:
    """Typed acceptance thresholds for measured research evidence."""

    minimum_cost_adjusted_return: float = 0.0001
    minimum_deflated_sharpe: float = 0.95
    minimum_walk_forward_windows: int = 3
    minimum_walk_forward_pass_fraction: float = 0.67
    maximum_backtest_overfitting_probability: float = 0.25
    maximum_portfolio_correlation: float = 0.8
    minimum_bootstrap_observations: int = 30
    version: str = "research-evidence/v1"
    bootstrap_method: str = "moving_block_bootstrap_v1"
    multiple_testing_method: str = "bailey_lopez_de_prado_dsr_v1"
    pbo_method: str = "combinatorial_purged_pbo_v1"

    @property
    def policy_hash(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "minimum_cost_adjusted_return": self.minimum_cost_adjusted_return,
                "minimum_deflated_sharpe": self.minimum_deflated_sharpe,
                "minimum_walk_forward_windows": self.minimum_walk_forward_windows,
                "minimum_walk_forward_pass_fraction": self.minimum_walk_forward_pass_fraction,
                "maximum_backtest_overfitting_probability": self.maximum_backtest_overfitting_probability,
                "maximum_portfolio_correlation": self.maximum_portfolio_correlation,
                "minimum_bootstrap_observations": self.minimum_bootstrap_observations,
                "bootstrap_method": self.bootstrap_method,
                "multiple_testing_method": self.multiple_testing_method,
                "pbo_method": self.pbo_method,
            }
        )

    def accepts(self, stage: str, evidence: Mapping[str, Any], controls: tuple[str, ...]) -> bool:
        if evidence.get("evidence_policy_hash") != self.policy_hash:
            return False
        validators = _STAGE_EVIDENCE_VALIDATORS.get(stage, {})
        if not validators or any(
            not validator(evidence.get(name), self) for name, validator in validators.items()
        ):
            return False
        if stage == "robustness":
            results = evidence.get("negative_control_results")
            if not isinstance(results, Mapping) or any(
                not _passed_mapping(results.get(control)) for control in controls
            ):
                return False
        return True


def _passed_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("passed") is True


def _true(value: object, _policy: EvidencePolicy) -> bool:
    return value is True


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if __import__("math").isfinite(result) else None


def _nonnegative(value: object, _policy: EvidencePolicy) -> bool:
    measured = _finite(value)
    return measured is not None and measured >= 0


def _positive(value: object, _policy: EvidencePolicy) -> bool:
    measured = _finite(value)
    return measured is not None and measured > 0


def _return_passes(value: object, policy: EvidencePolicy) -> bool:
    measured = _finite(value)
    return measured is not None and measured >= policy.minimum_cost_adjusted_return


def _deflated_sharpe_passes(value: object, policy: EvidencePolicy) -> bool:
    measured = _finite(value)
    return measured is not None and measured >= policy.minimum_deflated_sharpe


def _pbo_passes(value: object, policy: EvidencePolicy) -> bool:
    measured = _finite(value)
    return measured is not None and 0 <= measured <= policy.maximum_backtest_overfitting_probability


def _walk_forward_passes(value: object, policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    windows = value.get("window_count")
    fraction = _finite(value.get("pass_fraction"))
    return (
        isinstance(windows, int)
        and not isinstance(windows, bool)
        and windows >= policy.minimum_walk_forward_windows
        and fraction is not None
        and fraction >= policy.minimum_walk_forward_pass_fraction
    )


def _mapping_passes(value: object, _policy: EvidencePolicy) -> bool:
    if not _passed_mapping(value):
        return False
    assert isinstance(value, Mapping)
    return any(
        key != "passed"
        and value[key] not in (None, (), [], {})
        and not isinstance(value[key], bool)
        for key in value
    )


def _parameter_stability_passes(value: object, _policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    results = value.get("results")
    tested = value.get("neighbours_tested")
    if not isinstance(results, list | tuple) or not isinstance(tested, int) or tested < 2:
        return False
    if len(results) != tested:
        return False
    return all(
        isinstance(item, Mapping)
        and item.get("passed") is True
        and isinstance(item.get("observations"), int)
        and int(item["observations"]) > 0
        and _finite(item.get("return")) is not None
        and _valid_hash(item.get("input_hash"))
        for item in results
    )


def _cross_symbol_stability_passes(value: object, _policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    per_symbol = value.get("per_symbol")
    symbols = value.get("symbols")
    if not isinstance(per_symbol, Mapping) or not isinstance(symbols, int) or symbols <= 0:
        return False
    return len(per_symbol) == symbols and all(
        isinstance(item, Mapping)
        and item.get("passed") is True
        and isinstance(item.get("observations"), int)
        and int(item["observations"]) >= 2
        and _finite(item.get("return")) is not None
        and _valid_hash(item.get("input_hash"))
        for item in per_symbol.values()
    )


def _portfolio_overlap_passes(value: object, policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    comparisons = value.get("comparisons")
    count = value.get("active_strategy_count")
    maximum = _finite(value.get("maximum_correlation"))
    threshold = _finite(value.get("threshold"))
    if (
        not isinstance(comparisons, list | tuple)
        or not isinstance(count, int)
        or count <= 0
        or len(comparisons) != count
        or maximum is None
        or threshold is None
        or threshold < 0.0
        or threshold > policy.maximum_portfolio_correlation
        or maximum > threshold
    ):
        return False
    return all(
        isinstance(item, Mapping)
        and isinstance(item.get("observations"), int)
        and int(item["observations"]) >= 2
        and _finite(item.get("correlation")) is not None
        and _valid_hash(item.get("input_hash"))
        for item in comparisons
    )


def _valid_hash(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _statistical_procedures_pass(value: object, policy: EvidencePolicy) -> bool:
    return isinstance(value, Mapping) and value == {
        "bootstrap": policy.bootstrap_method,
        "multiple_testing": policy.multiple_testing_method,
        "pbo": policy.pbo_method,
    }


def _negative_control_results_pass(value: object, _policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        _passed_mapping(value.get(name))
        and isinstance(value[name].get("observations"), int)
        and int(value[name]["observations"]) > 0
        and _valid_hash(value[name].get("input_hash"))
        for name in REQUIRED_NEGATIVE_CONTROLS
    )


def _bootstrap_passes(value: object, policy: EvidencePolicy) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    observations = value.get("observations")
    lower = _finite(value.get("lower_bound"))
    return (
        isinstance(observations, int)
        and observations >= policy.minimum_bootstrap_observations
        and lower is not None
        and lower >= 0.0
    )


def _mapping(value: object, _policy: EvidencePolicy) -> bool:
    return isinstance(value, Mapping) and bool(value)


EvidenceValidator = Callable[[object, EvidencePolicy], bool]


_STAGE_EVIDENCE_VALIDATORS: dict[str, dict[str, EvidenceValidator]] = {
    "screening": {
        "compiled": _true,
        "features_valid": _true,
        "causality_valid": _true,
        "signal_frequency": _positive,
        "turnover": _nonnegative,
    },
    "development": {
        "chronological": _true,
        "cost_adjusted_return": _return_passes,
        "fees": _nonnegative,
        "slippage": _nonnegative,
        "funding": _nonnegative,
        "regime_breakdown": _mapping_passes,
        "parameter_stability": _parameter_stability_passes,
        "sample_evidence": _mapping_passes,
        "cross_symbol_stability": _cross_symbol_stability_passes,
        "universe_evidence": _mapping_passes,
        "portfolio_overlap": _portfolio_overlap_passes,
    },
    "robustness": {
        "walk_forward": _walk_forward_passes,
        "purged": _true,
        "embargo": _positive,
        "cost_stress": _mapping_passes,
        "delay_stress": _mapping_passes,
        "adverse_fill_stress": _mapping_passes,
        "missing_data_stress": _mapping_passes,
        "funding_stress": _mapping_passes,
        "monte_carlo_trade_order": _mapping_passes,
        "bootstrap_confidence": _bootstrap_passes,
        "probability_backtest_overfitting": _pbo_passes,
        "deflated_sharpe": _deflated_sharpe_passes,
        "statistical_procedures": _statistical_procedures_pass,
        "drawdown_stability": _mapping_passes,
        "null_results": _mapping_passes,
        "negative_control_results": _negative_control_results_pass,
    },
    "forward": {
        "production_equivalent": _mapping_passes,
        "exact_strategy_identity": _mapping_passes,
        "exact_artefact_hash": _mapping_passes,
        "exact_engine_hash": _mapping_passes,
        "exact_cost_model": _mapping_passes,
        "drift_checks": _mapping_passes,
        "duration": _positive,
        "evidence_units": _positive,
    },
}


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
        accepted, result = self.evaluator(
            {
                "claim_id": claim_id,
                "strategy_version_id": strategy_version_id,
                "dataset_snapshot_id": dataset_snapshot_id,
                "cohort_id": cohort_id,
                "source_hashes": dict(source_hashes),
            }
        )
        if not isinstance(accepted, bool) or not isinstance(result, Mapping):
            raise EvaluationContractError("protected evaluator returned an invalid sealed outcome")
        sealed = result.get("sealed_result")
        if not isinstance(sealed, Mapping) or not sealed:
            raise EvaluationContractError("protected evaluator returned no sealed outcome")
        outcome_id = self.repository.record_outcome(
            claim_id=claim_id,
            evaluated_at=evaluated_at,
            accepted=accepted,
            outcome=sealed,
        )
        return claim_id, outcome_id, accepted, dict(sealed)


class CanonicalResearchEvaluator:
    def __init__(
        self,
        store: SqlResearchStore,
        *,
        executors: ProviderExecutorRegistry | None = None,
        provider_context: Mapping[str, Any] | None = None,
        protected_worker: ProtectedHoldoutWorker | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ):
        self.store = store
        self.validation = SqlValidationRepository(store.engine)
        self.protected = ProtectedEvaluationBoundary(store.engine)
        self.executors = executors or ProviderExecutorRegistry.default()
        self.provider_context = dict(provider_context or {})
        self.protected_worker = protected_worker
        self.evidence_policy = evidence_policy or EvidencePolicy()

    def _negative_controls(self, thesis_id: str) -> tuple[str, ...]:
        try:
            return SqlThesisRegistry(self.store.engine).get(thesis_id).negative_controls
        except ThesisError:
            return ()

    def evaluate(self, request: EvaluationRequest) -> StageEvaluation:
        if request.dataset_roles:
            expected_role = {
                "screening": "screening",
                "development": "development",
                "robustness": "robustness",
                "protected": "protected_holdout",
                "forward": "forward_observation",
            }[request.requested_stage]
            if sum(role == expected_role for role in request.dataset_roles.values()) != 1:
                raise EvaluationContractError(
                    f"dataset_roles must contain exactly one {expected_role} snapshot"
                )
            if request.requested_stage != "protected" and "protected_holdout" in set(
                request.dataset_roles.values()
            ):
                raise EvaluationContractError(
                    "adaptive evaluation cannot contain a protected_holdout snapshot"
                )
        candidate = self.store.get_candidate(request.candidate_id)
        if not set(request.dataset_snapshot_ids).issubset(set(candidate.dataset_snapshot_hashes)):
            raise EvaluationContractError(
                "dataset_snapshot_ids are not a subset of the candidate's immutable dataset identities"
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
        stage_snapshot_ids = request.snapshot_ids_for_stage(request.requested_stage)
        protected_snapshot_id = (
            (
                request.protected_snapshot_id()
                if request.dataset_roles
                else request.dataset_snapshot_ids[0]
            )
            if request.requested_stage == "protected"
            else None
        )
        if request.requested_stage == "protected":
            stage_snapshot_ids = tuple(
                snapshot_id
                for snapshot_id in request.dataset_snapshot_ids
                if snapshot_id != protected_snapshot_id
            )
        adaptive_roles = {
            snapshot_id: request.dataset_roles[snapshot_id]
            for snapshot_id in stage_snapshot_ids
            if request.dataset_roles and snapshot_id in request.dataset_roles
        }
        context = {
            "candidate_id": request.candidate_id,
            "strategy_version_id": definition.strategy_version_id,
            "evaluation_policy_id": request.evaluation_policy_id,
            "dataset_snapshot_ids": list(stage_snapshot_ids),
            "dataset_roles": adaptive_roles,
            "evidence_policy_hash": self.evidence_policy.policy_hash,
            "maximum_portfolio_correlation": self.evidence_policy.maximum_portfolio_correlation,
            "minimum_bootstrap_observations": self.evidence_policy.minimum_bootstrap_observations,
            "walk_forward_windows": self.evidence_policy.minimum_walk_forward_windows,
            "minimum_walk_forward_pass_fraction": (
                self.evidence_policy.minimum_walk_forward_pass_fraction
            ),
            "maximum_backtest_overfitting_probability": (
                self.evidence_policy.maximum_backtest_overfitting_probability
            ),
            "minimum_deflated_sharpe": self.evidence_policy.minimum_deflated_sharpe,
            "bootstrap_method": self.evidence_policy.bootstrap_method,
            "multiple_testing_method": self.evidence_policy.multiple_testing_method,
            "pbo_method": self.evidence_policy.pbo_method,
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
            request.requested_stage,
            candidate,
            context,
            protected_snapshot_id=protected_snapshot_id,
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
        self,
        stage: str,
        candidate,
        context: Mapping[str, Any],
        *,
        protected_snapshot_id: str | None = None,
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
            accepted = (
                fields_valid
                and causality_valid
                and self.evidence_policy.accepts(
                    "screening", measured, self._negative_controls(candidate.thesis_id)
                )
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
                    claim_id, outcome_id, holdout_accepted, sealed_outcome = (
                        self.protected_worker.claim_and_evaluate(
                            strategy_version_id=definition.strategy_version_id,
                            dataset_snapshot_id=str(protected_snapshot_id or ""),
                            cohort_id=f"protected:{context['candidate_id']}",
                            source_hashes=source_hashes,
                            evaluated_at=str(context["evaluated_at"]),
                        )
                    )
                else:
                    claim_id = SqlHoldoutRepository(self.store.engine).claim(
                        strategy_version_id=definition.strategy_version_id,
                        data_snapshot_id=str(protected_snapshot_id or ""),
                        cohort_id=f"protected:{context['candidate_id']}",
                        source_hashes=source_hashes,
                        claimed_at=str(context["evaluated_at"]),
                    )
                    outcome_id = None
                    holdout_accepted = False
                    sealed_outcome = {}
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
                "data_commitment": canonical_hash(
                    {
                        "dataset_snapshot_id": protected_snapshot_id,
                        "source_hashes": source_hashes,
                    }
                ),
                "code_hash": context["code_hash"],
                "holdout_outcome": dict(outcome) if isinstance(outcome, Mapping) else {},
                "holdout_outcome_id": outcome_id,
                "sealed_outcome": dict(sealed_outcome),
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
        required_fields: tuple[str, ...] = {
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
        controls = self._negative_controls(candidate.thesis_id)
        policy_fields = _STAGE_EVIDENCE_VALIDATORS[stage]
        missing = [
            field
            for field in required_fields
            if field not in policy_fields
            or not policy_fields[field](evidence.get(field), self.evidence_policy)
        ]
        accepted = not missing and self.evidence_policy.accepts(stage, evidence, controls)
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
