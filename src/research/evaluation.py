"""Canonical, identity-based research-stage evaluation.

Workers receive references to immutable inputs. They never receive an
acceptance flag or an evidence map to trust. Each stage creates a durable run,
then appends a validation-stage record that points to that run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from src.data.database import experiment, holdout_claim
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.research.canonical import SqlHoldoutRepository, SqlValidationRepository
from src.research.datasets import CanonicalDatasetResolver
from src.research.evidence import (
    EvidenceProfile,
    cross_symbol_stability_passes,
    data_integrity_passes,
    drawdown_passes,
    family_evidence_passes,
    monte_carlo_passes,
    parameter_stability_passes,
    realistic_costs_passes,
    regime_breakdown_passes,
    sample_evidence_passes,
    select_profile,
    semantic_parity_passes,
)
from src.research.executors import ExecutionResult, ExecutorError, ProviderExecutorRegistry
from src.research.objectives import objective_is_available, objective_passes
from src.research.store import SqlResearchStore
from src.research.theses import SqlThesisRegistry, ThesisError

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
        "artefact_hash",
        "artefact_created_at",
    }
)


class EvaluationContractError(ValueError):
    """A research job tried to submit outcomes or an invalid identity."""


@dataclass(frozen=True)
class _ExecutionAttempt:
    result: ExecutionResult | None
    error: str | None = None


@dataclass(frozen=True)
class _ProtectedClaim:
    outcome_id: str | None
    accepted: bool
    error: str | None = None


_REQUIRED_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "development": (
        "chronological",
        "data_integrity",
        "semantic_parity",
        "realistic_costs",
        "family_evidence",
        "cost_adjusted_return",
        "funding",
        "regime_breakdown",
        "parameter_stability",
        "sample_evidence",
        "cross_symbol_stability",
        "universe_evidence",
    ),
    "robustness": (
        "data_integrity",
        "semantic_parity",
        "realistic_costs",
        "family_evidence",
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
        "data_integrity",
        "semantic_parity",
        "realistic_costs",
        "family_evidence",
        "production_equivalent",
        "exact_strategy_identity",
        "exact_artefact_hash",
        "exact_engine_hash",
        "exact_cost_model",
        "drift_checks",
        "duration",
        "evidence_units",
        "sample_evidence",
        "forward_duration",
    ),
    "protected": (
        "chronological",
        "data_integrity",
        "semantic_parity",
        "realistic_costs",
        "family_evidence",
        "cost_adjusted_return",
        "sample_evidence",
        "drawdown_stability",
    ),
}


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
    artefact_hash: str | None = None
    artefact_created_at: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvaluationRequest:
        payload = _validate_evaluation_request_payload(payload)
        candidate_id = _request_identity(payload, "candidate_id")
        policy_id = non_empty(
            str(
                payload.get("evaluation_policy_id")
                or f"canonical:{payload.get('evaluator_version')}"
            ),
            field="evaluation_policy_id",
        )
        snapshots = _request_snapshot_ids(payload)
        stage = _request_stage(payload)
        evaluated_at = timestamp(str(payload.get("evaluated_at") or ""), field="evaluated_at")
        hashes = _request_hashes(payload)
        evaluator_version = non_empty(
            str(payload.get("evaluator_version") or ""), field="evaluator_version"
        )
        producer_identity = non_empty(
            str(payload.get("producer_identity") or ""), field="producer_identity"
        )
        content_hash = _request_content_hash(payload)
        roles = _request_dataset_roles(payload, snapshots=snapshots, stage=stage)
        artefact_hash, artefact_created_at = _request_artefact(
            payload, stage=stage, evaluated_at=evaluated_at
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
            artefact_hash=artefact_hash,
            artefact_created_at=artefact_created_at,
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


def _validate_evaluation_request_payload(payload: object) -> Mapping[str, Any]:
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
    return payload


def _request_identity(payload: Mapping[str, Any], field: str) -> str:
    value = non_empty(str(payload.get(field) or ""), field=field)
    if not value.startswith("sha256:") or len(value) != 71:
        raise EvaluationContractError(f"{field} must be a sha256: identity")
    return value


def _request_snapshot_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw_snapshots = payload.get("dataset_snapshot_ids")
    if not isinstance(raw_snapshots, list | tuple) or not raw_snapshots:
        raise EvaluationContractError("dataset_snapshot_ids must be a non-empty list")
    snapshots = tuple(
        non_empty(str(value), field="dataset_snapshot_ids[]") for value in raw_snapshots
    )
    if len(set(snapshots)) != len(snapshots) or any(
        not value.startswith("sha256:") or len(value) != 71 for value in snapshots
    ):
        raise EvaluationContractError("dataset_snapshot_ids must contain unique sha256: identities")
    return snapshots


def _request_stage(payload: Mapping[str, Any]) -> str:
    stage = non_empty(str(payload.get("requested_stage") or ""), field="requested_stage")
    if stage not in STAGES:
        raise EvaluationContractError(f"requested_stage must be one of {STAGES}")
    return stage


def _request_hashes(payload: Mapping[str, Any]) -> dict[str, str | None]:
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
        hashes[field] = None if value is None else _request_identity({field: value}, field)
    return hashes


def _request_content_hash(payload: Mapping[str, Any]) -> str:
    content_hash = _request_identity(payload, "content_hash")
    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    if canonical_hash(unsigned) != content_hash:
        raise EvaluationContractError("content_hash does not match the evaluation request")
    return content_hash


def _request_dataset_roles(
    payload: Mapping[str, Any], *, snapshots: tuple[str, ...], stage: str
) -> dict[str, str]:
    raw_roles = payload.get("dataset_roles")
    if raw_roles is None:
        raise EvaluationContractError("evaluate_candidate requests require explicit dataset roles")
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
    if stage == "protected" and (
        len(snapshots) != 1 or roles.get(snapshots[0]) != "protected_holdout"
    ):
        raise EvaluationContractError(
            "protected evaluation requests may contain only the protected_holdout snapshot"
        )
    if stage != "protected" and "protected_holdout" in roles.values():
        raise EvaluationContractError(
            "adaptive evaluation cannot contain a protected_holdout snapshot"
        )
    return roles


def _request_artefact(
    payload: Mapping[str, Any], *, stage: str, evaluated_at: str
) -> tuple[str | None, str | None]:
    artefact_hash = payload.get("artefact_hash")
    if artefact_hash is not None:
        artefact_hash = _request_identity({"artefact_hash": artefact_hash}, "artefact_hash")
    artefact_created_at = payload.get("artefact_created_at")
    if artefact_created_at is not None:
        artefact_created_at = timestamp(str(artefact_created_at), field="artefact_created_at")
    if stage == "forward":
        if artefact_hash is None or artefact_created_at is None:
            raise EvaluationContractError(
                "forward evaluations require the exact artefact hash and creation time"
            )
        if artefact_created_at >= evaluated_at:
            raise EvaluationContractError("forward artefact creation must precede evaluation time")
    return artefact_hash, artefact_created_at


@dataclass(frozen=True)
class StageEvaluation:
    candidate_id: str
    stage: str
    accepted: bool
    reason_code: str | None
    run_id: str
    evidence_hash: str
    evidence: Mapping[str, Any]


class EvidenceStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class EvidencePolicy:
    """Typed acceptance thresholds for measured research evidence."""

    minimum_cost_adjusted_return: float = 0.0001
    minimum_deflated_sharpe: float = 0.95
    minimum_walk_forward_windows: int = 5
    minimum_walk_forward_pass_fraction: float = 0.67
    maximum_backtest_overfitting_probability: float = 0.25
    maximum_portfolio_correlation: float = 0.8
    minimum_bootstrap_observations: int = 30
    version: str = "research-evidence/v1"
    bootstrap_method: str = "moving_block_bootstrap_v1"
    multiple_testing_method: str = "bailey_lopez_de_prado_dsr_v1"
    pbo_method: str = "combinatorial_purged_pbo_v1"
    profiles: tuple[EvidenceProfile, ...] = ()

    @classmethod
    def from_configuration(cls, configuration: Mapping[str, Any]) -> EvidencePolicy:
        required = {
            "version",
            "minimum_cost_adjusted_return",
            "minimum_deflated_sharpe",
            "minimum_walk_forward_windows",
            "minimum_walk_forward_pass_fraction",
            "maximum_backtest_overfitting_probability",
            "maximum_portfolio_correlation",
            "minimum_bootstrap_observations",
            "bootstrap_method",
            "multiple_testing_method",
            "pbo_method",
        }
        missing = sorted(required - set(configuration))
        if missing:
            raise EvaluationContractError(
                "research evidence policy is missing versioned fields: " + ", ".join(missing)
            )
        raw_profiles = configuration.get("profiles", ())
        if not isinstance(raw_profiles, list | tuple):
            raise EvaluationContractError("research evidence policy profiles must be a list")
        try:
            profiles = tuple(EvidenceProfile.from_mapping(item) for item in raw_profiles)
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(f"invalid research evidence profile: {exc}") from exc
        return cls(
            **{str(key): configuration[key] for key in required},
            profiles=profiles,
        )

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
                "profiles": [profile.to_payload() for profile in self.profiles],
            }
        )

    def profile_for(
        self,
        stage: str,
        *,
        product_id: str | None = None,
        family: str | None = None,
        horizon: str | None = None,
        evidence_type: str | None = None,
    ) -> EvidenceProfile:
        selected = select_profile(
            self.profiles,
            stage=stage,
            product_id=product_id,
            family=family,
            horizon=horizon,
        )
        if selected is None and evidence_type is not None and evidence_type != family:
            selected = select_profile(
                self.profiles,
                stage=stage,
                product_id=product_id,
                family=evidence_type,
                horizon=horizon,
            )
        return selected or EvidenceProfile()

    def accepts(
        self,
        stage: str,
        evidence: Mapping[str, Any],
        controls: tuple[str, ...],
        *,
        product_id: str | None = None,
        family: str | None = None,
        horizon: str | None = None,
        evidence_type: str | None = None,
    ) -> bool:
        if evidence.get("evidence_policy_hash") != self.policy_hash:
            return False
        statuses = self.statuses(
            stage,
            evidence,
            controls,
            product_id=product_id,
            family=family,
            horizon=horizon,
            evidence_type=evidence_type,
        )
        required_statuses = {
            name: status
            for name, status in statuses.items()
            if name not in _OPTIONAL_EVIDENCE_VALIDATORS.get(stage, {})
        }
        if not required_statuses or any(
            status not in {EvidenceStatus.PASS, EvidenceStatus.NOT_APPLICABLE}
            for status in required_statuses.values()
        ):
            return False
        if product_id is not None and stage in {
            "development",
            "robustness",
            "protected",
            "forward",
        }:
            profile = self.profile_for(
                stage,
                product_id=product_id,
                family=family,
                horizon=horizon,
                evidence_type=evidence_type,
            )
            if not objective_passes(
                evidence,
                product_id=product_id,
                minimum_excess_fraction=(
                    profile.minimum_cost_adjusted_return
                    if profile.minimum_cost_adjusted_return is not None
                    else self.minimum_cost_adjusted_return
                ),
            ):
                return False
        return True

    def statuses(
        self,
        stage: str,
        evidence: Mapping[str, Any],
        controls: tuple[str, ...],
        *,
        product_id: str | None = None,
        family: str | None = None,
        horizon: str | None = None,
        evidence_type: str | None = None,
    ) -> dict[str, EvidenceStatus]:
        validators = _STAGE_EVIDENCE_VALIDATORS.get(stage, {})
        optional = _OPTIONAL_EVIDENCE_VALIDATORS.get(stage, {})
        profile = self.profile_for(
            stage,
            product_id=product_id,
            family=family,
            horizon=horizon,
            evidence_type=evidence_type,
        )
        statuses: dict[str, EvidenceStatus] = {}
        for name, validator in {**validators, **optional}.items():
            value = evidence.get(name)
            if name == "negative_control_results" and stage == "robustness":
                statuses[name] = _negative_control_status(value, controls)
            elif _is_not_applicable(value):
                statuses[name] = EvidenceStatus.NOT_APPLICABLE
            elif _is_unavailable(value):
                statuses[name] = EvidenceStatus.UNAVAILABLE
            else:
                statuses[name] = (
                    EvidenceStatus.PASS if validator(value, self, profile) else EvidenceStatus.FAIL
                )
        if product_id is not None and stage in {
            "development",
            "robustness",
            "protected",
            "forward",
        }:
            if not objective_is_available(evidence, product_id=product_id):
                statuses["objective_excess_fraction"] = EvidenceStatus.UNAVAILABLE
            else:
                statuses["objective_excess_fraction"] = (
                    EvidenceStatus.PASS
                    if objective_passes(
                        evidence,
                        product_id=product_id,
                        minimum_excess_fraction=(
                            profile.minimum_cost_adjusted_return
                            if profile.minimum_cost_adjusted_return is not None
                            else self.minimum_cost_adjusted_return
                        ),
                    )
                    else EvidenceStatus.FAIL
                )
        return statuses


def _passed_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("passed") is True


def _is_not_applicable(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("status") == EvidenceStatus.NOT_APPLICABLE


def _is_unavailable(value: object) -> bool:
    return value is None or (
        isinstance(value, Mapping) and value.get("status") == EvidenceStatus.UNAVAILABLE
    )


def _negative_control_status(value: object, controls: tuple[str, ...]) -> EvidenceStatus:
    if not controls:
        return EvidenceStatus.NOT_APPLICABLE
    if not isinstance(value, Mapping):
        return EvidenceStatus.UNAVAILABLE
    statuses = [
        EvidenceStatus.NOT_APPLICABLE
        if _is_not_applicable(value.get(control))
        else EvidenceStatus.UNAVAILABLE
        if _is_unavailable(value.get(control))
        else EvidenceStatus.PASS
        if _passed_mapping(value.get(control))
        else EvidenceStatus.FAIL
        for control in controls
    ]
    if any(status is EvidenceStatus.UNAVAILABLE for status in statuses):
        return EvidenceStatus.UNAVAILABLE
    if any(status is EvidenceStatus.FAIL for status in statuses):
        return EvidenceStatus.FAIL
    if all(status is EvidenceStatus.NOT_APPLICABLE for status in statuses):
        return EvidenceStatus.NOT_APPLICABLE
    return EvidenceStatus.PASS


def _true(value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None) -> bool:
    return value is True


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nonnegative(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    measured = _finite(value)
    return measured is not None and measured >= 0


def _positive(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    measured = _finite(value)
    return measured is not None and measured > 0


def _return_passes(
    value: object, policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    measured = _finite(value)
    minimum = (
        profile.minimum_cost_adjusted_return
        if profile is not None and profile.minimum_cost_adjusted_return is not None
        else policy.minimum_cost_adjusted_return
    )
    return measured is not None and measured >= minimum


def _deflated_sharpe_passes(
    value: object, policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    measured = _finite(value)
    minimum = (
        profile.minimum_deflated_sharpe
        if profile is not None and profile.minimum_deflated_sharpe is not None
        else policy.minimum_deflated_sharpe
    )
    return measured is not None and measured >= minimum


def _pbo_passes(
    value: object, policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    if _is_not_applicable(value):
        return True
    measured = _finite(value)
    maximum = (
        profile.maximum_backtest_overfitting_probability
        if profile is not None and profile.maximum_backtest_overfitting_probability is not None
        else policy.maximum_backtest_overfitting_probability
    )
    return measured is not None and 0 <= measured <= maximum


def _walk_forward_passes(
    value: object, policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    windows = value.get("window_count")
    fraction = _finite(value.get("pass_fraction"))
    minimum_windows = (
        profile.minimum_walk_forward_windows
        if profile is not None and profile.minimum_walk_forward_windows is not None
        else policy.minimum_walk_forward_windows
    )
    minimum_fraction = (
        profile.minimum_walk_forward_pass_fraction
        if profile is not None and profile.minimum_walk_forward_pass_fraction is not None
        else policy.minimum_walk_forward_pass_fraction
    )
    return (
        isinstance(windows, int)
        and not isinstance(windows, bool)
        and windows >= minimum_windows
        and fraction is not None
        and fraction >= minimum_fraction
    )


def _mapping_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    if not _passed_mapping(value):
        return False
    assert isinstance(value, Mapping)
    return any(
        key != "passed"
        and value[key] not in (None, (), [], {})
        and not isinstance(value[key], bool)
        for key in value
    )


def _data_integrity_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return data_integrity_passes(value)


def _semantic_parity_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return semantic_parity_passes(value)


def _realistic_costs_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return realistic_costs_passes(value)


def _family_evidence_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return family_evidence_passes(value)


def _regime_breakdown_passes(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return regime_breakdown_passes(value)


def _forward_duration_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    profile = profile or EvidenceProfile()
    calendar_days = _finite(value.get("calendar_days"))
    trading_days = value.get("trading_days")
    cycles = value.get("cycles")
    if calendar_days is None or calendar_days < profile.minimum_calendar_days:
        return False
    if (
        not isinstance(trading_days, int)
        or isinstance(trading_days, bool)
        or trading_days < profile.minimum_trading_days
    ):
        return False
    return (
        isinstance(cycles, int)
        and not isinstance(cycles, bool)
        and cycles >= profile.minimum_cycles
    )


def _parameter_stability_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    return parameter_stability_passes(value, profile or EvidenceProfile())


def _cross_symbol_stability_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    return cross_symbol_stability_passes(value, profile or EvidenceProfile())


def _portfolio_overlap_passes(
    value: object, policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    if _is_not_applicable(value):
        return True
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
        and _valid_hash(item.get("run_id"))
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


def _statistical_procedures_pass(
    value: object, policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return isinstance(value, Mapping) and value == {
        "bootstrap": policy.bootstrap_method,
        "multiple_testing": policy.multiple_testing_method,
        "pbo": policy.pbo_method,
    }


def _negative_control_results_pass(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        _passed_mapping(value.get(name))
        and isinstance(value[name].get("observations"), int)
        and int(value[name]["observations"]) > 0
        and _valid_hash(value[name].get("input_hash"))
        for name in value
    )


def _bootstrap_passes(
    value: object, policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    observations = value.get("observations")
    lower = _finite(value.get("lower_bound"))
    minimum = (
        profile.minimum_bootstrap_observations
        if profile is not None and profile.minimum_bootstrap_observations is not None
        else policy.minimum_bootstrap_observations
    )
    return (
        isinstance(observations, int)
        and observations >= minimum
        and lower is not None
        and lower >= 0.0
    )


def _sample_evidence_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    return sample_evidence_passes(value, profile or EvidenceProfile())


def _monte_carlo_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    return monte_carlo_passes(value, profile or EvidenceProfile())


def _drawdown_passes(
    value: object, _policy: EvidencePolicy, profile: EvidenceProfile | None = None
) -> bool:
    return drawdown_passes(value, profile or EvidenceProfile())


def _mapping(
    value: object, _policy: EvidencePolicy, _profile: EvidenceProfile | None = None
) -> bool:
    return isinstance(value, Mapping) and bool(value)


EvidenceValidator = Callable[[object, EvidencePolicy, EvidenceProfile], bool]


_STAGE_EVIDENCE_VALIDATORS: dict[str, dict[str, EvidenceValidator]] = {
    "screening": {
        "compiled": _true,
        "features_valid": _true,
        "causality_valid": _true,
        "data_integrity": _data_integrity_passes,
        "semantic_parity": _semantic_parity_passes,
        "realistic_costs": _realistic_costs_passes,
        "family_evidence": _family_evidence_passes,
        "signal_frequency": _positive,
        "turnover": _nonnegative,
    },
    "development": {
        "chronological": _true,
        "data_integrity": _data_integrity_passes,
        "semantic_parity": _semantic_parity_passes,
        "realistic_costs": _realistic_costs_passes,
        "family_evidence": _family_evidence_passes,
        "cost_adjusted_return": _return_passes,
        "fees": _nonnegative,
        "slippage": _nonnegative,
        "funding": _nonnegative,
        "regime_breakdown": _regime_breakdown_passes,
        "parameter_stability": _parameter_stability_passes,
        "sample_evidence": _sample_evidence_passes,
        "cross_symbol_stability": _cross_symbol_stability_passes,
        "universe_evidence": _mapping_passes,
    },
    "robustness": {
        "data_integrity": _data_integrity_passes,
        "semantic_parity": _semantic_parity_passes,
        "realistic_costs": _realistic_costs_passes,
        "family_evidence": _family_evidence_passes,
        "walk_forward": _walk_forward_passes,
        "purged": _true,
        "embargo": _positive,
        "cost_stress": _mapping_passes,
        "delay_stress": _mapping_passes,
        "adverse_fill_stress": _mapping_passes,
        "missing_data_stress": _mapping_passes,
        "funding_stress": _mapping_passes,
        "monte_carlo_trade_order": _monte_carlo_passes,
        "bootstrap_confidence": _bootstrap_passes,
        "probability_backtest_overfitting": _pbo_passes,
        "deflated_sharpe": _deflated_sharpe_passes,
        "statistical_procedures": _statistical_procedures_pass,
        "drawdown_stability": _drawdown_passes,
        "null_results": _mapping_passes,
        "negative_control_results": _negative_control_results_pass,
    },
    "forward": {
        "data_integrity": _data_integrity_passes,
        "semantic_parity": _semantic_parity_passes,
        "realistic_costs": _realistic_costs_passes,
        "family_evidence": _family_evidence_passes,
        "production_equivalent": _mapping_passes,
        "exact_strategy_identity": _mapping_passes,
        "exact_artefact_hash": _mapping_passes,
        "exact_engine_hash": _mapping_passes,
        "exact_cost_model": _mapping_passes,
        "drift_checks": _mapping_passes,
        "duration": _positive,
        "evidence_units": _positive,
        "sample_evidence": _sample_evidence_passes,
        "forward_duration": _forward_duration_passes,
    },
    "protected": {
        "chronological": _true,
        "data_integrity": _data_integrity_passes,
        "semantic_parity": _semantic_parity_passes,
        "realistic_costs": _realistic_costs_passes,
        "family_evidence": _family_evidence_passes,
        "cost_adjusted_return": _return_passes,
        "sample_evidence": _sample_evidence_passes,
        "drawdown_stability": _drawdown_passes,
    },
}


_OPTIONAL_EVIDENCE_VALIDATORS: dict[str, dict[str, EvidenceValidator]] = {
    "development": {"portfolio_overlap": _portfolio_overlap_passes},
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
        *,
        dataset_resolver: CanonicalDatasetResolver | None = None,
        feature_manifest_id: str | None = None,
        cost_model_id: str | None = None,
        parameter_set_id: str | None = None,
    ) -> None:
        self.repository = SqlHoldoutRepository(engine)
        self.evaluator = evaluator
        self.dataset_resolver = dataset_resolver
        self.dataset_identities = {
            "feature_manifest_hash": feature_manifest_id,
            "cost_model_hash": cost_model_id,
            "parameter_set_hash": parameter_set_id,
        }

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
        protected_context: Mapping[str, Any] | None = None
        if self.dataset_resolver is not None:
            if any(value is None for value in self.dataset_identities.values()):
                raise EvaluationContractError(
                    "protected holdout resolution requires canonical dataset identities"
                )
            protected_context = self.dataset_resolver.resolve_context(
                snapshot_ids=(dataset_snapshot_id,),
                feature_manifest_id=str(self.dataset_identities["feature_manifest_hash"]),
                cost_model_id=str(self.dataset_identities["cost_model_hash"]),
                parameter_set_id=str(self.dataset_identities["parameter_set_hash"]),
                allowed_roles=frozenset({"protected_holdout"}),
            )
        callback_payload = {
            "claim_id": claim_id,
            "strategy_version_id": strategy_version_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "cohort_id": cohort_id,
            "source_hashes": dict(source_hashes),
        }
        if protected_context is not None:
            # This object is scoped to the callback and is never returned by
            # the worker. Adaptive research only receives the sealed summary.
            callback_payload["protected_context"] = protected_context
        accepted, result = self.evaluator(callback_payload)
        if not isinstance(accepted, bool) or not isinstance(result, Mapping):
            raise EvaluationContractError("protected evaluator returned an invalid sealed outcome")
        sealed = result.get("sealed_result")
        if not isinstance(sealed, Mapping) or not sealed:
            raise EvaluationContractError("protected evaluator returned no sealed outcome")
        if sealed.get("passed") is not accepted:
            raise EvaluationContractError("protected sealed outcome does not match its decision")
        outcome_id = self.repository.record_outcome(
            claim_id=claim_id,
            evaluated_at=evaluated_at,
            accepted=accepted,
            outcome=sealed,
        )
        # Keep protected metrics and dataset identities in the sealed database
        # row. The adaptive caller receives only a non-row-level summary.
        return claim_id, outcome_id, accepted, {"passed": accepted, "outcome_id": outcome_id}


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
        family, horizon, evidence_type = _evidence_dimensions(candidate)
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
            "product_id": definition.product,
            "strategy_family": family,
            "strategy_horizon": horizon,
            "evidence_type": evidence_type,
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
            "negative_controls": list(self._negative_controls(candidate.thesis_id)),
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
            "artefact_hash": request.artefact_hash,
            "artefact_created_at": request.artefact_created_at,
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

    def _execute_stage(
        self,
        stage: str,
        candidate,
        context: Mapping[str, Any],
    ) -> _ExecutionAttempt:
        if stage not in {"screening", "development", "robustness", "forward"}:
            return _ExecutionAttempt(None)
        context_error = self.provider_context.get("provider_context_error")
        if isinstance(context_error, str) and context_error:
            return _ExecutionAttempt(None, context_error)
        try:
            return _ExecutionAttempt(
                self.executors.execute(candidate, {**self.provider_context, **context})
            )
        except (ExecutorError, KeyError, ValueError, TypeError) as exc:
            return _ExecutionAttempt(None, f"{type(exc).__name__}: {exc}")

    def _screening_stage(
        self,
        candidate,
        context: Mapping[str, Any],
        *,
        identity: str,
        family: str,
        horizon: str,
        evidence_type: str,
        execution: ExecutionResult,
    ) -> tuple[
        dict[str, Any],
        bool,
        str | None,
        Mapping[str, Any] | None,
        Mapping[str, float],
    ]:
        definition = candidate.definition
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
            token in json_value(definition.feature_graph, field="feature_graph").__repr__().lower()
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
                "screening",
                measured,
                self._negative_controls(candidate.thesis_id),
                product_id=definition.product,
                family=family,
                horizon=horizon,
                evidence_type=evidence_type,
            )
        )
        return (
            evidence,
            accepted,
            None if accepted else "screening_measured_evidence_missing",
            execution.receipt,
            execution.metrics,
        )

    def _protected_stage(
        self,
        candidate,
        context: Mapping[str, Any],
        *,
        identity: str,
        protected_snapshot_id: str | None,
    ) -> tuple[
        dict[str, Any],
        bool,
        str | None,
        Mapping[str, Any] | None,
        Mapping[str, float],
    ]:
        definition = candidate.definition
        source_hashes = {
            "code_hash": str(context["code_hash"]),
            "feature_manifest_id": str(context["feature_manifest_id"]),
            "cost_model_id": str(context["cost_model_id"]),
        }
        claim = self._claim_protected_holdout(
            definition,
            context,
            source_hashes=source_hashes,
            protected_snapshot_id=protected_snapshot_id,
        )
        if claim.error is not None:
            return (
                {
                    "identity": identity,
                    "context": dict(context),
                    "holdout_claim_error": claim.error,
                },
                False,
                "protected_holdout_claim_failed",
                None,
                {},
            )
        claims = self.protected.claim_for(definition.strategy_version_id)
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
            "holdout_outcome_id": claim.outcome_id,
            "sealed_outcome": {
                "passed": claim.accepted,
                "outcome_id": claim.outcome_id,
            },
            "context": dict(context),
        }
        accepted = bool(claims) and claim.accepted if self.protected_worker is not None else False
        return (
            evidence,
            accepted,
            None if accepted else "protected_holdout_outcome_missing",
            None,
            {},
        )

    def _claim_protected_holdout(
        self,
        definition,
        context: Mapping[str, Any],
        *,
        source_hashes: Mapping[str, str],
        protected_snapshot_id: str | None,
    ) -> _ProtectedClaim:
        try:
            if self.protected_worker is not None:
                _claim_id, outcome_id, accepted, _sealed_outcome = (
                    self.protected_worker.claim_and_evaluate(
                        strategy_version_id=definition.strategy_version_id,
                        dataset_snapshot_id=str(protected_snapshot_id or ""),
                        cohort_id=f"protected:{context['candidate_id']}",
                        source_hashes=source_hashes,
                        evaluated_at=str(context["evaluated_at"]),
                    )
                )
                return _ProtectedClaim(outcome_id, accepted)
            SqlHoldoutRepository(self.store.engine).claim(
                strategy_version_id=definition.strategy_version_id,
                data_snapshot_id=str(protected_snapshot_id or ""),
                cohort_id=f"protected:{context['candidate_id']}",
                source_hashes=source_hashes,
                claimed_at=str(context["evaluated_at"]),
            )
            return _ProtectedClaim(None, False)
        except Exception as exc:
            return _ProtectedClaim(None, False, f"{type(exc).__name__}: {exc}")

    def _authoritative_runs(
        self, context: Mapping[str, Any], execution: ExecutionResult | None
    ) -> tuple[list[dict[str, Any]], set[str]]:
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
        return authoritative_runs, run_names

    @staticmethod
    def _merge_authoritative_runs(
        authoritative_runs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, float], Mapping[str, Any] | None]:
        run_measured: dict[str, Any] = {}
        metrics: dict[str, float] = {}
        receipt: Mapping[str, Any] | None = None
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
            raw_receipt = payload.get("receipt")
            if receipt is None and isinstance(raw_receipt, Mapping) and raw_receipt:
                receipt = raw_receipt
        return run_measured, metrics, receipt

    def _adaptive_stage(
        self,
        stage: str,
        candidate,
        context: Mapping[str, Any],
        *,
        identity: str,
        family: str,
        horizon: str,
        evidence_type: str,
        execution: ExecutionResult | None,
    ) -> tuple[
        dict[str, Any],
        bool,
        str | None,
        Mapping[str, Any] | None,
        Mapping[str, float],
    ]:
        required_runs = {
            "development": ("bar_portfolio",),
            "robustness": ("bar_portfolio", "event_replay"),
            "forward": ("forward_paper",),
        }
        authoritative_runs, run_names = self._authoritative_runs(context, execution)
        required_runs_for_stage = required_runs[stage]
        missing_runs = (
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
        if missing_runs:
            return evidence, False, "missing_authoritative_run", None, {}
        run_measured, metrics, receipt = self._merge_authoritative_runs(authoritative_runs)
        evidence.update(run_measured)
        required_fields = _REQUIRED_STAGE_FIELDS[stage]
        requires_objective = _requires_product_objective(candidate)
        if requires_objective and stage in {"development", "robustness", "forward"}:
            required_fields = (*required_fields, "objective_excess_fraction")
        controls = self._negative_controls(candidate.thesis_id)
        policy_fields = _STAGE_EVIDENCE_VALIDATORS[stage]
        product_id = candidate.definition.product if requires_objective else None
        evidence_status = self.evidence_policy.statuses(
            stage,
            evidence,
            controls,
            product_id=product_id,
            family=family,
            horizon=horizon,
            evidence_type=evidence_type,
        )
        missing = [
            field
            for field in required_fields
            if field not in policy_fields
            and field != "objective_excess_fraction"
            or evidence_status.get(field)
            not in {EvidenceStatus.PASS, EvidenceStatus.NOT_APPLICABLE}
        ]
        accepted = not missing and self.evidence_policy.accepts(
            stage,
            evidence,
            controls,
            product_id=product_id,
            family=family,
            horizon=horizon,
            evidence_type=evidence_type,
        )
        evidence["missing_evidence"] = missing
        evidence["evidence_status"] = {
            field: status.value for field, status in evidence_status.items()
        }
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
        family, horizon, evidence_type = _evidence_dimensions(candidate)
        identity = canonical_hash(
            {
                "definition_hash": definition.definition_hash,
                "context": dict(context),
            }
        )
        attempt = self._execute_stage(stage, candidate, context)
        if attempt.error is not None:
            return (
                {"identity": identity, "context": dict(context), "executor_error": attempt.error},
                False,
                "candidate_execution_failed",
                None,
                {},
            )
        execution = attempt.result
        if stage == "screening":
            assert execution is not None
            return self._screening_stage(
                candidate,
                context,
                identity=identity,
                family=family,
                horizon=horizon,
                evidence_type=evidence_type,
                execution=execution,
            )

        if stage == "protected":
            return self._protected_stage(
                candidate,
                context,
                identity=identity,
                protected_snapshot_id=protected_snapshot_id,
            )

        return self._adaptive_stage(
            stage,
            candidate,
            context,
            identity=identity,
            family=family,
            horizon=horizon,
            evidence_type=evidence_type,
            execution=execution,
        )


def _requires_product_objective(candidate: Any) -> bool:
    product = str(candidate.definition.product)
    if product not in {"btc_accumulation", "active_income"}:
        return False
    metadata = candidate.definition.metadata
    if metadata.get("diagnostic") is True or metadata.get("promotable") is False:
        return False
    return metadata.get("promotable") is True or metadata.get("executable_registry_entry") is True


def _evidence_dimensions(candidate: Any) -> tuple[str, str, str]:
    definition = candidate.definition
    validation = definition.validation_policy
    evidence_type = str(validation.get("evidence_type") or "")
    horizon = str(
        validation.get("horizon")
        or definition.position_model.get("horizon")
        or definition.position_model.get("horizon_bars")
        or "*"
    )
    return str(definition.family), horizon, evidence_type
