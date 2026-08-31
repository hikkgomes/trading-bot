"""Strict, result-free contracts for work submitted to the platform queue.

Queue payloads are commands.  They identify immutable inputs and the code that
submitted them.  Workers load outcomes from canonical stores after claiming a
command.  This module is deliberately dependency-light so every service can
validate the boundary without importing research or execution code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from src.domain._codec import canonical_hash, json_value, timestamp


class JobSchemaError(ValueError):
    """A queue command is malformed or contains a result injection."""


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise JobSchemaError(f"{field} must be a non-empty string")
    return value.strip()


def _hash_id(payload: Mapping[str, Any], field: str) -> str:
    value = _required_text(payload, field)
    if not value.startswith("sha256:") or len(value) != 71:
        raise JobSchemaError(f"{field} must be a sha256: identity")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise JobSchemaError(f"{field} must be a sha256: identity") from exc
    return value


def _hash_ids(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = payload.get(field)
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise JobSchemaError(f"{field} must be a non-empty list")
    result = tuple(_hash_id({field: value}, field) for value in values)
    if len(set(result)) != len(result):
        raise JobSchemaError(f"{field} must not contain duplicates")
    return result


def _strict_fields(payload: Mapping[str, Any], allowed: frozenset[str], *, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise JobSchemaError(f"{name} must be an object")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise JobSchemaError(f"{name} contains unknown fields: {', '.join(unknown)}")


RESULT_FIELDS = frozenset(
    {
        "accepted",
        "backtest",
        "evidence",
        "events",
        "fills",
        "forward_result",
        "holdout_result",
        "limits",
        "metrics",
        "orders",
        "positions",
        "reason_code",
        "returns",
        "risk_decision",
        "target_steps",
        "targets",
        "validation",
    }
)


@dataclass(frozen=True)
class ProducerBinding:
    producer_identity: str
    content_hash: str

    ALLOWED: ClassVar[frozenset[str]] = frozenset({"producer_identity", "content_hash"})

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProducerBinding:
        _strict_fields(payload, cls.ALLOWED, name="producer binding")
        identity = _required_text(payload, "producer_identity")
        content_hash = _hash_id(payload, "content_hash")
        return cls(identity, content_hash)


@dataclass(frozen=True)
class ResearchJobRequest:
    """Canonical research command with immutable input identities only."""

    candidate_id: str
    dataset_snapshot_ids: tuple[str, ...]
    feature_manifest_id: str
    cost_model_id: str
    parameter_set_id: str
    evaluator_version: str
    requested_stage: str
    evaluated_at: str
    producer_identity: str
    content_hash: str
    dataset_roles: dict[str, str] | None = None
    artefact_hash: str | None = None
    artefact_created_at: str | None = None

    ALLOWED: ClassVar[frozenset[str]] = frozenset(
        {
            "candidate_id",
            "dataset_snapshot_ids",
            "feature_manifest_id",
            "cost_model_id",
            "parameter_set_id",
            "evaluator_version",
            "requested_stage",
            "evaluated_at",
            "producer_identity",
            "content_hash",
            "dataset_roles",
            "artefact_hash",
            "artefact_created_at",
        }
    )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, require_dataset_roles: bool = False
    ) -> ResearchJobRequest:
        _strict_fields(payload, cls.ALLOWED, name="research job")
        _reject_result_fields(payload)
        (
            candidate_id,
            snapshots,
            feature_manifest_id,
            cost_model_id,
            parameter_set_id,
        ) = _research_identity_fields(payload)
        requested_stage = _research_stage(payload)
        evaluated_at = timestamp(_required_text(payload, "evaluated_at"), field="evaluated_at")
        producer = _required_text(payload, "producer_identity")
        content_hash = _hash_id(payload, "content_hash")
        dataset_roles = _parse_dataset_roles(payload.get("dataset_roles"), snapshots)
        artefact_hash, artefact_created_at = _research_artefact_fields(payload)
        _validate_research_stage(
            requested_stage=requested_stage,
            evaluated_at=evaluated_at,
            artefact_hash=artefact_hash,
            artefact_created_at=artefact_created_at,
            snapshots=snapshots,
            dataset_roles=dataset_roles,
            require_dataset_roles=require_dataset_roles,
        )
        unsigned = dict(payload)
        unsigned.pop("content_hash", None)
        expected = canonical_hash(unsigned)
        if content_hash != expected:
            raise JobSchemaError("research job content_hash does not match its payload")
        return cls(
            candidate_id,
            snapshots,
            feature_manifest_id,
            cost_model_id,
            parameter_set_id,
            _required_text(payload, "evaluator_version"),
            requested_stage,
            evaluated_at,
            producer,
            content_hash,
            dataset_roles,
            artefact_hash,
            artefact_created_at,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset_snapshot_ids": list(self.dataset_snapshot_ids),
            "feature_manifest_id": self.feature_manifest_id,
            "cost_model_id": self.cost_model_id,
            "parameter_set_id": self.parameter_set_id,
            "evaluator_version": self.evaluator_version,
            "requested_stage": self.requested_stage,
            "evaluated_at": self.evaluated_at,
            "producer_identity": self.producer_identity,
            "content_hash": self.content_hash,
            **({"dataset_roles": dict(self.dataset_roles)} if self.dataset_roles else {}),
            **({"artefact_hash": self.artefact_hash} if self.artefact_hash else {}),
            **(
                {"artefact_created_at": self.artefact_created_at}
                if self.artefact_created_at
                else {}
            ),
        }


def _parse_dataset_roles(value: object, snapshots: tuple[str, ...]) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(snapshots):
        raise JobSchemaError("dataset_roles must map every dataset snapshot identity to one role")
    allowed = {
        "screening",
        "development",
        "robustness",
        "protected_holdout",
        "forward_observation",
        "unspecified",
    }
    result = {str(key): str(role) for key, role in value.items()}
    if any(role not in allowed for role in result.values()):
        raise JobSchemaError("dataset_roles contains an unsupported role")
    return result


def _reject_result_fields(payload: Mapping[str, Any]) -> None:
    forbidden = RESULT_FIELDS & set(payload)
    if forbidden:
        fields = sorted(forbidden)
        raise JobSchemaError("research jobs cannot contain results: " + ", ".join(fields))


def _research_identity_fields(
    payload: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], str, str, str]:
    return (
        _hash_id(payload, "candidate_id"),
        _hash_ids(payload, "dataset_snapshot_ids"),
        _hash_id(payload, "feature_manifest_id"),
        _hash_id(payload, "cost_model_id"),
        _hash_id(payload, "parameter_set_id"),
    )


def _research_stage(payload: Mapping[str, Any]) -> str:
    requested_stage = _required_text(payload, "requested_stage")
    if requested_stage not in {
        "screening",
        "development",
        "robustness",
        "protected",
        "forward",
    }:
        raise JobSchemaError("requested_stage is unsupported")
    return requested_stage


def _research_artefact_fields(
    payload: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    artefact_hash = (
        _hash_id(payload, "artefact_hash") if payload.get("artefact_hash") is not None else None
    )
    artefact_created_at = (
        timestamp(_required_text(payload, "artefact_created_at"), field="artefact_created_at")
        if payload.get("artefact_created_at") is not None
        else None
    )
    return artefact_hash, artefact_created_at


def _validate_research_stage(
    *,
    requested_stage: str,
    evaluated_at: str,
    artefact_hash: str | None,
    artefact_created_at: str | None,
    snapshots: tuple[str, ...],
    dataset_roles: dict[str, str] | None,
    require_dataset_roles: bool,
) -> None:
    _validate_forward_fields(
        requested_stage,
        evaluated_at=evaluated_at,
        artefact_hash=artefact_hash,
        artefact_created_at=artefact_created_at,
    )
    if require_dataset_roles and dataset_roles is None:
        raise JobSchemaError("research jobs require explicit dataset roles")
    if dataset_roles is not None:
        _validate_dataset_roles(
            requested_stage,
            snapshots=snapshots,
            dataset_roles=dataset_roles,
        )


def _validate_forward_fields(
    requested_stage: str,
    *,
    evaluated_at: str,
    artefact_hash: str | None,
    artefact_created_at: str | None,
) -> None:
    if requested_stage != "forward":
        return
    if artefact_hash is None or artefact_created_at is None:
        raise JobSchemaError(
            "forward research jobs require the exact artefact hash and immutable creation time"
        )
    if artefact_created_at >= evaluated_at:
        raise JobSchemaError("forward artefact creation must precede evaluation time")


def _validate_dataset_roles(
    requested_stage: str,
    *,
    snapshots: tuple[str, ...],
    dataset_roles: Mapping[str, str],
) -> None:
    expected_role = {
        "screening": "screening",
        "development": "development",
        "robustness": "robustness",
        "protected": "protected_holdout",
        "forward": "forward_observation",
    }[requested_stage]
    if sum(role == expected_role for role in dataset_roles.values()) != 1:
        raise JobSchemaError(
            f"research jobs require exactly one {expected_role} snapshot for {requested_stage}"
        )
    if requested_stage == "protected" and (
        len(snapshots) != 1 or dataset_roles.get(snapshots[0]) != "protected_holdout"
    ):
        raise JobSchemaError(
            "protected research jobs may contain only the protected_holdout snapshot"
        )
    if requested_stage != "protected" and "protected_holdout" in dataset_roles.values():
        raise JobSchemaError(
            "adaptive research jobs must not contain protected holdout snapshot identities"
        )
    if requested_stage != "forward" and "forward_observation" in dataset_roles.values():
        raise JobSchemaError(
            "adaptive research jobs must not contain forward observation snapshot identities"
        )


@dataclass(frozen=True)
class RiskAssessmentRequest:
    """Risk command containing snapshot and policy identities, never values."""

    assessment_id: str
    product_id: str
    event_id: str
    target_position_snapshot_id: str
    account_snapshot_id: str
    positions_snapshot_id: str
    balances_snapshot_id: str
    market_data_snapshot_id: str
    risk_policy_ids: tuple[str, ...]
    evaluated_at: str
    producer_identity: str
    content_hash: str

    ALLOWED: ClassVar[frozenset[str]] = frozenset(
        {
            "assessment_id",
            "product_id",
            "event_id",
            "target_position_snapshot_id",
            "account_snapshot_id",
            "positions_snapshot_id",
            "balances_snapshot_id",
            "market_data_snapshot_id",
            "risk_policy_ids",
            "evaluated_at",
            "producer_identity",
            "content_hash",
        }
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RiskAssessmentRequest:
        _strict_fields(payload, cls.ALLOWED, name="risk job")
        forbidden = RESULT_FIELDS & set(payload)
        if forbidden:
            raise JobSchemaError("risk jobs cannot contain decisions or values")
        text_fields = ("assessment_id", "product_id", "event_id", "producer_identity")
        values = {field: _required_text(payload, field) for field in text_fields}
        snapshot_fields = (
            "target_position_snapshot_id",
            "account_snapshot_id",
            "positions_snapshot_id",
            "balances_snapshot_id",
            "market_data_snapshot_id",
        )
        snapshots = {field: _hash_id(payload, field) for field in snapshot_fields}
        raw_policies = payload.get("risk_policy_ids")
        if (
            not isinstance(raw_policies, Sequence)
            or isinstance(raw_policies, str)
            or not raw_policies
        ):
            raise JobSchemaError("risk_policy_ids must be a non-empty list")
        policies = tuple(_required_text({"value": value}, "value") for value in raw_policies)
        if len(set(policies)) != len(policies):
            raise JobSchemaError("risk_policy_ids must not contain duplicates")
        evaluated_at = timestamp(_required_text(payload, "evaluated_at"), field="evaluated_at")
        content_hash = _hash_id(payload, "content_hash")
        unsigned = dict(payload)
        unsigned.pop("content_hash", None)
        if content_hash != canonical_hash(unsigned):
            raise JobSchemaError("risk job content_hash does not match its payload")
        return cls(
            **values,
            **snapshots,
            risk_policy_ids=policies,
            evaluated_at=evaluated_at,
            content_hash=content_hash,
        )


def _strict_command(
    payload: Mapping[str, Any],
    *,
    name: str,
    allowed: frozenset[str],
    required: frozenset[str],
    hash_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    _strict_fields(payload, allowed, name=name)
    missing = sorted(required - set(payload))
    if missing:
        raise JobSchemaError(f"{name} is missing: {', '.join(missing)}")
    for field in hash_fields:
        _hash_id(payload, field)
    for field in required - hash_fields - {"horizon_seconds", "feature_ids"}:
        _required_text(payload, field)
    if "horizon_seconds" in payload:
        value = payload["horizon_seconds"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise JobSchemaError("horizon_seconds must be a positive integer")
    evaluated_at = timestamp(_required_text(payload, "evaluated_at"), field="evaluated_at")
    content_hash = _hash_id(payload, "content_hash")
    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    if content_hash != canonical_hash(unsigned):
        raise JobSchemaError(f"{name} content_hash does not match its payload")
    clean = dict(payload)
    clean["evaluated_at"] = evaluated_at
    return clean


def validate_job_payload(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a typed command and return its canonical JSON payload.

    Legacy low-level jobs remain available for paper fixtures, but all
    authority-bearing research and risk jobs use these strict contracts.
    """

    validators = {
        "evaluate_candidate": _validate_evaluate_candidate,
        "risk_assessment": _validate_risk_assessment,
        "strategy_evaluation": _validate_strategy_evaluation,
        "portfolio_target_build": _validate_portfolio_target_build,
        "emergency_reduction": _validate_emergency_reduction,
    }
    validator = validators.get(name)
    return (
        validator(payload)
        if validator is not None
        else json_value(dict(payload), field=f"{name} payload")
    )


def _validate_evaluate_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    return ResearchJobRequest.from_mapping(payload, require_dataset_roles=True).to_payload()


def _validate_risk_assessment(payload: Mapping[str, Any]) -> dict[str, Any]:
    request = RiskAssessmentRequest.from_mapping(payload)
    return {
        "assessment_id": request.assessment_id,
        "product_id": request.product_id,
        "event_id": request.event_id,
        "target_position_snapshot_id": request.target_position_snapshot_id,
        "account_snapshot_id": request.account_snapshot_id,
        "positions_snapshot_id": request.positions_snapshot_id,
        "balances_snapshot_id": request.balances_snapshot_id,
        "market_data_snapshot_id": request.market_data_snapshot_id,
        "risk_policy_ids": list(request.risk_policy_ids),
        "evaluated_at": request.evaluated_at,
        "producer_identity": request.producer_identity,
        "content_hash": request.content_hash,
    }


def _validate_strategy_evaluation(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = _strict_command(
        payload,
        name="strategy evaluation",
        allowed=frozenset(
            {
                "event_id",
                "product_id",
                "instrument_id",
                "assignment_id",
                "feature_ids",
                "feature_set_version",
                "market_data_snapshot_id",
                "input_reference_id",
                "evaluated_at",
                "horizon_seconds",
                "producer_identity",
                "content_hash",
            }
        ),
        required=frozenset(
            {
                "event_id",
                "product_id",
                "instrument_id",
                "assignment_id",
                "feature_ids",
                "feature_set_version",
                "evaluated_at",
                "producer_identity",
                "content_hash",
            }
        ),
        hash_fields=frozenset({"event_id", "assignment_id", "content_hash"}),
    )
    _hash_ids(clean, "feature_ids")
    if "market_data_snapshot_id" in clean:
        _hash_id(clean, "market_data_snapshot_id")
    if "input_reference_id" in clean:
        _hash_id(clean, "input_reference_id")
    return clean


def _validate_portfolio_target_build(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _strict_command(
        payload,
        name="portfolio target build",
        allowed=frozenset(
            {
                "event_id",
                "product_id",
                "forecast_id",
                "market_data_snapshot_id",
                "evaluated_at",
                "producer_identity",
                "content_hash",
            }
        ),
        required=frozenset(
            {
                "event_id",
                "product_id",
                "forecast_id",
                "evaluated_at",
                "producer_identity",
                "content_hash",
            }
        ),
        hash_fields=frozenset({"event_id", "forecast_id", "content_hash"}),
    )


def _validate_emergency_reduction(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = frozenset(
        {
            "product_id",
            "portfolio_id",
            "instrument_id",
            "stop_id",
            "position_quantity",
            "reason_code",
            "evaluated_at",
            "producer_identity",
            "content_hash",
        }
    )
    _strict_fields(payload, allowed, name="emergency reduction")
    for field in (
        "product_id",
        "portfolio_id",
        "instrument_id",
        "stop_id",
        "reason_code",
        "producer_identity",
    ):
        _required_text(payload, field)
    position_quantity = payload.get("position_quantity")
    if isinstance(position_quantity, bool):
        raise JobSchemaError("position_quantity must be numeric")
    try:
        numeric_quantity = float(position_quantity)
    except (TypeError, ValueError) as exc:
        raise JobSchemaError("position_quantity must be numeric") from exc
    if not math.isfinite(numeric_quantity) or abs(numeric_quantity) <= 0:
        raise JobSchemaError("position_quantity must be finite and non-zero")
    evaluated_at = timestamp(_required_text(payload, "evaluated_at"), field="evaluated_at")
    content_hash = _hash_id(payload, "content_hash")
    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    if content_hash != canonical_hash(unsigned):
        raise JobSchemaError("emergency reduction content_hash does not match its payload")
    return {**dict(payload), "position_quantity": numeric_quantity, "evaluated_at": evaluated_at}


def build_content_hash(payload: Mapping[str, Any]) -> str:
    """Build the producer-independent hash used by strict commands."""

    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    return canonical_hash(unsigned)
