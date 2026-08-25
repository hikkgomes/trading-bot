"""Strict, result-free contracts for work submitted to the platform queue.

Queue payloads are commands.  They identify immutable inputs and the code that
submitted them.  Workers load outcomes from canonical stores after claiming a
command.  This module is deliberately dependency-light so every service can
validate the boundary without importing research or execution code.
"""

from __future__ import annotations

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
        }
    )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, require_dataset_roles: bool = False
    ) -> ResearchJobRequest:
        _strict_fields(payload, cls.ALLOWED, name="research job")
        if RESULT_FIELDS & set(payload):
            fields = sorted(RESULT_FIELDS & set(payload))
            raise JobSchemaError("research jobs cannot contain results: " + ", ".join(fields))
        candidate_id = _hash_id(payload, "candidate_id")
        snapshots = _hash_ids(payload, "dataset_snapshot_ids")
        feature_manifest_id = _hash_id(payload, "feature_manifest_id")
        cost_model_id = _hash_id(payload, "cost_model_id")
        parameter_set_id = _hash_id(payload, "parameter_set_id")
        requested_stage = _required_text(payload, "requested_stage")
        if requested_stage not in {
            "screening",
            "development",
            "robustness",
            "protected",
            "forward",
        }:
            raise JobSchemaError("requested_stage is unsupported")
        evaluated_at = timestamp(_required_text(payload, "evaluated_at"), field="evaluated_at")
        producer = _required_text(payload, "producer_identity")
        content_hash = _hash_id(payload, "content_hash")
        dataset_roles = _parse_dataset_roles(payload.get("dataset_roles"), snapshots)
        if require_dataset_roles and dataset_roles is None:
            raise JobSchemaError("research jobs require explicit dataset roles")
        if dataset_roles is not None:
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

    if name == "evaluate_candidate":
        research_request = ResearchJobRequest.from_mapping(payload, require_dataset_roles=True)
        return research_request.to_payload()
    if name == "risk_assessment":
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
    if name == "strategy_evaluation":
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
    if name == "portfolio_target_build":
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
    return json_value(dict(payload), field=f"{name} payload")


def build_content_hash(payload: Mapping[str, Any]) -> str:
    """Build the producer-independent hash used by strict commands."""

    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    return canonical_hash(unsigned)
