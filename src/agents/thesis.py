"""Validation of untrusted OpenClaw economic-thesis payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.domain._codec import non_empty, timestamp
from src.domain.strategies import MechanismCategory, ResearchThesis


class AgentThesisError(ValueError):
    """An OpenClaw thesis is not a complete safe economic contract."""


THESIS_FIELDS = frozenset(
    {
        "mechanism_category",
        "market_rationale",
        "expected_causal_chain",
        "expected_direction",
        "expected_horizon",
        "required_data",
        "permitted_features",
        "instrument_universe",
        "generalisation_scope",
        "failure_regimes",
        "falsification_tests",
        "negative_controls",
        "execution_capacity_assumptions",
        "parent_thesis_ids",
        "cumulative_trial_budget",
    }
)


def _strings(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise AgentThesisError(f"OpenClaw thesis {field} must be a non-empty list")
    return tuple(non_empty(str(item), field=field) for item in value)


def _mapping(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise AgentThesisError(f"OpenClaw thesis {field} must be an object")
    return dict(value)


def parse_openclaw_thesis(
    payload: Mapping[str, Any], *, product_id: str, created_at: str
) -> ResearchThesis:
    if not isinstance(payload, Mapping):
        raise AgentThesisError("OpenClaw thesis must be an object")
    unknown = set(payload) - THESIS_FIELDS
    if unknown:
        raise AgentThesisError(
            "OpenClaw thesis contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    scope = _mapping(payload, "generalisation_scope")
    if str(scope.get("product") or "") != product_id:
        raise AgentThesisError("OpenClaw thesis product scope does not match the proposal")
    raw_budget = payload.get("cumulative_trial_budget")
    if isinstance(raw_budget, bool) or not isinstance(raw_budget, int) or not 1 <= raw_budget <= 50:
        raise AgentThesisError("OpenClaw thesis trial budget must be an integer from 1 to 50")
    parents = _strings(payload, "parent_thesis_ids") if payload.get("parent_thesis_ids") else ()
    if any(not item.startswith("sha256:") or len(item) != 71 for item in parents):
        raise AgentThesisError("OpenClaw thesis parents must be SHA-256 identities")
    return ResearchThesis(
        mechanism_category=MechanismCategory(str(payload.get("mechanism_category") or "")),
        market_rationale=non_empty(
            str(payload.get("market_rationale") or ""), field="market_rationale"
        ),
        expected_causal_chain=_strings(payload, "expected_causal_chain"),
        expected_direction=non_empty(
            str(payload.get("expected_direction") or ""), field="expected_direction"
        ),
        expected_horizon=non_empty(
            str(payload.get("expected_horizon") or ""), field="expected_horizon"
        ),
        required_data=_strings(payload, "required_data"),
        permitted_features=_strings(payload, "permitted_features"),
        instrument_universe=_strings(payload, "instrument_universe"),
        generalisation_scope=scope,
        failure_regimes=_strings(payload, "failure_regimes"),
        falsification_tests=_strings(payload, "falsification_tests"),
        negative_controls=_strings(payload, "negative_controls"),
        execution_capacity_assumptions=_mapping(payload, "execution_capacity_assumptions"),
        parent_thesis_ids=parents,
        cumulative_trial_budget=raw_budget,
        created_at=timestamp(created_at, field="created_at"),
        creator_identity="openclaw/untrusted-thesis/v1",
    )
