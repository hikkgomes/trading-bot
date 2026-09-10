"""Outcome vocabulary shared by execution, evaluation and operator reports."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class OutcomeCategory(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    BLOCKED_DATA = "blocked_data"
    BLOCKED_INFRASTRUCTURE = "blocked_infrastructure"
    UNSUPPORTED = "unsupported"


class TrialDisposition(StrEnum):
    PENDING = "pending"
    FINALIST = "finalist"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    BLOCKED_DATA = "blocked_data"
    INFRASTRUCTURE_EXHAUSTED = "infrastructure_exhausted"
    UNSUPPORTED = "unsupported"


def terminal_trial_disposition(
    state: str, metadata: Mapping[str, Any] | None = None
) -> TrialDisposition | None:
    terminal = {
        "infrastructure_exhausted": TrialDisposition.INFRASTRUCTURE_EXHAUSTED,
        "unsupported": TrialDisposition.UNSUPPORTED,
    }.get(state)
    deferred = (metadata or {}).get("evaluation_deferred", {}).get("outcome_category")
    if terminal is None and deferred in {"blocked_data", "inconclusive", "unsupported"}:
        return TrialDisposition(deferred)
    return terminal


def forward_outcome(accepted: bool, reason: str | None) -> OutcomeCategory:
    if accepted:
        return OutcomeCategory.ACCEPTED
    if reason in {
        "forward_data_gaps",
        "forward_data_uptime_insufficient",
        "forward_execution_reference_missing",
        "forward_window_missing",
        "forward_window_superseded",
        "forward_window_engine_changed",
        "forward_window_corrupt",
        "research_candidate_binding_invalid",
        "research_inference_invalid",
        "research_interval_exposed",
    }:
        return OutcomeCategory.BLOCKED_DATA
    if reason in {
        "forward_evidence_duration_insufficient",
        "forward_decisions_insufficient",
        "forward_trading_days_insufficient",
        "forward_cycles_insufficient",
        "forward_completed_episodes_insufficient",
        "forward_effective_trades_insufficient",
        "forward_fill_rate_not_applicable",
        "research_policy_context_unavailable",
        "research_cohort_unavailable",
        "research_policy_unapproved",
        "research_inference_inconclusive",
    }:
        return OutcomeCategory.INCONCLUSIVE
    return OutcomeCategory.REJECTED
