"""Deterministic strategy lifecycle and promotion policy."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import promotion_event
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.services.scheduler import DatabaseJobQueue


class LifecycleState(StrEnum):
    REGISTERED = "registered"
    DEVELOPMENT = "development"
    FORWARD_PAPER = "forward_paper"
    LIVE_CANARY = "live_canary"
    LIVE = "live"
    SUSPENDED = "suspended"
    RETIRED = "retired"


@dataclass(frozen=True)
class PromotionPolicy:
    automatic_paper_promotion: bool
    automatic_live_canary_promotion: bool
    canary_capital_limit: float
    required_forward_evidence_days: int
    maximum_drawdown: float
    maximum_execution_drift: float
    maximum_model_drift: float


@dataclass(frozen=True)
class PromotionEvidence:
    strategy_artefact_hash: str
    source_commit_hash: str
    validation_accepted: bool
    protected_holdout_accepted: bool
    forward_evidence_days: int
    forward_evidence_accepted: bool
    drawdown: float
    execution_drift: float
    model_drift: float
    portfolio_capacity: float
    requested_capital: float
    risk_budget_available: float
    live_approval: bool
    fresh_preflight: bool


@dataclass(frozen=True)
class PromotionDecision:
    strategy_version_id: str
    prior_state: LifecycleState
    next_state: LifecycleState
    accepted: bool
    reason_code: str
    evaluated_at: str
    capital_limit: float
    evidence_hash: str


def decide_promotion(
    *,
    strategy_version_id: str,
    current_state: LifecycleState,
    evidence: PromotionEvidence,
    policy: PromotionPolicy,
    evaluated_at: str,
) -> PromotionDecision:
    strategy_version_id = non_empty(strategy_version_id, field="strategy_version_id")
    evaluated_at = timestamp(evaluated_at, field="evaluated_at")
    if evidence.drawdown > policy.maximum_drawdown:
        next_state = (
            LifecycleState.SUSPENDED
            if current_state
            in {
                LifecycleState.FORWARD_PAPER,
                LifecycleState.LIVE_CANARY,
                LifecycleState.LIVE,
            }
            else current_state
        )
        return _decision(
            strategy_version_id,
            current_state,
            next_state,
            False,
            "drawdown_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if evidence.execution_drift > policy.maximum_execution_drift:
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.SUSPENDED,
            False,
            "execution_drift_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if evidence.model_drift > policy.maximum_model_drift:
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.SUSPENDED,
            False,
            "model_drift_limit",
            evaluated_at,
            0.0,
            evidence,
        )
    if current_state in {LifecycleState.REGISTERED, LifecycleState.DEVELOPMENT}:
        if not evidence.validation_accepted:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "validation_not_accepted",
                evaluated_at,
                0.0,
                evidence,
            )
        if not evidence.protected_holdout_accepted:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "protected_holdout_not_accepted",
                evaluated_at,
                0.0,
                evidence,
            )
        if not policy.automatic_paper_promotion:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                "automatic_paper_promotion_disabled",
                evaluated_at,
                0.0,
                evidence,
            )
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.FORWARD_PAPER,
            True,
            "forward_paper_promoted",
            evaluated_at,
            0.0,
            evidence,
        )
    if current_state is LifecycleState.FORWARD_PAPER:
        checks = (
            (
                evidence.forward_evidence_days < policy.required_forward_evidence_days,
                "forward_evidence_duration_insufficient",
            ),
            (not evidence.forward_evidence_accepted, "forward_evidence_not_accepted"),
            (not policy.automatic_live_canary_promotion, "automatic_live_canary_disabled"),
            (not evidence.live_approval, "live_approval_missing"),
            (not evidence.fresh_preflight, "fresh_preflight_missing"),
            (evidence.portfolio_capacity <= 0, "portfolio_capacity_unavailable"),
            (evidence.risk_budget_available <= 0, "risk_budget_unavailable"),
        )
        reason = next((reason for failed, reason in checks if failed), None)
        if reason:
            return _decision(
                strategy_version_id,
                current_state,
                current_state,
                False,
                reason,
                evaluated_at,
                0.0,
                evidence,
            )
        capital = min(
            policy.canary_capital_limit,
            evidence.requested_capital,
            evidence.portfolio_capacity,
            evidence.risk_budget_available,
        )
        return _decision(
            strategy_version_id,
            current_state,
            LifecycleState.LIVE_CANARY,
            True,
            "live_canary_promoted",
            evaluated_at,
            capital,
            evidence,
        )
    return _decision(
        strategy_version_id,
        current_state,
        current_state,
        True,
        "lifecycle_state_maintained",
        evaluated_at,
        min(evidence.requested_capital, evidence.risk_budget_available),
        evidence,
    )


def _decision(
    strategy_version_id: str,
    prior_state: LifecycleState,
    next_state: LifecycleState,
    accepted: bool,
    reason_code: str,
    evaluated_at: str,
    capital_limit: float,
    evidence: PromotionEvidence,
) -> PromotionDecision:
    return PromotionDecision(
        strategy_version_id=strategy_version_id,
        prior_state=prior_state,
        next_state=next_state,
        accepted=accepted,
        reason_code=reason_code,
        evaluated_at=evaluated_at,
        capital_limit=max(0.0, capital_limit),
        evidence_hash=canonical_hash(evidence),
    )


class SqlPromotionStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, decision: PromotionDecision) -> str:
        payload = json_value(
            {
                "strategy_version_id": decision.strategy_version_id,
                "prior_state": decision.prior_state.value,
                "next_state": decision.next_state.value,
                "accepted": decision.accepted,
                "reason_code": decision.reason_code,
                "evaluated_at": decision.evaluated_at,
                "capital_limit": decision.capital_limit,
                "evidence_hash": decision.evidence_hash,
            },
            field="promotion decision",
        )
        identity = canonical_hash(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(promotion_event.c.payload).where(promotion_event.c.id == identity)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("promotion decision identity collision")
                return identity
            connection.execute(
                insert(promotion_event).values(
                    id=identity,
                    created_at=decision.evaluated_at,
                    payload=payload,
                )
            )
        return identity

    def latest(self, strategy_version_id: str) -> PromotionDecision | None:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(promotion_event.c.payload).order_by(promotion_event.c.created_at.desc())
            ).scalars()
            for payload in payloads:
                if payload.get("strategy_version_id") != strategy_version_id:
                    continue
                return PromotionDecision(
                    strategy_version_id=payload["strategy_version_id"],
                    prior_state=LifecycleState(payload["prior_state"]),
                    next_state=LifecycleState(payload["next_state"]),
                    accepted=bool(payload["accepted"]),
                    reason_code=payload["reason_code"],
                    evaluated_at=payload["evaluated_at"],
                    capital_limit=float(payload["capital_limit"]),
                    evidence_hash=payload["evidence_hash"],
                )
        return None


class DatabasePromotionWorker:
    """Evaluate deterministic lifecycle jobs without granting execution authority."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlPromotionStore,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("promotion_evaluation",),
        )
        if claimed is None:
            return {"reason_code": "promotion_queue_empty"}
        try:
            evidence = claimed.payload.get("evidence")
            policy = claimed.payload.get("policy")
            if not isinstance(evidence, dict) or not isinstance(policy, dict):
                raise ValueError("promotion evidence and policy must be objects")
            decision = decide_promotion(
                strategy_version_id=str(claimed.payload["strategy_version_id"]),
                current_state=LifecycleState(str(claimed.payload["current_state"])),
                evidence=PromotionEvidence(**evidence),
                policy=PromotionPolicy(**policy),
                evaluated_at=str(claimed.payload.get("evaluated_at") or now),
            )
            identity = self.store.append(decision)
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "promotion_evaluation_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": decision.reason_code,
            "job_id": claimed.job_id,
            "promotion_event_id": identity,
            "accepted": decision.accepted,
            "next_state": decision.next_state.value,
            "capital_limit": decision.capital_limit,
        }


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
