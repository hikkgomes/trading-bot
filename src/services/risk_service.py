"""Lease-based deterministic six-level risk assessment service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

from src.domain.risk import RiskDecision
from src.risk.account import AccountRiskLimits, assess_account_risk
from src.risk.engine import SqlRiskDecisionStore, combine_risk_decisions
from src.risk.global_risk import GlobalRiskLimits, assess_global_risk
from src.risk.instrument import InstrumentRiskLimits, assess_instrument_risk
from src.risk.product import ProductRiskLimits, assess_product_risk
from src.risk.sleeve import SleeveRiskLimits, assess_sleeve_risk
from src.risk.strategy import StrategyRiskLimits, assess_strategy_risk
from src.services.scheduler import DatabaseJobQueue

Evaluator = Callable[..., RiskDecision]

_SCOPES: tuple[tuple[str, Evaluator, type], ...] = (
    ("strategy", assess_strategy_risk, StrategyRiskLimits),
    ("instrument", assess_instrument_risk, InstrumentRiskLimits),
    ("sleeve", assess_sleeve_risk, SleeveRiskLimits),
    ("product", assess_product_risk, ProductRiskLimits),
    ("account", assess_account_risk, AccountRiskLimits),
    ("global", assess_global_risk, GlobalRiskLimits),
)


class DatabaseRiskWorker:
    """Create one immutable aggregate from six explicitly supplied snapshots."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlRiskDecisionStore,
        lease_seconds: int = 60,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("risk_assessment",),
        )
        if claimed is None:
            return {"reason_code": "risk_queue_empty"}
        try:
            product_id = str(claimed.payload["product_id"])
            assessment_id = str(claimed.payload["assessment_id"])
            decisions = tuple(
                self._evaluate_scope(claimed.payload, scope, evaluator, limits_type)
                for scope, evaluator, limits_type in _SCOPES
            )
            assessment = combine_risk_decisions(
                decisions,
                assessment_id=assessment_id,
                product_id=product_id,
                store=self.store,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "risk_assessment_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": (
                "risk_assessment_accepted" if assessment.accepted else "risk_assessment_rejected"
            ),
            "job_id": claimed.job_id,
            "assessment_id": assessment.aggregate.decision_id,
            "accepted": assessment.accepted,
            "first_rejected_scope": assessment.aggregate.input_snapshot["first_rejected_scope"],
        }

    @staticmethod
    def _evaluate_scope(
        payload: Mapping[str, Any],
        scope: str,
        evaluator: Evaluator,
        limits_type: type,
    ) -> RiskDecision:
        raw = payload.get(scope)
        if not isinstance(raw, Mapping):
            raise ValueError(f"risk job is missing {scope} input")
        inputs = raw.get("inputs")
        limits = raw.get("limits")
        if not isinstance(inputs, Mapping) or not isinstance(limits, Mapping):
            raise ValueError(f"risk job {scope} inputs and limits must be objects")
        decision_id = str(raw.get("decision_id") or "")
        return evaluator(
            decision_id=decision_id,
            **dict(inputs),
            limits=limits_type(**dict(limits)),
        )


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
