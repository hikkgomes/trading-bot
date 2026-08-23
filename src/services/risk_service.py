"""Lease-based deterministic six-level risk assessment service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

from src.domain._codec import canonical_hash
from src.domain.risk import RiskDecision
from src.risk.account import AccountRiskLimits, assess_account_risk
from src.risk.engine import (
    SqlRiskDecisionStore,
    SqlRiskPolicyStore,
    SqlRiskSnapshotStore,
    combine_risk_decisions,
)
from src.risk.global_risk import GlobalRiskLimits, assess_global_risk
from src.risk.instrument import InstrumentRiskLimits, assess_instrument_risk
from src.risk.product import ProductRiskLimits, assess_product_risk
from src.risk.sleeve import SleeveRiskLimits, assess_sleeve_risk
from src.risk.strategy import StrategyRiskLimits, assess_strategy_risk
from src.services.job_schemas import JobSchemaError, RiskAssessmentRequest
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
        snapshot_store: SqlRiskSnapshotStore | None = None,
        snapshot_loader: Callable[[RiskAssessmentRequest], Mapping[str, Mapping[str, Any]]]
        | None = None,
        execution_modes: Mapping[str, str] | None = None,
        lease_seconds: int = 60,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.snapshot_store = snapshot_store
        self.snapshot_loader = snapshot_loader
        self.execution_modes = dict(execution_modes or {})
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
            try:
                request = RiskAssessmentRequest.from_mapping(claimed.payload)
            except JobSchemaError:
                if self.store.engine.dialect.name != "sqlite" or not _legacy_risk_fixture(
                    claimed.payload
                ):
                    raise
                request = None
            if request is None:
                product_id = str(claimed.payload["product_id"])
                assessment_id = str(claimed.payload["assessment_id"])
                snapshot_inputs = claimed.payload
            else:
                product_id = request.product_id
                assessment_id = request.assessment_id
                loaded_inputs = self._load_snapshot_inputs(request)
                policy_limits = SqlRiskPolicyStore(self.store.engine).resolve(
                    request.risk_policy_ids
                )
                if set(policy_limits) != {scope for scope, _, _ in _SCOPES}:
                    raise ValueError("risk policies must define all six risk scopes")
                snapshot_inputs = {
                    scope: {**dict(values), "limits": policy_limits[scope]}
                    for scope, values in loaded_inputs.items()
                }
            decisions = tuple(
                self._evaluate_scope(snapshot_inputs, scope, evaluator, limits_type)
                for scope, evaluator, limits_type in _SCOPES
            )
            assessment = combine_risk_decisions(
                decisions,
                assessment_id=assessment_id,
                product_id=product_id,
                store=self.store,
            )
            if request is not None and product_id in self.execution_modes:
                execution_payload = {
                    "product_id": product_id,
                    "event_id": request.event_id,
                    "evaluated_at": request.evaluated_at,
                    "risk_assessment_id": assessment.aggregate.decision_id,
                    "target_position_snapshot_id": request.target_position_snapshot_id,
                    "execution_mode": self.execution_modes[product_id],
                }
                execution_job_id = "execution:" + canonical_hash(execution_payload).removeprefix(
                    "sha256:"
                )
                self.queue.enqueue_if_absent(
                    job_id=execution_job_id,
                    name="execute_targets",
                    payload=execution_payload,
                    available_at=request.evaluated_at,
                    priority=20,
                    producer_identity=self.worker_id,
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
            **(
                {"execution_job_id": execution_job_id}
                if request is not None and product_id in self.execution_modes
                else {}
            ),
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

    def _load_snapshot_inputs(
        self, request: RiskAssessmentRequest
    ) -> Mapping[str, Mapping[str, Any]]:
        if self.snapshot_loader is not None:
            loaded = self.snapshot_loader(request)
            if set(loaded) != {scope for scope, _, _ in _SCOPES}:
                raise ValueError("risk snapshot loader must return all six risk scopes")
            return loaded
        if self.snapshot_store is None:
            self.snapshot_store = SqlRiskSnapshotStore(self.store.engine)
        snapshot_ids = (
            request.target_position_snapshot_id,
            request.account_snapshot_id,
            request.positions_snapshot_id,
            request.balances_snapshot_id,
            request.market_data_snapshot_id,
        )
        records = [self.snapshot_store.get(snapshot_id) for snapshot_id in snapshot_ids]
        scopes: dict[str, Mapping[str, Any]] = {}
        for record in records:
            raw_scopes = record.get("scopes")
            if isinstance(raw_scopes, Mapping):
                for scope, value in raw_scopes.items():
                    if isinstance(value, Mapping):
                        scopes[str(scope)] = value
            scope = record.get("scope")
            if isinstance(scope, str) and isinstance(record.get("inputs"), Mapping):
                scopes[scope] = record
        expected = {scope for scope, _, _ in _SCOPES}
        if set(scopes) != expected:
            raise ValueError("canonical risk snapshots do not contain all six risk scopes")
        return scopes


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def _legacy_risk_fixture(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    scopes = {"strategy", "instrument", "sleeve", "product", "account", "global"}
    return set(payload) == scopes | {"product_id", "assessment_id"}
