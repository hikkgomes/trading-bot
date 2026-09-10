"""Lease-based deterministic six-level risk assessment service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from src.domain._codec import canonical_hash, non_empty
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
from src.services.job_schemas import RiskAssessmentRequest
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
        execution_modes: Mapping[str, str] | None = None,
        lease_seconds: int = 60,
        job_name: str = "risk_assessment",
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.snapshot_store = snapshot_store
        self.execution_modes = dict(execution_modes or {})
        self.lease_seconds = lease_seconds
        self.job_name = job_name

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=(self.job_name,),
        )
        if claimed is None:
            return {"reason_code": "risk_queue_empty"}
        try:
            request = RiskAssessmentRequest.from_mapping(claimed.payload)
            product_id = request.product_id
            assessment_id = request.assessment_id
            loaded_inputs = self._load_snapshot_inputs(request)
            policy_limits = self._policy_limits(request)
            if set(policy_limits) != {scope for scope, _, _ in _SCOPES}:
                raise ValueError("risk policies must define all six risk scopes")
            snapshot_inputs = {
                scope: {**dict(values), "limits": policy_limits[scope]}
                for scope, values in loaded_inputs.items()
            }
            decisions = tuple(
                self._evaluate_scope(
                    snapshot_inputs, scope, evaluator, limits_type, request.evaluated_at
                )
                for scope, evaluator, limits_type in _SCOPES
            )
            assessment = combine_risk_decisions(
                decisions,
                assessment_id=assessment_id,
                product_id=product_id,
                store=self.store,
            )
            if product_id in self.execution_modes:
                target_snapshot = (
                    self.snapshot_store or SqlRiskSnapshotStore(self.store.engine)
                ).get(request.target_position_snapshot_id)
                execution_payload = {
                    "product_id": product_id,
                    "event_id": request.event_id,
                    "evaluated_at": request.evaluated_at,
                    "risk_assessment_id": assessment.aggregate.decision_id,
                    "target_position_snapshot_id": request.target_position_snapshot_id,
                    "execution_mode": target_snapshot.get(
                        "execution_mode", self.execution_modes[product_id]
                    ),
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
                {"execution_job_id": execution_job_id} if product_id in self.execution_modes else {}
            ),
        }

    def _policy_limits(self, request: RiskAssessmentRequest) -> dict[str, dict]:
        snapshots = self.snapshot_store or SqlRiskSnapshotStore(self.store.engine)
        account = snapshots.get(request.account_snapshot_id)
        if account.get("product_id") != request.product_id:
            raise ValueError("risk account snapshot belongs to another product")
        raw_account_id = account.get("account_id")
        if not isinstance(raw_account_id, str):
            raise ValueError("risk snapshot has no account identity")
        account_id = non_empty(raw_account_id, field="risk snapshot account id")
        return SqlRiskPolicyStore(self.store.engine).resolve(
            request.risk_policy_ids, account_id=account_id
        )

    @staticmethod
    def _evaluate_scope(
        payload: Mapping[str, Any],
        scope: str,
        evaluator: Evaluator,
        limits_type: type,
        evaluated_at: str,
    ) -> RiskDecision:
        raw = payload.get(scope)
        if not isinstance(raw, Mapping):
            raise ValueError(f"risk job is missing {scope} input")
        inputs = raw.get("inputs")
        limits = raw.get("limits")
        if not isinstance(inputs, Mapping) or not isinstance(limits, Mapping):
            raise ValueError(f"risk job {scope} inputs and limits must be objects")
        decision_id = str(raw.get("decision_id") or "")
        return replace(
            evaluator(
                decision_id=decision_id,
                **dict(inputs),
                limits=limits_type(**dict(limits)),
            ),
            evaluated_at=evaluated_at,
        )

    def _load_snapshot_inputs(
        self, request: RiskAssessmentRequest
    ) -> Mapping[str, Mapping[str, Any]]:
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
        self._validate_sources(request, records)
        targets = records[0].get("scopes")
        if not isinstance(targets, Mapping) or set(targets) != {
            "strategy",
            "instrument",
            "sleeve",
            "product",
        }:
            raise ValueError("target snapshot must contain exactly its four risk scopes")
        scopes = {**targets, "account": records[1], "global": records[4]}
        if any(not isinstance(value, Mapping) for value in scopes.values()):
            raise ValueError("canonical risk scopes must be objects")
        return scopes

    def _validate_sources(
        self, request: RiskAssessmentRequest, records: list[dict[str, object]]
    ) -> None:
        kinds = (
            "target_position",
            "account_risk_input",
            "positions_risk_input",
            "balances_risk_input",
            "global_risk_input",
        )
        for kind, record in zip(kinds, records, strict=True):
            if record.get("kind") != kind:
                raise ValueError("risk source snapshot has the wrong kind")
            if record.get("product_id") != request.product_id:
                raise ValueError("risk source snapshot belongs to another product")
        state_id = records[0].get("canonical_state_id")
        if (
            not isinstance(state_id, str)
            or not state_id
            or any(record.get("canonical_state_id") != state_id for record in records)
        ):
            raise ValueError("risk source snapshots must share one canonical state")
        state = SqlRiskSnapshotStore(self.store.engine).get(state_id)
        if (
            state.get("kind") != "canonical_portfolio_risk_state"
            or state.get("product_id") != request.product_id
        ):
            raise ValueError("risk source state has the wrong identity")
        if state.get("risk_policy_ids") != list(request.risk_policy_ids):
            raise ValueError("risk request policies differ from its canonical state")


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
