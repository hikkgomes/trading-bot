"""Lease-based live account reconciliation and durable recovery planning."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

from src.domain._codec import canonical_hash, timestamp
from src.execution.reconciler import ReconciliationResult
from src.execution.recovery import (
    RecoveryAction,
    RecoveryActionType,
    SqlRecoveryStore,
    plan_recovery,
)
from src.services.scheduler import DatabaseJobQueue


class DatabaseLiveRecoveryWorker:
    """Consume ambiguous live state without repeating exchange side effects."""

    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlRecoveryStore,
        reconcile_product: Callable[[str], ReconciliationResult],
        account_products: Mapping[str, str],
        execute_action: Callable[[str, RecoveryAction], Mapping[str, Any]] | None = None,
        backfill_account: Callable[[str, str], Mapping[str, Any]] | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.reconcile_product = reconcile_product
        self.account_products = dict(account_products)
        self.execute_action = execute_action
        self.backfill_account = backfill_account
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=("live_order_recovery", "live_account_backfill"),
        )
        if claimed is None:
            return {"reason_code": "live_recovery_queue_empty"}
        try:
            product_id = str(claimed.payload.get("product_id") or "")
            if not product_id:
                product_id = self.account_products[str(claimed.payload["account_id"])]
            self._recover_protective_child(product_id, claimed.payload)
            backfill = self._backfill(product_id, claimed.name, now)
            if backfill is not None and backfill.get("history_complete") is False:
                self.queue.defer(
                    claimed, available_at=_retry_at(now, 1), reason="income_history_pending"
                )
                return {"reason_code": "income_history_pending", "backfill": dict(backfill)}
            reconciliation = self.reconcile_product(product_id)
            plan = plan_recovery(reconciliation, created_at=now, store=self.store)
            if plan is None:
                recovery_kind = claimed.payload.get("recovery_kind")
                if recovery_kind in {
                    "user_stream_reconnect",
                    "emergency_ambiguous_submission",
                    "protective_child",
                }:
                    self.queue.complete(claimed, completed_at=now)
                    return {
                        "reason_code": "live_recovery_verified",
                        "job_id": claimed.job_id,
                        "product_id": product_id,
                        "recovery_kind": recovery_kind,
                        **({"backfill": dict(backfill)} if backfill is not None else {}),
                    }
                if claimed.name == "live_account_backfill":
                    self.queue.complete(claimed, completed_at=now)
                    return {
                        "reason_code": "live_account_backfill_completed",
                        "job_id": claimed.job_id,
                        "product_id": product_id,
                        **({"backfill": dict(backfill)} if backfill is not None else {}),
                    }
                raise ValueError("recovery job found no exchange-state difference")
            action_results: tuple[dict[str, Any], ...] = ()
            if self.execute_action is not None:
                action_results = tuple(
                    dict(self.execute_action(product_id, action)) for action in plan.actions
                )
                verification = self.reconcile_product(product_id)
                if verification.recovery_required:
                    self.queue.defer(
                        claimed,
                        available_at=_retry_at(now, min(self.lease_seconds, 30)),
                        reason="live_recovery_reconciliation_pending",
                    )
                    return {
                        "reason_code": "live_recovery_reconciliation_pending",
                        "job_id": claimed.job_id,
                        "product_id": product_id,
                        "recovery_plan_id": plan.plan_id,
                        "actions": len(plan.actions),
                        "action_results": list(action_results),
                        "operator_review_required": plan.requires_operator_review,
                    }
                resolution_id = self.store.resolve(
                    plan.plan_id,
                    resolved_at=now,
                    verification_hash=canonical_hash(
                        {
                            "product_id": product_id,
                            "recovery_plan_id": plan.plan_id,
                            "matched": verification.matched,
                        }
                    ),
                )
            else:
                resolution_id = None
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "live_recovery_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": (
                "live_recovery_actions_executed"
                if self.execute_action is not None
                else "live_recovery_plan_created"
            ),
            "job_id": claimed.job_id,
            "product_id": product_id,
            "recovery_plan_id": plan.plan_id,
            "actions": len(plan.actions),
            "operator_review_required": plan.requires_operator_review,
            "action_results": list(action_results),
            "resolution_id": resolution_id,
            **({"backfill": dict(backfill)} if backfill is not None else {}),
        }

    def _recover_protective_child(self, product_id: str, payload: Mapping[str, Any]) -> None:
        child_id = payload.get("protective_child_order_id")
        if not child_id:
            if payload.get("recovery_kind") == "protective_child":
                raise RuntimeError("protective trigger is waiting for its exchange child identity")
            return
        if self.execute_action is None:
            raise RuntimeError("protective child recovery requires an order reconciler")
        self.execute_action(
            product_id, RecoveryAction(RecoveryActionType.RECONCILE_ORDER, str(child_id))
        )

    def _backfill(self, product_id: str, name: str, now: str) -> Mapping[str, Any] | None:
        if self.backfill_account is None:
            if name == "live_account_backfill":
                raise RuntimeError("periodic live backfill has no account backfill handler")
            return None
        return self.backfill_account(product_id, now)


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(timestamp(value, field="now"))
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
