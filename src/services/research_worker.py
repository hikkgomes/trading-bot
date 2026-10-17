"""Lease-based research worker that cannot access order submission."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

from src.research.outcomes import OutcomeCategory
from src.services.heavy_compute import HeavyComputeLeaseStore
from src.services.runtime import ServiceRuntime, utc_now
from src.services.scheduler import ClaimedJob, DatabaseJobQueue

JobHandler = Callable[[ClaimedJob, Callable[[], ClaimedJob]], Mapping[str, Any] | None]


class ResearchWorker:
    def __init__(
        self,
        *,
        runtime: ServiceRuntime,
        queue: DatabaseJobQueue,
        worker_id: str,
        handlers: Mapping[str, JobHandler],
        lease_seconds: int = 120,
        heavy_compute: HeavyComputeLeaseStore | None = None,
        forbidden_secret_names: tuple[str, ...] = (),
        maximum_runtime_seconds: int | None = None,
    ) -> None:
        if runtime.node.operating_system != "linux":
            raise ValueError("research workers must run on Linux")
        if runtime.can_submit_orders:
            raise PermissionError("research workers cannot submit exchange orders")
        forbidden = {
            "EXCHANGE_API_KEY",
            "EXCHANGE_API_SECRET",
            *(str(name).strip() for name in forbidden_secret_names if str(name).strip()),
        }
        if any(name in os.environ for name in forbidden):
            raise PermissionError("research workers must not receive exchange credentials")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.runtime = runtime
        self.queue = queue
        self.worker_id = worker_id
        self.handlers = dict(handlers)
        self.lease_seconds = lease_seconds
        self.heavy_compute = heavy_compute
        if maximum_runtime_seconds is not None and maximum_runtime_seconds <= 0:
            raise ValueError("research deadline must be positive")
        self.maximum_runtime_seconds = maximum_runtime_seconds

    def run_once(self, *, now: str | None = None) -> dict[str, Any]:
        now = now or utc_now()
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=tuple(sorted(self.handlers)),
        )
        if claimed is None:
            return {"reason_code": "research_queue_empty"}
        if claimed.name == "agent_review":
            # Metadata-only review preparation must not wait for a long dataset build.
            return self._execute_handler(claimed, self.handlers[claimed.name], None)
        compute_lease = self._acquire_compute_lease(claimed, now=now)
        if self.heavy_compute is not None and compute_lease is None:
            self.queue.defer(
                claimed,
                available_at=_add_seconds(now, min(self.lease_seconds, 60)),
                reason="heavy_compute_slot_unavailable",
            )
            return {"reason_code": "heavy_compute_slot_unavailable", "job_id": claimed.job_id}
        handler = self.handlers.get(claimed.name)
        if handler is None:
            return self._fail_unsupported(claimed, now=now)
        return self._execute_handler(claimed, handler, compute_lease)

    def _acquire_compute_lease(self, claimed: ClaimedJob, *, now: str):
        if self.heavy_compute is None:
            return None
        return self.heavy_compute.acquire(
            owner=self.worker_id,
            job_id=claimed.job_id,
            now=now,
            lease_seconds=self.lease_seconds,
        )

    def _fail_unsupported(self, claimed: ClaimedJob, *, now: str) -> dict[str, Any]:
        self.queue.fail(
            claimed,
            completed_at=now,
            error="unsupported_job_type",
            retry_at=_add_seconds(now, self.lease_seconds),
        )
        return {
            "reason_code": "unsupported_job_type",
            "job_id": claimed.job_id,
            "job_name": claimed.name,
        }

    def _execute_handler(
        self, claimed: ClaimedJob, handler: JobHandler, compute_lease
    ) -> dict[str, Any]:
        current = claimed
        heavy_compute = self.heavy_compute
        deadline = time.monotonic() + (self.maximum_runtime_seconds or self.lease_seconds)

        def renew() -> ClaimedJob:
            nonlocal current
            lease_seconds = max(1, min(self.lease_seconds, int(deadline - time.monotonic())))
            current = self.queue.heartbeat(
                current,
                now=utc_now(),
                lease_seconds=lease_seconds,
            )
            if compute_lease is not None and heavy_compute is not None:
                # Renewal is deliberately tied to the queue lease so a dead
                # worker cannot keep the singleton slot indefinitely.
                heavy_compute.renew(
                    compute_lease,
                    now=utc_now(),
                    lease_seconds=lease_seconds,
                )
            return current

        try:
            with _job_deadline(self.maximum_runtime_seconds):
                result = dict(handler(current, renew) or {})
        except Exception as exc:
            if compute_lease is not None and heavy_compute is not None:
                heavy_compute.release(compute_lease)
            completed_at = utc_now()
            self.queue.fail(
                current,
                completed_at=completed_at,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_add_seconds(completed_at, self.lease_seconds),
            )
            return {
                "reason_code": "research_job_failed",
                "job_id": current.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if result.pop("_defer", False):
            defer_reason = str(result.pop("_defer_reason", "research_work_deferred"))
            defer_at = _add_seconds(utc_now(), min(self.lease_seconds, 300))
            if compute_lease is not None and heavy_compute is not None:
                heavy_compute.release(compute_lease)
            category = OutcomeCategory(
                result.get("outcome_category") or OutcomeCategory.BLOCKED_INFRASTRUCTURE
            )
            self.queue.defer(
                current, available_at=defer_at, reason=defer_reason, outcome_category=category
            )
            return {
                "reason_code": "research_job_deferred",
                "job_id": current.job_id,
                **result,
                "defer_reason": defer_reason,
            }
        self.queue.complete(current, completed_at=utc_now(), outcome=result)
        if compute_lease is not None and heavy_compute is not None:
            heavy_compute.release(compute_lease)
        return _completed_outcome(current.job_id, result)


@contextmanager
def _job_deadline(seconds):
    timer = threading.Timer(seconds, lambda: os._exit(124)) if seconds is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()


def _add_seconds(value: str, seconds: int) -> str:
    import datetime as dt

    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def _completed_outcome(job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    handler_reason = result.pop("reason_code", None)
    return {
        "reason_code": "research_job_completed",
        "job_id": job_id,
        **result,
        **({"handler_reason_code": handler_reason} if handler_reason is not None else {}),
    }
