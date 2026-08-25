"""Lease-based research worker that cannot access order submission."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

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
    ) -> None:
        if runtime.node.operating_system != "linux":
            raise ValueError("research workers must run on Linux")
        if runtime.can_submit_orders:
            raise PermissionError("research workers cannot submit exchange orders")
        if any(name in os.environ for name in ("EXCHANGE_API_KEY", "EXCHANGE_API_SECRET")):
            raise PermissionError("research workers must not receive exchange credentials")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.runtime = runtime
        self.queue = queue
        self.worker_id = worker_id
        self.handlers = dict(handlers)
        self.lease_seconds = lease_seconds
        self.heavy_compute = heavy_compute

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
        compute_lease = None
        if self.heavy_compute is not None:
            compute_lease = self.heavy_compute.acquire(
                owner=self.worker_id,
                job_id=claimed.job_id,
                now=now,
                lease_seconds=self.lease_seconds,
            )
            if compute_lease is None:
                self.queue.fail(
                    claimed,
                    completed_at=now,
                    error="heavy_compute_slot_unavailable",
                    retry_at=_add_seconds(now, min(self.lease_seconds, 60)),
                )
                return {"reason_code": "heavy_compute_slot_unavailable", "job_id": claimed.job_id}
        handler = self.handlers.get(claimed.name)
        if handler is None:
            retry_at = _add_seconds(now, self.lease_seconds)
            self.queue.fail(
                claimed,
                completed_at=now,
                error="unsupported_job_type",
                retry_at=retry_at,
            )
            return {
                "reason_code": "unsupported_job_type",
                "job_id": claimed.job_id,
                "job_name": claimed.name,
            }
        current = claimed
        heavy_compute = self.heavy_compute

        def renew() -> ClaimedJob:
            nonlocal current
            current = self.queue.heartbeat(
                current,
                now=utc_now(),
                lease_seconds=self.lease_seconds,
            )
            if compute_lease is not None and heavy_compute is not None:
                # Renewal is deliberately tied to the queue lease so a dead
                # worker cannot keep the singleton slot indefinitely.
                heavy_compute.renew(
                    compute_lease,
                    now=current.lease_expires_at,
                    lease_seconds=self.lease_seconds,
                )
            return current

        try:
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
        self.queue.complete(current, completed_at=utc_now())
        if compute_lease is not None and heavy_compute is not None:
            heavy_compute.release(compute_lease)
        return {"reason_code": "research_job_completed", "job_id": current.job_id, **result}


def _add_seconds(value: str, seconds: int) -> str:
    import datetime as dt

    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
