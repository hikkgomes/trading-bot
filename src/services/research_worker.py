"""Lease-based research worker that cannot access order submission."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

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
    ) -> None:
        if runtime.node.operating_system != "macos":
            raise ValueError("research workers must run on the macOS research node")
        if runtime.can_submit_orders:
            raise PermissionError("research workers cannot submit exchange orders")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.runtime = runtime
        self.queue = queue
        self.worker_id = worker_id
        self.handlers = dict(handlers)
        self.lease_seconds = lease_seconds

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

        def renew() -> ClaimedJob:
            nonlocal current
            current = self.queue.heartbeat(
                current,
                now=utc_now(),
                lease_seconds=self.lease_seconds,
            )
            return current

        try:
            result = dict(handler(current, renew) or {})
        except Exception as exc:
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
        return {"reason_code": "research_job_completed", "job_id": current.job_id, **result}


def _add_seconds(value: str, seconds: int) -> str:
    import datetime as dt

    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
