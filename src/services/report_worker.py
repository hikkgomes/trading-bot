"""Immutable operator-report materialisation from PostgreSQL state."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from src.domain._codec import canonical_hash, timestamp
from src.observability.reports import DatabasePlatformReport
from src.services.scheduler import DatabaseJobQueue


class DatabaseReportWorker:
    def __init__(
        self,
        *,
        engine: Engine,
        root: Path,
        queue: DatabaseJobQueue | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.report = DatabasePlatformReport(engine)
        self.root = root.resolve()
        self.queue = queue
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        now = timestamp(now, field="now")
        claimed = None
        if self.queue is not None:
            if self.worker_id is None:
                raise ValueError("queued report worker requires a worker identity")
            claimed = self.queue.claim(
                worker_id=self.worker_id,
                now=now,
                lease_seconds=self.lease_seconds,
                names=("reporting",),
            )
            if claimed is None:
                return {"reason_code": "report_queue_empty"}
        report = {**self.report.build(), "generated_at": now}
        report_hash = canonical_hash(report)
        date = now[:10]
        destination = self.root / date / f"{report_hash.removeprefix('sha256:')}.json"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        result = {
            "reason_code": "operator_report_written",
            "report_hash": report_hash,
            "path": str(destination),
        }
        if claimed is not None and self.queue is not None:
            self.queue.complete(claimed, completed_at=now)
            result["job_id"] = claimed.job_id
        return result
