"""Immutable operator-report materialisation from PostgreSQL state."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from src.domain._codec import canonical_hash, timestamp
from src.observability.reports import DatabasePlatformReport
from src.services.alerting import AlertSeverity, SqlAlertService
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
        account_stale_after_seconds: int = 60,
        market_data_stale_after_seconds: int = 5,
        alerts: SqlAlertService | None = None,
        minimum_valid_screenings_before_progress: int = 10,
        backup_root: Path | None = None,
        backup_max_age_seconds: int = 172_800,
        minimum_free_bytes: int = 536_870_912,
    ) -> None:
        self.report = DatabasePlatformReport(
            engine,
            account_stale_after_seconds=account_stale_after_seconds,
            market_data_stale_after_seconds=market_data_stale_after_seconds,
        )
        self.root = root.resolve()
        self.queue = queue
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.alerts = alerts
        if minimum_valid_screenings_before_progress <= 0:
            raise ValueError("minimum valid screenings threshold must be positive")
        self.minimum_valid_screenings_before_progress = minimum_valid_screenings_before_progress
        self.backup_root = backup_root.resolve() if backup_root is not None else None
        if backup_max_age_seconds <= 0 or minimum_free_bytes < 0:
            raise ValueError("backup and disk thresholds are invalid")
        self.backup_max_age_seconds = backup_max_age_seconds
        self.minimum_free_bytes = minimum_free_bytes

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
        report = {**self.report.build(now=now), "generated_at": now}
        operations = report.setdefault("operations", {})
        operations["disk"] = self._disk_status()
        if self.backup_root is not None:
            operations["backups"] = self._backup_status(now)
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
        emitted_alerts = self._emit_sli_alerts(report, now=now)
        result = {
            "reason_code": "operator_report_written",
            "report_hash": report_hash,
            "path": str(destination),
            "alert_ids": emitted_alerts,
        }
        if claimed is not None and self.queue is not None:
            self.queue.complete(claimed, completed_at=now)
            result["job_id"] = claimed.job_id
        return result

    def _disk_status(self) -> dict[str, Any]:
        path = self.root.parent
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            return {
                "healthy": False,
                "path": str(path),
                "reason": f"disk_usage_failed:{type(exc).__name__}",
            }
        return {
            "healthy": usage.free >= self.minimum_free_bytes,
            "path": str(path),
            "free_bytes": usage.free // (1024 * 1024) * (1024 * 1024),
            "total_bytes": usage.total // (1024 * 1024) * (1024 * 1024),
            "minimum_free_bytes": self.minimum_free_bytes,
        }

    def _backup_status(self, now: str) -> dict[str, Any]:
        assert self.backup_root is not None
        latest: dict[str, float] = {}
        if self.backup_root.is_dir():
            for path in self.backup_root.iterdir():
                if path.is_symlink() or not path.is_dir() or not (path / "manifest.json").is_file():
                    continue
                kind = path.name.split("-", 1)[0]
                if kind in {"postgresql", "parquet"}:
                    latest[kind] = max(latest.get(kind, 0.0), path.stat().st_mtime)
        current = dt.datetime.fromisoformat(now).timestamp()
        ages = {kind: max(0.0, current - observed) for kind, observed in sorted(latest.items())}
        missing = sorted({"postgresql", "parquet"} - set(ages))
        stale = sorted(kind for kind, age in ages.items() if age > self.backup_max_age_seconds)
        return {
            "healthy": not missing and not stale,
            "root": str(self.backup_root),
            "ages_seconds": ages,
            "missing": missing,
            "stale": stale,
            "maximum_age_seconds": self.backup_max_age_seconds,
        }

    def _emit_sli_alerts(self, report: dict[str, Any], *, now: str) -> list[str]:
        if self.alerts is None:
            return []
        funnel = report.get("research", {}).get("funnel", {})
        operations = report.get("operations", {})
        slis = operations.get("slis", {})
        emitted: list[str] = []
        emitted.extend(self._funnel_alerts(funnel, now=now))
        emitted.extend(self._risk_alerts(slis, now=now))
        emitted.extend(self._resource_alerts(operations, now=now))
        return emitted

    def _funnel_alerts(self, funnel: Mapping[str, Any], *, now: str) -> list[str]:
        emitted: list[str] = []
        blocking_states = {
            "queued",
            "screening",
            "waiting_for_dataset:screening",
            "waiting_for_dataset:development",
            "waiting_for_dataset:robustness",
            "waiting_for_dataset:protected_holdout",
        }
        age_by_state = funnel.get("candidate_age_by_state", {})
        progressed = sum(
            int(values.get("count", 0))
            for state, values in age_by_state.items()
            if state not in blocking_states and isinstance(values, dict)
        )
        if (
            int(funnel.get("candidates_generated", 0)) > 0
            and int(funnel.get("candidates_compiled", 0))
            >= self.minimum_valid_screenings_before_progress
            and progressed == 0
        ):
            emitted.append(
                self._emit(
                    event_type="candidate_funnel_stalled",
                    severity=AlertSeverity.WARNING,
                    dedupe_key="research:candidate-funnel-stalled",
                    target="research",
                    message="no research candidate has reached development",
                    now=now,
                    payload={
                        "valid_screenings": int(funnel.get("candidates_compiled", 0)),
                        "minimum_valid_screenings": self.minimum_valid_screenings_before_progress,
                    },
                )
            )
        if int(funnel.get("missing_stage_dataset_count", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="research_dataset_missing",
                    severity=AlertSeverity.WARNING,
                    dedupe_key="research:missing-stage-dataset",
                    target="research",
                    message="research candidates are waiting for missing stage datasets",
                    now=now,
                    payload={"missing_stage_datasets": funnel.get("missing_stage_datasets", {})},
                )
            )
        if funnel.get("candidates_without_job_or_reason"):
            emitted.append(
                self._emit(
                    event_type="research_candidate_without_job",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="research:candidate-without-job",
                    target="research",
                    message="research candidates have no active job or waiting reason",
                    now=now,
                    payload={"candidate_ids": funnel["candidates_without_job_or_reason"]},
                )
            )
        schedule_progress = funnel.get("scheduled_versus_started_jobs", {})
        if int(schedule_progress.get("not_started", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="research_schedule_not_started",
                    severity=AlertSeverity.WARNING,
                    dedupe_key="research:schedule-not-started",
                    target="platform-scheduler",
                    message="scheduled platform jobs have not started",
                    now=now,
                    payload={"scheduled_versus_started_jobs": schedule_progress},
                )
            )
        return emitted

    def _risk_alerts(self, slis: Mapping[str, Any], *, now: str) -> list[str]:
        emitted: list[str] = []
        if int(slis.get("unresolved_recovery_count", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="live_recovery_required",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="live:recovery-required",
                    target="execution-engine",
                    message="live recovery work requires reconciliation",
                    now=now,
                    payload={"recovery": slis.get("unresolved_recovery", {})},
                )
            )
        checks = (
            (
                "stale_account_authority",
                "account_authority_stale",
                "risk:stale-account-authority",
                "account-reconciliation",
                "authenticated account authority is stale or unknown",
            ),
            (
                "stale_market_data",
                "market_data_stale",
                "risk:stale-market-data",
                "market-gateway",
                "market data authority is stale",
            ),
            (
                "missing_risk_data",
                "risk_state_data_missing",
                "risk:state-data-missing",
                "portfolio-state-service",
                "portfolio risk state contains unavailable measurements",
            ),
        )
        for value_key, event_type, dedupe_key, target, message in checks:
            value = slis.get(value_key, {})
            if int(value.get("count", 0)) > 0:
                emitted.append(
                    self._emit(
                        event_type=event_type,
                        severity=AlertSeverity.CRITICAL,
                        dedupe_key=dedupe_key,
                        target=target,
                        message=message,
                        now=now,
                        payload={value_key: value},
                    )
                )
        conflicts = slis.get("execution_authority_conflicts", {})
        if conflicts.get("conflict") is True:
            emitted.append(
                self._emit(
                    event_type="execution_authority_conflict",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="execution:authority-conflict",
                    target="platform",
                    message="old and new execution authorities are active together",
                    now=now,
                    payload={"execution_authority_conflicts": conflicts},
                )
            )
        return emitted

    def _resource_alerts(self, operations: Mapping[str, Any], *, now: str) -> list[str]:
        emitted: list[str] = []
        disk = operations.get("disk", {})
        if isinstance(disk, Mapping) and disk.get("healthy") is False:
            emitted.append(
                self._emit(
                    event_type="disk_capacity_low",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="platform:disk-capacity-low",
                    target="control-monitoring",
                    message="platform disk capacity is below the configured minimum",
                    now=now,
                    payload={"disk": disk},
                )
            )
        backups = operations.get("backups", {})
        if isinstance(backups, Mapping) and backups.get("healthy") is False:
            emitted.append(
                self._emit(
                    event_type="backup_failure_or_stale",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="platform:backup-failure-or-stale",
                    target="control-monitoring",
                    message="a required platform backup is missing or stale",
                    now=now,
                    payload={"backups": backups},
                )
            )
        delivery_failures = [
            item
            for item in operations.get("alerts", [])
            if isinstance(item, Mapping) and item.get("event_type") == "delivery_failed"
        ]
        if delivery_failures:
            emitted.append(
                self._emit(
                    event_type="alert_delivery_failed",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="platform:alert-delivery-failed",
                    target="control-monitoring",
                    message="external alert delivery failures are recorded",
                    now=now,
                    payload={"count": len(delivery_failures)},
                )
            )
        return emitted

    def _emit(
        self,
        *,
        event_type: str,
        severity: AlertSeverity,
        dedupe_key: str,
        target: str,
        message: str,
        now: str,
        payload: dict[str, Any],
    ) -> str:
        return self.alerts.emit(
            event_type=event_type,
            severity=severity,
            dedupe_key=dedupe_key,
            target=target,
            message=message,
            emitted_at=now,
            payload=payload,
        ).alert_id
