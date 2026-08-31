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

    def _emit_sli_alerts(self, report: dict[str, Any], *, now: str) -> list[str]:
        if self.alerts is None:
            return []
        funnel = report.get("research", {}).get("funnel", {})
        operations = report.get("operations", {})
        slis = operations.get("slis", {})
        emitted: list[str] = []
        age_by_state = funnel.get("candidate_age_by_state", {})
        blocking_states = {
            "queued",
            "screening",
            "waiting_for_dataset:screening",
            "waiting_for_dataset:development",
            "waiting_for_dataset:robustness",
            "waiting_for_dataset:protected_holdout",
        }
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
                    payload={
                        "candidate_ids": funnel["candidates_without_job_or_reason"],
                    },
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
        stale_account = slis.get("stale_account_authority", {})
        if int(stale_account.get("count", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="account_authority_stale",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="risk:stale-account-authority",
                    target="account-reconciliation",
                    message="authenticated account authority is stale or unknown",
                    now=now,
                    payload={"stale_account_authority": stale_account},
                )
            )
        stale_market = slis.get("stale_market_data", {})
        if int(stale_market.get("count", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="market_data_stale",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="risk:stale-market-data",
                    target="market-gateway",
                    message="market data authority is stale",
                    now=now,
                    payload={"stale_market_data": stale_market},
                )
            )
        missing_risk = slis.get("missing_risk_data", {})
        if int(missing_risk.get("count", 0)) > 0:
            emitted.append(
                self._emit(
                    event_type="risk_state_data_missing",
                    severity=AlertSeverity.CRITICAL,
                    dedupe_key="risk:state-data-missing",
                    target="portfolio-state-service",
                    message="portfolio risk state contains unavailable measurements",
                    now=now,
                    payload={"missing_risk_data": missing_risk},
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
