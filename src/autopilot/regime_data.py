"""Status helpers for regime-tagged research datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.autopilot.config import JobConfig
from src.config import PROJECT_ROOT


def _project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _command_value(command: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for part in command:
        if part.startswith(prefix):
            value = part[len(prefix) :]
            return value or None
    try:
        index = command.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    value = command[value_index]
    if value.startswith("--"):
        return None
    return value


def _is_regime_job(job: JobConfig) -> bool:
    command = list(job.command or [])
    return "src.regime" in command


def build_regime_data_statuses(jobs: list[JobConfig]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for job in jobs:
        if not _is_regime_job(job):
            continue
        command = list(job.command or [])
        report_path = _project_path(_command_value(command, "--report"))
        output_path = _project_path(_command_value(command, "--output"))
        status: dict[str, Any] = {
            "name": job.name,
            "enabled": bool(job.enabled),
            "report_path": str(report_path) if report_path else None,
            "output_path": str(output_path) if output_path else None,
            "available": None if not job.enabled else False,
            "ok": True if not job.enabled else False,
            "reason": "disabled" if not job.enabled else None,
        }
        if not job.enabled:
            statuses.append(status)
            continue
        if report_path is None:
            status.update(reason="missing_report_argument")
            statuses.append(status)
            continue
        if not report_path.exists():
            status.update(reason="missing_report")
            statuses.append(status)
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status.update(reason="report_read_error", error=str(exc))
            statuses.append(status)
            continue

        output_from_report = _project_path(payload.get("output"))
        if output_path is None:
            output_path = output_from_report
            status["output_path"] = str(output_path) if output_path else None
        output_exists = bool(output_path and output_path.exists())
        skipped = bool(payload.get("skipped"))
        rows = int(payload.get("rows") or 0)
        available = bool(payload.get("ok") and not skipped and output_exists and rows > 0)
        status.update(
            {
                "ok": available,
                "available": available,
                "reason": "ready" if available else payload.get("reason") or "not_ready",
                "skipped": skipped,
                "rows": rows,
                "output_exists": output_exists,
                "regime_counts": payload.get("regime_counts") or {},
                "input": payload.get("input"),
                "daily_input": payload.get("daily_input"),
                "error": payload.get("error"),
            }
        )
        statuses.append(status)
    return statuses
