"""Independent scheduled-job loop for responsive trading supervision.

The trading supervisor must never sit inside a 30-minute data/research
subprocess while an operator is asking it to pause or flatten.  Deployment runs
this worker as a separate systemd service sharing only the atomic job-state and
control files with the supervisor.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, load_config
from src.autopilot.control import is_job_paused, load_control, unknown_control_selectors
from src.autopilot.io import write_json_atomic
from src.autopilot.jobs import run_due_jobs
from src.autopilot.notifications import emit_alert
from src.autopilot.reporting import utc_now
from src.autopilot.runtime import acquire_runtime_lock, validate_config
from src.run_bot import configure_logging

LOGGER = logging.getLogger("autopilot.jobs")


def job_worker_lock_path(config: AutopilotConfig) -> Path:
    suffix = config.lock_file.suffix or ".lock"
    return config.lock_file.with_name(f"{config.lock_file.stem}.jobs{suffix}")


def job_worker_status_path(config: AutopilotConfig) -> Path:
    return config.job_state_file.with_name("job_worker_status.json")


def _worker_alert(config: AutopilotConfig, report: dict[str, Any]) -> dict[str, Any] | None:
    if not config.alerts_enabled or report.get("ok"):
        return None
    try:
        return emit_alert(
            alert_file=config.alert_file,
            state_file=config.alert_state_file,
            severity="error",
            title="autopilot scheduled job worker failed",
            detail={
                "control_error": report.get("control_error"),
                "jobs": [item for item in report.get("jobs", []) if not item.get("ok")],
            },
            cooldown_seconds=config.alert_cooldown_seconds,
            webhook_url_env=config.webhook_url_env,
        )
    except Exception as exc:  # the worker status must survive alert failures
        return {"sent": False, "error": f"{type(exc).__name__}: {exc}"}


def run_worker_once(config: AutopilotConfig) -> dict[str, Any]:
    control = load_control(config.control_file)
    report: dict[str, Any] = {
        "ok": True,
        "generated_at": utc_now(),
        "control": control,
        "jobs": [],
    }
    selectors = unknown_control_selectors(control, config)
    if control.get("control_error") or selectors:
        report["ok"] = False
        report["control_error"] = control.get("control_error") or (
            "unknown control selectors: " + json.dumps(selectors, sort_keys=True)
        )
    elif control.get("paused") or control.get("pause_jobs"):
        report["skipped"] = True
        report["reason"] = "paused"
    else:
        try:
            paused_jobs = {job.name for job in config.jobs if is_job_paused(control, job.name)}
            report["jobs"] = run_due_jobs(
                config.jobs,
                config.job_state_file,
                paused_jobs=paused_jobs,
                max_jobs_per_cycle=config.max_jobs_per_cycle,
            )
            report["ok"] = all(bool(item.get("ok")) for item in report["jobs"])
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("Scheduled job worker cycle failed")
            report["ok"] = False
            report["jobs"] = [
                {
                    "name": "scheduler",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
    alert = _worker_alert(config, report)
    if alert is not None:
        report["alert"] = alert
    write_json_atomic(job_worker_status_path(config), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scheduled autopilot jobs outside trading supervision."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sleep", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    errors = validate_config(
        config,
        require_core_products=True,
        require_core_jobs=True,
        verify_job_imports=True,
    )
    if errors:
        raise SystemExit("\n".join(errors))
    sleep_seconds = config.loop_sleep_seconds if args.sleep is None else args.sleep
    if sleep_seconds <= 0:
        raise SystemExit("sleep seconds must be positive")
    try:
        with acquire_runtime_lock(job_worker_lock_path(config)):
            while True:
                report = run_worker_once(config)
                print(
                    json.dumps(
                        {"ok": report["ok"], "status_file": str(job_worker_status_path(config))},
                        sort_keys=True,
                    )
                )
                if args.once:
                    return
                time.sleep(sleep_seconds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
