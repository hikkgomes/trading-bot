import json
import sys

from src.autopilot.config import AutopilotConfig, JobConfig
from src.autopilot.job_worker import (
    job_worker_lock_path,
    job_worker_status_path,
    run_worker_once,
)


def worker_config(tmp_path) -> AutopilotConfig:
    return AutopilotConfig(
        control_file=tmp_path / "control.json",
        status_file=tmp_path / "supervisor_status.json",
        lock_file=tmp_path / "autopilot.lock",
        job_state_file=tmp_path / "job_state.json",
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
        alerts_enabled=False,
        jobs=[
            JobConfig(
                name="smoke",
                enabled=True,
                command=[sys.executable, "-c", "print('ok')"],
                cadence_seconds=60,
                timeout_seconds=5,
                working_dir=tmp_path,
            )
        ],
    )


def test_worker_uses_separate_lock_and_status_paths(tmp_path):
    config = worker_config(tmp_path)

    assert job_worker_lock_path(config) == tmp_path / "autopilot.jobs.lock"
    assert job_worker_status_path(config) == tmp_path / "job_worker_status.json"
    assert job_worker_lock_path(config) != config.lock_file
    assert job_worker_status_path(config) != config.status_file


def test_worker_runs_due_job_and_writes_own_status(tmp_path):
    config = worker_config(tmp_path)

    report = run_worker_once(config)

    assert report["ok"] is True
    assert report["jobs"][0]["name"] == "smoke"
    assert report["jobs"][0]["ok"] is True
    saved = json.loads(job_worker_status_path(config).read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert not config.status_file.exists()


def test_worker_pause_keeps_jobs_responsive_without_running_subprocess(tmp_path):
    config = worker_config(tmp_path)
    config.control_file.write_text(json.dumps({"pause_jobs": True}), encoding="utf-8")

    report = run_worker_once(config)

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["reason"] == "paused"
    assert report["jobs"] == []
    assert not config.job_state_file.exists()


def test_worker_malformed_control_fails_closed(tmp_path):
    config = worker_config(tmp_path)
    config.control_file.write_text("{", encoding="utf-8")

    report = run_worker_once(config)

    assert report["ok"] is False
    assert "JSONDecodeError" in report["control_error"]
    assert report["jobs"] == []
