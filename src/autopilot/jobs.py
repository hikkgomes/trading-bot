"""Cadenced maintenance/research jobs for the autopilot.

Jobs are subprocess command lists, never shell strings. That keeps the runtime
simple and avoids accidentally granting shell expansion or command injection to
JSON config.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.autopilot.config import JobConfig
from src.autopilot.io import write_json_atomic
from src.autopilot.market_data import default_1m_candle_path

FAILED_JOB_RETRY_SECONDS = 15 * 60
MAX_FAILED_JOB_RETRY_SECONDS = 6 * 60 * 60
MAX_JOB_OUTPUT_CAPTURE_BYTES = 1_000_000
JOB_OUTPUT_TAIL_BYTES = 2_000
MAX_STRUCTURED_REPORT_STATUS_BYTES = 4000
MAX_STRUCTURED_REPORT_SAMPLE_ITEMS = 3
STRUCTURED_REPORT_SCALAR_KEYS = (
    "ok",
    "status",
    "reason",
    "error",
    "generated_at",
    "skipped",
    "count",
    "executable",
    "research_only",
)
STRUCTURED_REPORT_COUNT_KEYS = (
    "checks",
    "errors",
    "exports",
    "hypotheses",
    "products",
    "proposals",
    "scenarios",
    "strategies",
)
STRUCTURED_REPORT_SAMPLE_KEYS = ("errors",)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def load_job_state(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"job state must not be a symlink: {path}")
    if not path.exists():
        return {"version": 1, "jobs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"job state must be a JSON object: {path}")
    payload.setdefault("version", 1)
    payload.setdefault("jobs", {})
    if not isinstance(payload["jobs"], dict):
        raise ValueError(f"job state jobs must be a JSON object: {path}")
    return payload


def save_job_state(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"job state must not be a symlink: {path}")
    write_json_atomic(path, payload)


def job_definition_fingerprint(job: JobConfig) -> str:
    payload = {
        "command": list(job.command),
        "cadence_seconds": int(job.cadence_seconds),
        "timeout_seconds": int(job.timeout_seconds),
        "working_dir": str(job.working_dir),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def job_definition_changed(job: JobConfig, entry: dict[str, Any]) -> bool:
    previous = entry.get("definition_fingerprint")
    return bool(previous and previous != job_definition_fingerprint(job))


def effective_job_cadence_seconds(job: JobConfig, entry: dict[str, Any]) -> int:
    if entry and entry.get("last_ok") is False:
        try:
            consecutive_failures = max(1, int(entry.get("consecutive_failures") or 1))
        except (TypeError, ValueError):
            consecutive_failures = 1
        retry_seconds = min(
            MAX_FAILED_JOB_RETRY_SECONDS,
            FAILED_JOB_RETRY_SECONDS * (2 ** min(consecutive_failures - 1, 8)),
        )
        return min(job.cadence_seconds, retry_seconds)
    return job.cadence_seconds


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _now_ts(now: float | None) -> float:
    raw = time.time() if now is None else now
    try:
        timestamp = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("job scheduler now timestamp must be numeric") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("job scheduler now timestamp must be finite and non-negative")
    return timestamp


def _scheduler_start_index(state: dict[str, Any], jobs: list[JobConfig]) -> int:
    scheduler = state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    raw_index = scheduler.get("next_index", 0)
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return 0
    return index % len(jobs) if jobs else 0


def _rotated_jobs(jobs: list[JobConfig], start_index: int) -> list[tuple[int, JobConfig]]:
    if not jobs:
        return []
    ordered = list(enumerate(jobs))
    start_index = start_index % len(jobs)
    return ordered[start_index:] + ordered[:start_index]


def _set_scheduler_next_index(state: dict[str, Any], *, job_index: int, total_jobs: int) -> None:
    if total_jobs <= 0:
        return
    scheduler = state.get("scheduler") if isinstance(state.get("scheduler"), dict) else {}
    scheduler = dict(scheduler)
    scheduler["next_index"] = (job_index + 1) % total_jobs
    state["scheduler"] = scheduler


def _command_value(command: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for part in command:
        if part.startswith(prefix):
            value = part[len(prefix):]
            return value or None
    try:
        index = command.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    value = command[value_index]
    return None if value.startswith("--") else value


def _command_path_value(job: JobConfig, flag: str) -> Path | None:
    value = _command_value(list(job.command), flag)
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else job.working_dir / path


def _positive_int(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except ValueError:
        return False


def _missing_bootstrap_seed_due(job: JobConfig, entry: dict[str, Any]) -> bool:
    if entry.get("last_ok") is False:
        return False
    command = list(job.command)
    legacy_bootstrap = (
        "src.update_candles" in command
        and _positive_int(_command_value(command, "--bootstrap-days"))
    )
    native_history_bootstrap = "src.autopilot.history_bootstrap" in command
    if not legacy_bootstrap and not native_history_bootstrap:
        return False
    market = _command_value(command, "--market")
    if market not in {"spot", "futures"}:
        return False
    return not default_1m_candle_path(market=market).exists()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stale_source_due(
    *,
    job: JobConfig,
    source_key: str,
) -> bool:
    input_path = _command_path_value(job, "--input")
    output_path = _command_path_value(job, "--output")
    if input_path is None or output_path is None:
        return False
    input_payload = _load_json_object(input_path)
    input_generated_at = input_payload.get("generated_at")
    if not input_generated_at:
        return False
    output_payload = _load_json_object(output_path)
    if not output_payload:
        return True
    source = output_payload.get("source")
    if not isinstance(source, dict):
        return True
    return source.get(source_key) != input_generated_at


def _stale_research_handoff_due(job: JobConfig) -> bool:
    command = list(job.command)
    if "src.autopilot.mutation_plan" in command:
        return _stale_source_due(job=job, source_key="research_generated_at")
    if "src.autopilot.mutation_batch" in command:
        return _stale_source_due(job=job, source_key="plan_generated_at")
    return False


def _mutation_batch_marker(payload: dict[str, Any]) -> dict[str, Any] | None:
    generated_at = payload.get("generated_at")
    if not generated_at:
        return None
    count = payload.get("count")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    hypotheses = count if count is not None else summary.get("hypotheses")
    return {
        "status": payload.get("status"),
        "generated_at": generated_at,
        "hypotheses": hypotheses,
    }


def _state_mutation_batch_marker_current(state_value: Any, marker: dict[str, Any]) -> bool:
    if isinstance(state_value, str):
        try:
            state_marker = json.loads(state_value)
        except json.JSONDecodeError:
            return False
    elif isinstance(state_value, dict):
        state_marker = state_value
    else:
        return False
    return all(state_marker.get(key) == marker.get(key) for key in ("status", "generated_at", "hypotheses"))


def _mutation_batch_awaiting_research_due(job: JobConfig, job_state: dict[str, Any]) -> bool:
    command = list(job.command)
    if "src.autopilot.research_cycle" not in command or "--include-mutations" not in command:
        return False
    state_path = _command_path_value(job, "--state")
    mutation_batch_path = _command_path_value(job, "--mutation-batch")
    if state_path is None or mutation_batch_path is None:
        return False
    mutation_batch = _load_json_object(mutation_batch_path)
    marker = _mutation_batch_marker(mutation_batch)
    if marker is None:
        return False
    state = _load_json_object(state_path)
    if _state_mutation_batch_marker_current(state.get("last_mutation_batch_marker"), marker):
        return False

    job_entries = job_state.get("jobs", {}) if isinstance(job_state.get("jobs"), dict) else {}
    research_entry = job_entries.get(job.name) if isinstance(job_entries.get(job.name), dict) else {}
    mutation_entry = job_entries.get("mutation_batch") if isinstance(job_entries.get("mutation_batch"), dict) else {}
    try:
        research_started = float(research_entry.get("last_started_ts"))
        mutation_started = float(mutation_entry.get("last_started_ts"))
    except (TypeError, ValueError):
        return True
    # Avoid a tight research->plan->batch->research loop: if the scheduler
    # created the next mutation batch after this research job last ran, let the
    # normal research cadence decide when to evaluate that next generation.
    return mutation_started <= research_started


def _generated_batch_marker(payload: dict[str, Any]) -> dict[str, Any] | None:
    generated_at = payload.get("generated_at")
    if not generated_at:
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "status": "loaded",
        "generated_at": generated_at,
        "hypotheses": summary.get("hypotheses", len(payload.get("hypotheses") or [])),
        "scenarios": len(summary.get("by_space") or {}),
        "cumulative_trials": summary.get("cumulative_trials", 0),
    }


def _state_generated_batch_marker_current(state_value: Any, marker: dict[str, Any]) -> bool:
    if isinstance(state_value, str):
        try:
            state_marker = json.loads(state_value)
        except json.JSONDecodeError:
            return False
    elif isinstance(state_value, dict):
        state_marker = state_value
    else:
        return False
    return all(
        state_marker.get(key) == marker.get(key)
        for key in ("status", "generated_at", "hypotheses", "scenarios", "cumulative_trials")
    )


def _generated_batch_awaiting_research_due(job: JobConfig) -> bool:
    command = list(job.command)
    if "src.autopilot.research_cycle" not in command or "--include-generated" not in command:
        return False
    state_path = _command_path_value(job, "--state")
    generated_batch_path = _command_path_value(job, "--generated-batch")
    if state_path is None or generated_batch_path is None:
        return False
    marker = _generated_batch_marker(_load_json_object(generated_batch_path))
    if marker is None:
        return False
    state = _load_json_object(state_path)
    return not _state_generated_batch_marker_current(
        state.get("last_generated_batch_marker"),
        marker,
    )


def _blocked_export_products(payload: dict[str, Any]) -> set[str]:
    exports = payload.get("exports")
    if not isinstance(exports, list):
        return set()
    products: set[str] = set()
    for item in exports:
        if not isinstance(item, dict):
            continue
        if item.get("exported") is not False or item.get("reason") != "open_positions_block_export":
            continue
        product = item.get("product")
        if isinstance(product, str) and product:
            products.add(product)
    return products


def _state_has_no_open_positions(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _load_json_object(path)
    if not payload:
        return False
    open_positions = payload.get("open_positions")
    if open_positions is None:
        return True
    return isinstance(open_positions, dict) and not open_positions


def _open_position_blocked_export_due(job: JobConfig) -> bool:
    command = list(job.command)
    if "src.autopilot.research_cycle" not in command:
        return False
    output_path = _command_path_value(job, "--output")
    if output_path is None:
        return False
    blocked_products = _blocked_export_products(_load_json_object(output_path))
    if not blocked_products:
        return False
    for product in blocked_products:
        state_path = job.working_dir / "runtime" / f"{product}_state.json"
        if not _state_has_no_open_positions(state_path):
            return False
    return True


def job_due(job: JobConfig, job_state: dict[str, Any], now: float | None = None) -> bool:
    if not job.enabled:
        return False
    now = _now_ts(now)
    entry = job_state.get("jobs", {}).get(job.name, {})
    if job_definition_changed(job, entry):
        return True
    if _missing_bootstrap_seed_due(job, entry):
        return True
    if _stale_research_handoff_due(job):
        return True
    if _mutation_batch_awaiting_research_due(job, job_state):
        return True
    if _generated_batch_awaiting_research_due(job):
        return True
    if _open_position_blocked_export_due(job):
        return True
    if not entry or entry.get("last_started_ts") is None:
        return True
    last_started = _float_or_none(entry.get("last_started_ts"))
    if last_started is None:
        return True
    if last_started > now:
        return True
    return now - last_started >= effective_job_cadence_seconds(job, entry)


def parse_structured_stdout(stdout: str) -> dict[str, Any] | None:
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    decoder = json.JSONDecoder()
    for index in reversed([position for position, char in enumerate(stripped) if char == "{"]):
        try:
            payload, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and not stripped[index + end :].strip():
            return payload
    return None


def _structured_report_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def summarize_structured_report(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        key: payload[key]
        for key in STRUCTURED_REPORT_SCALAR_KEYS
        if key in payload
    }
    for key in ("summary", "source"):
        value = payload.get(key)
        if isinstance(value, dict):
            summary[key] = value
    for key in STRUCTURED_REPORT_COUNT_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
    for key in STRUCTURED_REPORT_SAMPLE_KEYS:
        value = payload.get(key)
        if isinstance(value, list) and value:
            summary[key] = value[:MAX_STRUCTURED_REPORT_SAMPLE_ITEMS]
    return summary


def structured_failure_detail(payload: dict[str, Any]) -> str | None:
    detail = payload.get("error") or payload.get("reason")
    if detail:
        return str(detail)
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if isinstance(first, dict):
        prefix = first.get("task") or first.get("scope") or first.get("name") or first.get("code")
        message = first.get("error") or first.get("message") or first.get("reason")
        if prefix and message:
            return f"{prefix}: {message}"
        if message:
            return str(message)
    return str(first)


def structured_error_state(payload: dict[str, Any]) -> dict[str, Any]:
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return {}
    try:
        errors_count = int(payload.get("errors_count", len(errors)))
    except (TypeError, ValueError):
        errors_count = len(errors)
    if errors_count < 0:
        errors_count = len(errors)
    return {
        "last_structured_errors_count": errors_count,
        "last_structured_errors": errors[:MAX_STRUCTURED_REPORT_SAMPLE_ITEMS],
    }


def _structured_report_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    size_bytes = _structured_report_size(payload)
    if size_bytes <= MAX_STRUCTURED_REPORT_STATUS_BYTES:
        return {"structured_report": payload}
    return {
        "structured_report_summary": summarize_structured_report(payload),
        "structured_report_truncated": True,
        "structured_report_bytes": size_bytes,
    }


def _decode_output(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _read_output_capture(handle) -> tuple[str, str, bool, int]:
    handle.flush()
    handle.seek(0, 2)
    size = handle.tell()
    truncated = size > MAX_JOB_OUTPUT_CAPTURE_BYTES
    if truncated:
        full = ""
        handle.seek(max(0, size - JOB_OUTPUT_TAIL_BYTES))
        tail = _decode_output(handle.read())
    else:
        handle.seek(0)
        full = _decode_output(handle.read())
        tail = full[-JOB_OUTPUT_TAIL_BYTES:]
    return full, tail, truncated, size


def run_job(job: JobConfig) -> dict[str, Any]:
    if not job.command:
        raise ValueError(f"job {job.name}: command cannot be empty")
    started_ts = time.time()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            result = subprocess.run(
                job.command,
                cwd=job.working_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=job.timeout_seconds,
                check=False,
            )
            stdout, stdout_tail, stdout_truncated, stdout_bytes = _read_output_capture(stdout_file)
            _, stderr_tail, stderr_truncated, stderr_bytes = _read_output_capture(stderr_file)
            structured_report = None if stdout_truncated else parse_structured_stdout(stdout)
            structured_ok = structured_report.get("ok") if structured_report else None
            ok = result.returncode == 0 and structured_ok is not False
            failure_detail = None
            if structured_ok is False:
                failure_detail = structured_failure_detail(structured_report) or "structured report failed"
            return {
                "name": job.name,
                "ok": ok,
                "returncode": result.returncode,
                "started_at": utc_now(),
                "started_ts": started_ts,
                "duration_seconds": round(time.time() - started_ts, 3),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                **({"stdout_truncated": True, "stdout_bytes": stdout_bytes} if stdout_truncated else {}),
                **({"stderr_truncated": True, "stderr_bytes": stderr_bytes} if stderr_truncated else {}),
                **(_structured_report_status_payload(structured_report) if structured_report is not None else {}),
                **({"error": str(failure_detail)} if failure_detail else {}),
            }
        except subprocess.TimeoutExpired:
            _, stdout_tail, stdout_truncated, stdout_bytes = _read_output_capture(stdout_file)
            _, stderr_tail, stderr_truncated, stderr_bytes = _read_output_capture(stderr_file)
            return {
                "name": job.name,
                "ok": False,
                "returncode": None,
                "started_at": utc_now(),
                "started_ts": started_ts,
                "duration_seconds": round(time.time() - started_ts, 3),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                **({"stdout_truncated": True, "stdout_bytes": stdout_bytes} if stdout_truncated else {}),
                **({"stderr_truncated": True, "stderr_bytes": stderr_bytes} if stderr_truncated else {}),
                "error": f"timed out after {job.timeout_seconds}s",
            }


def run_due_jobs(
    jobs: list[JobConfig],
    state_path: Path,
    now: float | None = None,
    paused_jobs: set[str] | None = None,
    max_jobs_per_cycle: int | None = None,
) -> list[dict[str, Any]]:
    if max_jobs_per_cycle is not None and max_jobs_per_cycle <= 0:
        raise ValueError("max_jobs_per_cycle must be positive")
    state = load_job_state(state_path)
    state.setdefault("jobs", {})
    results: list[dict[str, Any]] = []
    paused_jobs = paused_jobs or set()
    scheduler_now = _now_ts(now) if now is not None else None
    executed_jobs = 0
    start_index = _scheduler_start_index(state, jobs)
    for job_index, job in _rotated_jobs(jobs, start_index):
        if not job.enabled:
            continue
        if job.name in paused_jobs:
            results.append(
                {
                    "name": job.name,
                    "ok": True,
                    "skipped": True,
                    "reason": "paused",
                    "started_at": utc_now(),
                    "started_ts": now if now is not None else time.time(),
                }
            )
            continue
        if not job_due(job, state, now=now):
            continue
        if max_jobs_per_cycle is not None and executed_jobs >= max_jobs_per_cycle:
            deferred_at = utc_now()
            deferred_ts = now if now is not None else time.time()
            previous_entry = state["jobs"].get(job.name, {})
            previous_entry = previous_entry if isinstance(previous_entry, dict) else {}
            try:
                previous_deferrals = int(previous_entry.get("consecutive_deferrals") or 0)
            except (TypeError, ValueError):
                previous_deferrals = 0
            state["jobs"][job.name] = {
                **previous_entry,
                "last_deferred_at": deferred_at,
                "last_deferred_ts": deferred_ts,
                "last_deferred_reason": "cycle_job_limit",
                "consecutive_deferrals": previous_deferrals + 1,
                "definition_fingerprint": job_definition_fingerprint(job),
            }
            save_job_state(state_path, state)
            results.append(
                {
                    "name": job.name,
                    "ok": True,
                    "skipped": True,
                    "reason": "cycle_job_limit",
                    "started_at": deferred_at,
                    "started_ts": deferred_ts,
                }
            )
            continue
        result = run_job(job)
        executed_jobs += 1
        if scheduler_now is not None:
            result["started_ts"] = scheduler_now
        structured_report = result.get("structured_report")
        if not isinstance(structured_report, dict):
            structured_report = result.get("structured_report_summary")
        if not isinstance(structured_report, dict):
            structured_report = {}
        previous_entry = state["jobs"].get(job.name, {})
        try:
            previous_failures = int(previous_entry.get("consecutive_failures") or 0)
        except (TypeError, ValueError):
            previous_failures = 0
        consecutive_failures = 0 if result["ok"] else previous_failures + 1
        state["jobs"][job.name] = {
            "last_started_at": result["started_at"],
            "last_started_ts": result["started_ts"],
            "last_ok": result["ok"],
            "last_returncode": result["returncode"],
            "last_duration_seconds": result["duration_seconds"],
            "consecutive_failures": consecutive_failures,
            "definition_fingerprint": job_definition_fingerprint(job),
            **(
                {"last_stdout_truncated": True, "last_stdout_bytes": result["stdout_bytes"]}
                if result.get("stdout_truncated")
                else {}
            ),
            **(
                {"last_stderr_truncated": True, "last_stderr_bytes": result["stderr_bytes"]}
                if result.get("stderr_truncated")
                else {}
            ),
            **({"last_error": result["error"]} if result.get("error") else {}),
            **(
                {
                    "last_reason": structured_report["reason"],
                }
                if structured_report.get("reason")
                else {}
            ),
            **structured_error_state(structured_report),
        }
        results.append(result)
        _set_scheduler_next_index(state, job_index=job_index, total_jobs=len(jobs))
        save_job_state(state_path, state)
    return results
