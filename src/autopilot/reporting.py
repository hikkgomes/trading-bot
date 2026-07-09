"""Status reporting helpers for the autopilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.autopilot.approvals import is_valid_approval_actor, is_valid_revocation_reason
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, load_config
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.jobs import effective_job_cadence_seconds, job_due
from src.autopilot.market_data import (
    build_indicator_feature_statuses,
    build_market_data_statuses,
    required_indicator_features_by_market,
)
from src.autopilot.regime_data import build_regime_data_statuses
from src.autopilot.testnet_rehearsal import summarize_testnet_rehearsal_report

LOGGER = logging.getLogger("autopilot.reporting")

BROKER_EXIT_NUMERIC_FIELDS = (
    "broker_exit_qty",
    "broker_exit_price",
    "broker_exit_fee",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["generated_at"] = utc_now()
    write_json_atomic(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "_load_error": {
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        }
    if not isinstance(payload, dict):
        return {
            "_load_error": {
                "path": str(path),
                "error": f"TypeError: expected JSON object, got {type(payload).__name__}",
            }
        }
    return payload


def _runtime_load_errors(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for name, payload in payloads.items():
        load_error = payload.get("_load_error") if isinstance(payload, dict) else None
        if isinstance(load_error, dict):
            errors.append({"name": name, **load_error})
    return errors


def _runtime_shape_errors(
    *,
    status: dict[str, Any],
    job_state: dict[str, Any],
    approval_ledger: dict[str, Any],
    config: AutopilotConfig,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def add_collection_errors(name: str, path: Path, payload: dict[str, Any], key: str, expected: type) -> None:
        if payload.get("_load_error") or key not in payload:
            return
        value = payload.get(key)
        if not isinstance(value, expected):
            errors.append(
                {
                    "name": name,
                    "path": str(path),
                    "field": key,
                    "error": f"expected {expected.__name__}, got {type(value).__name__}",
                }
            )
            return
        if expected is not list:
            return
        invalid_entries = [
            {"index": index, "type": type(item).__name__}
            for index, item in enumerate(value)
            if not isinstance(item, dict)
        ]
        if invalid_entries:
            errors.append(
                {
                    "name": name,
                    "path": str(path),
                    "field": key,
                    "error": "expected list entries to be JSON objects",
                    "invalid_entries": invalid_entries[:10],
                }
            )

    add_collection_errors("status", config.status_file, status, "products", list)
    add_collection_errors("status", config.status_file, status, "jobs", list)
    add_collection_errors("job_state", config.job_state_file, job_state, "jobs", dict)
    add_collection_errors("approval_ledger", config.approval_ledger, approval_ledger, "approvals", dict)
    return errors


def _dict_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _status_heartbeat(status: dict[str, Any], config: AutopilotConfig, now_ts: float | None = None) -> dict[str, Any]:
    generated_at = status.get("generated_at")
    generated_ts = _parse_timestamp(generated_at)
    now_ts = now_ts if now_ts is not None else dt.datetime.now(dt.timezone.utc).timestamp()
    limit_seconds = max(float(config.loop_sleep_seconds) * 3.0, 300.0)
    age_seconds = None
    fresh = None
    reason = None
    if generated_ts is not None:
        if generated_ts > now_ts:
            fresh = False
            reason = "future_generated_at"
        else:
            age_seconds = now_ts - generated_ts
            fresh = age_seconds <= limit_seconds
            if not fresh:
                reason = "stale"
    heartbeat = {
        "generated_at": generated_at,
        "fresh": fresh,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "limit_seconds": round(limit_seconds, 3),
    }
    if reason:
        heartbeat["reason"] = reason
    return heartbeat


def _trade_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "trades": 0,
        "wins": 0,
        "net_return_sum": 0.0,
        "sized_return_sum": 0.0,
        "last_exit_time": None,
        "invalid_rows": 0,
        "numeric_errors": [],
    }
    if not path.exists():
        return summary
    invalid_lines: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            summary["trades"] += 1
            net_return = _trade_float(row, "net_return", line_number, summary, invalid_lines, required=True)
            sized_return = _trade_float(row, "sized_return", line_number, summary, invalid_lines, required=True)
            broker_fields_present = any((row.get(field) or "") != "" for field in BROKER_EXIT_NUMERIC_FIELDS)
            if broker_fields_present:
                _trade_float(
                    row,
                    "broker_exit_qty",
                    line_number,
                    summary,
                    invalid_lines,
                    required=True,
                    positive=True,
                )
                _trade_float(
                    row,
                    "broker_exit_price",
                    line_number,
                    summary,
                    invalid_lines,
                    required=True,
                    positive=True,
                )
                _trade_float(
                    row,
                    "broker_exit_fee",
                    line_number,
                    summary,
                    invalid_lines,
                    required=True,
                    non_negative=True,
                )
            summary["net_return_sum"] += net_return
            summary["sized_return_sum"] += sized_return
            summary["wins"] += int(net_return > 0)
            summary["last_exit_time"] = row.get("exit_time") or summary["last_exit_time"]
    summary["invalid_rows"] = len(invalid_lines)
    if summary["invalid_rows"]:
        summary["issue"] = f"trade log has {summary['invalid_rows']} row(s) with invalid numeric fields"
    trades = summary["trades"]
    summary["win_rate"] = (summary["wins"] / trades) if trades else None
    return summary


def _trade_float(
    row: dict[str, str],
    field: str,
    line_number: int,
    summary: dict[str, Any],
    invalid_lines: set[int],
    *,
    required: bool = False,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    raw = row.get(field)
    if raw in (None, ""):
        if required:
            invalid_lines.add(line_number)
            summary["numeric_errors"].append(
                {
                    "line": line_number,
                    "field": field,
                    "value": raw,
                }
            )
        return 0.0
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        invalid_lines.add(line_number)
        summary["numeric_errors"].append(
            {
                "line": line_number,
                "field": field,
                "value": raw,
            }
        )
        return 0.0
    if (
        not math.isfinite(parsed)
        or (positive and parsed <= 0)
        or (non_negative and parsed < 0)
    ):
        invalid_lines.add(line_number)
        summary["numeric_errors"].append(
            {
                "line": line_number,
                "field": field,
                "value": raw,
            }
        )
        return 0.0
    return parsed


def _state_error_issue(state_errors: Any) -> str:
    if not state_errors:
        return ""
    if not isinstance(state_errors, list):
        return f"state_errors malformed: expected list, got {type(state_errors).__name__}"
    first_error = next((item for item in state_errors if isinstance(item, dict)), None)
    if first_error is None:
        return "state error: malformed state error entry"
    field = first_error.get("field") or "state"
    error = first_error.get("error") or "invalid"
    suffix = ""
    if len(state_errors) > 1:
        suffix = f" (+{len(state_errors) - 1} more)"
    return f"state error {field}: {error}{suffix}"


def _float_report_value(value: Any, *, field: str, invalid_reasons: list[str]) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        invalid_reasons.append(f"invalid job state {field}: {value!r}")
        return None
    if parsed < 0:
        invalid_reasons.append(f"invalid job state {field}: {value!r}")
        return None
    return parsed


def _int_report_value(value: Any, *, field: str, invalid_reasons: list[str], default: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        invalid_reasons.append(f"invalid job state {field}: {value!r}")
        return default
    if parsed < 0:
        invalid_reasons.append(f"invalid job state {field}: {value!r}")
        return default
    return parsed


def _market_data_job_market(job: Any) -> str | None:
    command = list(getattr(job, "command", []) or [])
    if "src.update_candles" not in command:
        return None
    return _command_value(command, "--market")


def _scheduled_job_summary(
    config: AutopilotConfig,
    now_ts: float | None = None,
    *,
    job_state: dict[str, Any] | None = None,
    market_data_by_market: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    state = job_state if job_state is not None else _load_json(config.job_state_file)
    entries = state.get("jobs", {}) if isinstance(state.get("jobs", {}), dict) else {}
    now_ts = now_ts if now_ts is not None else dt.datetime.now(dt.timezone.utc).timestamp()
    jobs = []
    for job in config.jobs:
        entry = entries.get(job.name, {}) if isinstance(entries.get(job.name, {}), dict) else {}
        last_started_ts = entry.get("last_started_ts")
        age_seconds = None
        invalid_state_reasons: list[str] = []
        if last_started_ts is not None:
            try:
                parsed_started_ts = float(last_started_ts)
            except (TypeError, ValueError):
                invalid_state_reasons.append(f"invalid job state last_started_ts: {last_started_ts!r}")
            else:
                if not math.isfinite(parsed_started_ts) or parsed_started_ts < 0:
                    invalid_state_reasons.append(f"invalid job state last_started_ts: {last_started_ts!r}")
                elif parsed_started_ts > now_ts:
                    invalid_state_reasons.append(f"invalid job state future last_started_ts: {last_started_ts!r}")
                else:
                    age_seconds = now_ts - parsed_started_ts
        last_duration_seconds = _float_report_value(
            entry.get("last_duration_seconds"),
            field="last_duration_seconds",
            invalid_reasons=invalid_state_reasons,
        )
        consecutive_failures = _int_report_value(
            entry.get("consecutive_failures"),
            field="consecutive_failures",
            invalid_reasons=invalid_state_reasons,
        )
        last_stdout_bytes = _int_report_value(
            entry.get("last_stdout_bytes"),
            field="last_stdout_bytes",
            invalid_reasons=invalid_state_reasons,
            default=0,
        )
        last_stderr_bytes = _int_report_value(
            entry.get("last_stderr_bytes"),
            field="last_stderr_bytes",
            invalid_reasons=invalid_state_reasons,
            default=0,
        )
        structured_errors_count = _int_report_value(
            entry.get("last_structured_errors_count"),
            field="last_structured_errors_count",
            invalid_reasons=invalid_state_reasons,
        )
        structured_errors = entry.get("last_structured_errors")
        if not isinstance(structured_errors, list):
            structured_errors = []
        if not job.enabled:
            status = "disabled"
        elif entry.get("last_deferred_reason") == "cycle_job_limit" and job_due(job, state, now=now_ts):
            status = "deferred"
        elif not entry:
            status = "never_run"
        elif entry.get("last_ok"):
            status = "ok"
        else:
            status = "fail"
        last_error = entry.get("last_error")
        invalid_state_reason = "; ".join(invalid_state_reasons)
        last_reason = entry.get("last_reason") or invalid_state_reason or None
        job_market = _market_data_job_market(job)
        if status == "fail" and job_market and market_data_by_market:
            market_status = market_data_by_market.get(job_market) or {}
            if market_status.get("ok"):
                status = "recovered"
                last_error = None
                last_reason = "last failure resolved; current market data is ready"
        effective_cadence = effective_job_cadence_seconds(job, entry)
        due = job_due(job, state, now=now_ts)
        jobs.append(
            {
                "name": job.name,
                "enabled": job.enabled,
                "status": status,
                "due": due,
                "cadence_seconds": job.cadence_seconds,
                "effective_cadence_seconds": effective_cadence,
                "timeout_seconds": job.timeout_seconds,
                "last_started_at": entry.get("last_started_at"),
                "last_ok": entry.get("last_ok"),
                "last_returncode": entry.get("last_returncode"),
                "last_duration_seconds": last_duration_seconds,
                "consecutive_failures": consecutive_failures,
                "consecutive_deferrals": _int_report_value(
                    entry.get("consecutive_deferrals"),
                    field="consecutive_deferrals",
                    invalid_reasons=invalid_state_reasons,
                    default=0,
                ),
                "last_deferred_at": entry.get("last_deferred_at"),
                "last_deferred_reason": entry.get("last_deferred_reason"),
                "last_stdout_truncated": entry.get("last_stdout_truncated") is True,
                "last_stdout_bytes": last_stdout_bytes,
                "last_stderr_truncated": entry.get("last_stderr_truncated") is True,
                "last_stderr_bytes": last_stderr_bytes,
                "last_reason": last_reason,
                "last_error": last_error,
                "last_structured_errors_count": structured_errors_count,
                "last_structured_errors": structured_errors,
                "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            }
        )
    return jobs


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


def _promotion_review_summaries(config: AutopilotConfig, *, now_ts: float | None = None) -> list[dict[str, Any]]:
    now_ts = now_ts if now_ts is not None else dt.datetime.now(dt.timezone.utc).timestamp()
    summaries = []
    for job in config.jobs:
        if "promotion" not in job.name:
            continue
        output = _command_value(job.command, "--output-json")
        product = _command_value(job.command, "--product")
        path = Path(output) if output else None
        if path is not None and not path.is_absolute():
            path = job.working_dir / path
        payload = _load_json(path) if path is not None else {}
        generated_at = payload.get("generated_at")
        generated_ts = _parse_timestamp(generated_at)
        max_age_seconds = max(float(job.cadence_seconds) * 2.0, 3600.0)
        age_seconds = None
        fresh = None
        reason = payload.get("reason")
        if generated_ts is not None:
            if generated_ts > now_ts:
                fresh = False
                reason = "future_generated_at"
            else:
                age_seconds = now_ts - generated_ts
                fresh = age_seconds <= max_age_seconds
                if not fresh and not reason:
                    reason = "stale"
        strategies = payload.get("strategies") if isinstance(payload.get("strategies"), list) else []
        recommendations: dict[str, int] = {}
        approval_commands = []
        for item in strategies:
            if not isinstance(item, dict):
                continue
            recommendation = str(item.get("recommendation") or "unknown")
            recommendations[recommendation] = recommendations.get(recommendation, 0) + 1
            if recommendation == "needs_approval" and item.get("approval_command"):
                approval_commands.append(str(item["approval_command"]))
        summaries.append(
            {
                "job": job.name,
                "product": product or (payload.get("product") or {}).get("name"),
                "enabled": job.enabled,
                "path": str(path) if path is not None else None,
                "exists": bool(path and path.exists()),
                "status": payload.get("status") or ("ready" if strategies else "missing"),
                "generated_at": generated_at,
                "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
                "max_age_seconds": round(max_age_seconds, 3),
                "fresh": fresh,
                "reason": reason,
                "strategies": len(strategies),
                "recommendations": recommendations,
                "needs_approval": recommendations.get("needs_approval", 0),
                "approval_commands": approval_commands[:3],
            }
        )
    return summaries


def _approval_event_ts(record: dict[str, Any]) -> float:
    timestamp = record.get("event_at")
    parsed = _parse_timestamp(timestamp)
    return parsed if parsed is not None else -1.0


def _approval_product_label(product: dict[str, Any] | None) -> str:
    if not isinstance(product, dict) or not product:
        return "unscoped"
    name = product.get("name") or "unknown"
    symbol = product.get("symbol") or "unknown"
    return f"{name}/{str(symbol).upper()}"


def _approval_event_record(
    *,
    fingerprint: str,
    entry: dict[str, Any],
    event: str,
    event_at: Any,
    actor: Any,
) -> dict[str, Any]:
    fingerprint_text = str(fingerprint)
    fingerprint_short = fingerprint_text.split(":")[-1][:12]
    product = entry.get("product")
    return {
        "event": event,
        "event_at": event_at,
        "actor": actor,
        "status": entry.get("status"),
        "strategy_id": entry.get("strategy_id"),
        "fingerprint": fingerprint_text,
        "fingerprint_short": fingerprint_short,
        "product": product,
        "product_label": _approval_product_label(product),
        "artifact_path": entry.get("artifact_path"),
        "revocation_reason": entry.get("revocation_reason"),
        "audit_reasons": entry.get("audit_reasons"),
    }


def _revocation_audit_reasons(entry: dict[str, Any]) -> list[str]:
    reasons = []
    if not is_valid_approval_actor(entry.get("revoked_by")):
        reasons.append("invalid_revoked_by")
    if not is_valid_revocation_reason(entry.get("revocation_reason")):
        reasons.append("missing_revocation_reason")
    return reasons


def _approval_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    approvals = ledger.get("approvals", {}) if isinstance(ledger.get("approvals"), dict) else {}
    counts: dict[str, int] = {}
    by_product: dict[str, dict[str, int]] = {}
    events: list[dict[str, Any]] = []
    for fingerprint, raw_entry in approvals.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        status = str(entry.get("status") or "unknown")
        if status == "approved" and not is_valid_approval_actor(entry.get("approved_by")):
            status = "invalid_actor"
        elif status == "approved" and entry.get("fingerprint") != str(fingerprint):
            status = "fingerprint_mismatch"
        elif status == "revoked" and (audit_reasons := _revocation_audit_reasons(entry)):
            status = "invalid_revocation_audit"
            entry = dict(entry)
            entry["audit_reasons"] = audit_reasons
        counts[status] = counts.get(status, 0) + 1
        product_label = _approval_product_label(entry.get("product"))
        product_counts = by_product.setdefault(product_label, {})
        product_counts[status] = product_counts.get(status, 0) + 1
        if entry.get("revoked_at"):
            event = "revoked"
            event_at = entry.get("revoked_at")
            actor = entry.get("revoked_by")
        else:
            event = "approved"
            event_at = entry.get("approved_at")
            actor = entry.get("approved_by")
        events.append(
            _approval_event_record(
                fingerprint=str(fingerprint),
                entry=entry,
                event=event,
                event_at=event_at,
                actor=actor,
            )
        )
        for history_item in entry.get("history", []) if isinstance(entry.get("history"), list) else []:
            if not isinstance(history_item, dict):
                continue
            history_event = str(history_item.get("event") or "changed")
            history_event_at = history_item.get("event_at")
            history_actor = history_item.get("actor")
            history_entry = dict(entry)
            history_status = history_item.get("status", entry.get("status"))
            history_entry.update(
                {
                    "status": history_status,
                    "strategy_id": history_item.get("strategy_id", entry.get("strategy_id")),
                    "product": history_item.get("product", entry.get("product")),
                    "artifact_path": history_item.get("artifact_path", entry.get("artifact_path")),
                    "revocation_reason": history_item.get("revocation_reason"),
                }
            )
            if history_status == "revoked":
                history_reasons = _revocation_audit_reasons(history_entry)
                if history_reasons:
                    history_entry["audit_reasons"] = history_reasons
            events.append(
                _approval_event_record(
                    fingerprint=str(fingerprint),
                    entry=history_entry,
                    event=history_event,
                    event_at=history_event_at,
                    actor=history_actor,
                )
            )
    events.sort(key=_approval_event_ts, reverse=True)
    return {
        "total": len(approvals),
        "counts": counts,
        "approved": counts.get("approved", 0),
        "revoked": counts.get("revoked", 0),
        "by_product": by_product,
        "latest_event": events[0] if events else None,
        "recent_events": events[:10],
    }


def _payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _compact_artifact_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    keep_keys: tuple[str, ...],
    count_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not payload:
        return payload
    if payload.get("_load_error"):
        return payload
    compacted = {key: payload[key] for key in keep_keys if key in payload}
    for key in count_keys:
        value = payload.get(key)
        if isinstance(value, list):
            compacted[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            compacted[f"{key}_count"] = len(value)
    compacted.update(
        {
            "compacted": True,
            "artifact": str(path),
            "raw_size_bytes": _payload_size_bytes(payload),
        }
    )
    return compacted


def _testnet_rehearsal_status(config: AutopilotConfig, *, now_ts: float | None = None) -> dict[str, Any]:
    required_products = [
        product
        for product in config.products
        if product.enabled and product.require_testnet_rehearsal
    ]
    if required_products:
        product = required_products[0]
        path = product.testnet_rehearsal_report or Path("runtime/testnet_rehearsal_report.json")
        status = summarize_testnet_rehearsal_report(
            path,
            max_age_seconds=product.testnet_rehearsal_max_age_seconds,
            now_ts=now_ts,
            expected_product=product,
        )
        status["required"] = True
        status["required_by"] = [item.name for item in required_products]
        if status.get("product") is None:
            status["product"] = product.name
        return status
    status = summarize_testnet_rehearsal_report(now_ts=now_ts)
    status["required"] = False
    status["required_by"] = []
    return status


def build_operator_report(config: AutopilotConfig, *, now_ts: float | None = None) -> dict[str, Any]:
    status = _load_json(config.status_file)
    approval_ledger = _load_json(config.approval_ledger)
    approval_summary = _approval_summary(approval_ledger)
    research_smoke = _load_json(config.research_smoke_file)
    strategy_smoke = _load_json(config.strategy_smoke_file)
    research_cycle = _load_json(config.research_cycle_file)
    incubation_candidates = _load_json(config.incubation_candidates_file)
    mutation_plan = _load_json(config.mutation_plan_file)
    mutation_batch = _load_json(config.mutation_batch_file)
    artifact_hygiene = _load_json(config.artifact_hygiene_file)
    backup_report = _load_json(config.backup_report_file)
    job_state = _load_json(config.job_state_file)
    loaded_payloads = {
        "status": status,
        "job_state": job_state,
        "approval_ledger": approval_ledger,
        "research_smoke": research_smoke,
        "strategy_smoke": strategy_smoke,
        "research_cycle": research_cycle,
        "incubation_candidates": incubation_candidates,
        "mutation_plan": mutation_plan,
        "mutation_batch": mutation_batch,
        "artifact_hygiene": artifact_hygiene,
        "backup_report": backup_report,
    }
    testnet_rehearsal = _testnet_rehearsal_status(config, now_ts=now_ts)
    product_statuses = {
        item.get("product", {}).get("name"): item
        for item in _dict_entries(status.get("products"))
    }
    products = []
    for product in config.products:
        cycle = product_statuses.get(product.name, {})
        products.append(
            {
                "name": product.name,
                "objective": product.objective,
                "market": product.market,
                "mode": product.execution_mode,
                "enabled": product.enabled,
                "cycle_ok": cycle.get("ok"),
                "action": cycle.get("action"),
                "skipped": cycle.get("skipped", False),
                "reason": cycle.get("reason"),
                "error": cycle.get("error"),
                "close_error": cycle.get("close_error"),
                "detail": cycle.get("detail"),
                "broker": cycle.get("broker"),
                "flattened": cycle.get("flattened"),
                "fill": cycle.get("fill"),
                "spot_step_aside": cycle.get("spot_step_aside"),
                "local_state": cycle.get("local_state"),
                "position_before": cycle.get("position_before"),
                "position_after": cycle.get("position_after"),
                "position_after_error": cycle.get("position_after_error"),
                "position_after_attempt": cycle.get("position_after_attempt"),
                "position_after_attempt_error": cycle.get("position_after_attempt_error"),
                "cycle_errors": cycle.get("cycle_errors", []),
                "state_errors": cycle.get("state_errors", []),
                "equity": cycle.get("equity"),
                "open_positions": cycle.get("open_positions"),
                "open_position_details": cycle.get("open_position_details", []),
                "trade_summary": _trade_summary(product.trade_log),
                "require_testnet_rehearsal": product.require_testnet_rehearsal,
                "testnet_rehearsal_report": (
                    str(product.testnet_rehearsal_report)
                    if product.testnet_rehearsal_report is not None
                    else None
                ),
            }
        )
    markets = sorted({product.market for product in config.products}) or ["futures"]
    market_data_by_market = build_market_data_statuses(markets)
    indicator_features_by_market = build_indicator_feature_statuses(
        markets,
        required_features_by_market=required_indicator_features_by_market(markets, jobs=config.jobs),
    )
    aggregate_market_data = {
        "ok": all(item.get("ok") for item in market_data_by_market.values()),
        "markets": market_data_by_market,
    }
    aggregate_indicator_features = {
        "ok": all(item.get("ok") for item in indicator_features_by_market.values()),
        "markets": indicator_features_by_market,
    }
    regime_data = build_regime_data_statuses(config.jobs)
    research_cycle_view = _compact_artifact_payload(
        research_cycle,
        path=config.research_cycle_file,
        keep_keys=(
            "ok",
            "generated_at",
            "skipped",
            "reason",
            "state_recovered",
            "state_error",
            "summary",
            "market_data",
            "mutation_batch",
            "last_market_timestamp",
            "last_mutation_batch_marker",
            "incubation_review",
        ),
        count_keys=("scenarios", "exports"),
    )
    incubation_candidates_view = _compact_artifact_payload(
        incubation_candidates,
        path=config.incubation_candidates_file,
        keep_keys=(
            "ok",
            "generated_at",
            "schema",
            "research_only",
            "executable",
            "paper_trade_allowed",
            "live_allowed",
            "promotion_eligible",
            "reason",
            "summary",
        ),
        count_keys=("products",),
    )
    mutation_plan_view = _compact_artifact_payload(
        mutation_plan,
        path=config.mutation_plan_file,
        keep_keys=("ok", "generated_at", "status", "source", "summary"),
        count_keys=("proposals", "skipped_scenarios"),
    )
    mutation_batch_view = _compact_artifact_payload(
        mutation_batch,
        path=config.mutation_batch_file,
        keep_keys=(
            "ok",
            "generated_at",
            "status",
            "error",
            "source",
            "research_only",
            "executable",
            "paper_trade_allowed",
            "promotion_allowed",
            "live_allowed",
            "requires_full_validation_before_export",
            "count",
            "families",
            "summary",
        ),
        count_keys=("hypotheses", "mutation_metadata", "skipped"),
    )
    return {
        "generated_at": utc_now(),
        "status_file": str(config.status_file),
        "status_generated_at": status.get("generated_at"),
        "status_heartbeat": _status_heartbeat(status, config, now_ts=now_ts),
        "runtime_load_errors": _runtime_load_errors(loaded_payloads),
        "runtime_shape_errors": _runtime_shape_errors(
            status=status,
            job_state=job_state,
            approval_ledger=approval_ledger,
            config=config,
        ),
        "ok": status.get("ok"),
        "control": status.get("control", {}),
        "control_error": status.get("control_error") or (status.get("control") or {}).get("control_error"),
        "control_clear": status.get("control_clear", []),
        "unknown_control_selectors": status.get("unknown_control_selectors"),
        "approval_count": approval_summary["total"],
        "approval_summary": approval_summary,
        "market_data": aggregate_market_data,
        "market_data_by_market": market_data_by_market,
        "indicator_features": aggregate_indicator_features,
        "indicator_features_by_market": indicator_features_by_market,
        "regime_data": {
            "ok": all(item.get("available") is not False for item in regime_data),
            "datasets": regime_data,
        },
        "research_smoke": research_smoke,
        "strategy_smoke": strategy_smoke,
        "research_cycle": research_cycle_view,
        "incubation_candidates": incubation_candidates_view,
        "mutation_plan": mutation_plan_view,
        "mutation_batch": mutation_batch_view,
        "promotion_reviews": _promotion_review_summaries(config, now_ts=now_ts),
        "artifact_hygiene": artifact_hygiene,
        "backup_report": backup_report,
        "testnet_rehearsal": testnet_rehearsal,
        "products": products,
        "scheduled_jobs": _scheduled_job_summary(
            config,
            now_ts=now_ts,
            job_state=job_state,
            market_data_by_market=market_data_by_market,
        ),
        "jobs": _dict_entries(status.get("jobs")),
        "data_update": status.get("data_update"),
        "reporting": status.get("reporting"),
        "alert": status.get("alert"),
        "readiness_alert": status.get("readiness_alert"),
        "research_handoff_alert": status.get("research_handoff_alert"),
        "research_progress_alert": status.get("research_progress_alert"),
        "testnet_rehearsal_alert": status.get("testnet_rehearsal_alert"),
        "promotion_alert": status.get("promotion_alert"),
    }


def _fmt_bool(value: Any) -> str:
    if value is None:
        return "unknown"
    return "ok" if bool(value) else "fail"


def _fmt_age(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{float(seconds):.0f}s"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _truncate(value: str, limit: int = 180) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _alert_detail(alert: dict[str, Any] | None) -> str:
    if not alert:
        return "none"
    state_error = alert.get("state_error")
    suffix = f", state error: {_truncate(str(state_error), 80)}" if state_error else ""
    if alert.get("sent"):
        return f"sent{suffix}"
    reason = alert.get("reason") or alert.get("error")
    return f"not sent ({reason}{suffix})" if reason else f"not sent{suffix}"


def _structured_error_detail(error: Any) -> str:
    if isinstance(error, dict):
        prefix = error.get("task") or error.get("scope") or error.get("name") or error.get("code")
        message = error.get("error") or error.get("message") or error.get("reason")
        if prefix and message:
            return f"{prefix}: {message}"
        if message:
            return str(message)
    return str(error)


def _scheduled_job_structured_errors_detail(job: dict[str, Any]) -> str:
    errors = job.get("last_structured_errors")
    if not isinstance(errors, list) or not errors:
        return ""
    count = job.get("last_structured_errors_count")
    try:
        parsed_count = int(count)
    except (TypeError, ValueError):
        parsed_count = len(errors)
    if parsed_count < len(errors):
        parsed_count = len(errors)
    primary_issue = str(job.get("last_error") or job.get("last_reason") or "")
    details = []
    for error in errors:
        detail = _structured_error_detail(error)
        if detail == primary_issue:
            continue
        details.append(_truncate(detail, 60))
    if not details:
        return ""
    shown_count = len(details)
    label = str(parsed_count) if parsed_count == shown_count else f"{parsed_count} total, {shown_count} shown"
    return f"structured errors ({label}): {'; '.join(details)}"


def _top_count_key(counts: dict[str, Any]) -> str:
    best_name = "none"
    best_count = -1
    for name, value in counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        if count > best_count:
            best_name = str(name)
            best_count = count
    return best_name


def _scenario_opportunity(scenario: dict[str, Any]) -> str:
    explicit = scenario.get("opportunity_type")
    if explicit:
        return str(explicit)
    if scenario.get("product") == "btc_accumulation":
        return "btc_accumulation"
    if scenario.get("product") == "active_income":
        timeframe = scenario.get("base_tf")
        if timeframe == "1m":
            return "scalping"
        if timeframe == "5m":
            return "day_trading"
        if timeframe in {"15m", "30m", "1h"}:
            return "swing_trading"
    return "research"


def _ordered_opportunities(values: set[str]) -> list[str]:
    preferred = ["scalping", "day_trading", "swing_trading", "btc_accumulation", "research"]
    ordered = [name for name in preferred if name in values]
    ordered.extend(sorted(name for name in values if name not in preferred))
    return ordered


def _research_coverage_detail(research_cycle: dict[str, Any]) -> str:
    summary = research_cycle.get("summary") or {}
    by_product_payload = summary.get("opportunity_types_by_product")
    by_product: dict[str, set[str]] = {}
    if isinstance(by_product_payload, dict) and by_product_payload:
        for product, counts in by_product_payload.items():
            if isinstance(counts, dict):
                active = set()
                for name, count in counts.items():
                    try:
                        count_value = int(count or 0)
                    except (TypeError, ValueError):
                        count_value = 0
                    if count_value > 0:
                        active.add(str(name))
                by_product[str(product)] = active
    else:
        for scenario in research_cycle.get("scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            product = str(scenario.get("product", "unknown"))
            by_product.setdefault(product, set()).add(_scenario_opportunity(scenario))
    if not by_product:
        return ""
    parts = []
    for product in sorted(by_product):
        opportunities = ", ".join(_ordered_opportunities(by_product[product]))
        if opportunities:
            parts.append(f"{product}: {opportunities}")
    return "; ".join(parts)


def _source_freshness_detail(
    *,
    child: dict[str, Any],
    parent: dict[str, Any],
    source_key: str,
    child_label: str,
    parent_label: str,
) -> str:
    source = child.get("source") if isinstance(child.get("source"), dict) else {}
    child_source_at = source.get(source_key)
    parent_generated_at = parent.get("generated_at")
    if not child_source_at or not parent_generated_at:
        return "source unknown"
    if child_source_at == parent_generated_at:
        return "source current"
    return (
        f"source stale ({parent_label} {parent_generated_at}, "
        f"{child_label} source {child_source_at})"
    )


def _strategy_smoke_detail(strategy_smoke: dict[str, Any]) -> str:
    parts = []
    for scenario in strategy_smoke.get("scenarios") or []:
        name = scenario.get("name") or "scenario"
        if scenario.get("skipped"):
            parts.append(f"{name}: skipped {scenario.get('reason') or 'unknown'}")
            continue
        if not scenario.get("ok"):
            parts.append(f"{name}: failed {_truncate(str(scenario.get('error') or 'unknown'), 80)}")
            continue
        detail = f"{name}: rows {scenario.get('rows', 0)}"
        if scenario.get("scored_rows"):
            detail += f", scored {scenario['scored_rows']}"
        if scenario.get("best_strategy"):
            detail += f", best {scenario['best_strategy']}"
        if scenario.get("best_dsr") is not None:
            detail += f", dsr {float(scenario['best_dsr']):.3f}"
        parts.append(detail)
    return "; ".join(parts) if parts else "no scenarios"


def _promotion_reviews_detail(reviews: list[dict[str, Any]]) -> str:
    if not reviews:
        return "none configured"
    parts = []
    for review in reviews:
        product = review.get("product") or review.get("job") or "unknown"
        status = review.get("status") or "unknown"
        recommendations = review.get("recommendations") or {}
        needs_approval = int(review.get("needs_approval") or 0)
        action = " approval command available" if needs_approval else ""
        if recommendations:
            rec_detail = ", ".join(f"{key} {value}" for key, value in sorted(recommendations.items()))
            parts.append(f"{product}: {status} ({rec_detail}{action})")
        elif review.get("reason"):
            parts.append(f"{product}: {status} ({_truncate(str(review['reason']), 80)})")
        else:
            parts.append(f"{product}: {status}")
    return "; ".join(parts)


def _testnet_rehearsal_detail(status: dict[str, Any] | None) -> str:
    if not status:
        return "unknown"
    state = status.get("status") or "unknown"
    parts = [str(state)]
    next_action = status.get("next_action") if isinstance(status.get("next_action"), dict) else {}
    if state == "missing" and next_action.get("status_command"):
        parts.append(f"next {next_action['status_command']}")
    if status.get("product"):
        parts.append(str(status["product"]))
    if status.get("notional_usd") is not None:
        parts.append(f"notional ${float(status['notional_usd']):g}")
    if status.get("generated_at"):
        parts.append(f"generated {status['generated_at']}")
    if status.get("fresh") is False:
        parts.append(f"stale age {_fmt_age(status.get('age_seconds'))}")
    if status.get("final_position_flat") is False:
        parts.append("final position not flat")
    invalid_reasons = status.get("invalid_reasons")
    if invalid_reasons:
        reasons = ", ".join(str(item) for item in invalid_reasons)
        parts.append(f"invalid: {reasons}")
    if status.get("error"):
        parts.append(_truncate(str(status["error"]), 100))
    if state != "missing" and next_action.get("status_command") and status.get("ok") is not True:
        parts.append(f"next {next_action['status_command']}")
    return ", ".join(parts)


def _control_clear_detail(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    parts = []
    for item in items:
        name = item.get("name")
        label = str(name) if name is not None else "all"
        if item.get("skipped"):
            reason = item.get("reason") or "skipped"
            parts.append(f"{label}: skipped ({reason}){_control_clear_targets(item)}")
        elif item.get("ok") is True:
            parts.append(f"{label}: cleared{_control_clear_targets(item)}")
        else:
            error = item.get("error") or "failed"
            parts.append(f"{label}: failed ({error}){_control_clear_targets(item)}")
    return "; ".join(parts)


def _control_clear_targets(item: dict[str, Any]) -> str:
    targets = item.get("targets")
    if not isinstance(targets, list):
        return ""
    parts = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        name = target.get("product_name") or target.get("name") or "unknown"
        if target.get("ok") is True:
            status = "ok"
        elif target.get("ok") is False:
            status = "failed"
        else:
            status = "unknown"
        if target.get("skipped"):
            status = f"skipped {target.get('reason') or 'unknown'}"
        elif target.get("reason"):
            status = f"{status} {target['reason']}"
        elif target.get("error"):
            status = f"{status} {target['error']}"
        parts.append(f"{name} {status}")
    if not parts:
        return ""
    return " [" + ", ".join(_truncate(part, 80) for part in parts) + "]"


def _backup_report_detail(backup: dict[str, Any]) -> str:
    output = backup.get("output") or "unknown"
    size = backup.get("archive_size_bytes")
    manifest = backup.get("manifest") or {}
    verification = backup.get("verification") or {}
    retention = backup.get("retention") or {}
    parts = [str(output)]
    if size is not None:
        parts.append(f"{int(size)} bytes")
    if manifest:
        parts.append(
            f"included {int(manifest.get('included_files') or 0)}, "
            f"missing {int(manifest.get('missing_files') or 0)}, "
            f"skipped {int(manifest.get('skipped_files') or 0)}"
        )
    if verification:
        parts.append(
            f"verified {_fmt_bool(verification.get('ok'))}"
            f", checked {int(verification.get('checked_files') or 0)}"
        )
    if retention:
        parts.append(
            f"retention keep {int(retention.get('keep') or 0)}, "
            f"archives {int(retention.get('archives') or 0)}, "
            f"deleted {int(retention.get('deleted_archives') or 0)}"
        )
    return "; ".join(parts)


def _runtime_load_error_detail(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return "none"
    parts = []
    for item in errors[:3]:
        label = item.get("name") or item.get("path") or "runtime_file"
        error = item.get("error") or "unreadable"
        parts.append(f"{label}: {_truncate(str(error), 80)}")
    if len(errors) > 3:
        parts.append(f"+{len(errors) - 3} more")
    return "; ".join(parts)


def _report_error_detail(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return "none"
    parts = []
    for item in errors[:3]:
        code = item.get("code") or "report_error"
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        error = detail.get("error") or item.get("message") or "failed"
        parts.append(f"{code}: {_truncate(str(error), 80)}")
    if len(errors) > 3:
        parts.append(f"+{len(errors) - 3} more")
    return "; ".join(parts)


def _market_remediation_detail(market_data: dict[str, Any]) -> str:
    commands = []
    for market, item in sorted((market_data.get("markets") or {}).items()):
        remediation = item.get("remediation") or {}
        command = remediation.get("command")
        if isinstance(command, list) and command:
            commands.append(f"{market}: {' '.join(str(part) for part in command)}")
    return "; ".join(commands)


def _regime_data_detail(regime_data: dict[str, Any]) -> str:
    datasets = regime_data.get("datasets") or []
    if not datasets:
        return "none configured"
    parts = []
    for item in datasets:
        name = item.get("name") or "regime"
        if item.get("available"):
            counts = item.get("regime_counts") or {}
            bucket_detail = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
            parts.append(
                f"{name}: rows {int(item.get('rows') or 0)}"
                + (f", regimes {bucket_detail}" if bucket_detail else "")
            )
        elif item.get("enabled") is False:
            parts.append(f"{name}: disabled")
        else:
            parts.append(f"{name}: {item.get('reason') or 'not_ready'}")
    return "; ".join(parts)


def _approval_detail(summary: dict[str, Any] | None, fallback_count: int) -> str:
    if not summary:
        return f"`{fallback_count}`"
    total = int(summary.get("total") or 0)
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    count_parts = []
    for status in ("approved", "revoked", "unknown"):
        count = int(counts.get(status) or 0)
        if count:
            count_parts.append(f"{status} {count}")
    for status, count in sorted(counts.items()):
        if status in {"approved", "revoked", "unknown"}:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            count_int = 0
        if count_int:
            count_parts.append(f"{status} {count_int}")
    detail = f"`{total}`"
    if count_parts:
        detail += f" ({', '.join(count_parts)})"
    latest = summary.get("latest_event")
    if isinstance(latest, dict) and latest:
        strategy_id = latest.get("strategy_id") or "<unknown>"
        fingerprint = latest.get("fingerprint_short") or "unknown"
        product = latest.get("product_label") or "unscoped"
        actor = latest.get("actor") or "-"
        event_at = latest.get("event_at") or "unknown"
        event = latest.get("event") or "changed"
        latest_detail = (
            f"{event} {strategy_id} {fingerprint} for {product} "
            f"by {actor} at {event_at}"
        )
        reason = latest.get("revocation_reason")
        if reason:
            latest_detail += f", reason {_truncate(str(reason), 80)}"
        audit_reasons = latest.get("audit_reasons")
        if audit_reasons:
            latest_detail += f", audit {'/'.join(map(str, audit_reasons))}"
        detail += f"; latest {_truncate(latest_detail, 180)}"
    return detail


def _table_cell(value: Any, *, max_length: int = 80) -> str:
    if value is None:
        return ""
    return _truncate(str(value).replace("|", "\\|"), max_length)


def _open_position_broker_detail(position: dict[str, Any]) -> str:
    keys = (
        "broker_symbol",
        "broker_side",
        "broker_qty",
        "broker_requested_qty",
        "broker_fill_ratio",
        "broker_entry_quote_value",
        "broker_exit_sizing",
    )
    if not any(position.get(key) is not None for key in keys):
        return ""
    parts = []
    if position.get("broker_symbol") is not None:
        parts.append(str(position.get("broker_symbol")))
    if position.get("broker_side") is not None:
        parts.append(str(position.get("broker_side")))
    if position.get("broker_qty") is not None and position.get("broker_requested_qty") is not None:
        parts.append(f"qty {position.get('broker_qty')}/{position.get('broker_requested_qty')}")
    elif position.get("broker_qty") is not None:
        parts.append(f"qty {position.get('broker_qty')}")
    if position.get("broker_fill_ratio") is not None:
        parts.append(f"fill {_fmt_pct(position.get('broker_fill_ratio'))}")
    if position.get("broker_entry_quote_value") is not None:
        parts.append(f"quote {position.get('broker_entry_quote_value')}")
    if position.get("broker_exit_sizing") is not None:
        parts.append(str(position.get("broker_exit_sizing")))
    return ", ".join(parts)


def _open_position_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    generated_ts = _parse_timestamp(report.get("generated_at"))
    rows: list[dict[str, str]] = []
    for product in report.get("products", []) or []:
        if not isinstance(product, dict):
            continue
        for position in product.get("open_position_details", []) or []:
            if not isinstance(position, dict):
                continue
            entry_time = position.get("entry_time")
            entry_ts = _parse_timestamp(entry_time)
            age_seconds = None
            if generated_ts is not None and entry_ts is not None:
                age_seconds = generated_ts - entry_ts
            try:
                stale_after_seconds = float(position.get("stale_after_seconds"))
            except (TypeError, ValueError):
                stale_after_seconds = None
            if age_seconds is None or stale_after_seconds is None or stale_after_seconds <= 0:
                stale = "unknown"
            else:
                stale = "yes" if age_seconds > stale_after_seconds else "no"
            base_timeframe = position.get("base_timeframe")
            horizon_bars = position.get("horizon_bars")
            horizon = ""
            if base_timeframe is not None and horizon_bars is not None:
                horizon = f"{base_timeframe} x {horizon_bars}"
            elif base_timeframe is not None:
                horizon = str(base_timeframe)
            elif horizon_bars is not None:
                horizon = str(horizon_bars)
            rows.append(
                {
                    "product": _table_cell(product.get("name"), max_length=40),
                    "mode": _table_cell(product.get("mode"), max_length=16),
                    "market": _table_cell(product.get("market"), max_length=16),
                    "strategy": _table_cell(position.get("strategy_id"), max_length=60),
                    "side": _table_cell(position.get("direction"), max_length=16),
                    "broker": _table_cell(_open_position_broker_detail(position), max_length=90),
                    "size": _table_cell(position.get("position_size"), max_length=16),
                    "entry_price": _table_cell(position.get("entry_price"), max_length=18),
                    "stop": _table_cell(position.get("sl_price"), max_length=18),
                    "target": _table_cell(position.get("tp_price"), max_length=18),
                    "entry": _table_cell(entry_time, max_length=40),
                    "age": _fmt_age(age_seconds),
                    "horizon": _table_cell(horizon, max_length=24),
                    "stale_after": _fmt_age(stale_after_seconds),
                    "stale": stale,
                }
            )
    return rows


def render_operator_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Autopilot Operator Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Last status: `{report.get('status_generated_at') or 'missing'}`",
        f"- Overall status: `{_fmt_bool(report.get('ok'))}`",
        f"- Runtime file issues: `{_runtime_load_error_detail((report.get('runtime_load_errors') or []) + (report.get('runtime_shape_errors') or []))}`",
        f"- Report issues: `{_report_error_detail(report.get('report_errors') or [])}`",
        f"- Strategy approvals: {_approval_detail(report.get('approval_summary'), int(report.get('approval_count') or 0))}",
        f"- Error alert: `{_alert_detail(report.get('alert'))}`",
        f"- Readiness alert: `{_alert_detail(report.get('readiness_alert'))}`",
        f"- Research handoff alert: `{_alert_detail(report.get('research_handoff_alert'))}`",
        f"- Research progress alert: `{_alert_detail(report.get('research_progress_alert'))}`",
        f"- Testnet rehearsal alert: `{_alert_detail(report.get('testnet_rehearsal_alert'))}`",
        f"- Promotion alert: `{_alert_detail(report.get('promotion_alert'))}`",
    ]
    heartbeat = report.get("status_heartbeat") or {}
    lines.append(
        f"- Status heartbeat: `{_fmt_bool(heartbeat.get('fresh'))}` "
        f"(age {_fmt_age(heartbeat.get('age_seconds'))}, "
        f"limit {_fmt_age(heartbeat.get('limit_seconds'))})"
    )
    market_data = report.get("market_data") or {}
    market_details = []
    for market, item in sorted((market_data.get("markets") or {}).items()):
        detail = item.get("reason") or "unknown"
        if item.get("last_timestamp"):
            detail = f"{detail}, last candle {item['last_timestamp']}"
        market_details.append(f"{market}: {detail}")
    lines.append(
        f"- Market data: `{_fmt_bool(market_data.get('ok'))}` "
        f"({'; '.join(market_details) or 'unknown'})"
    )
    market_remediation = _market_remediation_detail(market_data)
    if market_remediation:
        lines.append(f"- Market data remediation: `{market_remediation}`")
    indicator_features = report.get("indicator_features") or {}
    feature_markets = indicator_features.get("markets") or {}
    missing_feature_details = []
    if feature_markets:
        feature_iter = (
            (f"{market}/{timeframe}", entry)
            for market, market_status in sorted(feature_markets.items())
            for timeframe, entry in (market_status.get("timeframes") or {}).items()
        )
    else:
        feature_iter = (
            (timeframe, entry)
            for timeframe, entry in (indicator_features.get("timeframes") or {}).items()
        )
    for label, entry in feature_iter:
        missing = entry.get("missing_features") or []
        if missing:
            missing_feature_details.append(f"{label}: {', '.join(missing)}")
    feature_detail = "ready" if not missing_feature_details else "missing " + "; ".join(missing_feature_details)
    lines.append(f"- Indicator features: `{_fmt_bool(indicator_features.get('ok'))}` ({feature_detail})")
    regime_data = report.get("regime_data") or {}
    if regime_data.get("datasets"):
        lines.append(
            f"- Regime data: `{_fmt_bool(regime_data.get('ok'))}` "
            f"({_regime_data_detail(regime_data)})"
        )
    research = report.get("research_smoke") or {}
    if research:
        scenario_count = len(research.get("scenarios") or [])
        lines.append(
            f"- Research smoke: `{_fmt_bool(research.get('ok'))}` "
            f"({scenario_count} synthetic scenarios, generated `{research.get('generated_at', 'unknown')}`)"
        )
    else:
        lines.append("- Research smoke: `unknown` (missing report)")
    strategy_smoke = report.get("strategy_smoke") or {}
    if strategy_smoke:
        lines.append(
            f"- Strategy smoke: `{_fmt_bool(strategy_smoke.get('ok'))}` "
            f"({_strategy_smoke_detail(strategy_smoke)}, generated `{strategy_smoke.get('generated_at', 'unknown')}`)"
        )
    else:
        lines.append("- Strategy smoke: `unknown` (missing report)")
    research_cycle = report.get("research_cycle") or {}
    if research_cycle:
        scenario_count = len(research_cycle.get("scenarios") or [])
        export_count = sum(1 for item in research_cycle.get("exports") or [] if item.get("exported"))
        skipped = research_cycle.get("skipped", False)
        summary = research_cycle.get("summary") or {}
        if skipped:
            detail = "skipped, market data unchanged"
        elif summary:
            top_reasons = summary.get("top_reasons") or {}
            top_reason = _top_count_key(top_reasons)
            next_action = (summary.get("next_actions") or ["none"])[0]
            detail = (
                f"{summary.get('scenarios', scenario_count)} real scenarios, "
                f"keepers {summary.get('keepers', 0)}, "
                f"watchlist {summary.get('incubation_candidates', 0)}, "
                f"exports {summary.get('exported', export_count)}, "
                f"top reason {top_reason}, next {_truncate(str(next_action), 90)}"
            )
            mutation_effectiveness = summary.get("mutation_effectiveness")
            if isinstance(mutation_effectiveness, dict):
                detail += (
                    f", mutations {mutation_effectiveness.get('evaluated_hypotheses', 0)} tested"
                    f"/{mutation_effectiveness.get('keepers', 0)} keepers"
                    f" ({mutation_effectiveness.get('outcome', 'unknown')})"
                )
            coverage_detail = _research_coverage_detail(research_cycle)
            if coverage_detail:
                detail += f", coverage {coverage_detail}"
        else:
            detail = f"{scenario_count} real scenarios, exports {export_count}"
            coverage_detail = _research_coverage_detail(research_cycle)
            if coverage_detail:
                detail += f", coverage {coverage_detail}"
        recovery_notes = []
        if research_cycle.get("state_recovered"):
            recovery_notes.append("state recovered")
        cycle_mutation_batch = research_cycle.get("mutation_batch")
        if isinstance(cycle_mutation_batch, dict) and cycle_mutation_batch.get("status") == "read_error":
            recovery_notes.append("mutation batch read_error")
        if recovery_notes:
            detail += f", {'; '.join(recovery_notes)}"
        lines.append(
            f"- Research cycle: `{_fmt_bool(research_cycle.get('ok'))}` "
            f"({detail}, generated `{research_cycle.get('generated_at', 'unknown')}`)"
        )
    else:
        lines.append("- Research cycle: `unknown` (missing report)")
    incubation_candidates = report.get("incubation_candidates") or {}
    if incubation_candidates:
        summary = incubation_candidates.get("summary") or {}
        by_product = summary.get("by_product") or {}
        product_detail = ", ".join(
            f"{product} {count}" for product, count in sorted(by_product.items())
        )
        safety = (
            f"research_only `{bool(incubation_candidates.get('research_only', False))}`, "
            f"executable `{bool(incubation_candidates.get('executable', True))}`, "
            f"paper `{bool(incubation_candidates.get('paper_trade_allowed', False))}`, "
            f"live `{bool(incubation_candidates.get('live_allowed', False))}`, "
            f"promotion `{bool(incubation_candidates.get('promotion_eligible', False))}`"
        )
        lines.append(
            f"- Incubation queue: `{_fmt_bool(incubation_candidates.get('ok'))}` "
            f"({summary.get('candidates', 0)} research-only candidates"
            f"{', ' + product_detail if product_detail else ''}, {safety}, generated "
            f"`{incubation_candidates.get('generated_at', 'unknown')}`)"
        )
    else:
        lines.append("- Incubation queue: `unknown` (missing report)")
    mutation_plan = report.get("mutation_plan") or {}
    if mutation_plan:
        mutation_summary = mutation_plan.get("summary") or {}
        by_product = mutation_summary.get("by_product") or {}
        product_detail = ", ".join(
            f"{product} {count}" for product, count in sorted(by_product.items())
        )
        suppressed_by_product = mutation_summary.get("suppressed_by_product") or {}
        suppressed_by_reason = mutation_summary.get("suppressed_by_reason") or {}
        suppressed_product_detail = ", ".join(
            f"{product} {count}" for product, count in sorted(suppressed_by_product.items())
        )
        suppressed_reason_detail = ", ".join(
            f"{reason} {count}" for reason, count in sorted(suppressed_by_reason.items())
        )
        suppressed_detail = f"suppressed repeats {mutation_summary.get('suppressed_repeated_sources', 0)}"
        if suppressed_product_detail:
            suppressed_detail += f" ({suppressed_product_detail})"
        if suppressed_reason_detail:
            suppressed_detail += f", suppressed reasons {suppressed_reason_detail}"
        source_detail = _source_freshness_detail(
            child=mutation_plan,
            parent=research_cycle,
            source_key="research_generated_at",
            child_label="plan",
            parent_label="research",
        ) if research_cycle else "source unknown"
        lines.append(
            f"- Mutation plan: `{_fmt_bool(mutation_plan.get('ok'))}` "
            f"({mutation_summary.get('proposals', 0)} research-only proposals"
            f"{', ' + product_detail if product_detail else ''}, "
            f"skipped scenarios {mutation_summary.get('skipped_scenarios', 0)}, "
            f"{suppressed_detail}, "
            f"{source_detail}, generated "
            f"`{mutation_plan.get('generated_at', 'unknown')}`)"
        )
    else:
        lines.append("- Mutation plan: `unknown` (missing report)")
    mutation_batch = report.get("mutation_batch") or {}
    if mutation_batch:
        mutation_summary = mutation_batch.get("summary") or {}
        by_product = mutation_summary.get("by_product") or {}
        product_detail = ", ".join(
            f"{product} {count}" for product, count in sorted(by_product.items())
        )
        skipped = int(mutation_summary.get("skipped") or 0)
        batch_source_detail = (
            _source_freshness_detail(
                child=mutation_batch,
                parent=mutation_plan,
                source_key="plan_generated_at",
                child_label="batch",
                parent_label="plan",
            )
            if mutation_plan
            else "source unknown"
        )
        status_detail = (
            f"{mutation_batch.get('count', mutation_summary.get('hypotheses', 0))} research-only hypotheses"
            f"{', ' + product_detail if product_detail else ''}, skipped {skipped}, "
            f"executable `{bool(mutation_batch.get('executable', False))}`, "
            f"{batch_source_detail}, generated "
            f"`{mutation_batch.get('generated_at', 'unknown')}`"
        )
        if mutation_batch.get("status"):
            status_detail = f"{mutation_batch.get('status')}, {status_detail}"
        if mutation_batch.get("error"):
            status_detail += f", error {_truncate(str(mutation_batch.get('error')), 120)}"
        lines.append(f"- Mutation batch: `{_fmt_bool(mutation_batch.get('ok'))}` ({status_detail})")
    else:
        lines.append("- Mutation batch: `unknown` (missing report)")
    lines.append(f"- Promotion reviews: `{_promotion_reviews_detail(report.get('promotion_reviews') or [])}`")
    hygiene = report.get("artifact_hygiene") or {}
    if hygiene:
        summary = hygiene.get("summary") or {}
        errors = hygiene.get("errors") if isinstance(hygiene.get("errors"), list) else []
        error_detail = ""
        if errors:
            first_error = errors[0] if isinstance(errors[0], dict) else {"error": str(errors[0])}
            error_detail = (
                f", errors {len(errors)}, "
                f"first {_truncate(str(first_error.get('scope') or 'artifact_hygiene'), 40)}: "
                f"{_truncate(str(first_error.get('error') or 'unknown'), 80)}"
            )
        lines.append(
            f"- Artifact hygiene: `{_fmt_bool(hygiene.get('ok'))}` "
            f"({summary.get('quarantine_candidates', 0)} quarantine candidates, "
            f"{summary.get('unreferenced_active_artifacts', 0)} unreferenced active artifacts, "
            f"dry-run `{bool(hygiene.get('dry_run', True))}`{error_detail})"
        )
    else:
        lines.append("- Artifact hygiene: `unknown` (missing report)")
    backup_report = report.get("backup_report") or {}
    if backup_report:
        backup_ok = bool(backup_report.get("ok")) and bool((backup_report.get("verification") or {}).get("ok"))
        lines.append(
            f"- Backup: `{_fmt_bool(backup_ok)}` "
            f"({_backup_report_detail(backup_report)})"
        )
    else:
        lines.append("- Backup: `unknown` (missing report)")
    testnet_rehearsal = report.get("testnet_rehearsal") or {}
    testnet_label = testnet_rehearsal.get("status") or _fmt_bool(testnet_rehearsal.get("ok"))
    lines.append(
        f"- Testnet rehearsal: `{testnet_label}` "
        f"({_testnet_rehearsal_detail(testnet_rehearsal)})"
    )
    control = report.get("control") or {}
    lines.extend(
        [
            f"- Paused: `{bool(control.get('paused', False))}`",
            f"- Paused products: `{', '.join(control.get('paused_products', [])) or 'none'}`",
            f"- Pause jobs: `{bool(control.get('pause_jobs', False))}`",
            f"- Paused jobs: `{', '.join(control.get('paused_jobs', [])) or 'none'}`",
            f"- Flatten products: `{', '.join(control.get('flatten_products', [])) or 'none'}`",
        ]
    )
    if report.get("control_error"):
        lines.append(f"- Control issue: `{_truncate(str(report['control_error']))}`")
    control_clear = report.get("control_clear") or []
    if control_clear:
        lines.append(f"- Control clear: `{_control_clear_detail(control_clear)}`")
    lines.extend(
        [
            "",
            "## Products",
            "",
            "| Product | Mode | Market | Cycle | Action | Open | Equity | Trades | Win Rate | Sized Return | Issue |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for product in report["products"]:
        trades = product["trade_summary"]
        issue = product.get("error") or product.get("close_error") or product.get("detail") or product.get("reason") or ""
        if not product.get("error") and product.get("close_error") and product.get("position_after_attempt"):
            position = product["position_after_attempt"]
            issue = f"{issue}; after attempt qty {position.get('qty')}"
        if not product.get("error") and product.get("position_after") and product.get("cycle_ok") is False:
            position = product["position_after"]
            issue = issue or f"position after qty {position.get('qty')}"
        if not issue and product.get("cycle_errors"):
            first_error = product["cycle_errors"][0]
            issue = f"{first_error.get('stage', 'cycle')}: {first_error.get('error', 'failed')}"
        if not issue and product.get("state_errors"):
            issue = _state_error_issue(product.get("state_errors"))
        if not issue and trades.get("issue"):
            issue = trades["issue"]
        issue_text = _truncate(str(issue).replace("|", "\\|"))
        lines.append(
            "| {name} | {mode} | {market} | {cycle} | {action} | {open_positions} | "
            "{equity} | {trades} | {win_rate} | {sized_return} | {issue} |".format(
                name=product["name"],
                mode=product["mode"],
                market=product["market"],
                cycle=_fmt_bool(product.get("cycle_ok")),
                action=product.get("action") or ("skipped" if product.get("skipped") else "cycle"),
                open_positions=product.get("open_positions")
                if product.get("open_positions") is not None
                else "n/a",
                equity=f"{product['equity']:.4f}" if isinstance(product.get("equity"), (int, float)) else "n/a",
                trades=trades["trades"],
                win_rate=_fmt_pct(trades.get("win_rate")),
                sized_return=_fmt_pct(trades.get("sized_return_sum", 0.0)),
                issue=issue_text,
            )
        )
    open_position_rows = _open_position_rows(report)
    if open_position_rows:
        lines.extend(
            [
                "",
                "## Open Positions",
                "",
                "| Product | Mode | Market | Strategy | Side | Broker | Size | Entry Price | Stop | Target | Entry Time | Age | Horizon | Stale After | Stale |",
                "|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---|",
            ]
        )
        for row in open_position_rows:
            lines.append(
                "| {product} | {mode} | {market} | {strategy} | {side} | {broker} | {size} | "
                "{entry_price} | {stop} | {target} | {entry} | "
                "{age} | {horizon} | {stale_after} | {stale} |".format(**row)
            )
    lines.extend(["", "## Jobs", ""])
    scheduled_jobs = report.get("scheduled_jobs") or []
    if scheduled_jobs:
        lines.extend(["| Job | Enabled | State | Due | Last Run | Issue |", "|---|---:|---:|---:|---|---|"])
        for job in scheduled_jobs:
            last_run = job.get("last_started_at") or "never"
            issue = job.get("last_error") or job.get("last_reason") or job.get("last_deferred_reason") or ""
            output_warnings = []
            structured_errors = _scheduled_job_structured_errors_detail(job)
            if structured_errors:
                output_warnings.append(structured_errors)
            if job.get("last_stdout_truncated"):
                output_warnings.append(f"stdout truncated ({job.get('last_stdout_bytes')} bytes)")
            if job.get("last_stderr_truncated"):
                output_warnings.append(f"stderr truncated ({job.get('last_stderr_bytes')} bytes)")
            if output_warnings:
                issue = "; ".join([str(issue), *output_warnings]).strip("; ")
            issue_text = _truncate(str(issue).replace("|", "\\|"))
            lines.append(
                f"| {job.get('name', 'unknown')} | `{bool(job.get('enabled'))}` | "
                f"`{job.get('status', 'unknown')}` | `{bool(job.get('due'))}` | {last_run} | {issue_text} |"
            )
        lines.append("")
    jobs = report.get("jobs") or []
    if not jobs:
        lines.append("No jobs ran in the latest cycle.")
    else:
        lines.extend(["Latest cycle:", "", "| Job | Status | Detail |", "|---|---|---|"])
        for job in jobs:
            detail = job.get("error") or job.get("stderr_tail") or job.get("stdout_tail") or ""
            detail_text = str(detail).strip().replace("|", "\\|")[:160]
            lines.append(
                f"| {job.get('name', 'unknown')} | `{_fmt_bool(job.get('ok'))}` | "
                f"{detail_text} |"
            )
    if report.get("alert"):
        lines.extend(["", "## Last Alert", "", f"```json\n{json.dumps(report['alert'], indent=2)}\n```"])
    return "\n".join(lines) + "\n"


def _operator_failure_report(config_path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "status_generated_at": None,
        "ok": False,
        "approval_count": 0,
        "approval_summary": {},
        "runtime_load_errors": [],
        "report_errors": [
            {
                "code": "operator_report_build_failed",
                "message": "operator report could not load config or build its payload",
                "detail": {"config": str(config_path), "error": f"{type(exc).__name__}: {exc}"},
            }
        ],
        "market_data": {},
        "indicator_features": {},
        "regime_data": {},
        "research_smoke": {},
        "strategy_smoke": {},
        "research_cycle": {},
        "artifact_hygiene": {},
        "testnet_rehearsal": {},
        "control": {},
        "products": [],
        "scheduled_jobs": [],
        "jobs": [],
    }


def _append_report_error(report: dict[str, Any], code: str, message: str, detail: dict[str, Any]) -> None:
    report.setdefault("report_errors", []).append({"code": code, "message": message, "detail": detail})
    report["ok"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact autopilot operator report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("runtime/operator_report.md"))
    parser.add_argument("--json-output", type=Path, help="Optional path for the structured report JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_failed = False
    try:
        report = build_operator_report(load_config(args.config))
    except Exception as exc:
        LOGGER.exception("Failed to build operator report")
        report = _operator_failure_report(args.config, exc)
    try:
        write_text_atomic(args.output, render_operator_markdown(report))
    except Exception as exc:
        LOGGER.exception("Failed to write operator markdown")
        output_failed = True
        _append_report_error(
            report,
            "operator_report_markdown_write_failed",
            "operator report could not write markdown output",
            {"path": str(args.output), "error": f"{type(exc).__name__}: {exc}"},
        )
    if args.json_output:
        try:
            write_json_atomic(args.json_output, report)
        except Exception as exc:
            LOGGER.exception("Failed to write operator JSON")
            output_failed = True
            _append_report_error(
                report,
                "operator_report_json_write_failed",
                "operator report could not write JSON output",
                {"path": str(args.json_output), "error": f"{type(exc).__name__}: {exc}"},
            )
    if output_failed:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(str(args.output))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
