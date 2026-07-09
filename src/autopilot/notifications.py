"""Low-noise alerting for the autopilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from src.autopilot.io import append_json_line, write_json_atomic
from src.autopilot.reporting import utc_now

VOLATILE_FINGERPRINT_KEYS = {
    "age_seconds",
    "free_bytes",
    "generated_at",
    "generated_ts",
    "total_bytes",
    "used_bytes",
}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _fingerprint_value(item)
            for key, item in value.items()
            if str(key) not in VOLATILE_FINGERPRINT_KEYS
        }
    if isinstance(value, list):
        return [_fingerprint_value(item) for item in value]
    return value


def alert_fingerprint(severity: str, title: str, detail: dict[str, Any]) -> str:
    fingerprint_detail = _fingerprint_value(detail)
    stable = json.dumps(
        {"severity": severity, "title": title, "detail": fingerprint_detail},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _fresh_state(*, load_error: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"version": 1, "alerts": {}}
    if load_error is not None:
        payload["_load_error"] = load_error
    return payload


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _fresh_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fresh_state(
            load_error={"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        )
    if not isinstance(payload, dict):
        return _fresh_state(
            load_error={
                "path": str(path),
                "error": f"TypeError: expected JSON object, got {type(payload).__name__}",
            }
        )
    alerts = payload.get("alerts", {})
    if not isinstance(alerts, dict):
        return _fresh_state(
            load_error={
                "path": str(path),
                "error": f"TypeError: expected alerts object, got {type(alerts).__name__}",
            }
        )
    payload.setdefault("version", 1)
    payload["alerts"] = alerts
    return payload


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    append_json_line(path, payload)


def _post_webhook(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.post(url, json=payload, timeout=10)
    return {"status_code": response.status_code, "ok": 200 <= response.status_code < 300}


def emit_alert(
    *,
    alert_file: Path,
    state_file: Path,
    severity: str,
    title: str,
    detail: dict[str, Any],
    cooldown_seconds: int = 900,
    webhook_url_env: str = "AUTOPILOT_WEBHOOK_URL",
    now: float | None = None,
) -> dict[str, Any]:
    raw_now = time.time() if now is None else now
    try:
        now = float(raw_now)
    except (TypeError, ValueError) as exc:
        raise ValueError("alert now timestamp must be numeric") from exc
    if not math.isfinite(now) or now < 0:
        raise ValueError("alert now timestamp must be finite and non-negative")
    try:
        cooldown_seconds = float(cooldown_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("alert cooldown_seconds must be numeric") from exc
    if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError("alert cooldown_seconds must be finite and non-negative")
    fingerprint = alert_fingerprint(severity, title, detail)
    state = _load_state(state_file)
    state_load_error = state.pop("_load_error", None)
    entry = state["alerts"].get(fingerprint, {})
    if not isinstance(entry, dict):
        entry = {}
    entry_recovery = None
    try:
        last_sent_ts = float(entry.get("last_sent_ts") or 0.0)
    except (TypeError, ValueError):
        last_sent_ts = 0.0
    if not math.isfinite(last_sent_ts) or last_sent_ts < 0:
        entry_recovery = {
            "fingerprint": fingerprint,
            "field": "last_sent_ts",
            "value": entry.get("last_sent_ts"),
            "reason": "invalid_timestamp",
        }
        last_sent_ts = 0.0
    elif last_sent_ts > now:
        entry_recovery = {
            "fingerprint": fingerprint,
            "field": "last_sent_ts",
            "value": last_sent_ts,
            "reason": "future_timestamp",
        }
        last_sent_ts = 0.0
    if now - last_sent_ts < cooldown_seconds:
        return {"sent": False, "reason": "cooldown", "fingerprint": fingerprint}

    payload = {
        "generated_at": utc_now(),
        "severity": severity,
        "title": title,
        "detail": detail,
        "fingerprint": fingerprint,
    }
    if state_load_error is not None:
        payload["alert_state_recovered"] = state_load_error
    if entry_recovery is not None:
        payload["alert_state_entry_recovered"] = entry_recovery
    webhook_url = os.environ.get(webhook_url_env, "").strip()
    if webhook_url:
        try:
            payload["webhook"] = _post_webhook(webhook_url, payload)
        except Exception as exc:  # local alerting/cooldown must survive webhook outages
            payload["webhook"] = {"ok": False, "error": str(exc)}

    _write_jsonl(alert_file, payload)

    state["alerts"][fingerprint] = {
        "last_sent_at": payload["generated_at"],
        "last_sent_ts": now,
        "severity": severity,
        "title": title,
    }
    result = {"sent": True, "fingerprint": fingerprint}
    if "webhook" in payload:
        result["webhook"] = payload["webhook"]
    try:
        _save_state(state_file, state)
    except Exception as exc:
        result["state_error"] = f"{type(exc).__name__}: {exc}"
    return result


def failure_detail(report: dict[str, Any]) -> dict[str, Any]:
    control = None
    if report.get("control_error"):
        raw_control = report.get("control") if isinstance(report.get("control"), dict) else {}
        control = {
            "error": report.get("control_error"),
            "reason": raw_control.get("reason"),
            "paused": raw_control.get("paused"),
            "pause_jobs": raw_control.get("pause_jobs"),
        }
        if report.get("unknown_control_selectors") is not None:
            control["unknown_selectors"] = report.get("unknown_control_selectors")
    products = []
    for product in _dict_list(report.get("products")):
        if not product.get("ok"):
            detail = {
                "name": product.get("product", {}).get("name"),
                "mode": product.get("product", {}).get("execution_mode"),
                "market": product.get("product", {}).get("market"),
                "action": product.get("action"),
                "error": product.get("error") or product.get("close_error") or product.get("reason"),
                "cycle_errors": product.get("cycle_errors", []),
                "state_errors": product.get("state_errors", []),
            }
            for key in (
                "broker",
                "position_before",
                "position_after",
                "position_after_error",
                "position_after_attempt",
                "position_after_attempt_error",
                "flattened",
                "fill",
                "spot_step_aside",
                "local_state",
            ):
                if product.get(key) is not None:
                    detail[key] = product.get(key)
            products.append(detail)
    jobs = []
    for job in _dict_list(report.get("jobs")):
        if not job.get("ok"):
            jobs.append(
                {
                    "name": job.get("name"),
                    "error": job.get("error") or job.get("stderr_tail") or job.get("returncode"),
                }
            )
    detail = {"products": products, "jobs": jobs, "data_update": report.get("data_update")}
    if control is not None:
        detail["control"] = control
    return detail


def readiness_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    for check in _dict_list(report.get("checks")):
        if check.get("ok") or check.get("level") != "warning":
            continue
        name = check.get("name")
        if name == "market data seed and freshness":
            markets = {}
            detail = check.get("detail") if isinstance(check.get("detail"), dict) else {}
            for market, item in detail.items():
                if not isinstance(item, dict) or item.get("ok"):
                    continue
                markets[market] = {
                    "reason": item.get("reason"),
                    "path": item.get("path"),
                }
            if markets:
                warnings.append({"name": name, "markets": markets})
        elif name == "indicator feature readiness":
            missing = {}
            detail = check.get("detail") if isinstance(check.get("detail"), dict) else {}
            for market, item in detail.items():
                if not isinstance(item, dict) or item.get("ok"):
                    continue
                market_missing = {}
                for timeframe, tf_item in (item.get("timeframes") or {}).items():
                    if not isinstance(tf_item, dict) or tf_item.get("ok"):
                        continue
                    market_missing[timeframe] = {
                        "reason": tf_item.get("reason"),
                        "missing_features": tf_item.get("missing_features") or [],
                    }
                if market_missing:
                    missing[market] = market_missing
            if missing:
                warnings.append({"name": name, "missing": missing})
        elif name == "runtime filesystem free space":
            detail = check.get("detail") or {}
            if isinstance(detail, dict):
                warnings.append(
                    {
                        "name": name,
                        "path": detail.get("path"),
                        "checked_path": detail.get("checked_path"),
                        "free_bytes": detail.get("free_bytes"),
                        "min_free_bytes": detail.get("min_free_bytes"),
                        "reason": detail.get("reason"),
                    }
                )
        elif name == "strategy framework smoke":
            detail = check.get("detail") or {}
            if isinstance(detail, dict):
                warnings.append(
                    {
                        "name": name,
                        "reason": detail.get("reason"),
                        "path": detail.get("path"),
                        "scenario_count": detail.get("scenario_count"),
                        "failures": detail.get("failures") or [],
                    }
                )
        elif name in {
            "approval ledger actor audit",
            "approval ledger fingerprint audit",
            "approval ledger revocation audit",
        }:
            detail = check.get("detail") or {}
            if isinstance(detail, dict):
                warning = {"name": name, "entries": detail.get("entries") or []}
                for key in (
                    "invalid_actor_count",
                    "fingerprint_mismatch_count",
                    "invalid_revocation_count",
                ):
                    if key in detail:
                        warning[key] = detail.get(key)
                warnings.append(warning)
    return {"warnings": warnings}


def promotion_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    for review in _dict_list(report.get("promotion_reviews")):
        recommendations = review.get("recommendations") or {}
        failed_count = int(recommendations.get("approved_review_failed") or 0)
        product = review.get("product") or review.get("job") or "unknown"
        status = review.get("status") or "unknown"
        path = review.get("path")
        generated_at = review.get("generated_at")
        if failed_count > 0:
            warnings.append(
                {
                    "name": "approved_review_failed",
                    "product": product,
                    "status": status,
                    "approved_review_failed": failed_count,
                    "path": path,
                    "generated_at": generated_at,
                }
            )
        if review.get("enabled") is not False and review.get("exists") is True and (
            review.get("fresh") is False or generated_at is None
        ):
            warning = {
                "name": "promotion_review_stale",
                "product": product,
                "status": status,
                "path": path,
                "generated_at": generated_at,
                "fresh": review.get("fresh"),
                "age_seconds": review.get("age_seconds"),
                "max_age_seconds": review.get("max_age_seconds"),
            }
            if review.get("needs_approval") is not None:
                warning["needs_approval"] = review.get("needs_approval")
            warnings.append({key: value for key, value in warning.items() if value is not None})
    return {"warnings": warnings}


def research_handoff_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    research_cycle = report.get("research_cycle") if isinstance(report.get("research_cycle"), dict) else {}
    mutation_plan = report.get("mutation_plan") if isinstance(report.get("mutation_plan"), dict) else {}
    mutation_batch = report.get("mutation_batch") if isinstance(report.get("mutation_batch"), dict) else {}

    def unsafe_flags(payload: dict[str, Any]) -> list[str]:
        flags = [
            key
            for key in ("executable", "paper_trade_allowed", "promotion_allowed", "live_allowed")
            if payload.get(key) is True
        ]
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        flags.extend(
            f"summary.{key}"
            for key in ("executable", "paper_trade_allowed", "promotion_allowed", "live_allowed")
            if summary.get(key) is True
        )
        flags.extend(str(flag) for flag in summary.get("unsafe_flags") or [])
        return sorted(set(flags))

    if mutation_plan:
        plan_unsafe_flags = unsafe_flags(mutation_plan)
        if mutation_plan.get("ok") is False or plan_unsafe_flags:
            warning = {
                "name": "mutation_plan_unhealthy",
                "ok": mutation_plan.get("ok"),
                "status": mutation_plan.get("status"),
                "generated_at": mutation_plan.get("generated_at"),
                "path": mutation_plan.get("path"),
                "unsafe_flags": plan_unsafe_flags,
            }
            warnings.append(
                {
                    key: value
                    for key, value in warning.items()
                    if value is not None and value != []
                }
            )

    if mutation_batch:
        batch_unsafe_flags = unsafe_flags(mutation_batch)
        if mutation_batch.get("ok") is False or batch_unsafe_flags:
            warning = {
                "name": "mutation_batch_unhealthy",
                "ok": mutation_batch.get("ok"),
                "status": mutation_batch.get("status"),
                "generated_at": mutation_batch.get("generated_at"),
                "path": mutation_batch.get("path"),
                "unsafe_flags": batch_unsafe_flags,
                "skipped": (mutation_batch.get("summary") or {}).get("skipped")
                if isinstance(mutation_batch.get("summary"), dict)
                else None,
            }
            warnings.append(
                {
                    key: value
                    for key, value in warning.items()
                    if value is not None and value != []
                }
            )

    research_generated_at = research_cycle.get("generated_at")
    plan_source = mutation_plan.get("source") if isinstance(mutation_plan.get("source"), dict) else {}
    plan_research_generated_at = plan_source.get("research_generated_at")
    if research_generated_at and plan_research_generated_at and research_generated_at != plan_research_generated_at:
        warnings.append(
            {
                "name": "mutation_plan_stale_source",
                "research_generated_at": research_generated_at,
                "mutation_plan_source_research_generated_at": plan_research_generated_at,
                "mutation_plan_generated_at": mutation_plan.get("generated_at"),
            }
        )

    plan_generated_at = mutation_plan.get("generated_at")
    batch_source = mutation_batch.get("source") if isinstance(mutation_batch.get("source"), dict) else {}
    batch_plan_generated_at = batch_source.get("plan_generated_at")
    if plan_generated_at and batch_plan_generated_at and plan_generated_at != batch_plan_generated_at:
        warnings.append(
            {
                "name": "mutation_batch_stale_source",
                "mutation_plan_generated_at": plan_generated_at,
                "mutation_batch_source_plan_generated_at": batch_plan_generated_at,
                "mutation_batch_generated_at": mutation_batch.get("generated_at"),
            }
        )
    return {"warnings": warnings}


def research_progress_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize research that is healthy but not yet producing paper artifacts."""

    research_cycle = report.get("research_cycle") if isinstance(report.get("research_cycle"), dict) else {}
    summary = research_cycle.get("summary") if isinstance(research_cycle.get("summary"), dict) else {}
    if research_cycle.get("ok") is not True:
        return {"warnings": []}
    if int(summary.get("hypotheses") or 0) <= 0:
        return {"warnings": []}
    export_reasons = summary.get("export_reasons") if isinstance(summary.get("export_reasons"), dict) else {}
    if int(export_reasons.get("open_positions_block_export") or 0) > 0:
        open_products = []
        for product in _dict_list(report.get("products")):
            open_positions = product.get("open_positions")
            try:
                open_count = int(open_positions or 0)
            except (TypeError, ValueError):
                open_count = 0
            if open_count <= 0:
                continue
            open_products.append(
                {
                    "name": product.get("name"),
                    "objective": product.get("objective"),
                    "market": product.get("market"),
                    "open_positions": open_count,
                }
            )
        return {
            "warnings": [
                {
                    "name": "research_export_blocked_open_positions",
                    "generated_at": research_cycle.get("generated_at"),
                    "keepers": summary.get("keepers"),
                    "exported": summary.get("exported"),
                    "export_reasons": export_reasons,
                    "next_actions": summary.get("next_actions") or [],
                    "open_products": open_products,
                }
            ]
        }
    if int(summary.get("keepers") or 0) > 0 or int(summary.get("exported") or 0) > 0:
        return {"warnings": []}

    waiting_products = []
    for product in _dict_list(report.get("products")):
        if product.get("enabled") is False:
            continue
        if product.get("mode") != "paper":
            continue
        if product.get("reason") != "waiting_for_strategy_artifact":
            continue
        waiting_products.append(
            {
                "name": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
            }
        )
    if not waiting_products:
        return {"warnings": []}

    mutation_effectiveness = summary.get("mutation_effectiveness")
    warning = {
        "name": "research_cycle_no_exportable_strategies",
        "generated_at": research_cycle.get("generated_at"),
        "hypotheses": summary.get("hypotheses"),
        "top_reasons": summary.get("top_reasons") or {},
        "next_actions": summary.get("next_actions") or [],
        "waiting_products": waiting_products,
    }
    if isinstance(mutation_effectiveness, dict):
        warning["mutation_effectiveness"] = mutation_effectiveness
    return {"warnings": [warning]}


def required_testnet_rehearsal_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize missing or stale exchange-facing rehearsal evidence."""

    rehearsal = report.get("testnet_rehearsal") if isinstance(report.get("testnet_rehearsal"), dict) else {}
    rehearsal_required = bool(rehearsal.get("required")) or any(
        product.get("enabled") is not False
        and bool(product.get("require_testnet_rehearsal"))
        for product in _dict_list(report.get("products"))
    )
    if not rehearsal_required or rehearsal.get("ok") is True:
        return {"warnings": []}

    warning = {
        "name": "required_testnet_rehearsal_not_ready",
        "status": rehearsal.get("status") or "unknown",
        "path": rehearsal.get("path"),
        "required_by": rehearsal.get("required_by") or [],
        "product": rehearsal.get("product"),
        "generated_at": rehearsal.get("generated_at"),
        "fresh": rehearsal.get("fresh"),
        "age_seconds": rehearsal.get("age_seconds"),
        "max_age_seconds": rehearsal.get("max_age_seconds"),
        "testnet": rehearsal.get("testnet"),
        "final_position_flat": rehearsal.get("final_position_flat"),
        "invalid_reasons": rehearsal.get("invalid_reasons"),
        "report_product": rehearsal.get("report_product"),
        "expected_product": rehearsal.get("expected_product"),
        "risk_controls": rehearsal.get("risk_controls"),
        "error": rehearsal.get("error"),
        "next_action": rehearsal.get("next_action"),
    }
    return {"warnings": [{key: value for key, value in warning.items() if value is not None}]}
