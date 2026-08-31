"""Low-noise alerting for the autopilot."""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.autopilot.alert_settings import alert_environment
from src.autopilot.io import append_json_line, write_json_atomic
from src.autopilot.locking import acquire_file_update_lock
from src.autopilot.reporting import utc_now
from src.autopilot.telegram_edge import redact_sensitive, send_alert_from_environment

LOGGER = logging.getLogger(__name__)
REMOTE_DELIVERY_QUEUE_LIMIT = 256
_REMOTE_DELIVERY_QUEUE: queue.Queue[tuple[Path, dict[str, Any], str, dict[str, str]]] = queue.Queue(
    maxsize=REMOTE_DELIVERY_QUEUE_LIMIT
)
_REMOTE_WORKER_LOCK = threading.Lock()
_REMOTE_WORKER: threading.Thread | None = None

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


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


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
        return _fresh_state(load_error={"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
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

    parsed = urlsplit(url)
    hostname = parsed.hostname
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("webhook URL must have a host and must not contain user credentials")
    loopback = hostname.lower() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("webhook URL must use HTTPS; HTTP is allowed only for loopback testing")
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=10,
            allow_redirects=False,
        )
    except Exception as exc:
        raise RuntimeError(f"webhook request failed: {type(exc).__name__}") from exc
    return {"status_code": response.status_code, "ok": 200 <= response.status_code < 300}


def _normalise_alert_inputs(
    cooldown_seconds: int,
    dedupe_key: str | None,
    now: float | None,
    detail: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    raw_now = time.time() if now is None else now
    try:
        normalised_now = float(raw_now)
    except (TypeError, ValueError) as exc:
        raise ValueError("alert now timestamp must be numeric") from exc
    if not math.isfinite(normalised_now) or normalised_now < 0:
        raise ValueError("alert now timestamp must be finite and non-negative")
    try:
        normalised_cooldown = float(cooldown_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("alert cooldown_seconds must be numeric") from exc
    if not math.isfinite(normalised_cooldown) or normalised_cooldown < 0:
        raise ValueError("alert cooldown_seconds must be finite and non-negative")
    if dedupe_key is None:
        return normalised_now, normalised_cooldown, detail
    if not isinstance(dedupe_key, str) or not dedupe_key.strip():
        raise ValueError("alert dedupe_key must be a non-empty string")
    if len(dedupe_key) > 512:
        raise ValueError("alert dedupe_key must be at most 512 characters")
    return normalised_now, normalised_cooldown, {"dedupe_key": dedupe_key.strip()}


def emit_alert(
    *,
    alert_file: Path,
    state_file: Path,
    severity: str,
    title: str,
    detail: dict[str, Any],
    cooldown_seconds: int = 900,
    dedupe_key: str | None = None,
    webhook_url_env: str = "AUTOPILOT_WEBHOOK_URL",
    now: float | None = None,
) -> dict[str, Any]:
    now, cooldown_seconds, fingerprint_detail = _normalise_alert_inputs(
        cooldown_seconds, dedupe_key, now, detail
    )
    fingerprint = alert_fingerprint(severity, title, fingerprint_detail)
    with acquire_file_update_lock(state_file, label="alert cooldown state"):
        result, payload = _emit_alert_locked(
            alert_file=alert_file,
            state_file=state_file,
            severity=severity,
            title=title,
            detail=detail,
            cooldown_seconds=cooldown_seconds,
            now=now,
            fingerprint=fingerprint,
        )
    if payload is not None:
        try:
            operations_environment = alert_environment(os.environ)
        except Exception as exc:
            result["remote_delivery"] = {
                "status": "invalid_settings",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return result
        webhook_url = operations_environment.get(webhook_url_env, "").strip()
        remote_configured = bool(
            webhook_url
            or operations_environment.get("AUTOPILOT_TELEGRAM_BOT_TOKEN", "").strip()
            or operations_environment.get("AUTOPILOT_TELEGRAM_SETTINGS_FILE", "").strip()
        )
        if remote_configured:
            result["remote_delivery"] = _enqueue_remote_delivery(
                alert_file,
                payload,
                webhook_url,
                operations_environment,
            )
        else:
            result["remote_delivery"] = {"status": "not_configured"}
    return result


def _emit_alert_locked(
    *,
    alert_file: Path,
    state_file: Path,
    severity: str,
    title: str,
    detail: dict[str, Any],
    cooldown_seconds: float,
    now: float,
    fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
        return (
            {"sent": False, "reason": "cooldown", "fingerprint": fingerprint},
            None,
        )

    payload = {
        "schema": "autopilot.alert/v1",
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
    _write_jsonl(alert_file, payload)

    state["alerts"][fingerprint] = {
        "last_sent_at": payload["generated_at"],
        "last_sent_ts": now,
        "severity": severity,
        "title": title,
    }
    result = {"sent": True, "fingerprint": fingerprint}
    try:
        _save_state(state_file, state)
    except Exception as exc:
        result["state_error"] = f"{type(exc).__name__}: {exc}"
    return result, payload


def _deliver_remote_alert(
    alert_file: Path,
    payload: dict[str, Any],
    webhook_url: str,
    operations_environment: dict[str, str],
) -> None:
    """Deliver one already-durable local alert without blocking supervision."""

    delivery: dict[str, Any] = {
        "schema": "autopilot.alert_delivery/v1",
        "generated_at": utc_now(),
        "fingerprint": payload["fingerprint"],
    }
    if webhook_url:
        try:
            remote_payload = redact_sensitive(payload)
            if not isinstance(remote_payload, dict):
                raise ValueError("sanitized webhook payload must be a JSON object")
            delivery["webhook"] = _post_webhook(webhook_url, remote_payload)
        except Exception as exc:
            delivery["webhook"] = {"ok": False, "error": str(exc)}
    telegram_configured = bool(
        operations_environment.get("AUTOPILOT_TELEGRAM_BOT_TOKEN", "").strip()
        or operations_environment.get("AUTOPILOT_TELEGRAM_SETTINGS_FILE", "").strip()
    )
    if telegram_configured:
        try:
            telegram = send_alert_from_environment(payload, environ=operations_environment)
            delivery["telegram"] = (
                telegram
                if telegram is not None
                else {
                    "ok": False,
                    "error": "configured Telegram settings resolved to no delivery client",
                }
            )
        except Exception as exc:
            delivery["telegram"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if "webhook" not in delivery and "telegram" not in delivery:
        return
    _write_jsonl(alert_file, delivery)


def _remote_delivery_worker() -> None:
    while True:
        alert_file, payload, webhook_url, operations_environment = _REMOTE_DELIVERY_QUEUE.get()
        try:
            _deliver_remote_alert(
                alert_file,
                payload,
                webhook_url,
                operations_environment,
            )
        except Exception:
            # The originating local alert and cooldown state are already
            # durable. A delivery-record write failure is still visible in the
            # service journal without destabilizing trading supervision.
            LOGGER.exception(
                "Asynchronous remote alert delivery failed for fingerprint %s",
                payload.get("fingerprint"),
            )
        finally:
            _REMOTE_DELIVERY_QUEUE.task_done()


def _ensure_remote_worker() -> None:
    global _REMOTE_WORKER
    with _REMOTE_WORKER_LOCK:
        if _REMOTE_WORKER is not None and _REMOTE_WORKER.is_alive():
            return
        _REMOTE_WORKER = threading.Thread(
            target=_remote_delivery_worker,
            name="autopilot-alert-delivery",
            daemon=True,
        )
        _REMOTE_WORKER.start()


def _enqueue_remote_delivery(
    alert_file: Path,
    payload: dict[str, Any],
    webhook_url: str,
    operations_environment: dict[str, str],
) -> dict[str, Any]:
    _ensure_remote_worker()
    try:
        _REMOTE_DELIVERY_QUEUE.put_nowait(
            (alert_file, payload, webhook_url, dict(operations_environment))
        )
    except queue.Full:
        return {
            "status": "queue_full",
            "queue_limit": REMOTE_DELIVERY_QUEUE_LIMIT,
        }
    return {"status": "queued"}


def wait_for_remote_alerts(timeout_seconds: float = 5.0) -> bool:
    """Wait for queued deliveries; intended for tests and orderly shutdowns."""

    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while _REMOTE_DELIVERY_QUEUE.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


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
                "error": product.get("error")
                or product.get("close_error")
                or product.get("reason"),
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
    detail = {
        "products": products,
        "jobs": jobs,
        "job_config_errors": report.get("job_config_errors", []),
        "data_update": report.get("data_update"),
    }
    if control is not None:
        detail["control"] = control
    return detail


def _market_data_readiness_warning(check: dict[str, Any]) -> dict[str, Any] | None:
    detail = check.get("detail") if isinstance(check.get("detail"), dict) else {}
    markets = {
        market: {"reason": item.get("reason"), "path": item.get("path")}
        for market, item in detail.items()
        if isinstance(item, dict) and not item.get("ok")
    }
    return {"name": check.get("name"), "markets": markets} if markets else None


def _indicator_readiness_warning(check: dict[str, Any]) -> dict[str, Any] | None:
    detail = check.get("detail") if isinstance(check.get("detail"), dict) else {}
    missing: dict[str, Any] = {}
    for market, item in detail.items():
        if not isinstance(item, dict) or item.get("ok"):
            continue
        market_missing = {
            timeframe: {
                "reason": tf_item.get("reason"),
                "missing_features": tf_item.get("missing_features") or [],
            }
            for timeframe, tf_item in (item.get("timeframes") or {}).items()
            if isinstance(tf_item, dict) and not tf_item.get("ok")
        }
        if market_missing:
            missing[market] = market_missing
    return {"name": check.get("name"), "missing": missing} if missing else None


def _simple_readiness_warning(check: dict[str, Any]) -> dict[str, Any] | None:
    detail = check.get("detail") or {}
    if not isinstance(detail, dict):
        return None
    name = check.get("name")
    if name == "runtime filesystem free space":
        return {
            "name": name,
            "path": detail.get("path"),
            "checked_path": detail.get("checked_path"),
            "free_bytes": detail.get("free_bytes"),
            "min_free_bytes": detail.get("min_free_bytes"),
            "reason": detail.get("reason"),
        }
    if name == "strategy framework smoke":
        return {
            "name": name,
            "reason": detail.get("reason"),
            "path": detail.get("path"),
            "scenario_count": detail.get("scenario_count"),
            "failures": detail.get("failures") or [],
        }
    return None


def _ledger_readiness_warning(check: dict[str, Any]) -> dict[str, Any] | None:
    detail = check.get("detail") or {}
    if not isinstance(detail, dict):
        return None
    warning = {"name": check.get("name"), "entries": detail.get("entries") or []}
    for key in (
        "invalid_actor_count",
        "fingerprint_mismatch_count",
        "invalid_revocation_count",
    ):
        if key in detail:
            warning[key] = detail.get(key)
    return warning


def _readiness_check_warning(check: dict[str, Any]) -> dict[str, Any] | None:
    if check.get("ok") or check.get("level") != "warning":
        return None
    name = check.get("name")
    if name == "market data seed and freshness":
        return _market_data_readiness_warning(check)
    if name == "indicator feature readiness":
        return _indicator_readiness_warning(check)
    if name in {"runtime filesystem free space", "strategy framework smoke"}:
        return _simple_readiness_warning(check)
    if name in {
        "approval ledger actor audit",
        "approval ledger fingerprint audit",
        "approval ledger revocation audit",
    }:
        return _ledger_readiness_warning(check)
    return None


def readiness_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    for check in _dict_list(report.get("checks")):
        warning = _readiness_check_warning(check)
        if warning is not None:
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
        if (
            review.get("enabled") is not False
            and review.get("exists") is True
            and (review.get("fresh") is False or generated_at is None)
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


def _unsafe_handoff_flags(payload: dict[str, Any]) -> list[str]:
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


def _handoff_health_warning(
    name: str,
    payload: dict[str, Any],
    *,
    include_skipped: bool = False,
) -> dict[str, Any] | None:
    unsafe_flags = _unsafe_handoff_flags(payload)
    if payload.get("ok") is not False and not unsafe_flags:
        return None
    warning: dict[str, Any] = {
        "name": name,
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "generated_at": payload.get("generated_at"),
        "path": payload.get("path"),
        "unsafe_flags": unsafe_flags,
    }
    if include_skipped:
        summary = payload.get("summary")
        warning["skipped"] = summary.get("skipped") if isinstance(summary, dict) else None
    return {key: value for key, value in warning.items() if value is not None and value != []}


def _handoff_source_warnings(
    research_cycle: dict[str, Any],
    mutation_plan: dict[str, Any],
    mutation_batch: dict[str, Any],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    research_generated_at = research_cycle.get("generated_at")
    plan_source = (
        mutation_plan.get("source") if isinstance(mutation_plan.get("source"), dict) else {}
    )
    plan_research_generated_at = plan_source.get("research_generated_at")
    if (
        research_generated_at
        and plan_research_generated_at
        and research_generated_at != plan_research_generated_at
    ):
        warnings.append(
            {
                "name": "mutation_plan_stale_source",
                "research_generated_at": research_generated_at,
                "mutation_plan_source_research_generated_at": plan_research_generated_at,
                "mutation_plan_generated_at": mutation_plan.get("generated_at"),
            }
        )
    plan_generated_at = mutation_plan.get("generated_at")
    batch_source = (
        mutation_batch.get("source") if isinstance(mutation_batch.get("source"), dict) else {}
    )
    batch_plan_generated_at = batch_source.get("plan_generated_at")
    if (
        plan_generated_at
        and batch_plan_generated_at
        and plan_generated_at != batch_plan_generated_at
    ):
        warnings.append(
            {
                "name": "mutation_batch_stale_source",
                "mutation_plan_generated_at": plan_generated_at,
                "mutation_batch_source_plan_generated_at": batch_plan_generated_at,
                "mutation_batch_generated_at": mutation_batch.get("generated_at"),
            }
        )
    return warnings


def _generated_batch_handoff_warning(
    research_cycle: dict[str, Any], generated_batch: dict[str, Any]
) -> dict[str, Any] | None:
    research_generated_at = _parse_timestamp(research_cycle.get("generated_at"))
    generated_batch_at = generated_batch.get("generated_at")
    generated_batch_ts = _parse_timestamp(generated_batch_at)
    if (
        research_generated_at is None
        or generated_batch_ts is None
        or generated_batch_ts <= research_generated_at
    ):
        return None
    return {
        "name": "generated_batch_unconsumed",
        "generated_batch_generated_at": generated_batch_at,
        "research_cycle_generated_at": research_cycle.get("generated_at"),
        "hypotheses": generated_batch.get(
            "hypotheses_count",
            (generated_batch.get("summary") or {}).get("hypotheses"),
        ),
    }


def _freshness_handoff_warnings(
    research_cycle: dict[str, Any], generated_batch: dict[str, Any]
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for name, payload in (
        ("research_cycle_stale", research_cycle),
        ("generated_batch_stale", generated_batch),
    ):
        if payload.get("fresh") is False:
            warnings.append(
                {
                    "name": name,
                    "generated_at": payload.get("generated_at"),
                    "age_seconds": payload.get("age_seconds"),
                    "max_age_seconds": payload.get("max_age_seconds"),
                    "reason": payload.get("freshness_reason"),
                }
            )
    return warnings


def research_handoff_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    research_cycle = (
        report.get("research_cycle") if isinstance(report.get("research_cycle"), dict) else {}
    )
    mutation_plan = (
        report.get("mutation_plan") if isinstance(report.get("mutation_plan"), dict) else {}
    )
    mutation_batch = (
        report.get("mutation_batch") if isinstance(report.get("mutation_batch"), dict) else {}
    )
    generated_batch = (
        report.get("generated_batch") if isinstance(report.get("generated_batch"), dict) else {}
    )
    for name, payload, include_skipped in (
        ("mutation_plan_unhealthy", mutation_plan, False),
        ("mutation_batch_unhealthy", mutation_batch, True),
    ):
        if payload:
            warning = _handoff_health_warning(name, payload, include_skipped=include_skipped)
            if warning is not None:
                warnings.append(warning)
    warnings.extend(_handoff_source_warnings(research_cycle, mutation_plan, mutation_batch))
    generated_warning = _generated_batch_handoff_warning(research_cycle, generated_batch)
    if generated_warning is not None:
        warnings.append(generated_warning)
    warnings.extend(_freshness_handoff_warnings(research_cycle, generated_batch))
    return {"warnings": warnings}


def _open_research_products(report: dict[str, Any]) -> list[dict[str, Any]]:
    products = []
    for product in _dict_list(report.get("products")):
        try:
            open_count = int(product.get("open_positions") or 0)
        except (TypeError, ValueError):
            open_count = 0
        if open_count <= 0:
            continue
        products.append(
            {
                "name": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "open_positions": open_count,
            }
        )
    return products


def _open_position_research_warning(
    report: dict[str, Any],
    research_cycle: dict[str, Any],
    summary: dict[str, Any],
    export_reasons: dict[str, Any],
) -> dict[str, Any] | None:
    if int(export_reasons.get("open_positions_block_export") or 0) <= 0:
        return None
    return {
        "warnings": [
            {
                "name": "research_export_blocked_open_positions",
                "generated_at": research_cycle.get("generated_at"),
                "keepers": summary.get("keepers"),
                "exported": summary.get("exported"),
                "export_reasons": export_reasons,
                "next_actions": summary.get("next_actions") or [],
                "open_products": _open_research_products(report),
            }
        ]
    }


def _waiting_research_products(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": product.get("name"),
            "objective": product.get("objective"),
            "market": product.get("market"),
        }
        for product in _dict_list(report.get("products"))
        if product.get("enabled") is not False
        and product.get("mode") == "paper"
        and product.get("reason") == "waiting_for_strategy_artifact"
    ]


def research_progress_warning_detail(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize research that is healthy but not yet producing paper artifacts."""

    research_cycle = (
        report.get("research_cycle") if isinstance(report.get("research_cycle"), dict) else {}
    )
    summary = (
        research_cycle.get("summary") if isinstance(research_cycle.get("summary"), dict) else {}
    )
    if research_cycle.get("ok") is not True:
        return {"warnings": []}
    if int(summary.get("hypotheses") or 0) <= 0:
        return {"warnings": []}
    export_reasons = (
        summary.get("export_reasons") if isinstance(summary.get("export_reasons"), dict) else {}
    )
    open_position_warning = _open_position_research_warning(
        report, research_cycle, summary, export_reasons
    )
    if open_position_warning is not None:
        return open_position_warning
    if int(summary.get("keepers") or 0) > 0 or int(summary.get("exported") or 0) > 0:
        return {"warnings": []}

    waiting_products = _waiting_research_products(report)
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

    rehearsal = (
        report.get("testnet_rehearsal") if isinstance(report.get("testnet_rehearsal"), dict) else {}
    )
    rehearsal_required = bool(rehearsal.get("required")) or any(
        product.get("enabled") is not False and bool(product.get("require_testnet_rehearsal"))
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
