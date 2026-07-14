"""Machine-readable autopilot healthcheck for external watchdogs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.notifications import (
    emit_alert,
    research_handoff_warning_detail,
    wait_for_remote_alerts,
)
from src.autopilot.readiness import build_readiness_report
from src.autopilot.reporting import build_operator_report

LOGGER = logging.getLogger("autopilot.healthcheck")
REMOTE_ALERT_DRAIN_SECONDS = 30.0


def _drain_oneshot_remote_alert(health: dict[str, Any]) -> None:
    """Keep a oneshot watchdog alive until its queued remote alert is delivered.

    Long-running supervision remains non-blocking, but a daemon delivery thread
    cannot outlive a systemd ``Type=oneshot`` process.  The local JSONL alert is
    already durable; this bounded drain gives webhook/Telegram delivery enough
    time to finish and records a timeout without hiding the health result.
    """

    alert = health.get("healthcheck_alert")
    if not isinstance(alert, dict):
        return
    remote = alert.get("remote_delivery")
    if not isinstance(remote, dict) or remote.get("status") != "queued":
        return
    try:
        drained = wait_for_remote_alerts(REMOTE_ALERT_DRAIN_SECONDS)
    except Exception as exc:  # alert delivery must not suppress watchdog output
        LOGGER.exception("Failed while draining the healthcheck remote-alert queue")
        remote.update(drained=False, drain_error=f"{type(exc).__name__}: {exc}")
        return
    remote["drained"] = bool(drained)
    if not drained:
        remote["drain_error"] = (
            f"remote alert queue did not drain within {REMOTE_ALERT_DRAIN_SECONDS:g} seconds"
        )


def _issue(code: str, message: str, *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail:
        payload["detail"] = detail
    return payload


def _dict_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _malformed_dict_list_detail(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    if key not in payload:
        return None
    value = payload.get(key)
    if not isinstance(value, list):
        return {
            "section": key,
            "error": f"expected list, got {type(value).__name__}",
        }
    invalid_entries = [
        {"index": index, "type": type(item).__name__}
        for index, item in enumerate(value)
        if not isinstance(item, dict)
    ]
    if not invalid_entries:
        return None
    return {"section": key, "invalid_entries": invalid_entries[:10]}


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


def _backup_stale_limit_seconds(
    operator_report: dict[str, Any],
    fallback_seconds: float = 48 * 60 * 60,
) -> float:
    schedule = operator_report.get("backup_schedule")
    if isinstance(schedule, dict) and schedule.get("enabled") is True:
        cadence = schedule.get("cadence_seconds")
        try:
            cadence_seconds = float(cadence)
        except (TypeError, ValueError):
            cadence_seconds = 0.0
        if cadence_seconds > 0:
            return max(cadence_seconds * 2.0, 3600.0)
    for job in _dict_list(operator_report, "scheduled_jobs"):
        if not job.get("enabled"):
            continue
        if "backup" not in str(job.get("name") or ""):
            continue
        cadence = job.get("effective_cadence_seconds") or job.get("cadence_seconds")
        try:
            cadence_seconds = float(cadence)
        except (TypeError, ValueError):
            continue
        if cadence_seconds > 0:
            return max(cadence_seconds * 2.0, 3600.0)
    return fallback_seconds


def _enabled_backup_jobs(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        {
            "name": job.get("name"),
            "status": job.get("status"),
            "effective_cadence_seconds": job.get("effective_cadence_seconds"),
            "cadence_seconds": job.get("cadence_seconds"),
        }
        for job in _dict_list(operator_report, "scheduled_jobs")
        if job.get("enabled") and "backup" in str(job.get("name") or "")
    ]
    schedule = operator_report.get("backup_schedule")
    if isinstance(schedule, dict) and schedule.get("enabled") is True:
        jobs.append(
            {
                "name": schedule.get("name") or "runtime_backup_timer",
                "status": "dedicated_timer",
                "effective_cadence_seconds": schedule.get("cadence_seconds"),
                "cadence_seconds": schedule.get("cadence_seconds"),
            }
        )
    return jobs


def _scheduled_job_state_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    invalid_jobs = []
    for job in _dict_list(operator_report, "scheduled_jobs"):
        if not job.get("enabled"):
            continue
        last_reason = job.get("last_reason")
        if not isinstance(last_reason, str) or "invalid job state" not in last_reason:
            continue
        invalid_jobs.append(
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "due": job.get("due"),
                "last_started_at": job.get("last_started_at"),
                "age_seconds": job.get("age_seconds"),
                "last_reason": last_reason,
            }
        )
    return invalid_jobs


def _scheduled_job_output_truncation_warnings(
    operator_report: dict[str, Any],
) -> list[dict[str, Any]]:
    noisy_jobs = []
    for job in _dict_list(operator_report, "scheduled_jobs"):
        if not job.get("enabled"):
            continue
        stdout_truncated = job.get("last_stdout_truncated") is True
        stderr_truncated = job.get("last_stderr_truncated") is True
        if not stdout_truncated and not stderr_truncated:
            continue
        noisy_jobs.append(
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "last_started_at": job.get("last_started_at"),
                "stdout_truncated": stdout_truncated,
                "stdout_bytes": job.get("last_stdout_bytes"),
                "stderr_truncated": stderr_truncated,
                "stderr_bytes": job.get("last_stderr_bytes"),
            }
        )
    return noisy_jobs


def _scheduled_job_deferral_warnings(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    deferred_jobs = []
    for job in _dict_list(operator_report, "scheduled_jobs"):
        if not job.get("enabled"):
            continue
        if job.get("status") != "deferred" and job.get("last_deferred_reason") != "cycle_job_limit":
            continue
        deferred_jobs.append(
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "due": job.get("due"),
                "last_deferred_at": job.get("last_deferred_at"),
                "last_deferred_reason": job.get("last_deferred_reason"),
                "consecutive_deferrals": job.get("consecutive_deferrals"),
            }
        )
    return deferred_jobs


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scheduled_job_deferral_limit_issues(
    operator_report: dict[str, Any],
    *,
    max_consecutive_job_deferrals: int,
) -> list[dict[str, Any]]:
    starved_jobs = []
    for job in _dict_list(operator_report, "scheduled_jobs"):
        if not job.get("enabled"):
            continue
        if job.get("status") != "deferred" and job.get("last_deferred_reason") != "cycle_job_limit":
            continue
        consecutive_deferrals = _int_value(job.get("consecutive_deferrals"))
        if consecutive_deferrals is None or consecutive_deferrals < max_consecutive_job_deferrals:
            continue
        starved_jobs.append(
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "due": job.get("due"),
                "last_deferred_at": job.get("last_deferred_at"),
                "last_deferred_reason": job.get("last_deferred_reason"),
                "consecutive_deferrals": consecutive_deferrals,
                "max_consecutive_job_deferrals": max_consecutive_job_deferrals,
            }
        )
    return starved_jobs


def _scheduled_job_failure_detail(job: dict[str, Any]) -> dict[str, Any]:
    detail = {
        "name": job.get("name"),
        "status": job.get("status"),
        "consecutive_failures": job.get("consecutive_failures"),
        "last_error": job.get("last_error"),
        "last_reason": job.get("last_reason"),
    }
    structured_errors = job.get("last_structured_errors")
    if isinstance(structured_errors, list) and structured_errors:
        detail["last_structured_errors"] = structured_errors
        detail["last_structured_errors_count"] = job.get("last_structured_errors_count")
    return detail


def _paper_products_waiting_for_artifacts(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    waiting = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False:
            continue
        if product.get("mode") != "paper":
            continue
        if product.get("reason") != "waiting_for_strategy_artifact":
            continue
        waiting.append(product)
    return waiting


def _market_data_issue_detail(operator_report: dict[str, Any]) -> dict[str, Any] | None:
    market_data = operator_report.get("market_data")
    if not isinstance(market_data, dict) or market_data.get("ok") is not False:
        return None
    markets = market_data.get("markets")
    bad_markets = []
    if isinstance(markets, dict):
        market_items = sorted(markets.items())
    else:
        market_items = [(market_data.get("market") or "default", market_data)]
    for market, item in market_items:
        if not isinstance(item, dict) or item.get("ok") is not False:
            continue
        bad_markets.append(
            {
                key: item.get(key)
                for key in (
                    "market",
                    "path",
                    "exists",
                    "reason",
                    "error",
                    "rows",
                    "first_timestamp",
                    "last_timestamp",
                    "age_seconds",
                    "max_age_seconds",
                    "remediation",
                )
                if key in item
            }
            or {"market": market}
        )
        if "market" not in bad_markets[-1]:
            bad_markets[-1]["market"] = market
    return {"markets": bad_markets} if bad_markets else {"market_data": market_data}


def _stale_promotion_reviews(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    stale_reviews = []
    for review in _dict_list(operator_report, "promotion_reviews"):
        if review.get("enabled") is False:
            continue
        if review.get("exists") is False:
            continue
        if review.get("fresh") is not False and review.get("generated_at") is not None:
            continue
        stale_reviews.append(
            {
                key: review.get(key)
                for key in (
                    "job",
                    "product",
                    "path",
                    "status",
                    "generated_at",
                    "age_seconds",
                    "max_age_seconds",
                    "fresh",
                    "reason",
                    "needs_approval",
                )
                if key in review
            }
        )
    return stale_reviews


def _product_state_error_details(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    state_error_details = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        raw_errors = product.get("state_errors")
        if not raw_errors:
            continue
        if isinstance(raw_errors, list):
            state_errors = [
                item if isinstance(item, dict) else {"error": str(item)} for item in raw_errors
            ]
        else:
            state_errors = [
                {
                    "field": "state_errors",
                    "error": f"expected list, got {type(raw_errors).__name__}",
                }
            ]
        state_error_details.append(
            {
                "product": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("mode"),
                "state_errors": state_errors,
            }
        )
    return state_error_details


def _product_drawdown_halt_details(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    halted_products = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        if product.get("drawdown_halted") is not True:
            continue
        halted_products.append(
            {
                "product": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("mode"),
                "equity": product.get("equity"),
                "peak_equity": product.get("peak_equity"),
                "drawdown_fraction": product.get("drawdown_fraction"),
                "drawdown_limit_fraction": product.get("drawdown_limit_fraction"),
                "drawdown_halted_at": product.get("drawdown_halted_at"),
                "drawdown_halt_reason": product.get("drawdown_halt_reason"),
            }
        )
    return halted_products


def _product_exit_accounting_intent_details(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    pending_products = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        intent = product.get("exit_accounting_intent")
        if intent is None:
            continue
        pending_products.append(
            {
                "product": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("mode"),
                "intent": intent,
            }
        )
    return pending_products


def _product_recovery_state_details(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    pending_products = []
    state_keys = (
        "pending_order",
        "pending_entry_recovery",
        "risk_recovery_incident",
        "flatten_intent",
    )
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        states = {key: product[key] for key in state_keys if product.get(key) is not None}
        if not states:
            continue
        pending_products.append(
            {
                "product": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("mode"),
                "states": states,
            }
        )
    return pending_products


def _product_trade_log_issue_details(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    trade_log_issues = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        trade_summary = product.get("trade_summary")
        if not isinstance(trade_summary, dict):
            continue
        invalid_rows = trade_summary.get("invalid_rows")
        issue = trade_summary.get("issue")
        if not issue and not invalid_rows:
            continue
        trade_log_issues.append(
            {
                "product": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("mode"),
                "path": trade_summary.get("path"),
                "invalid_rows": invalid_rows,
                "issue": issue,
                "numeric_errors": trade_summary.get("numeric_errors", [])[:10]
                if isinstance(trade_summary.get("numeric_errors"), list)
                else [],
                **(
                    {
                        "exit_event_id_errors": trade_summary.get("exit_event_id_errors", [])[:10]
                        if isinstance(trade_summary.get("exit_event_id_errors"), list)
                        else []
                    }
                    if "exit_event_id_errors" in trade_summary
                    else {}
                ),
            }
        )
    return trade_log_issues


def _products_requiring_testnet_rehearsal(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    required_products = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False:
            continue
        if not bool(product.get("require_testnet_rehearsal")):
            continue
        required_products.append(product)
    return required_products


def _reporting_failure_detail(operator_report: dict[str, Any]) -> dict[str, Any] | None:
    reporting = operator_report.get("reporting")
    if not isinstance(reporting, dict):
        return None
    if reporting.get("ok") is not False:
        return None
    detail: dict[str, Any] = {}
    errors = reporting.get("errors")
    if isinstance(errors, list):
        detail["errors"] = [
            item if isinstance(item, dict) else {"error": str(item)} for item in errors
        ]
    elif errors:
        detail["errors"] = [{"error": str(errors)}]
    elif reporting.get("error"):
        detail["errors"] = [{"error": str(reporting.get("error"))}]
    else:
        detail["errors"] = []
    outputs = reporting.get("outputs")
    if isinstance(outputs, dict):
        detail["outputs"] = outputs
    for key in (
        "operator_report",
        "operator_report_json",
        "readiness_report",
        "readiness_report_json",
    ):
        if key in reporting and key not in detail:
            detail[key] = reporting.get(key)
    return detail


def _cycle_failure_detail(operator_report: dict[str, Any]) -> dict[str, Any] | None:
    control_error = operator_report.get("control_error")
    unknown_control_selectors = operator_report.get("unknown_control_selectors")
    control_clear = [
        item for item in _dict_list(operator_report, "control_clear") if item.get("ok") is False
    ]
    products = []
    for product in _dict_list(operator_report, "products"):
        cycle_ok = product.get("cycle_ok")
        if cycle_ok is not False and product.get("ok") is not False:
            continue
        detail = {
            key: product.get(key)
            for key in (
                "name",
                "objective",
                "market",
                "mode",
                "action",
                "reason",
                "error",
                "close_error",
                "broker",
                "flattened",
                "fill",
                "spot_step_aside",
                "local_state",
                "position_before",
                "position_after",
                "position_after_error",
                "position_after_attempt",
                "position_after_attempt_error",
                "cycle_errors",
                "state_errors",
            )
            if product.get(key) is not None
        }
        products.append(detail)

    jobs = []
    for job in _dict_list(operator_report, "jobs"):
        if job.get("ok") is not False:
            continue
        jobs.append(
            {
                key: job.get(key)
                for key in ("name", "returncode", "error", "stderr_tail", "stdout_tail")
                if job.get(key) is not None
            }
        )

    data_update = operator_report.get("data_update")
    failed_data_update = (
        data_update if isinstance(data_update, dict) and data_update.get("ok") is False else None
    )
    if (
        not products
        and not jobs
        and failed_data_update is None
        and not control_error
        and not unknown_control_selectors
        and not control_clear
    ):
        return None
    detail = {
        "ok": operator_report.get("ok"),
        "products": products,
        "jobs": jobs,
        "data_update": failed_data_update,
    }
    if control_error:
        detail["control_error"] = control_error
    if unknown_control_selectors:
        detail["unknown_control_selectors"] = unknown_control_selectors
    if control_clear:
        detail["control_clear"] = control_clear
    return detail


def _active_control_detail(operator_report: dict[str, Any]) -> dict[str, Any] | None:
    if operator_report.get("control_error"):
        return None
    control = operator_report.get("control")
    if not isinstance(control, dict):
        return None
    detail = {
        "paused": bool(control.get("paused")),
        "pause_jobs": bool(control.get("pause_jobs")),
        "paused_products": control.get("paused_products")
        if isinstance(control.get("paused_products"), list)
        else [],
        "paused_jobs": control.get("paused_jobs")
        if isinstance(control.get("paused_jobs"), list)
        else [],
        "flatten_all": bool(control.get("flatten_all")),
        "flatten_products": control.get("flatten_products")
        if isinstance(control.get("flatten_products"), list)
        else [],
    }
    active = (
        detail["paused"]
        or detail["pause_jobs"]
        or bool(detail["paused_products"])
        or bool(detail["paused_jobs"])
        or detail["flatten_all"]
        or bool(detail["flatten_products"])
    )
    if not active:
        return None
    reason = control.get("reason")
    if reason:
        detail["reason"] = reason
    return detail


def _testnet_rehearsal_issue_detail(
    testnet_rehearsal: dict[str, Any],
    *,
    live_products: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detail = {
        key: testnet_rehearsal.get(key)
        for key in (
            "status",
            "path",
            "required_by",
            "product",
            "generated_at",
            "fresh",
            "age_seconds",
            "max_age_seconds",
            "testnet",
            "final_position_flat",
            "invalid_reasons",
            "report_product",
            "expected_product",
            "risk_controls",
            "error",
            "next_action",
        )
        if key in testnet_rehearsal
    }
    if live_products:
        detail["live_products"] = [
            {
                key: product.get(key)
                for key in ("name", "objective", "market", "mode")
                if product.get(key) is not None
            }
            for product in live_products
        ]
    return detail


def _stale_open_positions(operator_report: dict[str, Any], now_ts: float) -> list[dict[str, Any]]:
    stale_positions = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False:
            continue
        for position in product.get("open_position_details", []) or []:
            if not isinstance(position, dict):
                continue
            entry_ts = _parse_timestamp(position.get("entry_time"))
            try:
                stale_after_seconds = float(position.get("stale_after_seconds"))
            except (TypeError, ValueError):
                continue
            if entry_ts is None or stale_after_seconds <= 0:
                continue
            age_seconds = now_ts - entry_ts
            if age_seconds < 0:
                continue
            if age_seconds <= stale_after_seconds:
                continue
            detail = {
                "product": product.get("name"),
                "mode": product.get("mode"),
                "market": product.get("market"),
                "strategy_id": position.get("strategy_id"),
                "direction": position.get("direction"),
                "entry_time": position.get("entry_time"),
                "age_seconds": round(age_seconds, 3),
                "stale_after_seconds": round(stale_after_seconds, 3),
                "base_timeframe": position.get("base_timeframe"),
                "horizon_bars": position.get("horizon_bars"),
            }
            if product.get("objective") is not None:
                detail["objective"] = product.get("objective")
            stale_positions.append(detail)
    return stale_positions


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _non_negative_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _open_position_risk_issues(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    risk_issues = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        for position in product.get("open_position_details", []) or []:
            if not isinstance(position, dict):
                continue
            reasons: list[str] = []
            position_size = _positive_float(position.get("position_size"))
            entry_price = _positive_float(position.get("entry_price"))
            sl_price = _positive_float(position.get("sl_price"))
            tp_price = _positive_float(position.get("tp_price"))
            if position_size is None:
                reasons.append("invalid_position_size")
            if entry_price is None:
                reasons.append("invalid_entry_price")
            if sl_price is None:
                reasons.append("invalid_stop_price")
            if tp_price is None:
                reasons.append("invalid_target_price")
            direction = str(position.get("direction") or "").lower()
            if direction not in {"long", "short"}:
                reasons.append("invalid_direction")
            elif entry_price is not None and sl_price is not None and tp_price is not None:
                if direction == "long" and not (sl_price < entry_price < tp_price):
                    reasons.append("invalid_long_stop_target_order")
                if direction == "short" and not (tp_price < entry_price < sl_price):
                    reasons.append("invalid_short_stop_target_order")
            if reasons:
                risk_issues.append(
                    {
                        "product": product.get("name"),
                        "objective": product.get("objective"),
                        "market": product.get("market"),
                        "strategy_id": position.get("strategy_id"),
                        "direction": position.get("direction"),
                        "position_size": position.get("position_size"),
                        "entry_price": position.get("entry_price"),
                        "sl_price": position.get("sl_price"),
                        "tp_price": position.get("tp_price"),
                        "reasons": reasons,
                    }
                )
    return risk_issues


def _live_open_position_risk_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    return _open_position_risk_issues(operator_report, mode="live")


def _paper_open_position_risk_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    return _open_position_risk_issues(operator_report, mode="paper")


def _live_open_position_broker_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    broker_issues = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != "live":
            continue
        try:
            open_count = int(product.get("open_positions"))
        except (TypeError, ValueError):
            continue
        if open_count <= 0:
            continue
        for position in product.get("open_position_details", []) or []:
            if not isinstance(position, dict):
                continue
            reasons: list[str] = []
            broker_symbol = position.get("broker_symbol")
            if not isinstance(broker_symbol, str) or not broker_symbol:
                reasons.append("invalid_broker_symbol")
            broker_side = str(position.get("broker_side") or "").lower()
            if broker_side not in {"buy", "sell"}:
                reasons.append("invalid_broker_side")
            direction = str(position.get("direction") or "").lower()
            if direction == "long" and broker_side != "buy":
                reasons.append("broker_side_direction_mismatch")
            if direction == "short" and broker_side != "sell":
                reasons.append("broker_side_direction_mismatch")
            broker_qty = _positive_float(position.get("broker_qty"))
            if broker_qty is None:
                reasons.append("invalid_broker_qty")
            broker_entry_price = _positive_float(position.get("broker_entry_price"))
            if broker_entry_price is None:
                reasons.append("invalid_broker_entry_price")
            broker_entry_fee = _non_negative_float(position.get("broker_entry_fee"))
            if broker_entry_fee is None:
                reasons.append("invalid_broker_entry_fee")
            requested_qty = _positive_float(position.get("broker_requested_qty"))
            if requested_qty is None:
                reasons.append("invalid_broker_requested_qty")
            fill_ratio = _positive_float(position.get("broker_fill_ratio"))
            if fill_ratio is None:
                reasons.append("invalid_broker_fill_ratio")
            elif abs(fill_ratio - 1.0) > 1e-6:
                reasons.append("broker_fill_ratio_not_complete")
            if (
                broker_qty is not None
                and requested_qty is not None
                and abs(broker_qty - requested_qty) > max(requested_qty * 1e-6, 1e-9)
            ):
                reasons.append("broker_qty_mismatch_requested")
            if product.get("market") == "futures":
                if _positive_float(position.get("broker_entry_balance")) is None:
                    reasons.append("invalid_broker_entry_balance")
                stop_order_id = position.get("broker_stop_order_id")
                if not isinstance(stop_order_id, str) or not stop_order_id.strip():
                    reasons.append("invalid_broker_stop_order_id")
                stop_client_id = position.get("broker_stop_client_id")
                if not isinstance(stop_client_id, str) or not stop_client_id.strip():
                    reasons.append("invalid_broker_stop_client_id")
                stop_trigger = _positive_float(position.get("broker_stop_trigger_price"))
                if stop_trigger is None:
                    reasons.append("invalid_broker_stop_trigger_price")
                else:
                    strategy_stop = _positive_float(position.get("sl_price"))
                    if strategy_stop is not None and abs(stop_trigger - strategy_stop) > max(
                        strategy_stop * 1e-9,
                        1e-12,
                    ):
                        reasons.append("broker_stop_trigger_mismatch_strategy_stop")
            if product.get("objective") == "btc_accumulation" and product.get("market") == "spot":
                exit_sizing = position.get("broker_exit_sizing")
                if direction == "short" and exit_sizing != "quote_reinvest":
                    reasons.append("invalid_spot_step_aside_exit_sizing")
                if (
                    direction == "short"
                    and _positive_float(position.get("broker_entry_quote_value")) is None
                ):
                    reasons.append("invalid_spot_step_aside_quote_value")
            if reasons:
                broker_issues.append(
                    {
                        "product": product.get("name"),
                        "objective": product.get("objective"),
                        "market": product.get("market"),
                        "strategy_id": position.get("strategy_id"),
                        "direction": position.get("direction"),
                        "broker_symbol": position.get("broker_symbol"),
                        "broker_side": position.get("broker_side"),
                        "broker_qty": position.get("broker_qty"),
                        "broker_requested_qty": position.get("broker_requested_qty"),
                        "broker_fill_ratio": position.get("broker_fill_ratio"),
                        "broker_entry_fee": position.get("broker_entry_fee"),
                        "broker_entry_balance": position.get("broker_entry_balance"),
                        "broker_stop_order_id": position.get("broker_stop_order_id"),
                        "broker_stop_client_id": position.get("broker_stop_client_id"),
                        "broker_stop_trigger_price": position.get("broker_stop_trigger_price"),
                        "reasons": reasons,
                    }
                )
    return broker_issues


def _open_position_monitoring_issues(
    operator_report: dict[str, Any], *, mode: str, now_ts: float
) -> list[dict[str, Any]]:
    monitoring_issues = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        for position in product.get("open_position_details", []) or []:
            if not isinstance(position, dict):
                continue
            reasons: list[str] = []
            entry_ts = _parse_timestamp(position.get("entry_time"))
            if entry_ts is None:
                reasons.append("invalid_entry_time")
            elif entry_ts > now_ts:
                reasons.append("future_entry_time")
            try:
                stale_after_seconds = float(position.get("stale_after_seconds"))
            except (TypeError, ValueError):
                stale_after_seconds = 0.0
            if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
                reasons.append("invalid_stale_after_seconds")
            if reasons:
                monitoring_issues.append(
                    {
                        "product": product.get("name"),
                        "objective": product.get("objective"),
                        "market": product.get("market"),
                        "strategy_id": position.get("strategy_id"),
                        "direction": position.get("direction"),
                        "entry_time": position.get("entry_time"),
                        "base_timeframe": position.get("base_timeframe"),
                        "horizon_bars": position.get("horizon_bars"),
                        "stale_after_seconds": position.get("stale_after_seconds"),
                        "reasons": reasons,
                    }
                )
    return monitoring_issues


def _live_open_position_monitoring_issues(
    operator_report: dict[str, Any], *, now_ts: float
) -> list[dict[str, Any]]:
    return _open_position_monitoring_issues(operator_report, mode="live", now_ts=now_ts)


def _paper_open_position_monitoring_issues(
    operator_report: dict[str, Any], *, now_ts: float
) -> list[dict[str, Any]]:
    return _open_position_monitoring_issues(operator_report, mode="paper", now_ts=now_ts)


def _open_position_visibility_issues(
    operator_report: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    visibility_issues = []
    for product in _dict_list(operator_report, "products"):
        if product.get("enabled") is False or product.get("mode") != mode:
            continue
        raw_count = product.get("open_positions")
        if raw_count is None:
            continue
        reasons: list[str] = []
        try:
            open_count_float = float(raw_count)
        except (TypeError, ValueError):
            open_count_float = float("nan")
        if (
            not math.isfinite(open_count_float)
            or open_count_float < 0
            or not open_count_float.is_integer()
        ):
            reasons.append("invalid_open_position_count")
            open_count = None
        else:
            open_count = int(open_count_float)
        details = product.get("open_position_details")
        if not isinstance(details, list):
            detail_count = None
            reasons.append("invalid_open_position_details")
        else:
            detail_count = len(details)
            if open_count is not None and open_count > detail_count:
                reasons.append("missing_open_position_details")
            if open_count is not None and detail_count > open_count:
                reasons.append("open_position_count_detail_mismatch")
        if reasons:
            visibility_issues.append(
                {
                    "product": product.get("name"),
                    "objective": product.get("objective"),
                    "market": product.get("market"),
                    "open_positions": raw_count,
                    "open_position_details_count": detail_count,
                    "reasons": reasons,
                }
            )
    return visibility_issues


def _live_open_position_visibility_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    return _open_position_visibility_issues(operator_report, mode="live")


def _paper_open_position_visibility_issues(operator_report: dict[str, Any]) -> list[dict[str, Any]]:
    return _open_position_visibility_issues(operator_report, mode="paper")


def evaluate_health(
    operator_report: dict[str, Any],
    *,
    readiness_report: dict[str, Any] | None = None,
    fail_on_job_failures: bool = True,
    fail_on_job_overdue: bool = True,
    now_ts: float | None = None,
    max_backup_age_seconds: float | None = None,
    max_consecutive_job_deferrals: int = 3,
) -> dict[str, Any]:
    """Convert report payloads into a compact health status and issue list."""

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    now_ts = now_ts if now_ts is not None else dt.datetime.now(dt.UTC).timestamp()
    malformed_sections = [
        detail
        for key in ("products", "jobs", "scheduled_jobs")
        if (detail := _malformed_dict_list_detail(operator_report, key)) is not None
    ]
    if malformed_sections:
        issues.append(
            _issue(
                "operator_report_malformed",
                "operator report has malformed collection sections",
                detail={"sections": malformed_sections},
            )
        )
    heartbeat = operator_report.get("status_heartbeat") or {}
    if heartbeat.get("fresh") is not True:
        issues.append(
            _issue(
                "stale_status",
                "autopilot status heartbeat is stale or missing",
                detail={
                    "generated_at": heartbeat.get("generated_at"),
                    "age_seconds": heartbeat.get("age_seconds"),
                    "limit_seconds": heartbeat.get("limit_seconds"),
                    "reason": heartbeat.get("reason"),
                },
            )
        )
    cycle_failure_detail = _cycle_failure_detail(operator_report)
    if operator_report.get("ok") is not True or cycle_failure_detail is not None:
        issues.append(
            _issue(
                "cycle_failed",
                "latest autopilot cycle did not complete successfully",
                detail=cycle_failure_detail or {"ok": operator_report.get("ok")},
            )
        )
    active_control_detail = _active_control_detail(operator_report)
    if active_control_detail is not None:
        warnings.append(
            _issue(
                "operator_control_active",
                "operator control is actively pausing or flattening the autopilot",
                detail=active_control_detail,
            )
        )
    runtime_load_errors = operator_report.get("runtime_load_errors") or []
    if runtime_load_errors:
        issues.append(
            _issue(
                "runtime_file_unreadable",
                "one or more runtime JSON files could not be read",
                detail={"files": runtime_load_errors},
            )
        )
    runtime_shape_errors = operator_report.get("runtime_shape_errors") or []
    if runtime_shape_errors:
        issues.append(
            _issue(
                "runtime_file_shape_invalid",
                "one or more runtime JSON files have malformed fields",
                detail={"files": runtime_shape_errors},
            )
        )
    reporting_failure = _reporting_failure_detail(operator_report)
    if reporting_failure is not None:
        warnings.append(
            _issue(
                "runtime_reporting_failed",
                "latest autopilot cycle could not refresh all operator reports",
                detail=reporting_failure,
            )
        )
    artifact_hygiene = operator_report.get("artifact_hygiene")
    if (
        isinstance(artifact_hygiene, dict)
        and artifact_hygiene
        and artifact_hygiene.get("ok") is False
    ):
        summary = (
            artifact_hygiene.get("summary")
            if isinstance(artifact_hygiene.get("summary"), dict)
            else {}
        )
        errors = (
            artifact_hygiene.get("errors")
            if isinstance(artifact_hygiene.get("errors"), list)
            else []
        )
        warnings.append(
            _issue(
                "artifact_hygiene_unhealthy",
                "artifact hygiene reported cleanup or inspection failures",
                detail={
                    "summary": {
                        "quarantine_candidates": summary.get("quarantine_candidates"),
                        "unreferenced_active_artifacts": summary.get(
                            "unreferenced_active_artifacts"
                        ),
                        "historical_search_outputs": summary.get("historical_search_outputs"),
                        "errors": summary.get("errors", len(errors)),
                        "quarantined": summary.get("quarantined"),
                    },
                    "errors": errors,
                },
            )
        )
    market_data_issue = _market_data_issue_detail(operator_report)
    if market_data_issue is not None:
        issues.append(
            _issue(
                "market_data_unhealthy",
                "one or more market data feeds are missing, stale, invalid, or timestamped in the future",
                detail=market_data_issue,
            )
        )
    live_state_errors = _product_state_error_details(operator_report, mode="live")
    if live_state_errors:
        issues.append(
            _issue(
                "live_product_state_invalid",
                "one or more live products reported invalid local state",
                detail={"products": live_state_errors},
            )
        )
    paper_state_errors = _product_state_error_details(operator_report, mode="paper")
    if paper_state_errors:
        warnings.append(
            _issue(
                "paper_product_state_invalid",
                "one or more paper products reported invalid local state",
                detail={"products": paper_state_errors},
            )
        )
    live_recovery_states = _product_recovery_state_details(operator_report, mode="live")
    if live_recovery_states:
        issues.append(
            _issue(
                "live_product_recovery_pending",
                "one or more live products have unresolved broker intents or safety-recovery incidents",
                detail={"products": live_recovery_states},
            )
        )
    paper_recovery_states = _product_recovery_state_details(operator_report, mode="paper")
    if paper_recovery_states:
        warnings.append(
            _issue(
                "paper_product_recovery_pending",
                "one or more paper products have unresolved broker intents or safety-recovery incidents",
                detail={"products": paper_recovery_states},
            )
        )
    live_drawdown_halts = _product_drawdown_halt_details(operator_report, mode="live")
    if live_drawdown_halts:
        issues.append(
            _issue(
                "live_product_drawdown_halted",
                "one or more live products hit the sticky peak-equity drawdown circuit breaker",
                detail={"products": live_drawdown_halts},
            )
        )
    paper_drawdown_halts = _product_drawdown_halt_details(operator_report, mode="paper")
    if paper_drawdown_halts:
        warnings.append(
            _issue(
                "paper_product_drawdown_halted",
                "one or more paper products hit the sticky peak-equity drawdown circuit breaker",
                detail={"products": paper_drawdown_halts},
            )
        )
    live_exit_accounting = _product_exit_accounting_intent_details(operator_report, mode="live")
    if live_exit_accounting:
        issues.append(
            _issue(
                "live_exit_accounting_pending",
                "one or more live products have an unresolved idempotent exit accounting intent",
                detail={"products": live_exit_accounting},
            )
        )
    paper_exit_accounting = _product_exit_accounting_intent_details(operator_report, mode="paper")
    if paper_exit_accounting:
        warnings.append(
            _issue(
                "paper_exit_accounting_pending",
                "one or more paper products have an unresolved idempotent exit accounting intent",
                detail={"products": paper_exit_accounting},
            )
        )
    live_trade_log_issues = _product_trade_log_issue_details(operator_report, mode="live")
    if live_trade_log_issues:
        issues.append(
            _issue(
                "live_trade_log_invalid",
                "one or more live products have invalid trade-log audit fields",
                detail={"products": live_trade_log_issues},
            )
        )
    paper_trade_log_issues = _product_trade_log_issue_details(operator_report, mode="paper")
    if paper_trade_log_issues:
        warnings.append(
            _issue(
                "paper_trade_log_invalid",
                "one or more paper products have invalid trade-log audit fields",
                detail={"products": paper_trade_log_issues},
            )
        )
    if readiness_report is not None and readiness_report.get("ok") is not True:
        blocking_checks = [
            {"name": item.get("name"), "level": item.get("level"), "detail": item.get("detail")}
            for item in _dict_list(readiness_report, "checks")
            if item.get("level") == "error" and not item.get("ok")
        ]
        issues.append(
            _issue(
                "readiness_blocked",
                "autopilot readiness has blocking failures",
                detail={"blocking_checks": blocking_checks},
            )
        )
    if readiness_report is not None:
        malformed_readiness_checks = _malformed_dict_list_detail(readiness_report, "checks")
        if malformed_readiness_checks is not None:
            issues.append(
                _issue(
                    "readiness_report_malformed",
                    "readiness report has malformed check entries",
                    detail=malformed_readiness_checks,
                )
            )
        warning_checks = [
            {"name": item.get("name"), "level": item.get("level"), "detail": item.get("detail")}
            for item in _dict_list(readiness_report, "checks")
            if item.get("level") == "warning" and not item.get("ok")
        ]
        if warning_checks:
            warnings.append(
                _issue(
                    "readiness_warning",
                    "autopilot readiness has warning-level failures",
                    detail={"warning_checks": warning_checks},
                )
            )
    raw_candidate_paper = operator_report.get("candidate_paper")
    candidate_paper = raw_candidate_paper if isinstance(raw_candidate_paper, dict) else {}
    if raw_candidate_paper is not None and not isinstance(raw_candidate_paper, dict):
        issues.append(
            _issue(
                "candidate_paper_status_malformed",
                "candidate paper status summary must be a JSON object",
                detail={"type": type(raw_candidate_paper).__name__},
            )
        )
    if candidate_paper.get("configured") is True and candidate_paper.get("enabled") is True:
        if candidate_paper.get("exists") is not True:
            warnings.append(
                _issue(
                    "candidate_paper_status_missing",
                    "candidate paper cycle is enabled but has not produced a status file",
                    detail={
                        "job": candidate_paper.get("job"),
                        "path": candidate_paper.get("path"),
                    },
                )
            )
        elif candidate_paper.get("ok") is not True:
            issues.append(
                _issue(
                    "candidate_paper_unhealthy",
                    "latest staged-candidate paper status is stale, invalid, or failed",
                    detail={
                        "job": candidate_paper.get("job"),
                        "path": candidate_paper.get("path"),
                        "status": candidate_paper.get("status"),
                        "generated_at": candidate_paper.get("generated_at"),
                        "age_seconds": candidate_paper.get("age_seconds"),
                        "max_age_seconds": candidate_paper.get("max_age_seconds"),
                        "fresh": candidate_paper.get("fresh"),
                        "reason": candidate_paper.get("reason"),
                        "error": candidate_paper.get("error"),
                        "errors": candidate_paper.get("errors") or [],
                        "open_positions": candidate_paper.get("open_positions"),
                        "activation_ready_products": candidate_paper.get(
                            "activation_ready_products"
                        )
                        or [],
                    },
                )
            )
        halted_products = candidate_paper.get("drawdown_halted_products") or []
        if halted_products:
            warnings.append(
                _issue(
                    "candidate_paper_drawdown_halted",
                    "one or more staged candidates are halted by paper drawdown controls",
                    detail={"products": halted_products},
                )
            )
    backup_report = operator_report.get("backup_report") or {}
    if backup_report:
        verification = backup_report.get("verification") or {}
        manifest = backup_report.get("manifest") or {}
        critical_skipped = [
            {
                "path": item.get("path"),
                "role": item.get("role"),
                "reason": item.get("reason"),
            }
            for item in _dict_list(manifest, "files")
            if item.get("required_if_present") is True
            and item.get("exists") is True
            and item.get("included") is not True
        ]
        reported_critical_skipped = _int_value(manifest.get("critical_skipped_files")) or 0
        if critical_skipped or reported_critical_skipped > 0:
            issues.append(
                _issue(
                    "backup_incomplete",
                    "latest backup omitted one or more existing recovery files",
                    detail={
                        "output": backup_report.get("output"),
                        "critical_skipped_files": max(
                            reported_critical_skipped,
                            len(critical_skipped),
                        ),
                        "files": critical_skipped[:10],
                    },
                )
            )
        if backup_report.get("ok") is not True or verification.get("ok") is not True:
            issues.append(
                _issue(
                    "backup_unhealthy",
                    "latest backup report failed or did not verify",
                    detail={
                        "ok": backup_report.get("ok"),
                        "output": backup_report.get("output"),
                        "verification_ok": verification.get("ok"),
                        "verification_issues": verification.get("issues") or [],
                    },
                )
            )
        else:
            generated_at = manifest.get("generated_at") or backup_report.get("generated_at")
            generated_ts = _parse_timestamp(generated_at)
            stale_limit = (
                float(max_backup_age_seconds)
                if max_backup_age_seconds is not None
                else _backup_stale_limit_seconds(operator_report)
            )
            if generated_ts is None:
                issues.append(
                    _issue(
                        "backup_timestamp_missing",
                        "latest backup report has no valid generation timestamp",
                        detail={
                            "output": backup_report.get("output"),
                            "generated_at": generated_at,
                        },
                    )
                )
            elif generated_ts > now_ts:
                issues.append(
                    _issue(
                        "backup_timestamp_future",
                        "latest verified backup is timestamped in the future",
                        detail={
                            "output": backup_report.get("output"),
                            "generated_at": generated_at,
                        },
                    )
                )
            else:
                age_seconds = now_ts - generated_ts
                if age_seconds > stale_limit:
                    issues.append(
                        _issue(
                            "backup_stale",
                            "latest verified backup is stale",
                            detail={
                                "output": backup_report.get("output"),
                                "generated_at": generated_at,
                                "age_seconds": round(age_seconds, 3),
                                "limit_seconds": round(stale_limit, 3),
                            },
                        )
                    )
    else:
        backup_jobs = _enabled_backup_jobs(operator_report)
        if backup_jobs:
            issues.append(
                _issue(
                    "backup_report_missing",
                    "backup job is enabled but no backup report is available",
                    detail={"scheduled_jobs": backup_jobs},
                )
            )
    scheduled_job_state_issues = _scheduled_job_state_issues(operator_report)
    if scheduled_job_state_issues:
        issues.append(
            _issue(
                "scheduled_job_state_invalid",
                "one or more enabled scheduled jobs reported invalid scheduler state",
                detail={"jobs": scheduled_job_state_issues},
            )
        )
    truncated_job_outputs = _scheduled_job_output_truncation_warnings(operator_report)
    if truncated_job_outputs:
        warnings.append(
            _issue(
                "scheduled_job_output_truncated",
                "one or more enabled scheduled jobs produced truncated output",
                detail={"jobs": truncated_job_outputs},
            )
        )
    deferred_jobs = _scheduled_job_deferral_warnings(operator_report)
    if deferred_jobs:
        warnings.append(
            _issue(
                "scheduled_job_deferred",
                "one or more enabled scheduled jobs were deferred by the per-cycle job limit",
                detail={"jobs": deferred_jobs},
            )
        )
    excessive_deferred_jobs = _scheduled_job_deferral_limit_issues(
        operator_report,
        max_consecutive_job_deferrals=max_consecutive_job_deferrals,
    )
    if excessive_deferred_jobs:
        issues.append(
            _issue(
                "scheduled_job_deferral_limit",
                "one or more enabled scheduled jobs exceeded the consecutive deferral limit",
                detail={"jobs": excessive_deferred_jobs},
            )
        )
    if fail_on_job_failures:
        failed_jobs = [
            _scheduled_job_failure_detail(job)
            for job in _dict_list(operator_report, "scheduled_jobs")
            if job.get("enabled") and job.get("status") == "fail"
        ]
        if failed_jobs:
            issues.append(
                _issue(
                    "scheduled_job_failed",
                    "one or more enabled scheduled jobs are failing",
                    detail={"jobs": failed_jobs},
                )
            )
    if fail_on_job_overdue:
        never_run_jobs = []
        overdue_jobs = []
        for job in _dict_list(operator_report, "scheduled_jobs"):
            if not job.get("enabled"):
                continue
            if job.get("status") in {"disabled", "fail"}:
                continue
            if job.get("due") is not True:
                continue
            if job.get("status") == "never_run":
                never_run_jobs.append(
                    {
                        "name": job.get("name"),
                        "status": job.get("status"),
                        "due": job.get("due"),
                        "cadence_seconds": job.get("cadence_seconds"),
                        "effective_cadence_seconds": job.get("effective_cadence_seconds"),
                        "timeout_seconds": job.get("timeout_seconds"),
                    }
                )
                continue
            try:
                age_seconds = float(job.get("age_seconds"))
                cadence_seconds = float(
                    job.get("effective_cadence_seconds") or job.get("cadence_seconds")
                )
            except (TypeError, ValueError):
                continue
            if age_seconds < 0 or cadence_seconds <= 0:
                continue
            limit_seconds = max(
                cadence_seconds * 2.0, cadence_seconds + float(job.get("timeout_seconds") or 0)
            )
            if age_seconds > limit_seconds:
                overdue_jobs.append(
                    {
                        "name": job.get("name"),
                        "status": job.get("status"),
                        "due": job.get("due"),
                        "age_seconds": round(age_seconds, 3),
                        "limit_seconds": round(limit_seconds, 3),
                        "effective_cadence_seconds": round(cadence_seconds, 3),
                        "last_started_at": job.get("last_started_at"),
                        "last_reason": job.get("last_reason"),
                    }
                )
        if overdue_jobs:
            issues.append(
                _issue(
                    "scheduled_job_overdue",
                    "one or more enabled scheduled jobs are overdue",
                    detail={"jobs": overdue_jobs},
                )
            )
        if never_run_jobs:
            issues.append(
                _issue(
                    "scheduled_job_never_ran",
                    "one or more enabled scheduled jobs are due but have never run",
                    detail={"jobs": never_run_jobs},
                )
            )
    waiting_paper_products = _paper_products_waiting_for_artifacts(operator_report)
    for product in waiting_paper_products:
        warnings.append(
            _issue(
                "paper_product_waiting_for_strategy_artifact",
                "paper product is waiting for an exported strategy artifact",
                detail={
                    key: product.get(key)
                    for key in ("name", "objective", "market", "strategy_artifact", "detail")
                    if product.get(key) is not None
                },
            )
        )
    stale_positions = _stale_open_positions(operator_report, now_ts)
    live_stale_positions = [
        position for position in stale_positions if position.get("mode") == "live"
    ]
    paper_stale_positions = [
        position for position in stale_positions if position.get("mode") != "live"
    ]
    if live_stale_positions:
        issues.append(
            _issue(
                "live_open_position_stale",
                "one or more live open positions are older than their expected strategy horizon",
                detail={"positions": live_stale_positions},
            )
        )
    if paper_stale_positions:
        warnings.append(
            _issue(
                "open_position_stale",
                "one or more open positions are older than their expected strategy horizon",
                detail={"positions": paper_stale_positions},
            )
        )
    live_risk_issues = _live_open_position_risk_issues(operator_report)
    if live_risk_issues:
        issues.append(
            _issue(
                "live_open_position_risk_invalid",
                "one or more live open positions have missing or invalid risk metadata",
                detail={"positions": live_risk_issues},
            )
        )
    live_broker_issues = _live_open_position_broker_issues(operator_report)
    if live_broker_issues:
        issues.append(
            _issue(
                "live_open_position_broker_invalid",
                "one or more live open positions have missing or invalid broker metadata",
                detail={"positions": live_broker_issues},
            )
        )
    paper_risk_issues = _paper_open_position_risk_issues(operator_report)
    if paper_risk_issues:
        warnings.append(
            _issue(
                "paper_open_position_risk_invalid",
                "one or more paper open positions have missing or invalid risk metadata",
                detail={"positions": paper_risk_issues},
            )
        )
    live_monitoring_issues = _live_open_position_monitoring_issues(operator_report, now_ts=now_ts)
    if live_monitoring_issues:
        issues.append(
            _issue(
                "live_open_position_monitoring_invalid",
                "one or more live open positions have missing or invalid monitoring metadata",
                detail={"positions": live_monitoring_issues},
            )
        )
    paper_monitoring_issues = _paper_open_position_monitoring_issues(operator_report, now_ts=now_ts)
    if paper_monitoring_issues:
        warnings.append(
            _issue(
                "paper_open_position_monitoring_invalid",
                "one or more paper open positions have missing or invalid monitoring metadata",
                detail={"positions": paper_monitoring_issues},
            )
        )
    live_visibility_issues = _live_open_position_visibility_issues(operator_report)
    if live_visibility_issues:
        issues.append(
            _issue(
                "live_open_position_visibility_invalid",
                "one or more live products report open positions without matching position details",
                detail={"products": live_visibility_issues},
            )
        )
    paper_visibility_issues = _paper_open_position_visibility_issues(operator_report)
    if paper_visibility_issues:
        warnings.append(
            _issue(
                "paper_open_position_visibility_invalid",
                "one or more paper products report open positions without matching position details",
                detail={"products": paper_visibility_issues},
            )
        )
    research_cycle = (
        operator_report.get("research_cycle")
        if isinstance(operator_report.get("research_cycle"), dict)
        else {}
    )
    research_summary = (
        research_cycle.get("summary") if isinstance(research_cycle.get("summary"), dict) else {}
    )
    if research_cycle.get("state_recovered") is True:
        warnings.append(
            _issue(
                "research_cycle_state_recovered",
                "research cycle recovered a corrupt or invalid state file",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "state_error": research_cycle.get("state_error"),
                },
            )
        )
    cycle_mutation_batch = research_cycle.get("mutation_batch")
    if (
        isinstance(cycle_mutation_batch, dict)
        and cycle_mutation_batch.get("status") == "read_error"
    ):
        warnings.append(
            _issue(
                "research_cycle_mutation_batch_read_error",
                "research cycle ignored a mutation batch it could not read",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "path": cycle_mutation_batch.get("path"),
                    "error": cycle_mutation_batch.get("error"),
                },
            )
        )
    cycle_generated_batch = research_cycle.get("generated_batch")
    if isinstance(cycle_generated_batch, dict) and cycle_generated_batch.get("status") in {
        "read_error",
        "invalid",
        "ignored",
    }:
        warnings.append(
            _issue(
                "research_cycle_generated_batch_unhealthy",
                "research cycle could not safely consume the generated strategy batch",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "status": cycle_generated_batch.get("status"),
                    "path": cycle_generated_batch.get("path"),
                    "reason": cycle_generated_batch.get("reason"),
                    "error": cycle_generated_batch.get("error"),
                },
            )
        )
    elif research_cycle.get("error") == "generated_batch_not_ready":
        warnings.append(
            _issue(
                "research_cycle_generated_batch_missing",
                "generated-only research is waiting for a valid generated strategy batch",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "status": (
                        cycle_generated_batch.get("status")
                        if isinstance(cycle_generated_batch, dict)
                        else None
                    ),
                    "path": (
                        cycle_generated_batch.get("path")
                        if isinstance(cycle_generated_batch, dict)
                        else None
                    ),
                },
            )
        )

    generated_batch = (
        operator_report.get("generated_batch")
        if isinstance(operator_report.get("generated_batch"), dict)
        else {}
    )
    if generated_batch and not generated_batch.get("_load_error"):
        unsafe_flags = [
            name
            for name, expected in (
                ("schema", "autopilot.generative_strategy_batch/v1"),
                ("research_only", True),
                ("executable", False),
                ("paper_trade_allowed", False),
                ("promotion_allowed", False),
                ("live_allowed", False),
                ("requires_full_validation_before_export", True),
            )
            if generated_batch.get(name) != expected
        ]
        if generated_batch.get("ok") is False or unsafe_flags:
            warnings.append(
                _issue(
                    "generated_batch_unhealthy",
                    "latest generated strategy batch failed or violates its research-only contract",
                    detail={
                        "generated_at": generated_batch.get("generated_at"),
                        "artifact": generated_batch.get("artifact"),
                        "error": generated_batch.get("error"),
                        "unsafe_flags": unsafe_flags,
                    },
                )
            )
        generated_summary = (
            generated_batch.get("summary")
            if isinstance(generated_batch.get("summary"), dict)
            else {}
        )
        generated_count = generated_batch.get("hypotheses_count")
        if generated_count is None:
            generated_count = generated_summary.get("hypotheses")
        if generated_batch.get("ok") is True and int(generated_count or 0) == 0:
            warnings.append(
                _issue(
                    "generative_search_empty",
                    "strategy factory completed without emitting any research hypotheses",
                    detail={
                        "generated_at": generated_batch.get("generated_at"),
                        "rejected_attempts": generated_summary.get("rejected_attempts"),
                        "unique_behavioral_specs": generated_summary.get("unique_behavioral_specs"),
                    },
                )
            )

    experiment_memory = (
        operator_report.get("experiment_memory")
        if isinstance(operator_report.get("experiment_memory"), dict)
        else {}
    )
    if experiment_memory:
        integrity = (
            experiment_memory.get("integrity")
            if isinstance(experiment_memory.get("integrity"), dict)
            else {}
        )
        if (
            experiment_memory.get("status") == "error"
            or experiment_memory.get("ok") is False
            or integrity.get("ok") is False
        ):
            issues.append(
                _issue(
                    "experiment_memory_unhealthy",
                    "strategy experiment memory failed integrity validation",
                    detail={
                        "path": experiment_memory.get("path"),
                        "status": experiment_memory.get("status"),
                        "error": experiment_memory.get("error"),
                        "integrity": integrity,
                    },
                )
            )
        if experiment_memory.get("protected_holdout_results_excluded") is not True:
            issues.append(
                _issue(
                    "experiment_memory_feedback_scope_invalid",
                    "experiment-memory reporting does not confirm protected holdout exclusion",
                    detail={
                        "path": experiment_memory.get("path"),
                        "adaptive_evidence_scope": experiment_memory.get("adaptive_evidence_scope"),
                    },
                )
            )
        enabled_factory_jobs = [
            job
            for job in _dict_list(operator_report, "scheduled_jobs")
            if job.get("enabled") and job.get("name") == "research_factory"
        ]
        factory_has_run = any(job.get("last_started_at") for job in enabled_factory_jobs)
        if experiment_memory.get("status") == "missing" and factory_has_run:
            warnings.append(
                _issue(
                    "experiment_memory_missing",
                    "strategy factory has run but durable experiment memory is missing",
                    detail={
                        "path": experiment_memory.get("path"),
                        "factory_jobs": [job.get("name") for job in enabled_factory_jobs],
                    },
                )
            )
    history_coverage = (
        research_cycle.get("history_coverage")
        if isinstance(research_cycle.get("history_coverage"), dict)
        else {}
    )
    coverage_failures = int(
        research_summary.get("coverage_failures") or history_coverage.get("failure_count") or 0
    )
    coverage_failed_scenarios = (
        research_summary.get("coverage_failed_scenarios")
        or history_coverage.get("failed_scenarios")
        or []
    )
    if coverage_failures > 0:
        warnings.append(
            _issue(
                "research_history_coverage_insufficient",
                "one or more curated research scenarios lack their required history depth",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "coverage_failures": coverage_failures,
                    "scenarios": coverage_failed_scenarios,
                    "next_actions": research_summary.get("next_actions") or [],
                },
            )
        )
    if (
        waiting_paper_products
        and research_cycle.get("ok") is True
        and int(research_summary.get("hypotheses") or 0) > 0
        and int(research_summary.get("keepers") or 0) == 0
        and int(research_summary.get("exported") or 0) == 0
    ):
        mutation_effectiveness = research_summary.get("mutation_effectiveness")
        warnings.append(
            _issue(
                "research_cycle_no_exportable_strategies",
                "research has run but found no exportable strategy candidates",
                detail={
                    "generated_at": research_cycle.get("generated_at"),
                    "hypotheses": research_summary.get("hypotheses"),
                    "top_reasons": research_summary.get("top_reasons") or {},
                    "next_actions": research_summary.get("next_actions") or [],
                    "waiting_products": [
                        {
                            "name": product.get("name"),
                            "objective": product.get("objective"),
                            "market": product.get("market"),
                        }
                        for product in waiting_paper_products
                    ],
                    **(
                        {"mutation_effectiveness": mutation_effectiveness}
                        if isinstance(mutation_effectiveness, dict)
                        else {}
                    ),
                },
            )
        )
    research_handoff_detail = research_handoff_warning_detail(operator_report)
    if research_handoff_detail["warnings"]:
        warnings.append(
            _issue(
                "research_handoff_warning",
                "research handoff artifacts are stale, unsafe, or failed",
                detail=research_handoff_detail,
            )
        )
    stale_promotion_reviews = _stale_promotion_reviews(operator_report)
    if stale_promotion_reviews:
        warnings.append(
            _issue(
                "promotion_review_stale",
                "one or more promotion review packets are stale or missing a valid timestamp",
                detail={"reviews": stale_promotion_reviews},
            )
        )
    testnet_rehearsal = (
        operator_report.get("testnet_rehearsal")
        if isinstance(operator_report.get("testnet_rehearsal"), dict)
        else {}
    )
    required_testnet_products = _products_requiring_testnet_rehearsal(operator_report)
    live_testnet_products = [
        product for product in required_testnet_products if product.get("mode") == "live"
    ]
    rehearsal_required = bool(testnet_rehearsal.get("required")) or bool(required_testnet_products)
    if rehearsal_required and testnet_rehearsal.get("ok") is not True:
        detail = _testnet_rehearsal_issue_detail(
            testnet_rehearsal, live_products=live_testnet_products
        )
        if live_testnet_products:
            issues.append(
                _issue(
                    "live_required_testnet_rehearsal_not_ready",
                    "live product requires a successful recent exchange testnet rehearsal",
                    detail=detail,
                )
            )
        else:
            warnings.append(
                _issue(
                    "required_testnet_rehearsal_not_ready",
                    "required exchange testnet rehearsal is missing, stale, or failed",
                    detail=detail,
                )
            )
    for key, code, message in (
        ("readiness_alert", "readiness_warning_alert", "autopilot has active readiness warnings"),
        (
            "research_handoff_alert",
            "research_handoff_warning_alert",
            "autopilot has active research handoff warnings",
        ),
        (
            "research_progress_alert",
            "research_progress_warning_alert",
            "autopilot has active research progress warnings",
        ),
        (
            "testnet_rehearsal_alert",
            "testnet_rehearsal_warning_alert",
            "autopilot has active testnet rehearsal warnings",
        ),
        ("promotion_alert", "promotion_warning_alert", "autopilot has active promotion warnings"),
    ):
        alert = operator_report.get(key)
        if not isinstance(alert, dict):
            continue
        if alert.get("sent") is not True and alert.get("reason") != "cooldown":
            continue
        detail = {
            item_key: alert.get(item_key)
            for item_key in ("sent", "reason", "fingerprint", "webhook", "state_error")
            if item_key in alert
        }
        warnings.append(_issue(code, message, detail=detail))
    memory_feedback = (
        experiment_memory.get("feedback")
        if isinstance(experiment_memory.get("feedback"), dict)
        else {}
    )
    memory_totals = (
        memory_feedback.get("totals") if isinstance(memory_feedback.get("totals"), dict) else {}
    )
    generated_summary = (
        generated_batch.get("summary") if isinstance(generated_batch.get("summary"), dict) else {}
    )
    generative_research = {
        "batch_ok": generated_batch.get("ok"),
        "batch_generated_at": generated_batch.get("generated_at"),
        "batch_hypotheses": generated_batch.get(
            "hypotheses_count", generated_summary.get("hypotheses")
        ),
        "unique_behavioral_specs": memory_totals.get("strategies"),
        "recorded_evaluations": memory_totals.get("evaluations"),
        "memory_status": experiment_memory.get("status"),
        "memory_integrity_ok": (
            (experiment_memory.get("integrity") or {}).get("ok")
            if isinstance(experiment_memory.get("integrity"), dict)
            else None
        ),
        "adaptive_evidence_scope": experiment_memory.get("adaptive_evidence_scope"),
        "protected_holdout_results_excluded": experiment_memory.get(
            "protected_holdout_results_excluded"
        ),
    }
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "status_generated_at": operator_report.get("status_generated_at"),
        "operator_report_generated_at": operator_report.get("generated_at"),
        "readiness_ok": None if readiness_report is None else readiness_report.get("ok"),
        "generative_research": generative_research,
        "candidate_paper": candidate_paper,
    }


def build_healthcheck(
    config: AutopilotConfig,
    *,
    include_readiness: bool = True,
    fail_on_job_failures: bool = True,
    fail_on_job_overdue: bool = True,
    max_backup_age_seconds: float | None = None,
    emit_failure_alert: bool = False,
) -> dict[str, Any]:
    operator_report = build_operator_report(config)
    readiness_report = (
        build_readiness_report(config, require_core_products=True, require_core_jobs=True)
        if include_readiness
        else None
    )
    health = evaluate_health(
        operator_report,
        readiness_report=readiness_report,
        fail_on_job_failures=fail_on_job_failures,
        fail_on_job_overdue=fail_on_job_overdue,
        max_backup_age_seconds=max_backup_age_seconds,
        max_consecutive_job_deferrals=config.max_consecutive_job_deferrals,
    )
    if emit_failure_alert and health.get("issues"):
        try:
            health["healthcheck_alert"] = emit_alert(
                alert_file=config.alert_file,
                state_file=config.alert_state_file,
                severity="critical",
                title="autopilot healthcheck failed",
                detail={
                    "issues": health.get("issues", []),
                    "warning_count": len(health.get("warnings", [])),
                    "status_generated_at": health.get("status_generated_at"),
                    "operator_report_generated_at": health.get("operator_report_generated_at"),
                    "readiness_ok": health.get("readiness_ok"),
                },
            )
        except Exception as exc:  # healthcheck output must survive alert I/O failures
            LOGGER.exception("Failed to emit healthcheck alert")
            health["healthcheck_alert"] = {"sent": False, "error": str(exc)}
    return health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exit nonzero when the autopilot needs operator attention."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--skip-readiness",
        action="store_true",
        help="Only check the last status/report payload, not current readiness.",
    )
    parser.add_argument(
        "--ignore-job-failures",
        action="store_true",
        help="Do not fail when an enabled scheduled job is still marked failing.",
    )
    parser.add_argument(
        "--ignore-job-overdue",
        action="store_true",
        help="Do not fail when an enabled scheduled job is more than two cadences overdue.",
    )
    parser.add_argument(
        "--max-backup-age-hours",
        type=float,
        help="Fail if the latest verified backup is older than this. Defaults to twice the backup job cadence.",
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Do not emit a configured alert when healthcheck finds blocking issues.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        health = build_healthcheck(
            config,
            include_readiness=not args.skip_readiness,
            fail_on_job_failures=not args.ignore_job_failures,
            fail_on_job_overdue=not args.ignore_job_overdue,
            max_backup_age_seconds=(
                args.max_backup_age_hours * 3600.0
                if args.max_backup_age_hours is not None
                else None
            ),
            emit_failure_alert=not args.no_alert,
        )
        health["config"] = str(args.config)
    except Exception as exc:
        LOGGER.exception("Failed to build healthcheck")
        health = {
            "ok": False,
            "issues": [
                _issue(
                    "healthcheck_build_failed",
                    "healthcheck could not load config or build its report",
                    detail={"config": str(args.config), "error": f"{type(exc).__name__}: {exc}"},
                )
            ],
            "warnings": [],
            "config": str(args.config),
        }
    _drain_oneshot_remote_alert(health)
    if args.output:
        try:
            write_json_atomic(args.output, health)
        except Exception as exc:
            health["ok"] = False
            health.setdefault("issues", []).append(
                _issue(
                    "healthcheck_output_write_failed",
                    "healthcheck could not write its JSON output file",
                    detail={"path": str(args.output), "error": f"{type(exc).__name__}: {exc}"},
                )
            )
            health["output_error"] = {"path": str(args.output), "error": str(exc)}
    payload = json.dumps(health, indent=2, sort_keys=True)
    print(payload)
    raise SystemExit(0 if health["ok"] else 1)


if __name__ == "__main__":
    main()
