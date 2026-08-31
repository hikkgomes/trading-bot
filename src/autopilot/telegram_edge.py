"""Deterministic Telegram alerts and a deliberately tiny operator channel.

The bot is not an execution interface.  Incoming messages are restricted to a
sanitized status view and irreversible-in-the-safe-direction pause operations.
There is intentionally no command for approval, activation, resume, flattening,
risk changes, configuration changes, or order placement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, load_config
from src.autopilot.control import load_control, update_control
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now
from src.config import PROJECT_ROOT
from src.envfile import parse_env_value

TELEGRAM_API_ROOT = "https://api.telegram.org"
TELEGRAM_MAX_TEXT = 4096
TELEGRAM_SAFE_TEXT = 4000
DEFAULT_POLL_STATE = Path("runtime/telegram/telegram_poll_state.json")
LEGACY_POLL_STATE = Path("runtime/telegram_poll_state.json")
DEFAULT_JOB_WORKER_STATUS = Path("runtime/job_worker_status.json")
DEFAULT_RESEARCH_CYCLE = Path("runtime/research_cycle.json")
DEFAULT_MARKET_UNIVERSE = Path("runtime/market_universe.json")
DEFAULT_OPENCLAW_REVIEW_AUDIT = Path("runtime/openclaw/review_audit.jsonl")
DEFAULT_OPENCLAW_INGEST_STATUS = Path("runtime/research_inbox/openclaw/ingest_status.json")
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "runtime" / "telegram.env"
ALLOWED_API_METHODS = frozenset({"getUpdates", "sendMessage"})
ALLOWED_UPDATE_TYPES = ["message", "channel_post"]
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "signature",
    "token",
    "webhook",
)
TELEGRAM_SETTINGS_KEYS = frozenset(
    {
        "AUTOPILOT_TELEGRAM_BOT_TOKEN",
        "AUTOPILOT_TELEGRAM_CHAT_ID",
        "AUTOPILOT_TELEGRAM_PAUSE_COMMANDS",
        "AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS",
    }
)
PROTECTED_RESEARCH_KEY_MARKERS = (
    "holdout",
    "final_holdout",
    "final_metrics",
    "final_outcome",
    "final_result",
    "final_test",
    "locked_test",
    "test_set",
)
DIAGNOSTIC_KEYS = frozenset(
    {
        "diagnostic",
        "diagnostics",
        "error",
        "errors",
        "exception",
        "exceptions",
        "raw_request",
        "raw_response",
        "request_headers",
        "request_url",
        "response_body",
        "response_headers",
        "stack",
        "stack_trace",
        "stderr",
        "stderr_tail",
        "stdout",
        "stdout_tail",
        "trace",
        "traceback",
    }
)
_PROTECTED_RESEARCH_TEXT = re.compile(
    r"(?i)(?:\b[A-Za-z0-9_-]*holdout[A-Za-z0-9_-]*\b|"
    r"\b[A-Za-z0-9_-]*final[\s_-]*(?:test|metrics?|outcome|result)"
    r"[A-Za-z0-9_-]*\b|\b[A-Za-z0-9_-]*locked[\s_-]*test[A-Za-z0-9_-]*\b|"
    r"\b[A-Za-z0-9_-]*test[\s_-]*set[A-Za-z0-9_-]*\b)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|apikey|api[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|authorization|credential|password|private[_-]?key|"
    r"secret|signature|token|x-mbx-apikey|exchange_api_key|"
    r"exchange_api_secret|awsaccesskeyid|x-amz-(?:credential|signature|security-token)|"
    r"x-goog-(?:credential|signature))\b\s*[:=]\s*)([^\s,;&]+)"
)
_CREDENTIAL_QUERY_PARAMETER = re.compile(
    r"(?i)([?&](?:api[_-]?key|apikey|api[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|authorization|auth|credential|password|secret|"
    r"signature|token|awsaccesskeyid|x-amz-(?:credential|signature|security-token)|"
    r"x-goog-(?:credential|signature))=)([^&#\s]+)"
)
_URL_USERINFO = re.compile(r"(?i)(https?://)([^/@\s:]+):([^/@\s]+)@")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
_TELEGRAM_BOT_TOKEN = re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{6,}\b")
_OMIT = object()
SAFE_CONTROL_KEYS = (
    "paused",
    "pause_jobs",
    "paused_products",
    "paused_jobs",
    "flatten_all",
    "flatten_products",
    "reason",
)
SAFE_RESEARCH_SUMMARY_KEYS = (
    "active_exports",
    "available_hypotheses",
    "coverage_failures",
    "export_reasons",
    "exported",
    "generative_search",
    "hypotheses",
    "incubation_candidates",
    "keepers",
    "mutation_effectiveness",
    "opportunity_types",
    "opportunity_types_by_product",
    "scenarios",
    "selected_hypotheses",
    "scenario_errors",
    "staged",
    "top_reasons",
    "unprotected_epoch_deferrals",
    "unsupported_hypotheses",
    "verdicts",
)


class TelegramError(RuntimeError):
    """Raised for Telegram configuration or API failures without exposing tokens."""


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str = field(repr=False)
    chat_id: str
    allowed_user_ids: frozenset[int] = frozenset()
    pause_commands_enabled: bool = False

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        required: bool = False,
    ) -> TelegramSettings | None:
        source = os.environ if environ is None else environ
        values: Mapping[str, str] = _environment_with_private_settings(source)
        token = str(values.get("AUTOPILOT_TELEGRAM_BOT_TOKEN", "")).strip()
        chat_id = str(values.get("AUTOPILOT_TELEGRAM_CHAT_ID", "")).strip()
        if not token and not chat_id and not required:
            return None
        if not token or not chat_id:
            raise TelegramError(
                "Telegram requires AUTOPILOT_TELEGRAM_BOT_TOKEN and "
                "AUTOPILOT_TELEGRAM_CHAT_ID together"
            )
        allowed_user_ids = _parse_id_set(
            str(values.get("AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS", "")),
            field_name="AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS",
        )
        pause_enabled = _parse_bool(
            str(values.get("AUTOPILOT_TELEGRAM_PAUSE_COMMANDS", "0")),
            field_name="AUTOPILOT_TELEGRAM_PAUSE_COMMANDS",
        )
        if pause_enabled and not allowed_user_ids:
            raise TelegramError(
                "Telegram pause commands require an explicit "
                "AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS allowlist"
            )
        return cls(
            bot_token=token,
            chat_id=chat_id,
            allowed_user_ids=allowed_user_ids,
            pause_commands_enabled=pause_enabled,
        )


def _environment_with_private_settings(environ: Mapping[str, str]) -> dict[str, str]:
    """Overlay Telegram-only settings without loading the trading ``.env``."""

    values = dict(environ)
    configured_path = str(values.get("AUTOPILOT_TELEGRAM_SETTINGS_FILE", "")).strip()
    if not configured_path:
        return values
    path = Path(configured_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return values
    if path.is_symlink() or not path.is_file():
        raise TelegramError(f"Telegram settings must be a non-symlink regular file: {path}")
    if path.stat().st_size > 64 * 1024:
        raise TelegramError(f"Telegram settings file is unexpectedly large: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise TelegramError(f"Telegram settings must not be group/world accessible: {path}")
    file_values = load_settings_file(path)
    for key, value in file_values.items():
        values.setdefault(key, value)
    return values


def _scan_env_value_syntax(raw: str) -> tuple[str | None, bool, int | None]:
    quote: str | None = None
    escaped = False
    comment_at: int | None = None
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (index == 0 or raw[index - 1].isspace()):
            comment_at = index
            break
    return quote, escaped, comment_at


def _validate_env_value_syntax(raw: str, *, line_number: int) -> None:
    quote, escaped, comment_at = _scan_env_value_syntax(raw)
    if quote is not None or escaped or "\x00" in raw:
        raise TelegramError(f"Telegram settings line {line_number} has malformed value syntax")
    syntax = raw[:comment_at].strip() if comment_at is not None else raw.strip()
    quote_chars = {"'", '"'}
    if syntax and syntax[0] in quote_chars:
        if len(syntax) < 2 or syntax[-1] != syntax[0]:
            raise TelegramError(f"Telegram settings line {line_number} has malformed value syntax")
    elif any(char in quote_chars for char in syntax):
        raise TelegramError(f"Telegram settings line {line_number} has malformed value syntax")


def _parse_settings_line(raw: str, *, line_number: int) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        raise TelegramError(
            f"Telegram settings line {line_number} must be a KEY=value assignment"
        )
    key, _, raw_value = line.partition("=")
    key = key.strip()
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
        raise TelegramError(f"Telegram settings line {line_number} has malformed key")
    if key not in TELEGRAM_SETTINGS_KEYS:
        raise TelegramError(f"Telegram settings line {line_number} uses an unknown key")
    _validate_env_value_syntax(raw_value, line_number=line_number)
    return key, parse_env_value(raw_value)


def load_settings_file(path: Path) -> dict[str, str]:
    """Strictly load the Telegram-only settings file without exposing values."""

    path = Path(path)
    if path.is_symlink() or not path.exists() or not path.is_file():
        raise TelegramError(f"Telegram settings must be a non-symlink regular file: {path}")
    if path.stat().st_size > 64 * 1024:
        raise TelegramError(f"Telegram settings file is unexpectedly large: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise TelegramError(f"Telegram settings must not be group/world accessible: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TelegramError(f"cannot read Telegram settings file: {type(exc).__name__}") from exc
    values: dict[str, str] = {}
    for line_number, raw in enumerate(lines, start=1):
        parsed = _parse_settings_line(raw, line_number=line_number)
        if parsed is None:
            continue
        key, parsed_value = parsed
        if key in values:
            raise TelegramError(f"Telegram settings line {line_number} duplicates a key")
        values[key] = parsed_value
    return values


def validate_settings_file(path: Path) -> dict[str, Any]:
    values = load_settings_file(path)
    settings = TelegramSettings.from_environment(values, required=True)
    if settings is None:
        raise TelegramError("required Telegram settings unexpectedly resolved to empty")
    return {
        "ok": True,
        "settings_file": str(path),
        "pause_commands_enabled": settings.pause_commands_enabled,
        "allowed_user_count": len(settings.allowed_user_ids),
    }


def _parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise TelegramError(f"{field_name} must be 0 or 1")


def _parse_id_set(value: str, *, field_name: str) -> frozenset[int]:
    identifiers: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise TelegramError(f"{field_name} must be a comma-separated integer list") from exc
        if parsed <= 0:
            raise TelegramError(f"{field_name} must contain positive user IDs")
        identifiers.add(parsed)
    return frozenset(identifiers)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)


def _is_protected_research_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized == "final" or any(
        marker in normalized for marker in PROTECTED_RESEARCH_KEY_MARKERS
    )


def _is_diagnostic_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized in DIAGNOSTIC_KEYS
        or normalized.startswith(("error_", "exception_", "stderr_", "stdout_", "traceback_"))
        or normalized.endswith("_error")
        or normalized.endswith("_errors")
        or normalized.endswith("_exception")
        or normalized.endswith("_exceptions")
        or normalized.endswith("_traceback")
    )


def _redact_secret_patterns(value: str) -> str:
    value = _BEARER_TOKEN.sub("Bearer [redacted]", value)
    value = _CREDENTIAL_QUERY_PARAMETER.sub(r"\1[redacted]", value)
    value = _CREDENTIAL_ASSIGNMENT.sub(r"\1[redacted]", value)
    value = _URL_USERINFO.sub(r"\1[redacted]@", value)
    value = _JWT.sub("[redacted-jwt]", value)
    return _TELEGRAM_BOT_TOKEN.sub("[redacted-telegram-token]", value)


def _sanitize_telegram_mapping(value: dict[Any, Any], *, depth: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= 100:
            result["_truncated"] = True
            break
        if _is_protected_research_key(key):
            continue
        if _is_sensitive_key(key):
            result[str(key)] = "[redacted]"
            continue
        if _is_diagnostic_key(key):
            result[str(key)] = "[redacted diagnostic]"
            continue
        sanitized = _sanitize_telegram_value(item, depth=depth + 1)
        if sanitized is not _OMIT:
            result[str(key)] = sanitized
    return result


def _sanitize_telegram_list(value: list[Any], *, depth: int) -> list[Any]:
    result = []
    for item in value[:100]:
        sanitized = _sanitize_telegram_value(item, depth=depth + 1)
        if sanitized is not _OMIT:
            result.append(sanitized)
    return result


def _sanitize_telegram_scalar(value: Any) -> Any:
    if isinstance(value, str):
        if _PROTECTED_RESEARCH_TEXT.search(value):
            return _OMIT
        return _redact_secret_patterns(value[:1000])
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    rendered = str(value)[:1000]
    if _PROTECTED_RESEARCH_TEXT.search(rendered):
        return _OMIT
    return _redact_secret_patterns(rendered)


def _sanitize_telegram_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, dict):
        return _sanitize_telegram_mapping(value, depth=depth)
    if isinstance(value, list):
        return _sanitize_telegram_list(value, depth=depth)
    return _sanitize_telegram_scalar(value)


def redact_sensitive(value: Any, *, depth: int = 0) -> Any:
    """Return bounded Telegram-safe data with secrets and protected tests removed."""

    sanitized = _sanitize_telegram_value(value, depth=depth)
    return "[omitted protected research result]" if sanitized is _OMIT else sanitized


def _api_call(
    settings: TelegramSettings,
    method: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
    post: Callable[..., Any] | None = None,
) -> Any:
    if method not in ALLOWED_API_METHODS:
        raise TelegramError(f"Telegram API method is not allowed: {method}")
    if timeout_seconds <= 0 or timeout_seconds > 65:
        raise TelegramError("Telegram API timeout must be between 1 and 65 seconds")
    if post is None:
        import requests

        post = requests.post
    url = f"{TELEGRAM_API_ROOT}/bot{settings.bot_token}/{method}"
    try:
        response = post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        raise TelegramError(f"Telegram {method} request failed: {type(exc).__name__}") from exc
    if not isinstance(body, dict) or body.get("ok") is not True:
        description = body.get("description") if isinstance(body, dict) else None
        safe_description = redact_sensitive(
            str(description or "API returned an invalid response")[:240]
        )
        raise TelegramError(f"Telegram {method} rejected the request: {safe_description}")
    return body.get("result")


def send_text(
    settings: TelegramSettings,
    text: str,
    *,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    rendered = _redact_secret_patterns(str(text).strip())
    if _PROTECTED_RESEARCH_TEXT.search(rendered):
        rendered = "[omitted protected research result]"
    if not rendered:
        raise TelegramError("Telegram message cannot be empty")
    if len(rendered) > TELEGRAM_SAFE_TEXT:
        rendered = rendered[: TELEGRAM_SAFE_TEXT - 16] + "\n… [truncated]"
    result = _api_call(
        settings,
        "sendMessage",
        {
            "chat_id": settings.chat_id,
            "text": rendered,
            "link_preview_options": {"is_disabled": True},
        },
        timeout_seconds=15,
        post=post,
    )
    message_id = result.get("message_id") if isinstance(result, dict) else None
    return {"ok": True, "message_id": message_id}


def _friendly_name(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").strip()


def _formatted_number(value: Any, *, decimals: int = 2) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rendered = f"{number:,.{decimals}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _formatted_percent(value: Any) -> str | None:
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return f"{number:+.2f}%"


def _format_position_alert(title: str, detail: dict[str, Any]) -> str:
    event_type = str(detail.get("event_type") or "").lower()
    opened = event_type == "opened" or title.endswith("opened")
    mode = str(detail.get("configured_mode") or detail.get("execution") or "unknown").upper()
    symbol = str(detail.get("symbol") or "unknown")
    direction = str(detail.get("direction") or "unknown").upper()
    product = _friendly_name(detail.get("product") or detail.get("objective"))
    lines = [
        f"{'Position opened' if opened else 'Position closed'} · {mode}",
        f"{symbol} · {direction} · {product}",
    ]
    entry = _formatted_number(detail.get("entry_price"))
    if opened:
        if entry is not None:
            lines.append(f"Entry: {entry}")
        size = _formatted_percent(detail.get("position_size"))
        if size is not None:
            lines.append(f"Position size: {size.removeprefix('+')} of equity")
        stop = _formatted_number(detail.get("sl_price"))
        target = _formatted_number(detail.get("tp_price"))
        if stop is not None or target is not None:
            lines.append(f"Stop: {stop or 'n/a'} · Target: {target or 'n/a'}")
        strategy = detail.get("strategy_id")
        if strategy:
            lines.append(f"Strategy: {_friendly_name(strategy)}")
    else:
        exit_price = _formatted_number(detail.get("exit_price"))
        if entry is not None or exit_price is not None:
            lines.append(f"Entry: {entry or 'n/a'} → Exit: {exit_price or 'n/a'}")
        result = _formatted_percent(detail.get("sized_return"))
        if result is not None:
            lines.append(f"Portfolio result: {result}")
        reason = detail.get("exit_reason")
        if reason:
            lines.append(f"Reason: {_friendly_name(reason)}")
        equity = _formatted_number(detail.get("equity_after"))
        if equity is not None:
            lines.append(f"Equity after close: {equity}")
    lines.append("No action required.")
    return "\n".join(lines)


def _format_health_alert(detail: dict[str, Any]) -> str:
    issues = detail.get("issues")
    issues = issues if isinstance(issues, list) else []
    lines = ["System issue detected"]
    rendered_issue = False
    only_research_jobs = True
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or "unknown")
        issue_detail = issue.get("detail")
        issue_detail = issue_detail if isinstance(issue_detail, dict) else {}
        if code == "scheduled_job_failed":
            jobs = issue_detail.get("jobs")
            names = (
                [
                    _friendly_name(job.get("name"))
                    for job in jobs
                    if isinstance(job, dict) and job.get("name")
                ]
                if isinstance(jobs, list)
                else []
            )
            if names:
                lines.append(f"Research/data jobs failing ({len(names)}):")
                lines.extend(f"• {name}" for name in names)
            else:
                lines.append("One or more research/data jobs are failing.")
            rendered_issue = True
            continue
        only_research_jobs = False
        message = issue.get("message")
        lines.append(f"• {_friendly_name(message or code)}")
        rendered_issue = True
    if not rendered_issue:
        lines.append("The healthcheck found a blocking operational problem.")
        only_research_jobs = False
    if only_research_jobs:
        lines.append("Position supervision is still running.")
    lines.append("You will receive one recovery message when this clears.")
    return "\n".join(lines)


def _append_readable_detail(
    lines: list[str],
    value: Any,
    *,
    label: str,
    depth: int = 0,
) -> None:
    if len(lines) >= 14 or depth > 2:
        return
    if isinstance(value, bool):
        lines.append(f"{label}: {'yes' if value else 'no'}")
    elif isinstance(value, str | int | float) or value is None:
        lines.append(f"{label}: {'none' if value is None else value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            child = _friendly_name(key).capitalize()
            _append_readable_detail(lines, item, label=child, depth=depth + 1)
    elif isinstance(value, list):
        scalar_items = [str(item) for item in value if isinstance(item, str | int | float)]
        if scalar_items:
            lines.append(f"{label}: {', '.join(scalar_items[:5])}")
        for item in value:
            if isinstance(item, dict):
                _append_readable_detail(lines, item, label=label, depth=depth + 1)


def _format_generic_alert(safe: dict[str, Any]) -> str:
    severity = str(safe.get("severity") or "info").upper()
    title = str(safe.get("title") or "Autopilot notification")[:200]
    detail = safe.get("detail") if isinstance(safe.get("detail"), dict) else {}
    lines = [f"{severity.title()} · {title}"]
    for key, value in detail.items():
        if key in {"autonomous", "event_id", "operator_action_required", "schema"}:
            continue
        _append_readable_detail(lines, value, label=_friendly_name(key).capitalize())
    return "\n".join(lines)


def format_alert_message(payload: dict[str, Any]) -> str:
    safe = redact_sensitive(payload)
    title = str(safe.get("title") or "Autopilot notification")[:200]
    detail = safe.get("detail") if isinstance(safe.get("detail"), dict) else {}
    if title in {"autonomous position opened", "autonomous position closed"}:
        return _format_position_alert(title, detail)
    if title == "autopilot healthcheck failed":
        return _format_health_alert(detail)
    if title == "autopilot healthcheck recovered":
        cleared = detail.get("cleared_issue_codes")
        cleared = cleared if isinstance(cleared, list) else []
        lines = ["System recovered"]
        if cleared:
            lines.append("Cleared: " + ", ".join(_friendly_name(item) for item in cleared))
        lines.append("No action required.")
        return "\n".join(lines)
    if title == "autopilot cycle failed":
        lines = ["Trading supervision cycle failed"]
        products = detail.get("products")
        product_names = (
            [
                _friendly_name(item.get("name"))
                for item in products
                if isinstance(item, dict) and item.get("name")
            ]
            if isinstance(products, list)
            else []
        )
        jobs = detail.get("jobs")
        job_names = (
            [
                _friendly_name(item.get("name"))
                for item in jobs
                if isinstance(item, dict) and item.get("name")
            ]
            if isinstance(jobs, list)
            else []
        )
        if product_names:
            lines.append("Affected products: " + ", ".join(product_names))
        if job_names:
            lines.append("Failed jobs: " + ", ".join(job_names))
        if detail.get("control"):
            lines.append("Operator control state is invalid.")
        lines.append("Check the operator report. This will not repeat unless the incident changes.")
        return "\n".join(lines)
    if title == "autopilot cycle recovered":
        return "Trading supervision recovered\nNo action required."

    return _format_generic_alert(safe)


def _read_last_json_line(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[-1]) if lines else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def send_alert_from_environment(
    payload: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Send an alert when Telegram is configured; otherwise remain a no-op."""

    settings = TelegramSettings.from_environment(environ)
    if settings is None:
        return None
    return send_text(settings, format_alert_message(payload), post=post)


def _read_json_object(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.exists() or not path.is_file():
        return {}
    try:
        if path.stat().st_size > max_bytes:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_scalar_tree(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    sanitized = _sanitize_telegram_value(value, depth=depth)
    return None if sanitized is _OMIT else sanitized


def build_status_snapshot(
    *,
    status_path: Path,
    control_path: Path,
    job_worker_status_path: Path = DEFAULT_JOB_WORKER_STATUS,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    market_universe_path: Path = DEFAULT_MARKET_UNIVERSE,
    openclaw_review_audit_path: Path = DEFAULT_OPENCLAW_REVIEW_AUDIT,
    openclaw_ingest_status_path: Path = DEFAULT_OPENCLAW_INGEST_STATUS,
) -> dict[str, Any]:
    status = _read_json_object(status_path)
    worker = _read_json_object(job_worker_status_path)
    research = _read_json_object(research_cycle_path)
    universe = _read_json_object(market_universe_path)
    openclaw_review = _read_last_json_line(openclaw_review_audit_path)
    openclaw_ingest = _read_json_object(openclaw_ingest_status_path)
    control = load_control(control_path)

    products: list[dict[str, Any]] = []
    for raw in status.get("products") or []:
        if not isinstance(raw, dict):
            continue
        product = raw.get("product") if isinstance(raw.get("product"), dict) else raw
        products.append(
            {
                "name": product.get("name"),
                "objective": product.get("objective"),
                "market": product.get("market"),
                "mode": product.get("execution_mode") or product.get("mode"),
                "ok": raw.get("ok"),
                "action": _safe_scalar_tree(raw.get("action")),
                "skipped": bool(raw.get("skipped", False)),
                "reason": _safe_scalar_tree(raw.get("reason")),
                "paused": bool(raw.get("paused", False)),
                "open_positions": raw.get("open_positions"),
                "drawdown_halted": bool(raw.get("drawdown_halted", False)),
            }
        )

    jobs: list[dict[str, Any]] = []
    for raw in worker.get("jobs") or []:
        if not isinstance(raw, dict):
            continue
        jobs.append(
            {
                "name": raw.get("name"),
                "ok": raw.get("ok"),
                "skipped": bool(raw.get("skipped", False)),
                "reason": _safe_scalar_tree(raw.get("reason")),
                "returncode": raw.get("returncode"),
            }
        )
    summary = research.get("summary") if isinstance(research.get("summary"), dict) else {}
    safe_summary = {
        key: _safe_scalar_tree(summary.get(key))
        for key in SAFE_RESEARCH_SUMMARY_KEYS
        if key in summary
    }
    return {
        "schema": "autopilot.telegram_status/v1",
        "generated_at": utc_now(),
        "supervisor": {
            "ok": status.get("ok"),
            "generated_at": status.get("generated_at"),
        },
        "control": {key: _safe_scalar_tree(control.get(key)) for key in SAFE_CONTROL_KEYS},
        "products": products,
        "job_worker": {
            "ok": worker.get("ok"),
            "generated_at": worker.get("generated_at"),
            "skipped": bool(worker.get("skipped", False)),
            "reason": _safe_scalar_tree(worker.get("reason")),
            "jobs": jobs,
        },
        "research": {
            "ok": research.get("ok"),
            "generated_at": research.get("generated_at"),
            "skipped": bool(research.get("skipped", False)),
            "reason": _safe_scalar_tree(research.get("reason")),
            "summary": safe_summary,
        },
        "universe": {
            "ok": universe.get("ok"),
            "generated_at": universe.get("generated_at"),
            "research_symbols": _safe_scalar_tree(universe.get("research_symbols")) or [],
            "eligible_research_symbols": (
                _safe_scalar_tree(universe.get("eligible_research_symbols")) or []
            ),
        },
        "openclaw_review": {
            "recorded_at": openclaw_review.get("recorded_at"),
            "proposal_count": openclaw_review.get("proposal_count"),
            "summary": _safe_scalar_tree(openclaw_review.get("summary")),
        },
        "openclaw_ingest": {
            "ok": openclaw_ingest.get("ok"),
            "degraded": bool(openclaw_ingest.get("degraded", False)),
            "degraded_reasons": _safe_scalar_tree(openclaw_ingest.get("degraded_reasons")) or [],
            "generated_at": openclaw_ingest.get("generated_at"),
            "accepted": len(openclaw_ingest.get("accepted") or []),
            "rejected": len(openclaw_ingest.get("rejected") or []),
            "remaining": openclaw_ingest.get("remaining"),
        },
    }


def _status_attention_needed(snapshot: dict[str, Any]) -> bool:
    control = snapshot.get("control") or {}
    research = snapshot.get("research") or {}
    summary = research.get("summary") or {}
    blocked = any(
        int(summary.get(key) or 0) > 0
        for key in ("coverage_failures", "scenario_errors", "unsupported_hypotheses")
    )
    return bool(
        control.get("paused")
        or control.get("pause_jobs")
        or (snapshot.get("supervisor") or {}).get("ok") is False
        or (snapshot.get("job_worker") or {}).get("ok") is False
        or research.get("ok") is False
        or (snapshot.get("universe") or {}).get("ok") is False
        or (snapshot.get("openclaw_ingest") or {}).get("ok") is False
        or (snapshot.get("openclaw_ingest") or {}).get("degraded")
        or blocked
    )


def _status_research_lines(
    snapshot: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    lines = ["", "Research"]
    generation = summary.get("generative_search") or {}
    if generation:
        lines.append(
            f"- Selected {generation.get('batch_hypotheses') or 0}: "
            f"{generation.get('new_hypotheses') or 0} new, "
            f"{generation.get('resumed_pending') or 0} resumed, "
            f"{generation.get('revalidation_pending') or 0} revalidations."
        )
    verdict_text = ", ".join(
        f"{count} {str(verdict).replace('_', ' ')}"
        for verdict, count in sorted((summary.get("verdicts") or {}).items())
        if count
    )
    lines.append(
        f"- Evaluated {summary.get('hypotheses', 0)} hypotheses: "
        f"{verdict_text or 'no completed verdicts'}; "
        f"{summary.get('keepers', 0)} keepers, {summary.get('staged', 0)} staged."
    )
    reasons = ", ".join(
        f"{str(reason).replace('_', ' ')} ({count})"
        for reason, count in list((summary.get("top_reasons") or {}).items())[:4]
        if count
    )
    if reasons:
        lines.append(f"- Main outcomes: {reasons}.")
    universe = snapshot.get("universe") or {}
    configured = universe.get("research_symbols") or []
    if configured:
        lines.append(
            f"- Futures universe: {', '.join(configured)}. "
            f"Currently eligible: {', '.join(universe.get('eligible_research_symbols') or []) or 'none'}."
        )
    return lines


def _status_bridge_lines(
    snapshot: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    lines: list[str] = []
    review = snapshot.get("openclaw_review") or {}
    ingest = snapshot.get("openclaw_ingest") or {}
    generation = summary.get("generative_search") or {}
    if review.get("recorded_at"):
        lines.append(
            f"- OpenClaw proposed {review.get('proposal_count') or 0} ideas in its latest review."
        )
    if ingest.get("generated_at"):
        lines.append(
            f"- Bridge pass: {ingest.get('accepted') or 0} accepted, "
            f"{ingest.get('rejected') or 0} rejected, "
            f"{ingest.get('remaining') or 0} awaiting ingestion."
        )
        if ingest.get("ok") is False or ingest.get("degraded"):
            reasons = ", ".join(
                str(reason).replace("_", " ") for reason in ingest.get("degraded_reasons") or []
            )
            lines.append(f"- Bridge needs attention: {reasons or 'operational failure'}.")
    if generation:
        lines.append(
            f"- Strategy factory consumed "
            f"{generation.get('openclaw_proposals_seen') or 0} OpenClaw proposals."
        )
    return lines


def format_status_message(snapshot: dict[str, Any]) -> str:
    supervisor = snapshot.get("supervisor") or {}
    control = snapshot.get("control") or {}
    worker = snapshot.get("job_worker") or {}
    research = snapshot.get("research") or {}
    summary = research.get("summary") or {}
    paused = bool(control.get("paused") or control.get("pause_jobs"))
    attention = _status_attention_needed(snapshot)
    overall = "Attention needed" if attention else "Healthy"
    lines = [
        "Trading research update",
        f"Overall: {overall}.",
    ]
    if paused:
        lines.append(f"Automation is paused ({control.get('reason') or 'no reason recorded'}).")
    lines.extend(["", "Products"])
    lines.extend(
        f"- {str(product.get('name') or 'unknown').replace('_', ' ').title()}: "
        f"{_status_word(product.get('ok'))}; {product.get('mode') or 'unknown'} mode; "
        f"{product.get('open_positions') or 0} open positions."
        for product in snapshot.get("products") or []
    )
    lines.extend(_status_research_lines(snapshot, summary))
    lines.extend(_status_bridge_lines(snapshot, summary))
    lines.extend(
        [
            "",
            f"Operations: supervisor is {_status_word(supervisor.get('ok'))}; "
            f"job worker is {_status_word(worker.get('ok'))}. "
            f"Last supervisor heartbeat: {supervisor.get('generated_at') or 'unknown'}.",
        ]
    )
    return "\n".join(lines)


def _status_word(value: Any) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "failed"
    return "unknown"


def _message_from_update(update: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("message", "channel_post"):
        message = update.get(key)
        if isinstance(message, dict):
            return message
    return None


def _parse_command(text: str) -> tuple[str, list[str]] | None:
    parts = text.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None
    command = parts[0].split("@", 1)[0].lower()
    return command, parts[1:]


def _handle_pause_command(
    command: str,
    arguments: list[str],
    *,
    update: dict[str, Any],
    sender_id: int,
    settings: TelegramSettings,
    control_path: Path,
    control_audit_path: Path,
    product_names: set[str],
) -> dict[str, Any]:
    pause_commands = {"/pause", "/pause_jobs", "/pause_product"}
    refused = {
        "handled": True,
        "command": command,
        "refused": True,
    }
    if command not in pause_commands:
        return {
            **refused,
            "reply": "Command refused. This channel supports status and pause-only controls.",
        }
    if not settings.pause_commands_enabled:
        return {
            **refused,
            "reply": "Pause commands are disabled; use /status or the local control CLI.",
        }
    if not isinstance(sender_id, int) or sender_id not in settings.allowed_user_ids:
        return {
            **refused,
            "reply": "Pause command refused for this Telegram user.",
        }
    actor = f"telegram:{sender_id}"
    reason = f"pause requested through Telegram update {update.get('update_id', 'unknown')}"
    if command == "/pause" and not arguments:
        control = update_control(
            control_path, "pause", reason=reason, audit_path=control_audit_path, actor=actor
        )
        reply = "Global pause requested. Resuming requires the local control CLI."
    elif command == "/pause_jobs" and not arguments:
        control = update_control(
            control_path,
            "pause-jobs",
            reason=reason,
            audit_path=control_audit_path,
            actor=actor,
        )
        reply = "Scheduled-job pause requested. Resuming requires the local control CLI."
    elif command == "/pause_product" and len(arguments) == 1:
        product_name = arguments[0]
        if product_name not in product_names:
            allowed = ", ".join(sorted(product_names)) or "none"
            return {
                **refused,
                "reply": f"Unknown product. Configured products: {allowed}.",
            }
        control = update_control(
            control_path,
            "pause-product",
            name=product_name,
            reason=reason,
            audit_path=control_audit_path,
            actor=actor,
        )
        reply = f"Product {product_name} pause requested. Resuming requires the local control CLI."
    else:
        return {
            **refused,
            "reply": "Invalid pause command syntax. Use /help.",
        }
    return {
        "handled": True,
        "command": command,
        "reply": reply,
        "control": {key: control.get(key) for key in SAFE_CONTROL_KEYS},
    }


def handle_update(
    update: dict[str, Any],
    *,
    settings: TelegramSettings,
    status_path: Path,
    control_path: Path,
    control_audit_path: Path,
    product_names: set[str],
    job_worker_status_path: Path = DEFAULT_JOB_WORKER_STATUS,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
) -> dict[str, Any]:
    """Process one update without performing network I/O."""

    message = _message_from_update(update)
    if message is None:
        return {"handled": False, "reason": "unsupported_update"}
    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    if str(chat.get("id")) != settings.chat_id:
        return {"handled": False, "reason": "unauthorized_chat"}
    parsed = _parse_command(str(message.get("text") or ""))
    if parsed is None:
        return {"handled": False, "reason": "not_a_command"}
    command, arguments = parsed
    sender = message.get("from") if isinstance(message.get("from"), dict) else {}
    sender_id = sender.get("id")

    if command == "/status" and not arguments:
        snapshot = build_status_snapshot(
            status_path=status_path,
            control_path=control_path,
            job_worker_status_path=job_worker_status_path,
            research_cycle_path=research_cycle_path,
        )
        return {"handled": True, "command": command, "reply": format_status_message(snapshot)}
    if command == "/help" and not arguments:
        return {"handled": True, "command": command, "reply": _help_text(settings)}

    return _handle_pause_command(
        command,
        arguments,
        update=update,
        sender_id=sender_id,
        settings=settings,
        control_path=control_path,
        control_audit_path=control_audit_path,
        product_names=product_names,
    )


def _help_text(settings: TelegramSettings) -> str:
    lines = ["Allowed commands:", "/status — sanitized runtime/research status"]
    if settings.pause_commands_enabled:
        lines.extend(
            [
                "/pause — pause all new work",
                "/pause_jobs — pause scheduled research/data jobs",
                "/pause_product NAME — pause one product",
            ]
        )
    lines.append(
        "Approval, activation, resume, risk, flatten, and order commands are never available."
    )
    return "\n".join(lines)


def _load_poll_offset(path: Path) -> int:
    source = path
    if (
        not source.exists()
        and source.resolve(strict=False) == DEFAULT_POLL_STATE.resolve(strict=False)
        and LEGACY_POLL_STATE.exists()
    ):
        source = LEGACY_POLL_STATE
    payload = _read_json_object(source, max_bytes=64 * 1024)
    raw = payload.get("next_update_id", 0)
    return raw if isinstance(raw, int) and raw >= 0 else 0


def poll_once(
    *,
    settings: TelegramSettings,
    status_path: Path,
    control_path: Path,
    control_audit_path: Path,
    poll_state_path: Path,
    product_names: set[str],
    job_worker_status_path: Path = DEFAULT_JOB_WORKER_STATUS,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    long_poll_seconds: int = 30,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if long_poll_seconds < 0 or long_poll_seconds > 50:
        raise TelegramError("long poll seconds must be between 0 and 50")
    next_update_id = _load_poll_offset(poll_state_path)
    result = _api_call(
        settings,
        "getUpdates",
        {
            "offset": next_update_id,
            "limit": 50,
            "timeout": long_poll_seconds,
            "allowed_updates": ALLOWED_UPDATE_TYPES,
        },
        timeout_seconds=max(10, long_poll_seconds + 5),
        post=post,
    )
    updates = result if isinstance(result, list) else []
    handled = 0
    refused = 0
    errors: list[dict[str, Any]] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        update_id = update.get("update_id")
        try:
            outcome = handle_update(
                update,
                settings=settings,
                status_path=status_path,
                control_path=control_path,
                control_audit_path=control_audit_path,
                product_names=product_names,
                job_worker_status_path=job_worker_status_path,
                research_cycle_path=research_cycle_path,
            )
            if outcome.get("handled"):
                handled += 1
                refused += int(bool(outcome.get("refused")))
                send_text(settings, str(outcome.get("reply") or "Command processed."), post=post)
        except Exception as exc:
            errors.append(
                {
                    "update_id": update_id,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            )
            # Do not acknowledge a failed pause/status update. A retried pause
            # is idempotent and safer than silently losing the operator intent.
            break
        if isinstance(update_id, int) and update_id >= next_update_id:
            next_update_id = update_id + 1
    write_json_atomic(
        poll_state_path,
        {
            "schema": "autopilot.telegram_poll_state/v1",
            "updated_at": utc_now(),
            "next_update_id": next_update_id,
        },
    )
    return {
        "ok": not errors,
        "updates": len(updates),
        "handled": handled,
        "refused": refused,
        "errors": errors,
        "next_update_id": next_update_id,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the restricted Telegram operator edge.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_FILE)
    parser.add_argument(
        "--validate-settings",
        action="store_true",
        help="Strictly validate the private Telegram-only settings file and exit.",
    )
    parser.add_argument(
        "--validate-service-paths",
        action="store_true",
        help="Validate the installer's dedicated writable path boundary and exit.",
    )
    parser.add_argument("--expected-control-file", type=Path)
    parser.add_argument("--expected-control-audit", type=Path)
    parser.add_argument("--poll-state", type=Path, default=DEFAULT_POLL_STATE)
    parser.add_argument("--job-worker-status", type=Path, default=DEFAULT_JOB_WORKER_STATUS)
    parser.add_argument("--research-cycle", type=Path, default=DEFAULT_RESEARCH_CYCLE)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--status", action="store_true", help="Print sanitized status without Telegram."
    )
    parser.add_argument(
        "--send-status", action="store_true", help="Send one sanitized status message."
    )
    parser.add_argument("--long-poll-seconds", type=int, default=30)
    parser.add_argument("--retry-seconds", type=int, default=10)
    return parser.parse_args(argv)


def _validate_service_paths(
    config: Any,
    *,
    expected_control_file: Path | None,
    expected_control_audit: Path | None,
    poll_state: Path,
) -> None:
    expected = {
        "control_file": expected_control_file,
        "control_audit_file": expected_control_audit,
        "poll_state": poll_state,
    }
    actual = {
        "control_file": config.control_file,
        "control_audit_file": config.control_audit_file,
        "poll_state": poll_state,
    }
    missing = [name for name, path in expected.items() if path is None]
    mismatched = [
        name
        for name, expected_path in expected.items()
        if expected_path is not None
        and actual[name].resolve(strict=False) != expected_path.resolve(strict=False)
    ]
    if missing or mismatched:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Telegram service writable paths do not match the dedicated boundary",
                    "missing": missing,
                    "mismatched": mismatched,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1)
    unsafe_paths = [
        name
        for name, path in actual.items()
        if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir()
    ]
    if unsafe_paths:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Telegram service writable paths require safe existing parent directories",
                    "unsafe": unsafe_paths,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1)
    writable_parents = {path.parent.resolve(strict=False) for path in actual.values()}
    runtime_root = (PROJECT_ROOT / "runtime").resolve(strict=False)
    if runtime_root in writable_parents:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Telegram service paths must not require the runtime root to be writable",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {"ok": True, "writable_directories": sorted(map(str, writable_parents))},
            sort_keys=True,
        )
    )


def _run_telegram_loop(
    settings: TelegramSettings,
    config: Any,
    args: argparse.Namespace,
    product_names: set[str],
) -> None:
    while True:
        try:
            report = poll_once(
                settings=settings,
                status_path=config.status_file,
                control_path=config.control_file,
                control_audit_path=config.control_audit_file,
                poll_state_path=args.poll_state,
                product_names=product_names,
                job_worker_status_path=args.job_worker_status,
                research_cycle_path=args.research_cycle,
                long_poll_seconds=args.long_poll_seconds,
            )
            print(json.dumps(report, sort_keys=True))
        except TelegramError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            if args.once:
                raise SystemExit(str(exc)) from exc
            time.sleep(args.retry_seconds)
            continue
        if args.once:
            return


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_settings:
        try:
            result = validate_settings_file(args.settings_file)
        except TelegramError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            raise SystemExit("Telegram settings validation failed") from exc
        print(json.dumps(result, sort_keys=True))
        return
    config = load_config(args.config, strict_jobs=False)
    if args.validate_service_paths:
        _validate_service_paths(
            config,
            expected_control_file=args.expected_control_file,
            expected_control_audit=args.expected_control_audit,
            poll_state=args.poll_state,
        )
        return
    snapshot = build_status_snapshot(
        status_path=config.status_file,
        control_path=config.control_file,
        job_worker_status_path=args.job_worker_status,
        research_cycle_path=args.research_cycle,
    )
    if args.status:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    settings_environment = dict(os.environ)
    settings_environment["AUTOPILOT_TELEGRAM_SETTINGS_FILE"] = str(args.settings_file)
    settings = TelegramSettings.from_environment(
        _environment_with_private_settings(settings_environment), required=True
    )
    if settings is None:
        raise TelegramError("required Telegram settings unexpectedly resolved to empty")
    if args.send_status:
        print(json.dumps(send_text(settings, format_status_message(snapshot)), sort_keys=True))
        return
    if args.retry_seconds <= 0:
        raise SystemExit("retry seconds must be positive")
    product_names = {product.name for product in config.products}
    _run_telegram_loop(settings, config, args, product_names)


if __name__ == "__main__":
    main()
