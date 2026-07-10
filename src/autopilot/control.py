"""Tiny file-based control channel.

This keeps the Linux server dependency-free. Operators can pause the whole
system, one product, or selected scheduled jobs by editing ``runtime/control.json``
or replacing it with the examples documented in README.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.autopilot.config import (
    DEFAULT_CONTROL_AUDIT_FILE,
    DEFAULT_CONTROL_FILE,
    AutopilotConfig,
    load_config,
)
from src.autopilot.io import append_json_line, write_json_atomic
from src.autopilot.reporting import utc_now

DEFAULT_CONTROL = {
    "paused": False,
    "pause_jobs": False,
    "paused_products": [],
    "paused_jobs": [],
    "flatten_products": [],
    "flatten_all": False,
    "reason": "",
}

LOGGER = logging.getLogger("autopilot.control")
TRUE_STRINGS = {"1", "true", "yes", "on"}
FALSE_STRINGS = {"0", "false", "no", "off"}


class ControlConflictError(RuntimeError):
    """Raised when an automatic mutation is based on a stale control snapshot."""


@contextmanager
def control_update_lock(path: Path):
    """Serialize control read-modify-write transactions across processes.

    The lock is deliberately a sibling file, rather than the atomically replaced
    control inode.  Locking the control file itself would stop protecting later
    readers after ``os.replace`` installs a new inode.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    if lock_path.is_symlink():
        raise ValueError(f"control lock file must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open control lock {lock_path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"control lock must be a regular file: {lock_path}")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fail_closed(path: Path, error: str) -> dict[str, Any]:
    control = dict(DEFAULT_CONTROL)
    control.update(
        paused=True,
        pause_jobs=True,
        reason="invalid_control_file",
        control_error=f"{path}: {error}",
    )
    return control


def _symlink_error(path: Path) -> str:
    return f"control file must not be a symlink: {path}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_STRINGS:
            return True
        if normalized in FALSE_STRINGS:
            return False
        raise ValueError(f"invalid boolean string: {value!r}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"invalid boolean number: {value!r}")
    if value is None:
        raise ValueError("invalid boolean null")
    raise ValueError(f"invalid boolean type: {type(value).__name__}")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _selector_list_error(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return None
    if not isinstance(value, list):
        return f"{key} must be a string or list of strings"
    invalid = [
        {"index": index, "type": type(item).__name__}
        for index, item in enumerate(value)
        if not isinstance(item, str) or not item.strip()
    ]
    if invalid:
        return f"{key} must contain only non-empty strings: {invalid}"
    return None


def _selector_payload_error(payload: dict[str, Any]) -> str | None:
    for key in ("paused_products", "paused_jobs", "flatten_products"):
        if key not in payload:
            continue
        error = _selector_list_error(key, payload.get(key))
        if error is not None:
            return error
    return None


def _boolean_payload_error(payload: dict[str, Any]) -> str | None:
    for key in ("paused", "pause_jobs", "flatten_all"):
        if key not in payload:
            continue
        try:
            _as_bool(payload.get(key))
        except ValueError as exc:
            return f"{key} must be a boolean, 0/1, or true/false string: {exc}"
    return None


def _control_payload_error(payload: dict[str, Any]) -> str | None:
    return _selector_payload_error(payload) or _boolean_payload_error(payload)


def _normalize_control(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_CONTROL)
    merged.update(payload)
    merged["paused"] = _as_bool(merged.get("paused"))
    merged["pause_jobs"] = _as_bool(merged.get("pause_jobs"))
    merged["flatten_all"] = _as_bool(merged.get("flatten_all"))
    merged["paused_products"] = _as_string_list(merged.get("paused_products"))
    merged["paused_jobs"] = _as_string_list(merged.get("paused_jobs"))
    merged["flatten_products"] = _as_string_list(merged.get("flatten_products"))
    merged["reason"] = str(merged.get("reason") or "")
    return merged


def load_control(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return _fail_closed(path, _symlink_error(path))
    if not path.exists():
        return dict(DEFAULT_CONTROL)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail_closed(path, f"{type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        return _fail_closed(path, "control payload must be a JSON object")
    control_error = _control_payload_error(payload)
    if control_error is not None:
        return _fail_closed(path, control_error)
    return _normalize_control(payload)


def _load_editable_control(path: Path) -> tuple[dict[str, Any], str | None]:
    """Load existing operator intent for mutation.

    Unlike ``load_control`` this recovers from malformed JSON by starting from
    defaults, so the CLI can repair a broken control file with ``clear`` or any
    explicit command.
    """
    if path.is_symlink():
        return dict(DEFAULT_CONTROL), _symlink_error(path)
    if not path.exists():
        return dict(DEFAULT_CONTROL), None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return dict(DEFAULT_CONTROL), f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return dict(DEFAULT_CONTROL), f"TypeError: control payload must be a JSON object, got {type(payload).__name__}"
    control_error = _control_payload_error(payload)
    if control_error is not None:
        return dict(DEFAULT_CONTROL), control_error
    return _normalize_control(payload), None


def write_control(path: Path, control: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(_symlink_error(path))
    payload = _normalize_control(control)
    payload.pop("control_error", None)
    write_json_atomic(path, payload)
    return payload


def _set_reason(control: dict[str, Any], reason: str | None) -> None:
    if reason is not None:
        control["reason"] = reason


def _append_unique(control: dict[str, Any], key: str, value: str) -> None:
    value = value.strip()
    if not value:
        raise ValueError(f"{key} requires a non-empty name")
    values = _as_string_list(control.get(key))
    if value not in values:
        values.append(value)
    control[key] = values


def _remove_value(control: dict[str, Any], key: str, value: str) -> None:
    control[key] = [item for item in _as_string_list(control.get(key)) if item != value]


def _default_operator() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def _audit_event(
    audit_path: Path | None,
    *,
    path: Path,
    command: str,
    name: str | None,
    reason: str | None,
    actor: str | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if audit_path is None:
        return
    append_json_line(
        audit_path,
        {
            "at": utc_now(),
            "actor": actor or _default_operator(),
            "command": command,
            "control_file": str(path),
            "name": name,
            "reason": reason or "",
            "before": before,
            "after": after,
        },
    )


def _control_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_control(control)
    return {key: normalized[key] for key in DEFAULT_CONTROL}


def _update_control_locked(
    path: Path,
    command: str,
    *,
    name: str | None = None,
    reason: str | None = None,
    audit_path: Path | None = None,
    actor: str | None = None,
    enforce_flatten_pause: bool = False,
    expected_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control, recovered_control_error = _load_editable_control(path)
    if expected_control is not None:
        if recovered_control_error is not None:
            raise ControlConflictError(
                "control changed or became invalid after the runtime snapshot; "
                f"refusing stale {command!r} mutation: {recovered_control_error}"
            )
        if _control_snapshot(control) != _control_snapshot(expected_control):
            raise ControlConflictError(
                "control changed after the runtime snapshot; "
                f"refusing stale {command!r} mutation"
            )
    before = dict(control)
    if recovered_control_error is not None:
        before["recovered_control_error"] = recovered_control_error
    if command == "clear":
        control = dict(DEFAULT_CONTROL)
        _set_reason(control, reason)
    elif command == "panic":
        control["paused"] = True
        control["pause_jobs"] = True
        control["flatten_all"] = True
        _set_reason(control, reason)
    elif command == "pause":
        control["paused"] = True
        _set_reason(control, reason)
    elif command == "resume":
        control["paused"] = False
        _set_reason(control, reason)
    elif command == "pause-product":
        if not name:
            raise ValueError("pause-product requires a product name")
        _append_unique(control, "paused_products", name)
        _set_reason(control, reason)
    elif command == "resume-product":
        if not name:
            raise ValueError("resume-product requires a product name")
        _remove_value(control, "paused_products", name)
        _set_reason(control, reason)
    elif command == "pause-job":
        if not name:
            raise ValueError("pause-job requires a job name")
        _append_unique(control, "paused_jobs", name)
        _set_reason(control, reason)
    elif command == "resume-job":
        if not name:
            raise ValueError("resume-job requires a job name")
        _remove_value(control, "paused_jobs", name)
        _set_reason(control, reason)
    elif command == "pause-jobs":
        control["pause_jobs"] = True
        _set_reason(control, reason)
    elif command == "resume-jobs":
        control["pause_jobs"] = False
        _set_reason(control, reason)
    elif command == "flatten":
        if not name:
            raise ValueError("flatten requires a product name")
        # Flattening is an emergency/risk-reduction action. Keep the product
        # paused after the one-shot request is auto-cleared so it cannot open a
        # fresh position before the operator reconciles fills and accounting.
        _append_unique(control, "paused_products", name)
        _append_unique(control, "flatten_products", name)
        _set_reason(control, reason)
    elif command == "flatten-all":
        control["paused"] = True
        control["flatten_all"] = True
        _set_reason(control, reason)
    elif command == "clear-flatten":
        if enforce_flatten_pause:
            if name:
                _append_unique(control, "paused_products", name)
            else:
                control["paused"] = True
        if name:
            _remove_value(control, "flatten_products", name)
        else:
            control["flatten_products"] = []
            control["flatten_all"] = False
        _set_reason(control, reason)
    else:
        raise ValueError(f"unknown control command: {command}")
    payload = write_control(path, control)
    if recovered_control_error is not None:
        payload["recovered_control_error"] = recovered_control_error
    try:
        _audit_event(
            audit_path,
            path=path,
            command=command,
            name=name,
            reason=reason,
            actor=actor,
            before=before,
            after=payload,
        )
    except Exception as exc:
        LOGGER.exception("Failed to append control audit event")
        payload["audit_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def update_control(
    path: Path,
    command: str,
    *,
    name: str | None = None,
    reason: str | None = None,
    audit_path: Path | None = None,
    actor: str | None = None,
    enforce_flatten_pause: bool = False,
    expected_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically mutate operator control without losing concurrent intent.

    Interactive commands always apply to the latest state under the lock.
    Runtime auto-clears additionally pass the snapshot they acted on; a newer
    operator command then wins and the stale auto-clear fails closed.
    """

    with control_update_lock(path):
        return _update_control_locked(
            path,
            command,
            name=name,
            reason=reason,
            audit_path=audit_path,
            actor=actor,
            enforce_flatten_pause=enforce_flatten_pause,
            expected_control=expected_control,
        )


def is_product_paused(control: dict[str, Any], product_name: str) -> bool:
    return bool(control.get("paused")) or product_name in set(control.get("paused_products", []))


def is_job_paused(control: dict[str, Any], job_name: str) -> bool:
    return bool(control.get("paused")) or bool(control.get("pause_jobs")) or job_name in set(control.get("paused_jobs", []))


def should_flatten_product(control: dict[str, Any], product_name: str) -> bool:
    return bool(control.get("flatten_all")) or product_name in set(control.get("flatten_products", []))


def unknown_control_selectors(control: dict[str, Any], config: AutopilotConfig) -> dict[str, list[str]]:
    product_names = {product.name for product in config.products}
    job_names = {job.name for job in config.jobs}
    unknown: dict[str, list[str]] = {}
    paused_products = sorted({name for name in control.get("paused_products", []) if name not in product_names})
    flatten_products = sorted({name for name in control.get("flatten_products", []) if name not in product_names})
    paused_jobs = sorted({name for name in control.get("paused_jobs", []) if name not in job_names})
    if paused_products:
        unknown["paused_products"] = paused_products
    if flatten_products:
        unknown["flatten_products"] = flatten_products
    if paused_jobs:
        unknown["paused_jobs"] = paused_jobs
    return unknown


def _validate_selector_against_config(config: AutopilotConfig | None, command: str, name: str | None) -> None:
    if config is None or name is None:
        return
    if command in {"pause-product", "resume-product", "flatten", "clear-flatten"}:
        product_names = {product.name for product in config.products}
        if name not in product_names:
            allowed = ", ".join(sorted(product_names)) or "<none>"
            raise ValueError(f"unknown product {name!r}; configured products: {allowed}")
    if command in {"pause-job", "resume-job"}:
        job_names = {job.name for job in config.jobs}
        if name not in job_names:
            allowed = ", ".join(sorted(job_names)) or "<none>"
            raise ValueError(f"unknown job {name!r}; configured jobs: {allowed}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read or update the autopilot file-based control channel.")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL_FILE, help="Path to control JSON file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional autopilot config used to reject unknown product/job selectors before writing.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Path to append JSONL control audit events. Defaults on for the runtime control file.",
    )
    parser.add_argument("--operator", default=None, help="Operator name stored in audit events.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print the normalized effective control state.")

    def command(name: str, *, needs_name: str | None = None, help_text: str) -> None:
        sub = subparsers.add_parser(name, help=help_text)
        if needs_name:
            sub.add_argument(needs_name)
        sub.add_argument("--reason", default=None, help="Optional operator note stored in the control file.")

    command("clear", help_text="Reset all pause and flatten controls to defaults.")
    command("panic", help_text="Pause everything and request emergency flatten for all live products.")
    command("pause", help_text="Pause all products and scheduled jobs.")
    command("resume", help_text="Resume global product/job supervision.")
    command("pause-product", needs_name="product", help_text="Pause one product.")
    command("resume-product", needs_name="product", help_text="Resume one product.")
    command("pause-job", needs_name="job", help_text="Pause one scheduled job.")
    command("resume-job", needs_name="job", help_text="Resume one scheduled job.")
    command("pause-jobs", help_text="Pause all scheduled jobs while products can continue.")
    command("resume-jobs", help_text="Resume scheduled jobs.")
    command("flatten", needs_name="product", help_text="Request emergency flatten for one live product.")
    command("flatten-all", help_text="Request emergency flatten for all live products.")
    command("clear-flatten", needs_name="product", help_text="Clear one product's flatten request.")
    sub = subparsers.add_parser("clear-all-flatten", help="Clear all flatten requests.")
    sub.add_argument("--reason", default=None, help="Optional operator note stored in the control file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config) if args.config is not None else None
    audit_path = args.audit
    if audit_path is None and args.control == DEFAULT_CONTROL_FILE:
        audit_path = DEFAULT_CONTROL_AUDIT_FILE
    if args.command == "status":
        payload = load_control(args.control)
        if config is not None:
            unknown = unknown_control_selectors(payload, config)
            payload["selector_validation"] = {"ok": not unknown}
            if unknown:
                payload["selector_validation"]["unknown_selectors"] = unknown
    elif args.command == "clear-all-flatten":
        payload = update_control(
            args.control,
            "clear-flatten",
            reason=args.reason,
            audit_path=audit_path,
            actor=args.operator,
        )
    else:
        name = getattr(args, "product", None) or getattr(args, "job", None)
        try:
            _validate_selector_against_config(config, args.command, name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        payload = update_control(
            args.control,
            args.command,
            name=name,
            reason=args.reason,
            audit_path=audit_path,
            actor=args.operator,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
