"""Strict, operations-only environment loading for outbound alerts."""

from __future__ import annotations

import argparse
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.envfile import parse_env_value

ALERT_SETTINGS_FILE_ENV = "AUTOPILOT_ALERT_SETTINGS_FILE"
ALERT_SETTINGS_KEYS = frozenset(
    {
        "AUTOPILOT_WEBHOOK_URL",
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE",
    }
)
ALERT_RUNTIME_KEYS = ALERT_SETTINGS_KEYS | frozenset(
    {
        "AUTOPILOT_TELEGRAM_BOT_TOKEN",
        "AUTOPILOT_TELEGRAM_CHAT_ID",
        "AUTOPILOT_TELEGRAM_PAUSE_COMMANDS",
        "AUTOPILOT_TELEGRAM_ALLOWED_USER_IDS",
    }
)
MAX_ALERT_SETTINGS_BYTES = 64 * 1024


class AlertSettingsError(ValueError):
    """Raised when the operations-only alert settings boundary is invalid."""


def _resolve_settings_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_alert_settings_lines(path: Path) -> list[str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        if path.is_symlink():
            raise AlertSettingsError(f"alert settings must not be a symlink: {path}") from exc
        raise AlertSettingsError(f"cannot open alert settings {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AlertSettingsError(f"alert settings must be a regular file: {path}")
        if metadata.st_uid != os.getuid():
            raise AlertSettingsError(f"alert settings must be owned by the service user: {path}")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o600:
            raise AlertSettingsError(f"alert settings must have mode 0600, got {mode:04o}: {path}")
        if metadata.st_size > MAX_ALERT_SETTINGS_BYTES:
            raise AlertSettingsError(f"alert settings file is unexpectedly large: {path}")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            lines = stream.read(MAX_ALERT_SETTINGS_BYTES + 1).splitlines()
    except (OSError, UnicodeError) as exc:
        raise AlertSettingsError(f"cannot read alert settings {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return lines


def _parse_alert_settings_lines(path: Path, lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AlertSettingsError(
                f"alert settings line {line_number} must be a KEY=value assignment"
            )
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise AlertSettingsError(f"alert settings line {line_number} has malformed key")
        if key not in ALERT_SETTINGS_KEYS:
            raise AlertSettingsError(
                f"alert settings line {line_number} uses forbidden key {key!r}"
            )
        if key in values:
            raise AlertSettingsError(f"alert settings line {line_number} duplicates {key}")
        if "\x00" in raw_value or "\r" in raw_value:
            raise AlertSettingsError(
                f"alert settings line {line_number} contains a control character"
            )
        values[key] = parse_env_value(raw_value)
    return values


def load_alert_settings_file(path: Path) -> dict[str, str]:
    """Load an owner-private file containing only alert-routing assignments."""

    path = _resolve_settings_path(path)
    lines = _read_alert_settings_lines(path)
    return _parse_alert_settings_lines(path, lines)


def alert_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only allowlisted alert/Telegram values from process + private file."""

    source = os.environ if environ is None else environ
    values = {
        key: str(source[key])
        for key in ALERT_RUNTIME_KEYS
        if key in source and str(source[key]).strip()
    }
    configured_path = str(source.get(ALERT_SETTINGS_FILE_ENV, "")).strip()
    if configured_path:
        values.update(load_alert_settings_file(_resolve_settings_path(configured_path)))
    return values


def validate_alert_settings_file(path: Path) -> dict[str, Any]:
    values = load_alert_settings_file(path)
    return {
        "ok": True,
        "settings_file": str(_resolve_settings_path(path)),
        "exists": _resolve_settings_path(path).exists(),
        "configured_keys": sorted(key for key, value in values.items() if value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate operations-only alert settings.")
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_alert_settings_file(args.file)
    except AlertSettingsError as exc:
        raise SystemExit(str(exc)) from exc
    print(report)


if __name__ == "__main__":
    main()
