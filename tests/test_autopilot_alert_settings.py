from pathlib import Path

import pytest

from src.autopilot.alert_settings import (
    AlertSettingsError,
    alert_environment,
    load_alert_settings_file,
    validate_alert_settings_file,
)


def _settings(path: Path, text: str, *, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_alert_settings_load_only_operations_routing_keys(tmp_path):
    path = _settings(
        tmp_path / "alerts.env",
        "AUTOPILOT_WEBHOOK_URL=https://alerts.example/hook\n"
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE=runtime/telegram.env\n",
    )

    assert load_alert_settings_file(path) == {
        "AUTOPILOT_WEBHOOK_URL": "https://alerts.example/hook",
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE": "runtime/telegram.env",
    }
    assert validate_alert_settings_file(path)["configured_keys"] == [
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE",
        "AUTOPILOT_WEBHOOK_URL",
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("TRADING_LIVE=1\n", "forbidden key 'TRADING_LIVE'"),
        ("EXCHANGE_API_KEY=secret\n", "forbidden key 'EXCHANGE_API_KEY'"),
        (
            "AUTOPILOT_WEBHOOK_URL=a\nAUTOPILOT_WEBHOOK_URL=b\n",
            "duplicates AUTOPILOT_WEBHOOK_URL",
        ),
        ("not-an-assignment\n", "must be a KEY=value assignment"),
    ],
)
def test_alert_settings_reject_forbidden_or_ambiguous_assignments(
    tmp_path,
    text,
    message,
):
    path = _settings(tmp_path / "alerts.env", text)

    with pytest.raises(AlertSettingsError, match=message):
        load_alert_settings_file(path)


def test_alert_settings_require_owner_only_regular_file(tmp_path):
    insecure = _settings(tmp_path / "insecure.env", "AUTOPILOT_WEBHOOK_URL=\n", mode=0o640)
    target = _settings(tmp_path / "target.env", "AUTOPILOT_WEBHOOK_URL=\n")
    linked = tmp_path / "linked.env"
    linked.symlink_to(target)

    with pytest.raises(AlertSettingsError, match="mode 0600"):
        load_alert_settings_file(insecure)
    with pytest.raises(AlertSettingsError, match="must not be a symlink"):
        load_alert_settings_file(linked)


def test_alert_environment_filters_process_secrets_and_overlays_private_file(tmp_path):
    path = _settings(
        tmp_path / "alerts.env",
        "AUTOPILOT_WEBHOOK_URL=https://file.example/hook\n"
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE=runtime/telegram.env\n",
    )

    values = alert_environment(
        {
            "AUTOPILOT_ALERT_SETTINGS_FILE": str(path),
            "AUTOPILOT_WEBHOOK_URL": "https://manager.example/hook",
            "AUTOPILOT_TELEGRAM_BOT_TOKEN": "legacy-direct-token",
            "EXCHANGE_API_KEY": "must-not-cross-boundary",
            "TRADING_LIVE": "1",
        }
    )

    assert values == {
        "AUTOPILOT_WEBHOOK_URL": "https://file.example/hook",
        "AUTOPILOT_TELEGRAM_BOT_TOKEN": "legacy-direct-token",
        "AUTOPILOT_TELEGRAM_SETTINGS_FILE": "runtime/telegram.env",
    }


def test_missing_alert_settings_file_is_optional(tmp_path):
    path = tmp_path / "missing.env"

    assert load_alert_settings_file(path) == {}
    assert validate_alert_settings_file(path)["exists"] is False
