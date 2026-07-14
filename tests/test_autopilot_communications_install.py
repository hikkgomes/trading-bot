import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_telegram_service_uses_telegram_only_environment_and_pause_edge():
    script = Path("scripts/install_communications_service.sh").read_text(encoding="utf-8")

    assert 'TELEGRAM_ENV="${TELEGRAM_ENV:-$REPO/runtime/telegram.env}"' in script
    assert "EnvironmentFile=" not in script
    assert "--settings-file $TELEGRAM_ENV_UNIT --validate-settings" in script
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT "
        "--settings-file $TELEGRAM_ENV_UNIT" in script
    )
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.telegram_edge --config $CONFIG_UNIT "
        "--settings-file $TELEGRAM_ENV_UNIT --send-status" in script
    )
    assert "OnUnitActiveSec=$REPORT_INTERVAL" in script
    assert "InaccessiblePaths=$TRADING_ENV_UNIT" in script
    assert "InaccessiblePaths=-$APPROVALS_UNIT" in script
    assert "ReadOnlyPaths=$TELEGRAM_ENV_UNIT" in script
    assert "ReadWritePaths=$CONTROL_STATE_DIR_UNIT" in script
    assert "ReadWritePaths=$TELEGRAM_STATE_DIR_UNIT" in script
    assert "ReadWritePaths=$RUNTIME_UNIT" not in script
    assert "NoNewPrivileges=true" in script
    assert "MemoryMax=192M" in script
    assert "WorkingDirectory=$REPO_WORKING_DIRECTORY" in script
    assert 'systemd-analyze --user verify "$@"' in script
    verify_call = script.index(
        'verify_unit_files "$UNIT_FILE" "$REPORT_SERVICE_FILE" "$REPORT_TIMER_FILE"'
    )
    assert verify_call < script.index('if [ "$DRY_RUN" = "1" ]', verify_call)
    assert verify_call < script.index('systemctl --user enable --now "$SERVICE_NAME"')
    assert "prepare_unit_staging()" in script
    assert "publish_unit_files()" in script
    assert verify_call < script.index("publish_unit_files", verify_call)


def test_openclaw_timer_never_launches_openclaw_or_loads_trading_environment():
    script = Path("scripts/install_openclaw_bridge_timer.sh").read_text(encoding="utf-8")

    assert "EnvironmentFile=" not in script
    assert "ExecStart=$PYTHON_UNIT -m src.autopilot.openclaw_bridge export" in script
    assert "ExecStart=$PYTHON_UNIT -m src.autopilot.openclaw_bridge ingest" in script
    assert "ExecStart=openclaw" not in script
    assert " -m openclaw " not in script
    assert "RestrictAddressFamilies=AF_UNIX" in script
    assert "InaccessiblePaths=$TRADING_ENV_UNIT" in script
    assert "InaccessiblePaths=-$APPROVALS_UNIT" in script
    assert "InaccessiblePaths=-$CONTROL_UNIT" in script
    assert "WorkingDirectory=$REPO_WORKING_DIRECTORY" in script
    assert 'systemd-analyze --user verify "$@"' in script
    verify_call = script.index('verify_unit_files "$SERVICE_FILE" "$TIMER_FILE"')
    assert verify_call < script.index('if [ "$DRY_RUN" = "1" ]', verify_call)
    assert verify_call < script.index('systemctl --user enable --now "$TIMER_NAME"')
    assert "prepare_unit_staging()" in script
    assert "publish_unit_files()" in script
    assert verify_call < script.index("publish_unit_files", verify_call)


def test_openclaw_shared_user_mode_uses_narrow_named_user_acls():
    script = Path("scripts/install_openclaw_bridge_timer.sh").read_text(encoding="utf-8")

    assert "deny_immediate_children()" in script
    assert "grant_minimal_parent_traversal()" in script
    assert 'setfacl -m "d:u:$OPENCLAW_USER:---"' in script
    assert 'setfacl -m "u:$OPENCLAW_USER:--x"' in script
    assert 'setfacl -m "u:$OPENCLAW_USER:r-x"' in script
    assert 'setfacl -m "u:$OPENCLAW_USER:rwx"' in script
    assert "foreign_path_is_traversable" in script
    assert 'setfacl -m "g:$OPENCLAW_GROUP:--x" "$traverse_path"' not in script
    assert "setfacl -b" not in script
    assert "chmod go-rwx" not in script
    assert "Shared-user mode requires both OPENCLAW_USER and OPENCLAW_GROUP" in script


def test_openclaw_shared_user_dry_run_generates_group_unit_without_changing_checkout_modes(
    tmp_path,
):
    source_repo = Path.cwd()
    test_repo = tmp_path / "private-checkout"
    test_repo.mkdir()
    sentinel = test_repo / "sensitive.txt"
    sentinel.write_text("private\n", encoding="utf-8")
    sentinel.chmod(0o644)
    unit_dir = tmp_path / "units"

    result = subprocess.run(
        ["bash", str(source_repo / "scripts" / "install_openclaw_bridge_timer.sh")],
        cwd=source_repo,
        env={
            **os.environ,
            "REPO": str(test_repo),
            "PYTHON": str(source_repo / ".venv" / "bin" / "python"),
            "UNIT_DIR": str(unit_dir),
            "DRY_RUN": "1",
            "OPENCLAW_GROUP": "example-bridge",
            "OPENCLAW_USER": "example-openclaw",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "ACL changes were not applied" in result.stdout
    assert sentinel.stat().st_mode & 0o777 == 0o644
    assert not (test_repo / "runtime").exists()
    unit = (unit_dir / "trading-bot-openclaw-bridge.service").read_text(encoding="utf-8")
    assert "Environment=OPENCLAW_SHARED_GROUP=1" in unit


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("setfacl") is None or shutil.which("getfacl") is None,
    reason="requires Linux POSIX ACL tools",
)
def test_openclaw_shared_user_install_applies_narrow_acl_boundary(tmp_path):
    try:
        pwd.getpwnam("nobody")
    except KeyError:
        pytest.skip("requires a local nobody account for named-user ACL verification")

    source_repo = Path.cwd()
    test_repo = tmp_path / "checkout"
    context_dir = test_repo / "runtime" / "openclaw"
    inbox_root = test_repo / "runtime" / "research_inbox" / "openclaw"
    context_dir.mkdir(parents=True)
    for name in ("incoming", "accepted", "rejected", "archive"):
        (inbox_root / name).mkdir(parents=True)
    context = context_dir / "research_context.json"
    context.write_text("{}\n", encoding="utf-8")
    private_data = test_repo / "data"
    private_data.mkdir()
    (private_data / "candles.parquet").write_text("private\n", encoding="utf-8")

    primary_group = subprocess.check_output(["id", "-gn"], text=True).strip()
    current_user = subprocess.check_output(["id", "-un"], text=True).strip()
    real_id = shutil.which("id")
    assert real_id is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "id").write_text(
        f"""#!/bin/sh
if [ "$1" = "-un" ]; then echo {current_user}; exit 0; fi
if [ "$1" = "-nG" ]; then echo {primary_group}; exit 0; fi
if [ "$1" = "nobody" ]; then exit 0; fi
exec {real_id} "$@"
""",
        encoding="utf-8",
    )
    (fake_bin / "getent").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "systemd-analyze").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for path in fake_bin.iterdir():
        path.chmod(0o700)

    result = subprocess.run(
        ["bash", str(source_repo / "scripts" / "install_openclaw_bridge_timer.sh")],
        cwd=source_repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "REPO": str(test_repo),
            "PYTHON": str(source_repo / ".venv" / "bin" / "python"),
            "UNIT_DIR": str(tmp_path / "units"),
            "DRY_RUN": "0",
            "OPENCLAW_GROUP": primary_group,
            "OPENCLAW_USER": "nobody",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    data_acl = subprocess.check_output(["getfacl", "-cp", private_data], text=True)
    runtime_acl = subprocess.check_output(["getfacl", "-cp", test_repo / "runtime"], text=True)
    context_acl = subprocess.check_output(["getfacl", "-cp", context], text=True)
    incoming_acl = subprocess.check_output(["getfacl", "-cp", inbox_root / "incoming"], text=True)
    assert "user:nobody:---" in data_acl
    assert "default:user:nobody:---" in runtime_acl
    assert "user:nobody:r--" in context_acl
    assert "user:nobody:rwx" in incoming_acl
    assert (private_data / "candles.parquet").stat().st_mode & 0o777 == 0o644


def test_communications_installers_generate_hardened_units_in_dry_run(tmp_path):
    repo = Path.cwd()
    runtime = repo / "runtime"
    runtime.mkdir(exist_ok=True)
    telegram_env = tmp_path / "telegram.env"
    telegram_env.write_text(
        "AUTOPILOT_TELEGRAM_BOT_TOKEN=fake\nAUTOPILOT_TELEGRAM_CHAT_ID=123\n",
        encoding="utf-8",
    )
    telegram_env.chmod(0o600)
    common = {
        **os.environ,
        "REPO": str(repo),
        "PYTHON": str(repo / ".venv" / "bin" / "python"),
        "UNIT_DIR": str(tmp_path / "units"),
        "DRY_RUN": "1",
    }

    telegram = subprocess.run(
        ["bash", "scripts/install_communications_service.sh"],
        env={**common, "TELEGRAM_ENV": str(telegram_env)},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    bridge = subprocess.run(
        ["bash", "scripts/install_openclaw_bridge_timer.sh"],
        env=common,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert telegram.returncode == 0, telegram.stderr
    assert bridge.returncode == 0, bridge.stderr
    telegram_unit = (tmp_path / "units" / "trading-bot-telegram.service").read_text()
    report_unit = (tmp_path / "units" / "trading-bot-telegram-report.service").read_text()
    report_timer = (tmp_path / "units" / "trading-bot-telegram-report.timer").read_text()
    bridge_unit = (tmp_path / "units" / "trading-bot-openclaw-bridge.service").read_text()
    working_directory = str(repo).replace("%", "%%")
    assert f"WorkingDirectory={working_directory}" in telegram_unit
    assert f"WorkingDirectory={working_directory}" in report_unit
    assert f"WorkingDirectory={working_directory}" in bridge_unit
    assert "ProtectSystem=strict" in telegram_unit
    assert str(repo / ".env") in telegram_unit
    assert "EnvironmentFile=" not in telegram_unit
    assert f'--settings-file "{telegram_env}"' in telegram_unit
    assert "--validate-settings" in telegram_unit
    assert f'ReadOnlyPaths="{telegram_env}"' in telegram_unit
    assert f'ReadWritePaths="{repo / "runtime" / "operator-control"}"' in telegram_unit
    assert f'ReadWritePaths="{repo / "runtime" / "telegram"}"' in telegram_unit
    assert f'ReadWritePaths="{runtime}"' not in telegram_unit
    assert (
        f'--poll-state "{repo / "runtime" / "telegram" / "telegram_poll_state.json"}"'
        in telegram_unit
    )
    assert "--send-status" in report_unit
    assert f'ReadOnlyPaths="{telegram_env}"' in report_unit
    assert "ReadWritePaths=" not in report_unit
    assert "OnUnitActiveSec=24h" in report_timer
    assert "EnvironmentFile=" not in bridge_unit
    assert "Environment=OPENCLAW_SHARED_GROUP=0" in bridge_unit
    assert "RestrictAddressFamilies=AF_UNIX" in bridge_unit


def test_telegram_installer_rejects_unknown_duplicate_and_malformed_settings(tmp_path):
    repo = Path.cwd()
    base_environment = {
        **os.environ,
        "REPO": str(repo),
        "PYTHON": str(repo / ".venv" / "bin" / "python"),
        "DRY_RUN": "1",
    }
    cases = {
        "unknown": (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=valid-token\n"
            "AUTOPILOT_TELEGRAM_CHAT_ID=123\n"
            "EXCHANGE_API_SECRET=must-never-appear\n",
            "must-never-appear",
        ),
        "duplicate": (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=first-private-value\n"
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=second-private-value\n"
            "AUTOPILOT_TELEGRAM_CHAT_ID=123\n",
            "second-private-value",
        ),
        "malformed": (
            "AUTOPILOT_TELEGRAM_BOT_TOKEN=valid-token\n"
            "this is not an assignment and contains private-material\n",
            "private-material",
        ),
    }

    for name, (content, forbidden) in cases.items():
        settings_file = tmp_path / f"{name}.env"
        settings_file.write_text(content, encoding="utf-8")
        settings_file.chmod(0o600)
        unit_dir = tmp_path / f"units-{name}"

        result = subprocess.run(
            ["bash", "scripts/install_communications_service.sh"],
            env={
                **base_environment,
                "TELEGRAM_ENV": str(settings_file),
                "UNIT_DIR": str(unit_dir),
            },
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

        rendered = result.stdout + result.stderr
        assert result.returncode != 0
        assert forbidden not in rendered
        assert not list(unit_dir.glob("*.service"))


def test_telegram_installer_copies_legacy_state_into_dedicated_directories(tmp_path):
    source_repo = Path.cwd()
    repo = tmp_path / "repo"
    runtime = repo / "runtime"
    config_dir = repo / "config"
    runtime.mkdir(parents=True)
    config_dir.mkdir()
    control_dir = runtime / "operator-control"
    config = config_dir / "autopilot.json"
    config.write_text(
        json.dumps(
            {
                "control_file": str(control_dir / "control.json"),
                "control_audit_file": str(control_dir / "control_audit.jsonl"),
                "status_file": str(runtime / "status.json"),
                "jobs": [],
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    legacy_control = runtime / "control.json"
    legacy_audit = runtime / "control_audit.jsonl"
    legacy_poll = runtime / "telegram_poll_state.json"
    legacy_control.write_text('{"paused": true, "reason": "legacy pause"}\n', encoding="utf-8")
    legacy_audit.write_text('{"command": "pause"}\n', encoding="utf-8")
    legacy_poll.write_text('{"next_update_id": 42}\n', encoding="utf-8")
    telegram_env = runtime / "telegram.env"
    telegram_env.write_text(
        "AUTOPILOT_TELEGRAM_BOT_TOKEN=fake\nAUTOPILOT_TELEGRAM_CHAT_ID=123\n",
        encoding="utf-8",
    )
    telegram_env.chmod(0o600)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o700)
    systemd_analyze = fake_bin / "systemd-analyze"
    systemd_analyze.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    systemd_analyze.chmod(0o700)

    result = subprocess.run(
        ["bash", str(source_repo / "scripts" / "install_communications_service.sh")],
        cwd=source_repo,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "REPO": str(repo),
            "PYTHON": str(source_repo / ".venv" / "bin" / "python"),
            "CONFIG": str(config),
            "TELEGRAM_ENV": str(telegram_env),
            "UNIT_DIR": str(tmp_path / "units"),
            "DRY_RUN": "0",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (control_dir / "control.json").read_text(encoding="utf-8") == legacy_control.read_text(
        encoding="utf-8"
    )
    assert (control_dir / "control_audit.jsonl").read_text(
        encoding="utf-8"
    ) == legacy_audit.read_text(encoding="utf-8")
    assert (runtime / "telegram" / "telegram_poll_state.json").read_text(
        encoding="utf-8"
    ) == legacy_poll.read_text(encoding="utf-8")
    assert legacy_control.exists()
    assert legacy_audit.exists()
    assert legacy_poll.exists()
