import os
import subprocess
import sys
from pathlib import Path


def systemd_unit_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def test_systemd_installer_validates_config_before_starting_service():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert '"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate' in script
    assert '"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"' in script
    assert "ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --validate" in script
    assert "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT" in script
    assert script.index('"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate') < script.index(
        '"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"'
    )
    assert script.index('"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"') < script.index(
        'systemctl --user enable --now "$SERVICE_NAME"'
    )


def test_systemd_installer_validates_unit_names_before_deriving_paths():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "validate_unit_name()" in script
    assert 'validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"' in script
    assert (
        'validate_unit_name "$HEALTHCHECK_SERVICE_NAME" ".service" "HEALTHCHECK_SERVICE_NAME"'
        in script
    )
    assert 'validate_unit_name "$HEALTHCHECK_TIMER_NAME" ".timer" "HEALTHCHECK_TIMER_NAME"' in script
    assert script.index('validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"') < script.index(
        'UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"'
    )


def test_systemd_installer_validates_raw_unit_values_before_deriving_paths():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "validate_unit_value()" in script
    assert "validate_positive_integer()" in script
    assert "validate_zero_or_one()" in script
    assert 'validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"' in script
    assert 'validate_positive_integer "$AUTOPILOT_TASKS_MAX" "AUTOPILOT_TASKS_MAX"' in script
    assert 'validate_unit_value "$AUTOPILOT_MEMORY_MAX" "AUTOPILOT_MEMORY_MAX"' in script
    assert 'validate_unit_value "$AUTOPILOT_CPU_QUOTA" "AUTOPILOT_CPU_QUOTA"' in script
    assert 'validate_unit_value "$HEALTHCHECK_ON_BOOT" "HEALTHCHECK_ON_BOOT"' in script
    assert 'validate_unit_value "$HEALTHCHECK_INTERVAL" "HEALTHCHECK_INTERVAL"' in script
    assert 'validate_zero_or_one "$DRY_RUN" "DRY_RUN"' in script
    assert script.index('validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"') < script.index(
        'UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"'
    )


def test_systemd_unit_runs_readiness_before_service_start():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT "
        "--output $READINESS_REPORT_UNIT "
        "--json-output $READINESS_JSON_UNIT"
    ) in script
    assert script.index(
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --validate"
    ) < script.index(
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT"
    )
    assert script.index(
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT"
    ) < script.index(
        "ExecStart=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT"
    )


def test_systemd_unit_loads_env_and_has_restart_bounds():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "EnvironmentFile=$ENV_FILE_UNIT" in script
    assert "Restart=always" in script
    assert "RestartSec=10" in script
    assert "StartLimitIntervalSec=300" in script
    assert "StartLimitBurst=5" in script
    assert "TimeoutStopSec=30" in script
    assert "KillSignal=SIGINT" in script


def test_systemd_units_limit_scientific_python_thread_pools():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'AUTOPILOT_THREADS="${AUTOPILOT_THREADS:-2}"' in script
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        assert f"Environment={variable}=$AUTOPILOT_THREADS" in script


def test_systemd_units_have_configurable_cgroup_resource_limits():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'AUTOPILOT_MEMORY_MAX="${AUTOPILOT_MEMORY_MAX:-1G}"' in script
    assert 'AUTOPILOT_CPU_QUOTA="${AUTOPILOT_CPU_QUOTA:-75%}"' in script
    assert 'AUTOPILOT_TASKS_MAX="${AUTOPILOT_TASKS_MAX:-128}"' in script
    assert "MemoryAccounting=true" in script
    assert "MemoryMax=$AUTOPILOT_MEMORY_MAX" in script
    assert "CPUAccounting=true" in script
    assert "CPUQuota=$AUTOPILOT_CPU_QUOTA" in script
    assert "TasksAccounting=true" in script
    assert "TasksMax=$AUTOPILOT_TASKS_MAX" in script


def test_systemd_unit_has_light_server_sandboxing_and_private_logs():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "umask 077" in script
    assert "NoNewPrivileges=true" in script
    assert "PrivateTmp=true" in script
    assert "PrivateDevices=true" in script
    assert "ProtectClock=true" in script
    assert "ProtectControlGroups=true" in script
    assert "ProtectKernelLogs=true" in script
    assert "ProtectKernelModules=true" in script
    assert "ProtectKernelTunables=true" in script
    assert "RestrictSUIDSGID=true" in script
    assert "LockPersonality=true" in script
    assert "UMask=0077" in script
    assert "Nice=5" in script
    assert "IOSchedulingPriority=7" in script
    assert "StandardOutput=journal" in script
    assert "StandardError=journal" in script


def test_systemd_installer_creates_healthcheck_timer():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'HEALTHCHECK_SERVICE_NAME="${HEALTHCHECK_SERVICE_NAME:-trading-bot-autopilot-healthcheck.service}"' in script
    assert 'HEALTHCHECK_TIMER_NAME="${HEALTHCHECK_TIMER_NAME:-trading-bot-autopilot-healthcheck.timer}"' in script
    assert 'HEALTHCHECK_INTERVAL="${HEALTHCHECK_INTERVAL:-5min}"' in script
    assert 'UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"' in script
    assert "cat > \"$HEALTHCHECK_SERVICE_FILE\" <<UNIT" in script
    assert "cat > \"$HEALTHCHECK_TIMER_FILE\" <<UNIT" in script
    assert "ExecStart=$PYTHON_UNIT -m src.autopilot.healthcheck --config $CONFIG_UNIT --output $HEALTHCHECK_JSON_UNIT" in script
    assert "OnUnitActiveSec=$HEALTHCHECK_INTERVAL" in script
    assert "Unit=$HEALTHCHECK_SERVICE_NAME" in script
    assert 'systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"' in script
    assert script.index("systemctl --user enable --now \"$SERVICE_NAME\"") < script.index(
        "systemctl --user enable --now \"$HEALTHCHECK_TIMER_NAME\""
    )


def test_healthcheck_systemd_unit_loads_env_and_uses_private_runtime_settings():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    healthcheck_start = script.index('cat > "$HEALTHCHECK_SERVICE_FILE" <<UNIT')
    healthcheck_end = script.index('cat > "$HEALTHCHECK_TIMER_FILE" <<UNIT', healthcheck_start)
    healthcheck_block = script[healthcheck_start:healthcheck_end]
    assert "Type=oneshot" in healthcheck_block
    assert "WorkingDirectory=$REPO_UNIT" in healthcheck_block
    assert "EnvironmentFile=$ENV_FILE_UNIT" in healthcheck_block
    assert "NoNewPrivileges=true" in healthcheck_block
    assert "PrivateTmp=true" in healthcheck_block
    assert "PrivateDevices=true" in healthcheck_block
    assert "ProtectClock=true" in healthcheck_block
    assert "ProtectControlGroups=true" in healthcheck_block
    assert "ProtectKernelLogs=true" in healthcheck_block
    assert "ProtectKernelModules=true" in healthcheck_block
    assert "ProtectKernelTunables=true" in healthcheck_block
    assert "RestrictSUIDSGID=true" in healthcheck_block
    assert "LockPersonality=true" in healthcheck_block
    assert "UMask=0077" in healthcheck_block
    assert "StandardOutput=journal" in healthcheck_block
    assert "StandardError=journal" in healthcheck_block


def test_installer_dry_run_generates_units_without_systemctl(tmp_path):
    repo = Path.cwd()
    service_repo = tmp_path / "autopilot repo % path"
    python_dir = tmp_path / "python bin % path"
    python_dir.mkdir(parents=True)
    python_target = repo / ".venv" / "bin" / "python"
    if not python_target.exists():
        python_target = Path(sys.executable)
    python_link = python_dir / "python with space"
    python_link.write_text(f'#!/bin/sh\nexec "{python_target}" "$@"\n', encoding="utf-8")
    python_link.chmod(0o700)
    config_dir = tmp_path / "config % path"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "autopilot config.json"
    config_file.write_text((repo / "config" / "autopilot.json").read_text(encoding="utf-8"), encoding="utf-8")

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(tmp_path / "dry-run-units"),
        "REPO": str(service_repo),
        "PYTHON": str(python_link),
        "CONFIG": str(config_file),
        "DRY_RUN": "1",
        "SERVICE_NAME": "test-autopilot.service",
        "HEALTHCHECK_SERVICE_NAME": "test-autopilot-healthcheck.service",
        "HEALTHCHECK_TIMER_NAME": "test-autopilot-healthcheck.timer",
        "AUTOPILOT_THREADS": "1",
        "AUTOPILOT_MEMORY_MAX": "512M",
        "AUTOPILOT_CPU_QUOTA": "50%",
        "AUTOPILOT_TASKS_MAX": "64",
    }

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run complete. Wrote unit files:" in result.stdout
    unit_dir = tmp_path / "dry-run-units"
    service = (unit_dir / "test-autopilot.service").read_text(encoding="utf-8")
    health_service = (unit_dir / "test-autopilot-healthcheck.service").read_text(encoding="utf-8")
    timer = (unit_dir / "test-autopilot-healthcheck.timer").read_text(encoding="utf-8")
    repo_unit = systemd_unit_quote(str(service_repo))
    python_unit = systemd_unit_quote(str(python_link))
    config_unit = systemd_unit_quote(str(config_file))
    env_file_unit = systemd_unit_quote("-" + str(service_repo / ".env"))
    readiness_report_unit = systemd_unit_quote(str(service_repo / "runtime" / "readiness_report.md"))
    readiness_json_unit = systemd_unit_quote(str(service_repo / "runtime" / "readiness_report.json"))
    healthcheck_json_unit = systemd_unit_quote(str(service_repo / "runtime" / "healthcheck.json"))
    assert f"WorkingDirectory={repo_unit}" in service
    assert f"EnvironmentFile={env_file_unit}" in service
    assert f"ExecStartPre={python_unit} -m src.autopilot.runtime --config {config_unit} --validate" in service
    assert (
        f"ExecStartPre={python_unit} -m src.autopilot.readiness --config {config_unit} "
        f"--output {readiness_report_unit} --json-output {readiness_json_unit}"
    ) in service
    assert f"ExecStart={python_unit} -m src.autopilot.runtime --config {config_unit}" in service
    assert f"WorkingDirectory={repo_unit}" in health_service
    assert f"EnvironmentFile={env_file_unit}" in health_service
    assert f"ExecStart={python_unit} -m src.autopilot.healthcheck --config {config_unit} --output {healthcheck_json_unit}" in health_service
    assert "Environment=OMP_NUM_THREADS=1" in service
    assert "Environment=OPENBLAS_NUM_THREADS=1" in service
    assert "Environment=LOKY_MAX_CPU_COUNT=1" in health_service
    assert "MemoryMax=512M" in service
    assert "CPUQuota=50%" in service
    assert "TasksMax=64" in service
    assert "MemoryMax=512M" in health_service
    assert "CPUQuota=50%" in health_service
    assert "TasksMax=64" in health_service
    assert "Unit=test-autopilot-healthcheck.service" in timer
    assert "OnUnitActiveSec=5min" in timer
    assert (unit_dir / "test-autopilot.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-autopilot-healthcheck.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-autopilot-healthcheck.timer").stat().st_mode & 0o777 == 0o600


def test_installer_rejects_invalid_unit_name_before_writing_files(tmp_path):
    repo = Path.cwd()
    unit_dir = tmp_path / "dry-run-units"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(unit_dir),
        "REPO": str(tmp_path / "service-repo"),
        "PYTHON": str(tmp_path / "missing-python"),
        "CONFIG": str(tmp_path / "missing-config.json"),
        "DRY_RUN": "1",
        "SERVICE_NAME": "../escape.service",
    }

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "SERVICE_NAME must be a systemd .service unit name" in result.stderr
    assert not unit_dir.exists()


def test_installer_rejects_invalid_raw_unit_values_before_writing_files(tmp_path):
    repo = Path.cwd()
    unit_dir = tmp_path / "dry-run-units"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(unit_dir),
        "REPO": str(tmp_path / "service-repo"),
        "PYTHON": str(tmp_path / "missing-python"),
        "CONFIG": str(tmp_path / "missing-config.json"),
        "DRY_RUN": "1",
        "AUTOPILOT_THREADS": "0",
    }

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "AUTOPILOT_THREADS must be a positive integer" in result.stderr
    assert not unit_dir.exists()


def test_installer_rejects_control_characters_in_raw_unit_values(tmp_path):
    repo = Path.cwd()
    unit_dir = tmp_path / "dry-run-units"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(unit_dir),
        "REPO": str(tmp_path / "service-repo"),
        "PYTHON": str(tmp_path / "missing-python"),
        "CONFIG": str(tmp_path / "missing-config.json"),
        "DRY_RUN": "1",
        "AUTOPILOT_CPU_QUOTA": "75%\nNoNewPrivileges=false",
    }

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "AUTOPILOT_CPU_QUOTA must be non-empty" in result.stderr
    assert not unit_dir.exists()


def test_installer_rejects_ambiguous_dry_run_flag_before_writing_files(tmp_path):
    repo = Path.cwd()
    unit_dir = tmp_path / "dry-run-units"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(unit_dir),
        "REPO": str(tmp_path / "service-repo"),
        "PYTHON": str(tmp_path / "missing-python"),
        "CONFIG": str(tmp_path / "missing-config.json"),
        "DRY_RUN": "true",
    }

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "DRY_RUN must be 0 or 1" in result.stderr
    assert not unit_dir.exists()
