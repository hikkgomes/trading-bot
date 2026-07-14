import os
import re
import subprocess
import sys
from pathlib import Path


def systemd_unit_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)


def real_install_env(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    fake_python = fake_bin / "python"
    write_executable(fake_python, "#!/bin/sh\nexit 0\n")
    write_executable(fake_bin / "id", "#!/bin/sh\necho autopilot-test\n")
    config = tmp_path / "autopilot.json"
    config.write_text("{}\n", encoding="utf-8")
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "USER": "autopilot-test",
        "UNIT_DIR": str(tmp_path / "units"),
        "REPO": str(tmp_path / "repo"),
        "PYTHON": str(fake_python),
        "CONFIG": str(config),
        "DRY_RUN": "0",
    }


def test_systemd_installer_validates_config_before_starting_service():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert '"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate' in script
    assert '"$PYTHON" -m src.autopilot.alert_settings --file "$ALERT_ENV_FILE"' in script
    assert '"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"' in script
    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT "
        "--validate --skip-jobs"
    ) in script
    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT" not in script
    )
    assert script.index(
        '"$PYTHON" -m src.autopilot.runtime --config "$CONFIG" --validate'
    ) < script.index('"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"')
    assert script.index('"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"') < script.index(
        'systemctl --user enable --now "$SERVICE_NAME"'
    )


def test_systemd_installer_validates_unit_names_before_deriving_paths():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "validate_unit_name()" in script
    assert 'validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"' in script
    assert 'validate_unit_name "$JOB_SERVICE_NAME" ".service" "JOB_SERVICE_NAME"' in script
    assert (
        'validate_unit_name "$HEALTHCHECK_SERVICE_NAME" ".service" "HEALTHCHECK_SERVICE_NAME"'
        in script
    )
    assert (
        'validate_unit_name "$HEALTHCHECK_TIMER_NAME" ".timer" "HEALTHCHECK_TIMER_NAME"' in script
    )
    assert script.index(
        'validate_unit_name "$SERVICE_NAME" ".service" "SERVICE_NAME"'
    ) < script.index('UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"')


def test_systemd_installer_validates_raw_unit_values_before_deriving_paths():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "validate_unit_value()" in script
    assert "validate_positive_integer()" in script
    assert "validate_zero_or_one()" in script
    assert 'validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"' in script
    assert 'validate_positive_integer "$AUTOPILOT_TASKS_MAX" "AUTOPILOT_TASKS_MAX"' in script
    assert 'validate_unit_value "$AUTOPILOT_MEMORY_MAX" "AUTOPILOT_MEMORY_MAX"' in script
    assert 'validate_unit_value "$AUTOPILOT_CPU_QUOTA" "AUTOPILOT_CPU_QUOTA"' in script
    assert 'validate_unit_value "$ALERT_ENV_FILE" "ALERT_ENV_FILE"' in script
    assert 'validate_unit_value "$TELEGRAM_ENV_FILE" "TELEGRAM_ENV_FILE"' in script
    assert 'validate_unit_value "$HEALTHCHECK_ON_BOOT" "HEALTHCHECK_ON_BOOT"' in script
    assert 'validate_unit_value "$HEALTHCHECK_INTERVAL" "HEALTHCHECK_INTERVAL"' in script
    assert 'validate_zero_or_one "$DRY_RUN" "DRY_RUN"' in script
    assert script.index(
        'validate_positive_integer "$AUTOPILOT_THREADS" "AUTOPILOT_THREADS"'
    ) < script.index('UNIT_FILE="$UNIT_DIR/$SERVICE_NAME"')


def test_systemd_installer_verifies_linger_before_enabling_units():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert "ensure_user_linger()" in script
    assert 'loginctl show-user "$target_user" --property=Linger' in script
    assert 'TARGET_USER="$(id -un)"' in script
    assert 'ensure_user_linger "$TARGET_USER"' in script
    assert script.index('ensure_user_linger "$TARGET_USER"') < script.index(
        'systemctl --user enable --now "$SERVICE_NAME"'
    )
    dry_run_exit = script.index('if [ "$DRY_RUN" = "1" ]')
    assert dry_run_exit < script.index('ensure_user_linger "$TARGET_USER"')
    assert 'loginctl enable-linger "$USER" >/dev/null 2>&1 || true' not in script


def test_real_installer_fails_actionably_when_linger_cannot_be_enabled(tmp_path):
    fake_bin = tmp_path / "bin"
    write_executable(
        fake_bin / "loginctl",
        """#!/bin/sh
if [ "$1" = "show-user" ]; then
  echo "Linger=no"
  exit 0
fi
if [ "$1" = "enable-linger" ]; then
  exit 1
fi
exit 2
""",
    )
    write_executable(fake_bin / "systemctl", "#!/bin/sh\necho systemctl-invoked >&2\nexit 99\n")

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=Path.cwd(),
        env=real_install_env(tmp_path, fake_bin),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert "Could not enable user lingering for autopilot-test" in result.stderr
    assert "sudo loginctl enable-linger autopilot-test" in result.stderr
    assert "systemctl-invoked" not in result.stderr


def test_real_installer_enables_and_verifies_linger_before_systemd(tmp_path):
    fake_bin = tmp_path / "bin"
    linger_state = tmp_path / "linger-enabled"
    systemctl_log = tmp_path / "systemctl.log"
    write_executable(
        fake_bin / "loginctl",
        f"""#!/bin/sh
if [ "$1" = "show-user" ]; then
  if [ -f "{linger_state}" ]; then echo "Linger=yes"; else echo "Linger=no"; fi
  exit 0
fi
if [ "$1" = "enable-linger" ]; then
  : > "{linger_state}"
  exit 0
fi
exit 2
""",
    )
    write_executable(
        fake_bin / "systemctl",
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{systemctl_log}"\nexit 0\n',
    )

    result = subprocess.run(
        ["bash", "scripts/install_autopilot_service.sh"],
        cwd=Path.cwd(),
        env=real_install_env(tmp_path, fake_bin),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert linger_state.exists()
    calls = systemctl_log.read_text(encoding="utf-8").splitlines()
    assert "--user daemon-reload" in calls
    assert "--user enable --now trading-bot-autopilot.service" in calls
    assert "--user enable --now trading-bot-autopilot-jobs.service" in calls
    assert "--user enable --now trading-bot-autopilot-healthcheck.timer" in calls


def test_server_requirements_cover_enabled_autopilot_job_dependencies():
    requirement_lines = [
        line.strip()
        for line in Path("requirements-bot.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    requirements = {
        re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower() for line in requirement_lines
    }

    assert {
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "scipy",
        "scikit-learn",
        "ta-lib",
        "ccxt",
    } <= requirements
    assert {
        "aiodns",
        "aiohttp",
        "certifi",
        "cffi",
        "charset-normalizer",
        "coincurve",
        "cryptography",
        "idna",
        "joblib",
        "pycares",
        "python-dateutil",
        "threadpoolctl",
        "typing_extensions",
        "urllib3",
        "yarl",
    } <= requirements
    assert all("==" in line.split(";", 1)[0] for line in requirement_lines)


def test_systemd_installer_gates_initial_enablement_but_restart_preserves_management():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert '"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"' in script
    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT "
        "--validate --skip-jobs"
    ) in script
    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.readiness --config $CONFIG_UNIT" not in script
    )
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --skip-jobs"
        in script
    )


def test_systemd_separates_trading_supervision_from_scheduled_jobs():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'JOB_SERVICE_NAME="${JOB_SERVICE_NAME:-trading-bot-autopilot-jobs.service}"' in script
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.runtime --config $CONFIG_UNIT --skip-jobs"
        in script
    )
    assert "ExecStart=$PYTHON_UNIT -m src.autopilot.job_worker --config $CONFIG_UNIT" in script
    assert 'systemctl --user enable --now "$JOB_SERVICE_NAME"' in script


def test_scheduled_job_unit_cannot_inherit_live_credentials_or_read_approvals():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")
    job_start = script.index('cat > "$JOB_SERVICE_FILE" <<UNIT')
    job_end = script.index('cat > "$HEALTHCHECK_SERVICE_FILE" <<UNIT', job_start)
    job_block = script[job_start:job_end]

    assert "EnvironmentFile=$ENV_FILE_UNIT" not in job_block
    assert (
        "UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD "
        "TRADING_LIVE EXCHANGE_TESTNET" in job_block
    )
    assert "ProtectSystem=strict" in job_block
    assert "ReadOnlyPaths=$REPO_UNIT" in job_block
    assert "ReadWritePaths=$RUNTIME_UNIT" in job_block
    assert "ReadWritePaths=$DATA_UNIT" in job_block
    assert "ReadWritePaths=$OUTPUTS_UNIT" in job_block
    assert "InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT" in job_block
    assert "InaccessiblePaths=$JOB_APPROVALS_INACCESSIBLE_UNIT" in job_block
    assert "InaccessiblePaths=$ALERT_ENV_FILE_UNIT" in job_block
    assert "InaccessiblePaths=$TELEGRAM_ENV_FILE_UNIT" in job_block
    assert "AUTOPILOT_WEBHOOK_URL" in job_block
    assert "AUTOPILOT_TELEGRAM_BOT_TOKEN" in job_block
    assert "AUTOPILOT_ALERT_SETTINGS_FILE" in job_block


def test_candidate_paper_has_a_dedicated_hardened_sub_minute_timer():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'CANDIDATE_PAPER_INTERVAL="${CANDIDATE_PAPER_INTERVAL:-45s}"' in script
    assert 'CANDIDATE_PAPER_TIMEOUT="${CANDIDATE_PAPER_TIMEOUT:-240}"' in script
    assert 'cat > "$CANDIDATE_PAPER_SERVICE_FILE" <<UNIT' in script
    assert 'cat > "$CANDIDATE_PAPER_TIMER_FILE" <<UNIT' in script
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.candidate_paper --config $CONFIG_UNIT "
        "--output $CANDIDATE_PAPER_STATUS_UNIT --lock $CANDIDATE_PAPER_LOCK_UNIT"
    ) in script
    assert "TimeoutStartSec=$CANDIDATE_PAPER_TIMEOUT" in script
    assert "OnUnitActiveSec=$CANDIDATE_PAPER_INTERVAL" in script
    assert "Unit=$CANDIDATE_PAPER_SERVICE_NAME" in script
    assert 'systemctl --user enable --now "$CANDIDATE_PAPER_TIMER_NAME"' in script


def test_backup_has_a_dedicated_credential_free_timer_with_read_only_approvals():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    assert 'BACKUP_INTERVAL="${BACKUP_INTERVAL:-24h}"' in script
    assert 'BACKUP_TIMEOUT="${BACKUP_TIMEOUT:-60}"' in script
    assert 'cat > "$BACKUP_SERVICE_FILE" <<UNIT' in script
    assert 'cat > "$BACKUP_TIMER_FILE" <<UNIT' in script
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.backup --config $CONFIG_UNIT "
        "--report $BACKUP_REPORT_UNIT --max-file-bytes 52428800 --max-backups 30"
    ) in script
    assert "ReadOnlyPaths=$BACKUP_APPROVALS_READ_ONLY_UNIT" in script
    assert 'systemctl --user enable --now "$BACKUP_TIMER_NAME"' in script


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

    assert (
        'HEALTHCHECK_SERVICE_NAME="${HEALTHCHECK_SERVICE_NAME:-trading-bot-autopilot-healthcheck.service}"'
        in script
    )
    assert (
        'HEALTHCHECK_TIMER_NAME="${HEALTHCHECK_TIMER_NAME:-trading-bot-autopilot-healthcheck.timer}"'
        in script
    )
    assert 'HEALTHCHECK_INTERVAL="${HEALTHCHECK_INTERVAL:-5min}"' in script
    assert 'UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"' in script
    assert 'cat > "$HEALTHCHECK_SERVICE_FILE" <<UNIT' in script
    assert 'cat > "$HEALTHCHECK_TIMER_FILE" <<UNIT' in script
    assert (
        "ExecStart=$PYTHON_UNIT -m src.autopilot.healthcheck --config $CONFIG_UNIT "
        "--output $HEALTHCHECK_JSON_UNIT --skip-readiness" in script
    )
    assert "OnUnitActiveSec=$HEALTHCHECK_INTERVAL" in script
    assert "Unit=$HEALTHCHECK_SERVICE_NAME" in script
    assert 'systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"' in script
    assert script.index('systemctl --user enable --now "$SERVICE_NAME"') < script.index(
        'systemctl --user enable --now "$HEALTHCHECK_TIMER_NAME"'
    )


def test_healthcheck_systemd_unit_uses_only_private_operations_credentials():
    script = Path("scripts/install_autopilot_service.sh").read_text(encoding="utf-8")

    healthcheck_start = script.index('cat > "$HEALTHCHECK_SERVICE_FILE" <<UNIT')
    healthcheck_end = script.index('cat > "$HEALTHCHECK_TIMER_FILE" <<UNIT', healthcheck_start)
    healthcheck_block = script[healthcheck_start:healthcheck_end]
    assert "Type=oneshot" in healthcheck_block
    assert "WorkingDirectory=$REPO_UNIT" in healthcheck_block
    assert "EnvironmentFile=$ENV_FILE_UNIT" not in healthcheck_block
    assert "EnvironmentFile=" not in healthcheck_block
    assert "Environment=$ALERT_ENV_ASSIGNMENT_UNIT" in healthcheck_block
    assert (
        "ExecStartPre=$PYTHON_UNIT -m src.autopilot.alert_settings --file $ALERT_ENV_PATH_UNIT"
    ) in healthcheck_block
    assert "UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET" in healthcheck_block
    assert "--skip-readiness" in healthcheck_block
    assert "ProtectSystem=strict" in healthcheck_block
    assert "ReadOnlyPaths=$REPO_UNIT" in healthcheck_block
    assert "ReadWritePaths=$RUNTIME_UNIT" in healthcheck_block
    assert "ReadOnlyPaths=$TELEGRAM_ENV_FILE_UNIT" in healthcheck_block
    assert "InaccessiblePaths=$JOB_ENV_INACCESSIBLE_UNIT" in healthcheck_block
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
    config_file.write_text(
        (repo / "config" / "autopilot.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "UNIT_DIR": str(tmp_path / "dry-run-units"),
        "REPO": str(service_repo),
        "PYTHON": str(python_link),
        "CONFIG": str(config_file),
        "DRY_RUN": "1",
        "SERVICE_NAME": "test-autopilot.service",
        "JOB_SERVICE_NAME": "test-autopilot-jobs.service",
        "CANDIDATE_PAPER_SERVICE_NAME": "test-candidate-paper.service",
        "CANDIDATE_PAPER_TIMER_NAME": "test-candidate-paper.timer",
        "BACKUP_SERVICE_NAME": "test-autopilot-backup.service",
        "BACKUP_TIMER_NAME": "test-autopilot-backup.timer",
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
    job_service = (unit_dir / "test-autopilot-jobs.service").read_text(encoding="utf-8")
    candidate_service = (unit_dir / "test-candidate-paper.service").read_text(encoding="utf-8")
    candidate_timer = (unit_dir / "test-candidate-paper.timer").read_text(encoding="utf-8")
    backup_service = (unit_dir / "test-autopilot-backup.service").read_text(encoding="utf-8")
    backup_timer = (unit_dir / "test-autopilot-backup.timer").read_text(encoding="utf-8")
    health_service = (unit_dir / "test-autopilot-healthcheck.service").read_text(encoding="utf-8")
    timer = (unit_dir / "test-autopilot-healthcheck.timer").read_text(encoding="utf-8")
    repo_unit = systemd_unit_quote(str(service_repo))
    python_unit = systemd_unit_quote(str(python_link))
    config_unit = systemd_unit_quote(str(config_file))
    env_file_unit = systemd_unit_quote("-" + str(service_repo / ".env"))
    alert_env_unit = systemd_unit_quote("-" + str(service_repo / "runtime" / "alerts.env"))
    alert_env_path_unit = systemd_unit_quote(str(service_repo / "runtime" / "alerts.env"))
    alert_env_assignment_unit = systemd_unit_quote(
        "AUTOPILOT_ALERT_SETTINGS_FILE=" + str(service_repo / "runtime" / "alerts.env")
    )
    telegram_env_unit = systemd_unit_quote("-" + str(service_repo / "runtime" / "telegram.env"))
    approvals_unit = systemd_unit_quote("-" + str(service_repo / "runtime" / "approvals.json"))
    runtime_unit = systemd_unit_quote(str(service_repo / "runtime"))
    data_unit = systemd_unit_quote(str(service_repo / "data"))
    outputs_unit = systemd_unit_quote(str(service_repo / "outputs"))
    healthcheck_json_unit = systemd_unit_quote(str(service_repo / "runtime" / "healthcheck.json"))
    candidate_status_unit = systemd_unit_quote(
        str(service_repo / "runtime" / "candidate_paper_status.json")
    )
    candidate_lock_unit = systemd_unit_quote(str(service_repo / "runtime" / "candidate_paper.lock"))
    backup_report_unit = systemd_unit_quote(str(service_repo / "runtime" / "backup_report.json"))
    assert f"WorkingDirectory={repo_unit}" in service
    assert f"EnvironmentFile={env_file_unit}" in service
    assert f"Environment={alert_env_assignment_unit}" in service
    assert f"EnvironmentFile={alert_env_unit}" not in service
    assert (
        f"ExecStartPre={python_unit} -m src.autopilot.alert_settings --file {alert_env_path_unit}"
    ) in service
    assert (
        f"ExecStartPre={python_unit} -m src.autopilot.runtime --config {config_unit} "
        "--validate --skip-jobs"
    ) in service
    assert "src.autopilot.readiness" not in service
    assert (
        f"ExecStart={python_unit} -m src.autopilot.runtime --config {config_unit} --skip-jobs"
        in service
    )
    assert f"WorkingDirectory={repo_unit}" in job_service
    assert f"EnvironmentFile={env_file_unit}" not in job_service
    assert (
        "UnsetEnvironment=EXCHANGE_API_KEY EXCHANGE_API_SECRET EXCHANGE_API_PASSWORD "
        "TRADING_LIVE EXCHANGE_TESTNET" in job_service
    )
    assert "ProtectSystem=strict" in job_service
    assert f"ReadOnlyPaths={repo_unit}" in job_service
    assert f"ReadWritePaths={runtime_unit}" in job_service
    assert f"ReadWritePaths={data_unit}" in job_service
    assert f"ReadWritePaths={outputs_unit}" in job_service
    assert f"InaccessiblePaths={env_file_unit}" in job_service
    assert f"InaccessiblePaths={approvals_unit}" in job_service
    assert f"InaccessiblePaths={alert_env_unit}" in job_service
    assert f"InaccessiblePaths={telegram_env_unit}" in job_service
    assert (
        f"ExecStart={python_unit} -m src.autopilot.job_worker --config {config_unit}" in job_service
    )
    assert "EnvironmentFile=" not in candidate_service
    assert (
        f"ExecStart={python_unit} -m src.autopilot.candidate_paper --config {config_unit} "
        f"--output {candidate_status_unit} --lock {candidate_lock_unit}"
    ) in candidate_service
    assert "TimeoutStartSec=240" in candidate_service
    assert f"ReadWritePaths={runtime_unit}" in candidate_service
    assert f"InaccessiblePaths={env_file_unit}" in candidate_service
    assert f"InaccessiblePaths={approvals_unit}" in candidate_service
    assert f"InaccessiblePaths={alert_env_unit}" in candidate_service
    assert f"InaccessiblePaths={telegram_env_unit}" in candidate_service
    assert "AUTOPILOT_WEBHOOK_URL" in candidate_service
    assert "AUTOPILOT_TELEGRAM_BOT_TOKEN" in candidate_service
    assert "MemoryMax=512M" in candidate_service
    assert "CPUQuota=50%" in candidate_service
    assert "TasksMax=64" in candidate_service
    assert "OnUnitActiveSec=45s" in candidate_timer
    assert "AccuracySec=5s" in candidate_timer
    assert "Unit=test-candidate-paper.service" in candidate_timer
    assert "EnvironmentFile=" not in backup_service
    assert (
        f"ExecStart={python_unit} -m src.autopilot.backup --config {config_unit} "
        f"--report {backup_report_unit} --max-file-bytes 52428800 --max-backups 30"
    ) in backup_service
    assert "TimeoutStartSec=60" in backup_service
    assert f"ReadWritePaths={runtime_unit}" in backup_service
    assert f"ReadOnlyPaths={approvals_unit}" in backup_service
    assert f"InaccessiblePaths={env_file_unit}" in backup_service
    assert f"InaccessiblePaths={alert_env_unit}" in backup_service
    assert f"InaccessiblePaths={telegram_env_unit}" in backup_service
    assert "AUTOPILOT_WEBHOOK_URL" in backup_service
    assert "AUTOPILOT_TELEGRAM_BOT_TOKEN" in backup_service
    assert f"InaccessiblePaths={approvals_unit}" not in backup_service
    assert "OnUnitActiveSec=24h" in backup_timer
    assert "Unit=test-autopilot-backup.service" in backup_timer
    assert f"WorkingDirectory={repo_unit}" in health_service
    assert f"EnvironmentFile={env_file_unit}" not in health_service
    assert "EnvironmentFile=" not in health_service
    assert f"Environment={alert_env_assignment_unit}" in health_service
    assert (
        f"ExecStartPre={python_unit} -m src.autopilot.alert_settings --file {alert_env_path_unit}"
    ) in health_service
    assert f"ReadOnlyPaths={telegram_env_unit}" in health_service
    assert f"InaccessiblePaths={env_file_unit}" in health_service
    assert (
        f"ExecStart={python_unit} -m src.autopilot.healthcheck --config {config_unit} "
        f"--output {healthcheck_json_unit} --skip-readiness" in health_service
    )
    assert "Environment=OMP_NUM_THREADS=1" in service
    assert "Environment=OPENBLAS_NUM_THREADS=1" in service
    assert "Environment=LOKY_MAX_CPU_COUNT=1" in health_service
    assert "MemoryMax=512M" in service
    assert "CPUQuota=50%" in service
    assert "TasksMax=64" in service
    assert "MemoryMax=512M" in job_service
    assert "CPUQuota=50%" in job_service
    assert "TasksMax=64" in job_service
    assert "MemoryMax=512M" in health_service
    assert "CPUQuota=50%" in health_service
    assert "TasksMax=64" in health_service
    assert "Unit=test-autopilot-healthcheck.service" in timer
    assert "OnUnitActiveSec=5min" in timer
    assert (unit_dir / "test-autopilot.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-autopilot-jobs.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-candidate-paper.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-candidate-paper.timer").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-autopilot-backup.service").stat().st_mode & 0o777 == 0o600
    assert (unit_dir / "test-autopilot-backup.timer").stat().st_mode & 0o777 == 0o600
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
