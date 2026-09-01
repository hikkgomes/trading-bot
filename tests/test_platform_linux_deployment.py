from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
INSTALLED_ROOT = Path(os.environ.get("TRADING_PLATFORM_INSTALL_ROOT", "/opt/trading-bot"))


def test_linux_deployment_declares_shared_traversal_and_exact_writable_paths() -> None:
    installer = (ROOT / "scripts/install_platform_services.sh").read_text()
    runtime = (ROOT / "deploy/systemd/trading-platform-runtime.service").read_text()
    research = (ROOT / "deploy/systemd/trading-platform-research.service").read_text()
    agent = (ROOT / "deploy/systemd/trading-platform-agent.service").read_text()
    control = (ROOT / "deploy/systemd/trading-platform-control.service").read_text()
    migration = (ROOT / "deploy/systemd/trading-platform-migration.service").read_text()
    instance = (ROOT / "deploy/systemd/trading-platform@.service").read_text()

    assert "trading-platform" in installer
    assert 'REPO="${REPO:-/home/alfred/trading-bot}"' in installer
    assert "install -d -m 0750 -o root -g trading-platform /etc/trading-platform" in installer
    assert "runtime:trading-runtime" in installer
    assert "research:trading-research" in installer
    assert "agent:trading-agent" in installer
    assert "migration:trading-platform-owner" in installer
    assert "EnvironmentFile=/etc/trading-platform/runtime.env" in runtime
    assert "EnvironmentFile=/etc/trading-platform/research.env" in research
    assert "EnvironmentFile=/etc/trading-platform/agent.env" in agent
    assert "EnvironmentFile=/etc/trading-platform/runtime.env" in control
    assert "EnvironmentFile=/etc/trading-platform/migration.env" in migration
    assert "TimeoutStartSec=300" in instance
    assert "TimeoutStopSec=120" in instance
    assert "ExecStartPre=/opt/trading-bot/.venv-runtime/bin/python" in instance
    assert 'install_platform_unit "$REPO/deploy/systemd/trading-platform@.service"' in installer
    assert "common.env" not in runtime + research + agent + migration
    assert 'install -d -m 0750 -o root -g trading-platform "$REPO/data"' in installer
    assert (
        'setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-agent:--x "$REPO/data"'
        in installer
    )
    assert 'setfacl -m u:trading-research:rx "$REPO/data/$directory"' in installer
    assert 'setfacl -R -m u:trading-research:r-X "$REPO/data/$directory"' in installer
    assert (
        'setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-agent:--x "$REPO/runtime"'
        in installer
    )
    assert "/opt/trading-bot/data/artefacts" in research
    assert "/opt/trading-bot/data/reports" not in research
    assert "/opt/trading-bot/runtime/research" in research
    assert "NUMBA_CACHE_DIR=/opt/trading-bot/runtime/research/numba-cache" in research
    assert (
        "install -d -m 2770 -o trading-research -g trading-research \\\n"
        '    "$REPO/runtime/research/numba-cache"'
    ) in installer
    assert (
        "ReadOnlyPaths=/opt/trading-bot/data/raw /opt/trading-bot/data/bars "
        "/opt/trading-bot/data/features"
    ) in research
    assert "TRADING_PLATFORM_AGENT_WORKTREE_ROOT=/opt/trading-bot/runtime/agent-worktrees" in agent
    assert "NUMBA_CACHE_DIR=/opt/trading-bot/runtime/agent-worktrees/numba-cache" in agent
    assert "GIT_CONFIG_KEY_0=safe.directory" in agent
    assert "GIT_CONFIG_VALUE_0=/opt/trading-bot" in agent
    assert (
        "install -d -m 2770 -o trading-agent -g trading-agent \\\n"
        '    "$REPO/runtime/agent-worktrees/numba-cache"'
    ) in installer
    assert "ReadWritePaths=/opt/trading-bot/runtime/agent-worktrees" in agent
    assert 'SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}"' in installer
    assert 'if [[ "$SKIP_SYSTEMD" == "1" ]]; then' in installer
    assert 'if [[ "$REPO" == /home/*/* ]]; then' in installer
    assert 'setfacl -m g:trading-platform:--x "$repository_home"' in installer
    assert (
        'setfacl -R -m g:trading-platform:r-X "$REPO/src" "$REPO/config" "$REPO/alembic"'
        in installer
    )
    assert 'setfacl -R -m u:trading-agent:r-X "$REPO/.git"' in installer
    assert '"$REPO/.venv-runtime"' in installer
    assert '"$REPO/.venv-research"' in installer
    assert '"$REPO/.venv-agent"' in installer
    assert "s|/opt/trading-bot|$REPO|g" in installer
    assert "s|ProtectHome=true|ProtectHome=$PROTECT_HOME|g" in installer
    assert (
        'install_platform_unit "$REPO/deploy/systemd/trading-platform-runtime.service"' in installer
    )
    assert (
        'install_platform_unit "$REPO/deploy/systemd/trading-platform-research.service"'
        in installer
    )
    assert (
        'install_platform_unit "$REPO/deploy/systemd/trading-platform-agent.service"' in installer
    )
    assert (
        'install_platform_unit "$REPO/deploy/systemd/trading-platform-control.service"' in installer
    )
    assert "systemctl enable trading-platform-runtime.service" in installer
    assert "systemctl enable trading-platform-research.service" in installer
    assert "trading-bot-candidate-paper.timer" in installer
    assert "trading-bot-openclaw-bridge.timer" in installer
    assert "trading-bot-telegram-report.timer" in installer
    assert "systemctl disable --now" in installer


def test_deployment_runbook_targets_the_actual_optiplex_checkout() -> None:
    runbook = (ROOT / "docs/DEPLOYMENT.md").read_text()

    assert 'git clone "$REPOSITORY_URL" /home/alfred/trading-bot' in runbook
    assert "REPO=/home/alfred/trading-bot NODE=linux-optiplex" in runbook
    assert "never against `trading_platform`" in runbook
    assert "createdb -O trading_platform_migrator trading_platform_smoke" in runbook
    assert "TRADING_CONTROL_TOKEN=<RANDOM_64_HEX_CONTROL_TOKEN>" in runbook
    assert "http://127.0.0.1:8088/status" in runbook


@pytest.mark.skipif(
    os.geteuid() != 0 or not INSTALLED_ROOT.is_dir(),
    reason="requires the installed Linux platform and root user",
)
def test_installed_service_users_can_traverse_and_write_their_real_paths() -> None:
    checks = {
        "trading-runtime": (
            str(INSTALLED_ROOT / "data/raw"),
            str(INSTALLED_ROOT / "data/bars"),
            str(INSTALLED_ROOT / "data/features"),
            str(INSTALLED_ROOT / "runtime"),
        ),
        "trading-research": (
            str(INSTALLED_ROOT / "data/research"),
            str(INSTALLED_ROOT / "data/artefacts"),
            str(INSTALLED_ROOT / "runtime/research"),
        ),
        "trading-agent": (str(INSTALLED_ROOT / "runtime/agent-worktrees"),),
    }
    for user, paths in checks.items():
        for index, path in enumerate(paths):
            probe = f"{path}/.platform-write-probe-{os.getpid()}-{index}"
            result = subprocess.run(
                [
                    "runuser",
                    "-u",
                    user,
                    "--",
                    "sh",
                    "-c",
                    ': > "$1" && rm -f "$1"',
                    "platform-write-probe",
                    probe,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"{user} cannot write {path}: {result.stderr}"


@pytest.mark.skipif(
    os.geteuid() != 0
    or not INSTALLED_ROOT.is_dir()
    or not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql")
    or not all(
        Path(path).is_file()
        for path in (
            INSTALLED_ROOT / ".venv-runtime/bin/python",
            INSTALLED_ROOT / ".venv-research/bin/python",
            INSTALLED_ROOT / ".venv-agent/bin/python",
        )
    ),
    reason="requires the installed Linux platform, service runtimes, and PostgreSQL",
)
def test_installed_service_users_can_run_one_real_service_cycle() -> None:
    services = {
        "trading-runtime": (
            str(INSTALLED_ROOT / ".venv-runtime/bin/python"),
            "product-supervisor",
        ),
        "trading-research": (
            str(INSTALLED_ROOT / ".venv-research/bin/python"),
            "research-worker",
        ),
        "trading-agent": (
            str(INSTALLED_ROOT / ".venv-agent/bin/python"),
            "agent-sandbox",
        ),
    }
    for user, (python, service) in services.items():
        environment = [
            f"TRADING_PLATFORM_DATABASE_URL={os.environ['TRADING_PLATFORM_DATABASE_URL']}",
        ]
        if user == "trading-agent":
            environment.append(
                f"TRADING_PLATFORM_AGENT_WORKTREE_ROOT={INSTALLED_ROOT}/runtime/agent-worktrees"
            )
        result = subprocess.run(
            [
                "runuser",
                "-u",
                user,
                "--",
                "env",
                *environment,
                python,
                "-m",
                "src.services.supervisor",
                "--config",
                str(INSTALLED_ROOT / "config/platform.json"),
                "--node",
                "linux-optiplex",
                "--service",
                service,
                "--once",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{user} could not run {service}: {result.stderr}"
