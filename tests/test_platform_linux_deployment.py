from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_linux_deployment_declares_shared_traversal_and_exact_writable_paths() -> None:
    installer = (ROOT / "scripts/install_platform_services.sh").read_text()
    research = (ROOT / "deploy/systemd/trading-platform-research@.service").read_text()
    agent = (ROOT / "deploy/systemd/trading-platform-agent@.service").read_text()

    assert "trading-platform" in installer
    assert 'install -d -m 0750 -o trading-runtime -g trading-platform "$REPO/data"' in installer
    assert "/opt/trading-bot/data/artefacts" in research
    assert "/opt/trading-bot/data/reports" in research
    assert "/opt/trading-bot/runtime/research" in research
    assert "TRADING_PLATFORM_AGENT_WORKTREE_ROOT=/opt/trading-bot/runtime/agent-worktrees" in agent
    assert "ReadWritePaths=/opt/trading-bot/runtime/agent-worktrees" in agent


@pytest.mark.skipif(
    os.geteuid() != 0 or not Path("/opt/trading-bot").is_dir(),
    reason="requires the installed Linux platform and root user",
)
def test_installed_service_users_can_traverse_and_write_their_real_paths() -> None:
    checks = {
        "trading-runtime": ("/opt/trading-bot/data", "/opt/trading-bot/runtime"),
        "trading-research": (
            "/opt/trading-bot/data/research",
            "/opt/trading-bot/data/artefacts",
            "/opt/trading-bot/data/reports",
            "/opt/trading-bot/runtime/research",
        ),
        "trading-agent": ("/opt/trading-bot/runtime/agent-worktrees",),
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
    or not Path("/opt/trading-bot").is_dir()
    or not os.environ.get("TRADING_PLATFORM_DATABASE_URL", "").startswith("postgresql")
    or not all(
        Path(path).is_file()
        for path in (
            "/opt/trading-bot/.venv-runtime/bin/python",
            "/opt/trading-bot/.venv-research/bin/python",
            "/opt/trading-bot/.venv-agent/bin/python",
        )
    ),
    reason="requires the installed Linux platform, service runtimes, and PostgreSQL",
)
def test_installed_service_users_can_run_one_real_service_cycle() -> None:
    services = {
        "trading-runtime": (
            "/opt/trading-bot/.venv-runtime/bin/python",
            "product-supervisor",
        ),
        "trading-research": (
            "/opt/trading-bot/.venv-research/bin/python",
            "research-worker",
        ),
        "trading-agent": (
            "/opt/trading-bot/.venv-agent/bin/python",
            "agent-sandbox",
        ),
    }
    for user, (python, service) in services.items():
        environment = [
            f"TRADING_PLATFORM_DATABASE_URL={os.environ['TRADING_PLATFORM_DATABASE_URL']}",
        ]
        if user == "trading-agent":
            environment.append(
                "TRADING_PLATFORM_AGENT_WORKTREE_ROOT=/opt/trading-bot/runtime/agent-worktrees"
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
                "/opt/trading-bot/config/platform.json",
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
