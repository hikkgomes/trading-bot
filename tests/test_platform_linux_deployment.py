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
    assert "/opt/trading-bot/data/reports" in research
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
    assert (
        "install -d -m 2770 -o trading-agent -g trading-agent \\\n"
        '    "$REPO/runtime/agent-worktrees/numba-cache"'
    ) in installer
    assert "ReadWritePaths=/opt/trading-bot/runtime/agent-worktrees" in agent
    assert 'SKIP_SYSTEMD="${SKIP_SYSTEMD:-0}"' in installer
    assert 'if [[ "$SKIP_SYSTEMD" == "1" ]]; then' in installer


def test_ci_runs_real_service_user_permission_rehearsal() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "sudo apt-get install -y acl" in workflow
    assert 'sudo mv "$GITHUB_WORKSPACE" /opt/trading-bot' in workflow
    assert 'sudo ln -s /opt/trading-bot "$GITHUB_WORKSPACE"' in workflow
    assert "REPO=/opt/trading-bot" in workflow
    assert "jq '.postgresql.require_tls = false'" in workflow
    assert "TRADING_PLATFORM_CONFIG=/opt/trading-bot/config/platform.ci.json" in workflow
    assert "SKIP_SYSTEMD=1" in workflow
    assert "scripts/install_platform_services.sh" in workflow
    assert "scripts/verify_platform_service_install.sh" in workflow
    assert 'sudo chown -R "$USER:$(id -gn)" /opt/trading-bot/data' in workflow


@pytest.mark.skipif(
    os.geteuid() != 0 or not Path("/opt/trading-bot").is_dir(),
    reason="requires the installed Linux platform and root user",
)
def test_installed_service_users_can_traverse_and_write_their_real_paths() -> None:
    checks = {
        "trading-runtime": (
            "/opt/trading-bot/data/raw",
            "/opt/trading-bot/data/bars",
            "/opt/trading-bot/data/features",
            "/opt/trading-bot/runtime",
        ),
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
