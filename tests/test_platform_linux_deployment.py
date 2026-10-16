from __future__ import annotations

import os
import shlex
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
    gateway = (ROOT / "deploy/systemd/trading-platform-market-gateway.service").read_text()
    data_writer = (ROOT / "deploy/systemd/trading-platform-data-writer.service").read_text()
    operations = (ROOT / "deploy/systemd/trading-platform-operations.service").read_text()

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
    assert "--process market-gateway" in gateway
    assert "--process data-writer" in data_writer
    assert "--process scheduler-operations" in operations
    assert 'install_platform_unit "$REPO/deploy/systemd/trading-platform@.service"' not in installer
    assert not (ROOT / "deploy/systemd/trading-platform@.service").exists()
    assert "common.env" not in runtime + research + agent + migration
    assert 'install -d -m 0750 -o root -g trading-platform "$REPO/data"' in installer
    assert (
        'setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-evaluator:rx,u:trading-agent:--x "$REPO/data"'
        in installer
    )
    assert (
        'setfacl -m u:trading-research:rx,u:trading-evaluator:rx "$REPO/data/$directory"'
        in installer
    )
    assert (
        'setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X "$REPO/data/$directory"'
        in installer
    )
    assert (
        'setfacl -m u:trading-runtime:rwx,u:trading-research:rx,u:trading-evaluator:rx,u:trading-agent:--x "$REPO/runtime"'
        in installer
    )
    assert "/opt/trading-bot/data/artefacts" not in research
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
    assert (
        'setfacl -R -m u:trading-research:r-X,u:trading-evaluator:r-X,u:trading-agent:--- "$REPO/.git"'
        in installer
    )
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
    assert "systemctl enable trading-platform-market-gateway.service" in installer
    assert "systemctl enable trading-platform-data-writer.service" in installer
    assert "systemctl enable trading-platform-operations.service" in installer
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
            str(INSTALLED_ROOT / "runtime/numba-cache"),
        ),
        "trading-research": (
            str(INSTALLED_ROOT / "data/research"),
            str(INSTALLED_ROOT / "runtime/research"),
        ),
        "trading-evaluator": (
            str(INSTALLED_ROOT / "data/artefacts"),
            str(INSTALLED_ROOT / "runtime/evaluation"),
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
    typed_probes = (
        (
            "trading-runtime",
            INSTALLED_ROOT / ".venv-runtime/bin/python",
            INSTALLED_ROOT / "data/bars",
            "parquet",
            "import pathlib, sys; import pyarrow as pa; import pyarrow.parquet as pq; p = pathlib.Path(sys.argv[1]); pq.write_table(pa.table({'probe': [1]}), p); assert pq.read_table(p).num_rows == 1; p.unlink()",
        ),
        (
            "trading-research",
            INSTALLED_ROOT / ".venv-research/bin/python",
            INSTALLED_ROOT / "data/research",
            "parquet",
            "import pathlib, sys; import pyarrow as pa; import pyarrow.parquet as pq; p = pathlib.Path(sys.argv[1]); pq.write_table(pa.table({'probe': [1]}), p); assert pq.read_table(p).num_rows == 1; p.unlink()",
        ),
        (
            "trading-runtime",
            INSTALLED_ROOT / ".venv-runtime/bin/python",
            INSTALLED_ROOT / "runtime",
            "json",
            "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text('{}'); assert json.loads(p.read_text()) == {}; p.unlink()",
        ),
        (
            "trading-research",
            INSTALLED_ROOT / ".venv-research/bin/python",
            INSTALLED_ROOT / "runtime/research",
            "json",
            "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text('{}'); assert json.loads(p.read_text()) == {}; p.unlink()",
        ),
        (
            "trading-evaluator",
            INSTALLED_ROOT / ".venv-runtime/bin/python",
            INSTALLED_ROOT / "data/artefacts",
            "json",
            "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text('{}'); assert json.loads(p.read_text()) == {}; p.unlink()",
        ),
        (
            "trading-agent",
            INSTALLED_ROOT / ".venv-agent/bin/python",
            INSTALLED_ROOT / "runtime/agent-worktrees",
            "json",
            "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text('{}'); assert json.loads(p.read_text()) == {}; p.unlink()",
        ),
    )
    for index, (user, python, directory, kind, code) in enumerate(typed_probes):
        probe = directory / f".platform-{kind}-probe-{os.getpid()}-{index}.{kind}"
        result = subprocess.run(
            [
                "runuser",
                "-u",
                user,
                "--",
                "env",
                "-i",
                f"PATH={os.defpath}",
                str(python),
                "-c",
                code,
                str(probe),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{user} cannot write {kind} in {directory}: {result.stderr}"
    private_paths = {
        "trading-agent": (
            INSTALLED_ROOT / ".git/config",
            INSTALLED_ROOT / "runtime/research",
            INSTALLED_ROOT / "runtime/backups",
            Path("/etc/trading-platform/runtime.env"),
            Path("/etc/trading-platform/research.env"),
            Path("/etc/trading-platform/evaluator.env"),
        ),
        "trading-research": (
            INSTALLED_ROOT / "data/artefacts",
            INSTALLED_ROOT / "runtime/backups",
            Path("/etc/trading-platform/runtime.env"),
            Path("/etc/trading-platform/evaluator.env"),
        ),
        "trading-evaluator": (
            INSTALLED_ROOT / "runtime/backups",
            Path("/etc/trading-platform/runtime.env"),
        ),
    }
    for user, paths in private_paths.items():
        for path in paths:
            result = subprocess.run(
                ["runuser", "-u", user, "--", "test", "-r", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 1, f"{user} can read protected path: {path}"


@pytest.mark.skipif(
    os.geteuid() != 0
    or not INSTALLED_ROOT.is_dir()
    or not Path("/etc/trading-platform/runtime.env").is_file()
    or not all(
        os.environ.get(f"TRADING_PLATFORM_INSTALLED_{domain}_DATABASE_URL", "").startswith(
            "postgresql"
        )
        for domain in ("RUNTIME", "RESEARCH", "EVALUATOR")
    )
    or not all(
        Path(path).is_file()
        for path in (
            INSTALLED_ROOT / ".venv-runtime/bin/python",
            INSTALLED_ROOT / ".venv-research/bin/python",
        )
    ),
    reason="requires the installed Linux platform, service runtimes, and PostgreSQL",
)
def test_installed_database_service_users_can_run_one_real_service_cycle() -> None:
    from sqlalchemy.engine import make_url

    domains = ("RUNTIME", "RESEARCH", "EVALUATOR")
    database_urls = {
        domain: make_url(os.environ[f"TRADING_PLATFORM_INSTALLED_{domain}_DATABASE_URL"])
        for domain in domains
    }
    production_value = None
    with Path("/etc/trading-platform/runtime.env").open(encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.partition("=")
            if separator and key.strip() == "TRADING_PLATFORM_DATABASE_URL":
                parsed = shlex.split(value.strip(), comments=False)
                production_value = parsed[0] if parsed else None
    assert production_value
    production_url = make_url(production_value)
    production_database = (production_url.database or "").casefold()
    target_identities = set()
    logins = set()
    for domain, url in database_urls.items():
        assert url.drivername.startswith("postgresql")
        assert url.database
        assert url.database.casefold() not in {"trading_platform", production_database}, domain
        assert url.username
        host = (url.host or "").casefold()
        if host in {"localhost", "127.0.0.1", "::1"}:
            host = "loopback"
        target_identities.add((host, url.port or 5432, url.database.casefold()))
        logins.add(url.username)
    assert len(target_identities) == 1
    assert len(logins) == 3

    services = (
        ("trading-runtime", ".venv-runtime", "product-supervisor", "RUNTIME"),
        ("trading-research", ".venv-research", "research-worker", "RESEARCH"),
        ("trading-research", ".venv-research", "agent-coordinator", "RESEARCH"),
        ("trading-evaluator", ".venv-runtime", "evaluation-worker", "EVALUATOR"),
    )
    for user, environment_name, service, domain in services:
        database_url = os.environ[f"TRADING_PLATFORM_INSTALLED_{domain}_DATABASE_URL"]
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "TRADING_PLATFORM_DATABASE_URL": database_url,
            "TRADING_LIVE": "0",
            "EXCHANGE_TESTNET": "1",
            "NUMBA_CACHE_DIR": str(
                INSTALLED_ROOT
                / {
                    "RUNTIME": "runtime/numba-cache",
                    "RESEARCH": "runtime/research/numba-cache",
                    "EVALUATOR": "runtime/evaluation/numba-cache",
                }[domain]
            ),
            "TRADING_PLATFORM_AGENT_WORKTREE_ROOT": str(
                INSTALLED_ROOT / "runtime/research/agent-worktrees"
            ),
        }
        result = subprocess.run(
            [
                "runuser",
                "-u",
                user,
                "--",
                str(INSTALLED_ROOT / environment_name / "bin/python"),
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
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{user} could not run {service}: {result.stderr}"
