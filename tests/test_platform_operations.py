"""Operational boundaries exercised without changing the host service manager."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_service_drains_completed_jobs_and_polls_live_inputs_without_cycle_barrier():
    from unittest.mock import Mock, call

    from src.services.process import _service_loop

    stopping = Mock()
    stopping.is_set.side_effect = [False, False, True]
    runtime = SimpleNamespace(
        service_name="data-writer",
        run_once=Mock(
            side_effect=[
                SimpleNamespace(healthy=True, detail={"job_id": "batch-1"}),
                SimpleNamespace(healthy=True, detail={"reason_code": "market_event_queue_empty"}),
            ]
        ),
    )
    assert _service_loop(
        runtime, Mock(), stopping, SimpleNamespace(once=False, interval_seconds=15)
    )
    assert stopping.wait.call_args_list == [call(0.01), call(1.0)]


def test_fast_work_keeps_heartbeat_writes_bounded_but_publishes_health_changes(tmp_path):
    from src.data.database import PlatformDatabase
    from src.services.config import load_platform_config
    from src.services.health import DatabaseHeartbeatStore
    from src.services.runtime import ServiceRuntime

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'heartbeats.sqlite3'}")
    database.create_schema()
    store = DatabaseHeartbeatStore(database.engine)
    runtime = ServiceRuntime(
        config=load_platform_config(Path("config/platform.json")),
        node_id="linux-optiplex",
        service_name="data-writer",
        heartbeat_store=store,
        heartbeat_interval_seconds=15,
    )
    first = "2026-10-15T00:00:00+00:00"
    runtime.run_once(lambda: {"reason_code": "market_event_written"}, observed_at=first)
    runtime.run_once(
        lambda: {"reason_code": "market_event_queue_empty"}, observed_at="2026-10-15T00:00:01+00:00"
    )
    assert store.latest()[0].observed_at == first
    runtime.run_once(
        lambda: {"reason_code": "market_event_write_failed", "error": "disk full"},
        observed_at="2026-10-15T00:00:02+00:00",
    )
    assert store.latest()[0].healthy is False
    runtime.run_once(
        lambda: {"reason_code": "market_event_written"}, observed_at="2026-10-15T00:00:03+00:00"
    )
    assert store.latest()[0].healthy is True
    assert store.latest()[0].observed_at == "2026-10-15T00:00:03+00:00"
    database.dispose()


def test_parquet_backup_preserves_source_and_quarantined_evidence(tmp_path, monkeypatch):
    from src.services import backup_service

    data = tmp_path / "data"
    originals = {
        "bars/current.parquet": b"current-bars",
        "quarantine/original.parquet": b"audit-bars",
    }
    for name, content in originals.items():
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    config = SimpleNamespace(paths={"parquet": str(data), "backups": str(tmp_path / "backups")})
    monkeypatch.setattr(backup_service, "load_platform_config", lambda _: config)
    identity = backup_service.create_parquet_backup(config_path=tmp_path / "config.json")
    restored = tmp_path / "restored"
    backup_service.restore_parquet_backup(
        config_path=tmp_path / "config.json", backup_id=identity, destination=restored
    )
    for name, content in originals.items():
        assert (data / name).read_bytes() == content
        assert (restored / name).read_bytes() == content


@pytest.mark.parametrize("target", ["platform-testnet-connected", "platform-testnet-emergency"])
def test_testnet_commands_require_confirmation_before_running_python(tmp_path, target):
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "python-invoked"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        ["make", target, f"PY={executable}", "CONFIRM=0"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Set CONFIRM=1" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "scenario",
    [
        "backup_failure",
        "inventory_failure",
        "lingering_agent",
        "migration_failure",
        "user_authority",
    ],
)
def test_upgrade_stops_all_trading_writers_before_backup(tmp_path, scenario):
    root = Path(__file__).resolve().parents[1]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls.jsonl"
    mock = binaries / "mock"
    mock.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
name, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(os.environ["TEST_CALLS"], "a") as handle:
    handle.write(json.dumps([name, *args]) + "\\n")
scenario = os.environ["TEST_SCENARIO"]
if name == "uname": print("Linux")
elif name == "id": print("0")
elif name == "getent":
    if scenario == "user_authority":
        print("legacy:x:1000:1000::" + os.environ["TEST_ACCOUNT_HOME"] + ":/bin/sh")
elif name == "runuser":
    print("trading-bot-candidate-paper.service loaded active running")
elif name == "python":
    sys.exit(0 if scenario == "migration_failure" and "src.services.backup_service" in args else 37)
elif name == "systemctl":
    if args[0] == "list-units":
        if scenario == "inventory_failure": sys.exit(1)
        print("trading-platform-agent@agent-sandbox.service loaded active running")
        print("trading-platform-backup@parquet.service loaded activating start")
        print("trading-platform-backup-parquet.timer loaded active waiting")
    elif args[0] == "show":
        print("activating" if scenario == "lingering_agent" and "agent@" in args[1] else "inactive")
    elif args[0] == "start":
        sys.exit(37 if scenario == "backup_failure" else 0)
else: sys.exit(99)
"""
    )
    mock.chmod(0o755)
    for name in ("uname", "id", "getent", "runuser", "systemctl", "python"):
        (binaries / name).symlink_to(mock)
    account_home = tmp_path / "legacy"
    (account_home / ".config/systemd/user").mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(root / "scripts/upgrade_platform.sh")],
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "REPO": str(root),
            "PYTHON": str(binaries / "python"),
            "PLATFORM_ENV_FILE": str(tmp_path / "absent.env"),
            "TRADING_PLATFORM_DATABASE_URL": "fixture-only",
            "TEST_CALLS": str(calls),
            "TEST_SCENARIO": scenario,
            "TEST_ACCOUNT_HOME": str(account_home),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert result.returncode != 0
    python_calls = [call for call in recorded if call[0] == "python"]
    if scenario == "user_authority":
        assert python_calls == []
        assert not any(call[:2] == ["systemctl", "stop"] for call in recorded)
        assert "Retire active legacy user units" in result.stderr
        return
    if scenario in {"backup_failure", "migration_failure"}:
        stops = next(call[2:] for call in recorded if call[:2] == ["systemctl", "stop"])
        assert {
            "trading-platform-agent@agent-sandbox.service",
            "trading-platform-backup@parquet.service",
            "trading-platform-backup-parquet.timer",
        }.issubset(stops)
        backup_calls = [call[2] for call in recorded if call[:2] == ["systemctl", "start"]]
        assert backup_calls == [
            f"trading-platform-backup@{mode}.service"
            for mode in (
                ("postgresql",)
                if scenario == "backup_failure"
                else ("postgresql", "parquet", "verify")
            )
        ]
        assert len(python_calls) == (0 if scenario == "backup_failure" else 1)
        if scenario == "migration_failure":
            assert python_calls[0][-1] == "src.services.platform_upgrade"
    else:
        assert python_calls == []
        assert (
            "Cannot inspect" in result.stderr
            if scenario == "inventory_failure"
            else "did not stop" in result.stderr
        )


def test_agent_process_uses_transport_without_opening_a_database(monkeypatch):
    from src.services import process
    from src.services.config import load_split_configuration

    config_path = Path(__file__).resolve().parents[1] / "config/platform.json"
    configuration = load_split_configuration(config_path.parent)
    secret_names = {"TRADING_PLATFORM_DATABASE_URL"}
    secret_names.update(
        str(account[field])
        for account in configuration["accounts"]["accounts"]
        for field in ("api_key_env", "api_secret_env")
        if account.get(field)
    )
    for name in secret_names:
        monkeypatch.delenv(name, raising=False)
    calls = []

    def no_database(*_args, **_kwargs):
        pytest.fail("agent process attempted to open a database")

    def consume():
        calls.append("transport")
        return {"state": "idle"}

    monkeypatch.setattr(process, "PlatformDatabase", no_database)
    monkeypatch.setattr("src.agents.consumer.consume_configured_review", consume)
    args = process.parse_args(
        [
            "--config",
            str(config_path),
            "--node",
            "linux-optiplex",
            "--process",
            "agent-runtime",
            "--once",
        ]
    )
    assert process.run(args) == 0
    assert calls == ["transport"]
    for name in secret_names:
        with monkeypatch.context() as patch:
            patch.setenv(name, "forbidden-fixture-credential")
            with pytest.raises(PermissionError, match="must not receive"):
                process.run(args)
    assert calls == ["transport"]


@pytest.mark.parametrize(
    "scenario", ["enabled_timer", "active_service", "unreadable_manager", "quiet_user"]
)
def test_installer_checks_user_authorities_before_system_mutations(tmp_path, scenario):
    root = Path(__file__).resolve().parents[1]
    account_home = tmp_path / "legacy-user"
    units = account_home / ".config/systemd/user"
    units.mkdir(parents=True)
    if scenario == "enabled_timer":
        wanted = units / "default.target.wants"
        wanted.mkdir()
        (wanted / "trading-bot-candidate-paper.timer").symlink_to(
            "../trading-bot-candidate-paper.timer"
        )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls.jsonl"
    mock = binaries / "mock"
    mock.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
if name == "id":
    print("0")
elif name == "getent":
    print("legacy-user:x:1000:1000::" + os.environ["TEST_ACCOUNT_HOME"] + ":/bin/sh")
elif name == "runuser":
    if os.environ["TEST_SCENARIO"] == "unreadable_manager":
        sys.exit(1)
    if os.environ["TEST_SCENARIO"] == "active_service":
        print("trading-bot-candidate-paper.service loaded activating start legacy paper")
elif name == "systemctl":
    with open(os.environ["TEST_CALLS"], "a") as handle:
        handle.write(json.dumps(args) + "\\n")
    if args[0] == "stop":
        sys.exit(73)
else:
    sys.exit(99)
"""
    )
    mock.chmod(0o755)
    for name in ("id", "getent", "runuser", "systemctl"):
        (binaries / name).symlink_to(mock)
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "REPO": str(root),
        "NODE": "linux-optiplex",
        "DRY_RUN": "0",
        "TEST_ACCOUNT_HOME": str(account_home),
        "TEST_SCENARIO": scenario,
        "TEST_CALLS": str(calls),
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/install_platform_services.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    recorded = (
        [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    )
    assert result.returncode != 0
    if scenario == "quiet_user":
        assert any(call[0] == "stop" for call in recorded)
        assert "current unit did not stop" in result.stderr
    else:
        assert not any(
            call[0] in {"stop", "disable", "enable", "daemon-reload"} for call in recorded
        )
        expected = "Cannot inspect" if scenario == "unreadable_manager" else "legacy user units"
        assert expected in result.stderr


def test_external_heartbeat_withholds_stale_unhealthy_and_other_node_state():
    from src.services.health import ServiceHealth, publish_external_heartbeat

    sent = []
    store = SimpleNamespace(latest=lambda: tuple(rows))
    sink = SimpleNamespace(send=sent.append)
    now = "2026-09-09T00:01:00+00:00"
    rows = [ServiceHealth("runtime", "host", now, True, {"private": "never transmit"})]
    options = {"sink": sink, "node_id": "host", "required_services": ("runtime",), "now": now}
    assert publish_external_heartbeat(store, **options) == "delivered"
    assert len(sent) == 1 and sent[0].payload == {}
    for observed, healthy, node in (
        ("2026-09-08T23:59:59+00:00", True, "host"),
        ("2026-09-09T00:01:01+00:00", True, "host"),
        (now, False, "host"),
        (now, True, "other-host"),
    ):
        rows[:] = [ServiceHealth("runtime", node, observed, healthy, {})]
        assert publish_external_heartbeat(store, **options) == "withheld_unhealthy"
    assert len(sent) == 1
    rows[:] = [ServiceHealth("runtime", "host", now, True, {})]

    def unavailable(_record):
        raise RuntimeError("network unavailable")

    sink.send = unavailable
    assert publish_external_heartbeat(store, **options) == "delivery_failed"


def test_archive_verification_has_bounded_memory_for_many_files(tmp_path):
    import tarfile
    import tracemalloc

    from src.services.backups import verify_directory_archive

    path = tmp_path / "many.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("data")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for index in range(20_000):
            archive.addfile(tarfile.TarInfo(f"data/{index}.parquet"))
    del archive
    tracemalloc.start()
    try:
        assert verify_directory_archive(path) == 20_001
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024


def test_corrupt_archive_never_publishes_a_restore(tmp_path):
    from src.services.backups import create_directory_archive, restore_directory_archive

    source = tmp_path / "data"
    source.mkdir()
    (source / "source.parquet").write_bytes(b"retained evidence")
    archive = create_directory_archive(source, tmp_path / "data.tar.gz")
    corrupted = bytearray(archive.read_bytes())
    corrupted[-8] ^= 1
    archive.write_bytes(corrupted)
    destination = tmp_path / "restored"
    with pytest.raises(OSError):
        restore_directory_archive(archive, destination)
    assert not destination.exists()
    assert (source / "source.parquet").read_bytes() == b"retained evidence"


def test_backup_retention_keeps_recent_daily_and_last_verified_recovery_points(tmp_path):
    import datetime as dt

    from src.services.backups import BackupStore

    store = BackupStore(tmp_path / "backups")
    source = tmp_path / "source"
    dates = {
        "db-expired": "2026-01-01T00:00:00+00:00",
        "db-daily-old": "2026-01-29T08:00:00+00:00",
        "db-daily-new": "2026-01-29T20:00:00+00:00",
        "db-recent": "2026-01-30T14:00:00+00:00",
        "db-latest": "2026-01-31T00:00:00+00:00",
        "parquet-only": "2025-12-01T00:00:00+00:00",
    }
    for identity, created in dates.items():
        source.write_text(identity)
        name = "parquet.tar.gz" if identity == "parquet-only" else "database.dump"
        store.create(backup_id=identity, created_at=created, files={name: source})
    now = dt.datetime(2026, 1, 31, 12, tzinfo=dt.UTC).timestamp()
    removed = store.prune(older_than_timestamp=now - 30 * 86400, thin_before_timestamp=now - 86400)
    assert set(removed) == {"db-expired", "db-daily-old"}
    for identity in ("db-daily-new", "db-recent", "db-latest", "parquet-only"):
        store.verify(identity)
    source.write_text("new")
    store.create(
        backup_id="db-new", created_at="2026-01-31T06:00:00+00:00", files={"database.dump": source}
    )
    (store.root / "db-new/database.dump").write_text("corrupt")
    with pytest.raises(ValueError, match="content hash"):
        store.prune(older_than_timestamp=now, thin_before_timestamp=now)
    assert (store.root / "db-latest").is_dir()
    assert (store.root / "parquet-only").is_dir()


def test_research_hard_deadline_terminates_a_stuck_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.services.research_worker import _job_deadline; import time\nwith _job_deadline(.1): time.sleep(10)",
        ],
        timeout=60,
        capture_output=True,
    )
    assert result.returncode == 124


def test_backup_verification_ignores_unpublished_builds(tmp_path, monkeypatch):
    from src.services import backup_service
    from src.services.backups import BackupStore

    root = tmp_path / "backups"
    source = tmp_path / "source"
    source.write_bytes(b"verified recovery data")
    store = BackupStore(root)
    store.create(
        backup_id="published", created_at="2026-01-01T00:00:00+00:00", files={"evidence": source}
    )
    for name in (".parquet-build-pending", ".postgresql-build-pending", ".incomplete.tmp"):
        (root / name).mkdir()
        (root / name / "partial").write_bytes(b"incomplete")
    config = SimpleNamespace(paths={"backups": str(root)})
    monkeypatch.setattr(backup_service, "load_platform_config", lambda _: config)
    assert backup_service.verify_backups(config_path=tmp_path / "config") == 1


def test_backup_verification_rehearses_newest_database_and_writes_bound_receipt(
    tmp_path, monkeypatch
):
    from src.services import backup_service
    from src.services.backups import BackupStore, file_hash

    root = tmp_path / "backups"
    source = tmp_path / "database.dump"
    store = BackupStore(root)
    for identity, created_at in (
        ("postgresql-old", "2026-10-14T00:00:00+00:00"),
        ("postgresql-new", "2026-10-15T00:00:00+00:00"),
    ):
        source.write_bytes(identity.encode())
        store.create(
            backup_id=identity,
            created_at=created_at,
            files={"database.dump": source},
        )
    config = SimpleNamespace(paths={"backups": str(root)})
    monkeypatch.setattr(backup_service, "load_platform_config", lambda _: config)
    verified_dumps = []
    rehearsed_dumps = []
    summary = {
        "schema_revision": "040",
        "table_counts": {"job": 2},
    }
    monkeypatch.setattr(
        backup_service,
        "verify_postgresql_dump",
        lambda path: verified_dumps.append(path),
    )
    monkeypatch.setattr(
        backup_service,
        "rehearse_postgresql_restore",
        lambda path, scratch_root: (rehearsed_dumps.append(path), summary)[1],
    )

    assert backup_service.verify_backups(config_path=tmp_path / "config") == 2
    assert verified_dumps == [
        root / "postgresql-new/database.dump",
        root / "postgresql-old/database.dump",
    ]
    assert rehearsed_dumps == [root / "postgresql-new/database.dump"]
    receipt_path = root / "restore-rehearsal-postgresql-new.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["backup_id"] == "postgresql-new"
    assert receipt["backup_created_at"] == "2026-10-15T00:00:00+00:00"
    assert receipt["backup_manifest_hash"] == file_hash(root / "postgresql-new/manifest.json")
    assert receipt["database_dump_hash"] == file_hash(root / "postgresql-new/database.dump")
    assert receipt["restore"] == summary
    assert not (root / "restore-rehearsal-postgresql-old.json").exists()
    monkeypatch.setattr(
        backup_service,
        "prune_backups",
        lambda **_: pytest.fail("verification must not prune recovery points"),
    )
    backup_service.main(["--mode", "verify", "--config", str(tmp_path / "config")])
    assert (root / "postgresql-old/database.dump").is_file()


def test_archive_creation_bounds_inode_metadata_memory(tmp_path):
    from src.services.backups import verify_directory_archive

    source = tmp_path / "many-files"
    source.mkdir()
    for index in range(20_000):
        (source / f"{index}.parquet").touch()
    archive = tmp_path / "many-files.tar.gz"
    # Measure only the archiver, excluding pytest and background test allocations.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, tracemalloc; "
            "from src.services.backups import create_directory_archive; "
            "tracemalloc.start(); "
            "create_directory_archive(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])); "
            "print(tracemalloc.get_traced_memory()[1])",
            str(source),
            str(archive),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    peak = int(result.stdout.strip())
    assert peak < 8 * 1024 * 1024
    assert verify_directory_archive(archive) == 20_001
