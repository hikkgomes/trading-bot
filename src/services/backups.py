"""Content-verified backup bundles and restore checks."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"backup input must be a regular file: {path}")
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    result = path.lstat()
    if not stat.S_ISREG(result.st_mode):
        raise ValueError(f"backup input must be a regular file: {path}")
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _move_hashed_source(
    source: Path, target: Path, logical_name: str
) -> tuple[str, tuple[int, int, int, int, int]]:
    source_identity = _file_identity(source)
    digest = file_hash(source)
    if _file_identity(source) != source_identity:
        raise RuntimeError(f"backup input changed while hashing: {logical_name}")
    os.replace(source, target)
    try:
        moved_identity = _file_identity(target)
        if moved_identity[:4] != source_identity[:4]:
            raise RuntimeError(f"backup input changed while moving: {logical_name}")
    except Exception:
        if target.exists() and not source.exists():
            os.replace(target, source)
        raise
    return digest, moved_identity


@dataclass(frozen=True)
class PostgreSQLTools:
    bindir: Path
    executables: dict[str, Path]
    major_version: int


_POSTGRESQL_TOOL_NAMES = ("initdb", "pg_ctl", "pg_restore", "postgres", "psql", "createdb")
# The retained event database must also restore on the server's shared spinning disk.
_RESTORE_TIMEOUT_SECONDS = 90 * 60
_RESTORE_ROLES = ("trading_runtime", "trading_research", "trading_evaluator")


def _tool_environment() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}


def _tool_major_version(executable: Path) -> int:
    result = subprocess.run(
        [str(executable), "--version"],
        env=_tool_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    match = re.search(r"PostgreSQL\)?\s+(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"cannot read PostgreSQL version from {executable.name}")
    return int(match.group(1))


def discover_postgresql_tools() -> PostgreSQLTools:
    pg_config = shutil.which("pg_config")
    if pg_config is None:
        raise RuntimeError("PostgreSQL restore rehearsal requires pg_config")
    result = subprocess.run(
        [pg_config, "--bindir"],
        env=_tool_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    bindir = Path(result.stdout.strip())
    if not bindir.is_absolute() or not bindir.is_dir():
        raise RuntimeError("pg_config returned an invalid PostgreSQL bindir")
    executables = {name: bindir / name for name in _POSTGRESQL_TOOL_NAMES}
    missing = [
        name
        for name, path in executables.items()
        if not path.is_file() or not os.access(path, os.X_OK)
    ]
    if missing:
        raise RuntimeError(f"PostgreSQL bindir is missing tools: {', '.join(missing)}")
    versions = {_tool_major_version(path) for path in executables.values()}
    config_version = _tool_major_version(Path(pg_config))
    if versions != {config_version}:
        raise RuntimeError("PostgreSQL tools do not share the pg_config major version")
    return PostgreSQLTools(bindir, executables, config_version)


def _safe_logical_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("backup logical names must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value == "manifest.json":
        raise ValueError("backup logical names must be safe relative paths")
    return value


def _safe_backup_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or Path(value).name != value
    ):
        raise ValueError("backup IDs must be safe directory names")
    return value


@dataclass(frozen=True)
class BackupBundle:
    backup_id: str
    created_at: str
    files: dict[str, str]


def write_postgresql_restore_receipt(
    root: Path,
    bundle: BackupBundle,
    summary: dict[str, Any],
    *,
    verified_at: dt.datetime | None = None,
) -> Path:
    backup_id = _safe_backup_id(bundle.backup_id)
    if "database.dump" not in bundle.files:
        raise ValueError("restore receipt requires a PostgreSQL dump")
    verified_at = verified_at or dt.datetime.now(dt.UTC)
    if verified_at.tzinfo is None:
        raise ValueError("restore receipt time must include a timezone")
    manifest_hash = file_hash(root / backup_id / "manifest.json")
    payload = {
        "receipt_version": 1,
        "backup_id": backup_id,
        "backup_created_at": bundle.created_at,
        "backup_manifest_hash": manifest_hash,
        "database_dump_hash": bundle.files["database.dump"],
        "verified_at": verified_at.astimezone(dt.UTC).isoformat(),
        "restore": summary,
    }
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"restore-rehearsal-{backup_id}.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(root)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


class BackupStore:
    def __init__(self, root: Path):
        self.root = root

    def discard_abandoned_builds(self) -> tuple[str, ...]:
        """Remove unpublished builds while the caller owns the maintenance lock."""
        pattern = re.compile(
            r"\.(?:postgresql|parquet)-build-[A-Za-z0-9_]+"
            r"|\.(?:postgresql|parquet)-[0-9]{8}T[0-9]{6}Z\.tmp"
        )
        removed = []
        for path in self.root.iterdir():
            if not pattern.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            if path.stat().st_uid != os.getuid():
                continue
            shutil.rmtree(path)
            removed.append(path.name)
        if removed:
            _fsync_directory(self.root)
        return tuple(sorted(removed))

    def create(
        self,
        *,
        backup_id: str,
        created_at: str,
        files: dict[str, Path],
        move_sources: bool = False,
    ) -> BackupBundle:
        backup_id = _safe_backup_id(backup_id)
        if not backup_id or not files:
            raise ValueError("backup_id and files are required")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / backup_id
        if destination.exists():
            raise ValueError(f"backup already exists: {backup_id}")
        temporary = self.root / f".{backup_id}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        hashes: dict[str, str] = {}
        moved_sources: list[tuple[Path, Path]] = []
        try:
            for logical_name, source in sorted(files.items()):
                logical_name = _safe_logical_name(logical_name)
                target = temporary / logical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                moved_identity: tuple[int, int, int, int, int] | None = None
                if move_sources:
                    digest, moved_identity = _move_hashed_source(source, target, logical_name)
                    moved_sources.append((source, target))
                else:
                    digest = file_hash(source)
                    shutil.copyfile(source, target)
                if not move_sources and file_hash(target) != digest:
                    raise RuntimeError(f"backup verification failed: {logical_name}")
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                if moved_identity is not None and _file_identity(target) != moved_identity:
                    raise RuntimeError(f"backup input changed while creating: {logical_name}")
                hashes[logical_name] = digest
            bundle = BackupBundle(backup_id=backup_id, created_at=created_at, files=hashes)
            manifest = temporary / "manifest.json"
            manifest.write_text(
                json.dumps(bundle.__dict__, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with manifest.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(self.root)
        except Exception:
            for source, target in reversed(moved_sources):
                if target.exists() and not source.exists():
                    os.replace(target, source)
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return bundle

    def _read_bundle(self, backup_id: str) -> BackupBundle:
        backup_id = _safe_backup_id(backup_id)
        directory = self.root / backup_id
        manifest = directory / "manifest.json"
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or manifest.is_symlink()
            or not manifest.is_file()
        ):
            raise ValueError("backup directory and manifest must be regular paths")
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
            raise ValueError("backup manifest is invalid")
        files: dict[str, str] = {}
        for logical_name, expected_hash in payload["files"].items():
            safe_name = _safe_logical_name(logical_name)
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 71
                or not expected_hash.startswith("sha256:")
            ):
                raise ValueError(f"backup manifest has an invalid hash: {safe_name}")
            try:
                int(expected_hash[7:], 16)
            except ValueError as exc:
                raise ValueError(f"backup manifest has an invalid hash: {safe_name}") from exc
            files[safe_name] = expected_hash
        bundle = BackupBundle(
            backup_id=str(payload.get("backup_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            files=files,
        )
        if bundle.backup_id != backup_id:
            raise ValueError("backup manifest ID does not match its directory")
        return bundle

    def verify(self, backup_id: str) -> BackupBundle:
        bundle = self._read_bundle(backup_id)
        for logical_name, expected_hash in bundle.files.items():
            if file_hash(self.root / backup_id / logical_name) != expected_hash:
                raise ValueError(f"backup content hash does not match: {logical_name}")
        return bundle

    def restore(
        self, *, backup_id: str, destination: Path, overwrite: bool = False
    ) -> tuple[Path, ...]:
        bundle = self.verify(backup_id)
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError("restore destination must be a regular directory")
        destination.mkdir(parents=True, exist_ok=True)
        destination_root = destination.resolve()
        source_root = self.root / backup_id
        restored: list[Path] = []
        for logical_name, expected_hash in sorted(bundle.files.items()):
            target = destination / logical_name
            try:
                target.resolve(strict=False).relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"restore target escapes destination: {target}") from exc
            if target.exists() and not overwrite:
                raise FileExistsError(f"restore refuses to overwrite existing file: {target}")
            if target.is_symlink():
                raise ValueError(f"restore target must not be a symlink: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.restore.tmp")
            shutil.copyfile(source_root / logical_name, temporary)
            if file_hash(temporary) != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"restore verification failed: {logical_name}")
            os.replace(temporary, target)
            restored.append(target)
        return tuple(restored)

    def prune(
        self,
        *,
        older_than_timestamp: float,
        thin_before_timestamp: float | None = None,
        logical_name: str | None = None,
    ) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        if logical_name is not None:
            logical_name = _safe_logical_name(logical_name)
        bundles = []
        for directory in self.root.iterdir():
            if (
                directory.is_dir()
                and not directory.is_symlink()
                and not directory.name.startswith(".")
                and (directory / "manifest.json").is_file()
            ):
                bundle = self._read_bundle(directory.name)
                if logical_name is None or tuple(sorted(bundle.files)) == (logical_name,):
                    bundles.append(bundle)
        removed, latest = _retention_plan(bundles, older_than_timestamp, thin_before_timestamp)
        # Never delete an earlier recovery point unless its replacement verifies.
        for backup_id in latest:
            self.verify(backup_id)
        for backup_id in removed:
            shutil.rmtree(self.root / backup_id)
        return removed


def _backup_epoch(bundle: BackupBundle) -> float:
    value = dt.datetime.fromisoformat(bundle.created_at)
    if value.tzinfo is None or not bundle.files:
        raise ValueError("backup retention requires a dated, non-empty manifest")
    return value.timestamp()


def _retention_plan(bundles, cutoff: float, recent: float | None):
    ordered = sorted(
        ((_backup_epoch(bundle), bundle) for bundle in bundles),
        key=lambda item: (item[0], item[1].backup_id),
        reverse=True,
    )
    latest = {}
    days = set()
    removed = []
    for created, bundle in ordered:
        kind = tuple(sorted(bundle.files))
        day = (kind, dt.datetime.fromtimestamp(created, dt.UTC).date())
        if kind not in latest:
            latest[kind] = bundle.backup_id
        elif created < cutoff or (recent is not None and created < recent and day in days):
            removed.append(bundle.backup_id)
        days.add(day)
    return tuple(removed), tuple(latest.values())


def postgresql_environment(database_url: str) -> dict[str, str]:
    parts = urlsplit(database_url.replace("postgresql+psycopg://", "postgresql://", 1))
    if parts.scheme != "postgresql" or not parts.hostname or not parts.path.strip("/"):
        raise ValueError("backup database URL must identify a PostgreSQL database")
    query = parse_qs(parts.query)
    environment = {
        "PGHOST": parts.hostname,
        "PGPORT": str(parts.port or 5432),
        "PGDATABASE": unquote(parts.path.lstrip("/")),
        "PGCONNECT_TIMEOUT": str(query.get("connect_timeout", ["5"])[-1]),
    }
    if parts.username:
        environment["PGUSER"] = unquote(parts.username)
    if parts.password:
        environment["PGPASSWORD"] = unquote(parts.password)
    if query.get("sslmode"):
        environment["PGSSLMODE"] = str(query["sslmode"][-1])
    return environment


class PostgreSQLBackup:
    """Run PostgreSQL native backup tools without exposing credentials in argv."""

    def __init__(self, database_url: str):
        self.environment = {**os.environ, **postgresql_environment(database_url)}

    def create(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "pg_dump",
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    "--file",
                    str(temporary),
                ],
                env=self.environment,
                capture_output=True,
                text=True,
                check=True,
            )
            self.verify(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def verify(self, dump_path: Path) -> None:
        verify_postgresql_dump(dump_path)

    def restore(self, dump_path: Path) -> None:
        """Restore into the configured target, which must be prepared and empty."""
        self.verify(dump_path)
        subprocess.run(
            [
                "pg_restore",
                "--exit-on-error",
                "--no-owner",
                "--no-privileges",
                "--dbname",
                self.environment["PGDATABASE"],
                str(dump_path),
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )


def verify_postgresql_dump(dump_path: Path) -> None:
    if dump_path.is_symlink() or not dump_path.is_file():
        raise ValueError("PostgreSQL dump must be a regular file")
    pg_restore = shutil.which("pg_restore")
    if pg_restore is None:
        raise RuntimeError("PostgreSQL backup verification requires pg_restore")
    subprocess.run(
        [pg_restore, "--list", str(dump_path)],
        env=_tool_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )


def _run_restore_tool(
    executable: Path,
    arguments: list[str],
    environment: Mapping[str, str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _restore_socket_environment(
    bindir: Path, socket_directory: Path, scratch: Path
) -> dict[str, str]:
    return {
        "PATH": str(bindir),
        "HOME": str(scratch),
        "LC_ALL": "C",
        "PGHOST": str(socket_directory),
        "PGPORT": "5432",
        "PGUSER": "restore_owner",
        "PGDATABASE": "postgres",
        "PGCONNECT_TIMEOUT": "5",
    }


def _restore_server_options(socket_directory: Path) -> str:
    options = (
        "-c",
        "listen_addresses=",
        "-c",
        f"unix_socket_directories={socket_directory}",
        "-c",
        "unix_socket_permissions=0700",
        "-c",
        "port=5432",
        "-c",
        "max_connections=2",
        "-c",
        "superuser_reserved_connections=1",
        "-c",
        "shared_buffers=64MB",
        "-c",
        "work_mem=4MB",
        "-c",
        "maintenance_work_mem=64MB",
        "-c",
        "temp_buffers=8MB",
        "-c",
        "max_parallel_workers=0",
        "-c",
        "max_parallel_workers_per_gather=0",
        "-c",
        "autovacuum=off",
        "-c",
        "statement_timeout=1800000",
        "-c",
        "idle_in_transaction_session_timeout=60000",
    )
    return shlex.join(options)


def _restore_server_running(tools: PostgreSQLTools, data_directory: Path) -> bool:
    result = _run_restore_tool(
        tools.executables["pg_ctl"],
        ["-D", str(data_directory), "status"],
        _tool_environment(),
        timeout=10,
        check=False,
    )
    if result.returncode not in (0, 3):
        raise RuntimeError("cannot confirm local PostgreSQL restore cluster status")
    return result.returncode == 0


def _stop_restore_server(tools: PostgreSQLTools, data_directory: Path) -> None:
    if not _restore_server_running(tools, data_directory):
        return
    pg_ctl = tools.executables["pg_ctl"]
    try:
        _run_restore_tool(
            pg_ctl,
            ["-D", str(data_directory), "-m", "fast", "-w", "-t", "45", "stop"],
            _tool_environment(),
            timeout=50,
        )
    except subprocess.SubprocessError:
        if _restore_server_running(tools, data_directory):
            _run_restore_tool(
                pg_ctl,
                ["-D", str(data_directory), "-m", "immediate", "-w", "-t", "20", "stop"],
                _tool_environment(),
                timeout=25,
            )
    if _restore_server_running(tools, data_directory):
        raise RuntimeError("local PostgreSQL restore cluster did not stop")


def _restore_database_schema(tools: PostgreSQLTools, environment: Mapping[str, str]) -> None:
    psql = tools.executables["psql"]
    for role in _RESTORE_ROLES:
        _run_restore_tool(
            psql,
            [
                "-X",
                "-w",
                "-v",
                "ON_ERROR_STOP=1",
                "--dbname=postgres",
                "--command",
                f"CREATE ROLE {role} NOLOGIN",
            ],
            environment,
            timeout=30,
        )
    _run_restore_tool(
        tools.executables["createdb"],
        ["--no-password", "restore_rehearsal"],
        environment,
        timeout=30,
    )


def _restored_database_summary(
    tools: PostgreSQLTools, environment: Mapping[str, str]
) -> dict[str, Any]:
    query = """
        SELECT json_build_object(
            'server_version_num', current_setting('server_version_num')::integer,
            'schema_revision', (SELECT version_num FROM alembic_version LIMIT 1),
            'table_counts', json_build_object(
                'alembic_version', (SELECT count(*) FROM alembic_version),
                'schema_migration', (SELECT count(*) FROM schema_migration),
                'job', (SELECT count(*) FROM job),
                'experiment', (SELECT count(*) FROM experiment),
                'dataset_bundle', (SELECT count(*) FROM dataset_bundle),
                'platform_bootstrap', (SELECT count(*) FROM platform_bootstrap),
                'fill', (SELECT count(*) FROM fill),
                'financial_application', (SELECT count(*) FROM financial_application),
                'accounting_entry', (SELECT count(*) FROM accounting_entry)
            )
        )::text
    """
    result = _run_restore_tool(
        tools.executables["psql"],
        [
            "-X",
            "-w",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "--dbname=restore_rehearsal",
            "--command",
            query,
        ],
        environment,
        timeout=60,
    )
    summary = json.loads(result.stdout)
    if not isinstance(summary, dict) or not summary.get("schema_revision"):
        raise ValueError("restored PostgreSQL database has no schema revision")
    counts = summary.get("table_counts")
    if not isinstance(counts, dict) or len(counts) != 9 or counts.get("alembic_version") != 1:
        raise ValueError("restored PostgreSQL database is missing expected tables")
    return summary


def rehearse_postgresql_restore(dump_path: Path, scratch_root: Path) -> dict[str, Any]:
    """Restore one verified dump into a private, temporary local PostgreSQL cluster."""
    if dump_path.is_symlink() or not dump_path.is_file():
        raise ValueError("PostgreSQL dump must be a regular file")
    tools = discover_postgresql_tools()
    scratch_root.mkdir(parents=True, exist_ok=True)
    data_directory = Path(tempfile.mkdtemp(prefix=".restore-rehearsal-", dir=scratch_root))
    socket_directory: Path | None = None
    started = False
    try:
        socket_directory = Path(tempfile.mkdtemp(prefix="pg-restore-sock-", dir="/tmp"))
        environment = _restore_socket_environment(tools.bindir, socket_directory, data_directory)
        _run_restore_tool(
            tools.executables["initdb"],
            [
                "-D",
                str(data_directory),
                "--no-locale",
                "--encoding=UTF8",
                "--username=restore_owner",
                "--auth-local=trust",
                "--auth-host=reject",
            ],
            _tool_environment(),
            timeout=120,
        )
        started = True
        _run_restore_tool(
            tools.executables["pg_ctl"],
            [
                "-D",
                str(data_directory),
                "-l",
                str(data_directory / "restore.log"),
                "-o",
                _restore_server_options(socket_directory),
                "-w",
                "-t",
                "60",
                "start",
            ],
            environment,
            timeout=70,
        )
        _restore_database_schema(tools, environment)
        _run_restore_tool(
            tools.executables["pg_restore"],
            [
                "--exit-on-error",
                "--no-owner",
                "--no-privileges",
                "--jobs=1",
                "--dbname=restore_rehearsal",
                str(dump_path),
            ],
            environment,
            timeout=_RESTORE_TIMEOUT_SECONDS,
        )
        return _restored_database_summary(tools, environment)
    finally:
        stopped = not started
        if started:
            try:
                try:
                    _stop_restore_server(tools, data_directory)
                    stopped = True
                except Exception as exc:
                    raise RuntimeError(
                        "local PostgreSQL restore cluster may still be running; "
                        f"private scratch retained at {data_directory} and {socket_directory}"
                    ) from exc
            finally:
                if stopped:
                    shutil.rmtree(data_directory)
                    if socket_directory is not None:
                        shutil.rmtree(socket_directory)
        else:
            shutil.rmtree(data_directory)
            if socket_directory is not None:
                shutil.rmtree(socket_directory)


def create_directory_archive(source: Path, destination: Path) -> Path:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a regular directory: {source}")
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("backup destination must be outside its source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with gzip.open(temporary, "wb", compresslevel=3) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                _archive_committed_tree(archive, source)
        verify_directory_archive(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _archive_committed_tree(archive, source: Path) -> None:
    for directory_name, directories, files in os.walk(source):
        directory = Path(directory_name)
        if any((directory / name).is_symlink() for name in directories):
            raise ValueError("backup source contains a symbolic link")
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        if directory.is_symlink():
            raise ValueError("backup source contains a symbolic link")
        archive.addfile(
            archive.gettarinfo(str(directory), str(directory.relative_to(source.parent)))
        )
        archive.members.clear()
        archive.inodes.clear()
        for name in sorted(files):
            if name.startswith(".") or name.endswith((".tmp", ".lock")):
                continue
            path = directory / name
            if path.is_symlink():
                raise ValueError("backup source contains a symbolic link")
            # Metadata and bytes refer to one opened inode even if a writer replaces the path.
            with path.open("rb") as handle:
                member = archive.gettarinfo(
                    fileobj=handle, arcname=str(path.relative_to(source.parent))
                )
                archive.addfile(member, handle)
                archive.members.clear()
                archive.inodes.clear()


@contextmanager
def _stream_archive(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("backup archive must be a regular file")
    with gzip.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            yield archive
        # Consume the gzip trailer as well, so truncation and CRC failures fail verification.
        while compressed.read(1024 * 1024):
            pass


def _archive_members(archive):
    root = None
    while (member := archive.next()) is not None:
        # Python 3.11 retains headers even in pipe mode. No backwards lookup is needed here.
        archive.members.clear()
        parts = Path(member.name).parts
        if (
            not parts
            or member.name.startswith("/")
            or ".." in parts
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError("backup archive contains an unsafe member")
        if root is None:
            if len(parts) != 1 or not member.isdir():
                raise ValueError("backup archive must start with its directory root")
            root = parts[0]
        if parts[0] != root:
            raise ValueError("backup archive must contain one directory root")
        yield member
    if root is None:
        raise ValueError("backup archive is empty")


def verify_directory_archive(archive_path: Path) -> int:
    count = 0
    with _stream_archive(archive_path) as archive:
        for member in _archive_members(archive):
            count += 1
            if member.isfile():
                payload = archive.extractfile(member)
                if payload is None:
                    raise ValueError("backup file has no readable payload")
                with payload:
                    while payload.read(1024 * 1024):
                        pass
    return count


def restore_directory_archive(archive_path: Path, destination: Path) -> int:
    """Stream a verified archive into a private directory, returning its file count."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent)
    )
    count = 0
    root_name = None
    try:
        with _stream_archive(archive_path) as archive:
            for member in _archive_members(archive):
                root_name = root_name or Path(member.name).parts[0]
                target = temporary / member.name
                if not target.parent.is_dir():
                    raise ValueError("backup directory entries must precede their children")
                archive.extract(member, temporary)  # noqa: S202 - only checked regular files/directories
                target.chmod(0o700 if member.isdir() else 0o600)
                count += int(member.isfile())
        if root_name is None:
            raise ValueError("backup archive has no directory root")
        os.replace(temporary / root_name, destination)
        _fsync_directory(destination.parent)
        return count
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
