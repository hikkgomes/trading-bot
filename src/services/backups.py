"""Content-verified backup bundles and restore checks."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
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


def _safe_logical_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("backup logical names must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value == "manifest.json":
        raise ValueError("backup logical names must be safe relative paths")
    return value


def _safe_backup_id(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("backup IDs must be safe directory names")
    return value


@dataclass(frozen=True)
class BackupBundle:
    backup_id: str
    created_at: str
    files: dict[str, str]


class BackupStore:
    def __init__(self, root: Path):
        self.root = root

    def create(
        self,
        *,
        backup_id: str,
        created_at: str,
        files: dict[str, Path],
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
        try:
            for logical_name, source in sorted(files.items()):
                logical_name = _safe_logical_name(logical_name)
                digest = file_hash(source)
                target = temporary / logical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_hash(target) != digest:
                    raise RuntimeError(f"backup verification failed: {logical_name}")
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
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
        self, *, older_than_timestamp: float, thin_before_timestamp: float | None = None
    ) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        bundles = [
            self._read_bundle(directory.name)
            for directory in self.root.iterdir()
            if directory.is_dir()
            and not directory.is_symlink()
            and (directory / "manifest.json").is_file()
        ]
        removed, latest = _retention_plan(bundles, older_than_timestamp, thin_before_timestamp)
        # Never delete an earlier recovery point unless its replacement verifies.
        for backup_id in latest:
            self.verify(backup_id)
        for backup_id in removed:
            self.verify(backup_id)
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
        if dump_path.is_symlink() or not dump_path.is_file():
            raise ValueError("PostgreSQL dump must be a regular file")
        subprocess.run(
            ["pg_restore", "--list", str(dump_path)],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )

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
        subprocess.run(
            ["tar", "-czf", str(temporary), "-C", str(source.parent), "--", source.name],
            check=True,
            capture_output=True,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        verify_directory_archive(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


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
