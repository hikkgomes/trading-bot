"""Content-verified backup bundles and restore checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"backup input must be a regular file: {path}")
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


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
                if (
                    not logical_name
                    or Path(logical_name).is_absolute()
                    or ".." in Path(logical_name).parts
                ):
                    raise ValueError("backup logical names must be safe relative paths")
                digest = file_hash(source)
                target = temporary / logical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_hash(target) != digest:
                    raise RuntimeError(f"backup verification failed: {logical_name}")
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
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return bundle

    def verify(self, backup_id: str) -> BackupBundle:
        directory = self.root / backup_id
        payload: Any = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
            raise ValueError("backup manifest is invalid")
        bundle = BackupBundle(
            backup_id=str(payload.get("backup_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            files={str(key): str(value) for key, value in payload["files"].items()},
        )
        if bundle.backup_id != backup_id:
            raise ValueError("backup manifest ID does not match its directory")
        for logical_name, expected_hash in bundle.files.items():
            if file_hash(directory / logical_name) != expected_hash:
                raise ValueError(f"backup content hash does not match: {logical_name}")
        return bundle

    def restore(self, *, backup_id: str, destination: Path) -> tuple[Path, ...]:
        bundle = self.verify(backup_id)
        source_root = self.root / backup_id
        restored: list[Path] = []
        for logical_name, expected_hash in sorted(bundle.files.items()):
            target = destination / logical_name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.restore.tmp")
            shutil.copyfile(source_root / logical_name, temporary)
            if file_hash(temporary) != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"restore verification failed: {logical_name}")
            os.replace(temporary, target)
            restored.append(target)
        return tuple(restored)

    def prune(self, *, older_than_timestamp: float) -> tuple[str, ...]:
        removed: list[str] = []
        if not self.root.exists():
            return ()
        for directory in sorted(self.root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            manifest = directory / "manifest.json"
            if not manifest.is_file() or manifest.stat().st_mtime >= older_than_timestamp:
                continue
            self.verify(directory.name)
            shutil.rmtree(directory)
            removed.append(directory.name)
        return tuple(removed)


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
    source = source.resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a regular directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(source, arcname=source.name, recursive=True)
        with tarfile.open(temporary, "r:gz") as archive:
            for member in archive.getmembers():
                if member.islnk() or member.issym() or member.name.startswith("/"):
                    raise ValueError("backup archive contains an unsafe member")
                if ".." in Path(member.name).parts:
                    raise ValueError("backup archive escapes its restore root")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_directory_archive(archive_path: Path) -> int:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("backup archive must be a regular file")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.islnk() or member.issym() or member.name.startswith("/"):
                raise ValueError("backup archive contains an unsafe member")
            if ".." in Path(member.name).parts:
                raise ValueError("backup archive escapes its restore root")
            if not (member.isfile() or member.isdir()):
                raise ValueError("backup archive contains a non-file member")
        with tempfile.TemporaryDirectory(prefix="platform-restore-check-") as raw_directory:
            archive.extractall(Path(raw_directory))  # noqa: S202 - members are validated above
    return len(members)
