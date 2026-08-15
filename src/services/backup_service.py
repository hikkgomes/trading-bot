"""Scheduled PostgreSQL and Parquet backup entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
import time
from pathlib import Path

from src.services.backups import (
    BackupStore,
    PostgreSQLBackup,
    create_directory_archive,
    verify_directory_archive,
)
from src.services.config import load_platform_config


def _backup_id(kind: str, now: dt.datetime) -> str:
    return f"{kind}-{now.strftime('%Y%m%dT%H%M%SZ')}"


def create_database_backup(*, config_path: Path, now: dt.datetime | None = None) -> str:
    config = load_platform_config(config_path)
    database = PostgreSQLBackup(config.database_url())
    now = now or dt.datetime.now(dt.UTC)
    backup_id = _backup_id("postgresql", now)
    store = BackupStore(Path(config.paths["backups"]))
    with tempfile.TemporaryDirectory(prefix="platform-postgresql-backup-") as raw_directory:
        dump = database.create(Path(raw_directory) / "database.dump")
        store.create(
            backup_id=backup_id,
            created_at=now.replace(microsecond=0).isoformat(),
            files={"database.dump": dump},
        )
    store.verify(backup_id)
    return backup_id


def create_parquet_backup(*, config_path: Path, now: dt.datetime | None = None) -> str:
    config = load_platform_config(config_path)
    now = now or dt.datetime.now(dt.UTC)
    backup_id = _backup_id("parquet", now)
    store = BackupStore(Path(config.paths["backups"]))
    with tempfile.TemporaryDirectory(prefix="platform-parquet-backup-") as raw_directory:
        archive = create_directory_archive(
            Path(config.paths["parquet"]), Path(raw_directory) / "parquet.tar.gz"
        )
        verify_directory_archive(archive)
        store.create(
            backup_id=backup_id,
            created_at=now.replace(microsecond=0).isoformat(),
            files={"parquet.tar.gz": archive},
        )
    store.verify(backup_id)
    return backup_id


def verify_backups(*, config_path: Path) -> int:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    verified = 0
    for directory in sorted(store.root.iterdir() if store.root.exists() else ()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        bundle = store.verify(directory.name)
        archive = directory / "parquet.tar.gz"
        dump = directory / "database.dump"
        if archive.exists():
            verify_directory_archive(archive)
        if dump.exists():
            PostgreSQLBackup(config.database_url()).verify(dump)
        if not bundle.files:
            raise ValueError(f"backup has no content: {directory.name}")
        verified += 1
    return verified


def prune_backups(*, config_path: Path) -> tuple[str, ...]:
    config = load_platform_config(config_path)
    retention_days = int(config.backup["retention_days"])
    cutoff = time.time() - retention_days * 86_400
    return BackupStore(Path(config.paths["backups"])).prune(older_than_timestamp=cutoff)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify platform backups.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--mode", choices=("postgresql", "parquet", "verify"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "postgresql":
        result: object = create_database_backup(config_path=args.config)
    elif args.mode == "parquet":
        result = create_parquet_backup(config_path=args.config)
    else:
        result = verify_backups(config_path=args.config)
    removed = prune_backups(config_path=args.config)
    print({"ok": True, "mode": args.mode, "result": result, "removed": list(removed)})


if __name__ == "__main__":
    main()
