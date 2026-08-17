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
    restore_directory_archive,
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


def restore_postgresql_backup(
    *, config_path: Path, backup_id: str, target_database_url: str | None = None
) -> str:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    bundle = store.verify(backup_id)
    dump = Path(config.paths["backups"]) / backup_id / "database.dump"
    if "database.dump" not in bundle.files:
        raise ValueError(f"backup does not contain a PostgreSQL dump: {backup_id}")
    PostgreSQLBackup(target_database_url or config.database_url()).restore(dump)
    return backup_id


def restore_parquet_backup(
    *, config_path: Path, backup_id: str, destination: Path
) -> tuple[str, ...]:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    bundle = store.verify(backup_id)
    archive = Path(config.paths["backups"]) / backup_id / "parquet.tar.gz"
    if "parquet.tar.gz" not in bundle.files:
        raise ValueError(f"backup does not contain a Parquet archive: {backup_id}")
    restored = restore_directory_archive(archive, destination)
    return tuple(str(path) for path in restored)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create, verify, or restore platform backups.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument(
        "--mode",
        choices=("postgresql", "parquet", "verify", "restore-postgresql", "restore-parquet"),
        required=True,
    )
    parser.add_argument("--backup-id")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--target-database-url")
    parser.add_argument("--confirm-restore", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "postgresql":
        result: object = create_database_backup(config_path=args.config)
    elif args.mode == "parquet":
        result = create_parquet_backup(config_path=args.config)
    elif args.mode == "verify":
        result = verify_backups(config_path=args.config)
    elif args.mode == "restore-postgresql":
        if not args.confirm_restore or not args.backup_id:
            raise SystemExit("restore-postgresql requires --backup-id and --confirm-restore")
        result = restore_postgresql_backup(
            config_path=args.config,
            backup_id=args.backup_id,
            target_database_url=args.target_database_url,
        )
    else:
        if not args.confirm_restore or not args.backup_id or args.destination is None:
            raise SystemExit(
                "restore-parquet requires --backup-id, --destination, and --confirm-restore"
            )
        result = restore_parquet_backup(
            config_path=args.config,
            backup_id=args.backup_id,
            destination=args.destination,
        )
    removed = (
        prune_backups(config_path=args.config)
        if args.mode in {"postgresql", "parquet", "verify"}
        else ()
    )
    print({"ok": True, "mode": args.mode, "result": result, "removed": list(removed)})


if __name__ == "__main__":
    main()
