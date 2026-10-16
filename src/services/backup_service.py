"""Scheduled PostgreSQL and Parquet backup entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
import time
from pathlib import Path

from src.data.locking import acquire_file_update_lock
from src.services.backups import (
    BackupBundle,
    BackupStore,
    PostgreSQLBackup,
    create_directory_archive,
    rehearse_postgresql_restore,
    restore_directory_archive,
    verify_directory_archive,
    verify_postgresql_dump,
    write_postgresql_restore_receipt,
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
    store.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".postgresql-build-", dir=store.root) as raw_directory:
        dump = database.create(Path(raw_directory) / "database.dump")
        store.create(
            backup_id=backup_id,
            created_at=now.replace(microsecond=0).isoformat(),
            files={"database.dump": dump},
            move_sources=True,
        )
    store.verify(backup_id)
    return backup_id


def create_parquet_backup(*, config_path: Path, now: dt.datetime | None = None) -> str:
    config = load_platform_config(config_path)
    now = now or dt.datetime.now(dt.UTC)
    backup_id = _backup_id("parquet", now)
    store = BackupStore(Path(config.paths["backups"]))
    store.root.mkdir(parents=True, exist_ok=True)
    # Backup is read-only. Compaction must not change the data referenced by
    # the matching database snapshot; quarantine also contains audit evidence.
    with tempfile.TemporaryDirectory(prefix=".parquet-build-", dir=store.root) as raw_directory:
        archive = create_directory_archive(
            Path(config.paths["parquet"]),
            Path(raw_directory) / "parquet.tar.gz",
        )
        store.create(
            backup_id=backup_id,
            created_at=now.replace(microsecond=0).isoformat(),
            files={"parquet.tar.gz": archive},
            move_sources=True,
        )
    store.verify(backup_id)
    return backup_id


def verify_backups(*, config_path: Path) -> int:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    verified = 0
    database_bundles: list[BackupBundle] = []
    for directory in sorted(store.root.iterdir() if store.root.exists() else ()):
        if directory.name.startswith(".") or directory.is_symlink() or not directory.is_dir():
            continue
        bundle = store.verify(directory.name)
        archive = directory / "parquet.tar.gz"
        dump = directory / "database.dump"
        if "parquet.tar.gz" in bundle.files:
            verify_directory_archive(archive)
        if "database.dump" in bundle.files:
            verify_postgresql_dump(dump)
            database_bundles.append(bundle)
        if not bundle.files:
            raise ValueError(f"backup has no content: {directory.name}")
        verified += 1
    if database_bundles:
        newest = max(database_bundles, key=_backup_created_timestamp)
        dump = store.root / newest.backup_id / "database.dump"
        summary = rehearse_postgresql_restore(dump, scratch_root=store.root)
        if store.verify(newest.backup_id) != newest:
            raise ValueError("backup changed during PostgreSQL restore rehearsal")
        write_postgresql_restore_receipt(store.root, newest, summary)
    return verified


def _backup_created_timestamp(bundle: BackupBundle) -> float:
    try:
        created_at = dt.datetime.fromisoformat(bundle.created_at)
    except ValueError as exc:
        raise ValueError(f"backup has an invalid creation time: {bundle.backup_id}") from exc
    if created_at.tzinfo is None:
        raise ValueError(f"backup has no timezone-aware creation time: {bundle.backup_id}")
    return created_at.timestamp()


def prune_backups(*, config_path: Path) -> tuple[str, ...]:
    config = load_platform_config(config_path)
    retention_days = int(config.backup["retention_days"])
    recent_hours = int(config.backup.get("hourly_retention_hours", 24))
    if not 1 <= recent_hours <= retention_days * 24:
        raise ValueError("hourly backup retention must fit within the daily retention window")
    now = time.time()
    return BackupStore(Path(config.paths["backups"])).prune(
        older_than_timestamp=now - retention_days * 86_400,
        thin_before_timestamp=now - recent_hours * 3_600,
    )


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


def restore_parquet_backup(*, config_path: Path, backup_id: str, destination: Path) -> int:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    bundle = store.verify(backup_id)
    archive = Path(config.paths["backups"]) / backup_id / "parquet.tar.gz"
    if "parquet.tar.gz" not in bundle.files:
        raise ValueError(f"backup does not contain a Parquet archive: {backup_id}")
    return restore_directory_archive(archive, destination)


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
    config = load_platform_config(args.config)
    lock_target = Path(config.paths["backups"]) / "maintenance"
    with acquire_file_update_lock(lock_target, label="backup maintenance"):
        if args.mode in {"postgresql", "parquet"}:
            prune_backups(config_path=args.config)
        _run_command(args)


def _run_command(args: argparse.Namespace) -> None:
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
        if args.mode in {"postgresql", "parquet"}
        else ()
    )
    print({"ok": True, "mode": args.mode, "result": result, "removed": list(removed)})


if __name__ == "__main__":
    main()
