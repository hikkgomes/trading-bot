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
    _backup_directories,
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
    return backup_id


def verify_backups(*, config_path: Path) -> int:
    config = load_platform_config(config_path)
    root = Path(config.paths["backups"])
    store = BackupStore(root)
    verified_ids: set[str] = set()
    database_bundles: list[BackupBundle] = []
    for kind, logical_name in (
        ("postgresql", "database.dump"),
        ("parquet", "parquet.tar.gz"),
    ):
        lock_target = root / f"maintenance-{kind}"
        with acquire_file_update_lock(lock_target, label=f"{kind} backup verification"):
            for directory in _backup_directories(root, kind=kind):
                bundle = store._read_bundle(directory.name)
                if tuple(sorted(bundle.files)) != (logical_name,):
                    raise ValueError(f"backup kind does not match its ID: {directory.name}")
                store.verify(bundle.backup_id)
                if kind == "parquet":
                    verify_directory_archive(directory / logical_name)
                else:
                    verify_postgresql_dump(directory / logical_name)
                    database_bundles.append(bundle)
                verified_ids.add(bundle.backup_id)
            if kind == "postgresql" and database_bundles:
                newest = max(database_bundles, key=_backup_created_timestamp)
                dump = root / newest.backup_id / "database.dump"
                summary = rehearse_postgresql_restore(dump, scratch_root=root)
                if store.verify(newest.backup_id) != newest:
                    raise ValueError("backup changed during PostgreSQL restore rehearsal")
                write_postgresql_restore_receipt(root, newest, summary)
    return len(verified_ids)


def _backup_created_timestamp(bundle: BackupBundle) -> float:
    try:
        created_at = dt.datetime.fromisoformat(bundle.created_at)
    except ValueError as exc:
        raise ValueError(f"backup has an invalid creation time: {bundle.backup_id}") from exc
    if created_at.tzinfo is None:
        raise ValueError(f"backup has no timezone-aware creation time: {bundle.backup_id}")
    return created_at.timestamp()


def prune_backups(*, config_path: Path, logical_name: str | None = None) -> tuple[str, ...]:
    config = load_platform_config(config_path)
    retention_days = int(config.backup["retention_days"])
    recent_hours = int(config.backup.get("hourly_retention_hours", 24))
    if not 1 <= recent_hours <= retention_days * 24:
        raise ValueError("hourly backup retention must fit within the daily retention window")
    now = time.time()
    return BackupStore(Path(config.paths["backups"])).prune(
        older_than_timestamp=now - retention_days * 86_400,
        thin_before_timestamp=now - recent_hours * 3_600,
        logical_name=logical_name,
    )


def restore_postgresql_backup(
    *, config_path: Path, backup_id: str, target_database_url: str | None = None
) -> str:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    bundle = store._read_bundle(backup_id)
    if tuple(sorted(bundle.files)) != ("database.dump",):
        raise ValueError(f"backup does not contain only a PostgreSQL dump: {backup_id}")
    store.verify(backup_id)
    dump = Path(config.paths["backups"]) / backup_id / "database.dump"
    PostgreSQLBackup(target_database_url or config.database_url()).restore(dump)
    return backup_id


def restore_parquet_backup(*, config_path: Path, backup_id: str, destination: Path) -> int:
    config = load_platform_config(config_path)
    store = BackupStore(Path(config.paths["backups"]))
    bundle = store._read_bundle(backup_id)
    if tuple(sorted(bundle.files)) != ("parquet.tar.gz",):
        raise ValueError(f"backup does not contain only a Parquet archive: {backup_id}")
    store.verify(backup_id)
    archive = Path(config.paths["backups"]) / backup_id / "parquet.tar.gz"
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
    with acquire_file_update_lock(
        Path(config.paths["backups"]) / "maintenance-backup", label="backup operation"
    ):
        _run_maintenance(args, config)


def _run_maintenance(args: argparse.Namespace, config) -> None:
    if args.mode == "verify":
        _run_command(args)
        return
    kind = (
        "postgresql" if args.mode.endswith("postgresql") or args.mode == "postgresql" else "parquet"
    )
    lock_target = Path(config.paths["backups"]) / f"maintenance-{kind}"
    with acquire_file_update_lock(lock_target, label=f"{kind} backup maintenance"):
        discarded: tuple[str, ...] = ()
        removed: tuple[str, ...] = ()
        if args.mode in {"postgresql", "parquet"}:
            store = BackupStore(lock_target.parent)
            discarded = store.discard_abandoned_builds(kind=kind)
            logical_name = "database.dump" if kind == "postgresql" else "parquet.tar.gz"
            removed = prune_backups(config_path=args.config, logical_name=logical_name)
        _run_command(args, discarded_temporary=discarded, removed_backups=removed)


def _run_command(
    args: argparse.Namespace,
    *,
    discarded_temporary: tuple[str, ...] = (),
    removed_backups: tuple[str, ...] = (),
) -> None:
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
    print(
        {
            "ok": True,
            "mode": args.mode,
            "result": result,
            "removed": list(removed_backups),
            "discarded_temporary_builds": list(discarded_temporary),
        }
    )


if __name__ == "__main__":
    main()
