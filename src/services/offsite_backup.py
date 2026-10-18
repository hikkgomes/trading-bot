"""Encrypted, incremental recovery points managed by restic."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.data.atomic import write_json_atomic
from src.data.locking import acquire_file_update_lock
from src.services.backup_service import create_database_backup
from src.services.backups import BackupBundle, BackupStore, _fsync_directory, file_hash
from src.services.config import load_platform_config

_HOST = "linux-optiplex"
_TAG = "trading-platform"
_SCHEMA = "platform.offsite_backup/v1"


def offsite_status(root: Path, *, now: dt.datetime, maximum_age_seconds: int) -> dict[str, Any]:
    path = root / "offsite-latest.json"
    try:
        if path.is_symlink() or path.stat().st_size > 16_384:
            raise ValueError("invalid receipt file")
        receipt = json.loads(path.read_text())
        verified = dt.datetime.fromisoformat(receipt["verified_at"])
        age = (now - verified).total_seconds()
        full_age = (
            now - dt.datetime.fromisoformat(receipt["full_repository_read_at"])
        ).total_seconds()
        valid = (
            receipt["schema"] == _SCHEMA
            and re.fullmatch(r"[0-9a-f]{64}", receipt["snapshot_id"]) is not None
            and receipt["database_dump_roundtrip_verified"] is True
            and receipt["data_file_roundtrip_verified"] is True
            and receipt["retention_succeeded"] is True
            and receipt["repository_check"] in {"full", "sampled"}
            and 0 <= full_age <= 8 * 86400
            and 0 <= age <= maximum_age_seconds
        )
        return {"ok": valid, "age_seconds": age, "snapshot_id": receipt["snapshot_id"]}
    except (OSError, ValueError, TypeError, KeyError):
        return {"ok": False, "age_seconds": None, "snapshot_id": None}


def _restic(root: Path, *arguments: str) -> list[dict[str, Any]]:
    # Keep potentially large progress output off the service's memory and journal.
    with tempfile.TemporaryFile(mode="w+") as output, tempfile.TemporaryFile(mode="w+") as errors:
        command = ["restic"]
        if os.environ.get("RESTIC_REPOSITORY", "").startswith("rclone:"):
            command.extend(("-o", "rclone.timeout=5m"))
        command.extend(arguments)
        result = subprocess.run(
            command,
            stdout=output,
            stderr=errors,
            timeout=8 * 60 * 60,
            check=False,
        )
        if result.returncode:
            errors.seek(0)
            (root / "offsite-last-error.txt").write_text(errors.read(64 * 1024))
            raise RuntimeError(f"restic {arguments[0]} failed with exit {result.returncode}")
        output.seek(0)
        first = ""
        while character := output.read(1):
            if not character.isspace():
                first = character
                break
        output.seek(0)
        if first == "[":
            values = json.load(output)
            if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
                raise RuntimeError("restic returned an invalid JSON array")
            return values
        summaries = []
        for line in output:
            if not line.startswith("{"):
                continue
            value = json.loads(line)
            if isinstance(value, dict) and value.get("message_type") == "summary":
                summaries.append(value)
        return summaries


def _recent_verified_database_bundle(
    store: BackupStore, *, now: dt.datetime
) -> BackupBundle | None:
    candidates = []
    if not store.root.exists():
        return None
    for directory in sorted(store.root.iterdir()):
        if (
            not directory.name.startswith("postgresql-")
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            continue
        try:
            bundle = store._read_bundle(directory.name)
            created_at = dt.datetime.fromisoformat(bundle.created_at)
        except (OSError, ValueError, TypeError):
            continue
        if tuple(sorted(bundle.files)) != ("database.dump",) or created_at.tzinfo is None:
            continue
        age = (now - created_at.astimezone(dt.UTC)).total_seconds()
        if 0 <= age <= 3600:
            candidates.append((created_at, bundle.backup_id))
    for _created_at, backup_id in sorted(candidates, reverse=True):
        try:
            return store.verify(backup_id)
        except (OSError, ValueError):
            continue
    return None


def _pin_database(
    config_path: Path, root: Path, *, now: dt.datetime | None = None
) -> tuple[Path, dict[str, Any]]:
    store = BackupStore(root)
    if now is not None and now.tzinfo is None:
        raise ValueError("offsite database selection time must include a timezone")
    with acquire_file_update_lock(
        root / "maintenance-postgresql", label="offsite database capture"
    ):
        current = now or dt.datetime.now(dt.UTC)
        bundle = _recent_verified_database_bundle(store, now=current.astimezone(dt.UTC))
        if bundle is None:
            identity = create_database_backup(config_path=config_path)
            bundle = store.verify(identity)
        if tuple(sorted(bundle.files)) != ("database.dump",):
            raise ValueError("offsite database input must contain only database.dump")
        pinned = root / ".offsite-source"
        pinned.mkdir(mode=0o700, exist_ok=True)
        temporary = pinned / "database.dump.new"
        temporary.unlink(missing_ok=True)
        os.link(root / bundle.backup_id / "database.dump", temporary)
        os.replace(temporary, pinned / "database.dump")
        manifest = {
            "backup_id": bundle.backup_id,
            "created_at": bundle.created_at,
            "files": bundle.files,
        }
        write_json_atomic(pinned / "manifest.json", manifest)
    return pinned, manifest


def _verify_snapshot(root: Path, snapshot: str, files: dict[Path, str], *, full: bool) -> None:
    check = "--read-data" if full else f"--read-data-subset={dt.datetime.now(dt.UTC).day % 7 + 1}/7"
    _restic(root, "check", check)
    with tempfile.TemporaryDirectory(prefix=".offsite-restore-", dir=root) as temporary:
        destination = Path(temporary)
        include = [argument for source in files for argument in ("--include", str(source))]
        _restic(root, "restore", snapshot, "--target", str(destination), *include)
        for source, expected_hash in files.items():
            restored = destination.joinpath(*source.parts[1:])
            if file_hash(restored) != expected_hash:
                raise RuntimeError("cloud restore differs from the captured backup input")


def _pin_data_file(data_root: Path, pinned: Path) -> Path:
    source = next(
        (p for p in data_root.rglob("*.parquet") if p.is_file() and not p.is_symlink()), None
    )
    if source is None:
        raise ValueError("offsite recovery point requires a Parquet data file")
    # Read one inode: live writers replace partitions, and backups may use a different disk.
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    temporary = None
    try:
        with (
            os.fdopen(descriptor, "rb") as handle,
            tempfile.NamedTemporaryFile(dir=pinned, delete=False) as target,
        ):
            temporary = Path(target.name)
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("backup data probe must be a regular file")
            shutil.copyfileobj(handle, target)
            target.flush()
            os.fsync(target.fileno())
        destination = pinned / "data.parquet"
        os.replace(temporary, destination)
        _fsync_directory(pinned)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_offsite_backup(config_path: Path) -> dict[str, Any]:
    for name in ("RESTIC_REPOSITORY", "RESTIC_PASSWORD_FILE", "RCLONE_CONFIG"):
        if not os.environ.get(name):
            raise ValueError(f"offsite backup requires {name}")
    config = load_platform_config(config_path)
    root = Path(config.paths["backups"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (
        acquire_file_update_lock(root / "maintenance-backup", label="backup operation"),
        acquire_file_update_lock(root / "maintenance-offsite", label="offsite backup"),
    ):
        return _capture_offsite(config_path, config, root)


def _capture_offsite(config_path: Path, config: Any, root: Path) -> dict[str, Any]:
    _restic(root, "snapshots", "--json")
    now = dt.datetime.now(dt.UTC)
    previous = offsite_status(root, now=now, maximum_age_seconds=7 * 86400)
    prior = json.loads((root / "offsite-latest.json").read_text()) if previous["ok"] else {}
    last_full = dt.datetime.fromisoformat(
        prior.get("full_repository_read_at", "1970-01-01T00:00:00+00:00")
    )
    full = (now - last_full).total_seconds() >= 7 * 86400
    pinned, manifest = _pin_database(config_path, root)
    data_root = Path(config.paths["parquet"]).resolve()
    data_probe = _pin_data_file(data_root, pinned)
    restored_files = {
        pinned / "database.dump": manifest["files"]["database.dump"],
        data_probe: file_hash(data_probe),
    }
    artefacts = Path(config.paths["artefacts"]).resolve()
    extra = [str(artefacts)] if not artefacts.is_relative_to(data_root) else []
    quarantine = config_path.resolve().parent.parent / "runtime" / "quarantine"
    if quarantine.is_dir():
        extra.append(str(quarantine))
    summaries = _restic(
        root,
        "backup",
        "--json",
        "--host",
        _HOST,
        "--tag",
        _TAG,
        "--read-concurrency",
        "1",
        "--exclude",
        str(data_root / "**" / ".*"),
        "--exclude",
        "*.tmp",
        "--exclude",
        "*.lock",
        str(data_root),
        str(pinned),
        str(config_path.parent.resolve()),
        *extra,
    )
    snapshot = summaries[-1].get("snapshot_id", "") if summaries else ""
    if not re.fullmatch(r"[0-9a-f]{64}", snapshot):
        raise RuntimeError("restic did not confirm a complete snapshot")
    _verify_snapshot(root, snapshot, restored_files, full=full)
    verified = dt.datetime.now(dt.UTC).isoformat()
    receipt = {
        "schema": _SCHEMA,
        "snapshot_id": snapshot,
        "database_backup": manifest,
        "database_dump_roundtrip_verified": True,
        "data_file_roundtrip_verified": True,
        "restored_file_hashes": {str(path): digest for path, digest in restored_files.items()},
        "repository_check": "full" if full else "sampled",
        "full_repository_read_at": verified if full else last_full.isoformat(),
        "verified_at": verified,
        "retention_succeeded": False,
    }
    write_json_atomic(root / "offsite-latest.json", receipt)
    _restic(
        root,
        "forget",
        "--host",
        _HOST,
        "--tag",
        _TAG,
        "--group-by",
        "host,tags",
        "--keep-last",
        "1",
        "--keep-hourly",
        str(config.backup["hourly_retention_hours"]),
        "--keep-daily",
        str(config.backup["retention_days"]),
        "--prune",
    )
    receipt["retention_succeeded"] = True
    write_json_atomic(root / "offsite-latest.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and verify an encrypted offsite backup.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    args = parser.parse_args()
    print(json.dumps(run_offsite_backup(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
