"""Compact valid Parquet inputs without deleting the source evidence first."""

from __future__ import annotations

import argparse
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.parquet_store import _fsync_directory
from src.domain._codec import canonical_hash


class StorageCompactionError(RuntimeError):
    """A partition could not be compacted safely."""


def compact_data_root(
    root: Path,
    *,
    quarantine_root: Path | None = None,
    maximum_rows: int = 5_000,
    quarantine_retention_seconds: int = 7 * 86_400,
    quarantine_max_bytes: int = 10 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Merge small files by partition and quarantine superseded files."""

    if maximum_rows <= 0 or quarantine_retention_seconds <= 0 or quarantine_max_bytes <= 0:
        raise ValueError("storage compaction bounds must be positive")
    root = root.resolve()
    quarantine = (quarantine_root or root.parent / "runtime" / "quarantine" / "storage").resolve()
    groups = _partition_groups(root)
    compacted = 0
    quarantined = 0
    rows_total = 0
    outputs: list[str] = []
    for directory, paths in sorted(groups.items(), key=lambda item: str(item[0])):
        rows = _read_partition(paths)
        if not rows:
            continue
        destinations = _write_chunks(directory, _deduplicate(rows), maximum_rows)
        outputs.extend(str(path) for path in destinations)
        compacted += len(destinations)
        rows_total += len(rows)
        # A destination is fsynced before any source is moved. The active tree
        # therefore always contains either the old valid files or the new
        # content-addressed partition.
        quarantined += _quarantine(
            [path for path in paths if path not in destinations],
            root=root,
            quarantine=quarantine,
        )
    quarantine_roots = tuple(dict.fromkeys((quarantine, root / "quarantine")))
    pruned_files = 0
    pruned_bytes = 0
    for quarantine_path in quarantine_roots:
        removed_files, removed_bytes = _prune_quarantine(
            quarantine_path,
            retention_seconds=quarantine_retention_seconds,
            maximum_bytes=quarantine_max_bytes,
        )
        pruned_files += removed_files
        pruned_bytes += removed_bytes
    quarantine_files = 0
    quarantine_bytes = 0
    for quarantine_path in quarantine_roots:
        files, bytes_used = _quarantine_metrics(quarantine_path)
        quarantine_files += files
        quarantine_bytes += bytes_used
    return {
        "schema": "platform.storage_compaction/v1",
        "partitions": compacted,
        "rows": rows_total,
        "quarantined_files": quarantined,
        "quarantine_files": quarantine_files,
        "quarantine_bytes": quarantine_bytes,
        "pruned_quarantine_files": pruned_files,
        "pruned_quarantine_bytes": pruned_bytes,
        "outputs": outputs,
    }


def _partition_groups(root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for kind in ("raw", "bars", "features"):
        base = root / kind
        if not base.is_dir():
            continue
        for path in base.rglob("*.parquet"):
            if path.is_file() and not path.is_symlink():
                groups[path.parent].append(path)
    return groups


def _read_partition(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            values = pq.ParquetFile(path).read().to_pylist()
        except Exception as exc:
            raise StorageCompactionError(f"cannot read {path}") from exc
        rows.extend(dict(row) for row in values if isinstance(row, dict))
    return rows


def _write_chunks(
    directory: Path, rows: list[dict[str, Any]], maximum_rows: int
) -> tuple[Path, ...]:
    destinations: list[Path] = []
    for offset in range(0, len(rows), maximum_rows):
        chunk = tuple(rows[offset : offset + maximum_rows])
        digest = canonical_hash(chunk).removeprefix("sha256:")
        destination = directory / f"compact-{digest}.parquet"
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise StorageCompactionError(
                    f"compact destination is not a regular file: {destination}"
                )
            existing_digest = canonical_hash(
                tuple(dict(row) for row in pq.ParquetFile(destination).read().to_pylist())
            ).removeprefix("sha256:")
            if existing_digest != digest:
                raise StorageCompactionError(f"compact destination hash collision: {destination}")
        else:
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            keys = sorted({key for row in chunk for key in row})
            table = pa.table({key: [row.get(key) for row in chunk] for key in keys})
            try:
                pq.write_table(table, temporary, compression="zstd", row_group_size=maximum_rows)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        destinations.append(destination)
    return tuple(destinations)


def _quarantine(paths: list[Path], *, root: Path, quarantine: Path) -> int:
    quarantine.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in paths:
        target = quarantine / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.name}.{uuid.uuid4().hex}")
        os.replace(path, target)
        count += 1
    if count:
        _fsync_directory(quarantine)
    return count


def _quarantine_metrics(quarantine: Path) -> tuple[int, int]:
    files = (
        [path for path in quarantine.rglob("*") if path.is_file() and not path.is_symlink()]
        if quarantine.is_dir()
        else []
    )
    return len(files), sum(path.stat().st_size for path in files)


def _prune_quarantine(
    quarantine: Path, *, retention_seconds: int, maximum_bytes: int
) -> tuple[int, int]:
    if not quarantine.is_dir():
        return 0, 0
    now = time.time()
    files = [path for path in quarantine.rglob("*") if path.is_file() and not path.is_symlink()]
    files.sort(key=lambda path: (path.stat().st_mtime, str(path)))
    total = sum(path.stat().st_size for path in files)
    removed_files = 0
    removed_bytes = 0
    for path in files:
        age_expired = now - path.stat().st_mtime > retention_seconds
        over_limit = total > maximum_bytes
        if not age_expired and not over_limit:
            continue
        size = path.stat().st_size
        path.unlink()
        total -= size
        removed_files += 1
        removed_bytes += size
    for directory in sorted(
        (path for path in quarantine.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed_files, removed_bytes


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("event_id", "feature_id", "open_time_ms", "close_time_ms", "source_close_time")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = next((str(row[key]) for key in keys if row.get(key) is not None), None)
        identity = identity or canonical_hash(row)
        result.setdefault(identity, row)
    return sorted(result.values(), key=lambda row: tuple(str(row.get(key) or "") for key in keys))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compact platform Parquet data.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--quarantine-root", type=Path)
    parser.add_argument("--maximum-rows", type=int, default=5_000)
    parser.add_argument("--quarantine-retention-seconds", type=int, default=7 * 86_400)
    parser.add_argument("--quarantine-max-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    args = parser.parse_args(argv)
    import json

    print(
        json.dumps(
            compact_data_root(
                args.root,
                quarantine_root=args.quarantine_root,
                maximum_rows=args.maximum_rows,
                quarantine_retention_seconds=args.quarantine_retention_seconds,
                quarantine_max_bytes=args.quarantine_max_bytes,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
