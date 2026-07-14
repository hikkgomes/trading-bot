"""Small filesystem helpers for runtime state files."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from src.autopilot.locking import acquire_file_update_lock


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text by replacing the target after the full payload reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        # ``fsync`` on the temporary file protects its contents; syncing the
        # containing directory makes the rename itself durable across a host
        # crash.  This matters for order intents and accounting WAL records.
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one durable JSONL record."""
    with acquire_file_update_lock(path, label="JSONL append"):
        _append_json_line_locked(path, payload)


def _append_json_line_locked(path: Path, payload: dict[str, Any]) -> None:
    """Append while the shared sibling update lock is held."""

    if path.is_symlink():
        raise ValueError(f"jsonl path must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"jsonl path must be a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        record = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        written = os.write(descriptor, record)
        if written != len(record):
            raise OSError(f"short JSONL append to {path}: wrote {written} of {len(record)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
