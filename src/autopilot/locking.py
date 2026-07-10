"""Shared process lock for runtime cycles and operator state transitions."""

from __future__ import annotations

import datetime as dt
import fcntl
import os
import stat
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


@contextmanager
def acquire_file_update_lock(target: Path, *, label: str):
    """Take a blocking sibling lock for an atomic read-modify-write transaction."""

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    if lock_path.is_symlink():
        raise RuntimeError(f"{label} lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open {label} lock {lock_path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"{label} lock must be a regular file: {lock_path}")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def acquire_runtime_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"runtime lock must not be a symlink: {path}")
    handle = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"autopilot already running; lock is held: {path}") from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at={_utc_now()}\n")
        handle.flush()
        yield handle
    finally:
        if acquired:
            _release_runtime_lock(handle)
        else:
            handle.close()


def _release_runtime_lock(handle: TextIOWrapper) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
