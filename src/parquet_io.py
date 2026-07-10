"""Crash-safe helpers for replacing generated Parquet artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically publish on success.

    Writers may stream arbitrarily large output to the temporary path.  The
    previous artifact remains readable until the new file is fully closed and
    fsynced.  Any exception removes only the temporary file.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"output path must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_atomic(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a DataFrame without exposing a partial or missing final file."""
    with atomic_output_path(path) as temporary:
        frame.to_parquet(temporary, **kwargs)
