"""Private per-process cache directories for research commands.

Research entry points import plotting and model libraries that may need a
writable cache even when the service account has no writable home directory.
Predictable shared paths under ``/tmp`` permit cross-user symlink and cache
poisoning attacks, so missing cache variables are backed by one mode-0700
temporary directory owned by this process.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_PROCESS_CACHE: tempfile.TemporaryDirectory[str] | None = None


def configure_private_process_cache() -> None:
    """Set missing plotting/cache variables to a private temporary root."""

    global _PROCESS_CACHE
    if "MPLCONFIGDIR" in os.environ and "XDG_CACHE_HOME" in os.environ:
        return
    if _PROCESS_CACHE is None:
        _PROCESS_CACHE = tempfile.TemporaryDirectory(prefix="trading-bot-cache-")
    root = Path(_PROCESS_CACHE.name)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "xdg"))
