"""Deterministic identity for the code and dependencies that can affect execution."""

from __future__ import annotations

import hashlib
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

from src.config import PROJECT_ROOT


def _execution_source_paths(root: Path) -> list[Path]:
    paths = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("research_exploration").rglob("*.py"),
        root / "build_binance_indicator_dataset.py",
        root / "requirements-bot.txt",
    ]
    return sorted({path for path in paths if path.exists()}, key=lambda path: path.as_posix())


def _installed_distribution_versions() -> tuple[tuple[str, str], ...]:
    """Return the complete installed environment in deterministic form.

    The runtime lock pins transitive dependencies, not just top-level packages.
    Hashing the complete environment also fails closed if an operator adds or
    replaces a low-level networking/crypto package without updating the lock.
    """

    installed: dict[str, str] = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip() or not raw_version:
            raise RuntimeError("installed distribution has incomplete package metadata")
        name = re.sub(r"[-_.]+", "-", raw_name.strip()).lower()
        version_value = str(raw_version).strip()
        previous = installed.get(name)
        if previous is not None and previous != version_value:
            raise RuntimeError(
                f"multiple installed versions found for {name}: {previous}, {version_value}"
            )
        installed[name] = version_value
    if not installed:
        raise RuntimeError("execution identity found no installed Python distributions")
    return tuple(sorted(installed.items()))


def execution_engine_digest(*, root: Path = PROJECT_ROOT) -> str:
    """Hash the execution-capable source tree and installed runtime versions.

    Hashing the broad source surface is intentionally conservative: changing a
    helper that later becomes part of signal construction cannot silently reuse
    an older human approval. Symlinks are rejected so the identity never follows
    code outside the reviewed checkout.
    """

    root = root.resolve()
    paths = _execution_source_paths(root)
    if not paths:
        raise RuntimeError(f"execution identity found no source files under {root}")
    digest = hashlib.sha256()
    digest.update(
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n".encode()
    )
    for distribution, installed_version in _installed_distribution_versions():
        digest.update(f"dependency={distribution}=={installed_version}\n".encode())
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"execution identity source must not be a symlink: {path}")
        try:
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read execution identity source {path}: {exc}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    # File modes and mtimes do not affect behavior. Environment/package
    # versions above cover the runtime.
    return f"sha256:{digest.hexdigest()}"
