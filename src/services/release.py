"""Observed source identity for service receipts, without reading secrets."""

import hashlib
import subprocess
from functools import cache
from pathlib import Path


def _source_hash(root: Path) -> str:
    files = set(root.glob("requirements*.txt")) | {root / "Makefile", root / "pyproject.toml"}
    for directory in ("src", "config", "alembic", "deploy", "scripts"):
        files.update(
            path
            for path in (root / directory).rglob("*")
            if path.suffix
            in {
                ".py",
                ".json",
                ".sh",
                ".service",
                ".timer",
                ".slice",
                ".toml",
            }
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("release source must remain within the checkout")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


@cache
def observed_release() -> dict[str, str | bool | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        source_hash = _source_hash(root)
    except (OSError, ValueError):
        source_hash = None
    git = ["git", "-c", f"safe.directory={root}"]
    try:
        revision = subprocess.check_output(
            [*git, "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = subprocess.check_output(
            [
                *git,
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src",
                "config",
                "alembic",
                "deploy",
                "scripts",
                "Makefile",
                "pyproject.toml",
                "requirements*.txt",
            ],
            cwd=root,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "source_modified": None, "source_hash": source_hash}
    return {
        "revision": revision,
        "source_modified": bool(dirty.strip()),
        "source_hash": source_hash,
    }
