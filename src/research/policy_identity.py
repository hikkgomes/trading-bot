"""Identity of research inference, independent of deployment-only dependencies."""

import hashlib
import sys
from importlib.metadata import version
from pathlib import Path

from src.config import PROJECT_ROOT

POLICY_SOURCES = (
    "src/research/selection.py",
    "src/research/calibration.py",
    "src/research/cohorts.py",
    "src/research/policy_approval.py",
    "src/research/policy_identity.py",
    "src/research/outcomes.py",
    "src/research/economic_series.py",
)
POLICY_DEPENDENCIES = ("numpy", "scipy", "SQLAlchemy", "psycopg")


def research_policy_engine_digest(*, root: Path = PROJECT_ROOT) -> str:
    """Bind the approved statistical procedure; live approval has its own full identity."""
    root = root.resolve()
    digest = hashlib.sha256()
    digest.update(f"python={sys.version_info[:3]}\n".encode())
    for name in POLICY_DEPENDENCIES:
        digest.update(f"{name}=={version(name)}\n".encode())
    for name in POLICY_SOURCES:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("research policy source must stay inside the checkout")
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()
