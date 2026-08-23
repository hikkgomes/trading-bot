"""Adapters that expose the registered strategy library to the unified queue."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import Candidate
from src.strategies import library  # noqa: F401
from src.strategies.manifest import (
    assert_manifest_complete,
    manifest_by_name,
    manifest_description,
    manifest_source_type,
)
from src.strategies.registry import available, describe


def registered_strategy_source_hash(name: str) -> str:
    """Hash executable strategy provenance, not its display name.

    The hash includes the registered class source, imported local indicator and
    feature modules, defaults, runtime lock, Python version, and Git commit.
    A source change therefore produces a new strategy identity and cannot
    silently reuse approval or evidence from an older implementation.
    """

    strategy = __import__("src.strategies.registry", fromlist=["get"]).get(name)
    repository = Path(__file__).resolve().parents[2]
    module_files: dict[str, str] = {}
    module = inspect.getmodule(strategy)
    if module is not None:
        module_path = getattr(module, "__file__", None)
        if module_path:
            path = Path(module_path).resolve()
            if path.is_file() and repository in path.parents:
                module_files[str(path.relative_to(repository))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for directory in (repository / "src" / "indicators", repository / "src" / "features"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*.py")):
                if path.is_file() and not path.is_symlink():
                    module_files[str(path.relative_to(repository))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    lock_files = {}
    for filename in ("requirements-bot.txt", "requirements-runtime.txt", "pyproject.toml"):
        path = repository / filename
        if path.is_file():
            lock_files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "strategy": name,
        "module_files": module_files,
        "default_params": strategy.default_params(),
        "default_config": strategy.default_config(),
        "runtime_lock": lock_files,
        "python": sys.version,
        "git_commit": commit,
    }
    return "sha256:" + hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def registered_strategy_candidates(
    *,
    product: str,
    dataset_snapshot_hashes: Iterable[str],
) -> tuple[Candidate, ...]:
    """Create common-contract candidates for every registered strategy.

    Parameter search remains a research concern. This adapter prevents the
    named strategy library from being excluded merely because it is not DSL.
    """
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    assert_manifest_complete()
    descriptions = describe()
    manifest = manifest_by_name()
    candidates: list[Candidate] = []
    names = sorted(set(available()) | set(manifest))
    for name in names:
        source_hash = (
            registered_strategy_source_hash(name)
            if name in available()
            else "sha256:"
            + hashlib.sha256(canonical_manifest_payload(name).encode("utf-8")).hexdigest()
        )
        entry = manifest.get(name)
        evidence_type = (
            "btc_allocation"
            if product == "btc_accumulation"
            else (entry.evidence_type if entry is not None else "swing")
        )
        definition = StrategyDefinition(
            identity=name,
            version="registered-v1",
            family="registered",
            product=product,
            universe={"dynamic": True},
            data_requirements={},
            feature_graph={},
            signal_model={"registered_strategy": name},
            position_model={},
            execution_preferences={},
            risk_policy={},
            validation_policy={"evidence_type": evidence_type},
            # The common queue owns execution of these catalogue entries. The
            # family-specific source type remains explicit metadata until a
            # concrete evaluator is registered for that family.
            source_type=StrategySourceType.REGISTERED_PYTHON,
            source_hash=source_hash,
            metadata={
                "description": descriptions.get(name) or manifest_description(name),
                "manifest_family": entry.family if entry is not None else "supplemental",
                "canonical_source_type": manifest_source_type(name),
                "executable_registry_entry": name in available(),
            },
        )
        candidates.append(
            Candidate(
                definition=definition,
                provider="registered_strategy_catalogue",
                dataset_snapshot_hashes=tuple(dataset_snapshot_hashes),
                submitted_at=now,
            )
        )
    return tuple(candidates)


def canonical_manifest_payload(name: str) -> str:
    """Return stable provenance for manifest-only strategies without code."""

    entry = manifest_by_name().get(name)
    return repr(
        {
            "name": name,
            "family": entry.family if entry else "supplemental",
            "evidence_type": entry.evidence_type if entry else "swing",
            "python": sys.version,
        }
    )
