"""Adapters that expose the registered strategy library to the unified queue."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import inspect
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from src.domain._codec import canonical_hash
from src.domain.strategies import (
    MechanismCategory,
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.coordinator import Candidate
from src.research.theses import REQUIRED_NEGATIVE_CONTROLS
from src.strategies import library  # noqa: F401
from src.strategies.manifest import (
    assert_manifest_complete,
    manifest_by_name,
    manifest_description,
    manifest_source_type,
)
from src.strategies.registry import available, describe


def registered_strategy_theses(
    *, product: str, instrument_universe: Iterable[str]
) -> dict[str, ResearchThesis]:
    universe = tuple(sorted(set(instrument_universe)))
    if not universe:
        raise ValueError("registered strategy research requires a predeclared universe")
    assert_manifest_complete()
    manifest = manifest_by_name()
    return {
        name: ResearchThesis(
            mechanism_category=MechanismCategory.BEHAVIOURAL,
            market_rationale=f"Test the predeclared {name} mechanism without result-led changes.",
            expected_causal_chain=("observable market state", name, "subsequent return"),
            expected_direction="as declared by the strategy signal",
            expected_horizon="strategy validation horizon",
            required_data=("canonical closed bars",),
            permitted_features=(name,),
            instrument_universe=universe,
            generalisation_scope={"product": product},
            failure_regimes=("structural break", "insufficient liquidity"),
            falsification_tests=("chronological holdout", "cost stress"),
            negative_controls=REQUIRED_NEGATIVE_CONTROLS,
            execution_capacity_assumptions={"market_impact_model_required": True},
            parent_thesis_ids=(),
            cumulative_trial_budget=12,
            created_at="2026-01-01T00:00:00+00:00",
            creator_identity="registered_strategy_catalogue/v1",
        )
        for name in available()
        if name not in {"ml_classifier", "ml_regressor"} and name in manifest
    }


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
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                direct_modules: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        direct_modules.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        direct_modules.add(node.module)
                for imported in sorted(direct_modules):
                    if not (
                        imported.startswith("src.strategies.indicators")
                        or imported.startswith("src.features")
                        or imported.startswith("src.indicators")
                    ):
                        continue
                    dependency = repository.joinpath(*imported.split(".")).with_suffix(".py")
                    if dependency.is_file():
                        module_files[str(dependency.relative_to(repository))] = hashlib.sha256(
                            dependency.read_bytes()
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
        "position_model": {"signal_timing": "next_bar", "default": strategy.default_config()},
        "risk_policy": {
            "take_profit": strategy.default_config().take_profit,
            "stop_loss": strategy.default_config().stop_loss,
            "horizon_bars": strategy.default_config().horizon_bars,
        },
        "cost_model": {
            "fee_bps": strategy.default_config().fee_bps,
            "slippage_bps": strategy.default_config().slippage_bps,
            "pnl_unit": strategy.default_config().pnl_unit,
        },
        "runtime_lock": lock_files,
        "python": sys.version,
        "git_commit": commit,
    }
    return "sha256:" + hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def registered_strategy_candidates(
    *,
    product: str,
    dataset_snapshot_hashes: Iterable[str],
    instrument_universe: Iterable[str] = ("BTCUSDT",),
) -> tuple[Candidate, ...]:
    """Create common-contract candidates for every registered strategy.

    Parameter search remains a research concern. This adapter prevents the
    named strategy library from being excluded merely because it is not DSL.
    """
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    assert_manifest_complete()
    descriptions = describe()
    manifest = manifest_by_name()
    theses = registered_strategy_theses(
        product=product, instrument_universe=instrument_universe
    )
    candidates: list[Candidate] = []
    # Only executable registry entries enter research. The manifest is
    # descriptive and cannot manufacture a candidate for missing code.
    names = [
        name
        for name in available()
        if name not in {"ml_classifier", "ml_regressor"}
    ]
    for name in names:
        source_hash = registered_strategy_source_hash(name)
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
                "executable_registry_entry": True,
            },
        )
        candidates.append(
            Candidate(
                definition=definition,
                thesis_id=theses[name].thesis_id,
                lineage_id=canonical_hash({"thesis_id": theses[name].thesis_id, "root": name}),
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
