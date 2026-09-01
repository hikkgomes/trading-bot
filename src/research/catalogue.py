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
from typing import Any, cast

from src.domain._codec import canonical_hash
from src.domain.instruments import MarketType, canonical_instrument_id
from src.domain.strategies import (
    ResearchThesis,
    StrategyDefinition,
    StrategySourceType,
)
from src.research.coordinator import Candidate
from src.research.theses import StrategyThesisFactory
from src.strategies import library  # noqa: F401
from src.strategies.manifest import (
    assert_manifest_complete,
    manifest_by_name,
    manifest_description,
    manifest_source_type,
    registered_feature_contract,
)
from src.strategies.registry import available, describe, get


def registered_strategy_theses(
    *, product: str, instrument_universe: Iterable[str]
) -> dict[str, ResearchThesis]:
    universe = tuple(sorted(set(instrument_universe)))
    if not universe:
        raise ValueError("registered strategy research requires a predeclared universe")
    if product == "btc_accumulation" and universe != ("BTCUSDT",):
        raise ValueError("BTC accumulation research is restricted to BTCUSDT spot")
    assert_manifest_complete()
    manifest = manifest_by_name()
    factory = StrategyThesisFactory.default()
    return {
        name: factory.build(
            name=name,
            family=entry.family,
            product=product,
            instrument_universe=universe,
        )
        for name, entry in manifest.items()
    }


def _digest_path(path: Path, repository: Path) -> tuple[str, str]:
    return str(path.relative_to(repository)), hashlib.sha256(path.read_bytes()).hexdigest()


def _imported_strategy_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _direct_strategy_dependency_files(path: Path, repository: Path) -> dict[str, str]:
    module_files: dict[str, str] = {}
    for imported in sorted(_imported_strategy_modules(path)):
        if not (
            imported.startswith("src.strategies.indicators")
            or imported.startswith("src.features")
            or imported.startswith("src.indicators")
        ):
            continue
        dependency = repository.joinpath(*imported.split(".")).with_suffix(".py")
        if dependency.is_file():
            relative, digest = _digest_path(dependency, repository)
            module_files[relative] = digest
    return module_files


def _strategy_executable_files(strategy: object, repository: Path) -> dict[str, str]:
    module_files: dict[str, str] = {}
    module = inspect.getmodule(strategy)
    module_path = getattr(module, "__file__", None) if module is not None else None
    if module_path:
        path = Path(module_path).resolve()
        if path.is_file() and repository in path.parents:
            relative, digest = _digest_path(path, repository)
            module_files[relative] = digest
            module_files.update(_direct_strategy_dependency_files(path, repository))
    for directory in (repository / "src" / "indicators", repository / "src" / "features"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if path.is_file() and not path.is_symlink():
                relative, digest = _digest_path(path, repository)
                module_files[relative] = digest
    return module_files


def _runtime_lock(repository: Path) -> dict[str, str]:
    lock_files: dict[str, str] = {}
    for filename in ("requirements-bot.txt", "requirements-runtime.txt", "pyproject.toml"):
        path = repository / filename
        if path.is_file():
            lock_files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return lock_files


def _strategy_identity_policy(strategy: Any) -> dict[str, object]:
    configuration = strategy.default_config()
    return {
        "default_params": strategy.default_params(),
        "position_model": {"signal_timing": "next_bar", "default": configuration},
        "risk_policy": {
            "take_profit": configuration.take_profit,
            "stop_loss": configuration.stop_loss,
            "horizon_bars": configuration.horizon_bars,
        },
        "cost_model": {
            "fee_bps": configuration.fee_bps,
            "slippage_bps": configuration.slippage_bps,
            "pnl_unit": configuration.pnl_unit,
        },
    }


def _registered_strategy_identity_payload(name: str) -> dict[str, object]:
    """Build the economic behaviour identity without deployment metadata."""

    strategy = __import__("src.strategies.registry", fromlist=["get"]).get(name)
    repository = Path(__file__).resolve().parents[2]
    identity_policy = _strategy_identity_policy(strategy)
    return {
        "strategy": name,
        "module_files": _strategy_executable_files(strategy, repository),
        **identity_policy,
    }


def _registered_strategy_deployment_payload(name: str) -> dict[str, object]:
    """Build non-authoritative deployment provenance for one strategy."""

    repository = Path(__file__).resolve().parents[2]
    return {
        "strategy": name,
        "behaviour_source_hash": canonical_hash(_registered_strategy_identity_payload(name)),
        "runtime_lock": _runtime_lock(repository),
        "python": sys.version,
    }


def _repository_commit(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def registered_strategy_provenance(name: str) -> dict[str, object]:
    """Return immutable executable identity plus non-authoritative provenance."""

    repository = Path(__file__).resolve().parents[2]
    identity_payload = _registered_strategy_identity_payload(name)
    deployment_payload = _registered_strategy_deployment_payload(name)
    return {
        "source_hash": canonical_hash(identity_payload),
        "deployment_hash": canonical_hash(deployment_payload),
        "git_commit": _repository_commit(repository),
        "identity_schema": "registered_strategy_executable/v3",
    }


def registered_strategy_source_hash(name: str) -> str:
    """Hash executable strategy identity, independent of the current Git commit."""

    return str(registered_strategy_provenance(name)["source_hash"])


def registered_strategy_candidates(
    *,
    product: str,
    dataset_snapshot_hashes: Iterable[str],
    instrument_universe: Iterable[str] = ("BTCUSDT",),
    submitted_at: str | None = None,
    dataset_bundle_id: str | None = None,
    universe_snapshot_id: str | None = None,
) -> tuple[Candidate, ...]:
    """Create common-contract candidates for every registered strategy.

    Parameter search remains a research concern. This adapter prevents the
    named strategy library from being excluded merely because it is not DSL.
    """
    now = submitted_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    assert_manifest_complete()
    descriptions = describe()
    manifest = manifest_by_name()
    universe = tuple(sorted(set(instrument_universe)))
    if product == "btc_accumulation" and universe != ("BTCUSDT",):
        raise ValueError("BTC accumulation research is restricted to BTCUSDT spot")
    theses = registered_strategy_theses(product=product, instrument_universe=universe)
    candidates: list[Candidate] = []
    # Only executable registry entries enter research. The manifest is
    # descriptive and cannot manufacture a candidate for missing code.
    names = [name for name in available() if name not in {"ml_classifier", "ml_regressor"}]
    for name in names:
        source_hash = registered_strategy_source_hash(name)
        entry = manifest.get(name)
        if entry is None:
            raise ValueError(f"registered strategy is missing from the manifest: {name}")
        feature_nodes, _ = registered_feature_contract(name)
        strategy_class = cast(Any, get(name))
        parameters = strategy_class.default_params()
        market_type = MarketType.SPOT if product == "btc_accumulation" else MarketType.FUTURES
        universe_definition = {
            "type": "fixed",
            "symbols": list(universe),
            "instrument_ids": [
                canonical_instrument_id(
                    symbol,
                    market_type=market_type,
                    settlement_asset="USDT" if market_type is MarketType.FUTURES else None,
                )
                for symbol in universe
            ],
        }
        if universe_snapshot_id is not None:
            universe_definition["type"] = "point_in_time"
            universe_definition["universe_snapshot_id"] = universe_snapshot_id
        evidence_type = (
            "btc_allocation"
            if product == "btc_accumulation"
            else (entry.evidence_type if entry is not None else "swing")
        )
        definition = StrategyDefinition(
            identity=name,
            version="registered-v1",
            family=entry.family,
            product=product,
            universe=universe_definition,
            data_requirements={"closed_ohlcv_bars": True},
            feature_graph={"version": "canonical-features/v2", "required_nodes": feature_nodes},
            signal_model={
                "registered_strategy": name,
                "parameters": parameters,
                "behaviour_contract": "registered_strategy/v1",
            },
            position_model={"kind": "volatility_scaled"},
            execution_preferences={"policy": "market"},
            risk_policy={"product_policy": product},
            validation_policy={"evidence_type": evidence_type},
            # The common queue owns execution of these catalogue entries. The
            # family-specific source type remains explicit metadata until a
            # concrete evaluator is registered for that family.
            source_type=StrategySourceType.REGISTERED_PYTHON,
            source_hash=source_hash,
            metadata={
                "description": descriptions.get(name) or manifest_description(name),
                "manifest_family": entry.family,
                "canonical_source_type": manifest_source_type(name),
                "executable_registry_entry": True,
                "promotable": True,
                "source_provenance": registered_strategy_provenance(name),
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
                dataset_bundle_id=dataset_bundle_id,
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
