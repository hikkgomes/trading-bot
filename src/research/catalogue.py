"""Adapters that expose the registered strategy library to the unified queue."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable

from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import Candidate
from src.strategies import library  # noqa: F401
from src.strategies.registry import available, describe


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
    descriptions = describe()
    candidates: list[Candidate] = []
    for name in available():
        source_hash = "sha256:" + hashlib.sha256(name.encode()).hexdigest()
        evidence_type = (
            "btc_allocation"
            if product == "btc_accumulation"
            else "ml"
            if name.startswith("ml_")
            else "swing"
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
            source_type=StrategySourceType.REGISTERED_PYTHON,
            source_hash=source_hash,
            metadata={"description": descriptions.get(name, "")},
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
