"""Adapters for every strategy source entering the common research queue."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.domain._codec import canonical_hash
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.coordinator import Candidate
from src.research.datasets import CandidateDatasetPlan


def provider_candidate(
    *,
    identity: str,
    version: str,
    family: str,
    product: str,
    thesis_id: str,
    lineage_id: str,
    provider: str,
    source_type: StrategySourceType,
    source_payload: Mapping[str, Any],
    dataset_snapshot_hashes: tuple[str, ...],
    submitted_at: str,
    universe: Mapping[str, Any] | None = None,
    data_requirements: Mapping[str, Any] | None = None,
    feature_graph: Mapping[str, Any] | None = None,
    position_model: Mapping[str, Any] | None = None,
    execution_preferences: Mapping[str, Any] | None = None,
    risk_policy: Mapping[str, Any] | None = None,
    validation_policy: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    dataset_bundle_id: str | None = None,
    dataset_plan: CandidateDatasetPlan | None = None,
) -> Candidate:
    definition = StrategyDefinition(
        identity=identity,
        version=version,
        family=family,
        product=product,
        universe=universe or {"dynamic": True},
        data_requirements=data_requirements or {},
        feature_graph=feature_graph or {},
        signal_model=dict(source_payload),
        position_model=position_model or {},
        execution_preferences=execution_preferences or {},
        risk_policy=risk_policy or {},
        validation_policy=validation_policy or {},
        source_type=source_type,
        source_hash=canonical_hash(source_payload),
        metadata=metadata or {},
    )
    return Candidate(
        definition=definition,
        thesis_id=thesis_id,
        lineage_id=lineage_id,
        provider=provider,
        dataset_snapshot_hashes=dataset_snapshot_hashes,
        submitted_at=submitted_at,
        metadata=metadata or {},
        dataset_bundle_id=dataset_bundle_id,
        dataset_plan=dataset_plan,
    )
