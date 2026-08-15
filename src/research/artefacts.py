"""Immutable, reproducible deployable strategy artefacts."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive
from src.domain.strategies import StrategyDefinition


def _hashes(values: tuple[str, ...], *, field_name: str, allow_empty: bool = False) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not item.startswith("sha256:") or len(item) != 71 for item in values):
        raise ValueError(f"{field_name} must contain SHA-256 hashes")


@dataclass(frozen=True)
class StrategyArtefact:
    definition: StrategyDefinition
    dependency_hash: str
    dataset_snapshot_hashes: tuple[str, ...]
    feature_set_version: str
    cost_model_version: str
    validation_evidence: Mapping[str, Any]
    holdout_claim: Mapping[str, Any]
    forward_evidence: Mapping[str, Any]
    promotion_policy: Mapping[str, Any]
    position_limits: Mapping[str, Any]
    risk_limits: Mapping[str, Any]
    model_hashes: tuple[str, ...]
    supported_products: tuple[str, ...]
    supported_instruments: tuple[str, ...]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _hashes((self.dependency_hash,), field_name="dependency_hash")
        _hashes(self.dataset_snapshot_hashes, field_name="dataset_snapshot_hashes")
        _hashes(self.model_hashes, field_name="model_hashes", allow_empty=True)
        if not self.supported_products or not self.supported_instruments:
            raise ValueError("artefacts need supported products and instruments")
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        for field_name in (
            "validation_evidence",
            "holdout_claim",
            "forward_evidence",
            "promotion_policy",
            "position_limits",
            "risk_limits",
            "metadata",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be an object")
            object.__setattr__(self, field_name, json_value(dict(value), field=field_name))

    @property
    def artefact_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "platform.strategy_artefact/v1",
            "definition": to_primitive(self.definition),
            "definition_hash": self.definition.definition_hash,
            "dependency_hash": self.dependency_hash,
            "dataset_snapshot_hashes": list(self.dataset_snapshot_hashes),
            "feature_set_version": self.feature_set_version,
            "cost_model_version": self.cost_model_version,
            "validation_evidence": dict(self.validation_evidence),
            "holdout_claim": dict(self.holdout_claim),
            "forward_evidence": dict(self.forward_evidence),
            "promotion_policy": dict(self.promotion_policy),
            "position_limits": dict(self.position_limits),
            "risk_limits": dict(self.risk_limits),
            "model_hashes": list(self.model_hashes),
            "supported_products": list(self.supported_products),
            "supported_instruments": list(self.supported_instruments),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["artefact_hash"] = self.artefact_hash
        return payload


class StrategyArtefactStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, artefact: StrategyArtefact) -> Path:
        digest = artefact.artefact_hash.removeprefix("sha256:")
        destination = self.root / digest[:2] / f"{digest}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(artefact.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        if destination.exists():
            if destination.is_symlink() or destination.read_text(encoding="utf-8") != encoded:
                raise RuntimeError("immutable strategy artefact hash collision")
            return destination
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_text(encoding="utf-8") != encoded:
                    raise RuntimeError("immutable strategy artefact hash collision") from None
        finally:
            temporary.unlink(missing_ok=True)
        return destination
