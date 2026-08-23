"""Frozen machine-learning artefact with an exact ordered feature manifest."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.domain._codec import canonical_hash


@dataclass(frozen=True)
class FrozenModelForecast:
    score: float
    expected_return: float
    feature_vector_hash: str


@dataclass(frozen=True)
class FrozenLinearModel:
    artefact_hash: str
    feature_manifest_hash: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_artefact_hash: str,
        expected_feature_manifest_hash: str,
    ) -> FrozenLinearModel:
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen model artefact must be a regular file")
        raw = path.read_bytes()
        artefact_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        if artefact_hash != expected_artefact_hash:
            raise ValueError("frozen model artefact hash mismatch")
        payload = json.loads(raw)
        allowed = {"model_type", "feature_names", "weights", "intercept"}
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise ValueError("frozen model artefact schema is invalid")
        if payload["model_type"] != "linear_return_v1":
            raise ValueError("unsupported frozen model type")
        names = tuple(str(item) for item in payload["feature_names"])
        weights = tuple(float(item) for item in payload["weights"])
        if not names or len(names) != len(set(names)) or len(names) != len(weights):
            raise ValueError("frozen model feature manifest and weights differ")
        if any(not math.isfinite(value) for value in (*weights, float(payload["intercept"]))):
            raise ValueError("frozen model parameters must be finite")
        manifest_hash = canonical_hash({"feature_names": names})
        if manifest_hash != expected_feature_manifest_hash:
            raise ValueError("frozen model feature manifest hash mismatch")
        return cls(
            artefact_hash=artefact_hash,
            feature_manifest_hash=manifest_hash,
            feature_names=names,
            weights=weights,
            intercept=float(payload["intercept"]),
        )

    def evaluate(self, features: Mapping[str, float]) -> FrozenModelForecast:
        if tuple(features) != self.feature_names:
            raise ValueError("live feature vector does not match the frozen ordered manifest")
        values = tuple(float(features[name]) for name in self.feature_names)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("frozen model features must be finite")
        expected_return = self.intercept + sum(
            weight * value for weight, value in zip(self.weights, values, strict=True)
        )
        score = max(-1.0, min(1.0, expected_return))
        return FrozenModelForecast(
            score=score,
            expected_return=expected_return,
            feature_vector_hash=canonical_hash(
                {"manifest": self.feature_manifest_hash, "values": values}
            ),
        )
