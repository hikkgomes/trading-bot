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


SAFE_MODEL_TYPES = frozenset(
    {
        "linear_return_v1",
        "logistic_probability_v1",
        "lightgbm_classifier_json_v1",
        "lightgbm_regressor_json_v1",
        "cross_sectional_ranker_json_v1",
        "triple_barrier_classifier_json_v1",
        "return_regressor_json_v1",
        "meta_labelling_json_v1",
        "probability_calibration_json_v1",
        "regime_classifier_json_v1",
    }
)


@dataclass(frozen=True)
class FrozenSafeModel:
    """Data-only frozen model. Arbitrary pickle and executable code are rejected."""

    artefact_hash: str
    feature_manifest_hash: str
    model_type: str
    feature_names: tuple[str, ...]
    payload: Mapping[str, object]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_artefact_hash: str,
        expected_feature_manifest_hash: str,
    ) -> FrozenSafeModel:
        if path.suffix.lower() not in {".json", ".ubj"} or path.is_symlink() or not path.is_file():
            raise ValueError("production models must use a safe frozen JSON or UBJSON format")
        raw = path.read_bytes()
        if path.suffix.lower() == ".ubj":
            raise ValueError("UBJSON loading requires the optional validated LightGBM runtime")
        artefact_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        if artefact_hash != expected_artefact_hash:
            raise ValueError("frozen model artefact hash mismatch")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("model_type") not in SAFE_MODEL_TYPES:
            raise ValueError("unsupported safe frozen model type")
        names = tuple(str(item) for item in payload.get("feature_names", ()))
        if not names or len(names) != len(set(names)):
            raise ValueError("frozen model needs a unique ordered feature manifest")
        manifest_hash = canonical_hash({"feature_names": names})
        if manifest_hash != expected_feature_manifest_hash:
            raise ValueError("frozen model feature manifest hash mismatch")
        return cls(artefact_hash, manifest_hash, str(payload["model_type"]), names, payload)

    def evaluate(self, features: Mapping[str, float]) -> FrozenModelForecast:
        if tuple(features) != self.feature_names:
            raise ValueError("live feature vector does not match the frozen ordered manifest")
        values = [float(features[name]) for name in self.feature_names]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("frozen model features must be finite")
        raw = _safe_model_raw(self.payload, self.feature_names, values)
        if (
            "logistic" in self.model_type
            or "classifier" in self.model_type
            or "labelling" in self.model_type
            or "calibration" in self.model_type
        ):
            probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))
            expected = probability - 0.5
            score = expected * 2.0
        else:
            expected = raw
            score = max(-1.0, min(1.0, raw))
        return FrozenModelForecast(
            score=score,
            expected_return=expected,
            feature_vector_hash=canonical_hash(
                {"manifest": self.feature_manifest_hash, "values": values}
            ),
        )


def _safe_model_raw(
    payload: Mapping[str, object], feature_names: tuple[str, ...], values: list[float]
) -> float:
    trees = payload.get("trees")
    if isinstance(trees, list):
        feature_values = dict(zip(feature_names, values, strict=True))
        learning_rate = float(payload.get("learning_rate", 1.0))
        return float(payload.get("base_score", 0.0)) + learning_rate * sum(
            _tree_value(tree, feature_values) for tree in trees
        )
    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != len(values):
        raise ValueError("safe frozen model has no validated numeric weights or trees")
    raw = float(payload.get("intercept", 0.0)) + sum(
        float(weight) * value for weight, value in zip(weights, values, strict=True)
    )
    calibration = payload.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, Mapping) or set(calibration) != {"slope", "intercept"}:
            raise ValueError("safe frozen model calibration is invalid")
        raw = float(calibration["slope"]) * raw + float(calibration["intercept"])
    if not math.isfinite(raw):
        raise ValueError("safe frozen model output is not finite")
    return raw


def _tree_value(node: object, features: Mapping[str, float], *, depth: int = 0) -> float:
    if depth > 64 or not isinstance(node, Mapping):
        raise ValueError("safe frozen tree is invalid")
    if set(node) == {"value"}:
        value = float(node["value"])
        if not math.isfinite(value):
            raise ValueError("safe frozen tree leaf is not finite")
        return value
    if set(node) != {"feature", "threshold", "left", "right"}:
        raise ValueError("safe frozen tree node schema is invalid")
    feature = str(node["feature"])
    if feature not in features:
        raise ValueError(f"safe frozen tree uses undeclared feature: {feature}")
    threshold = float(node["threshold"])
    branch = node["left"] if features[feature] <= threshold else node["right"]
    return _tree_value(branch, features, depth=depth + 1)
