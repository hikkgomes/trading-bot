"""Safe JSON serialization and inference for sklearn gradient-boosting models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SCHEMA = "autopilot.frozen_gradient_boosting/v1"
MAX_FEATURES = 500
MAX_TREES = 2_000
MAX_TOTAL_NODES = 2_000_000


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def export_sklearn_gradient_boosting(strategy: Any) -> dict[str, Any]:
    """Export a fitted project ML strategy without pickle or executable code."""
    model = getattr(strategy, "_model", None)
    features = getattr(strategy, "_features", None)
    params = getattr(strategy, "params", None)
    if (
        model is None
        or not isinstance(features, list)
        or not features
        or not isinstance(params, dict)
    ):
        raise ValueError("ML strategy must be fitted before frozen export")
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise ValueError("only sklearn gradient-boosting estimators support frozen JSON export")
    kind = "classifier" if hasattr(model, "predict_proba") else "regressor"
    if kind == "classifier":
        prior = float(model.init_.class_prior_[1])
        prior = min(1 - 1e-15, max(1e-15, prior))
        initial = math.log(prior / (1 - prior))
        thresholds = {
            "long_threshold": float(params["long_threshold"]),
            "short_threshold": float(params["short_threshold"]),
        }
    else:
        initial = float(model.init_.constant_.reshape(-1)[0])
        thresholds = {"min_edge": float(params["min_edge"])}
    trees = []
    for raw in estimators.reshape(-1):
        tree = raw.tree_
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": tree.value[:, 0, 0].astype(float).tolist(),
            }
        )
    payload = {
        "schema": SCHEMA,
        "kind": kind,
        "feature_names": features,
        "learning_rate": float(model.learning_rate),
        "initial_prediction": initial,
        "trees": trees,
        **thresholds,
    }
    FrozenGradientBoostingModel.from_dict(payload)
    return payload


@dataclass(frozen=True)
class FrozenGradientBoostingModel:
    kind: str
    feature_names: tuple[str, ...]
    learning_rate: float
    initial_prediction: float
    trees: tuple[dict[str, tuple[Any, ...]], ...]
    long_threshold: float | None = None
    short_threshold: float | None = None
    min_edge: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenGradientBoostingModel:
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("frozen gradient-boosting schema is invalid")
        kind = payload.get("kind")
        if kind not in {"classifier", "regressor"}:
            raise ValueError("frozen gradient-boosting kind is invalid")
        names = payload.get("feature_names")
        if (
            not isinstance(names, list)
            or not names
            or len(names) > MAX_FEATURES
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("frozen gradient-boosting features are invalid")
        raw_trees = payload.get("trees")
        if not isinstance(raw_trees, list) or not raw_trees or len(raw_trees) > MAX_TREES:
            raise ValueError("frozen gradient-boosting tree count is invalid")
        trees = []
        total_nodes = 0
        for raw in raw_trees:
            if not isinstance(raw, dict) or set(raw) != {
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "value",
            }:
                raise ValueError("frozen gradient-boosting tree fields are invalid")
            arrays = {key: raw[key] for key in raw}
            if any(not isinstance(value, list) for value in arrays.values()):
                raise ValueError("frozen gradient-boosting tree arrays are invalid")
            lengths = {len(value) for value in arrays.values()}
            if len(lengths) != 1:
                raise ValueError("frozen gradient-boosting tree arrays are invalid")
            nodes = next(iter(lengths), 0)
            if nodes < 1:
                raise ValueError("frozen gradient-boosting tree is empty")
            total_nodes += nodes
            if total_nodes > MAX_TOTAL_NODES:
                raise ValueError("frozen gradient-boosting node budget exceeded")
            left = tuple(int(value) for value in raw["children_left"])
            right = tuple(int(value) for value in raw["children_right"])
            features = tuple(int(value) for value in raw["feature"])
            thresholds = tuple(_finite(value, "tree threshold") for value in raw["threshold"])
            values = tuple(_finite(value, "tree value") for value in raw["value"])
            for index, (child_left, child_right, feature) in enumerate(
                zip(left, right, features, strict=True)
            ):
                leaf = child_left == -1 and child_right == -1
                if leaf:
                    continue
                if not (0 <= child_left < nodes and 0 <= child_right < nodes):
                    raise ValueError(f"tree node {index} has invalid children")
                if not 0 <= feature < len(names):
                    raise ValueError(f"tree node {index} has invalid feature")
            trees.append(
                {
                    "children_left": left,
                    "children_right": right,
                    "feature": features,
                    "threshold": thresholds,
                    "value": values,
                }
            )
        long_threshold = payload.get("long_threshold")
        short_threshold = payload.get("short_threshold")
        min_edge = payload.get("min_edge")
        if kind == "classifier":
            long_threshold = _finite(long_threshold, "long_threshold")
            short_threshold = _finite(short_threshold, "short_threshold")
            if not 0 < short_threshold < long_threshold < 1:
                raise ValueError("classifier thresholds are invalid")
            min_edge = None
        else:
            min_edge = _finite(min_edge, "min_edge")
            if min_edge <= 0:
                raise ValueError("regressor min_edge must be positive")
            long_threshold = short_threshold = None
        learning_rate = _finite(payload.get("learning_rate"), "learning_rate")
        if not 0 < learning_rate <= 10:
            raise ValueError("learning_rate must be in (0, 10]")
        return cls(
            kind=str(kind),
            feature_names=tuple(names),
            learning_rate=learning_rate,
            initial_prediction=_finite(payload.get("initial_prediction"), "initial_prediction"),
            trees=tuple(trees),
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            min_edge=min_edge,
        )

    @staticmethod
    def _tree_value(tree: dict[str, tuple[Any, ...]], values: tuple[float, ...]) -> float:
        node = 0
        visited = 0
        while tree["children_left"][node] != -1:
            visited += 1
            if visited > len(tree["value"]):
                raise ValueError("frozen gradient-boosting tree contains a cycle")
            feature = tree["feature"][node]
            node = (
                tree["children_left"][node]
                if values[feature] <= tree["threshold"][node]
                else tree["children_right"][node]
            )
        return float(tree["value"][node])

    def raw_prediction(self, row: dict[str, Any]) -> float:
        values = tuple(_finite(row.get(name), f"feature {name}") for name in self.feature_names)
        return self.initial_prediction + self.learning_rate * sum(
            self._tree_value(tree, values) for tree in self.trees
        )

    def prediction(self, row: dict[str, Any]) -> float:
        raw = self.raw_prediction(row)
        if self.kind == "regressor":
            return raw
        if raw >= 0:
            return 1.0 / (1.0 + math.exp(-raw))
        exp_raw = math.exp(raw)
        return exp_raw / (1.0 + exp_raw)

    def triggered(self, row: dict[str, Any], direction: str) -> bool:
        if direction not in {"long", "short"}:
            raise ValueError("frozen ML direction must be long or short")
        prediction = self.prediction(row)
        if self.kind == "classifier":
            return (
                prediction > float(self.long_threshold)
                if direction == "long"
                else prediction < float(self.short_threshold)
            )
        return (
            prediction > float(self.min_edge)
            if direction == "long"
            else prediction < -float(self.min_edge)
        )
