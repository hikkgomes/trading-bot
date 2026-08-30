"""Production-only dispatch of immutable strategy artefacts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from src.domain._codec import canonical_hash


class ArtefactDispatchError(RuntimeError):
    pass


Evaluator = Callable[[Mapping[str, float], Mapping[str, Any]], Mapping[str, Any]]


class ArtefactDispatcher:
    """Resolve an artefact source type to one exact production evaluator."""

    def __init__(self, evaluators: Mapping[str, Evaluator] | None = None) -> None:
        self._evaluators = dict(evaluators or {})

    @classmethod
    def default(cls) -> ArtefactDispatcher:
        dispatcher = cls()
        dispatcher.register("registered_python", _registered_python)
        dispatcher.register("parameter_search", _derived_registered_python)
        dispatcher.register("mutation", _derived_registered_python)
        dispatcher.register("crossover", _derived_registered_python)
        dispatcher.register("agent_generated_python", _agent_registered_python)
        dispatcher.register("generated_dsl", _generated_dsl)
        dispatcher.register("machine_learning", _machine_learning)
        dispatcher.register("cross_sectional", _cross_sectional)
        dispatcher.register("relative_value", _relative_value)
        dispatcher.register("microstructure", _microstructure)
        dispatcher.register("ensemble", _ensemble)
        return dispatcher

    def register(self, source_type: str, evaluator: Evaluator) -> None:
        if source_type in self._evaluators:
            raise ValueError(f"production evaluator already registered for {source_type}")
        self._evaluators[source_type] = evaluator

    def evaluate(
        self, features: Mapping[str, float], artefact: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        _verify_artefact(artefact)
        definition = artefact.get("definition")
        if not isinstance(definition, Mapping):
            raise ArtefactDispatchError("artefact has no immutable strategy definition")
        source_type = str(definition.get("source_type") or "")
        evaluator = self._evaluators.get(source_type)
        if evaluator is None:
            raise ArtefactDispatchError(
                f"no production evaluator is registered for source type {source_type!r}"
            )
        values = dict(evaluator(features, artefact))
        required = {
            "direction",
            "score",
            "expected_return",
            "confidence",
            "target_volatility",
            "maximum_position",
        }
        if not required.issubset(values):
            raise ArtefactDispatchError("production evaluator returned an incomplete forecast")
        receipt = {
            "artefact_hash": artefact.get("artefact_hash"),
            "definition_hash": artefact.get("definition_hash"),
            "source_type": source_type,
            "feature_values_hash": canonical_hash(dict(features)),
        }
        values["execution_receipt"] = {**receipt, "receipt_hash": canonical_hash(receipt)}
        return values


def _verify_artefact(artefact: Mapping[str, Any]) -> None:
    expected = artefact.get("artefact_hash")
    content = dict(artefact)
    content.pop("artefact_hash", None)
    if not isinstance(expected, str) or canonical_hash(content) != expected:
        raise ArtefactDispatchError("strategy artefact content hash is invalid")
    definition = artefact.get("definition")
    if not isinstance(definition, Mapping):
        raise ArtefactDispatchError("artefact has no immutable strategy definition")
    declared = artefact.get("definition_hash")
    if not isinstance(declared, str):
        raise ArtefactDispatchError("strategy definition hash is missing")
    authoritative = dict(definition)
    authoritative.pop("version", None)
    authoritative.pop("metadata", None)
    if canonical_hash(authoritative) != declared:
        raise ArtefactDispatchError("strategy definition hash is invalid")


def _forecast(
    score: float, artefact: Mapping[str, Any], *, confidence: float = 0.7
) -> Mapping[str, Any]:
    if not math.isfinite(score):
        raise ArtefactDispatchError("strategy score is not finite")
    signed = max(-1.0, min(1.0, float(score)))
    limits = artefact.get("position_limits")
    limits = limits if isinstance(limits, Mapping) else {}
    maximum = float(limits.get("maximum_position", limits.get("maximum_fraction", 0.1)))
    maximum = max(0.0, min(1.0, maximum))
    direction = "long" if signed > 0 else "short" if signed < 0 else "flat"
    return {
        "direction": direction,
        "score": abs(signed),
        "expected_return": signed * float(limits.get("return_scale", 0.01)),
        "confidence": max(0.0, min(1.0, confidence)),
        "target_volatility": max(0.0, float(limits.get("target_volatility", 0.1))),
        "maximum_position": maximum if direction != "flat" else 0.0,
    }


def _definition(artefact: Mapping[str, Any]) -> Mapping[str, Any]:
    value = artefact["definition"]
    assert isinstance(value, Mapping)
    return value


def _feature(features: Mapping[str, float], *names: str) -> float:
    for name in names:
        if name in features:
            value = float(features[name])
            if math.isfinite(value):
                return value
    raise ArtefactDispatchError("declared strategy feature is unavailable")


def _registered_python(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = _definition(artefact).get("signal_model")
    model = model if isinstance(model, Mapping) else {}
    rule = model.get("production_rule")
    if not isinstance(rule, Mapping) or rule.get("kind") != "linear_feature_score/v1":
        raise ArtefactDispatchError("registered strategy has no immutable production rule")
    terms = rule.get("terms")
    if not isinstance(terms, list | tuple) or not terms:
        raise ArtefactDispatchError("registered strategy production rule has no terms")
    score = 0.0
    for term in terms:
        if not isinstance(term, Mapping):
            raise ArtefactDispatchError("registered strategy production term is invalid")
        scale = float(term.get("scale", 0.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ArtefactDispatchError("registered strategy production scale is invalid")
        value = _feature(features, str(term.get("feature") or ""))
        score += (value - float(term.get("centre", 0.0))) / scale * float(term.get("weight", 0.0))
    return _forecast(score, artefact)


def _derived_registered_python(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Evaluate derived research families only through a sealed production rule."""
    metadata = _definition(artefact).get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("derived_from") is not None:
        if not isinstance(metadata.get("derived_from"), str):
            raise ArtefactDispatchError("derived strategy lineage identity is invalid")
    return _registered_python(features, artefact)


def _agent_registered_python(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    metadata = _definition(artefact).get("metadata")
    if not isinstance(metadata, Mapping) or not metadata.get("sandbox_receipt"):
        raise ArtefactDispatchError("agent-generated strategy needs a verified sandbox receipt")
    return _registered_python(features, artefact)


def _generated_dsl(features: Mapping[str, float], artefact: Mapping[str, Any]) -> Mapping[str, Any]:
    model = _definition(artefact).get("signal_model")
    rule = model.get("rule") if isinstance(model, Mapping) else None
    if not isinstance(rule, Mapping):
        raise ArtefactDispatchError("generated DSL artefact has no typed rule")
    value = _feature(features, str(rule.get("feature") or ""))
    threshold = float(rule.get("threshold", 0.0))
    operator = str(rule.get("operator") or "")
    passed = {
        "gt": value > threshold,
        "ge": value >= threshold,
        "lt": value < threshold,
        "le": value <= threshold,
    }.get(operator)
    if passed is None:
        raise ArtefactDispatchError("generated DSL operator is unsupported")
    return _forecast(1.0 if passed else 0.0, artefact)


def _machine_learning(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = _definition(artefact).get("signal_model")
    model = model if isinstance(model, Mapping) else {}
    names = model.get("feature_names")
    weights = model.get("weights")
    if (
        not isinstance(names, list | tuple)
        or not isinstance(weights, list | tuple)
        or len(names) != len(weights)
    ):
        raise ArtefactDispatchError("frozen model schema is invalid")
    score = float(model.get("intercept", 0.0)) + sum(
        float(weight) * _feature(features, str(name))
        for name, weight in zip(names, weights, strict=True)
    )
    return _forecast(score, artefact, confidence=float(model.get("calibration", 0.7)))


def _cross_sectional(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _forecast(_feature(features, "cross_sectional_rank", "funding_rank"), artefact)


def _relative_value(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _forecast(
        -_feature(
            features,
            "spot_perpetual_basis",
            "spread_zscore",
            "basis_zscore",
            "basis",
        ),
        artefact,
    )


def _microstructure(
    features: Mapping[str, float], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    score = 0.5 * _feature(features, "depth_imbalance", "microprice_displacement") + 0.5 * float(
        features.get("aggressor_flow", 0.0)
    )
    return _forecast(score, artefact)


def _ensemble(features: Mapping[str, float], artefact: Mapping[str, Any]) -> Mapping[str, Any]:
    values = [float(value) for key, value in features.items() if key.startswith("forecast_")]
    if not values:
        values = [_feature(features, "signal", "bar_return")]
    return _forecast(sum(values) / len(values), artefact)
