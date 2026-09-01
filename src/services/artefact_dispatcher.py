"""Production-only dispatch of immutable strategy artefacts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from src.domain._codec import canonical_hash
from src.strategies.behaviour import (
    RegisteredStrategyBehaviour,
    StrategyBehaviourError,
    TypedRuleBehaviour,
    behaviour_hash_for_definition,
)
from src.strategies.semantic import (
    SEMANTIC_STRATEGIES,
    SemanticEvaluationError,
    semantic_forecast_from_output,
    semantic_input_from_features,
    semantic_strategy_name,
)


class ArtefactDispatchError(RuntimeError):
    pass


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


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
        for source_type in ("cross_sectional", "relative_value", "microstructure", "ensemble"):
            dispatcher.register(source_type, _semantic)
        return dispatcher

    def register(self, source_type: str, evaluator: Evaluator) -> None:
        if source_type in self._evaluators:
            raise ValueError(f"production evaluator already registered for {source_type}")
        self._evaluators[source_type] = evaluator

    def evaluate(
        self, features: Mapping[str, Any], artefact: Mapping[str, Any]
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
            "deployment_hash": artefact.get("artefact_hash"),
            "definition_hash": artefact.get("definition_hash"),
            "source_type": source_type,
            "feature_values_hash": canonical_hash(_feature_input_payload(features)),
        }
        if values.get("behaviour_hash") is not None:
            receipt["behaviour_hash"] = values["behaviour_hash"]
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
    declared_behaviour = artefact.get("behaviour_hash")
    if declared_behaviour is not None:
        try:
            expected_behaviour = behaviour_hash_for_definition(_definition(artefact))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtefactDispatchError("strategy behaviour identity is invalid") from exc
        if declared_behaviour != expected_behaviour:
            raise ArtefactDispatchError("strategy behaviour hash is invalid")


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
    features: Mapping[str, Any], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = _definition(artefact).get("signal_model")
    model = model if isinstance(model, Mapping) else {}
    if model.get("registered_strategy"):
        return _registered_strategy_behaviour(features, artefact)
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


def _registered_strategy_behaviour(
    features: Mapping[str, Any], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    try:
        behaviour = RegisteredStrategyBehaviour.from_definition(_definition(artefact))
        frame = features.get("market_frame")
        if frame is not None:
            parity = behaviour.parity_receipt(frame)
            signal = int(parity["signals"][-1]) if parity["signals"] else 0
            input_hash = behaviour.frame_input_hash(frame)
            input_status = "market_frame"
        else:
            raw_signal = features.get("registered_signal", 0)
            signal = behaviour._signal(raw_signal)
            input_hash = canonical_hash(
                {"behaviour_hash": behaviour.behaviour_hash, "registered_signal": signal}
            )
            input_status = (
                "registered_signal" if "registered_signal" in features else "history_unavailable"
            )
    except StrategyBehaviourError as exc:
        raise ArtefactDispatchError(str(exc)) from exc
    forecast = dict(_forecast(float(signal), artefact))
    forecast["expected_return"] = float(signal) * float(forecast["target_volatility"])
    forecast["behaviour_hash"] = behaviour.behaviour_hash
    forecast["behaviour_input_hash"] = input_hash
    forecast["behaviour_input_status"] = input_status
    if frame is not None:
        forecast["parity_receipt"] = parity
    return forecast


def _feature_input_payload(features: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in features.items():
        if name == "market_frame" and hasattr(value, "to_dict"):
            rows = value.to_dict(orient="records")
            payload[name] = [
                {
                    str(key): float(item)
                    if isinstance(item, int | float) and not isinstance(item, bool)
                    else item
                    for key, item in row.items()
                }
                for row in rows
            ]
        else:
            payload[str(name)] = value
    return payload


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
    try:
        behaviour = TypedRuleBehaviour.from_definition(_definition(artefact))
        signal = behaviour.signal(features)
    except StrategyBehaviourError as exc:
        raise ArtefactDispatchError(str(exc)) from exc
    result = dict(_forecast(float(signal), artefact))
    result["behaviour_hash"] = behaviour.behaviour_hash
    result["behaviour_input_hash"] = canonical_hash(
        {"behaviour_hash": behaviour.behaviour_hash, "features": dict(features)}
    )
    return result


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


def _semantic(
    features: Mapping[str, Any], artefact: Mapping[str, Any]
) -> Mapping[str, Any]:
    definition = _definition(artefact)
    model = definition.get("signal_model")
    model = model if isinstance(model, Mapping) else {}
    source_type = str(definition.get("source_type") or "")
    name = semantic_strategy_name(source_type, model.get("semantic_strategy"))
    instrument_id = str(
        features.get("instrument_id")
        or next(iter(artefact.get("supported_instruments", ())), "")
    )
    if not instrument_id:
        raise ArtefactDispatchError("semantic evaluation requires an instrument identity")
    try:
        semantic_input = semantic_input_from_features(name, features)
        output = SEMANTIC_STRATEGIES.get(name).evaluate(semantic_input)
        result = semantic_forecast_from_output(
            output,
            instrument_id=instrument_id,
            position_limits=artefact.get("position_limits"),
        )
    except (SemanticEvaluationError, KeyError, TypeError, ValueError) as exc:
        raise ArtefactDispatchError(str(exc)) from exc
    result["semantic_strategy"] = name
    input_hash = canonical_hash(semantic_input)
    output_hash = canonical_hash(output)
    result["semantic_input_hash"] = input_hash
    result["behaviour_hash"] = artefact.get("behaviour_hash") or behaviour_hash_for_definition(
        definition
    )
    parity_payload = {
        "schema": "semantic_parity/v1",
        "behaviour_hash": result["behaviour_hash"],
        "strategy": name,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "instrument_id": instrument_id,
        "signal": result["semantic_signal"],
    }
    result["parity_receipt"] = {
        **parity_payload,
        "receipt_hash": canonical_hash(parity_payload),
    }
    return result
