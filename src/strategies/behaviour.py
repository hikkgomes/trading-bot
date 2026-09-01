"""One executable behaviour contract shared by research and production."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty


class StrategyBehaviourError(ValueError):
    """A registered strategy behaviour cannot be evaluated safely."""


@dataclass(frozen=True)
class TypedRuleBehaviour:
    """Immutable typed-rule behaviour shared by research and production."""

    rule: Mapping[str, Any]
    contract_version: str = "typed_rule/v1"

    def __post_init__(self) -> None:
        if self.contract_version != "typed_rule/v1":
            raise StrategyBehaviourError("unsupported typed-rule behaviour contract")
        if not isinstance(self.rule, Mapping):
            raise StrategyBehaviourError("typed strategy rule must be an object")
        object.__setattr__(self, "rule", json_value(dict(self.rule), field="typed strategy rule"))
        if not any(
            key in self.rule
            for key in (
                "conditions",
                "positive_conditions",
                "negative_conditions",
                "long_conditions",
                "short_conditions",
                "exit_conditions",
            )
        ):
            _validate_condition(self.rule, field="rule")
        else:
            entry_groups = (
                "conditions",
                "positive_conditions",
                "negative_conditions",
                "long_conditions",
                "short_conditions",
            )
            if not any(key in self.rule for key in entry_groups):
                raise StrategyBehaviourError("typed strategy rule has no entry conditions")
            for key in (*entry_groups, "exit_conditions"):
                if key in self.rule:
                    _conditions(self.rule, key, required=key in entry_groups)
        direction = str(self.rule.get("direction") or "long")
        if direction not in {"long", "short", "signed", "market_neutral", "hedged"}:
            raise StrategyBehaviourError("typed strategy rule direction is invalid")

    @classmethod
    def from_definition(cls, definition: Any) -> TypedRuleBehaviour:
        signal_model = (
            definition.get("signal_model")
            if isinstance(definition, Mapping)
            else getattr(definition, "signal_model", None)
        )
        rule = signal_model.get("rule") if isinstance(signal_model, Mapping) else None
        if not isinstance(rule, Mapping):
            raise StrategyBehaviourError("strategy definition has no typed rule")
        return cls(rule)

    @property
    def behaviour_hash(self) -> str:
        return canonical_hash({"contract_version": self.contract_version, "rule": self.rule})

    def signal(self, features: Mapping[str, Any]) -> int:
        direction = str(self.rule.get("direction") or "long")
        if _has_composite_conditions(self.rule):
            return _composite_signal(self.rule, features, direction=direction)
        threshold = float(self.rule.get("threshold", 0.0))
        feature = str(self.rule["feature"])
        if feature not in features:
            raise StrategyBehaviourError(f"typed strategy feature is unavailable: {feature}")
        try:
            value = float(features[feature])
        except (TypeError, ValueError) as exc:
            raise StrategyBehaviourError("typed strategy feature value is invalid") from exc
        if not math.isfinite(value):
            raise StrategyBehaviourError("typed strategy feature value is not finite")
        if direction in {"signed", "market_neutral", "hedged"}:
            magnitude = abs(threshold)
            long_operator = str(self.rule.get("positive_operator") or "gt")
            short_operator = str(self.rule.get("negative_operator") or "lt")
            long_threshold = float(self.rule.get("positive_threshold", magnitude))
            short_threshold = float(self.rule.get("negative_threshold", -magnitude))
            _validate_operator(long_operator, field="positive_operator")
            _validate_operator(short_operator, field="negative_operator")
            if not math.isfinite(long_threshold) or not math.isfinite(short_threshold):
                raise StrategyBehaviourError("typed strategy directional thresholds are invalid")
            if _compare(value, long_operator, long_threshold):
                return 1
            if _compare(value, short_operator, short_threshold):
                return -1
            return 0
        passed = _compare(value, str(self.rule["operator"]), threshold)
        return -1 if passed and direction == "short" else 1 if passed else 0

    def generate_signals(self, rows: Any) -> tuple[int, ...]:
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        if not isinstance(rows, list | tuple) or not rows:
            raise StrategyBehaviourError("typed strategy rows must be a non-empty sequence")
        if not all(isinstance(row, Mapping) for row in rows):
            raise StrategyBehaviourError("typed strategy rows must be objects")
        return tuple(self.signal(row) for row in rows)

    def parity_receipt(self, rows: Any) -> Mapping[str, Any]:
        signals = self.generate_signals(rows)
        input_payload = rows.to_dict(orient="records") if hasattr(rows, "to_dict") else rows
        payload = {
            "schema": "typed_rule_parity/v1",
            "behaviour_hash": self.behaviour_hash,
            "input_hash": canonical_hash(input_payload),
            "signals": list(signals),
        }
        return {**payload, "receipt_hash": canonical_hash(payload)}


def _hash_identity(value: str, *, field: str) -> str:
    if not value.startswith("sha256:") or len(value) != 71:
        raise StrategyBehaviourError(f"{field} must be a SHA-256 identity")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise StrategyBehaviourError(f"{field} must be a SHA-256 identity") from exc
    return value


@dataclass(frozen=True)
class RegisteredStrategyBehaviour:
    """Immutable reference to the exact registered strategy implementation."""

    name: str
    parameters: Mapping[str, Any]
    source_hash: str
    contract_version: str = "registered_strategy/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", non_empty(self.name, field="strategy_name"))
        object.__setattr__(
            self, "source_hash", _hash_identity(self.source_hash, field="source_hash")
        )
        if self.contract_version != "registered_strategy/v1":
            raise StrategyBehaviourError("unsupported registered strategy behaviour contract")
        if not isinstance(self.parameters, Mapping):
            raise StrategyBehaviourError("registered strategy parameters must be an object")
        object.__setattr__(
            self,
            "parameters",
            json_value(dict(self.parameters), field="registered strategy parameters"),
        )

    @classmethod
    def from_definition(cls, definition: Any) -> RegisteredStrategyBehaviour:
        if isinstance(definition, Mapping):
            signal_model = definition.get("signal_model")
            source_hash = definition.get("source_hash")
        else:
            signal_model = getattr(definition, "signal_model", None)
            source_hash = getattr(definition, "source_hash", None)
        if not isinstance(signal_model, Mapping) or not isinstance(source_hash, str):
            raise StrategyBehaviourError("strategy definition has no registered behaviour")
        name = signal_model.get("registered_strategy")
        if not isinstance(name, str) or not name:
            raise StrategyBehaviourError("strategy definition has no registered strategy name")
        parameters = signal_model.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise StrategyBehaviourError("registered strategy parameters must be an object")
        return cls(name=name, parameters=parameters, source_hash=source_hash)

    @property
    def behaviour_hash(self) -> str:
        return canonical_hash(
            {
                "contract_version": self.contract_version,
                "strategy_name": self.name,
                "parameters": self.parameters,
                "source_hash": self.source_hash,
            }
        )

    def generate_signals(self, frame: Any) -> tuple[int, ...]:
        from src.strategies import library  # noqa: F401
        from src.strategies.registry import get

        if frame is None or not hasattr(frame, "__len__"):
            raise StrategyBehaviourError("registered strategy requires an immutable market frame")
        try:
            from src.research.catalogue import registered_strategy_source_hash

            current_source_hash = registered_strategy_source_hash(self.name)
        except (ImportError, KeyError, TypeError, ValueError) as exc:
            raise StrategyBehaviourError(
                "registered strategy source identity is unavailable"
            ) from exc
        if current_source_hash != self.source_hash:
            raise StrategyBehaviourError("registered strategy source identity is stale")
        if isinstance(frame, list | tuple):
            if not frame or not all(isinstance(row, Mapping) for row in frame):
                raise StrategyBehaviourError("market frame rows must be objects")
            import pandas as pd

            frame = pd.DataFrame(tuple(dict(row) for row in frame))
        try:
            output = get(self.name)(**dict(self.parameters)).generate_signals(frame)
        except (KeyError, TypeError, ValueError) as exc:
            raise StrategyBehaviourError(f"registered strategy evaluation failed: {exc}") from exc
        values = tuple(self._signal(value) for value in output)
        if len(values) != len(frame):
            raise StrategyBehaviourError("registered strategy output is not frame-aligned")
        return values

    def latest_signal(self, frame: Any) -> int:
        values = self.generate_signals(frame)
        return values[-1] if values else 0

    def frame_payload(self, frame: Any) -> tuple[dict[str, Any], ...]:
        if hasattr(frame, "to_dict"):
            raw_rows = frame.to_dict(orient="records")
        elif isinstance(frame, list | tuple):
            raw_rows = frame
        else:
            raise StrategyBehaviourError("market frame must be a dataframe or row sequence")
        if not isinstance(raw_rows, list | tuple) or not raw_rows:
            raise StrategyBehaviourError("market frame cannot be empty")
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise StrategyBehaviourError("market frame rows must be objects")
            rows.append(
                {
                    str(key): float(value)
                    if isinstance(value, int | float) and not isinstance(value, bool)
                    else value
                    for key, value in raw.items()
                }
            )
        return tuple(json_value(rows, field="market frame"))

    def frame_input_hash(self, frame: Any) -> str:
        return canonical_hash(
            {
                "behaviour_hash": self.behaviour_hash,
                "market_frame": self.frame_payload(frame),
            }
        )

    def parity_receipt(self, frame: Any) -> Mapping[str, Any]:
        """Return a deterministic bar-by-bar receipt for runtime parity."""

        signals = self.generate_signals(frame)
        payload = {
            "schema": "registered_strategy_parity/v1",
            "behaviour_hash": self.behaviour_hash,
            "input_hash": self.frame_input_hash(frame),
            "signals": list(signals),
        }
        return {**payload, "receipt_hash": canonical_hash(payload)}

    @staticmethod
    def _signal(value: Any) -> int:
        if isinstance(value, bool):
            raise StrategyBehaviourError("strategy signal cannot be boolean")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise StrategyBehaviourError("strategy signal must be an integer") from exc
        if number not in {-1, 0, 1} or float(value) != number:
            raise StrategyBehaviourError("strategy signal must be -1, 0, or 1")
        return number


def _validate_condition(value: Mapping[str, Any], *, field: str) -> None:
    feature = str(value.get("feature") or "")
    operator = str(value.get("operator") or "")
    threshold = value.get("threshold")
    if not feature or operator not in {"gt", "ge", "lt", "le"}:
        raise StrategyBehaviourError(f"typed strategy {field} feature or operator is invalid")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise StrategyBehaviourError(f"typed strategy {field} threshold is invalid")
    if not math.isfinite(float(threshold)):
        raise StrategyBehaviourError(f"typed strategy {field} threshold is not finite")


def _has_composite_conditions(rule: Mapping[str, Any]) -> bool:
    return any(
        key in rule
        for key in (
            "conditions",
            "positive_conditions",
            "negative_conditions",
            "long_conditions",
            "short_conditions",
            "exit_conditions",
        )
    )


def _composite_signal(
    rule: Mapping[str, Any], features: Mapping[str, Any], *, direction: str
) -> int:
    if _conditions_pass(rule, "exit_conditions", features):
        return 0
    default = next(
        key
        for key in (
            "conditions",
            "long_conditions",
            "positive_conditions",
            "short_conditions",
            "negative_conditions",
        )
        if key in rule
    )
    long_group = (
        "long_conditions"
        if "long_conditions" in rule
        else "positive_conditions"
        if "positive_conditions" in rule
        else default
    )
    short_group = (
        "short_conditions"
        if "short_conditions" in rule
        else "negative_conditions"
        if "negative_conditions" in rule
        else default
    )
    if direction in {"signed", "market_neutral", "hedged"}:
        positive_group = "positive_conditions" if "positive_conditions" in rule else long_group
        negative_group = "negative_conditions" if "negative_conditions" in rule else short_group
        return (
            1
            if _conditions_pass(rule, positive_group, features)
            else -1
            if _conditions_pass(rule, negative_group, features)
            else 0
        )
    passed = _conditions_pass(rule, long_group, features)
    return -1 if passed and direction == "short" else 1 if passed else 0


def _conditions(
    rule: Mapping[str, Any], key: str, *, required: bool
) -> tuple[Mapping[str, Any], ...]:
    raw = rule.get(key)
    if raw is None:
        if required:
            raise StrategyBehaviourError(f"typed strategy rule has no {key}")
        return ()
    if not isinstance(raw, list | tuple) or not raw:
        raise StrategyBehaviourError(f"typed strategy {key} must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    for index, condition in enumerate(raw):
        if not isinstance(condition, Mapping):
            raise StrategyBehaviourError(f"typed strategy {key}[{index}] is invalid")
        _validate_condition(condition, field=f"{key}[{index}]")
        result.append(condition)
    return tuple(result)


def _conditions_pass(rule: Mapping[str, Any], key: str, features: Mapping[str, Any]) -> bool:
    conditions = _conditions(rule, key, required=False)
    if not conditions:
        return False
    mode = str(rule.get("condition_mode") or "all")
    if mode not in {"all", "any"}:
        raise StrategyBehaviourError("typed strategy condition_mode is invalid")
    results = []
    for condition in conditions:
        feature = str(condition["feature"])
        if feature not in features:
            raise StrategyBehaviourError(f"typed strategy feature is unavailable: {feature}")
        try:
            value = float(features[feature])
        except (TypeError, ValueError) as exc:
            raise StrategyBehaviourError("typed strategy feature value is invalid") from exc
        if not math.isfinite(value):
            raise StrategyBehaviourError("typed strategy feature value is not finite")
        results.append(_compare(value, str(condition["operator"]), float(condition["threshold"])))
    return any(results) if mode == "any" else all(results)


def _validate_operator(value: str, *, field: str) -> None:
    if value not in {"gt", "ge", "lt", "le"}:
        raise StrategyBehaviourError(f"typed strategy {field} is invalid")


def _compare(value: float, operator: str, threshold: float) -> bool:
    _validate_operator(operator, field="operator")
    return {
        "gt": value > threshold,
        "ge": value >= threshold,
        "lt": value < threshold,
        "le": value <= threshold,
    }[operator]


def behaviour_hash_for_definition(definition: Any) -> str:
    """Return the behaviour identity without including deployment content."""

    signal_model = (
        definition.get("signal_model")
        if isinstance(definition, Mapping)
        else getattr(definition, "signal_model", None)
    )
    if isinstance(signal_model, Mapping) and signal_model.get("registered_strategy"):
        return RegisteredStrategyBehaviour.from_definition(definition).behaviour_hash
    if isinstance(signal_model, Mapping) and isinstance(signal_model.get("rule"), Mapping):
        return TypedRuleBehaviour.from_definition(definition).behaviour_hash
    definition_hash = (
        definition.definition_hash
        if hasattr(definition, "definition_hash")
        else canonical_hash(
            {
                key: definition[key]
                for key in (
                    "identity",
                    "family",
                    "product",
                    "universe",
                    "data_requirements",
                    "feature_graph",
                    "signal_model",
                    "position_model",
                    "execution_preferences",
                    "risk_policy",
                    "validation_policy",
                    "source_type",
                    "source_hash",
                )
                if key in definition
            }
        )
    )
    return canonical_hash(
        {
            "contract_version": "definition_behaviour/v1",
            "definition_hash": definition_hash,
        }
    )
