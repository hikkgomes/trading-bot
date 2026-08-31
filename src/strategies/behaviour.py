"""One executable behaviour contract shared by research and production."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty


class StrategyBehaviourError(ValueError):
    """A registered strategy behaviour cannot be evaluated safely."""


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


def behaviour_hash_for_definition(definition: Any) -> str:
    """Return the behaviour identity without including deployment content."""

    signal_model = (
        definition.get("signal_model")
        if isinstance(definition, Mapping)
        else getattr(definition, "signal_model", None)
    )
    if isinstance(signal_model, Mapping) and signal_model.get("registered_strategy"):
        return RegisteredStrategyBehaviour.from_definition(definition).behaviour_hash
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
