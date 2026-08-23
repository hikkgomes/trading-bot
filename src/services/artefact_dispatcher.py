"""Production-only dispatch of immutable strategy artefacts."""

from __future__ import annotations

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

    def register(self, source_type: str, evaluator: Evaluator) -> None:
        if source_type in self._evaluators:
            raise ValueError(f"production evaluator already registered for {source_type}")
        self._evaluators[source_type] = evaluator

    def evaluate(
        self, features: Mapping[str, float], artefact: Mapping[str, Any]
    ) -> Mapping[str, Any]:
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
        receipt = {
            "artefact_hash": artefact.get("artefact_hash"),
            "definition_hash": artefact.get("definition_hash"),
            "source_type": source_type,
            "feature_values_hash": canonical_hash(dict(features)),
        }
        values["execution_receipt"] = {**receipt, "receipt_hash": canonical_hash(receipt)}
        return values
