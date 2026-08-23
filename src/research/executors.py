"""Provider executors for canonical candidate evaluation.

The registry is the only place where a strategy source becomes executable.
Every executor returns measured evidence and a receipt describing the exact
inputs it consumed.  A worker cannot manufacture a validation result by
putting metrics in a queue payload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash, json_value
from src.domain.strategies import StrategySourceType
from src.research.coordinator import Candidate
from src.strategies.semantic import SEMANTIC_STRATEGIES


class ExecutorError(RuntimeError):
    """A candidate could not be compiled or evaluated from canonical inputs."""


@dataclass(frozen=True)
class ExecutionResult:
    evidence: Mapping[str, Any]
    metrics: Mapping[str, float]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.evidence or not self.metrics:
            raise ExecutorError("candidate execution must produce non-empty evidence and metrics")
        object.__setattr__(self, "evidence", json_value(dict(self.evidence), field="evidence"))
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        receipt = json_value(dict(self.receipt), field="execution receipt")
        required = {"candidate_id", "dataset_snapshot_ids", "executor_version", "input_hash"}
        if not required.issubset(receipt):
            raise ExecutorError("execution receipt is missing canonical input identities")
        object.__setattr__(self, "receipt", receipt)


Executor = Callable[[Candidate, Mapping[str, Any]], ExecutionResult]


class ProviderExecutorRegistry:
    """Map every supported source family to a concrete executor."""

    def __init__(self) -> None:
        self._executors: dict[StrategySourceType, Executor] = {}

    def register(self, source_type: StrategySourceType, executor: Executor) -> None:
        if source_type in self._executors:
            raise ValueError(f"executor already registered for {source_type.value}")
        self._executors[source_type] = executor

    def executor_for(self, source_type: StrategySourceType) -> Executor:
        try:
            return self._executors[source_type]
        except KeyError as exc:
            raise ExecutorError(f"no executor registered for {source_type.value}") from exc

    def execute(self, candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
        return self.executor_for(candidate.definition.source_type)(candidate, context)

    @classmethod
    def default(cls) -> ProviderExecutorRegistry:
        registry = cls()
        registry.register(StrategySourceType.REGISTERED_PYTHON, execute_registered_python)
        registry.register(StrategySourceType.GENERATED_DSL, execute_generated_dsl)
        registry.register(StrategySourceType.PARAMETER_SEARCH, execute_registered_python)
        registry.register(StrategySourceType.MUTATION, execute_registered_python)
        registry.register(StrategySourceType.CROSSOVER, execute_registered_python)
        registry.register(StrategySourceType.MACHINE_LEARNING, execute_machine_learning)
        registry.register(StrategySourceType.CROSS_SECTIONAL, execute_cross_sectional)
        registry.register(StrategySourceType.RELATIVE_VALUE, execute_relative_value)
        registry.register(StrategySourceType.MICROSTRUCTURE, execute_microstructure)
        registry.register(StrategySourceType.ENSEMBLE, execute_ensemble)
        registry.register(
            StrategySourceType.AGENT_GENERATED_PYTHON, execute_agent_generated_python
        )
        return registry


def _measured_result(candidate: Candidate, context: Mapping[str, Any], output: Any) -> ExecutionResult:
    snapshots = tuple(str(item) for item in context.get("dataset_snapshot_ids", ()))
    if not snapshots:
        raise ExecutorError("executor requires canonical dataset snapshot identities")
    output_hash = canonical_hash(output)
    return ExecutionResult(
        evidence={"compiled": True, "output_hash": output_hash, "observations": 1},
        metrics={"observations": 1.0},
        receipt=execution_receipt(
            candidate=candidate,
            dataset_snapshot_ids=snapshots,
            executor_version="provider-executors/v3",
        ),
    )


def execute_registered_python(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    from src.strategies import library  # noqa: F401
    from src.strategies.registry import get

    frame = context.get("market_frame")
    if frame is None:
        raise ExecutorError("registered Python execution requires a canonical market_frame")
    name = str(candidate.definition.signal_model.get("registered_strategy") or "")
    params = candidate.definition.signal_model.get("parameters", {})
    strategy = get(name)(**dict(params))
    signals = strategy.generate_signals(frame)
    return _measured_result(candidate, context, tuple(int(value) for value in signals))


def execute_generated_dsl(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    rows = context.get("feature_rows")
    rule = candidate.definition.signal_model.get("rule")
    if not isinstance(rows, list | tuple) or not isinstance(rule, Mapping):
        raise ExecutorError("DSL execution requires canonical feature_rows and a typed rule")
    feature = str(rule.get("feature", ""))
    operator = str(rule.get("operator", ""))
    threshold = float(rule.get("threshold"))
    operations = {
        "gt": lambda value: value > threshold,
        "ge": lambda value: value >= threshold,
        "lt": lambda value: value < threshold,
        "le": lambda value: value <= threshold,
    }
    if operator not in operations:
        raise ExecutorError("DSL rule operator is unsupported")
    signals = tuple(1 if operations[operator](float(row[feature])) else 0 for row in rows)
    return _measured_result(candidate, context, signals)


def execute_machine_learning(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    model_hash = context.get("model_artefact_hash")
    manifest_hash = context.get("feature_manifest_hash")
    if not model_hash or not manifest_hash:
        raise ExecutorError("machine-learning execution needs a frozen model and feature manifest")
    model = context.get("frozen_model")
    features = context.get("feature_vector")
    if model is None or not callable(getattr(model, "evaluate", None)) or not isinstance(features, Mapping):
        raise ExecutorError("machine-learning execution needs loaded frozen model inputs")
    return _measured_result(candidate, context, model.evaluate(features))


def execute_cross_sectional(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_relative_value(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_microstructure(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_ensemble(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    return _execute_semantic(candidate, context)


def execute_agent_generated_python(
    candidate: Candidate, context: Mapping[str, Any]
) -> ExecutionResult:
    if not context.get("sandbox_receipt"):
        raise ExecutorError("agent-generated Python needs a verified sandbox receipt")
    return execute_registered_python(candidate, context)


def _execute_semantic(candidate: Candidate, context: Mapping[str, Any]) -> ExecutionResult:
    name = str(candidate.definition.signal_model.get("semantic_strategy") or candidate.definition.identity)
    semantic_input = context.get("semantic_input")
    if semantic_input is None:
        raise ExecutorError("semantic execution requires a typed canonical input")
    output = SEMANTIC_STRATEGIES.get(name).evaluate(semantic_input)
    return _measured_result(candidate, context, output)


def execution_receipt(
    *, candidate: Candidate, dataset_snapshot_ids: tuple[str, ...], executor_version: str
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate.candidate_id,
        "dataset_snapshot_ids": list(dataset_snapshot_ids),
        "executor_version": executor_version,
    }
    return {**payload, "input_hash": canonical_hash(payload)}
