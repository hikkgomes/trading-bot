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
        supported = (
            StrategySourceType.REGISTERED_PYTHON,
            StrategySourceType.GENERATED_DSL,
            StrategySourceType.MACHINE_LEARNING,
            StrategySourceType.CROSS_SECTIONAL,
            StrategySourceType.RELATIVE_VALUE,
            StrategySourceType.MICROSTRUCTURE,
            StrategySourceType.AGENT_GENERATED_PYTHON,
        )
        for source_type in supported:
            registry.register(source_type, _execute_from_canonical_inputs)
        return registry


def _execute_from_canonical_inputs(
    candidate: Candidate, context: Mapping[str, Any]
) -> ExecutionResult:
    """Execute a provider supplied by the research runtime.

    The registry requires an injected callable in production.  Falling back to
    an error is intentional: no empty run or fabricated metric can satisfy a
    validation stage.
    """

    callback = context.get("provider_callable")
    if not callable(callback):
        raise ExecutorError(
            f"{candidate.definition.source_type.value} executor has no canonical data runner"
        )
    result = callback(candidate, context)
    if not isinstance(result, ExecutionResult):
        raise ExecutorError("provider executor returned an invalid execution result")
    return result


def execution_receipt(
    *, candidate: Candidate, dataset_snapshot_ids: tuple[str, ...], executor_version: str
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate.candidate_id,
        "dataset_snapshot_ids": list(dataset_snapshot_ids),
        "executor_version": executor_version,
    }
    return {**payload, "input_hash": canonical_hash(payload)}
