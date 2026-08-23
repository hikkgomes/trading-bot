from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.domain.strategies import StrategySourceType
from src.research.executors import ExecutorError, ProviderExecutorRegistry
from src.services.job_schemas import JobSchemaError


def test_default_executor_does_not_turn_missing_execution_into_evidence() -> None:
    registry = ProviderExecutorRegistry.default()
    candidate = SimpleNamespace(
        definition=SimpleNamespace(source_type=StrategySourceType.REGISTERED_PYTHON)
    )
    with pytest.raises(ExecutorError, match="no canonical data runner"):
        registry.execute(candidate, {})  # type: ignore[arg-type]


def test_result_fields_are_not_accepted_as_a_research_command() -> None:
    from tests.test_research_job_authority import _research_request

    with pytest.raises(JobSchemaError):
        from src.services.job_schemas import ResearchJobRequest

        ResearchJobRequest.from_mapping({**_research_request(), "accepted": False})
