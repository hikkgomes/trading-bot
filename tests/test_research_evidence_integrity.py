from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import insert

from src.data.database import PlatformDatabase, strategy_definition, strategy_version
from src.domain._codec import canonical_hash, to_primitive
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    CanonicalEvidenceError,
    SqlForwardEvidenceRepository,
    SqlStrategyArtefactRepository,
)
from src.research.datasets import (
    CanonicalDatasetResolver,
    DatasetResolutionError,
    ResolvedDataset,
)
from src.research.evaluation import EvaluationRequest
from src.research.executors import ExecutorError, ProviderExecutorRegistry
from src.services.job_schemas import JobSchemaError


class _DatasetRepository:
    def __init__(self, *datasets: ResolvedDataset) -> None:
        self.datasets = {dataset.snapshot_id: dataset for dataset in datasets}

    def resolve(self, snapshot_id: str) -> ResolvedDataset:
        return self.datasets[snapshot_id]


def _dataset(role: str, availability: str) -> ResolvedDataset:
    payload = {"role": role, "bars": [{"close": 100.0}, {"close": 101.0}]}
    return ResolvedDataset(
        snapshot_id=canonical_hash({"snapshot": role}),
        content_hash=canonical_hash(payload),
        interval={
            "start": "2026-08-20T00:00:00+00:00",
            "end": "2026-08-21T00:00:00+00:00",
        },
        universe_snapshot_id=canonical_hash({"universe": role}),
        availability_timestamp=availability,
        feature_manifest_hash=canonical_hash({"feature": "v1"}),
        cost_model_hash=canonical_hash({"cost": "v1"}),
        parameter_set_hash=canonical_hash({"parameters": "v1"}),
        product_id="active_income",
        instrument_scope=("BTCUSDT",),
        engine_version="research/v1",
        payload=payload,
        role=role,
    )


def test_development_resolution_cannot_load_protected_data() -> None:
    development = _dataset("development", "2026-08-22T00:00:00+00:00")
    protected = _dataset("protected_holdout", "2026-08-22T00:00:00+00:00")
    resolver = CanonicalDatasetResolver(_DatasetRepository(development, protected))
    kwargs = {
        "feature_manifest_id": development.feature_manifest_hash,
        "cost_model_id": development.cost_model_hash,
        "parameter_set_id": development.parameter_set_hash,
    }
    resolved = resolver.resolve_context(
        snapshot_ids=(development.snapshot_id,),
        allowed_roles=frozenset({"development"}),
        **kwargs,
    )
    assert resolved["dataset_snapshot_ids"] == [development.snapshot_id]
    with pytest.raises(DatasetResolutionError, match="role is not permitted"):
        resolver.resolve_context(
            snapshot_ids=(protected.snapshot_id,),
            allowed_roles=frozenset({"development"}),
            **kwargs,
        )
    with pytest.raises(DatasetResolutionError, match="explicit protected boundary"):
        resolver.resolve_context(snapshot_ids=(protected.snapshot_id,), **kwargs)


def test_research_stages_select_distinct_dataset_roles() -> None:
    development = canonical_hash({"snapshot": "development"})
    protected = canonical_hash({"snapshot": "protected"})
    forward = canonical_hash({"snapshot": "forward"})
    request = EvaluationRequest(
        candidate_id=canonical_hash({"candidate": "roles"}),
        evaluation_policy_id="policy",
        dataset_snapshot_ids=(development, protected, forward),
        requested_stage="development",
        evaluated_at="2026-08-23T00:00:00+00:00",
        dataset_roles={
            development: "development",
            protected: "protected_holdout",
            forward: "forward_observation",
        },
    )
    assert request.snapshot_ids_for_stage("development") == (development,)
    assert request.snapshot_ids_for_stage("protected") == (protected,)
    assert request.snapshot_ids_for_stage("forward") == (forward,)
    assert protected not in request.snapshot_ids_for_stage("development")


def test_forward_resolution_requires_post_artefact_availability() -> None:
    forward = _dataset("forward_observation", "2026-08-24T00:00:00+00:00")
    resolver = CanonicalDatasetResolver(_DatasetRepository(forward))
    kwargs = {
        "feature_manifest_id": forward.feature_manifest_hash,
        "cost_model_id": forward.cost_model_hash,
        "parameter_set_id": forward.parameter_set_hash,
        "allowed_roles": frozenset({"forward_observation"}),
    }
    resolver.resolve_context(
        snapshot_ids=(forward.snapshot_id,),
        minimum_availability_timestamp="2026-08-23T00:00:00+00:00",
        **kwargs,
    )
    with pytest.raises(DatasetResolutionError, match="after artefact creation"):
        resolver.resolve_context(
            snapshot_ids=(forward.snapshot_id,),
            minimum_availability_timestamp="2026-08-24T00:00:00+00:00",
            **kwargs,
        )


def test_forward_observation_must_follow_artefact_creation(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'forward.sqlite3'}")
    database.create_schema()
    definition = StrategyDefinition(
        identity="forward-test",
        version="v1",
        family="time_series",
        product="active_income",
        universe={"symbols": ["BTCUSDT"]},
        data_requirements={"bars": "1h"},
        feature_graph={"required_nodes": ["bar_return"]},
        signal_model={"kind": "test"},
        position_model={"kind": "volatility_scaled"},
        execution_preferences={"policy": "market"},
        risk_policy={"id": "risk-v1"},
        validation_policy={"id": "validation-v1"},
        source_type=StrategySourceType.REGISTERED_PYTHON,
        source_hash=canonical_hash({"source": "forward-test"}),
    )
    created_at = "2026-08-23T00:00:00+00:00"
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=definition.definition_hash,
                identity=definition.identity,
                product_id=definition.product,
                source_type=definition.source_type.value,
                source_hash=definition.source_hash,
                definition=to_primitive(definition),
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id=definition.strategy_version_id,
                definition_id=definition.definition_hash,
                version=definition.version,
                created_at=created_at,
                payload={"definition_hash": definition.definition_hash},
            )
        )
    artefact = StrategyArtefact(
        definition=definition,
        dependency_hash=canonical_hash({"dependency": "v1"}),
        dataset_snapshot_hashes=(canonical_hash({"dataset": "v1"}),),
        feature_set_version="features-v1",
        cost_model_version="costs-v1",
        validation_evidence={},
        holdout_claim={},
        forward_evidence={},
        promotion_policy={},
        position_limits={"maximum_position": 0.1},
        risk_limits={"policy": "risk-v1"},
        model_hashes=(),
        supported_products=("active_income",),
        supported_instruments=("BTCUSDT",),
        created_at=created_at,
        authoritative_evidence={"test": True},
        portfolio_id="portfolio-active-income",
        account_id="account-usdt",
        promotion_policy_id="promotion-v1",
        engine_version="engine-v1",
    )
    SqlStrategyArtefactRepository(database.engine).put(
        artefact.artefact_hash, artefact.to_dict(), created_at=created_at
    )
    repository = SqlForwardEvidenceRepository(database.engine)
    with pytest.raises(CanonicalEvidenceError, match="after artefact creation"):
        repository.append(
            strategy_version_id=definition.strategy_version_id,
            product_id="active_income",
            instrument_id="BTCUSDT",
            observed_at=created_at,
            artefact_hash=artefact.artefact_hash,
            observation={"direction": "flat"},
        )
    observation_id = repository.append(
        strategy_version_id=definition.strategy_version_id,
        product_id="active_income",
        instrument_id="BTCUSDT",
        observed_at="2026-08-24T00:00:00+00:00",
        artefact_hash=artefact.artefact_hash,
        observation={"direction": "flat"},
    )
    assert observation_id.startswith("sha256:")


def test_default_executor_does_not_turn_missing_execution_into_evidence() -> None:
    registry = ProviderExecutorRegistry.default()
    candidate = SimpleNamespace(
        definition=SimpleNamespace(source_type=StrategySourceType.REGISTERED_PYTHON)
    )
    with pytest.raises(ExecutorError, match="canonical market_frame"):
        registry.execute(candidate, {})  # type: ignore[arg-type]


def test_result_fields_are_not_accepted_as_a_research_command() -> None:
    from tests.test_research_job_authority import _research_request

    with pytest.raises(JobSchemaError):
        from src.services.job_schemas import ResearchJobRequest

        ResearchJobRequest.from_mapping({**_research_request(), "accepted": False})
