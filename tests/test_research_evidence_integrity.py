from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import insert

from src.data.database import (
    PlatformDatabase,
    alpha_forecast,
    strategy_definition,
    strategy_version,
)
from src.domain._codec import canonical_hash, to_primitive
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    CanonicalEvidenceError,
    SqlActiveStrategyAssignmentRepository,
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
from src.services.forward_observation import DatabaseForwardObservationWorker
from src.services.job_schemas import JobSchemaError
from src.services.scheduler import DatabaseJobQueue


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
    with pytest.raises(DatasetResolutionError, match="evaluation time"):
        resolver.resolve_context(
            snapshot_ids=(forward.snapshot_id,),
            minimum_availability_timestamp="2026-08-23T00:00:00+00:00",
            maximum_availability_timestamp="2026-08-23T23:59:59+00:00",
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

    forecast_payload = {
        "strategy_version_id": definition.strategy_version_id,
        "product_id": "active_income",
        "instrument_id": "BTCUSDT",
        "direction": "long",
    }
    forecast_id = canonical_hash(forecast_payload)
    with database.engine.begin() as connection:
        connection.execute(
            insert(alpha_forecast).values(
                id=forecast_id,
                created_at="2026-08-25T00:00:00+00:00",
                payload=forecast_payload,
            )
        )

    def facts(*, net_pnl: float, benchmark_pnl: float = 0.0) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "platform.forward_evidence_facts/v1",
            "window_start": "2026-08-23T00:00:00+00:00",
            "source_event_ids": [artefact.artefact_hash],
            "metrics": {
                "net_pnl": net_pnl,
                "benchmark_pnl": benchmark_pnl,
                "drawdown": 0.0,
                "execution_drift": 0.0,
                "model_drift": 0.0,
                "portfolio_capacity": 100.0,
                "risk_budget_available": 10.0,
                "data_gaps": 0,
                "effective_trades": 1,
                "fill_rate": 1.0,
                "slippage": 0.0,
                "data_uptime": 1.0,
                "rejected_orders": 0,
            },
            "forecast_hash": canonical_hash(forecast_payload),
            "target_hash": None,
        }
        payload["facts_hash"] = canonical_hash(payload)
        return payload

    with pytest.raises(CanonicalEvidenceError, match="after artefact creation"):
        repository.append(
            strategy_version_id=definition.strategy_version_id,
            product_id="active_income",
            instrument_id="BTCUSDT",
            observed_at=created_at,
            artefact_hash=artefact.artefact_hash,
            observation={"direction": "flat"},
        )
    with pytest.raises(CanonicalEvidenceError, match="activate before artefact creation"):
        SqlActiveStrategyAssignmentRepository(database.engine).assign(
            product_id="active_income",
            portfolio_id="portfolio-active-income",
            strategy_version_id=definition.strategy_version_id,
            artefact_hash=artefact.artefact_hash,
            lifecycle_state="forward_paper",
            execution_mode="paper",
            capital_limit=100.0,
            assigned_at="2026-08-22T23:59:59+00:00",
            assigned_by="test",
            instrument_id="BTCUSDT",
        )
    observation_id = repository.append(
        strategy_version_id=definition.strategy_version_id,
        product_id="active_income",
        instrument_id="BTCUSDT",
        observed_at="2026-08-24T00:00:00+00:00",
        artefact_hash=artefact.artefact_hash,
        observation={"decision_id": "decision-1", "facts": facts(net_pnl=0.0)},
    )
    assert observation_id.startswith("sha256:")

    assignment_id = SqlActiveStrategyAssignmentRepository(database.engine).assign(
        product_id="active_income",
        portfolio_id="portfolio-active-income",
        strategy_version_id=definition.strategy_version_id,
        artefact_hash=artefact.artefact_hash,
        lifecycle_state="forward_paper",
        execution_mode="paper",
        capital_limit=100.0,
        assigned_at="2026-08-24T01:00:00+00:00",
        assigned_by="test",
        instrument_id="BTCUSDT",
    )
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="test:forward-observer",
        node_id="test",
        role="promotion-engine",
        capabilities=("forward_paper_observation",),
        observed_at="2026-08-26T00:00:00+00:00",
    )
    queue.enqueue(
        job_id="test:forward-observation",
        name="forward_paper_observation",
        payload={
            "assignment_id": assignment_id,
            "strategy_version_id": definition.strategy_version_id,
            "product_id": "active_income",
            "instrument_id": "BTCUSDT",
            "artefact_hash": artefact.artefact_hash,
            "evaluation_time": "2026-08-26T00:00:00+00:00",
        },
        available_at="2026-08-26T00:00:00+00:00",
    )
    recorded = DatabaseForwardObservationWorker(
        engine=database.engine,
        queue=queue,
        worker_id="test:forward-observer",
    ).run_once(now="2026-08-26T00:00:00+00:00")

    assert recorded["reason_code"] == "forward_observation_recorded"
    assert recorded["observed_at"] == "2026-08-25T00:00:00+00:00"
    second_observation_id = repository.append(
        strategy_version_id=definition.strategy_version_id,
        product_id="active_income",
        instrument_id="BTCUSDT",
        observed_at="2026-08-26T00:00:00+00:00",
        artefact_hash=artefact.artefact_hash,
        observation={
            "decision_id": "decision-2",
            "facts": facts(net_pnl=1.0, benchmark_pnl=0.25),
        },
    )
    summary_id, summary = repository.build_summary(
        strategy_version_id=definition.strategy_version_id,
        product_id="active_income",
        artefact_hash=artefact.artefact_hash,
        observed_at="2026-08-26T00:00:00+00:00",
    )
    assert summary.independent_decisions == 3
    assert summary.net_pnl == pytest.approx(1.0)
    assert summary.benchmark_pnl == pytest.approx(0.25)
    assert summary.excess_benchmark_pnl == pytest.approx(0.75)
    assert second_observation_id in summary.observation_ids
    decision_id, accepted, reason_code = repository.decide_summary(
        summary_id,
        decided_at="2026-08-26T00:00:01+00:00",
        minimum_days=1,
        minimum_decisions=2,
    )
    assert decision_id.startswith("sha256:")
    assert accepted is True
    assert reason_code is None
    with pytest.raises(CanonicalEvidenceError, match="unknown observation identity"):
        repository.append_summary(
            strategy_version_id=definition.strategy_version_id,
            product_id="active_income",
            observed_at="2026-08-26T00:00:02+00:00",
            artefact_hash=artefact.artefact_hash,
            evidence={
                "observation_ids": ["sha256:" + "f" * 64],
                "observed_from": "2026-08-24T00:00:00+00:00",
                "observed_until": "2026-08-26T00:00:00+00:00",
            },
        )
    with pytest.raises(CanonicalEvidenceError, match="artefact-bound"):
        repository.append_summary(
            strategy_version_id=definition.strategy_version_id,
            product_id="active_income",
            observed_at="2026-08-24T00:00:00+00:00",
            evidence={"accepted": True},
        )


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
