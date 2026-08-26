from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import insert

from src.data.binance_market import normalise_public_event
from src.data.database import PlatformDatabase, strategy_definition, strategy_version
from src.data.feature_store import SqlFeatureStore
from src.domain._codec import canonical_hash, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlStrategyArtefactRepository,
)
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue
from src.services.strategy_evaluator import DatabaseStrategyEvaluator

NOW = dt.datetime(2026, 8, 25, tzinfo=dt.UTC).isoformat()
AUTONOMOUS_SOURCE_TYPES = (
    StrategySourceType.PARAMETER_SEARCH,
    StrategySourceType.MUTATION,
    StrategySourceType.CROSSOVER,
    StrategySourceType.AGENT_GENERATED_PYTHON,
)


@pytest.mark.parametrize("source_type", AUTONOMOUS_SOURCE_TYPES)
def test_every_autonomous_source_type_runs_a_complete_paper_service_chain(
    tmp_path: Path, source_type: StrategySourceType
) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    for worker_id, role, capability in (
        ("linux-data", "data-writer", "market_event_write"),
        ("linux-feature", "feature-service", "live_feature_calculation"),
        ("linux-strategy", "strategy-evaluator", "strategy_evaluation"),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex",
            role=role,
            capabilities=(capability,),
            observed_at=NOW,
        )

    instrument = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=3,
        minimum_quantity=0.001,
        minimum_notional=5.0,
    )
    product_id = f"paper_{source_type.value}"
    portfolio_id = f"portfolio_{source_type.value}"
    account_id = f"account_{source_type.value}"
    definition = StrategyDefinition(
        identity=f"source-chain:{source_type.value}",
        version="paper-chain-v1",
        family="time_series",
        product=product_id,
        universe={"symbols": [instrument.exchange_symbol]},
        data_requirements={"bars": "1m", "closed_only": True},
        feature_graph={"version": "core-bars-v1", "required_nodes": ["bar_return"]},
        signal_model={
            "production_rule": {
                "kind": "linear_feature_score/v1",
                "terms": [{"feature": "bar_return", "scale": 1.0, "weight": 1.0}],
            }
        },
        position_model={"kind": "volatility_scaled"},
        execution_preferences={"policy": "market"},
        risk_policy={"id": "paper-chain-risk"},
        validation_policy={"id": "paper-chain-validation"},
        source_type=source_type,
        source_hash=canonical_hash({"source_type": source_type.value, "product": product_id}),
        metadata=(
            {"sandbox_receipt": "sha256:" + "a" * 64}
            if source_type is StrategySourceType.AGENT_GENERATED_PYTHON
            else {"derived_from": "sha256:" + "b" * 64}
        ),
    )
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=definition.definition_hash,
                identity=definition.identity,
                product_id=product_id,
                source_type=source_type.value,
                source_hash=definition.source_hash,
                definition=to_primitive(definition),
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id=definition.strategy_version_id,
                definition_id=definition.definition_hash,
                version=definition.version,
                created_at=NOW,
                payload={"definition_hash": definition.definition_hash},
            )
        )
    artefact = StrategyArtefact(
        definition=definition,
        dependency_hash=canonical_hash({"dependencies": "paper-chain"}),
        dataset_snapshot_hashes=(canonical_hash({"dataset": product_id}),),
        feature_set_version="core-bars-v1",
        cost_model_version="paper-chain-costs-v1",
        validation_evidence={"accepted": True},
        holdout_claim={"accepted": True},
        forward_evidence={"accepted": True},
        promotion_policy={"paper": True},
        position_limits={"maximum_position": 0.1, "target_volatility": 0.1},
        risk_limits={"risk_policy_id": "paper-chain-risk"},
        model_hashes=(),
        supported_products=(product_id,),
        supported_instruments=(instrument.instrument_id,),
        created_at=NOW,
        authoritative_evidence={"paper_chain": True},
        product_id=product_id,
        portfolio_id=portfolio_id,
        account_id=account_id,
        promotion_policy_id="paper-chain-policy",
        engine_version="strategy-evaluator/v1",
    )
    SqlStrategyArtefactRepository(database.engine).put(
        artefact.artefact_hash, artefact.to_dict(), created_at=NOW
    )
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    assignment_id = assignments.assign(
        product_id=product_id,
        portfolio_id=portfolio_id,
        strategy_version_id=definition.strategy_version_id,
        artefact_hash=artefact.artefact_hash,
        lifecycle_state="forward_paper",
        execution_mode="paper",
        capital_limit=0.1,
        risk_budget=0.1,
        assigned_at=NOW,
        assigned_by="paper-chain-test",
        assignment_reason="source dispatch coverage",
        instrument_id=instrument.instrument_id,
    )

    close_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": "100",
                "h": "103",
                "l": "99",
                "c": "102",
                "v": "25",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id=f"source-chain:{source_type.value}",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=NOW,
    )
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
    )
    feature_store = SqlFeatureStore(database.engine)
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=feature_store,
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
        active_assignments=lambda instrument_id: tuple(
            item
            for item in assignments.active_assignments(product_id)
            if item.get("instrument_id") == instrument_id
        ),
        feature_graph_for_assignment=lambda _assignment: {"required_nodes": ["bar_return"]},
    )
    strategy_worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id="linux-strategy",
        feature_store=feature_store,
        portfolio=SqlPortfolioRepository(database.engine, require_pipeline_identity=True),
        assignments=assignments,
    )

    written = writer.run_once(now=NOW)
    featured = feature_worker.run_once(now=NOW)
    evaluated = strategy_worker.run_once(now=NOW)

    assert written["reason_code"] == "market_event_written"
    assert written["bar_path"]
    assert featured["reason_code"] == "features_persisted"
    assert featured["features"] >= 1
    assert evaluated["reason_code"] == "alpha_forecast_persisted"
    forecast = SqlPortfolioRepository(database.engine, require_pipeline_identity=True).forecast(
        evaluated["forecast_id"]
    )
    assert forecast.direction.value == "long"
    assert forecast.metadata["assignment_id"] == assignment_id
    assert forecast.metadata["execution_receipt"]["source_type"] == source_type.value
