from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert

from src.data.binance_market import normalise_public_event
from src.data.database import PlatformDatabase, strategy_definition, strategy_version
from src.data.feature_store import SqlFeatureStore
from src.domain._codec import canonical_hash, to_primitive
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.instruments import Instrument, MarketType
from src.domain.market_events import MarketEvent, MarketEventType
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlStrategyArtefactRepository,
)
from src.risk.engine import SqlRiskSnapshotStore
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue
from src.services.strategy_evaluator import DatabaseStrategyEvaluator

NOW = dt.datetime(2026, 8, 25, tzinfo=dt.UTC).isoformat()
PROMOTABLE_SOURCE_TYPES = tuple(StrategySourceType)


def _source_contract(
    source_type: StrategySourceType,
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]:
    if source_type is StrategySourceType.GENERATED_DSL:
        return (
            ("bar_return",),
            {"rule": {"feature": "bar_return", "operator": "gt", "threshold": 0.0}},
            {},
        )
    if source_type is StrategySourceType.MACHINE_LEARNING:
        return (
            ("bar_return",),
            {"feature_names": ["bar_return"], "weights": [1.0], "intercept": 0.0},
            {},
        )
    if source_type is StrategySourceType.CROSS_SECTIONAL:
        return (("cross_sectional_rank",), {"production_rule": {}}, {})
    if source_type is StrategySourceType.RELATIVE_VALUE:
        return (("spot_perpetual_basis",), {"production_rule": {}}, {})
    if source_type is StrategySourceType.MICROSTRUCTURE:
        return (("depth_imbalance", "aggressor_flow"), {"production_rule": {}}, {})
    if source_type is StrategySourceType.ENSEMBLE:
        return (("bar_return",), {"production_rule": {}}, {})
    metadata = (
        {"sandbox_receipt": "sha256:" + "a" * 64}
        if source_type is StrategySourceType.AGENT_GENERATED_PYTHON
        else (
            {"derived_from": "sha256:" + "b" * 64}
            if source_type
            in {
                StrategySourceType.PARAMETER_SEARCH,
                StrategySourceType.MUTATION,
                StrategySourceType.CROSSOVER,
            }
            else {}
        )
    )
    return (
        ("bar_return",),
        {
            "production_rule": {
                "kind": "linear_feature_score/v1",
                "terms": [{"feature": "bar_return", "scale": 1.0, "weight": 1.0}],
            }
        },
        metadata,
    )


def _candle_event(
    *, market: str, symbol: str, close_ms: int, open_price: float, close_price: float
) -> MarketEvent:
    return normalise_public_event(
        market=market,
        stream=f"{symbol.lower()}@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": symbol,
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": str(open_price),
                "h": str(max(open_price, close_price) + 1.0),
                "l": str(min(open_price, close_price) - 1.0),
                "c": str(close_price),
                "v": "25",
                "x": True,
            },
        },
    )


def _support_events(
    source_type: StrategySourceType, base_ms: int
) -> tuple[tuple[str, str, MarketEvent], ...]:
    if source_type is StrategySourceType.CROSS_SECTIONAL:
        return tuple(
            (
                "futures",
                symbol,
                _candle_event(
                    market="futures",
                    symbol=symbol,
                    close_ms=base_ms - offset * 60_000,
                    open_price=close,
                    close_price=close,
                ),
            )
            for symbol, prices in (("BTCUSDT", (100.0, 101.0)), ("ETHUSDT", (200.0, 198.0)))
            for offset, close in zip((3, 2), prices, strict=True)
        )
    if source_type is StrategySourceType.RELATIVE_VALUE:
        return (
            (
                "spot",
                "BTCUSDT",
                _candle_event(
                    market="spot",
                    symbol="BTCUSDT",
                    close_ms=base_ms - 60_000,
                    open_price=99.0,
                    close_price=100.0,
                ),
            ),
            (
                "futures",
                "BTCUSDT",
                _candle_event(
                    market="futures",
                    symbol="BTCUSDT",
                    close_ms=base_ms - 60_000,
                    open_price=100.0,
                    close_price=101.0,
                ),
            ),
        )
    if source_type is StrategySourceType.MICROSTRUCTURE:
        book = normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=NOW,
            payload={
                "e": "bookTicker",
                "E": base_ms - 2_000,
                "s": "BTCUSDT",
                "b": "99.5",
                "B": "12",
                "a": "100.5",
                "A": "8",
            },
        )
        trade = normalise_public_event(
            market="futures",
            stream="btcusdt@aggTrade",
            receive_timestamp=NOW,
            payload={
                "e": "aggTrade",
                "E": base_ms - 1_000,
                "s": "BTCUSDT",
                "p": "100",
                "q": "4",
                "m": False,
            },
        )
        return (("futures", "BTCUSDT", book), ("futures", "BTCUSDT", trade))
    if source_type is StrategySourceType.MACHINE_LEARNING:
        manifest = normalise_public_event(
            market="futures",
            stream="btcusdt@aggTrade",
            receive_timestamp=NOW,
            payload={
                "e": "aggTrade",
                "E": base_ms - 1_000,
                "s": "BTCUSDT",
                "p": "100",
                "q": "4",
                "m": False,
                "feature_vector": {"bar_return": 0.01},
            },
        )
        return (("futures", "BTCUSDT", manifest),)
    return ()


@pytest.mark.parametrize("source_type", PROMOTABLE_SOURCE_TYPES)
def test_every_promotable_source_type_runs_a_complete_paper_service_chain(
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
    required_nodes, source_signal_model, metadata = _source_contract(source_type)
    definition = StrategyDefinition(
        identity=f"source-chain:{source_type.value}",
        version="paper-chain-v1",
        family="time_series",
        product=product_id,
        universe={"symbols": [instrument.exchange_symbol]},
        data_requirements={"bars": "1m", "closed_only": True},
        feature_graph={"version": "core-bars-v1", "required_nodes": list(required_nodes)},
        signal_model=source_signal_model,
        position_model={"kind": "volatility_scaled"},
        execution_preferences={"policy": "market"},
        risk_policy={"id": "paper-chain-risk"},
        validation_policy={"id": "paper-chain-validation"},
        source_type=source_type,
        source_hash=canonical_hash({"source_type": source_type.value, "product": product_id}),
        metadata=metadata,
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
    feature_store = SqlFeatureStore(database.engine)
    graph_enabled = False
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
    )
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=feature_store,
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        active_assignments=lambda instrument_id: tuple(
            item
            for item in assignments.active_assignments(product_id)
            if graph_enabled and item.get("instrument_id") == instrument_id
        ),
        feature_graph_for_assignment=lambda _assignment: {"required_nodes": list(required_nodes)},
    )

    def write_event(market: str, symbol: str, event: MarketEvent, label: str) -> None:
        queue.enqueue(
            job_id=f"source-chain:{source_type.value}:{label}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": market,
                "symbol": symbol,
                "event": to_primitive(event),
            },
            available_at=NOW,
        )
        written = writer.run_once(now=NOW)
        assert written["reason_code"] == "market_event_written"
        if event.event_type is MarketEventType.CANDLE:
            featured = feature_worker.run_once(now=NOW)
            assert featured["reason_code"] == "features_persisted"

    base_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1
    for index, (market, symbol, event) in enumerate(_support_events(source_type, base_ms)):
        write_event(market, symbol, event, f"support-{index}")

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
    graph_enabled = True
    if source_type is StrategySourceType.ENSEMBLE:
        SqlPortfolioRepository(database.engine).save_forecast(
            AlphaForecast(
                strategy_version_id="upstream-strategy",
                product_id=product_id,
                instrument_id=instrument.instrument_id,
                direction=ForecastDirection.LONG,
                score=0.5,
                expected_return=0.01,
                confidence=0.8,
                horizon_seconds=60,
                valid_from=NOW,
                valid_until="2026-08-25T01:00:00+00:00",
                target_volatility=0.1,
                maximum_position=0.1,
            )
        )
    final_event = _candle_event(
        market="futures",
        symbol="BTCUSDT",
        close_ms=base_ms,
        open_price=100.0,
        close_price=102.0,
    )
    write_event("futures", "BTCUSDT", final_event, "final")

    strategy_worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id="linux-strategy",
        feature_store=feature_store,
        portfolio=SqlPortfolioRepository(database.engine, require_pipeline_identity=True),
        assignments=assignments,
        snapshot_store=SqlRiskSnapshotStore(database.engine),
    )
    evaluated = strategy_worker.run_once(now=NOW)

    assert evaluated["reason_code"] == "alpha_forecast_persisted"
    forecast = SqlPortfolioRepository(database.engine, require_pipeline_identity=True).forecast(
        evaluated["forecast_id"]
    )
    assert forecast.direction.value in {"long", "short"}
    assert forecast.metadata["assignment_id"] == assignment_id
    assert forecast.metadata["execution_receipt"]["source_type"] == source_type.value
