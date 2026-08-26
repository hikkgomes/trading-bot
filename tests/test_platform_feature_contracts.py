from __future__ import annotations

import datetime as dt
import statistics

import pytest

from src.data.binance_market import normalise_public_event
from src.data.database import PlatformDatabase
from src.data.feature_graph import (
    AvailableValue,
    FeatureGraphError,
    FeatureGraphRegistry,
    default_feature_engine,
)
from src.data.feature_store import SqlFeatureStore
from src.domain._codec import to_primitive
from src.domain.instruments import Instrument, MarketType
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.risk.engine import SqlRiskSnapshotStore
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.platform_smoke import _seed_strategy
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.scheduler import DatabaseJobQueue
from src.services.strategy_evaluator import DatabaseStrategyEvaluator

NOW = "2026-08-24T00:00:00+00:00"


def _inputs(closes: list[float]):
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in closes]
    return {
        "open": AvailableValue(closes[-1] - 0.5, NOW, NOW),
        "close": AvailableValue(closes[-1], NOW, NOW),
        "high": AvailableValue(highs[-1], NOW, NOW),
        "low": AvailableValue(lows[-1], NOW, NOW),
        "close_history": AvailableValue(closes, NOW, NOW),
        "high_history": AvailableValue(highs, NOW, NOW),
        "low_history": AvailableValue(lows, NOW, NOW),
    }


def test_real_indicators_use_full_history_and_handle_flat_series() -> None:
    registry = FeatureGraphRegistry.default()
    graph = registry.graph(("sma", "rsi", "macd"))
    engine = default_feature_engine()

    rising = engine.evaluate(
        graph, information_timestamp=NOW, inputs=_inputs([100.0 + i for i in range(100)])
    )
    flat = engine.evaluate(graph, information_timestamp=NOW, inputs=_inputs([100.0] * 100))

    assert rising["sma"] == statistics.fmean(range(180, 200))
    assert rising["rsi"] > 50.0
    assert rising["macd"] > 0.0
    assert flat["sma"] == 100.0
    assert flat["rsi"] == 50.0
    assert flat["macd"] == 0.0


def test_real_atr_adx_and_supertrend_use_historical_ranges() -> None:
    registry = FeatureGraphRegistry.default()
    graph = registry.graph(("atr", "adx", "supertrend"))
    values = default_feature_engine().evaluate(
        graph,
        information_timestamp=NOW,
        inputs=_inputs([100.0 + i * 0.25 for i in range(100)]),
    )

    assert values["atr"] > 0.0
    assert 0.0 <= values["adx"] <= 100.0
    assert values["supertrend"] != 0.0


def test_structured_live_inputs_are_typed_and_fail_closed() -> None:
    registry = FeatureGraphRegistry.default()
    graph = registry.graph(("cross_sectional_rank", "funding", "open_interest"))
    inputs = {
        "cross_sectional_values": AvailableValue(
            {"BTCUSDT": 0.4, "ETHUSDT": -0.2, "__current__": 0.4}, NOW, NOW
        ),
        "funding_rate": AvailableValue(0.001, NOW, NOW),
        "open_interest": AvailableValue(10_000.0, NOW, NOW),
    }
    values = default_feature_engine().evaluate(graph, information_timestamp=NOW, inputs=inputs)
    assert values["cross_sectional_rank"] == 1.0
    assert values["funding"] == 0.001
    assert values["open_interest"] == 10_000.0
    with pytest.raises(FeatureGraphError, match="at least 2 mapping values"):
        default_feature_engine().evaluate(
            graph,
            information_timestamp=NOW,
            inputs={
                **inputs,
                "cross_sectional_values": AvailableValue({"__current__": 0.4}, NOW, NOW),
            },
        )


def test_correlation_and_beta_require_point_in_time_return_histories() -> None:
    graph = FeatureGraphRegistry.default().graph(("beta", "correlation"))
    values = default_feature_engine().evaluate(
        graph,
        information_timestamp=NOW,
        inputs={
            "asset_returns": AvailableValue([0.01, 0.02, 0.03], NOW, NOW),
            "benchmark_returns": AvailableValue([0.01, 0.01, 0.02], NOW, NOW),
        },
    )

    assert values["beta"] > 0.0
    assert 0.0 < values["correlation"] <= 1.0

    with pytest.raises(FeatureGraphError, match="at least 2 historical values"):
        default_feature_engine().evaluate(
            graph,
            information_timestamp=NOW,
            inputs={
                "asset_returns": AvailableValue([0.01], NOW, NOW),
                "benchmark_returns": AvailableValue([0.01], NOW, NOW),
            },
        )


def test_feature_worker_expands_frozen_ml_vector_into_manifest_features(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'ml-features.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-feature",
        node_id="linux-optiplex",
        role="feature-worker",
        capabilities=("historical_feature_calculation",),
        observed_at=NOW,
    )
    queue.register_worker(
        worker_id="linux-strategy",
        node_id="linux-optiplex",
        role="strategy-evaluator",
        capabilities=("strategy_evaluation",),
        observed_at=NOW,
    )
    queue.enqueue(
        job_id="ml-feature-job",
        name="historical_feature_calculation",
        payload={
            "feature_set_version": "core-bars-v1",
            "instrument_id": "binance:futures:BTCUSDT",
            "source_event_time": NOW,
            "source_close_time": NOW,
            "availability_time": NOW,
            "source_market_event_id": "sha256:" + "b" * 64,
            "inputs": {
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 50.0,
                "feature_vector": {"momentum": 0.2, "funding": -0.01},
            },
            "producer_identity": "historical-test",
        },
        available_at=NOW,
    )
    worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=SqlFeatureStore(database.engine),
        job_names=("historical_feature_calculation",),
        active_assignments=lambda _instrument_id: (
            {
                "id": "sha256:" + "a" * 64,
                "product_id": "active_income",
                "instrument_id": "binance:futures:BTCUSDT",
            },
        ),
        feature_graph_for_assignment=lambda _assignment: {
            "required_nodes": ["frozen_ml_feature_vector"]
        },
    )

    result = worker.run_once(now=NOW)

    assert result["reason_code"] == "features_persisted", result
    values = SqlFeatureStore(database.engine).available(
        instrument_id="binance:futures:BTCUSDT",
        at=NOW,
        feature_set_version="core-bars-v1",
    )
    assert {item.feature_name for item in values} >= {"momentum", "funding"}
    assert "frozen_ml_feature_vector" not in {item.feature_name for item in values}
    strategy_job = queue.claim(
        worker_id="linux-strategy",
        now=NOW,
        lease_seconds=60,
        names=("strategy_evaluation",),
    )
    assert strategy_job is not None
    assert len(strategy_job.payload["feature_ids"]) == 2


def test_market_data_writer_publishes_authoritative_market_snapshot(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-data",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
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
                "spread_bps": "1.5",
                "visible_depth": "1000",
                "volatility": "0.2",
                "funding": "0.001",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id="market-snapshot-1",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=NOW,
    )
    result = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    ).run_once(now=NOW)

    assert result["market_snapshot_ids"]
    _snapshot_id, snapshot = SqlRiskSnapshotStore(database.engine).latest(
        kind="market_data_input", product_id="active_income", at=NOW
    )
    assert snapshot["values"] == {
        "close": 102.0,
        "spread_bps": 1.5,
        "visible_depth": 1000.0,
        "volatility": 0.2,
        "funding": 0.001,
    }


def test_market_data_writer_publishes_partial_book_snapshot(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'book.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="linux-data",
        node_id="linux-optiplex",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
    )
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@bookTicker",
        receive_timestamp=NOW,
        payload={
            "e": "bookTicker",
            "E": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
            "s": "BTCUSDT",
            "b": "99.5",
            "B": "12",
            "a": "100.5",
            "A": "8",
        },
    )
    queue.enqueue(
        job_id="book-snapshot-1",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(event),
        },
        available_at=NOW,
    )
    result = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    ).run_once(now=NOW)

    assert result["market_snapshot_ids"]
    _snapshot_id, snapshot = SqlRiskSnapshotStore(database.engine).latest(
        kind="market_data_input", product_id="active_income", at=NOW
    )
    assert snapshot["values"] == {
        "close": 100.0,
        "spread_bps": 100.0,
        "visible_depth": 20.0,
    }


def test_live_sma_chain_uses_100_bars_and_produces_active_and_flat_forecasts(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    start = dt.datetime(2026, 8, 23, tzinfo=dt.UTC)
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
            observed_at=start.isoformat(),
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
    product = {
        "product_id": "active_income",
        "portfolio_id": "active-income-portfolio",
        "account_id": "futures",
        "risk_policy_id": "active-income",
        "base_accounting_asset": "USDT",
    }
    _seed_strategy(
        database,
        product,
        instrument,
        "sha256:" + "1" * 64,
        start.isoformat(),
        "live-sma-chain",
        strategy_name="sma_cross",
    )
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    graph_enabled = False
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
            for item in assignments.active_assignments("active_income")
            if item.get("instrument_id") == instrument_id and graph_enabled
        ),
        feature_graph_for_assignment=lambda _assignment: {
            "required_nodes": ["sma_fast", "sma_slow", "rsi", "macd"]
        },
    )
    strategy_worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id="linux-strategy",
        feature_store=feature_store,
        portfolio=SqlPortfolioRepository(database.engine, require_pipeline_identity=True),
        assignments=assignments,
    )
    forecasts = []
    last_time = start
    for index in range(150):
        last_time = start + dt.timedelta(minutes=index + 1)
        close = 100.0 + index * 0.1 if index < 100 else 110.0
        close_ms = int(last_time.timestamp() * 1_000)
        event = normalise_public_event(
            market="futures",
            stream="btcusdt@kline_1m",
            receive_timestamp=last_time.isoformat(),
            payload={
                "e": "kline",
                "E": close_ms + 1,
                "s": "BTCUSDT",
                "k": {
                    "t": close_ms - 59_999,
                    "T": close_ms,
                    "i": "1m",
                    "o": str(close - 0.05),
                    "h": str(close + 0.1),
                    "l": str(close - 0.1),
                    "c": str(close),
                    "v": "25",
                    "x": True,
                },
            },
        )
        queue.enqueue(
            job_id=f"live-sma-event:{index}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": "futures",
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=last_time.isoformat(),
        )
        assert writer.run_once(now=last_time.isoformat())["reason_code"] == "market_event_written"
        if index == 99:
            graph_enabled = True
        featured = feature_worker.run_once(now=last_time.isoformat())
        assert featured["reason_code"] == "features_persisted"
        if graph_enabled:
            evaluated = strategy_worker.run_once(now=last_time.isoformat())
            assert evaluated["reason_code"] == "alpha_forecast_persisted"
            forecasts.append(
                SqlPortfolioRepository(database.engine).forecast(evaluated["forecast_id"])
            )

    assert (
        len(
            feature_store.available(
                instrument_id=instrument.instrument_id,
                at=(last_time + dt.timedelta(minutes=1)).isoformat(),
                feature_set_version="core-bars-v1",
            )
        )
        >= 150
    )
    assert forecasts[0].direction.value == "long"
    assert forecasts[-1].direction.value == "flat"
    feature_values = feature_store.available(
        instrument_id=instrument.instrument_id,
        at=(last_time + dt.timedelta(minutes=1)).isoformat(),
        feature_set_version="core-bars-v1",
    )
    rsi_values = [item.value for item in feature_values if item.feature_name == "rsi"]
    macd_values = [item.value for item in feature_values if item.feature_name == "macd"]
    assert rsi_values and max(rsi_values) > 50.0
    assert macd_values and max(macd_values) > 0.0
