from __future__ import annotations

import datetime as dt
import json
import statistics

import pytest
from sqlalchemy import select

from src.data.binance_market import normalise_public_event
from src.data.database import PlatformDatabase, job, market_input_buffer, risk_snapshot
from src.data.feature_graph import (
    AvailableValue,
    FeatureGraphError,
    FeatureGraphRegistry,
    default_feature_engine,
)
from src.data.feature_store import SqlFeatureStore
from src.data.parquet_store import partition_files
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


def test_gateway_keeps_latest_quote_in_each_flush_and_ignores_delayed_updates(tmp_path):
    from src.data.parquet_store import DurableMarketBatchSpool
    from src.services.data_writer import market_input_snapshot_rows
    from src.services.market_gateway import DatabaseGatewaySink

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'quote-coalescing.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    base = dt.datetime.now(dt.UTC).replace(microsecond=0)
    later = (base + dt.timedelta(seconds=3)).isoformat()
    queue.register_worker(
        worker_id="writer",
        node_id="fixture",
        role="writer",
        capabilities=("market_event_write", "market_event_batch_write"),
        observed_at=NOW,
    )
    spool = DurableMarketBatchSpool(tmp_path / "legacy-quote-spool", max_rows=1, max_bytes=10_000)
    sink = DatabaseGatewaySink(queue, spool=spool, coalesce_quotes=True)
    for offset in (1, 2, 0):
        observed = base + dt.timedelta(seconds=offset)
        event = normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=later,
            payload={
                "e": "bookTicker",
                "E": int(observed.timestamp() * 1000),
                "s": "BTCUSDT",
                "b": str(100 + offset),
                "a": str(101 + offset),
                "B": "10",
                "A": "20",
            },
        )
        sink._write_market_event(event, market="futures", symbol="BTCUSDT")
    assert queue.claim(worker_id="writer", now=later, lease_seconds=60) is None
    sink.flush()
    assert queue.claim(worker_id="writer", now=later, lease_seconds=60) is None
    quote_batch = sink.take_quote_batch()
    assert len(quote_batch) == 1
    market, symbol, event = quote_batch[0]
    assert (market, symbol) == ("futures", "BTCUSDT")
    assert event.payload["data"]["b"] == "102"
    assert event.payload["capture_policy"] == "latest_quote_per_flush/v1"
    inputs = market_input_snapshot_rows(event, ("active_income",))
    identities = SqlRiskSnapshotStore(database.engine).save_market_inputs(
        inputs, retention_at=later
    )
    assert len(identities) == 1
    persisted = SqlRiskSnapshotStore(database.engine).get(identities[0])
    assert persisted["source_event_id"] == event.event_id
    assert persisted["source_payload"]["capture_policy"] == "latest_quote_per_flush/v1"

    legacy_segment = spool.append(event, venue="binance", market="futures", symbol="BTCUSDT")
    assert legacy_segment is not None
    sink._enqueue_segment(legacy_segment)
    result = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="writer",
        root=tmp_path / "legacy-quote-market",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    ).run_once(now=later)
    assert result["reason_code"] == "market_event_batch_written"
    assert result["market_snapshot_ids"] == list(identities)
    database.dispose()


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
    market_events = (
        normalise_public_event(
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
        ),
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=NOW,
            payload={
                "e": "bookTicker",
                "E": close_ms + 2,
                "s": "BTCUSDT",
                "b": "101.99",
                "B": "500",
                "a": "102.01",
                "A": "500",
            },
        ),
        normalise_public_event(
            market="futures",
            stream="btcusdt@markPrice@1s",
            receive_timestamp=NOW,
            payload={
                "e": "markPriceUpdate",
                "E": close_ms + 3,
                "s": "BTCUSDT",
                "p": "102",
                "r": "0.001",
                "T": close_ms + 3,
            },
        ),
    )
    for index, event in enumerate(market_events):
        queue.enqueue(
            job_id=f"market-snapshot-{index}",
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
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    )
    results = [writer.run_once(now=NOW) for _ in market_events]

    assert all(result["market_snapshot_ids"] for result in results)
    assert partition_files(writer.store.root, "raw/binance/futures/best_bid_ask/**/*.parquet") == ()
    assert partition_files(writer.store.root, "raw/binance/futures/mark_price/**/*.parquet") == ()
    assert partition_files(writer.store.root, "raw/binance/futures/candle/**/*.parquet")
    with database.engine.connect() as connection:
        immutable_ids = set(connection.execute(select(risk_snapshot.c.id)).scalars())
        buffered_ids = set(connection.execute(select(market_input_buffer.c.id)).scalars())
    assert results[0]["market_snapshot_ids"][0] in immutable_ids
    assert {result["market_snapshot_ids"][0] for result in results[1:]} <= buffered_ids
    snapshots = [
        SqlRiskSnapshotStore(database.engine).get(result["market_snapshot_ids"][0])
        for result in results
    ]
    assert snapshots[0]["values"] == {
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 25.0,
    }
    assert snapshots[1]["values"] == {
        "close": 102.0,
        "spread_bps": pytest.approx(1.9607843137),
        "visible_depth_quote": 50995.0,
        "bid_depth_quote": 50995.0,
        "ask_depth_quote": 51005.0,
    }
    assert snapshots[2]["values"] == {"close": 102.0, "funding": 0.001}


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
        "visible_depth_quote": 804.0,
        "bid_depth_quote": 1194.0,
        "ask_depth_quote": 804.0,
    }


def test_feature_enqueue_pins_only_asof_microstructure_inputs(tmp_path, monkeypatch) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'feature-market-inputs.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    for worker_id, role, capability in (
        ("linux-data", "data-writer", "market_event_write"),
        ("linux-feature", "feature-service", "live_feature_calculation"),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="fixture",
            role=role,
            capabilities=(capability,),
            observed_at=NOW,
        )
    store = SqlRiskSnapshotStore(database.engine)
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
        snapshot_store=store,
        product_ids_by_market={"futures": ("active_income",)},
    )
    candle_close_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1
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
    _seed_strategy(
        database,
        {
            "product_id": "active_income",
            "portfolio_id": "active-income-portfolio",
            "account_id": "futures",
            "risk_policy_id": "active-income",
            "promotion_policy_id": "active-income",
            "base_accounting_asset": "USDT",
        },
        instrument,
        "sha256:" + "1" * 64,
        NOW,
        "core-only-feature-scope",
        strategy_name="sma_cross",
    )
    assert (
        writer._active_feature_input_families(
            event=normalise_public_event(
                market="futures",
                stream="btcusdt@bookTicker",
                receive_timestamp=NOW,
                payload={
                    "e": "bookTicker",
                    "E": candle_close_ms,
                    "s": "BTCUSDT",
                    "b": "99",
                    "a": "100",
                    "B": "1",
                    "A": "1",
                },
            ),
            market="futures",
        )
        == set()
    )
    monkeypatch.setattr(
        writer,
        "_active_feature_input_families",
        lambda **_kwargs: {"order_book", "funding_open_interest"},
    )
    events = (
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=NOW,
            payload={
                "e": "bookTicker",
                "E": candle_close_ms - 2_000,
                "s": "BTCUSDT",
                "b": "99.9",
                "B": "12",
                "a": "100.1",
                "A": "8",
            },
        ),
        normalise_public_event(
            market="futures",
            stream="btcusdt@markPrice@1s",
            receive_timestamp=NOW,
            payload={
                "e": "markPriceUpdate",
                "E": candle_close_ms - 1_000,
                "s": "BTCUSDT",
                "p": "100",
                "r": "0.001",
                "T": candle_close_ms - 1_000,
            },
        ),
        normalise_public_event(
            market="futures",
            stream="btcusdt@bookTicker",
            receive_timestamp=NOW,
            payload={
                "e": "bookTicker",
                "E": candle_close_ms + 1_000,
                "s": "BTCUSDT",
                "b": "109.9",
                "B": "12",
                "a": "110.1",
                "A": "8",
            },
        ),
    )
    for index, event in enumerate(events):
        queue.enqueue(
            job_id=f"microstructure-{index}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": "futures",
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=NOW,
        )
        assert writer.run_once(now=NOW)["reason_code"] == "market_event_written"
    candle = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": candle_close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": candle_close_ms - 59_999,
                "T": candle_close_ms,
                "i": "1m",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100",
                "v": "1",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id="microstructure-candle",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(candle),
        },
        available_at=NOW,
    )
    result = writer.run_once(now=NOW)
    assert result["reason_code"] == "market_event_written"

    with database.engine.connect() as connection:
        payload = connection.execute(
            select(job.c.payload).where(job.c.id == result["feature_job_id"])
        ).scalar_one()
        buffered_ids = set(connection.execute(select(market_input_buffer.c.id)).scalars())
    references = payload["input_references"]
    book_ids = references["order_book"]["source_snapshot_ids"]
    funding_ids = references["funding_open_interest"]["source_snapshot_ids"]
    assert len(book_ids) == 1
    assert len(funding_ids) == 1
    book = store.get(book_ids[0])
    funding = store.get(funding_ids[0])
    assert book["source_payload"]["data"]["b"] == "99.9"
    assert book["source_sequence"] == events[0].sequence
    assert book["source_event_time"] == events[0].exchange_timestamp
    assert book["source_receive_time"] == events[0].receive_timestamp
    assert funding["source_payload"]["data"]["r"] == "0.001"
    assert events[2].event_id not in {book["source_event_id"], funding["source_event_id"]}
    assert result["feature_job_id"] is not None

    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=SqlFeatureStore(database.engine),
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
        snapshot_store=store,
    )
    decoded = feature_worker._resolve_input_references(
        payload,
        required_families={"order_book", "funding_open_interest"},
        history_limit=1,
    )
    assert decoded["bid_price"] == pytest.approx(99.9)
    assert decoded["ask_price"] == pytest.approx(100.1)
    assert decoded["funding_rate"] == pytest.approx(0.001)
    core_only = feature_worker._resolve_input_references(
        payload, required_families=set(), history_limit=1
    )
    assert "bid_price" not in core_only
    assert "funding_rate" not in core_only
    assert set(book_ids + funding_ids).isdisjoint(buffered_ids)
    assert partition_files(writer.store.root, "raw/binance/futures/best_bid_ask/**/*.parquet") == ()
    assert partition_files(writer.store.root, "raw/binance/futures/mark_price/**/*.parquet") == ()
    assert partition_files(writer.store.root, "raw/binance/futures/candle/**/*.parquet")
    database.dispose()


def test_live_sma_chain_holds_entry_until_configured_horizon(tmp_path) -> None:
    from src.domain.orders import Fill, OrderIntent, OrderSide, OrderType
    from src.execution.order_manager import OrderManager, SqlOrderStore
    from src.services.financial_application import FinancialApplication, FinancialScope

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
        "promotion_policy_id": "active-income",
        "base_accounting_asset": "USDT",
    }
    assignment_id = _seed_strategy(
        database,
        product,
        instrument,
        "sha256:" + "1" * 64,
        start.isoformat(),
        "live-sma-chain",
        strategy_name="sma_cross",
    )
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    book_id = assignments.by_id(assignment_id)["payload"]["book_id"]
    snapshots = SqlRiskSnapshotStore(database.engine)
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
        snapshot_store=snapshots,
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
        snapshot_store=snapshots,
    )
    forecasts = []
    orders = OrderManager(SqlOrderStore(database.engine))
    application = FinancialApplication(
        database.engine, FinancialScope("active_income", book_id, "paper", "binance", "USDT")
    )
    held = False
    last_time = start
    for index in range(450):
        last_time = start + dt.timedelta(minutes=index + 1)
        close = 120.0 - index * 0.1 if index < 225 else 200.0
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
        if index == 225:
            graph_enabled = True
        featured = feature_worker.run_once(now=last_time.isoformat())
        assert featured["reason_code"] == "features_persisted"
        if graph_enabled:
            evaluated = strategy_worker.run_once(now=last_time.isoformat())
            assert evaluated["reason_code"] == "alpha_forecast_persisted"
            forecasts.append(
                SqlPortfolioRepository(database.engine).forecast(evaluated["forecast_id"])
            )
            forecast = forecasts[-1]
            if len(forecasts) == 1 or (held and forecast.direction.value == "flat"):
                side = OrderSide.SELL if held else OrderSide.BUY
                order = OrderIntent(
                    f"sma-order-{index}",
                    book_id,
                    instrument.instrument_id,
                    side,
                    1,
                    OrderType.MARKET,
                    created_at=last_time.isoformat(),
                    strategy_contributions={forecast.strategy_version_id: 1},
                    metadata={
                        "reference_price": close,
                        "target_metadata": {
                            "assignment_id": forecast.metadata["assignment_id"],
                            "account_id": book_id,
                            "book_id": book_id,
                            "execution_environment": "paper",
                        },
                    },
                )
                orders.create(order)
                application.apply_fill(
                    Fill(
                        f"sma-fill-{index}",
                        order.order_id,
                        instrument.instrument_id,
                        side,
                        1,
                        close,
                        0,
                        (last_time + dt.timedelta(milliseconds=1)).isoformat(),
                        fee_asset="USDT",
                    )
                )
                held = not held

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
    assert forecasts[0].metadata["parity_receipt"]["prior_position_event_id"] is None
    assert forecasts[1].metadata["parity_receipt"]["prior_position_event_id"]
    assert forecasts[50].direction.value == "long"
    horizon = forecasts[0].metadata["position_rules"]["horizon_bars"]
    assert forecasts[horizon - 1].direction.value == "long"
    assert forecasts[horizon].direction.value == "flat"
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


def test_closed_candle_auxiliary_filter_uses_close_and_availability_times() -> None:
    through = dt.datetime.fromisoformat(NOW)
    decision_time = through + dt.timedelta(seconds=3)
    through_ms = int(through.timestamp() * 1_000)
    rows = (
        ("closed_after_close", 0, True, 0),
        ("late_arrival", 0, True, 1_000),
        ("open_update", 0, False, 0),
        ("future_close", 1_000, True, 0),
    )

    resolved = {}
    for event_id, close_offset_ms, closed, availability_offset_ms in rows:
        close_ms = through_ms + close_offset_ms
        close_time = dt.datetime.fromtimestamp(close_ms / 1_000, dt.UTC)
        emitted_at = close_time + dt.timedelta(seconds=1)
        available_at = decision_time + dt.timedelta(milliseconds=availability_offset_ms)
        resolved[event_id] = DatabaseFeatureWorker._auxiliary_row(
            {
                "event_id": event_id,
                "instrument_id": "binance:futures:BTCUSDT:USDT",
                "event_type": "candle",
                "sequence": None,
                "exchange_timestamp": emitted_at.isoformat(),
                "availability_time": available_at.isoformat(),
                "payload_json": json.dumps(
                    {"data": {"k": {"T": close_ms, "x": closed, "c": "100"}}}
                ),
            },
            source_ids=set(),
            event_types={"candle"},
            instrument_id="binance:futures:BTCUSDT:USDT",
            same_instrument=True,
            available_at=decision_time,
            through=through,
        )

    accepted = resolved["closed_after_close"]
    assert accepted is not None
    assert accepted["source_time"] == through
    assert accepted["exchange_timestamp"] > through
    assert resolved["late_arrival"] is None
    assert resolved["open_update"] is None
    assert resolved["future_close"] is None


def test_snapshot_store_keeps_active_multitimeframe_candle_inputs(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'multitimeframe-inputs.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    for worker_id, role, capability in (
        ("linux-data", "data-writer", "market_event_write"),
        ("linux-feature", "feature-service", "live_feature_calculation"),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="fixture",
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
    product = {
        "product_id": "active_income",
        "portfolio_id": "active-income-portfolio",
        "account_id": "futures",
        "risk_policy_id": "active-income",
        "promotion_policy_id": "active-income",
        "base_accounting_asset": "USDT",
    }
    _seed_strategy(
        database,
        product,
        instrument,
        "multi-timeframe-feature-scope",
        NOW,
        "multi-timeframe-feature-scope",
        strategy_name="multi_tf_trend",
    )
    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    snapshots = SqlRiskSnapshotStore(database.engine)
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="linux-data",
        root=tmp_path / "data",
        snapshot_store=snapshots,
        product_ids_by_market={"futures": ("active_income",)},
    )
    target_close_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 1
    for close_ms, close in ((target_close_ms - 600_000, 100), (target_close_ms - 300_000, 110)):
        event = normalise_public_event(
            market="futures",
            stream="btcusdt@kline_5m",
            receive_timestamp=NOW,
            payload={
                "e": "kline",
                "E": close_ms + 1,
                "s": "BTCUSDT",
                "k": {
                    "t": close_ms - 299_999,
                    "T": close_ms,
                    "i": "5m",
                    "o": str(close - 1),
                    "h": str(close + 1),
                    "l": str(close - 1),
                    "c": str(close),
                    "v": "10",
                    "x": True,
                },
            },
        )
        writer.store.put(event, venue="binance", market="futures", symbol="BTCUSDT")
    target = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": target_close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": target_close_ms - 59_999,
                "T": target_close_ms,
                "i": "1m",
                "o": "119",
                "h": "121",
                "l": "119",
                "c": "120",
                "v": "5",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id="multitimeframe-target-candle",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": to_primitive(target),
        },
        available_at=NOW,
    )
    result = writer.run_once(now=NOW)
    assert result["reason_code"] == "market_event_written"
    with database.engine.connect() as connection:
        payload = connection.execute(
            select(job.c.payload).where(job.c.id == result["feature_job_id"])
        ).scalar_one()
    higher_timeframe = payload["input_references"]["higher_timeframe"]
    assert higher_timeframe["relative_pattern"].startswith("raw/binance/futures/")
    feature_worker = DatabaseFeatureWorker(
        queue=queue,
        worker_id="linux-feature",
        store=SqlFeatureStore(database.engine),
        job_names=("live_feature_calculation",),
        parquet_root=tmp_path / "data",
        snapshot_store=snapshots,
        active_assignments=lambda instrument_id: tuple(
            assignment
            for assignment in assignments.active_assignments("active_income", at=NOW)
            if assignment.get("instrument_id") == instrument_id
        ),
        feature_graph_for_assignment=lambda _assignment: {"required_nodes": ["multi_timeframe"]},
    )
    resolved = feature_worker._resolve_input_references(
        payload, required_families={"higher_timeframe"}, history_limit=1
    )
    assert resolved["higher_timeframe_return"] == pytest.approx(120 / 110 - 1)
    assert feature_worker.run_once(now=NOW)["reason_code"] == "features_persisted"
    database.dispose()
