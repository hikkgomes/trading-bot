from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from sqlalchemy import insert, select, update

from src.accounting.paper_account import PaperAccountLedger
from src.agents.openclaw_bridge import OpenClawAgentBridge
from src.agents.store import SqlAgentStore
from src.data.binance_market import normalise_public_event
from src.data.database import (
    PlatformDatabase,
    funding_entry,
    heavy_compute_lease,
    historical_bootstrap_progress,
    job,
    job_quarantine,
    platform_schedule,
)
from src.data.feature_graph import FeatureGraphRegistry
from src.data.feature_store import FeatureValue, SqlFeatureStore
from src.data.parquet_store import (
    DurableFeatureBatchSpool,
    DurableMarketBatchSpool,
    PartitionedBarStore,
    PartitionedFeatureStore,
    PartitionedMarketEventStore,
)
from src.domain._codec import canonical_hash
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderType
from src.domain.positions import Position, PositionStatus
from src.research.accounting import FuturesIncomeAccounting
from src.research.backtest.bar_engine import BarPortfolioEngine, BarStep
from src.research.backtest.event_engine import (
    EventReplayEngine,
    ReplayEvent,
    SimulatedLimitOrder,
    SimulatedOrderSide,
)
from src.research.evidence import regime_breakdown_passes
from src.research.generation import CAMPAIGNS, GenerationError, build_hypothesis
from src.research.returns import PositionReturnLedger
from src.services.alerting import AlertSeverity, SqlAlertService
from src.services.config import load_split_configuration
from src.services.feature_worker import DatabaseFeatureWorker
from src.services.forward_metrics import ForwardEvidenceCollector, ForwardEvidenceMetrics
from src.services.heavy_compute import HeavyComputeLeaseStore
from src.services.historical_bootstrap import HistoricalBootstrapError, HistoricalBootstrapService
from src.services.platform_bootstrap import PlatformBootstrap
from src.services.platform_upgrade import quarantine_legacy_research_jobs
from src.services.promotion import PromotionPolicy, overlay_family_policy
from src.services.research_worker import ResearchWorker
from src.services.scheduler import DatabaseJobQueue, PlatformScheduler

NOW = "2026-09-01T00:00:00+00:00"


class _NoopLimiter:
    def acquire(self) -> None:
        return None


@dataclass
class _Response:
    payload: object

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _HistorySession:
    def __init__(self, *, start_ms: int, interval_ms: int) -> None:
        self.klines = [
            [
                start_ms + index * interval_ms,
                "100",
                "101",
                "99",
                "100.5",
                "1",
                start_ms + index * interval_ms + interval_ms - 1,
                "100",
                "3",
                "0.5",
                "50",
            ]
            for index in range(4)
        ]
        self.funding = [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "-0.001",
                "fundingTime": start_ms,
                "markPrice": "100",
            },
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.002",
                "fundingTime": start_ms + 8 * 3_600_000,
                "markPrice": "110",
            },
        ]

    def get(self, url: str, *, params=None, timeout: float):
        if "exchangeInfo" in url:
            return _Response({"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]})
        if "fundingRate" in url:
            start = int(params["startTime"])
            end = int(params["endTime"])
            return _Response(
                [row for row in self.funding if start <= int(row["fundingTime"]) <= end]
            )
        start = int(params["startTime"])
        end = int(params["endTime"])
        return _Response(
            [row for row in self.klines if start <= int(row[0]) <= end][: int(params["limit"])]
        )


@dataclass
class _FlakyAlertSink:
    fail: bool = True
    name: str = "test-alert"

    def send(self, _record) -> None:
        if self.fail:
            raise OSError("temporary sink failure")


def _database(tmp_path: Path) -> PlatformDatabase:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    return database


def _candle(index: int = 0):
    close = dt.datetime.fromisoformat(NOW) + dt.timedelta(minutes=index + 1)
    close_ms = int(close.timestamp() * 1_000)
    return normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=close.isoformat(),
        payload={
            "e": "kline",
            "E": close_ms + 1,
            "s": "BTCUSDT",
            "k": {
                "t": close_ms - 59_999,
                "T": close_ms,
                "i": "1m",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": str(100 + index),
                "v": "10",
                "x": True,
            },
        },
    )


@pytest.mark.parametrize("count", [1, 401])
def test_legacy_research_jobs_are_quarantined_once(tmp_path: Path, count: int) -> None:
    database = _database(tmp_path)
    try:
        with database.engine.begin() as connection:
            connection.execute(
                insert(job),
                [
                    {
                        "id": f"legacy-job-{index}",
                        "name": "bounded_backtest",
                        "state": "pending",
                        "priority": 1,
                        "available_at": NOW,
                        "attempts": 0,
                        "max_attempts": 3,
                        "producer_identity": "legacy-autopilot",
                        "content_hash": "sha256:" + "1" * 64,
                        "payload": {"legacy": True},
                    }
                    for index in range(count)
                ],
            )
        first = quarantine_legacy_research_jobs(database.engine, now=NOW)
        second = quarantine_legacy_research_jobs(database.engine, now=NOW)
        assert first["count"] == count
        assert second["already_applied"] is True
        with database.engine.connect() as connection:
            assert set(connection.execute(select(job.c.state)).scalars()) == {"quarantined"}
            assert set(connection.execute(select(job_quarantine.c.job_id)).scalars()) == {
                f"legacy-job-{index}" for index in range(count)
            }
    finally:
        database.dispose()


def test_scheduler_reuses_stable_work_without_timestamp_identity_collisions(tmp_path: Path) -> None:
    database = _database(tmp_path)
    try:
        configuration = load_split_configuration(Path("config"))
        PlatformBootstrap(engine=database.engine, configuration=configuration).ensure(now=NOW)
        scheduler = PlatformScheduler(
            engine=database.engine,
            products={
                str(item["product_id"]): dict(item)
                for item in configuration["products"]["products"]
            },
            node_id="linux-optiplex",
        )
        scheduler.run_once(now=NOW)
        future = "2026-09-01T01:00:00+00:00"
        with database.engine.begin() as connection:
            connection.execute(update(platform_schedule).values(next_run_at=future))
        scheduler.run_once(now=future)
    finally:
        database.dispose()


def test_alert_delivery_retries_and_resolves_the_latest_attempt(tmp_path: Path) -> None:
    database = _database(tmp_path)
    sink = _FlakyAlertSink()
    service = SqlAlertService(database.engine, sinks=(sink,), default_cooldown_seconds=0)
    try:
        record = service.emit(
            event_type="emergency_action_pending",
            severity=AlertSeverity.CRITICAL,
            dedupe_key="emergency:one",
            target="active_income",
            message="waiting for flat reconciliation",
            emitted_at=NOW,
        )
        sink.fail = False
        delivery = service.deliver_pending(now="2026-09-01T00:01:00+00:00")
        assert delivery.delivered == (record.alert_id,)
        attempts = service.delivery_attempts(alert_id=record.alert_id)
        assert attempts[-1]["status"] == "delivered"
        assert service.deliver_pending(now="2026-09-01T00:02:00+00:00").delivered == ()
    finally:
        database.dispose()


def test_history_bootstrap_resumes_and_records_native_funding(tmp_path: Path) -> None:
    database = _database(tmp_path)
    start_ms = 1_700_000_040_000
    interval_ms = 60_000
    session = _HistorySession(start_ms=start_ms, interval_ms=interval_ms)
    service = HistoricalBootstrapService(
        engine=database.engine,
        root=tmp_path / "data",
        session=session,
        rate_limiter=_NoopLimiter(),
        page_size=2,
    )
    try:
        with pytest.raises(HistoricalBootstrapError, match="page budget"):
            service.bootstrap(
                product_id="active_income",
                market="futures",
                symbol="BTCUSDT",
                timeframe="1m",
                range_start_ms=start_ms,
                range_end_ms=start_ms + 4 * interval_ms,
                maximum_pages=1,
            )
        resumed = service.bootstrap(
            product_id="active_income",
            market="futures",
            symbol="BTCUSDT",
            timeframe="1m",
            range_start_ms=start_ms,
            range_end_ms=start_ms + 4 * interval_ms,
            maximum_pages=2,
        )
        assert resumed.resumed is True
        assert resumed.completed is True
        assert len(list((tmp_path / "data/bars").rglob("*.parquet"))) == 1
        with database.engine.connect() as connection:
            progress = connection.execute(
                select(historical_bootstrap_progress.c.status).where(
                    historical_bootstrap_progress.c.product_id == "active_income",
                    historical_bootstrap_progress.c.timeframe == "1m",
                )
            ).scalar_one()
        assert progress == "completed"
        funding_end = start_ms + 8 * 3_600_000 + 1
        funding = service.bootstrap_funding(
            product_id="active_income",
            symbol="BTCUSDT",
            range_start_ms=start_ms,
            range_end_ms=funding_end,
        )
        assert funding.completed is True
        assert funding.rows == 2
        with database.engine.connect() as connection:
            payloads = connection.execute(select(funding_entry.c.payload)).scalars().all()
        assert {float(payload["funding_rate"]) for payload in payloads} == {-0.001, 0.002}
    finally:
        database.dispose()


def test_history_bootstrap_retries_persisted_gaps_after_restart(tmp_path: Path) -> None:
    database = _database(tmp_path)
    start_ms = 1_700_000_040_000
    interval_ms = 60_000
    session = _HistorySession(start_ms=start_ms, interval_ms=interval_ms)
    missing = session.klines.pop(1)
    service = HistoricalBootstrapService(
        engine=database.engine,
        root=tmp_path / "data",
        session=session,
        rate_limiter=_NoopLimiter(),
        page_size=10,
    )
    try:
        first = service.bootstrap(
            product_id="active_income",
            market="futures",
            symbol="BTCUSDT",
            timeframe="1m",
            range_start_ms=start_ms,
            range_end_ms=start_ms + 4 * interval_ms,
        )
        assert first.completed is False
        session.klines.insert(1, missing)
        second = service.bootstrap(
            product_id="active_income",
            market="futures",
            symbol="BTCUSDT",
            timeframe="1m",
            range_start_ms=start_ms,
            range_end_ms=start_ms + 4 * interval_ms,
        )
        assert second.resumed is True
        assert second.completed is True
        path = next((tmp_path / "data/bars").rglob("*.parquet"))
        assert len(pq.ParquetFile(path).read().to_pylist()) == 4
    finally:
        database.dispose()


def test_market_spool_and_partition_replay_are_bounded_and_idempotent(tmp_path: Path) -> None:
    event = _candle()
    spool = DurableMarketBatchSpool(tmp_path / "spool", max_rows=1)
    segment = spool.append(event, venue="binance", market="futures", symbol="BTCUSDT")
    assert segment is not None
    claimed = spool.claim(segment.path)
    committed = spool.commit(claimed)
    assert spool.recoverable() == (committed,)
    spool.delete(committed)
    assert tuple((tmp_path / "spool" / "committed").glob("*")) == ()

    store = PartitionedMarketEventStore(tmp_path / "data")
    first = store.put_batch(((event, "binance", "futures", "BTCUSDT"),))
    second = store.put_batch(((event, "binance", "futures", "BTCUSDT"),))
    assert first == second
    assert len(list((tmp_path / "data/raw").rglob("*.parquet"))) == 1

    bars = PartitionedBarStore(tmp_path / "data")
    bars.put(event, venue="binance", market="futures", symbol="BTCUSDT")
    bars.put(event, venue="binance", market="futures", symbol="BTCUSDT")
    assert len(list((tmp_path / "data/bars").rglob("*.parquet"))) == 1

    values = tuple(
        FeatureValue(
            feature_set_version="features-v1",
            feature_name=name,
            instrument_id=event.instrument_id,
            source_event_time=event.exchange_timestamp,
            source_close_time=event.close_timestamp,
            availability_time=event.availability_time,
            value=float(index),
        )
        for index, name in enumerate(("bar_return", "range_fraction"), 1)
    )
    feature_store = PartitionedFeatureStore(tmp_path / "data")
    feature_store.put_batch(
        values[:1], venue="binance", market="futures", symbol="BTCUSDT", timeframe="1m"
    )
    feature_store.put_batch(
        values[1:], venue="binance", market="futures", symbol="BTCUSDT", timeframe="1m"
    )
    assert len(list((tmp_path / "data/features").rglob("*.parquet"))) == 1


def test_feature_spool_splits_cross_day_batches_before_partition_write(tmp_path: Path) -> None:
    database = _database(tmp_path)
    root = tmp_path / "data"
    spool = DurableFeatureBatchSpool(
        tmp_path / "runtime/events/feature-spool", max_rows=10, flush_seconds=10**9
    )
    values = tuple(
        FeatureValue(
            feature_set_version="features-v1",
            feature_name=name,
            instrument_id="binance:futures:BTCUSDT:USDT",
            source_event_time=f"2026-09-0{day}T00:00:00+00:00",
            source_close_time=f"2026-09-0{day}T00:00:00+00:00",
            availability_time=f"2026-09-0{day}T00:00:01+00:00",
            value=float(day),
        )
        for day, name in ((1, "bar_return"), (2, "bar_return"))
    )
    try:
        for value in values:
            spool.append(
                (value,),
                venue="binance",
                market="futures",
                symbol="BTCUSDT",
                timeframe="1m",
            )
        segment = spool.flush()
        assert segment is not None
        worker = DatabaseFeatureWorker(
            queue=DatabaseJobQueue(database.engine),
            worker_id="feature-worker",
            store=SqlFeatureStore(database.engine),
            job_names=("live_feature_calculation",),
            parquet_root=root,
        )
        paths = worker._write_feature_segment(segment)
        assert len(paths) == 2
        assert len(list((root / "features").rglob("*.parquet"))) == 2
        assert spool.recoverable() == ()
    finally:
        database.dispose()


def test_accounting_is_terminal_costed_fractional_and_symbol_isolated() -> None:
    returns = PositionReturnLedger(fee_rate=0.01).measure(
        [0.0, 1.0], [0.1], terminal_position=1.0, liquidate_terminal=True
    )
    assert returns.turnover == pytest.approx(2.0)
    assert returns.terminal_liquidation_cost == pytest.approx(0.01)
    drawdown = PositionReturnLedger().measure([1.0, 1.0, 1.0], [0.1, -0.05])
    assert drawdown.maximum_drawdown == pytest.approx(0.05 / 1.1)

    report = FuturesIncomeAccounting().evaluate(
        initial_cash=1_000.0,
        terminal_fee_rate=0.01,
        events=(
            {
                "type": "fill",
                "timestamp": NOW,
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 1,
                "price": 100,
            },
            {
                "type": "fill",
                "timestamp": "2026-09-01T00:01:00+00:00",
                "symbol": "ETHUSDT",
                "side": "buy",
                "quantity": 2,
                "price": 50,
            },
            {
                "type": "funding",
                "timestamp": "2026-09-01T00:02:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 100,
                "funding_rate": 0.01,
            },
            {
                "type": "fill",
                "timestamp": "2026-09-01T00:03:00+00:00",
                "symbol": "BTCUSDT",
                "side": "sell",
                "quantity": 1,
                "price": 110,
            },
            {
                "type": "mark",
                "timestamp": "2026-09-01T00:04:00+00:00",
                "symbol": "ETHUSDT",
                "mark_price": 60,
            },
        ),
    )
    assert report.realised_pnl == pytest.approx(10.0)
    assert report.funding_pnl == pytest.approx(-1.0)
    assert report.terminal_liquidation_cost == pytest.approx(1.2)


def test_generated_campaigns_emit_family_specific_executable_contracts() -> None:
    expected = {
        "btc_accumulation": "btc_target_allocation",
        "time_series": "signed_directional_forecast",
        "mean_reversion": "signed_directional_forecast",
        "cross_sectional": "ranked_long_short_panel",
        "relative_value": "paired_legs_hedged",
        "microstructure": "event_time_microstructure",
        "meta_strategy": "forecast_ensemble",
    }
    for campaign in CAMPAIGNS:
        symbols = ("BTCUSDT",) if campaign.product == "btc_accumulation" else ("BTCUSDT", "ETHUSDT")
        if campaign.unavailable_reason is not None:
            with pytest.raises(GenerationError, match="unsupported campaign"):
                build_hypothesis(
                    campaign,
                    variant=0,
                    instrument_universe=symbols,
                    dataset_snapshot_hashes=(canonical_hash({"campaign": campaign.name}),),
                    submitted_at=NOW,
                )
            continue
        hypothesis = build_hypothesis(
            campaign,
            variant=0,
            instrument_universe=symbols,
            dataset_snapshot_hashes=(canonical_hash({"campaign": campaign.name}),),
            submitted_at=NOW,
        )
        model = hypothesis.candidate.definition.signal_model
        expected_kind = (
            expected["btc_accumulation"]
            if campaign.product == "btc_accumulation"
            else expected.get(campaign.family, "signed_directional_forecast")
        )
        assert model["kind"] == expected_kind
        assert model["parameters"]["threshold_unit"]


def test_feature_graph_availability_does_not_enable_unimplemented_pair_campaigns() -> None:
    registry = FeatureGraphRegistry.default()
    for campaign in CAMPAIGNS:
        registry.graph(tuple(campaign.permitted_features))
        if campaign.family == "relative_value":
            with pytest.raises(GenerationError, match="unsupported campaign"):
                build_hypothesis(
                    campaign,
                    variant=0,
                    instrument_universe=("BTCUSDT", "ETHUSDT"),
                    dataset_snapshot_hashes=(canonical_hash({"campaign": campaign.name}),),
                    submitted_at=NOW,
                )


def test_forward_policy_is_selected_by_exact_product_and_family() -> None:
    base = PromotionPolicy(
        automatic_paper_promotion=True,
        automatic_live_canary_promotion=False,
        canary_capital_limit=0.01,
        required_forward_evidence_days=60,
        maximum_drawdown=0.08,
        maximum_execution_drift=0.1,
        maximum_model_drift=0.1,
        paper_capital_limit=10_000.0,
    )
    selected = overlay_family_policy(
        base,
        product_id="active_income",
        family="scalping",
        profiles={
            ("active_income", "scalping"): {
                "required_forward_evidence_days": 14,
                "minimum_forward_independent_decisions": 200,
                "minimum_forward_effective_trades": 200,
            },
            ("active_income", "swing"): {"required_forward_evidence_days": 60},
        },
    )
    assert selected.required_forward_evidence_days == 14
    assert selected.minimum_forward_independent_decisions == 200
    assert selected.minimum_forward_effective_trades == 200
    assert selected.maximum_drawdown == base.maximum_drawdown


def test_paper_account_state_survives_reinstantiation_and_reconciliation(tmp_path: Path) -> None:
    database = _database(tmp_path)
    account = PaperAccountLedger(
        engine=database.engine,
        account_id="paper-futures",
        product_id="active_income",
        accounting_asset="USDT",
        starting_balances={"USDT": 1_000.0},
    )
    order = OrderIntent(
        order_id="paper-order",
        portfolio_id="paper-portfolio",
        instrument_id="binance:futures:BTCUSDT:USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        order_type=OrderType.MARKET,
        created_at=NOW,
        strategy_contributions={"strategy-1": 1.0},
    )
    fill = Fill(
        fill_id="paper-fill",
        order_id=order.order_id,
        instrument_id=order.instrument_id,
        side=OrderSide.BUY,
        quantity=0.1,
        price=100.0,
        fee=0.1,
        occurred_at=NOW,
        fee_asset="USDT",
        metadata={"slippage_cost": 0.2},
    )
    previous = Position(
        portfolio_id=order.portfolio_id,
        instrument_id=order.instrument_id,
        quantity=0.0,
        average_entry_price=0.0,
        status=PositionStatus.FLAT,
        updated_at=NOW,
    )
    current = Position(
        portfolio_id=order.portfolio_id,
        instrument_id=order.instrument_id,
        quantity=0.1,
        average_entry_price=100.0,
        status=PositionStatus.OPEN,
        updated_at=NOW,
    )
    account.record_fill(order=order, fill=fill, position=current, previous_position=previous)
    restarted = PaperAccountLedger(
        engine=database.engine,
        account_id="paper-futures",
        product_id="active_income",
        accounting_asset="USDT",
        starting_balances={"USDT": 1_000.0},
    )
    assert restarted.state() is not None
    assert restarted.state().balances["USDT"] == pytest.approx(999.9)
    assert restarted.state().positions[order.instrument_id] == pytest.approx(0.1)
    database.dispose()


@pytest.mark.parametrize(
    "initial,events,expected",
    [
        (
            {},
            [("a1", "a", "buy", 1, 0), ("b1", "b", "sell", 1, 0), ("a2", "a", "buy", 1, 0)],
            ("b1",),
        ),
        (
            {"BTC": 1},
            [("s1", "s", "sell", 1, 0), ("b1", "b", "buy", 1, 0), ("s2", "s", "sell", 1, 0)],
            ("s1", "s2"),
        ),
        ({}, [("a1", "a", "buy", 1, 0.001), ("b1", "b", "sell", 0.999, 0)], ("b1",)),
    ],
)
def test_forward_episodes_follow_incremental_fills_not_order_groups(initial, events, expected):
    rows = tuple(
        {
            "id": identity,
            "created_at": NOW,
            "payload": {
                "order_id": order,
                "instrument_id": "BTC",
                "side": side,
                "quantity": quantity,
                "occurred_at": f"2026-08-30T10:00:00.{index + 1:03d}+00:00",
                "metadata": {"base_fee_quantity": fee},
            },
        }
        for index, (identity, order, side, quantity, fee) in enumerate(events)
    )
    assert (
        ForwardEvidenceCollector._closed_episode_ids(
            tuple(reversed(rows)) + rows[:1], initial_positions=initial
        )
        == expected
    )
    conflicting = {**rows[0], "payload": {**rows[0]["payload"], "quantity": 99}}
    with pytest.raises(ValueError, match="conflicting content"):
        ForwardEvidenceCollector._closed_episode_ids(
            (*rows, conflicting), initial_positions=initial
        )


def test_forward_facts_count_closed_episodes_and_do_not_invent_fill_rate() -> None:
    rows = (
        {
            "id": "fill-open-a",
            "created_at": NOW,
            "payload": {
                "order_id": "open-a",
                "instrument_id": "BTC",
                "side": "buy",
                "quantity": 0.5,
            },
        },
        {
            "id": "fill-open-b",
            "created_at": NOW,
            "payload": {
                "order_id": "open-a",
                "instrument_id": "BTC",
                "side": "buy",
                "quantity": 0.5,
            },
        },
        {
            "id": "fill-close-a",
            "created_at": NOW,
            "payload": {
                "order_id": "close-a",
                "instrument_id": "BTC",
                "side": "sell",
                "quantity": 1.0,
            },
        },
    )
    assert ForwardEvidenceCollector._closed_episode_ids(rows) == ("fill-close-a",)
    assert ForwardEvidenceCollector._data_uptime(0, expected_observations=8) == 0.0
    facts = ForwardEvidenceMetrics(
        net_pnl=0.0,
        benchmark_pnl=0.0,
        drawdown=0.0,
        execution_drift=0.0,
        model_drift=0.0,
        portfolio_capacity=100.0,
        risk_budget_available=10.0,
        data_gaps=0,
        effective_trades=0,
        fill_rate=0.0,
        slippage=0.0,
        data_uptime=1.0,
        rejected_orders=0,
        source_event_ids=("source",),
        window_start=NOW,
    )
    assert facts.fill_rate_applicable is False


def test_backtests_charge_terminal_exposure_and_preserve_terminal_state() -> None:
    bars = BarPortfolioEngine(initial_equity=1_000.0, fee_bps=10.0, slippage_bps=5.0).simulate(
        (
            BarStep(
                timestamp=NOW,
                prices={"BTC": 100.0},
                target_fractions={"BTC": 0.5},
            ),
        )
    )
    assert bars.terminal_exposure == pytest.approx(500.0)
    assert bars.terminal_liquidation_cost == pytest.approx(0.75)
    assert bars.quantities == {"BTC": 0.0}
    assert bars.equity_curve[-1][1] == pytest.approx(998.5)

    events = EventReplayEngine(terminal_fee_bps=10.0, terminal_slippage_bps=5.0).simulate(
        events=(
            ReplayEvent(
                event_time=NOW,
                receive_time=NOW,
                instrument_id="BTC",
                best_bid=99.0,
                best_ask=101.0,
                ask_depth=1.0,
                mark_price=100.0,
            ),
        ),
        orders=(
            SimulatedLimitOrder(
                order_id="terminal-order",
                instrument_id="BTC",
                side=SimulatedOrderSide.BUY,
                quantity=1.0,
                limit_price=101.0,
                submitted_at=NOW,
                expires_at="2026-09-01T00:05:00+00:00",
            ),
        ),
    )
    assert events.terminal_exposure == pytest.approx(100.0)
    assert events.terminal_liquidation_cost == pytest.approx(0.15)


def test_regime_evidence_rejects_placeholder_single_bucket() -> None:
    assert (
        regime_breakdown_passes(
            {
                "passed": True,
                "status": "measured",
                "regimes": {"all": 0.04},
                "observations_by_regime": {"all": 100},
                "regime_count": 1,
                "concentration_fraction": 1.0,
            }
        )
        is False
    )


def test_ambiguous_emergency_submission_never_reports_success_without_reconcile() -> None:
    from src.services.platform_testnet_emergency import EmergencyStateMachine

    calls = {"submit": 0, "query": 0, "reconcile": 0}

    def submit():
        calls["submit"] += 1
        raise TimeoutError

    def query(_client_id: str):
        calls["query"] += 1
        return SimpleNamespace(status="filled")

    def reconcile() -> bool:
        calls["reconcile"] += 1
        return calls["reconcile"] > 1

    result = EmergencyStateMachine().run(
        order_id="order",
        client_order_id="client",
        submit=submit,
        query=query,
        reconcile=reconcile,
    )
    assert result.state == "completed"
    assert result.reconciled_flat is True
    assert result.history == (
        "requested",
        "entries_blocked",
        "exchange_state_refreshed",
        "close_planned",
        "close_submitted",
        "close_acknowledged",
        "fills_observed",
        "reconciled_flat",
        "completed",
    )
    assert calls == {"submit": 1, "query": 2, "reconcile": 2}


def test_research_worker_rejects_configured_account_secrets(monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_FUTURES_API_KEY", "secret")
    runtime = SimpleNamespace(
        node=SimpleNamespace(operating_system="linux"),
        can_submit_orders=False,
    )
    with pytest.raises(PermissionError, match="exchange credentials"):
        ResearchWorker(
            runtime=runtime,
            queue=SimpleNamespace(),
            worker_id="research",
            handlers={},
            forbidden_secret_names=("BINANCE_FUTURES_API_KEY",),
        )


def test_resource_deferral_does_not_consume_a_research_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    database = _database(tmp_path)
    queue = DatabaseJobQueue(database.engine)
    worker_id = "research-worker"
    queue.register_worker(
        worker_id=worker_id,
        node_id="linux-optiplex",
        role="research-worker",
        capabilities=("bounded_backtest",),
        observed_at=NOW,
    )
    with database.engine.begin() as connection:
        connection.execute(
            insert(job).values(
                id="heavy-job",
                name="bounded_backtest",
                state="pending",
                priority=1,
                available_at=NOW,
                attempts=0,
                max_attempts=3,
                producer_identity="test",
                content_hash="sha256:" + "2" * 64,
                payload={},
            )
        )
        connection.execute(
            insert(heavy_compute_lease).values(
                slot_id=HeavyComputeLeaseStore.SLOT_ID,
                owner="other-worker",
                job_id="other-job",
                acquired_at=NOW,
                expires_at="2026-09-01T01:00:00+00:00",
                status="active",
            )
        )
    runtime = SimpleNamespace(
        node=SimpleNamespace(operating_system="linux"),
        can_submit_orders=False,
    )
    result = ResearchWorker(
        runtime=runtime,
        queue=queue,
        worker_id=worker_id,
        handlers={"bounded_backtest": lambda _claimed, _renew: {}},
        heavy_compute=HeavyComputeLeaseStore(database.engine),
    ).run_once(now=NOW)
    assert result["reason_code"] == "heavy_compute_slot_unavailable"
    with database.engine.connect() as connection:
        state, attempts = connection.execute(
            select(job.c.state, job.c.attempts).where(job.c.id == "heavy-job")
        ).one()
    assert (state, attempts) == ("pending", 0)
    database.dispose()


def test_openclaw_scheduled_review_round_trip_is_bound_to_request(
    tmp_path: Path, monkeypatch
) -> None:
    from src.agents.context import AgentContext
    from src.agents.transport import sign_review_response
    from src.services.agent_transport import AgentReviewTransport

    database = _database(tmp_path)
    queue = DatabaseJobQueue(database.engine)
    bridge = OpenClawAgentBridge(store=SqlAgentStore(database.engine), queue=queue)
    review_id = "review-1"
    request = {
        "schema": "platform.agent_review_request/v1",
        "review_id": review_id,
        "context": {"failure_reasons": {"capacity": 1}, "product_id": "active_income"},
        "context_hash": AgentContext(
            created_at=NOW,
            values={"failure_reasons": {"capacity": 1}, "product_id": "active_income"},
        ).content_hash,
        "transport": "postgresql_agent_review",
        "response_required": True,
        "requested_at": NOW,
        "expires_at": "2026-09-01T01:00:00+00:00",
    }
    SqlAgentStore(database.engine).save_review(
        proposal_id=review_id,
        created_at=NOW,
        payload=request,
    )
    response = {
        "schema": "openclaw.agent_proposal/v1",
        "source": "openclaw",
        "proposal_id": "proposal-1",
        "role": "researcher",
        "action": "create_thesis",
        "product_id": "active_income",
        "created_at": NOW,
        "thesis": "A bounded market hypothesis must explain a causal return mechanism.",
        "provenance": {"review_id": review_id},
    }
    key = b"test-review-transport-secret-only" * 2
    request = bridge.review_request(review_id)
    bridge.enqueue_review_delivery(review_id, available_at=NOW)
    transport = AgentReviewTransport(queue, key=key)
    transport.bridge = bridge
    assert transport.next_request(now=NOW)["request"] == request
    envelope = sign_review_response(request, response, key=key)
    with pytest.raises(PermissionError, match="signature"):
        bridge.ingest_review_response(envelope, key=b"wrong-key" * 8, now=NOW)
    with pytest.raises(PermissionError, match="expired"):
        bridge.ingest_review_response(envelope, key=key, now=request["expires_at"])
    future_dated = sign_review_response(
        request, {**response, "created_at": request["expires_at"]}, key=key
    )
    with pytest.raises(ValueError, match="timestamp is outside"):
        bridge.ingest_review_response(future_dated, key=key, now=NOW)
    assert not SqlAgentStore(database.engine).records("proposal")

    def interrupted_intake(_payload):
        raise RuntimeError("crash after response acceptance")

    with monkeypatch.context() as patch:
        patch.setattr(bridge, "ingest", interrupted_intake)
        with pytest.raises(RuntimeError, match="crash after response"):
            bridge.ingest_review_response(envelope, key=key, now=NOW)
    assert not SqlAgentStore(database.engine).records("proposal")
    assert transport.next_request(now=request["expires_at"])["request"] is None
    assert queue.state(f"agent-delivery:{review_id}") == "completed"
    proposal = bridge.ingest_review_response(envelope, key=key, now=request["expires_at"])
    assert proposal.proposal_id == "proposal-1"
    assert bridge.ingest_review_response(envelope, key=key, now=NOW) == proposal
    conflicting = sign_review_response(request, {**response, "proposal_id": "other"}, key=key)
    with pytest.raises(ValueError, match="different response"):
        bridge.ingest_review_response(conflicting, key=key, now=NOW)
    assert len(SqlAgentStore(database.engine).records("proposal")) == 1
    assert any(
        item.get("schema") == "platform.agent_review_response/v1"
        and item.get("review_id") == review_id
        for item in SqlAgentStore(database.engine).records("review")
    )
    database.dispose()
