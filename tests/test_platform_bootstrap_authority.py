from __future__ import annotations

import copy
import datetime as dt
import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, insert, select, update

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.binance_market import normalise_public_event
from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    cost_model_manifest,
    feature_manifest,
    instrument,
    job,
    platform_bootstrap,
    platform_rehearsal_report,
    platform_schedule,
    universe_snapshot,
)
from src.data.universe import (
    InstrumentObservation,
    SqlUniverseStore,
    UniverseEligibilityPolicy,
    eligibility_reason,
)
from src.domain._codec import canonical_hash, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.domain.market_events import MarketEventType
from src.domain.orders import OrderIntent, OrderSide, OrderType
from src.execution.broker import (
    BrokerFill,
    BrokerIncome,
    BrokerOrderState,
    FuturesPositionIdentity,
    OpenOrderIdentity,
)
from src.execution.broker import (
    OrderSide as BrokerOrderSide,
)
from src.execution.ccxt_broker import CcxtBroker
from src.execution.config import ExchangeConfig
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.observability.decision_trace import SqlDecisionTraceStore
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlApprovalRepository,
    SqlPreflightRepository,
    SqlStrategyArtefactRepository,
)
from src.research.store import SqlResearchStore
from src.risk.engine import SqlRiskSnapshotStore
from src.services.account_reconciliation import AccountAuthorityError, AccountReconciliationService
from src.services.config import load_platform_config, load_split_configuration
from src.services.data_writer import DatabaseMarketDataWriter, _market_snapshot_values
from src.services.health import DatabaseHeartbeatStore
from src.services.live_execution import ApprovedLiveExecution, live_authority_configuration_hash
from src.services.market_gateway import UserStreamAccount, _stream_endpoints
from src.services.order_execution import DatabasePaperExecutionWorker
from src.services.paper_diagnostic import DatabaseDiagnosticPaperWorker
from src.services.platform_bootstrap import PlatformBootstrap
from src.services.platform_smoke import _product_fixture, _seed_strategy
from src.services.platform_testnet_connected import (
    ConnectedTestnetError,
    _connected_gateway,
    _emergency_restore_position,
    _verify_recovery_lookup,
    validate_connected_testnet_configuration,
)
from src.services.platform_testnet_connected import (
    _execution_engine_identity as connected_execution_engine_identity,
)
from src.services.portfolio_state import DatabasePortfolioSourceService, portfolio_state_policies
from src.services.readiness import _execution_engine_identity, _live_product_checks
from src.services.runtime import ServiceRuntime
from src.services.scheduler import DatabaseJobQueue, PlatformScheduler
from src.services.supervisor import _active_assignments, _research_cycle
from src.services.universe_service import BinanceUniverseClient

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-27T10:00:00+00:00"


def _database(tmp_path: Path) -> PlatformDatabase:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.create_schema()
    return database


def test_platform_bootstrap_and_scheduler_are_idempotent(tmp_path: Path) -> None:
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    bootstrap = PlatformBootstrap(engine=database.engine, configuration=configuration)

    first = bootstrap.ensure(now=NOW)
    second = bootstrap.ensure(now="2026-08-27T11:00:00+00:00")

    assert first == second
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(func.count()).select_from(platform_bootstrap)).scalar_one()
            == 1
        )
        assert (
            connection.execute(select(func.count()).select_from(universe_snapshot)).scalar_one()
            == 2
        )
        assert (
            connection.execute(select(func.count()).select_from(account_snapshot)).scalar_one() == 2
        )
        assert (
            connection.execute(select(func.count()).select_from(feature_manifest)).scalar_one() == 2
        )
        assert (
            connection.execute(select(func.count()).select_from(cost_model_manifest)).scalar_one()
            == 2
        )
        instruments = set(connection.execute(select(instrument.c.id)).scalars())
    assert instruments == {"binance:spot:BTCUSDT", "binance:futures:BTCUSDT:USDT"}
    for instrument_id in instruments:
        assert _active_assignments(database.engine)(instrument_id) == ()

    scheduler = PlatformScheduler(
        engine=database.engine,
        products={
            str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
        },
        node_id="linux-optiplex",
    )
    scheduled = scheduler.run_once(now=NOW)
    repeated = scheduler.run_once(now=NOW)

    assert scheduled["schedules"] == 13
    assert scheduled["jobs_enqueued"] == 7
    assert repeated["jobs_enqueued"] == 0
    assert scheduled["maintenance"]["reason_code"] == "platform_maintenance_completed"
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(func.count()).select_from(platform_schedule)).scalar_one()
            == 13
        )
        assert connection.execute(select(func.count()).select_from(job)).scalar_one() == 13
        assert "diagnostic_paper_open" in set(connection.execute(select(job.c.name)).scalars())
        assert "dataset_snapshot_validate" in set(connection.execute(select(job.c.name)).scalars())
        assert "universe_refresh" in set(connection.execute(select(job.c.name)).scalars())
        assert "agent_review" in set(connection.execute(select(job.c.name)).scalars())
        assert "register_ml_candidate" not in set(connection.execute(select(job.c.name)).scalars())

    with database.engine.begin() as connection:
        connection.execute(
            update(platform_schedule)
            .where(platform_schedule.c.id == "platform:register_strategy_catalogue")
            .values(next_run_at=NOW)
        )
    restarted = scheduler.run_once(now="2026-08-27T10:00:01+00:00")
    assert restarted["jobs_enqueued"] == 0

    testnet_configuration = copy.deepcopy(configuration)
    next(
        item
        for item in testnet_configuration["products"]["products"]
        if item["product_id"] == "active_income"
    )["execution_mode"] = "live"
    next(
        item
        for item in testnet_configuration["accounts"]["accounts"]
        if item["account_id"] == "binance-futures-main"
    )["environment"] = "testnet"
    PlatformBootstrap(
        engine=database.engine,
        configuration=testnet_configuration,
    ).ensure(now="2026-08-27T12:00:00+00:00")


def test_bootstrap_is_idempotent_after_initial_job_completion(tmp_path: Path) -> None:
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    bootstrap = PlatformBootstrap(engine=database.engine, configuration=configuration)
    bootstrap.ensure(now=NOW)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="test:universe-worker",
        node_id="test",
        role="universe-worker",
        capabilities=("universe_refresh",),
        observed_at=NOW,
    )
    claimed = queue.claim(
        worker_id="test:universe-worker",
        now=NOW,
        lease_seconds=60,
        names=("universe_refresh",),
    )
    assert claimed is not None
    queue.complete(claimed, completed_at="2026-08-27T10:00:01+00:00")

    bootstrap.ensure(now="2026-08-27T11:00:00+00:00")

    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(job).where(job.c.id == claimed.job_id)
            ).scalar_one()
            == 1
        )


def test_fresh_platform_state_policies_leave_measured_risk_to_runtime() -> None:
    configuration = load_split_configuration(ROOT / "config")
    products = {
        str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
    }

    policies = portfolio_state_policies(configuration, products)

    for policy in policies.values():
        assert "product_drawdown_fraction" not in policy
        assert "daily_pnl_fraction" not in policy
        assert "global_drawdown_fraction" not in policy
        assert "trades_today" not in policy


def test_scheduler_research_jobs_are_processed_automatically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    PlatformBootstrap(engine=database.engine, configuration=configuration).ensure(now=NOW)
    PlatformScheduler(
        engine=database.engine,
        products={
            str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
        },
        node_id="linux-optiplex",
    ).run_once(now=NOW)
    runtime = ServiceRuntime(
        config=load_platform_config(ROOT / "config/platform.json"),
        node_id="linux-optiplex",
        service_name="research-worker",
        heartbeat_store=DatabaseHeartbeatStore(database.engine),
    )
    research = _research_cycle(
        database=database,
        node_id="linux-optiplex",
        service_name="research-worker",
        runtime=runtime,
        maximum_runtime_seconds=60,
        parquet_root=tmp_path / "parquet",
        artefact_root=tmp_path / "artefacts",
        research_configuration=configuration["research"],
    )

    results = [research() for _ in range(6)]

    assert all(result["reason_code"] == "research_job_completed" for result in results)
    assert any(
        result.get("handler_reason_code") == "historical_bars_incomplete" for result in results
    )
    assert SqlResearchStore(database.engine).load_candidates() == ()


def test_platform_smoke_runs_after_bootstrap(tmp_path: Path) -> None:
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    smoke_now = "2026-08-23T00:00:00+00:00"
    PlatformBootstrap(engine=database.engine, configuration=configuration).ensure(now=smoke_now)
    products = {
        str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
    }
    accounts = {
        str(item["account_id"]): dict(item) for item in configuration["accounts"]["accounts"]
    }

    AccountReconciliationService(
        engine=database.engine,
        products=products,
        accounts=accounts,
    ).reconcile_once(now=smoke_now)

    first_products = [
        {
            **dict(product),
            "product_id": f"{product['product_id']}:smoke:first",
            "product_family": str(product["product_id"]),
            "portfolio_id": f"{product['portfolio_id']}:smoke:first",
        }
        for product in configuration["products"]["products"]
    ]
    results = [
        _product_fixture(
            database,
            dict(product),
            accounts,
            tmp_path,
            index,
            run_id="first-run",
            observed=dt.datetime(2026, 8, 23, tzinfo=dt.UTC),
        )
        for index, product in enumerate(first_products)
    ]
    second_products = [
        {
            **dict(product),
            "product_id": f"{product['product_id']}:smoke:second",
            "product_family": str(product["product_id"]),
            "portfolio_id": f"{product['portfolio_id']}:smoke:second",
        }
        for product in configuration["products"]["products"]
    ]
    repeated_results = [
        _product_fixture(
            database,
            dict(product),
            accounts,
            tmp_path,
            index,
            run_id="second-run",
            observed=dt.datetime(2026, 8, 24, tzinfo=dt.UTC),
        )
        for index, product in enumerate(second_products)
    ]

    assert all(result["ok"] for result in results), results
    assert all(result["ok"] for result in repeated_results), repeated_results


def test_bootstrap_diagnostic_paper_round_trip_is_automatic(tmp_path: Path) -> None:
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    PlatformBootstrap(engine=database.engine, configuration=configuration).ensure(now=NOW)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="test:paper-engine",
        node_id="test",
        role="paper-engine",
        capabilities=(
            "diagnostic_paper_open",
            "diagnostic_paper_close",
            "paper_order_submit",
            "paper_order_continue",
        ),
        observed_at=NOW,
    )
    order_manager = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    products = {
        str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
    }
    ledgers = {
        product_id: Ledger(
            product_id=product_id,
            accounting_asset=str(product["base_accounting_asset"]),
            store=SqlLedgerStore(database.engine, product_id=product_id),
        )
        for product_id, product in products.items()
    }
    diagnostic = DatabaseDiagnosticPaperWorker(
        queue=queue,
        worker_id="test:paper-engine",
        order_manager=order_manager,
        positions=positions,
        products=products,
    )
    paper = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id="test:paper-engine",
        order_manager=order_manager,
        positions=positions,
        ledgers=ledgers,
        trace_store=SqlDecisionTraceStore(database.engine),
    )

    for minute in range(8):
        now = f"2026-08-27T10:{minute:02d}:00+00:00"
        diagnostic.run_once(now=now)
        paper.run_once(now=now)

    order_manager.reload()
    positions.reload()
    assert len(order_manager.all()) == 4
    assert all(order.status.value == "filled" for order in order_manager.all())
    assert positions.get(
        "btc-accumulation-portfolio", "binance:spot:BTCUSDT"
    ).quantity == pytest.approx(0.0, abs=1e-12)
    assert positions.get(
        "active-income-portfolio", "binance:futures:BTCUSDT:USDT"
    ).quantity == pytest.approx(0.0, abs=1e-12)
    with database.engine.connect() as connection:
        diagnostic_states = connection.execute(
            select(job.c.name, job.c.state).where(job.c.name.like("diagnostic_paper_%"))
        ).all()
    assert len(diagnostic_states) == 4
    assert all(state == "completed" for _, state in diagnostic_states)


def test_spot_universe_does_not_require_funding_or_open_interest() -> None:
    observation = InstrumentObservation(
        instrument=Instrument(
            venue="binance",
            market_type=MarketType.SPOT,
            base_asset="BTC",
            quote_asset="USDT",
            settlement_asset=None,
            exchange_symbol="BTCUSDT",
            price_precision=2,
            quantity_precision=6,
            minimum_quantity=0.000001,
            minimum_notional=5.0,
        ),
        listing_age_days=365,
        quote_volume=1_000_000_000,
        trade_count=1_000_000,
        spread_bps=1,
        open_interest=0,
        funding_rate=0,
        realised_volatility=0.2,
        depth_notional=10_000_000,
        data_completeness=1,
    )

    assert eligibility_reason(observation, UniverseEligibilityPolicy()) == "eligible"


def test_binance_spot_universe_uses_filter_precision_and_listing_history() -> None:
    client = BinanceUniverseClient()
    old_listing = 1_500_000_000_000

    def response(path: str, **params):
        if path == "/api/v3/depth":
            return {"bids": [["100", "10000"]], "asks": [["100.01", "10000"]]}
        if path == "/api/v3/klines" and params.get("startTime") == 0:
            return [[old_listing, "1", "1", "1", "1"]]
        if path == "/api/v3/klines":
            return [
                [old_listing + index * 3_600_000, "100", "101", "99", str(100 + index)]
                for index in range(168)
            ]
        raise AssertionError(path)

    client._get = response  # type: ignore[method-assign]
    observation = client._observation(
        {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "5.0"},
            ],
        },
        {"quoteVolume": "1000000000", "count": 1000000},
        {"bidPrice": "100", "askPrice": "100.01"},
        {},
        1_787_824_800_000,
        "spot",
    )

    assert observation.instrument.price_precision == 2
    assert observation.instrument.quantity_precision == 5
    assert observation.instrument.minimum_notional == 5.0
    assert observation.listing_age_days > 30
    assert eligibility_reason(observation, UniverseEligibilityPolicy()) == "eligible"


def test_spot_market_source_defaults_funding_but_futures_rejects_missing_or_stale_funding(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    store = SqlRiskSnapshotStore(database.engine)
    for product_id, instrument_id in (
        ("btc_accumulation", "binance:spot:BTCUSDT"),
        ("active_income", "binance:futures:BTCUSDT:USDT"),
    ):
        store.save(
            {
                "kind": "market_data_input",
                "product_id": product_id,
                "instrument_id": instrument_id,
                "values": {
                    "close": 100.0,
                    "spread_bps": 1.0,
                    "visible_depth": 1_000_000.0,
                    "volatility": 0.2,
                },
            },
            created_at=NOW,
        )
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={
            "btc_accumulation": {"account_id": "spot"},
            "active_income": {"account_id": "futures"},
        },
        accounts={
            "spot": {"market": "spot"},
            "futures": {"market": "usdt_futures"},
        },
    )

    spot, _ = source._market("btc_accumulation", NOW)
    missing_futures, _ = source._market("active_income", NOW)

    assert spot["binance:spot:BTCUSDT"]["funding"] == 0.0
    assert spot["binance:spot:BTCUSDT"]["market_type"] == "spot"
    assert missing_futures == {}

    store.save(
        {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": "binance:futures:BTCUSDT:USDT",
            "values": {"funding": 0.0001},
        },
        created_at=NOW,
    )
    fresh_futures, _ = source._market("active_income", NOW)
    stale_at = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=28_801)).isoformat()
    stale_futures, _ = source._market("active_income", stale_at)

    assert fresh_futures["binance:futures:BTCUSDT:USDT"]["funding"] == 0.0001
    assert stale_futures == {}


def test_closed_kline_does_not_carry_derived_market_fields() -> None:
    event = normalise_public_event(
        market="spot",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": 1787824800000,
            "s": "BTCUSDT",
            "k": {
                "T": 1787824800000,
                "i": "1m",
                "c": "100.0",
                "x": True,
                "spread_bps": 1.0,
                "visible_depth": 1_000_000.0,
                "volatility": 0.2,
                "funding": 0.0001,
            },
        },
    )

    assert _market_snapshot_values(event) == {"close": 100.0}


def test_funding_event_is_persisted_without_a_fabricated_candle(tmp_path: Path) -> None:
    database = _database(tmp_path)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="test:data-writer",
        node_id="test",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
    )
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@markPrice@1s",
        receive_timestamp=NOW,
        payload={
            "e": "markPriceUpdate",
            "E": 1787824800000,
            "s": "BTCUSDT",
            "p": "100.0",
            "r": "0.0001",
            "T": 1787824800000,
        },
    )
    raw_funding = {
        "instrument_id": event.instrument_id,
        "event_type": MarketEventType.FUNDING_RATE,
        "exchange_timestamp": event.exchange_timestamp,
        "receive_timestamp": event.receive_timestamp,
        "sequence": event.sequence,
        "payload": {
            "data": {"funding_rate": 0.0001},
            "source_event_id": event.event_id,
        },
    }
    queue.enqueue(
        job_id="test:funding",
        name="market_event_write",
        payload={
            "venue": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "event": raw_funding,
        },
        available_at=NOW,
    )
    result = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="test:data-writer",
        root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    ).run_once(now=NOW)

    assert result["reason_code"] == "market_event_written"
    snapshot = SqlRiskSnapshotStore(database.engine).get(result["market_snapshot_ids"][0])
    assert snapshot["values"] == {"funding": 0.0001}


def test_realistic_market_events_build_complete_spot_and_futures_state(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="test:data-writer",
        node_id="test",
        role="data-writer",
        capabilities=("market_event_write",),
        observed_at=NOW,
    )
    events = []
    start = dt.datetime.fromisoformat(NOW) - dt.timedelta(minutes=4)
    for market, closes in (
        ("spot", (100.0, 101.0, 103.0)),
        ("futures", (100.0, 102.0, 101.0)),
    ):
        for index, close in enumerate(closes):
            observed_at = (start + dt.timedelta(minutes=index)).isoformat()
            close_ms = int(dt.datetime.fromisoformat(observed_at).timestamp() * 1_000) - 1
            events.append(
                (
                    market,
                    normalise_public_event(
                        market=market,
                        stream="btcusdt@kline_1m",
                        receive_timestamp=observed_at,
                        payload={
                            "e": "kline",
                            "E": close_ms + 1,
                            "s": "BTCUSDT",
                            "k": {
                                "t": close_ms - 59_999,
                                "T": close_ms,
                                "i": "1m",
                                "o": str(close - 1),
                                "h": str(close + 1),
                                "l": str(close - 2),
                                "c": str(close),
                                "v": "25",
                                "x": True,
                            },
                        },
                    ),
                )
            )
        observed_at = (start + dt.timedelta(minutes=3)).isoformat()
        event_ms = int(dt.datetime.fromisoformat(observed_at).timestamp() * 1_000)
        events.append(
            (
                market,
                normalise_public_event(
                    market=market,
                    stream="btcusdt@bookTicker",
                    receive_timestamp=observed_at,
                    payload={
                        "e": "bookTicker",
                        "E": event_ms,
                        "s": "BTCUSDT",
                        "b": "100.99",
                        "B": "500",
                        "a": "101.01",
                        "A": "500",
                    },
                ),
            )
        )
    funding_at = (start + dt.timedelta(minutes=3, seconds=1)).isoformat()
    funding_ms = int(dt.datetime.fromisoformat(funding_at).timestamp() * 1_000)
    events.append(
        (
            "futures",
            normalise_public_event(
                market="futures",
                stream="btcusdt@markPrice@1s",
                receive_timestamp=funding_at,
                payload={
                    "e": "markPriceUpdate",
                    "E": funding_ms,
                    "s": "BTCUSDT",
                    "p": "101.0",
                    "r": "0.0001",
                    "T": funding_ms,
                },
            ),
        )
    )
    for index, (market, event) in enumerate(events):
        queue.enqueue(
            job_id=f"real-market-event:{index:02d}",
            name="market_event_write",
            payload={
                "venue": "binance",
                "market": market,
                "symbol": "BTCUSDT",
                "event": to_primitive(event),
            },
            available_at=event.receive_timestamp,
        )
    writer = DatabaseMarketDataWriter(
        queue=queue,
        worker_id="test:data-writer",
        root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={
            "spot": ("btc_accumulation",),
            "futures": ("active_income",),
        },
    )
    assert all(writer.run_once(now=NOW)["reason_code"] == "market_event_written" for _ in events)
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=SqlRiskSnapshotStore(database.engine),
        products={
            "btc_accumulation": {"account_id": "spot"},
            "active_income": {"account_id": "futures"},
        },
        accounts={
            "spot": {"market": "spot"},
            "futures": {"market": "usdt_futures"},
        },
    )

    spot, _ = source._market("btc_accumulation", NOW)
    futures, _ = source._market("active_income", NOW)

    assert set(spot) == {"binance:spot:BTCUSDT"}
    assert spot["binance:spot:BTCUSDT"]["funding"] == 0.0
    assert spot["binance:spot:BTCUSDT"]["market_type"] == "spot"
    assert spot["binance:spot:BTCUSDT"]["volatility"] > 0.0
    assert set(futures) == {"binance:futures:BTCUSDT:USDT"}
    assert futures["binance:futures:BTCUSDT:USDT"]["funding"] == 0.0001
    assert futures["binance:futures:BTCUSDT:USDT"]["market_type"] == "futures"
    assert futures["binance:futures:BTCUSDT:USDT"]["volatility"] > 0.0


def test_paper_account_authority_requires_explicit_balances() -> None:
    with pytest.raises(AccountAuthorityError, match="explicit starting balances"):
        AccountReconciliationService._paper(
            {"account_id": "paper", "paper_starting_positions": {}},
            {"product_id": "product"},
        )


def test_paper_account_authority_rejects_invalid_balances() -> None:
    with pytest.raises(AccountAuthorityError, match="paper account state is invalid"):
        AccountReconciliationService._paper(
            {
                "account_id": "paper",
                "paper_starting_balances": {"USDT": -1.0},
                "paper_starting_positions": {},
            },
            {"product_id": "product"},
        )


def test_authenticated_reconciliation_detects_external_product_exposure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = _database(tmp_path)
    monkeypatch.setenv("TEST_BINANCE_KEY", "test-key")
    monkeypatch.setenv("TEST_BINANCE_SECRET", "test-secret")

    class Broker:
        account_fingerprint = "sha256:" + "1" * 64

        @classmethod
        def account_snapshot(cls, *, expected_symbols):
            assert expected_symbols == ("BTCUSDT",)
            return {
                "balances": {"USDT": 1_000.0},
                "free_balances": {"USDT": 900.0},
                "positions": {"binance:futures:BTCUSDT:USDT": 0.1},
                "regular_orders": [
                    {
                        "symbol": "BTCUSDT",
                        "order_id": "external-order",
                        "client_id": "external-client",
                        "status": "open",
                    }
                ],
                "conditional_orders": [
                    {
                        "symbol": "BTCUSDT",
                        "order_id": "external-stop",
                        "client_id": "external-stop-client",
                        "status": "open",
                    }
                ],
                "used_margin": 100.0,
                "maintenance_margin": 10.0,
                "used_margin_fraction": 0.1,
                "liquidation_buffer_fraction": 0.99,
                "account_mode": "one_way",
                "unknown_exposure": {},
                "account_state_known": True,
                "account_state_authority": "authenticated_rest",
                "account_fingerprint": cls.account_fingerprint,
            }

    service = AccountReconciliationService(
        engine=database.engine,
        products={
            "active_income": {
                "product_id": "active_income",
                "portfolio_id": "active-income-portfolio",
                "account_id": "futures",
                "execution_mode": "live",
                "exchange_symbols": ["BTCUSDT"],
            }
        },
        accounts={
            "futures": {
                "account_id": "futures",
                "market": "usdt_futures",
                "api_key_env": "TEST_BINANCE_KEY",
                "api_secret_env": "TEST_BINANCE_SECRET",
            }
        },
        broker_factory=lambda _account, _market: Broker(),
    )

    result = service.reconcile_once(now=NOW)

    assert result["accounts"][0]["unknown_exposure"] == {
        "position:binance:futures:BTCUSDT:USDT": {
            "exchange_quantity": 0.1,
            "platform_quantity": 0.0,
        },
        "external_order:BTCUSDT:external-order": "open",
        "external_order:BTCUSDT:external-stop": "open",
    }


def test_spot_account_snapshot_detects_assets_and_orders_outside_product_scope() -> None:
    class Client:
        @staticmethod
        def fetch_balance():
            return {
                "total": {"BTC": 0.1, "USDT": 1000.0, "ETH": 2.0},
                "free": {"BTC": 0.1, "USDT": 1000.0, "ETH": 2.0},
                "info": {},
            }

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        api_key="test-key",
        testnet=True,
        quote_asset="USDT",
    )
    broker._client = Client()
    broker.list_account_open_orders = lambda *, conditional: (  # type: ignore[method-assign]
        OpenOrderIdentity(
            symbol="ETHUSDT",
            order_id="external-order",
            client_id="external-client",
            status="open",
            conditional=conditional,
        ),
    )

    snapshot = broker.account_snapshot(expected_symbols=("BTCUSDT",))

    assert snapshot["positions"] == {"binance:spot:BTCUSDT": 0.1}
    assert snapshot["unknown_exposure"] == {
        "asset:ETH": 2.0,
        "ETHUSDT:external-order": "open",
    }


def test_futures_account_snapshot_uses_platform_instrument_identity() -> None:
    class Client:
        @staticmethod
        def fetch_balance():
            return {
                "total": {"USDT": 1000.0},
                "free": {"USDT": 900.0},
                "info": {
                    "totalInitialMargin": "100",
                    "totalMaintMargin": "10",
                    "totalMarginBalance": "1000",
                },
            }

        @staticmethod
        def fetch_position_mode():
            return {"hedged": False}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="test-key",
        testnet=True,
        quote_asset="USDT",
    )
    broker._client = Client()
    broker.list_account_futures_positions = lambda: (  # type: ignore[method-assign]
        FuturesPositionIdentity(symbol="BTC/USDT:USDT", qty=0.1, avg_price=100000.0),
    )
    broker.list_account_open_orders = lambda *, conditional: ()  # type: ignore[method-assign]

    snapshot = broker.account_snapshot(expected_symbols=("BTCUSDT",))

    assert snapshot["positions"] == {"binance:futures:BTCUSDT:USDT": 0.1}
    assert snapshot["unknown_exposure"] == {}


def test_spot_account_snapshot_separates_conditional_order_legs() -> None:
    class Client:
        @staticmethod
        def fetch_balance():
            return {
                "total": {"BTC": 0.1, "USDT": 1000.0},
                "free": {"BTC": 0.1, "USDT": 1000.0},
                "info": {},
            }

        @staticmethod
        def fetch_open_orders(_symbol, params):
            assert params in ({}, {"trigger": True})
            return [
                {
                    "id": "regular-order",
                    "clientOrderId": "regular-client",
                    "symbol": "BTCUSDT",
                    "status": "open",
                    "type": "LIMIT",
                    "stopPrice": "0",
                },
                {
                    "id": "stop-order",
                    "clientOrderId": "stop-client",
                    "symbol": "BTCUSDT",
                    "status": "open",
                    "type": "STOP_LOSS_LIMIT",
                    "stopPrice": "95000",
                },
            ]

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        api_key="test-key",
        testnet=True,
        quote_asset="USDT",
    )
    broker._client = Client()

    snapshot = broker.account_snapshot(expected_symbols=("BTCUSDT",))

    assert [item["order_id"] for item in snapshot["regular_orders"]] == ["regular-order"]
    assert [item["order_id"] for item in snapshot["conditional_orders"]] == ["stop-order"]


def test_live_readiness_accepts_exact_connected_authority(tmp_path: Path, monkeypatch) -> None:
    database = _database(tmp_path)
    configuration = load_split_configuration(ROOT / "config")
    promotion_policies = {
        str(item["policy_id"]): dict(item) for item in configuration["promotion"]["policies"]
    }
    product = copy.deepcopy(
        next(
            item
            for item in configuration["products"]["products"]
            if item["product_id"] == "active_income"
        )
    )
    account = copy.deepcopy(
        next(
            item
            for item in configuration["accounts"]["accounts"]
            if item["account_id"] == product["account_id"]
        )
    )
    product["execution_mode"] = "live"
    account["environment"] = "production"
    instrument_value = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    universe_id = SqlUniverseStore(database.engine).record_snapshot(
        universe_id=str(product["universe_id"]),
        observed_at=NOW,
        observations=(
            InstrumentObservation(
                instrument_value,
                365,
                1_000_000_000,
                1_000_000,
                1.0,
                1_000_000_000,
                0.0001,
                0.2,
                10_000_000,
                1.0,
            ),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    assignment_id = _seed_strategy(
        database,
        product,
        instrument_value,
        universe_id,
        NOW,
        "live-readiness",
        execution_mode="live",
        sleeve_id="directional",
    )
    assignment = SqlActiveStrategyAssignmentRepository(database.engine).by_id(assignment_id)
    assert assignment is not None
    gate_now = "2026-08-27T10:00:01+00:00"
    artefact = SqlStrategyArtefactRepository(database.engine).get(str(assignment["artefact_hash"]))
    monkeypatch.setenv(str(account["api_key_env"]), "test-key")
    monkeypatch.setenv(str(account["api_secret_env"]), "test-secret")
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY", "test-signing-key")
    fingerprint = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="test-key",
        testnet=False,
    ).account_fingerprint
    instrument_payload = dict(to_primitive(instrument_value))
    instrument_payload["instrument_id"] = instrument_value.instrument_id
    configuration_hash = live_authority_configuration_hash(
        product=product,
        account=account,
        instrument_payload=instrument_payload,
        artefact=artefact,
        sleeve_id="directional",
        promotion_policy=promotion_policies[str(product["promotion_policy_id"])],
        risk_configuration=configuration["risk"],
    )
    preflight_id = SqlPreflightRepository(database.engine).append(
        {
            "schema": "platform.production-preflight/v1",
            "strategy_version_id": assignment["strategy_version_id"],
            "product_id": product["product_id"],
            "account_id": account["account_id"],
            "artefact_hash": assignment["artefact_hash"],
            "source_commit_hash": artefact["source_commit_hash"],
            "engine_version": artefact["engine_version"],
            "capital_cap": 0.1,
            "checked_at": gate_now,
            "accepted": True,
            "environment": account["environment"],
            "account_fingerprint": fingerprint,
            "execution_engine_identity": _execution_engine_identity(),
            "instrument_id": instrument_value.instrument_id,
            "sleeve_id": "directional",
            "configuration_hash": configuration_hash,
        }
    )
    SqlApprovalRepository(database.engine).append(
        strategy_version_id=str(assignment["strategy_version_id"]),
        product_id=str(product["product_id"]),
        account_id=str(account["account_id"]),
        artefact_hash=str(assignment["artefact_hash"]),
        source_commit_hash=str(artefact["source_commit_hash"]),
        engine_version=str(artefact["engine_version"]),
        capital_cap=0.1,
        actor="operator",
        approved_at=gate_now,
        payload={
            "schema": "platform.strategy-approval/v1",
            "preflight_id": preflight_id,
            "instrument_id": instrument_value.instrument_id,
            "sleeve_id": "directional",
            "environment": account["environment"],
            "account_fingerprint": fingerprint,
            "execution_engine_identity": _execution_engine_identity(),
            "configuration_hash": configuration_hash,
        },
    )
    rehearsal_fingerprint = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="testnet-key",
        testnet=True,
    ).account_fingerprint
    account_payload = {
        "account_id": account["account_id"],
        "product_id": product["product_id"],
        "balances": {"USDT": 1000.0},
        "free_balances": {"USDT": 1000.0},
        "positions": {},
        "regular_orders": [],
        "conditional_orders": [],
        "used_margin": 0.0,
        "maintenance_margin": 0.0,
        "used_margin_fraction": 0.0,
        "liquidation_buffer_fraction": 1.0,
        "account_mode": "one_way",
        "unknown_exposure": {},
        "account_state_known": True,
        "account_state_authority": "authenticated_rest",
        "account_fingerprint": fingerprint,
        "observed_at": gate_now,
    }
    report_payload = {
        "schema": "platform.connected-testnet-report/v1",
        "environment": "testnet",
        "real_exchange": True,
        "product_id": product["product_id"],
        "account_id": account["account_id"],
        "assignment_id": assignment_id,
        "artefact_hash": assignment["artefact_hash"],
        "forecast_id": "sha256:" + "1" * 64,
        "target_position_snapshot_id": "sha256:" + "2" * 64,
        "risk_assessment_id": "sha256:" + "3" * 64,
        "risk_scopes": ["strategy", "instrument", "sleeve", "product", "account", "global"],
        "risk_accepted": True,
        "execution_engine_identity": _execution_engine_identity(),
        "account_fingerprint": rehearsal_fingerprint,
        "open_acknowledged": True,
        "close_acknowledged": True,
        "user_stream_fill": True,
        "accounting_reconciled": True,
        "recovery_identifiers": {"open": {}, "close": {}},
        "recovery_lookup": True,
        "flat_reconciliation": True,
    }
    assert report_payload["execution_engine_identity"] == connected_execution_engine_identity()
    report_hash = canonical_hash(report_payload)
    report = {
        **report_payload,
        "report_hash": report_hash,
        "signature": hmac.new(
            b"test-signing-key", report_hash.encode(), hashlib.sha256
        ).hexdigest(),
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(account_snapshot).values(
                id=canonical_hash({"account": account_payload}),
                account_id=account["account_id"],
                observed_at=gate_now,
                source="authenticated_rest",
                content_hash=canonical_hash(account_payload),
                payload=account_payload,
            )
        )
        connection.execute(
            insert(platform_rehearsal_report).values(
                id=report_hash,
                product_id=product["product_id"],
                account_id=account["account_id"],
                created_at=gate_now,
                content_hash=report_hash,
                accepted=True,
                payload=report,
            )
        )
    with database.engine.connect() as connection:
        result = _live_product_checks(
            connection=connection,
            product=product,
            accounts={str(account["account_id"]): account},
            promotion_policies=promotion_policies,
            risk_configuration=configuration["risk"],
            now=gate_now,
        )

    assert {key for key, value in result.items() if value is False} == set(), result
    assert [
        item["id"] for item in _active_assignments(database.engine)(instrument_value.instrument_id)
    ] == [assignment_id]

    drifted_product = copy.deepcopy(product)
    drifted_product["account_snapshot_max_age_seconds"] = 30
    with database.engine.connect() as connection:
        drifted = _live_product_checks(
            connection=connection,
            product=drifted_product,
            accounts={str(account["account_id"]): account},
            promotion_policies=promotion_policies,
            risk_configuration=configuration["risk"],
            now=gate_now,
        )
    assert drifted["approval"] is False
    assert drifted["preflight"] is False
    assert drifted["ok"] is False

    assignments = SqlActiveStrategyAssignmentRepository(database.engine)
    assignments.deactivate(str(product["product_id"]))
    after_deactivation = (
        dt.datetime.now(dt.UTC).replace(microsecond=0) + dt.timedelta(seconds=1)
    ).isoformat()
    with database.engine.connect() as connection:
        deactivated = _live_product_checks(
            connection=connection,
            product=product,
            accounts={str(account["account_id"]): account},
            promotion_policies=promotion_policies,
            risk_configuration=configuration["risk"],
            now=after_deactivation,
        )

    assert deactivated["assignment"] is False
    assert _active_assignments(database.engine)(instrument_value.instrument_id) == ()


def test_connected_rehearsal_rejects_test_doubles() -> None:
    with pytest.raises(ConnectedTestnetError, match="injected broker"):
        validate_connected_testnet_configuration(
            {
                "environment": "testnet",
                "queue_backend": "postgresql",
                "product_id": "active_income",
                "injected_broker": object(),
            }
        )


def test_connected_rehearsal_accepts_btc_spot_product() -> None:
    result = validate_connected_testnet_configuration(
        {
            "environment": "testnet",
            "queue_backend": "postgresql",
            "product_id": "btc_accumulation",
        }
    )

    assert result["product_id"] == "btc_accumulation"


def test_connected_rehearsal_emergency_close_restores_futures_position() -> None:
    class Broker:
        closed = []

        @classmethod
        def close_position(cls, symbol):
            cls.closed.append(symbol)

        @staticmethod
        def account_snapshot(*, expected_symbols):
            assert expected_symbols == ("BTCUSDT",)
            return {
                "positions": {},
                "regular_orders": [],
                "conditional_orders": [],
                "unknown_exposure": {},
            }

    _emergency_restore_position(
        broker=Broker(),
        symbol="BTCUSDT",
        market="usdt_futures",
        opening_side=OrderSide.BUY,
        open_quantity=0.001,
        initial_positions={},
    )

    assert Broker.closed == ["BTCUSDT"]


def test_connected_rehearsal_uses_exchange_and_client_order_recovery() -> None:
    calls = []

    class Broker:
        @staticmethod
        def query_order(*, symbol, exchange_order_id, client_order_id):
            calls.append((symbol, exchange_order_id, client_order_id))
            return SimpleNamespace(
                exchange_order_id="exchange-order",
                client_order_id="client-order",
            )

    result = _verify_recovery_lookup(
        venue=SimpleNamespace(broker=Broker()),
        symbol="BTCUSDT",
        submission={
            "exchange_order_id": "exchange-order",
            "client_order_id": "client-order",
        },
    )

    assert result == {
        "exchange_order_id": "exchange-order",
        "client_order_id": "client-order",
    }
    assert calls == [
        ("BTCUSDT", "exchange-order", "client-order"),
        ("BTCUSDT", "", "client-order"),
    ]


def test_ccxt_order_recovery_resolves_client_order_id() -> None:
    class Client:
        calls = []

        @classmethod
        def fetch_order(cls, order_id, symbol, params):
            cls.calls.append((order_id, symbol, params))
            return {
                "id": "exchange-order",
                "clientOrderId": "client-order",
                "status": "closed",
                "filled": 0.001,
                "average": 100_000.0,
            }

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="test-key",
        testnet=True,
        quote_asset="USDT",
    )
    broker._client = Client()
    broker._submission_responses = {}

    state = broker.query_order(
        symbol="BTCUSDT",
        exchange_order_id="",
        client_order_id="client-order",
    )

    assert state.exchange_order_id == "exchange-order"
    assert state.client_order_id == "client-order"
    assert Client.calls == [("", "BTC/USDT:USDT", {"origClientOrderId": "client-order"})]


def test_ccxt_rest_recovery_reads_exact_trades_and_signed_income() -> None:
    class Client:
        trade_calls = []
        income_calls = []

        @classmethod
        def fetch_my_trades(cls, symbol, since, limit, params):
            cls.trade_calls.append((symbol, since, limit, params))
            return [
                {
                    "id": "trade-1",
                    "order": "exchange-order",
                    "clientOrderId": "client-order",
                    "symbol": symbol,
                    "side": "sell",
                    "amount": 0.001,
                    "price": 100_000.0,
                    "fee": {"cost": 0.1, "currency": "USDT"},
                    "timestamp": 1_777_000_000_000,
                }
            ]

        @classmethod
        def fapiPrivateGetIncome(cls, params):
            cls.income_calls.append(params)
            return [
                {
                    "tranId": "income-1",
                    "symbol": "BTCUSDT",
                    "incomeType": "FUNDING_FEE",
                    "income": "-0.25",
                    "asset": "USDT",
                    "time": 1_777_000_000_000,
                }
            ]

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="test-key",
        testnet=True,
        quote_asset="USDT",
    )
    broker._client = Client()

    fills = broker.query_order_fills(
        symbol="BTCUSDT",
        exchange_order_id="exchange-order",
        client_order_id="client-order",
    )
    income = broker.query_income(since=1_776_000_000.0)

    assert len(fills) == 1
    assert fills[0].fee_asset == "USDT"
    assert fills[0].quantity == pytest.approx(0.001)
    assert income[0].amount == pytest.approx(-0.25)
    assert income[0].income_type == "FUNDING_FEE"
    assert Client.trade_calls == [("BTC/USDT:USDT", None, None, {"orderId": "exchange-order"})]
    assert Client.income_calls == [{"startTime": 1_776_000_000_000}]


def test_approved_live_recovery_applies_rest_trade_fills_to_position_and_ledger(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    orders = OrderManager(SqlOrderStore(database.engine))
    positions = PositionManager(SqlPositionStore(database.engine))
    order_id = "rest-recovery-order"
    order = OrderIntent(
        order_id=order_id,
        portfolio_id="active-income-portfolio",
        instrument_id="binance:futures:BTCUSDT:USDT",
        side=OrderSide.SELL,
        quantity=0.01,
        order_type=OrderType.MARKET,
        created_at=NOW,
        metadata={
            "exchange_order_id": "exchange-order",
            "client_order_id": "client-order",
            "reference_price": 100.0,
        },
    )
    orders.create(order)
    orders.persist_for_submission(order_id)
    orders.submitted(order_id)
    orders.acknowledged(order_id, event_at=NOW)
    orders.recovery_required(order_id)

    class Broker:
        def query_order(self, **_kwargs):
            return BrokerOrderState("exchange-order", "client-order", "closed", 0.01, 90.0)

        def query_order_fills(self, **_kwargs):
            return (
                BrokerFill(
                    trade_id="trade-1",
                    exchange_order_id="exchange-order",
                    client_order_id="client-order",
                    symbol="BTCUSDT",
                    side=BrokerOrderSide.SELL,
                    quantity=0.01,
                    price=90.0,
                    fee=0.01,
                    occurred_at=1_777_000_000.0,
                    fee_asset="USDT",
                ),
            )

        def query_income(self, **_kwargs):
            return (
                BrokerIncome(
                    income_id="funding-1",
                    symbol="BTCUSDT",
                    income_type="FUNDING_FEE",
                    amount=-0.25,
                    asset="USDT",
                    occurred_at=dt.datetime.fromisoformat(NOW).timestamp(),
                ),
            )

    venue = SimpleNamespace(
        broker=Broker(),
        instruments={"binance:futures:BTCUSDT:USDT": SimpleNamespace(exchange_symbol="BTCUSDT")},
    )
    approved = ApprovedLiveExecution.__new__(ApprovedLiveExecution)
    approved.order_manager = orders
    approved.positions = positions
    approved.venues = {"active_income": venue}
    approved.ledgers = {
        "active_income": Ledger(
            product_id="active_income",
            accounting_asset="USDT",
            store=SqlLedgerStore(database.engine, product_id="active_income"),
        )
    }

    result = approved._reconcile_missing_order("active_income", order_id)

    assert result["status"] == "reconciled"
    assert result["recovered_fills"] == 1
    assert orders.get(order_id).status.value == "reconciled"
    assert positions.get(
        "active-income-portfolio", "binance:futures:BTCUSDT:USDT"
    ).quantity == pytest.approx(-0.01)
    assert len(approved.ledgers["active_income"].entries) == 1
    first_backfill = approved.backfill_account("active_income", NOW)
    second_backfill = approved.backfill_account("active_income", NOW)
    assert first_backfill["recorded"] == 1
    assert second_backfill["recorded"] == 0
    assert len(approved.ledgers["active_income"].entries) == 2


def test_ccxt_futures_testnet_uses_binance_demo_routing(monkeypatch) -> None:
    import sys

    class Client:
        def __init__(self, options):
            self.options = options
            self.demo_calls = []
            self.sandbox_calls = []

        def enable_demo_trading(self, enabled):
            self.demo_calls.append(enabled)

        def set_sandbox_mode(self, enabled):
            self.sandbox_calls.append(enabled)

    monkeypatch.setitem(
        sys.modules,
        "ccxt",
        SimpleNamespace(__version__="4.5.64", binanceusdm=Client),
    )
    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="test-key",
        testnet=True,
    )

    client = broker._build_client()

    assert client.demo_calls == [True]
    assert client.sandbox_calls == []
    assert client.options["options"]["warnOnFetchOpenOrdersWithoutSymbol"] is False


def test_connected_gateway_uses_testnet_public_endpoints(tmp_path: Path, monkeypatch) -> None:
    database = _database(tmp_path)
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")

    gateway = _connected_gateway(
        database=database,
        config_path=ROOT / "config" / "platform.json",
        account_payload={
            "account_id": "binance-futures-main",
            "market": "usdt_futures",
            "environment": "testnet",
            "api_key_env": "BINANCE_API_KEY",
            "api_secret_env": "BINANCE_API_SECRET",
        },
    )

    assert gateway.testnet is True
    assert {source.url for source in gateway.capture_config.sources} == {
        "wss://stream.testnet.binance.vision/stream",
        "wss://demo-fstream.binance.com/stream",
    }
    assert _stream_endpoints(
        UserStreamAccount(
            account_id="futures",
            market="futures",
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
        )
    ) == (
        "https://demo-fapi.binance.com/fapi/v1/listenKey",
        "wss://demo-fstream.binance.com/ws",
    )
