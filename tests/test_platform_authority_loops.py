from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select

from src.data.binance_market import normalise_public_event
from src.data.database import PlatformDatabase, job, risk_snapshot, service_heartbeat
from src.domain._codec import to_primitive
from src.risk.engine import SqlRiskSnapshotStore
from src.services.accounting_service import AccountingService, DatabaseAccountingWorker
from src.services.data_writer import DatabaseMarketDataWriter
from src.services.health import DatabaseHeartbeatStore
from src.services.portfolio_state import (
    DatabasePortfolioSourceService,
    DatabasePortfolioStateWorker,
)
from src.services.scheduler import DatabaseJobQueue

NOW = "2026-08-24T00:00:00+00:00"


def _policy() -> dict[str, object]:
    return {
        "maximum_state_age_seconds": 30,
        "product_drawdown_fraction": 0.0,
        "daily_pnl_fraction": 0.0,
        "global_drawdown_fraction": 0.0,
        "risk_policy_ids": ["policy"],
        "portfolio_risk_budget": 1.0,
        "maximum_symbol_fraction": 1.0,
        "maximum_abs_beta": 1.0,
        "maximum_correlation": 1.0,
        "maximum_turnover_fraction": 1.0,
        "maximum_cluster_fraction": 1.0,
        "maximum_product_drawdown_fraction": 1.0,
        "maximum_depth_participation": 1.0,
        "sleeve_budgets": {},
        "clusters": {},
        "cluster_fraction_caps": {},
        "trades_today": 0,
    }


def test_unchanged_portfolio_sources_remain_idle_for_10000_cycles(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'authority.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="state-worker",
        node_id="linux-optiplex",
        role="portfolio-state-service",
        capabilities=("portfolio_state_publish",),
        observed_at=NOW,
    )
    store = SqlRiskSnapshotStore(database.engine)
    values = {
        "balances": {"balances": {"USDT": 100.0}},
        "positions": {"positions": {}},
        "open_orders": {"open_orders": []},
        "account": {
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "unknown_exposure": {},
        },
        "market": {
            "market": {
                "instrument": {
                    "price": 1.0,
                    "spread_bps": 1.0,
                    "visible_depth_quote": 100.0,
                    "volatility": 0.1,
                    "funding": 0.0,
                }
            },
            "correlations": {},
            "beta": {},
            "risk_data_available": True,
            "risk_data_missing": [],
        },
        "health": {
            "data_age_seconds": 0.0,
            "clock_skew_seconds": 0.0,
            "exchange_connected": True,
            "database_healthy": True,
        },
        "drift": {"execution_drift": False, "model_drift": False},
    }
    for kind, source_values in values.items():
        store.save(
            {"kind": kind, "product_id": "product", "observed_at": NOW, "values": source_values},
            created_at=NOW,
        )
    worker = DatabasePortfolioStateWorker(queue=queue, worker_id="state-worker", store=store)

    heartbeat = DatabaseHeartbeatStore(database.engine, retention_per_service=32)
    for index in range(10_000):
        now = (dt.datetime.fromisoformat(NOW) + dt.timedelta(seconds=index)).isoformat()
        result = worker.run_once(now=now)
        if result["reason_code"] == "portfolio_state_queue_empty":
            scheduled = worker.schedule_from_latest(
                products={"product": {}}, state_policies={"product": _policy()}, now=now
            )
            if scheduled:
                result = worker.run_once(now=now)
        if index % 250 == 0:
            heartbeat.record(
                service_name="portfolio-state-service",
                node_id="linux-optiplex",
                observed_at=now,
                healthy=True,
                payload={"reason_code": result["reason_code"]},
            )
    assert result["reason_code"] in {
        "portfolio_state_queue_empty",
        "canonical_portfolio_state_published",
    }
    with database.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(job)).scalar_one() == 1
        assert connection.execute(select(func.count()).select_from(risk_snapshot)).scalar_one() == 8
        assert (
            connection.execute(select(func.count()).select_from(service_heartbeat)).scalar_one()
            == 32
        )


def test_heartbeat_retention_is_bounded(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'heartbeat.sqlite3'}")
    database.create_schema()
    store = DatabaseHeartbeatStore(database.engine, retention_per_service=32)
    for index in range(100):
        store.record(
            service_name="service",
            node_id="node",
            observed_at=f"2026-08-24T00:00:{index % 60:02d}+00:00",
            healthy=True,
            payload={"index": index},
        )
    with database.engine.connect() as connection:
        assert (
            connection.execute(select(func.count()).select_from(service_heartbeat)).scalar_one()
            == 32
        )


def test_portfolio_source_readers_fail_closed_without_complete_market_or_health(
    tmp_path,
) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'source-authority.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    store.save(
        {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": "binance:futures:BTCUSDT",
            "values": {"close": 100.0, "spread_bps": 1.0},
        },
        created_at=NOW,
    )
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={},
        accounts={},
    )

    market, market_at = source._market("active_income", NOW)
    health, health_at = source._health(NOW)

    assert market == {}
    assert market_at is None
    assert health == {}
    assert health_at is None


def test_portfolio_source_account_snapshots_are_scoped_and_honour_cutoffs(tmp_path) -> None:
    from src.data.database import account_snapshot, balance_snapshot

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'account-source-cutoff.sqlite3'}")
    database.create_schema()
    direct_rows = (
        ("account-a-tie-a", "account-a", "2026-08-24T00:00:00+00:00", 1.0),
        ("account-a-tie-z", "account-a", "2026-08-24T00:00:00+00:00", 2.0),
        ("account-b-newer", "account-b", "2026-08-24T00:01:00+00:00", 90.0),
        ("account-a-future", "account-a", "2026-08-24T00:02:00+00:00", 3.0),
    )
    fallback_rows = (
        ("fallback-a-tie-a", "fallback-a", "2026-08-24T00:00:00+00:00", 4.0),
        ("fallback-a-tie-z", "fallback-a", "2026-08-24T00:00:00+00:00", 5.0),
        ("fallback-b-newer", "fallback-b", "2026-08-24T00:01:00+00:00", 90.0),
        ("fallback-a-future", "fallback-a", "2026-08-24T00:02:00+00:00", 6.0),
    )
    with database.engine.begin() as connection:
        for identity, account_id, observed_at, cash in direct_rows:
            connection.execute(
                account_snapshot.insert().values(
                    id=identity,
                    account_id=account_id,
                    observed_at=observed_at,
                    source="test",
                    content_hash=f"hash-{identity}",
                    payload={"account_id": account_id, "balances": {"USDT": cash}},
                )
            )
        for identity, account_id, created_at, cash in fallback_rows:
            connection.execute(
                balance_snapshot.insert().values(
                    id=identity,
                    created_at=created_at,
                    payload={"account_id": account_id, "balances": {"USDT": cash}},
                )
            )

    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=SqlRiskSnapshotStore(database.engine),
        products={},
        accounts={},
    )
    direct, direct_at = source._latest_payload(
        balance_snapshot, "account-a", "2026-08-24T00:01:30+00:00"
    )
    fallback, fallback_at = source._latest_payload(
        balance_snapshot, "fallback-a", "2026-08-24T00:01:30+00:00"
    )
    future, future_at = source._latest_payload(
        balance_snapshot, "fallback-a", "2026-08-24T00:02:00+00:00"
    )

    assert direct == {"account_id": "account-a", "balances": {"USDT": 2.0}}
    assert direct_at == "2026-08-24T00:00:00+00:00"
    assert fallback == {"account_id": "fallback-a", "balances": {"USDT": 5.0}}
    assert fallback_at == "2026-08-24T00:00:00+00:00"
    assert future == {"account_id": "fallback-a", "balances": {"USDT": 6.0}}
    assert future_at == "2026-08-24T00:02:00+00:00"
    database.dispose()


def test_portfolio_position_and_order_reads_select_latest_as_of_state(tmp_path) -> None:
    from src.data.database import exchange_order, order_intent, position

    database = PlatformDatabase(
        f"sqlite+pysqlite:///{tmp_path / 'portfolio-history-cutoff.sqlite3'}"
    )
    database.create_schema()
    position_rows = (
        ("btc-a", "2026-08-24T00:00:00+00:00", "portfolio-a", "BTCUSDT", 1.0, 100.0),
        ("btc-z", "2026-08-24T00:00:00+00:00", "portfolio-a", "BTCUSDT", 2.0, 101.0),
        ("eth-open", "2026-08-24T00:01:00+00:00", "portfolio-a", "ETHUSDT", 0.5, 50.0),
        ("btc-other", "2026-08-24T00:02:00+00:00", "portfolio-b", "BTCUSDT", 90.0, 1.0),
        ("btc-flat", "2026-08-24T00:03:00+00:00", "portfolio-a", "BTCUSDT", 0.0, 101.0),
        ("btc-reopen", "2026-08-24T00:05:00+00:00", "portfolio-a", "BTCUSDT", 3.0, 110.0),
    )
    with database.engine.begin() as connection:
        for (
            identity,
            created_at,
            portfolio_id,
            instrument_id,
            quantity,
            entry_price,
        ) in position_rows:
            connection.execute(
                position.insert().values(
                    id=identity,
                    created_at=created_at,
                    payload={
                        "portfolio_id": portfolio_id,
                        "instrument_id": instrument_id,
                        "quantity": quantity,
                        "average_entry_price": entry_price,
                    },
                )
            )
        for order_id, portfolio_id in (
            ("order-open", "portfolio-a"),
            ("order-terminal", "portfolio-a"),
            ("order-checkpoint", "portfolio-a"),
            ("order-other", "portfolio-b"),
        ):
            connection.execute(
                order_intent.insert().values(
                    id=order_id,
                    created_at="2026-08-24T00:00:00+00:00",
                    payload={"order_id": order_id, "portfolio_id": portfolio_id},
                )
            )
        order_rows = (
            ("order-open", 0, "2026-08-24T00:00:00+00:00", "acknowledged"),
            ("order-open", 1, "2026-08-24T00:04:00+00:00", "filled"),
            ("order-terminal", 0, "2026-08-24T00:00:00+00:00", "acknowledged"),
            ("order-terminal", 1, "2026-08-24T00:02:00+00:00", "cancelled"),
            ("order-checkpoint", 0, "2026-08-24T00:03:00+00:00", "acknowledged"),
            ("order-checkpoint", 1, "2026-08-24T00:01:00+00:00", "cancelled"),
            ("order-other", 0, "2026-08-24T00:04:00+00:00", "acknowledged"),
        )
        for order_id, sequence, created_at, status in order_rows:
            connection.execute(
                exchange_order.insert().values(
                    id=f"{order_id}:{sequence}",
                    order_id=order_id,
                    sequence=sequence,
                    created_at=created_at,
                    status=status,
                    payload={
                        "event_type": status,
                        "intent": {
                            "order_id": order_id,
                            "portfolio_id": (
                                "portfolio-b" if order_id == "order-other" else "portfolio-a"
                            ),
                            "instrument_id": "BTCUSDT",
                            "remaining_quantity": 1.0,
                        },
                    },
                )
            )

    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=SqlRiskSnapshotStore(database.engine),
        products={},
        accounts={},
    )
    before_close, before_close_at = source._positions("portfolio-a", "2026-08-24T00:01:30+00:00")
    after_close, after_close_at = source._positions("portfolio-a", "2026-08-24T00:03:30+00:00")
    open_before_terminal, open_before_terminal_at = source._open_orders(
        "portfolio-a", "2026-08-24T00:01:00+00:00"
    )
    open_after_terminal, open_after_terminal_at = source._open_orders(
        "portfolio-a", "2026-08-24T00:03:00+00:00"
    )
    latest_orders, latest_orders_at = source._open_orders(
        "portfolio-a", "2026-08-24T00:04:00+00:00"
    )

    assert before_close == {"BTCUSDT": 2.0, "ETHUSDT": 0.5}
    assert before_close_at == "2026-08-24T00:01:00+00:00"
    assert after_close == {"BTCUSDT": 0.0, "ETHUSDT": 0.5}
    assert after_close_at == "2026-08-24T00:03:00+00:00"
    assert source._position_terms("portfolio-a", "2026-08-24T00:03:30+00:00") == {
        "ETHUSDT": (0.5, 50.0)
    }
    assert [row["order_id"] for row in open_before_terminal] == ["order-open", "order-terminal"]
    assert open_before_terminal_at == "2026-08-24T00:01:00+00:00"
    assert [row["order_id"] for row in open_after_terminal] == ["order-open"]
    assert open_after_terminal_at == "2026-08-24T00:02:00+00:00"
    assert latest_orders == ()
    assert latest_orders_at == "2026-08-24T00:04:00+00:00"
    database.dispose()


def test_portfolio_health_source_can_scope_diagnostic_heartbeats(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'scoped-health.sqlite3'}")
    database.create_schema()
    heartbeat = DatabaseHeartbeatStore(database.engine)
    heartbeat.record(
        service_name="feature-service",
        node_id="linux-optiplex",
        observed_at=NOW,
        healthy=False,
        payload={"reason_code": "service_cycle_failed"},
    )
    heartbeat.record(
        service_name="data-writer",
        node_id="diagnostic-node",
        observed_at=NOW,
        healthy=True,
        payload={"reason_code": "market_event_written"},
    )

    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=SqlRiskSnapshotStore(database.engine),
        products={},
        accounts={},
        health_node_id="diagnostic-node",
    )

    health, observed_at = source._health(NOW)

    assert observed_at == NOW
    assert health["exchange_connected"] is True
    assert health["services"] == {"data-writer@diagnostic-node": True}


def test_portfolio_market_source_derives_realised_volatility_from_closes(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'market-volatility.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    for index, close in enumerate((100.0, 101.0, 103.0)):
        observed_at = (dt.datetime.fromisoformat(NOW) + dt.timedelta(minutes=index)).isoformat()
        store.save(
            {
                "kind": "market_data_input",
                "product_id": "active_income",
                "instrument_id": "binance:futures:BTCUSDT",
                "source_event_type": "candle",
                "source_candle_closed": True,
                "source_timeframe": "1m",
                "source_close_time_ms": int(
                    dt.datetime.fromisoformat(observed_at).timestamp() * 1000
                )
                - 1,
                "availability_time": observed_at,
                "values": {
                    "close": close,
                    "spread_bps": 1.0,
                    "visible_depth_quote": 100.0,
                    "funding": 0.0,
                },
            },
            created_at=observed_at,
        )
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={},
        accounts={},
    )

    market, market_at = source._market("active_income", "2026-08-24T00:02:00+00:00")

    assert market_at == "2026-08-24T00:02:00+00:00"
    assert market["binance:futures:BTCUSDT"]["volatility"] == pytest.approx(
        abs((103 / 101 - 1) - 0.01) / 2
    )


def test_normal_market_and_account_events_publish_all_portfolio_sources(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'normal-events.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    for worker_id, role, capability in (
        ("data", "data-writer", "market_event_write"),
        ("accounting", "accounting-service", "accounting_event"),
        ("state", "portfolio-state-service", "portfolio_state_publish"),
    ):
        queue.register_worker(
            worker_id=worker_id,
            node_id="linux-optiplex",
            role=role,
            capabilities=(capability,),
            observed_at=NOW,
        )
    product = {
        "product_id": "active_income",
        "portfolio_id": "active-income-portfolio",
        "account_id": "futures",
    }
    event_time_ms = int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000)
    market_events = (
        normalise_public_event(
            market="futures",
            stream="btcusdt@kline_1m",
            receive_timestamp=NOW,
            payload={
                "e": "kline",
                "E": event_time_ms,
                "s": "BTCUSDT",
                "k": {
                    "t": event_time_ms - 59_999,
                    "T": event_time_ms,
                    "i": "1m",
                    "o": "100",
                    "h": "101",
                    "l": "99",
                    "c": "100.5",
                    "v": "10",
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
                "E": event_time_ms + 1,
                "s": "BTCUSDT",
                "b": "100.49",
                "B": "500",
                "a": "100.51",
                "A": "500",
            },
        ),
        normalise_public_event(
            market="futures",
            stream="btcusdt@markPrice@1s",
            receive_timestamp=NOW,
            payload={
                "e": "markPriceUpdate",
                "E": event_time_ms + 2,
                "s": "BTCUSDT",
                "p": "100.5",
                "r": "0.0001",
                "T": event_time_ms + 2,
            },
        ),
    )
    candle = market_events[0].payload["data"]
    history = tuple(
        normalise_public_event(
            market="futures",
            stream="btcusdt@kline_1m",
            receive_timestamp=NOW,
            payload={
                **candle,
                "E": event_time_ms - offset * 60000,
                "k": {
                    **candle["k"],
                    "t": event_time_ms - offset * 60000 - 59999,
                    "T": event_time_ms - offset * 60000,
                    "c": close,
                },
            },
        )
        for offset, close in ((2, "100"), (1, "101"))
    )
    market_events = (*history, *market_events)
    for index, event in enumerate(market_events):
        queue.enqueue(
            job_id=f"normal-market-event-{index}",
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
        worker_id="data",
        root=tmp_path / "data",
        snapshot_store=SqlRiskSnapshotStore(database.engine),
        product_ids_by_market={"futures": ("active_income",)},
    )
    for _event in market_events:
        assert writer.run_once(now=NOW)["reason_code"] == "market_event_written"
    queue.enqueue(
        job_id="normal-account-event",
        name="accounting_event",
        payload={
            "kind": "balance",
            "product_id": "active_income",
            "account_id": "futures",
            "observed_at": NOW,
            "balances": {"USDT": 10_000.0},
        },
        available_at=NOW,
    )
    accounting = DatabaseAccountingWorker(
        queue=queue,
        worker_id="accounting",
        service=AccountingService(
            engine=database.engine,
            ledgers={},
            snapshot_store=SqlRiskSnapshotStore(database.engine),
        ),
    )
    assert accounting.run_once(now=NOW)["reason_code"] == "accounting_event_recorded"

    snapshots = SqlRiskSnapshotStore(database.engine)
    source_service = DatabasePortfolioSourceService(
        engine=database.engine,
        store=snapshots,
        products={"active_income": product},
        accounts={"futures": {}},
    )
    heartbeat = DatabaseHeartbeatStore(database.engine)
    heartbeat.record(
        service_name="data-writer",
        node_id="linux-optiplex",
        observed_at=NOW,
        healthy=True,
        payload={"reason_code": "market_event_written"},
    )
    worker = DatabasePortfolioStateWorker(
        queue=queue,
        worker_id="state",
        store=snapshots,
        refresh_sources=source_service.refresh,
    )
    assert (
        worker.schedule_from_latest(
            products={"active_income": product},
            state_policies={"active_income": _policy()},
            now=NOW,
        )
        == 1
    )
    assert worker.run_once(now=NOW)["reason_code"] == "canonical_portfolio_state_published"
    with database.engine.connect() as connection:
        risk_snapshot_count = connection.execute(
            select(func.count()).select_from(risk_snapshot)
        ).scalar_one()

    later = "2026-08-24T00:00:01+00:00"
    heartbeat.record(
        service_name="data-writer",
        node_id="linux-optiplex",
        observed_at=later,
        healthy=True,
        payload={"reason_code": "market_event_written"},
    )
    source_service.refresh("active_income", later)

    with database.engine.connect() as connection:
        assert (
            connection.execute(select(func.count()).select_from(risk_snapshot)).scalar_one()
            == risk_snapshot_count + 1
        )
        source_kinds = {
            str(row[0])
            for row in connection.execute(
                select(risk_snapshot.c.payload["kind"]).where(
                    risk_snapshot.c.payload["product_id"].as_string() == "active_income"
                )
            )
        }
    assert DatabasePortfolioStateWorker.REQUIRED_SOURCES.issubset(source_kinds)
