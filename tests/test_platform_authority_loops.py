from __future__ import annotations

import datetime as dt

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
                    "visible_depth": 100.0,
                    "volatility": 0.1,
                    "funding": 0.0,
                }
            },
            "correlations": {},
            "beta": {},
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
    event = normalise_public_event(
        market="futures",
        stream="btcusdt@kline_1m",
        receive_timestamp=NOW,
        payload={
            "e": "kline",
            "E": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
            "s": "BTCUSDT",
            "k": {
                "t": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000) - 59_999,
                "T": int(dt.datetime.fromisoformat(NOW).timestamp() * 1_000),
                "i": "1m",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100.5",
                "v": "10",
                "x": True,
            },
        },
    )
    queue.enqueue(
        job_id="normal-market-event",
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
            == risk_snapshot_count
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
