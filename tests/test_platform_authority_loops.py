from __future__ import annotations

from sqlalchemy import func, select

from src.data.database import PlatformDatabase, job, service_heartbeat
from src.risk.engine import SqlRiskSnapshotStore
from src.services.health import DatabaseHeartbeatStore
from src.services.portfolio_state import DatabasePortfolioStateWorker
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

    assert (
        worker.schedule_from_latest(
            products={"product": {}}, state_policies={"product": _policy()}, now=NOW
        )
        == 1
    )
    assert worker.run_once(now=NOW)["reason_code"] == "canonical_portfolio_state_published"
    assert all(
        worker.schedule_from_latest(
            products={"product": {}},
            state_policies={"product": _policy()},
            now="2026-08-24T00:00:01+00:00",
        )
        == 0
        for _ in range(10_000)
    )
    with database.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(job)).scalar_one() == 1


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
