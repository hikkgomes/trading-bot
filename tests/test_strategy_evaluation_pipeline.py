from __future__ import annotations

from sqlalchemy import insert

from src.data.database import PlatformDatabase, balance_snapshot
from src.domain._codec import canonical_hash
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.risk.engine import SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash, validate_job_payload
from src.services.portfolio_engine import DatabasePortfolioTargetBuilder
from src.services.portfolio_service import SqlPortfolioRepository


def test_strategy_evaluation_command_binds_event_features_and_assignment() -> None:
    payload = {
        "event_id": "sha256:" + "1" * 64,
        "product_id": "active_income",
        "instrument_id": "binance:futures:BTCUSDT:USDT",
        "assignment_id": "sha256:" + "2" * 64,
        "feature_ids": ["sha256:" + "3" * 64],
        "feature_set_version": "core-bars-v1",
        "evaluated_at": "2026-08-23T00:00:00+00:00",
        "horizon_seconds": 60,
        "producer_identity": "feature-service",
    }
    payload["content_hash"] = build_content_hash(payload)
    clean = validate_job_payload("strategy_evaluation", payload)
    assert clean["feature_ids"] == payload["feature_ids"]


def test_target_builder_consumes_canonical_market_and_balance_snapshots(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'platform.sqlite3'}")
    database.migrate()
    repository = SqlPortfolioRepository(database.engine)
    forecast = AlphaForecast(
        strategy_version_id="strategy-v1",
        product_id="active_income",
        instrument_id="binance:futures:BTCUSDT:USDT",
        direction=ForecastDirection.LONG,
        score=0.8,
        expected_return=0.01,
        confidence=0.8,
        horizon_seconds=60,
        valid_from="2026-08-23T00:00:00+00:00",
        valid_until="2026-08-23T00:01:00+00:00",
        target_volatility=0.1,
        maximum_position=0.1,
        metadata={
            "market_event_id": "sha256:" + "1" * 64,
            "feature_ids": ["sha256:" + "2" * 64],
            "artefact_hash": "sha256:" + "3" * 64,
            "engine_version": "test",
        },
    )
    forecast_id = repository.save_forecast(forecast)
    snapshots = SqlRiskSnapshotStore(database.engine)
    market_id = snapshots.save(
        {
            "kind": "market_data_input",
            "values": {"close": 100.0},
        },
        created_at=forecast.valid_from,
    )
    balance_id = canonical_hash({"account_id": "futures", "balances": {"USDT": 1000.0}})
    with database.engine.begin() as connection:
        connection.execute(
            insert(balance_snapshot).values(
                id=balance_id,
                created_at=forecast.valid_from,
                payload={"account_id": "futures", "balances": {"USDT": 1000.0}},
            )
        )
    builder = DatabasePortfolioTargetBuilder(
        repository=repository,
        snapshot_store=snapshots,
        positions=PositionManager(SqlPositionStore(database.engine)),
        product_configuration={
            "active_income": {
                "account_id": "futures",
                "portfolio_id": "active-income-portfolio",
                "risk_policy_id": "active-income",
            }
        },
        account_configuration={},
    )
    refs = builder(
        {
            "event_id": "sha256:" + "4" * 64,
            "product_id": "active_income",
            "forecast_id": forecast_id,
            "evaluated_at": forecast.valid_from,
            "market_data_snapshot_id": market_id,
        }
    )
    assert refs.target_position_snapshot_id.startswith("sha256:")
    assert refs.market_data_snapshot_id.startswith("sha256:")
