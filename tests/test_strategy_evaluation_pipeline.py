from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.data.database import PlatformDatabase
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.risk.engine import SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash, validate_job_payload
from src.services.portfolio_engine import DatabasePortfolioTargetBuilder
from src.services.portfolio_service import SqlPortfolioRepository
from src.services.strategy_evaluator import DatabaseStrategyEvaluator


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


def test_strategy_evaluator_completes_queued_diagnostic_assignment() -> None:
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
    claimed = SimpleNamespace(job_id="diagnostic-evaluation", payload=payload)
    queue = MagicMock()
    queue.claim.return_value = claimed
    assignments = MagicMock()
    assignments.by_id.return_value = {
        "id": payload["assignment_id"],
        "active": True,
        "product_id": payload["product_id"],
        "payload": {"diagnostic": True, "promotable": False},
    }
    worker = DatabaseStrategyEvaluator(
        queue=queue,
        worker_id="linux-strategy",
        feature_store=MagicMock(),
        portfolio=MagicMock(),
        assignments=assignments,
    )

    result = worker.run_once(now=payload["evaluated_at"])

    assert result["reason_code"] == "diagnostic_strategy_evaluation_skipped"
    queue.complete.assert_called_once_with(claimed, completed_at=payload["evaluated_at"])
    queue.fail.assert_not_called()


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
    eth_instrument = "binance:futures:ETHUSDT:USDT"
    repository.save_forecast(
        replace(
            forecast,
            strategy_version_id="strategy-v2",
            instrument_id=eth_instrument,
            expected_return=0.008,
        )
    )
    snapshots = SqlRiskSnapshotStore(database.engine)
    market_id = snapshots.save(
        {
            "kind": "market_data_input",
            "values": {"close": 100.0},
        },
        created_at=forecast.valid_from,
    )
    snapshots.save(
        {
            "kind": "canonical_portfolio_risk_state",
            "product_id": "active_income",
            "observed_at": forecast.valid_from,
            "source_snapshot_ids": {
                name: "sha256:" + str(index) * 64
                for index, name in enumerate(
                    (
                        "balances",
                        "positions",
                        "open_orders",
                        "account",
                        "market",
                        "health",
                        "drift",
                    ),
                    1,
                )
            },
            "maximum_state_age_seconds": 5,
            "balances": {"USDT": 1000.0},
            "positions": {},
            "open_orders": [],
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "unknown_exposure": {},
            "market": {
                forecast.instrument_id: {
                    "price": 100.0,
                    "spread_bps": 1.0,
                    "visible_depth": 100_000.0,
                    "volatility": 0.1,
                    "funding": 0.0,
                },
                eth_instrument: {
                    "price": 10.0,
                    "spread_bps": 2.0,
                    "visible_depth": 100_000.0,
                    "volatility": 0.1,
                    "funding": 0.0,
                },
            },
            "correlations": {},
            "beta": {forecast.instrument_id: 0.0, eth_instrument: 0.0},
            "product_drawdown_fraction": 0.0,
            "daily_pnl_fraction": 0.0,
            "global_drawdown_fraction": 0.0,
            "data_age_seconds": 0.0,
            "clock_skew_seconds": 0.0,
            "exchange_connected": True,
            "database_healthy": True,
            "execution_drift": False,
            "model_drift": False,
            "risk_policy_ids": ["active-income", "futures"],
            "portfolio_risk_budget": 0.5,
            "maximum_symbol_fraction": 0.2,
            "maximum_abs_beta": 0.5,
            "maximum_correlation": 0.8,
            "maximum_turnover_fraction": 1.0,
            "maximum_cluster_fraction": 1.0,
            "maximum_product_drawdown_fraction": 0.1,
            "maximum_depth_participation": 0.1,
            "sleeve_budgets": {"directional": 0.5},
            "clusters": {forecast.instrument_id: "btc", eth_instrument: "eth"},
            "cluster_fraction_caps": {"btc": 0.5, "eth": 0.5},
            "trades_today": 0,
        },
        created_at=forecast.valid_from,
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
    target_snapshot = snapshots.get(refs.target_position_snapshot_id)
    assert len(target_snapshot["targets"]) == 2
