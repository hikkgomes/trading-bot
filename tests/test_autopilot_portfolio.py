import json

import pandas as pd
import pytest

from research_exploration.hypothesis_schema import (
    EntryScoreRule,
    ExitRule,
    Hypothesis,
    Predicate,
)
from research_exploration.predicates import entry_mask, entry_score_series
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.portfolio import (
    ALPHA_AGGREGATION_SCHEMA,
    ALPHA_FORECAST_SCHEMA,
    AlphaForecast,
    PortfolioPolicy,
    PortfolioPosition,
    PortfolioRiskModel,
    aggregate_forecasts,
    allocate_forecasts,
    forecast_from_strategy,
)
from src.autopilot.runtime import _active_income_portfolio_decision


def _scored_hypothesis():
    return Hypothesis(
        id="scored",
        family="weighted",
        idea="weighted test",
        market_logic="two strong votes can exceed the threshold",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[Predicate("5m", "rsi_14", "ge", reference=50)],
        setup=[Predicate("5m", "adx_14", "ge", reference=20)],
        trigger=[Predicate("5m", "volume_z_20", "ge", reference=1)],
        exit=ExitRule(take_profit=0.02, stop_loss=0.01, horizon_bars=12),
        entry_score=EntryScoreRule(weights=(0.25, 0.25, 0.5), threshold=0.7),
    )


def test_scored_hypothesis_uses_normalized_weighted_vote_and_round_trips():
    hypothesis = _scored_hypothesis()
    frame = pd.DataFrame(
        {
            "tf_5m_rsi_14": [60.0],
            "tf_5m_adx_14": [10.0],
            "tf_5m_volume_z_20": [2.0],
        }
    )

    assert entry_score_series(frame, hypothesis).iloc[-1] == pytest.approx(0.75)
    assert bool(entry_mask(frame, hypothesis).iloc[-1]) is True
    assert Hypothesis.from_dict(hypothesis.to_dict()) == hypothesis


def test_entry_score_requires_one_weight_per_predicate():
    payload = _scored_hypothesis().to_dict()
    payload["entry_score"]["weights"] = [1.0]

    with pytest.raises(ValueError, match="one-to-one"):
        Hypothesis.from_dict(payload)


def test_boolean_strategy_is_adapted_to_unified_alpha_forecast():
    forecast = forecast_from_strategy(
        {
            "id": "trend",
            "direction": "long",
            "base_timeframe": "5m",
            "horizon_bars": 12,
            "take_profit": 0.02,
            "stop_loss": 0.01,
            "baseline_win_rate": 0.55,
            "fees": {"fee_bps": 5, "slippage_bps": 2},
            "metrics": {"dsr_deflated": 0.72},
        },
        product="active_income",
        market="futures",
        symbol="BTCUSDT",
    )

    assert forecast.to_dict()["schema"] == ALPHA_FORECAST_SCHEMA
    assert forecast.score == 1.0
    assert forecast.confidence == pytest.approx(0.72)
    assert forecast.expected_return == pytest.approx(0.0051)
    assert forecast.horizon_seconds == 3600


def _forecast(source_id, symbol, direction, score, confidence, expected_return):
    return AlphaForecast(
        source_id=source_id,
        product="active_income",
        market="futures",
        symbol=symbol,
        direction=direction,
        score=score,
        confidence=confidence,
        expected_return=expected_return,
        horizon_seconds=300,
        generated_at="2026-08-10T00:00:00+00:00",
    )


def test_portfolio_allocator_ranks_alpha_and_enforces_symbol_gross_and_net_caps():
    report = allocate_forecasts(
        [
            _forecast("btc_long", "BTCUSDT", "long", 0.9, 0.8, 0.01),
            _forecast("btc_short", "BTCUSDT", "short", 0.8, 0.8, 0.01),
            _forecast("eth_long", "ETHUSDT", "long", 0.8, 0.8, 0.008),
            _forecast("weak", "SOLUSDT", "long", 0.4, 0.9, 0.02),
        ],
        existing_positions=[PortfolioPosition("active_income", "XRPUSDT", "short", 0.1)],
        requested_fractions={
            "btc_long": 0.25,
            "btc_short": 0.25,
            "eth_long": 0.25,
            "weak": 0.25,
        },
        policy=PortfolioPolicy(
            max_positions=3,
            max_gross_fraction=0.5,
            max_net_fraction=0.25,
            max_symbol_fraction=0.25,
            min_confidence=0.5,
            min_score=0.55,
        ),
    )

    decisions = {item["source_id"]: item for item in report["decisions"]}
    assert decisions["btc_long"]["allocated_fraction"] == pytest.approx(0.25)
    assert decisions["btc_short"]["reason"] == "symbol_already_allocated"
    assert decisions["weak"]["reason"] == "score_below_minimum"
    assert report["exposure"]["gross_fraction"] <= 0.5
    assert abs(report["exposure"]["net_fraction"]) <= 0.25


def _risk_model():
    return PortfolioRiskModel.from_dict(
        {
            "schema": "autopilot.portfolio_risk_model/v1",
            "ok": True,
            "generated_at": "2026-08-10T00:00:00+00:00",
            "benchmark_symbol": "BTCUSDT",
            "correlations": {
                "BTCUSDT": {"BTCUSDT": 1.0, "ETHUSDT": 0.9, "SOLUSDT": -0.8},
                "ETHUSDT": {"BTCUSDT": 0.9, "ETHUSDT": 1.0, "SOLUSDT": -0.7},
                "SOLUSDT": {"BTCUSDT": -0.8, "ETHUSDT": -0.7, "SOLUSDT": 1.0},
            },
            "beta_by_symbol": {"BTCUSDT": 1.0, "ETHUSDT": 1.4, "SOLUSDT": -0.8},
        }
    )


def test_portfolio_allocator_caps_benchmark_beta_exposure():
    report = allocate_forecasts(
        [_forecast("eth_long", "ETHUSDT", "long", 0.9, 0.9, 0.02)],
        existing_positions=[PortfolioPosition("active_income", "BTCUSDT", "long", 0.2)],
        requested_fractions={"eth_long": 0.25},
        policy=PortfolioPolicy(
            max_gross_fraction=0.8,
            max_net_fraction=0.8,
            max_symbol_fraction=0.5,
            max_correlated_fraction=0.5,
            max_abs_beta_fraction=0.4,
        ),
        risk_model=_risk_model(),
    )

    decision = report["decisions"][0]
    assert decision["correlated_existing_fraction"] == pytest.approx(0.18)
    assert decision["allocated_fraction"] == pytest.approx((0.4 - 0.2) / 1.4)
    assert report["exposure"]["benchmark_beta_fraction"] == pytest.approx(0.4)


def test_portfolio_allocator_caps_correlated_exposure():
    report = allocate_forecasts(
        [_forecast("eth_long", "ETHUSDT", "long", 0.9, 0.9, 0.02)],
        existing_positions=[PortfolioPosition("active_income", "BTCUSDT", "long", 0.2)],
        requested_fractions={"eth_long": 0.25},
        policy=PortfolioPolicy(
            max_gross_fraction=0.8,
            max_net_fraction=0.8,
            max_symbol_fraction=0.5,
            max_correlated_fraction=0.3,
            max_abs_beta_fraction=1.0,
        ),
        risk_model=_risk_model(),
    )

    assert report["decisions"][0]["allocated_fraction"] == pytest.approx(0.12)


def test_portfolio_allocator_blocks_new_risk_at_aggregate_drawdown_limit():
    report = allocate_forecasts(
        [_forecast("btc_long", "BTCUSDT", "long", 0.9, 0.9, 0.02)],
        requested_fractions={"btc_long": 0.2},
        policy=PortfolioPolicy(max_drawdown_fraction=0.1),
        portfolio_drawdown_fraction=-0.1,
    )

    assert report["decisions"][0]["reason"] == "portfolio_drawdown_limit_reached"
    assert report["decisions"][0]["allowed"] is False


def test_alpha_aggregator_reinforces_agreement_and_removes_strategy_order_bias():
    forecasts = [
        _forecast("trend", "BTCUSDT", "long", 0.8, 0.8, 0.01),
        _forecast("flow", "BTCUSDT", "long", 0.7, 0.9, 0.008),
        _forecast("reversion", "BTCUSDT", "short", 0.3, 0.5, 0.004),
    ]

    first = aggregate_forecasts(forecasts)
    second = aggregate_forecasts(reversed(forecasts))

    assert first["schema"] == ALPHA_AGGREGATION_SCHEMA
    assert first == second
    assert first["allowed"] is True
    assert first["forecast"]["direction"] == "long"
    assert 0 < first["conflict_fraction"] < 1


def test_alpha_aggregator_blocks_balanced_conflict():
    result = aggregate_forecasts(
        [
            _forecast("long", "BTCUSDT", "long", 0.8, 0.8, 0.01),
            _forecast("short", "BTCUSDT", "short", 0.8, 0.8, 0.01),
        ]
    )

    assert result["allowed"] is False
    assert result["reason"] == "conflicting_alpha_below_minimum_net_score"


def test_alpha_aggregator_blocks_neutral_zero_score_forecasts_without_error():
    result = aggregate_forecasts(
        [
            _forecast("neutral_long", "BTCUSDT", "long", 0.0, 0.8, 0.01),
            _forecast("neutral_short", "BTCUSDT", "short", 0.0, 0.9, 0.01),
        ]
    )

    assert result["allowed"] is False
    assert result["reason"] == "conflicting_alpha_below_minimum_net_score"
    assert result["agreement"] == 0.0


def test_runtime_portfolio_decision_uses_durable_cross_product_exposure(tmp_path):
    state = tmp_path / "eth_state.json"
    state.write_text(
        json.dumps({"open_positions": {"eth_alpha": {"direction": "long", "position_size": 0.2}}}),
        encoding="utf-8",
    )
    product = ProductConfig(
        name="active_income__ethusdt",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="ETHUSDT",
        strategies_path=tmp_path / "artifact.json",
        state_file=state,
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )
    config = AutopilotConfig(
        products=[product],
        active_income_max_open_positions=3,
        active_income_max_gross_fraction=0.6,
        active_income_max_net_fraction=0.4,
        active_income_max_symbol_fraction=0.25,
    )
    forecast = _forecast("btc_long", "BTCUSDT", "long", 0.9, 0.8, 0.01)

    decision = _active_income_portfolio_decision(
        config,
        {"forecast": forecast.to_dict(), "requested_fraction": 0.25},
    )

    assert decision["allowed"] is True
    assert decision["allocated_fraction"] == pytest.approx(0.2)
    assert decision["portfolio_exposure"]["gross_fraction"] == pytest.approx(0.4)
    assert decision["portfolio_exposure"]["net_fraction"] == pytest.approx(0.4)
