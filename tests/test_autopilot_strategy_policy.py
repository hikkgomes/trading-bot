import json

import pytest

from src.autopilot.config import ProductConfig
from src.autopilot.strategy_policy import (
    StrategyPolicyError,
    assert_strategy_artifact_allowed,
    validate_strategy_artifact,
)


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "active.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def strategy(**overrides):
    payload = {
        "id": "policy_r1",
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 8,
        "take_profit": 0.02,
        "stop_loss": 0.01,
        "pnl_unit": "usdt",
        "conditions": [
            {
                "feature": "tf_5m_rsi_14",
                "kind": "value_ge",
                "threshold": 50.0,
                "description": "tf_5m_rsi_14 >= 50.0",
            }
        ],
        "risk": {
            "risk_per_trade": 0.003,
            "max_position_fraction": 0.25,
            "daily_stop_loss": -0.02,
            "max_consecutive_losses": 3,
            "cooldown_bars": 24,
            "max_trades_per_day": 4,
        },
        "fees": {"fee_bps": 5.0, "slippage_bps": 2.0},
        "metrics": {"holdout_total_return": 0.03, "dsr_deflated": 0.72},
    }
    payload.update(overrides)
    return payload


def live_artifact(strategies, **overrides):
    payload = {
        "version": 1,
        "market": "futures",
        "symbol": "BTCUSDT",
        "pnl_unit": "usdt",
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "strategies": strategies,
    }
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def test_strategy_policy_accepts_active_income_default(tmp_path):
    artifact = live_artifact([strategy()])

    assert validate_strategy_artifact(product(tmp_path), artifact) == []


@pytest.mark.parametrize("strategies", ["bad", {"id": "bad"}, 123])
def test_strategy_policy_rejects_non_list_strategies_payload(tmp_path, strategies):
    artifact = live_artifact(strategies)

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["active_income: artifact strategies must be a list."]


def test_strategy_policy_rejects_non_object_strategy_payloads(tmp_path):
    artifact = live_artifact([strategy(), "bad", 123])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["active_income: artifact strategies must be JSON objects; invalid indexes: 1, 2."]


def test_strategy_policy_rejects_missing_live_eligibility_flags(tmp_path):
    artifact = {"version": 1, "market": "futures", "strategies": [strategy()]}

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == [
        "active_income: artifact must explicitly allow paper trading before live review.",
        "active_income: artifact must explicitly allow live trading.",
        "active_income: artifact must explicitly be eligible for promotion.",
    ]


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("paper_trade_allowed", "artifact must explicitly allow paper trading before live review."),
        ("live_allowed", "artifact must explicitly allow live trading."),
        ("promotion_eligible", "artifact must explicitly be eligible for promotion."),
    ],
)
def test_strategy_policy_rejects_non_boolean_live_eligibility_flags(tmp_path, field, message):
    artifact = live_artifact([strategy()])
    artifact[field] = "true"

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == [f"active_income: {message}"]


def test_strategy_policy_rejects_research_only_incubation_artifacts(tmp_path):
    artifact = {
        "schema": "autopilot.incubation_candidates/v1",
        "research_only": True,
        "executable": False,
        "paper_trade_allowed": False,
        "live_allowed": False,
        "promotion_eligible": False,
        "products": {"active_income": []},
    }

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == [
        "active_income: artifact is research-only.",
        "active_income: artifact is marked non-executable.",
        "active_income: artifact is not allowed for paper trading.",
        "active_income: artifact is not allowed for live trading.",
        "active_income: artifact is not eligible for promotion.",
    ]


def test_strategy_policy_allows_paper_only_artifact_for_paper_mode(tmp_path):
    artifact_path = tmp_path / "active.json"
    paper_strategy = strategy(metrics={})
    artifact = {
        "version": 1,
        "market": "futures",
        "paper_trade_allowed": True,
        "live_allowed": False,
        "promotion_eligible": False,
        "strategies": [paper_strategy],
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert validate_strategy_artifact(
        product(tmp_path),
        artifact,
        require_live_eligible=False,
    ) == []
    detail = assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))
    assert detail["ok"] is True
    assert detail["strategies"] == 1


def test_strategy_policy_rejects_non_object_artifact_file(tmp_path):
    artifact_path = tmp_path / "active.json"
    artifact_path.write_text("[]", encoding="utf-8")

    with pytest.raises(StrategyPolicyError, match="must be a JSON object"):
        assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))


def test_strategy_policy_rejects_invalid_json_artifact_file(tmp_path):
    artifact_path = tmp_path / "active.json"
    artifact_path.write_text('{"version": 1,', encoding="utf-8")

    with pytest.raises(StrategyPolicyError, match="must be valid JSON"):
        assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))


def test_strategy_policy_rejects_malformed_strategy_entries(tmp_path):
    artifact_path = tmp_path / "active.json"
    artifact_path.write_text(
        json.dumps(live_artifact(["bad"])),
        encoding="utf-8",
    )

    with pytest.raises(StrategyPolicyError, match="strategies must be JSON objects"):
        assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))


def test_strategy_policy_rejects_paper_only_artifact_for_live_mode(tmp_path):
    artifact_path = tmp_path / "active.json"
    artifact_path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "paper_trade_allowed": True,
                "live_allowed": False,
                "promotion_eligible": False,
                "strategies": [strategy()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StrategyPolicyError, match="not allowed for live trading"):
        assert_strategy_artifact_allowed(
            product(tmp_path, strategies_path=artifact_path, execution_mode="live")
        )


def test_assert_strategy_artifact_allowed_rejects_research_only_file(tmp_path):
    artifact_path = tmp_path / "incubation_candidates.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "autopilot.incubation_candidates/v1",
                "research_only": True,
                "executable": False,
                "paper_trade_allowed": False,
                "live_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StrategyPolicyError, match="artifact is research-only"):
        assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))


def test_strategy_policy_rejects_duplicate_strategy_ids(tmp_path):
    duplicate = strategy()
    artifact = live_artifact([strategy(), duplicate])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert "active_income: duplicate strategy id 'policy_r1'." in errors


@pytest.mark.parametrize("strategy_id", ["", "   ", 123, None])
def test_strategy_policy_rejects_invalid_strategy_id(tmp_path, strategy_id):
    artifact = live_artifact([strategy(id=strategy_id)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert "strategy[0]: id must be a non-empty string." in errors


def test_strategy_policy_rejects_missing_artifact_market(tmp_path):
    artifact = live_artifact([strategy()], market=None)

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "active_income: artifact missing market."


def test_strategy_policy_rejects_missing_live_artifact_symbol(tmp_path):
    artifact = live_artifact([strategy()], symbol=None)

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "active_income: artifact missing symbol."


def test_strategy_policy_rejects_missing_live_artifact_pnl_unit(tmp_path):
    artifact = live_artifact([strategy()], pnl_unit=None)

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "active_income: artifact missing pnl_unit."


def test_strategy_policy_rejects_missing_strategy_market(tmp_path):
    artifact = live_artifact([strategy(market=None)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "policy_r1: missing strategy market."


def test_strategy_policy_rejects_missing_live_strategy_symbol(tmp_path):
    artifact = live_artifact([strategy(symbol=None)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "policy_r1: missing strategy symbol."


def test_strategy_policy_rejects_missing_live_strategy_pnl_unit(tmp_path):
    artifact = live_artifact([strategy(pnl_unit=None)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "policy_r1: missing strategy pnl_unit."


def test_strategy_policy_rejects_artifact_market_mismatch(tmp_path):
    artifact = live_artifact([strategy()], market="spot")

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "active_income: artifact market 'spot' does not match product market 'futures'."


def test_strategy_policy_rejects_strategy_market_mismatch(tmp_path):
    artifact = live_artifact([strategy(market="spot")])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "policy_r1: strategy market 'spot' does not match product market 'futures'."


def test_strategy_policy_accepts_equivalent_symbol_forms(tmp_path):
    artifact = live_artifact([strategy(symbol="BTC/USDT")], symbol="BTCUSDT")

    errors = validate_strategy_artifact(product(tmp_path, symbol="BTC/USDT:USDT"), artifact)

    assert errors == []


def test_strategy_policy_rejects_artifact_symbol_mismatch(tmp_path):
    artifact = live_artifact([strategy()], symbol="ETHUSDT")

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "active_income: artifact symbol 'ETHUSDT' does not match product symbol 'BTCUSDT'."


def test_strategy_policy_rejects_strategy_symbol_mismatch(tmp_path):
    artifact = live_artifact([strategy(symbol="ETHUSDT")])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors[0] == "policy_r1: strategy symbol 'ETHUSDT' does not match product symbol 'BTCUSDT'."


def test_strategy_policy_rejects_negative_holdout(tmp_path):
    artifact = live_artifact([strategy(metrics={"holdout_total_return": -0.01})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any("holdout_total_return -0.010000 must be positive" in error for error in errors)


def test_strategy_policy_rejects_active_income_without_dsr_evidence(tmp_path):
    artifact = live_artifact([strategy(metrics={"holdout_total_return": 0.03})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: missing active income DSR metric."]


@pytest.mark.parametrize("value", [0.0, 0.59])
def test_strategy_policy_rejects_active_income_low_dsr(tmp_path, value):
    artifact = live_artifact([strategy(metrics={"holdout_total_return": 0.03, "dsr_deflated": value})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == [f"policy_r1: active income DSR {value:.6f} below 0.600000."]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_strategy_policy_rejects_invalid_active_income_dsr(tmp_path, value):
    artifact = live_artifact([strategy(metrics={"holdout_total_return": 0.03, "dsr_deflated": value})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any("active income DSR" in error and ("must be finite" in error or "must be numeric" in error) for error in errors)


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, float("nan"), float("inf"), "not-a-number"])
def test_strategy_policy_rejects_invalid_baseline_win_rate(tmp_path, value):
    artifact = live_artifact([strategy(baseline_win_rate=value)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any("baseline_win_rate" in error for error in errors)


def test_strategy_policy_rejects_over_risk_artifact(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                risk={
                    "risk_per_trade": 0.02,
                    "max_position_fraction": 0.75,
                    "daily_stop_loss": -0.10,
                    "max_consecutive_losses": 10,
                    "cooldown_bars": 1,
                    "max_trades_per_day": 99,
                }
            )
        ]
    )

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any("risk_per_trade" in error for error in errors)
    assert any("max_position_fraction" in error for error in errors)
    assert any("daily_stop_loss" in error for error in errors)
    assert any("max_consecutive_losses" in error for error in errors)
    assert any("cooldown_bars" in error for error in errors)
    assert any("max_trades_per_day" in error for error in errors)


def test_strategy_policy_rejects_active_income_strategy_leverage_above_one(tmp_path):
    artifact = live_artifact([strategy(leverage=2)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: leverage must be 1 for active_income."]


def test_strategy_policy_rejects_active_income_artifact_leverage_above_one(tmp_path):
    artifact = live_artifact([strategy()], leverage=2)

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["active_income: leverage must be 1 for active_income."]


def test_strategy_policy_rejects_active_income_cross_margin_metadata(tmp_path):
    artifact = live_artifact([strategy(margin_mode="cross")])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: futures margin_mode must be isolated."]


@pytest.mark.parametrize("value", [0, -1, 4.5, float("nan"), float("inf"), "not-a-number"])
def test_strategy_policy_rejects_invalid_horizon(tmp_path, value):
    artifact = live_artifact([strategy(horizon_bars=value)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any("horizon_bars" in error for error in errors)


def test_strategy_policy_rejects_missing_conditions_for_condition_entry(tmp_path):
    artifact = live_artifact([strategy(conditions=[])])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: conditions entry must include at least one condition."]


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ({"feature": "", "kind": "value_ge", "threshold": 50.0}, "condition[0].feature"),
        ({"feature": "tf_5m_rsi_14", "kind": "unknown", "threshold": 50.0}, "unsupported condition kind"),
        ({"feature": "tf_5m_rsi_14", "kind": "value_ge", "threshold": float("nan")}, "threshold: must be finite"),
        ({"feature": "tf_5m_rsi_14", "kind": "value_ge", "threshold": "bad"}, "threshold: must be numeric"),
        ({"feature": "tf_5m_rsi_14", "kind": "ratio_ge", "threshold": 1.0}, "feature_b: required"),
        (
            {"feature": "tf_5m_rsi_14", "kind": "slope_3_ge", "threshold": 0.0, "lookback": 0},
            "lookback: must be positive",
        ),
        (
            {"feature": "tf_5m_rsi_14", "kind": "value_ge", "threshold": 50.0, "quantile": 1.5},
            "quantile: must be between 0 and 1",
        ),
    ],
)
def test_strategy_policy_rejects_invalid_conditions(tmp_path, condition, message):
    artifact = live_artifact([strategy(conditions=[condition])])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any(message in error for error in errors)


def test_strategy_policy_accepts_supported_condition_variants(tmp_path):
    conditions = [
        {"feature": "tf_5m_rsi_14", "kind": "slope_3_ge", "threshold": 0.0, "lookback": 3},
        {"feature": "tf_5m_rsi_14", "feature_b": "tf_5m_sma_20", "kind": "cross_above", "threshold": 0.0},
        {"feature": "tf_5m_rsi_14", "kind": "divergence_bull_10", "threshold": 0.0},
    ]
    artifact = live_artifact([strategy(conditions=conditions)])

    assert validate_strategy_artifact(product(tmp_path), artifact) == []


def test_strategy_policy_rejects_hypothesis_entry_without_payload(tmp_path):
    artifact = live_artifact([strategy(entry_type="hypothesis", conditions=None, hypothesis=None)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: hypothesis entry must include a hypothesis object."]


def test_strategy_policy_rejects_unknown_entry_type(tmp_path):
    artifact = live_artifact([strategy(entry_type="external")])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert errors == ["policy_r1: entry_type must be 'conditions' or 'hypothesis'."]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_loss", float("nan")),
        ("stop_loss", float("inf")),
        ("take_profit", float("nan")),
        ("take_profit", float("inf")),
        ("take_profit", "not-a-number"),
    ],
)
def test_strategy_policy_rejects_invalid_tp_sl_numbers(tmp_path, field, value):
    artifact = live_artifact([strategy(**{field: value})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any(
        field in error
        and ("must be finite" in error or "must be numeric" in error or "must be an integer" in error)
        for error in errors
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_per_trade", float("nan")),
        ("risk_per_trade", float("inf")),
        ("max_position_fraction", float("nan")),
        ("max_position_fraction", float("inf")),
        ("daily_stop_loss", float("nan")),
        ("daily_stop_loss", float("inf")),
        ("max_consecutive_losses", float("nan")),
        ("max_consecutive_losses", float("inf")),
        ("max_consecutive_losses", 2.5),
        ("cooldown_bars", float("nan")),
        ("cooldown_bars", float("inf")),
        ("cooldown_bars", 24.5),
        ("cooldown_bars", "not-a-number"),
        ("max_trades_per_day", float("nan")),
        ("max_trades_per_day", float("inf")),
        ("max_trades_per_day", 1.5),
        ("max_trades_per_day", "not-a-number"),
    ],
)
def test_strategy_policy_rejects_invalid_risk_numbers(tmp_path, field, value):
    risk = {
        "risk_per_trade": 0.003,
        "max_position_fraction": 0.25,
        "daily_stop_loss": -0.02,
        "max_consecutive_losses": 3,
        "cooldown_bars": 24,
        "max_trades_per_day": 4,
    }
    risk[field] = value
    artifact = live_artifact([strategy(risk=risk)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any(
        field in error
        and ("must be finite" in error or "must be numeric" in error or "must be an integer" in error)
        for error in errors
    )


def test_strategy_policy_rejects_malformed_risk_and_fee_blocks(tmp_path):
    artifact = live_artifact([strategy(risk=["bad"], fees=["bad"])])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert "policy_r1: risk must be an object." in errors
    assert "policy_r1: fees must be an object." in errors


def test_strategy_policy_requires_daily_trade_cap(tmp_path):
    risk = {
        "risk_per_trade": 0.003,
        "max_position_fraction": 0.25,
        "daily_stop_loss": -0.02,
        "max_consecutive_losses": 3,
        "cooldown_bars": 24,
    }
    artifact = live_artifact([strategy(risk=risk)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert "policy_r1: missing required key max_trades_per_day." in errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fee_bps", float("nan"), "must be finite"),
        ("fee_bps", float("inf"), "must be finite"),
        ("fee_bps", -0.1, "must be non-negative"),
        ("slippage_bps", float("nan"), "must be finite"),
        ("slippage_bps", float("inf"), "must be finite"),
        ("slippage_bps", -0.1, "must be non-negative"),
        ("slippage_bps", "not-a-number", "must be numeric"),
    ],
)
def test_strategy_policy_rejects_invalid_fee_numbers(tmp_path, field, value, message):
    fees = {"fee_bps": 5.0, "slippage_bps": 2.0}
    fees[field] = value
    artifact = live_artifact([strategy(fees=fees)])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any(field in error and message in error for error in errors)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_strategy_policy_rejects_invalid_holdout_metric_numbers(tmp_path, value):
    artifact = live_artifact([strategy(metrics={"holdout_total_return": value})])

    errors = validate_strategy_artifact(product(tmp_path), artifact)

    assert any(
        "holdout_total_return" in error and ("must be finite" in error or "must be numeric" in error)
        for error in errors
    )


def test_assert_strategy_artifact_allowed_raises_with_policy_detail(tmp_path):
    artifact_path = tmp_path / "active.json"
    artifact_path.write_text(
        json.dumps({"version": 1, "strategies": [strategy(metrics={"holdout_total_return": 0.0})]}),
        encoding="utf-8",
    )

    with pytest.raises(StrategyPolicyError, match="violates policy"):
        assert_strategy_artifact_allowed(product(tmp_path, strategies_path=artifact_path))


def test_btc_accumulation_policy_requires_btc_pnl_unit(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="usdt",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": 0.02,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: BTC accumulation pnl_unit must be BTC."]


def test_btc_accumulation_policy_rejects_strategy_leverage_above_one(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="btc",
                leverage=2,
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": 0.02,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: leverage must be 1 for btc_accumulation."]


def test_btc_accumulation_policy_rejects_spot_margin_metadata(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="btc",
                margin_mode="isolated",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": 0.02,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: margin_mode is not allowed for spot strategies."]


def test_btc_accumulation_policy_requires_step_aside_short_direction(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="long",
                pnl_unit="btc",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": 0.02,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: BTC accumulation strategies must be spot step-aside shorts."]


def test_btc_accumulation_policy_requires_buy_hold_excess_metric(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="btc",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={"holdout_total_return": 0.03},
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: missing holdout_excess_return_vs_buy_hold metric."]


def test_btc_accumulation_policy_rejects_buy_hold_underperformance(tmp_path):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="btc",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": 0.0,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert errors == ["policy_r1: holdout_excess_return_vs_buy_hold 0.000000 must be positive."]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_btc_accumulation_policy_rejects_invalid_buy_hold_excess_metric(tmp_path, value):
    artifact = live_artifact(
        [
            strategy(
                market="spot",
                direction="short",
                pnl_unit="btc",
                risk={
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.35,
                    "daily_stop_loss": -0.005,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 1,
                },
                metrics={
                    "holdout_total_return": 0.03,
                    "holdout_excess_return_vs_buy_hold": value,
                },
            )
        ],
        market="spot",
        pnl_unit="btc",
    )
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
    )

    errors = validate_strategy_artifact(btc_product, artifact)

    assert any(
        "holdout_excess_return_vs_buy_hold" in error and ("must be finite" in error or "must be numeric" in error)
        for error in errors
    )
