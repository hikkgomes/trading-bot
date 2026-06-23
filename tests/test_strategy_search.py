import json

import pandas as pd

from src.discover_patterns import Condition
from src.strategy_search import (
    StrategyCandidate,
    _generate_pairs_flat,
    _generate_pairs_pool,
    combined_mask,
    regime_breakdown,
    score_candidate,
    simulate_trades,
    timeframe_for_feature,
    trade_metrics,
)


def test_timeframe_for_feature_extracts_prefix():
    assert timeframe_for_feature("tf_15m_rsi_14") == "15m"
    assert timeframe_for_feature("tf_4h_cmo_7") == "4h"


def test_combined_mask_requires_all_conditions():
    data = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    conditions = (
        Condition("a", "value_ge", 2, "a >= 2"),
        Condition("b", "value_ge", 2, "b >= 2"),
    )

    assert combined_mask(data, conditions).tolist() == [False, True, False]


def test_strategy_candidate_reports_timeframes_and_rule():
    candidate = StrategyCandidate(
        direction="long",
        horizon_bars=4,
        conditions=(
            Condition("tf_15m_rsi_14", "value_le", 30, "15m rsi low"),
            Condition("tf_1h_cmo_14", "value_ge", 10, "1h cmo high"),
        ),
    )

    assert candidate.timeframes == ("15m", "1h")
    assert candidate.rule == "15m rsi low AND 1h cmo high"


def test_simulate_long_trade_uses_next_open_and_take_profit():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "tf_15m_open": [100, 100, 100, 100, 100, 100],
            "tf_15m_high": [100, 101, 102, 100, 100, 100],
            "tf_15m_low": [100, 99.8, 99.8, 99.8, 99.8, 99.8],
            "tf_15m_close": [100, 100, 100, 100, 100, 100],
        }
    )
    signal = pd.Series([True, False, False, False, False, False])

    trades = simulate_trades(
        data,
        signal,
        direction="long",
        horizon_bars=4,
        fee_bps=0,
        slippage_bps=0,
        take_profit=0.01,
        stop_loss=0.01,
    )

    assert len(trades) == 1
    assert trades["exit_reason"].iloc[0] == "take_profit"
    assert round(trades["net_return"].iloc[0], 6) == 0.01


def test_trade_metrics_compounds_returns():
    trades = pd.DataFrame({"net_return": [0.1, -0.05]})

    metrics = trade_metrics(trades)

    assert metrics["trades"] == 2
    assert metrics["win_rate"] == 0.5
    assert round(metrics["total_return"], 6) == 0.045


def test_score_candidate_records_exit_scenario():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC"),
            "tf_15m_open": [100] * 8,
            "tf_15m_high": [100, 101, 101, 101, 101, 101, 101, 101],
            "tf_15m_low": [99] * 8,
            "tf_15m_close": [100] * 8,
            "signal": [1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    candidate = StrategyCandidate(
        direction="long",
        horizon_bars=4,
        conditions=(Condition("signal", "value_ge", 1, "signal >= 1"),),
    )

    row = score_candidate(
        data,
        data,
        candidate,
        fee_bps=0,
        slippage_bps=0,
        take_profit=0.01,
        stop_loss=0.02,
    )

    assert row["take_profit"] == 0.01
    assert row["stop_loss"] == 0.02
    assert row["train_trades"] == 1


def test_generate_pairs_pool_forces_cross_timeframe():
    conditions = [
        Condition("tf_15m_rsi_14", "value_ge", 70, "15m rsi high"),
        Condition("tf_15m_ema_20", "value_ge", 100, "15m ema high"),
        Condition("tf_4h_rsi_14", "value_le", 30, "4h rsi low"),
        Condition("tf_4h_ema_20", "value_le", 100, "4h ema low"),
    ]
    selected_indices = [0, 1, 2, 3]

    pairs = _generate_pairs_pool(conditions, selected_indices, max_pairs=100)

    for left_idx, right_idx in pairs:
        left_tf = timeframe_for_feature(conditions[left_idx].feature)
        right_tf = timeframe_for_feature(conditions[right_idx].feature)
        assert left_tf != right_tf, f"Expected cross-TF pair, got {left_tf} + {right_tf}"


def test_generate_pairs_flat_skips_same_feature():
    conditions = [
        Condition("feat_a", "value_ge", 1, "a high"),
        Condition("feat_a", "value_le", 0, "a low"),
        Condition("feat_b", "value_ge", 1, "b high"),
    ]
    selected_indices = [0, 1, 2]

    pairs = _generate_pairs_flat(conditions, selected_indices, max_pairs=100)

    for left_idx, right_idx in pairs:
        assert conditions[left_idx].feature != conditions[right_idx].feature


def test_condition_json_roundtrip_with_feature_b():
    original = Condition("a", "ratio_ge", 1.5, "a/b >= 1.5", feature_b="b")
    payload = {
        "feature": original.feature,
        "kind": original.kind,
        "threshold": original.threshold,
        "description": original.description,
        **({"feature_b": original.feature_b} if original.feature_b else {}),
    }
    serialized = json.dumps(payload)
    deserialized = Condition(**json.loads(serialized))

    assert deserialized == original
    assert deserialized.feature_b == "b"


def test_condition_json_roundtrip_without_feature_b():
    original = Condition("a", "value_ge", 1.0, "a >= 1")
    payload = {
        "feature": original.feature,
        "kind": original.kind,
        "threshold": original.threshold,
        "description": original.description,
    }
    deserialized = Condition(**json.loads(json.dumps(payload)))

    assert deserialized == original
    assert deserialized.feature_b is None


def test_regime_breakdown_reports_per_regime_stats():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC"),
            "tf_15m_open": [100] * 8,
            "tf_15m_high": [101] * 8,
            "tf_15m_low": [99] * 8,
            "tf_15m_close": [100] * 8,
            "signal": [1] * 8,
            "tf_1d_regime_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    candidate = StrategyCandidate("long", 1, (Condition("signal", "value_ge", 1, "signal"),))
    out = regime_breakdown(data, candidate, 0, 0, 0.005, 0.005)
    assert set(out) == {"0", "1"}
    assert "dsr" in out["0"]
