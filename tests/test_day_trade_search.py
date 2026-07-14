import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.day_trade_search as dts
import src.feature_screener as fs
from src.day_trade_search import (
    DayTradeConfig,
    StrategyCandidate,
    _generate_pairs_pool,
    combined_mask,
    day_trade_metrics,
    numeric_feature_columns,
    score_candidate,
    simulate_day_trades,
    timeframe_for_feature,
)
from src.discover_patterns import Condition


def _make_5m_data(n=100):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "tf_5m_open": close - rng.uniform(0, 0.05, n),
            "tf_5m_high": close + rng.uniform(0.1, 0.5, n),
            "tf_5m_low": close - rng.uniform(0.1, 0.5, n),
            "tf_5m_close": close,
            "tf_5m_rsi_14": rng.uniform(20, 80, n),
            "tf_15m_rsi_14": rng.uniform(20, 80, n),
        }
    )


def test_timeframe_for_feature():
    assert timeframe_for_feature("tf_5m_rsi_14") == "5m"
    assert timeframe_for_feature("tf_15m_close") == "15m"
    assert timeframe_for_feature("tf_1h_ema_20") == "1h"


def test_combined_mask():
    data = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    conditions = (
        Condition("a", "value_ge", 2, "a >= 2"),
        Condition("b", "value_ge", 2, "b >= 2"),
    )
    result = combined_mask(data, conditions)
    assert result.tolist() == [False, True, False]


def test_simulate_day_trades_take_profit():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"),
            "tf_5m_open": [100] * 20,
            "tf_5m_high": [100, 101, 101, 101, 101] + [100] * 15,
            "tf_5m_low": [100, 99.9, 99.9, 99.9, 99.9] + [100] * 15,
            "tf_5m_close": [100] * 20,
        }
    )
    signal = pd.Series([True] + [False] * 19)
    config = DayTradeConfig(
        take_profit=0.01,
        stop_loss=0.02,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=8,
    )
    trades = simulate_day_trades(data, signal, "long", config)
    assert len(trades) == 1
    assert trades["exit_reason"].iloc[0] == "take_profit"
    assert round(trades["net_return"].iloc[0], 6) == 0.01


def test_simulate_day_trades_stop_loss():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"),
            "tf_5m_open": [100] * 20,
            "tf_5m_high": [100, 100, 100, 100, 100] + [100] * 15,
            "tf_5m_low": [100, 98, 98, 98, 98] + [100] * 15,
            "tf_5m_close": [100] * 20,
        }
    )
    signal = pd.Series([True] + [False] * 19)
    config = DayTradeConfig(
        take_profit=0.05,
        stop_loss=0.01,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=8,
    )
    trades = simulate_day_trades(data, signal, "long", config)
    assert len(trades) == 1
    assert trades["exit_reason"].iloc[0] == "stop"
    assert round(trades["net_return"].iloc[0], 6) == -0.01


def test_simulate_day_trades_short_uses_linear_usdt_futures_return():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC"),
            "tf_5m_open": [100.0, 100.0, 90.0, 90.0],
            "tf_5m_high": [100.0, 100.0, 100.0, 90.0],
            "tf_5m_low": [100.0, 100.0, 90.0, 90.0],
            "tf_5m_close": [100.0, 100.0, 90.0, 90.0],
        }
    )
    config = DayTradeConfig(
        take_profit=0.5,
        stop_loss=0.5,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=1,
    )

    trades = simulate_day_trades(data, pd.Series([True, False, False, False]), "short", config)

    assert trades["gross_return"].iloc[0] == pytest.approx(0.10)


def test_simulate_day_trades_daily_stop():
    n = 100
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": [100.0] * n,
            "tf_5m_high": [100.0] * n,
            "tf_5m_low": [97.0] * n,
            "tf_5m_close": [100.0] * n,
        }
    )
    signal = pd.Series([True, False] * (n // 2))
    config = DayTradeConfig(
        take_profit=0.05,
        stop_loss=0.01,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=4,
        risk_per_trade=1.0,
        daily_stop_loss=-0.02,
        max_consecutive_losses=100,
        cooldown_bars=0,
    )
    trades = simulate_day_trades(data, signal, "long", config)
    daily_pnl = {}
    for _, t in trades.iterrows():
        day = pd.Timestamp(t["entry_time"]).date()
        daily_pnl[day] = daily_pnl.get(day, 0) + t["sized_return"]
    for day_total in daily_pnl.values():
        assert day_total >= -0.03


def test_simulate_day_trades_cooldown():
    n = 60
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": [100.0] * n,
            "tf_5m_high": [100.0] * n,
            "tf_5m_low": [97.0] * n,
            "tf_5m_close": [100.0] * n,
        }
    )
    signal = pd.Series([True, False, False, False, False, False] * 10)
    config = DayTradeConfig(
        take_profit=0.05,
        stop_loss=0.01,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=4,
        risk_per_trade=1.0,
        daily_stop_loss=-1.0,
        max_consecutive_losses=2,
        cooldown_bars=12,
    )
    trades = simulate_day_trades(data, signal, "long", config)
    assert trades.attrs.get("cooldown_triggers", 0) > 0
    no_cool = DayTradeConfig(
        take_profit=0.05,
        stop_loss=0.01,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=4,
        risk_per_trade=1.0,
        daily_stop_loss=-1.0,
        max_consecutive_losses=100,
        cooldown_bars=0,
    )
    trades_no_cool = simulate_day_trades(data, signal, "long", no_cool)
    assert len(trades) <= len(trades_no_cool)


def test_simulate_day_trades_position_sizing():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"),
            "tf_5m_open": [100] * 20,
            "tf_5m_high": [100, 101, 101, 101, 101] + [100] * 15,
            "tf_5m_low": [100] * 20,
            "tf_5m_close": [100] * 20,
        }
    )
    signal = pd.Series([True] + [False] * 19)
    config = DayTradeConfig(
        take_profit=0.01,
        stop_loss=0.005,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=8,
        risk_per_trade=0.003,
    )
    trades = simulate_day_trades(data, signal, "long", config)
    assert len(trades) == 1
    expected_size = 0.25
    assert abs(trades["position_size"].iloc[0] - expected_size) < 1e-9
    assert abs(trades["sized_return"].iloc[0] - trades["net_return"].iloc[0] * expected_size) < 1e-9


def test_simulate_day_trades_respects_explicit_position_cap():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC"),
            "tf_5m_open": [100] * 20,
            "tf_5m_high": [100, 101, 101, 101, 101] + [100] * 15,
            "tf_5m_low": [100] * 20,
            "tf_5m_close": [100] * 20,
        }
    )
    signal = pd.Series([True] + [False] * 19)
    config = DayTradeConfig(
        take_profit=0.01,
        stop_loss=0.005,
        fee_bps=0,
        slippage_bps=0,
        horizon_bars=8,
        risk_per_trade=0.003,
        max_position_fraction=0.1,
    )

    trades = simulate_day_trades(data, signal, "long", config)

    assert trades["position_size"].iloc[0] == pytest.approx(0.1)


def test_day_trade_metrics_basic():
    trades = pd.DataFrame(
        {
            "net_return": [0.01, -0.005, 0.008],
            "sized_return": [0.006, -0.003, 0.0048],
            "entry_time": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 10:00"]
            ),
            "holding_bars": [4, 3, 5],
        }
    )
    trades.attrs["daily_stop_hits"] = 1
    trades.attrs["cooldown_triggers"] = 0
    m = day_trade_metrics(trades)
    assert m["trades"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-6
    assert m["avg_trades_per_day"] == 1.5
    assert m["daily_stop_hits"] == 1


def test_day_trade_metrics_empty():
    trades = pd.DataFrame(
        columns=[
            "net_return",
            "sized_return",
            "entry_time",
            "holding_bars",
        ]
    )
    m = day_trade_metrics(trades)
    assert m["trades"] == 0
    assert m["total_return"] == 0.0


def test_score_candidate_runs():
    data = _make_5m_data(200)
    data["future_return_4_bars"] = data["tf_5m_close"].shift(-4) / data["tf_5m_close"] - 1
    data = data.dropna().reset_index(drop=True)
    train = data.iloc[:140].copy()
    test = data.iloc[140:].copy()
    candidate = StrategyCandidate(
        direction="long",
        horizon_bars=4,
        conditions=(Condition("tf_5m_rsi_14", "value_le", 50, "rsi low"),),
    )
    config = DayTradeConfig(
        take_profit=0.002,
        stop_loss=0.002,
        fee_bps=5,
        slippage_bps=2,
        horizon_bars=4,
    )
    row = score_candidate(train, test, candidate, config)
    assert "train_trades" in row
    assert "test_trades" in row
    assert "test_win_rate" in row
    assert row["direction"] == "long"


def test_strategy_candidate_timeframes():
    candidate = StrategyCandidate(
        direction="long",
        horizon_bars=4,
        conditions=(
            Condition("tf_5m_rsi_14", "value_le", 30, "5m rsi low"),
            Condition("tf_15m_ema_20", "value_ge", 100, "15m ema high"),
        ),
    )
    assert candidate.timeframes == ("15m", "5m")
    assert "AND" in candidate.rule


def test_generate_pairs_pool_cross_tf():
    conditions = [
        Condition("tf_5m_rsi_14", "value_ge", 70, "5m rsi high"),
        Condition("tf_5m_ema_20", "value_ge", 100, "5m ema high"),
        Condition("tf_15m_rsi_14", "value_le", 30, "15m rsi low"),
        Condition("tf_15m_ema_20", "value_le", 100, "15m ema low"),
    ]
    pairs = _generate_pairs_pool(conditions, [0, 1, 2, 3], max_pairs=100)
    for l_idx, r_idx in pairs:
        l_tf = timeframe_for_feature(conditions[l_idx].feature)
        r_tf = timeframe_for_feature(conditions[r_idx].feature)
        assert l_tf != r_tf


def test_condition_json_roundtrip():
    original = Condition("a", "ratio_ge", 1.5, "a/b >= 1.5", feature_b="b")
    payload = {
        "feature": original.feature,
        "kind": original.kind,
        "threshold": original.threshold,
        "description": original.description,
        "feature_b": original.feature_b,
    }
    deserialized = Condition(**json.loads(json.dumps(payload)))
    assert deserialized == original


def test_numeric_feature_columns_excludes_label_and_diagnostics():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC"),
            "tf_5m_open": np.arange(10, dtype=float),
            "tf_5m_high": np.arange(10, dtype=float),
            "tf_5m_low": np.arange(10, dtype=float),
            "tf_5m_close": np.arange(10, dtype=float),
            "tf_5m_rsi_14": np.arange(10, dtype=float),
            "label_long_tp50_sl30_h8": np.ones(10, dtype=int),
            "mfe_long_tp50_sl30_h8_exit": np.ones(10, dtype=float),
        }
    )
    cols = numeric_feature_columns(data, "tf_5m_")
    assert "tf_5m_rsi_14" in cols
    assert "label_long_tp50_sl30_h8" not in cols
    assert "mfe_long_tp50_sl30_h8_exit" not in cols


def test_walk_forward_last_screened_out_no_keyerror_and_schema(monkeypatch, tmp_path):
    n = 140
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": np.full(n, 100.0),
            "tf_5m_high": np.full(n, 100.5),
            "tf_5m_low": np.full(n, 99.5),
            "tf_5m_close": np.full(n, 100.0),
            "tf_5m_feat": np.linspace(0, 1, n),
        }
    )
    data["future_return_4_bars"] = data["tf_5m_close"].shift(-4) / data["tf_5m_close"] - 1
    data["label_long_tp50_sl30_h4"] = 1
    data = data.dropna().reset_index(drop=True)

    candidate = StrategyCandidate(
        "long",
        4,
        (Condition("tf_5m_feat", "value_ge", 0.2, "x", threshold_source="quantile", quantile=0.8),),
    )

    monkeypatch.setattr(dts, "load_dataset", lambda *a, **k: data)
    monkeypatch.setattr(dts, "make_candidates", lambda *a, **k: [candidate])
    monkeypatch.setattr(fs, "screen_features", lambda *a, **k: [])

    out = dts.run(
        input_path=Path("unused.parquet"),
        output_dir=tmp_path,
        walk_forward=True,
        horizons=(4,),
        take_profits=(0.005,),
        stop_losses=(0.003,),
        wf_train_bars=60,
        wf_test_bars=20,
        wf_step_bars=20,
        wf_min_windows=2,
        feature_screening="lightgbm",
        min_test_trades=1,
    )
    assert isinstance(out, pd.DataFrame)
    scored = pd.read_csv(tmp_path / "scored_strategies_all.csv")
    required = {
        "train_trades",
        "train_total_return",
        "test_trades",
        "test_total_return",
        "wf_total_windows",
    }
    assert required.issubset(set(scored.columns))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "wf_pass_rate" in report
    assert "train_total_return" not in report


def _fake_wf_row(candidate, take_profit, stop_loss, pass_rate, expectancy):
    row = {
        "direction": candidate.direction,
        "horizon_bars": candidate.horizon_bars,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "timeframes": ",".join(candidate.timeframes),
        "timeframe_count": len(candidate.timeframes),
        "conditions": len(candidate.conditions),
        "rule": candidate.rule,
    }
    row.update(dts._WF_ZEROED_COLUMNS)
    row.update(
        {
            "wf_windows": 2,
            "wf_total_windows": 2,
            "wf_scored_windows": 2,
            "wf_screened_out_windows": 0,
            "wf_pass_rate": pass_rate,
            "wf_expectancy": expectancy,
            "wf_profit_factor_median": 1.2,
            "wf_max_drawdown_worst": -0.02,
            "wf_avg_trades": 10.0,
            "wf_trade_count_stability": 0.1,
            "wf_passes_walk_forward": True,
            "wf_passes": True,
            "wf_returns_sharpe": expectancy * 100,
            "wf_returns_skew": 0.0,
            "wf_returns_kurt": 3.0,
            "test_sharpe_ci_low": 0.0,
            "test_sharpe_ci_high": 0.1,
            "wf_window_returns_json": json.dumps([expectancy, expectancy]),
        }
    )
    return row


def test_walk_forward_sort_uses_aggregate_metrics(monkeypatch, tmp_path):
    n = 140
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": np.full(n, 100.0),
            "tf_5m_high": np.full(n, 100.5),
            "tf_5m_low": np.full(n, 99.5),
            "tf_5m_close": np.full(n, 100.0),
            "tf_5m_feat": np.linspace(0, 1, n),
            "tf_5m_feat2": np.linspace(1, 0, n),
        }
    )
    data["future_return_4_bars"] = data["tf_5m_close"].shift(-4) / data["tf_5m_close"] - 1
    data = data.dropna().reset_index(drop=True)
    c1 = StrategyCandidate(
        "long",
        4,
        (Condition("tf_5m_feat", "value_ge", 0.2, "a", threshold_source="quantile", quantile=0.8),),
    )
    c2 = StrategyCandidate(
        "long",
        4,
        (
            Condition(
                "tf_5m_feat2", "value_ge", 0.2, "b", threshold_source="quantile", quantile=0.8
            ),
        ),
    )

    monkeypatch.setattr(dts, "load_dataset", lambda *a, **k: data)
    monkeypatch.setattr(dts, "make_candidates", lambda *a, **k: [c1, c2])
    monkeypatch.setattr(dts, "write_report", lambda *a, **k: None)

    def fake_score(engine, candidate, scenarios, **kwargs):
        if candidate.rule == "a":
            return [_fake_wf_row(candidate, tp, sl, 0.9, 0.02) for tp, sl in scenarios]
        return [_fake_wf_row(candidate, tp, sl, 0.6, 0.01) for tp, sl in scenarios]

    monkeypatch.setattr(dts, "_score_candidate_walk_forward_day", fake_score)
    out = dts.run(
        input_path=Path("unused.parquet"),
        output_dir=tmp_path,
        walk_forward=True,
        horizons=(4,),
        take_profits=(0.005,),
        stop_losses=(0.003,),
        wf_train_bars=60,
        wf_test_bars=20,
        wf_step_bars=20,
        wf_min_windows=2,
        min_test_trades=1,
        require_multitimeframe=False,
    )
    # Only candidate "a" clears the wf_pass_rate gate (0.9 >= 0.8); "b" (0.6) fails.
    assert len(out) >= 1
    assert out.iloc[0]["rule"] == "a"
    # Holdout columns are present, report-only.
    assert "holdout_total_return" in out.columns


def test_day_trade_walk_forward_end_to_end(tmp_path):
    n = 9_000
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.05, size=n))
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": close + rng.normal(0, 0.01, size=n),
            "tf_5m_high": close + rng.uniform(0.01, 0.2, size=n),
            "tf_5m_low": close - rng.uniform(0.01, 0.2, size=n),
            "tf_5m_close": close,
            "tf_5m_rsi_14": rng.uniform(10, 90, size=n),
            "tf_15m_rsi_14": rng.uniform(10, 90, size=n),
        }
    )
    input_path = tmp_path / "train.parquet"
    data.to_parquet(input_path, index=False)
    output_dir = tmp_path / "out"

    dts.run(
        input_path=input_path,
        output_dir=output_dir,
        walk_forward=True,
        horizons=(4,),
        max_features=2,
        top_conditions=4,
        max_pairs=4,
        max_triples=0,
        condition_depths=(1,),
        take_profits=(0.005,),
        stop_losses=(0.003,),
        enabled_kinds={"value", "delta"},
        ranking_method="spearman",
        wf_train_bars=6_000,
        wf_test_bars=500,
        wf_step_bars=500,
        wf_min_windows=2,
        min_test_trades=1,
        require_multitimeframe=False,
        holdout_fraction=0.2,
        checkpoint_every=2,
    )

    scored = pd.read_csv(output_dir / "scored_strategies_all.csv")
    assert not scored.empty
    for column in ("wf_pass_rate", "wf_expectancy", "wf_windows", "dsr", "candidate_index"):
        assert column in scored.columns
    assert "wf_window_returns_json" not in scored.columns
    assert not (output_dir / "checkpoint.csv").exists()
    assert not (output_dir / "checkpoint_meta.json").exists()
    summary = json.loads((output_dir / "filter_summary.json").read_text())
    assert "positive_wf_expectancy" in summary
    config = json.loads((output_dir / "config.json").read_text())
    assert config["holdout_fraction"] == 0.2


def test_make_candidates_feature_pattern_restricts_universe():
    # Needs enough rows that q10/q20 conditions clear the min_support=500 floor.
    data = _make_5m_data(6_000)
    data["future_return_4_bars"] = data["tf_5m_close"].shift(-4) / data["tf_5m_close"] - 1
    data = data.dropna().reset_index(drop=True)
    candidates = dts.make_candidates(
        data,
        horizons=(4,),
        directions=("long",),
        max_features=5,
        top_conditions=5,
        max_pairs=5,
        max_triples=0,
        rank_sample_rows=1_000,
        condition_depths=(1,),
        ranking_method="spearman",
        enabled_kinds={"value", "delta"},
        feature_pattern="tf_5m_rsi",
    )
    assert candidates
    for candidate in candidates:
        for condition in candidate.conditions:
            assert "tf_5m_rsi" in condition.feature

    with pytest.raises(ValueError, match="matches no feature columns"):
        dts.make_candidates(
            data,
            horizons=(4,),
            directions=("long",),
            max_features=5,
            top_conditions=5,
            max_pairs=5,
            max_triples=0,
            rank_sample_rows=1_000,
            condition_depths=(1,),
            ranking_method="spearman",
            enabled_kinds={"value", "delta"},
            feature_pattern="does_not_exist",
        )


def test_day_trade_regime_breakdown_reports_per_regime_stats():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC"),
            "tf_5m_open": [100.0] * 8,
            "tf_5m_high": [101.0] * 8,
            "tf_5m_low": [99.0] * 8,
            "tf_5m_close": [100.0] * 8,
            "signal": [1] * 8,
            "tf_1d_regime_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    candidate = dts.StrategyCandidate("long", 1, (Condition("signal", "value_ge", 1, "signal"),))
    config = dts.DayTradeConfig(0.005, 0.005, 0, 0, 1)
    out = dts.regime_breakdown(data, candidate, config, "tf_5m_")
    assert set(out) == {"0", "1"}
    assert "dsr" in out["0"]


def test_feature_screen_cache_reuses_same_fold_scenario(monkeypatch):
    n = 20
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_a": np.arange(n, dtype=float),
            "tf_5m_b": np.arange(n, dtype=float) * 2,
            "label_long_tp50_sl30_h8": [0, 1] * (n // 2),
        }
    )
    calls = {"count": 0}

    def fake_screen(*args, **kwargs):
        calls["count"] += 1
        return ["tf_5m_a"]

    monkeypatch.setattr(fs, "screen_features", fake_screen)
    cache = dts.FeatureScreenCache.create()
    features = ["tf_5m_a", "tf_5m_b"]
    first = dts.get_screened_features_cached(
        train,
        "label_long_tp50_sl30_h8",
        "long",
        8,
        0.005,
        0.003,
        features,
        1,
        cache,
        method="importance",
    )
    second = dts.get_screened_features_cached(
        train,
        "label_long_tp50_sl30_h8",
        "long",
        8,
        0.005,
        0.003,
        features,
        1,
        cache,
        method="importance",
    )
    assert first == second == {"tf_5m_a"}
    assert calls["count"] == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_feature_screen_cache_separates_fold_label_and_scenario(monkeypatch):
    n = 20
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n + 1, freq="5min", tz="UTC"),
            "tf_5m_a": np.arange(n + 1, dtype=float),
            "tf_5m_b": np.arange(n + 1, dtype=float) * 2,
            "label_long_tp50_sl30_h8": [0, 1] * 10 + [0],
            "label_short_tp50_sl30_h8": [1, 0] * 10 + [1],
        }
    )
    calls = {"count": 0}

    def fake_screen(*args, **kwargs):
        calls["count"] += 1
        return ["tf_5m_a"]

    monkeypatch.setattr(fs, "screen_features", fake_screen)
    cache = dts.FeatureScreenCache.create()
    features = ["tf_5m_a", "tf_5m_b"]
    common = {
        "direction": "long",
        "horizon_bars": 8,
        "take_profit": 0.005,
        "stop_loss": 0.003,
        "feature_columns": features,
        "max_features": 1,
        "cache": cache,
        "method": "importance",
    }
    dts.get_screened_features_cached(base.iloc[:n], "label_long_tp50_sl30_h8", **common)
    dts.get_screened_features_cached(base.iloc[1:], "label_long_tp50_sl30_h8", **common)
    dts.get_screened_features_cached(base.iloc[:n], "label_short_tp50_sl30_h8", **common)
    dts.get_screened_features_cached(
        base.iloc[:n],
        "label_long_tp50_sl30_h8",
        direction="long",
        horizon_bars=8,
        take_profit=0.008,
        stop_loss=0.003,
        feature_columns=features,
        max_features=1,
        cache=cache,
        method="importance",
    )
    assert calls["count"] == 4
    assert cache.hits == 0
    assert cache.misses == 4


def test_simulate_day_trades_atr_based():
    import pytest

    n = 10
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "tf_5m_open": [100.0] * n,
            "tf_5m_high": [100.5, 102.0] + [100.5] * (n - 2),
            "tf_5m_low": [99.5, 98.0] + [99.5] * (n - 2),
            "tf_5m_close": [100.0] * n,
            "tf_5m_atr": [1.0] * n,
        }
    )
    signal = pd.Series([False] * n)
    signal.iloc[0] = True

    config = dts.DayTradeConfig(
        take_profit=2.0,
        stop_loss=1.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        horizon_bars=5,
        risk_per_trade=0.01,
        use_atr_tp_sl=True,
    )

    trades = dts.simulate_day_trades(data, signal, "long", config)
    assert not trades.empty
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit"] == 99.0
    assert trade["position_size"] == pytest.approx(0.25)
