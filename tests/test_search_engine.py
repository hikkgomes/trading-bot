import json

import numpy as np
import pandas as pd
import pytest

import src.strategy_search as ss
from src.discover_patterns import Condition
from src.strategy_search import (
    SimArrays,
    StrategyCandidate,
    _load_checkpoint,
    _simulate_net_returns_python,
    simulate_net_returns,
    simulate_trades,
)


def make_ohlc(n=400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.4, size=n))
    high = close + rng.uniform(0.05, 0.8, size=n)
    low = close - rng.uniform(0.05, 0.8, size=n)
    open_ = close + rng.normal(0, 0.1, size=n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "tf_15m_open": open_,
            "tf_15m_high": high,
            "tf_15m_low": low,
            "tf_15m_close": close,
        }
    )


def test_numba_python_parity():
    data = make_ohlc()
    arrays = SimArrays.from_dataframe(data)
    rng = np.random.default_rng(11)
    signal = rng.random(len(data)) < 0.15
    for direction in ("long", "short"):
        for pnl_unit in ("usdt", "btc"):
            fast = simulate_net_returns(
                arrays,
                signal,
                direction,
                8,
                5.0,
                1.0,
                0.01,
                0.005,
                pnl_unit,
            )
            slow = _simulate_net_returns_python(
                arrays.open_,
                arrays.high,
                arrays.low,
                arrays.close,
                signal,
                direction == "long",
                8,
                0.01,
                0.005,
                2 * ((5.0 + 1.0) / 10_000),
                pnl_unit == "btc",
            )
            np.testing.assert_array_equal(fast, slow)


def test_fast_path_matches_legacy_simulate_trades():
    """Golden parity: the optimized array path must produce the exact same
    net returns as the original DataFrame simulation."""
    data = make_ohlc()
    arrays = SimArrays.from_dataframe(data)
    rng = np.random.default_rng(3)
    signal_values = rng.random(len(data)) < 0.2
    signal = pd.Series(signal_values, index=data.index)
    for direction in ("long", "short"):
        for pnl_unit in ("usdt", "btc"):
            legacy = simulate_trades(
                data,
                signal,
                direction,
                6,
                fee_bps=5.0,
                slippage_bps=1.0,
                take_profit=0.008,
                stop_loss=0.004,
                pnl_unit=pnl_unit,
            )
            fast = simulate_net_returns(
                arrays,
                signal_values,
                direction,
                6,
                5.0,
                1.0,
                0.008,
                0.004,
                pnl_unit,
            )
            legacy_returns = legacy["net_return"].to_numpy() if not legacy.empty else np.array([])
            np.testing.assert_allclose(fast, legacy_returns, rtol=0, atol=1e-12)


def make_search_dataset(n=9_000, seed=42):
    rng = np.random.default_rng(seed)
    data = make_ohlc(n, seed)
    data["tf_15m_rsi_14"] = rng.uniform(10, 90, size=n)
    data["tf_1h_rsi_14"] = rng.uniform(10, 90, size=n)
    return data


def test_run_walk_forward_end_to_end(tmp_path):
    data = make_search_dataset()
    input_path = tmp_path / "train.parquet"
    data.to_parquet(input_path, index=False)
    output_dir = tmp_path / "out"

    strategies = ss.run(
        input_path=input_path,
        output_dir=output_dir,
        horizons=(4,),
        max_features=2,
        top_conditions=4,
        max_pairs=4,
        max_triples=0,
        condition_depths=(1,),
        min_train_trades=1,
        min_test_trades=1,
        take_profits=(0.008,),
        stop_losses=(0.004,),
        enabled_kinds={"value", "delta"},
        walk_forward=True,
        wf_train_bars=6_000,
        wf_test_bars=500,
        wf_step_bars=500,
        wf_min_windows=2,
        wf_pass_rate=0.5,
        holdout_fraction=0.2,
        checkpoint_every=2,
    )

    scored = pd.read_csv(output_dir / "scored_strategies_all.csv")
    assert not scored.empty
    # Walk-forward stats and DSR present; raw window returns stripped from CSVs.
    for column in ("wf_pass_rate", "wf_expectancy", "wf_windows", "dsr", "candidate_index"):
        assert column in scored.columns
    assert "wf_window_returns_json" not in scored.columns
    # Zeroed split columns confirm no train/test selection happened in WF mode.
    assert (scored["test_total_return"] == 0).all()
    if not strategies.empty:
        assert "holdout_total_return" in strategies.columns
    # Checkpoint cleaned up after a successful run.
    assert not (output_dir / "checkpoint.csv").exists()
    assert not (output_dir / "checkpoint_meta.json").exists()
    assert (output_dir / "report.md").exists()
    config = json.loads((output_dir / "config.json").read_text())
    assert config["holdout_fraction"] == 0.2
    assert "git_sha" in config


def test_run_single_split_ranks_without_test_metrics(tmp_path):
    data = make_search_dataset(n=4_000)
    input_path = tmp_path / "train.parquet"
    data.to_parquet(input_path, index=False)
    output_dir = tmp_path / "out"

    ss.run(
        input_path=input_path,
        output_dir=output_dir,
        horizons=(4,),
        max_features=2,
        top_conditions=4,
        max_pairs=4,
        max_triples=0,
        condition_depths=(1,),
        min_train_trades=1,
        min_test_trades=1,
        take_profits=(0.008,),
        stop_losses=(0.004,),
        enabled_kinds={"value", "delta"},
        checkpoint_every=3,
    )
    scored = pd.read_csv(output_dir / "scored_strategies_all.csv")
    assert not scored.empty
    for column in ("dsr", "test_sharpe_ci_low", "test_sharpe_ci_high"):
        assert column in scored.columns
    # Ranking must be train-led: verify the sort-key column VALUES are in the
    # documented order (comparing values, not row identity, tolerates ties).
    sort_columns = ["dsr", "train_total_return", "train_avg_net_return", "train_trades"]
    resorted = scored.sort_values(sort_columns, ascending=[False] * 4).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        scored[sort_columns].reset_index(drop=True),
        resorted[sort_columns],
    )


def test_candidate_feature_columns_collects_all_references():
    from src.walk_forward import candidate_feature_columns

    candidates = [
        StrategyCandidate(
            "long",
            4,
            (
                Condition("tf_15m_rsi_14", "value_le", 30.0, "a"),
                Condition("tf_1h_ema_20", "ratio_ge", 1.0, "b", feature_b="tf_1h_close"),
            ),
        ),
        StrategyCandidate(
            "short",
            4,
            (
                Condition(
                    "tf_15m_macd", "cross_above", 0.0, "c", cross_feature="tf_15m_macd_signal"
                ),
            ),
        ),
    ]
    needed = candidate_feature_columns(candidates)
    assert needed == {
        "tf_15m_rsi_14",
        "tf_1h_ema_20",
        "tf_1h_close",
        "tf_15m_macd",
        "tf_15m_macd_signal",
    }


def test_load_dataset_column_projection_matches_full_load(tmp_path):
    data = make_search_dataset(n=600)
    path = tmp_path / "train.parquet"
    data.to_parquet(path, index=False)

    full = ss.load_dataset(path, horizons=(4,))
    pruned = ss.load_dataset(path, horizons=(4,), columns=["tf_15m_rsi_14"])

    # Same rows, base OHLC always present, unrequested features dropped.
    assert len(pruned) == len(full)
    assert "tf_15m_rsi_14" in pruned.columns
    assert "tf_1h_rsi_14" not in pruned.columns
    for column in ("timestamp", "tf_15m_open", "tf_15m_high", "tf_15m_low", "tf_15m_close"):
        assert column in pruned.columns
    np.testing.assert_array_equal(
        pruned["future_return_4_bars"].to_numpy(),
        full["future_return_4_bars"].to_numpy(),
    )


def test_load_checkpoint_skips_complete_candidates(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    meta = tmp_path / "checkpoint_meta.json"
    pd.DataFrame(
        [
            {"candidate_index": 0, "take_profit": 0.01, "stop_loss": 0.005},
            {"candidate_index": 0, "take_profit": 0.02, "stop_loss": 0.005},
            {"candidate_index": 1, "take_profit": 0.01, "stop_loss": 0.005},  # partial
        ]
    ).to_csv(checkpoint, index=False)
    meta.write_text(json.dumps({"config_hash": "abc"}), encoding="utf-8")

    rows, done = _load_checkpoint(checkpoint, meta, "abc", n_scenarios=2)
    assert done == {0}
    assert all(row["candidate_index"] == 0 for row in rows)
    assert len(rows) == 2


def test_load_checkpoint_rejects_mismatched_config(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    meta = tmp_path / "checkpoint_meta.json"
    pd.DataFrame([{"candidate_index": 0}]).to_csv(checkpoint, index=False)
    meta.write_text(json.dumps({"config_hash": "old"}), encoding="utf-8")
    with pytest.raises(ValueError, match="different search"):
        _load_checkpoint(checkpoint, meta, "new", n_scenarios=1)


def test_walk_forward_engine_caches_masks_across_candidates():
    data = make_search_dataset(n=2_000)
    windows = [(slice(0, 1_000), slice(1_000, 1_500)), (slice(500, 1_500), slice(1_500, 2_000))]
    engine = ss.WalkForwardEngine(data, windows)
    condition = Condition(
        "tf_15m_rsi_14",
        "value_le",
        0.0,
        "rsi low",
        threshold_source="quantile",
        quantile=0.2,
    )
    candidate_a = StrategyCandidate("long", 4, (condition,))
    candidate_b = StrategyCandidate("short", 4, (condition,))

    mask_a = engine.candidate_test_mask(0, candidate_a)
    cache_size = len(engine.mask_cache)
    mask_b = engine.candidate_test_mask(0, candidate_b)
    # Same condition, different candidate: no new cache entry, same mask object.
    assert len(engine.mask_cache) == cache_size
    np.testing.assert_array_equal(mask_a, mask_b)
    # Threshold is refit per window from that window's train slice.
    engine.candidate_test_mask(1, candidate_a)
    assert len(engine.threshold_cache) == 2
