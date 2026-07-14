import numpy as np

from src.metrics import (
    bootstrap_sharpe_ci,
    cluster_strategies_by_overlap,
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    sharpe_ratio,
)


def test_sharpe_ratio_handles_short_or_flat_series():
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_deflated_sharpe_penalizes_more_trials():
    low_trials = deflated_sharpe_ratio(
        0.3, n_trials=1, skew=0.0, kurt=3.0, n_obs=100, sr_std_trials=0.1
    )
    high_trials = deflated_sharpe_ratio(
        0.3, n_trials=10_000, skew=0.0, kurt=3.0, n_obs=100, sr_std_trials=0.1
    )
    assert 0.0 <= high_trials < low_trials <= 1.0


def test_deflated_sharpe_without_dispersion_is_probabilistic_sr():
    # sr_std_trials=0 disables deflation: result is the plain PSR, independent
    # of n_trials, and a clearly positive SR should score near 1.
    a = deflated_sharpe_ratio(0.5, n_trials=1, skew=0.0, kurt=3.0, n_obs=200)
    b = deflated_sharpe_ratio(0.5, n_trials=100_000, skew=0.0, kurt=3.0, n_obs=200)
    assert a == b
    assert a > 0.99


def test_deflated_sharpe_is_a_probability_not_degenerate():
    # The old implementation collapsed to ~0 for every realistic per-trade SR.
    # A strong SR with modest trial dispersion must keep a meaningful value.
    value = deflated_sharpe_ratio(
        0.4, n_trials=150_000, skew=0.0, kurt=3.0, n_obs=500, sr_std_trials=0.05
    )
    assert value > 0.5
    weak = deflated_sharpe_ratio(
        0.05, n_trials=150_000, skew=0.0, kurt=3.0, n_obs=500, sr_std_trials=0.05
    )
    assert weak < value


def test_bootstrap_sharpe_ci_returns_ordered_bounds():
    low, high = bootstrap_sharpe_ci([0.01, -0.005, 0.02, 0.0, 0.01], n_boot=100)
    assert low <= high


def test_probability_backtest_overfitting_returns_probability():
    matrix = np.array([[0.01, 0.02, -0.01, -0.02], [-0.01, 0.0, 0.02, 0.03]])
    pbo = probability_backtest_overfitting(matrix, n_splits=4)
    assert 0.0 <= pbo <= 1.0


def test_cluster_strategies_by_overlap_groups_similar_masks():
    clusters = cluster_strategies_by_overlap(
        {
            "a": [True, True, False, False],
            "b": [True, True, False, False],
            "c": [False, False, True, True],
        }
    )
    assert any(set(values) == {"a", "b"} for values in clusters.values())
