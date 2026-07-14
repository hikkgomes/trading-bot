from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.discover_patterns import Condition, condition_mask


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int
    test_bars: int
    step_bars: int
    min_windows: int = 6
    pass_rate: float = 0.8
    embargo_bars: int = 0


def generate_windows(n_rows: int, config: WalkForwardConfig) -> list[tuple[slice, slice]]:
    if min(config.train_bars, config.test_bars, config.step_bars) <= 0:
        raise ValueError("train_bars, test_bars and step_bars must be > 0")
    windows: list[tuple[slice, slice]] = []
    test_start = config.train_bars + config.embargo_bars
    while True:
        test_end = test_start + config.test_bars
        if test_end > n_rows:
            break
        train_end = test_start - config.embargo_bars
        train_start = train_end - config.train_bars
        if train_start < 0:
            break
        windows.append((slice(train_start, train_end), slice(test_start, test_end)))
        test_start += config.step_bars
    if len(windows) < config.min_windows:
        raise ValueError(
            f"Not enough rows for walk-forward min_windows={config.min_windows}: got {len(windows)}"
        )
    return windows


def generate_purged_kfold_windows(
    n_rows: int,
    k: int,
    horizon: int,
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if min(n_rows, k) <= 0:
        raise ValueError("n_rows and k must be > 0")
    if k > n_rows:
        raise ValueError("k cannot exceed n_rows")
    horizon = max(0, int(horizon))
    embargo = max(0, int(embargo))
    folds = np.array_split(np.arange(n_rows), int(k))
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        test_start = int(fold[0])
        test_end = int(fold[-1]) + 1
        purge_start = max(0, test_start - horizon - embargo)
        purge_end = min(n_rows, test_end + horizon + embargo)
        before = np.arange(0, purge_start, dtype=int)
        after = np.arange(purge_end, n_rows, dtype=int)
        train_index = np.concatenate([before, after])
        if train_index.size == 0:
            continue
        windows.append((train_index, fold.astype(int)))
    if not windows:
        raise ValueError("No purged k-fold windows could be generated")
    return windows


def _require_feature(train_window: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in train_window.columns:
        raise ValueError(f"Feature {feature!r} missing from train window")
    return train_window[feature].replace([np.inf, -np.inf], np.nan)


def _require_quantile(condition: Condition) -> float:
    if condition.quantile is None:
        raise ValueError(
            f"Condition {condition.kind} on {condition.feature} is missing quantile metadata"
        )
    return float(condition.quantile)


def _compute_threshold(train_window: pd.DataFrame, condition: Condition) -> float:
    source = condition.threshold_source
    series = _require_feature(train_window, condition.feature)
    if source == "quantile":
        return float(series.dropna().quantile(_require_quantile(condition)))
    if source == "delta_quantile":
        return float(series.diff().dropna().quantile(_require_quantile(condition)))
    if source == "slope_quantile":
        if condition.lookback is None:
            raise ValueError(
                f"Condition {condition.kind} on {condition.feature} is missing lookback metadata"
            )
        slope = (series - series.shift(int(condition.lookback))) / int(condition.lookback)
        return float(slope.dropna().quantile(_require_quantile(condition)))
    if source == "ratio_quantile":
        if condition.feature_b is None:
            raise ValueError(
                f"Condition {condition.kind} on {condition.feature} is missing feature_b metadata"
            )
        series_b = _require_feature(train_window, condition.feature_b).replace(0, np.nan)
        ratio = (series / series_b).replace([np.inf, -np.inf], np.nan)
        return float(ratio.dropna().quantile(_require_quantile(condition)))
    if source == "fixed":
        return float(condition.threshold)
    raise ValueError(
        f"Unknown threshold_source {source!r} for condition {condition.kind} on {condition.feature}"
    )


def refit_conditions(
    train_window: pd.DataFrame,
    candidate,
    base_prefix: str,
):
    _ = base_prefix
    updated: list[Condition] = []
    for c in candidate.conditions:
        threshold = _compute_threshold(train_window, c)
        updated.append(
            Condition(
                feature=c.feature,
                kind=c.kind,
                threshold=threshold,
                description=c.description,
                feature_b=c.feature_b,
                threshold_source=c.threshold_source,
                quantile=c.quantile,
                lookback=c.lookback,
                cross_feature=c.cross_feature,
            )
        )
    return type(candidate)(
        direction=candidate.direction,
        horizon_bars=candidate.horizon_bars,
        conditions=tuple(updated),
    )


def candidate_feature_columns(candidates) -> set:
    """All dataset columns referenced by the candidates' conditions.

    Scoring workers only need these plus base OHLC — loading the full
    multi-thousand-column training table per worker process exhausts RAM.
    """
    needed = set()
    for candidate in candidates:
        for condition in candidate.conditions:
            needed.add(condition.feature)
            if condition.feature_b:
                needed.add(condition.feature_b)
            if condition.cross_feature:
                needed.add(condition.cross_feature)
    return needed


def condition_cache_key(condition: Condition) -> tuple:
    refittable = condition.threshold_source in (
        "quantile",
        "delta_quantile",
        "slope_quantile",
        "ratio_quantile",
    )
    return (
        condition.feature,
        condition.kind,
        condition.threshold_source,
        condition.quantile,
        condition.lookback,
        condition.feature_b,
        condition.cross_feature,
        None if refittable else round(float(condition.threshold), 10),
    )


def with_threshold(condition: Condition, threshold: float) -> Condition:
    return Condition(
        feature=condition.feature,
        kind=condition.kind,
        threshold=threshold,
        description=condition.description,
        feature_b=condition.feature_b,
        threshold_source=condition.threshold_source,
        quantile=condition.quantile,
        lookback=condition.lookback,
        cross_feature=condition.cross_feature,
    )


class WindowConditionCache:
    """Per-window threshold refits and per-condition test masks.

    Both depend only on (window, condition) — never on the candidate or the
    TP/SL scenario — so caching them collapses the dominant cost of scoring a
    large candidate population across walk-forward windows.
    """

    def __init__(self, data: pd.DataFrame, windows: Sequence[tuple[slice, slice]]):
        self.windows = list(windows)
        self.train_frames = [data.iloc[train_slice] for train_slice, _ in self.windows]
        self.test_frames = [data.iloc[test_slice] for _, test_slice in self.windows]
        self.threshold_cache: dict[tuple, float] = {}
        self.mask_cache: dict[tuple, np.ndarray] = {}

    def condition_test_mask(self, window_index: int, condition: Condition) -> np.ndarray:
        key = (window_index, condition_cache_key(condition))
        cached = self.mask_cache.get(key)
        if cached is not None:
            return cached
        threshold = self.threshold_cache.get(key)
        if threshold is None:
            threshold = _compute_threshold(self.train_frames[window_index], condition)
            self.threshold_cache[key] = threshold
        refit = with_threshold(condition, threshold)
        mask = condition_mask(self.test_frames[window_index], refit).fillna(False).to_numpy()
        self.mask_cache[key] = mask
        return mask

    def candidate_test_mask(self, window_index: int, candidate) -> np.ndarray:
        mask = self.condition_test_mask(window_index, candidate.conditions[0])
        for condition in candidate.conditions[1:]:
            mask = mask & self.condition_test_mask(window_index, condition)
        return mask


def aggregate_walk_forward_results(
    window_results: Sequence[dict[str, float]],
    pass_rate_threshold: float = 0.8,
) -> dict[str, float]:
    if not window_results:
        return {"windows": 0, "pass_rate": 0.0, "passes_walk_forward": False}
    returns = np.array([r.get("test_total_return", 0.0) for r in window_results], dtype=float)
    profitable = returns > 0
    pass_rate = float(np.mean(profitable))
    expectancy = float(np.mean([r.get("test_avg_net_return", 0.0) for r in window_results]))
    pf_values = [r.get("test_profit_factor", 0.0) for r in window_results]
    mdd_values = [r.get("test_max_drawdown", 0.0) for r in window_results]
    trades = np.array([r.get("test_trades", 0) for r in window_results], dtype=float)
    screened_out = int(sum(bool(r.get("screened_out", False)) for r in window_results))
    trade_stability = float(np.std(trades) / np.mean(trades)) if np.mean(trades) > 0 else np.nan
    return {
        "windows": int(len(window_results)),
        "total_windows": int(len(window_results)),
        "scored_windows": int(len(window_results) - screened_out),
        "screened_out_windows": screened_out,
        "pass_rate": pass_rate,
        "expectancy": expectancy,
        "profit_factor_median": float(np.median(np.array(pf_values, dtype=float))),
        "max_drawdown_worst": float(np.min(np.array(mdd_values, dtype=float))),
        "avg_trades": float(np.mean(trades)),
        "trade_count_stability": trade_stability,
        "passes_walk_forward": pass_rate >= pass_rate_threshold,
    }


def walk_forward_score_candidate(
    data: pd.DataFrame,
    candidate,
    trade_config,
    wf_config: WalkForwardConfig,
    base_prefix: str,
    score_fn: Callable[[pd.DataFrame, pd.DataFrame, object, object, str], dict[str, float]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    window_results: list[dict[str, float]] = []
    windows = generate_windows(len(data), wf_config)
    for train_slice, test_slice in windows:
        train = data.iloc[train_slice].copy()
        test = data.iloc[test_slice].copy()
        wf_candidate = refit_conditions(train, candidate, base_prefix)
        window_results.append(score_fn(train, test, wf_candidate, trade_config, base_prefix))
    return window_results, aggregate_walk_forward_results(window_results, wf_config.pass_rate)
