from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import OHLCV

EXCLUDE_SUBSTRINGS = ("timestamp", "future_return", "label_", "bars_to_exit", "target")
OHLCV_SUFFIXES = ("_open", "_high", "_low", "_close", "_volume")


def candidate_feature_columns(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] | None,
    max_features: int,
) -> list[str]:
    if feature_cols:
        return [column for column in feature_cols if column in df.columns][: int(max_features)]
    numeric = df.select_dtypes(include=[np.number]).columns
    cols = [
        column
        for column in numeric
        if not any(sub in column for sub in EXCLUDE_SUBSTRINGS)
        and not column.endswith(OHLCV_SUFFIXES)
        and column not in ("open", "high", "low", "close", "volume")
    ]
    return cols[: int(max_features)]


def screen_feature_columns(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    feature_cols: Iterable[str] | None,
    max_features: int,
    method: str = "spearman",
    min_abs_corr: float = 0.0,
) -> list[str]:
    candidates = candidate_feature_columns(
        df,
        feature_cols=feature_cols,
        max_features=max(int(max_features) * 4, int(max_features)),
    )
    if method in ("none", None):
        return candidates[: int(max_features)]
    if method != "spearman":
        raise ValueError(f"Unsupported feature_screen method: {method!r}")

    scores: list[tuple[str, float]] = []
    target = pd.to_numeric(target, errors="coerce")
    for column in candidates:
        feature = pd.to_numeric(df[column], errors="coerce")
        valid = feature.notna() & target.notna()
        if int(valid.sum()) < 50:
            continue
        corr = feature[valid].corr(target[valid], method="spearman")
        if pd.notna(corr):
            score = abs(float(corr))
            if score >= float(min_abs_corr):
                scores.append((column, score))
    if not scores:
        return candidates[: int(max_features)]
    scores.sort(key=lambda item: (-item[1], item[0]))
    return [column for column, _ in scores[: int(max_features)]]


def triple_barrier_direction_target(
    ohlcv: OHLCV,
    index: pd.Index,
    *,
    horizon: int,
    take_profit: float,
    stop_loss: float,
) -> pd.Series:
    values: Any = np.full(len(index), np.nan, dtype=float)
    for i, entry_idx, end_idx, entry in _barrier_windows(ohlcv, horizon):
        upper = entry * (1.0 + take_profit)
        lower = entry * (1.0 - stop_loss)
        label = np.nan
        for j in range(entry_idx, end_idx + 1):
            hit_upper = ohlcv.high[j] >= upper
            hit_lower = ohlcv.low[j] <= lower
            if hit_upper and hit_lower:
                break
            if hit_upper:
                label = 1.0
                break
            if hit_lower:
                label = 0.0
                break
        if np.isnan(label):
            label = 1.0 if ohlcv.close[end_idx] > entry else 0.0
        values[i] = label
    return pd.Series(values, index=index)


def triple_barrier_return_target(
    ohlcv: OHLCV,
    index: pd.Index,
    *,
    horizon: int,
    take_profit: float,
    stop_loss: float,
) -> pd.Series:
    values: Any = np.full(len(index), np.nan, dtype=float)
    for i, entry_idx, end_idx, entry in _barrier_windows(ohlcv, horizon):
        upper = entry * (1.0 + take_profit)
        lower = entry * (1.0 - stop_loss)
        ret = np.nan
        for j in range(entry_idx, end_idx + 1):
            hit_upper = ohlcv.high[j] >= upper
            hit_lower = ohlcv.low[j] <= lower
            if hit_upper and hit_lower:
                break
            if hit_upper:
                ret = take_profit
                break
            if hit_lower:
                ret = -stop_loss
                break
        if np.isnan(ret):
            ret = float(ohlcv.close[end_idx] / entry - 1.0)
            ret = float(np.clip(ret, -stop_loss, take_profit))
        values[i] = ret
    return pd.Series(values, index=index)


def _barrier_windows(ohlcv: OHLCV, horizon: int):
    horizon = int(horizon)
    if horizon <= 0:
        return
    n = len(ohlcv.close)
    for i in range(0, n - horizon - 1):
        entry_idx = i + 1
        entry = float(ohlcv.open[entry_idx])
        if not np.isfinite(entry) or entry <= 0:
            continue
        yield i, entry_idx, entry_idx + horizon, entry
