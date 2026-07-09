import argparse
import datetime
import hashlib
import itertools
import json
import logging
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.build_dataset import TARGET_COLUMNS
from src.config import (
    DAY_TRADE_HIGHER_TFS,
    INDICATOR_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    WALK_FORWARD_DEFAULTS,
)
from src.discover_patterns import (
    DEFAULT_ENABLED_KINDS,
    Condition,
    build_all_conditions,
    condition_mask,
    detect_cross_feature_pairs,
    split_train_test,
)
from src.load_data import configure_logging
from src.metrics import (
    bootstrap_sharpe_ci,
    cluster_strategies_by_overlap,
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    sharpe_ratio,
)
from src.trade_utils import scan_tp_sl, scan_tp_sl_numba
from src.walk_forward import (
    WalkForwardConfig,
    WindowConditionCache,
    aggregate_walk_forward_results,
    candidate_feature_columns,
    generate_windows,
    with_threshold,
)
from src.strategy_search import _config_hash, _flush_rows, _load_checkpoint


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day_trade_search"


@dataclass(frozen=True)
class DayTradeConfig:
    take_profit: float
    stop_loss: float
    fee_bps: float
    slippage_bps: float
    horizon_bars: int
    risk_per_trade: float = 0.003
    max_position_fraction: float = 0.25
    daily_stop_loss: float = -0.02
    max_consecutive_losses: int = 3
    cooldown_bars: int = 24
    use_atr_tp_sl: bool = False


@dataclass(frozen=True)
class StrategyCandidate:
    direction: str
    horizon_bars: int
    conditions: Tuple[Condition, ...]

    @property
    def rule(self) -> str:
        return " AND ".join(c.description for c in self.conditions)

    @property
    def timeframes(self) -> Tuple[str, ...]:
        values = set()
        for c in self.conditions:
            values.add(timeframe_for_feature(c.feature))
        return tuple(sorted(values))


def timeframe_for_feature(feature: str) -> str:
    if feature.startswith("tf_"):
        parts = feature.split("_", 3)
        if len(parts) >= 3:
            return parts[1]
    return "unknown"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass
class FeatureScreenCache:
    values: Dict[Tuple[object, ...], Set[str]]
    hits: int = 0
    misses: int = 0

    @classmethod
    def create(cls) -> "FeatureScreenCache":
        return cls(values={})

    @property
    def enabled(self) -> bool:
        return True


def feature_columns_hash(feature_columns: Sequence[str]) -> str:
    payload = "\n".join(feature_columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_key_for_screening(
    train: pd.DataFrame,
    label_column: str,
    direction: str,
    horizon_bars: int,
    take_profit: float,
    stop_loss: float,
    feature_columns: Sequence[str],
    max_features: int,
    method: str,
) -> Tuple[object, ...]:
    if train.empty:
        start = None
        stop = None
    else:
        start = str(train["timestamp"].iloc[0]) if "timestamp" in train.columns else int(train.index[0])
        stop = str(train["timestamp"].iloc[-1]) if "timestamp" in train.columns else int(train.index[-1])
    return (
        start,
        stop,
        label_column,
        direction,
        int(horizon_bars),
        round(float(take_profit), 10),
        round(float(stop_loss), 10),
        int(max_features),
        method,
        feature_columns_hash(feature_columns),
    )


def get_screened_features_cached(
    train: pd.DataFrame,
    label_column: str,
    direction: str,
    horizon_bars: int,
    take_profit: float,
    stop_loss: float,
    feature_columns: Sequence[str],
    max_features: int,
    cache: FeatureScreenCache,
    method: str = "shap",
) -> Set[str]:
    key = cache_key_for_screening(
        train, label_column, direction, horizon_bars,
        take_profit, stop_loss, feature_columns,
        max_features, method,
    )
    if key in cache.values:
        cache.hits += 1
        return cache.values[key]
    from src.feature_screener import screen_features
    cache.misses += 1
    screened = set(screen_features(
        train,
        label_column,
        feature_columns,
        max_features=max_features,
        method=method,
    ))
    cache.values[key] = screened
    return screened


def _ensure_dataset(path: Path, base_timeframe: str) -> None:
    if path.exists():
        return
    LOGGER.info("Dataset %s not found — building it now", path)
    from src.build_dataset import run as build_run
    build_run(
        indicator_dir=INDICATOR_DATA_DIR,
        input_kind="indicators",
        output_path=path,
        report_path=path.with_suffix(".json"),
        include_duplicate_scan=False,
        base_timeframe=base_timeframe,
    )


def load_dataset(
    path: Path,
    horizons: Sequence[int],
    base_prefix: str = "tf_5m_",
    base_timeframe: str = "5m",
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    _ensure_dataset(path, base_timeframe)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    base_columns = {
        "timestamp",
        f"{base_prefix}open",
        f"{base_prefix}high",
        f"{base_prefix}low",
        f"{base_prefix}close",
    }
    read_columns = None
    if columns is not None:
        read_columns = sorted(set(columns) | base_columns)
    data = pd.read_parquet(path, columns=read_columns).sort_values("timestamp").reset_index(drop=True)
    required = base_columns
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    close_col = f"{base_prefix}close"
    for horizon in horizons:
        data[f"future_return_{horizon}_bars"] = (
            data[close_col].shift(-horizon) / data[close_col] - 1
        )
    return data.dropna(
        subset=[f"future_return_{h}_bars" for h in horizons]
    ).reset_index(drop=True)


def numeric_feature_columns(
    data: pd.DataFrame,
    base_prefix: str = "tf_5m_",
) -> List[str]:
    excluded = {
        "timestamp",
        f"{base_prefix}open",
        f"{base_prefix}high",
        f"{base_prefix}low",
        f"{base_prefix}close",
    } | set(TARGET_COLUMNS)
    excluded.update(c for c in data.columns if c.startswith("future_return_"))
    return [
        c for c in data.select_dtypes(include="number").columns
        if c not in excluded and c.startswith("tf_")
    ]


@dataclass
class PrecomputedArrays:
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray
    day_ids: np.ndarray
    atr: np.ndarray

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, base_prefix: str = "tf_5m_") -> "PrecomputedArrays":
        open_ = data[f"{base_prefix}open"].astype(float).to_numpy()
        high = data[f"{base_prefix}high"].astype(float).to_numpy()
        low = data[f"{base_prefix}low"].astype(float).to_numpy()
        close = data[f"{base_prefix}close"].astype(float).to_numpy()
        timestamps = data["timestamp"].to_numpy()
        ts_dates = pd.DatetimeIndex(timestamps).date
        unique_dates = sorted(set(ts_dates))
        date_to_id = {d: i for i, d in enumerate(unique_dates)}
        day_ids = np.array([date_to_id[d] for d in ts_dates], dtype=np.int64)
        atr_col = f"{base_prefix}atr" if f"{base_prefix}atr" in data.columns else f"{base_prefix}atr_14"
        if atr_col not in data.columns:
            atr_cols = [c for c in data.columns if c.endswith("_atr") or c.endswith("_atr_14")]
            atr_col = atr_cols[0] if atr_cols else None
        atr = data[atr_col].astype(float).to_numpy() if atr_col else np.zeros(len(data))
        return cls(open_=open_, high=high, low=low, close=close,
                   timestamps=timestamps, day_ids=day_ids, atr=atr)


try:
    from numba import njit

    @njit(cache=True)
    def _simulate_inner(
        open_, high, low, close, day_ids, signal,
        is_long, horizon_bars, take_profit, stop_loss,
        total_cost, position_size, daily_stop_loss,
        max_consecutive_losses, cooldown_bars,
        atr, use_atr_tp_sl, risk_per_trade, max_position_fraction,
    ):
        n = len(signal)
        max_entry_index = n - horizon_bars - 1
        max_trades = n
        signal_indices = np.empty(max_trades, dtype=np.int64)
        entry_indices = np.empty(max_trades, dtype=np.int64)
        exit_indices = np.empty(max_trades, dtype=np.int64)
        exit_reasons = np.empty(max_trades, dtype=np.int64)  # 0=time, 1=stop, 2=tp
        entries = np.empty(max_trades, dtype=np.float64)
        exits = np.empty(max_trades, dtype=np.float64)
        gross_returns = np.empty(max_trades, dtype=np.float64)
        net_returns = np.empty(max_trades, dtype=np.float64)
        sized_returns = np.empty(max_trades, dtype=np.float64)
        equities = np.empty(max_trades, dtype=np.float64)
        position_sizes = np.empty(max_trades, dtype=np.float64)

        trade_count = 0
        next_allowed = 0
        equity = 1.0
        daily_pnl = 0.0
        current_day = -1
        consecutive_losses = 0
        cooldown_until = 0
        daily_stop_hits = 0
        cooldown_triggers = 0

        for si in range(n):
            ei = si + 1
            if not signal[si] or ei < next_allowed or ei > max_entry_index:
                continue
            d = day_ids[si]
            if d != current_day:
                daily_pnl = 0.0
                current_day = d
            if daily_pnl <= daily_stop_loss:
                daily_stop_hits += 1
                continue
            if si < cooldown_until:
                continue

            entry = open_[ei]
            atr_val = atr[si]

            if use_atr_tp_sl and not np.isnan(atr_val) and atr_val > 0:
                sl_pct = (stop_loss * atr_val) / entry
                tp_pct = (take_profit * atr_val) / entry
            else:
                sl_pct = stop_loss
                tp_pct = take_profit

            if sl_pct <= 0:
                sl_pct = 0.01  # fallback 1%

            xi, xr = scan_tp_sl_numba(
                high, low, entry, is_long, tp_pct, sl_pct, ei, ei + horizon_bars
            )
            if xr == 1:
                xp = entry * (1.0 - sl_pct) if is_long else entry * (1.0 + sl_pct)
            elif xr == 2:
                xp = entry * (1.0 + tp_pct) if is_long else entry * (1.0 - tp_pct)
            else:
                xp = close[xi]

            if is_long:
                gr = xp / entry - 1.0
            else:
                gr = entry / xp - 1.0
            nr = gr - total_cost

            if use_atr_tp_sl:
                trade_position_size = min(risk_per_trade / sl_pct, max_position_fraction, 1.0)
            else:
                trade_position_size = position_size

            sr = nr * trade_position_size

            daily_pnl += sr
            equity *= 1.0 + sr

            if nr < 0:
                consecutive_losses += 1
                if consecutive_losses >= max_consecutive_losses:
                    cooldown_until = xi + cooldown_bars
                    cooldown_triggers += 1
                    consecutive_losses = 0
            else:
                consecutive_losses = 0

            signal_indices[trade_count] = si
            entry_indices[trade_count] = ei
            exit_indices[trade_count] = xi
            exit_reasons[trade_count] = xr
            entries[trade_count] = entry
            exits[trade_count] = xp
            gross_returns[trade_count] = gr
            net_returns[trade_count] = nr
            sized_returns[trade_count] = sr
            equities[trade_count] = equity
            position_sizes[trade_count] = trade_position_size
            trade_count += 1
            next_allowed = xi + 1

        return (
            signal_indices[:trade_count],
            entry_indices[:trade_count],
            exit_indices[:trade_count],
            exit_reasons[:trade_count],
            entries[:trade_count],
            exits[:trade_count],
            gross_returns[:trade_count],
            net_returns[:trade_count],
            sized_returns[:trade_count],
            equities[:trade_count],
            position_sizes[:trade_count],
            daily_stop_hits,
            cooldown_triggers,
        )

    _HAS_NUMBA = True
    LOGGER.info("Numba available — using JIT-compiled simulation")
except ImportError:
    _HAS_NUMBA = False


def combined_mask(data: pd.DataFrame, conditions: Sequence[Condition]) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for condition in conditions:
        mask &= condition_mask(data, condition).fillna(False)
    return mask


def _simulate_day_trades_python(
    data: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    config: DayTradeConfig,
    base_prefix: str = "tf_5m_",
    precomputed: Optional[PrecomputedArrays] = None,
) -> pd.DataFrame:
    if precomputed is not None:
        open_ = precomputed.open_
        high = precomputed.high
        low = precomputed.low
        close = precomputed.close
        timestamps = precomputed.timestamps
        atr = precomputed.atr
    else:
        open_ = data[f"{base_prefix}open"].astype(float).to_numpy()
        high = data[f"{base_prefix}high"].astype(float).to_numpy()
        low = data[f"{base_prefix}low"].astype(float).to_numpy()
        close = data[f"{base_prefix}close"].astype(float).to_numpy()
        timestamps = data["timestamp"].to_numpy()
        atr_col = f"{base_prefix}atr" if f"{base_prefix}atr" in data.columns else f"{base_prefix}atr_14"
        if atr_col not in data.columns:
            atr_cols = [c for c in data.columns if c.endswith("_atr") or c.endswith("_atr_14")]
            atr_col = atr_cols[0] if atr_cols else None
        atr = data[atr_col].astype(float).to_numpy() if atr_col else np.zeros(len(data))
    signal = signal_mask.fillna(False).to_numpy()

    total_cost = 2 * ((config.fee_bps + config.slippage_bps) / 10_000)
    position_size = (
        min(config.risk_per_trade / config.stop_loss, config.max_position_fraction, 1.0)
        if config.stop_loss > 0
        else config.max_position_fraction
    )

    trades = []
    next_allowed_entry = 0
    max_entry_index = len(data) - config.horizon_bars - 1

    equity = 1.0
    daily_pnl = 0.0
    current_day = None
    consecutive_losses = 0
    cooldown_until = 0
    daily_stop_hits = 0
    cooldown_triggers = 0

    for signal_index, should_enter in enumerate(signal):
        entry_index = signal_index + 1
        if (
            not should_enter
            or entry_index < next_allowed_entry
            or entry_index > max_entry_index
        ):
            continue

        day = pd.Timestamp(timestamps[signal_index]).date()
        if day != current_day:
            daily_pnl = 0.0
            current_day = day

        if daily_pnl <= config.daily_stop_loss:
            daily_stop_hits += 1
            continue

        if signal_index < cooldown_until:
            continue

        entry = open_[entry_index]
        atr_val = atr[signal_index]

        if config.use_atr_tp_sl and not np.isnan(atr_val) and atr_val > 0:
            sl_pct = (config.stop_loss * atr_val) / entry
            tp_pct = (config.take_profit * atr_val) / entry
        else:
            sl_pct = config.stop_loss
            tp_pct = config.take_profit

        if sl_pct <= 0:
            sl_pct = 0.01

        if config.use_atr_tp_sl:
            trade_position_size = min(config.risk_per_trade / sl_pct, config.max_position_fraction, 1.0)
        else:
            trade_position_size = position_size

        exit_index, reason_code = scan_tp_sl(
            high, low, entry, direction == "long",
            tp_pct, sl_pct, entry_index, entry_index + config.horizon_bars,
        )
        if reason_code == 1:
            exit_price = entry * (1 - sl_pct) if direction == "long" else entry * (1 + sl_pct)
            exit_reason = "stop"
        elif reason_code == 2:
            exit_price = entry * (1 + tp_pct) if direction == "long" else entry * (1 - tp_pct)
            exit_reason = "take_profit"
        else:
            exit_price = close[exit_index]
            exit_reason = "time"

        gross_return = (
            exit_price / entry - 1
            if direction == "long"
            else entry / exit_price - 1
        )
        net_return = gross_return - total_cost
        sized_return = net_return * trade_position_size

        daily_pnl += sized_return
        equity *= 1 + sized_return

        if net_return < 0:
            consecutive_losses += 1
            if consecutive_losses >= config.max_consecutive_losses:
                cooldown_until = exit_index + config.cooldown_bars
                cooldown_triggers += 1
                consecutive_losses = 0
        else:
            consecutive_losses = 0

        trades.append(
            {
                "signal_time": timestamps[signal_index],
                "entry_time": timestamps[entry_index],
                "exit_time": timestamps[exit_index],
                "direction": direction,
                "entry": entry,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "gross_return": gross_return,
                "net_return": net_return,
                "sized_return": sized_return,
                "position_size": trade_position_size,
                "holding_bars": exit_index - entry_index,
                "equity_after": equity,
            }
        )
        next_allowed_entry = exit_index + 1

    result = pd.DataFrame(trades)
    result.attrs["daily_stop_hits"] = daily_stop_hits
    result.attrs["cooldown_triggers"] = cooldown_triggers
    return result


def _simulate_day_trades_numba(
    data: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    config: DayTradeConfig,
    base_prefix: str = "tf_5m_",
    precomputed: Optional[PrecomputedArrays] = None,
) -> pd.DataFrame:
    if precomputed is not None:
        open_ = precomputed.open_
        high = precomputed.high
        low = precomputed.low
        close = precomputed.close
        timestamps = precomputed.timestamps
        day_ids = precomputed.day_ids
        atr = precomputed.atr
    else:
        open_ = data[f"{base_prefix}open"].astype(float).to_numpy()
        high = data[f"{base_prefix}high"].astype(float).to_numpy()
        low = data[f"{base_prefix}low"].astype(float).to_numpy()
        close = data[f"{base_prefix}close"].astype(float).to_numpy()
        timestamps = data["timestamp"].to_numpy()
        ts_dates = pd.DatetimeIndex(timestamps).date
        unique_dates = sorted(set(ts_dates))
        date_to_id = {d: i for i, d in enumerate(unique_dates)}
        day_ids = np.array([date_to_id[d] for d in ts_dates], dtype=np.int64)
        atr_col = f"{base_prefix}atr" if f"{base_prefix}atr" in data.columns else f"{base_prefix}atr_14"
        if atr_col not in data.columns:
            atr_cols = [c for c in data.columns if c.endswith("_atr") or c.endswith("_atr_14")]
            atr_col = atr_cols[0] if atr_cols else None
        atr = data[atr_col].astype(float).to_numpy() if atr_col else np.zeros(len(data))

    signal = signal_mask.fillna(False).to_numpy()

    total_cost = 2 * ((config.fee_bps + config.slippage_bps) / 10_000)
    position_size = (
        min(config.risk_per_trade / config.stop_loss, config.max_position_fraction, 1.0)
        if config.stop_loss > 0
        else config.max_position_fraction
    )

    (
        signal_indices, entry_indices, exit_indices, exit_reasons,
        entries, exits, gross_returns, net_returns, sized_returns,
        equities, position_sizes, daily_stop_hits, cooldown_triggers,
    ) = _simulate_inner(
        open_, high, low, close, day_ids, signal,
        direction == "long", config.horizon_bars, config.take_profit,
        config.stop_loss, total_cost, position_size, config.daily_stop_loss,
        config.max_consecutive_losses, config.cooldown_bars,
        atr, config.use_atr_tp_sl, config.risk_per_trade, config.max_position_fraction,
    )

    if len(signal_indices) == 0:
        result = pd.DataFrame(columns=[
            "signal_time", "entry_time", "exit_time", "direction",
            "entry", "exit", "exit_reason", "gross_return", "net_return",
            "sized_return", "position_size", "holding_bars", "equity_after",
        ])
        result.attrs["daily_stop_hits"] = int(daily_stop_hits)
        result.attrs["cooldown_triggers"] = int(cooldown_triggers)
        return result

    reason_map = {0: "time", 1: "stop", 2: "take_profit"}
    result = pd.DataFrame({
        "signal_time": timestamps[signal_indices],
        "entry_time": timestamps[entry_indices],
        "exit_time": timestamps[exit_indices],
        "direction": direction,
        "entry": entries,
        "exit": exits,
        "exit_reason": [reason_map[r] for r in exit_reasons],
        "gross_return": gross_returns,
        "net_return": net_returns,
        "sized_return": sized_returns,
        "position_size": position_sizes,
        "holding_bars": exit_indices - entry_indices,
        "equity_after": equities,
    })
    result.attrs["daily_stop_hits"] = int(daily_stop_hits)
    result.attrs["cooldown_triggers"] = int(cooldown_triggers)
    return result


def simulate_day_trades(
    data: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    config: DayTradeConfig,
    base_prefix: str = "tf_5m_",
    precomputed: Optional[PrecomputedArrays] = None,
) -> pd.DataFrame:
    if _HAS_NUMBA:
        return _simulate_day_trades_numba(data, signal_mask, direction, config, base_prefix, precomputed)
    return _simulate_day_trades_python(data, signal_mask, direction, config, base_prefix, precomputed)


def day_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_return": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": np.nan,
            "avg_trades_per_day": 0.0,
            "sharpe_ratio": np.nan,
            "avg_holding_bars": np.nan,
            "daily_stop_hits": 0,
            "cooldown_triggers": 0,
            "return_skew": 0.0,
            "return_kurt": 3.0,
        }
    returns = trades["net_return"].astype(float)
    sized_returns = trades["sized_return"].astype(float)
    equity = (1 + sized_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    entry_times = pd.to_datetime(trades["entry_time"])
    trading_days = entry_times.dt.date.nunique()
    avg_per_day = len(trades) / max(trading_days, 1)

    daily_returns = sized_returns.groupby(entry_times.dt.date).sum()
    sharpe = np.nan
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(
            daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        )

    return {
        "trades": int(len(trades)),
        "win_rate": float((returns > 0).mean()),
        "avg_net_return": float(returns.mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "avg_trades_per_day": round(avg_per_day, 2),
        "sharpe_ratio": sharpe,
        "avg_holding_bars": float(trades["holding_bars"].mean()),
        "daily_stop_hits": int(trades.attrs.get("daily_stop_hits", 0)),
        "cooldown_triggers": int(trades.attrs.get("cooldown_triggers", 0)),
        "return_skew": float(returns.skew()) if len(returns) > 2 else 0.0,
        "return_kurt": float(returns.kurt() + 3.0) if len(returns) > 3 else 3.0,
    }


def score_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: StrategyCandidate,
    config: DayTradeConfig,
    base_prefix: str = "tf_5m_",
    train_mask: Optional[pd.Series] = None,
    test_mask: Optional[pd.Series] = None,
    train_arrays: Optional[PrecomputedArrays] = None,
    test_arrays: Optional[PrecomputedArrays] = None,
) -> Dict[str, object]:
    if train_mask is None:
        train_mask = combined_mask(train, candidate.conditions)
    if test_mask is None:
        test_mask = combined_mask(test, candidate.conditions)
    train_trades = simulate_day_trades(
        train, train_mask, candidate.direction, config, base_prefix, train_arrays,
    )
    test_trades = simulate_day_trades(
        test, test_mask, candidate.direction, config, base_prefix, test_arrays,
    )
    train_m = day_trade_metrics(train_trades)
    test_m = day_trade_metrics(test_trades)
    return {
        "direction": candidate.direction,
        "horizon_bars": config.horizon_bars,
        "take_profit": config.take_profit,
        "stop_loss": config.stop_loss,
        "timeframes": ",".join(candidate.timeframes),
        "timeframe_count": len(candidate.timeframes),
        "conditions": len(candidate.conditions),
        "rule": candidate.rule,
        "train_returns": train_trades["net_return"].astype(float).to_numpy() if not train_trades.empty else np.array([], dtype=float),
        "test_returns": test_trades["net_return"].astype(float).to_numpy() if not test_trades.empty else np.array([], dtype=float),
        **{f"train_{k}": v for k, v in train_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    }


def _attach_statistical_metrics(
    strategies: pd.DataFrame,
    return_rows: Optional[Sequence[Dict[str, object]]] = None,
    walk_forward: bool = False,
) -> pd.DataFrame:
    if strategies.empty:
        return strategies
    if walk_forward:
        out = strategies.copy()
        sr = out["wf_returns_sharpe"].astype(float)
        skew = out["wf_returns_skew"].astype(float)
        kurt = out["wf_returns_kurt"].astype(float)
        n_obs = out["wf_windows"].astype(float)
        finite = sr[np.isfinite(sr)]
        sr_std_trials = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        n_trials = max(1, len(out))
        out["dsr"] = [
            deflated_sharpe_ratio(
                float(s) if np.isfinite(s) else 0.0,
                n_trials=n_trials,
                skew=float(sk) if np.isfinite(sk) else 0.0,
                kurt=float(ku) if np.isfinite(ku) else 3.0,
                n_obs=max(1, int(n) if np.isfinite(n) else 1),
                sr_std_trials=sr_std_trials,
            )
            for s, sk, ku, n in zip(sr, skew, kurt, n_obs)
        ]
        return out
    out = strategies.copy()
    n_trials = max(1, len(out))
    train_srs = [
        sharpe_ratio(np.asarray(row.get("train_returns", []), dtype=float))
        for row in return_rows
    ]
    finite_srs = np.asarray([sr for sr in train_srs if np.isfinite(sr)], dtype=float)
    sr_std_trials = float(np.std(finite_srs, ddof=1)) if finite_srs.size > 1 else 0.0
    dsr_values = []
    ci_low = []
    ci_high = []
    for row, train_sr in zip(return_rows, train_srs):
        train_returns = np.asarray(row.get("train_returns", []), dtype=float)
        test_returns = np.asarray(row.get("test_returns", []), dtype=float)
        dsr_values.append(
            deflated_sharpe_ratio(
                train_sr,
                n_trials=n_trials,
                skew=float(row.get("train_return_skew", 0.0)),
                kurt=float(row.get("train_return_kurt", 3.0)),
                n_obs=max(1, train_returns.size),
                sr_std_trials=sr_std_trials,
            )
        )
        low, high = bootstrap_sharpe_ci(test_returns, n_boot=200)
        ci_low.append(low)
        ci_high.append(high)
    out["dsr"] = dsr_values
    out["test_sharpe_ci_low"] = ci_low
    out["test_sharpe_ci_high"] = ci_high
    return out


def regime_breakdown(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    config: DayTradeConfig,
    base_prefix: str = "tf_5m_",
) -> Dict[str, Dict[str, float]]:
    if "tf_1d_regime_id" not in data.columns:
        return {}
    breakdown: Dict[str, Dict[str, float]] = {}
    for regime_id, regime_data in data.groupby("tf_1d_regime_id"):
        trades = simulate_day_trades(
            regime_data,
            combined_mask(regime_data, candidate.conditions),
            candidate.direction,
            config,
            base_prefix,
        )
        metrics = day_trade_metrics(trades)
        returns = trades["net_return"].astype(float).to_numpy() if not trades.empty else np.array([], dtype=float)
        breakdown[str(int(regime_id))] = {
            "trades": int(metrics["trades"]),
            "total_return": float(metrics["total_return"]),
            "win_rate": None if pd.isna(metrics["win_rate"]) else float(metrics["win_rate"]),
            "dsr": deflated_sharpe_ratio(
                sharpe_ratio(returns),
                n_trials=1,
                skew=float(metrics.get("return_skew", 0.0)),
                kurt=float(metrics.get("return_kurt", 3.0)),
                n_obs=max(1, returns.size),
            ),
        }
    return breakdown


def cluster_ranked_strategies(data: pd.DataFrame, strategies: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    out = strategies.copy()
    if out.empty or "conditions_json" not in out.columns:
        out["cluster_id"] = pd.Series(dtype=int)
        return out
    masks = {}
    for idx, row in out.iterrows():
        conditions = tuple(Condition(**payload) for payload in json.loads(row["conditions_json"]))
        masks[str(idx)] = combined_mask(data, conditions).to_numpy()
    clusters = cluster_strategies_by_overlap(masks, jaccard_threshold=threshold)
    representatives = []
    for cluster_id, members in clusters.items():
        member_indices = [int(member) for member in members]
        cluster_rows = out.loc[member_indices].copy()
        cluster_rows["cluster_id"] = int(cluster_id)
        sort_cols = [col for col in ["dsr", "test_total_return", "test_avg_net_return", "test_trades"] if col in cluster_rows.columns]
        cluster_rows = cluster_rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        representatives.append(cluster_rows.iloc[0])
    sort_cols = [col for col in ["dsr", "test_total_return", "test_avg_net_return", "test_trades"] if col in representatives[0].index]
    return pd.DataFrame(representatives).sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


def rank_features_by_direction(
    train: pd.DataFrame,
    features: Iterable[str],
    horizon: int,
    direction: str,
    max_features: int,
    sample_rows: int = 50_000,
) -> List[str]:
    feature_list = list(features)
    target = train[f"future_return_{horizon}_bars"]
    if direction == "short":
        target = -target
    if sample_rows and len(train) > sample_rows:
        index = np.linspace(0, len(train) - 1, sample_rows).astype(int)
        sample = train.iloc[index]
        target = target.iloc[index]
    else:
        sample = train
    feature_frame = sample.loc[:, feature_list].replace([np.inf, -np.inf], np.nan)
    correlations = feature_frame.corrwith(target, method="spearman").abs()
    correlations = correlations.dropna().sort_values(ascending=False)
    return correlations.head(max_features).index.tolist()


def _rank_features(
    train: pd.DataFrame,
    all_features: List[str],
    horizon: int,
    direction: str,
    max_features: int,
    ranking_method: str,
    rank_sample_rows: int,
) -> List[str]:
    if ranking_method == "spearman":
        return rank_features_by_direction(
            train, all_features, horizon, direction,
            max_features=max_features, sample_rows=rank_sample_rows,
        )
    from src.feature_ranking import rank_features_blended, rank_features_by_importance

    target_column = f"future_return_{horizon}_bars"
    if ranking_method == "importance":
        return rank_features_by_importance(
            train, all_features, target_column, max_features, direction=direction,
        )
    return rank_features_blended(
        train, all_features, target_column, max_features, direction=direction,
    )


def build_feature_conditions(train: pd.DataFrame, feature: str) -> List[Condition]:
    conditions = []
    clean = train[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return conditions
    quantiles = clean.quantile([0.1, 0.2, 0.8, 0.9]).to_dict()
    for q, threshold in quantiles.items():
        if pd.isna(threshold):
            continue
        if q < 0.5:
            conditions.append(
                Condition(feature, "value_le", threshold,
                          f"{feature} <= train q{int(q * 100)} ({threshold:.6g})",
                          threshold_source="quantile", quantile=float(q))
            )
        else:
            conditions.append(
                Condition(feature, "value_ge", threshold,
                          f"{feature} >= train q{int(q * 100)} ({threshold:.6g})",
                          threshold_source="quantile", quantile=float(q))
            )
    delta = train[feature].diff()
    delta_clean = delta.replace([np.inf, -np.inf], np.nan).dropna()
    if not delta_clean.empty:
        dq = delta_clean.quantile([0.1, 0.9]).to_dict()
        for q, threshold in dq.items():
            if pd.isna(threshold):
                continue
            if q < 0.5:
                conditions.append(
                    Condition(feature, "delta_le", threshold,
                              f"{feature} falling fast: 1-bar change <= train q10 ({threshold:.6g})",
                              threshold_source="delta_quantile", quantile=float(q))
                )
            else:
                conditions.append(
                    Condition(feature, "delta_ge", threshold,
                              f"{feature} rising fast: 1-bar change >= train q90 ({threshold:.6g})",
                              threshold_source="delta_quantile", quantile=float(q))
                )
    return conditions


def _build_conditions_for_features(
    train: pd.DataFrame,
    ranked_features: List[str],
    enabled_kinds: Set[str],
    cross_feature_pairs: Optional[Sequence[Tuple[str, str]]],
) -> List[Condition]:
    if enabled_kinds == {"value", "delta"}:
        return [
            c for feature in ranked_features
            for c in build_feature_conditions(train, feature)
        ]
    return build_all_conditions(
        train, ranked_features, enabled_kinds=enabled_kinds,
        cross_feature_pairs=cross_feature_pairs,
    )


def _score_and_select_conditions(
    train: pd.DataFrame,
    conditions: List[Condition],
    horizon: int,
    direction: str,
    top_conditions: int,
    min_support: int = 500,
) -> List[int]:
    target = train[f"future_return_{horizon}_bars"]
    if direction == "short":
        target = -target
    scores = []
    for idx, cond in enumerate(conditions):
        mask = condition_mask(train, cond).fillna(False)
        selected = target.loc[mask]
        if len(selected) < min_support:
            continue
        scores.append((idx, float(selected.mean()), int(len(selected))))
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [idx for idx, _, _ in scores[:top_conditions]]


def _generate_pairs_flat(
    conditions: List[Condition],
    selected_indices: List[int],
    max_pairs: int,
) -> List[Tuple[int, int]]:
    pairs = []
    for left_pos, left_idx in enumerate(selected_indices):
        for right_idx in selected_indices[left_pos + 1:]:
            if conditions[left_idx].feature == conditions[right_idx].feature:
                continue
            pairs.append((left_idx, right_idx))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _generate_pairs_pool(
    conditions: List[Condition],
    selected_indices: List[int],
    max_pairs: int,
) -> List[Tuple[int, int]]:
    pools: Dict[str, List[int]] = defaultdict(list)
    for idx in selected_indices:
        tf = timeframe_for_feature(conditions[idx].feature)
        pools[tf].append(idx)
    tf_keys = sorted(pools.keys())
    tf_combos = [
        (tf_keys[i], tf_keys[j])
        for i in range(len(tf_keys))
        for j in range(i + 1, len(tf_keys))
    ]
    if not tf_combos:
        return _generate_pairs_flat(conditions, selected_indices, max_pairs)
    pairs_per_combo = max(1, max_pairs // len(tf_combos))
    pairs: List[Tuple[int, int]] = []
    for tf_a, tf_b in tf_combos:
        count = 0
        for left_idx in pools[tf_a]:
            for right_idx in pools[tf_b]:
                if conditions[left_idx].feature == conditions[right_idx].feature:
                    continue
                pairs.append((left_idx, right_idx))
                count += 1
                if count >= pairs_per_combo:
                    break
            if count >= pairs_per_combo:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs[:max_pairs]


def make_candidates(
    train: pd.DataFrame,
    horizons: Sequence[int],
    directions: Sequence[str],
    max_features: int,
    top_conditions: int,
    max_pairs: int,
    max_triples: int,
    rank_sample_rows: int,
    condition_depths: Sequence[int],
    ranking_method: str = "blended",
    cross_tf_mode: str = "pool",
    enabled_kinds: Set[str] = DEFAULT_ENABLED_KINDS,
    base_prefix: str = "tf_5m_",
    feature_pattern: Optional[str] = None,
) -> List[StrategyCandidate]:
    all_features = numeric_feature_columns(train, base_prefix)
    if feature_pattern:
        compiled = re.compile(feature_pattern)
        all_features = [column for column in all_features if compiled.search(column)]
        if not all_features:
            raise ValueError(f"feature_pattern {feature_pattern!r} matches no feature columns")
        LOGGER.info("Feature pattern %r restricts the universe to %s columns", feature_pattern, len(all_features))
    candidates: List[StrategyCandidate] = []
    for horizon in horizons:
        for direction in directions:
            LOGGER.info(
                "Ranking features for direction=%s horizon=%s method=%s",
                direction, horizon, ranking_method,
            )
            ranked = _rank_features(
                train, all_features, horizon, direction,
                max_features, ranking_method, rank_sample_rows,
            )
            cross_pairs = (
                detect_cross_feature_pairs(ranked)
                if "cross" in enabled_kinds or "ratio" in enabled_kinds
                else None
            )
            conditions = _build_conditions_for_features(
                train, ranked, enabled_kinds, cross_pairs,
            )
            selected = _score_and_select_conditions(
                train, conditions, horizon, direction, top_conditions,
            )
            LOGGER.info(
                "Selected %s base conditions for direction=%s horizon=%s",
                len(selected), direction, horizon,
            )

            if 1 in condition_depths:
                for idx in selected:
                    candidates.append(
                        StrategyCandidate(direction, horizon, (conditions[idx],))
                    )

            if 2 in condition_depths or 3 in condition_depths:
                if cross_tf_mode == "pool":
                    pairs = _generate_pairs_pool(conditions, selected, max_pairs)
                else:
                    pairs = _generate_pairs_flat(conditions, selected, max_pairs)

                if 2 in condition_depths:
                    for l_idx, r_idx in pairs:
                        candidates.append(
                            StrategyCandidate(
                                direction, horizon,
                                (conditions[l_idx], conditions[r_idx]),
                            )
                        )

                if 3 in condition_depths:
                    triple_count = 0
                    seen: set = set()
                    for l_idx, r_idx in pairs:
                        used = {conditions[l_idx].feature, conditions[r_idx].feature}
                        for t_idx in selected:
                            if t_idx in {l_idx, r_idx}:
                                continue
                            if conditions[t_idx].feature in used:
                                continue
                            triple = tuple(sorted((l_idx, r_idx, t_idx)))
                            if triple in seen:
                                continue
                            seen.add(triple)
                            candidates.append(
                                StrategyCandidate(
                                    direction, horizon,
                                    tuple(conditions[i] for i in triple),
                                )
                            )
                            triple_count += 1
                            if triple_count >= max_triples:
                                break
                        if triple_count >= max_triples:
                            break
    seen_signatures = set()
    deduped: List[StrategyCandidate] = []
    for candidate in candidates:
        signature = (
            candidate.direction,
            candidate.horizon_bars,
            frozenset(condition.signature() for condition in candidate.conditions),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
    return deduped


def add_year_metrics(
    data: pd.DataFrame,
    strategies: pd.DataFrame,
    config_template: DayTradeConfig,
    base_prefix: str = "tf_5m_",
    top_n: int = 10,
) -> pd.DataFrame:
    if strategies.empty:
        return pd.DataFrame()
    rows = []
    for rank, (_, strategy) in enumerate(strategies.head(top_n).iterrows(), start=1):
        conditions = tuple(
            Condition(**c) for c in json.loads(strategy["conditions_json"])
        )
        candidate = StrategyCandidate(
            strategy["direction"], int(strategy["horizon_bars"]), conditions,
        )
        config = DayTradeConfig(
            take_profit=strategy["take_profit"],
            stop_loss=strategy["stop_loss"],
            fee_bps=config_template.fee_bps,
            slippage_bps=config_template.slippage_bps,
            horizon_bars=int(strategy["horizon_bars"]),
            risk_per_trade=config_template.risk_per_trade,
            max_position_fraction=config_template.max_position_fraction,
            daily_stop_loss=config_template.daily_stop_loss,
            max_consecutive_losses=config_template.max_consecutive_losses,
            cooldown_bars=config_template.cooldown_bars,
            use_atr_tp_sl=config_template.use_atr_tp_sl,
        )
        mask = combined_mask(data, candidate.conditions)
        trades = simulate_day_trades(
            data, mask, candidate.direction, config, base_prefix,
        )
        if trades.empty:
            continue
        entry_times = pd.to_datetime(trades["entry_time"])
        for year, group in trades.groupby(entry_times.dt.year):
            m = day_trade_metrics(group)
            rows.append({
                "strategy_rank": rank,
                "year": year,
                "direction": candidate.direction,
                "horizon_bars": config.horizon_bars,
                "take_profit": config.take_profit,
                "stop_loss": config.stop_loss,
                **m,
                "rule": candidate.rule,
            })
    return pd.DataFrame(rows)


def summarize_filter_rejections(
    scored: pd.DataFrame,
    min_train_trades: int,
    min_test_trades: int,
    require_multitimeframe: bool,
    walk_forward: bool = False,
) -> Dict[str, int]:
    if walk_forward:
        if scored.empty:
            return {
                "scored_candidates": 0,
                "positive_wf_expectancy": 0,
                "passes_wf_gate": 0,
                "enough_wf_trades": 0,
                "multitimeframe": 0,
            }
        return {
            "scored_candidates": int(len(scored)),
            "positive_wf_expectancy": int((scored["wf_expectancy"] > 0).sum()),
            "passes_wf_gate": int(scored["wf_passes"].astype(bool).sum()),
            "enough_wf_trades": int((scored["wf_avg_trades"] >= min_test_trades).sum()),
            "multitimeframe": (
                int((scored["timeframe_count"] >= 2).sum())
                if require_multitimeframe
                else int(len(scored))
            ),
        }
    if scored.empty:
        return {
            "scored_candidates": 0,
            "enough_train_trades": 0,
            "enough_test_trades": 0,
            "positive_train_return": 0,
            "positive_test_return": 0,
            "multitimeframe": 0,
        }
    summary = {
        "scored_candidates": int(len(scored)),
        "enough_train_trades": int((scored["train_trades"] >= min_train_trades).sum()),
        "enough_test_trades": int((scored["test_trades"] >= min_test_trades).sum()),
        "positive_train_return": int((scored["train_total_return"] > 0).sum()),
        "positive_test_return": int((scored["test_total_return"] > 0).sum()),
        "multitimeframe": int((scored["timeframe_count"] >= 2).sum()),
    }
    if not require_multitimeframe:
        summary["multitimeframe"] = int(len(scored))
    return summary


class DayTradeWindowEngine(WindowConditionCache):
    """WindowConditionCache plus per-window simulation arrays for day trading."""

    def __init__(self, data: pd.DataFrame, windows, base_prefix: str):
        super().__init__(data, windows)
        self.base_prefix = base_prefix
        self.test_arrays = [
            PrecomputedArrays.from_dataframe(frame, base_prefix) for frame in self.test_frames
        ]


_WF_ZEROED_COLUMNS = {
    f"{split}_{metric}": 0.0
    for split in ("train", "test")
    for metric in (
        "trades", "total_return", "win_rate", "avg_net_return",
        "profit_factor", "max_drawdown",
    )
}

_SCREENED_OUT_STATS = {
    "test_total_return": 0.0,
    "test_avg_net_return": 0.0,
    "test_profit_factor": 0.0,
    "test_max_drawdown": 0.0,
    "test_trades": 0,
    "screened_out": True,
}


def _conditions_payload_day(candidate: StrategyCandidate) -> str:
    return json.dumps([
        {
            "feature": c.feature,
            "kind": c.kind,
            "threshold": c.threshold,
            "description": c.description,
            **({"feature_b": c.feature_b} if c.feature_b else {}),
            **({"threshold_source": c.threshold_source} if c.threshold_source else {}),
            **({"quantile": c.quantile} if c.quantile is not None else {}),
            **({"lookback": c.lookback} if c.lookback is not None else {}),
            **({"cross_feature": c.cross_feature} if c.cross_feature else {}),
        }
        for c in candidate.conditions
    ])


def _score_candidate_walk_forward_day(
    engine: DayTradeWindowEngine,
    candidate: StrategyCandidate,
    scenarios: Sequence[Tuple[float, float]],
    fee_bps: float,
    slippage_bps: float,
    risk_per_trade: float,
    max_position_fraction: float,
    daily_stop_loss: float,
    max_consecutive_losses: int,
    cooldown_bars: int,
    use_atr_tp_sl: bool,
    wf_pass_rate: float,
    feature_screening: str = "none",
    screen_cache: Optional[FeatureScreenCache] = None,
    max_features: int = 60,
) -> List[Dict[str, object]]:
    per_scenario_stats: List[List[Dict[str, float]]] = [[] for _ in scenarios]
    for window_index in range(len(engine.windows)):
        mask = engine.candidate_test_mask(window_index, candidate)
        test_frame = engine.test_frames[window_index]
        arrays = engine.test_arrays[window_index]
        signal = pd.Series(mask, index=test_frame.index)
        for scenario_index, (take_profit, stop_loss) in enumerate(scenarios):
            if feature_screening == "lightgbm" and screen_cache is not None:
                tp_bps = int(round(take_profit * 10_000))
                sl_bps = int(round(stop_loss * 10_000))
                label_column = (
                    f"label_{candidate.direction}_tp{tp_bps}_sl{sl_bps}_h{candidate.horizon_bars}"
                )
                train_frame = engine.train_frames[window_index]
                if label_column in train_frame.columns:
                    feature_columns = numeric_feature_columns(train_frame, engine.base_prefix)
                    screened = get_screened_features_cached(
                        train_frame, label_column,
                        candidate.direction, candidate.horizon_bars,
                        take_profit, stop_loss,
                        feature_columns, max_features, screen_cache, method="shap",
                    )
                    if any(c.feature not in screened for c in candidate.conditions):
                        per_scenario_stats[scenario_index].append(dict(_SCREENED_OUT_STATS))
                        continue
            config = DayTradeConfig(
                take_profit=take_profit,
                stop_loss=stop_loss,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                horizon_bars=candidate.horizon_bars,
                risk_per_trade=risk_per_trade,
                max_position_fraction=max_position_fraction,
                daily_stop_loss=daily_stop_loss,
                max_consecutive_losses=max_consecutive_losses,
                cooldown_bars=cooldown_bars,
                use_atr_tp_sl=use_atr_tp_sl,
            )
            trades = simulate_day_trades(
                test_frame, signal, candidate.direction, config, engine.base_prefix, arrays,
            )
            metrics = day_trade_metrics(trades)
            per_scenario_stats[scenario_index].append(
                {
                    "test_total_return": metrics["total_return"],
                    "test_avg_net_return": metrics["avg_net_return"] if metrics["trades"] > 0 else 0.0,
                    "test_profit_factor": metrics["profit_factor"] if np.isfinite(metrics["profit_factor"]) else 0.0,
                    "test_max_drawdown": metrics["max_drawdown"],
                    "test_trades": metrics["trades"],
                    "screened_out": False,
                }
            )
    rows: List[Dict[str, object]] = []
    for scenario_index, (take_profit, stop_loss) in enumerate(scenarios):
        stats = per_scenario_stats[scenario_index]
        summary = aggregate_walk_forward_results(stats, wf_pass_rate)
        window_returns = np.array([s["test_avg_net_return"] for s in stats], dtype=float)
        returns_series = pd.Series(window_returns)
        ci_low, ci_high = bootstrap_sharpe_ci(window_returns, n_boot=200)
        row: Dict[str, object] = {
            "direction": candidate.direction,
            "horizon_bars": candidate.horizon_bars,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "timeframes": ",".join(candidate.timeframes),
            "timeframe_count": len(candidate.timeframes),
            "conditions": len(candidate.conditions),
            "rule": candidate.rule,
        }
        row.update(_WF_ZEROED_COLUMNS)
        row.update({f"wf_{key}": value for key, value in summary.items()})
        row["wf_passes"] = bool(summary["passes_walk_forward"])
        row.update(
            {
                "wf_returns_sharpe": sharpe_ratio(window_returns),
                "wf_returns_skew": float(returns_series.skew()) if len(window_returns) > 2 else 0.0,
                "wf_returns_kurt": float(returns_series.kurt() + 3.0) if len(window_returns) > 3 else 3.0,
                "test_sharpe_ci_low": ci_low,
                "test_sharpe_ci_high": ci_high,
                "wf_window_returns_json": json.dumps([round(float(r), 8) for r in window_returns]),
            }
        )
        rows.append(row)
    return rows


def _evaluate_holdout_day(
    holdout: pd.DataFrame,
    refit_frame: pd.DataFrame,
    strategies: pd.DataFrame,
    base_prefix: str,
    fee_bps: float,
    slippage_bps: float,
    risk_per_trade: float,
    max_position_fraction: float,
    daily_stop_loss: float,
    max_consecutive_losses: int,
    cooldown_bars: int,
    use_atr_tp_sl: bool,
    top_n: int = 30,
) -> pd.DataFrame:
    """Score top strategies on the untouched holdout with the FULL risk
    simulation. Report-only: never used for ranking or filtering."""
    from src.walk_forward import _compute_threshold

    out = strategies.copy()
    for column in ("holdout_trades", "holdout_total_return", "holdout_win_rate", "holdout_avg_net_return"):
        out[column] = np.nan
    if out.empty or holdout.empty:
        return out
    for idx, row in out.head(top_n).iterrows():
        conditions = [Condition(**payload) for payload in json.loads(row["conditions_json"])]
        refit = []
        for condition in conditions:
            try:
                threshold = _compute_threshold(refit_frame, condition)
            except ValueError:
                threshold = condition.threshold
            refit.append(with_threshold(condition, threshold))
        config = DayTradeConfig(
            take_profit=float(row["take_profit"]),
            stop_loss=float(row["stop_loss"]),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            horizon_bars=int(row["horizon_bars"]),
            risk_per_trade=risk_per_trade,
            max_position_fraction=max_position_fraction,
            daily_stop_loss=daily_stop_loss,
            max_consecutive_losses=max_consecutive_losses,
            cooldown_bars=cooldown_bars,
            use_atr_tp_sl=use_atr_tp_sl,
        )
        mask = combined_mask(holdout, refit)
        trades = simulate_day_trades(holdout, mask, str(row["direction"]), config, base_prefix)
        metrics = day_trade_metrics(trades)
        out.loc[idx, "holdout_trades"] = metrics["trades"]
        out.loc[idx, "holdout_total_return"] = metrics["total_return"]
        out.loc[idx, "holdout_win_rate"] = metrics["win_rate"]
        out.loc[idx, "holdout_avg_net_return"] = metrics["avg_net_return"]
    return out


_DT_WORKER: Dict[str, object] = {}


def _dt_worker_init(payload: Dict[str, object]) -> None:
    base_prefix = f"tf_{payload['base_timeframe']}_"
    data = load_dataset(
        Path(payload["input_path"]), payload["horizons"], base_prefix, payload["base_timeframe"],
        columns=payload.get("columns"),
    )
    _DT_WORKER.update(payload)
    _DT_WORKER["data"] = data
    core = data.iloc[: payload["core_rows"]]
    windows = generate_windows(len(core), WalkForwardConfig(**payload["wf_config"]))
    _DT_WORKER["engine"] = DayTradeWindowEngine(core, windows, base_prefix)
    _DT_WORKER["screen_cache"] = FeatureScreenCache.create()


def _dt_score_chunk(indices: Sequence[int]) -> List[Dict[str, object]]:
    candidates: List[StrategyCandidate] = _DT_WORKER["candidates"]
    scenarios: List[Tuple[float, float]] = _DT_WORKER["scenarios"]
    rows: List[Dict[str, object]] = []
    for index in indices:
        candidate = candidates[index]
        candidate_rows = _score_candidate_walk_forward_day(
            _DT_WORKER["engine"], candidate, scenarios,
            fee_bps=_DT_WORKER["fee_bps"],
            slippage_bps=_DT_WORKER["slippage_bps"],
            risk_per_trade=_DT_WORKER["risk_per_trade"],
            max_position_fraction=_DT_WORKER["max_position_fraction"],
            daily_stop_loss=_DT_WORKER["daily_stop_loss"],
            max_consecutive_losses=_DT_WORKER["max_consecutive_losses"],
            cooldown_bars=_DT_WORKER["cooldown_bars"],
            use_atr_tp_sl=_DT_WORKER["use_atr_tp_sl"],
            wf_pass_rate=_DT_WORKER["wf_pass_rate"],
            feature_screening=_DT_WORKER["feature_screening"],
            screen_cache=_DT_WORKER["screen_cache"],
            max_features=_DT_WORKER["max_features"],
        )
        conditions_json = _conditions_payload_day(candidate)
        for row in candidate_rows:
            row["candidate_index"] = index
            row["conditions_json"] = conditions_json
        rows.extend(candidate_rows)
    return rows


def write_report(
    strategies: pd.DataFrame,
    yearly: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rejection_summary: Dict[str, int],
    path: Path,
    base_timeframe: str = "5m",
    walk_forward: bool = False,
) -> None:
    lines = [
        "# Day Trade Strategy Search Report",
        "",
        f"Base timeframe: {base_timeframe}. Searches for high-frequency strategies with risk management.",
        "",
        "## Filter Summary",
        "",
        "```text",
    ]
    for key, value in rejection_summary.items():
        lines.append(f"{key}: {value}")
    if not diagnostics.empty and "pool_pbo" in diagnostics.columns:
        pool_pbo = diagnostics["pool_pbo"].dropna()
        if not pool_pbo.empty:
            lines.append(f"pool_pbo: {float(pool_pbo.iloc[0]):.4f}")
    lines.extend(["```", ""])

    lines.extend(["## Top Passing Strategies", ""])
    if strategies.empty:
        lines.append("No strategies passed the filters.")
    else:
        if walk_forward:
            columns = [
                "direction", "horizon_bars", "take_profit", "stop_loss",
                "timeframes", "wf_pass_rate", "wf_expectancy",
                "wf_profit_factor_median", "wf_max_drawdown_worst",
                "wf_avg_trades", "wf_total_windows",
                "wf_scored_windows", "wf_screened_out_windows", "dsr", "rule",
            ]
        else:
            columns = [
                "direction", "horizon_bars", "take_profit", "stop_loss",
                "timeframes", "train_trades", "train_total_return",
                "test_trades", "test_total_return", "test_win_rate",
                "test_avg_net_return", "test_max_drawdown",
                "test_avg_trades_per_day", "test_sharpe_ratio", "dsr", "rule",
            ]
        available = [c for c in columns if c in strategies.columns]
        lines.append("```text")
        lines.append(strategies[available].head(30).to_string(index=False))
        lines.append("```")

    lines.extend(["", "## Best Near Misses", ""])
    if diagnostics.empty:
        lines.append("No scored candidates were produced.")
    else:
        if walk_forward:
            columns = [
                "direction", "horizon_bars", "take_profit", "stop_loss",
                "timeframes", "wf_pass_rate", "wf_expectancy",
                "wf_profit_factor_median", "wf_max_drawdown_worst",
                "wf_avg_trades", "wf_total_windows",
                "wf_scored_windows", "wf_screened_out_windows",
                "dsr", "passes_filters", "rule",
            ]
        else:
            columns = [
                "direction", "horizon_bars", "take_profit", "stop_loss",
                "timeframes", "train_trades", "train_total_return",
                "test_trades", "test_total_return", "test_win_rate",
                "test_avg_net_return", "test_max_drawdown",
                "dsr", "passes_filters", "rule",
            ]
        available = [c for c in columns if c in diagnostics.columns]
        lines.append("```text")
        lines.append(diagnostics[available].head(30).to_string(index=False))
        lines.append("```")

    if walk_forward:
        lines.extend(["", "## Year Breakdown", "", "Walk-forward mode: year metrics disabled to avoid stale-threshold reporting."])
    elif not yearly.empty:
        lines.extend(["", "## Year Breakdown", "", "```text"])
        year_cols = [
            "strategy_rank", "year", "direction", "horizon_bars",
            "take_profit", "stop_loss", "trades", "total_return",
            "win_rate", "avg_net_return", "max_drawdown",
            "avg_trades_per_day", "sharpe_ratio",
        ]
        available = [c for c in year_cols if c in yearly.columns]
        lines.append(yearly[available].to_string(index=False))
        lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_timeframe: str = "5m",
    horizons: Sequence[int] = (4, 8, 16),
    train_fraction: float = 0.7,
    max_features: int = 60,
    top_conditions: int = 60,
    max_pairs: int = 2500,
    max_triples: int = 2500,
    rank_sample_rows: int = 50_000,
    condition_depths: Sequence[int] = (1, 2, 3),
    min_train_trades: int = 100,
    min_test_trades: int = 50,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    take_profits: Sequence[float] = (0.003, 0.005, 0.008, 0.012),
    stop_losses: Sequence[float] = (0.002, 0.004, 0.006, 0.01),
    require_multitimeframe: bool = True,
    ranking_method: str = "blended",
    cross_tf_mode: str = "pool",
    enabled_kinds: Set[str] = DEFAULT_ENABLED_KINDS,
    risk_per_trade: float = 0.003,
    max_position_fraction: float = 0.25,
    daily_stop_loss: float = -0.02,
    max_consecutive_losses: int = 3,
    cooldown_bars: int = 24,
    walk_forward: bool = False,
    wf_train_bars: Optional[int] = None,
    wf_test_bars: Optional[int] = None,
    wf_step_bars: Optional[int] = None,
    wf_pass_rate: float = 0.8,
    wf_min_windows: int = 6,
    wf_embargo_bars: Optional[int] = None,
    feature_screening: str = "none",
    dsr_threshold: float = 0.0,
    regime_conditional: bool = False,
    cluster_jaccard: float = 0.8,
    report: bool = False,
    use_atr_tp_sl: bool = False,
    holdout_fraction: float = 0.2,
    n_jobs: int = 1,
    checkpoint_every: int = 25,
    resume: bool = False,
    feature_pattern: Optional[str] = None,
) -> pd.DataFrame:
    base_prefix = f"tf_{base_timeframe}_"
    data = load_dataset(input_path, horizons, base_prefix, base_timeframe)
    if regime_conditional and "tf_1d_regime_id" not in data.columns:
        LOGGER.warning("--regime-conditional requested but tf_1d_regime_id is missing; skipping regime breakdown")
        regime_conditional = False
    directions = ("long", "short")
    scenarios = list(itertools.product(take_profits, stop_losses))
    rows: List[Dict[str, object]] = []
    return_rows: List[Dict[str, object]] = []
    screen_cache = FeatureScreenCache.create()
    holdout = data.iloc[0:0]
    core = data
    windows: List[Tuple[slice, slice]] = []
    checkpoint_path = output_dir / "checkpoint.csv"
    meta_path = output_dir / "checkpoint_meta.json"
    if not walk_forward:
        train, test = split_train_test(data, train_fraction)
        candidates = make_candidates(
            train, horizons, directions=directions,
            max_features=max_features, top_conditions=top_conditions,
            max_pairs=max_pairs, max_triples=max_triples,
            rank_sample_rows=rank_sample_rows,
            condition_depths=condition_depths,
            ranking_method=ranking_method,
            cross_tf_mode=cross_tf_mode,
            enabled_kinds=enabled_kinds,
            base_prefix=base_prefix,
            feature_pattern=feature_pattern,
        )
        LOGGER.info(
            "Scoring %s strategy candidates across %s TP/SL scenarios",
            len(candidates), len(scenarios),
        )
        LOGGER.info("Pre-computing OHLC arrays for train/test splits...")
        train_arrays = PrecomputedArrays.from_dataframe(train, base_prefix)
        test_arrays = PrecomputedArrays.from_dataframe(test, base_prefix)
        for index, candidate in enumerate(candidates, start=1):
            if index == 1 or index % 500 == 0:
                LOGGER.info("Scoring candidate %s/%s", index, len(candidates))
            conditions_json = _conditions_payload_day(candidate)
            train_mask = combined_mask(train, candidate.conditions)
            test_mask = combined_mask(test, candidate.conditions)
            for take_profit, stop_loss in scenarios:
                config = DayTradeConfig(
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    horizon_bars=candidate.horizon_bars,
                    risk_per_trade=risk_per_trade,
                    max_position_fraction=max_position_fraction,
                    daily_stop_loss=daily_stop_loss,
                    max_consecutive_losses=max_consecutive_losses,
                    cooldown_bars=cooldown_bars,
                    use_atr_tp_sl=use_atr_tp_sl,
                )
                row = score_candidate(
                    train, test, candidate, config, base_prefix,
                    train_mask=train_mask, test_mask=test_mask,
                    train_arrays=train_arrays, test_arrays=test_arrays,
                )
                if regime_conditional:
                    row["regime_breakdown_json"] = json.dumps(
                        regime_breakdown(data, candidate, config, base_prefix),
                        sort_keys=True,
                    )
                return_payload = {
                    **row,
                    "train_returns": row.pop("train_returns"),
                    "test_returns": row.pop("test_returns"),
                }
                row["conditions_json"] = conditions_json
                rows.append(row)
                return_rows.append(return_payload)
    else:
        if regime_conditional:
            LOGGER.warning(
                "Regime breakdown is not computed in walk-forward mode "
                "(it would mix stale thresholds with full-history data)."
            )
        defaults = WALK_FORWARD_DEFAULTS.get(base_timeframe, WALK_FORWARD_DEFAULTS["5m"])
        wf_cfg = WalkForwardConfig(
            train_bars=wf_train_bars or defaults["train_bars"],
            test_bars=wf_test_bars or defaults["test_bars"],
            step_bars=wf_step_bars or defaults["step_bars"],
            min_windows=wf_min_windows,
            pass_rate=wf_pass_rate,
            embargo_bars=wf_embargo_bars if wf_embargo_bars is not None else max(horizons),
        )
        core_rows = len(data) - int(len(data) * holdout_fraction) if holdout_fraction > 0 else len(data)
        core = data.iloc[:core_rows]
        holdout = data.iloc[core_rows:]
        windows = generate_windows(len(core), wf_cfg)
        # Candidates from the FIRST train window only: no walk-forward test
        # window (nor the holdout) influences candidate selection.
        first_train = core.iloc[windows[0][0]].copy()
        candidates = make_candidates(
            first_train, horizons, directions=directions,
            max_features=max_features, top_conditions=top_conditions,
            max_pairs=max_pairs, max_triples=max_triples,
            rank_sample_rows=rank_sample_rows,
            condition_depths=condition_depths,
            ranking_method=ranking_method,
            cross_tf_mode=cross_tf_mode,
            enabled_kinds=enabled_kinds,
            base_prefix=base_prefix,
            feature_pattern=feature_pattern,
        )
        LOGGER.info(
            "Walk-forward: %s windows over %s core rows, %s holdout rows reserved, %s candidates",
            len(windows), len(core), len(holdout), len(candidates),
        )
        # Prune to the columns scoring actually touches. The 5m training table
        # is ~3k columns (~16 GB as float64); loading it whole in every worker
        # process takes the machine down.
        if feature_screening == "lightgbm":
            worker_columns = None
            LOGGER.warning(
                "feature_screening=lightgbm needs the full %s-column table in every "
                "worker. On a memory-constrained machine use --n-jobs 1 or "
                "--feature-screening none.",
                len(data.columns),
            )
        else:
            base_columns = {
                "timestamp",
                f"{base_prefix}open", f"{base_prefix}high",
                f"{base_prefix}low", f"{base_prefix}close",
            }
            atr_columns = {
                c for c in data.columns
                if c.endswith("_atr") or c.endswith("_atr_14")
            }
            worker_columns = sorted(
                (candidate_feature_columns(candidates) & set(data.columns)) | atr_columns
            )
            n_core = len(core)
            keep = [c for c in data.columns if c in set(worker_columns) | base_columns]
            data = data[keep]
            core = data.iloc[:n_core]
            holdout = data.iloc[n_core:]
        hash_config = {
            "input_path": str(input_path),
            "base_timeframe": base_timeframe,
            "horizons": list(horizons),
            "train_fraction": train_fraction,
            "max_features": max_features,
            "top_conditions": top_conditions,
            "max_pairs": max_pairs,
            "max_triples": max_triples,
            "rank_sample_rows": rank_sample_rows,
            "condition_depths": list(condition_depths),
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "take_profits": list(take_profits),
            "stop_losses": list(stop_losses),
            "use_atr_tp_sl": use_atr_tp_sl,
            "ranking_method": ranking_method,
            "cross_tf_mode": cross_tf_mode,
            "enabled_kinds": sorted(enabled_kinds),
            "risk_per_trade": risk_per_trade,
            "max_position_fraction": max_position_fraction,
            "daily_stop_loss": daily_stop_loss,
            "max_consecutive_losses": max_consecutive_losses,
            "cooldown_bars": cooldown_bars,
            "wf_train_bars": wf_cfg.train_bars,
            "wf_test_bars": wf_cfg.test_bars,
            "wf_step_bars": wf_cfg.step_bars,
            "wf_pass_rate": wf_pass_rate,
            "wf_min_windows": wf_min_windows,
            "wf_embargo_bars": wf_cfg.embargo_bars,
            "feature_screening": feature_screening,
            "holdout_fraction": holdout_fraction,
            "require_multitimeframe": require_multitimeframe,
            "feature_pattern": feature_pattern,
        }
        config_hash = _config_hash(hash_config)
        output_dir.mkdir(parents=True, exist_ok=True)
        done: Set[int] = set()
        if resume:
            rows, done = _load_checkpoint(checkpoint_path, meta_path, config_hash, len(scenarios))
        elif checkpoint_path.exists():
            checkpoint_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
        meta_path.write_text(
            json.dumps(
                {"config_hash": config_hash, "n_candidates": len(candidates), "n_scenarios": len(scenarios)},
                indent=2,
            ),
            encoding="utf-8",
        )
        pending = [index for index in range(len(candidates)) if index not in done]
        chunks = [
            pending[start: start + checkpoint_every]
            for start in range(0, len(pending), checkpoint_every)
        ]
        worker_payload = {
            "input_path": str(input_path),
            "base_timeframe": base_timeframe,
            "horizons": list(horizons),
            "columns": worker_columns,
            "core_rows": len(core),
            "wf_config": {
                "train_bars": wf_cfg.train_bars,
                "test_bars": wf_cfg.test_bars,
                "step_bars": wf_cfg.step_bars,
                "min_windows": wf_min_windows,
                "pass_rate": wf_pass_rate,
                "embargo_bars": wf_cfg.embargo_bars,
            },
            "candidates": candidates,
            "scenarios": scenarios,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "risk_per_trade": risk_per_trade,
            "max_position_fraction": max_position_fraction,
            "daily_stop_loss": daily_stop_loss,
            "max_consecutive_losses": max_consecutive_losses,
            "cooldown_bars": cooldown_bars,
            "use_atr_tp_sl": use_atr_tp_sl,
            "wf_pass_rate": wf_pass_rate,
            "feature_screening": feature_screening,
            "max_features": max_features,
        }
        scored = len(done)
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_dt_worker_init, initargs=(worker_payload,),
            ) as pool:
                futures = [pool.submit(_dt_score_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    chunk_rows = future.result()
                    _flush_rows(chunk_rows, checkpoint_path)
                    rows.extend(chunk_rows)
                    scored += len({row["candidate_index"] for row in chunk_rows})
                    LOGGER.info("Scored candidate %s/%s", scored, len(candidates))
        else:
            _DT_WORKER.clear()
            _DT_WORKER.update(worker_payload)
            _DT_WORKER["data"] = data
            _DT_WORKER["engine"] = DayTradeWindowEngine(core, windows, base_prefix)
            _DT_WORKER["screen_cache"] = screen_cache
            for chunk in chunks:
                chunk_rows = _dt_score_chunk(chunk)
                _flush_rows(chunk_rows, checkpoint_path)
                rows.extend(chunk_rows)
                scored += len(chunk)
                LOGGER.info("Scored candidate %s/%s", scored, len(candidates))

    strategies = pd.DataFrame(rows)
    strategies = _attach_statistical_metrics(strategies, return_rows, walk_forward=walk_forward)
    if walk_forward and not strategies.empty and "wf_window_returns_json" in strategies.columns:
        matrix = [json.loads(str(payload)) for payload in strategies["wf_window_returns_json"]]
        lengths = {len(item) for item in matrix}
        strategies["pool_pbo"] = (
            probability_backtest_overfitting(np.asarray(matrix, dtype=float))
            if len(lengths) == 1
            else np.nan
        )
    else:
        strategies["pool_pbo"] = np.nan
    diagnostics = strategies.copy()
    if not diagnostics.empty:
        if walk_forward:
            diagnostics["passes_trade_count"] = diagnostics["wf_avg_trades"] >= min_test_trades
            diagnostics["passes_profitability"] = (
                (diagnostics["wf_pass_rate"] >= wf_pass_rate)
                & (diagnostics["wf_expectancy"] > 0)
            )
        else:
            diagnostics["passes_trade_count"] = (
                (diagnostics["train_trades"] >= min_train_trades)
                & (diagnostics["test_trades"] >= min_test_trades)
            )
            diagnostics["passes_profitability"] = (
                (diagnostics["train_total_return"] > 0)
                & (diagnostics["test_total_return"] > 0)
            )
        diagnostics["passes_multitimeframe"] = (
            diagnostics["timeframe_count"] >= 2
            if require_multitimeframe
            else True
        )
        diagnostics["passes_dsr"] = diagnostics["dsr"] >= dsr_threshold
        diagnostics["passes_filters"] = (
            diagnostics["passes_trade_count"]
            & diagnostics["passes_profitability"]
            & diagnostics["passes_multitimeframe"]
            & diagnostics["passes_dsr"]
        )
        if walk_forward:
            diagnostics["passes_filters"] = diagnostics["passes_filters"] & diagnostics["wf_passes"]
        if walk_forward:
            sort_cols = [
                "dsr", "wf_pass_rate", "wf_expectancy", "wf_profit_factor_median",
                "wf_max_drawdown_worst", "wf_avg_trades",
            ]
            sort_asc = [False, False, False, False, False, False]
        else:
            sort_cols = ["dsr", "test_total_return", "test_avg_net_return", "test_trades"]
            sort_asc = [False, False, False, False]
        diagnostics = diagnostics.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
        strategies = diagnostics[diagnostics["passes_filters"]].copy()
        if require_multitimeframe:
            strategies = strategies[strategies["timeframe_count"] >= 2].copy()
        strategies = strategies.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    if walk_forward and not holdout.empty and not strategies.empty:
        refit_frame = core.iloc[windows[-1][0]]
        strategies = _evaluate_holdout_day(
            holdout, refit_frame, strategies, base_prefix,
            fee_bps, slippage_bps, risk_per_trade, max_position_fraction, daily_stop_loss,
            max_consecutive_losses, cooldown_bars, use_atr_tp_sl,
        )

    rejection_summary = summarize_filter_rejections(
        diagnostics, min_train_trades, min_test_trades, require_multitimeframe,
        walk_forward=walk_forward,
    )

    config_template = DayTradeConfig(
        take_profit=take_profits[0],
        stop_loss=stop_losses[0],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        horizon_bars=horizons[0],
        risk_per_trade=risk_per_trade,
        max_position_fraction=max_position_fraction,
        daily_stop_loss=daily_stop_loss,
        max_consecutive_losses=max_consecutive_losses,
        cooldown_bars=cooldown_bars,
        use_atr_tp_sl=use_atr_tp_sl,
    )
    if walk_forward:
        yearly = pd.DataFrame()
    else:
        yearly = add_year_metrics(
            data, strategies, config_template, base_prefix, top_n=10,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "git_sha": get_git_sha(),
        "search_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_path": str(input_path),
        "base_timeframe": base_timeframe,
        "horizons": list(horizons),
        "train_fraction": train_fraction,
        "max_features": max_features,
        "top_conditions": top_conditions,
        "max_pairs": max_pairs,
        "max_triples": max_triples,
        "rank_sample_rows": rank_sample_rows,
        "condition_depths": list(condition_depths),
        "min_train_trades": min_train_trades,
        "min_test_trades": min_test_trades,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "take_profits": list(take_profits),
        "stop_losses": list(stop_losses),
        "use_atr_tp_sl": use_atr_tp_sl,
        "require_multitimeframe": require_multitimeframe,
        "ranking_method": ranking_method,
        "cross_tf_mode": cross_tf_mode,
        "enabled_kinds": sorted(enabled_kinds),
        "risk_per_trade": risk_per_trade,
        "max_position_fraction": max_position_fraction,
        "daily_stop_loss": daily_stop_loss,
        "max_consecutive_losses": max_consecutive_losses,
        "cooldown_bars": cooldown_bars,
        "walk_forward": walk_forward,
        "wf_train_bars": wf_train_bars,
        "wf_test_bars": wf_test_bars,
        "wf_step_bars": wf_step_bars,
        "wf_pass_rate": wf_pass_rate,
        "wf_min_windows": wf_min_windows,
        "wf_embargo_bars": wf_embargo_bars,
        "feature_screening": feature_screening,
        "dsr_threshold": dsr_threshold,
        "regime_conditional": regime_conditional,
        "cluster_jaccard": cluster_jaccard,
        "holdout_fraction": holdout_fraction if walk_forward else 0.0,
        "n_jobs": n_jobs,
        "checkpoint_every": checkpoint_every,
        "resume": resume,
        "feature_pattern": feature_pattern,
        "feature_screen_cache_enabled": bool(walk_forward and feature_screening == "lightgbm"),
        "feature_screen_cache_hits": screen_cache.hits,
        "feature_screen_cache_misses": screen_cache.misses,
        "feature_screen_cache_entries": len(screen_cache.values),
        "feature_screen_cache_key_fields": [
            "train_start",
            "train_stop",
            "label_column",
            "direction",
            "horizon_bars",
            "take_profit",
            "stop_loss",
            "max_features",
            "method",
            "feature_columns_hash",
        ],
        "report": report,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8",
    )
    heavy_columns = ["wf_window_returns_json"]
    diagnostics.drop(columns=heavy_columns, errors="ignore").to_csv(
        output_dir / "scored_strategies_all.csv", index=False,
    )
    (output_dir / "filter_summary.json").write_text(
        json.dumps(rejection_summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    strategies = strategies.drop(columns=heavy_columns, errors="ignore")
    strategies.to_csv(output_dir / "ranked_strategies.csv", index=False)
    clustered = cluster_ranked_strategies(data, strategies, threshold=cluster_jaccard)
    clustered.to_csv(output_dir / "ranked_strategies_clustered.csv", index=False)
    yearly.to_csv(output_dir / "ranked_strategies_by_year.csv", index=False)
    write_report(
        strategies, yearly, diagnostics, rejection_summary,
        output_dir / "report.md", base_timeframe, walk_forward=walk_forward,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    return strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for day trading strategies with risk management."
    )
    parser.add_argument(
        "--input-path", type=Path,
        default=PROCESSED_DATA_DIR / "train_5m_indicators.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-timeframe", "--base-tf", dest="base_timeframe", default="5m")
    parser.add_argument("--horizon", action="append", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--top-conditions", type=int, default=60)
    parser.add_argument("--max-pairs", type=int, default=2500)
    parser.add_argument("--max-triples", type=int, default=2500)
    parser.add_argument("--rank-sample-rows", type=int, default=50_000)
    parser.add_argument(
        "--condition-depth", action="append", type=int, choices=(1, 2, 3),
    )
    parser.add_argument("--min-train-trades", type=int, default=100)
    parser.add_argument("--min-test-trades", type=int, default=50)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--take-profit", action="append", type=float)
    parser.add_argument("--stop-loss", action="append", type=float)
    parser.add_argument("--use-atr-tp-sl", action="store_true", help="Use ATR-based dynamic stop loss and take profit.")
    parser.add_argument("--require-multitimeframe", action="store_true", default=True)
    parser.add_argument("--no-require-multitimeframe", dest="require_multitimeframe", action="store_false")
    parser.add_argument(
        "--ranking-method", choices=("spearman", "importance", "blended"),
        default="blended",
    )
    parser.add_argument(
        "--cross-tf-mode", choices=("none", "pool"), default="pool",
    )
    parser.add_argument("--enabled-kinds", nargs="+", default=None)
    parser.add_argument("--risk-per-trade", type=float, default=0.003)
    parser.add_argument("--max-position-fraction", type=float, default=0.25)
    parser.add_argument("--daily-stop-loss", type=float, default=-0.02)
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--cooldown-bars", type=int, default=24)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-train-bars", type=int, default=None)
    parser.add_argument("--wf-test-bars", type=int, default=None)
    parser.add_argument("--wf-step-bars", type=int, default=None)
    parser.add_argument("--wf-pass-rate", type=float, default=0.8)
    parser.add_argument("--wf-min-windows", type=int, default=6)
    parser.add_argument("--wf-embargo-bars", type=int, default=None)
    parser.add_argument("--feature-screening", choices=("none", "lightgbm"), default="none")
    parser.add_argument("--dsr-threshold", type=float, default=0.0)
    parser.add_argument("--regime-conditional", action="store_true")
    parser.add_argument("--cluster-jaccard", type=float, default=0.8)
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.2,
        help="Final fraction of data excluded from walk-forward; scored report-only.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker processes for walk-forward candidate scoring. 1 = sequential.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Flush scored candidates to the checkpoint file every N candidates (walk-forward mode).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a walk-forward run from an existing checkpoint in --output-dir.",
    )
    parser.add_argument(
        "--feature-pattern", default=None,
        help="Regex restricting the candidate feature universe (e.g. 'cvd_|taker_' "
             "to mine only order-flow features).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    bt = args.base_timeframe
    horizons = tuple(args.horizon) if args.horizon else (4, 8, 16)
    condition_depths = tuple(args.condition_depth) if args.condition_depth else (1, 2, 3)
    if args.use_atr_tp_sl:
        # In ATR mode TP/SL values are ATR MULTIPLES, not fractional returns.
        # The percent-scale defaults (0.003-0.012) would produce microscopic
        # stops (0.003 x ATR), so ATR mode gets its own default grid.
        take_profits = tuple(args.take_profit) if args.take_profit else (1.0, 1.5, 2.0, 3.0)
        stop_losses = tuple(args.stop_loss) if args.stop_loss else (0.75, 1.0, 1.5, 2.0)
    else:
        take_profits = tuple(args.take_profit) if args.take_profit else (0.003, 0.005, 0.008, 0.012)
        stop_losses = tuple(args.stop_loss) if args.stop_loss else (0.002, 0.004, 0.006, 0.01)
    enabled_kinds = set(args.enabled_kinds) if args.enabled_kinds else DEFAULT_ENABLED_KINDS
    input_path = args.input_path
    if str(input_path) == str(PROCESSED_DATA_DIR / "train_5m_indicators.parquet") and bt != "5m":
        input_path = PROCESSED_DATA_DIR / f"train_{bt}_indicators.parquet"

    strategies = run(
        input_path=input_path,
        output_dir=args.output_dir,
        base_timeframe=bt,
        horizons=horizons,
        train_fraction=args.train_fraction,
        max_features=args.max_features,
        top_conditions=args.top_conditions,
        max_pairs=args.max_pairs,
        max_triples=args.max_triples,
        rank_sample_rows=args.rank_sample_rows,
        condition_depths=condition_depths,
        min_train_trades=args.min_train_trades,
        min_test_trades=args.min_test_trades,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        take_profits=take_profits,
        stop_losses=stop_losses,
        require_multitimeframe=args.require_multitimeframe,
        ranking_method=args.ranking_method,
        cross_tf_mode=args.cross_tf_mode,
        enabled_kinds=enabled_kinds,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        daily_stop_loss=args.daily_stop_loss,
        max_consecutive_losses=args.max_consecutive_losses,
        cooldown_bars=args.cooldown_bars,
        walk_forward=args.walk_forward,
        wf_train_bars=args.wf_train_bars,
        wf_test_bars=args.wf_test_bars,
        wf_step_bars=args.wf_step_bars,
        wf_pass_rate=args.wf_pass_rate,
        wf_min_windows=args.wf_min_windows,
        wf_embargo_bars=args.wf_embargo_bars,
        feature_screening=args.feature_screening,
        dsr_threshold=args.dsr_threshold,
        regime_conditional=args.regime_conditional,
        cluster_jaccard=args.cluster_jaccard,
        report=args.report,
        use_atr_tp_sl=args.use_atr_tp_sl,
        holdout_fraction=args.holdout_fraction,
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        feature_pattern=args.feature_pattern,
    )
    print(f"Wrote {args.output_dir / 'ranked_strategies.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    print(f"Strategies passing filters: {len(strategies)}")


if __name__ == "__main__":
    main()
