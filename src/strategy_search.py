import argparse
import itertools
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_dataset import TARGET_COLUMNS
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
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
from src.trade_utils import gross_return_for_pnl_unit, gross_return_numba
from src.walk_forward import (
    WalkForwardConfig,
    WindowConditionCache,
    aggregate_walk_forward_results,
    candidate_feature_columns,
    condition_cache_key,
    generate_purged_kfold_windows,
    generate_windows,
    with_threshold,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "train_15m_indicators.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "strategy_search"
TIMEFRAME_PREFIXES = ("tf_15m_", "tf_30m_", "tf_1h_", "tf_4h_", "tf_1d_", "tf_1w_")


@dataclass(frozen=True)
class StrategyCandidate:
    direction: str
    horizon_bars: int
    conditions: tuple[Condition, ...]

    @property
    def rule(self) -> str:
        return " AND ".join(condition.description for condition in self.conditions)

    @property
    def timeframes(self) -> tuple[str, ...]:
        values = []
        for condition in self.conditions:
            values.append(timeframe_for_feature(condition.feature))
        return tuple(sorted(set(values)))


def timeframe_for_feature(feature: str) -> str:
    for prefix in TIMEFRAME_PREFIXES:
        if feature.startswith(prefix):
            return prefix.removeprefix("tf_").removesuffix("_")
    return "unknown"


@dataclass(frozen=True)
class TradeConfig:
    fee_bps: float
    slippage_bps: float
    take_profit: float
    stop_loss: float
    pnl_unit: str = "usdt"
    use_triple_barrier_labels: bool = False


@dataclass(frozen=True)
class SimArrays:
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, base_prefix: str = "tf_15m_") -> "SimArrays":
        return cls(
            open_=data[f"{base_prefix}open"].astype(float).to_numpy(),
            high=data[f"{base_prefix}high"].astype(float).to_numpy(),
            low=data[f"{base_prefix}low"].astype(float).to_numpy(),
            close=data[f"{base_prefix}close"].astype(float).to_numpy(),
            timestamps=data["timestamp"].to_numpy(),
        )


def _simulate_net_returns_python(
    open_, high, low, close, signal, is_long, horizon_bars,
    take_profit, stop_loss, total_cost, pnl_btc,
):
    n = len(signal)
    max_entry_index = n - horizon_bars - 1
    net_returns = np.empty(n, dtype=np.float64)
    count = 0
    next_allowed = 0
    for si in range(n):
        ei = si + 1
        if not signal[si] or ei < next_allowed or ei > max_entry_index:
            continue
        entry = open_[ei]
        xi = ei + horizon_bars
        xp = close[xi]
        for k in range(ei, ei + horizon_bars + 1):
            if is_long:
                if low[k] <= entry * (1.0 - stop_loss):
                    xi = k
                    xp = entry * (1.0 - stop_loss)
                    break
                if high[k] >= entry * (1.0 + take_profit):
                    xi = k
                    xp = entry * (1.0 + take_profit)
                    break
            else:
                if high[k] >= entry * (1.0 + stop_loss):
                    xi = k
                    xp = entry * (1.0 + stop_loss)
                    break
                if low[k] <= entry * (1.0 - take_profit):
                    xi = k
                    xp = entry * (1.0 - take_profit)
                    break
        gr = gross_return_numba(entry, xp, is_long, pnl_btc)
        net_returns[count] = gr - total_cost
        count += 1
        next_allowed = xi + 1
    return net_returns[:count]


try:
    from numba import njit as _njit

    _simulate_net_returns_numba = _njit(cache=True)(_simulate_net_returns_python)
    _HAS_NUMBA = True
except ImportError:
    _simulate_net_returns_numba = None
    _HAS_NUMBA = False


def simulate_net_returns(
    arrays: SimArrays,
    signal: np.ndarray,
    direction: str,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
    take_profit: float,
    stop_loss: float,
    pnl_unit: str = "usdt",
) -> np.ndarray:
    normalized_pnl_unit = str(pnl_unit).lower()
    if normalized_pnl_unit not in {"usdt", "btc"}:
        raise ValueError(f"Unsupported pnl_unit {pnl_unit!r}; expected 'usdt' or 'btc'")
    total_cost = 2 * ((fee_bps + slippage_bps) / 10_000)
    fn = _simulate_net_returns_numba if _HAS_NUMBA else _simulate_net_returns_python
    return fn(
        arrays.open_, arrays.high, arrays.low, arrays.close,
        signal, direction == "long", horizon_bars,
        take_profit, stop_loss, total_cost, normalized_pnl_unit == "btc",
    )


BASE_COLUMNS = ("timestamp", "tf_15m_open", "tf_15m_high", "tf_15m_low", "tf_15m_close")


def load_dataset(path: Path, horizons: Sequence[int], columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    read_columns = None
    if columns is not None:
        read_columns = sorted(set(columns) | set(BASE_COLUMNS))
    data = pd.read_parquet(path, columns=read_columns).sort_values("timestamp").reset_index(drop=True)
    required = set(BASE_COLUMNS)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    for horizon in horizons:
        data[f"future_return_{horizon}_bars"] = (
            data["tf_15m_close"].shift(-horizon) / data["tf_15m_close"] - 1
        )
    return data.dropna(subset=[f"future_return_{h}_bars" for h in horizons]).reset_index(
        drop=True
    )


def numeric_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "tf_15m_open",
        "tf_15m_high",
        "tf_15m_low",
        "tf_15m_close",
    } | set(TARGET_COLUMNS)
    excluded.update(column for column in data.columns if column.startswith("future_return_"))
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in excluded and column.startswith(TIMEFRAME_PREFIXES)
    ]


def _finite_quantiles(series: pd.Series, quantiles: Sequence[float]) -> dict[float, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {}
    values = clean.quantile(list(quantiles)).to_dict()
    return {float(key): float(value) for key, value in values.items() if pd.notna(value)}


def build_feature_conditions(train: pd.DataFrame, feature: str) -> list[Condition]:
    conditions = []
    for quantile, threshold in _finite_quantiles(train[feature], [0.1, 0.2, 0.8, 0.9]).items():
        kind = "value_le" if quantile < 0.5 else "value_ge"
        op = "<=" if quantile < 0.5 else ">="
        conditions.append(
            Condition(
                feature,
                kind,
                threshold,
                f"{feature} {op} train q{int(quantile * 100)} ({threshold:.6g})",
                threshold_source="quantile",
                quantile=float(quantile),
            )
        )
    delta = train[feature].diff()
    for quantile, threshold in _finite_quantiles(delta, [0.1, 0.9]).items():
        kind = "delta_le" if quantile < 0.5 else "delta_ge"
        trend = "falling" if quantile < 0.5 else "rising"
        bound = "<= train q10" if quantile < 0.5 else ">= train q90"
        conditions.append(
            Condition(
                feature,
                kind,
                threshold,
                f"{feature} {trend} fast: 1-bar change {bound} ({threshold:.6g})",
                threshold_source="delta_quantile",
                quantile=float(quantile),
            )
        )
    return conditions


def rank_features_by_direction(
    train: pd.DataFrame,
    features: Iterable[str],
    horizon: int,
    direction: str,
    max_features: int,
    sample_rows: int = 50_000,
) -> list[str]:
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


def combined_mask(data: pd.DataFrame, conditions: Sequence[Condition]) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for condition in conditions:
        mask &= condition_mask(data, condition).fillna(False)
    return mask


def simulate_trades(
    data: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
    take_profit: float,
    stop_loss: float,
    pnl_unit: str = "usdt",
    use_triple_barrier_labels: bool = False,
) -> pd.DataFrame:
    open_ = data["tf_15m_open"].astype(float).to_numpy()
    high = data["tf_15m_high"].astype(float).to_numpy()
    low = data["tf_15m_low"].astype(float).to_numpy()
    close = data["tf_15m_close"].astype(float).to_numpy()
    timestamps = data["timestamp"].to_numpy()
    signal = signal_mask.fillna(False).to_numpy()
    total_cost = 2 * ((fee_bps + slippage_bps) / 10_000)
    trades = []
    next_allowed_entry = 0
    max_entry_index = len(data) - horizon_bars - 1

    label_column = f"label_{direction}_tp{int(round(take_profit * 10_000))}_sl{int(round(stop_loss * 10_000))}_h{horizon_bars}"
    bars_column = f"bars_to_exit_{direction}_tp{int(round(take_profit * 10_000))}_sl{int(round(stop_loss * 10_000))}_h{horizon_bars}"
    can_use_labels = use_triple_barrier_labels and label_column in data.columns and bars_column in data.columns

    for signal_index, should_enter in enumerate(signal):
        entry_index = signal_index + 1
        if not should_enter or entry_index < next_allowed_entry or entry_index > max_entry_index:
            continue
        entry = open_[entry_index]
        exit_index = entry_index + horizon_bars
        exit_price = close[exit_index]
        exit_reason = "time"

        if can_use_labels:
            label = int(data[label_column].iloc[signal_index])
            bars_to_exit = int(data[bars_column].iloc[signal_index])
            exit_index = min(entry_index + max(1, bars_to_exit), len(data) - 1)
            if label > 0:
                exit_reason = "take_profit"
                exit_price = entry * (1 + take_profit) if direction == "long" else entry * (1 - take_profit)
            elif label < 0:
                exit_reason = "stop"
                exit_price = entry * (1 - stop_loss) if direction == "long" else entry * (1 + stop_loss)
            else:
                exit_reason = "time"
                exit_price = close[exit_index]
        else:
            for index in range(entry_index, entry_index + horizon_bars + 1):
                if direction == "long":
                    stop_hit = low[index] <= entry * (1 - stop_loss)
                    take_hit = high[index] >= entry * (1 + take_profit)
                    if stop_hit:
                        exit_index = index
                        exit_price = entry * (1 - stop_loss)
                        exit_reason = "stop"
                        break
                    if take_hit:
                        exit_index = index
                        exit_price = entry * (1 + take_profit)
                        exit_reason = "take_profit"
                        break
                else:
                    stop_hit = high[index] >= entry * (1 + stop_loss)
                    take_hit = low[index] <= entry * (1 - take_profit)
                    if stop_hit:
                        exit_index = index
                        exit_price = entry * (1 + stop_loss)
                        exit_reason = "stop"
                        break
                    if take_hit:
                        exit_index = index
                        exit_price = entry * (1 - take_profit)
                        exit_reason = "take_profit"
                        break

        gross_return = gross_return_for_pnl_unit(
            entry,
            exit_price,
            is_long=direction == "long",
            pnl_unit=pnl_unit,
        )
        net_return = gross_return - total_cost
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
                "holding_bars": exit_index - entry_index,
            }
        )
        next_allowed_entry = exit_index + 1

    return pd.DataFrame(trades)


def returns_metrics(returns: np.ndarray) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": np.nan,
            "return_skew": 0.0,
            "return_kurt": 3.0,
            "sharpe": 0.0,
        }
    series = pd.Series(returns)
    equity = np.cumprod(1 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "trades": int(returns.size),
        "win_rate": float((returns > 0).mean()),
        "avg_net_return": float(returns.mean()),
        "median_net_return": float(np.median(returns)),
        "total_return": float(equity[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "return_skew": float(series.skew()) if returns.size > 2 else 0.0,
        "return_kurt": float(series.kurt() + 3.0) if returns.size > 3 else 3.0,
        "sharpe": sharpe_ratio(returns),
    }


def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return returns_metrics(np.array([], dtype=float))
    return returns_metrics(trades["net_return"].astype(float).to_numpy())


def score_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: StrategyCandidate,
    fee_bps: float,
    slippage_bps: float,
    take_profit: float,
    stop_loss: float,
    pnl_unit: str = "usdt",
    use_triple_barrier_labels: bool = False,
) -> dict[str, object]:
    train_mask = combined_mask(train, candidate.conditions)
    test_mask = combined_mask(test, candidate.conditions)
    train_trades = simulate_trades(
        train,
        train_mask,
        candidate.direction,
        candidate.horizon_bars,
        fee_bps,
        slippage_bps,
        take_profit,
        stop_loss,
        pnl_unit=pnl_unit,
        use_triple_barrier_labels=use_triple_barrier_labels,
    )
    test_trades = simulate_trades(
        test,
        test_mask,
        candidate.direction,
        candidate.horizon_bars,
        fee_bps,
        slippage_bps,
        take_profit,
        stop_loss,
        pnl_unit=pnl_unit,
        use_triple_barrier_labels=use_triple_barrier_labels,
    )
    train_metrics = trade_metrics(train_trades)
    test_metrics = trade_metrics(test_trades)
    return {
        "direction": candidate.direction,
        "horizon_bars": candidate.horizon_bars,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "timeframes": ",".join(candidate.timeframes),
        "timeframe_count": len(candidate.timeframes),
        "conditions": len(candidate.conditions),
        "rule": candidate.rule,
        "train_returns": train_trades["net_return"].astype(float).to_numpy() if not train_trades.empty else np.array([], dtype=float),
        "test_returns": test_trades["net_return"].astype(float).to_numpy() if not test_trades.empty else np.array([], dtype=float),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def score_candidate_with_config(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: StrategyCandidate,
    trade_config: TradeConfig,
    base_prefix: str = "tf_15m_",
) -> dict[str, object]:
    _ = base_prefix
    return score_candidate(
        train,
        test,
        candidate,
        trade_config.fee_bps,
        trade_config.slippage_bps,
        trade_config.take_profit,
        trade_config.stop_loss,
        pnl_unit=trade_config.pnl_unit,
        use_triple_barrier_labels=trade_config.use_triple_barrier_labels,
    )


def _rank_features(
    train: pd.DataFrame,
    all_features: list[str],
    horizon: int,
    direction: str,
    max_features: int,
    ranking_method: str,
    rank_sample_rows: int,
) -> list[str]:
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


def _build_conditions_for_features(
    train: pd.DataFrame,
    ranked_features: list[str],
    enabled_kinds: set[str],
    cross_feature_pairs: Sequence[tuple[str, str]] | None,
) -> list[Condition]:
    if enabled_kinds == {"value", "delta"}:
        return [
            condition
            for feature in ranked_features
            for condition in build_feature_conditions(train, feature)
        ]
    return build_all_conditions(
        train, ranked_features, enabled_kinds=enabled_kinds,
        cross_feature_pairs=cross_feature_pairs,
    )


def _score_and_select_conditions(
    train: pd.DataFrame,
    conditions: list[Condition],
    horizon: int,
    direction: str,
    top_conditions: int,
    min_support: int = 500,
) -> list[int]:
    target = train[f"future_return_{horizon}_bars"]
    if direction == "short":
        target = -target
    condition_scores = []
    for index, condition in enumerate(conditions):
        mask = condition_mask(train, condition).fillna(False)
        selected = target.loc[mask]
        if len(selected) < min_support:
            continue
        condition_scores.append((index, float(selected.mean()), int(len(selected))))
    condition_scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [index for index, _, _ in condition_scores[:top_conditions]]


def _generate_pairs_flat(
    conditions: list[Condition],
    selected_indices: list[int],
    max_pairs: int,
) -> list[tuple[int, int]]:
    pairs = []
    for left_pos, left_index in enumerate(selected_indices):
        for right_index in selected_indices[left_pos + 1:]:
            if conditions[left_index].feature == conditions[right_index].feature:
                continue
            pairs.append((left_index, right_index))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _generate_pairs_pool(
    conditions: list[Condition],
    selected_indices: list[int],
    max_pairs: int,
) -> list[tuple[int, int]]:
    pools: dict[str, list[int]] = defaultdict(list)
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
    pairs: list[tuple[int, int]] = []
    for tf_a, tf_b in tf_combos:
        combo_count = 0
        for left_index in pools[tf_a]:
            for right_index in pools[tf_b]:
                if conditions[left_index].feature == conditions[right_index].feature:
                    continue
                pairs.append((left_index, right_index))
                combo_count += 1
                if combo_count >= pairs_per_combo:
                    break
            if combo_count >= pairs_per_combo:
                break
        if len(pairs) >= max_pairs:
            break
    return pairs[:max_pairs]


def _generate_pairs_shap(
    train: pd.DataFrame,
    conditions: list[Condition],
    selected_indices: list[int],
    all_features: list[str],
    horizon: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    from src.feature_ranking import suggest_feature_pairs

    target_column = f"future_return_{horizon}_bars"
    shap_pairs = suggest_feature_pairs(
        train, all_features, target_column, max_pairs=max_pairs * 2,
    )
    shap_feature_set = {(a, b) for a, b in shap_pairs} | {(b, a) for a, b in shap_pairs}

    pairs: list[tuple[int, int]] = []
    for left_pos, left_index in enumerate(selected_indices):
        for right_index in selected_indices[left_pos + 1:]:
            left_f = conditions[left_index].feature
            right_f = conditions[right_index].feature
            if left_f == right_f:
                continue
            if (left_f, right_f) in shap_feature_set:
                pairs.append((left_index, right_index))
                if len(pairs) >= max_pairs:
                    return pairs
    if not pairs:
        return _generate_pairs_flat(conditions, selected_indices, max_pairs)
    return pairs


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
    ranking_method: str = "spearman",
    cross_tf_mode: str = "none",
    enabled_kinds: set[str] = DEFAULT_ENABLED_KINDS,
    shap_screen: bool = False,
    shap_target: str = "sign",
) -> list[StrategyCandidate]:
    all_features = numeric_feature_columns(train)
    candidates: list[StrategyCandidate] = []
    for horizon in horizons:
        for direction in directions:
            LOGGER.info(
                "Ranking features for direction=%s horizon=%s method=%s",
                direction, horizon, ranking_method,
            )
            ranked_features = _rank_features(
                train, all_features, horizon, direction,
                max_features, ranking_method, rank_sample_rows,
            )
            if shap_screen:
                from src.feature_screener import screen_features

                train = train.copy()
                label_column = f"screen_label_{direction}_{horizon}"
                if shap_target == "triple-barrier":
                    matching = [
                        col for col in train.columns
                        if col.startswith(f"label_{direction}_") and col.endswith(f"_h{horizon}")
                    ]
                    if matching:
                        label_column = matching[0]
                    else:
                        LOGGER.warning(
                            "No triple-barrier SHAP target for direction=%s horizon=%s; using sign target",
                            direction,
                            horizon,
                        )
                        target = train[f"future_return_{horizon}_bars"]
                        train[label_column] = (target > 0).astype(int) if direction == "long" else (target < 0).astype(int)
                else:
                    target = train[f"future_return_{horizon}_bars"]
                    train[label_column] = (target > 0).astype(int) if direction == "long" else (target < 0).astype(int)
                ranked_features = screen_features(
                    train,
                    label_column,
                    ranked_features,
                    max_features=max(1, min(max_features, len(ranked_features))),
                )

            cross_pairs = detect_cross_feature_pairs(ranked_features) if "cross" in enabled_kinds or "ratio" in enabled_kinds else None
            conditions = _build_conditions_for_features(
                train, ranked_features, enabled_kinds, cross_pairs,
            )
            selected_indices = _score_and_select_conditions(
                train, conditions, horizon, direction, top_conditions,
            )
            LOGGER.info(
                "Selected %s base conditions for direction=%s horizon=%s",
                len(selected_indices), direction, horizon,
            )

            if 1 in condition_depths:
                for index in selected_indices:
                    candidates.append(
                        StrategyCandidate(direction, horizon, (conditions[index],))
                    )

            if 2 in condition_depths or 3 in condition_depths:
                if cross_tf_mode == "pool":
                    selected_pairs = _generate_pairs_pool(
                        conditions, selected_indices, max_pairs,
                    )
                elif cross_tf_mode == "shap":
                    selected_pairs = _generate_pairs_shap(
                        train, conditions, selected_indices,
                        all_features, horizon, max_pairs,
                    )
                else:
                    selected_pairs = _generate_pairs_flat(
                        conditions, selected_indices, max_pairs,
                    )

                if 2 in condition_depths:
                    for left_index, right_index in selected_pairs:
                        candidates.append(
                            StrategyCandidate(
                                direction, horizon,
                                (conditions[left_index], conditions[right_index]),
                            )
                        )

                if 3 in condition_depths:
                    triple_count = 0
                    seen_triples: set = set()
                    for left_index, right_index in selected_pairs:
                        used_features = {
                            conditions[left_index].feature,
                            conditions[right_index].feature,
                        }
                        for third_index in selected_indices:
                            if third_index in {left_index, right_index}:
                                continue
                            third = conditions[third_index]
                            if third.feature in used_features:
                                continue
                            triple = tuple(sorted((left_index, right_index, third_index)))
                            if triple in seen_triples:
                                continue
                            seen_triples.add(triple)
                            candidates.append(
                                StrategyCandidate(
                                    direction, horizon,
                                    tuple(conditions[index] for index in triple),
                                )
                            )
                            triple_count += 1
                            if triple_count >= max_triples:
                                break
                        if triple_count >= max_triples:
                            break
    seen = set()
    deduped: list[StrategyCandidate] = []
    for candidate in candidates:
        signature = (
            candidate.direction,
            candidate.horizon_bars,
            frozenset(condition.signature() for condition in candidate.conditions),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def _conditions_payload(candidate: StrategyCandidate) -> str:
    return json.dumps(
        [
            {
                "feature": condition.feature,
                "kind": condition.kind,
                "threshold": condition.threshold,
                "description": condition.description,
                "feature_b": condition.feature_b,
                "threshold_source": condition.threshold_source,
                "quantile": condition.quantile,
                "lookback": condition.lookback,
                "cross_feature": condition.cross_feature,
            }
            for condition in candidate.conditions
        ]
    )


def _attach_statistical_metrics(strategies: pd.DataFrame, walk_forward: bool = False) -> pd.DataFrame:
    """Compute DSR for every scored row from its stored distribution columns.

    The deflation term uses the dispersion of Sharpe estimates across ALL
    trials in this run, per Bailey/Lopez de Prado — so it must run after all
    rows are scored, never incrementally.
    """
    if strategies.empty:
        return strategies
    out = strategies.copy()
    if walk_forward:
        sr = out["wf_returns_sharpe"].astype(float)
        skew = out["wf_returns_skew"].astype(float)
        kurt = out["wf_returns_kurt"].astype(float)
        n_obs = out["wf_windows"].astype(float)
    else:
        sr = out["train_sharpe"].astype(float)
        skew = out["train_return_skew"].astype(float)
        kurt = out["train_return_kurt"].astype(float)
        n_obs = out["train_trades"].astype(float)
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
        for s, sk, ku, n in zip(sr, skew, kurt, n_obs, strict=False)
    ]
    return out


def regime_breakdown(
    data: pd.DataFrame,
    candidate: StrategyCandidate,
    fee_bps: float,
    slippage_bps: float,
    take_profit: float,
    stop_loss: float,
    pnl_unit: str = "usdt",
) -> dict[str, dict[str, float]]:
    if "tf_1d_regime_id" not in data.columns:
        return {}
    breakdown: dict[str, dict[str, float]] = {}
    for regime_id, regime_data in data.groupby("tf_1d_regime_id"):
        trades = simulate_trades(
            regime_data,
            combined_mask(regime_data, candidate.conditions),
            candidate.direction,
            candidate.horizon_bars,
            fee_bps,
            slippage_bps,
            take_profit,
            stop_loss,
            pnl_unit=pnl_unit,
        )
        metrics = trade_metrics(trades)
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


def cluster_ranked_strategies(
    data: pd.DataFrame,
    strategies: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    out = strategies.copy()
    if out.empty or "conditions_json" not in out.columns:
        out["cluster_id"] = pd.Series(dtype=int)
        return out
    masks: dict[str, np.ndarray] = {}
    for idx, row in out.iterrows():
        conditions = tuple(Condition(**payload) for payload in json.loads(row["conditions_json"]))
        masks[str(idx)] = combined_mask(data, conditions).to_numpy()
    clusters = cluster_strategies_by_overlap(masks, jaccard_threshold=threshold)
    representatives = []
    for cluster_id, members in clusters.items():
        member_indices = [int(member) for member in members]
        cluster_rows = out.loc[member_indices].copy()
        cluster_rows["cluster_id"] = int(cluster_id)
        cluster_rows = cluster_rows.sort_values(
            ["dsr", "test_total_return", "test_avg_net_return", "test_trades"],
            ascending=[False, False, False, False],
        )
        representatives.append(cluster_rows.iloc[0])
    return pd.DataFrame(representatives).sort_values(
        ["dsr", "test_total_return", "test_avg_net_return", "test_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def add_year_metrics(
    data: pd.DataFrame,
    strategies: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    top_n: int,
    pnl_unit: str = "usdt",
) -> pd.DataFrame:
    if strategies.empty:
        return pd.DataFrame()
    rows = []
    dated = data.copy()
    dated["year"] = pd.to_datetime(dated["timestamp"], utc=True).dt.year
    for rank, row in strategies.head(top_n).iterrows():
        conditions = [
            Condition(**payload)
            for payload in json.loads(row["conditions_json"])
        ]
        candidate = StrategyCandidate(row["direction"], int(row["horizon_bars"]), tuple(conditions))
        take_profit = float(row["take_profit"])
        stop_loss = float(row["stop_loss"])
        for year, year_data in dated.groupby("year"):
            trades = simulate_trades(
                year_data,
                combined_mask(year_data, candidate.conditions),
                candidate.direction,
                candidate.horizon_bars,
                fee_bps,
                slippage_bps,
                take_profit,
                stop_loss,
                pnl_unit=pnl_unit,
            )
            metrics = trade_metrics(trades)
            rows.append(
                {
                    "strategy_rank": int(rank) + 1,
                    "year": int(year),
                    "direction": candidate.direction,
                    "horizon_bars": candidate.horizon_bars,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "trades": metrics["trades"],
                    "win_rate": metrics["win_rate"],
                    "avg_net_return": metrics["avg_net_return"],
                    "total_return": metrics["total_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "rule": candidate.rule,
                }
            )
    return pd.DataFrame(rows)


def summarize_filter_rejections(
    scored: pd.DataFrame,
    min_train_trades: int,
    min_test_trades: int,
    require_multitimeframe: bool,
    walk_forward: bool = False,
) -> dict[str, int]:
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
            "passes_wf_gate": int(scored["wf_passes_walk_forward"].astype(bool).sum()),
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


def _get_git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


_condition_cache_key = condition_cache_key
_with_threshold = with_threshold


class WalkForwardEngine(WindowConditionCache):
    """WindowConditionCache plus per-window simulation arrays for the 15m search."""

    def __init__(self, data: pd.DataFrame, windows: Sequence[tuple[slice, slice]], base_prefix: str = "tf_15m_"):
        super().__init__(data, windows)
        self.test_arrays = [SimArrays.from_dataframe(frame, base_prefix) for frame in self.test_frames]


def _base_row(candidate: StrategyCandidate, take_profit: float, stop_loss: float) -> dict[str, object]:
    return {
        "direction": candidate.direction,
        "horizon_bars": candidate.horizon_bars,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "timeframes": ",".join(candidate.timeframes),
        "timeframe_count": len(candidate.timeframes),
        "conditions": len(candidate.conditions),
        "rule": candidate.rule,
    }


_ZEROED_SPLIT_COLUMNS = {
    f"{split}_{metric}": 0.0
    for split in ("train", "test")
    for metric in (
        "trades", "win_rate", "avg_net_return", "median_net_return", "total_return",
        "max_drawdown", "profit_factor", "return_skew", "return_kurt", "sharpe",
    )
}


def _score_candidate_walk_forward(
    engine: WalkForwardEngine,
    candidate: StrategyCandidate,
    scenarios: Sequence[tuple[float, float]],
    fee_bps: float,
    slippage_bps: float,
    pnl_unit: str,
    wf_pass_rate: float,
) -> list[dict[str, object]]:
    n_windows = len(engine.windows)
    per_scenario_stats: list[list[dict[str, float]]] = [[] for _ in scenarios]
    for window_index in range(n_windows):
        mask = engine.candidate_test_mask(window_index, candidate)
        arrays = engine.test_arrays[window_index]
        for scenario_index, (take_profit, stop_loss) in enumerate(scenarios):
            returns = simulate_net_returns(
                arrays, mask, candidate.direction, candidate.horizon_bars,
                fee_bps, slippage_bps, take_profit, stop_loss, pnl_unit,
            )
            metrics = returns_metrics(returns)
            per_scenario_stats[scenario_index].append(
                {
                    "test_total_return": metrics["total_return"],
                    "test_avg_net_return": metrics["avg_net_return"] if metrics["trades"] > 0 else 0.0,
                    "test_profit_factor": metrics["profit_factor"] if np.isfinite(metrics["profit_factor"]) else 0.0,
                    "test_max_drawdown": metrics["max_drawdown"],
                    "test_trades": metrics["trades"],
                }
            )
    rows = []
    for scenario_index, (take_profit, stop_loss) in enumerate(scenarios):
        stats = per_scenario_stats[scenario_index]
        summary = aggregate_walk_forward_results(stats, wf_pass_rate)
        window_returns = np.array([s["test_avg_net_return"] for s in stats], dtype=float)
        returns_series = pd.Series(window_returns)
        ci_low, ci_high = bootstrap_sharpe_ci(window_returns, n_boot=200)
        row = _base_row(candidate, take_profit, stop_loss)
        row.update(_ZEROED_SPLIT_COLUMNS)
        row.update({f"wf_{key}": value for key, value in summary.items()})
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


def _score_candidate_single_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_arrays: SimArrays,
    test_arrays: SimArrays,
    candidate: StrategyCandidate,
    scenarios: Sequence[tuple[float, float]],
    trade_config_template: TradeConfig,
) -> list[dict[str, object]]:
    rows = []
    if trade_config_template.use_triple_barrier_labels:
        for take_profit, stop_loss in scenarios:
            row = score_candidate(
                train, test, candidate,
                trade_config_template.fee_bps, trade_config_template.slippage_bps,
                take_profit, stop_loss,
                pnl_unit=trade_config_template.pnl_unit,
                use_triple_barrier_labels=True,
            )
            test_returns = np.asarray(row.pop("test_returns"), dtype=float)
            row.pop("train_returns")
            ci_low, ci_high = bootstrap_sharpe_ci(test_returns, n_boot=200)
            row["test_sharpe_ci_low"] = ci_low
            row["test_sharpe_ci_high"] = ci_high
            rows.append(row)
        return rows
    train_mask = combined_mask(train, candidate.conditions).to_numpy()
    test_mask = combined_mask(test, candidate.conditions).to_numpy()
    for take_profit, stop_loss in scenarios:
        train_returns = simulate_net_returns(
            train_arrays, train_mask, candidate.direction, candidate.horizon_bars,
            trade_config_template.fee_bps, trade_config_template.slippage_bps,
            take_profit, stop_loss, trade_config_template.pnl_unit,
        )
        test_returns = simulate_net_returns(
            test_arrays, test_mask, candidate.direction, candidate.horizon_bars,
            trade_config_template.fee_bps, trade_config_template.slippage_bps,
            take_profit, stop_loss, trade_config_template.pnl_unit,
        )
        ci_low, ci_high = bootstrap_sharpe_ci(test_returns, n_boot=200)
        row = _base_row(candidate, take_profit, stop_loss)
        row.update({f"train_{key}": value for key, value in returns_metrics(train_returns).items()})
        row.update({f"test_{key}": value for key, value in returns_metrics(test_returns).items()})
        row["test_sharpe_ci_low"] = ci_low
        row["test_sharpe_ci_high"] = ci_high
        rows.append(row)
    return rows


def _config_hash(config: dict[str, object]) -> str:
    import hashlib

    runtime_only = ("git_sha", "search_timestamp", "n_jobs", "checkpoint_every", "resume")
    relevant = {
        key: value
        for key, value in config.items()
        if key not in runtime_only
    }
    payload = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint(
    checkpoint_path: Path,
    meta_path: Path,
    config_hash: str,
    n_scenarios: int,
) -> tuple[list[dict[str, object]], set[int]]:
    if not checkpoint_path.exists() or not meta_path.exists():
        return [], set()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("config_hash") != config_hash:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} was produced with different search "
            "parameters. Delete it (and its meta file) or rerun with the original arguments."
        )
    frame = pd.read_csv(checkpoint_path)
    if frame.empty or "candidate_index" not in frame.columns:
        return [], set()
    counts = frame["candidate_index"].value_counts()
    complete = set(counts[counts == n_scenarios].index.astype(int))
    frame = frame[frame["candidate_index"].isin(complete)]
    LOGGER.info("Resuming from checkpoint: %s candidates already scored", len(complete))
    return frame.to_dict("records"), complete


_WORKER: dict[str, object] = {}


def _worker_init(payload: dict[str, object]) -> None:
    data = load_dataset(
        Path(payload["input_path"]), payload["horizons"], columns=payload.get("columns"),
    )
    _WORKER.update(payload)
    _WORKER["data"] = data
    if payload["walk_forward"]:
        wf_config = WalkForwardConfig(**payload["wf_config"])
        core = data.iloc[: payload["core_rows"]]
        windows = generate_windows(len(core), wf_config)
        _WORKER["engine"] = WalkForwardEngine(core, windows)
    else:
        train, test = split_train_test(data, payload["train_fraction"])
        _WORKER["train"] = train
        _WORKER["test"] = test
        _WORKER["train_arrays"] = SimArrays.from_dataframe(train)
        _WORKER["test_arrays"] = SimArrays.from_dataframe(test)


def _score_chunk(indices: Sequence[int]) -> list[dict[str, object]]:
    candidates: list[StrategyCandidate] = _WORKER["candidates"]
    scenarios: list[tuple[float, float]] = _WORKER["scenarios"]
    trade_config = TradeConfig(**_WORKER["trade_config"])
    rows: list[dict[str, object]] = []
    for index in indices:
        candidate = candidates[index]
        if _WORKER["walk_forward"]:
            candidate_rows = _score_candidate_walk_forward(
                _WORKER["engine"], candidate, scenarios,
                trade_config.fee_bps, trade_config.slippage_bps,
                trade_config.pnl_unit, _WORKER["wf_pass_rate"],
            )
        else:
            candidate_rows = _score_candidate_single_split(
                _WORKER["train"], _WORKER["test"],
                _WORKER["train_arrays"], _WORKER["test_arrays"],
                candidate, scenarios, trade_config,
            )
            if _WORKER.get("regime_conditional"):
                for row in candidate_rows:
                    row["regime_breakdown_json"] = json.dumps(
                        regime_breakdown(
                            _WORKER["data"], candidate,
                            _WORKER["fee_bps"], _WORKER["slippage_bps"],
                            float(row["take_profit"]), float(row["stop_loss"]),
                            pnl_unit=str(_WORKER["pnl_unit"]),
                        ),
                        sort_keys=True,
                    )
        conditions_json = _conditions_payload(candidate)
        for row in candidate_rows:
            row["candidate_index"] = index
            row["conditions_json"] = conditions_json
        rows.extend(candidate_rows)
    return rows


def _flush_rows(rows: list[dict[str, object]], checkpoint_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(
        checkpoint_path,
        mode="a",
        header=not checkpoint_path.exists(),
        index=False,
    )


def _evaluate_holdout(
    holdout: pd.DataFrame,
    refit_frame: pd.DataFrame,
    strategies: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    pnl_unit: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """Score top strategies on the untouched holdout segment. Report-only:
    these columns must never participate in ranking or filtering."""
    from src.walk_forward import _compute_threshold

    out = strategies.copy()
    for column in ("holdout_trades", "holdout_total_return", "holdout_win_rate", "holdout_avg_net_return"):
        out[column] = np.nan
    if out.empty or holdout.empty:
        return out
    arrays = SimArrays.from_dataframe(holdout)
    for idx, row in out.head(top_n).iterrows():
        conditions = [Condition(**payload) for payload in json.loads(row["conditions_json"])]
        refit = []
        for condition in conditions:
            try:
                threshold = _compute_threshold(refit_frame, condition)
            except ValueError:
                threshold = condition.threshold
            refit.append(_with_threshold(condition, threshold))
        mask = combined_mask(holdout, refit).to_numpy()
        returns = simulate_net_returns(
            arrays, mask, row["direction"], int(row["horizon_bars"]),
            fee_bps, slippage_bps, float(row["take_profit"]), float(row["stop_loss"]), pnl_unit,
        )
        metrics = returns_metrics(returns)
        out.loc[idx, "holdout_trades"] = metrics["trades"]
        out.loc[idx, "holdout_total_return"] = metrics["total_return"]
        out.loc[idx, "holdout_win_rate"] = metrics["win_rate"]
        out.loc[idx, "holdout_avg_net_return"] = metrics["avg_net_return"]
    return out


def write_report(
    strategies: pd.DataFrame,
    yearly: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rejection_summary: dict[str, int],
    path: Path,
    mode: str = "usdt",
    pnl_unit: str = "usdt",
    shap_screen: bool = False,
    walk_forward: bool = False,
) -> None:
    btc_mode = mode == "btc-position" or pnl_unit == "btc"
    if btc_mode:
        title = "# BTC Position Trading Report"
        subtitle = (
            "BTC-denominated strategy: hold BTC, swap to USDT on short signals to dodge pullbacks. "
            "Returns represent extra BTC accumulated above buy-and-hold."
        )
    else:
        title = "# Strategy Search Report"
        subtitle = "Ranks long and short indicator/timeframe patterns by fee-adjusted test trades."
    lines = [
        title,
        "",
        subtitle,
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
    if btc_mode and not strategies.empty:
        best = strategies.iloc[0]
        if walk_forward:
            lines.extend([
                "## BTC Accumulation Summary (Best Strategy)",
                "",
                f"- Walk-forward window pass rate: **{float(best.get('wf_pass_rate', np.nan)):.0%}**",
                f"- Avg extra BTC per trade (per-window expectancy): **{float(best.get('wf_expectancy', np.nan)):.4%}**",
                f"- Avg trades per window: {float(best.get('wf_avg_trades', np.nan)):.0f}",
            ])
            if "holdout_total_return" in strategies.columns and pd.notna(best.get("holdout_total_return")):
                lines.append(
                    f"- Holdout extra BTC accumulated (untouched data): **{float(best['holdout_total_return']):.2%}**"
                )
            lines.append("")
        else:
            total_trades = int(best["train_trades"] + best["test_trades"])
            total_return = best["train_total_return"] + best["test_total_return"]
            lines.extend([
                "## BTC Accumulation Summary (Best Strategy)",
                "",
                f"- Extra BTC accumulated (train+test): **{total_return:.2%}**",
                f"- Total trades: {total_trades}",
                f"- Trades per year: ~{total_trades / 6:.0f}",
                "",
            ])
    lines.extend(["## Top Passing Strategies", ""])
    if strategies.empty:
        lines.append("No strategies passed the filters.")
    else:
        if walk_forward:
            display = strategies
            columns = [
                "direction", "horizon_bars", "take_profit", "stop_loss",
                "timeframes", "wf_pass_rate", "wf_expectancy",
                "wf_profit_factor_median", "wf_max_drawdown_worst",
                "wf_avg_trades", "wf_windows", "dsr",
                "holdout_total_return", "holdout_trades", "rule",
            ]
        elif btc_mode:
            display = strategies.copy()
            display = display.rename(columns={
                "train_total_return": "train_btc_accumulated",
                "test_total_return": "test_btc_accumulated",
                "test_avg_net_return": "test_avg_btc_gain",
            })
            columns = [
                "horizon_bars",
                "take_profit",
                "stop_loss",
                "timeframes",
                "train_trades",
                "train_btc_accumulated",
                "test_trades",
                "test_btc_accumulated",
                "test_win_rate",
                "test_avg_btc_gain",
                "test_max_drawdown",
                "dsr",
                "rule",
            ]
        else:
            display = strategies
            columns = [
                "direction",
                "horizon_bars",
                "take_profit",
                "stop_loss",
                "timeframes",
                "train_trades",
                "train_total_return",
                "test_trades",
                "test_total_return",
                "test_win_rate",
                "test_avg_net_return",
                "test_max_drawdown",
                "dsr",
                "rule",
            ]
        available = [column for column in columns if column in display.columns]
        lines.append("```text")
        lines.append(display[available].head(30).to_string(index=False))
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
                "wf_avg_trades", "dsr", "passes_filters", "rule",
            ]
        else:
            columns = [
                "direction",
                "horizon_bars",
                "take_profit",
                "stop_loss",
                "timeframes",
                "train_trades",
                "train_total_return",
                "test_trades",
                "test_total_return",
                "test_win_rate",
                "test_avg_net_return",
                "test_max_drawdown",
                "passes_filters",
                "rule",
            ]
        available = [column for column in columns if column in diagnostics.columns]
        lines.append("```text")
        lines.append(diagnostics[available].head(30).to_string(index=False))
        lines.append("```")
    if not yearly.empty:
        lines.extend(["", "## Year Breakdown", "", "```text"])
        lines.append(
            yearly[
                [
                    "strategy_rank",
                    "year",
                    "direction",
                    "horizon_bars",
                    "take_profit",
                    "stop_loss",
                    "trades",
                    "total_return",
                    "win_rate",
                    "avg_net_return",
                    "max_drawdown",
                ]
            ].to_string(index=False)
        )
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: Sequence[int] = (4, 8, 16),
    train_fraction: float = 0.7,
    max_features: int = 60,
    top_conditions: int = 60,
    max_pairs: int = 2500,
    max_triples: int = 2500,
    rank_sample_rows: int = 50_000,
    condition_depths: Sequence[int] = (1, 2, 3),
    min_train_trades: int = 50,
    min_test_trades: int = 25,
    fee_bps: float = 5.0,
    slippage_bps: float = 1.0,
    take_profits: Sequence[float] = (0.006,),
    stop_losses: Sequence[float] = (0.004,),
    require_multitimeframe: bool = False,
    ranking_method: str = "spearman",
    cross_tf_mode: str = "none",
    enabled_kinds: set[str] = DEFAULT_ENABLED_KINDS,
    mode: str = "usdt",
    pnl_unit: str = "usdt",
    shap_screen: bool = False,
    shap_target: str = "sign",
    regime_conditional: bool = False,
    cluster_jaccard: float = 0.8,
    walk_forward: bool = False,
    purged_kfold: bool = False,
    wf_train_bars: int = 70_080,
    wf_test_bars: int = 17_520,
    wf_step_bars: int = 17_520,
    wf_min_windows: int = 3,
    wf_pass_rate: float = 0.6,
    embargo_bars: int | None = None,
    purged_kfold_splits: int = 5,
    use_triple_barrier_labels: bool = False,
    dsr_threshold: float = 0.0,
    holdout_fraction: float = 0.2,
    n_jobs: int = 1,
    checkpoint_every: int = 25,
    resume: bool = False,
) -> pd.DataFrame:
    data = load_dataset(input_path, horizons)
    if regime_conditional and "tf_1d_regime_id" not in data.columns:
        LOGGER.warning("--regime-conditional requested but tf_1d_regime_id is missing; skipping regime breakdown")
        regime_conditional = False
    if embargo_bars is None:
        embargo_bars = max(horizons)
    directions = ("short",) if mode == "btc-position" else ("long", "short")
    scenarios = list(itertools.product(take_profits, stop_losses))
    wf_engine_mode = walk_forward and not purged_kfold

    holdout = data.iloc[0:0]
    core = data
    windows: list[tuple[slice, slice]] = []
    wf_config: WalkForwardConfig | None = None
    train = test = None
    if wf_engine_mode:
        core_rows = len(data) - int(len(data) * holdout_fraction) if holdout_fraction > 0 else len(data)
        core = data.iloc[:core_rows]
        holdout = data.iloc[core_rows:]
        wf_config = WalkForwardConfig(
            train_bars=wf_train_bars,
            test_bars=wf_test_bars,
            step_bars=wf_step_bars,
            min_windows=wf_min_windows,
            pass_rate=wf_pass_rate,
            embargo_bars=embargo_bars,
        )
        windows = generate_windows(len(core), wf_config)
        # Candidates come from the FIRST train window only, so no walk-forward
        # test window (nor the holdout) ever influences candidate selection.
        candidate_frame = core.iloc[windows[0][0]]
        LOGGER.info(
            "Walk-forward: %s windows over %s core rows, %s holdout rows reserved",
            len(windows), len(core), len(holdout),
        )
    else:
        train, test = split_train_test(data, train_fraction)
        candidate_frame = train

    candidates = make_candidates(
        candidate_frame,
        horizons,
        directions=directions,
        max_features=max_features,
        top_conditions=top_conditions,
        max_pairs=max_pairs,
        max_triples=max_triples,
        rank_sample_rows=rank_sample_rows,
        condition_depths=condition_depths,
        ranking_method=ranking_method,
        cross_tf_mode=cross_tf_mode,
        enabled_kinds=enabled_kinds,
        shap_screen=shap_screen,
        shap_target=shap_target,
    )
    if use_triple_barrier_labels:
        missing_labels = []
        for direction in directions:
            for horizon in horizons:
                for take_profit, stop_loss in scenarios:
                    suffix = f"{direction}_tp{int(round(take_profit * 10_000))}_sl{int(round(stop_loss * 10_000))}_h{horizon}"
                    if f"label_{suffix}" not in data.columns or f"bars_to_exit_{suffix}" not in data.columns:
                        missing_labels.append(suffix)
        if missing_labels:
            LOGGER.warning(
                "--use-triple-barrier-labels requested but %s label scenarios are missing; those scenarios fall back to simulation",
                len(missing_labels),
            )

    # Scoring only touches the columns the candidates reference (plus OHLC).
    # The full table is thousands of columns; loading it once per worker
    # process exhausts RAM, so workers get a pruned column list and, in
    # walk-forward mode, the parent drops the unused columns too.
    extra_columns: set[str] = set()
    if use_triple_barrier_labels:
        extra_columns |= {c for c in data.columns if c.startswith(("label_", "bars_to_exit_"))}
    if regime_conditional and "tf_1d_regime_id" in data.columns:
        extra_columns.add("tf_1d_regime_id")
    worker_columns = sorted(
        (candidate_feature_columns(candidates) & set(data.columns)) | extra_columns
    )
    if wf_engine_mode:
        n_core = len(core)
        keep = [c for c in data.columns if c in set(worker_columns) | set(BASE_COLUMNS)]
        data = data[keep]
        core = data.iloc[:n_core]
        holdout = data.iloc[n_core:]

    config = {
        "git_sha": _get_git_sha(),
        "search_timestamp": pd.Timestamp.now("UTC").isoformat(),
        "input_path": str(input_path),
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
        "require_multitimeframe": require_multitimeframe,
        "ranking_method": ranking_method,
        "cross_tf_mode": cross_tf_mode,
        "enabled_kinds": sorted(enabled_kinds),
        "mode": mode,
        "pnl_unit": pnl_unit,
        "shap_screen": shap_screen,
        "shap_target": shap_target,
        "regime_conditional": regime_conditional,
        "cluster_jaccard": cluster_jaccard,
        "walk_forward": walk_forward,
        "purged_kfold": purged_kfold,
        "wf_train_bars": wf_train_bars,
        "wf_test_bars": wf_test_bars,
        "wf_step_bars": wf_step_bars,
        "wf_min_windows": wf_min_windows,
        "wf_pass_rate": wf_pass_rate,
        "embargo_bars": embargo_bars,
        "use_triple_barrier_labels": use_triple_barrier_labels,
        "dsr_threshold": dsr_threshold,
        "holdout_fraction": holdout_fraction if wf_engine_mode else 0.0,
        "n_jobs": n_jobs,
        "checkpoint_every": checkpoint_every,
        "resume": resume,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.csv"
    meta_path = output_dir / "checkpoint_meta.json"
    config_hash = _config_hash(config)
    rows: list[dict[str, object]] = []
    done: set[int] = set()
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
    LOGGER.info(
        "Scoring %s strategy candidates across %s TP/SL scenarios (%s pending, n_jobs=%s)",
        len(candidates), len(scenarios), len(pending), n_jobs,
    )

    trade_config_payload = {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "take_profit": scenarios[0][0],
        "stop_loss": scenarios[0][1],
        "pnl_unit": pnl_unit,
        "use_triple_barrier_labels": use_triple_barrier_labels,
    }

    if purged_kfold:
        scored = 0
        buffer: list[dict[str, object]] = []
        for index in pending:
            candidate = candidates[index]
            conditions_json = _conditions_payload(candidate)
            kfold_windows = generate_purged_kfold_windows(
                len(data), purged_kfold_splits, candidate.horizon_bars, embargo_bars,
            )
            for take_profit, stop_loss in scenarios:
                trade_config = TradeConfig(
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    pnl_unit=pnl_unit,
                    use_triple_barrier_labels=use_triple_barrier_labels,
                )
                row = score_candidate_with_config(train, test, candidate, trade_config)
                test_returns = np.asarray(row.pop("test_returns"), dtype=float)
                row.pop("train_returns")
                ci_low, ci_high = bootstrap_sharpe_ci(test_returns, n_boot=200)
                row["test_sharpe_ci_low"] = ci_low
                row["test_sharpe_ci_high"] = ci_high
                window_results = []
                for train_index, test_index in kfold_windows:
                    fold_row = score_candidate_with_config(
                        data.iloc[train_index].copy(),
                        data.iloc[test_index].copy(),
                        candidate,
                        trade_config,
                    )
                    fold_row.pop("train_returns")
                    fold_row.pop("test_returns")
                    window_results.append(fold_row)
                wf_summary = aggregate_walk_forward_results(window_results, wf_pass_rate)
                row.update({f"wf_{key}": value for key, value in wf_summary.items()})
                row["wf_window_returns_json"] = json.dumps(
                    [round(float(result.get("test_avg_net_return", 0.0) or 0.0), 8) for result in window_results]
                )
                if regime_conditional:
                    row["regime_breakdown_json"] = json.dumps(
                        regime_breakdown(
                            data, candidate, fee_bps, slippage_bps,
                            take_profit, stop_loss, pnl_unit=pnl_unit,
                        ),
                        sort_keys=True,
                    )
                row["conditions_json"] = conditions_json
                row["candidate_index"] = index
                buffer.append(row)
            scored += 1
            if scored % checkpoint_every == 0 or scored == len(pending):
                _flush_rows(buffer, checkpoint_path)
                rows.extend(buffer)
                buffer = []
                LOGGER.info("Scored candidate %s/%s", scored + len(done), len(candidates))
    else:
        worker_payload = {
            "input_path": str(input_path),
            "horizons": list(horizons),
            "walk_forward": wf_engine_mode,
            "train_fraction": train_fraction,
            "columns": worker_columns,
            "candidates": candidates,
            "scenarios": scenarios,
            "trade_config": trade_config_payload,
            "wf_pass_rate": wf_pass_rate,
            "regime_conditional": regime_conditional,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "pnl_unit": pnl_unit,
        }
        if wf_engine_mode:
            worker_payload["core_rows"] = len(core)
            worker_payload["wf_config"] = {
                "train_bars": wf_train_bars,
                "test_bars": wf_test_bars,
                "step_bars": wf_step_bars,
                "min_windows": wf_min_windows,
                "pass_rate": wf_pass_rate,
                "embargo_bars": embargo_bars,
            }
        chunks = [
            pending[start: start + checkpoint_every]
            for start in range(0, len(pending), checkpoint_every)
        ]
        scored = len(done)
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_worker_init, initargs=(worker_payload,),
            ) as pool:
                futures = [pool.submit(_score_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    chunk_rows = future.result()
                    _flush_rows(chunk_rows, checkpoint_path)
                    rows.extend(chunk_rows)
                    scored += len({row["candidate_index"] for row in chunk_rows})
                    LOGGER.info("Scored candidate %s/%s", scored, len(candidates))
        else:
            _WORKER.clear()
            _WORKER.update(worker_payload)
            _WORKER["data"] = data
            if wf_engine_mode:
                _WORKER["engine"] = WalkForwardEngine(core, windows)
            else:
                _WORKER["train"] = train
                _WORKER["test"] = test
                _WORKER["train_arrays"] = SimArrays.from_dataframe(train)
                _WORKER["test_arrays"] = SimArrays.from_dataframe(test)
            for chunk in chunks:
                chunk_rows = _score_chunk(chunk)
                _flush_rows(chunk_rows, checkpoint_path)
                rows.extend(chunk_rows)
                scored += len(chunk)
                LOGGER.info("Scored candidate %s/%s", scored, len(candidates))

    strategies = pd.DataFrame(rows)
    strategies = _attach_statistical_metrics(strategies, walk_forward=wf_engine_mode)
    if not strategies.empty and "wf_window_returns_json" in strategies.columns:
        matrix = [json.loads(str(payload)) for payload in strategies["wf_window_returns_json"]]
        lengths = {len(item) for item in matrix}
        if len(lengths) == 1:
            strategies["pool_pbo"] = probability_backtest_overfitting(np.asarray(matrix, dtype=float))
        else:
            strategies["pool_pbo"] = np.nan
    else:
        strategies["pool_pbo"] = np.nan
    diagnostics = strategies.copy()
    if wf_engine_mode:
        sort_columns = ["dsr", "wf_pass_rate", "wf_expectancy", "wf_profit_factor_median", "wf_avg_trades"]
    else:
        # Rank on in-sample stats + DSR only; test metrics are reported, never
        # used for selection (selecting on them un-blinds the test split).
        sort_columns = ["dsr", "train_total_return", "train_avg_net_return", "train_trades"]
    if not diagnostics.empty:
        if wf_engine_mode:
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
        if "wf_passes_walk_forward" in diagnostics.columns:
            diagnostics["passes_walk_forward_filter"] = diagnostics["wf_passes_walk_forward"].astype(bool)
        else:
            diagnostics["passes_walk_forward_filter"] = True
        diagnostics["passes_filters"] = (
            diagnostics["passes_trade_count"]
            & diagnostics["passes_profitability"]
            & diagnostics["passes_multitimeframe"]
            & diagnostics["passes_dsr"]
            & diagnostics["passes_walk_forward_filter"]
        )
        diagnostics = diagnostics.sort_values(
            sort_columns, ascending=[False] * len(sort_columns),
        ).reset_index(drop=True)
        strategies = diagnostics[diagnostics["passes_filters"]].copy()
        if require_multitimeframe:
            strategies = strategies[strategies["timeframe_count"] >= 2].copy()
        strategies = strategies.sort_values(
            sort_columns, ascending=[False] * len(sort_columns),
        ).reset_index(drop=True)
    rejection_summary = summarize_filter_rejections(
        diagnostics,
        min_train_trades,
        min_test_trades,
        require_multitimeframe,
        walk_forward=wf_engine_mode,
    )

    if wf_engine_mode and not holdout.empty and not strategies.empty:
        refit_frame = core.iloc[windows[-1][0]]
        strategies = _evaluate_holdout(
            holdout, refit_frame, strategies, fee_bps, slippage_bps, pnl_unit,
        )

    if wf_engine_mode:
        yearly = pd.DataFrame()
    else:
        yearly = add_year_metrics(
            data,
            strategies,
            fee_bps,
            slippage_bps,
            top_n=10,
            pnl_unit=pnl_unit,
        )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    heavy_columns = ["wf_window_returns_json"]
    diagnostics.drop(columns=heavy_columns, errors="ignore").to_csv(
        output_dir / "scored_strategies_all.csv", index=False,
    )
    (output_dir / "filter_summary.json").write_text(
        json.dumps(rejection_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    strategies = strategies.drop(columns=heavy_columns, errors="ignore")
    strategies.to_csv(output_dir / "ranked_strategies.csv", index=False)
    clustered = cluster_ranked_strategies(data, strategies, threshold=cluster_jaccard)
    clustered.to_csv(output_dir / "ranked_strategies_clustered.csv", index=False)
    yearly.to_csv(output_dir / "ranked_strategies_by_year.csv", index=False)
    write_report(
        strategies,
        yearly,
        diagnostics,
        rejection_summary,
        output_dir / "report.md",
        mode=mode,
        pnl_unit=pnl_unit,
        walk_forward=wf_engine_mode,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    return strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search long and short multi-timeframe indicator strategies."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", action="append", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--top-conditions", type=int, default=60)
    parser.add_argument("--max-pairs", type=int, default=2500)
    parser.add_argument("--max-triples", type=int, default=2500)
    parser.add_argument("--rank-sample-rows", type=int, default=50_000)
    parser.add_argument(
        "--condition-depth",
        action="append",
        type=int,
        choices=(1, 2, 3),
        help="Pattern depth to include. Repeat for multiple depths. Defaults to 1, 2, and 3.",
    )
    parser.add_argument("--min-train-trades", type=int, default=50)
    parser.add_argument("--min-test-trades", type=int, default=25)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument(
        "--take-profit",
        action="append",
        type=float,
        help="Take-profit return. Repeat to search multiple values.",
    )
    parser.add_argument(
        "--stop-loss",
        action="append",
        type=float,
        help="Stop-loss return. Repeat to search multiple values.",
    )
    parser.add_argument(
        "--require-multitimeframe",
        action="store_true",
        help="Only keep strategies whose conditions use at least two timeframes.",
    )
    parser.add_argument(
        "--ranking-method",
        choices=("spearman", "importance", "blended"),
        default="spearman",
        help="Feature ranking method. 'importance' and 'blended' use LightGBM.",
    )
    parser.add_argument(
        "--cross-tf-mode",
        choices=("none", "pool", "shap"),
        default="none",
        help="Cross-timeframe pair generation mode.",
    )
    parser.add_argument(
        "--enabled-kinds",
        nargs="+",
        default=None,
        help="Condition kinds to enable. Defaults to all.",
    )
    parser.add_argument(
        "--mode",
        choices=("usdt", "btc-position"),
        default="usdt",
        help="Trading mode. 'btc-position' treats BTC as base currency and searches short-only signals.",
    )
    parser.add_argument(
        "--pnl-unit",
        choices=("usdt", "btc"),
        default=None,
        help="PnL accounting unit. Defaults to btc for btc-position mode, otherwise usdt.",
    )
    parser.add_argument("--shap-screen", action="store_true")
    parser.add_argument("--shap-target", choices=("sign", "triple-barrier"), default="sign")
    parser.add_argument("--regime-conditional", action="store_true")
    parser.add_argument("--cluster-jaccard", type=float, default=0.8)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--purged-kfold", action="store_true")
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument(
        "--wf-train-bars", type=int, default=70_080,
        help="Walk-forward train window in base bars (default: 2 years of 15m bars).",
    )
    parser.add_argument("--wf-test-bars", type=int, default=17_520)
    parser.add_argument("--wf-step-bars", type=int, default=17_520)
    parser.add_argument("--wf-min-windows", type=int, default=3)
    parser.add_argument("--wf-pass-rate", type=float, default=0.6)
    parser.add_argument(
        "--embargo-bars", type=int, default=None,
        help="Bars purged between train and test windows. Defaults to max(horizons).",
    )
    parser.add_argument("--purged-kfold-splits", type=int, default=5)
    parser.add_argument("--use-triple-barrier-labels", action="store_true")
    parser.add_argument("--dsr-threshold", type=float, default=0.0)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.2,
        help="Final fraction of data excluded from walk-forward; scored report-only.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker processes for candidate scoring. 1 = sequential.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Flush scored candidates to the checkpoint file every N candidates.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint in --output-dir (same arguments required).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    horizons = tuple(args.horizon) if args.horizon else (4, 8, 16)
    condition_depths = tuple(args.condition_depth) if args.condition_depth else (1, 2, 3)
    take_profits = tuple(args.take_profit) if args.take_profit else (0.003, 0.005, 0.008, 0.012)
    stop_losses = tuple(args.stop_loss) if args.stop_loss else (0.002, 0.004, 0.006, 0.01)
    enabled_kinds = set(args.enabled_kinds) if args.enabled_kinds else DEFAULT_ENABLED_KINDS
    pnl_unit = args.pnl_unit or ("btc" if args.mode == "btc-position" else "usdt")
    strategies = run(
        input_path=args.input_path,
        output_dir=args.output_dir,
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
        mode=args.mode,
        pnl_unit=pnl_unit,
        shap_screen=args.shap_screen,
        shap_target=args.shap_target,
        regime_conditional=args.regime_conditional,
        cluster_jaccard=args.cluster_jaccard,
        walk_forward=args.walk_forward and not args.single_split,
        purged_kfold=args.purged_kfold and not args.single_split,
        wf_train_bars=args.wf_train_bars,
        wf_test_bars=args.wf_test_bars,
        wf_step_bars=args.wf_step_bars,
        wf_min_windows=args.wf_min_windows,
        wf_pass_rate=args.wf_pass_rate,
        embargo_bars=args.embargo_bars,
        purged_kfold_splits=args.purged_kfold_splits,
        use_triple_barrier_labels=args.use_triple_barrier_labels,
        dsr_threshold=args.dsr_threshold,
        holdout_fraction=args.holdout_fraction,
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    print(f"Wrote {args.output_dir / 'ranked_strategies.csv'}")
    print(f"Wrote {args.output_dir / 'ranked_strategies_by_year.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    print(f"Strategies passing filters: {len(strategies)}")


if __name__ == "__main__":
    main()
