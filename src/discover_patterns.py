import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.build_dataset import TARGET_COLUMNS
from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR
from src.load_data import configure_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "train_15m_indicators.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pattern_discovery"
DEFAULT_HORIZON_BARS = 4
DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_MIN_TEST_SUPPORT = 100


@dataclass(frozen=True)
class Condition:
    feature: str
    kind: str
    threshold: float
    description: str
    feature_b: Optional[str] = None
    threshold_source: Optional[str] = None
    quantile: Optional[float] = None
    lookback: Optional[int] = None
    cross_feature: Optional[str] = None

    def signature(self) -> Tuple[str, str, float, Optional[str]]:
        return (self.feature, self.kind, round(float(self.threshold), 6), self.feature_b)


def target_column_for_horizon(horizon_bars: int) -> str:
    return f"future_return_{horizon_bars}_bars"


def load_dataset(path: Path, horizon_bars: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    data = pd.read_parquet(path)
    required = {"timestamp", "tf_15m_close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.sort_values("timestamp").reset_index(drop=True)
    target_column = target_column_for_horizon(horizon_bars)
    data[target_column] = data["tf_15m_close"].shift(-horizon_bars) / data["tf_15m_close"] - 1
    data = data.dropna(subset=[target_column]).reset_index(drop=True)
    return data


def numeric_feature_columns(data: pd.DataFrame) -> List[str]:
    excluded = {"timestamp", "tf_15m_close"} | set(TARGET_COLUMNS)
    excluded.update(column for column in data.columns if column.startswith("future_return_"))
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in excluded
    ]


def split_train_test(data: pd.DataFrame, train_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = int(len(data) * train_fraction)
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows for the requested train/test split")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def rank_features(
    train: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    max_features: int,
) -> List[str]:
    target = train[target_column]
    scored = []
    for column in feature_columns:
        series = train[column]
        valid = series.notna() & target.notna()
        if valid.sum() < 1000 or series[valid].nunique(dropna=True) <= 1:
            continue
        corr = series[valid].corr(target[valid], method="spearman")
        if pd.notna(corr):
            scored.append((column, abs(float(corr)), float(corr)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [column for column, _, _ in scored[:max_features]]


def _finite_quantiles(series: pd.Series, quantiles: Sequence[float]) -> Dict[float, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {}
    values = clean.quantile(list(quantiles)).to_dict()
    return {float(key): float(value) for key, value in values.items() if pd.notna(value)}


def build_conditions(train: pd.DataFrame, features: Sequence[str]) -> List[Condition]:
    conditions: List[Condition] = []
    for feature in features:
        quantiles = _finite_quantiles(train[feature], [0.1, 0.2, 0.8, 0.9])
        for quantile, threshold in quantiles.items():
            if quantile in {0.1, 0.2}:
                conditions.append(
                    Condition(
                        feature=feature,
                        kind="value_le",
                        threshold=threshold,
                        description=f"{feature} <= train q{int(quantile * 100)} ({threshold:.6g})",
                        threshold_source="quantile",
                        quantile=float(quantile),
                    )
                )
            else:
                conditions.append(
                    Condition(
                        feature=feature,
                        kind="value_ge",
                        threshold=threshold,
                        description=f"{feature} >= train q{int(quantile * 100)} ({threshold:.6g})",
                        threshold_source="quantile",
                        quantile=float(quantile),
                    )
                )

        delta = train[feature].diff()
        delta_quantiles = _finite_quantiles(delta, [0.1, 0.9])
        if 0.1 in delta_quantiles:
            threshold = delta_quantiles[0.1]
            conditions.append(
                Condition(
                    feature=feature,
                    kind="delta_le",
                    threshold=threshold,
                    description=f"{feature} falling fast: 1-bar change <= train q10 ({threshold:.6g})",
                    threshold_source="delta_quantile",
                    quantile=0.1,
                )
            )
        if 0.9 in delta_quantiles:
            threshold = delta_quantiles[0.9]
            conditions.append(
                Condition(
                    feature=feature,
                    kind="delta_ge",
                    threshold=threshold,
                    description=f"{feature} rising fast: 1-bar change >= train q90 ({threshold:.6g})",
                    threshold_source="delta_quantile",
                    quantile=0.9,
                )
            )

    return conditions


DEFAULT_ENABLED_KINDS: Set[str] = {
    "value",
    "delta",
    "slope",
    "cross",
    "ratio",
    "divergence",
}


def build_slope_conditions(
    train: pd.DataFrame, feature: str, windows: Sequence[int] = (3, 5, 10)
) -> List[Condition]:
    conditions: List[Condition] = []
    for window in windows:
        slope = (train[feature] - train[feature].shift(window)) / window
        for quantile, threshold in _finite_quantiles(slope, [0.1, 0.9]).items():
            if quantile < 0.5:
                conditions.append(
                    Condition(
                        feature=feature,
                        kind=f"slope_{window}_le",
                        threshold=threshold,
                        description=f"{feature} slope({window}) <= q10 ({threshold:.6g})",
                        threshold_source="slope_quantile",
                        quantile=float(quantile),
                        lookback=int(window),
                    )
                )
            else:
                conditions.append(
                    Condition(
                        feature=feature,
                        kind=f"slope_{window}_ge",
                        threshold=threshold,
                        description=f"{feature} slope({window}) >= q90 ({threshold:.6g})",
                        threshold_source="slope_quantile",
                        quantile=float(quantile),
                        lookback=int(window),
                    )
                )
    return conditions


def build_cross_conditions(
    train: pd.DataFrame, feature_pairs: Sequence[Tuple[str, str]]
) -> List[Condition]:
    conditions: List[Condition] = []
    for feature_a, feature_b in feature_pairs:
        if feature_a not in train.columns or feature_b not in train.columns:
            continue
        conditions.append(
            Condition(
                feature=feature_a,
                kind="cross_above",
                threshold=0.0,
                description=f"{feature_a} crosses above {feature_b}",
                feature_b=feature_b,
                threshold_source="fixed",
            )
        )
        conditions.append(
            Condition(
                feature=feature_a,
                kind="cross_below",
                threshold=0.0,
                description=f"{feature_a} crosses below {feature_b}",
                feature_b=feature_b,
                threshold_source="fixed",
            )
        )
    return conditions


def build_ratio_conditions(
    train: pd.DataFrame, feature_pairs: Sequence[Tuple[str, str]]
) -> List[Condition]:
    conditions: List[Condition] = []
    for feature_a, feature_b in feature_pairs:
        if feature_a not in train.columns or feature_b not in train.columns:
            continue
        series_b = train[feature_b].replace(0, np.nan)
        ratio = train[feature_a] / series_b
        for quantile, threshold in _finite_quantiles(ratio, [0.1, 0.2, 0.8, 0.9]).items():
            if quantile < 0.5:
                conditions.append(
                    Condition(
                        feature=feature_a,
                        kind="ratio_le",
                        threshold=threshold,
                        description=f"{feature_a}/{feature_b} <= q{int(quantile * 100)} ({threshold:.6g})",
                        feature_b=feature_b,
                        threshold_source="ratio_quantile",
                        quantile=float(quantile),
                    )
                )
            else:
                conditions.append(
                    Condition(
                        feature=feature_a,
                        kind="ratio_ge",
                        threshold=threshold,
                        description=f"{feature_a}/{feature_b} >= q{int(quantile * 100)} ({threshold:.6g})",
                        feature_b=feature_b,
                        threshold_source="ratio_quantile",
                        quantile=float(quantile),
                    )
                )
    return conditions


def build_divergence_conditions(
    train: pd.DataFrame,
    features: Sequence[str],
    windows: Sequence[int] = (10, 20),
) -> List[Condition]:
    conditions: List[Condition] = []
    if "tf_15m_close" not in train.columns:
        return conditions
    for feature in features:
        for window in windows:
            conditions.append(
                Condition(
                    feature=feature,
                    kind=f"divergence_bull_{window}",
                    threshold=0.0,
                    description=f"bullish divergence: price new low but {feature} higher low ({window} bars)",
                    threshold_source="fixed",
                )
            )
            conditions.append(
                Condition(
                    feature=feature,
                    kind=f"divergence_bear_{window}",
                    threshold=0.0,
                    description=f"bearish divergence: price new high but {feature} lower high ({window} bars)",
                    threshold_source="fixed",
                )
            )
    return conditions


def detect_cross_feature_pairs(features: Sequence[str]) -> List[Tuple[str, str]]:
    by_base: Dict[str, List[str]] = {}
    for feature in features:
        parts = feature.split("_", 2)
        if len(parts) >= 3 and parts[0] == "tf":
            base = parts[2]
            by_base.setdefault(base, []).append(feature)

    pairs: List[Tuple[str, str]] = []
    for base, members in by_base.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        for i, a in enumerate(members_sorted):
            for b in members_sorted[i + 1:]:
                pairs.append((a, b))
    return pairs


def build_all_conditions(
    train: pd.DataFrame,
    features: Sequence[str],
    enabled_kinds: Set[str] = DEFAULT_ENABLED_KINDS,
    cross_feature_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Condition]:
    conditions: List[Condition] = []

    for feature in features:
        if "value" in enabled_kinds:
            quantiles = _finite_quantiles(train[feature], [0.1, 0.2, 0.8, 0.9])
            for quantile, threshold in quantiles.items():
                if quantile < 0.5:
                    conditions.append(
                        Condition(
                            feature=feature,
                            kind="value_le",
                            threshold=threshold,
                            description=f"{feature} <= train q{int(quantile * 100)} ({threshold:.6g})",
                            threshold_source="quantile",
                            quantile=float(quantile),
                        )
                    )
                else:
                    conditions.append(
                        Condition(
                            feature=feature,
                            kind="value_ge",
                            threshold=threshold,
                            description=f"{feature} >= train q{int(quantile * 100)} ({threshold:.6g})",
                            threshold_source="quantile",
                            quantile=float(quantile),
                        )
                    )

        if "delta" in enabled_kinds:
            delta = train[feature].diff()
            delta_quantiles = _finite_quantiles(delta, [0.1, 0.9])
            if 0.1 in delta_quantiles:
                threshold = delta_quantiles[0.1]
                conditions.append(
                    Condition(
                        feature=feature,
                        kind="delta_le",
                        threshold=threshold,
                        description=f"{feature} falling fast: 1-bar change <= train q10 ({threshold:.6g})",
                        threshold_source="delta_quantile",
                        quantile=0.1,
                    )
                )
            if 0.9 in delta_quantiles:
                threshold = delta_quantiles[0.9]
                conditions.append(
                    Condition(
                        feature=feature,
                        kind="delta_ge",
                        threshold=threshold,
                        description=f"{feature} rising fast: 1-bar change >= train q90 ({threshold:.6g})",
                        threshold_source="delta_quantile",
                        quantile=0.9,
                    )
                )

        if "slope" in enabled_kinds:
            conditions.extend(build_slope_conditions(train, feature))

        if "divergence" in enabled_kinds:
            conditions.extend(build_divergence_conditions(train, [feature]))

    if cross_feature_pairs is None:
        cross_feature_pairs = detect_cross_feature_pairs(features)

    if "cross" in enabled_kinds:
        conditions.extend(build_cross_conditions(train, cross_feature_pairs))

    if "ratio" in enabled_kinds:
        conditions.extend(build_ratio_conditions(train, cross_feature_pairs))

    return conditions


_SLOPE_RE = re.compile(r"^slope_(\d+)_(le|ge)$")
_DIVERGENCE_RE = re.compile(r"^divergence_(bull|bear)_(\d+)$")


def condition_mask(data: pd.DataFrame, condition: Condition) -> pd.Series:
    series = data[condition.feature]

    if condition.kind == "value_le":
        return series <= condition.threshold
    if condition.kind == "value_ge":
        return series >= condition.threshold
    if condition.kind == "delta_le":
        return series.diff() <= condition.threshold
    if condition.kind == "delta_ge":
        return series.diff() >= condition.threshold

    slope_match = _SLOPE_RE.match(condition.kind)
    if slope_match:
        window = int(slope_match.group(1))
        direction = slope_match.group(2)
        slope = (series - series.shift(window)) / window
        if direction == "le":
            return slope <= condition.threshold
        return slope >= condition.threshold

    if condition.kind in ("cross_above", "cross_below"):
        series_b = data[condition.feature_b]
        if condition.kind == "cross_above":
            return (series > series_b) & (series.shift(1) <= series_b.shift(1))
        return (series < series_b) & (series.shift(1) >= series_b.shift(1))

    if condition.kind in ("ratio_le", "ratio_ge"):
        series_b = data[condition.feature_b]
        ratio = series / series_b.replace(0, np.nan)
        if condition.kind == "ratio_le":
            return ratio <= condition.threshold
        return ratio >= condition.threshold

    div_match = _DIVERGENCE_RE.match(condition.kind)
    if div_match:
        direction = div_match.group(1)
        window = int(div_match.group(2))
        
        # Determine the timeframe for this feature to get the correct close price
        tf = "15m"
        if condition.feature.startswith("tf_"):
            parts = condition.feature.split("_", 3)
            if len(parts) >= 3:
                tf = parts[1]
        
        close_col = f"tf_{tf}_close"
        if close_col in data.columns:
            price = data[close_col]
        else:
            close_cols = [c for c in data.columns if c.endswith("_close")]
            if close_cols:
                price = data[close_cols[0]]
            else:
                raise KeyError(f"Could not find close column (tried {close_col} and others) in DataFrame.")

        if direction == "bull":
            return (
                (price == price.rolling(window, min_periods=window).min())
                & (series > series.rolling(window, min_periods=window).min())
            )
        return (
            (price == price.rolling(window, min_periods=window).max())
            & (series < series.rolling(window, min_periods=window).max())
        )

    raise ValueError(f"Unknown condition kind: {condition.kind}")


def evaluate_mask(mask: pd.Series, returns: pd.Series, baseline_mean: float) -> Dict[str, float]:
    selected = returns.loc[mask.fillna(False)]
    support = int(len(selected))
    if support == 0:
        return {
            "support": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "edge_vs_baseline": np.nan,
        }
    avg_return = float(selected.mean())
    return {
        "support": support,
        "win_rate": float((selected > 0).mean()),
        "avg_return": avg_return,
        "median_return": float(selected.median()),
        "edge_vs_baseline": avg_return - baseline_mean,
    }


def evaluate_rule(
    data: pd.DataFrame,
    conditions: Sequence[Condition],
    target_column: str,
) -> Dict[str, float]:
    mask = pd.Series(True, index=data.index)
    for condition in conditions:
        mask &= condition_mask(data, condition).fillna(False)
    return evaluate_mask(mask, data[target_column], float(data[target_column].mean()))


def score_single_conditions(
    train: pd.DataFrame,
    conditions: Sequence[Condition],
    target_column: str,
    min_support: int,
) -> pd.DataFrame:
    rows = []
    for index, condition in enumerate(conditions):
        metrics = evaluate_rule(train, [condition], target_column)
        if metrics["support"] < min_support:
            continue
        rows.append(
            {
                "condition_index": index,
                "rule": condition.description,
                "conditions": 1,
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["edge_vs_baseline", "support"],
        ascending=[False, False],
    ).reset_index(drop=True)


def score_condition_pairs(
    train: pd.DataFrame,
    conditions: Sequence[Condition],
    target_column: str,
    candidate_indices: Sequence[int],
    min_support: int,
    max_pairs: int,
) -> pd.DataFrame:
    rows = []
    count = 0
    for left_pos, left_index in enumerate(candidate_indices):
        left = conditions[left_index]
        for right_index in candidate_indices[left_pos + 1 :]:
            right = conditions[right_index]
            if left.feature == right.feature:
                continue
            count += 1
            if count > max_pairs:
                break
            metrics = evaluate_rule(train, [left, right], target_column)
            if metrics["support"] < min_support:
                continue
            rows.append(
                {
                    "condition_indices": json.dumps([left_index, right_index]),
                    "rule": f"{left.description} AND {right.description}",
                    "conditions": 2,
                    **metrics,
                }
            )
        if count > max_pairs:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["edge_vs_baseline", "support"],
        ascending=[False, False],
    ).reset_index(drop=True)


def add_validation_metrics(
    candidates: pd.DataFrame,
    conditions: Sequence[Condition],
    test: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        if row["conditions"] == 1:
            rule_conditions = [conditions[int(row["condition_index"])]]
        else:
            rule_conditions = [
                conditions[index] for index in json.loads(row["condition_indices"])
            ]
        test_metrics = evaluate_rule(test, rule_conditions, target_column)
        rows.append(
            {
                **row.to_dict(),
                "test_support": test_metrics["support"],
                "test_win_rate": test_metrics["win_rate"],
                "test_avg_return": test_metrics["avg_return"],
                "test_median_return": test_metrics["median_return"],
                "test_edge_vs_baseline": test_metrics["edge_vs_baseline"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["validated"] = (
        (frame["test_support"] > 0)
        & (frame["avg_return"] > 0)
        & (frame["test_avg_return"] > 0)
        & (frame["edge_vs_baseline"] > 0)
        & (frame["test_edge_vs_baseline"] > 0)
    )
    return frame.sort_values(
        ["validated", "test_edge_vs_baseline", "test_support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def add_year_metrics(
    candidates: pd.DataFrame,
    conditions: Sequence[Condition],
    data: pd.DataFrame,
    target_column: str,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    dated = data.copy()
    dated["year"] = pd.to_datetime(dated["timestamp"], utc=True).dt.year
    for candidate_rank, row in candidates.head(top_n).iterrows():
        if row["conditions"] == 1:
            rule_conditions = [conditions[int(row["condition_index"])]]
        else:
            rule_conditions = [
                conditions[index] for index in json.loads(row["condition_indices"])
            ]
        for year, year_data in dated.groupby("year"):
            metrics = evaluate_rule(year_data, rule_conditions, target_column)
            rows.append(
                {
                    "candidate_rank": int(candidate_rank) + 1,
                    "year": int(year),
                    "support": metrics["support"],
                    "win_rate": metrics["win_rate"],
                    "avg_return": metrics["avg_return"],
                    "edge_vs_baseline": metrics["edge_vs_baseline"],
                    "rule": row["rule"],
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    candidates: pd.DataFrame,
    yearly: pd.DataFrame,
    output_path: Path,
    horizon_bars: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    lines = [
        "# Pattern Discovery Report",
        "",
        f"Base timeframe: 15m",
        f"Future return horizon: {horizon_bars} bars ({horizon_bars * 15} minutes)",
        f"Train period: {train['timestamp'].min()} to {train['timestamp'].max()}",
        f"Test period: {test['timestamp'].min()} to {test['timestamp'].max()}",
        "",
        "## Top Validated Candidates",
        "",
    ]
    if candidates.empty:
        lines.append("No candidates passed the configured support filters.")
    else:
        display_columns = [
            "validated",
            "conditions",
            "support",
            "avg_return",
            "win_rate",
            "test_support",
            "test_avg_return",
            "test_win_rate",
            "test_edge_vs_baseline",
            "rule",
        ]
        lines.append("```text")
        lines.append(candidates[display_columns].head(25).to_string(index=False))
        lines.append("```")
    if not yearly.empty:
        lines.extend(["", "## Year Breakdown For Top Candidates", "", "```text"])
        lines.append(
            yearly[
                [
                    "candidate_rank",
                    "year",
                    "support",
                    "avg_return",
                    "win_rate",
                    "edge_vs_baseline",
                ]
            ].to_string(index=False)
        )
        lines.append("```")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    max_features: int = 80,
    top_single_conditions: int = 80,
    max_pairs: int = 5000,
    min_train_support: int = 500,
    min_test_support: int = DEFAULT_MIN_TEST_SUPPORT,
) -> pd.DataFrame:
    data = load_dataset(input_path, horizon_bars)
    train, test = split_train_test(data, train_fraction)
    target_column = target_column_for_horizon(horizon_bars)
    features = rank_features(
        train,
        numeric_feature_columns(train),
        target_column,
        max_features=max_features,
    )
    LOGGER.info("Selected %s ranked features", len(features))

    conditions = build_conditions(train, features)
    singles = score_single_conditions(
        train,
        conditions,
        target_column,
        min_support=min_train_support,
    )
    candidate_indices = (
        singles["condition_index"].head(top_single_conditions).astype(int).tolist()
        if not singles.empty
        else []
    )
    pairs = score_condition_pairs(
        train,
        conditions,
        target_column,
        candidate_indices=candidate_indices,
        min_support=min_train_support,
        max_pairs=max_pairs,
    )
    candidates = pd.concat([singles, pairs], ignore_index=True)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["edge_vs_baseline", "support"],
            ascending=[False, False],
        ).head(500)
    candidates = add_validation_metrics(candidates, conditions, test, target_column)
    if not candidates.empty:
        candidates = candidates[candidates["test_support"] >= min_test_support]
    yearly = add_year_metrics(candidates, conditions, data, target_column, top_n=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidate_patterns.csv", index=False)
    yearly.to_csv(output_dir / "candidate_patterns_by_year.csv", index=False)
    (output_dir / "selected_features.txt").write_text(
        "\n".join(features) + ("\n" if features else ""),
        encoding="utf-8",
    )
    write_markdown_report(
        candidates,
        yearly,
        output_dir / "report.md",
        horizon_bars,
        train,
        test,
    )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and validate human-readable indicator patterns."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON_BARS)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--top-single-conditions", type=int, default=80)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--min-train-support", type=int, default=500)
    parser.add_argument("--min-test-support", type=int, default=DEFAULT_MIN_TEST_SUPPORT)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    candidates = run(
        input_path=args.input_path,
        output_dir=args.output_dir,
        horizon_bars=args.horizon_bars,
        train_fraction=args.train_fraction,
        max_features=args.max_features,
        top_single_conditions=args.top_single_conditions,
        max_pairs=args.max_pairs,
        min_train_support=args.min_train_support,
        min_test_support=args.min_test_support,
    )
    print(f"Wrote {args.output_dir / 'candidate_patterns.csv'}")
    print(f"Wrote {args.output_dir / 'candidate_patterns_by_year.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    print(f"Candidates after test support filter: {len(candidates)}")


if __name__ == "__main__":
    main()
