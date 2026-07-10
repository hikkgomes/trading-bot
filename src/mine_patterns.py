import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.build_dataset import TARGET_COLUMNS
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT

DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "train_15m_indicators.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "patterns"
DEFAULT_TARGET = "target_return_next_4_bars"


def load_data(
    path: Path,
    max_features: int,
    target_column: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    data = pd.read_parquet(path)
    required = {"timestamp", target_column, *TARGET_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    feature_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in TARGET_COLUMNS
    ]

    ranked = rank_univariate_features(data, feature_columns, target_column, max_features)
    keep = ["timestamp", *TARGET_COLUMNS, *ranked]
    return data.loc[:, keep].dropna(subset=[target_column]).reset_index(drop=True)


def rank_univariate_features(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    max_features: int,
) -> list[str]:
    target = data[target_column]
    scores = []
    for column in feature_columns:
        series = data[column]
        valid = series.notna() & target.notna()
        if valid.sum() < 1000:
            continue
        if series[valid].nunique(dropna=True) <= 1:
            continue
        corr = series[valid].corr(target[valid], method="spearman")
        if pd.notna(corr):
            scores.append((column, abs(float(corr)), float(corr)))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [column for column, _, _ in scores[:max_features]]


def make_direction_labels(
    returns: pd.Series,
    min_abs_return: float,
) -> pd.Series:
    labels = pd.Series(np.nan, index=returns.index, dtype="float64")
    labels.loc[returns > min_abs_return] = 1.0
    labels.loc[returns < -min_abs_return] = 0.0
    return labels


def train_rule_tree(
    data: pd.DataFrame,
    target_column: str,
    min_abs_return: float,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    labels = make_direction_labels(data[target_column], min_abs_return)
    feature_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in TARGET_COLUMNS
    ]
    model_data = data.loc[labels.notna(), feature_columns].copy()
    model_labels = labels.loc[labels.notna()].astype(int)
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.fillna(model_data.median(numeric_only=True))

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(model_data, model_labels)
    return model, model_data, model_labels


def extract_rules(
    model: DecisionTreeClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    returns: pd.Series,
) -> list[dict[str, object]]:
    tree = model.tree_
    feature_names = features.columns.to_numpy()
    baseline_up_rate = float(labels.mean())
    rules = []

    def walk(node: int, conditions: list[str]) -> None:
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            walk(tree.children_left[node], [*conditions, f"{feature} <= {threshold:.8g}"])
            walk(tree.children_right[node], [*conditions, f"{feature} > {threshold:.8g}"])
            return

        leaf_mask = np.ones(len(features), dtype=bool)
        for condition in conditions:
            if " <= " in condition:
                feature, threshold = condition.split(" <= ")
                leaf_mask &= features[feature].to_numpy() <= float(threshold)
            else:
                feature, threshold = condition.split(" > ")
                leaf_mask &= features[feature].to_numpy() > float(threshold)

        support = int(leaf_mask.sum())
        if support == 0:
            return

        leaf_labels = labels.to_numpy()[leaf_mask]
        leaf_returns = returns.loc[labels.index].to_numpy()[leaf_mask]
        up_rate = float(np.mean(leaf_labels))
        avg_return = float(np.mean(leaf_returns))
        rules.append(
            {
                "rule": " AND ".join(conditions),
                "conditions": len(conditions),
                "support": support,
                "support_pct": support / len(features),
                "up_rate": up_rate,
                "down_rate": 1.0 - up_rate,
                "baseline_up_rate": baseline_up_rate,
                "lift_up": up_rate / baseline_up_rate if baseline_up_rate else np.nan,
                "avg_future_return": avg_return,
                "used_features": sorted(
                    {
                        condition.split(" <= ")[0].split(" > ")[0]
                        for condition in conditions
                    }
                ),
            }
        )

    walk(0, [])
    return rules


def write_outputs(rules: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rules)
    if frame.empty:
        frame.to_csv(output_dir / "rules.csv", index=False)
        (output_dir / "selected_features.txt").write_text("", encoding="utf-8")
        return

    frame["edge"] = (frame["up_rate"] - frame["baseline_up_rate"]).abs()
    frame = frame.sort_values(
        ["edge", "support", "conditions"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    frame["used_features"] = frame["used_features"].map(json.dumps)
    frame.to_csv(output_dir / "rules.csv", index=False)

    selected = sorted(
        {
            feature
            for features_json in frame.head(50)["used_features"]
            for feature in json.loads(features_json)
        }
    )
    (output_dir / "selected_features.txt").write_text(
        "\n".join(selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine human-readable indicator rules correlated with future price movement."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=TARGET_COLUMNS)
    parser.add_argument(
        "--min-abs-return",
        type=float,
        default=0.002,
        help="Ignore tiny future moves between -x and +x when learning direction rules.",
    )
    parser.add_argument("--max-features", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.input_path, args.max_features, target_column=args.target)
    model, features, labels = train_rule_tree(
        data=data,
        target_column=args.target,
        min_abs_return=args.min_abs_return,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    rules = extract_rules(
        model=model,
        features=features,
        labels=labels,
        returns=data[args.target],
    )
    write_outputs(rules, args.output_dir)
    print(f"Wrote {args.output_dir / 'rules.csv'}")
    print(f"Wrote {args.output_dir / 'selected_features.txt'}")


if __name__ == "__main__":
    main()
