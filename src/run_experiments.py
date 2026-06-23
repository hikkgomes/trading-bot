import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trading-bot-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trading-bot-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.build_dataset import TARGET_COLUMNS
from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR
from src.load_data import configure_logging
from src.train_model import get_feature_matrix, time_ordered_split, train_model


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "train_15m_indicators.parquet"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "outputs" / "indicators_research"
CLASSIFICATION_TARGETS = frozenset({"target_direction_next_4_bars"})
CRYPTO_15M_BARS_PER_DAY = 96
DEFAULT_WALK_FORWARD_TRAIN_BARS = CRYPTO_15M_BARS_PER_DAY * 365 * 3
DEFAULT_WALK_FORWARD_TEST_BARS = CRYPTO_15M_BARS_PER_DAY * 90
DEFAULT_WALK_FORWARD_SPLITS = 5


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing training table: {path}")
    data = pd.read_parquet(path)
    missing_targets = [target for target in TARGET_COLUMNS if target not in data.columns]
    if missing_targets:
        raise ValueError(f"Training table is missing targets: {missing_targets}")
    return data.dropna(subset=list(TARGET_COLUMNS)).reset_index(drop=True)


def _experiment_type(target: str) -> str:
    return "classification" if target in CLASSIFICATION_TARGETS else "regression"


def _objective_for_target(target: str) -> str:
    return "binary" if target in CLASSIFICATION_TARGETS else "regression"


def _evaluate_regression(
    model,
    target: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    return {
        "target": target,
        "type": "regression",
        "best_iteration": int(model.best_iteration_ or 0),
        "feature_count": int(x_train.shape[1]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_mae": float(mean_absolute_error(y_train, train_predictions)),
        "test_mae": float(mean_absolute_error(y_test, test_predictions)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_predictions))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "train_r2": float(r2_score(y_train, train_predictions)),
        "test_r2": float(r2_score(y_test, test_predictions)),
        "test_directional_accuracy": float(
            np.mean(np.sign(test_predictions) == np.sign(y_test.to_numpy()))
        ),
    }


def _classification_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def _evaluate_classification(
    model,
    target: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    train_probabilities = _classification_probabilities(model, x_train)
    test_probabilities = _classification_probabilities(model, x_test)
    train_predictions = (train_probabilities >= 0.5).astype(int)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    try:
        test_roc_auc: Optional[float] = float(roc_auc_score(y_test_int, test_probabilities))
    except ValueError:
        test_roc_auc = None

    return {
        "target": target,
        "type": "classification",
        "best_iteration": int(model.best_iteration_ or 0),
        "feature_count": int(x_train.shape[1]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_accuracy": float(accuracy_score(y_train_int, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test_int, test_predictions)),
        "test_roc_auc": test_roc_auc,
    }


def _get_feature_importance(model) -> pd.DataFrame:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    )
    return importance.sort_values(
        ["importance_gain", "importance_split"], ascending=False
    ).reset_index(drop=True)


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote %s", path)


def _save_feature_importance_chart(
    importance: pd.DataFrame,
    path: Path,
    title: str,
    top_n: int = 30,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).sort_values("importance_gain")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(top["feature"], top["importance_gain"], color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("Importance gain")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def _normalise_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def _compute_and_save_shap(model, x_test: pd.DataFrame, target: str, shap_dir: Path) -> None:
    try:
        import shap
    except ImportError:
        LOGGER.warning("Skipping SHAP for %s because shap is not installed", target)
        return

    shap_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(x_test)
    shap_values = _normalise_shap_values(raw_values)
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(shap_dir / f"{target}_shap_importance.csv", index=False)

    shap.summary_plot(shap_values, x_test, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(shap_dir / f"{target}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Wrote SHAP outputs for %s", target)


def _prediction_values(model, x_test: pd.DataFrame, target: str) -> np.ndarray:
    if _experiment_type(target) == "classification":
        return _classification_probabilities(model, x_test)
    return model.predict(x_test)


def _save_predictions(
    data: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    target: str,
    output_base: Path,
) -> None:
    output_dir = output_base / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = data.loc[x_test.index, ["timestamp", "tf_15m_close"]].copy()
    prediction_frame = prediction_frame.rename(columns={"tf_15m_close": "close"})
    prediction_frame["y_true"] = y_test.to_numpy()
    prediction_frame["y_pred"] = _prediction_values(model, x_test, target)
    prediction_frame.to_csv(output_dir / f"{target}_test_predictions.csv", index=False)
    LOGGER.info("Wrote predictions for %s", target)


def run_experiment(
    data: pd.DataFrame,
    target: str,
    train_fraction: float,
    output_base: Path,
    compute_shap: bool = False,
) -> Dict[str, object]:
    LOGGER.info("Running experiment for %s", target)
    features, labels = get_feature_matrix(data, target_column=target)
    x_train, x_test, y_train, y_test = time_ordered_split(
        features, labels, train_fraction
    )
    objective = _objective_for_target(target)
    model = train_model(x_train, y_train, objective=objective)

    if _experiment_type(target) == "classification":
        metrics = _evaluate_classification(model, target, x_train, y_train, x_test, y_test)
    else:
        metrics = _evaluate_regression(model, target, x_train, y_train, x_test, y_test)

    metrics_dir = output_base / "metrics"
    charts_dir = output_base / "charts"
    feature_reports_dir = output_base / "feature_reports"
    shap_dir = output_base / "shap"

    _save_json(metrics, metrics_dir / f"{target}_metrics.json")
    importance = _get_feature_importance(model)
    feature_reports_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(
        feature_reports_dir / f"{target}_feature_importance.csv",
        index=False,
    )
    LOGGER.info("Wrote feature importance for %s", target)
    _save_feature_importance_chart(
        importance,
        charts_dir / f"{target}_feature_importance.png",
        f"{target} feature importance",
    )
    if compute_shap:
        _compute_and_save_shap(model, x_test, target, shap_dir)
    _save_predictions(data, x_test, y_test, model, target, output_base)
    return metrics


def walk_forward_splits(
    row_count: int,
    train_rows: int = DEFAULT_WALK_FORWARD_TRAIN_BARS,
    test_rows: int = DEFAULT_WALK_FORWARD_TEST_BARS,
    n_splits: int = DEFAULT_WALK_FORWARD_SPLITS,
) -> List[Tuple[int, int, int, int]]:
    if train_rows <= 0 or test_rows <= 0 or n_splits <= 0:
        raise ValueError("Walk-forward train rows, test rows, and splits must be positive")
    if row_count < train_rows + test_rows:
        raise ValueError(
            "Not enough rows for walk-forward validation: "
            f"rows={row_count} train_rows={train_rows} test_rows={test_rows}"
        )

    max_splits = (row_count - train_rows) // test_rows
    actual_splits = min(n_splits, max_splits)
    first_test_start = row_count - (actual_splits * test_rows)
    splits = []
    for split_number in range(actual_splits):
        test_start = first_test_start + split_number * test_rows
        test_end = test_start + test_rows
        train_start = test_start - train_rows
        train_end = test_start
        splits.append((train_start, train_end, test_start, test_end))
    return splits


def _split_by_bounds(
    features: pd.DataFrame,
    labels: pd.Series,
    bounds: Tuple[int, int, int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_start, train_end, test_start, test_end = bounds
    return (
        features.iloc[train_start:train_end],
        features.iloc[test_start:test_end],
        labels.iloc[train_start:train_end],
        labels.iloc[test_start:test_end],
    )


def run_walk_forward_experiment(
    data: pd.DataFrame,
    target: str,
    output_base: Path,
    train_rows: int,
    test_rows: int,
    n_splits: int,
) -> List[Dict[str, object]]:
    LOGGER.info("Running walk-forward experiment for %s", target)
    features, labels = get_feature_matrix(data, target_column=target)
    bounds_list = walk_forward_splits(len(features), train_rows, test_rows, n_splits)
    objective = _objective_for_target(target)
    metrics_rows = []

    for fold_index, bounds in enumerate(bounds_list, start=1):
        x_train, x_test, y_train, y_test = _split_by_bounds(features, labels, bounds)
        model = train_model(x_train, y_train, objective=objective)
        if _experiment_type(target) == "classification":
            metrics = _evaluate_classification(model, target, x_train, y_train, x_test, y_test)
        else:
            metrics = _evaluate_regression(model, target, x_train, y_train, x_test, y_test)

        train_start, train_end, test_start, test_end = bounds
        metrics.update(
            {
                "fold": fold_index,
                "train_start": str(data["timestamp"].iloc[train_start]),
                "train_end": str(data["timestamp"].iloc[train_end - 1]),
                "test_start": str(data["timestamp"].iloc[test_start]),
                "test_end": str(data["timestamp"].iloc[test_end - 1]),
            }
        )
        metrics_rows.append(metrics)

    metrics_dir = output_base / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        metrics_dir / f"{target}_walk_forward_metrics.csv",
        index=False,
    )
    return metrics_rows


def summarize_walk_forward(all_metrics: Iterable[Dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(list(all_metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("# Walk-Forward Summary\n\nNo folds were run.\n", encoding="utf-8")
        return

    numeric_columns = [
        column
        for column in (
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_directional_accuracy",
            "test_accuracy",
            "test_roc_auc",
        )
        if column in frame.columns
    ]
    grouped = frame.groupby(["target", "type"], dropna=False)[numeric_columns].agg(
        ["mean", "std", "min", "max"]
    )
    lines = ["# Walk-Forward Summary", ""]
    lines.append("```text")
    lines.append(grouped.to_string())
    lines.append("```")
    lines.append("")
    lines.append(f"Folds per target: {int(frame.groupby('target').size().min())}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_markdown(all_metrics: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "target",
        "type",
        "best_iteration",
        "feature_count",
        "train_rows",
        "test_rows",
        "train_mae",
        "test_mae",
        "train_rmse",
        "test_rmse",
        "test_r2",
        "test_directional_accuracy",
        "test_accuracy",
        "test_roc_auc",
    ]
    lines = [
        "# Experiment Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for metrics in all_metrics:
        lines.append(
            "| "
            + " | ".join(_format_metric(metrics.get(header)) for header in headers)
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    train_fraction: float = 0.8,
    validation_mode: str = "single",
    compute_shap: bool = False,
    walk_forward_train_rows: int = DEFAULT_WALK_FORWARD_TRAIN_BARS,
    walk_forward_test_rows: int = DEFAULT_WALK_FORWARD_TEST_BARS,
    walk_forward_splits_count: int = DEFAULT_WALK_FORWARD_SPLITS,
) -> None:
    data = load_data(input_path)
    if validation_mode in {"single", "both"}:
        all_metrics = [
            run_experiment(
                data,
                target,
                train_fraction,
                output_base,
                compute_shap=compute_shap,
            )
            for target in TARGET_COLUMNS
        ]
        write_summary_markdown(all_metrics, output_base / "metrics" / "summary.md")

    if validation_mode in {"walk-forward", "both"}:
        all_walk_forward_metrics: List[Dict[str, object]] = []
        for target in TARGET_COLUMNS:
            all_walk_forward_metrics.extend(
                run_walk_forward_experiment(
                    data,
                    target,
                    output_base,
                    train_rows=walk_forward_train_rows,
                    test_rows=walk_forward_test_rows,
                    n_splits=walk_forward_splits_count,
                )
            )
        summarize_walk_forward(
            all_walk_forward_metrics,
            output_base / "metrics" / "walk_forward_summary.md",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all trading research experiments.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--validation-mode",
        choices=("single", "walk-forward", "both"),
        default="single",
    )
    parser.add_argument(
        "--with-shap",
        action="store_true",
        help="Compute SHAP outputs. This can be slow on the full indicator dataset.",
    )
    parser.add_argument(
        "--walk-forward-train-rows",
        type=int,
        default=DEFAULT_WALK_FORWARD_TRAIN_BARS,
    )
    parser.add_argument(
        "--walk-forward-test-rows",
        type=int,
        default=DEFAULT_WALK_FORWARD_TEST_BARS,
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=DEFAULT_WALK_FORWARD_SPLITS,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(
        input_path=args.input_path,
        output_base=args.output_base,
        train_fraction=args.train_fraction,
        validation_mode=args.validation_mode,
        compute_shap=args.with_shap,
        walk_forward_train_rows=args.walk_forward_train_rows,
        walk_forward_test_rows=args.walk_forward_test_rows,
        walk_forward_splits_count=args.walk_forward_splits,
    )


if __name__ == "__main__":
    main()
