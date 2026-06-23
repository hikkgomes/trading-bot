import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trading-bot-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/trading-bot-cache")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.build_dataset import TARGET_COLUMN, TARGET_COLUMNS
from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR
from src.load_data import configure_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "train_15m.parquet"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
DEFAULT_FEATURE_REPORTS_DIR = PROJECT_ROOT / "outputs" / "feature_reports"


def load_training_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing training table: {input_path}. Run python -m src.build_dataset first."
        )

    data = pd.read_parquet(input_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training table is missing target column {TARGET_COLUMN!r}")

    before = len(data)
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    LOGGER.info("Loaded %s rows from %s", before, input_path)
    LOGGER.info("Kept %s rows after removing missing targets", len(data))
    return data


def get_feature_matrix(
    data: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in data.columns:
        raise ValueError(f"Training table is missing target column {target_column!r}")

    excluded = {"timestamp"} | set(TARGET_COLUMNS)
    feature_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in excluded
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    features = data.loc[:, feature_columns]
    target = data[target_column]
    return features, target


def time_ordered_split(
    features: pd.DataFrame, target: pd.Series, train_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")

    split_index = int(len(features) * train_fraction)
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Not enough rows for the requested train/test split")

    x_train = features.iloc[:split_index]
    x_test = features.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]
    LOGGER.info("Time split: train=%s rows test=%s rows", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str = "regression",
) -> Union[lgb.LGBMRegressor, lgb.LGBMClassifier]:
    val_split = int(len(x_train) * 0.9)
    if val_split <= 0 or val_split >= len(x_train):
        raise ValueError("Not enough training rows to create an early-stopping slice")

    x_tr = x_train.iloc[:val_split]
    x_val = x_train.iloc[val_split:]
    y_tr = y_train.iloc[:val_split]
    y_val = y_train.iloc[val_split:]

    common_params = dict(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if objective == "binary":
        model = lgb.LGBMClassifier(**common_params)
        y_tr = y_tr.astype(int)
        y_val = y_val.astype(int)
    else:
        model = lgb.LGBMRegressor(objective=objective, **common_params)

    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    LOGGER.info("Best iteration: %s", model.best_iteration_)
    return model


def evaluate_model(
    model: lgb.LGBMRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    return {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(x_train.shape[1]),
        "best_iteration": int(model.best_iteration_ or 0),
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


def get_feature_importance(model: lgb.LGBMRegressor) -> pd.DataFrame:
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


def write_metrics(metrics: Dict[str, float], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_baseline_15m_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", output_path)
    return output_path


def write_top_features(importance: pd.DataFrame, output_dir: Path, top_n: int = 100) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_top_100_features_15m.csv"
    importance.head(top_n).to_csv(output_path, index=False)
    LOGGER.info("Wrote top %s features to %s", top_n, output_path)
    return output_path


def write_feature_importance_chart(
    importance: pd.DataFrame, output_dir: Path, top_n: int = 30
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_feature_importance_15m.png"
    top = importance.head(top_n).sort_values("importance_gain")

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(top["feature"], top["importance_gain"], color="#2563eb")
    ax.set_title(f"LightGBM Feature Importance by Gain - Top {top_n}")
    ax.set_xlabel("Importance gain")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    LOGGER.info("Wrote feature importance chart to %s", output_path)
    return output_path


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    charts_dir: Path = DEFAULT_CHARTS_DIR,
    feature_reports_dir: Path = DEFAULT_FEATURE_REPORTS_DIR,
    train_fraction: float = 0.8,
) -> None:
    data = load_training_data(input_path)
    features, target = get_feature_matrix(data)
    x_train, x_test, y_train, y_test = time_ordered_split(
        features, target, train_fraction
    )
    model = train_model(x_train, y_train, objective="regression")
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    importance = get_feature_importance(model)

    write_metrics(metrics, metrics_dir)
    write_feature_importance_chart(importance, charts_dir)
    write_top_features(importance, feature_reports_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline LightGBM regressor.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS_DIR)
    parser.add_argument(
        "--feature-reports-dir", type=Path, default=DEFAULT_FEATURE_REPORTS_DIR
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(
        input_path=args.input_path,
        metrics_dir=args.metrics_dir,
        charts_dir=args.charts_dir,
        feature_reports_dir=args.feature_reports_dir,
        train_fraction=args.train_fraction,
    )


if __name__ == "__main__":
    main()
