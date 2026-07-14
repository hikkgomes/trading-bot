import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT, WALK_FORWARD_DEFAULTS
from src.day_trade_search import numeric_feature_columns, simulate_day_trades
from src.feature_screener import screen_features
from src.walk_forward import WalkForwardConfig, generate_windows


def train_signal_model(train: pd.DataFrame, features: Sequence[str], label_column: str):
    x = train.loc[:, list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[label_column].astype(int)
    val_split = int(len(x) * 0.9)
    if val_split <= 0 or val_split >= len(x):
        raise ValueError("Not enough rows for model training")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x.iloc[:val_split],
        y.iloc[:val_split],
        eval_set=[(x.iloc[val_split:], y.iloc[val_split:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model, {"best_iteration": int(model.best_iteration_ or 0)}


def predict_signals(
    model,
    data: pd.DataFrame,
    features: Sequence[str],
    tp: float,
    sl: float,
    fee_cost: float,
    min_ev: float = 0.0,
):
    x = data.loc[:, list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prob = model.predict_proba(x)[:, 1]
    ev = prob * (tp - fee_cost) - (1.0 - prob) * (sl + fee_cost)
    return (
        pd.Series(ev > min_ev, index=data.index),
        pd.Series(prob, index=data.index),
        pd.Series(ev, index=data.index),
    )


def walk_forward_model_signals(
    data: pd.DataFrame,
    features: Sequence[str],
    label_column: str,
    wf_config: WalkForwardConfig,
    tp: float,
    sl: float,
    fee_cost: float,
    max_features: int = 80,
) -> pd.DataFrame:
    windows = generate_windows(len(data), wf_config)
    parts: list[pd.DataFrame] = []
    for train_slice, test_slice in windows:
        train = data.iloc[train_slice].copy()
        test = data.iloc[test_slice].copy()
        fold_features = screen_features(
            train, label_column, features, max_features=max_features, method="shap"
        )
        model, _ = train_signal_model(train, fold_features, label_column)
        sig, prob, ev = predict_signals(model, test, fold_features, tp, sl, fee_cost)
        out = test.loc[:, ["timestamp"]].copy()
        out["source_index"] = test.index.to_numpy()
        out["signal"] = sig.to_numpy()
        out["prob"] = prob.to_numpy()
        out["ev"] = ev.to_numpy()
        out["label_column"] = label_column
        parts.append(out)
    return pd.concat(parts, axis=0).reset_index(drop=True)


def backtest_model_signals(
    data: pd.DataFrame, signals_df: pd.DataFrame, trade_config, base_prefix: str = "tf_5m_"
) -> pd.DataFrame:
    if "source_index" not in signals_df.columns:
        raise ValueError("signals_df must include source_index")
    mask = pd.Series(False, index=data.index)
    for _, row in signals_df.iterrows():
        idx = int(row["source_index"])
        if pd.Timestamp(data.iloc[idx]["timestamp"]) != pd.Timestamp(row["timestamp"]):
            raise ValueError(f"Timestamp mismatch at source_index={idx}")
        mask.iloc[idx] = bool(row["signal"])
    direction_match = re.search(
        r"label_(long|short)_", str(signals_df.get("label_column", pd.Series([""])).iloc[0])
    )
    direction = direction_match.group(1) if direction_match else "long"
    return simulate_day_trades(data, mask, direction, trade_config, base_prefix=base_prefix)


def _infer_embargo_bars(label_column: str, fallback: int = 16) -> int:
    m = re.search(r"_h(\d+)$", label_column)
    if m:
        return int(m.group(1))
    return fallback


def run(
    input_path: Path,
    output_dir: Path,
    base_timeframe: str,
    label_column: str,
    tp: float,
    sl: float,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    embargo_bars: int | None = None,
    report: bool = False,
) -> Path:
    data = pd.read_parquet(input_path).sort_values("timestamp").reset_index(drop=True)
    features = numeric_feature_columns(data, base_prefix=f"tf_{base_timeframe}_")
    defaults = WALK_FORWARD_DEFAULTS.get(base_timeframe, WALK_FORWARD_DEFAULTS["5m"])
    resolved_embargo = (
        embargo_bars if embargo_bars is not None else _infer_embargo_bars(label_column, fallback=16)
    )
    wf = WalkForwardConfig(
        train_bars=defaults["train_bars"],
        test_bars=defaults["test_bars"],
        step_bars=defaults["step_bars"],
        embargo_bars=resolved_embargo,
    )
    fee_cost = 2 * ((fee_bps + slippage_bps) / 10_000)
    signals = walk_forward_model_signals(data, features, label_column, wf, tp, sl, fee_cost)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_signals.parquet"
    signals.to_parquet(output_path, index=False)
    if report:
        summary = {
            "rows": int(len(signals)),
            "signals": int(signals["signal"].sum()),
            "mean_prob": float(signals["prob"].mean()),
            "mean_ev": float(signals["ev"].mean()),
        }
        (output_dir / "report.json").write_text(
            pd.Series(summary).to_json(indent=2), encoding="utf-8"
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward model-driven signals.")
    parser.add_argument(
        "--input-path", type=Path, default=PROCESSED_DATA_DIR / "train_5m_indicators_labels.parquet"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "model_signals"
    )
    parser.add_argument("--base-timeframe", default="5m")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--tp", type=float, required=True)
    parser.add_argument("--sl", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run(
        args.input_path,
        args.output_dir,
        args.base_timeframe,
        args.label_column,
        args.tp,
        args.sl,
        args.fee_bps,
        args.slippage_bps,
        args.embargo_bars,
        args.report,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
