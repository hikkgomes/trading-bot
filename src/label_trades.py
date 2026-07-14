import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.trade_utils import scan_tp_sl


def compute_tp_sl_labels(
    open_, high, low, close, tp: float, sl: float, horizon: int, direction: str
):
    n = len(close)
    hit_tp = np.zeros(n, dtype=bool)
    bars_to_exit = np.full(n, -1, dtype=np.int64)
    mfe_full = np.full(n, np.nan, dtype=float)
    mae_full = np.full(n, np.nan, dtype=float)
    mfe_exit = np.full(n, np.nan, dtype=float)
    mae_exit = np.full(n, np.nan, dtype=float)
    is_long = direction == "long"
    for i in range(0, n - horizon - 1):
        entry_idx = i + 1
        entry = open_[entry_idx]
        end_idx = entry_idx + horizon
        exit_idx, reason = scan_tp_sl(high, low, entry, is_long, tp, sl, entry_idx, end_idx)
        bars_to_exit[i] = exit_idx - entry_idx
        hit_tp[i] = reason == 2
        highs_full = high[entry_idx : end_idx + 1]
        lows_full = low[entry_idx : end_idx + 1]
        highs_exit = high[entry_idx : exit_idx + 1]
        lows_exit = low[entry_idx : exit_idx + 1]
        if is_long:
            mfe_full[i] = float(np.max(highs_full) / entry - 1.0)
            mae_full[i] = float(np.min(lows_full) / entry - 1.0)
            mfe_exit[i] = float(np.max(highs_exit) / entry - 1.0)
            mae_exit[i] = float(np.min(lows_exit) / entry - 1.0)
        else:
            mfe_full[i] = float(entry / np.min(lows_full) - 1.0)
            mae_full[i] = float(entry / np.max(highs_full) - 1.0)
            mfe_exit[i] = float(entry / np.min(lows_exit) - 1.0)
            mae_exit[i] = float(entry / np.max(highs_exit) - 1.0)
    return hit_tp, bars_to_exit, mfe_full, mae_full, mfe_exit, mae_exit


def build_trade_labels(
    data: pd.DataFrame,
    tp_sl_pairs: Sequence[tuple[float, float]],
    horizons: Sequence[int],
    directions: Sequence[str],
    base_prefix: str,
) -> pd.DataFrame:
    open_ = data[f"{base_prefix}open"].astype(float).to_numpy()
    high = data[f"{base_prefix}high"].astype(float).to_numpy()
    low = data[f"{base_prefix}low"].astype(float).to_numpy()
    close = data[f"{base_prefix}close"].astype(float).to_numpy()
    columns = {}
    for direction in directions:
        for horizon in horizons:
            full_computed = False
            for tp, sl in tp_sl_pairs:
                tp_bps = int(round(tp * 10_000))
                sl_bps = int(round(sl * 10_000))
                label, bars, mfe_full, mae_full, mfe_exit, mae_exit = compute_tp_sl_labels(
                    open_, high, low, close, tp, sl, horizon, direction
                )
                prefix = f"{direction}_tp{tp_bps}_sl{sl_bps}_h{horizon}"
                columns[f"label_{prefix}"] = label
                columns[f"bars_to_exit_{prefix}"] = bars
                columns[f"mfe_{prefix}_exit"] = mfe_exit
                columns[f"mae_{prefix}_exit"] = mae_exit
                if not full_computed:
                    columns[f"mfe_{direction}_h{horizon}_full"] = mfe_full
                    columns[f"mae_{direction}_h{horizon}_full"] = mae_full
                    full_computed = True
    if not columns:
        return data.copy()
    return pd.concat([data.copy(), pd.DataFrame(columns, index=data.index)], axis=1).copy()


def run(
    input_path: Path,
    output_path: Path,
    base_timeframe: str = "5m",
    tp_sl_pairs: Sequence[tuple[float, float]] = (
        (0.003, 0.002),
        (0.005, 0.003),
        (0.008, 0.004),
        (0.012, 0.006),
    ),
    horizons: Sequence[int] = (4, 8, 16),
    directions: Sequence[str] = ("long", "short"),
) -> Path:
    data = pd.read_parquet(input_path).sort_values("timestamp").reset_index(drop=True)
    base_prefix = f"tf_{base_timeframe}_"
    out = build_trade_labels(data, tp_sl_pairs, horizons, directions, base_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TP-before-SL labels.")
    parser.add_argument(
        "--input-path",
        "--input",
        dest="input_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "train_5m_indicators.parquet",
    )
    parser.add_argument(
        "--output-path",
        "--output",
        dest="output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "train_5m_indicators_labels.parquet",
    )
    parser.add_argument("--base-timeframe", default="5m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input_path, args.output_path, args.base_timeframe)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
