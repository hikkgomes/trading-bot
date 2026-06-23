import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import PROCESSED_DATA_DIR


def add_regime_column(data: pd.DataFrame, price_column: str = "tf_1d_close") -> pd.DataFrame:
    if price_column not in data.columns:
        raise ValueError(f"Missing {price_column}")
    out = data.copy()
    daily = out[["timestamp", price_column]].copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    daily = daily.dropna(subset=[price_column])
    daily = daily.loc[daily[price_column].ne(daily[price_column].shift())].copy()
    returns = daily[price_column].astype(float).pct_change()
    features = pd.DataFrame(
        {
            "realized_vol_21d": returns.rolling(21, min_periods=5).std(),
            "abs_return_21d": returns.rolling(21, min_periods=5).mean().abs(),
        },
        index=daily.index,
    ).replace([np.inf, -np.inf], np.nan)
    valid = features.dropna()
    out["tf_1d_regime_id"] = -1
    if len(valid) < 4:
        return out
    n_clusters = min(4, len(valid))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    daily.loc[valid.index, "tf_1d_regime_id"] = model.fit_predict(valid)
    daily["tf_1d_regime_id"] = daily["tf_1d_regime_id"].ffill().fillna(-1).astype(int)
    out = pd.merge_asof(
        out.sort_values("timestamp"),
        daily[["timestamp", "tf_1d_regime_id"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    out["tf_1d_regime_id"] = out["tf_1d_regime_id_y"].fillna(out["tf_1d_regime_id_x"]).fillna(-1).astype(int)
    return out.drop(columns=["tf_1d_regime_id_x", "tf_1d_regime_id_y"])


def run(input_path: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or input_path
    data = pd.read_parquet(input_path).sort_values("timestamp").reset_index(drop=True)
    add_regime_column(data).to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add rolling-volatility regime labels.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "train_15m_indicators.parquet")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Wrote {run(args.input, args.output)}")


if __name__ == "__main__":
    main()
