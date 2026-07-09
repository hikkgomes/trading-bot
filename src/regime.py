import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import DEFAULT_SYMBOL, PROCESSED_DATA_DIR, indicator_data_dir


def _timestamped(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
        first = out.columns[0]
        if first != "timestamp":
            out = out.rename(columns={first: "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("Missing timestamp column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def _regime_frame(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    if price_column not in data.columns:
        raise ValueError(f"Missing {price_column}")
    daily = _timestamped(data)[["timestamp", price_column]].copy()
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
    daily["tf_1d_regime_id"] = -1
    if len(valid) < 4:
        return daily[["timestamp", "tf_1d_regime_id"]]
    n_clusters = min(4, len(valid))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    daily.loc[valid.index, "tf_1d_regime_id"] = model.fit_predict(valid)
    daily["tf_1d_regime_id"] = daily["tf_1d_regime_id"].ffill().fillna(-1).astype(int)
    return daily[["timestamp", "tf_1d_regime_id"]]


def add_regime_column(data: pd.DataFrame, price_column: str = "tf_1d_close") -> pd.DataFrame:
    out = _timestamped(data)
    regimes = _regime_frame(out, price_column)
    out = out.drop(columns=["tf_1d_regime_id"], errors="ignore")
    out = pd.merge_asof(
        out.sort_values("timestamp"),
        regimes.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    out["tf_1d_regime_id"] = out["tf_1d_regime_id"].fillna(-1).astype(int)
    return out


def add_regime_column_from_daily(
    data: pd.DataFrame,
    daily_data: pd.DataFrame,
    *,
    daily_price_column: str = "close",
) -> pd.DataFrame:
    out = _timestamped(data).drop(columns=["tf_1d_regime_id"], errors="ignore")
    regimes = _regime_frame(daily_data, daily_price_column)
    out = pd.merge_asof(
        out.sort_values("timestamp"),
        regimes.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    out["tf_1d_regime_id"] = out["tf_1d_regime_id"].fillna(-1).astype(int)
    return out


def _indicator_path(symbol: str, market: str, timeframe: str) -> Path:
    return indicator_data_dir(symbol, market, legacy_fallback=True) / f"{symbol}_{timeframe}_all_indicators.parquet"


def tag_regime_file(
    input_path: Path,
    output_path: Path,
    *,
    price_column: str = "tf_1d_close",
    daily_input_path: Path | None = None,
    daily_price_column: str = "close",
    skip_if_missing: bool = False,
) -> dict:
    if not input_path.exists():
        if skip_if_missing:
            return {
                "ok": True,
                "skipped": True,
                "reason": "missing_input",
                "input": str(input_path),
                "output": str(output_path),
            }
        raise FileNotFoundError(input_path)
    if daily_input_path is not None and not daily_input_path.exists():
        if skip_if_missing:
            return {
                "ok": True,
                "skipped": True,
                "reason": "missing_daily_input",
                "input": str(input_path),
                "daily_input": str(daily_input_path),
                "output": str(output_path),
            }
        raise FileNotFoundError(daily_input_path)

    data = pd.read_parquet(input_path)
    if daily_input_path is not None:
        daily = pd.read_parquet(daily_input_path)
        out = add_regime_column_from_daily(data, daily, daily_price_column=daily_price_column)
    else:
        out = add_regime_column(data, price_column=price_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    regimes = out["tf_1d_regime_id"]
    return {
        "ok": True,
        "skipped": False,
        "input": str(input_path),
        "daily_input": str(daily_input_path) if daily_input_path else None,
        "output": str(output_path),
        "rows": int(len(out)),
        "regime_counts": {str(int(k)): int(v) for k, v in regimes.value_counts().sort_index().items()},
    }


def run(input_path: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or input_path
    tag_regime_file(input_path, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add rolling-volatility regime labels.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--price-column", default=None)
    parser.add_argument("--daily-input", type=Path, default=None)
    parser.add_argument("--daily-price-column", default="close")
    parser.add_argument("--market", choices=("spot", "futures"), default=None)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--daily-timeframe", default="1d")
    parser.add_argument("--skip-if-missing", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _default_input(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return args.input
    if args.market:
        return _indicator_path(args.symbol, args.market, args.timeframe)
    return PROCESSED_DATA_DIR / "train_15m_indicators.parquet"


def _default_daily_input(args: argparse.Namespace) -> Path | None:
    if args.daily_input is not None:
        return args.daily_input
    if args.market:
        return _indicator_path(args.symbol, args.market, args.daily_timeframe)
    return None


def main() -> None:
    args = parse_args()
    input_path = _default_input(args)
    output_path = args.output or input_path
    price_column = args.price_column or ("close" if args.market else "tf_1d_close")
    report = tag_regime_file(
        input_path,
        output_path,
        price_column=price_column,
        daily_input_path=_default_daily_input(args),
        daily_price_column=args.daily_price_column,
        skip_if_missing=args.skip_if_missing,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report.get("skipped"):
        print(f"Skipped regime tagging: {report['reason']} ({input_path})")
    else:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
