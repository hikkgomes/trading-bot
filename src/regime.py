import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_SYMBOL, PROCESSED_DATA_DIR, indicator_data_dir
from src.parquet_io import write_parquet_atomic

REGIME_LABELS = {
    -1: "unknown",
    0: "range",
    1: "bull_trend",
    2: "bear_trend",
    3: "high_volatility",
}
REGIME_LOOKBACK_DAYS = 21
REGIME_MIN_HISTORY_DAYS = 20


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
    """Build causal, prefix-invariant daily regimes with stable semantics.

    Every row depends only on prices available at or before that row.  Numeric
    IDs have fixed meanings from :data:`REGIME_LABELS`; unlike full-sample
    clustering, appending future data cannot relabel history.
    """

    if price_column not in data.columns:
        raise ValueError(f"Missing {price_column}")
    daily = _timestamped(data)[["timestamp", price_column]].copy()
    daily[price_column] = pd.to_numeric(daily[price_column], errors="coerce")
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=[price_column])
    daily = daily.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    # ``add_regime_column`` also accepts an intraday frame carrying a repeated
    # already-closed daily feature. Collapse that input to the first available
    # value per UTC day, while retaining genuine unchanged closes in daily data.
    deltas = daily["timestamp"].diff().dropna()
    if not deltas.empty and deltas.median() < pd.Timedelta(hours=12):
        daily["_utc_day"] = daily["timestamp"].dt.floor("D")
        daily = daily.drop_duplicates("_utc_day", keep="first").drop(columns="_utc_day")
    prices = daily[price_column].astype(float)
    returns = prices.pct_change()
    realized_vol = returns.rolling(
        REGIME_LOOKBACK_DAYS,
        min_periods=REGIME_LOOKBACK_DAYS,
    ).std()
    trailing_return = prices.pct_change(REGIME_LOOKBACK_DAYS)
    # Compare current volatility with a threshold learned strictly from prior
    # observations. The expanding quantile is causal and prefix-invariant.
    high_vol_threshold = (
        realized_vol.expanding(min_periods=REGIME_MIN_HISTORY_DAYS).quantile(0.75).shift(1)
    )
    trend_band = pd.Series(
        np.maximum(0.02, realized_vol.to_numpy() * np.sqrt(REGIME_LOOKBACK_DAYS)),
        index=daily.index,
    )
    daily["tf_1d_regime_id"] = -1
    valid = realized_vol.notna() & trailing_return.notna() & high_vol_threshold.notna()
    daily.loc[valid, "tf_1d_regime_id"] = 0
    high_volatility = valid & (realized_vol > high_vol_threshold)
    daily.loc[high_volatility, "tf_1d_regime_id"] = 3
    daily.loc[valid & ~high_volatility & (trailing_return > trend_band), "tf_1d_regime_id"] = 1
    daily.loc[valid & ~high_volatility & (trailing_return < -trend_band), "tf_1d_regime_id"] = 2
    daily["tf_1d_regime_id"] = daily["tf_1d_regime_id"].astype(int)
    return daily[["timestamp", "tf_1d_regime_id"]]


def add_regime_column(data: pd.DataFrame, price_column: str = "tf_1d_close") -> pd.DataFrame:
    out = _timestamped(data)
    regimes = _regime_frame(out, price_column)
    source_deltas = out["timestamp"].sort_values().diff().dropna()
    if (
        price_column == "close"
        and not source_deltas.empty
        and source_deltas.median() >= pd.Timedelta(hours=12)
    ):
        # A direct daily ``close`` is known only after its timestamped candle.
        regimes["timestamp"] = regimes["timestamp"] + pd.Timedelta(days=1)
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
    # Daily candle timestamps denote candle opens. A label using that candle's
    # close becomes available to intraday consumers only when the day has closed.
    regimes["timestamp"] = regimes["timestamp"] + pd.Timedelta(days=1)
    out = pd.merge_asof(
        out.sort_values("timestamp"),
        regimes.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    out["tf_1d_regime_id"] = out["tf_1d_regime_id"].fillna(-1).astype(int)
    return out


def _indicator_path(symbol: str, market: str, timeframe: str) -> Path:
    return (
        indicator_data_dir(symbol, market, legacy_fallback=True)
        / f"{symbol}_{timeframe}_all_indicators.parquet"
    )


def tag_regime_file(
    input_path: Path,
    output_path: Path,
    *,
    price_column: str = "tf_1d_close",
    daily_input_path: Path | None = None,
    daily_price_column: str = "close",
    skip_if_missing: bool = False,
    compact: bool = False,
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

    compact_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    data = pd.read_parquet(input_path, columns=compact_columns if compact else None)
    if daily_input_path is not None:
        daily = pd.read_parquet(daily_input_path, columns=["timestamp", daily_price_column])
        out = add_regime_column_from_daily(data, daily, daily_price_column=daily_price_column)
    else:
        out = add_regime_column(data, price_column=price_column)
    if compact:
        out = out[[*compact_columns, "tf_1d_regime_id"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(out, output_path, index=False)
    regimes = out["tf_1d_regime_id"]
    return {
        "ok": True,
        "skipped": False,
        "input": str(input_path),
        "daily_input": str(daily_input_path) if daily_input_path else None,
        "output": str(output_path),
        "compact": compact,
        "rows": int(len(out)),
        "regime_labels": {str(key): value for key, value in REGIME_LABELS.items()},
        "regime_counts": {
            str(int(k)): int(v) for k, v in regimes.value_counts().sort_index().items()
        },
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
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write only timestamp, OHLCV, and regime id for the lightweight smoke workflow.",
    )
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
        compact=args.compact,
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
