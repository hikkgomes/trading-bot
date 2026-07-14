from collections.abc import Sequence

import numpy as np
import pandas as pd

POSITIVE_1M_COLUMNS = ("open", "high", "low", "close")
NON_NEGATIVE_1M_COLUMNS = (
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


def _with_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" in out.columns:
        return out
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("1m candles: missing timestamp column or DatetimeIndex")
    out = out.reset_index()
    first_column = out.columns[0]
    if first_column != "timestamp":
        out = out.rename(columns={first_column: "timestamp"})
    return out


def validate_1m_candles(
    frame: pd.DataFrame,
    *,
    candle_columns: Sequence[str],
    label: str = "1m candles",
) -> None:
    if frame.empty:
        return

    work = _with_timestamp_column(frame)
    required_columns = [column for column in candle_columns if column != "timestamp"]
    missing = [column for column in required_columns if column not in work.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")

    timestamps = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label}: invalid timestamps")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(f"{label}: timestamps must be strictly increasing")
    if not timestamps.dt.floor("1min").equals(timestamps):
        raise ValueError(f"{label}: timestamps must be aligned to 1-minute boundaries")
    if len(timestamps) > 1:
        deltas = timestamps.diff().iloc[1:]
        if not (deltas == pd.Timedelta(minutes=1)).all():
            raise ValueError(f"{label}: timestamps must be contiguous 1-minute intervals")

    numeric: dict[str, pd.Series] = {}
    for column in required_columns:
        values = pd.to_numeric(work[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError(f"{label}: {column} must be finite numeric")
        numeric[column] = values

    for column in POSITIVE_1M_COLUMNS:
        if (numeric[column] <= 0).any():
            raise ValueError(f"{label}: {column} must be positive")
    for column in NON_NEGATIVE_1M_COLUMNS:
        if (numeric[column] < 0).any():
            raise ValueError(f"{label}: {column} must be non-negative")

    high = numeric["high"]
    low = numeric["low"]
    open_ = numeric["open"]
    close = numeric["close"]
    if ((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).any():
        raise ValueError(f"{label}: OHLC values are internally inconsistent")
