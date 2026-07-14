import gc
import os
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import talib
from talib import abstract

from src.candle_validation import validate_1m_candles
from src.config import candle_data_dir, indicator_data_dir, normalize_market
from src.parquet_io import atomic_output_path, write_parquet_atomic

# =========================
# CONFIG
# =========================

# Change these for another pair or market.
SYMBOL = "BTCUSDT"
MARKET = normalize_market(os.environ.get("TRADING_DATA_MARKET", "futures"))  # "futures" or "spot"

# Inclusive monthly Binance download range, formatted as YYYY-MM.
START_MONTH = "2016-01"
END_MONTH = "2026-03"

# Higher timeframes are rebuilt from downloaded Binance 1m candles.
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# 1m stays chunked to avoid building a huge all-indicators DataFrame in RAM.
CHUNKED_INDICATOR_TIMEFRAMES = {"1m"}
CHUNK_SIZE = 100_000
WARMUP_ROWS = 1_000

TIMEPERIOD_VARIANTS = [5, 7, 10, 14, 20, 21, 30, 50, 100, 200]

REBUILD_CANDLES = True
REBUILD_INDICATORS = True
SAVE_CSV = False

RAW_DIR = Path("data/raw") / MARKET / SYMBOL / "1m"
CANDLE_DIR = candle_data_dir(SYMBOL, MARKET, legacy_fallback=True)
INDICATOR_DIR = indicator_data_dir(SYMBOL, MARKET, legacy_fallback=True)


def configure_dataset(
    *,
    symbol: str | None = None,
    market: str | None = None,
    legacy_fallback: bool = True,
) -> None:
    """Update module-level paths for scripts/tests that select a market at runtime."""
    global SYMBOL, MARKET, RAW_DIR, CANDLE_DIR, INDICATOR_DIR
    if symbol is not None:
        SYMBOL = symbol
    if market is not None:
        MARKET = normalize_market(market)
    else:
        MARKET = normalize_market(MARKET)
    RAW_DIR = Path("data/raw") / MARKET / SYMBOL / "1m"
    CANDLE_DIR = candle_data_dir(SYMBOL, MARKET, legacy_fallback=legacy_fallback)
    INDICATOR_DIR = indicator_data_dir(SYMBOL, MARKET, legacy_fallback=legacy_fallback)


TIMEFRAME_RULES = {
    "1m": None,
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-MON",
}

TIMEFRAME_PERIODS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
}

CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


# =========================
# BINANCE DOWNLOAD
# =========================


def parse_month(value: str) -> tuple[int, int]:
    year_text, month_text = value.split("-", 1)
    year = int(year_text)
    month = int(month_text)
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month: {value}")
    return year, month


def iter_months(start_month: str, end_month: str) -> Iterable[tuple[int, int]]:
    start_year, start_month_number = parse_month(start_month)
    end_year, end_month_number = parse_month(end_month)
    current_year = start_year
    current_month = start_month_number

    while (current_year, current_month) <= (end_year, end_month_number):
        yield current_year, current_month
        current_month += 1
        if current_month == 13:
            current_year += 1
            current_month = 1


def get_base_url() -> str:
    if MARKET == "futures":
        return "https://data.binance.vision/data/futures/um/monthly/klines"
    if MARKET == "spot":
        return "https://data.binance.vision/data/spot/monthly/klines"
    raise ValueError("MARKET must be 'futures' or 'spot'")


def download_binance_1m() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_url = get_base_url()

    for year, month in iter_months(START_MONTH, END_MONTH):
        filename = f"{SYMBOL}-1m-{year}-{month:02d}.zip"
        url = f"{base_url}/{SYMBOL}/1m/{filename}"
        path = RAW_DIR / filename

        if path.exists():
            print(f"Already exists: {filename}", flush=True)
            continue

        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            path.write_bytes(response.content)
            print(f"Downloaded: {filename}", flush=True)
        else:
            print(f"Missing from Binance: {filename} status={response.status_code}", flush=True)


# =========================
# CANDLE BUILDING
# =========================


def read_binance_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file inside {path}")
        with archive.open(csv_files[0]) as file:
            df = pd.read_csv(file, header=None, names=BINANCE_COLUMNS)

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    open_times = df["open_time"].to_numpy(dtype="float64")
    if df["open_time"].isna().any() or not np.isfinite(open_times).all() or (open_times < 0).any():
        raise ValueError(f"{path}: open_time must be finite and non-negative")

    for column in CANDLE_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df[CANDLE_COLUMNS]
    validate_1m_candles(out, candle_columns=CANDLE_COLUMNS, label=f"{path} raw 1m candles")
    return out


def load_1m_from_zips() -> pd.DataFrame:
    zip_files = sorted(RAW_DIR.glob(f"{SYMBOL}-1m-*.zip"))
    if not zip_files:
        raise RuntimeError(f"No Binance ZIP files found in {RAW_DIR}")

    frames = []
    for index, path in enumerate(zip_files, start=1):
        print(f"Reading ZIP {index}/{len(zip_files)}: {path.name}", flush=True)
        frames.append(read_binance_zip(path))

    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    validate_1m_candles(
        df, candle_columns=CANDLE_COLUMNS, label=f"{SYMBOL} {MARKET} loaded raw 1m candles"
    )
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df


def load_existing_1m_candles() -> pd.DataFrame | None:
    path = CANDLE_DIR / f"{SYMBOL}_1m.parquet"
    if not path.exists():
        return None

    print(f"Loading existing Binance-derived 1m candles: {path}", flush=True)
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        df = df.reset_index()
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")

    validate_1m_candles(df, candle_columns=CANDLE_COLUMNS, label=f"{path} stored 1m candles")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df


def build_timeframes(
    df_1m: pd.DataFrame,
    timeframes: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    first_timestamp = df_1m.index.min()
    complete_until = df_1m.index.max() + TIMEFRAME_PERIODS["1m"]
    selected = set(timeframes) if timeframes else set(TIMEFRAMES)
    unknown = selected - set(TIMEFRAMES)
    if unknown:
        raise ValueError(f"Unknown timeframes {sorted(unknown)}; available: {sorted(TIMEFRAMES)}")

    for timeframe in TIMEFRAMES:
        if timeframe not in selected:
            continue
        print(f"Building candles for {timeframe}", flush=True)
        if timeframe == "1m":
            df = df_1m.copy()
        else:
            df = df_1m.resample(
                TIMEFRAME_RULES[timeframe],
                label="left",
                closed="left",
            ).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "quote_asset_volume": "sum",
                    "number_of_trades": "sum",
                    "taker_buy_base_volume": "sum",
                    "taker_buy_quote_volume": "sum",
                }
            )
            period = TIMEFRAME_PERIODS[timeframe]
            df = df[(df.index >= first_timestamp) & (df.index + period <= complete_until)]

        df = df.dropna(subset=["open", "high", "low", "close"])
        df.index.name = "timestamp"

        output_path = CANDLE_DIR / f"{SYMBOL}_{timeframe}.parquet"
        write_parquet_atomic(df, output_path)
        datasets[timeframe] = df
        print(
            f"Saved candles: {output_path} | rows={len(df):,}, cols={len(df.columns):,}",
            flush=True,
        )

    return datasets


# =========================
# TA-LIB INDICATORS
# =========================


def make_talib_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "open": df["open"].astype(float).values,
        "high": df["high"].astype(float).values,
        "low": df["low"].astype(float).values,
        "close": df["close"].astype(float).values,
        "volume": df["volume"].astype(float).values,
    }


def normalise_output(result, output_names, prefix: str) -> dict[str, object]:
    output = {}

    if isinstance(result, pd.DataFrame):
        for column in result.columns:
            output[f"{prefix}_{column}"] = result[column].values
        return output

    if isinstance(result, pd.Series):
        output[prefix] = result.values
        return output

    if isinstance(result, dict):
        for key, values in result.items():
            output[f"{prefix}_{key}"] = values
        return output

    if isinstance(result, tuple | list):
        for index, values in enumerate(result):
            name = (
                output_names[index]
                if output_names and index < len(output_names)
                else f"output_{index}"
            )
            output[f"{prefix}_{name}"] = values
        return output

    values = np.asarray(result)
    if values.ndim == 1:
        output[prefix] = values
    elif values.ndim == 2:
        for index in range(values.shape[1]):
            name = (
                output_names[index]
                if output_names and index < len(output_names)
                else f"output_{index}"
            )
            output[f"{prefix}_{name}"] = values[:, index]

    return output


def run_indicator(
    function_name: str,
    inputs: dict[str, np.ndarray],
    params: dict[str, object] | None = None,
    suffix: str | None = None,
) -> dict[str, object]:
    function = abstract.Function(function_name)
    prefix = function_name.lower()
    if suffix:
        prefix = f"{prefix}_{suffix}"

    result = function(inputs, **params) if params else function(inputs)
    return normalise_output(result, function.output_names, prefix)


def get_variant_candidates() -> list[str]:
    candidates = []
    for function_name in talib.get_functions():
        try:
            function = abstract.Function(function_name)
            if "timeperiod" in function.parameters:
                candidates.append(function_name)
        except Exception:
            pass
    return candidates


FLOW_WINDOWS = [5, 20, 50, 100]
_PERIOD_FEATURE_RE = re.compile(r"^(.+)_(\d+)(?:_(.+))?$")


def flow_feature_names() -> set[str]:
    names = {"taker_buy_ratio", "taker_imbalance", "avg_trade_size"}
    for window in FLOW_WINDOWS:
        names.update(
            {
                f"cvd_{window}",
                f"taker_imbalance_ma_{window}",
                f"volume_z_{window}",
                f"trades_z_{window}",
                f"avg_trade_size_z_{window}",
            }
        )
    return names


def build_flow_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Order-flow features from Binance taker/volume candle fields.

    Captures who is aggressing (taker buys vs sells) — information OHLCV
    indicators cannot see. Strictly backward-looking rolling windows; returns
    {} when the flow columns are absent so callers can merge unconditionally.
    """
    required = {"volume", "taker_buy_base_volume", "number_of_trades"}
    if not required.issubset(df.columns):
        return {}
    volume = df["volume"].astype(float)
    taker_buy = df["taker_buy_base_volume"].astype(float)
    trades = df["number_of_trades"].astype(float)
    safe_volume = volume.replace(0, np.nan)
    # Net aggressive buy volume: buys minus sells, where sells = volume - buys.
    delta = 2.0 * taker_buy - volume
    imbalance = delta / safe_volume
    features: dict[str, pd.Series] = {
        "taker_buy_ratio": taker_buy / safe_volume,
        "taker_imbalance": imbalance,
        "avg_trade_size": volume / trades.replace(0, np.nan),
    }
    for window in FLOW_WINDOWS:
        rolling_volume = volume.rolling(window).sum().replace(0, np.nan)
        features[f"cvd_{window}"] = delta.rolling(window).sum() / rolling_volume
        features[f"taker_imbalance_ma_{window}"] = imbalance.rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        features[f"volume_z_{window}"] = (volume - volume.rolling(window).mean()) / volume_std
        trades_std = trades.rolling(window).std().replace(0, np.nan)
        features[f"trades_z_{window}"] = (trades - trades.rolling(window).mean()) / trades_std
        size = features["avg_trade_size"]
        size_std = size.rolling(window).std().replace(0, np.nan)
        features[f"avg_trade_size_z_{window}"] = (size - size.rolling(window).mean()) / size_std
    return features


def _normalised_feature_names(function_name: str, suffix: str | None = None) -> set[str]:
    function = abstract.Function(function_name)
    prefix = function_name.lower()
    if suffix:
        prefix = f"{prefix}_{suffix}"
    output_names = list(function.output_names or [])
    if len(output_names) > 1:
        return {f"{prefix}_{name}" for name in output_names}
    return {prefix}


def _required_indicator_specs(required_features: set[str]) -> set[tuple[str, int | None]]:
    all_functions = talib.get_functions()
    functions_by_lower = {name.lower(): name for name in all_functions}
    variant_candidates = set(get_variant_candidates())
    specs: set[tuple[str, int | None]] = set()
    unresolved = {
        feature
        for feature in required_features
        if feature not in CANDLE_COLUMNS and feature not in flow_feature_names()
    }

    for feature in list(unresolved):
        match = _PERIOD_FEATURE_RE.match(feature)
        if not match:
            continue
        base_name = match.group(1)
        period = int(match.group(2))
        function_name = functions_by_lower.get(base_name)
        if function_name in variant_candidates:
            specs.add((function_name, period))
            unresolved.remove(feature)

    for feature in list(unresolved):
        for function_name in all_functions:
            if feature in _normalised_feature_names(function_name):
                specs.add((function_name, None))
                unresolved.remove(feature)
                break
    return specs


def build_indicator_features(
    df: pd.DataFrame,
    timeframe: str,
    required_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    inputs = make_talib_inputs(df)
    features = {}
    required = None if required_features is None else set(required_features)
    if required is None or required & flow_feature_names():
        flow_features = build_flow_features(df)
        if required is None:
            features.update(flow_features)
        else:
            features.update(
                {name: values for name, values in flow_features.items() if name in required}
            )

    if required is not None:
        for function_name, period in sorted(
            _required_indicator_specs(required),
            key=lambda item: (item[0], -1 if item[1] is None else item[1]),
        ):
            try:
                if period is None:
                    features.update(run_indicator(function_name, inputs))
                else:
                    features.update(
                        run_indicator(
                            function_name,
                            inputs,
                            params={"timeperiod": period},
                            suffix=str(period),
                        )
                    )
            except Exception as exc:
                print(f"[{timeframe}] Skipped required {function_name}: {exc}", flush=True)
        feature_df = pd.DataFrame(features, index=df.index)
        final = pd.concat([df, feature_df], axis=1)
        final = final.replace([np.inf, -np.inf], np.nan)
        return final.loc[:, ~final.columns.duplicated()]

    all_functions = talib.get_functions()
    variant_candidates = get_variant_candidates()
    for index, function_name in enumerate(all_functions, start=1):
        print(f"[{timeframe}] Default {index}/{len(all_functions)}: {function_name}", flush=True)
        try:
            features.update(run_indicator(function_name, inputs))
        except Exception as exc:
            print(f"[{timeframe}] Skipped default {function_name}: {exc}", flush=True)

    variant_total = len(variant_candidates) * len(TIMEPERIOD_VARIANTS)
    variant_count = 0
    for function_name in variant_candidates:
        for period in TIMEPERIOD_VARIANTS:
            variant_count += 1
            print(
                f"[{timeframe}] Variant {variant_count}/{variant_total}: {function_name}_{period}",
                flush=True,
            )
            try:
                features.update(
                    run_indicator(
                        function_name,
                        inputs,
                        params={"timeperiod": period},
                        suffix=str(period),
                    )
                )
            except Exception as exc:
                print(f"[{timeframe}] Skipped variant {function_name}_{period}: {exc}", flush=True)

    feature_df = pd.DataFrame(features, index=df.index)
    final = pd.concat([df, feature_df], axis=1)
    final = final.replace([np.inf, -np.inf], np.nan)
    return final.loc[:, ~final.columns.duplicated()]


def reduce_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if column == "timestamp":
            continue
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].astype("float32")
        elif pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype("int32")
    return df


def base_columns_for(df: pd.DataFrame) -> list[str]:
    return [column for column in CANDLE_COLUMNS if column in df.columns]


def build_full_indicator_file(timeframe: str, df: pd.DataFrame) -> None:
    output_path = INDICATOR_DIR / f"{SYMBOL}_{timeframe}_all_indicators.parquet"
    if output_path.exists() and not REBUILD_INDICATORS:
        print(f"Output already exists, skipping: {output_path}", flush=True)
        return

    print(f"Building indicators for {timeframe}", flush=True)
    indicators = build_indicator_features(df, timeframe)
    indicators = reduce_numeric_dtypes(indicators)
    write_parquet_atomic(indicators, output_path)
    print(
        f"Saved indicators: {output_path} | rows={len(indicators):,}, cols={len(indicators.columns):,}",
        flush=True,
    )

    if SAVE_CSV:
        indicators.to_csv(INDICATOR_DIR / f"{SYMBOL}_{timeframe}_all_indicators.csv")


def build_chunked_indicator_file(timeframe: str, df: pd.DataFrame) -> None:
    output_path = INDICATOR_DIR / f"{SYMBOL}_{timeframe}_all_indicators.parquet"
    if output_path.exists() and not REBUILD_INDICATORS:
        print(f"Output already exists, skipping: {output_path}", flush=True)
        return

    if "timestamp" not in df.columns:
        df = df.reset_index()

    df = df[base_columns_for(df)].copy()
    for column in df.columns:
        if column != "timestamp":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError(f"Cannot build empty {timeframe} indicator artifact")
    with atomic_output_path(output_path) as temporary_path:
        writer = None
        try:
            for start in range(0, total_rows, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total_rows)
                warmup_start = max(0, start - WARMUP_ROWS)
                print(
                    f"[{timeframe}] Chunk rows {start:,} to {end:,} (warmup from {warmup_start:,})",
                    flush=True,
                )

                chunk = df.iloc[warmup_start:end].copy()
                out_chunk = build_indicator_features(chunk, timeframe)
                rows_to_drop = start - warmup_start
                if rows_to_drop > 0:
                    out_chunk = out_chunk.iloc[rows_to_drop:].copy()

                out_chunk = reduce_numeric_dtypes(out_chunk)
                table = pa.Table.from_pandas(out_chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
                writer.write_table(table)

                print(
                    f"[{timeframe}] Wrote chunk rows={len(out_chunk):,}, cols={len(out_chunk.columns):,}",
                    flush=True,
                )

                del chunk
                del out_chunk
                del table
                gc.collect()
        finally:
            if writer is not None:
                writer.close()

    print(f"Saved chunked indicators: {output_path}", flush=True)


def build_indicator_files(datasets: dict[str, pd.DataFrame]) -> None:
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)
    # Iterate only the timeframes actually provided (callers may pass a
    # subset), preserving the canonical TIMEFRAMES build order.
    for timeframe in TIMEFRAMES:
        if timeframe not in datasets:
            continue
        df = datasets[timeframe]
        if timeframe in CHUNKED_INDICATOR_TIMEFRAMES:
            build_chunked_indicator_file(timeframe, df)
        else:
            build_full_indicator_file(timeframe, df)


# =========================
# MAIN
# =========================


def main() -> None:
    print(
        f"Building {SYMBOL} {MARKET} dataset from Binance monthly 1m klines "
        f"{START_MONTH} through {END_MONTH}",
        flush=True,
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)

    if REBUILD_CANDLES:
        download_binance_1m()
        df_1m = load_1m_from_zips()
    else:
        df_1m = load_existing_1m_candles()
        if df_1m is None:
            raise FileNotFoundError(
                f"Missing existing 1m candle file: {CANDLE_DIR / f'{SYMBOL}_1m.parquet'}"
            )

    datasets = build_timeframes(df_1m)
    build_indicator_files(datasets)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
