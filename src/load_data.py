import argparse
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TIMEFRAMES

LOGGER = logging.getLogger(__name__)
TIMESTAMP_CANDIDATES = (
    "timestamp",
    "time",
    "date",
    "datetime",
    "time_utc",
    "utc_time",
    "open_time",
)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def standardize_column_name(column: object) -> str:
    name = str(column).strip()
    if "__" in name:
        name = name.split("__", 1)[1]
    name = name.lower()
    name = name.replace("%", "pct")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "unnamed"


def make_unique_columns(columns: Iterable[object]) -> list[str]:
    seen = {}
    unique = []
    for column in columns:
        base = standardize_column_name(column)
        candidate = base
        count = seen.get(base, 0)
        while candidate in seen:
            count += 1
            candidate = f"{base}_{count}"
        seen[base] = count
        seen[candidate] = 0
        unique.append(candidate)
    return unique


def detect_timestamp_column(df: pd.DataFrame) -> str:
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in normalized:
            return str(normalized[candidate])

    best_column = None
    best_valid = -1
    sample = df.head(1000)
    for column in sample.columns:
        parsed = parse_timestamp_series(sample[column])
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_column = str(column)

    if best_column is None or best_valid == 0:
        raise ValueError("Could not detect a timestamp column")
    LOGGER.warning(
        "Detected timestamp column %r by parsing sample values. TODO: confirm this is correct for this export.",
        best_column,
    )
    return best_column


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        median_abs = numeric.dropna().abs().median()
        if pd.isna(median_abs):
            return pd.to_datetime(series, errors="coerce", utc=True)
        if median_abs >= 1e17:
            unit = "ns"
        elif median_abs >= 1e14:
            unit = "us"
        elif median_abs >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)

    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = detect_timestamp_column(df)
    parsed_timestamp = parse_timestamp_series(df[timestamp_column])

    cleaned = df.copy()
    cleaned.columns = make_unique_columns(cleaned.columns)
    cleaned.insert(0, "timestamp", parsed_timestamp)

    original_timestamp_name = standardize_column_name(timestamp_column)
    if original_timestamp_name in cleaned.columns[1:]:
        cleaned = cleaned.drop(columns=[original_timestamp_name])

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["timestamp"])
    invalid_timestamps = before - len(cleaned)
    if invalid_timestamps:
        LOGGER.warning("Dropped %s rows with invalid timestamps", invalid_timestamps)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    exact_duplicates = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    timestamp_duplicates = before - len(cleaned)

    if exact_duplicates or timestamp_duplicates:
        LOGGER.info(
            "Dropped duplicates: exact=%s timestamp=%s",
            exact_duplicates,
            timestamp_duplicates,
        )

    return cleaned.sort_values("timestamp").reset_index(drop=True)


def find_raw_file(raw_dir: Path, timeframe: str) -> Path | None:
    expected = raw_dir / f"btcusdt_{timeframe}.csv"
    if expected.exists():
        return expected

    matches = sorted(raw_dir.glob(f"*_{timeframe}.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple raw files found for {timeframe}: {matches}. "
            "TODO: keep only one file per timeframe or define a merge policy."
        )
    return None


def clean_timeframe_file(raw_file: Path, output_file: Path) -> pd.DataFrame:
    LOGGER.info("Reading %s", raw_file)
    df = pd.read_csv(raw_file)
    cleaned = clean_dataframe(df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_file, index=False)
    LOGGER.info("Wrote %s rows to %s", len(cleaned), output_file)
    return cleaned


def run(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR) -> None:
    for timeframe in TIMEFRAMES:
        raw_file = find_raw_file(raw_dir, timeframe)
        if raw_file is None:
            LOGGER.warning("Missing raw CSV for %s in %s", timeframe, raw_dir)
            continue
        clean_timeframe_file(raw_file, processed_dir / f"btcusdt_{timeframe}.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean TradingView CSV exports.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
