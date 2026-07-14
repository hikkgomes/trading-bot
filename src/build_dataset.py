import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    BASE_TIMEFRAME,
    DAY_TRADE_HIGHER_TFS,
    INDICATOR_DATA_DIR,
    INDICATOR_TIMEFRAMES,
    PROCESSED_DATA_DIR,
    TIMEFRAMES,
)
from src.load_data import configure_logging

LOGGER = logging.getLogger(__name__)
TIMEFRAME_PREFIXES = {
    "1m": "tf_1m_",
    "5m": "tf_5m_",
    "15m": "tf_15m_",
    "30m": "tf_30m_",
    "60m": "tf_1h_",
    "1h": "tf_1h_",
    "240m": "tf_4h_",
    "4h": "tf_4h_",
    "1d": "tf_1d_",
    "1w": "tf_1w_",
}
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1h": 3600,
    "240m": 14400,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}
TIMEFRAME_JOIN_ORDER = (
    "1m",
    "5m",
    "15m",
    "30m",
    "60m",
    "1h",
    "240m",
    "4h",
    "1d",
    "1w",
)
TARGET_COLUMN = "target_return_next_4_bars"
TARGET_COLUMNS = (
    "target_return_next_1_bar",
    "target_return_next_4_bars",
    "target_direction_next_4_bars",
)
TARGET_HORIZON_BARS = {
    "target_return_next_1_bar": 1,
    "target_return_next_4_bars": 4,
    "target_direction_next_4_bars": 4,
}
DEFAULT_MOSTLY_EMPTY_THRESHOLD = 0.95


def prefix_columns(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    prefix = TIMEFRAME_PREFIXES.get(timeframe, f"tf_{timeframe}_")
    renamed = {column: f"{prefix}{column}" for column in df.columns if column != "timestamp"}
    return df.rename(columns=renamed)


def read_processed_timeframes(processed_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for timeframe in TIMEFRAMES:
        path = processed_dir / f"btcusdt_{timeframe}.parquet"
        if not path.exists():
            LOGGER.warning("Missing processed parquet for %s: %s", timeframe, path)
            continue
        df = normalize_timestamp_column(pd.read_parquet(path))
        frames[timeframe] = df
        LOGGER.info("Loaded %s rows for %s", len(df), timeframe)
    return frames


def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            raise ValueError("Dataframe is missing a timestamp column or timestamp index")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df.sort_values("timestamp").reset_index(drop=True)


def read_indicator_timeframes(
    indicator_dir: Path,
    timeframes: tuple = INDICATOR_TIMEFRAMES,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for timeframe in timeframes:
        path = indicator_dir / f"BTCUSDT_{timeframe}_all_indicators.parquet"
        if not path.exists():
            LOGGER.warning("Missing indicator parquet for %s: %s", timeframe, path)
            continue
        df = normalize_timestamp_column(pd.read_parquet(path))
        frames[timeframe] = df
        LOGGER.info("Loaded %s indicator rows for %s", len(df), timeframe)
    return frames


def build_multitimeframe_dataset(
    frames: dict[str, pd.DataFrame], base_timeframe: str = BASE_TIMEFRAME
) -> pd.DataFrame:
    if base_timeframe not in frames:
        raise ValueError(
            f"Base timeframe {base_timeframe!r} is missing. "
            "Run python -m src.load_data first or add the raw CSV."
        )

    base = frames[base_timeframe].sort_values("timestamp").reset_index(drop=True)
    close_column = "close"
    if close_column not in base.columns:
        raise ValueError(
            f"Base timeframe {base_timeframe!r} has no {close_column!r} column. "
            "TODO: define which price column should be used for target returns."
        )

    dataset = prefix_columns(base, base_timeframe)
    base_close_column = f"{TIMEFRAME_PREFIXES[base_timeframe]}{close_column}"
    base_close = dataset[base_close_column]
    dataset["target_return_next_1_bar"] = base_close.shift(-1) / base_close - 1
    dataset["target_return_next_4_bars"] = base_close.shift(-4) / base_close - 1
    dataset["target_direction_next_4_bars"] = (
        dataset["target_return_next_4_bars"]
        .map(lambda value: float(value > 0) if pd.notna(value) else pd.NA)
        .astype("Float64")
    )

    for timeframe in TIMEFRAME_JOIN_ORDER:
        if timeframe == base_timeframe or timeframe not in frames:
            continue
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Missing timeframe period for {timeframe!r}")
        right = prefix_columns(frames[timeframe], timeframe)
        right_shifted = right.copy()
        right_shifted["timestamp"] = right_shifted["timestamp"] + pd.Timedelta(
            seconds=TIMEFRAME_SECONDS[timeframe]
        )
        dataset = pd.merge_asof(
            dataset.sort_values("timestamp").reset_index(drop=True),
            right_shifted.sort_values("timestamp").reset_index(drop=True),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        LOGGER.info(
            "Joined %s onto %s rows by asof merge using last closed candle",
            timeframe,
            len(dataset),
        )

    dataset = keep_numeric_columns(dataset)
    dataset = dataset.dropna(subset=list(TARGET_COLUMNS)).reset_index(drop=True)
    return dataset


def keep_numeric_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = dataset.select_dtypes(include="number").columns.tolist()
    keep_columns = ["timestamp"] + numeric_columns
    return dataset.loc[:, keep_columns]


def find_constant_columns(dataset: pd.DataFrame) -> list[str]:
    constant_columns = []
    for column in dataset.columns:
        if column == "timestamp":
            continue
        if dataset[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)
    return constant_columns


def find_mostly_empty_columns(
    dataset: pd.DataFrame, threshold: float = DEFAULT_MOSTLY_EMPTY_THRESHOLD
) -> list[str]:
    empty_ratio = dataset.isna().mean()
    return [
        column
        for column, ratio in empty_ratio.items()
        if column != "timestamp" and ratio >= threshold
    ]


def find_duplicate_columns(dataset: pd.DataFrame) -> list[list[str]]:
    comparable = dataset.drop(columns=["timestamp"], errors="ignore")
    duplicate_groups = []
    visited = set()

    for index, column in enumerate(comparable.columns):
        if column in visited:
            continue
        duplicates = [
            other
            for other in comparable.columns[index + 1 :]
            if other not in visited and comparable[column].equals(comparable[other])
        ]
        if duplicates:
            group = [column] + duplicates
            duplicate_groups.append(group)
            visited.update(group)

    return duplicate_groups


def build_feature_report(
    dataset: pd.DataFrame,
    mostly_empty_threshold: float = DEFAULT_MOSTLY_EMPTY_THRESHOLD,
    include_duplicate_scan: bool = True,
) -> dict[str, object]:
    return {
        "number_of_rows": int(dataset.shape[0]),
        "number_of_columns": int(dataset.shape[1]),
        "constant_columns": find_constant_columns(dataset),
        "mostly_empty_columns": find_mostly_empty_columns(
            dataset, threshold=mostly_empty_threshold
        ),
        "mostly_empty_threshold": mostly_empty_threshold,
        "duplicate_columns_by_exact_equality": find_duplicate_columns(dataset)
        if include_duplicate_scan
        else [],
        "duplicate_column_scan_enabled": include_duplicate_scan,
    }


def drop_problem_columns(
    dataset: pd.DataFrame,
    mostly_empty_threshold: float = DEFAULT_MOSTLY_EMPTY_THRESHOLD,
    include_duplicate_scan: bool = True,
    base_timeframe: str = BASE_TIMEFRAME,
) -> pd.DataFrame:
    prefix = TIMEFRAME_PREFIXES.get(base_timeframe, f"tf_{base_timeframe}_")
    protected_columns = (
        {"timestamp"}
        | set(TARGET_COLUMNS)
        | {
            f"{prefix}open",
            f"{prefix}high",
            f"{prefix}low",
            f"{prefix}close",
            f"{prefix}volume",
            f"{prefix}atr",
            f"{prefix}atr_14",
        }
    )
    cols_to_drop = set()
    cols_to_drop.update(find_constant_columns(dataset))
    cols_to_drop.update(find_mostly_empty_columns(dataset, threshold=mostly_empty_threshold))
    if include_duplicate_scan:
        for group in find_duplicate_columns(dataset):
            cols_to_drop.update(group[1:])

    cols_to_drop = cols_to_drop - protected_columns
    if not cols_to_drop:
        return dataset

    LOGGER.info("Dropping %s problem columns before write", len(cols_to_drop))
    return dataset.drop(columns=sorted(cols_to_drop))


def write_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s rows and %s columns to %s", *dataset.shape, output_path)


def write_feature_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote feature report to %s", output_path)


def _indicator_timeframes_for_base(base_timeframe: str) -> tuple:
    if base_timeframe in DAY_TRADE_HIGHER_TFS:
        return (base_timeframe,) + DAY_TRADE_HIGHER_TFS[base_timeframe]
    return INDICATOR_TIMEFRAMES


def run(
    processed_dir: Path = PROCESSED_DATA_DIR,
    indicator_dir: Path = INDICATOR_DATA_DIR,
    input_kind: str = "processed",
    output_path: Path = PROCESSED_DATA_DIR / "train_15m.parquet",
    report_path: Path = PROCESSED_DATA_DIR / "feature_report_15m.json",
    mostly_empty_threshold: float = DEFAULT_MOSTLY_EMPTY_THRESHOLD,
    include_duplicate_scan: bool = True,
    base_timeframe: str = BASE_TIMEFRAME,
) -> None:
    if input_kind == "indicators":
        timeframes = _indicator_timeframes_for_base(base_timeframe)
        LOGGER.info("Building dataset with base=%s, timeframes=%s", base_timeframe, timeframes)
        frames = read_indicator_timeframes(indicator_dir, timeframes=timeframes)
    else:
        frames = read_processed_timeframes(processed_dir)
    dataset = build_multitimeframe_dataset(frames, base_timeframe=base_timeframe)
    dataset = drop_problem_columns(
        dataset,
        mostly_empty_threshold,
        include_duplicate_scan=include_duplicate_scan,
        base_timeframe=base_timeframe,
    )
    write_dataset(dataset, output_path)
    report = build_feature_report(
        dataset,
        mostly_empty_threshold,
        include_duplicate_scan=include_duplicate_scan,
    )
    write_feature_report(report, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-timeframe training table from cleaned parquet files."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--indicator-dir", type=Path, default=INDICATOR_DATA_DIR)
    parser.add_argument(
        "--input-kind",
        choices=("processed", "indicators"),
        default="processed",
        help="Use cleaned processed parquet files or full indicator parquet files.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--base-timeframe",
        default=BASE_TIMEFRAME,
        help="Base timeframe for the dataset. Default: 15m. Use 5m or 1m for day trading.",
    )
    parser.add_argument(
        "--mostly-empty-threshold",
        type=float,
        default=DEFAULT_MOSTLY_EMPTY_THRESHOLD,
    )
    parser.add_argument(
        "--skip-duplicate-scan",
        action="store_true",
        help="Skip exact duplicate-column detection. Useful for very wide indicator datasets.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    bt = args.base_timeframe
    output_path = args.output_path or PROCESSED_DATA_DIR / f"train_{bt}.parquet"
    report_path = args.report_path or PROCESSED_DATA_DIR / f"feature_report_{bt}.json"
    run(
        processed_dir=args.processed_dir,
        indicator_dir=args.indicator_dir,
        input_kind=args.input_kind,
        output_path=output_path,
        report_path=report_path,
        mostly_empty_threshold=args.mostly_empty_threshold,
        include_duplicate_scan=not args.skip_duplicate_scan,
        base_timeframe=bt,
    )


if __name__ == "__main__":
    main()
