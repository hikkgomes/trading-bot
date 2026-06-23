import argparse
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import requests

from src.config import PROJECT_ROOT
# Import configuration and building functions from build_binance_indicator_dataset
import build_binance_indicator_dataset as bbid

LOGGER = logging.getLogger("update_candles")

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def fetch_recent_candles(
    symbol: str,
    market: str,
    start_time_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetches 1m candles from Binance starting from start_time_ms (inclusive).
    """
    if market == "futures":
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_time_ms,
        "limit": limit,
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Binance API error: status={response.status_code} response={response.text}"
        )

    data = response.json()
    if not data:
        return pd.DataFrame()

    # Parse klines data
    df = pd.DataFrame(data, columns=bbid.BINANCE_COLUMNS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])

    for column in bbid.CANDLE_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[bbid.CANDLE_COLUMNS]

def update_1m_candles() -> pd.DataFrame:
    """
    Loads existing 1m candles, fetches missing candles from Binance REST API,
    merges them, and writes the updated dataset back to Parquet.
    """
    candle_path = bbid.CANDLE_DIR / f"{bbid.SYMBOL}_1m.parquet"
    LOGGER.info("Loading existing 1m candles from %s", candle_path)
    df_existing = bbid.load_existing_1m_candles()

    if df_existing is None:
        raise FileNotFoundError(
            f"No existing 1m candles found at {candle_path}. Run build_binance_indicator_dataset.py first."
        )

    last_timestamp = df_existing.index.max()
    # Refetch from the last stored candle INCLUSIVE: if a previous run persisted
    # a still-forming candle, this replaces it with the final closed values.
    start_time_ms = int(last_timestamp.value // 10**6)
    current_time_ms = int(time.time() * 1000)

    LOGGER.info("Last candle timestamp: %s (ms: %d)", last_timestamp, last_timestamp.value // 10**6)
    LOGGER.info("Fetching new candles starting from: %s", pd.to_datetime(start_time_ms, unit="ms", utc=True))

    new_frames = []
    fetch_start = start_time_ms
    calls = 0

    while fetch_start < current_time_ms:
        LOGGER.info("Fetching candles starting from ms %d...", fetch_start)
        try:
            df_new = fetch_recent_candles(bbid.SYMBOL, bbid.MARKET, fetch_start, limit=1000)
        except Exception as e:
            LOGGER.error("Error fetching candles: %s", e)
            break

        if df_new.empty:
            LOGGER.info("No more candles returned from Binance API.")
            break

        new_frames.append(df_new)
        calls += 1

        # Get the timestamp of the last retrieved candle
        last_fetched_ms = int(df_new["timestamp"].max().value // 10**6)
        LOGGER.info("Fetched %d candles. Last timestamp: %s", len(df_new), pd.to_datetime(last_fetched_ms, unit="ms", utc=True))

        # Check if we've reached the end of available data (if the last fetched time didn't advance)
        if last_fetched_ms <= fetch_start:
            break

        fetch_start = last_fetched_ms + 60000
        # Rate limit friendly sleep
        time.sleep(0.2)

    if new_frames:
        df_new_all = pd.concat(new_frames, ignore_index=True)
        # Drop duplicates by timestamp
        df_new_all = df_new_all.drop_duplicates("timestamp", keep="last")
        df_new_all = df_new_all.sort_values("timestamp").set_index("timestamp")
        df_new_all.index.name = "timestamp"

        # Concat existing and new; keep="last" prefers the freshly fetched
        # values so a previously stored partial candle gets overwritten.
        df_updated = pd.concat([df_existing, df_new_all], axis=0)
        df_updated = df_updated.loc[~df_updated.index.duplicated(keep="last")].sort_index()
        # Never persist the still-forming candle.
        cutoff = pd.Timestamp.now(tz="UTC").floor("1min")
        df_updated = df_updated.loc[df_updated.index < cutoff]

        LOGGER.info("Merged candles. Previous rows: %d, New rows: %d, Merged rows: %d",
                    len(df_existing), len(df_new_all), len(df_updated))

        # Save back to 1m Parquet
        candle_path.parent.mkdir(parents=True, exist_ok=True)
        df_updated.to_parquet(candle_path)
        LOGGER.info("Saved updated 1m candles to %s", candle_path)
        return df_updated
    else:
        LOGGER.info("No new candles fetched. Dataset is up to date.")
        return df_existing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally update candles and rebuild indicators.")
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Only rebuild indicator parquets for these timeframes (e.g. 5m 15m 30m 1h). "
             "Default rebuilds all. The chunked 1m build is by far the slowest.",
    )
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    LOGGER.info("Starting hybrid candle update for %s (%s)", bbid.SYMBOL, bbid.MARKET)

    # 1. Update 1m candles parquet file
    df_1m = update_1m_candles()

    # 2. Rebuild higher-timeframe candles
    LOGGER.info("Rebuilding higher timeframe candles...")
    datasets = bbid.build_timeframes(df_1m)

    # 3. Rebuild indicator datasets (optionally restricted to a TF subset)
    if args.timeframes:
        unknown = set(args.timeframes) - set(datasets)
        if unknown:
            raise ValueError(f"Unknown timeframes {sorted(unknown)}; available: {sorted(datasets)}")
        datasets = {tf: frame for tf, frame in datasets.items() if tf in set(args.timeframes)}
    LOGGER.info("Rebuilding indicator datasets for: %s", ", ".join(datasets))
    bbid.build_indicator_files(datasets)

    LOGGER.info("Update complete!")

if __name__ == "__main__":
    main()
