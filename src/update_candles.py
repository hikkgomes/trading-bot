import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Import configuration and building functions from build_binance_indicator_dataset
import build_binance_indicator_dataset as bbid
from src.candle_validation import validate_1m_candles
from src.parquet_io import write_parquet_atomic

LOGGER = logging.getLogger("update_candles")
DEFAULT_FETCH_ERROR_GRACE_SECONDS = 24 * 60 * 60


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def existing_1m_candle_path() -> Path:
    return bbid.CANDLE_DIR / f"{bbid.SYMBOL}_1m.parquet"


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
    if not isinstance(data, list):
        raise ValueError(f"{symbol} {market} fetched 1m candles: Binance response must be a list")
    expected_width = len(bbid.BINANCE_COLUMNS)
    for index, row in enumerate(data):
        if not isinstance(row, list | tuple) or len(row) != expected_width:
            raise ValueError(f"{symbol} {market} fetched 1m candles: malformed Binance row {index}")

    # Parse klines data
    df = pd.DataFrame(data, columns=bbid.BINANCE_COLUMNS)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    open_times = df["open_time"].to_numpy(dtype="float64")
    if df["open_time"].isna().any() or not np.isfinite(open_times).all() or (open_times < 0).any():
        raise ValueError(
            f"{symbol} {market} fetched 1m candles: open_time must be finite and non-negative"
        )

    for column in bbid.CANDLE_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df[bbid.CANDLE_COLUMNS]
    validate_1m_candles(
        out, candle_columns=bbid.CANDLE_COLUMNS, label=f"{symbol} {market} fetched 1m candles"
    )
    return out


def _normalise_1m_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
        first_column = out.columns[0]
        if first_column != "timestamp" and isinstance(frame.index, pd.DatetimeIndex):
            out = out.rename(columns={first_column: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    validate_1m_candles(out, candle_columns=bbid.CANDLE_COLUMNS, label="normalised 1m candles")
    out = out.set_index("timestamp")
    out.index.name = "timestamp"
    return out


def _seed_age_seconds(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    if isinstance(frame.index, pd.DatetimeIndex):
        last_timestamp = frame.index.max()
    elif "timestamp" in frame.columns:
        last_timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").max()
    else:
        return None
    if pd.isna(last_timestamp):
        return None
    last_timestamp = pd.Timestamp(last_timestamp)
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    else:
        last_timestamp = last_timestamp.tz_convert("UTC")
    return max(0.0, (pd.Timestamp.now(tz="UTC") - last_timestamp).total_seconds())


def _seed_is_fresh(
    frame: pd.DataFrame, *, max_age_seconds: int = DEFAULT_FETCH_ERROR_GRACE_SECONDS
) -> bool:
    age_seconds = _seed_age_seconds(frame)
    return age_seconds is not None and age_seconds <= max_age_seconds


def bootstrap_1m_candles(days: int) -> pd.DataFrame:
    """Create a bounded recent 1m seed via REST. This is intentionally not a
    replacement for the full historical monthly download."""
    if days <= 0:
        raise ValueError("bootstrap days must be positive")
    if days > 366:
        raise ValueError("bootstrap days must be <= 366")

    candle_path = existing_1m_candle_path()
    current_time_ms = int(time.time() * 1000)
    start_time_ms = current_time_ms - int(days * 24 * 60 * 60 * 1000)
    fetch_start = start_time_ms
    new_frames = []
    fetch_error = None

    LOGGER.info(
        "Bootstrapping %s %s 1m candles for the last %d days",
        bbid.SYMBOL,
        bbid.MARKET,
        days,
    )
    while fetch_start < current_time_ms:
        try:
            df_new = fetch_recent_candles(bbid.SYMBOL, bbid.MARKET, fetch_start, limit=1000)
        except Exception as exc:
            LOGGER.error("Error bootstrapping candles: %s", exc)
            fetch_error = str(exc)
            break
        if df_new.empty:
            break
        new_frames.append(df_new)
        last_fetched_ms = int(pd.to_datetime(df_new["timestamp"], utc=True).max().value // 10**6)
        if last_fetched_ms <= fetch_start:
            break
        fetch_start = last_fetched_ms + 60000
        time.sleep(0.2)

    if not new_frames:
        empty = pd.DataFrame()
        empty.attrs["fetch_error"] = fetch_error or "no bootstrap candles returned"
        empty.attrs["fetched_rows"] = 0
        empty.attrs["bootstrap_days"] = days
        return empty

    df_bootstrap = _normalise_1m_frame(pd.concat(new_frames, ignore_index=True))
    cutoff = pd.Timestamp.now(tz="UTC").floor("1min")
    df_bootstrap = df_bootstrap.loc[df_bootstrap.index < cutoff]
    if df_bootstrap.empty:
        df_bootstrap.attrs["fetch_error"] = "bootstrap returned no closed candles"
        df_bootstrap.attrs["fetched_rows"] = 0
        df_bootstrap.attrs["bootstrap_days"] = days
        return df_bootstrap

    validate_1m_candles(
        df_bootstrap,
        candle_columns=bbid.CANDLE_COLUMNS,
        label=f"{bbid.SYMBOL} {bbid.MARKET} bootstrapped 1m candles",
    )
    candle_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(df_bootstrap, candle_path)
    df_bootstrap.attrs["fetched_rows"] = int(sum(len(frame) for frame in new_frames))
    df_bootstrap.attrs["bootstrap_days"] = days
    if fetch_error:
        df_bootstrap.attrs["fetch_error"] = fetch_error
    LOGGER.info("Saved bootstrapped 1m candles to %s", candle_path)
    return df_bootstrap


def update_1m_candles() -> pd.DataFrame:
    """
    Loads existing 1m candles, fetches missing candles from Binance REST API,
    merges them, and writes the updated dataset back to Parquet.
    """
    candle_path = existing_1m_candle_path()
    LOGGER.info("Loading existing 1m candles from %s", candle_path)
    df_existing = bbid.load_existing_1m_candles()

    if df_existing is None:
        raise FileNotFoundError(
            f"No existing 1m candles found at {candle_path}. Run build_binance_indicator_dataset.py first."
        )

    validate_1m_candles(
        df_existing,
        candle_columns=bbid.CANDLE_COLUMNS,
        label=f"{bbid.SYMBOL} {bbid.MARKET} existing 1m candles",
    )
    df_existing = _normalise_1m_frame(df_existing)
    # Refetch from the last stored candle INCLUSIVE: if a previous run persisted
    # a still-forming candle, this replaces it with the final closed values.
    last_timestamp = df_existing.index.max()
    start_time_ms = int(last_timestamp.value // 10**6)
    current_time_ms = int(time.time() * 1000)

    LOGGER.info("Last candle timestamp: %s (ms: %d)", last_timestamp, last_timestamp.value // 10**6)
    LOGGER.info(
        "Fetching new candles starting from: %s", pd.to_datetime(start_time_ms, unit="ms", utc=True)
    )

    new_frames = []
    fetch_start = start_time_ms
    calls = 0
    fetch_error = None

    while fetch_start < current_time_ms:
        LOGGER.info("Fetching candles starting from ms %d...", fetch_start)
        try:
            df_new = fetch_recent_candles(bbid.SYMBOL, bbid.MARKET, fetch_start, limit=1000)
        except Exception as e:
            LOGGER.error("Error fetching candles: %s", e)
            fetch_error = str(e)
            break

        if df_new.empty:
            LOGGER.info("No more candles returned from Binance API.")
            break

        new_frames.append(df_new)
        calls += 1

        # Get the timestamp of the last retrieved candle
        last_fetched_ms = int(df_new["timestamp"].max().value // 10**6)
        LOGGER.info(
            "Fetched %d candles. Last timestamp: %s",
            len(df_new),
            pd.to_datetime(last_fetched_ms, unit="ms", utc=True),
        )

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
        validate_1m_candles(
            df_updated,
            candle_columns=bbid.CANDLE_COLUMNS,
            label=f"{bbid.SYMBOL} {bbid.MARKET} merged 1m candles",
        )

        LOGGER.info(
            "Merged candles. Previous rows: %d, New rows: %d, Merged rows: %d",
            len(df_existing),
            len(df_new_all),
            len(df_updated),
        )

        # Save back to 1m Parquet
        candle_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(df_updated, candle_path)
        df_updated.attrs["fetched_rows"] = int(len(df_new_all))
        if fetch_error:
            df_updated.attrs["fetch_error"] = fetch_error
        LOGGER.info("Saved updated 1m candles to %s", candle_path)
        return df_updated
    else:
        LOGGER.info("No new candles fetched. Dataset is up to date.")
        df_existing.attrs["fetched_rows"] = 0
        if fetch_error:
            df_existing.attrs["fetch_error"] = fetch_error
        return df_existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally update candles and rebuild indicators."
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Only rebuild indicator parquets for these timeframes (e.g. 5m 15m 30m 1h). "
        "Default rebuilds all. The chunked 1m build is by far the slowest.",
    )
    parser.add_argument(
        "--skip-if-missing",
        action="store_true",
        help="Exit successfully without network calls when the seed 1m candle parquet is missing.",
    )
    parser.add_argument(
        "--market",
        choices=("spot", "futures"),
        default=None,
        help="Market dataset to update. Defaults to build_binance_indicator_dataset.MARKET.",
    )
    parser.add_argument(
        "--bootstrap-days",
        type=int,
        default=0,
        help="If the 1m seed is missing, bootstrap this many recent days via REST "
        "(bounded to 366). Default 0 keeps skip/fail behavior.",
    )
    return parser.parse_args()


def _validate_timeframes(timeframes: list[str] | None) -> None:
    if not timeframes:
        return
    unknown = set(timeframes) - set(bbid.TIMEFRAMES)
    if unknown:
        raise ValueError(
            f"Unknown timeframes {sorted(unknown)}; available: {sorted(bbid.TIMEFRAMES)}"
        )


def run_update(
    timeframes: list[str] | None = None,
    *,
    skip_if_missing: bool = False,
    market: str | None = None,
    bootstrap_days: int = 0,
) -> dict[str, Any]:
    if market is not None:
        bbid.configure_dataset(market=market, legacy_fallback=True)
    if bootstrap_days < 0:
        raise ValueError("bootstrap_days must be non-negative")
    _validate_timeframes(timeframes)
    candle_path = existing_1m_candle_path()
    if skip_if_missing and not candle_path.exists() and bootstrap_days == 0:
        return {
            "ok": True,
            "skipped": True,
            "reason": "missing_seed_dataset",
            "candle_path": str(candle_path),
            "symbol": bbid.SYMBOL,
            "market": bbid.MARKET,
            "timeframes": timeframes or list(bbid.TIMEFRAMES),
        }

    LOGGER.info("Starting hybrid candle update for %s (%s)", bbid.SYMBOL, bbid.MARKET)

    # 1. Update 1m candles parquet file
    if not candle_path.exists() and bootstrap_days > 0:
        df_1m = bootstrap_1m_candles(bootstrap_days)
    else:
        df_1m = update_1m_candles()
    fetch_error = df_1m.attrs.get("fetch_error")
    if fetch_error:
        seed_age_seconds = _seed_age_seconds(df_1m)
        if _seed_is_fresh(df_1m):
            return {
                "ok": True,
                "skipped": True,
                "reason": "fetch_error_existing_seed_fresh",
                "warning": str(fetch_error),
                "candle_path": str(candle_path),
                "symbol": bbid.SYMBOL,
                "market": bbid.MARKET,
                "rows_1m": int(len(df_1m)),
                "fetched_rows": int(df_1m.attrs.get("fetched_rows") or 0),
                "bootstrap_days": int(df_1m.attrs.get("bootstrap_days") or 0),
                "seed_age_seconds": round(float(seed_age_seconds or 0.0), 3),
                "timeframes": timeframes or list(bbid.TIMEFRAMES),
            }
        return {
            "ok": False,
            "skipped": True,
            "reason": "fetch_error",
            "error": str(fetch_error),
            "candle_path": str(candle_path),
            "symbol": bbid.SYMBOL,
            "market": bbid.MARKET,
            "rows_1m": int(len(df_1m)),
            "fetched_rows": int(df_1m.attrs.get("fetched_rows") or 0),
            "bootstrap_days": int(df_1m.attrs.get("bootstrap_days") or 0),
            "timeframes": timeframes or list(bbid.TIMEFRAMES),
        }
    if df_1m.empty:
        return {
            "ok": False,
            "skipped": True,
            "reason": "empty_seed_dataset",
            "error": "1m candle dataset is empty; refusing to rebuild derived candles or indicators",
            "candle_path": str(candle_path),
            "symbol": bbid.SYMBOL,
            "market": bbid.MARKET,
            "rows_1m": 0,
            "fetched_rows": int(df_1m.attrs.get("fetched_rows") or 0),
            "bootstrap_days": int(df_1m.attrs.get("bootstrap_days") or 0),
            "timeframes": timeframes or list(bbid.TIMEFRAMES),
        }

    # 2. Rebuild higher-timeframe candles
    LOGGER.info("Rebuilding higher timeframe candles...")
    datasets = bbid.build_timeframes(df_1m, timeframes=timeframes)

    # 3. Rebuild indicator datasets.
    LOGGER.info("Rebuilding indicator datasets for: %s", ", ".join(datasets))
    bbid.build_indicator_files(datasets)

    LOGGER.info("Update complete!")
    return {
        "ok": True,
        "skipped": False,
        "candle_path": str(candle_path),
        "symbol": bbid.SYMBOL,
        "market": bbid.MARKET,
        "rows_1m": int(len(df_1m)),
        "fetched_rows": int(df_1m.attrs.get("fetched_rows") or 0),
        "bootstrap_days": int(df_1m.attrs.get("bootstrap_days") or 0),
        "timeframes": list(datasets),
    }


def main():
    configure_logging()
    args = parse_args()
    report = run_update(
        args.timeframes,
        skip_if_missing=args.skip_if_missing,
        market=args.market,
        bootstrap_days=args.bootstrap_days,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
