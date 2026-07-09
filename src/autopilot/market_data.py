"""Lightweight market-data status checks for operator visibility."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

import build_binance_indicator_dataset as bbid
from src.config import candle_data_dir, indicator_data_dir

DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60
DEFAULT_REQUIRED_INDICATOR_FEATURES = {
    "1m": ["volume_z_20"],
    "5m": ["volume_z_20"],
    "15m": ["volume_z_20"],
}
INDICATOR_CANARY_FEATURES = ["volume_z_20"]
DEFAULT_BOOTSTRAP_DAYS = {
    "futures": 90,
    "spot": 365,
}
DEFAULT_BOOTSTRAP_TIMEFRAMES = {
    "futures": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    "spot": ["1h", "4h", "1d", "1w"],
}


def default_1m_candle_path(*, market: str | None = None, symbol: str | None = None) -> Path:
    selected_symbol = symbol or bbid.SYMBOL
    if market is None:
        return bbid.CANDLE_DIR / f"{selected_symbol}_1m.parquet"
    return candle_data_dir(selected_symbol, market, legacy_fallback=True) / f"{selected_symbol}_1m.parquet"


def default_indicator_dir(*, market: str | None = None, symbol: str | None = None) -> Path:
    selected_symbol = symbol or bbid.SYMBOL
    if market is None:
        return bbid.INDICATOR_DIR
    return indicator_data_dir(selected_symbol, market, legacy_fallback=True)


def bootstrap_command_for_market(
    market: str,
    *,
    days: int | None = None,
    timeframes: Sequence[str] | None = None,
) -> list[str]:
    bootstrap_days = days if days is not None else DEFAULT_BOOTSTRAP_DAYS.get(market, 90)
    command = [
        ".venv/bin/python",
        "-m",
        "src.update_candles",
        "--market",
        market,
        "--bootstrap-days",
        str(bootstrap_days),
    ]
    selected_timeframes = list(timeframes or DEFAULT_BOOTSTRAP_TIMEFRAMES.get(market, ()))
    if selected_timeframes:
        command.extend(["--timeframes", *selected_timeframes])
    return command


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _validate_1m_seed_schema(parquet_file: pq.ParquetFile) -> None:
    names = set(parquet_file.schema.names)
    missing = [column for column in bbid.CANDLE_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"missing required 1m candle columns: {missing}")


def _timestamp_bounds_from_metadata(
    path: Path,
    *,
    parquet_file: pq.ParquetFile | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
    parquet_file = pq.ParquetFile(path) if parquet_file is None else parquet_file
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    rows = 0
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        rows += row_group.num_rows
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema != "timestamp":
                continue
            stats = column.statistics
            if not stats or not stats.has_min_max:
                return None
            minimum = _utc_timestamp(stats.min)
            maximum = _utc_timestamp(stats.max)
            first_timestamp = minimum if first_timestamp is None else min(first_timestamp, minimum)
            last_timestamp = maximum if last_timestamp is None else max(last_timestamp, maximum)
            break
        else:
            return None
    if first_timestamp is None or last_timestamp is None:
        return None
    return first_timestamp, last_timestamp, rows


def _timestamp_bounds_from_column(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    table = pq.read_table(path, columns=["timestamp"])
    timestamps = pd.to_datetime(table.column("timestamp").to_pandas(), utc=True, errors="coerce")
    timestamps = timestamps.dropna()
    if timestamps.empty:
        raise ValueError("empty timestamp column")
    return timestamps.min(), timestamps.max(), int(len(timestamps))


def build_market_data_status(
    path: Path | None = None,
    *,
    market: str | None = None,
    now: dt.datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")

    selected_market = market or bbid.MARKET
    path = default_1m_candle_path(market=market) if path is None else path
    now = utc_now() if now is None else now.astimezone(dt.timezone.utc)
    status: dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "exists": path.exists(),
        "symbol": bbid.SYMBOL,
        "market": selected_market,
        "max_age_seconds": max_age_seconds,
    }
    if not path.exists():
        status["reason"] = "missing_seed_dataset"
        status["remediation"] = {
            "action": "bootstrap_market_data",
            "command": bootstrap_command_for_market(selected_market),
            "note": "Run once on the server with network access to create the 1m seed and derived indicators.",
        }
        return status

    try:
        parquet_file = pq.ParquetFile(path)
        _validate_1m_seed_schema(parquet_file)
        bounds = _timestamp_bounds_from_metadata(path, parquet_file=parquet_file)
        if bounds is None:
            bounds = _timestamp_bounds_from_column(path)
    except ValueError as exc:
        status.update(reason="invalid_seed_dataset", error=str(exc))
        return status
    except Exception as exc:
        status.update(reason="read_error", error=str(exc))
        return status

    first_timestamp, last_timestamp, rows = bounds
    last_dt = last_timestamp.to_pydatetime()
    age_seconds = (now - last_dt).total_seconds()
    if age_seconds < 0:
        status.update(
            {
                "ok": False,
                "reason": "future_timestamp",
                "rows": rows,
                "first_timestamp": first_timestamp.isoformat(),
                "last_timestamp": last_timestamp.isoformat(),
                "age_seconds": round(age_seconds, 3),
            }
        )
        return status
    status.update(
        {
            "ok": age_seconds <= max_age_seconds,
            "reason": "fresh" if age_seconds <= max_age_seconds else "stale",
            "rows": rows,
            "first_timestamp": first_timestamp.isoformat(),
            "last_timestamp": last_timestamp.isoformat(),
            "age_seconds": round(age_seconds, 3),
        }
    )
    return status


def build_market_data_statuses(
    markets: list[str] | tuple[str, ...] | set[str],
    *,
    now: dt.datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> dict[str, dict[str, Any]]:
    return {
        market: build_market_data_status(
            market=market,
            now=now,
            max_age_seconds=max_age_seconds,
        )
        for market in sorted(set(markets))
    }


def _indicator_path(indicator_dir: Path, timeframe: str) -> Path:
    return indicator_dir / f"{bbid.SYMBOL}_{timeframe}_all_indicators.parquet"


def _command_value(command: Sequence[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for part in command:
        if part.startswith(prefix):
            value = part[len(prefix):]
            return value or None
    try:
        index = command.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    value = command[value_index]
    if value.startswith("--"):
        return None
    return value


def _command_values(command: Sequence[str], flag: str) -> list[str]:
    prefix = f"{flag}="
    inline_values = [part[len(prefix):] for part in command if part.startswith(prefix) and part[len(prefix):]]
    if inline_values:
        return inline_values
    try:
        index = command.index(flag)
    except ValueError:
        return []
    values: list[str] = []
    for value in command[index + 1:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def required_indicator_features_by_market(
    markets: Iterable[str],
    *,
    jobs: Iterable[Any] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Infer indicator canary checks from configured market-data jobs.

    Without explicit jobs we keep the historical futures defaults. With jobs, a
    market is checked only for the timeframes its enabled update command builds.
    """
    selected_markets = sorted(set(markets))
    if not jobs:
        return {market: dict(DEFAULT_REQUIRED_INDICATOR_FEATURES) for market in selected_markets}

    by_market: dict[str, dict[str, list[str]]] = {market: {} for market in selected_markets}
    selected = set(selected_markets)
    for job in jobs:
        if not getattr(job, "enabled", True):
            continue
        command = list(getattr(job, "command", []) or [])
        market = _command_value(command, "--market")
        if market not in selected:
            continue
        for timeframe in _command_values(command, "--timeframes"):
            by_market[market][timeframe] = list(INDICATOR_CANARY_FEATURES)
    return {
        market: features if features else dict(DEFAULT_REQUIRED_INDICATOR_FEATURES)
        for market, features in by_market.items()
    }


def _parquet_schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def build_indicator_feature_status(
    required_features: dict[str, list[str]] | None = None,
    *,
    market: str | None = None,
    indicator_dir: Path | None = None,
) -> dict[str, Any]:
    required_features = required_features or DEFAULT_REQUIRED_INDICATOR_FEATURES
    selected_market = market or bbid.MARKET
    indicator_dir = default_indicator_dir(market=market) if indicator_dir is None else indicator_dir

    status: dict[str, Any] = {
        "ok": True,
        "indicator_dir": str(indicator_dir),
        "symbol": bbid.SYMBOL,
        "market": selected_market,
        "timeframes": {},
    }
    for timeframe, required in required_features.items():
        path = _indicator_path(indicator_dir, timeframe)
        entry: dict[str, Any] = {
            "ok": False,
            "path": str(path),
            "exists": path.exists(),
            "required_features": list(required),
            "missing_features": list(required),
        }
        if not path.exists():
            entry["reason"] = "missing_indicator_dataset"
            status["ok"] = False
            status["timeframes"][timeframe] = entry
            continue

        try:
            columns = _parquet_schema_names(path)
        except Exception as exc:
            entry.update(reason="read_error", error=str(exc))
            status["ok"] = False
            status["timeframes"][timeframe] = entry
            continue

        missing = [feature for feature in required if feature not in columns]
        available_flow_features = sorted(
            column
            for column in columns
            if column.startswith(("volume_z_", "cvd_", "taker_imbalance"))
        )
        entry.update(
            {
                "ok": not missing,
                "reason": "ready" if not missing else "missing_required_features",
                "missing_features": missing,
                "available_flow_features": available_flow_features,
                "column_count": len(columns),
            }
        )
        status["ok"] = bool(status["ok"] and entry["ok"])
        status["timeframes"][timeframe] = entry
    return status


def build_indicator_feature_statuses(
    markets: list[str] | tuple[str, ...] | set[str],
    required_features: dict[str, list[str]] | None = None,
    *,
    required_features_by_market: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        market: build_indicator_feature_status(
            required_features_by_market.get(market, required_features)
            if required_features_by_market
            else required_features,
            market=market,
        )
        for market in sorted(set(markets))
    }
