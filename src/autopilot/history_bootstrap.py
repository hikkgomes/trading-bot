"""Bootstrap the exact native Binance history used by autonomous research.

This deliberately fetches native timeframe klines.  Reconstructing several
years of 5m/1h/4h/1d research data from a multi-year 1m archive wastes disk,
RAM, bandwidth, and server time. Each dataset is validated, checkpointed while
downloading, and atomically published with the feature inventory used by the
configured typed strategy grammar.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import build_binance_indicator_dataset as bbid
from research_exploration.strategy_grammar import DEFAULT_FEATURES
from src.autopilot.io import write_json_atomic
from src.autopilot.research_factory import DEFAULT_CONFIG as DEFAULT_FACTORY_CONFIG
from src.autopilot.research_factory import load_factory_config, search_spaces_for_symbol
from src.autopilot.research_history_contract import generated_history_contract
from src.config import candle_data_dir, indicator_data_dir
from src.parquet_io import write_parquet_atomic

LOGGER = logging.getLogger("autopilot.history_bootstrap")
SYMBOL = "BTCUSDT"
WARMUP_BARS = 250
OPERATIONAL_SEED_DAYS = 7
DEFAULT_CHECKPOINT_PAGES = 20
DEFAULT_REQUEST_DELAY_SECONDS = 0.2
DEFAULT_MAX_REQUEST_PAGES = 5_000
KLINE_LIMIT = 1_000
MAX_EXCHANGE_GAP_DURATION = pd.Timedelta(hours=12)
MAX_EXCHANGE_MISSING_FRACTION = 0.001
TIMEFRAME_DELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
}


class HistoryBootstrapDeferred(RuntimeError):
    """The bounded bootstrap saved a checkpoint and should resume later."""


def _defer_if_deadline_reached(
    deadline_monotonic: float | None,
    *,
    market: str,
    timeframe: str,
) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise HistoryBootstrapDeferred(
            f"{market} {timeframe}: bounded bootstrap time budget exhausted"
        )


@dataclass(frozen=True)
class HistoryRequirement:
    market: str
    timeframe: str
    start: pd.Timestamp
    required_features: frozenset[str]
    scenario_names: tuple[str, ...]
    build_indicators: bool = True


@dataclass(frozen=True)
class ExchangeGapPolicy:
    max_gap_duration: pd.Timedelta
    max_missing_fraction: float


EXCHANGE_GAP_POLICIES = {
    (SYMBOL, "spot", "1h"): ExchangeGapPolicy(
        max_gap_duration=MAX_EXCHANGE_GAP_DURATION,
        max_missing_fraction=MAX_EXCHANGE_MISSING_FRACTION,
    )
}


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _aligned_open_at_or_before(value: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    value = _utc_timestamp(value)
    if timeframe == "1w":
        return value.normalize() - pd.Timedelta(days=value.weekday())
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    return value.floor(aliases[timeframe])


def _last_closed_open(now: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    return _aligned_open_at_or_before(now, timeframe) - TIMEFRAME_DELTAS[timeframe]


def build_default_requirements(
    *,
    config_path: Path = DEFAULT_FACTORY_CONFIG,
    markets: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    exclude_timeframes: Iterable[str] | None = None,
    now: str | pd.Timestamp | None = None,
    symbol: str = SYMBOL,
    search_spaces: Iterable[Any] | None = None,
) -> tuple[HistoryRequirement, ...]:
    """Derive native datasets from the authoritative factory search spaces."""

    factory_config = load_factory_config(config_path)
    selected_markets = (
        set(markets)
        if markets is not None
        else {space.market for space in factory_config.search_spaces}
    )
    unknown_markets = selected_markets - {"spot", "futures"}
    if unknown_markets:
        raise ValueError(f"unsupported markets: {sorted(unknown_markets)}")
    selected_timeframes = set(timeframes) if timeframes is not None else set(TIMEFRAME_DELTAS)
    excluded_timeframes = set(exclude_timeframes or ())
    unknown_timeframes = (selected_timeframes | excluded_timeframes) - set(TIMEFRAME_DELTAS)
    if unknown_timeframes:
        raise ValueError(f"unsupported timeframes: {sorted(unknown_timeframes)}")
    overlap = selected_timeframes & excluded_timeframes
    if timeframes is not None and overlap:
        raise ValueError(f"timeframes cannot be both included and excluded: {sorted(overlap)}")
    selected_timeframes -= excluded_timeframes
    if not selected_timeframes:
        raise ValueError("timeframe selection cannot be empty")

    starts: dict[tuple[str, str], pd.Timestamp] = {}
    features: dict[tuple[str, str], set[str]] = defaultdict(set)
    scenario_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    resolved_spaces = (
        tuple(search_spaces)
        if search_spaces is not None
        else search_spaces_for_symbol(factory_config, symbol)
    )
    for space in resolved_spaces:
        if space.market not in selected_markets:
            continue
        configured_timeframes = {
            space.base_timeframe,
            space.regime_timeframe,
            space.setup_timeframe,
            space.trigger_timeframe,
        }
        unsupported = configured_timeframes - set(TIMEFRAME_DELTAS)
        if unsupported:
            raise ValueError(
                f"{space.name}: history bootstrap does not support timeframes {sorted(unsupported)}"
            )
        space_start = _utc_timestamp(generated_history_contract(space)["start"])
        for timeframe in configured_timeframes:
            if timeframe not in selected_timeframes:
                continue
            key = (space.market, timeframe)
            warm_start = space_start - WARMUP_BARS * TIMEFRAME_DELTAS[timeframe]
            starts[key] = min(starts.get(key, warm_start), warm_start)
            features[key].update(DEFAULT_FEATURES)
            scenario_names[key].add(space.name)

    # The runtime freshness watchdog intentionally uses a tiny 1m canary.  Spot
    # research itself remains coarse: this is seven days, not a multi-year 1m
    # reconstruction.  Its single volume canary is used by readiness checks.
    reference = _utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if "spot" in selected_markets and "1m" in selected_timeframes:
        key = ("spot", "1m")
        if key not in starts:
            starts[key] = _aligned_open_at_or_before(
                reference - pd.Timedelta(days=OPERATIONAL_SEED_DAYS), "1m"
            )
            scenario_names[key].add("operational_freshness_seed")
            features[key].add("volume_z_20")

    requirements = []
    for market, timeframe in sorted(
        starts,
        key=lambda item: (item[0], TIMEFRAME_DELTAS[item[1]]),
    ):
        required_values = set(features[(market, timeframe)])
        if required_values:
            # Readiness checks use this cheap flow feature as a schema canary
            # for every scheduled indicator parquet.
            required_values.add("volume_z_20")
        required = frozenset(required_values)
        requirements.append(
            HistoryRequirement(
                market=market,
                timeframe=timeframe,
                start=_aligned_open_at_or_before(starts[(market, timeframe)], timeframe),
                required_features=required,
                scenario_names=tuple(sorted(scenario_names[(market, timeframe)])),
                build_indicators=bool(required),
            )
        )
    return tuple(requirements)


def _frame_with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
        if "timestamp" not in out.columns and len(out.columns):
            out = out.rename(columns={out.columns[0]: "timestamp"})
    return out


def validate_candle_frame(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    label: str,
    require_contiguous: bool = True,
) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(f"{label}: unsupported timeframe {timeframe!r}")
    if frame.empty:
        raise ValueError(f"{label}: candle dataset is empty")
    out = _frame_with_timestamp(frame)
    missing = [column for column in bbid.CANDLE_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"{label}: missing required candle columns {missing}")
    timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label}: invalid timestamps")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(f"{label}: timestamps must be strictly increasing")
    aligned = [_aligned_open_at_or_before(value, timeframe) for value in timestamps]
    if any(actual != expected for actual, expected in zip(timestamps, aligned, strict=True)):
        raise ValueError(f"{label}: timestamps must align to {timeframe} boundaries")
    if require_contiguous and len(timestamps) > 1:
        deltas = timestamps.diff().iloc[1:]
        if not (deltas == TIMEFRAME_DELTAS[timeframe]).all():
            raise ValueError(f"{label}: timestamps must be contiguous {timeframe} intervals")

    numeric: dict[str, pd.Series] = {}
    for column in bbid.CANDLE_COLUMNS[1:]:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError(f"{label}: {column} must be finite numeric")
        numeric[column] = values
        out[column] = values
    for column in ("open", "high", "low", "close"):
        if (numeric[column] <= 0).any():
            raise ValueError(f"{label}: {column} must be positive")
    for column in (
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if (numeric[column] < 0).any():
            raise ValueError(f"{label}: {column} must be non-negative")
    if (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    ).any():
        raise ValueError(f"{label}: OHLC values are internally inconsistent")
    out["timestamp"] = timestamps
    return out[bbid.CANDLE_COLUMNS].set_index("timestamp")


def audit_exchange_gaps(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    label: str,
    max_gap_duration: pd.Timedelta = MAX_EXCHANGE_GAP_DURATION,
    max_missing_fraction: float | None = MAX_EXCHANGE_MISSING_FRACTION,
) -> list[dict[str, Any]]:
    """Describe bounded exchange-side outages without inventing tradable bars.

    Binance can omit klines while spot trading is suspended for maintenance.
    Keeping those intervals absent prevents a backtest from entering or exiting
    on synthetic prices. Large or pervasive gaps still fail closed as likely
    corruption, and every accepted range remains visible in the manifest.
    """

    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(f"{label}: unsupported timeframe {timeframe!r}")
    if frame.empty or len(frame.index) < 2:
        return []
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: expected a DatetimeIndex for gap auditing")

    period = TIMEFRAME_DELTAS[timeframe]
    gaps: list[dict[str, Any]] = []
    missing_total = 0
    previous = _utc_timestamp(frame.index[0])
    for raw_current in frame.index[1:]:
        current = _utc_timestamp(raw_current)
        delta = current - previous
        if delta == period:
            previous = current
            continue
        if delta <= period or delta % period != pd.Timedelta(0):
            raise ValueError(f"{label}: timestamps do not follow the {timeframe} cadence")
        missing_bars = int(delta / period) - 1
        missing_duration = missing_bars * period
        if missing_duration > max_gap_duration:
            raise ValueError(
                f"{label}: exchange gap of {missing_bars} {timeframe} bars exceeds "
                f"the {max_gap_duration} safety limit"
            )
        missing_total += missing_bars
        gaps.append(
            {
                "start": (previous + period).isoformat(),
                "end": (current - period).isoformat(),
                "missing_bars": missing_bars,
                "missing_duration_seconds": int(missing_duration.total_seconds()),
            }
        )
        previous = current

    expected_rows = len(frame) + missing_total
    missing_fraction = missing_total / expected_rows
    if max_missing_fraction is not None and missing_fraction > max_missing_fraction:
        raise ValueError(
            f"{label}: missing-candle fraction {missing_fraction:.6f} exceeds "
            f"the {max_missing_fraction:.6f} safety limit"
        )
    return gaps


def _exchange_gap_policy(
    symbol: str,
    market: str,
    timeframe: str,
) -> ExchangeGapPolicy | None:
    return EXCHANGE_GAP_POLICIES.get((symbol, market, timeframe))


def _endpoint(market: str) -> str:
    if market == "futures":
        return "https://fapi.binance.com/fapi/v1/klines"
    if market == "spot":
        return "https://api.binance.com/api/v3/klines"
    raise ValueError(f"unsupported market {market!r}")


def fetch_kline_page(
    *,
    symbol: str,
    market: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = KLINE_LIMIT,
    request_get: Callable[..., Any] = requests.get,
) -> pd.DataFrame:
    response = request_get(
        _endpoint(market),
        params={
            "symbol": symbol,
            "interval": timeframe,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": int(limit),
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Binance {market} {timeframe} API error: status={response.status_code} "
            f"response={response.text}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Binance {market} {timeframe} response must be a list")
    if not payload:
        return pd.DataFrame(columns=bbid.CANDLE_COLUMNS)
    for index, row in enumerate(payload):
        if not isinstance(row, list | tuple) or len(row) != len(bbid.BINANCE_COLUMNS):
            raise ValueError(f"Binance {market} {timeframe} malformed kline row {index}")
    raw = pd.DataFrame(payload, columns=bbid.BINANCE_COLUMNS)
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    if raw["open_time"].isna().any():
        raise ValueError(f"Binance {market} {timeframe} open_time must be numeric")
    raw["timestamp"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True, errors="coerce")
    for column in bbid.CANDLE_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    gap_policy = _exchange_gap_policy(symbol, market, timeframe)
    page = validate_candle_frame(
        raw[bbid.CANDLE_COLUMNS],
        timeframe,
        label=f"Binance {market} {timeframe} page",
        require_contiguous=gap_policy is None,
    )
    if gap_policy is not None:
        audit_exchange_gaps(
            page,
            timeframe,
            label=f"Binance {market} {timeframe} page",
            max_gap_duration=gap_policy.max_gap_duration,
            max_missing_fraction=None,
        )
    return page


def _load_candles(
    path: Path,
    timeframe: str,
    *,
    symbol: str,
    market: str,
    checkpoint: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=bbid.CANDLE_COLUMNS[1:], index=pd.DatetimeIndex([], name="timestamp", tz="UTC")
        )
    if path.is_symlink():
        raise ValueError(f"refusing symlinked candle input: {path}")
    gap_policy = _exchange_gap_policy(symbol, market, timeframe)
    frame = validate_candle_frame(
        pd.read_parquet(path),
        timeframe,
        label=str(path),
        require_contiguous=not checkpoint and gap_policy is None,
    )
    # A checkpoint can contain several disjoint repair ranges. Its schema,
    # ordering and alignment are validated above; cadence is assessed only
    # after it is merged with the published dataset.
    if not checkpoint and gap_policy is not None:
        audit_exchange_gaps(
            frame,
            timeframe,
            label=str(path),
            max_gap_duration=gap_policy.max_gap_duration,
            max_missing_fraction=gap_policy.max_missing_fraction,
        )
    return frame


def _merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=bbid.CANDLE_COLUMNS[1:], index=pd.DatetimeIndex([], name="timestamp", tz="UTC")
        )
    merged = pd.concat(nonempty).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged.index.name = "timestamp"
    return merged


def _manifest_prefix_complete(
    path: Path,
    requested_start: pd.Timestamp,
    available: pd.DataFrame,
) -> bool:
    """Trust a prior empty-prefix probe only while its saved prefix survives.

    A manifest may legitimately record that Binance history starts after the
    requested date.  It must not, however, hide accidental truncation of the
    candle parquet.  Binding the probe to the first saved timestamp preserves
    the listing-date case and forces a prefix repair when initial rows vanish.
    """

    if not path.exists() or path.is_symlink() or available.empty:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checked = _utc_timestamp(payload["prefix_checked_from"])
        recorded_first = _utc_timestamp(payload["first_timestamp"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    actual_first = _utc_timestamp(available.index.min())
    return bool(
        payload.get("prefix_complete") is True
        and checked <= requested_start
        and actual_first <= recorded_first
    )


def _missing_ranges(
    available: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    prefix_complete: bool,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    period = TIMEFRAME_DELTAS[timeframe]
    if end < start:
        return []
    if available.empty:
        return [(start, end)]
    selected = available.loc[(available.index >= start) & (available.index <= end)]
    if selected.empty:
        return [(start, end)]
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    first = selected.index.min()
    if first > start and not prefix_complete:
        ranges.append((start, min(end, first - period)))
    previous = first
    for current in selected.index[1:]:
        if current - previous > period:
            ranges.append((previous + period, min(end, current - period)))
        previous = current
    # Refetch the final saved candle to repair a prior partial value, then add
    # every newly closed bar.
    ranges.append((max(start, selected.index.max()), end))
    return [(left, right) for left, right in ranges if left <= right]


def _download_ranges(
    requirement: HistoryRequirement,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    symbol: str,
    checkpoint_path: Path,
    checkpoint: pd.DataFrame,
    checkpoint_pages: int,
    request_delay_seconds: float,
    max_request_pages: int,
    fetch_page: Callable[..., pd.DataFrame],
    request_pages_used: int = 0,
    deadline_monotonic: float | None = None,
) -> tuple[pd.DataFrame, int]:
    if checkpoint_pages <= 0:
        raise ValueError("checkpoint_pages must be positive")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")
    if (
        not isinstance(max_request_pages, int)
        or isinstance(max_request_pages, bool)
        or not 0 < max_request_pages <= 100_000
    ):
        raise ValueError("max_request_pages must be an integer in [1, 100000]")
    if (
        not isinstance(request_pages_used, int)
        or isinstance(request_pages_used, bool)
        or not 0 <= request_pages_used <= max_request_pages
    ):
        raise ValueError("request_pages_used must be an integer within the page budget")
    period_ms = int(TIMEFRAME_DELTAS[requirement.timeframe].total_seconds() * 1_000)
    downloaded = checkpoint
    pages_since_checkpoint = 0
    request_pages = request_pages_used
    try:
        for range_start, range_end in ranges:
            cursor_ms = int(range_start.value // 1_000_000)
            end_ms = int(range_end.value // 1_000_000)
            while cursor_ms <= end_ms:
                _defer_if_deadline_reached(
                    deadline_monotonic,
                    market=requirement.market,
                    timeframe=requirement.timeframe,
                )
                if request_pages >= max_request_pages:
                    raise RuntimeError(
                        f"{requirement.market} {requirement.timeframe}: API page budget "
                        f"exhausted at {max_request_pages}; rerun to resume the checkpoint"
                    )
                page = fetch_page(
                    symbol=symbol,
                    market=requirement.market,
                    timeframe=requirement.timeframe,
                    start_ms=cursor_ms,
                    end_ms=end_ms,
                    limit=KLINE_LIMIT,
                )
                request_pages += 1
                if page.empty:
                    break
                page = page.loc[(page.index >= range_start) & (page.index <= range_end)]
                if page.empty:
                    break
                last_ms = int(page.index.max().value // 1_000_000)
                if last_ms < cursor_ms:
                    raise RuntimeError(
                        f"{requirement.market} {requirement.timeframe}: Binance page did not advance"
                    )
                downloaded = _merge_frames(downloaded, page)
                cursor_ms = last_ms + period_ms
                pages_since_checkpoint += 1
                if pages_since_checkpoint >= checkpoint_pages:
                    write_parquet_atomic(downloaded, checkpoint_path)
                    pages_since_checkpoint = 0
                if request_delay_seconds:
                    time.sleep(request_delay_seconds)
    except Exception:
        if not downloaded.empty:
            write_parquet_atomic(downloaded, checkpoint_path)
        raise
    if not downloaded.empty:
        write_parquet_atomic(downloaded, checkpoint_path)
    return downloaded, request_pages


def sync_requirement(
    requirement: HistoryRequirement,
    *,
    symbol: str = SYMBOL,
    now: str | pd.Timestamp | None = None,
    checkpoint_pages: int = DEFAULT_CHECKPOINT_PAGES,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_request_pages: int = DEFAULT_MAX_REQUEST_PAGES,
    fetch_page: Callable[..., pd.DataFrame] = fetch_kline_page,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    reference = _utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    last_closed = _last_closed_open(reference, requirement.timeframe)
    candle_dir = candle_data_dir(symbol, requirement.market, legacy_fallback=False)
    indicator_dir = indicator_data_dir(symbol, requirement.market, legacy_fallback=False)
    candle_path = candle_dir / f"{symbol}_{requirement.timeframe}.parquet"
    checkpoint_path = candle_dir / f".{symbol}_{requirement.timeframe}.history_checkpoint.parquet"
    manifest_path = candle_dir / f".{symbol}_{requirement.timeframe}.history.json"
    existing = _load_candles(
        candle_path,
        requirement.timeframe,
        symbol=symbol,
        market=requirement.market,
    )
    checkpoint = _load_candles(
        checkpoint_path,
        requirement.timeframe,
        symbol=symbol,
        market=requirement.market,
        checkpoint=True,
    )
    available = _merge_frames(existing, checkpoint)
    prefix_complete = _manifest_prefix_complete(
        manifest_path,
        requirement.start,
        available,
    )
    ranges = _missing_ranges(
        available,
        start=requirement.start,
        end=last_closed,
        timeframe=requirement.timeframe,
        prefix_complete=prefix_complete,
    )
    downloaded, request_pages_used = (
        _download_ranges(
            requirement,
            ranges,
            symbol=symbol,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            checkpoint_pages=checkpoint_pages,
            request_delay_seconds=request_delay_seconds,
            max_request_pages=max_request_pages,
            fetch_page=fetch_page,
            deadline_monotonic=deadline_monotonic,
        )
        if ranges
        else (checkpoint, 0)
    )
    merged = _merge_frames(existing, downloaded)
    merged = merged.loc[(merged.index >= requirement.start) & (merged.index <= last_closed)]
    _defer_if_deadline_reached(
        deadline_monotonic,
        market=requirement.market,
        timeframe=requirement.timeframe,
    )
    gap_policy = _exchange_gap_policy(symbol, requirement.market, requirement.timeframe)
    merged = validate_candle_frame(
        merged,
        requirement.timeframe,
        label=f"{symbol} {requirement.market} {requirement.timeframe} merged history",
        require_contiguous=gap_policy is None,
    )
    targeted_gap_ranges: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    if gap_policy is not None:
        provisional_gaps = audit_exchange_gaps(
            merged,
            requirement.timeframe,
            label=f"{symbol} {requirement.market} {requirement.timeframe} merged history",
            max_gap_duration=gap_policy.max_gap_duration,
            max_missing_fraction=None,
        )
        provisional_ranges = [
            (_utc_timestamp(item["start"]), _utc_timestamp(item["end"]))
            for item in provisional_gaps
        ]
        requested_ranges = {(_utc_timestamp(left), _utc_timestamp(right)) for left, right in ranges}
        targeted_gap_ranges.update(
            gap_range for gap_range in provisional_ranges if gap_range in requested_ranges
        )
        repair_ranges = [
            gap_range for gap_range in provisional_ranges if gap_range not in requested_ranges
        ]
        if repair_ranges:
            downloaded, request_pages_used = _download_ranges(
                requirement,
                repair_ranges,
                symbol=symbol,
                checkpoint_path=checkpoint_path,
                checkpoint=downloaded,
                checkpoint_pages=checkpoint_pages,
                request_delay_seconds=request_delay_seconds,
                max_request_pages=max_request_pages,
                fetch_page=fetch_page,
                request_pages_used=request_pages_used,
                deadline_monotonic=deadline_monotonic,
            )
            targeted_gap_ranges.update(repair_ranges)
            merged = _merge_frames(existing, downloaded)
            merged = merged.loc[(merged.index >= requirement.start) & (merged.index <= last_closed)]
            merged = validate_candle_frame(
                merged,
                requirement.timeframe,
                label=f"{symbol} {requirement.market} {requirement.timeframe} merged history",
                require_contiguous=False,
            )
        exchange_gaps = audit_exchange_gaps(
            merged,
            requirement.timeframe,
            label=f"{symbol} {requirement.market} {requirement.timeframe} merged history",
            max_gap_duration=gap_policy.max_gap_duration,
            max_missing_fraction=gap_policy.max_missing_fraction,
        )
        for item in exchange_gaps:
            gap_start = _utc_timestamp(item["start"])
            gap_end = _utc_timestamp(item["end"])
            if not any(
                checked_start <= gap_start and checked_end >= gap_end
                for checked_start, checked_end in targeted_gap_ranges
            ):
                raise RuntimeError(
                    f"{symbol} {requirement.market} {requirement.timeframe}: "
                    "refusing to publish an exchange gap without a targeted recheck"
                )
    else:
        exchange_gaps = []
    if merged.index.max() < last_closed:
        raise RuntimeError(
            f"{symbol} {requirement.market} {requirement.timeframe}: incomplete Binance history; "
            f"last candle {merged.index.max().isoformat()} is before required "
            f"{last_closed.isoformat()}"
        )
    write_parquet_atomic(merged, candle_path)

    indicator_path: Path | None = None
    indicator_columns = 0
    if requirement.build_indicators:
        _defer_if_deadline_reached(
            deadline_monotonic,
            market=requirement.market,
            timeframe=requirement.timeframe,
        )
        indicator_path = indicator_dir / f"{symbol}_{requirement.timeframe}_all_indicators.parquet"
        indicators = bbid.build_indicator_features(
            merged,
            requirement.timeframe,
            required_features=requirement.required_features,
        )
        missing_features = sorted(requirement.required_features - set(indicators.columns))
        if missing_features:
            raise ValueError(
                f"{symbol} {requirement.market} {requirement.timeframe}: "
                f"indicator builder did not produce {missing_features}"
            )
        indicators = bbid.reduce_numeric_dtypes(indicators)
        write_parquet_atomic(indicators, indicator_path)
        indicator_columns = int(len(indicators.columns))

    exchange_gap_policy = (
        {
            "scope": f"{symbol}:{requirement.market}:{requirement.timeframe}",
            "max_gap_duration_seconds": int(gap_policy.max_gap_duration.total_seconds()),
            "max_missing_fraction": gap_policy.max_missing_fraction,
            "requires_targeted_recheck": True,
        }
        if gap_policy is not None
        else None
    )
    exchange_gap_rechecked_at = reference.isoformat() if exchange_gaps else None
    write_json_atomic(
        manifest_path,
        {
            "version": 2,
            "symbol": symbol,
            "market": requirement.market,
            "timeframe": requirement.timeframe,
            "prefix_checked_from": requirement.start.isoformat(),
            "prefix_complete": True,
            "first_timestamp": merged.index.min().isoformat(),
            "last_timestamp": merged.index.max().isoformat(),
            "rows": int(len(merged)),
            "exchange_gap_count": len(exchange_gaps),
            "exchange_missing_bars": sum(item["missing_bars"] for item in exchange_gaps),
            "exchange_gaps": exchange_gaps,
            "exchange_gap_policy": exchange_gap_policy,
            "exchange_gap_rechecked_at": exchange_gap_rechecked_at,
            "updated_at": reference.isoformat(),
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    return {
        "ok": True,
        "symbol": symbol,
        "market": requirement.market,
        "timeframe": requirement.timeframe,
        "requested_start": requirement.start.isoformat(),
        "first_timestamp": merged.index.min().isoformat(),
        "last_timestamp": merged.index.max().isoformat(),
        "rows": int(len(merged)),
        "exchange_gap_count": len(exchange_gaps),
        "exchange_missing_bars": sum(item["missing_bars"] for item in exchange_gaps),
        "exchange_gaps": exchange_gaps,
        "exchange_gap_policy": exchange_gap_policy,
        "exchange_gap_rechecked_at": exchange_gap_rechecked_at,
        "required_features": sorted(requirement.required_features),
        "indicator_columns": indicator_columns,
        "scenarios": list(requirement.scenario_names),
        "candle_path": str(candle_path),
        "indicator_path": str(indicator_path) if indicator_path is not None else None,
    }


def run_history_bootstrap(
    *,
    config_path: Path = DEFAULT_FACTORY_CONFIG,
    markets: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    exclude_timeframes: Iterable[str] | None = None,
    symbol: str = SYMBOL,
    now: str | pd.Timestamp | None = None,
    checkpoint_pages: int = DEFAULT_CHECKPOINT_PAGES,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    max_request_pages: int = DEFAULT_MAX_REQUEST_PAGES,
    fetch_page: Callable[..., pd.DataFrame] = fetch_kline_page,
    report_path: Path | None = None,
    search_spaces: Iterable[Any] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    generated_at = _utc_timestamp(
        now if now is not None else pd.Timestamp.now(tz="UTC")
    ).isoformat()
    try:
        requirements = build_default_requirements(
            config_path=config_path,
            markets=markets,
            timeframes=timeframes,
            exclude_timeframes=exclude_timeframes,
            now=now,
            symbol=symbol,
            search_spaces=search_spaces,
        )
    except (OSError, ValueError) as exc:
        report = {
            "ok": False,
            "generated_at": generated_at,
            "symbol": symbol,
            "research_factory_config": str(Path(config_path).resolve()),
            "dataset_count": 0,
            "required_dataset_count": 0,
            "datasets": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        if report_path:
            write_json_atomic(report_path, report)
        return report
    report: dict[str, Any] = {
        "ok": False,
        "generated_at": generated_at,
        "symbol": symbol,
        "research_factory_config": str(Path(config_path).resolve()),
        "api_page_budget_per_dataset": max_request_pages,
        "datasets": [],
    }
    for requirement in requirements:
        try:
            result = sync_requirement(
                requirement,
                symbol=symbol,
                now=now,
                checkpoint_pages=checkpoint_pages,
                request_delay_seconds=request_delay_seconds,
                max_request_pages=max_request_pages,
                fetch_page=fetch_page,
                deadline_monotonic=deadline_monotonic,
            )
        except HistoryBootstrapDeferred as exc:
            result = {
                "ok": True,
                "deferred": True,
                "reason": "time_budget_exhausted",
                "market": requirement.market,
                "timeframe": requirement.timeframe,
                "requested_start": requirement.start.isoformat(),
                "scenarios": list(requirement.scenario_names),
                "detail": str(exc),
                "remediation": "rerun the same command; the atomic checkpoint resumes completed pages",
            }
        except Exception as exc:
            result = {
                "ok": False,
                "market": requirement.market,
                "timeframe": requirement.timeframe,
                "requested_start": requirement.start.isoformat(),
                "scenarios": list(requirement.scenario_names),
                "error": f"{type(exc).__name__}: {exc}",
                "remediation": "rerun the same command; the atomic checkpoint resumes completed pages",
            }
        report["datasets"].append(result)
        if report_path:
            write_json_atomic(report_path, report)
        if result.get("deferred") or not result["ok"]:
            break
    report["ok"] = bool(requirements) and all(item.get("ok") for item in report["datasets"])
    report["dataset_count"] = len(report["datasets"])
    report["required_dataset_count"] = len(requirements)
    report["deferred"] = any(item.get("deferred") for item in report["datasets"])
    report["complete"] = bool(
        report["ok"] and not report["deferred"] and len(report["datasets"]) == len(requirements)
    )
    if report_path:
        write_json_atomic(report_path, report)
    return report


def _plan_payload(
    requirements: Iterable[HistoryRequirement],
    *,
    config_path: Path = DEFAULT_FACTORY_CONFIG,
) -> dict[str, Any]:
    datasets = [
        {
            "market": item.market,
            "timeframe": item.timeframe,
            "requested_start": item.start.isoformat(),
            "required_features": sorted(item.required_features),
            "build_indicators": item.build_indicators,
            "scenarios": list(item.scenario_names),
        }
        for item in requirements
    ]
    return {
        "ok": bool(datasets),
        "plan_only": True,
        "research_factory_config": str(Path(config_path).resolve()),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap native-timeframe Binance history for configured research spaces."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_FACTORY_CONFIG)
    parser.add_argument("--market", action="append", choices=("spot", "futures"), dest="markets")
    parser.add_argument("--timeframes", nargs="+", choices=tuple(TIMEFRAME_DELTAS))
    parser.add_argument(
        "--exclude-timeframes",
        nargs="+",
        choices=tuple(TIMEFRAME_DELTAS),
        help="Exclude a scheduled partition such as the separately refreshed 1m dataset.",
    )
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--checkpoint-pages", type=int, default=DEFAULT_CHECKPOINT_PAGES)
    parser.add_argument(
        "--request-delay-seconds", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS
    )
    parser.add_argument("--max-request-pages", type=int, default=DEFAULT_MAX_REQUEST_PAGES)
    parser.add_argument("--report", type=Path, default=Path("runtime/history_bootstrap.json"))
    parser.add_argument(
        "--plan", action="store_true", help="Print required datasets without network or writes."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.plan:
        report = run_history_bootstrap(
            config_path=args.config,
            markets=args.markets,
            timeframes=args.timeframes,
            exclude_timeframes=args.exclude_timeframes,
            symbol=args.symbol,
            checkpoint_pages=args.checkpoint_pages,
            request_delay_seconds=args.request_delay_seconds,
            max_request_pages=args.max_request_pages,
            report_path=args.report,
        )
    else:
        try:
            report = _plan_payload(
                build_default_requirements(
                    config_path=args.config,
                    markets=args.markets,
                    timeframes=args.timeframes,
                    exclude_timeframes=args.exclude_timeframes,
                    symbol=args.symbol,
                ),
                config_path=args.config,
            )
        except (OSError, ValueError) as exc:
            report = {
                "ok": False,
                "plan_only": True,
                "research_factory_config": str(args.config.resolve()),
                "dataset_count": 0,
                "datasets": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
