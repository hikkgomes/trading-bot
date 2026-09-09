"""Causal risk estimates from matching, consecutive closed-candle intervals."""

from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from collections.abc import Mapping
from itertools import combinations
from typing import Any

from sqlalchemy import BigInteger, select
from sqlalchemy.engine import Engine

from src.data.database import risk_snapshot
from src.domain._codec import canonical_hash, timeframe_seconds, timestamp

WINDOW_RETURNS = 256
MINIMUM_RETURNS = 2


def estimate_factors(
    engine: Engine, *, product_id: str, instrument_ids: tuple[str, ...], timeframe: str, at: str
) -> tuple[dict[str, dict[str, float]], dict[str, float], tuple[str, ...], dict[str, Any]]:
    cadence = timeframe_seconds(timeframe)
    through = dt.datetime.fromisoformat(
        timestamp(at, field="factor as-of", preserve_subseconds=True)
    )
    points = _closed_points(engine, product_id, instrument_ids, timeframe, through)
    returns = {key: _interval_returns(values, cadence) for key, values in points.items()}
    correlations: dict[str, dict[str, float]] = {}
    beta: dict[str, float] = {}
    receipts: dict[str, Any] = {}
    missing = set()
    benchmark = next((key for key in sorted(instrument_ids) if is_btc_benchmark(key)), None)
    for key in sorted(instrument_ids):
        if key == benchmark:
            beta[key] = 1.0
            receipts[f"beta:{key}"] = {"method": "BTC benchmark identity", "value": 1.0}
        elif benchmark is None:
            missing.add(f"beta:{key}")
        else:
            values, receipt = _paired(returns, points, key, benchmark, cadence, through)
            if values is None or statistics.pvariance(values[1]) == 0:
                missing.add(f"beta:{key}")
            else:
                beta[key] = _beta(*values)
                receipts[f"beta:{key}"] = {**receipt, "value": beta[key]}
        own = returns.get(key, {})
        if _usable(own, cadence, through):
            if statistics.pvariance(own.values()) > 0:
                correlations[key] = {key: 1.0}
            receipts[f"volatility:{key}"] = {
                **_receipt(points, (key,), sorted(own)),
                "method": "population standard deviation of consecutive candle returns",
                "unit": "return_volatility_ratio",
                "value": statistics.pstdev(own.values()),
            }
    for left, right in combinations(sorted(instrument_ids), 2):
        values, receipt = _paired(returns, points, left, right, cadence, through)
        if values is None or any(statistics.pvariance(series) == 0 for series in values):
            missing.add(f"correlation:{left}:{right}")
            continue
        value = statistics.correlation(*values)
        correlations.setdefault(left, {})[right] = value
        correlations.setdefault(right, {})[left] = value
        receipts[f"correlation:{left}:{right}"] = {**receipt, "value": value}
    return (
        correlations,
        beta,
        tuple(sorted(missing)),
        {
            "schema": "risk.factor_estimates/v1",
            "timeframe": timeframe,
            "window_returns": WINDOW_RETURNS,
            "minimum_returns": MINIMUM_RETURNS,
            "maximum_age_seconds": 2 * cadence,
            "benchmark_instrument_id": benchmark,
            "estimates": receipts,
        },
    )


def is_btc_benchmark(key: str) -> bool:
    return bool(re.search(r"(?:^|[:/ -])BTC/?USDT(?:$|[:/ -])", key.upper()))


def _closed_points(engine, product_id, instrument_ids, timeframe, through):
    cadence = timeframe_seconds(timeframe)
    oldest = int(through.timestamp() * 1000) - (WINDOW_RETURNS + 2) * cadence * 1000
    payload = risk_snapshot.c.payload
    statement = (
        select(risk_snapshot.c.id, payload, risk_snapshot.c.created_at)
        .where(
            payload["kind"].as_string() == "market_data_input",
            payload["product_id"].as_string() == product_id,
            payload["instrument_id"].as_string().in_(instrument_ids),
            payload["source_event_type"].as_string() == "candle",
            payload["source_candle_closed"].as_boolean().is_(True),
            payload["source_timeframe"].as_string() == timeframe,
            payload["source_close_time_ms"].as_string().cast(BigInteger) >= oldest,
            payload["source_close_time_ms"].as_string().cast(BigInteger)
            <= int(through.timestamp() * 1000),
            risk_snapshot.c.created_at <= through.isoformat(),
        )
        .order_by(risk_snapshot.c.created_at, risk_snapshot.c.id)
    )
    points: dict[str, dict[int, dict[str, Any]]] = {}
    with engine.connect() as connection:
        for row in connection.execute(statement).mappings():
            _add_point(points, row, through)
    return points


def _add_point(points, row, through):
    payload = row["payload"]
    if canonical_hash(payload) != row["id"]:
        raise ValueError("factor source content hash does not match its identity")
    available = dt.datetime.fromisoformat(
        timestamp(payload["availability_time"], field="bar availability", preserve_subseconds=True)
    )
    if available > through:
        return
    close_time = int(payload["source_close_time_ms"])
    if available.timestamp() * 1000 < close_time:
        raise ValueError("a closed candle cannot be available before its close")
    price = float(payload["values"]["close"])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("factor source requires a positive finite candle close")
    series = points.setdefault(payload["instrument_id"], {})
    previous = series.get(close_time)
    if previous is not None:
        if previous["close"] != price:
            raise ValueError("conflicting risk candle closes require explicit resolution")
        return
    series[close_time] = {
        "close": price,
        "source_id": row["id"],
        "availability_time": available.isoformat(),
    }


def _interval_returns(points: Mapping[int, Mapping[str, Any]], cadence: int):
    keys = sorted(points)[-(WINDOW_RETURNS + 1) :]
    return {
        (left, right): points[right]["close"] / points[left]["close"] - 1
        for left, right in zip(keys, keys[1:], strict=False)
        if right - left == cadence * 1000
    }


def _usable(returns, cadence, through):
    if len(returns) < MINIMUM_RETURNS:
        return False
    latest = max(right for _, right in returns)
    age_ms = int(through.timestamp() * 1000) - latest
    # Exchange candle endpoints include their final millisecond.
    return 0 <= age_ms <= 2 * cadence * 1000 + 1


def _paired(returns, points, left, right, cadence, through):
    left_returns, right_returns = returns.get(left, {}), returns.get(right, {})
    intervals = sorted(left_returns.keys() & right_returns.keys())
    aligned = [
        {interval: series[interval] for interval in intervals}
        for series in (left_returns, right_returns)
    ]
    if not all(_usable(series, cadence, through) for series in aligned):
        return None, {}
    return (
        [left_returns[key] for key in intervals],
        [right_returns[key] for key in intervals],
    ), _receipt(points, (left, right), intervals)


def _receipt(points, keys, intervals):
    source_points = [points[key][end] for key in keys for interval in intervals for end in interval]
    return {
        "method": "aligned consecutive candle returns",
        "instrument_ids": list(keys),
        "sample_count": len(intervals),
        "first_interval": list(intervals[0]),
        "last_interval": list(intervals[-1]),
        "source_snapshot_ids": sorted({point["source_id"] for point in source_points}),
        "available_at": max(point["availability_time"] for point in source_points),
    }


def _beta(values, benchmark):
    return statistics.covariance(values, benchmark) / statistics.variance(benchmark)
