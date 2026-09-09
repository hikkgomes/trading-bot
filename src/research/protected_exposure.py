"""Prevent new qualification on intervals already exposed by legacy research."""

import datetime as dt
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from src.data.database import legacy_research_archive


def legacy_exposure_conflicts(connection, snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    intervals = []
    for payload in connection.execute(select(legacy_research_archive.c.payload)).scalars():
        intervals.extend(payload["source_tables"].get("protected_intervals", ()))
    if not intervals:
        return ()
    interval = snapshot.get("interval")
    scope = snapshot.get("instrument_scope")
    if not isinstance(interval, Mapping) or not isinstance(scope, list | tuple) or not scope:
        raise ValueError(
            "protected dataset needs an interval and instrument scope for exposure checks"
        )
    start, end = _utc(interval["start"]), _utc(interval["end"])
    instruments = {_market_symbol(str(value), str(snapshot.get("product_id"))) for value in scope}
    return tuple(
        sorted(
            {
                str(row["interval_key"])
                for row in intervals
                if (row["market"], row["symbol"]) in instruments
                and start <= _utc(row["end_utc"])
                and end > _utc(row["start_utc"])
            }
        )
    )


def _utc(value: Any) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("protected exposure timestamps require an explicit timezone")
    return parsed.astimezone(dt.UTC)


def _market_symbol(instrument: str, product: str) -> tuple[str, str]:
    parts = instrument.split(":")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return ("spot" if product == "btc_accumulation" else "futures"), instrument
