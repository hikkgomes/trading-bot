"""Native settlement events, kept separate from indicative funding quotes."""

import datetime as dt
import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from typing import Any

from src.domain._codec import canonical_hash, timestamp
from src.research.outcomes import OutcomeCategory


class FundingUnavailable(ValueError):
    outcome_category = OutcomeCategory.BLOCKED_DATA


def merge_native_settlements(events, context):
    """Merge each instrument's measured settlements into the economic event stream."""
    rows = context.get("market_frame", context.get("bars"))
    if context.get("funding_available") is not True or not isinstance(rows, list | tuple):
        raise FundingUnavailable("panel funding coverage is unavailable")
    if any("funding_events" not in row for row in rows):
        raise FundingUnavailable(
            "panel bars need native settlements or explicit empty settlement intervals"
        )
    required = {event["symbol"] for event in events}
    covered = {row.get("instrument_id", row.get("symbol")) for row in rows}
    if not required.issubset(covered):
        raise FundingUnavailable("panel funding coverage omits an executed instrument")
    settlements = {}
    for row in rows:
        for event in row_settlements(row):
            identity = (event["symbol"], event["timestamp"])
            previous = settlements.setdefault(identity, event)
            if previous != event:
                raise ValueError("conflicting panel funding settlement")
    return [*events, *(settlements[key] for key in sorted(settlements))]


def legacy_settlement_id(payload: Mapping[str, Any]) -> str:
    legacy = {
        key: payload[key]
        for key in (
            "kind",
            "product_id",
            "instrument_id",
            "symbol",
            "funding_time",
            "funding_rate",
            "mark_price",
            "source",
        )
    }
    legacy["funding_time"] = timestamp(str(legacy["funding_time"]), field="legacy funding time")
    return canonical_hash(legacy)


def _available_rows(rows, product_id, created):
    for row in rows:
        payload = row["payload"]
        if payload.get("kind") != "historical_funding" or payload.get("product_id") != product_id:
            continue
        available = timestamp(
            str(payload.get("availability_time", row["created_at"])),
            field="funding availability",
            preserve_subseconds=True,
        )
        if available <= created:
            yield row


def settlement_index(
    rows: Iterable[Mapping[str, Any]], *, product_id: str, created: str
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    created = timestamp(created, field="funding as-of time", preserve_subseconds=True)
    native_rows = tuple(
        row
        for row in rows
        if row["payload"].get("kind") == "historical_funding"
        and row["payload"].get("product_id") == product_id
    )
    superseded = set()
    # Known corrupt timestamps stay quarantined even for an earlier as-of query.
    # The replacement's own availability still controls when it may be consumed.
    for row in native_rows:
        payload = row["payload"]
        previous_id = payload.get("supersedes_record_id")
        if previous_id is not None:
            if previous_id != legacy_settlement_id(payload):
                raise ValueError("funding precision correction does not match its legacy record")
            superseded.add(previous_id)
    for row in _available_rows(native_rows, product_id, created):
        if row.get("id") in superseded:
            continue
        payload = row["payload"]
        event = _settlement(payload)
        if event["timestamp"] > created:
            continue
        instrument = str(payload["instrument_id"])
        events = result.setdefault(instrument, {})
        previous = events.get(event["timestamp"])
        if previous is not None and previous != event:
            raise ValueError("conflicting native funding settlement")
        events[event["timestamp"]] = event
    return result


def _settlement(payload: Mapping[str, Any]) -> dict[str, Any]:
    rate = float(payload["funding_rate"])
    mark = payload.get("mark_price")
    if mark is None or not math.isfinite(float(mark)) or float(mark) <= 0:
        raise ValueError("native funding settlement has no measured mark price")
    if not math.isfinite(rate):
        raise ValueError("native funding settlement rate is not finite")
    observed = timestamp(
        str(payload["funding_time"]), field="funding time", preserve_subseconds=True
    )
    return {
        "type": "funding",
        "timestamp": observed,
        "symbol": payload["instrument_id"],
        "event_id": f"{payload['instrument_id']}:{observed}",
        "funding_rate": rate,
        "mark_price": float(mark),
        "mark_source": payload["source"],
        **(
            {"source_event_id": payload["source_event_id"]}
            if payload.get("source_event_id")
            else {}
        ),
    }


def attach_settlements(
    bars: tuple[dict[str, Any], ...],
    funding: Mapping[str, Mapping[str, dict[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    times = {instrument: sorted(events) for instrument, events in funding.items()}
    previous: dict[str, str] = {}
    result = []
    for row in sorted(bars, key=lambda item: (item["close_timestamp"], item["instrument_id"])):
        instrument = str(row["instrument_id"])
        close = str(row["close_timestamp"])
        index = times[instrument]
        lower = _interval_start(row, previous.get(instrument))
        start = (
            bisect_left(index, lower) if instrument not in previous else bisect_right(index, lower)
        )
        selected = [
            dict(funding[instrument][time]) for time in index[start : bisect_right(index, close)]
        ]
        result.append(
            {
                **row,
                "funding_events": selected,
                "funding_rate": sum(event["funding_rate"] for event in selected),
                "funding_event": bool(selected),
                "funding_available": True,
            }
        )
        previous[instrument] = close
    return tuple(result)


def _interval_start(row: Mapping[str, Any], previous: str | None) -> str:
    if previous is not None:
        return previous
    if row.get("open_time_ms") is not None:
        return dt.datetime.fromtimestamp(float(row["open_time_ms"]) / 1000, dt.UTC).isoformat()
    return str(row.get("open_timestamp", row["close_timestamp"]))


def row_settlements(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return only native events; an indicative rate cannot debit an account."""
    return tuple(dict(event) for event in row.get("funding_events", ()))
