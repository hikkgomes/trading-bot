"""Bounded, resumable native OKX history acquisition for offline research."""

import base64
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

INTERVALS = {"1m": 60_000, "15m": 900_000, "1H": 3_600_000, "4H": 14_400_000}
ENDPOINTS = {
    "trade": "history-candles",
    "mark": "history-mark-price-candles",
    "index": "history-index-candles",
}


def download_funding_history(
    *,
    instruments: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    root: Path,
    maximum_days: int = 93,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Read official daily archives; never infer missing funding as zero."""
    if not instruments or any(not re.fullmatch(r"[A-Z0-9]+-USDT-SWAP", x) for x in instruments):
        raise ValueError("unsupported funding instruments")
    first = dt.datetime.fromtimestamp(start_ms / 1000, dt.UTC).date()
    last = dt.datetime.fromtimestamp(end_ms / 1000, dt.UTC).date()
    days = (last - first).days + 1
    if start_ms >= end_ms or not 1 <= days <= maximum_days <= 93:
        raise ValueError("funding request exceeds its declared resource bounds")
    root.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    records: dict[tuple[str, int], dict[str, Any]] = {}
    sources = []
    for offset in range(days):
        day = first + dt.timedelta(days=offset)
        source = _funding_archive(day, root, client)
        sources.append({key: value for key, value in source.items() if key != "archive_base64"})
        _merge_funding_archive(records, source, day, instruments, start_ms, end_ms)
    payload = {
        "schema": "okx_funding_history/v1",
        "instruments": sorted(set(instruments)),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "sources": sources,
        "records": [records[key] for key in sorted(records)],
        "missing_instruments": sorted(set(instruments) - {key[0] for key in records}),
        "archive_coverage_complete": True,
        "event_completeness": "requires_historical_funding_schedule",
        "promotable": False,
    }
    identity = _digest(payload)
    path = root / f"funding-{identity}.json"
    _immutable_json(path, payload)
    return {"path": str(path.resolve()), "content_hash": identity, **payload}


def _funding_archive(day: dt.date, root: Path, client) -> dict[str, str]:
    name = f"allswap-fundingrates-{day.isoformat()}"
    url = f"https://static.okx.com/cdn/okex/traderecords/swaprates/daily/{day:%Y%m%d}/{name}.zip"
    path = root / f"{name}.json"
    if path.exists():
        if path.is_symlink() or path.stat().st_size > 2_700_000:
            raise ValueError("funding cache exceeds its trusted resource boundary")
        source = json.loads(path.read_text())
    else:
        content = bytearray()
        with client.get(url, timeout=20, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=65536):
                content.extend(chunk)
                if len(content) > 2_000_000:
                    raise ValueError("funding archive exceeds download budget")
        source = {
            "url": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "archive_base64": base64.b64encode(content).decode(),
        }
        _immutable_json(path, source)
    raw = base64.b64decode(source["archive_base64"], validate=True)
    if source["url"] != url or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("funding archive checksum or scope mismatch")
    return source


def _merge_funding_archive(records, source, day, instruments, start, end) -> None:
    raw = base64.b64decode(source["archive_base64"], validate=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        expected = f"allswap-fundingrates-{day.isoformat()}.csv"
        if archive.namelist() != [expected] or archive.getinfo(expected).file_size > 16_000_000:
            raise ValueError("unexpected funding archive content or size")
        with archive.open(expected) as handle:
            rows = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
            if rows.fieldnames != ["instrument_name", "funding_rate", "funding_time"]:
                raise ValueError("native funding archive fields changed")
            for row in rows:
                if row["instrument_name"] in instruments:
                    _merge_funding_row(records, row, source["sha256"], day, start, end)


def _merge_funding_row(records, row, source_hash, day, start, end) -> None:
    observed = int(row["funding_time"])
    rate = Decimal(row["funding_rate"])
    day_start = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp() * 1000) - 28_800_000
    if not rate.is_finite() or not day_start <= observed < day_start + 86_400_000:
        raise ValueError("funding rate or archive timestamp is invalid")
    if not start <= observed < end:
        return
    key = (row["instrument_name"], observed)
    record = {
        "instrument": key[0],
        "funding_time_ms": observed,
        "funding_rate": str(rate),
        "source_sha256": source_hash,
    }
    if key in records and Decimal(records[key]["funding_rate"]) != rate:
        raise ValueError("conflicting native funding event")
    records[key] = record


def download_history(
    *,
    instrument: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    root: Path,
    kind: str = "trade",
    maximum_pages: int = 25,
    maximum_rows: int = 200_000,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    _validate_request(instrument, timeframe, start_ms, end_ms, kind, maximum_pages, maximum_rows)
    scope = {
        "instrument": instrument,
        "timeframe": timeframe,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "kind": kind,
        "schema": "okx_history/v1",
    }
    identity = _digest(scope)
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"
    state: dict[str, Any] = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"cursor": end_ms, "rows": {}, "pages": []}
    )
    if state.get("complete"):
        return _complete_report(directory, state["report"], scope)
    state["rows"] = {}
    for page_hash in state["pages"]:
        page = json.loads((directory / f"page-{_safe_hash(page_hash)}.json").read_text())
        _merge_rows(state["rows"], page["data"], start_ms, end_ms, INTERVALS[timeframe], kind)
    cursor = int(state["cursor"])
    cursor = end_ms if cursor <= start_ms else cursor
    client = session or requests.Session()
    request_instrument = instrument.removesuffix("-SWAP") if kind == "index" else instrument
    for _ in range(maximum_pages):
        response = client.get(
            "https://www.okx.com/api/v5/market/" + ENDPOINTS[kind],
            params={
                "instId": request_instrument,
                "bar": timeframe,
                "after": str(cursor),
                "limit": "300",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0":
            raise ValueError(f"OKX history error: {payload.get('code')}")
        raw_rows = payload.get("data", [])
        if not raw_rows:
            break
        page_hash = _digest(payload)
        _immutable_json(directory / f"page-{page_hash}.json", payload)
        _merge_rows(state["rows"], raw_rows, start_ms, end_ms, INTERVALS[timeframe], kind)
        next_cursor = min(int(row[0]) for row in raw_rows)
        if next_cursor >= cursor:
            raise ValueError("OKX pagination did not advance")
        cursor = next_cursor
        state.update({"cursor": cursor, "pages": sorted({*state["pages"], page_hash})})
        _checkpoint(checkpoint, {key: value for key, value in state.items() if key != "rows"})
        if cursor <= start_ms:
            break
    rows = [state["rows"][key] for key in sorted(state["rows"], key=int)]
    gaps = _missing_ranges(rows, start_ms, end_ms, INTERVALS[timeframe])
    canonical = {"scope": scope, "bars": rows, "promotable": False}
    data_hash = _digest(canonical)
    bars_path = directory / f"bars-{data_hash}.json"
    _immutable_json(bars_path, canonical)
    report = {
        **scope,
        "content_hash": data_hash,
        "path": str(bars_path.resolve()),
        "rows": len(rows),
        "complete": not gaps,
        "missing_ranges": gaps,
        "page_hashes": state["pages"],
        "observed_at": dt.datetime.now(dt.UTC).isoformat(),
        "promotable": False,
    }
    state.update({"complete": not gaps, "report": report})
    _checkpoint(checkpoint, {key: value for key, value in state.items() if key != "rows"})
    return report


def _validate_request(instrument, timeframe, start, end, kind, pages, rows):
    if (
        not re.fullmatch(r"[A-Z0-9]+-USDT-SWAP", instrument)
        or timeframe not in INTERVALS
        or kind not in ENDPOINTS
    ):
        raise ValueError("unsupported native OKX history scope")
    if start >= end or start % INTERVALS[timeframe] or end % INTERVALS[timeframe]:
        raise ValueError("history range must be aligned and increasing")
    if pages < 1 or (end - start) // INTERVALS[timeframe] > rows:
        raise ValueError("history request exceeds its declared resource bounds")


def _merge_rows(stored, rows, start, end, interval, kind):
    expected = 9 if kind == "trade" else 6
    for row in rows:
        if len(row) != expected:
            raise ValueError("native OKX candle shape changed")
        time = int(row[0])
        if str(row[-1]) != "1" or not start <= time < end:
            continue
        if time % interval:
            raise ValueError("native candle timestamp is not aligned")
        _validate_prices(row)
        key = str(time)
        value = {
            "open_time_ms": time,
            "open": str(row[1]),
            "high": str(row[2]),
            "low": str(row[3]),
            "close": str(row[4]),
            "volume": str(row[5]) if kind == "trade" else None,
            "source_availability_time_ms": time + interval,
            "confirmed": True,
        }
        if key in stored and stored[key] != value:
            raise ValueError("conflicting native historical candle revision")
        stored[key] = value


def _validate_prices(row):
    opening, high, low, close = (float(value) for value in row[1:5])
    if not all(math.isfinite(value) and value > 0 for value in (opening, high, low, close)):
        raise ValueError("native candle prices must be finite and positive")
    if low > min(opening, close) or high < max(opening, close):
        raise ValueError("native candle prices are inconsistent")


def _missing_ranges(rows, start, end, interval):
    present = {row["open_time_ms"] for row in rows}
    gaps = []
    for time in range(start, end, interval):
        if time in present:
            continue
        if gaps and gaps[-1]["end_ms"] == time:
            gaps[-1]["end_ms"] += interval
        else:
            gaps.append({"start_ms": time, "end_ms": time + interval})
    return gaps


def _safe_hash(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid historical content identity")
    return value


def _complete_report(directory: Path, report, scope):
    digest = _safe_hash(report["content_hash"])
    payload = json.loads((directory / f"bars-{digest}.json").read_text())
    if payload["scope"] != scope or _digest(payload) != digest:
        raise ValueError("cached historical content failed verification")
    if _missing_ranges(
        payload["bars"], scope["start_ms"], scope["end_ms"], INTERVALS[scope["timeframe"]]
    ):
        raise ValueError("cached historical coverage is incomplete")
    return report


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _immutable_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("immutable history content collision")
        return
    with path.open("x") as handle:
        handle.write(encoded)


def _checkpoint(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)
