"""Bounded Binance history pagination; incomplete scans never certify recovery."""

from collections.abc import Callable, Iterator
from typing import Any

PAGE_SIZE = 1000
MAX_PAGES = 100


def trade_history(fetch: Callable[..., Any], symbol: str, params: dict) -> Iterator[dict]:
    raw_cursor = params.get("fromId", 0)
    if isinstance(raw_cursor, bool) or not str(raw_cursor).isdigit():
        raise ValueError("trade history cursor must be a non-negative integer")
    cursor = int(raw_cursor)
    seen: dict[str, dict] = {}
    for _ in range(MAX_PAGES):
        page = _page(fetch(symbol, None, PAGE_SIZE, {**params, "fromId": cursor}))
        for row in page:
            identity = str(_value(row, "id"))
            if _new_row(seen, identity, row):
                yield row
        if len(page) < PAGE_SIZE:
            return
        identifiers = [_trade_id(row) for row in page]
        next_cursor = max(identifiers) + 1
        if next_cursor <= cursor:
            raise RuntimeError("trade history pagination did not advance")
        cursor = next_cursor
    raise RuntimeError("trade history exceeds the bounded recovery page budget")


def income_history(fetch: Callable[..., Any], params: dict) -> Iterator[dict]:
    seen: dict[str, dict] = {}
    for number in range(1, MAX_PAGES + 1):
        page = _page(fetch({**params, "page": number, "limit": PAGE_SIZE}))
        additions = 0
        for row in page:
            identity = repr(
                tuple(
                    _value(row, field)
                    for field in (
                        "incomeType",
                        "tranId",
                        "symbol",
                        "asset",
                    )
                )
            )
            if _new_row(seen, identity, row):
                additions += 1
                yield row
        if len(page) < PAGE_SIZE:
            return
        if not additions:
            raise RuntimeError("income history pagination did not advance")
    raise RuntimeError("income history exceeds the bounded recovery page budget")


def _page(value: Any) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("exchange history response must be a list of objects")
    return value


def _value(row: dict, field: str) -> Any:
    info = row.get("info")
    return row.get(field, info.get(field) if isinstance(info, dict) else None)


def _new_row(seen: dict[str, dict], identity: str, row: dict) -> bool:
    previous = seen.get(identity)
    if previous is not None and previous != row:
        raise RuntimeError("exchange history changed an existing record identity")
    seen[identity] = row
    return previous is None


def _trade_id(row: dict) -> int:
    value = _value(row, "id")
    if isinstance(value, bool) or not isinstance(value, str | int) or not str(value).isdigit():
        raise RuntimeError("trade history cursor requires an exact non-negative integer ID")
    return int(value)
