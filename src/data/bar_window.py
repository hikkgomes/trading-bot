"""Bounded, chronological bar retention with explicit duplicate conflicts."""

import heapq
from collections.abc import Mapping
from typing import Any


class RecentBarWindow:
    def __init__(self, maximum_rows: int):
        if maximum_rows < 1:
            raise ValueError("bar window must retain at least one row")
        self.maximum_rows = maximum_rows
        self._times: list[str] = []
        self._rows: dict[str, dict[str, Any]] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        key = str(row["close_timestamp"])
        previous = self._rows.get(key)
        if previous is not None:
            _assert_duplicate(previous, row)
            first, second = sorted((previous, row), key=lambda item: str(item["availability_time"]))
            # Equivalent native observations may add metadata; retain first availability.
            self._rows[key] = {
                **second,
                **{name: value for name, value in first.items() if value is not None},
            }
            return
        if len(self._rows) == self.maximum_rows and key <= self._times[0]:
            return
        self._rows[key] = dict(row)
        heapq.heappush(self._times, key)
        if len(self._rows) > self.maximum_rows:
            del self._rows[heapq.heappop(self._times)]

    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[key] for key in sorted(self._rows)]


def _assert_duplicate(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    fields = ("open", "high", "low", "close", "volume")
    if any(first.get(key) != second.get(key) for key in fields):
        raise ValueError("conflicting historical bar revisions require explicit resolution")
    optional = (
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    if any(
        first.get(key) is not None and second.get(key) is not None and first[key] != second[key]
        for key in optional
    ):
        raise ValueError("conflicting historical bar metadata requires explicit resolution")
