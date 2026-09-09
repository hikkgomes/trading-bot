"""Lossless row provenance and explicit interpretation of historical exports."""

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

from src.domain._codec import canonical_hash

REPORT_MULTIPLIERS = {
    "BTC": Decimal("0.01"),
    "ATH": Decimal("100"),
    **{symbol: Decimal("1") for symbol in ("PI", "SOL", "SUI", "TON")},
}


@dataclass(frozen=True)
class SourceRow:
    sheet: str
    row_number: int
    raw_values: tuple[Any, ...]
    source_id: str

    @property
    def row_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class HistoricalOrder:
    source: SourceRow
    order_id: str
    id_status: str
    occurred_at: dt.datetime
    timestamp_precision: str
    timezone_status: str
    market: str
    symbol: str
    position_side: str
    action: str
    contracts: Decimal
    quantity: Decimal
    multiplier: Decimal
    multiplier_status: str
    price: Decimal
    recorded_pnl: Decimal
    fee: Decimal
    fee_currency: str
    order_type: str

    @property
    def pnl_status(self) -> str:
        return (
            "opening_no_realisation"
            if self.action == "open" and str(self.source.raw_values[13]) in {"None", ""}
            else "reported"
        )

    @property
    def fee_currency_status(self) -> str:
        return "inferred" if str(self.source.raw_values[15]).strip() in {"None", ""} else "reported"


def number(value: Any, *, field: str) -> Decimal:
    text = str(value).strip().replace(",", "").replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid historical {field}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite historical {field}")
    return result


def historical_time(
    value: Any, *, positions: bool = False, timezone: str | None = None
) -> dt.datetime:
    if isinstance(value, int | float) and not isinstance(value, bool):
        seconds = int((number(value, field="date serial") * 86400).to_integral_value())
        parsed = dt.datetime(1899, 12, 30) + dt.timedelta(seconds=seconds)
        return parsed.replace(tzinfo=ZoneInfo(timezone)) if timezone else parsed
    formats = ("%d/%m/%Y %H:%M:%S",) if positions else ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M:%S")
    for format in formats:
        try:
            parsed = dt.datetime.strptime(str(value).strip(), format)
            return parsed.replace(tzinfo=ZoneInfo(timezone)) if timezone else parsed
        except ValueError:
            continue
    raise ValueError("historical timestamp does not match its declared export format")


def import_order_rows(
    sheet: str,
    rows: Sequence[Sequence[Any]],
    *,
    source_id: str,
    timezone: str | None = None,
    id_cells: Mapping[int, Mapping[str, Any]] | None = None,
    multipliers: Mapping[str, Decimal] | None = None,
    assumed_fee_currency: str | None = None,
) -> tuple[HistoricalOrder, ...]:
    if not rows or str(rows[0][0]).strip() != "Order ID":
        raise ValueError("order export header is missing")
    result = []
    for row_number, row in enumerate(rows[1:], 2):
        if len(row) < 17 or row[2] not in {"Swap", "Spot"}:
            continue
        source = SourceRow(sheet, row_number, tuple(row), source_id)
        result.append(
            _order(
                source,
                timezone,
                (id_cells or {}).get(row_number),
                multipliers,
                assumed_fee_currency,
            )
        )
    return tuple(result)


def _order(
    source: SourceRow,
    timezone: str | None,
    id_cell: Mapping[str, Any] | None,
    multipliers: Mapping[str, Decimal] | None,
    assumed_fee_currency: str | None,
) -> HistoricalOrder:
    row = source.raw_values
    symbol = str(row[3]).split("-")[0].strip().upper()
    market = "swap" if row[2] == "Swap" else "spot"
    parts = str(row[5]).lower().split()
    action, side = (parts[0], parts[-1]) if market == "swap" else (parts[0], "spot")
    order_id, id_status = _order_identity(row[0], id_cell)
    multiplier = Decimal(1) if market == "spot" else (multipliers or REPORT_MULTIPLIERS)[symbol]
    contracts = number(row[8], field="filled quantity")
    price = number(row[11], field="fill price")
    if contracts <= 0 or price <= 0 or multiplier <= 0:
        raise ValueError("completed orders require positive quantity, multiplier and price")
    return HistoricalOrder(
        source,
        order_id,
        id_status,
        historical_time(row[1], timezone=timezone),
        "spreadsheet_serial" if isinstance(row[1], int | float) else "minute",
        "confirmed" if timezone else "unconfirmed",
        market,
        symbol,
        side,
        action,
        contracts,
        contracts * multiplier,
        multiplier,
        "explicit"
        if multipliers is not None or market == "spot"
        else "report_inferred_uncertified",
        price,
        _recorded_pnl(row[13], action),
        number(row[14], field="fee"),
        (assumed_fee_currency or "UNAVAILABLE")
        if str(row[15]).strip() in {"None", ""}
        else str(row[15]).strip().upper(),
        str(row[6]).strip(),
    )


def _order_identity(value: Any, cell: Mapping[str, Any] | None) -> tuple[str, str]:
    entered = (cell or {}).get("userEnteredValue", {})
    if "stringValue" in entered:
        return str(entered["stringValue"]), "exact_string"
    text = str(value)
    if (
        "numberValue" in entered
        or not isinstance(value, str)
        or not re.fullmatch(r"\d+", text.strip())
    ):
        return text, "numeric_precision_unresolved"
    return text, "source_string_unverified"


def _recorded_pnl(value: Any, action: str) -> Decimal:
    if action == "open" and str(value) in {"None", ""}:
        return Decimal(0)
    return number(value, field="PnL")


def google_id_cells(payload: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    result = {}
    for sheet in payload.get("sheets", ()):
        cells = {}
        for block in sheet.get("data", ()):
            start = int(block.get("startRow", 0) or 0)
            for index, row in enumerate(block.get("rowData", ())):
                values = row.get("values", ())
                if values:
                    cells[start + index + 1] = values[0]
        result[sheet["properties"]["title"]] = cells
    return result


def position_totals(rows: Sequence[Sequence[Any]]) -> dict[str, Decimal | int]:
    headers = [str(value).strip() for value in rows[0]]
    names = ("PnL", "Fee", "Funding Fee", "Liquidation Penalty", "Settled PnL", "Realized PnL")
    columns = {name: headers.index(name) for name in names}
    totals = {name: Decimal(0) for name in names}
    count = 0
    inapplicable_settlements = 0
    for row in rows[1:]:
        if len(row) <= max(columns.values()) or not row[0]:
            continue
        for name, index in columns.items():
            if (
                name == "Settled PnL"
                and str(row[index]) == "None"
                and row[headers.index("Instrument Type")] == "SWAP"
            ):
                inapplicable_settlements += 1
                continue
            totals[name] += number(row[index], field=name)
        count += 1
    return {
        **totals,
        "records": count,
        "recomputed_net": sum(totals[name] for name in names[:-1]),
        "inapplicable_swap_settlements": inapplicable_settlements,
    }
