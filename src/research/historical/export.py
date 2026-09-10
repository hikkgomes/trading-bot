"""Read bounded Google Sheets cells into the historical import's existing JSON format."""

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from src.research.historical.study import write_study

ORDER_SHEETS = (
    "Copy of Trading data 01- 14 July",
    "Copy of Trading data 01- 30 June",
    "Copy of Trading data 21 May- 31 May",
)
POSITION_SHEET = "Copy of OKX Positions History"


def export_workbook(
    session: requests.Session,
    spreadsheet_id: str,
    *,
    order_sheets: Sequence[str] = ORDER_SHEETS,
    position_sheet: str = POSITION_SHEET,
    batch_rows: int = 1000,
    maximum_rows: int = 100_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", spreadsheet_id):
        raise ValueError("invalid spreadsheet identifier")
    if not 1 <= batch_rows <= 1000 or not 1 <= maximum_rows <= 100_000:
        raise ValueError("historical export exceeds its bounded row budget")
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{quote(spreadsheet_id)}"
    metadata = _get(session, url, {"fields": "spreadsheetId,properties,sheets.properties"})
    titles = (*order_sheets, position_sheet)
    if not order_sheets or len(set(titles)) != len(titles):
        raise ValueError("historical source sheets must be distinct and explicit")
    properties = {item["properties"]["title"]: item["properties"] for item in metadata["sheets"]}
    if set(titles) - properties.keys():
        raise ValueError("a declared historical source sheet is unavailable")
    if sum(properties[title]["gridProperties"]["rowCount"] for title in titles) > maximum_rows:
        raise ValueError("historical export exceeds its declared total row budget")
    source: dict[str, Any] = {
        "schema": "historical.google_cells/v1",
        "source_id": "google-sheets:" + spreadsheet_id,
        "display_timezone": metadata["properties"].get("timeZone"),
        "timestamp_timezone_status": "unconfirmed",
        "sheets": {},
    }
    native_ids: dict[str, Any] = {"spreadsheetId": spreadsheet_id, "sheets": []}
    for title in titles:
        rows, id_rows = _read_sheet(
            session, url, properties[title], "Y" if title == position_sheet else "Q", batch_rows
        )
        source["sheets"][title] = rows
        if title in order_sheets:
            native_ids["sheets"].append(
                {
                    "properties": {"title": title},
                    "data": [{"startRow": 0, "startColumn": 0, "rowData": id_rows}],
                }
            )
    return source, native_ids


def _get(session, url, params) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=(10, 60))
    response.raise_for_status()
    return response.json()


def _read_sheet(session, url, properties, last_column, batch_rows):
    title = properties["title"]
    escaped = "'" + title.replace("'", "''") + "'"
    count = properties["gridProperties"]["rowCount"]
    rows: list[list[Any]] = [[] for _ in range(count)]
    id_rows: list[dict[str, Any]] = [{} for _ in range(count)]
    for start in range(0, count, batch_rows):
        end = min(count, start + batch_rows)
        payload = _get(
            session,
            url,
            {
                "ranges": f"{escaped}!A{start + 1}:{last_column}{end}",
                "fields": "sheets(properties(title),data(startRow,startColumn,rowData.values(userEnteredValue,effectiveValue,formattedValue)))",
            },
        )
        for sheet in payload.get("sheets", ()):
            if sheet["properties"]["title"] != title:
                raise ValueError("cell response belongs to another source sheet")
            _copy_cells(sheet.get("data", ()), rows, id_rows, start, end)
    while rows and not rows[-1]:
        rows.pop()
        id_rows.pop()
    return {"range": f"{escaped}!A1:{last_column}{count}", "values": rows}, id_rows


def _copy_cells(blocks, rows, id_rows, start, end):
    for block in blocks:
        offset = int(block.get("startRow", 0))
        if block.get("startColumn", 0) != 0:
            raise ValueError("historical cells must start at source column A")
        for index, row in enumerate(block.get("rowData", ())):
            if not start <= offset + index < end:
                raise ValueError("historical cells are outside their requested range")
            cells = row.get("values", ())
            values = [_cell_value(cell) for cell in cells]
            while values and values[-1] == "":
                values.pop()
            rows[offset + index] = values
            id_rows[offset + index] = {"values": [cells[0]]} if cells else {}


def _cell_value(cell: Mapping[str, Any]) -> Any:
    value = cell.get("effectiveValue", cell.get("userEnteredValue", {}))
    if "errorValue" in value or "formulaValue" in value:
        raise ValueError("historical source contains an unresolved formula or cell error")
    return next(iter(value.values()), "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spreadsheet-id", required=True)
    parser.add_argument("--order-sheet", action="append")
    parser.add_argument("--position-sheet", default=POSITION_SHEET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if not token:
        parser.error("GOOGLE_ACCESS_TOKEN must contain an authorised read-only Sheets access token")
    with requests.Session() as session:
        session.headers["Authorization"] = f"Bearer {token}"
        source, ids = export_workbook(
            session,
            args.spreadsheet_id,
            order_sheets=args.order_sheet or ORDER_SHEETS,
            position_sheet=args.position_sheet,
        )
    print(f"source: {write_study(args.output / 'values', source).resolve()}")
    print(f"id-cells: {write_study(args.output / 'id-cells', ids).resolve()}")


if __name__ == "__main__":
    main()
