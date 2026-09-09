"""Reproducible, non-promotable historical execution and structure studies."""

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.domain._codec import canonical_hash, non_empty
from src.research.historical.campaigns import hedged_exposure, reconstruct_campaigns
from src.research.historical.imports import (
    HistoricalOrder,
    google_id_cells,
    import_order_rows,
    position_totals,
)
from src.research.historical.structure import MarketBar, matched_pseudo_entries, structure_features


def import_study(
    source: Path, *, id_cells: Path | None = None, timezone: str | None = None
) -> dict[str, Any]:
    raw = source.read_bytes()
    payload = json.loads(raw)
    cells = google_id_cells(json.loads(id_cells.read_text())) if id_cells else {}
    orders: list[HistoricalOrder] = []
    positions = None
    for sheet in payload["sheets"].values():
        rows = sheet.get("values", ())
        title = sheet["range"].rsplit("!", 1)[0].strip("'").replace("''", "'")
        if rows and rows[0][0] == "Order ID":
            orders.extend(
                import_order_rows(
                    title,
                    rows,
                    source_id=payload["source_id"],
                    timezone=timezone,
                    id_cells=cells.get(title),
                    assumed_fee_currency="USDT",
                )
            )
        elif rows and "Realized PnL" in rows[0]:
            positions = position_totals(rows)
    reconstruction = reconstruct_campaigns(orders)
    balanced = [
        campaign
        for campaign in reconstruction.campaigns
        if campaign.closed and campaign.minimum_inventory >= 0
    ]
    return {
        "schema": "historical.execution_study/v1",
        "promotable": False,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_id": payload["source_id"],
        "timezone": timezone,
        "timezone_status": "confirmed" if timezone else "unconfirmed",
        "assumptions": {
            "missing_fee_currency": "USDT as specified by REPORT.md",
            "contract_multipliers": "report-inferred, historical metadata not certified",
        },
        "limitations": [
            "completed_orders_only",
            "equity_and_transfer_history_missing",
            "conditional_order_history_missing",
            "execution_records_are_not_entry_rules",
            "no_account_return_or_liquidation_cause_inferred",
        ],
        "orders": len(orders),
        "swap_orders": sum(order.market == "swap" for order in orders),
        "spot_orders": sum(order.market == "spot" for order in orders),
        "id_status": dict(Counter(order.id_status for order in orders)),
        "fee_currency_status": dict(Counter(order.fee_currency_status for order in orders)),
        "periods": _order_periods(orders),
        "position_history": positions,
        "balanced_campaigns": len(balanced),
        "reconciled_campaigns": sum(campaign.reconciled for campaign in balanced),
        "maximum_balanced_cashflow_error": max(
            (abs(c.gross_cashflow - c.recorded_gross) for c in balanced), default=Decimal(0)
        ),
        "campaigns": [asdict(campaign) for campaign in reconstruction.campaigns],
        "inventory_events": reconstruction.inventory_events,
        "hedged_exposure": hedged_exposure(reconstruction),
        "ambiguous_timestamps": reconstruction.ambiguous_timestamps,
    }


def _order_periods(orders: Sequence[HistoricalOrder]) -> dict[str, dict[str, Decimal | int]]:
    groups: dict[str, list[HistoricalOrder]] = defaultdict(list)
    for order in orders:
        if order.market != "swap":
            continue
        groups[order.occurred_at.strftime("%Y-%m")].append(order)
        groups["entire_export"].append(order)
        key = (
            "before_14_july"
            if order.occurred_at.date() < dt.date(2025, 7, 14)
            else "14_july_and_after"
        )
        groups[key].append(order)
    return {name: _totals(items) for name, items in groups.items()}


def _totals(orders: Sequence[HistoricalOrder]) -> dict[str, Decimal | int]:
    gross = sum((order.recorded_pnl for order in orders), Decimal(0))
    fees = sum((order.fee for order in orders), Decimal(0))
    return {
        "orders": len(orders),
        "gross_pnl": gross,
        "trading_fees": fees,
        "net_before_funding": gross + fees,
    }


def study_entry_structure(
    bars: Sequence[MarketBar],
    entry_indices: Sequence[int],
    *,
    instrument_id: str,
    direction: str,
    exposure_states: Sequence[str | None],
    seed: int,
    horizon: int,
    embargo: int,
    pivot_width: int = 3,
    lookback: int = 20,
) -> dict[str, Any]:
    instrument_id = non_empty(instrument_id, field="instrument_id")
    if direction not in {"long", "short"}:
        raise ValueError("historical study direction must be explicit")
    if len(set(entry_indices)) != len(entry_indices) or any(
        not 0 <= index < len(bars) - horizon for index in entry_indices
    ):
        raise ValueError("historical entries must be unique valid observation indices")
    features = structure_features(bars, pivot_width=pivot_width, lookback=lookback)
    matches = matched_pseudo_entries(
        features,
        entry_indices,
        exposure_states=exposure_states,
        seed=seed,
        horizon=horizon,
        embargo=embargo,
    )
    comparisons = []
    for entry, controls in matches.items():
        actual = bars[entry + horizon].close / bars[entry].close - 1
        pseudo = [bars[index + horizon].close / bars[index].close - 1 for index in controls]
        comparisons.append(
            {
                "entry_index": entry,
                "entry_timestamp": bars[entry].timestamp,
                "observed_price_return": actual,
                "pseudo_price_returns": pseudo,
                "observed_directional_return": actual if direction == "long" else -actual,
                "pseudo_directional_returns": [
                    value if direction == "long" else -value for value in pseudo
                ],
                "control_indices": controls,
                "status": "measured" if controls else "inconclusive",
            }
        )
    specification = {
        "schema": "historical.structure_specification/v1",
        "seed": seed,
        "horizon": horizon,
        "embargo": embargo,
        "pivot_width": pivot_width,
        "lookback": lookback,
        "matching": "hour_trend_fixed_volatility_exposure_bins/v2",
        "bar_hash": canonical_hash(bars),
        "entry_indices": list(entry_indices),
        "instrument_id": instrument_id,
        "direction": direction,
        "exposure_state_hash": canonical_hash(exposure_states),
    }
    return {
        "specification": specification,
        "specification_id": canonical_hash(specification),
        "features": [asdict(feature) for feature in features],
        "comparisons": comparisons,
        "promotable": False,
        "interpretation": "descriptive_price_response_only",
        "counterfactual_stops": {
            "status": "unsupported",
            "reason": "requires_intraperiod_execution_model",
        },
    }


def write_study(output: Path, result: Mapping[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(result, default=str, sort_keys=True, indent=2)
    identity = hashlib.sha256(serialised.encode()).hexdigest()
    destination = output / f"{identity}.json"
    if destination.exists():
        if destination.read_text() != serialised:
            raise ValueError("immutable historical study identity collision")
        return destination
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialised)
    destination.chmod(0o600)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a research-only Trading Study snapshot")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--id-cells", type=Path)
    parser.add_argument("--timezone")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = import_study(args.source, id_cells=args.id_cells, timezone=args.timezone)
    path = write_study(args.output, result)
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "promotable": False,
                "swap_orders": result["swap_orders"],
                "balanced_campaigns": result["balanced_campaigns"],
            }
        )
    )


if __name__ == "__main__":
    main()
