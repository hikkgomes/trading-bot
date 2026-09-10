"""Reproducible, non-promotable historical execution and structure studies."""

import argparse
import datetime as dt
import hashlib
import json
import os
from bisect import bisect_left
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
    raw_cells = id_cells.read_bytes() if id_cells else None
    cell_payload = json.loads(raw_cells) if raw_cells else {}
    source_workbook = payload["source_id"].removeprefix("google-sheets:")
    if cell_payload.get("spreadsheetId", source_workbook) != source_workbook:
        raise ValueError("native ID cells belong to another source workbook")
    cells = google_id_cells(cell_payload)
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
        "id_cells_sha256": hashlib.sha256(raw_cells).hexdigest() if raw_cells else None,
        "export_schema": payload.get("schema", "historical.google_values/legacy"),
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
        "order_records": [asdict(order) for order in orders],
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
    entry_exposure_states: Mapping[int, str | None] | None = None,
    excluded_entries: Sequence[int] = (),
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
        entry_exposure_states=entry_exposure_states,
        excluded_entries=excluded_entries,
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
        "matching": "hour_trend_fixed_volatility_execution_exposure/v3",
        "bar_hash": canonical_hash(bars),
        "entry_indices": list(entry_indices),
        "instrument_id": instrument_id,
        "direction": direction,
        "exposure_state_hash": canonical_hash(exposure_states),
        "entry_exposure_states": sorted((entry_exposure_states or {}).items()),
        "excluded_entry_indices": sorted(set(excluded_entries)),
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
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    serialised = json.dumps(result, default=str, sort_keys=True, indent=2)
    identity = hashlib.sha256(serialised.encode()).hexdigest()
    destination = output / f"{identity}.json"
    if destination.exists():
        if destination.read_text() != serialised:
            raise ValueError("immutable historical study identity collision")
        return destination
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialised)
    destination.chmod(0o600)
    return destination


def _native_structure_bars(history):
    from src.data.okx_history import INTERVALS

    raw = history.read_bytes()
    native = json.loads(raw)
    scope = native["scope"]
    if scope["schema"] != "okx_history/v1" or scope["kind"] != "trade":
        raise ValueError("entry studies require native OKX traded-price candles")
    symbol = scope["instrument"].removesuffix("-USDT-SWAP")
    if scope["instrument"] != symbol + "-USDT-SWAP":
        raise ValueError("historical entry study instrument is not an OKX USDT swap")
    interval = INTERVALS[scope["timeframe"]]
    rows = native["bars"]
    times = [int(row["source_availability_time_ms"]) for row in rows]
    if any(second - first != interval for first, second in zip(times, times[1:], strict=False)):
        raise ValueError("market cadence gaps cannot supply matched controls")
    if any(
        not row["confirmed"] or int(row["open_time_ms"]) + interval != time
        for row, time in zip(rows, times, strict=True)
    ):
        raise ValueError("historical bars have invalid confirmation times")
    bars = tuple(
        MarketBar(
            dt.datetime.fromtimestamp(time / 1000, dt.UTC).isoformat(),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        for row, time in zip(rows, times, strict=True)
    )
    return raw, scope, symbol, interval, times, bars


def align_campaign_structure(study, history: Path, *, horizon: int, embargo: int):
    """Associate each observed campaign open with strictly earlier native bars."""
    if study.get("timezone_status") != "confirmed":
        raise ValueError("market alignment requires confirmed execution timezone")
    if horizon < 1 or embargo < 0:
        raise ValueError("historical structure horizon and embargo are invalid")
    raw, scope, symbol, interval, times, bars = _native_structure_bars(history)
    exposures = _exposure_states(study, symbol, times)
    selected_campaigns = sorted(
        (row for row in study["campaigns"] if row["symbol"] == symbol),
        key=lambda row: (_milliseconds(row["started_at"]), row["campaign_id"]),
    )
    entry_states = _exposure_states(
        study, symbol, [_milliseconds(row["started_at"]) for row in selected_campaigns]
    )
    campaigns, excluded = defaultdict(list), []
    all_indices = []
    for campaign, exposure in zip(selected_campaigns, entry_states, strict=True):
        index = bisect_left(times, _milliseconds(campaign["started_at"])) - 1
        all_indices.append(index)
        reason = _alignment_exclusion(campaign, index, times, interval, horizon, exposure)
        if reason:
            excluded.append({"campaign_id": campaign["campaign_id"], "reason": reason})
        else:
            campaigns[campaign["side"]].append((index, campaign["campaign_id"], exposure))
    results = {
        side: _aligned_exposure_groups(
            bars, selected, exposures, all_indices, scope["instrument"], side, horizon, embargo
        )
        for side, selected in campaigns.items()
    }
    return {
        "schema": "historical.campaign_alignment/v2",
        "source_study_hash": canonical_hash(json.loads(json.dumps(study, default=str))),
        "native_history_sha256": hashlib.sha256(raw).hexdigest(),
        "native_scope": scope,
        "results": results,
        "excluded_campaigns": excluded,
        "promotable": False,
        "limitations": [
            "hourly_or_declared_bar_resolution",
            "coincident_entries_share_a_market_bar_cluster",
            "observed_export_inventory_only",
            "not_an_independent_strategy_validation",
        ],
    }


def _aligned_exposure_groups(
    bars, selected, exposures, all_indices, instrument, side, horizon, embargo
):
    studies, comparisons = [], []
    for state in sorted({state for _index, _identity, state in selected}):
        indices = sorted({index for index, _identity, exposure in selected if exposure == state})
        measured = study_entry_structure(
            bars,
            indices,
            instrument_id="okx:swap:" + instrument,
            direction=side,
            exposure_states=exposures,
            entry_exposure_states=dict.fromkeys(indices, state),
            excluded_entries=all_indices,
            seed=0,
            horizon=horizon,
            embargo=embargo,
        )
        studies.append(measured)
        comparisons.extend(
            {
                **row,
                "exposure_before_entry": state,
                "specification_id": measured["specification_id"],
            }
            for row in measured["comparisons"]
        )
    return {
        "studies": studies,
        "comparisons": comparisons,
        "promotable": False,
        "campaign_observations": [
            {
                "index": index,
                "campaign_id": identity,
                "available_at": bars[index].timestamp,
                "exposure_before_entry": state,
            }
            for index, identity, state in selected
        ],
    }


def _milliseconds(value):
    observed = dt.datetime.fromisoformat(value)
    if observed.utcoffset() is None:
        raise ValueError("historical alignment timestamp requires a timezone")
    return int(observed.timestamp() * 1000)


def _exposure_states(study, symbol, times):
    events = sorted(
        (row for row in study["inventory_events"] if row["symbol"] == symbol),
        key=lambda row: _milliseconds(row["timestamp"]),
    )
    states, inventory, cursor = [], {"long": Decimal(0), "short": Decimal(0)}, 0
    for time in times:
        while cursor < len(events) and _milliseconds(events[cursor]["timestamp"]) < time:
            event = events[cursor]
            inventory[event["side"]] = Decimal(str(event["quantity"]))
            cursor += 1
        if cursor == 0 or min(inventory.values()) < 0:
            states.append(None)
        else:
            active = [side for side, quantity in inventory.items() if quantity > 0]
            states.append("hedged" if len(active) == 2 else active[0] if active else "flat")
    return tuple(states)


def _alignment_exclusion(campaign, index, times, interval, horizon, exposure):
    if Decimal(str(campaign["minimum_inventory"])) < 0:
        return "opening_inventory_unresolved"
    if index < 19 or index + horizon >= len(times):
        return "market_warmup_or_outcome_missing"
    if _milliseconds(campaign["started_at"]) - times[index] > interval:
        return "entry_market_gap"
    window = times[index - 19 : index + horizon + 1]
    if any(second - first != interval for first, second in zip(window, window[1:], strict=False)):
        return "market_cadence_gap"
    if exposure is None:
        return "pre_entry_exposure_unresolved"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a research-only Trading Study snapshot")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--id-cells", type=Path)
    parser.add_argument("--timezone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market-history", type=Path, action="append", default=[])
    parser.add_argument("--structure-horizon-bars", type=int, default=4)
    parser.add_argument("--structure-embargo-bars", type=int, default=4)
    args = parser.parse_args()
    result = import_study(args.source, id_cells=args.id_cells, timezone=args.timezone)
    path = write_study(args.output, result)
    alignments = [
        str(
            write_study(
                args.output,
                align_campaign_structure(
                    result,
                    history,
                    horizon=args.structure_horizon_bars,
                    embargo=args.structure_embargo_bars,
                ),
            )
        )
        for history in args.market_history
    ]
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "promotable": False,
                "swap_orders": result["swap_orders"],
                "balanced_campaigns": result["balanced_campaigns"],
                "structure_studies": alignments,
            }
        )
    )


if __name__ == "__main__":
    main()
