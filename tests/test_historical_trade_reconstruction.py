from decimal import Decimal

import pytest

from src.research.historical.campaigns import reconstruct_campaigns
from src.research.historical.imports import import_order_rows
from src.research.historical.structure import (
    MarketBar,
    StructureObservation,
    matched_pseudo_entries,
    structure_features,
)

HEADER = [
    "Order ID",
    "Order Time",
    "Instrument",
    "Symbol",
    "Lvg",
    "Side",
    "Order Type",
    "Order Amount",
    "Filled Amount",
    "Amount Unit",
    "Order Price",
    "Avg. Filled Price",
    "Price Unit",
    "PNL",
    "Fee",
    "Fee Unit",
    "Status",
]


def _row(id, time, side, quantity, price, pnl, fee, market="Swap"):
    return [
        id,
        time,
        market,
        "SUI",
        "10",
        side,
        "Limit",
        quantity,
        quantity,
        "cont",
        price,
        price,
        "USDT",
        pnl,
        fee,
        "USDT",
        "Complete",
    ]


def test_partial_closes_reconcile_cashflow_and_preserve_exact_source_id():
    id = "90071992547409931234"
    rows = [
        HEADER,
        _row(id, "5/21/25 10:00", "Open long", "10", "10", "0", "-1"),
        _row("2", "5/21/25 10:01", "Close long", "4", "12", "8", "-0.4"),
        _row("3", "5/21/25 10:02", "Close long", "6", "9", "-6", "-0.6"),
    ]
    orders = import_order_rows(
        "orders",
        rows,
        source_id="fixture",
        id_cells={2: {"userEnteredValue": {"stringValue": id}}},
        multipliers={"SUI": Decimal(1)},
    )
    assert orders[0].order_id == id
    assert orders[0].id_status == "exact_string"
    assert orders[0].source.row_number == 2
    assert orders[0].timezone_status == "unconfirmed"
    campaign = reconstruct_campaigns(orders).campaigns[0]
    assert campaign.reconciled
    assert campaign.gross_cashflow == campaign.recorded_gross == Decimal(2)
    assert campaign.fees == Decimal(-2)
    assert campaign.remaining_quantity == 0


def test_missing_open_inventory_is_never_clipped_or_treated_as_balanced():
    rows = [
        HEADER,
        _row("1", "5/21/25 10:00", "Close long", "2", "10", "1", "-0.1"),
        _row("2", "5/21/25 10:01", "Open long", "1", "10", "0", "-0.1"),
    ]
    orders = import_order_rows("orders", rows, source_id="fixture", multipliers={"SUI": Decimal(1)})
    result = reconstruct_campaigns(orders)
    assert [event["quantity"] for event in result.inventory_events] == [Decimal(-2), Decimal(-1)]
    assert result.campaigns[0].minimum_inventory == -2
    assert not result.campaigns[0].reconciled


def test_spot_is_separate_and_numeric_ids_are_not_invented():
    rows = [HEADER, _row("1.234E+18", "5/21/25 10:00", "Sell", "1", "10", "0", "-0.1", "Spot")]
    orders = import_order_rows("orders", rows, source_id="fixture")
    assert orders[0].order_id == "1.234E+18"
    assert orders[0].id_status == "numeric_precision_unresolved"
    assert reconstruct_campaigns(orders).campaigns == ()


def test_pivots_become_available_only_after_confirmation_and_are_causal():
    bars = [
        MarketBar(f"2025-07-14T0{i}:00:00+00:00", value + 0.25, value - 0.25, value)
        for i, value in enumerate((1, 2, 5, 2, 1, 3))
    ]
    bars[4] = MarketBar("2025-07-14T04:00:00.013000+00:00", 1.25, 0.75, 1)
    original = structure_features(bars, pivot_width=2, lookback=3)
    assert original[2].high_anchor is None
    assert original[4].high_anchor.anchor_at == bars[2].timestamp
    assert original[4].high_anchor.available_at == bars[4].timestamp
    known_at_entry = [
        row for row in original if row.timestamp <= "2025-07-14T04:00:00.010000+00:00"
    ]
    assert known_at_entry[-1].high_anchor is None
    bars[-1] = MarketBar(bars[-1].timestamp, 1001, 999, 1000)
    assert structure_features(bars, pivot_width=2, lookback=3)[:5] == original[:5]


def test_pseudo_entries_match_covariates_without_using_outcomes():
    import datetime as dt

    start = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
    observations = [
        StructureObservation(
            (start + dt.timedelta(hours=i)).isoformat(), 0.01, 0.001, 0, None, None, None
        )
        for i in range(100)
    ]
    matches = matched_pseudo_entries(
        observations,
        [25],
        exposure_states=["flat"] * len(observations),
        seed=123,
        horizon=2,
        embargo=2,
    )
    assert set(matches[25]) == {1, 49, 73, 97}
    assert (
        matched_pseudo_entries(
            observations,
            [25],
            exposure_states=["flat"] * len(observations),
            seed=123,
            horizon=2,
            embargo=2,
        )
        == matches
    )
    exposure = ["flat"] * len(observations)
    exposure[1], exposure[49] = "hedged", None
    conditioned = matched_pseudo_entries(
        observations, [25], exposure_states=exposure, seed=123, horizon=2, embargo=2
    )
    assert set(conditioned[25]) == {73, 97}
    exposure[25] = None
    assert matched_pseudo_entries(
        observations, [25], exposure_states=exposure, seed=123, horizon=2, embargo=2
    ) == {25: ()}


def test_historical_structure_study_keeps_market_side_and_exposure_scope_explicit():
    import datetime as dt

    from src.research.historical.study import study_entry_structure

    start = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
    bars = [
        MarketBar((start + dt.timedelta(hours=i)).isoformat(), 101 + i, 99 + i, 100 + i)
        for i in range(100)
    ]
    result = study_entry_structure(
        bars,
        [25],
        instrument_id="BTC-USDT-SWAP",
        direction="short",
        exposure_states=["flat"] * len(bars),
        seed=123,
        horizon=2,
        embargo=2,
        lookback=3,
    )
    comparison = result["comparisons"][0]
    assert comparison["observed_price_return"] == pytest.approx(0.016)
    assert comparison["observed_directional_return"] == pytest.approx(-0.016)
    assert comparison["pseudo_directional_returns"]
    assert all(value < 0 for value in comparison["pseudo_directional_returns"])
    assert result["specification"]["instrument_id"] == "BTC-USDT-SWAP"
    assert result["specification"]["direction"] == "short"
    assert result["promotable"] is False


def test_invalid_structure_windows_are_rejected():
    with pytest.raises(ValueError, match="windows"):
        structure_features([], lookback=2)


def test_campaign_alignment_converts_dubai_time_and_excludes_unresolved_inventory(tmp_path):
    import datetime as dt
    import json

    from src.research.historical.study import align_campaign_structure

    start = dt.datetime(2025, 7, 12, tzinfo=dt.UTC)
    rows = [
        {
            "open_time_ms": int((start + dt.timedelta(hours=index)).timestamp() * 1000),
            "source_availability_time_ms": int(
                (start + dt.timedelta(hours=index + 1)).timestamp() * 1000
            ),
            "high": 102 + index,
            "low": 100 + index,
            "close": 101 + index,
            "confirmed": True,
        }
        for index in range(100)
    ]
    history = tmp_path / "native.json"
    payload = {
        "scope": {
            "schema": "okx_history/v1",
            "kind": "trade",
            "instrument": "BTC-USDT-SWAP",
            "timeframe": "1H",
        },
        "bars": rows,
    }
    history.write_text(json.dumps(payload))
    study = {
        "timezone_status": "confirmed",
        "campaigns": [
            {
                "campaign_id": "july-loss",
                "symbol": "BTC",
                "side": "short",
                "started_at": "2025-07-14T04:00:00+04:00",
                "minimum_inventory": "0",
            },
            {
                "campaign_id": "missing-open",
                "symbol": "BTC",
                "side": "long",
                "started_at": "2025-07-14T04:00:00+04:00",
                "minimum_inventory": "-1",
            },
        ],
        "inventory_events": [
            {
                "symbol": "BTC",
                "side": "short",
                "quantity": "0",
                "timestamp": "2025-07-12T12:00:00+04:00",
            },
            {
                "symbol": "BTC",
                "side": "long",
                "quantity": "1",
                "timestamp": "2025-07-14T03:30:00+04:00",
            },
        ],
        "order_records": [
            {
                "occurred_at": dt.datetime.fromisoformat("2025-07-14T04:00:00+04:00"),
                "fee": Decimal("-0.01"),
            }
        ],
    }
    result = align_campaign_structure(study, history, horizon=4, embargo=4)
    observation = result["results"]["short"]["campaign_observations"][0]
    assert observation == {
        "campaign_id": "july-loss",
        "index": 46,
        "available_at": "2025-07-13T23:00:00+00:00",
        "exposure_before_entry": "long",
    }
    assert result["results"]["short"]["studies"][0]["specification"]["entry_exposure_states"] == [
        (46, "long")
    ]
    reloaded = json.loads(json.dumps(study, default=str))
    assert (
        align_campaign_structure(reloaded, history, horizon=4, embargo=4)["source_study_hash"]
        == result["source_study_hash"]
    )
    assert result["excluded_campaigns"] == [
        {"campaign_id": "missing-open", "reason": "opening_inventory_unresolved"}
    ]
    assert result["promotable"] is False
    with pytest.raises(ValueError, match="confirmed execution timezone"):
        align_campaign_structure(
            {**study, "timezone_status": "unconfirmed"}, history, horizon=4, embargo=4
        )
    payload["bars"].pop(60)
    history.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cadence gaps"):
        align_campaign_structure(study, history, horizon=4, embargo=4)


def test_bounded_cell_export_roundtrip_preserves_blank_rows_native_ids_and_provenance(tmp_path):
    import json

    from src.research.historical.export import export_workbook
    from src.research.historical.study import import_study, write_study

    exact_id = "90071992547409931234"
    orders = [
        HEADER,
        _row(exact_id, "5/21/25 10:00", "Open long", "10", "10", "0", "-1"),
        [],
        _row(90071992547409940000, "5/21/25 10:01", "Close long", "10", "12", "20", "-1"),
    ]
    positions = [
        [
            "ID",
            "Instrument Type",
            "PnL",
            "Fee",
            "Funding Fee",
            "Liquidation Penalty",
            "Settled PnL",
            "Realized PnL",
        ]
    ]

    class Session:
        def get(self, url, *, params, timeout):
            assert timeout == (10, 60)
            assert url.endswith("/fixture")
            if "ranges" not in params:
                payload = {
                    "spreadsheetId": "fixture",
                    "properties": {"timeZone": "America/New_York"},
                    "sheets": [
                        {
                            "properties": {
                                "title": title,
                                "gridProperties": {"rowCount": 4},
                            }
                        }
                        for title in ("Orders", "Positions")
                    ],
                }
            else:
                import re

                match = re.fullmatch(r"'(\w+)'!A(\d+):[QY](\d+)", params["ranges"])
                title, start, end = match.groups()
                start, end = int(start) - 1, int(end)
                data = orders if title == "Orders" else positions
                payload = {
                    "sheets": [
                        {
                            "properties": {"title": title},
                            "data": [
                                {
                                    "startRow": start,
                                    "rowData": [
                                        {
                                            "values": [
                                                {
                                                    "userEnteredValue": {
                                                        "numberValue"
                                                        if isinstance(v, int)
                                                        else "stringValue": v
                                                    },
                                                    "effectiveValue": {
                                                        "numberValue"
                                                        if isinstance(v, int)
                                                        else "stringValue": v
                                                    },
                                                }
                                                for v in row
                                            ]
                                        }
                                        for row in data[start:end]
                                    ],
                                }
                            ],
                        }
                    ]
                }

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return payload

            return Response()

    source, ids = export_workbook(
        Session(), "fixture", order_sheets=["Orders"], position_sheet="Positions", batch_rows=2
    )
    source_path = write_study(tmp_path / "values", source)
    ids_path = write_study(tmp_path / "ids", ids)
    assert write_study(tmp_path / "values", source) == source_path
    assert source_path.stat().st_mode & 0o777 == 0o600
    result = import_study(source_path, id_cells=ids_path)
    assert result["orders"] == 2
    assert result["id_status"] == {"exact_string": 1, "numeric_precision_unresolved": 1}
    assert result["order_records"][1]["source"]["row_number"] == 4
    assert result["order_records"][0]["order_id"] == exact_id
    assert result["periods"]["entire_export"]["net_before_funding"] == Decimal(18)
    assert result["balanced_campaigns"] == result["reconciled_campaigns"] == 1
    assert result["timezone_status"] == "unconfirmed"
    assert result["promotable"] is False
    ids["sheets"][0]["data"][0]["rowData"][1]["values"][0]["userEnteredValue"] = {
        "numberValue": 90071992547409940000
    }
    changed = write_study(tmp_path / "ids", ids)
    assert (
        import_study(source_path, id_cells=changed)["id_cells_sha256"] != result["id_cells_sha256"]
    )
    with pytest.raises(ValueError, match="row budget"):
        export_workbook(
            Session(),
            "fixture",
            order_sheets=["Orders"],
            position_sheet="Positions",
            maximum_rows=7,
        )
    ids["spreadsheetId"] = "different"
    changed.write_text(json.dumps(ids))
    with pytest.raises(ValueError, match="another source workbook"):
        import_study(source_path, id_cells=changed)
