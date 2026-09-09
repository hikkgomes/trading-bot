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
    original = structure_features(bars, pivot_width=2, lookback=3)
    assert original[2].high_anchor is None
    assert original[4].high_anchor.anchor_at == bars[2].timestamp
    assert original[4].high_anchor.available_at == bars[4].timestamp
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
