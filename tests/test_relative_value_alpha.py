import math

import pytest

from src.alpha.relative_value import (
    MULTI_LEG_SCHEMA,
    basis_forecast,
    cross_sectional_forecasts,
    pairs_forecast,
)


def test_basis_forecast_is_hedged_and_permanently_non_promotable():
    forecast = basis_forecast(
        symbol="BTCUSDT",
        spot_price=100,
        perpetual_price=101,
        funding_rate=0.0001,
        generated_at="2026-08-10T00:00:00+00:00",
    )

    assert forecast is not None
    payload = forecast.to_dict()
    assert payload["schema"] == MULTI_LEG_SCHEMA
    assert [(leg["market"], leg["side"]) for leg in payload["legs"]] == [
        ("spot", "buy"),
        ("futures", "sell"),
    ]
    assert payload["live_allowed"] is False
    assert payload["promotion_eligible"] is False


def test_basis_forecast_ignores_opportunity_inside_entry_band():
    assert (
        basis_forecast(
            symbol="BTCUSDT",
            spot_price=100,
            perpetual_price=100.01,
            funding_rate=0,
            entry_threshold=0.001,
        )
        is None
    )


def test_cross_sectional_forecasts_are_symmetric_top_and_bottom_selection():
    forecasts = cross_sectional_forecasts(
        {"BTCUSDT": 0.01, "ETHUSDT": 0.03, "SOLUSDT": -0.02, "XRPUSDT": -0.01},
        top_k=1,
        generated_at="2026-08-10T00:00:00+00:00",
    )

    by_symbol = {forecast.symbol: forecast for forecast in forecasts}
    assert set(by_symbol) == {"ETHUSDT", "SOLUSDT"}
    assert by_symbol["ETHUSDT"].direction == "long"
    assert by_symbol["SOLUSDT"].direction == "short"


def test_pairs_forecast_emits_opposing_legs_after_large_divergence():
    second = [100 + index * 0.2 for index in range(60)]
    first = [2 * value for value in second]
    first[-1] *= 1.10

    forecast = pairs_forecast(
        first_symbol="ETHUSDT",
        second_symbol="BTCUSDT",
        first_prices=first,
        second_prices=second,
        entry_z=2,
        generated_at="2026-08-10T00:00:00+00:00",
    )

    assert forecast is not None
    assert forecast.legs[0].side == "sell"
    assert forecast.legs[1].side == "buy"
    assert forecast.score > 0.5
    assert sum(leg.weight for leg in forecast.legs) == pytest.approx(1.0)
    assert forecast.metadata["hedge_ratio"] > 0


def test_pairs_forecast_rejects_non_positive_prices():
    with pytest.raises(ValueError, match="positive"):
        pairs_forecast(
            first_symbol="ETHUSDT",
            second_symbol="BTCUSDT",
            first_prices=[1.0] * 30,
            second_prices=[1.0] * 29 + [math.nan],
        )
