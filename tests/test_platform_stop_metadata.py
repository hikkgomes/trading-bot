from __future__ import annotations

import pytest

from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets

NOW = "2026-08-31T10:00:00+00:00"
LATER = "2026-08-31T11:00:00+00:00"


def _forecast(direction: ForecastDirection, expected_return: float) -> AlphaForecast:
    return AlphaForecast(
        strategy_version_id="strategy-1",
        product_id="active_income",
        instrument_id="BTCUSDT",
        direction=direction,
        score=1.0,
        expected_return=expected_return,
        confidence=1.0,
        horizon_seconds=3600,
        valid_from=NOW,
        valid_until=LATER,
        target_volatility=0.1,
        maximum_position=0.2,
    )


def _targets(forecast: AlphaForecast):
    return optimise_targets(
        (forecast,),
        prices={"BTCUSDT": 100.0},
        valid_until=LATER,
        constraints=PortfolioConstraints(portfolio_id="portfolio", equity=10_000.0),
        protective_stop_fraction=0.02,
    )


def test_long_target_carries_a_below_market_protective_stop() -> None:
    target = _targets(_forecast(ForecastDirection.LONG, 0.01))[0]
    assert target.target_quantity > 0
    assert target.metadata["reference_price"] == pytest.approx(100.0)
    assert target.metadata["protective_stop_price"] == pytest.approx(98.0)


def test_short_target_carries_an_above_market_protective_stop() -> None:
    target = _targets(_forecast(ForecastDirection.SHORT, -0.01))[0]
    assert target.target_quantity < 0
    assert target.metadata["reference_price"] == pytest.approx(100.0)
    assert target.metadata["protective_stop_price"] == pytest.approx(102.0)
