"""Multi-symbol USDT futures portfolio product."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.domain.forecasts import AlphaForecast
from src.domain.portfolios import TargetPosition
from src.portfolio.aggregation import aggregate_forecasts
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets


@dataclass(frozen=True)
class ActiveIncomePortfolio:
    constraints: PortfolioConstraints

    def target_positions(
        self,
        forecasts: Iterable[AlphaForecast],
        *,
        prices: Mapping[str, float],
        valid_until: str,
        correlations: Mapping[str, Mapping[str, float]] | None = None,
        beta_by_instrument: Mapping[str, float] | None = None,
        observed_volatility: Mapping[str, float] | None = None,
        liquidity_fraction_caps: Mapping[str, float] | None = None,
        funding_rates: Mapping[str, float] | None = None,
        current_quantities: Mapping[str, float] | None = None,
        sleeve_budgets: Mapping[str, float] | None = None,
        cluster_by_instrument: Mapping[str, str] | None = None,
        cluster_fraction_caps: Mapping[str, float] | None = None,
        product_drawdown_fraction: float = 0.0,
        available_margin_fraction: float = 1.0,
        risk_budget: float = 1.0,
    ) -> tuple[TargetPosition, ...]:
        return optimise_targets(
            aggregate_forecasts(forecasts),
            prices=prices,
            valid_until=valid_until,
            constraints=self.constraints,
            correlations=correlations,
            beta_by_instrument=beta_by_instrument,
            observed_volatility=observed_volatility,
            liquidity_fraction_caps=liquidity_fraction_caps,
            funding_rates=funding_rates,
            current_quantities=current_quantities,
            sleeve_budgets=sleeve_budgets,
            cluster_by_instrument=cluster_by_instrument,
            cluster_fraction_caps=cluster_fraction_caps,
            product_drawdown_fraction=product_drawdown_fraction,
            available_margin_fraction=available_margin_fraction,
            risk_budget=risk_budget,
        )
