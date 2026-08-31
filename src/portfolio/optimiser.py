"""Deterministic, bounded multi-symbol target-position allocator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from src.domain._codec import finite
from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.portfolios import TargetPosition


@dataclass(frozen=True)
class PortfolioConstraints:
    portfolio_id: str
    equity: float
    max_positions: int = 12
    max_gross_fraction: float = 1.0
    max_net_fraction: float = 0.5
    max_symbol_fraction: float = 0.2
    max_abs_beta: float = 0.5
    max_correlation: float = 0.85
    max_margin_fraction: float = 1.0
    max_turnover_fraction: float = 1.0
    max_abs_funding_rate: float = 1.0
    max_cluster_fraction: float = 1.0
    max_drawdown_fraction: float = 1.0
    min_confidence: float = 0.0
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.portfolio_id:
            raise ValueError("portfolio_id cannot be empty")
        if self.equity <= 0:
            raise ValueError("equity must be positive")
        if not 1 <= self.max_positions <= 1_000:
            raise ValueError("max_positions must be in [1, 1000]")
        for field in (
            "max_net_fraction",
            "max_symbol_fraction",
            "max_correlation",
            "max_turnover_fraction",
            "max_abs_funding_rate",
            "max_cluster_fraction",
            "max_drawdown_fraction",
        ):
            value = float(getattr(self, field))
            if not 0 < value <= 1:
                raise ValueError(f"{field} must be in (0, 1]")
        for field in ("max_gross_fraction", "max_abs_beta", "max_margin_fraction"):
            value = float(getattr(self, field))
            if not 0 < value <= 3:
                raise ValueError(f"{field} must be in (0, 3]")
        for field in ("min_confidence", "min_score"):
            value = float(getattr(self, field))
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be in [0, 1]")


def _correlation(correlations: Mapping[str, Mapping[str, float]], first: str, second: str) -> float:
    if first == second:
        return 1.0
    value = correlations.get(first, {}).get(second, correlations.get(second, {}).get(first, 0.0))
    return abs(float(value))


@dataclass
class _AllocationState:
    selected: list[tuple[AlphaForecast, float]] = dataclass_field(default_factory=list)
    gross: float = 0.0
    net: float = 0.0
    beta: float = 0.0
    turnover: float = 0.0
    sleeve_usage: dict[str, float] = dataclass_field(default_factory=dict)
    cluster_usage: dict[str, float] = dataclass_field(default_factory=dict)


def _eligible_forecasts(
    forecasts: Iterable[AlphaForecast],
    *,
    prices: Mapping[str, float],
    funding_rates: Mapping[str, float],
    constraints: PortfolioConstraints,
    product_drawdown_fraction: float,
) -> list[AlphaForecast]:
    candidates = [
        forecast
        for forecast in forecasts
        if forecast.direction is not ForecastDirection.FLAT
        and forecast.confidence >= constraints.min_confidence
        and forecast.score >= constraints.min_score
        and forecast.maximum_position > 0
        and float(prices.get(forecast.instrument_id, 0.0)) > 0
        and abs(float(funding_rates.get(forecast.instrument_id, 0.0)))
        <= constraints.max_abs_funding_rate
        and product_drawdown_fraction <= constraints.max_drawdown_fraction
    ]
    candidates.sort(
        key=lambda item: (item.utility, item.confidence, item.strategy_version_id), reverse=True
    )
    return candidates


def _sized_forecast(
    forecast: AlphaForecast,
    *,
    prices: Mapping[str, float],
    funding_rates: Mapping[str, float],
    observed_volatility: Mapping[str, float],
    liquidity_fraction_caps: Mapping[str, float],
    current_quantities: Mapping[str, float],
    sleeve_budgets: Mapping[str, float],
    cluster_by_instrument: Mapping[str, str],
    cluster_fraction_caps: Mapping[str, float],
    sleeve_usage: Mapping[str, float],
    cluster_usage: Mapping[str, float],
    constraints: PortfolioConstraints,
    gross_limit: float,
    gross: float,
) -> tuple[float, float, str, str] | None:
    funding_rate = float(funding_rates.get(forecast.instrument_id, 0.0))
    direction_sign = 1.0 if forecast.direction is ForecastDirection.LONG else -1.0
    gross_expected_return = direction_sign * forecast.expected_return
    net_expected_return = gross_expected_return - direction_sign * funding_rate
    if net_expected_return <= 0:
        return None
    signal_scale = forecast.score * forecast.confidence
    volatility = float(observed_volatility.get(forecast.instrument_id, 0.0))
    volatility_scale = (
        min(1.0, forecast.target_volatility / volatility)
        if volatility > 0 and forecast.target_volatility > 0
        else 1.0
    )
    funding_scale = min(
        1.0,
        net_expected_return / gross_expected_return if gross_expected_return > 0 else 0.0,
    )
    fraction = (
        min(constraints.max_symbol_fraction, forecast.maximum_position)
        * signal_scale
        * volatility_scale
        * funding_scale
    )
    fraction = min(
        fraction,
        max(0.0, float(liquidity_fraction_caps.get(forecast.instrument_id, 1.0))),
        max(0.0, gross_limit - gross),
    )
    current_fraction = (
        float(current_quantities.get(forecast.instrument_id, 0.0))
        * float(prices[forecast.instrument_id])
        / constraints.equity
    )
    sleeve = str(forecast.metadata.get("sleeve") or "directional")
    sleeve_limit = max(0.0, float(sleeve_budgets.get(sleeve, 1.0)))
    fraction = min(fraction, max(0.0, sleeve_limit - sleeve_usage.get(sleeve, 0.0)))
    cluster = str(
        cluster_by_instrument.get(
            forecast.instrument_id,
            forecast.metadata.get("cluster") or "unclassified",
        )
    )
    cluster_limit = min(
        constraints.max_cluster_fraction,
        max(0.0, float(cluster_fraction_caps.get(cluster, 1.0))),
    )
    fraction = min(fraction, max(0.0, cluster_limit - cluster_usage.get(cluster, 0.0)))
    return fraction, current_fraction, sleeve, cluster


def _try_allocate_forecast(
    state: _AllocationState,
    forecast: AlphaForecast,
    *,
    prices: Mapping[str, float],
    constraints: PortfolioConstraints,
    correlations: Mapping[str, Mapping[str, float]],
    beta_by_instrument: Mapping[str, float],
    observed_volatility: Mapping[str, float],
    liquidity_fraction_caps: Mapping[str, float],
    funding_rates: Mapping[str, float],
    current_quantities: Mapping[str, float],
    sleeve_budgets: Mapping[str, float],
    cluster_by_instrument: Mapping[str, str],
    cluster_fraction_caps: Mapping[str, float],
    gross_limit: float,
) -> bool:
    if any(
        _correlation(correlations, forecast.instrument_id, previous.instrument_id)
        > constraints.max_correlation
        for previous, _ in state.selected
    ):
        return False
    sizing = _sized_forecast(
        forecast,
        prices=prices,
        funding_rates=funding_rates,
        observed_volatility=observed_volatility,
        liquidity_fraction_caps=liquidity_fraction_caps,
        current_quantities=current_quantities,
        sleeve_budgets=sleeve_budgets,
        cluster_by_instrument=cluster_by_instrument,
        cluster_fraction_caps=cluster_fraction_caps,
        sleeve_usage=state.sleeve_usage,
        cluster_usage=state.cluster_usage,
        constraints=constraints,
        gross_limit=gross_limit,
        gross=state.gross,
    )
    if sizing is None:
        return False
    fraction, current_fraction, sleeve, cluster = sizing
    if fraction <= 1e-12:
        return False
    signed = fraction if forecast.direction is ForecastDirection.LONG else -fraction
    candidate_turnover = state.turnover + abs(signed - current_fraction)
    candidate_beta = state.beta + signed * float(
        beta_by_instrument.get(forecast.instrument_id, 0.0)
    )
    if (
        abs(state.net + signed) > constraints.max_net_fraction + 1e-12
        or abs(candidate_beta) > constraints.max_abs_beta + 1e-12
        or candidate_turnover > constraints.max_turnover_fraction + 1e-12
    ):
        return False
    state.selected.append((forecast, signed))
    state.gross += fraction
    state.net += signed
    state.beta = candidate_beta
    state.turnover = candidate_turnover
    state.sleeve_usage[sleeve] = state.sleeve_usage.get(sleeve, 0.0) + fraction
    state.cluster_usage[cluster] = state.cluster_usage.get(cluster, 0.0) + fraction
    return True


def _target_metadata(
    forecast: AlphaForecast,
    *,
    funding_rates: Mapping[str, float],
    observed_volatility: Mapping[str, float],
    liquidity_fraction_caps: Mapping[str, float],
    cluster_by_instrument: Mapping[str, str],
    state: _AllocationState,
) -> dict[str, object]:
    funding_rate = float(funding_rates.get(forecast.instrument_id, 0.0))
    direction_sign = 1.0 if forecast.direction is ForecastDirection.LONG else -1.0
    gross_expected_return = direction_sign * forecast.expected_return
    net_expected_return = gross_expected_return - direction_sign * funding_rate
    metadata: dict[str, object] = {
        "expected_return": forecast.expected_return,
        "directional_expected_return": gross_expected_return,
        "net_expected_return": net_expected_return,
        "confidence": forecast.confidence,
        "score": forecast.score,
        "observed_volatility": observed_volatility.get(forecast.instrument_id),
        "funding_rate": funding_rate,
        "liquidity_fraction_cap": liquidity_fraction_caps.get(forecast.instrument_id, 1.0),
        "sleeve": forecast.metadata.get("sleeve") or "directional",
        "cluster": cluster_by_instrument.get(
            forecast.instrument_id,
            forecast.metadata.get("cluster") or "unclassified",
        ),
        "portfolio_gross_fraction": state.gross,
        "portfolio_net_fraction": state.net,
        "portfolio_beta": state.beta,
        "portfolio_turnover_fraction": state.turnover,
    }
    if forecast.metadata.get("order_group_key"):
        metadata["order_group_key"] = str(forecast.metadata["order_group_key"])
    if forecast.metadata.get("recovery_policy"):
        metadata["recovery_policy"] = str(forecast.metadata["recovery_policy"])
    return metadata


def _build_open_targets(
    state: _AllocationState,
    *,
    constraints: PortfolioConstraints,
    prices: Mapping[str, float],
    valid_until: str,
    risk_budget: float,
    available_margin_fraction: float,
    product_drawdown_fraction: float,
    funding_rates: Mapping[str, float],
    observed_volatility: Mapping[str, float],
    liquidity_fraction_caps: Mapping[str, float],
    cluster_by_instrument: Mapping[str, str],
    protective_stop_fraction: float | None,
) -> list[TargetPosition]:
    targets: list[TargetPosition] = []
    for forecast, fraction in state.selected:
        price = float(prices[forecast.instrument_id])
        metadata = _target_metadata(
            forecast,
            funding_rates=funding_rates,
            observed_volatility=observed_volatility,
            liquidity_fraction_caps=liquidity_fraction_caps,
            cluster_by_instrument=cluster_by_instrument,
            state=state,
        )
        metadata.update(
            {
                "available_margin_fraction": available_margin_fraction,
                "product_drawdown_fraction": product_drawdown_fraction,
            }
        )
        if protective_stop_fraction is not None:
            metadata.update(_protective_stop_metadata(price, fraction, protective_stop_fraction))
        targets.append(
            TargetPosition(
                portfolio_id=constraints.portfolio_id,
                instrument_id=forecast.instrument_id,
                target_quantity=constraints.equity * fraction / price,
                target_notional=constraints.equity * fraction,
                target_fraction=fraction,
                strategy_contributions={forecast.strategy_version_id: fraction},
                risk_budget=risk_budget,
                valid_until=valid_until,
                metadata=metadata,
            )
        )
    return targets


def _build_closing_targets(
    current_quantities: Mapping[str, float],
    *,
    selected_instruments: set[str],
    prices: Mapping[str, float],
    constraints: PortfolioConstraints,
    valid_until: str,
    risk_budget: float,
    exit_reason: str,
) -> list[TargetPosition]:
    targets: list[TargetPosition] = []
    for instrument_id, quantity in sorted(current_quantities.items()):
        if abs(float(quantity)) <= 1e-12 or instrument_id in selected_instruments:
            continue
        price = float(prices.get(instrument_id, 0.0))
        if price <= 0:
            raise ValueError(f"price is required to close existing position {instrument_id}")
        targets.append(
            TargetPosition(
                portfolio_id=constraints.portfolio_id,
                instrument_id=instrument_id,
                target_quantity=0.0,
                target_notional=0.0,
                target_fraction=0.0,
                strategy_contributions={f"portfolio:{exit_reason}": 0.0},
                risk_budget=risk_budget,
                valid_until=valid_until,
                metadata={"reason_code": exit_reason, "current_quantity": quantity},
            )
        )
    return targets


def optimise_targets(
    forecasts: Iterable[AlphaForecast],
    *,
    prices: Mapping[str, float],
    valid_until: str,
    constraints: PortfolioConstraints,
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
    protective_stop_fraction: float | None = None,
) -> tuple[TargetPosition, ...]:
    """Allocate compatible forecasts to simultaneous target positions.

    The optimiser intentionally uses simple deterministic sizing. It accepts
    correlation and beta inputs, enforces gross/net limits, and leaves order
    creation to :mod:`src.execution.order_planner`.
    """
    correlations = correlations or {}
    beta_by_instrument = beta_by_instrument or {}
    observed_volatility = observed_volatility or {}
    liquidity_fraction_caps = liquidity_fraction_caps or {}
    funding_rates = funding_rates or {}
    current_quantities = current_quantities or {}
    sleeve_budgets = sleeve_budgets or {}
    cluster_by_instrument = cluster_by_instrument or {}
    cluster_fraction_caps = cluster_fraction_caps or {}
    if not 0 <= available_margin_fraction <= 1:
        raise ValueError("available_margin_fraction must be in [0, 1]")
    if not 0 <= product_drawdown_fraction <= 1:
        raise ValueError("product_drawdown_fraction must be in [0, 1]")
    if protective_stop_fraction is not None:
        protective_stop_fraction = finite(
            protective_stop_fraction, field="protective_stop_fraction", minimum=0.0
        )
        if not 0 < protective_stop_fraction < 1:
            raise ValueError("protective_stop_fraction must be in (0, 1)")
    gross_limit = min(
        constraints.max_gross_fraction,
        constraints.max_margin_fraction,
        available_margin_fraction,
    )
    candidates = _eligible_forecasts(
        forecasts,
        prices=prices,
        funding_rates=funding_rates,
        constraints=constraints,
        product_drawdown_fraction=product_drawdown_fraction,
    )
    state = _AllocationState()
    for forecast in candidates:
        if len(state.selected) >= constraints.max_positions:
            break
        _try_allocate_forecast(
            state,
            forecast,
            prices=prices,
            constraints=constraints,
            correlations=correlations,
            beta_by_instrument=beta_by_instrument,
            observed_volatility=observed_volatility,
            liquidity_fraction_caps=liquidity_fraction_caps,
            funding_rates=funding_rates,
            current_quantities=current_quantities,
            sleeve_budgets=sleeve_budgets,
            cluster_by_instrument=cluster_by_instrument,
            cluster_fraction_caps=cluster_fraction_caps,
            gross_limit=gross_limit,
        )
    targets = _build_open_targets(
        state,
        constraints=constraints,
        prices=prices,
        valid_until=valid_until,
        risk_budget=risk_budget,
        available_margin_fraction=available_margin_fraction,
        product_drawdown_fraction=product_drawdown_fraction,
        funding_rates=funding_rates,
        observed_volatility=observed_volatility,
        liquidity_fraction_caps=liquidity_fraction_caps,
        cluster_by_instrument=cluster_by_instrument,
        protective_stop_fraction=protective_stop_fraction,
    )
    selected_instruments = {target.instrument_id for target in targets}
    exit_reason = (
        "product_drawdown_limit"
        if product_drawdown_fraction > constraints.max_drawdown_fraction
        else "no_valid_forecast"
    )
    targets.extend(
        _build_closing_targets(
            current_quantities,
            selected_instruments=selected_instruments,
            prices=prices,
            constraints=constraints,
            valid_until=valid_until,
            risk_budget=risk_budget,
            exit_reason=exit_reason,
        )
    )
    return tuple(targets)


def _protective_stop_metadata(
    price: float, signed_fraction: float, stop_fraction: float
) -> dict[str, float]:
    if price <= 0:
        raise ValueError("protective stop reference price must be positive")
    trigger = price * (1.0 - stop_fraction if signed_fraction > 0 else 1.0 + stop_fraction)
    return {
        "reference_price": price,
        "protective_stop_price": trigger,
        "protective_stop_fraction": stop_fraction,
    }
