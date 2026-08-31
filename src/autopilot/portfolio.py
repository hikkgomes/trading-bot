"""Unified alpha forecast and deterministic portfolio-allocation contracts."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.build_dataset import TIMEFRAME_SECONDS

ALPHA_FORECAST_SCHEMA = "autopilot.alpha_forecast/v1"
PORTFOLIO_DECISION_SCHEMA = "autopilot.portfolio_decision/v1"
ALPHA_AGGREGATION_SCHEMA = "autopilot.alpha_aggregation/v1"
PORTFOLIO_RISK_MODEL_SCHEMA = "autopilot.portfolio_risk_model/v1"


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class AlphaForecast:
    source_id: str
    product: str
    market: str
    symbol: str
    direction: str
    score: float
    expected_return: float
    confidence: float
    horizon_seconds: int
    generated_at: str

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("alpha source_id cannot be empty")
        if self.direction not in {"long", "short"}:
            raise ValueError("alpha direction must be long or short")
        if not 0 <= self.score <= 1:
            raise ValueError("alpha score must be in [0, 1]")
        if not math.isfinite(self.expected_return):
            raise ValueError("alpha expected_return must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("alpha confidence must be in [0, 1]")
        if self.horizon_seconds <= 0:
            raise ValueError("alpha horizon_seconds must be positive")

    @property
    def signed_score(self) -> float:
        return self.score if self.direction == "long" else -self.score

    @property
    def utility(self) -> float:
        return max(0.0, self.expected_return) * self.confidence * self.score

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ALPHA_FORECAST_SCHEMA, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlphaForecast:
        if payload.get("schema") != ALPHA_FORECAST_SCHEMA:
            raise ValueError("alpha forecast schema is invalid")
        return cls(
            source_id=str(payload["source_id"]),
            product=str(payload["product"]),
            market=str(payload["market"]),
            symbol=str(payload["symbol"]),
            direction=str(payload["direction"]),
            score=float(payload["score"]),
            expected_return=float(payload["expected_return"]),
            confidence=float(payload["confidence"]),
            horizon_seconds=int(payload["horizon_seconds"]),
            generated_at=str(payload["generated_at"]),
        )


@dataclass(frozen=True)
class PortfolioPosition:
    product: str
    symbol: str
    direction: str
    fraction: float

    def __post_init__(self) -> None:
        if self.direction not in {"long", "short"}:
            raise ValueError("portfolio position direction must be long or short")
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError("portfolio position fraction must be in [0, 1]")

    @property
    def signed_fraction(self) -> float:
        return self.fraction if self.direction == "long" else -self.fraction


@dataclass(frozen=True)
class PortfolioPolicy:
    max_positions: int = 3
    max_gross_fraction: float = 0.60
    max_net_fraction: float = 0.40
    max_symbol_fraction: float = 0.25
    min_confidence: float = 0.50
    min_score: float = 0.55
    max_correlated_fraction: float = 0.40
    max_abs_beta_fraction: float = 0.40
    max_drawdown_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 1 <= self.max_positions <= 100:
            raise ValueError("portfolio max_positions must be in [1, 100]")
        for field in (
            "max_gross_fraction",
            "max_net_fraction",
            "max_symbol_fraction",
            "max_correlated_fraction",
            "max_abs_beta_fraction",
            "max_drawdown_fraction",
        ):
            value = float(getattr(self, field))
            if not 0 < value <= 1:
                raise ValueError(f"portfolio {field} must be in (0, 1]")
        for field in ("min_confidence", "min_score"):
            value = float(getattr(self, field))
            if not 0 <= value <= 1:
                raise ValueError(f"portfolio {field} must be in [0, 1]")


@dataclass(frozen=True)
class PortfolioRiskModel:
    generated_at: str
    benchmark_symbol: str
    correlations: Mapping[str, Mapping[str, float]]
    beta_by_symbol: Mapping[str, float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioRiskModel:
        if payload.get("schema") != PORTFOLIO_RISK_MODEL_SCHEMA or payload.get("ok") is not True:
            raise ValueError("portfolio risk model schema/status is invalid")
        correlations = payload.get("correlations")
        beta = payload.get("beta_by_symbol")
        if not isinstance(correlations, Mapping) or not isinstance(beta, Mapping):
            raise ValueError("portfolio risk model correlations/beta must be objects")
        normalized_correlations: dict[str, dict[str, float]] = {}
        for first, row in correlations.items():
            if not isinstance(first, str) or not isinstance(row, Mapping):
                raise ValueError("portfolio correlation matrix is malformed")
            normalized_correlations[first.upper()] = {}
            for second, raw in row.items():
                value = float(raw)
                if not isinstance(second, str) or not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("portfolio correlation value is invalid")
                normalized_correlations[first.upper()][second.upper()] = value
        normalized_beta = {str(symbol).upper(): float(value) for symbol, value in beta.items()}
        if any(not math.isfinite(value) for value in normalized_beta.values()):
            raise ValueError("portfolio beta value is invalid")
        benchmark = str(payload.get("benchmark_symbol") or "").upper()
        if not benchmark or benchmark not in normalized_beta:
            raise ValueError("portfolio risk benchmark is missing")
        return cls(
            generated_at=str(payload.get("generated_at") or ""),
            benchmark_symbol=benchmark,
            correlations=normalized_correlations,
            beta_by_symbol=normalized_beta,
        )

    def correlation(self, first: str, second: str) -> float | None:
        first, second = first.upper(), second.upper()
        if first == second:
            return 1.0
        value = self.correlations.get(first, {}).get(second)
        if value is None:
            value = self.correlations.get(second, {}).get(first)
        return value


def forecast_from_strategy(
    strategy: Mapping[str, Any],
    *,
    product: str,
    market: str,
    symbol: str,
    signal_detail: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> AlphaForecast:
    """Adapt Boolean and scored strategies to one normalized alpha contract."""
    signal_detail = signal_detail or {}
    metrics = strategy.get("metrics") if isinstance(strategy.get("metrics"), Mapping) else {}
    score = min(1.0, max(0.0, _finite(signal_detail.get("alpha_score"), default=1.0)))
    confidence = _finite(metrics.get("dsr_deflated"), default=0.0)
    if confidence <= 0:
        confidence = _finite(strategy.get("baseline_win_rate"), default=0.5)
    confidence = min(1.0, max(0.0, confidence))
    trades = max(1.0, _finite(metrics.get("holdout_trades"), default=1.0))
    expected_return = _finite(metrics.get("holdout_total_return"), default=0.0) / trades
    if expected_return == 0:
        win_probability = _finite(strategy.get("baseline_win_rate"), default=0.5)
        take_profit = _finite(strategy.get("take_profit"), default=0.0)
        stop_loss = _finite(strategy.get("stop_loss"), default=0.0)
        fees = strategy.get("fees") if isinstance(strategy.get("fees"), Mapping) else {}
        round_trip_cost = (
            2 * (_finite(fees.get("fee_bps")) + _finite(fees.get("slippage_bps"))) / 10_000
        )
        expected_return = (
            win_probability * take_profit - (1.0 - win_probability) * stop_loss - round_trip_cost
        )
    base_timeframe = str(strategy.get("base_timeframe") or "5m")
    horizon_bars = max(1, int(strategy.get("horizon_bars") or 1))
    horizon_seconds = int(TIMEFRAME_SECONDS.get(base_timeframe, 300) * horizon_bars)
    return AlphaForecast(
        source_id=str(strategy.get("id") or "unknown"),
        product=product,
        market=market,
        symbol=symbol.upper(),
        direction=str(strategy.get("direction") or ""),
        score=score,
        expected_return=expected_return,
        confidence=confidence,
        horizon_seconds=horizon_seconds,
        generated_at=generated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    )


def _exposure(positions: Iterable[PortfolioPosition]) -> tuple[float, float, dict[str, float]]:
    gross = 0.0
    net = 0.0
    symbols: dict[str, float] = {}
    for position in positions:
        gross += position.fraction
        net += position.signed_fraction
        symbols[position.symbol.upper()] = symbols.get(position.symbol.upper(), 0.0) + abs(
            position.fraction
        )
    return gross, net, symbols


def aggregate_forecasts(
    forecasts: Iterable[AlphaForecast],
    *,
    minimum_net_score: float = 0.10,
) -> dict[str, Any]:
    """Combine reinforcing/conflicting same-instrument forecasts without order bias."""
    items = list(forecasts)
    if not items:
        raise ValueError("alpha aggregation requires at least one forecast")
    identity = {(item.product, item.market, item.symbol.upper()) for item in items}
    if len(identity) != 1:
        raise ValueError("alpha aggregation requires one product/market/symbol identity")
    if not 0 <= minimum_net_score <= 1:
        raise ValueError("minimum_net_score must be in [0, 1]")
    vote_weights = [max(item.confidence, 1e-12) for item in items]
    signed_votes = [
        weight * item.score * (1.0 if item.direction == "long" else -1.0)
        for item, weight in zip(items, vote_weights, strict=True)
    ]
    total_weight = sum(vote_weights)
    net_vote = sum(signed_votes) / total_weight
    absolute_vote = sum(abs(value) for value in signed_votes)
    # Zero scores are valid forecasts (for example, a model can deliberately
    # emit a neutral reading).  They must fail closed as no usable consensus,
    # not turn a reporting/allocation cycle into a division-by-zero failure.
    agreement = abs(sum(signed_votes)) / absolute_vote if absolute_vote else 0.0
    edge_weights = [weight * item.score for item, weight in zip(items, vote_weights, strict=True)]
    total_edge_weight = sum(edge_weights)
    expected_signed = (
        sum(
            edge_weight * item.expected_return * (1.0 if item.direction == "long" else -1.0)
            for item, edge_weight in zip(items, edge_weights, strict=True)
        )
        / total_edge_weight
        if total_edge_weight
        else 0.0
    )
    contributors = [
        {
            "source_id": item.source_id,
            "direction": item.direction,
            "score": item.score,
            "confidence": item.confidence,
            "expected_return": item.expected_return,
            "vote_weight": weight,
        }
        for item, weight in sorted(
            zip(items, vote_weights, strict=True),
            key=lambda pair: pair[0].source_id,
        )
    ]
    blocked_reason = None
    if abs(net_vote) < minimum_net_score:
        blocked_reason = "conflicting_alpha_below_minimum_net_score"
    elif expected_signed == 0 or math.copysign(1, expected_signed) != math.copysign(1, net_vote):
        blocked_reason = "alpha_vote_expected_return_conflict"
    payload = {
        "schema": ALPHA_AGGREGATION_SCHEMA,
        "allowed": blocked_reason is None,
        "reason": blocked_reason,
        "net_score": abs(net_vote),
        "agreement": agreement,
        "conflict_fraction": 1.0 - agreement,
        "contributors": contributors,
        "forecast": None,
    }
    if blocked_reason is not None:
        return payload
    product, market, symbol = next(iter(identity))
    lineage = json.dumps(
        [
            (item.source_id, item.direction)
            for item in sorted(items, key=lambda item: item.source_id)
        ],
        separators=(",", ":"),
    )
    ensemble = AlphaForecast(
        source_id="ensemble:" + hashlib.sha256(lineage.encode()).hexdigest()[:20],
        product=product,
        market=market,
        symbol=symbol,
        direction="long" if net_vote > 0 else "short",
        score=min(1.0, abs(net_vote)),
        expected_return=abs(expected_signed),
        confidence=min(
            1.0,
            agreement
            * sum(
                item.confidence * weight for item, weight in zip(items, vote_weights, strict=True)
            )
            / total_weight,
        ),
        horizon_seconds=int(
            round(
                sum(
                    item.horizon_seconds * weight
                    for item, weight in zip(items, vote_weights, strict=True)
                )
                / total_weight
            )
        ),
        generated_at=max(item.generated_at for item in items),
    )
    payload["forecast"] = ensemble.to_dict()
    return payload


def _allocation_context(
    existing_positions: Iterable[PortfolioPosition],
    requested_fractions: Mapping[str, float] | None,
    policy: PortfolioPolicy,
    risk_model: PortfolioRiskModel | None,
    portfolio_drawdown_fraction: float,
) -> dict[str, Any]:
    positions = list(existing_positions)
    requested_fractions = requested_fractions or {}
    gross, net, symbols = _exposure(positions)
    if not math.isfinite(portfolio_drawdown_fraction) or portfolio_drawdown_fraction > 0:
        raise ValueError("portfolio_drawdown_fraction must be finite and non-positive")
    missing_existing_beta = (
        sorted(
            {
                position.symbol.upper()
                for position in positions
                if position.symbol.upper() not in risk_model.beta_by_symbol
            }
        )
        if risk_model is not None
        else []
    )
    if missing_existing_beta:
        raise ValueError(
            "portfolio risk model is missing beta for existing positions: "
            + ", ".join(missing_existing_beta)
        )
    beta_exposure = (
        sum(
            position.signed_fraction * risk_model.beta_by_symbol[position.symbol.upper()]
            for position in positions
        )
        if risk_model is not None
        else None
    )
    return {
        "positions": positions,
        "requested_fractions": requested_fractions,
        "gross": gross,
        "net": net,
        "symbols": symbols,
        "beta_exposure": beta_exposure,
        "occupied_symbols": {position.symbol.upper(): position.direction for position in positions},
    }


def _forecast_restriction_reason(
    forecast: AlphaForecast,
    *,
    policy: PortfolioPolicy,
    positions: list[PortfolioPosition],
    occupied_symbols: Mapping[str, str],
    portfolio_drawdown_fraction: float,
) -> str | None:
    checks = (
        (forecast.expected_return <= 0, "non_positive_expected_return"),
        (forecast.confidence < policy.min_confidence, "confidence_below_minimum"),
        (forecast.score < policy.min_score, "score_below_minimum"),
        (
            portfolio_drawdown_fraction <= -policy.max_drawdown_fraction,
            "portfolio_drawdown_limit_reached",
        ),
        (len(positions) >= policy.max_positions, "position_capacity_reached"),
        (forecast.symbol.upper() in occupied_symbols, "symbol_already_allocated"),
    )
    for failed, reason in checks:
        if failed:
            return reason
    return None


def _forecast_risk_rooms(
    forecast: AlphaForecast,
    *,
    policy: PortfolioPolicy,
    positions: list[PortfolioPosition],
    risk_model: PortfolioRiskModel | None,
    gross: float,
    net: float,
    symbols: Mapping[str, float],
    beta_exposure: float | None,
    reason: str | None,
) -> tuple[str | None, float, float, float, float | None]:
    gross_room = max(0.0, policy.max_gross_fraction - gross)
    symbol_room = max(
        0.0,
        policy.max_symbol_fraction - symbols.get(forecast.symbol.upper(), 0.0),
    )
    direction = 1.0 if forecast.direction == "long" else -1.0
    net_room = (
        max(0.0, policy.max_net_fraction - net)
        if direction > 0
        else max(0.0, policy.max_net_fraction + net)
    )
    correlation_room = policy.max_correlated_fraction
    beta_room = policy.max_abs_beta_fraction
    correlated_exposure = 0.0
    candidate_beta = None
    if risk_model is not None:
        candidate_beta = risk_model.beta_by_symbol.get(forecast.symbol.upper())
        if candidate_beta is None:
            reason = reason or "portfolio_beta_missing"
        for position in positions:
            correlation = risk_model.correlation(forecast.symbol, position.symbol)
            if correlation is None:
                reason = reason or "portfolio_correlation_missing"
                continue
            same_direction = (
                correlation * direction * (1.0 if position.direction == "long" else -1.0)
            )
            correlated_exposure += max(0.0, same_direction) * position.fraction
        correlation_room = max(0.0, policy.max_correlated_fraction - correlated_exposure)
        if candidate_beta is not None and beta_exposure is not None:
            beta_per_fraction = direction * candidate_beta
            if beta_per_fraction > 0:
                beta_room = max(
                    0.0,
                    (policy.max_abs_beta_fraction - beta_exposure) / beta_per_fraction,
                )
            elif beta_per_fraction < 0:
                beta_room = max(
                    0.0,
                    (policy.max_abs_beta_fraction + beta_exposure) / -beta_per_fraction,
                )
            else:
                beta_room = policy.max_symbol_fraction
    return (
        reason,
        correlated_exposure,
        gross_room,
        min(symbol_room, net_room, correlation_room, beta_room),
        candidate_beta,
    )


def _forecast_allocation_decision(
    forecast: AlphaForecast,
    *,
    context: dict[str, Any],
    policy: PortfolioPolicy,
    risk_model: PortfolioRiskModel | None,
    portfolio_drawdown_fraction: float,
) -> dict[str, Any]:
    requested = min(
        policy.max_symbol_fraction,
        max(
            0.0,
            _finite(
                context["requested_fractions"].get(forecast.source_id),
                default=0.0,
            ),
        ),
    )
    reason = _forecast_restriction_reason(
        forecast,
        policy=policy,
        positions=context["positions"],
        occupied_symbols=context["occupied_symbols"],
        portfolio_drawdown_fraction=portfolio_drawdown_fraction,
    )
    reason, correlated_exposure, gross_room, other_room, candidate_beta = _forecast_risk_rooms(
        forecast,
        policy=policy,
        positions=context["positions"],
        risk_model=risk_model,
        gross=context["gross"],
        net=context["net"],
        symbols=context["symbols"],
        beta_exposure=context["beta_exposure"],
        reason=reason,
    )
    allocated = min(requested, gross_room, other_room) if reason is None else 0.0
    if reason is None and allocated <= 0:
        reason = "exposure_capacity_reached"
    return {
        "source_id": forecast.source_id,
        "symbol": forecast.symbol,
        "direction": forecast.direction,
        "utility": forecast.utility,
        "requested_fraction": requested,
        "allocated_fraction": allocated,
        "allowed": allocated > 0,
        "reason": reason,
        "correlated_existing_fraction": correlated_exposure,
        "candidate_beta": candidate_beta,
        "forecast": forecast.to_dict(),
    }


def allocate_forecasts(
    forecasts: Iterable[AlphaForecast],
    *,
    existing_positions: Iterable[PortfolioPosition] = (),
    requested_fractions: Mapping[str, float] | None = None,
    policy: PortfolioPolicy = PortfolioPolicy(),
    risk_model: PortfolioRiskModel | None = None,
    portfolio_drawdown_fraction: float = 0.0,
) -> dict[str, Any]:
    """Greedily allocate the strongest independent alpha under hard exposure caps."""
    context = _allocation_context(
        existing_positions,
        requested_fractions,
        policy,
        risk_model,
        portfolio_drawdown_fraction,
    )
    positions = context["positions"]
    decisions: list[dict[str, Any]] = []
    ranked = sorted(forecasts, key=lambda item: (item.utility, item.source_id), reverse=True)
    for forecast in ranked:
        decision = _forecast_allocation_decision(
            forecast,
            context=context,
            policy=policy,
            risk_model=risk_model,
            portfolio_drawdown_fraction=portfolio_drawdown_fraction,
        )
        allocated = decision["allocated_fraction"]
        if allocated > 0:
            position = PortfolioPosition(
                product=forecast.product,
                symbol=forecast.symbol,
                direction=forecast.direction,
                fraction=allocated,
            )
            positions.append(position)
            context["gross"] += allocated
            context["net"] += position.signed_fraction
            if context["beta_exposure"] is not None and decision["candidate_beta"] is not None:
                context["beta_exposure"] += position.signed_fraction * decision["candidate_beta"]
            symbol = forecast.symbol.upper()
            context["symbols"][symbol] = context["symbols"].get(symbol, 0.0) + allocated
            context["occupied_symbols"][symbol] = forecast.direction
        decisions.append(decision)
    return {
        "schema": PORTFOLIO_DECISION_SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "policy": dataclasses.asdict(policy),
        "exposure": {
            "positions": len(positions),
            "gross_fraction": round(context["gross"], 12),
            "net_fraction": round(context["net"], 12),
            "symbol_fractions": {
                key: round(value, 12) for key, value in sorted(context["symbols"].items())
            },
            "benchmark_beta_fraction": (
                round(context["beta_exposure"], 12)
                if context["beta_exposure"] is not None
                else None
            ),
            "portfolio_drawdown_fraction": portfolio_drawdown_fraction,
        },
        "decisions": decisions,
    }
