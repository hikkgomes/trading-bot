"""Research-paper contracts for basis, cross-sectional, and pairs alpha."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

from src.autopilot.portfolio import AlphaForecast

MULTI_LEG_SCHEMA = "autopilot.multi_leg_alpha_forecast/v1"


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


@dataclass(frozen=True)
class HedgeLeg:
    market: str
    symbol: str
    side: str
    weight: float

    def __post_init__(self) -> None:
        if self.market not in {"spot", "futures"}:
            raise ValueError("hedge leg market must be spot or futures")
        if self.side not in {"buy", "sell"}:
            raise ValueError("hedge leg side must be buy or sell")
        if not self.symbol or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("hedge leg requires a symbol and positive weight")


@dataclass(frozen=True)
class MultiLegAlphaForecast:
    source_id: str
    family: str
    legs: tuple[HedgeLeg, ...]
    score: float
    expected_return: float
    confidence: float
    horizon_seconds: int
    generated_at: str
    requires_borrow: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {"spot_perp_basis", "statistical_pair"}:
            raise ValueError("unsupported multi-leg alpha family")
        if len(self.legs) < 2:
            raise ValueError("multi-leg alpha requires at least two legs")
        if not 0 <= self.score <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("multi-leg score/confidence must be in [0, 1]")
        if not math.isfinite(self.expected_return) or self.horizon_seconds <= 0:
            raise ValueError("multi-leg return/horizon is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MULTI_LEG_SCHEMA,
            **dataclasses.asdict(self),
            "research_only": True,
            "paper_trade_allowed": True,
            "live_allowed": False,
            "promotion_eligible": False,
            "activation_blocked_reason": "atomic_multi_leg_execution_not_approved",
        }


def basis_forecast(
    *,
    symbol: str,
    spot_price: float,
    perpetual_price: float,
    funding_rate: float,
    expected_funding_intervals: int = 3,
    entry_threshold: float = 0.001,
    horizon_seconds: int = 86_400,
    generated_at: str | None = None,
) -> MultiLegAlphaForecast | None:
    """Construct a hedged cash-and-carry/reverse-basis research forecast."""
    spot_price = _positive(spot_price, "spot_price")
    perpetual_price = _positive(perpetual_price, "perpetual_price")
    if not math.isfinite(funding_rate) or expected_funding_intervals < 0:
        raise ValueError("funding assumptions are invalid")
    basis = perpetual_price / spot_price - 1
    carry = funding_rate * expected_funding_intervals
    opportunity = basis + carry
    if abs(opportunity) < entry_threshold:
        return None
    rich_perpetual = opportunity > 0
    legs = (
        HedgeLeg("spot", symbol.upper(), "buy" if rich_perpetual else "sell", 0.5),
        HedgeLeg("futures", symbol.upper(), "sell" if rich_perpetual else "buy", 0.5),
    )
    return MultiLegAlphaForecast(
        source_id=f"basis:{symbol.upper()}",
        family="spot_perp_basis",
        legs=legs,
        score=min(1.0, abs(opportunity) / max(entry_threshold * 4, 1e-12)),
        expected_return=abs(opportunity),
        confidence=min(1.0, 0.5 + abs(opportunity) / max(entry_threshold * 10, 1e-12)),
        horizon_seconds=horizon_seconds,
        generated_at=generated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        requires_borrow=not rich_perpetual,
        metadata={
            "spot_price": spot_price,
            "perpetual_price": perpetual_price,
            "basis": basis,
            "funding_rate_per_8h": funding_rate,
            "expected_funding_intervals": expected_funding_intervals,
            "opportunity": opportunity,
        },
    )


def cross_sectional_forecasts(
    returns_by_symbol: dict[str, float],
    *,
    top_k: int = 2,
    confidence: float = 0.6,
    horizon_seconds: int = 3_600,
    generated_at: str | None = None,
) -> list[AlphaForecast]:
    """Rank a liquid universe and emit symmetric strongest/weakest forecasts."""
    if top_k < 1 or len(returns_by_symbol) < top_k * 2:
        raise ValueError("cross-sectional universe must contain at least 2 * top_k symbols")
    clean = {symbol.upper(): float(value) for symbol, value in returns_by_symbol.items()}
    if any(not math.isfinite(value) for value in clean.values()):
        raise ValueError("cross-sectional returns must be finite")
    mean = sum(clean.values()) / len(clean)
    dispersion = math.sqrt(sum((value - mean) ** 2 for value in clean.values()) / len(clean))
    if dispersion <= 0:
        return []
    ranked = sorted(clean.items(), key=lambda item: (item[1], item[0]))
    selected = [*(ranked[:top_k]), *(ranked[-top_k:])]
    timestamp = generated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    forecasts = []
    for symbol, value in selected:
        relative = value - mean
        forecasts.append(
            AlphaForecast(
                source_id=f"cross_sectional:{symbol}",
                product="active_income",
                market="futures",
                symbol=symbol,
                direction="long" if relative > 0 else "short",
                score=min(1.0, abs(relative) / (3 * dispersion)),
                expected_return=abs(relative),
                confidence=confidence,
                horizon_seconds=horizon_seconds,
                generated_at=timestamp,
            )
        )
    return sorted(forecasts, key=lambda item: item.utility, reverse=True)


def pairs_forecast(
    *,
    first_symbol: str,
    second_symbol: str,
    first_prices: list[float],
    second_prices: list[float],
    entry_z: float = 2.0,
    horizon_seconds: int = 14_400,
    generated_at: str | None = None,
) -> MultiLegAlphaForecast | None:
    """Fit a causal log-price hedge ratio and trade a temporary spread divergence."""
    if len(first_prices) != len(second_prices) or len(first_prices) < 30:
        raise ValueError("pairs forecast requires aligned price histories with at least 30 rows")
    x = [math.log(_positive(value, "second_price")) for value in second_prices]
    y = [math.log(_positive(value, "first_price")) for value in first_prices]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    variance = sum((value - x_mean) ** 2 for value in x)
    if variance <= 0:
        return None
    hedge = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)) / variance
    intercept = y_mean - hedge * x_mean
    spread = [b - (intercept + hedge * a) for a, b in zip(x, y, strict=True)]
    mean = sum(spread) / len(spread)
    std = math.sqrt(sum((value - mean) ** 2 for value in spread) / len(spread))
    if std <= 0:
        return None
    z_score = (spread[-1] - mean) / std
    if abs(z_score) < entry_z:
        return None
    first_rich = z_score > 0
    hedge_weight = abs(hedge)
    total_weight = 1.0 + hedge_weight
    return MultiLegAlphaForecast(
        source_id=f"pair:{first_symbol.upper()}:{second_symbol.upper()}",
        family="statistical_pair",
        legs=(
            HedgeLeg(
                "futures",
                first_symbol.upper(),
                "sell" if first_rich else "buy",
                1.0 / total_weight,
            ),
            HedgeLeg(
                "futures",
                second_symbol.upper(),
                "buy" if first_rich else "sell",
                hedge_weight / total_weight,
            ),
        ),
        score=min(1.0, abs(z_score) / 4),
        expected_return=min(1.0, abs(spread[-1] - mean)),
        confidence=min(0.9, 0.5 + abs(z_score) / 10),
        horizon_seconds=horizon_seconds,
        generated_at=generated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        requires_borrow=False,
        metadata={
            "hedge_ratio": hedge,
            "z_score": z_score,
            "spread": spread[-1],
            "spread_mean": mean,
            "spread_std": std,
        },
    )
