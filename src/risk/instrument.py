"""Instrument-level notional, liquidity, spread, and volatility limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.risk import RiskDecision
from src.risk._common import decision


@dataclass(frozen=True)
class InstrumentRiskLimits:
    max_position_notional: float
    max_order_notional: float
    max_visible_depth_fraction: float
    max_spread_bps: float
    max_volatility: float
    max_concentration_fraction: float


def assess_instrument_risk(
    *,
    decision_id: str,
    position_notional: float,
    order_notional: float,
    visible_depth_fraction: float,
    spread_bps: float,
    volatility: float,
    concentration_fraction: float,
    limits: InstrumentRiskLimits,
) -> RiskDecision:
    snapshot = {
        "position_notional": position_notional,
        "order_notional": order_notional,
        "visible_depth_fraction": visible_depth_fraction,
        "spread_bps": spread_bps,
        "volatility": volatility,
        "concentration_fraction": concentration_fraction,
    }
    reason = None
    if abs(position_notional) > limits.max_position_notional:
        reason = "instrument_position_limit"
    elif abs(order_notional) > limits.max_order_notional:
        reason = "instrument_order_limit"
    elif visible_depth_fraction > limits.max_visible_depth_fraction:
        reason = "instrument_depth_limit"
    elif spread_bps > limits.max_spread_bps:
        reason = "instrument_spread_limit"
    elif volatility > limits.max_volatility:
        reason = "instrument_volatility_limit"
    elif abs(concentration_fraction) > limits.max_concentration_fraction:
        reason = "instrument_concentration_limit"
    return decision(
        decision_id=decision_id,
        scope="instrument",
        snapshot=snapshot,
        limits=asdict(limits),
        reason_code=reason,
    )
