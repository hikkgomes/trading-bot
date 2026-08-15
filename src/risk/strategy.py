"""Strategy-level turnover, position, cost, and activity limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.risk import RiskDecision
from src.risk._common import decision


@dataclass(frozen=True)
class StrategyRiskLimits:
    max_position_fraction: float
    max_turnover_fraction: float
    max_trades_per_day: int
    max_slippage_bps: float
    max_funding_cost_fraction: float


def assess_strategy_risk(
    *,
    decision_id: str,
    position_fraction: float,
    turnover_fraction: float,
    trades_today: int,
    expected_slippage_bps: float,
    expected_funding_cost_fraction: float,
    limits: StrategyRiskLimits,
) -> RiskDecision:
    snapshot = {
        "position_fraction": position_fraction,
        "turnover_fraction": turnover_fraction,
        "trades_today": trades_today,
        "expected_slippage_bps": expected_slippage_bps,
        "expected_funding_cost_fraction": expected_funding_cost_fraction,
    }
    reason = None
    if abs(position_fraction) > limits.max_position_fraction:
        reason = "strategy_position_limit"
    elif turnover_fraction > limits.max_turnover_fraction:
        reason = "strategy_turnover_limit"
    elif trades_today >= limits.max_trades_per_day:
        reason = "strategy_trade_limit"
    elif expected_slippage_bps > limits.max_slippage_bps:
        reason = "strategy_slippage_limit"
    elif expected_funding_cost_fraction > limits.max_funding_cost_fraction:
        reason = "strategy_funding_limit"
    return decision(
        decision_id=decision_id,
        scope="strategy",
        snapshot=snapshot,
        limits=asdict(limits),
        reason_code=reason,
    )
