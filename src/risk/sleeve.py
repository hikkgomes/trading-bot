"""Portfolio-sleeve capital and factor-risk limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.risk import RiskDecision
from src.risk._common import decision


@dataclass(frozen=True)
class SleeveRiskLimits:
    max_capital_fraction: float
    max_drawdown_fraction: float
    max_correlation: float
    max_abs_beta: float
    max_turnover_fraction: float


def assess_sleeve_risk(
    *,
    decision_id: str,
    capital_fraction: float,
    drawdown_fraction: float,
    maximum_correlation: float,
    beta: float,
    turnover_fraction: float,
    limits: SleeveRiskLimits,
) -> RiskDecision:
    snapshot = {
        "capital_fraction": capital_fraction,
        "drawdown_fraction": drawdown_fraction,
        "maximum_correlation": maximum_correlation,
        "beta": beta,
        "turnover_fraction": turnover_fraction,
    }
    reason = None
    if capital_fraction > limits.max_capital_fraction:
        reason = "sleeve_capital_limit"
    elif drawdown_fraction > limits.max_drawdown_fraction:
        reason = "sleeve_drawdown_limit"
    elif abs(maximum_correlation) > limits.max_correlation:
        reason = "sleeve_correlation_limit"
    elif abs(beta) > limits.max_abs_beta:
        reason = "sleeve_beta_limit"
    elif turnover_fraction > limits.max_turnover_fraction:
        reason = "sleeve_turnover_limit"
    return decision(
        decision_id=decision_id,
        scope="sleeve",
        snapshot=snapshot,
        limits=asdict(limits),
        reason_code=reason,
    )
