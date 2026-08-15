"""Product-level exposure, margin, drawdown, and loss limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.risk import RiskDecision
from src.risk._common import decision


@dataclass(frozen=True)
class ProductRiskLimits:
    max_gross_fraction: float
    max_net_fraction: float
    max_drawdown_fraction: float
    max_margin_fraction: float
    max_daily_loss_fraction: float


def assess_product_risk(
    *,
    decision_id: str,
    gross_fraction: float,
    net_fraction: float,
    drawdown_fraction: float,
    margin_fraction: float,
    daily_pnl_fraction: float,
    limits: ProductRiskLimits,
) -> RiskDecision:
    snapshot = {
        "gross_fraction": gross_fraction,
        "net_fraction": net_fraction,
        "drawdown_fraction": drawdown_fraction,
        "margin_fraction": margin_fraction,
        "daily_pnl_fraction": daily_pnl_fraction,
    }
    reason = None
    if gross_fraction > limits.max_gross_fraction:
        reason = "product_gross_limit"
    elif abs(net_fraction) > limits.max_net_fraction:
        reason = "product_net_limit"
    elif drawdown_fraction > limits.max_drawdown_fraction:
        reason = "product_drawdown_limit"
    elif margin_fraction > limits.max_margin_fraction:
        reason = "product_margin_limit"
    elif daily_pnl_fraction < -limits.max_daily_loss_fraction:
        reason = "product_daily_loss_limit"
    return decision(
        decision_id=decision_id,
        scope="product",
        snapshot=snapshot,
        limits=asdict(limits),
        reason_code=reason,
    )
