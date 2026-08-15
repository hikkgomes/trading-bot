"""Global platform health and drawdown limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.risk import RiskDecision
from src.risk._common import decision


@dataclass(frozen=True)
class GlobalRiskLimits:
    max_drawdown_fraction: float
    max_data_age_seconds: float
    max_clock_skew_seconds: float


def assess_global_risk(
    *,
    decision_id: str,
    drawdown_fraction: float,
    exchange_connected: bool,
    data_age_seconds: float,
    clock_skew_seconds: float,
    database_healthy: bool,
    execution_drift: bool,
    model_drift: bool,
    limits: GlobalRiskLimits,
) -> RiskDecision:
    snapshot = {
        "drawdown_fraction": drawdown_fraction,
        "exchange_connected": exchange_connected,
        "data_age_seconds": data_age_seconds,
        "clock_skew_seconds": clock_skew_seconds,
        "database_healthy": database_healthy,
        "execution_drift": execution_drift,
        "model_drift": model_drift,
    }
    reason = None
    if drawdown_fraction > limits.max_drawdown_fraction:
        reason = "global_drawdown_limit"
    elif not exchange_connected:
        reason = "exchange_disconnected"
    elif data_age_seconds > limits.max_data_age_seconds:
        reason = "global_data_stale"
    elif abs(clock_skew_seconds) > limits.max_clock_skew_seconds:
        reason = "global_clock_skew"
    elif not database_healthy:
        reason = "database_unhealthy"
    elif execution_drift:
        reason = "execution_drift"
    elif model_drift:
        reason = "model_drift"
    return decision(
        decision_id=decision_id,
        scope="global",
        snapshot=snapshot,
        limits=asdict(limits),
        reason_code=reason,
    )
