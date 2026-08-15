"""Account-level risk gates for simultaneous portfolio target changes."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass

from src.domain.risk import RiskDecision


@dataclass(frozen=True)
class AccountRiskLimits:
    max_used_margin_fraction: float
    min_liquidation_buffer_fraction: float
    reject_unknown_exposure: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_used_margin_fraction <= 1:
            raise ValueError("max_used_margin_fraction must be in (0, 1]")
        if not 0 <= self.min_liquidation_buffer_fraction <= 1:
            raise ValueError("min_liquidation_buffer_fraction must be in [0, 1]")


def assess_account_risk(
    *,
    decision_id: str,
    used_margin_fraction: float,
    liquidation_buffer_fraction: float,
    unknown_positions: Mapping[str, float],
    limits: AccountRiskLimits,
) -> RiskDecision:
    snapshot = {
        "used_margin_fraction": used_margin_fraction,
        "liquidation_buffer_fraction": liquidation_buffer_fraction,
        "unknown_positions": dict(unknown_positions),
    }
    reason_code = None
    if limits.reject_unknown_exposure and unknown_positions:
        reason_code = "unknown_exchange_exposure"
    elif used_margin_fraction > limits.max_used_margin_fraction:
        reason_code = "account_margin_limit"
    elif liquidation_buffer_fraction < limits.min_liquidation_buffer_fraction:
        reason_code = "liquidation_buffer_limit"
    return RiskDecision(
        decision_id=decision_id,
        scope="account",
        accepted=reason_code is None,
        reason_code=reason_code,
        evaluated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        input_snapshot=snapshot,
        limits={
            "max_used_margin_fraction": limits.max_used_margin_fraction,
            "min_liquidation_buffer_fraction": limits.min_liquidation_buffer_fraction,
            "reject_unknown_exposure": limits.reject_unknown_exposure,
        },
    )
