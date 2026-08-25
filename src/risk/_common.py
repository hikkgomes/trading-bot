"""Shared deterministic risk-decision construction."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

from src.domain.risk import RiskDecision


def decision(
    *,
    decision_id: str,
    scope: str,
    snapshot: Mapping[str, Any],
    limits: Mapping[str, Any],
    reason_code: str | None,
) -> RiskDecision:
    return RiskDecision(
        decision_id=decision_id,
        scope=scope,
        accepted=reason_code is None,
        reason_code=reason_code,
        evaluated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        input_snapshot=snapshot,
        limits=limits,
    )
