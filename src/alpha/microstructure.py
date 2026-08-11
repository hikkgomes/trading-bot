"""Typed short-horizon alpha from causal order-book and trade-flow features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.autopilot.portfolio import AlphaForecast


@dataclass(frozen=True)
class MicrostructureAlphaPolicy:
    minimum_abs_score: float = 0.35
    maximum_spread_bps: float = 8.0
    minimum_total_depth: float = 0.01
    minimum_liquidity_vacuum_ratio: float = 0.35
    horizon_seconds: int = 30
    depth_weight: float = 0.25
    aggressor_weight: float = 0.20
    microprice_weight: float = 0.20
    cancel_add_weight: float = 0.15
    liquidation_weight: float = 0.20

    def __post_init__(self) -> None:
        for label in ("minimum_abs_score", "minimum_liquidity_vacuum_ratio"):
            value = float(getattr(self, label))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{label} must be in (0, 1]")
        if not math.isfinite(self.maximum_spread_bps) or self.maximum_spread_bps <= 0:
            raise ValueError("maximum_spread_bps must be finite and positive")
        if not math.isfinite(self.minimum_total_depth) or self.minimum_total_depth <= 0:
            raise ValueError("minimum_total_depth must be finite and positive")
        if self.horizon_seconds < 1 or self.horizon_seconds > 3600:
            raise ValueError("horizon_seconds must be in [1, 3600]")
        weights = (
            self.depth_weight,
            self.aggressor_weight,
            self.microprice_weight,
            self.cancel_add_weight,
            self.liquidation_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("microstructure alpha weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("microstructure alpha weights must sum to 1")


def _bounded(value: Any, scale: float = 1.0) -> float:
    number = float(value or 0.0) / scale
    return max(-1.0, min(1.0, number)) if math.isfinite(number) else 0.0


def forecast_from_microstructure(
    features: dict[str, Any],
    *,
    market: str = "futures",
    policy: MicrostructureAlphaPolicy = MicrostructureAlphaPolicy(),
    generated_at: str,
) -> tuple[AlphaForecast | None, dict[str, Any]]:
    """Convert one causal feature snapshot into a normalized alpha forecast."""
    if features.get("ok") is not True:
        return None, {"eligible": False, "reason": str(features.get("reason") or "invalid_book")}
    spread = float(features["spread_bps"])
    total_depth = float(features["bid_depth"]) + float(features["ask_depth"])
    liquidity = float(features["liquidity_vacuum_ratio"])
    gates = {
        "spread": spread <= policy.maximum_spread_bps,
        "depth": total_depth >= policy.minimum_total_depth,
        "liquidity": liquidity >= policy.minimum_liquidity_vacuum_ratio,
    }
    components = {
        "weighted_depth": _bounded(features.get("weighted_depth_imbalance")),
        "aggressor": _bounded(features.get("aggressor_imbalance")),
        "microprice": _bounded(features.get("microprice_dislocation_bps"), 5.0),
        "cancel_add": _bounded(features.get("cancel_add_pressure")),
        "liquidation": _bounded(features.get("liquidation_imbalance")),
    }
    score = (
        components["weighted_depth"] * policy.depth_weight
        + components["aggressor"] * policy.aggressor_weight
        + components["microprice"] * policy.microprice_weight
        + components["cancel_add"] * policy.cancel_add_weight
        + components["liquidation"] * policy.liquidation_weight
    )
    detail = {
        "eligible": all(gates.values()) and abs(score) >= policy.minimum_abs_score,
        "gates": gates,
        "components": components,
        "signed_score": score,
    }
    if not all(gates.values()):
        detail["reason"] = "microstructure_liquidity_gate"
        return None, detail
    if abs(score) < policy.minimum_abs_score:
        detail["reason"] = "microstructure_score_below_threshold"
        return None, detail
    symbol = str(features.get("symbol") or "").upper()
    if not symbol:
        raise ValueError("microstructure features require a symbol")
    expected_return = max(
        0.0001, abs(score) * 0.001 + abs(float(features["microprice_dislocation_bps"])) / 10_000
    )
    forecast = AlphaForecast(
        source_id=f"microstructure:{symbol}",
        product="active_income",
        market=market,
        symbol=symbol,
        direction="long" if score > 0 else "short",
        score=min(1.0, abs(score)),
        expected_return=expected_return,
        confidence=min(0.95, 0.5 + abs(score) / 2),
        horizon_seconds=policy.horizon_seconds,
        generated_at=generated_at,
    )
    return forecast, detail
