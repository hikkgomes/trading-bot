"""Shared effective risk policy for research, export, and paper/live artifacts.

Generated hypotheses carry an expressive :class:`RiskRule`, while deployment
also imposes product-level safety envelopes.  Keeping the conversion here makes
the research simulator and exported artifact use the same effective values.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_exploration.hypothesis_schema import Hypothesis

DEFAULT_DAILY_STOP_LOSS = -0.02
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3
DEFAULT_COOLDOWN_BARS = 12

EXECUTION_RISK_ENVELOPES = {
    ("futures", "usdt"): {
        "max_risk_per_trade": 0.005,
        "max_position_fraction": 0.25,
        "max_daily_loss": 0.03,
        "max_consecutive_losses": 4,
        "max_trades_per_day": 8,
        "min_cooldown_bars": 12,
        "default_max_trades_per_day": 4,
    },
    ("spot", "btc"): {
        "max_risk_per_trade": 0.003,
        "max_position_fraction": 0.35,
        "max_daily_loss": 0.01,
        "max_consecutive_losses": 3,
        "max_trades_per_day": 2,
        "min_cooldown_bars": 24,
        "default_max_trades_per_day": 1,
    },
}


@dataclass(frozen=True)
class EffectiveRisk:
    risk_per_trade: float
    daily_stop_loss: float
    max_position_fraction: float
    max_consecutive_losses: int
    cooldown_bars: int
    max_trades_per_day: int

    def position_fraction(self, stop_loss_fraction: float) -> float:
        """Return the exact fraction used by ``PaperTradingBot`` for an entry."""

        if stop_loss_fraction <= 0:
            return 0.0
        return min(
            self.risk_per_trade / float(stop_loss_fraction),
            self.max_position_fraction,
            1.0,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "risk_per_trade": self.risk_per_trade,
            "daily_stop_loss": self.daily_stop_loss,
            "max_position_fraction": self.max_position_fraction,
            "max_consecutive_losses": self.max_consecutive_losses,
            "cooldown_bars": self.cooldown_bars,
            "max_trades_per_day": self.max_trades_per_day,
        }


def _execution_risk_envelope(market: str, pnl_unit: str) -> dict[str, float | int]:
    key = (str(market).lower(), str(pnl_unit).lower())
    envelope = EXECUTION_RISK_ENVELOPES.get(key)
    if envelope is None:
        raise ValueError(f"No execution risk envelope for market={key[0]!r}, pnl_unit={key[1]!r}")
    return envelope


def effective_risk(hypothesis: Hypothesis, *, market: str, pnl_unit: str) -> EffectiveRisk:
    """Resolve hypothesis risk through the same product envelope used at export."""

    envelope = _execution_risk_envelope(market, pnl_unit)
    requested_risk_per_trade = float(hypothesis.risk.risk_per_trade or 0.01)
    risk_per_trade = min(
        requested_risk_per_trade,
        float(envelope["max_risk_per_trade"]),
    )
    max_position_fraction = min(
        float(hypothesis.risk.max_position_fraction),
        float(envelope["max_position_fraction"]),
    )
    if hypothesis.risk.max_daily_loss_r:
        daily_stop_loss = -abs(float(hypothesis.risk.max_daily_loss_r) * requested_risk_per_trade)
    else:
        daily_stop_loss = DEFAULT_DAILY_STOP_LOSS
    daily_stop_loss = max(daily_stop_loss, -float(envelope["max_daily_loss"]))
    max_consecutive_losses = min(
        DEFAULT_MAX_CONSECUTIVE_LOSSES,
        int(envelope["max_consecutive_losses"]),
    )
    cooldown_bars = max(
        int(hypothesis.risk.cooldown_bars or DEFAULT_COOLDOWN_BARS),
        int(envelope["min_cooldown_bars"]),
    )
    if hypothesis.risk.max_trades_per_day is None:
        max_trades_per_day = int(envelope["default_max_trades_per_day"])
    else:
        max_trades_per_day = min(
            int(hypothesis.risk.max_trades_per_day),
            int(envelope["max_trades_per_day"]),
        )
    return EffectiveRisk(
        risk_per_trade=risk_per_trade,
        daily_stop_loss=daily_stop_loss,
        max_position_fraction=max_position_fraction,
        max_consecutive_losses=max_consecutive_losses,
        cooldown_bars=cooldown_bars,
        max_trades_per_day=max_trades_per_day,
    )


def effective_risk_block(
    hypothesis: Hypothesis,
    *,
    market: str,
    pnl_unit: str,
) -> dict[str, float | int]:
    """Artifact-shaped wrapper used by the exporter."""

    return effective_risk(hypothesis, market=market, pnl_unit=pnl_unit).to_dict()
