"""Keltner channel breakout — momentum entry on a close beyond the ATR bands."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class KeltnerBreakoutStrategy(Strategy):
    name = "keltner_breakout"
    description = "Long on a close above the upper Keltner band; short below the lower band."

    @classmethod
    def default_params(cls):
        return {"window": 20, "mult": 2.0, "atr_period": 14, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        lower, _, upper = ind.keltner_channels(
            high,
            low,
            close,
            int(self.params["window"]),
            float(self.params["mult"]),
            int(self.params["atr_period"]),
        )
        prev = close.shift(1)
        sig = self._empty_signals(df)
        sig[(close > upper) & (prev <= upper.shift(1))] = 1
        if self.params["allow_short"]:
            sig[(close < lower) & (prev >= lower.shift(1))] = -1
        return sig
