"""ATR volatility breakout — enter when price jumps an ATR multiple from the prior close.

A volatility-scaled breakout: the trigger distance adapts to current ATR, so it
fires on genuine expansion moves rather than a fixed percentage move.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class AtrChannelBreakoutStrategy(Strategy):
    name = "atr_channel_breakout"
    description = "Long when close exceeds prior close + k*ATR; short below prior close - k*ATR."

    @classmethod
    def default_params(cls):
        return {"atr_period": 14, "mult": 1.5, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=144)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        a = ind.atr(high, low, close, int(self.params["atr_period"]))
        k = float(self.params["mult"])
        prev_close = close.shift(1)
        upper = prev_close + k * a.shift(1)
        lower = prev_close - k * a.shift(1)
        sig = self._empty_signals(df)
        # False -> True transition so each expansion fires once.
        long_mask = close > upper
        short_mask = close < lower
        sig[long_mask & ~long_mask.shift(1, fill_value=False)] = 1
        if self.params["allow_short"]:
            sig[short_mask & ~short_mask.shift(1, fill_value=False)] = -1
        return sig
