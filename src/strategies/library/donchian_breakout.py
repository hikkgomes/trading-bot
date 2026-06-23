"""Donchian channel breakout — classic momentum/turtle-style entry."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class DonchianBreakoutStrategy(Strategy):
    name = "donchian_breakout"
    description = "Long on a close above the prior N-bar high; short below the prior N-bar low."

    @classmethod
    def default_params(cls):
        return {"channel": 55, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.08, stop_loss=0.04, horizon_bars=288)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ohlcv = self.ohlcv(df)
        close = pd.Series(ohlcv.close, index=df.index)
        high = pd.Series(ohlcv.high, index=df.index)
        low = pd.Series(ohlcv.low, index=df.index)
        n = int(self.params["channel"])
        # Prior-bar channel (shifted) so the breakout bar itself isn't included.
        upper = ind.rolling_high(high, n).shift(1)
        lower = ind.rolling_low(low, n).shift(1)
        prev_close = close.shift(1)
        sig = self._empty_signals(df)
        sig[(close > upper) & (prev_close <= upper)] = 1
        if self.params["allow_short"]:
            sig[(close < lower) & (prev_close >= lower)] = -1
        return sig
