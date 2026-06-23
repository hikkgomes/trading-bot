"""SuperTrend — ATR trailing-stop trend follower; trade on the trend flip."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class SupertrendStrategy(Strategy):
    name = "supertrend"
    description = "Long when SuperTrend flips up, short when it flips down (ATR trailing stop)."

    @classmethod
    def default_params(cls):
        return {"period": 10, "mult": 3.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.04, horizon_bars=288)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        direction = ind.supertrend(high, low, close, int(self.params["period"]), float(self.params["mult"]))
        prev = direction.shift(1)
        sig = self._empty_signals(df)
        sig[(direction > 0) & (prev <= 0)] = 1
        if self.params["allow_short"]:
            sig[(direction < 0) & (prev >= 0)] = -1
        return sig
