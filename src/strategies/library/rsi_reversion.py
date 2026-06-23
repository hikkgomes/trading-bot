"""RSI mean-reversion — buy oversold, sell overbought."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"
    description = "Long when RSI dips below oversold; short when RSI rises above overbought."

    @classmethod
    def default_params(cls):
        return {"period": 14, "oversold": 30.0, "overbought": 70.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.03, stop_loss=0.02, horizon_bars=96)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ohlcv = self.ohlcv(df)
        close = pd.Series(ohlcv.close, index=df.index)
        r = ind.rsi(close, int(self.params["period"]))
        prev = r.shift(1)
        sig = self._empty_signals(df)
        # Trigger on the bar that *crosses into* the zone (avoids re-firing).
        sig[(r < self.params["oversold"]) & (prev >= self.params["oversold"])] = 1
        if self.params["allow_short"]:
            sig[(r > self.params["overbought"]) & (prev <= self.params["overbought"])] = -1
        return sig
