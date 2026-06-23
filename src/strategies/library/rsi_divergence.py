"""RSI divergence — trade momentum/price disagreement at swing pivots.

Bullish divergence (price lower low, RSI higher low) flags downside exhaustion;
bearish divergence (price higher high, RSI lower high) flags upside exhaustion.
Signals fire only when the pivot is confirmed (``pivot`` bars after it forms),
so the strategy is lookahead-safe.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class RsiDivergenceStrategy(Strategy):
    name = "rsi_divergence"
    description = "Long on bullish RSI divergence, short on bearish RSI divergence at swing pivots."

    @classmethod
    def default_params(cls):
        return {"rsi_period": 14, "pivot": 5, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.04, stop_loss=0.025, horizon_bars=144)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        rsi = ind.rsi(close, int(self.params["rsi_period"]))
        bull, bear = ind.regular_divergence(high, low, rsi, int(self.params["pivot"]))
        sig = self._empty_signals(df)
        sig[bull] = 1
        if self.params["allow_short"]:
            sig[bear & ~bull] = -1
        return sig
