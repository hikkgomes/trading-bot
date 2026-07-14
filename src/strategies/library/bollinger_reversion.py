"""Bollinger Band mean-reversion — fade moves that pierce the outer bands."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"
    description = "Long when close crosses below the lower band; short above the upper band."

    @classmethod
    def default_params(cls):
        return {"window": 20, "num_std": 2.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.03, stop_loss=0.02, horizon_bars=96)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        lower, _, upper = ind.bollinger_bands(
            close, int(self.params["window"]), float(self.params["num_std"])
        )
        prev = close.shift(1)
        sig = self._empty_signals(df)
        # Trigger on the bar that crosses *into* the extreme (not every bar beyond).
        sig[(close < lower) & (prev >= lower.shift(1))] = 1
        if self.params["allow_short"]:
            sig[(close > upper) & (prev <= upper.shift(1))] = -1
        return sig
