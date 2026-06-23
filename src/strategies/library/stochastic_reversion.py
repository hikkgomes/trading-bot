"""Stochastic oscillator reversion — buy oversold/sell overbought %K-%D crosses."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class StochasticReversionStrategy(Strategy):
    name = "stochastic_reversion"
    description = "Long on %K/%D cross up from oversold; short on cross down from overbought."

    @classmethod
    def default_params(cls):
        return {"k_period": 14, "d_period": 3, "oversold": 20.0, "overbought": 80.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.03, stop_loss=0.02, horizon_bars=96)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        k, d = ind.stochastic(high, low, close, int(self.params["k_period"]), int(self.params["d_period"]))
        sig = self._empty_signals(df)
        sig[ind.crossover(k, d) & (d < float(self.params["oversold"]))] = 1
        if self.params["allow_short"]:
            sig[ind.crossunder(k, d) & (d > float(self.params["overbought"]))] = -1
        return sig
