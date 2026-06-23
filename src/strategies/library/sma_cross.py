"""Moving-average crossover — the canonical trend-following baseline."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class SmaCrossStrategy(Strategy):
    name = "sma_cross"
    description = "Long when fast SMA crosses above slow SMA; short on the cross down."

    @classmethod
    def default_params(cls):
        return {"fast": 20, "slow": 50, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ohlcv = self.ohlcv(df)
        close = pd.Series(ohlcv.close, index=df.index)
        fast = ind.sma(close, int(self.params["fast"]))
        slow = ind.sma(close, int(self.params["slow"]))
        sig = self._empty_signals(df)
        sig[ind.crossover(fast, slow)] = 1
        if self.params["allow_short"]:
            sig[ind.crossunder(fast, slow)] = -1
        return sig
