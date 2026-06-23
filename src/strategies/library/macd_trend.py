"""MACD trend-following — trade the MACD/signal line crossover."""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class MacdTrendStrategy(Strategy):
    name = "macd_trend"
    description = "Long when the MACD line crosses above its signal line; short on the cross down."

    @classmethod
    def default_params(cls):
        return {"fast": 12, "slow": 26, "signal": 9, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        macd_line, signal_line, _ = ind.macd(
            close, int(self.params["fast"]), int(self.params["slow"]), int(self.params["signal"])
        )
        sig = self._empty_signals(df)
        sig[ind.crossover(macd_line, signal_line)] = 1
        if self.params["allow_short"]:
            sig[ind.crossunder(macd_line, signal_line)] = -1
        return sig
