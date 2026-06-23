"""Z-score mean-reversion — fade statistically extreme deviations from the mean.

A pure, parameter-light reversion baseline: standardize price against its own
rolling mean/std and trade when the deviation exceeds ``entry_z`` standard
deviations, betting on reversion toward the mean.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class ZScoreReversionStrategy(Strategy):
    name = "zscore_reversion"
    description = "Long when price z-score drops below -entry_z; short above +entry_z."

    @classmethod
    def default_params(cls):
        return {"window": 50, "entry_z": 2.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.025, stop_loss=0.02, horizon_bars=96)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        z = ind.zscore(close, int(self.params["window"]))
        prev = z.shift(1)
        entry = float(self.params["entry_z"])
        sig = self._empty_signals(df)
        sig[(z < -entry) & (prev >= -entry)] = 1
        if self.params["allow_short"]:
            sig[(z > entry) & (prev <= entry)] = -1
        return sig
