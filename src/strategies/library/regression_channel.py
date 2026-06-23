"""Linear-regression channel — the blueprint's channel as a trend-aware band.

Two trendlines acting as guardrails, built mechanically from a rolling linear
regression of price. ``mode='revert'`` fades touches of the outer band back to
the sloping mean (range/channel trading); ``mode='breakout'`` trades closes that
escape the channel (trend continuation).
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class RegressionChannelStrategy(Strategy):
    name = "regression_channel"
    description = "Trade a rolling linear-regression channel — fade the bands (revert) or follow breaks."

    @classmethod
    def default_params(cls):
        return {"window": 100, "num_std": 2.0, "mode": "revert", "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.04, stop_loss=0.025, horizon_bars=144)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        lower, _, upper = ind.regression_channel(
            close, int(self.params["window"]), float(self.params["num_std"])
        )
        prev = close.shift(1)
        sig = self._empty_signals(df)
        if self.params["mode"] == "breakout":
            long_mask = (close > upper) & (prev <= upper.shift(1))
            short_mask = (close < lower) & (prev >= lower.shift(1))
        else:  # revert: buy the lower band, sell the upper band
            long_mask = (close < lower) & (prev >= lower.shift(1))
            short_mask = (close > upper) & (prev <= upper.shift(1))
        sig[long_mask] = 1
        if self.params["allow_short"]:
            sig[short_mask & ~long_mask] = -1
        return sig
