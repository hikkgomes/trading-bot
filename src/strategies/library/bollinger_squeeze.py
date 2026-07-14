"""Bollinger squeeze breakout — trade the expansion out of a low-volatility coil.

Classic "squeeze" logic: when Bollinger bandwidth compresses into the lowest
``squeeze_pct`` quantile of its recent history, volatility is coiled. The first
close that breaks the band after the squeeze is taken as a directional breakout.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class BollingerSqueezeStrategy(Strategy):
    name = "bollinger_squeeze"
    description = "Break out of a low-volatility Bollinger squeeze in the breakout direction."

    @classmethod
    def default_params(cls):
        return {
            "window": 20,
            "num_std": 2.0,
            "lookback": 120,  # window for ranking bandwidth tightness
            "squeeze_pct": 0.25,  # bandwidth below this quantile == squeeze
            "allow_short": True,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        window, num_std = int(self.params["window"]), float(self.params["num_std"])
        lower, _, upper = ind.bollinger_bands(close, window, num_std)
        bw = ind.bollinger_bandwidth(close, window, num_std)
        lookback = int(self.params["lookback"])
        thresh = bw.rolling(lookback, min_periods=lookback).quantile(
            float(self.params["squeeze_pct"])
        )
        # Squeezed on the *previous* bar, then a band break on this bar.
        was_squeezed = (bw.shift(1) <= thresh.shift(1)).fillna(False)
        sig = self._empty_signals(df)
        sig[was_squeezed & (close > upper)] = 1
        if self.params["allow_short"]:
            sig[was_squeezed & (close < lower)] = -1
        return sig
