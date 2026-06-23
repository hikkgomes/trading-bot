"""ADX-filtered directional trend — only trade DI crossovers in a strong trend.

ADX measures trend *strength* without direction; +DI/-DI give direction. We
enter only when ADX confirms a real trend (``adx_min``), which filters out the
chop where DI crossovers whipsaw.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class AdxTrendStrategy(Strategy):
    name = "adx_trend"
    description = "Long/short on +DI/-DI crossovers, gated by ADX trend-strength threshold."

    @classmethod
    def default_params(cls):
        return {"period": 14, "adx_min": 25.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        adx_line, plus_di, minus_di = ind.adx(high, low, close, int(self.params["period"]))
        strong = adx_line > float(self.params["adx_min"])
        sig = self._empty_signals(df)
        sig[ind.crossover(plus_di, minus_di) & strong] = 1
        if self.params["allow_short"]:
            sig[ind.crossunder(plus_di, minus_di) & strong] = -1
        return sig
