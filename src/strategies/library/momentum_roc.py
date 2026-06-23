"""Time-series momentum — ride sustained rate-of-change, optionally trend-gated.

The simplest expression of the momentum anomaly: if N-bar return is strongly
positive, go long (and vice versa). An optional long-EMA trend filter keeps
entries aligned with the prevailing regime.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class MomentumRocStrategy(Strategy):
    name = "momentum_roc"
    description = "Long when N-bar ROC exceeds +threshold; short below -threshold (optional trend gate)."

    @classmethod
    def default_params(cls):
        return {"period": 24, "threshold": 3.0, "trend_ema": 200, "use_trend_gate": True, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=144)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        r = ind.roc(close, int(self.params["period"]))
        prev = r.shift(1)
        thr = float(self.params["threshold"])
        if self.params["use_trend_gate"]:
            trend = ind.ema(close, int(self.params["trend_ema"]))
            up_regime = close > trend
        else:
            up_regime = pd.Series(True, index=df.index)
        sig = self._empty_signals(df)
        sig[(r > thr) & (prev <= thr) & up_regime] = 1
        if self.params["allow_short"]:
            sig[(r < -thr) & (prev >= -thr) & ~up_regime] = -1
        return sig
