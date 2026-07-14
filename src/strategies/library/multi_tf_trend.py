"""Multi-timeframe trend alignment.

A higher-timeframe trend filter gates lower-timeframe entries: only take longs
while the higher TF is trending up, and shorts while it trends down. Demonstrates
how to consume the project's ``tf_{timeframe}_`` columns when present, with a
graceful fallback (a long EMA on the base close) so it still runs on plain OHLCV.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class MultiTimeframeTrendStrategy(Strategy):
    name = "multi_tf_trend"
    description = "Higher-TF EMA trend filter gating base-TF EMA-cross entries."

    @classmethod
    def default_params(cls):
        return {
            "higher_tf": "4h",  # tf_4h_close used as the trend filter if present
            "higher_ema": 50,
            "fast": 12,
            "slow": 26,
            "fallback_trend_ema": 200,  # used when the higher-TF column is absent
            "allow_short": True,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.03, horizon_bars=192)

    def _trend(self, df: pd.DataFrame, base_close: pd.Series) -> pd.Series:
        col = f"tf_{self.params['higher_tf']}_close"
        if col in df.columns:
            htf = df[col].astype(float)
            return (htf > ind.ema(htf, int(self.params["higher_ema"]))).astype(int) * 2 - 1
        # Fallback: slope of a long EMA on the base close.
        trend_ema = ind.ema(base_close, int(self.params["fallback_trend_ema"]))
        return (base_close > trend_ema).astype(int) * 2 - 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ohlcv = self.ohlcv(df)
        close = pd.Series(ohlcv.close, index=df.index)
        fast = ind.ema(close, int(self.params["fast"]))
        slow = ind.ema(close, int(self.params["slow"]))
        trend = self._trend(df, close)  # +1 up, -1 down

        sig = self._empty_signals(df)
        long_entry = ind.crossover(fast, slow) & (trend > 0)
        sig[long_entry] = 1
        if self.params["allow_short"]:
            short_entry = ind.crossunder(fast, slow) & (trend < 0)
            sig[short_entry] = -1
        return sig
