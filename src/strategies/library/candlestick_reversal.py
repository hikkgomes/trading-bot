"""Candlestick reversal patterns — multi-candle bullish/bearish rejection entries.

Trades the classic single/two-candle reversal signals (engulfing, hammer,
shooting star). An optional trend filter only takes bullish reversals during a
pullback (price below an SMA) and bearish reversals during a pop above it — the
"reversal at an extreme" idea from the blueprint, which raises signal quality.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class CandlestickReversalStrategy(Strategy):
    name = "candlestick_reversal"
    description = "Long on bullish engulfing/hammer, short on bearish engulfing/shooting star."

    @classmethod
    def default_params(cls):
        return {"trend_sma": 50, "use_trend_filter": True, "wick_ratio": 2.0, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.03, stop_loss=0.02, horizon_bars=96)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        open_ = pd.Series(o.open, index=df.index)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        wr = float(self.params["wick_ratio"])

        bull = ind.bullish_engulfing(open_, high, low, close) | ind.hammer(
            open_, high, low, close, wr
        )
        bear = ind.bearish_engulfing(open_, high, low, close) | ind.shooting_star(
            open_, high, low, close, wr
        )

        if self.params["use_trend_filter"]:
            ma = ind.sma(close, int(self.params["trend_sma"]))
            bull &= close < ma  # bullish reversal only when pulled back below the mean
            bear &= close > ma  # bearish reversal only when stretched above it

        sig = self._empty_signals(df)
        sig[bull] = 1
        if self.params["allow_short"]:
            sig[bear & ~bull] = -1
        return sig
