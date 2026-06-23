"""BTC accumulation guard — step aside near cycle tops and on trend breaks.

Built for the position / BTC-accumulation product (``pnl_unit='btc'``), where a
SHORT signal means "step out of BTC" and a realised drop while stepped aside is
captured as relative BTC gain (being long == simply holding). It emits a
step-aside signal when *either*:

* the market is overheated — Mayer Multiple above ``mayer_top`` or a Pi-Cycle Top
  cross (the blueprint's "extreme greed precedes risk"), or
* the macro trend breaks — close falls below a long trend EMA ("if price breaks
  below these key levels the trend has likely shifted").

It never emits longs: in BTC mode you accumulate by *holding* and only act to
dodge drawdowns. The cycle indicators are tuned for daily/weekly bars; on
intraday data they behave as long-horizon stretch/trend filters.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class BtcCycleGuardStrategy(Strategy):
    name = "btc_cycle_guard"
    description = "Step aside (short) near cycle tops (Mayer/Pi-Cycle) or on a trend-EMA break; hold otherwise."

    @classmethod
    def default_params(cls):
        return {
            "trend_ema": 200,
            "mayer_window": 200,
            "mayer_top": 2.4,
            "use_pi_cycle": True,
            "use_mayer": True,
            "use_trend_break": True,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        # BTC-denominated, long horizon: a step-aside lasts until the dip plays out.
        return BacktestConfig(pnl_unit="btc", take_profit=0.10, stop_loss=0.05, horizon_bars=384)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        step_aside = pd.Series(False, index=df.index)

        if self.params["use_trend_break"]:
            step_aside |= close < ind.ema(close, int(self.params["trend_ema"]))
        if self.params["use_mayer"]:
            mm = ind.mayer_multiple(close, int(self.params["mayer_window"]))
            step_aside |= mm > float(self.params["mayer_top"])
        if self.params["use_pi_cycle"]:
            step_aside |= ind.pi_cycle_top(close)

        step_aside = step_aside.fillna(False)
        sig = self._empty_signals(df)
        # Act once when the guard first triggers (False -> True transition).
        sig[step_aside & ~step_aside.shift(1, fill_value=False)] = -1
        return sig
