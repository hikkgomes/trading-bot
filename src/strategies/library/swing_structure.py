"""Market-structure trend — trade breaks of swing structure (HH/HL vs LH/LL).

The blueprint's trend definition made mechanical: track the most recent
*confirmed* swing high and swing low (pivots), then go long when price breaks
above the last swing high (a higher high / break of structure up) and short when
it breaks below the last swing low. Distinct from the library's moving-average
trend strategies because it keys off actual price pivots, not smoothed averages.
"""

from __future__ import annotations

import pandas as pd

from src.strategies import indicators as ind
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class SwingStructureStrategy(Strategy):
    name = "swing_structure"
    description = "Long on a break above the last swing high; short on a break below the last swing low."

    @classmethod
    def default_params(cls):
        return {"pivot": 5, "allow_short": True}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.06, stop_loss=0.03, horizon_bars=192)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = self.ohlcv(df)
        high = pd.Series(o.high, index=df.index)
        low = pd.Series(o.low, index=df.index)
        close = pd.Series(o.close, index=df.index)
        pivot = int(self.params["pivot"])
        swing_high = ind.last_swing_high(high, pivot)
        swing_low = ind.last_swing_low(low, pivot)

        above = close > swing_high
        below = close < swing_low
        sig = self._empty_signals(df)
        # Entry only on the bar that breaks structure (False -> True transition).
        sig[above & ~above.shift(1, fill_value=False)] = 1
        if self.params["allow_short"]:
            short = below & ~below.shift(1, fill_value=False)
            sig[short & (sig != 1)] = -1
        return sig
