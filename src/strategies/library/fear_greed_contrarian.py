"""Fear & Greed contrarian — buy extreme fear, sell extreme greed.

Requires a ``fear_greed`` column (0-100) on the dataframe; add it with
``src.fear_greed.add_fear_greed_column(df)``. Goes long when sentiment crosses
*into* extreme fear and short when it crosses into extreme greed — the
blueprint's "do the opposite of the crowd at the extremes."
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class FearGreedContrarianStrategy(Strategy):
    name = "fear_greed_contrarian"
    description = "Long on extreme fear, short on extreme greed (needs a `fear_greed` column)."

    @classmethod
    def default_params(cls):
        return {
            "extreme_fear": 25,
            "extreme_greed": 75,
            "column": "fear_greed",
            "allow_short": True,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        # Sentiment is daily and slow-moving — give trades room and time.
        return BacktestConfig(take_profit=0.08, stop_loss=0.05, horizon_bars=288)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        col = self.params["column"]
        if col not in df.columns:
            raise KeyError(
                f"fear_greed_contrarian needs a {col!r} column. "
                f"Add it with src.fear_greed.add_fear_greed_column(df)."
            )
        fng = df[col].astype(float)
        prev = fng.shift(1)
        fear, greed = float(self.params["extreme_fear"]), float(self.params["extreme_greed"])
        sig = self._empty_signals(df)
        sig[(fng <= fear) & (prev > fear)] = 1
        if self.params["allow_short"]:
            sig[(fng >= greed) & (prev < greed)] = -1
        return sig
