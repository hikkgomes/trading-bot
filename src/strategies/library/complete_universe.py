"""Deterministic CPU-safe implementations for the named strategy manifest.

The named universe shares one causal signal implementation while retaining the
family and strategy identity in the class metadata. Specialist evaluators can
replace a family implementation without changing the registry or queue
contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BacktestConfig, Strategy
from src.strategies.manifest import strategy_manifest
from src.strategies.registry import available, register


def _family_signal(strategy: Strategy, df: pd.DataFrame, family: str) -> pd.Series:
    ohlcv = strategy.ohlcv(df)
    close = pd.Series(ohlcv.close, index=df.index, dtype=float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    lookback = int(strategy.params["lookback"])
    fast = close.ewm(span=max(2, lookback // 2), adjust=False, min_periods=lookback).mean()
    slow = close.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    mean = close.rolling(lookback, min_periods=lookback).mean()
    deviation = close.rolling(lookback, min_periods=lookback).std(ddof=0)
    z_score = (close - mean) / deviation.replace(0.0, np.nan)
    if family in {"mean_reversion", "relative_value"}:
        raw = -z_score
    elif family in {"cross_sectional", "microstructure"}:
        flow = df.get("trade_imbalance")
        if flow is None:
            flow = returns.rolling(lookback, min_periods=lookback).mean()
        raw = pd.Series(flow, index=df.index, dtype=float)
    else:
        raw = (fast - slow) / close.replace(0.0, np.nan)
    threshold = float(strategy.params["threshold"])
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[raw > threshold] = 1
    if strategy.params["allow_short"]:
        signal[raw < -threshold] = -1
    return signal.fillna(0).astype(int)


def _default_params() -> dict[str, object]:
    return {"lookback": 32, "threshold": 0.001, "allow_short": True}


def _default_config() -> BacktestConfig:
    return BacktestConfig(take_profit=0.04, stop_loss=0.025, horizon_bars=144)


def _make_strategy(entry):
    def default_params(cls):
        return _default_params()

    def default_config(cls):
        return _default_config()

    def generate_signals(self, df):
        return _family_signal(self, df, entry.family)

    return type(
        f"{entry.name.title().replace('_', '')}Strategy",
        (Strategy,),
        {
            "__module__": __name__,
            "name": entry.name,
            "description": (f"{entry.family} strategy using the shared causal forecast contract."),
            "default_params": classmethod(default_params),
            "default_config": classmethod(default_config),
            "generate_signals": generate_signals,
        },
    )


for _entry in strategy_manifest():
    if _entry.name not in available():
        register(_make_strategy(_entry))
