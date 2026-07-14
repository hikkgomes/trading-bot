"""Strategy base class + shared data helpers.

Every strategy — whether a one-line moving-average cross, a multi-timeframe
trend filter, an ML classifier, or a wrapper around the condition-grid search —
implements the same contract:

    signals = strategy.generate_signals(df)   # int Series in {-1, 0, +1}

A value of +1 at bar *i* means "open a long at the open of bar i+1"; -1 means
open a short; 0 means do nothing. The backtester (``src.strategies.backtester``)
turns those entry signals into non-overlapping trades with fees, slippage and
TP/SL/time exits — the *same* trade model the search engine uses, so framework
results are comparable to search results.

Strategies are deliberately decoupled from the heavy precomputed dataset: they
read OHLCV (and any extra feature columns they need) off a plain DataFrame.
``extract_ohlcv`` accepts both a plain ``open/high/low/close`` schema and the
project's ``tf_{timeframe}_{field}`` prefixed schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

_OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class OHLCV:
    """Price/volume arrays + the index they came from."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None
    index: pd.Index


def _find_column(df: pd.DataFrame, field_name: str, base_tf: str | None) -> str | None:
    candidates = []
    if base_tf:
        candidates.append(f"tf_{base_tf}_{field_name}")
    candidates.append(field_name)
    # Fall back to the first tf_*_<field> column if no explicit base tf given.
    for col in candidates:
        if col in df.columns:
            return col
    if base_tf is None:
        prefixed = [c for c in df.columns if c.endswith(f"_{field_name}") and c.startswith("tf_")]
        if len(prefixed) == 1:
            return prefixed[0]
    return None


def extract_ohlcv(df: pd.DataFrame, base_tf: str | None = None) -> OHLCV:
    """Pull OHLCV arrays from ``df`` supporting plain and ``tf_{tf}_`` schemas.

    ``volume`` is optional (set to None when absent). Raises if OHLC are missing.
    """
    cols = {f: _find_column(df, f, base_tf) for f in _OHLCV_FIELDS}
    missing = [f for f in ("open", "high", "low", "close") if cols[f] is None]
    if missing:
        raise KeyError(
            f"Could not resolve OHLC column(s) {missing} (base_tf={base_tf!r}). "
            f"Available columns sample: {list(df.columns)[:12]}"
        )
    vol = df[cols["volume"]].to_numpy(dtype=float) if cols["volume"] else None
    return OHLCV(
        open=df[cols["open"]].to_numpy(dtype=float),
        high=df[cols["high"]].to_numpy(dtype=float),
        low=df[cols["low"]].to_numpy(dtype=float),
        close=df[cols["close"]].to_numpy(dtype=float),
        volume=vol,
        index=df.index,
    )


@dataclass
class BacktestConfig:
    """Trade-model parameters shared with the search engine.

    ``horizon_bars`` is the time-stop (max holding bars). ``pnl_unit='btc'``
    switches to the BTC-accumulation convention where only shorts realise a
    return (being long == simply holding BTC, so it dodges dips).
    """

    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    take_profit: float = 0.05
    stop_loss: float = 0.03
    horizon_bars: int = 96
    pnl_unit: str = "usdt"  # "usdt" | "btc"
    initial_equity: float = 10_000.0

    @property
    def round_trip_cost(self) -> float:
        """Entry+exit cost as a fraction, matching strategy_search."""
        return 2.0 * ((self.fee_bps + self.slippage_bps) / 10_000.0)


class Strategy(ABC):
    """Abstract base for all strategies.

    Subclasses set a unique class-level ``name`` (and ``description``), declare
    their tunable ``default_params``, and implement ``generate_signals``. ML
    strategies additionally override ``fit``.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, **params):
        self.params: dict = {**self.default_params(), **params}
        # Set by the backtester/runner so generate_signals can resolve the
        # right tf_{tf}_ OHLCV columns when several timeframes are present.
        self.base_tf: str | None = None

    @classmethod
    def default_params(cls) -> dict:
        """Tunable parameters and their defaults. Override in subclasses."""
        return {}

    @classmethod
    def default_config(cls) -> BacktestConfig:
        """Backtest config this strategy was designed for. Callers may override
        any field. Override to ship sensible TP/SL/horizon defaults."""
        return BacktestConfig()

    def fit(self, df: pd.DataFrame) -> Strategy:
        """Optional training hook (ML strategies). No-op for rule strategies."""
        return self

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return an int Series in {-1, 0, +1} aligned to ``df.index``."""
        raise NotImplementedError

    # -- helpers available to every subclass --------------------------------
    def ohlcv(self, df: pd.DataFrame) -> OHLCV:
        """Resolve OHLCV using this strategy's ``base_tf`` (set by the runner)."""
        return extract_ohlcv(df, base_tf=self.base_tf)

    def _empty_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index, dtype=int)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, params={self.params})"
