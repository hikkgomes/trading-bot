"""Unified strategy framework.

One contract for every strategy paradigm (simple rule, multi-timeframe, ML,
condition-grid). Importing this package registers the bundled library so
``available()`` / ``get()`` see them.

    from src.strategies import get, available, run_backtest, BacktestConfig

    strat = get("sma_cross")(fast=10, slow=30)
    result = run_backtest(strat, df)
    print(result.summary())
"""

# Import the library for its registration side effects.
from src.strategies import library  # noqa: E402,F401
from src.strategies.backtester import BacktestResult, run_backtest
from src.strategies.base import BacktestConfig, Strategy, extract_ohlcv
from src.strategies.registry import available, describe, get, register

__all__ = [
    "Strategy",
    "BacktestConfig",
    "BacktestResult",
    "extract_ohlcv",
    "run_backtest",
    "register",
    "get",
    "available",
    "describe",
]
