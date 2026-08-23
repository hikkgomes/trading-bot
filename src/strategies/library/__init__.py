"""Importing this package registers every bundled strategy.

Add a new strategy by dropping a module here that defines a ``Strategy``
subclass decorated with ``@register`` from ``src.strategies.registry``, then
import it below.
"""

from src.strategies.library import (  # noqa: F401,E402,I001
    adx_trend,
    atr_channel_breakout,
    bollinger_reversion,
    bollinger_squeeze,
    btc_cycle_guard,
    candlestick_reversal,
    condition_grid,
    donchian_breakout,
    fear_greed_contrarian,
    keltner_breakout,
    macd_trend,
    ml_classifier,
    ml_regressor,
    momentum_roc,
    multi_tf_trend,
    regime_filter,
    regression_channel,
    rsi_divergence,
    rsi_reversion,
    sma_cross,
    stochastic_reversion,
    supertrend,
    swing_structure,
    zscore_reversion,
)
