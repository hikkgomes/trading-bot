"""Pure-pandas indicator helpers for the strategy library.

These intentionally avoid TA-Lib so framework strategies are self-contained and
testable on synthetic data on any machine. The heavy dataset build still uses
TA-Lib; these are for strategies that compute their own features on the fly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (only gains) RSI is 100; when avg_gain == 0 it's 0.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_high(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).max()


def rolling_low(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).min()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where ``fast`` crosses from below to above ``slow``."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where ``fast`` crosses from above to below ``slow``."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change (percent) over ``period`` bars."""
    return close.pct_change(period) * 100.0


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (s - rolling mean) / rolling std."""
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / std.replace(0.0, np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (lower, middle, upper) Bollinger bands."""
    middle = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return middle - num_std * std, middle, middle + num_std * std


def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """(upper - lower) / middle — a normalized volatility/squeeze gauge."""
    lower, middle, upper = bollinger_bands(close, window, num_std)
    return (upper - lower) / middle.replace(0.0, np.nan)


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    mult: float = 2.0,
    atr_period: int = 14,
):
    """Return (lower, middle, upper) Keltner channels (EMA mid + ATR bands)."""
    middle = ema(close, window)
    band = mult * atr(high, low, close, atr_period)
    return middle - band, middle, middle + band


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
):
    """Return (%K, %D) of the stochastic oscillator (0..100)."""
    hh = rolling_high(high, k_period)
    ll = rolling_low(low, k_period)
    percent_k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    percent_d = percent_k.rolling(d_period, min_periods=d_period).mean()
    return percent_k, percent_d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R (-100..0). -80 is oversold, -20 overbought."""
    hh = rolling_high(high, period)
    ll = rolling_low(low, period)
    return -100.0 * (hh - close) / (hh - ll).replace(0.0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index via Wilder smoothing.

    Returns (adx, plus_di, minus_di). ADX measures trend *strength* (not
    direction); +DI/-DI give direction. A common filter is ``adx > 25``.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    # Wilder smoothing == EWM with alpha = 1/period.
    atr_w = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_w
    )
    minus_di = (
        100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_w
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_line = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_line, plus_di, minus_di


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0
) -> pd.Series:
    """SuperTrend direction as a Series in {-1, +1} (+1 = uptrend).

    Standard ATR-band trailing-stop construction with the usual stateful band
    carry-forward. Implemented with an explicit loop because each bar's band
    depends on the prior bar's final band and trend state.
    """
    hl2 = (high + low) / 2.0
    atr_s = atr(high, low, close, period)
    upper_basic = (hl2 + mult * atr_s).to_numpy()
    lower_basic = (hl2 - mult * atr_s).to_numpy()
    close_a = close.to_numpy()
    n = len(close_a)
    direction: Any = np.ones(n, dtype=int)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)

    for i in range(n):
        if i == 0 or np.isnan(upper_basic[i]) or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper_basic[i]
            final_lower[i] = lower_basic[i]
            direction[i] = 1
            continue
        final_upper[i] = (
            upper_basic[i]
            if (upper_basic[i] < final_upper[i - 1] or close_a[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower_basic[i]
            if (lower_basic[i] > final_lower[i - 1] or close_a[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )
        if close_a[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close_a[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=close.index)


# --------------------------------------------------------------------------- candles
def _candle_parts(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    """Return (body, upper_wick, lower_wick) magnitudes for each candle."""
    body = (close - open_).abs()
    top = pd.concat([open_, close], axis=1).max(axis=1)
    bot = pd.concat([open_, close], axis=1).min(axis=1)
    return body, high - top, bot - low


def bullish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Current green candle whose body fully engulfs the prior red candle's body."""
    prev_red = close.shift(1) < open_.shift(1)
    curr_green = close > open_
    engulf = (close >= open_.shift(1)) & (open_ <= close.shift(1))
    return (prev_red & curr_green & engulf).fillna(False)


def bearish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Current red candle whose body fully engulfs the prior green candle's body."""
    prev_green = close.shift(1) > open_.shift(1)
    curr_red = close < open_
    engulf = (open_ >= close.shift(1)) & (close <= open_.shift(1))
    return (prev_green & curr_red & engulf).fillna(False)


def hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, wick_ratio: float = 2.0
) -> pd.Series:
    """Small body near the top with a long lower wick (bullish rejection)."""
    body, upper, lower = _candle_parts(open_, high, low, close)
    return ((lower >= wick_ratio * body) & (upper <= body) & (body > 0)).fillna(False)


def shooting_star(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, wick_ratio: float = 2.0
) -> pd.Series:
    """Small body near the bottom with a long upper wick (bearish rejection)."""
    body, upper, lower = _candle_parts(open_, high, low, close)
    return ((upper >= wick_ratio * body) & (lower <= body) & (body > 0)).fillna(False)


def doji(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, max_body_frac: float = 0.1
) -> pd.Series:
    """Indecision candle: body is a tiny fraction of the total range."""
    body = (close - open_).abs()
    rng = (high - low).replace(0.0, np.nan)
    return (body / rng <= max_body_frac).fillna(False)


# --------------------------------------------------------------------------- swing structure
def last_swing_high(high: pd.Series, pivot: int = 5) -> pd.Series:
    """Most recent *confirmed* swing-high level, forward-filled (lookahead-safe).

    A bar is a swing high if it is the max over ``pivot`` bars on each side; that
    pivot is only *confirmed* ``pivot`` bars later, so the level is shifted
    forward by ``pivot`` before being forward-filled. At any bar t the result
    uses only pivots already confirmed by t.
    """
    window = 2 * pivot + 1
    is_pivot = high.eq(high.rolling(window, center=True).max())
    level = high.where(is_pivot)
    return level.shift(pivot).ffill()


def last_swing_low(low: pd.Series, pivot: int = 5) -> pd.Series:
    """Most recent confirmed swing-low level, forward-filled (lookahead-safe)."""
    window = 2 * pivot + 1
    is_pivot = low.eq(low.rolling(window, center=True).min())
    level = low.where(is_pivot)
    return level.shift(pivot).ffill()


# --------------------------------------------------------------------------- BTC macro/cycle
def mayer_multiple(close: pd.Series, window: int = 200) -> pd.Series:
    """Price / its long SMA. >~2.4 has historically flagged overheated BTC tops
    (tuned for daily bars; on intraday it is just a long-MA stretch ratio)."""
    return close / sma(close, window)


def pi_cycle_top(close: pd.Series, fast: int = 111, slow: int = 350) -> pd.Series:
    """Pi-Cycle Top signal: True when the fast SMA crosses above 2x the slow SMA.

    A well-known BTC cycle-top indicator (111DMA vs 2*350DMA), tuned for daily
    bars. Returns a boolean Series (rare True spikes near macro tops)."""
    return crossover(sma(close, fast), 2.0 * sma(close, slow))


# --------------------------------------------------------------------------- divergence
def _pivot_positions(values: np.ndarray, pivot: int, kind: str) -> np.ndarray:
    """Indices that are local extrema over +/- ``pivot`` bars (kind: 'low'|'high')."""
    n = values.size
    out = []
    for i in range(pivot, n - pivot):
        window = values[i - pivot : i + pivot + 1]
        if kind == "low" and values[i] == window.min():
            out.append(i)
        elif kind == "high" and values[i] == window.max():
            out.append(i)
    return np.asarray(out, dtype=int)


def regular_divergence(high: pd.Series, low: pd.Series, osc: pd.Series, pivot: int = 5):
    """Detect regular RSI/oscillator divergence at confirmed swing pivots.

    Returns (bullish, bearish) boolean Series. Bullish = price prints a *lower
    low* while the oscillator prints a *higher low* (downside exhaustion);
    bearish = price *higher high* while oscillator *lower high*. Lookahead-safe:
    each signal is placed at the bar where the pivot is confirmed (``pivot`` bars
    after it forms) and only compares against the prior confirmed pivot.
    """
    bull = pd.Series(False, index=low.index)
    bear = pd.Series(False, index=high.index)
    low_a, high_a, osc_a = low.to_numpy(), high.to_numpy(), osc.to_numpy()
    n = low_a.size

    prev = None
    for p in _pivot_positions(low_a, pivot, "low"):
        conf = p + pivot
        if conf >= n or np.isnan(osc_a[p]):
            prev = p
            continue
        if prev is not None and low_a[p] < low_a[prev] and osc_a[p] > osc_a[prev]:
            bull.iloc[conf] = True
        prev = p

    prev = None
    for p in _pivot_positions(high_a, pivot, "high"):
        conf = p + pivot
        if conf >= n or np.isnan(osc_a[p]):
            prev = p
            continue
        if prev is not None and high_a[p] > high_a[prev] and osc_a[p] < osc_a[prev]:
            bear.iloc[conf] = True
        prev = p

    return bull, bear


# --------------------------------------------------------------------------- regression channel
def regression_channel(close: pd.Series, window: int = 100, num_std: float = 2.0):
    """Rolling linear-regression channel: (lower, mid, upper).

    ``mid`` is the regression fit value at the end of each trailing ``window``;
    the bands are +/- ``num_std`` residual standard deviations. A trend-aware
    cousin of Bollinger bands (the centre line slopes with the trend). Vectorised
    via a shared pseudo-inverse since the x-grid is identical for every window.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    y = close.to_numpy(dtype=float)
    n = y.size
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    if n >= window:
        x = np.arange(window)
        A = np.vstack([x, np.ones(window)]).T  # (window, 2)
        pinv = np.linalg.pinv(A)  # (2, window)
        win = sliding_window_view(y, window)  # (n-window+1, window)
        coeffs = win @ pinv.T  # (m, 2): slope, intercept
        fit = coeffs[:, [0]] * x + coeffs[:, [1]]  # (m, window)
        resid_std = (win - fit).std(axis=1)
        fit_last = coeffs[:, 0] * (window - 1) + coeffs[:, 1]
        sl = slice(window - 1, n)
        mid[sl] = fit_last
        lower[sl] = fit_last - num_std * resid_std
        upper[sl] = fit_last + num_std * resid_std
    idx = close.index
    return (pd.Series(lower, index=idx), pd.Series(mid, index=idx), pd.Series(upper, index=idx))
