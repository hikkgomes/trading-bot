"""Causal predicate evaluation on a ``tf_{tf}_``-prefixed aligned frame.

Split out of ``evaluate.py`` so the execution bot can import the *exact* mask
logic that validated a hypothesis without dragging in the research stack
(``evaluate`` pulls ``src.metrics`` -> scipy, which is deliberately not part of
``requirements-bot.txt``). This module depends on numpy/pandas only.

Research (``evaluate.py`` / ``validation.py``) and execution (``src.run_bot``)
both call :func:`entry_mask`, so a strategy trades live with the same code path
that scored it — the whole point of the research -> bot contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research_exploration.hypothesis_schema import Hypothesis, Predicate
from src.build_dataset import TIMEFRAME_PREFIXES, TIMEFRAME_SECONDS


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise KeyError(f"Column {name!r} not in aligned frame (have {len(frame.columns)} cols).")
    return frame[name].replace([np.inf, -np.inf], np.nan)


def timeframe_ratio(tf: str, base_tf: str | None) -> int:
    """How many base-tf bars make up one ``tf`` bar (>=1). 1 when base unknown
    or same tf. Raises if a timeframe duration is unknown (caller treats this as
    an ``invalid_timeframe_window`` rejection)."""
    if base_tf is None or tf == base_tf:
        return 1
    if tf not in TIMEFRAME_SECONDS or base_tf not in TIMEFRAME_SECONDS:
        raise KeyError(f"Unknown timeframe duration for {tf!r} or {base_tf!r}")
    ratio = TIMEFRAME_SECONDS[tf] / TIMEFRAME_SECONDS[base_tf]
    return max(1, round(ratio))


def effective_rolling_window(p: Predicate, base_tf: str | None) -> int:
    """Rolling window expressed in NATIVE bars of the predicate's timeframe,
    converted to rows of the ``base_tf``-aligned table.

    A higher-tf column is forward-filled onto the base frame, so each native
    value repeats ``timeframe_ratio`` times. Scaling the window by that ratio
    makes ``window`` mean "N native HTF bars", not "N base rows" — and because
    the fill is uniform, the empirical quantile over the scaled window equals the
    quantile over the N native values."""
    win = int(p.window or 100)
    return win * timeframe_ratio(p.timeframe, base_tf)


def predicate_history_bars(p: Predicate, base_tf: str | None) -> int:
    """Minimum base-tf rows of history needed for this predicate's mask to be
    defined at the last row (rolling windows, shifts, lookbacks). Used by the
    executor to size live candle fetches so live == research."""
    ratio = timeframe_ratio(p.timeframe, base_tf)
    if p.op in ("q_ge", "q_le"):
        return effective_rolling_window(p, base_tf) + ratio
    if p.op in ("rising", "falling", "slope_up", "slope_down"):
        return (int(p.lookback or 1) + 1) * ratio
    if p.op in ("cross_above", "cross_below"):
        return (2 + int(p.shift_b or 0)) * ratio
    if p.shift_b:
        return (1 + int(p.shift_b)) * ratio
    return ratio


def predicate_mask(frame: pd.DataFrame, p: Predicate, base_tf: str | None = None) -> pd.Series:
    a = _col(frame, p.column())
    op = p.op

    def _b() -> pd.Series:
        b = _col(frame, p.column_b())  # type: ignore[arg-type]
        return b.shift(p.shift_b) if p.shift_b else b

    if op in ("gt", "ge", "lt", "le"):
        ref = float(p.reference)
        return {"gt": a > ref, "ge": a >= ref, "lt": a < ref, "le": a <= ref}[op]
    if op in ("gt_feature", "lt_feature"):
        b = _b()
        return a > b if op == "gt_feature" else a < b
    if op in ("cross_above", "cross_below"):
        b = _b()
        if op == "cross_above":
            return (a > b) & (a.shift(1) <= b.shift(1))
        return (a < b) & (a.shift(1) >= b.shift(1))
    if op in ("rising", "falling"):
        lb = int(p.lookback or 1)
        return a > a.shift(lb) if op == "rising" else a < a.shift(lb)
    if op in ("slope_up", "slope_down"):
        lb = int(p.lookback or 1)
        slope = (a - a.shift(lb)) / lb
        ref = float(p.reference or 0.0)
        return slope > ref if op == "slope_up" else slope < ref
    if op in ("pct_above", "pct_below"):
        b = _b()
        pct = a / b.replace(0, np.nan) - 1.0
        ref = float(p.reference)
        return pct > ref if op == "pct_above" else pct < ref
    if op == "between":
        return (a >= float(p.low)) & (a <= float(p.high))
    if op in ("q_ge", "q_le"):
        win = effective_rolling_window(p, base_tf)
        q = float(p.quantile)
        thresh = a.rolling(win, min_periods=max(10, win // 4)).quantile(q)
        return a >= thresh if op == "q_ge" else a <= thresh
    if op in ("bullish", "bearish", "nonzero"):
        return {"bullish": a > 0, "bearish": a < 0, "nonzero": a != 0}[op]
    raise ValueError(f"Unhandled op {op!r}")


def entry_mask(frame: pd.DataFrame, hyp: Hypothesis, cfg=None) -> pd.Series:
    """AND of every regime/setup/trigger predicate, plus optional vol filters."""
    mask = pd.Series(True, index=frame.index)
    for p in hyp.all_predicates():
        mask &= predicate_mask(frame, p, base_tf=hyp.base_timeframe).fillna(False)
    # Risk volatility filter (skip too-quiet / too-wild bars), if NATR available.
    natr_col = f"{TIMEFRAME_PREFIXES[hyp.base_timeframe]}natr_14"
    if natr_col in frame.columns and (hyp.risk.min_atr_pct or hyp.risk.max_atr_pct):
        natr_pct = frame[natr_col] / 100.0
        if hyp.risk.min_atr_pct:
            mask &= natr_pct >= hyp.risk.min_atr_pct
        if hyp.risk.max_atr_pct:
            mask &= natr_pct <= hyp.risk.max_atr_pct
    return mask.fillna(False)


def hypothesis_history_requirements(hyp: Hypothesis) -> dict[str, int]:
    """Per-timeframe minimum bars (in NATIVE bars of each tf) the executor must
    have on hand for every predicate of ``hyp`` to be defined on the last closed
    bar. Includes the base timeframe, sized to cover the largest base-row
    requirement across predicates."""
    base_tf = hyp.base_timeframe
    base_rows = 1
    native: dict[str, int] = {base_tf: 1}
    for p in hyp.all_predicates():
        base_rows = max(base_rows, predicate_history_bars(p, base_tf))
        ratio = timeframe_ratio(p.timeframe, base_tf)
        native_bars = -(-predicate_history_bars(p, base_tf) // ratio)  # ceil div
        native[p.timeframe] = max(native.get(p.timeframe, 1), native_bars)
    native[base_tf] = max(native[base_tf], base_rows)
    return native
