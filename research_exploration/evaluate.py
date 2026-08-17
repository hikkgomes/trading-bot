"""Adapter: turn a structured :class:`Hypothesis` into trades and metrics.

Design choices that keep results honest and comparable to the rest of the repo:

* **Same trade model.** Entries are simulated with the *exact* canonical engine
  (``src.strategies.backtester._simulate``): next-bar-open entry, stop-before-TP
  intrabar, non-overlapping trades, round-trip cost ``2*(fee+slip)/1e4``.
* **No lookahead.** Higher timeframes are aligned onto the base frame with the
  closed-candle ``merge_asof`` convention from ``src.build_dataset`` (the HTF
  candle is shifted forward by its own duration, so it's only visible *after* it
  closes). Every predicate uses only current/past bars.
* **Causal thresholds.** Quantile predicates use a *rolling* window, never a
  global fit, so a candidate is testable out-of-sample by construction.

This module ships two frame builders:
  - ``build_aligned_frame``   – real data; loads only the needed columns and a
    bounded time window. **Heavy** — only call with the user's approval.
  - ``build_synthetic_aligned_frame`` – tiny in-memory frame for smoke tests.

Run (safe, synthetic):  python -m research_exploration.evaluate --synthetic
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.predicates import (  # noqa: F401  (re-exported API)
    _col,
    effective_rolling_window,
    entry_mask,
    predicate_mask,
    timeframe_ratio,
)
from research_exploration.risk_policy import EffectiveRisk, effective_risk
from src import metrics
from src.build_dataset import TIMEFRAME_PREFIXES, TIMEFRAME_SECONDS
from src.config import INDICATOR_DATA_DIR
from src.strategies.backtester import _simulate
from src.strategies.base import BacktestConfig
from src.trade_utils import gross_return_for_pnl_unit


def _configured_market() -> str:
    try:
        import build_binance_indicator_dataset as bbid

        market = str(getattr(bbid, "MARKET", "futures")).lower()
    except Exception:
        market = "futures"
    if market not in {"spot", "futures"}:
        raise ValueError(f"Unsupported market {market!r}; expected 'spot' or 'futures'")
    return market


@dataclass
class EvalConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    pnl_unit: str = "usdt"
    market: str = field(default_factory=_configured_market)
    initial_equity: float = 10_000.0
    use_atr_exits: bool = False  # convert ExitRule ATR multiples -> fractional via median NATR%
    min_trades: int = 30  # below this, verdict can't be "keep"
    entry_delay_bars: int = 0  # stress-only: additional bars after the signal
    adverse_fill_bps: float = 0.0  # stress-only: worsen the entry fill by direction
    exit_delay_bars: int = 0  # stress-only: defer exit execution to a later open
    adverse_exit_bps: float = 0.0  # stress-only: worsen the delayed exit fill
    funding_bps_per_8h: float = 0.0  # futures-only conservative holding debit

    def __post_init__(self) -> None:
        self.market = str(self.market).lower()
        if self.market not in {"spot", "futures"}:
            raise ValueError(f"market must be 'spot' or 'futures', got {self.market!r}")
        self.pnl_unit = str(self.pnl_unit).lower()
        if self.pnl_unit not in {"btc", "usdt"}:
            raise ValueError(f"pnl_unit must be 'btc' or 'usdt', got {self.pnl_unit!r}")
        for label, value in (
            ("fee_bps", self.fee_bps),
            ("slippage_bps", self.slippage_bps),
            ("adverse_fill_bps", self.adverse_fill_bps),
            ("adverse_exit_bps", self.adverse_exit_bps),
            ("funding_bps_per_8h", self.funding_bps_per_8h),
        ):
            if not np.isfinite(value) or not 0 <= float(value) < 10_000:
                raise ValueError(f"{label} must be finite and in [0, 10000)")
        for label, value in (
            ("entry_delay_bars", self.entry_delay_bars),
            ("exit_delay_bars", self.exit_delay_bars),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not np.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError("initial_equity must be finite and positive")
        if (
            not isinstance(self.min_trades, int)
            or isinstance(self.min_trades, bool)
            or self.min_trades <= 0
        ):
            raise ValueError("min_trades must be a positive integer")


def signals_from_hypothesis(
    frame: pd.DataFrame, hyp: Hypothesis, cfg: EvalConfig | None = None
) -> pd.Series:
    mask = entry_mask(frame, hyp, cfg)
    sig = np.where(mask.to_numpy(), 1 if hyp.direction == "long" else -1, 0)
    return pd.Series(sig, index=frame.index, dtype=int)


# --------------------------------------------------------------------------- #
# Backtest + metrics
# --------------------------------------------------------------------------- #
def _resolve_exit(frame: pd.DataFrame, hyp: Hypothesis, cfg: EvalConfig) -> BacktestConfig:
    tp, sl = hyp.exit.take_profit, hyp.exit.stop_loss
    if cfg.use_atr_exits and hyp.exit.atr_take_profit and hyp.exit.atr_stop_loss:
        natr_col = f"{TIMEFRAME_PREFIXES[hyp.base_timeframe]}natr_14"
        if natr_col in frame.columns:
            med = float((frame[natr_col] / 100.0).median())
            if med and np.isfinite(med):
                tp = hyp.exit.atr_take_profit * med
                sl = hyp.exit.atr_stop_loss * med
    return BacktestConfig(
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        take_profit=float(tp),
        stop_loss=float(sl),
        horizon_bars=int(hyp.exit.horizon_bars),
        pnl_unit=cfg.pnl_unit,
        initial_equity=cfg.initial_equity,
    )


def _ohlc(frame: pd.DataFrame, base_tf: str):
    pfx = TIMEFRAME_PREFIXES[base_tf]
    cols = {f: f"{pfx}{f}" for f in ("open", "high", "low", "close")}
    missing = [c for c in cols.values() if c not in frame.columns]
    if missing:
        raise KeyError(f"Aligned frame missing base OHLC columns: {missing}")
    return (
        frame[cols["open"]].to_numpy(float),
        frame[cols["high"]].to_numpy(float),
        frame[cols["low"]].to_numpy(float),
        frame[cols["close"]].to_numpy(float),
    )


def _breakdown(trades: pd.DataFrame, by: str) -> list[dict]:
    if trades.empty:
        return []
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True).dt.tz_localize(None)
    key = t["entry_time"].dt.year if by == "year" else t["entry_time"].dt.to_period("M").astype(str)
    rows = []
    for k, g in t.groupby(key):
        r = g["net_return"].to_numpy()
        rows.append(
            {
                by: str(k),
                "trades": int(len(g)),
                "win_rate": round(float((r > 0).mean()), 4),
                "total_return": round(float(np.prod(1 + r) - 1), 5),
                "avg_net": round(float(r.mean()), 6),
            }
        )
    return rows


def _trade_index(frame: pd.DataFrame) -> pd.Index:
    """Index passed to the simulator as trade entry/exit times. Real aligned
    frames carry timestamps as a *column* over a RangeIndex — use it, otherwise
    breakdowns would date every trade to the 1970 epoch."""
    if "timestamp" in frame.columns:
        return pd.Index(pd.to_datetime(frame["timestamp"], utc=True))
    return frame.index


def _daily_session(index: pd.Index, position: int, base_timeframe: str) -> object:
    """Return the UTC trading day, with a deterministic bar-day fallback."""

    if isinstance(index, pd.DatetimeIndex):
        value = index[position]
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return value.date()
    bars_per_day = max(1, 86_400 // TIMEFRAME_SECONDS[base_timeframe])
    return position // bars_per_day


def _accumulate_daily_return(current: float, sized_return: float) -> float:
    """Match the paper/live daily-stop accounting's conservative update."""

    additive = current + sized_return
    compounded = ((1.0 + current) * (1.0 + sized_return)) - 1.0
    return min(additive, compounded)


def _simulate_one_entry(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    index: pd.Index,
    *,
    signal_index: int,
    signal: int,
    cfg: BacktestConfig,
    entry_delay_bars: int,
    adverse_fill_bps: float,
    exit_delay_bars: int,
    adverse_exit_bps: float,
) -> tuple[dict, int, int] | None:
    """Use the canonical simulator for one accepted, possibly stressed entry."""

    local_start = signal_index + entry_delay_bars
    entry_index = local_start + 1
    local_stop = entry_index + int(cfg.horizon_bars) + int(exit_delay_bars) + 1
    if local_stop > len(close):
        return None
    local_open = open_[local_start:local_stop].copy()
    local_high = high[local_start:local_stop].copy()
    local_low = low[local_start:local_stop].copy()
    local_close = close[local_start:local_stop]
    local_direction = np.zeros(len(local_close), dtype=int)
    local_direction[0] = signal
    if adverse_fill_bps:
        adjustment = float(adverse_fill_bps) / 10_000.0
        local_open[1] *= 1.0 + adjustment if signal > 0 else 1.0 - adjustment
        local_high[1] = max(local_high[1], local_open[1])
        local_low[1] = min(local_low[1], local_open[1])
    rows = _simulate(
        local_open,
        local_high,
        local_low,
        local_close,
        local_direction,
        index[local_start:local_stop],
        cfg,
    )
    if not rows:
        return None
    trade = rows[0]
    trade["signal_time"] = index[signal_index]
    exit_index = entry_index + int(trade["holding_bars"])
    if exit_delay_bars or adverse_exit_bps:
        original_exit_index = exit_index
        original_exit_price = float(trade["exit"])
        exit_index += int(exit_delay_bars)
        if exit_index >= len(open_):
            return None
        delayed_exit = float(open_[exit_index]) if exit_delay_bars else original_exit_price
        if adverse_exit_bps:
            adjustment = float(adverse_exit_bps) / 10_000.0
            delayed_exit *= 1.0 - adjustment if signal > 0 else 1.0 + adjustment
        gross = gross_return_for_pnl_unit(
            float(trade["entry"]),
            delayed_exit,
            is_long=signal > 0,
            pnl_unit=cfg.pnl_unit,
        )
        trade.update(
            original_exit_time=index[original_exit_index],
            original_exit=float(original_exit_price),
            exit_time=index[exit_index],
            exit=float(delayed_exit),
            gross_return=float(gross),
            net_return=float(gross - cfg.round_trip_cost),
            holding_bars=int(exit_index - entry_index),
            exit_delay_bars=int(exit_delay_bars),
            adverse_exit_bps=float(adverse_exit_bps),
        )
    return trade, entry_index, exit_index


def _simulate_with_effective_risk(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: np.ndarray,
    index: pd.Index,
    hypothesis: Hypothesis,
    eval_cfg: EvalConfig,
    backtest_cfg: BacktestConfig,
    risk: EffectiveRisk,
) -> list[dict]:
    """Apply paper/live-equivalent sizing and stateful entry risk gates."""

    trades: list[dict] = []
    next_allowed_entry = 0
    cooldown_until_entry = 0
    consecutive_losses = 0
    current_session: object | None = None
    daily_pnl = 0.0
    daily_trades = 0
    position_fraction = risk.position_fraction(backtest_cfg.stop_loss)
    if position_fraction <= 0:
        return trades

    def enter_session(session: object) -> None:
        nonlocal current_session, daily_pnl, daily_trades
        if session != current_session:
            current_session = session
            daily_pnl = 0.0
            daily_trades = 0

    for signal_index, raw_signal in enumerate(direction):
        signal = int(raw_signal)
        if signal == 0:
            continue
        entry_index = signal_index + 1 + eval_cfg.entry_delay_bars
        if entry_index >= len(close) or entry_index < next_allowed_entry:
            continue
        entry_session = _daily_session(index, entry_index, hypothesis.base_timeframe)
        enter_session(entry_session)
        if entry_index < cooldown_until_entry:
            continue
        if daily_pnl <= risk.daily_stop_loss or daily_trades >= risk.max_trades_per_day:
            continue
        simulated = _simulate_one_entry(
            open_,
            high,
            low,
            close,
            index,
            signal_index=signal_index,
            signal=signal,
            cfg=backtest_cfg,
            entry_delay_bars=eval_cfg.entry_delay_bars,
            adverse_fill_bps=eval_cfg.adverse_fill_bps,
            exit_delay_bars=eval_cfg.exit_delay_bars,
            adverse_exit_bps=eval_cfg.adverse_exit_bps,
        )
        if simulated is None:
            continue
        trade, entry_index, exit_index = simulated
        raw_net_return = float(trade["net_return"])
        funding_cost = 0.0
        if eval_cfg.market == "futures" and eval_cfg.funding_bps_per_8h:
            holding_hours = (
                float(trade["holding_bars"])
                * TIMEFRAME_SECONDS[hypothesis.base_timeframe]
                / 3_600.0
            )
            funding_cost = float(eval_cfg.funding_bps_per_8h) / 10_000.0 * holding_hours / 8.0
            raw_net_return -= funding_cost
        sized_return = raw_net_return * position_fraction
        trade["unsized_net_return"] = raw_net_return
        trade["position_size"] = position_fraction
        trade["sized_return"] = sized_return
        # All research metrics consume net_return, so make it the account-level
        # return that paper/live accounting actually applies.
        trade["net_return"] = sized_return
        trade["entry_delay_bars"] = eval_cfg.entry_delay_bars
        trade["adverse_fill_bps"] = float(eval_cfg.adverse_fill_bps)
        trade["exit_delay_bars"] = eval_cfg.exit_delay_bars
        trade["adverse_exit_bps"] = float(eval_cfg.adverse_exit_bps)
        trade["funding_bps_per_8h"] = float(eval_cfg.funding_bps_per_8h)
        trade["funding_cost_fraction"] = funding_cost
        trades.append(trade)
        daily_trades += 1
        next_allowed_entry = exit_index + 1

        exit_session = _daily_session(index, exit_index, hypothesis.base_timeframe)
        enter_session(exit_session)
        daily_pnl = _accumulate_daily_return(daily_pnl, sized_return)
        if sized_return < 0:
            consecutive_losses += 1
            if consecutive_losses >= risk.max_consecutive_losses:
                cooldown_until_entry = exit_index + risk.cooldown_bars
                consecutive_losses = 0
        else:
            consecutive_losses = 0
    return trades


def hypothesis_trades(
    frame: pd.DataFrame, hyp: Hypothesis, cfg: EvalConfig | None = None
) -> tuple[pd.DataFrame, int, BacktestConfig]:
    """Simulate one hypothesis with the canonical engine.

    Returns ``(trades, n_signals, resolved_backtest_config)`` so callers (the
    validation harness, regime breakdowns) can work with the raw trades without
    re-implementing the signal->simulate wiring."""
    trades, direction, bt = _hypothesis_trades_with_signals(frame, hyp, cfg)
    return trades, int((direction != 0).sum()), bt


def _hypothesis_trades_with_signals(
    frame: pd.DataFrame, hyp: Hypothesis, cfg: EvalConfig | None = None
) -> tuple[pd.DataFrame, np.ndarray, BacktestConfig]:
    cfg = cfg or EvalConfig()
    bt = _resolve_exit(frame, hyp, cfg)
    o, hi, lo, c = _ohlc(frame, hyp.base_timeframe)
    direction = signals_from_hypothesis(frame, hyp, cfg).to_numpy()
    risk = effective_risk(hyp, market=cfg.market, pnl_unit=cfg.pnl_unit)
    trades = pd.DataFrame(
        _simulate_with_effective_risk(
            o,
            hi,
            lo,
            c,
            direction,
            _trade_index(frame),
            hyp,
            cfg,
            bt,
            risk,
        )
    )
    return trades, direction, bt


def _signal_frequency(
    frame: pd.DataFrame,
    hyp: Hypothesis,
    direction: np.ndarray,
) -> dict[str, float | int | None]:
    indices = np.flatnonzero(direction != 0)
    timeframe_seconds = TIMEFRAME_SECONDS[hyp.base_timeframe]
    observed_days = max(len(frame) * timeframe_seconds / 86_400.0, timeframe_seconds / 86_400.0)
    gaps = np.diff(indices)
    regime_mask = pd.Series(True, index=frame.index)
    for predicate in hyp.regime:
        regime_mask &= predicate_mask(frame, predicate, base_tf=hyp.base_timeframe).fillna(False)
    regime_eligible_bars = int(regime_mask.sum())
    months_with_signal_fraction: float | None = None
    timestamps: pd.Series | None = None
    if "timestamp" in frame.columns:
        timestamps = pd.Series(pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"))
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.Series(pd.to_datetime(frame.index, utc=True, errors="coerce"))
    if timestamps is not None and timestamps.notna().any():
        all_months = timestamps.dt.tz_convert(None).dt.to_period("M")
        observed_months = int(all_months.nunique())
        signal_months = int(all_months.iloc[indices].nunique()) if indices.size else 0
        if observed_months:
            months_with_signal_fraction = signal_months / observed_months
    return {
        "signals": int(indices.size),
        "signals_per_day": round(float(indices.size / observed_days), 6),
        "signals_per_week": round(float(indices.size / observed_days * 7), 6),
        "median_signal_gap_bars": round(float(np.median(gaps)), 3) if gaps.size else None,
        "p95_signal_gap_bars": round(float(np.percentile(gaps, 95)), 3) if gaps.size else None,
        "median_signal_gap_seconds": (
            round(float(np.median(gaps) * timeframe_seconds), 3) if gaps.size else None
        ),
        "p95_signal_gap_seconds": (
            round(float(np.percentile(gaps, 95) * timeframe_seconds), 3) if gaps.size else None
        ),
        "months_with_signal_fraction": (
            round(float(months_with_signal_fraction), 6)
            if months_with_signal_fraction is not None
            else None
        ),
        "signal_bar_fraction": round(float(indices.size / max(1, len(frame))), 8),
        "regime_eligible_bars": regime_eligible_bars,
        "regime_coverage": round(float(regime_eligible_bars / max(1, len(frame))), 8),
    }


def evaluate_hypothesis(
    frame: pd.DataFrame, hyp: Hypothesis, cfg: EvalConfig | None = None
) -> dict:
    """Run one hypothesis over an aligned frame; return metrics + breakdowns."""
    cfg = cfg or EvalConfig()
    trades, direction, bt = _hypothesis_trades_with_signals(frame, hyp, cfg)
    n_signals = int((direction != 0).sum())
    signal_frequency = _signal_frequency(frame, hyp, direction)
    resolved_risk = effective_risk(hyp, market=cfg.market, pnl_unit=cfg.pnl_unit).to_dict()
    if trades.empty:
        return {
            "hypothesis_id": hyp.id,
            "family": hyp.family,
            "direction": hyp.direction,
            "n_signals": n_signals,
            "signal_frequency": signal_frequency,
            "trades": 0,
            "win_rate": float("nan"),
            "total_return": 0.0,
            "avg_net_return": float("nan"),
            "sharpe": 0.0,
            "psr": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": float("nan"),
            "effective_risk": resolved_risk,
            "by_year": [],
            "by_month": [],
            "exit_reasons": {},
        }

    r = trades["net_return"].to_numpy()
    n = int(r.size)
    equity = np.cumprod(1 + r)
    dd = equity / np.maximum.accumulate(equity) - 1
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    skew = float(pd.Series(r).skew()) if n > 2 else 0.0
    kurt = float(pd.Series(r).kurt() + 3.0) if n > 3 else 3.0
    sharpe = float(metrics.sharpe_ratio(r))
    psr = float(metrics.deflated_sharpe_ratio(sharpe, n_trials=1, skew=skew, kurt=kurt, n_obs=n))
    return {
        "hypothesis_id": hyp.id,
        "family": hyp.family,
        "direction": hyp.direction,
        "base_tf": hyp.base_timeframe,
        "n_signals": n_signals,
        "signal_frequency": signal_frequency,
        "trades": n,
        "win_rate": round(float((r > 0).mean()), 4),
        "total_return": round(float(equity[-1] - 1.0), 5),
        "avg_net_return": round(float(r.mean()), 6),
        "median_holding_bars": int(np.median(trades["holding_bars"])),
        "sharpe": round(sharpe, 4),
        "psr": round(psr, 4),
        "max_drawdown": round(float(dd.min()), 4),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else float("inf"),
        "tp_sl": (round(bt.take_profit, 5), round(bt.stop_loss, 5), bt.horizon_bars),
        "effective_risk": resolved_risk,
        "by_year": _breakdown(trades, "year"),
        "by_month": _breakdown(trades, "month"),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def verdict_from_metrics(m: dict, cfg: EvalConfig | None = None) -> str:
    cfg = cfg or EvalConfig()
    if m["trades"] < cfg.min_trades:
        return "inconclusive"
    if m["total_return"] > 0 and m["sharpe"] > 0.3 and m["psr"] > 0.6:
        return "keep"
    return "reject"


# --------------------------------------------------------------------------- #
# Frame builders
# --------------------------------------------------------------------------- #
def _needed_columns(hyps: Iterable[Hypothesis]) -> dict[str, set]:
    """Map timeframe -> set of *unprefixed* feature roots needed (+ OHLC/NATR)."""
    by_tf: dict[str, set] = {}
    base_tfs = set()
    for hyp in hyps:
        base_tfs.add(hyp.base_timeframe)
        for p in hyp.all_predicates():
            by_tf.setdefault(p.timeframe, set()).add(p.feature)
            if p.feature_b:
                by_tf[p.timeframe].add(p.feature_b)
    for tf in base_tfs | set(by_tf):
        by_tf.setdefault(tf, set()).update({"open", "high", "low", "close", "natr_14"})
    return by_tf


def build_aligned_frame(
    hyps: list[Hypothesis],
    base_tf: str,
    start: str | None = None,
    end: str | None = None,
    indicator_dir: Path = INDICATOR_DATA_DIR,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Load only the needed columns/window and align HTFs onto ``base_tf``.

    HEAVY I/O (reads real parquets). Only call with the user's approval and a
    bounded [start, end] window — never load full history of 1m/5m unbounded.
    """
    needed = _needed_columns(hyps)
    ts_filter = []
    if start:
        ts_filter.append(("timestamp", ">=", pd.Timestamp(start, tz="UTC")))
    if end:
        ts_filter.append(("timestamp", "<=", pd.Timestamp(end, tz="UTC")))

    def load_prefixed(tf: str) -> pd.DataFrame:
        cols = ["timestamp"] + sorted(needed.get(tf, set()))
        path = indicator_dir / f"{symbol}_{tf}_all_indicators.parquet"
        df = pd.read_parquet(path, columns=cols, filters=ts_filter or None)
        # These parquets store timestamp as a named index, not a data column.
        if "timestamp" not in df.columns:
            df = df.reset_index()
        if "timestamp" not in df.columns:
            raise KeyError(f"No 'timestamp' column/index in {path.name} (got {list(df.columns)})")
        # Pin to ns resolution so merge_asof keys match across timeframes
        # (adding a Timedelta can otherwise promote ms -> us and break the merge).
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df.rename(
            columns={c: f"{TIMEFRAME_PREFIXES[tf]}{c}" for c in df.columns if c != "timestamp"}
        )

    dataset = load_prefixed(base_tf)
    for tf in sorted(needed, key=lambda t: TIMEFRAME_SECONDS.get(t, 0)):
        if tf == base_tf:
            continue
        right = load_prefixed(tf)
        # Closed-candle visibility: a HTF candle is only usable after it closes.
        right["timestamp"] = (
            right["timestamp"] + pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf])
        ).astype("datetime64[ns, UTC]")
        dataset = pd.merge_asof(
            dataset.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
    return dataset.reset_index(drop=True)


def build_synthetic_aligned_frame(
    hyps: list[Hypothesis], n: int = 4000, seed: int = 7
) -> pd.DataFrame:
    """Build a small in-memory aligned frame with every column the hypotheses
    reference, derived from one synthetic price path. For smoke tests only — the
    numbers are meaningless, the wiring is what's exercised."""
    rng = np.random.default_rng(seed)
    timeframes = sorted(
        {p.timeframe for hyp in hyps for p in hyp.all_predicates()}
        | {hyp.base_timeframe for hyp in hyps}
    )
    ret = rng.normal(0, 0.002, n).cumsum()
    close = 30_000 * np.exp(ret)
    idx = pd.RangeIndex(n)
    frame = pd.DataFrame(index=idx)
    needed = _needed_columns(hyps)
    for tf in timeframes:
        pfx = TIMEFRAME_PREFIXES[tf]
        # smooth the base path a touch per-tf so HTF columns vary more slowly
        span = {"1m": 1, "5m": 2, "15m": 4, "30m": 6, "1h": 8, "4h": 16, "1d": 48, "1w": 96}.get(
            tf, 4
        )
        c = pd.Series(close).ewm(span=span).mean().to_numpy()
        o = np.concatenate([[c[0]], c[:-1]])
        hi = np.maximum(o, c) * (1 + rng.uniform(0, 0.003, n))
        lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.003, n))
        base_cols = {"open": o, "high": hi, "low": lo, "close": c, "volume": rng.uniform(1, 100, n)}
        for feat in needed.get(tf, set()):
            col = f"{pfx}{feat}"
            if feat in base_cols:
                frame[col] = base_cols[feat]
            elif feat.startswith(("max_", "min_")):
                w = int(feat.split("_")[1])
                s = pd.Series(c)
                frame[col] = (
                    s.rolling(w, min_periods=1).max()
                    if feat.startswith("max")
                    else s.rolling(w, min_periods=1).min()
                ).to_numpy()
            elif feat.startswith("rsi"):
                frame[col] = 50 + 30 * np.sin(np.linspace(0, 20, n)) + rng.normal(0, 5, n)
            elif feat.startswith(("ema", "sma", "bbands", "macd", "linearreg", "mom")):
                sp = 20 if "20" in feat else (50 if "50" in feat else 100)
                frame[col] = pd.Series(c).ewm(span=sp).mean().to_numpy()
            elif feat.startswith(
                (
                    "adx",
                    "natr",
                    "stddev",
                    "atr",
                    "volume_z",
                    "trades_z",
                    "cvd",
                    "taker",
                    "avg_trade",
                    "willr",
                    "cci",
                    "aroon",
                    "stoch",
                )
            ):
                frame[col] = np.abs(rng.normal(20, 8, n))
            else:
                frame[col] = rng.normal(0, 1, n)
    # ensure base OHLC present
    return frame


# --------------------------------------------------------------------------- #
# CLI (safe synthetic smoke by default)
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hypotheses (synthetic smoke unless --real)."
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run the synthetic smoke (default)."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real data (HEAVY; needs --base-tf, --start, --end).",
    )
    parser.add_argument("--base-tf", default="5m")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    from research_exploration.hypothesis_generator import first_smoke_set

    cfg = EvalConfig()

    if args.real:
        hyps = [h for h in first_smoke_set() if h.base_timeframe == args.base_tf]
        frame = build_aligned_frame(hyps, base_tf=args.base_tf, start=args.start, end=args.end)
        print(f"Aligned real frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
    else:
        hyps = first_smoke_set()
        frame = build_synthetic_aligned_frame(hyps)
        print(f"Synthetic frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols (smoke only)")

    for hyp in hyps[:8]:
        try:
            m = evaluate_hypothesis(frame, hyp, cfg)
            print(
                f"  {hyp.id:42} signals={m['n_signals']:>5} trades={m['trades']:>4} "
                f"wr={m['win_rate']} ret={m['total_return']} sharpe={m['sharpe']} -> {verdict_from_metrics(m, cfg)}"
            )
        except Exception as e:  # smoke wiring check
            print(f"  {hyp.id:42} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
