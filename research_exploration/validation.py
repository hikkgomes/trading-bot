"""Controlled validation harness: from "looks positive once" to a defensible verdict.

This is the missing discipline layer on top of ``evaluate.py``. One evaluation
over one window says almost nothing; the old pipeline's documented flaw was that
its holdout was *report-only* — strategies that lost on their own holdout still
shipped. Here the holdout **gates**: a hypothesis can only be kept if it survives
every stage, in order:

    1. TRAIN       (first ``train_frac``)  — must show an edge at all
    2. VALIDATION  (next ``val_frac``)     — the edge must repeat out-of-sample
    3. OOS WINDOWS (train+val chunked)     — the edge must be spread across time,
                                             not one lucky cluster
    4. SENSITIVITY (perturbed variants)    — the edge must survive ±25% exits and
                                             stricter/looser entry thresholds
    5. STRESS      (train+validation only) — higher costs, delayed/adverse entry
                                             and exit, funding, and deterministic
                                             feed gaps must hold
    6. HOLDOUT     (final ``1-train-val``) — untouched until stages 1–5 pass;
                                             a negative holdout REJECTS

Because every predicate is causal (rolling windows, current/past bars only)
there are no fitted parameters to refit per window — the "walk-forward" here is
honest out-of-sample chunking of a *fixed* rule. Splits are chronological only.

Extra breakdowns per hypothesis: per-regime (bull/bear/range off a trailing
causal return), yearly/monthly (from ``evaluate.py``), long/short is inherent
(every hypothesis is directional and the batch carries both sides).

Batch-level honesty: DSR is deflated by ``n_trials`` = number of hypotheses
validated together, so testing 40 ideas costs each of them evidence.

Run (safe, synthetic):  python -m research_exploration.validation --synthetic
Real data (HEAVY, bounded window, needs approval):
    python -m research_exploration.validation --real --base-tf 5m \
        --start 2024-01-01 --end 2024-07-01
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research_exploration.dsr import DSR_METHOD, MIN_TRIAL_SHARPE_STD
from research_exploration.evaluate import (
    EvalConfig,
    _trade_index,
    evaluate_hypothesis,
    hypothesis_trades,
)
from research_exploration.experiment_log import (
    DEFAULT_LOG,
    ExperimentRecord,
    fingerprint,
    log_result,
)
from research_exploration.hypothesis_schema import Hypothesis, Predicate
from src import metrics
from src.build_dataset import TIMEFRAME_SECONDS

REJECTION_REASONS = (
    "insufficient_train_trades",
    "no_train_edge",
    "insufficient_validation_trades",
    "failed_validation",
    "unstable_across_windows",
    "parameter_fragile",
    "failed_execution_stress",
    "holdout_already_consumed",
    "insufficient_holdout_trades",
    "failed_holdout",
)


@dataclass
class ValidationConfig:
    """Split sizes, stage gates and perturbation sizes."""

    train_frac: float = 0.6
    val_frac: float = 0.2  # holdout = 1 - train_frac - val_frac
    min_trades_train: int = 30
    min_trades_val: int = 10
    min_trades_holdout: int = 5
    min_train_sharpe: float = 0.0  # train must at least be positive
    oos_windows: int = 6  # train+val chunked into this many OOS windows
    min_window_pass_rate: float = 0.5
    exit_delta: float = 0.25  # ±25% on TP / SL / horizon
    reference_jitter: float = 0.10  # ±10% on numeric entry thresholds
    quantile_jitter: float = 0.05  # ±0.05 on rolling-quantile predicates
    min_sensitivity_pass: float = 0.6
    regime_lookback_bars: int | None = None  # None -> ~30 days of base-tf bars
    regime_band: float = 0.10  # trailing return beyond ±band => bull/bear
    n_trials: int = 1  # DSR deflation; validate_batch enforces at least len(batch)
    sr_std_trials: float = 0.0  # effective cross-trial Sharpe dispersion used by DSR
    trial_sharpe_count: int = 0
    trial_sharpe_observed_std: float = 0.0
    trial_sharpe_conservative_floor: float = 0.0
    stress_cost_multiplier: float = 2.0
    stress_min_extra_cost_bps: float = 1.0
    stress_entry_delay_bars: int = 1
    stress_adverse_fill_bps: float = 5.0
    stress_exit_delay_bars: int = 1
    stress_adverse_exit_bps: float = 5.0
    stress_funding_bps_per_8h: float = 3.0
    stress_missing_bar_stride: int = 97
    stress_gap_bars: int = 2
    min_stress_pass_rate: float = 2.0 / 3.0
    min_stress_trade_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.train_frac < 1 or not 0 < self.val_frac < 1:
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave room for a holdout")
        if not math.isfinite(self.stress_cost_multiplier) or self.stress_cost_multiplier <= 1.0:
            raise ValueError("stress_cost_multiplier must be finite and greater than 1")
        if not math.isfinite(self.stress_min_extra_cost_bps) or self.stress_min_extra_cost_bps <= 0:
            raise ValueError("stress_min_extra_cost_bps must be finite and positive")
        if (
            not isinstance(self.stress_entry_delay_bars, int)
            or isinstance(self.stress_entry_delay_bars, bool)
            or self.stress_entry_delay_bars < 1
        ):
            raise ValueError("stress_entry_delay_bars must be a positive integer")
        if (
            not math.isfinite(self.stress_adverse_fill_bps)
            or not 0 < self.stress_adverse_fill_bps < 10_000
        ):
            raise ValueError("stress_adverse_fill_bps must be finite and in (0, 10000)")
        if (
            not isinstance(self.stress_exit_delay_bars, int)
            or isinstance(self.stress_exit_delay_bars, bool)
            or self.stress_exit_delay_bars < 1
        ):
            raise ValueError("stress_exit_delay_bars must be a positive integer")
        for label, value in (
            ("stress_adverse_exit_bps", self.stress_adverse_exit_bps),
            ("stress_funding_bps_per_8h", self.stress_funding_bps_per_8h),
        ):
            if not math.isfinite(value) or not 0 < value < 10_000:
                raise ValueError(f"{label} must be finite and in (0, 10000)")
        if (
            not isinstance(self.stress_missing_bar_stride, int)
            or isinstance(self.stress_missing_bar_stride, bool)
            or self.stress_missing_bar_stride < 3
        ):
            raise ValueError("stress_missing_bar_stride must be an integer of at least 3")
        if (
            not isinstance(self.stress_gap_bars, int)
            or isinstance(self.stress_gap_bars, bool)
            or not 1 <= self.stress_gap_bars < self.stress_missing_bar_stride
        ):
            raise ValueError("stress_gap_bars must be positive and below stress_missing_bar_stride")
        for label, value in (
            ("min_stress_pass_rate", self.min_stress_pass_rate),
            ("min_stress_trade_fraction", self.min_stress_trade_fraction),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{label} must be finite and in (0, 1]")
        if (
            not isinstance(self.n_trials, int)
            or isinstance(self.n_trials, bool)
            or self.n_trials < 1
        ):
            raise ValueError("n_trials must be a positive integer")
        if (
            not isinstance(self.trial_sharpe_count, int)
            or isinstance(self.trial_sharpe_count, bool)
            or self.trial_sharpe_count < 0
        ):
            raise ValueError("trial_sharpe_count must be a non-negative integer")
        for label, value in (
            ("sr_std_trials", self.sr_std_trials),
            ("trial_sharpe_observed_std", self.trial_sharpe_observed_std),
            ("trial_sharpe_conservative_floor", self.trial_sharpe_conservative_floor),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")


# --------------------------------------------------------------------------- #
# Chronological splits
# --------------------------------------------------------------------------- #
def split_frame(frame: pd.DataFrame, cfg: ValidationConfig) -> dict[str, pd.DataFrame]:
    """Contiguous, disjoint, chronological train / validation / holdout slices."""
    n = len(frame)
    t_end = int(n * cfg.train_frac)
    v_end = int(n * (cfg.train_frac + cfg.val_frac))
    return {
        "train": frame.iloc[:t_end],
        "validation": frame.iloc[t_end:v_end],
        "holdout": frame.iloc[v_end:],
    }


def _segment_bounds(seg: pd.DataFrame) -> dict:
    out: dict = {"rows": int(len(seg))}
    if len(seg) and "timestamp" in seg.columns:
        ts = pd.to_datetime(seg["timestamp"], utc=True)
        out["start"], out["end"] = str(ts.iloc[0]), str(ts.iloc[-1])
    return out


# --------------------------------------------------------------------------- #
# Regime tagging + per-regime breakdown (causal: trailing return only)
# --------------------------------------------------------------------------- #
def tag_regimes(
    frame: pd.DataFrame,
    base_tf: str,
    lookback_bars: int | None = None,
    lookback_days: int = 30,
    band: float = 0.10,
) -> pd.Series:
    """Label each bar bull/bear/range from the *trailing* ``lookback`` return of
    the base close. Uses only past bars — safe to group trades by. Bars inside
    the warmup are labelled 'unknown'."""
    close = frame[f"tf_{base_tf}_close"].astype(float)
    if lookback_bars is None:
        bars_per_day = max(1, 86_400 // TIMEFRAME_SECONDS[base_tf])
        lookback_bars = lookback_days * bars_per_day
    lookback_bars = max(1, min(int(lookback_bars), max(1, len(frame) - 1)))
    trailing = close / close.shift(lookback_bars) - 1.0
    labels = np.where(trailing > band, "bull", np.where(trailing < -band, "bear", "range"))
    labels = np.where(trailing.isna().to_numpy(), "unknown", labels)
    return pd.Series(labels, index=frame.index, name="regime")


def regime_breakdown(
    frame: pd.DataFrame, hyp: Hypothesis, eval_cfg: EvalConfig, cfg: ValidationConfig
) -> list[dict]:
    """Split the hypothesis' trades by the regime active at entry."""
    trades, _, _ = hypothesis_trades(frame, hyp, eval_cfg)
    if trades.empty:
        return []
    regimes = tag_regimes(
        frame, hyp.base_timeframe, lookback_bars=cfg.regime_lookback_bars, band=cfg.regime_band
    )
    lookup = pd.Series(regimes.to_numpy(), index=_trade_index(frame))
    trade_regime = lookup.reindex(pd.Index(trades["entry_time"])).fillna("unknown").to_numpy()
    rows = []
    for regime in ("bull", "bear", "range", "unknown"):
        r = trades.loc[trade_regime == regime, "net_return"].to_numpy()
        if r.size == 0:
            continue
        rows.append(
            {
                "regime": regime,
                "trades": int(r.size),
                "win_rate": round(float((r > 0).mean()), 4),
                "total_return": round(float(np.prod(1 + r) - 1), 5),
                "avg_net": round(float(r.mean()), 6),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# OOS window stability (fixed rule -> honest chronological chunking)
# --------------------------------------------------------------------------- #
def oos_window_stats(
    frame: pd.DataFrame, hyp: Hypothesis, eval_cfg: EvalConfig, n_windows: int
) -> dict:
    """Chunk ``frame`` into ``n_windows`` equal chronological windows and check
    the edge appears in most of them (a zero-trade window counts as a fail)."""
    trades, _, _ = hypothesis_trades(frame, hyp, eval_cfg)
    idx = _trade_index(frame)
    edges = np.linspace(0, len(frame), n_windows + 1).astype(int)
    entry = pd.Series(trades["entry_time"]) if not trades.empty else pd.Series(dtype=object)
    rows = []
    for i in range(n_windows):
        lo_pos, hi_pos = edges[i], edges[i + 1]
        if hi_pos <= lo_pos:
            continue
        if trades.empty:
            in_win = pd.Series(False, index=entry.index)
        else:
            in_win = (entry >= idx[lo_pos]) & (entry <= idx[hi_pos - 1])
        r = (
            trades.loc[in_win.to_numpy(), "net_return"].to_numpy()
            if not trades.empty
            else np.array([])
        )
        rows.append(
            {
                "window": i + 1,
                "trades": int(r.size),
                "total_return": round(float(np.prod(1 + r) - 1), 5) if r.size else 0.0,
                "win_rate": round(float((r > 0).mean()), 4) if r.size else float("nan"),
            }
        )
    profitable = sum(1 for w in rows if w["trades"] > 0 and w["total_return"] > 0)
    return {
        "windows": len(rows),
        "profitable_windows": int(profitable),
        "pass_rate": round(profitable / len(rows), 4) if rows else 0.0,
        "window_stats": rows,
    }


# --------------------------------------------------------------------------- #
# Parameter sensitivity (a real edge should survive small perturbations)
# --------------------------------------------------------------------------- #
_STRICTER_UP = {"gt", "ge", "slope_up", "pct_above"}
_STRICTER_DOWN = {"lt", "le", "slope_down", "pct_below"}


def _jitter_predicate(p: Predicate, cfg: ValidationConfig, stricter: bool) -> Predicate:
    """Nudge one predicate's numeric threshold so the entry gets stricter (fires
    less) or looser (fires more). Ops without a tunable scalar pass through."""
    if p.op in _STRICTER_UP or p.op in _STRICTER_DOWN:
        ref = p.reference
        if ref is None or ref == 0:
            return p
        up = (p.op in _STRICTER_UP) == stricter
        delta = cfg.reference_jitter * abs(float(ref))
        return dataclasses.replace(p, reference=float(ref) + (delta if up else -delta))
    if p.op in ("q_ge", "q_le") and p.quantile is not None:
        up = (p.op == "q_ge") == stricter
        q = float(p.quantile) + (cfg.quantile_jitter if up else -cfg.quantile_jitter)
        return dataclasses.replace(p, quantile=float(np.clip(q, 0.01, 0.99)))
    return p


def perturbed_variants(hyp: Hypothesis, cfg: ValidationConfig) -> list[tuple[str, Hypothesis]]:
    """Small, named perturbations of one hypothesis: ±exit_delta on TP / SL /
    horizon (one axis at a time) and entry thresholds jittered stricter/looser."""
    out: list[tuple[str, Hypothesis]] = []

    def with_exit(label: str, **changes) -> None:
        new_exit = dataclasses.replace(hyp.exit, **changes)
        out.append((label, dataclasses.replace(hyp, id=f"{hyp.id}~{label}", exit=new_exit)))

    d = cfg.exit_delta
    with_exit("tp_down", take_profit=hyp.exit.take_profit * (1 - d))
    with_exit("tp_up", take_profit=hyp.exit.take_profit * (1 + d))
    with_exit("sl_down", stop_loss=hyp.exit.stop_loss * (1 - d))
    with_exit("sl_up", stop_loss=hyp.exit.stop_loss * (1 + d))
    with_exit("horizon_down", horizon_bars=max(1, round(hyp.exit.horizon_bars * (1 - d))))
    with_exit("horizon_up", horizon_bars=max(1, round(hyp.exit.horizon_bars * (1 + d))))

    for label, stricter in (("entry_stricter", True), ("entry_looser", False)):
        stages = {
            stage: [_jitter_predicate(p, cfg, stricter) for p in getattr(hyp, stage)]
            for stage in ("regime", "setup", "trigger")
        }
        if any(stages[s][i] != getattr(hyp, s)[i] for s in stages for i in range(len(stages[s]))):
            out.append((label, dataclasses.replace(hyp, id=f"{hyp.id}~{label}", **stages)))
    return out


def sensitivity_check(
    frame: pd.DataFrame, hyp: Hypothesis, eval_cfg: EvalConfig, cfg: ValidationConfig
) -> dict:
    """Re-evaluate every perturbed variant on pre-holdout data. A robust edge
    stays profitable for most neighbours of the chosen parameters."""
    variants = perturbed_variants(hyp, cfg)
    rows, profitable = [], 0
    for label, variant in variants:
        m = evaluate_hypothesis(frame, variant, eval_cfg)
        ok = bool(m["trades"] > 0 and m["total_return"] > 0)
        profitable += ok
        rows.append(
            {
                "variant": label,
                "trades": m["trades"],
                "total_return": m["total_return"],
                "sharpe": m["sharpe"],
                "profitable": ok,
            }
        )
    frac = round(profitable / len(rows), 4) if rows else 0.0
    return {
        "variants": rows,
        "pass_fraction": frac,
        "passed": bool(rows) and frac >= cfg.min_sensitivity_pass,
    }


# --------------------------------------------------------------------------- #
# Pre-holdout execution/data stress (all variants remain adaptive-safe)
# --------------------------------------------------------------------------- #
def _frame_with_missing_bars(
    frame: pd.DataFrame,
    stride: int,
    gap_bars: int,
) -> pd.DataFrame:
    """Remove deterministic interior bars to emulate feed gaps without randomness."""

    if len(frame) < stride * 2:
        return frame.copy()
    keep = np.ones(len(frame), dtype=bool)
    for start in range(stride - 1, len(frame) - 1, stride):
        keep[start : min(start + gap_bars, len(frame) - 1)] = False
    return frame.iloc[keep].copy()


def execution_data_stress_check(
    frame: pd.DataFrame,
    hyp: Hypothesis,
    eval_cfg: EvalConfig,
    cfg: ValidationConfig,
) -> dict:
    """Gate candidates on adverse costs/fills and deterministic missing bars."""

    baseline = evaluate_hypothesis(frame, hyp, eval_cfg)
    baseline_trades = int(baseline.get("trades") or 0)
    required_trades = max(1, math.ceil(baseline_trades * cfg.min_stress_trade_fraction))
    variants: list[tuple[str, pd.DataFrame, EvalConfig]] = [
        (
            "higher_costs",
            frame,
            dataclasses.replace(
                eval_cfg,
                fee_bps=max(
                    float(eval_cfg.fee_bps) * cfg.stress_cost_multiplier,
                    float(eval_cfg.fee_bps) + cfg.stress_min_extra_cost_bps,
                ),
                slippage_bps=max(
                    float(eval_cfg.slippage_bps) * cfg.stress_cost_multiplier,
                    float(eval_cfg.slippage_bps) + cfg.stress_min_extra_cost_bps,
                ),
            ),
        ),
        (
            "delayed_adverse_entry",
            frame,
            dataclasses.replace(
                eval_cfg,
                entry_delay_bars=eval_cfg.entry_delay_bars + cfg.stress_entry_delay_bars,
                adverse_fill_bps=eval_cfg.adverse_fill_bps + cfg.stress_adverse_fill_bps,
            ),
        ),
        (
            "delayed_adverse_exit_funding",
            frame,
            dataclasses.replace(
                eval_cfg,
                exit_delay_bars=eval_cfg.exit_delay_bars + cfg.stress_exit_delay_bars,
                adverse_exit_bps=(eval_cfg.adverse_exit_bps + cfg.stress_adverse_exit_bps),
                funding_bps_per_8h=(
                    eval_cfg.funding_bps_per_8h + cfg.stress_funding_bps_per_8h
                    if eval_cfg.market == "futures"
                    else 0.0
                ),
            ),
        ),
        (
            "missing_bars",
            _frame_with_missing_bars(
                frame,
                cfg.stress_missing_bar_stride,
                cfg.stress_gap_bars,
            ),
            eval_cfg,
        ),
    ]
    rows: list[dict] = []
    passes = 0
    for label, variant_frame, variant_cfg in variants:
        metrics = evaluate_hypothesis(variant_frame, hyp, variant_cfg)
        trades = int(metrics.get("trades") or 0)
        profitable = bool(trades >= required_trades and metrics.get("total_return", 0.0) > 0)
        passes += int(profitable)
        rows.append(
            {
                "variant": label,
                "rows": int(len(variant_frame)),
                "trades": trades,
                "required_trades": required_trades,
                "total_return": metrics.get("total_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe": metrics.get("sharpe"),
                "passed": profitable,
                "eval_overrides": {
                    "fee_bps": variant_cfg.fee_bps,
                    "slippage_bps": variant_cfg.slippage_bps,
                    "entry_delay_bars": variant_cfg.entry_delay_bars,
                    "adverse_fill_bps": variant_cfg.adverse_fill_bps,
                    "exit_delay_bars": variant_cfg.exit_delay_bars,
                    "adverse_exit_bps": variant_cfg.adverse_exit_bps,
                    "funding_bps_per_8h": variant_cfg.funding_bps_per_8h,
                },
            }
        )
    pass_fraction = passes / len(rows) if rows else 0.0
    return {
        "baseline_trades": baseline_trades,
        "required_trades": required_trades,
        "variants": rows,
        "pass_fraction": round(pass_fraction, 4),
        "minimum_pass_fraction": cfg.min_stress_pass_rate,
        "passed": bool(rows) and pass_fraction + 1e-12 >= cfg.min_stress_pass_rate,
    }


def with_trial_sharpe_dispersion(
    frame: pd.DataFrame,
    hypotheses: list[Hypothesis],
    cfg: ValidationConfig,
    eval_cfg: EvalConfig,
) -> ValidationConfig:
    """Estimate the searched trial distribution without touching holdout rows.

    DSR needs the cross-trial Sharpe dispersion to estimate the best Sharpe a
    multiple-testing search would produce by chance.  A single/homogeneous
    current batch cannot honestly estimate zero dispersion, especially when
    ``n_trials`` includes earlier cycles, so the observed value is floored by
    both 0.10 and the largest finite-sample null standard error among the
    current pre-holdout trials.
    """

    # This makes the helper idempotent. Autonomous orchestration computes the
    # evidence before sealing its durable protocol; validate_batch may then
    # receive that exact config without silently changing its identity.
    if (
        cfg.trial_sharpe_count > 0
        or cfg.trial_sharpe_observed_std > 0
        or cfg.trial_sharpe_conservative_floor > 0
    ):
        return cfg

    pre_holdout_rows = int(len(frame) * (cfg.train_frac + cfg.val_frac))
    pre_holdout = frame.iloc[:pre_holdout_rows]
    sharpes: list[float] = []
    observation_counts: list[int] = []
    for hypothesis in hypotheses:
        trades, _, _ = hypothesis_trades(pre_holdout, hypothesis, eval_cfg)
        if "net_return" not in trades:
            continue
        returns = pd.to_numeric(trades["net_return"], errors="coerce").to_numpy(float)
        returns = returns[np.isfinite(returns)]
        if returns.size <= 3:
            continue
        sharpe = float(metrics.sharpe_ratio(returns))
        if math.isfinite(sharpe):
            sharpes.append(sharpe)
            observation_counts.append(int(returns.size))

    observed_std = (
        float(np.std(np.asarray(sharpes, dtype=float), ddof=1)) if len(sharpes) >= 2 else 0.0
    )
    conservative_floor = 0.0
    if cfg.n_trials > 1:
        sampling_floor = max(
            (1.0 / math.sqrt(max(count - 1, 1)) for count in observation_counts),
            default=0.0,
        )
        conservative_floor = max(MIN_TRIAL_SHARPE_STD, sampling_floor)
    effective = max(float(cfg.sr_std_trials), observed_std, conservative_floor)
    return dataclasses.replace(
        cfg,
        sr_std_trials=effective,
        trial_sharpe_count=len(sharpes),
        trial_sharpe_observed_std=observed_std,
        trial_sharpe_conservative_floor=conservative_floor,
    )


def _deflated_sharpe_evidence(
    returns: np.ndarray,
    cfg: ValidationConfig,
) -> dict[str, float | int | str | None]:
    """Calculate versioned DSR evidence from one trial's pre-holdout returns."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    effective_dispersion = float(cfg.sr_std_trials)
    conservative_floor = float(cfg.trial_sharpe_conservative_floor)
    if cfg.n_trials > 1:
        per_trial_floor = max(
            MIN_TRIAL_SHARPE_STD,
            1.0 / math.sqrt(max(int(values.size) - 1, 1)),
        )
        conservative_floor = max(conservative_floor, per_trial_floor)
        effective_dispersion = max(effective_dispersion, conservative_floor)
    evidence: dict[str, float | int | str | None] = {
        "dsr_method": DSR_METHOD,
        "n_trials": int(cfg.n_trials),
        "sr_std_trials": effective_dispersion,
        "trial_sharpe_count": int(cfg.trial_sharpe_count),
        "trial_sharpe_observed_std": float(cfg.trial_sharpe_observed_std),
        "trial_sharpe_conservative_floor": conservative_floor,
        "dsr_deflated": None,
    }
    if values.size <= 3:
        return evidence
    sharpe = float(metrics.sharpe_ratio(values))
    series = pd.Series(values)
    skew = float(series.skew())
    kurt = float(series.kurt() + 3.0)
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    evidence["dsr_deflated"] = round(
        float(
            metrics.deflated_sharpe_ratio(
                sharpe,
                n_trials=max(1, cfg.n_trials),
                skew=skew,
                kurt=kurt,
                n_obs=int(values.size),
                sr_std_trials=effective_dispersion,
            )
        ),
        4,
    )
    return evidence


# --------------------------------------------------------------------------- #
# Staged validation of one hypothesis
# --------------------------------------------------------------------------- #
def validate_hypothesis(
    frame: pd.DataFrame,
    hyp: Hypothesis,
    cfg: ValidationConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    before_holdout: Callable[[Hypothesis, dict], bool | None] | None = None,
) -> dict:
    """Run the full staged pipeline. The holdout is only ever *touched* after
    every earlier stage passes, and a negative holdout rejects — it gates."""
    cfg = cfg or ValidationConfig()
    eval_cfg = eval_cfg or EvalConfig()
    segs = split_frame(frame, cfg)
    result: dict = {
        "hypothesis_id": hyp.id,
        "family": hyp.family,
        "direction": hyp.direction,
        "splits": {name: _segment_bounds(seg) for name, seg in segs.items()},
        "train": None,
        "validation": None,
        "holdout": None,
        "oos": None,
        "sensitivity": None,
        "stress": None,
        "regimes": None,
        "dsr_deflated": None,
        "dsr_method": DSR_METHOD,
        "n_trials": cfg.n_trials,
        "sr_std_trials": cfg.sr_std_trials,
        "trial_sharpe_count": cfg.trial_sharpe_count,
        "trial_sharpe_observed_std": cfg.trial_sharpe_observed_std,
        "trial_sharpe_conservative_floor": cfg.trial_sharpe_conservative_floor,
        "verdict": None,
        "reasons": [],
    }

    def finish(verdict: str, reason: str | None = None) -> dict:
        result["verdict"] = verdict
        if reason:
            result["reasons"].append(reason)
        return result

    # 1. TRAIN — is there an edge at all?
    train = evaluate_hypothesis(segs["train"], hyp, eval_cfg)
    result["train"] = train
    if train["trades"] < cfg.min_trades_train:
        return finish("inconclusive", "insufficient_train_trades")
    if train["total_return"] <= 0 or train["sharpe"] <= cfg.min_train_sharpe:
        return finish("reject", "no_train_edge")

    # 2. VALIDATION — does it repeat out-of-sample?
    val = evaluate_hypothesis(segs["validation"], hyp, eval_cfg)
    result["validation"] = val
    if val["trades"] < cfg.min_trades_val:
        return finish("inconclusive", "insufficient_validation_trades")
    if val["total_return"] <= 0:
        return finish("reject", "failed_validation")

    # Pre-holdout region (train+val) for stability / sensitivity / regimes / DSR.
    pre_holdout = frame.iloc[: len(segs["train"]) + len(segs["validation"])]
    result["regimes"] = regime_breakdown(pre_holdout, hyp, eval_cfg, cfg)

    trades, _, _ = hypothesis_trades(pre_holdout, hyp, eval_cfg)
    r = trades["net_return"].to_numpy()
    result.update(_deflated_sharpe_evidence(r, cfg))

    # 3. OOS WINDOWS — is the edge spread across time?
    oos = oos_window_stats(pre_holdout, hyp, eval_cfg, cfg.oos_windows)
    result["oos"] = oos
    if oos["pass_rate"] < cfg.min_window_pass_rate:
        return finish("reject", "unstable_across_windows")

    # 4. SENSITIVITY — does it survive neighbouring parameters?
    sens = sensitivity_check(pre_holdout, hyp, eval_cfg, cfg)
    result["sensitivity"] = sens
    if not sens["passed"]:
        return finish("reject", "parameter_fragile")

    # 5. EXECUTION/DATA STRESS — still pre-holdout, so its full evidence may
    # safely guide later generations without exposing final evaluation data.
    stress = execution_data_stress_check(pre_holdout, hyp, eval_cfg, cfg)
    result["stress"] = stress
    if not stress["passed"]:
        return finish("reject", "failed_execution_stress")

    # 6. HOLDOUT — untouched until now, and it GATES.  Autonomous callers use
    # this hook to durably claim the canonical behavior+snapshot before a read.
    # Returning False means an earlier attempt (including a killed process)
    # already consumed this holdout; fail closed rather than looking again.
    if before_holdout is not None and before_holdout(hyp, result) is False:
        return finish("inconclusive", "holdout_already_consumed")
    holdout = evaluate_hypothesis(segs["holdout"], hyp, eval_cfg)
    result["holdout"] = holdout
    if holdout["trades"] < cfg.min_trades_holdout:
        return finish("inconclusive", "insufficient_holdout_trades")
    if holdout["total_return"] <= 0:
        return finish("reject", "failed_holdout")
    return finish("keep")


# --------------------------------------------------------------------------- #
# Batch runner (+ experiment log integration)
# --------------------------------------------------------------------------- #
def validate_batch(
    frame: pd.DataFrame,
    hyps: list[Hypothesis],
    cfg: ValidationConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    log_path: Path | None = None,
    before_holdout: Callable[[Hypothesis, dict], bool | None] | None = None,
    after_candidate: Callable[[Hypothesis, dict], None] | None = None,
) -> list[dict]:
    """Validate a batch with DSR deflated by at least the batch size.

    Callers that rotate a slice of a larger candidate universe can set
    ``cfg.n_trials`` higher than ``len(hyps)`` so DSR still pays for the
    hypotheses being searched over time. Optionally append one experiment-log
    record per hypothesis; skips nothing silently.
    """
    base_cfg = cfg or ValidationConfig()
    cfg = dataclasses.replace(base_cfg, n_trials=max(1, int(base_cfg.n_trials), len(hyps)))
    eval_cfg = eval_cfg or EvalConfig()
    cfg = with_trial_sharpe_dispersion(frame, hyps, cfg, eval_cfg)
    results = []
    for hyp in hyps:
        res = validate_hypothesis(
            frame,
            hyp,
            cfg,
            eval_cfg,
            before_holdout=before_holdout,
        )
        results.append(res)
        # Autonomous callers use this as the per-candidate durable checkpoint.
        # Invoke it before the advisory JSONL log so a timeout cannot erase the
        # completed work of earlier candidates in a long sequential batch.
        if after_candidate is not None:
            after_candidate(hyp, res)
        if log_path is not None:
            headline = {
                key: res[key]
                for key in (
                    "dsr_deflated",
                    "dsr_method",
                    "n_trials",
                    "sr_std_trials",
                    "trial_sharpe_count",
                    "trial_sharpe_observed_std",
                    "trial_sharpe_conservative_floor",
                )
            }
            headline["reasons"] = res["reasons"]
            for seg in ("train", "validation", "holdout"):
                m = res.get(seg)
                if m:
                    headline[seg] = {
                        k: m[k] for k in ("trades", "total_return", "win_rate", "sharpe")
                    }
            if res.get("oos"):
                headline["oos_pass_rate"] = res["oos"]["pass_rate"]
            if res.get("sensitivity"):
                headline["sensitivity_pass_fraction"] = res["sensitivity"]["pass_fraction"]
            if res.get("stress"):
                headline["stress"] = res["stress"]
            config = {"validation": dataclasses.asdict(cfg), "eval": dataclasses.asdict(eval_cfg)}
            log_result(
                ExperimentRecord(
                    hypothesis_id=hyp.id,
                    family=hyp.family,
                    direction=hyp.direction,
                    fingerprint=fingerprint(hyp.to_dict(), config),
                    verdict=res["verdict"],
                    metrics=headline,
                    config=config,
                    data_window=res["splits"],
                    notes="validation.py staged pipeline",
                    hypothesis=hyp.to_dict(),
                ),
                log_path,
            )
    return results


def summarize_results(results: list[dict]) -> str:
    lines = [
        f"{'hypothesis':44} {'verdict':13} {'train':>8} {'val':>8} "
        f"{'holdout':>8} {'oos':>5} {'sens':>5} {'stress':>6}  reasons"
    ]

    def seg_ret(r, seg):
        m = r.get(seg)
        return f"{m['total_return']:+.3f}" if m else "-"

    for r in sorted(results, key=lambda x: (x["verdict"] != "keep", x["hypothesis_id"])):
        oos = f"{r['oos']['pass_rate']:.2f}" if r.get("oos") else "-"
        sens = f"{r['sensitivity']['pass_fraction']:.2f}" if r.get("sensitivity") else "-"
        stress = f"{r['stress']['pass_fraction']:.2f}" if r.get("stress") else "-"
        lines.append(
            f"{r['hypothesis_id']:44} {r['verdict']:13} {seg_ret(r, 'train'):>8} "
            f"{seg_ret(r, 'validation'):>8} {seg_ret(r, 'holdout'):>8} {oos:>5} "
            f"{sens:>5} {stress:>6}  "
            f"{','.join(r['reasons']) or '-'}"
        )
    from collections import Counter

    counts = Counter(r["verdict"] for r in results)
    lines.append("")
    lines.append("Verdicts: " + "  ".join(f"{v}={n}" for v, n in counts.most_common()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI (safe synthetic smoke by default; --real is heavy and gated)
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged validation (train/val/OOS/sensitivity/GATING holdout)."
    )
    parser.add_argument("--synthetic", action="store_true", help="Synthetic smoke (default).")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Real data (HEAVY; needs --base-tf, --start, --end; results are logged).",
    )
    parser.add_argument("--base-tf", default="5m")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Hypotheses JSON (default: the generator's smoke set).",
    )
    parser.add_argument(
        "--position",
        action="store_true",
        help="Use the scenario-1 position/BTC-accumulation batch (pair with --pnl-unit btc).",
    )
    parser.add_argument(
        "--with-guards",
        action="store_true",
        help="Use the Family-F guarded variant of the smoke set.",
    )
    parser.add_argument(
        "--pnl-unit",
        choices=("usdt", "btc"),
        default="usdt",
        help="Scoring unit: usdt = day-trade/flow bot, btc = position/"
        "accumulation bot (shorts only realise returns).",
    )
    parser.add_argument(
        "--market",
        choices=("spot", "futures"),
        default=None,
        help="Market provenance for the validation data. Defaults to the "
        "configured build_binance_indicator_dataset.MARKET.",
    )
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--min-trades-train",
        type=int,
        default=None,
        help="Stage-1 minimum trades (default 30; 15 with --position — "
        "swing cadence can't match day-trade counts).",
    )
    parser.add_argument(
        "--min-trades-val",
        type=int,
        default=None,
        help="Stage-2 minimum trades (default 10; 5 with --position).",
    )
    parser.add_argument(
        "--min-trades-holdout",
        type=int,
        default=None,
        help="Stage-5 minimum trades (default 5; 3 with --position).",
    )
    parser.add_argument(
        "--log", action="store_true", help="Also log synthetic runs (real runs always log)."
    )
    args = parser.parse_args()

    from research_exploration.evaluate import (
        build_aligned_frame,
        build_synthetic_aligned_frame,
    )
    from research_exploration.hypothesis_generator import (
        first_smoke_set,
        load_batch,
        position_trading_set,
    )

    if args.batch:
        hyps = load_batch(args.batch)
    elif args.position:
        hyps = position_trading_set(with_guards=args.with_guards)
    else:
        hyps = first_smoke_set(with_guards=args.with_guards)
    eval_kwargs = {
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "pnl_unit": args.pnl_unit,
    }
    if args.market is not None:
        eval_kwargs["market"] = args.market
    eval_cfg = EvalConfig(**eval_kwargs)
    # Stage minimums: day-trade cadence by default; --position candidates trade
    # weekly-to-monthly, so unchanged minimums would leave everything
    # "inconclusive" no matter how long the window.
    default_min = (15, 5, 3) if args.position else (30, 10, 5)
    min_train = args.min_trades_train if args.min_trades_train is not None else default_min[0]
    min_val = args.min_trades_val if args.min_trades_val is not None else default_min[1]
    min_hold = args.min_trades_holdout if args.min_trades_holdout is not None else default_min[2]
    if args.real:
        hyps = [h for h in hyps if h.base_timeframe == args.base_tf]
        frame = build_aligned_frame(hyps, base_tf=args.base_tf, start=args.start, end=args.end)
        print(f"Aligned real frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
        cfg = ValidationConfig(
            min_trades_train=min_train, min_trades_val=min_val, min_trades_holdout=min_hold
        )
        log_path = DEFAULT_LOG
    else:
        frame = build_synthetic_aligned_frame(hyps, n=6000)
        print(f"Synthetic frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols (smoke only)")
        # tiny frame: relax trade minimums + regime lookback so stages exercise
        cfg = ValidationConfig(
            min_trades_train=5, min_trades_val=2, min_trades_holdout=1, regime_lookback_bars=200
        )
        log_path = DEFAULT_LOG if args.log else None

    results = validate_batch(frame, hyps, cfg, eval_cfg=eval_cfg, log_path=log_path)
    print(summarize_results(results))
    if log_path:
        print(f"\nLogged {len(results)} records to {log_path}")


if __name__ == "__main__":
    main()
