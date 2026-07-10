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
    5. HOLDOUT     (final ``1-train-val``) — untouched until stages 1–4 pass;
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    "holdout_already_consumed",
    "insufficient_holdout_trades",
    "failed_holdout",
)


@dataclass
class ValidationConfig:
    """Split sizes, stage gates and perturbation sizes."""

    train_frac: float = 0.6
    val_frac: float = 0.2          # holdout = 1 - train_frac - val_frac
    min_trades_train: int = 30
    min_trades_val: int = 10
    min_trades_holdout: int = 5
    min_train_sharpe: float = 0.0  # train must at least be positive
    oos_windows: int = 6           # train+val chunked into this many OOS windows
    min_window_pass_rate: float = 0.5
    exit_delta: float = 0.25       # ±25% on TP / SL / horizon
    reference_jitter: float = 0.10 # ±10% on numeric entry thresholds
    quantile_jitter: float = 0.05  # ±0.05 on rolling-quantile predicates
    min_sensitivity_pass: float = 0.6
    regime_lookback_bars: int | None = None  # None -> ~30 days of base-tf bars
    regime_band: float = 0.10      # trailing return beyond ±band => bull/bear
    n_trials: int = 1              # DSR deflation; validate_batch enforces at least len(batch)

    def __post_init__(self) -> None:
        if not 0 < self.train_frac < 1 or not 0 < self.val_frac < 1:
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave room for a holdout")


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
def tag_regimes(frame: pd.DataFrame, base_tf: str,
                lookback_bars: int | None = None,
                lookback_days: int = 30, band: float = 0.10) -> pd.Series:
    """Label each bar bull/bear/range from the *trailing* ``lookback`` return of
    the base close. Uses only past bars — safe to group trades by. Bars inside
    the warmup are labelled 'unknown'."""
    close = frame[f"tf_{base_tf}_close"].astype(float)
    if lookback_bars is None:
        bars_per_day = max(1, 86_400 // TIMEFRAME_SECONDS[base_tf])
        lookback_bars = lookback_days * bars_per_day
    lookback_bars = max(1, min(int(lookback_bars), max(1, len(frame) - 1)))
    trailing = close / close.shift(lookback_bars) - 1.0
    labels = np.where(trailing > band, "bull",
                      np.where(trailing < -band, "bear", "range"))
    labels = np.where(trailing.isna().to_numpy(), "unknown", labels)
    return pd.Series(labels, index=frame.index, name="regime")


def regime_breakdown(frame: pd.DataFrame, hyp: Hypothesis,
                     eval_cfg: EvalConfig, cfg: ValidationConfig) -> list[dict]:
    """Split the hypothesis' trades by the regime active at entry."""
    trades, _, _ = hypothesis_trades(frame, hyp, eval_cfg)
    if trades.empty:
        return []
    regimes = tag_regimes(frame, hyp.base_timeframe,
                          lookback_bars=cfg.regime_lookback_bars, band=cfg.regime_band)
    lookup = pd.Series(regimes.to_numpy(), index=_trade_index(frame))
    trade_regime = lookup.reindex(pd.Index(trades["entry_time"])).fillna("unknown").to_numpy()
    rows = []
    for regime in ("bull", "bear", "range", "unknown"):
        r = trades.loc[trade_regime == regime, "net_return"].to_numpy()
        if r.size == 0:
            continue
        rows.append({
            "regime": regime, "trades": int(r.size),
            "win_rate": round(float((r > 0).mean()), 4),
            "total_return": round(float(np.prod(1 + r) - 1), 5),
            "avg_net": round(float(r.mean()), 6),
        })
    return rows


# --------------------------------------------------------------------------- #
# OOS window stability (fixed rule -> honest chronological chunking)
# --------------------------------------------------------------------------- #
def oos_window_stats(frame: pd.DataFrame, hyp: Hypothesis,
                     eval_cfg: EvalConfig, n_windows: int) -> dict:
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
        r = trades.loc[in_win.to_numpy(), "net_return"].to_numpy() if not trades.empty else np.array([])
        rows.append({
            "window": i + 1, "trades": int(r.size),
            "total_return": round(float(np.prod(1 + r) - 1), 5) if r.size else 0.0,
            "win_rate": round(float((r > 0).mean()), 4) if r.size else float("nan"),
        })
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
        if any(stages[s][i] != getattr(hyp, s)[i]
               for s in stages for i in range(len(stages[s]))):
            out.append((label, dataclasses.replace(hyp, id=f"{hyp.id}~{label}", **stages)))
    return out


def sensitivity_check(frame: pd.DataFrame, hyp: Hypothesis,
                      eval_cfg: EvalConfig, cfg: ValidationConfig) -> dict:
    """Re-evaluate every perturbed variant on pre-holdout data. A robust edge
    stays profitable for most neighbours of the chosen parameters."""
    variants = perturbed_variants(hyp, cfg)
    rows, profitable = [], 0
    for label, variant in variants:
        m = evaluate_hypothesis(frame, variant, eval_cfg)
        ok = bool(m["trades"] > 0 and m["total_return"] > 0)
        profitable += ok
        rows.append({"variant": label, "trades": m["trades"],
                     "total_return": m["total_return"], "sharpe": m["sharpe"],
                     "profitable": ok})
    frac = round(profitable / len(rows), 4) if rows else 0.0
    return {
        "variants": rows,
        "pass_fraction": frac,
        "passed": bool(rows) and frac >= cfg.min_sensitivity_pass,
    }


# --------------------------------------------------------------------------- #
# Staged validation of one hypothesis
# --------------------------------------------------------------------------- #
def validate_hypothesis(frame: pd.DataFrame, hyp: Hypothesis,
                        cfg: ValidationConfig | None = None,
                        eval_cfg: EvalConfig | None = None,
                        before_holdout: Callable[[Hypothesis, dict], bool | None] | None = None) -> dict:
    """Run the full staged pipeline. The holdout is only ever *touched* after
    every earlier stage passes, and a negative holdout rejects — it gates."""
    cfg = cfg or ValidationConfig()
    eval_cfg = eval_cfg or EvalConfig()
    segs = split_frame(frame, cfg)
    result: dict = {
        "hypothesis_id": hyp.id, "family": hyp.family, "direction": hyp.direction,
        "splits": {name: _segment_bounds(seg) for name, seg in segs.items()},
        "train": None, "validation": None, "holdout": None,
        "oos": None, "sensitivity": None, "regimes": None,
        "dsr_deflated": None, "n_trials": cfg.n_trials,
        "verdict": None, "reasons": [],
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
    if r.size > 3:
        sharpe = float(metrics.sharpe_ratio(r))
        result["dsr_deflated"] = round(float(metrics.deflated_sharpe_ratio(
            sharpe, n_trials=max(1, cfg.n_trials),
            skew=float(pd.Series(r).skew()), kurt=float(pd.Series(r).kurt() + 3.0),
            n_obs=int(r.size))), 4)

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

    # 5. HOLDOUT — untouched until now, and it GATES.  Autonomous callers use
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
def validate_batch(frame: pd.DataFrame, hyps: list[Hypothesis],
                   cfg: ValidationConfig | None = None,
                   eval_cfg: EvalConfig | None = None,
                   log_path: Path | None = None,
                   before_holdout: Callable[[Hypothesis, dict], bool | None] | None = None,
                   after_candidate: Callable[[Hypothesis, dict], None] | None = None) -> list[dict]:
    """Validate a batch with DSR deflated by at least the batch size.

    Callers that rotate a slice of a larger candidate universe can set
    ``cfg.n_trials`` higher than ``len(hyps)`` so DSR still pays for the
    hypotheses being searched over time. Optionally append one experiment-log
    record per hypothesis; skips nothing silently.
    """
    base_cfg = cfg or ValidationConfig()
    cfg = dataclasses.replace(base_cfg, n_trials=max(1, int(base_cfg.n_trials), len(hyps)))
    eval_cfg = eval_cfg or EvalConfig()
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
            headline = {"dsr_deflated": res["dsr_deflated"], "n_trials": cfg.n_trials,
                        "reasons": res["reasons"]}
            for seg in ("train", "validation", "holdout"):
                m = res.get(seg)
                if m:
                    headline[seg] = {k: m[k] for k in
                                     ("trades", "total_return", "win_rate", "sharpe")}
            if res.get("oos"):
                headline["oos_pass_rate"] = res["oos"]["pass_rate"]
            if res.get("sensitivity"):
                headline["sensitivity_pass_fraction"] = res["sensitivity"]["pass_fraction"]
            config = {"validation": dataclasses.asdict(cfg),
                      "eval": dataclasses.asdict(eval_cfg)}
            log_result(ExperimentRecord(
                hypothesis_id=hyp.id, family=hyp.family, direction=hyp.direction,
                fingerprint=fingerprint(hyp.to_dict(), config),
                verdict=res["verdict"], metrics=headline, config=config,
                data_window=res["splits"], notes="validation.py staged pipeline",
                hypothesis=hyp.to_dict(),
            ), log_path)
    return results


def summarize_results(results: list[dict]) -> str:
    lines = [f"{'hypothesis':44} {'verdict':13} {'train':>8} {'val':>8} "
             f"{'holdout':>8} {'oos':>5} {'sens':>5}  reasons"]
    def seg_ret(r, seg):
        m = r.get(seg)
        return f"{m['total_return']:+.3f}" if m else "-"

    for r in sorted(results, key=lambda x: (x["verdict"] != "keep", x["hypothesis_id"])):
        oos = f"{r['oos']['pass_rate']:.2f}" if r.get("oos") else "-"
        sens = f"{r['sensitivity']['pass_fraction']:.2f}" if r.get("sensitivity") else "-"
        lines.append(f"{r['hypothesis_id']:44} {r['verdict']:13} {seg_ret(r, 'train'):>8} "
                     f"{seg_ret(r, 'validation'):>8} {seg_ret(r, 'holdout'):>8} {oos:>5} {sens:>5}  "
                     f"{','.join(r['reasons']) or '-'}")
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
        description="Staged validation (train/val/OOS/sensitivity/GATING holdout).")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic smoke (default).")
    parser.add_argument("--real", action="store_true",
                        help="Real data (HEAVY; needs --base-tf, --start, --end; results are logged).")
    parser.add_argument("--base-tf", default="5m")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--batch", type=Path, default=None,
                        help="Hypotheses JSON (default: the generator's smoke set).")
    parser.add_argument("--position", action="store_true",
                        help="Use the scenario-1 position/BTC-accumulation batch "
                             "(pair with --pnl-unit btc).")
    parser.add_argument("--with-guards", action="store_true",
                        help="Use the Family-F guarded variant of the smoke set.")
    parser.add_argument("--pnl-unit", choices=("usdt", "btc"), default="usdt",
                        help="Scoring unit: usdt = day-trade/flow bot, btc = position/"
                             "accumulation bot (shorts only realise returns).")
    parser.add_argument("--market", choices=("spot", "futures"), default=None,
                        help="Market provenance for the validation data. Defaults to the "
                             "configured build_binance_indicator_dataset.MARKET.")
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--min-trades-train", type=int, default=None,
                        help="Stage-1 minimum trades (default 30; 15 with --position — "
                             "swing cadence can't match day-trade counts).")
    parser.add_argument("--min-trades-val", type=int, default=None,
                        help="Stage-2 minimum trades (default 10; 5 with --position).")
    parser.add_argument("--min-trades-holdout", type=int, default=None,
                        help="Stage-5 minimum trades (default 5; 3 with --position).")
    parser.add_argument("--log", action="store_true",
                        help="Also log synthetic runs (real runs always log).")
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
        frame = build_aligned_frame(hyps, base_tf=args.base_tf,
                                    start=args.start, end=args.end)
        print(f"Aligned real frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
        cfg = ValidationConfig(min_trades_train=min_train, min_trades_val=min_val,
                               min_trades_holdout=min_hold)
        log_path = DEFAULT_LOG
    else:
        frame = build_synthetic_aligned_frame(hyps, n=6000)
        print(f"Synthetic frame: {frame.shape[0]:,} rows x {frame.shape[1]} cols (smoke only)")
        # tiny frame: relax trade minimums + regime lookback so stages exercise
        cfg = ValidationConfig(min_trades_train=5, min_trades_val=2, min_trades_holdout=1,
                               regime_lookback_bars=200)
        log_path = DEFAULT_LOG if args.log else None

    results = validate_batch(frame, hyps, cfg, eval_cfg=eval_cfg, log_path=log_path)
    print(summarize_results(results))
    if log_path:
        print(f"\nLogged {len(results)} records to {log_path}")


if __name__ == "__main__":
    main()
