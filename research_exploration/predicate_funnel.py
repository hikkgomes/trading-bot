"""Predicate-level funnel diagnostic.

The smoke test told us *that* hypotheses were rejected. This tells us *why*, at
the level of the individual predicate. For each hypothesis it walks the
conjunction stage by stage and records how many rows survive:

    total -> regime -> setup -> trigger -> risk filter -> entry signals -> trades

It reports, per stage, the combined coverage and each predicate's standalone
coverage; a cumulative funnel that pinpoints the exact predicate that collapses
the survivor count to zero (the "killer"); the exit-reason mix and basic PnL; and
a single rejection reason from a fixed vocabulary.

It is **read-only**: no optimisation, no parameter changes, no new features, no
search. It reuses the same bounded real-data sample as the smoke
(``build_aligned_frame`` over a [start, end] window) and the same causal
``predicate_mask`` / canonical simulator as ``evaluate``.

Run:  python -m research_exploration.predicate_funnel \
          --base-tf 5m --start 2024-01-01 --end 2024-07-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from research_exploration.evaluate import (
    EvalConfig,
    build_aligned_frame,
    effective_rolling_window,
    entry_mask,
    evaluate_hypothesis,
    predicate_mask,
)
from research_exploration.hypothesis_generator import first_smoke_set
from research_exploration.hypothesis_schema import Hypothesis, Predicate
from src.build_dataset import TIMEFRAME_PREFIXES

# Fixed rejection vocabulary (ordered by precedence in `classify`).
REJECTION_REASONS = (
    "missing_columns",
    "invalid_timeframe_mapping",
    "invalid_timeframe_window",
    "regime_never_fires",
    "setup_never_fires",
    "trigger_never_fires",
    "combination_too_strict",
    "enough_signals_but_no_trades",
    "too_few_trades",
    "trades_exist_but_negative_expectancy",
    "candidate_positive",
)


@dataclass
class PredicateCoverage:
    stage: str
    description: str
    column: str
    true_rows: int
    valid_rows: int          # rows where the inputs are non-NaN (excludes warmup / pre-fill)
    total_rows: int

    @property
    def true_pct(self) -> float:
        return self.true_rows / self.total_rows if self.total_rows else 0.0


@dataclass
class FunnelStep:
    stage: str
    description: str
    standalone_true: int
    cum_before: int
    cum_after: int

    @property
    def dropped(self) -> int:
        return self.cum_before - self.cum_after


@dataclass
class HypothesisFunnel:
    hypothesis_id: str
    family: str
    direction: str
    base_tf: str
    regime_tf: str
    setup_tf: str
    trigger_tf: str
    total_rows: int
    missing_columns: list[str] = field(default_factory=list)
    per_predicate: list[PredicateCoverage] = field(default_factory=list)
    stage_combined: dict[str, int] = field(default_factory=dict)   # stage -> rows where stage-AND True
    cumulative: list[FunnelStep] = field(default_factory=list)
    predicate_entry_rows: int = 0      # AND of all predicates (before risk filter)
    risk_filtered_rows: int = 0        # after the NATR risk filter (== actual signals)
    trades: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)
    win_rate: float = float("nan")
    total_return: float = 0.0
    avg_net_return: float = float("nan")
    sharpe: float = 0.0
    psr: float = 0.0
    profit_factor: float = float("nan")
    max_drawdown: float = 0.0
    rejection_reason: str = ""
    killer_stage: str = ""
    killer_predicate: str = ""
    killer_kind: str = ""              # standalone_zero | combination | trade_level | none
    tightest_filter: str = ""          # most-restrictive predicate (context, esp. for trade-level)
    notes: list[str] = field(default_factory=list)


def _coverage(frame: pd.DataFrame, p: Predicate, stage: str, total: int,
              base_tf: str) -> PredicateCoverage:
    mask = predicate_mask(frame, p, base_tf=base_tf).fillna(False)
    cols = [c for c in p.columns_used() if c in frame.columns]
    if cols:
        valid = frame[cols].notna().all(axis=1)
        valid_rows = int(valid.sum())
    else:
        valid_rows = total
    return PredicateCoverage(
        stage=stage, description=p.describe(), column=p.column(),
        true_rows=int(mask.sum()), valid_rows=valid_rows, total_rows=total,
    )


def analyze_hypothesis(frame: pd.DataFrame, hyp: Hypothesis,
                       cfg: EvalConfig | None = None) -> HypothesisFunnel:
    cfg = cfg or EvalConfig()
    total = len(frame)
    f = HypothesisFunnel(
        hypothesis_id=hyp.id, family=hyp.family, direction=hyp.direction,
        base_tf=hyp.base_timeframe, regime_tf=hyp.regime_timeframe,
        setup_tf=hyp.setup_timeframe, trigger_tf=hyp.trigger_timeframe,
        total_rows=total,
    )

    # --- missing-column / invalid-mapping guards (graceful, no crash) ---------
    missing = [c for c in hyp.feature_columns() if c not in frame.columns]
    if missing:
        f.missing_columns = missing
        f.rejection_reason = "missing_columns"
        f.killer_kind = "none"
        f.killer_predicate = ", ".join(missing)
        return f
    base_pfx = TIMEFRAME_PREFIXES.get(hyp.base_timeframe)
    if base_pfx is None or f"{base_pfx}close" not in frame.columns:
        f.rejection_reason = "invalid_timeframe_mapping"
        f.killer_kind = "none"
        f.notes.append(f"base OHLC for {hyp.base_timeframe} not resolvable in frame")
        return f

    # --- timeframe-aware rolling window guard ---------------------------------
    # A rolling-window predicate on a higher tf is scaled to native HTF bars; if
    # that scaled window can't fit in this bounded sample, reject explicitly
    # rather than silently using a too-short (wrong-length) window.
    for p in hyp.all_predicates():
        if p.op in ("q_ge", "q_le"):
            try:
                eff = effective_rolling_window(p, hyp.base_timeframe)
            except KeyError as exc:
                f.rejection_reason = "invalid_timeframe_window"
                f.killer_kind = "none"
                f.killer_predicate = p.describe()
                f.notes.append(f"cannot scale window: {exc}")
                return f
            if eff >= total:
                f.rejection_reason = "invalid_timeframe_window"
                f.killer_kind = "none"
                f.killer_stage = "window"
                f.killer_predicate = p.describe()
                f.notes.append(f"native window {p.window}×{p.timeframe} = {eff:,} base bars "
                               f">= {total:,} sample rows — sample too short to express it")
                return f

    # --- per-stage coverage + cumulative funnel -------------------------------
    stages = [("regime", hyp.regime), ("setup", hyp.setup), ("trigger", hyp.trigger)]
    cum = pd.Series(True, index=frame.index)
    cum_count = total
    for stage, preds in stages:
        stage_mask = pd.Series(True, index=frame.index)
        for p in preds:
            cov = _coverage(frame, p, stage, total, hyp.base_timeframe)
            f.per_predicate.append(cov)
            pm = predicate_mask(frame, p, base_tf=hyp.base_timeframe).fillna(False)
            stage_mask &= pm
            before = cum_count
            cum &= pm
            cum_count = int(cum.sum())
            f.cumulative.append(FunnelStep(stage=stage, description=p.describe(),
                                           standalone_true=int(pm.sum()),
                                           cum_before=before, cum_after=cum_count))
        f.stage_combined[stage] = int(stage_mask.sum()) if preds else total

    f.predicate_entry_rows = cum_count

    # --- risk NATR filter (part of the real signal, applied after predicates) -
    signals = entry_mask(frame, hyp, cfg)
    f.risk_filtered_rows = int(signals.sum())
    if f.risk_filtered_rows < f.predicate_entry_rows:
        f.cumulative.append(FunnelStep(stage="risk", description="NATR volatility filter",
                                       standalone_true=f.risk_filtered_rows,
                                       cum_before=f.predicate_entry_rows,
                                       cum_after=f.risk_filtered_rows))

    # --- trades + PnL (canonical simulator) -----------------------------------
    m = evaluate_hypothesis(frame, hyp, cfg)
    f.trades = m["trades"]
    f.exit_reasons = m.get("exit_reasons", {})
    f.win_rate = m["win_rate"]
    f.total_return = m["total_return"]
    f.avg_net_return = m["avg_net_return"]
    f.sharpe = m["sharpe"]
    f.psr = m["psr"]
    f.profit_factor = m["profit_factor"]
    f.max_drawdown = m["max_drawdown"]

    classify(f, cfg)
    return f


def classify(f: HypothesisFunnel, cfg: EvalConfig) -> None:
    """Assign the single rejection reason + the killer predicate."""
    # stage never-fires (a required stage's AND is empty in this sample)
    for stage in ("regime", "setup", "trigger"):
        if f.stage_combined.get(stage, 1) == 0:
            f.rejection_reason = f"{stage}_never_fires"
            _set_stage_killer(f, stage)
            return

    if f.risk_filtered_rows == 0:
        # every stage fired alone but the combination (or risk filter) is empty
        f.rejection_reason = "combination_too_strict"
        _set_combination_killer(f)
        return

    if f.trades == 0:
        f.rejection_reason = "enough_signals_but_no_trades"
        f.killer_kind = "trade_level"
        f.killer_predicate = "non-overlap / horizon / max-entry-index rules"
        f.killer_stage = "trade_rules"
        return

    # trades exist: record the tightest predicate (context), judge PnL at trade level
    _record_tightest_filter(f)
    if f.trades < cfg.min_trades:
        f.rejection_reason = "too_few_trades"
        f.killer_kind = "trade_level"
        f.killer_stage = "trades"
        f.killer_predicate = f"too few entries survive trade rules (tightest filter: {f.tightest_filter})"
    elif f.avg_net_return is None or not (f.avg_net_return > 0) or f.total_return <= 0:
        f.rejection_reason = "trades_exist_but_negative_expectancy"
        f.killer_kind = "trade_level"
        f.killer_stage = "trades"
        f.killer_predicate = "negative trade expectancy — no single predicate; the entries lose money"
    else:
        f.rejection_reason = "candidate_positive"
        f.killer_kind = "none"
        f.killer_predicate = ""


def _set_stage_killer(f: HypothesisFunnel, stage: str) -> None:
    stage_preds = [c for c in f.per_predicate if c.stage == stage]
    zeros = [c for c in stage_preds if c.true_rows == 0]
    if zeros:
        worst = min(zeros, key=lambda c: c.true_rows)
        f.killer_kind = "standalone_zero"
        f.killer_predicate = worst.description
    else:
        worst = min(stage_preds, key=lambda c: c.true_rows)
        f.killer_kind = "combination"
        f.killer_predicate = f"{stage} combination (tightest: {worst.description})"
    f.killer_stage = stage


def _set_combination_killer(f: HypothesisFunnel) -> None:
    # first cumulative step that drove survivors to zero
    for step in f.cumulative:
        if step.cum_after == 0 and step.cum_before > 0:
            f.killer_stage = step.stage
            f.killer_predicate = step.description
            standalone = next((c.true_rows for c in f.per_predicate
                               if c.description == step.description), None)
            f.killer_kind = "standalone_zero" if standalone == 0 else "combination"
            return
    # never hit exactly zero in the listed steps -> risk filter or rounding
    f.killer_stage = "risk"
    f.killer_predicate = "NATR volatility filter"
    f.killer_kind = "combination"


def _record_tightest_filter(f: HypothesisFunnel) -> None:
    """The predicate that removed the most rows — context for trade-level rejects."""
    if not f.cumulative:
        return
    tightest = max(f.cumulative, key=lambda s: s.dropped)
    f.tightest_filter = tightest.description


def bottleneck(f: HypothesisFunnel) -> str:
    """The most informative single line about what limited this candidate."""
    if f.killer_kind == "trade_level" and f.tightest_filter:
        return f.tightest_filter
    return f.killer_predicate or "—"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
_IMPOSSIBLE = {"missing_columns", "invalid_timeframe_mapping", "invalid_timeframe_window",
               "regime_never_fires", "setup_never_fires", "trigger_never_fires"}
_TOO_STRICT = {"combination_too_strict", "enough_signals_but_no_trades"}
_BAD_TRADES = {"too_few_trades", "trades_exist_but_negative_expectancy"}


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.1f}%" if total else "—"


def build_report(funnels: list[HypothesisFunnel], meta: dict) -> str:
    L: list[str] = []
    L.append("# Predicate Funnel Report")
    L.append("")
    bases = ", ".join(f"{b} ({r:,} rows)" for b, r in meta["bases"])
    L.append(f"_Read-only diagnostic. Sample: **{meta['start']} → {meta['end']}**, "
             f"base timeframes {bases}. Fees {meta['fee_bps']}+{meta['slippage_bps']}bps, "
             f"min_trades={meta['min_trades']}. {len(funnels)} hypotheses._")
    L.append("")
    L.append("No optimisation, no new features, no walk-forward — this only explains "
             "*why* each hypothesis from the bounded smoke was rejected. Coverage %s are "
             "relative to each hypothesis's own base-timeframe row count.")
    L.append("")

    # ---- summary table ----
    L.append("## Summary")
    L.append("")
    L.append("| hypothesis | family | dir | base | regime | setup | trigger | entry | trades | ret% | reason |")
    L.append("|---|---|---|---|--:|--:|--:|--:|--:|--:|---|")
    for f in funnels:
        reg = _pct(f.stage_combined.get("regime", 0), f.total_rows)
        setp = _pct(f.stage_combined.get("setup", 0), f.total_rows)
        trg = _pct(f.stage_combined.get("trigger", 0), f.total_rows)
        ret = f"{f.total_return*100:+.2f}" if f.trades else "—"
        L.append(f"| {f.hypothesis_id} | {f.family} | {f.direction} | {f.base_tf} | {reg} | {setp} | "
                 f"{trg} | {f.risk_filtered_rows} | {f.trades} | {ret} | **{f.rejection_reason}** |")
    L.append("")

    # ---- buckets ----
    impossible = [f for f in funnels if f.rejection_reason in _IMPOSSIBLE]
    too_strict = [f for f in funnels if f.rejection_reason in _TOO_STRICT]
    bad_trades = [f for f in funnels if f.rejection_reason in _BAD_TRADES]
    positive = [f for f in funnels if f.rejection_reason == "candidate_positive"]

    L.append("## Verdict buckets")
    L.append("")
    L.append(f"- **Impossible with current data** ({len(impossible)}): a required predicate "
             "never fires in this sample (or a column is missing).")
    for f in impossible:
        L.append(f"  - `{f.hypothesis_id}` — {f.rejection_reason}; killer: **{f.killer_predicate}** "
                 f"({f.killer_stage}, {f.killer_kind})")
    L.append(f"- **Too strict** ({len(too_strict)}): every stage fires alone, but the combination "
             "(or risk filter) leaves no usable entries.")
    for f in too_strict:
        L.append(f"  - `{f.hypothesis_id}` — {f.rejection_reason}; killer: **{f.killer_predicate}** "
                 f"({f.killer_stage}, {f.killer_kind})")
    L.append(f"- **Signals but bad trade outcomes** ({len(bad_trades)}): entries exist, the trades "
             "lose or are too few to judge.")
    for f in bad_trades:
        L.append(f"  - `{f.hypothesis_id}` — {f.rejection_reason}; {f.trades} trades, "
                 f"ret {f.total_return*100:+.2f}%, win {f.win_rate*100:.0f}%, PF {f.profit_factor}")
    if positive:
        L.append(f"- **Positive candidates** ({len(positive)}): survived this single window — "
                 "still needs walk-forward + holdout before any belief.")
        for f in positive:
            L.append(f"  - `{f.hypothesis_id}` — {f.trades} trades, ret {f.total_return*100:+.2f}%, "
                     f"PSR {f.psr}")
    L.append("")

    # ---- readiness for entry-edge diagnostic ----
    ready = [f for f in funnels if f.trades >= meta["min_trades"]]
    L.append("## Ready for an entry-edge diagnostic")
    L.append("")
    L.append(f"A hypothesis is *ready* when it produces enough trades ({meta['min_trades']}+) to "
             "study its entry edge — independent of whether this single window was profitable. "
             "Impossible / too-strict / too-few-trade candidates are **not** ready (no sample).")
    L.append("")
    if ready:
        for f in sorted(ready, key=lambda x: -x.trades):
            L.append(f"- `{f.hypothesis_id}` ({f.family}/{f.direction}, base {f.base_tf}) — "
                     f"{f.trades} trades, win {f.win_rate*100:.0f}%, ret {f.total_return*100:+.2f}%, "
                     f"PF {f.profit_factor}, exits {f.exit_reasons}")
    else:
        L.append("- _none in this sample._")
    L.append("")

    # ---- family refinement summary ----
    L.append("## Which families deserve refinement")
    L.append("")
    fams: dict[str, list[HypothesisFunnel]] = {}
    for f in funnels:
        fams.setdefault(f.family, []).append(f)
    L.append("| family | n | dominant reason | typical bottleneck |")
    L.append("|---|--:|---|---|")
    for fam, fs in fams.items():
        from collections import Counter
        reasons = Counter(f.rejection_reason for f in fs)
        dom = reasons.most_common(1)[0][0]
        necks = Counter(bottleneck(f) for f in fs if bottleneck(f) != "—")
        typ = necks.most_common(1)[0][0] if necks else "—"
        L.append(f"| {fam} | {len(fs)} | {dom} | {typ} |")
    L.append("")

    # ---- per-hypothesis detail ----
    L.append("## Per-hypothesis funnels")
    L.append("")
    for f in funnels:
        L.append(f"### `{f.hypothesis_id}` — {f.family} / {f.direction}  (base {f.base_tf})")
        L.append("")
        if f.missing_columns:
            L.append(f"- **missing columns:** {', '.join(f.missing_columns)}")
            L.append("")
            continue
        # stage coverage
        L.append(f"TFs: regime={f.regime_tf} · setup={f.setup_tf} · trigger={f.trigger_tf}")
        L.append("")
        L.append("```")
        L.append(f"stage coverage (rows firing / {f.total_rows:,})")
        for stage in ("regime", "setup", "trigger"):
            sc = f.stage_combined.get(stage, 0)
            L.append(f"  {stage:7} AND : {sc:>8,} ({_pct(sc, f.total_rows)})")
            for c in [c for c in f.per_predicate if c.stage == stage]:
                warm = "" if c.valid_rows == f.total_rows else f"  [valid {c.valid_rows:,}]"
                L.append(f"      - {c.description:55} {c.true_rows:>8,} ({c.true_pct*100:4.1f}%){warm}")
        L.append("")
        L.append("cumulative funnel (regime -> setup -> trigger -> risk):")
        L.append(f"  {'start':55} {f.total_rows:>8,}")
        for step in f.cumulative:
            L.append(f"  + {step.description:53} {step.cum_after:>8,}  (-{step.dropped:,})")
        L.append(f"  => entry signals {' '*39} {f.risk_filtered_rows:>8,}")
        if f.trades:
            exits = " / ".join(f"{k} {v}" for k, v in f.exit_reasons.items())
            L.append(f"  => trades {' '*46} {f.trades:>8,}")
            L.append(f"     exits: {exits}")
            L.append(f"     win {f.win_rate*100:.0f}% | ret {f.total_return*100:+.2f}% | "
                     f"avg {f.avg_net_return*100:+.3f}% | PF {f.profit_factor} | "
                     f"PSR {f.psr} | maxDD {f.max_drawdown*100:.1f}%")
        L.append("```")
        L.append(f"**Reason:** `{f.rejection_reason}` — killer: **{f.killer_predicate}** "
                 f"(stage={f.killer_stage}, kind={f.killer_kind})")
        if f.killer_kind == "trade_level" and f.tightest_filter:
            L.append(f"  - tightest predicate (most rows removed): {f.tightest_filter}")
        if f.notes:
            for n in f.notes:
                L.append(f"  - note: {n}")
        L.append("")

    # ---- cross-cutting notes ----
    L.append("## Cross-cutting observations")
    L.append("")
    # surface the max_/min_ breakout self-reference if any trigger never fired on it
    breakout_zero = [f for f in funnels if f.rejection_reason == "trigger_never_fires"
                     and ("max_" in f.killer_predicate or "20-bar high" in f.killer_predicate
                          or "20-bar low" in f.killer_predicate)]
    if breakout_zero:
        L.append("- **STILL BROKEN: a breakout trigger references a current-bar-inclusive rolling "
                 f"extreme and never fires** ({len(breakout_zero)} candidate(s)). Expected 0 after "
                 "the `shift_b` fix — investigate.")
    else:
        L.append("- **FIXED — breakout/sweep now compare against the *prior* range.** Predicates "
                 "comparing `close`/`high`/`low` to `max_N`/`min_N` carry `shift_b=1`, i.e. "
                 "`close > max_20.shift(1)` (the prior 20-bar max-close, since TA-Lib `MAX`/`MIN` "
                 "run on close). This removes the old tautology (`close > max_20` was impossible; "
                 "`close > min_20` was always-true).")
    L.append("- **FIXED — rolling-quantile windows are now scaled to native HTF bars.** A "
             "`q_le(window=180)` on a 4h column is evaluated over `180 × (4h/base)` base rows, so "
             "it means 180 *4h* candles, not 180 base rows. Because the HTF column is forward-filled "
             "uniformly, the quantile over the scaled window equals the quantile over the 180 native "
             "values. If the scaled window can't fit the bounded sample, the hypothesis is rejected "
             "with `invalid_timeframe_window` instead of silently using the wrong length.")
    L.append("- **Risk NATR filters silently cut entries.** `trend_continuation` (min_atr_pct) and "
             "`mean_reversion` (max_atr_pct) drop signals after the predicate conjunction; the "
             "funnel's `risk` step shows how many.")
    L.append("- A single bounded window cannot tell *too strict* from *rare-but-real*. Treat "
             "`combination_too_strict` as 'revisit the conjunction', not 'delete the idea'.")
    L.append("")
    return "\n".join(L) + "\n"


def run_funnel(start: str, end: str, base_tfs: list[str], cfg: EvalConfig | None = None,
               hyps: list[Hypothesis] | None = None):
    """Build one aligned frame per base timeframe, analyze each base's hypotheses,
    and return (funnels, bases) where bases = [(base_tf, rows), ...].
    ``hyps`` defaults to the day-trade smoke set; pass any batch (e.g. the
    position set) to diagnose it instead."""
    cfg = cfg or EvalConfig()
    candidates = hyps if hyps is not None else first_smoke_set()
    funnels: list[HypothesisFunnel] = []
    bases: list[tuple[str, int]] = []
    for base_tf in base_tfs:
        base_hyps = [h for h in candidates if h.base_timeframe == base_tf]
        if not base_hyps:
            continue
        frame = build_aligned_frame(base_hyps, base_tf=base_tf, start=start, end=end)
        bases.append((base_tf, len(frame)))
        funnels.extend(analyze_hypothesis(frame, h, cfg) for h in base_hyps)
    return funnels, bases


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicate-level funnel diagnostic (read-only).")
    parser.add_argument("--base-tfs", default="5m,15m,30m",
                        help="Comma-separated base timeframes to cover (default: 5m,15m,30m).")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-07-01")
    parser.add_argument("--batch", type=Path, default=None,
                        help="Hypotheses JSON to diagnose (default: the smoke set).")
    parser.add_argument("--position", action="store_true",
                        help="Diagnose the scenario-1 position/BTC batch (use with "
                             "--base-tfs 4h,1h and a multi-year window).")
    parser.add_argument("--with-guards", action="store_true",
                        help="Guarded variant of the generated batch.")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/research_exploration/predicate_funnel_report.md"))
    args = parser.parse_args()

    from research_exploration.hypothesis_generator import load_batch, position_trading_set

    if args.batch:
        hyps = load_batch(args.batch)
    elif args.position:
        hyps = position_trading_set(with_guards=args.with_guards)
    else:
        hyps = first_smoke_set(with_guards=args.with_guards)

    cfg = EvalConfig()
    base_tfs = [b.strip() for b in args.base_tfs.split(",") if b.strip()]
    funnels, bases = run_funnel(args.start, args.end, base_tfs, cfg, hyps=hyps)
    if not funnels:
        have = sorted({h.base_timeframe for h in hyps})
        raise SystemExit(f"No hypotheses in this batch with base timeframes in {base_tfs} "
                         f"(batch bases: {have})")

    meta = {"start": args.start, "end": args.end, "bases": bases, "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps, "min_trades": cfg.min_trades}
    report = build_report(funnels, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    # console summary
    from collections import Counter
    reasons = Counter(f.rejection_reason for f in funnels)
    base_str = ", ".join(f"{b}={r:,}" for b, r in bases)
    print(f"Analyzed {len(funnels)} hypotheses over {args.start}..{args.end} (bases: {base_str})")
    for r, n in reasons.most_common():
        print(f"  {r:38} {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
