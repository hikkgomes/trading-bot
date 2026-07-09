"""Generate a curated batch of concrete multi-timeframe hypotheses.

This is *not* a brute-force grid. It walks the named families in
``strategy_families`` across a small set of intentional timeframe combinations
and both directions, producing structured :class:`Hypothesis` objects ready to
test. The flow is: market idea -> precise hypothesis -> (later) controlled test.

Run:  python -m research_exploration.hypothesis_generator            # write batch
      python -m research_exploration.hypothesis_generator --print    # also echo
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.strategy_families import FAMILIES, apply_no_trade_guards, build_family

# Scenario 1 (position / BTC accumulation): coarse stacks, short-side only.
# With ``pnl_unit='btc'`` longs are equivalent to holding, so the only way to
# end up with MORE BTC than buy-and-hold is to be short during drawdowns —
# these are "dodge the pullback" hypotheses, validated in BTC terms.
POSITION_TF_COMBOS = [("1w", "1d", "4h"), ("1d", "4h", "1h"), ("1d", "4h", "4h")]
POSITION_EXIT_SCALE = 3.0     # day-trade exits are far too tight for multi-day swings
POSITION_HORIZON_SCALE = 2.0


def generate_batch(per_family_combos: int | None = None,
                   directions=("long", "short"),
                   with_guards: bool = False) -> list[Hypothesis]:
    """Build the candidate batch. ``per_family_combos`` caps timeframe combos
    per family (None == use all the family's default combos).

    ``with_guards`` layers the Family-F regime-avoidance guards (anti-chop +
    volatility band on the regime timeframe) onto every A–E candidate — a
    *stricter* variant, not a larger search. Default off so the canonical batch
    (and anything already logged against it) stays stable."""
    out: list[Hypothesis] = []
    for key, fam in FAMILIES.items():
        combos = fam.default_tf_combos
        if per_family_combos is not None:
            combos = combos[:per_family_combos]
        idx = 0
        for combo in combos:
            for direction in directions:
                if direction not in fam.directions:
                    continue
                idx += 1
                hyp = build_family(key, direction, combo, idx)
                if with_guards:
                    hyp = apply_no_trade_guards(hyp)
                out.append(hyp)
    return out


def first_smoke_set(directions=("long", "short"),
                    with_guards: bool = False) -> list[Hypothesis]:
    """A small (~20) diversified set: 2 timeframe combos per family, both
    directions. Good for the initial controlled smoke test."""
    return generate_batch(per_family_combos=2, directions=directions, with_guards=with_guards)


def position_trading_set(with_guards: bool = False,
                         exit_scale: float = POSITION_EXIT_SCALE,
                         horizon_scale: float = POSITION_HORIZON_SCALE) -> list[Hypothesis]:
    """Scenario 1: BTC-accumulation candidates. Same named families, but on
    coarse timeframe stacks, SHORT side only, with exits widened for multi-day
    swings. Validate with ``--pnl-unit btc``: a keep means the rule ended with
    more BTC than just holding across every validation stage."""
    out: list[Hypothesis] = []
    for key, fam in FAMILIES.items():
        if "short" not in fam.directions:
            continue
        for idx, combo in enumerate(POSITION_TF_COMBOS, start=1):
            hyp = build_family(key, "short", combo, idx)
            exit_rule = dataclasses.replace(
                hyp.exit,
                take_profit=round(hyp.exit.take_profit * exit_scale, 6),
                stop_loss=round(hyp.exit.stop_loss * exit_scale, 6),
                horizon_bars=max(1, round(hyp.exit.horizon_bars * horizon_scale)),
            )
            hyp = dataclasses.replace(
                hyp,
                id=f"POS_{hyp.id}",
                exit=exit_rule,
                expected_holding="days to weeks (position swing)",
                tags=[*hyp.tags, "position_btc"],
            )
            if with_guards:
                hyp = apply_no_trade_guards(hyp)
            out.append(hyp)
    return out


def write_batch(hyps: list[Hypothesis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "research_exploration.hypothesis_schema/v1",
        "count": len(hyps),
        "families": sorted({h.family for h in hyps}),
        "hypotheses": [h.to_dict() for h in hyps],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_batch(path: Path) -> list[Hypothesis]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Hypothesis.from_dict(d) for d in payload["hypotheses"]]


def summarize(hyps: list[Hypothesis]) -> str:
    from collections import Counter
    by_family = Counter(h.family for h in hyps)
    by_tf = Counter(
        f"{h.regime_timeframe}+{h.setup_timeframe}+{h.trigger_timeframe}" for h in hyps
    )
    lines = [f"Generated {len(hyps)} hypotheses.", "", "By family:"]
    lines += [f"  {fam:24} {n}" for fam, n in by_family.most_common()]
    lines += ["", "By timeframe stack:"]
    lines += [f"  {stack:20} {n}" for stack, n in by_tf.most_common()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a curated hypothesis batch.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON (default: hypotheses_batch_001.json, or "
                             "hypotheses_position_001.json with --position).")
    parser.add_argument("--smoke", action="store_true",
                        help="Only the small diversified smoke set (~20).")
    parser.add_argument("--position", action="store_true",
                        help="Scenario-1 batch: BTC-accumulation shorts on coarse stacks "
                             "(validate with --pnl-unit btc).")
    parser.add_argument("--with-guards", dest="with_guards", action="store_true",
                        help="Layer Family-F regime-avoidance guards (anti-chop + vol band) onto every candidate.")
    parser.add_argument("--print", dest="echo", action="store_true",
                        help="Echo each hypothesis' human summary.")
    args = parser.parse_args()

    if args.position:
        hyps = position_trading_set(with_guards=args.with_guards)
        default_out = Path("outputs/research_exploration/hypotheses_position_001.json")
    else:
        hyps = (first_smoke_set(with_guards=args.with_guards) if args.smoke
                else generate_batch(with_guards=args.with_guards))
        default_out = Path("outputs/research_exploration/hypotheses_batch_001.json")
    args.out = args.out or default_out
    write_batch(hyps, args.out)
    print(summarize(hyps))
    print(f"\nWrote {args.out}")
    if args.echo:
        print("\n" + "=" * 70)
        for h in hyps:
            print(h.human_summary())
            print("-" * 70)


if __name__ == "__main__":
    main()
