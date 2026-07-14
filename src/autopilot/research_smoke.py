"""Cheap autonomous research wiring check.

This intentionally uses synthetic data only. It proves that hypothesis
generation, staged validation, and BTC/USDT evaluation units still execute under
the 24/7 scheduler without launching heavy real-data research.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from research_exploration.evaluate import EvalConfig, build_synthetic_aligned_frame
from research_exploration.strategy_grammar import (
    build_fresh_hypothesis,
    mutate_hypothesis,
)
from research_exploration.validation import ValidationConfig, validate_batch
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now
from src.autopilot.research_factory import DEFAULT_CONFIG, load_factory_config


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str(result.get("verdict", "unknown")) for result in results)
    reasons = Counter(reason for result in results for reason in result.get("reasons", []))
    return {
        "hypotheses": len(results),
        "verdicts": dict(sorted(verdicts.items())),
        "top_reasons": dict(reasons.most_common(8)),
    }


def _run_scenario(
    space,
    *,
    synthetic_rows: int,
    seed: int,
) -> dict[str, Any]:
    parent = build_fresh_hypothesis(space, rng=random.Random(seed))
    second = build_fresh_hypothesis(space, rng=random.Random(seed + 1))
    child = mutate_hypothesis(
        parent.hypothesis,
        space,
        parent_hash="sha256:" + f"{seed:064x}"[-64:],
        rng=random.Random(seed + 2),
    )
    hypotheses = [
        dataclasses.replace(parent.hypothesis, id=f"SMOKE_{space.name}_FRESH_A"),
        dataclasses.replace(second.hypothesis, id=f"SMOKE_{space.name}_FRESH_B"),
        dataclasses.replace(child.hypothesis, id=f"SMOKE_{space.name}_MUTATION"),
    ]
    frame = build_synthetic_aligned_frame(hypotheses, n=synthetic_rows)
    validation_cfg = ValidationConfig(
        min_trades_train=3,
        min_trades_val=1,
        min_trades_holdout=1,
        regime_lookback_bars=max(50, min(200, synthetic_rows // 5)),
    )
    eval_cfg = EvalConfig(pnl_unit=space.pnl_unit, market=space.market)
    results = validate_batch(frame, hypotheses, validation_cfg, eval_cfg=eval_cfg, log_path=None)
    return {
        "ok": True,
        "name": space.name,
        "product": space.product,
        "pnl_unit": space.pnl_unit,
        "market": space.market,
        "opportunity_type": space.opportunity_type,
        "base_timeframe": space.base_timeframe,
        "synthetic_rows": synthetic_rows,
        "generation_methods": ["grammar_sample", "grammar_sample", "recursive_mutation"],
        **_summarize_results(results),
    }


def run_research_smoke(
    synthetic_rows: int = 3000,
    with_guards: bool = False,
    *,
    factory_config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if synthetic_rows < 500:
        raise ValueError("synthetic_rows must be at least 500")
    scenarios: list[dict[str, Any]] = []
    config = load_factory_config(factory_config_path)
    for index, space in enumerate(config.search_spaces, start=1):
        try:
            scenarios.append(
                _run_scenario(
                    space,
                    synthetic_rows=synthetic_rows,
                    seed=100 * index,
                )
            )
        except Exception as exc:
            scenarios.append(
                {
                    "ok": False,
                    "name": space.name,
                    "product": space.product,
                    "pnl_unit": space.pnl_unit,
                    "opportunity_type": space.opportunity_type,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": all(bool(scenario.get("ok")) for scenario in scenarios),
        "generated_at": utc_now(),
        "synthetic_only": True,
        "legacy_guards_requested": with_guards,
        "generator": "typed_compositional_grammar",
        "products": sorted({scenario.get("product") for scenario in scenarios}),
        "opportunity_types": sorted({scenario.get("opportunity_type") for scenario in scenarios}),
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cheap synthetic research wiring check.")
    parser.add_argument("--synthetic-rows", type=int, default=3000)
    parser.add_argument("--with-guards", action="store_true")
    parser.add_argument("--factory-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_research_smoke(
        args.synthetic_rows,
        with_guards=args.with_guards,
        factory_config_path=args.factory_config,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
