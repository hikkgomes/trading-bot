"""Cheap autonomous research wiring check.

This intentionally uses synthetic data only. It proves that hypothesis
generation, staged validation, and BTC/USDT evaluation units still execute under
the 24/7 scheduler without launching heavy real-data research.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research_exploration.evaluate import EvalConfig, build_synthetic_aligned_frame
from research_exploration.hypothesis_generator import first_smoke_set, position_trading_set
from research_exploration.validation import ValidationConfig, validate_batch
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str(result.get("verdict", "unknown")) for result in results)
    reasons = Counter(
        reason
        for result in results
        for reason in result.get("reasons", [])
    )
    return {
        "hypotheses": len(results),
        "verdicts": dict(sorted(verdicts.items())),
        "top_reasons": dict(reasons.most_common(8)),
    }


def _run_scenario(
    name: str,
    *,
    pnl_unit: str,
    position: bool,
    synthetic_rows: int,
    with_guards: bool = False,
) -> dict[str, Any]:
    hypotheses = (
        position_trading_set(with_guards=with_guards)
        if position
        else first_smoke_set(with_guards=with_guards)
    )
    frame = build_synthetic_aligned_frame(hypotheses, n=synthetic_rows)
    validation_cfg = ValidationConfig(
        min_trades_train=3,
        min_trades_val=1,
        min_trades_holdout=1,
        regime_lookback_bars=max(50, min(200, synthetic_rows // 5)),
    )
    eval_cfg = EvalConfig(pnl_unit=pnl_unit)
    results = validate_batch(frame, hypotheses, validation_cfg, eval_cfg=eval_cfg, log_path=None)
    return {
        "ok": True,
        "name": name,
        "pnl_unit": pnl_unit,
        "position": position,
        "with_guards": with_guards,
        "synthetic_rows": synthetic_rows,
        **_summarize_results(results),
    }


def run_research_smoke(synthetic_rows: int = 3000, with_guards: bool = False) -> dict[str, Any]:
    if synthetic_rows < 500:
        raise ValueError("synthetic_rows must be at least 500")
    scenarios: list[dict[str, Any]] = []
    for payload in (
        {"name": "active_income", "pnl_unit": "usdt", "position": False},
        {"name": "btc_accumulation", "pnl_unit": "btc", "position": True},
    ):
        try:
            scenarios.append(
                _run_scenario(
                    payload["name"],
                    pnl_unit=payload["pnl_unit"],
                    position=bool(payload["position"]),
                    synthetic_rows=synthetic_rows,
                    with_guards=with_guards,
                )
            )
        except Exception as exc:
            scenarios.append(
                {
                    "ok": False,
                    "name": payload["name"],
                    "pnl_unit": payload["pnl_unit"],
                    "position": payload["position"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": all(bool(scenario.get("ok")) for scenario in scenarios),
        "generated_at": utc_now(),
        "synthetic_only": True,
        "with_guards": with_guards,
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cheap synthetic research wiring check.")
    parser.add_argument("--synthetic-rows", type=int, default=3000)
    parser.add_argument("--with-guards", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_research_smoke(args.synthetic_rows, with_guards=args.with_guards)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
