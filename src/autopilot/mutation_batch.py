"""Build non-executable research hypotheses from mutation-plan proposals."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research_exploration.hypothesis_generator import (
    first_smoke_set,
    generate_batch,
    position_trading_set,
)
from research_exploration.hypothesis_schema import ExitRule, Hypothesis, Predicate
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now

DEFAULT_INPUT = Path("runtime/mutation_plan.json")
DEFAULT_OUTPUT = Path("runtime/mutation_hypotheses.json")
SCHEMA = "research_exploration.hypothesis_schema/v1"
SAFETY = {
    "research_only": True,
    "executable": False,
    "paper_trade_allowed": False,
    "promotion_allowed": False,
    "live_allowed": False,
    "requires_full_validation_before_export": True,
}
UNSAFE_PLAN_FLAGS = ("executable", "paper_trade_allowed", "promotion_allowed", "live_allowed")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_universe(scope: dict[str, Any]) -> list[Hypothesis]:
    candidate_set = str(scope.get("candidate_set") or "full")
    with_guards = bool(scope.get("with_guards", False))
    if candidate_set == "mutation":
        return []
    if candidate_set == "position":
        return position_trading_set(with_guards=with_guards)
    if candidate_set == "smoke":
        return first_smoke_set(with_guards=with_guards)
    return generate_batch(with_guards=with_guards)


def _find_source_hypothesis(proposal: dict[str, Any]) -> Hypothesis | None:
    source_id = str(proposal.get("source_candidate_id") or "")
    if not source_id:
        return None
    scope = (
        proposal.get("validation_scope")
        if isinstance(proposal.get("validation_scope"), dict)
        else {}
    )
    universe = _candidate_universe(scope)
    by_id = {hyp.id: hyp for hyp in universe}
    if source_id in by_id:
        return by_id[source_id]

    # Defensive fallback for older plans whose guarded flag was not persisted.
    alternate_scope = dict(scope)
    alternate_scope["with_guards"] = not bool(scope.get("with_guards", False))
    for hyp in _candidate_universe(alternate_scope):
        if hyp.id == source_id:
            return hyp
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _with_note(pred: Predicate, suffix: str) -> Predicate:
    note = pred.note
    if suffix not in note:
        note = f"{note} [{suffix}]" if note else suffix
    return dataclasses.replace(pred, note=note)


def _mutate_predicate(pred: Predicate, reason: str) -> Predicate:
    suffix = f"mutation:{reason}"
    if reason == "insufficient_train_trades":
        if pred.op in {"ge", "gt"} and pred.reference is not None:
            if pred.feature == "volume_z_20":
                return _with_note(
                    dataclasses.replace(pred, reference=round(max(0.0, pred.reference - 0.25), 4)),
                    suffix,
                )
            if pred.feature == "adx_14":
                return _with_note(
                    dataclasses.replace(pred, reference=round(max(10.0, pred.reference - 3.0), 4)),
                    suffix,
                )
        if pred.op in {"le", "lt"} and pred.reference is not None and pred.feature == "rsi_14":
            return _with_note(
                dataclasses.replace(pred, reference=round(min(85.0, pred.reference + 4.0), 4)),
                suffix,
            )
        if pred.op in {"ge", "gt"} and pred.reference is not None and pred.feature == "rsi_14":
            return _with_note(
                dataclasses.replace(pred, reference=round(max(15.0, pred.reference - 4.0), 4)),
                suffix,
            )
        if pred.op in {"q_ge", "q_le"} and pred.window is not None:
            return _with_note(
                dataclasses.replace(pred, window=max(30, round(pred.window * 0.75))), suffix
            )
        if pred.op == "between" and pred.low is not None and pred.high is not None:
            return _with_note(
                dataclasses.replace(
                    pred,
                    low=round(_clamp(pred.low - 3.0, 0.0, 100.0), 4),
                    high=round(_clamp(pred.high + 3.0, 0.0, 100.0), 4),
                ),
                suffix,
            )
        return pred

    if reason == "no_train_edge":
        if pred.op in {"ge", "gt"} and pred.reference is not None:
            if pred.feature == "volume_z_20":
                return _with_note(
                    dataclasses.replace(pred, reference=round(pred.reference + 0.25, 4)), suffix
                )
            if pred.feature == "adx_14":
                return _with_note(
                    dataclasses.replace(pred, reference=round(pred.reference + 3.0, 4)), suffix
                )
        if pred.op == "q_le" and pred.quantile is not None:
            return _with_note(
                dataclasses.replace(
                    pred, quantile=round(_clamp(pred.quantile - 0.05, 0.05, 0.95), 4)
                ),
                suffix,
            )
        if pred.op == "q_ge" and pred.quantile is not None:
            return _with_note(
                dataclasses.replace(
                    pred, quantile=round(_clamp(pred.quantile + 0.05, 0.05, 0.95), 4)
                ),
                suffix,
            )
        if (
            pred.op == "between"
            and pred.low is not None
            and pred.high is not None
            and pred.high - pred.low > 8.0
        ):
            return _with_note(
                dataclasses.replace(
                    pred, low=round(pred.low + 2.0, 4), high=round(pred.high - 2.0, 4)
                ),
                suffix,
            )
        return pred

    return pred


def _mutate_exit(exit_rule: ExitRule, reason: str) -> ExitRule:
    if reason == "no_train_edge":
        return dataclasses.replace(exit_rule, take_profit=round(exit_rule.take_profit * 1.1, 6))
    if reason == "insufficient_train_trades":
        return dataclasses.replace(
            exit_rule, horizon_bars=max(1, round(exit_rule.horizon_bars * 1.15))
        )
    return exit_rule


def _mutation_id(source_id: str, reason: str, index: int) -> str:
    reason_part = (
        "".join(ch if ch.isalnum() else "_" for ch in reason.upper()).strip("_") or "UNKNOWN"
    )
    return f"MUT_{source_id}_{reason_part}_{index:03d}"


def _explicit_unsafe_flags(payload: dict[str, Any]) -> list[str]:
    return [flag for flag in UNSAFE_PLAN_FLAGS if payload.get(flag) is True]


def _plan_unsafe_flags(plan: dict[str, Any]) -> list[str]:
    flags = _explicit_unsafe_flags(plan)
    summary = plan.get("summary")
    if isinstance(summary, dict):
        flags.extend(f"summary.{flag}" for flag in _explicit_unsafe_flags(summary))
    return flags


def _proposal_unsafe_flags(proposal: dict[str, Any]) -> list[str]:
    flags = _explicit_unsafe_flags(proposal)
    safety = proposal.get("safety")
    if isinstance(safety, dict):
        flags.extend(f"safety.{flag}" for flag in _explicit_unsafe_flags(safety))
    return flags


def mutate_hypothesis(source: Hypothesis, proposal: dict[str, Any], index: int) -> Hypothesis:
    reason = str(proposal.get("reason") or "unknown")
    mutated = dataclasses.replace(
        source,
        id=_mutation_id(source.id, reason, index),
        idea=f"{source.idea} Mutation from autopilot watchlist: {reason}.",
        regime=[_mutate_predicate(pred, reason) for pred in source.regime],
        setup=[_mutate_predicate(pred, reason) for pred in source.setup],
        trigger=[_mutate_predicate(pred, reason) for pred in source.trigger],
        exit=_mutate_exit(source.exit, reason),
        invalidation=(
            f"{source.invalidation} Mutation remains research-only until full validation and explicit approval."
        ),
        tags=sorted({*source.tags, "mutation_plan", f"reason:{reason}", "research_only"}),
    )
    return Hypothesis.from_dict(mutated.to_dict())


def build_mutation_batch(plan: dict[str, Any], *, max_total: int | None = None) -> dict[str, Any]:
    unsafe_plan_flags = _plan_unsafe_flags(plan)
    if unsafe_plan_flags:
        return {
            "ok": False,
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "status": "unsafe_mutation_plan",
            "source": {
                "path": str(DEFAULT_INPUT),
                "plan_generated_at": plan.get("generated_at"),
                "plan_proposals": (plan.get("summary") or {}).get("proposals", 0),
            },
            **SAFETY,
            "count": 0,
            "families": [],
            "summary": {
                "hypotheses": 0,
                "skipped": 0,
                "by_product": {},
                "executable": False,
                "unsafe_flags": unsafe_plan_flags,
            },
            "mutation_metadata": [],
            "skipped": [],
            "hypotheses": [],
        }
    proposals = [item for item in plan.get("proposals") or [] if isinstance(item, dict)]
    if max_total is not None:
        proposals = proposals[:max_total]
    hypotheses: list[Hypothesis] = []
    mutation_metadata: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        unsafe_proposal_flags = _proposal_unsafe_flags(proposal)
        if unsafe_proposal_flags:
            skipped.append(
                {
                    "source_candidate_id": proposal.get("source_candidate_id"),
                    "reason": "unsafe_proposal",
                    "unsafe_flags": unsafe_proposal_flags,
                }
            )
            continue
        source = _find_source_hypothesis(proposal)
        if source is None:
            skipped.append(
                {
                    "source_candidate_id": proposal.get("source_candidate_id"),
                    "reason": "source_candidate_not_found",
                }
            )
            continue
        mutated = mutate_hypothesis(source, proposal, index)
        hypotheses.append(mutated)
        mutation_metadata.append(
            {
                "id": mutated.id,
                "source_candidate_id": source.id,
                "source_scenario": proposal.get("source_scenario"),
                "product": proposal.get("product"),
                "market": proposal.get("market"),
                "opportunity_type": proposal.get("opportunity_type"),
                "reason": proposal.get("reason"),
                "validation_scope": proposal.get("validation_scope") or {},
                "safety": dict(SAFETY),
            }
        )
    by_product = Counter(str(item.get("product") or "unknown") for item in mutation_metadata)
    return {
        "ok": True,
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "source": {
            "path": str(DEFAULT_INPUT),
            "plan_generated_at": plan.get("generated_at"),
            "plan_proposals": (plan.get("summary") or {}).get("proposals", len(proposals)),
        },
        **SAFETY,
        "count": len(hypotheses),
        "families": sorted({hyp.family for hyp in hypotheses}),
        "summary": {
            "hypotheses": len(hypotheses),
            "skipped": len(skipped),
            "by_product": dict(sorted(by_product.items())),
            "executable": False,
        },
        "mutation_metadata": mutation_metadata,
        "skipped": skipped,
        "hypotheses": [hyp.to_dict() for hyp in hypotheses],
    }


def run(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    max_total: int | None = None,
) -> dict[str, Any]:
    plan = _load_json(input_path)
    if not plan:
        payload = {
            "ok": True,
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "status": "waiting_for_mutation_plan",
            "source": {"path": str(input_path)},
            **SAFETY,
            "count": 0,
            "families": [],
            "summary": {"hypotheses": 0, "skipped": 0, "by_product": {}, "executable": False},
            "mutation_metadata": [],
            "skipped": [],
            "hypotheses": [],
        }
    else:
        payload = build_mutation_batch(plan, max_total=max_total)
        payload["source"]["path"] = str(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-executable mutation hypotheses for research validation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-total", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(input_path=args.input, output_path=args.output, max_total=args.max_total)
    compact = {
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "count": payload.get("count", 0),
        "skipped": (payload.get("summary") or {}).get("skipped", 0),
        "research_only": payload.get("research_only"),
        "executable": payload.get("executable"),
        "output": str(args.output),
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
