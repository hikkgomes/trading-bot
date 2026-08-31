"""Non-executable mutation planning from research watchlist candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research_exploration.experiment_log import DEFAULT_LOG, load_log
from research_exploration.hypothesis_schema import Hypothesis
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.reporting import utc_now

DEFAULT_INPUT = Path("runtime/research_cycle.json")
DEFAULT_OUTPUT = Path("runtime/mutation_plan.json")
DEFAULT_MARKDOWN = Path("runtime/mutation_plan.md")
DEFAULT_EXPLORATION_STATUS = Path("runtime/exploration_paper/status.json")
SKIPPED_CANDIDATE_SETS = {"mutation"}
RETIRED_REASONS = {"failed_holdout"}


FAMILY_ACTIONS = {
    "volatility_breakout": {
        "insufficient_train_trades": [
            "shorten compression lookback by one notch",
            "relax volume_z trigger by 0.25",
            "try the same breakout on the next finer configured base timeframe",
        ],
        "no_train_edge": [
            "discard this exact breakout trigger or require stronger regime direction",
            "test a prior-range lookback of 10 instead of 20",
            "compare with guarded and unguarded variants before expanding search",
        ],
    },
    "momentum_continuation": {
        "insufficient_train_trades": [
            "relax ADX threshold by 3 points",
            "widen the RSI continuation band",
            "try one finer trigger timeframe with the same regime stack",
        ],
        "no_train_edge": [
            "discard this exact momentum trigger or require stronger volume confirmation",
            "test a longer momentum lookback before retesting",
            "compare long and short symmetry before expanding search",
        ],
    },
    "trend_continuation": {
        "insufficient_train_trades": [
            "relax pullback RSI threshold by 3 points",
            "make volume confirmation optional in the mutation",
            "try the next lower setup timeframe for more samples",
        ],
        "no_train_edge": [
            "discard this exact pullback trigger or require cleaner trend context",
            "test a stricter EMA trend filter",
            "compare MACD cross with slope-based trigger",
        ],
    },
    "mean_reversion": {
        "insufficient_train_trades": [
            "relax RSI extreme threshold by 5 points",
            "test band touch without requiring close outside the band",
            "try one finer trigger timeframe for the reclaim",
        ],
        "no_train_edge": [
            "discard this exact fade or require volatility cap",
            "tighten invalidation before retesting",
            "compare fast reclaim trigger with candle-color trigger",
        ],
    },
    "liquidity_sweep": {
        "insufficient_train_trades": [
            "shorten sweep lookback from 50 to 30 bars",
            "relax reclaim volume_z threshold by 0.25",
            "try one finer trigger timeframe for the reclaim",
        ],
        "no_train_edge": [
            "discard this exact sweep trigger or require stronger higher-timeframe trend",
            "test reclaim against prior 10-bar level instead of prior 20-bar level",
            "compare RSI flip with candle-close reclaim only",
        ],
    },
}

REASON_ACTIONS = {
    "regime_never_fires": [
        "soften or replace the dominant regime predicate only",
        "leave setup, trigger, exit, and risk unchanged for attribution",
    ],
    "setup_never_fires": [
        "soften or replace the dominant setup predicate only",
        "leave regime, trigger, exit, and risk unchanged for attribution",
    ],
    "trigger_never_fires": [
        "soften or replace the dominant trigger predicate only",
        "leave regime, setup, exit, and risk unchanged for attribution",
    ],
    "trades_exist_but_negative_expectancy": [
        "leave signal frequency and entry predicates unchanged",
        "mutate the exit envelope before considering any entry change",
    ],
    "failed_validation": [
        "mutate thresholds only; do not expand family count yet",
        "retest the same family on the next market-data refresh",
    ],
    "unstable_across_windows": [
        "add or tighten a regime filter before retesting",
        "avoid promoting until profitable windows are evenly distributed",
    ],
    "parameter_fragile": [
        "widen exit robustness or simplify the entry condition",
        "prefer mutations with fewer sensitive numeric thresholds",
    ],
}

DEFAULT_ACTIONS = [
    "keep in bounded rotation",
    "retest only after new market data or a deterministic mutation",
]
MAX_SUPPRESSED_SOURCES_REPORT = 20


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_reason(candidate: dict[str, Any]) -> str:
    adaptive_reason = candidate.get("adaptive_mutation_reason")
    if isinstance(adaptive_reason, str) and adaptive_reason:
        return adaptive_reason
    reasons = candidate.get("reasons") or []
    return str(reasons[0]) if reasons else "unknown"


def _retired_reason(candidate: dict[str, Any]) -> str | None:
    reasons = {str(reason) for reason in candidate.get("reasons") or []}
    return next((reason for reason in RETIRED_REASONS if reason in reasons), None)


def _retired_candidate(
    scenario: dict[str, Any], candidate: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "source_scenario": scenario.get("name"),
        "product": scenario.get("product"),
        "source_candidate_id": candidate.get("id"),
        "reason": reason,
        "source_verdict": candidate.get("verdict"),
        "stage_reached": candidate.get("stage_reached"),
        "score": candidate.get("score"),
        "disposition": "retired_from_autonomous_mutation",
    }


def _actions_for(candidate: dict[str, Any]) -> list[str]:
    family = str(candidate.get("family") or "unknown")
    reason = _primary_reason(candidate)
    family_actions = FAMILY_ACTIONS.get(family, {})
    actions = family_actions.get(reason) or REASON_ACTIONS.get(reason) or DEFAULT_ACTIONS
    return list(actions)


def _proposal(
    scenario: dict[str, Any],
    candidate: dict[str, Any],
    *,
    source_hypothesis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reason = _primary_reason(candidate)
    proposal = {
        "id": f"mutate:{candidate.get('id', 'unknown')}:{reason}",
        "source_candidate_id": candidate.get("id"),
        "source_scenario": scenario.get("name"),
        "product": scenario.get("product"),
        "market": scenario.get("market"),
        "opportunity_type": scenario.get("opportunity_type"),
        "base_timeframe": scenario.get("base_tf"),
        "family": candidate.get("family"),
        "direction": candidate.get("direction"),
        "reason": reason,
        "source_verdict": candidate.get("verdict"),
        "stage_reached": candidate.get("stage_reached"),
        "score": candidate.get("score"),
        "actions": _actions_for(candidate),
        **(
            {
                "adaptive_feedback": candidate["adaptive_feedback"],
                "mutation_focus_stage": candidate["adaptive_feedback"].get("mutation_focus_stage"),
            }
            if isinstance(candidate.get("adaptive_feedback"), dict)
            else {}
        ),
        "validation_scope": {
            "candidate_set": scenario.get("candidate_set"),
            "pnl_unit": scenario.get("pnl_unit"),
            "with_guards": bool(scenario.get("with_guards", False)),
        },
        "safety": {
            "executable": False,
            "paper_trade_allowed": False,
            "promotion_allowed": False,
            "live_allowed": False,
            "requires_full_validation_before_export": True,
        },
    }
    if source_hypothesis is not None:
        parsed = Hypothesis.from_dict(source_hypothesis)
        if parsed.id != proposal["source_candidate_id"]:
            raise ValueError("source hypothesis identity does not match mutation proposal")
        proposal["source_hypothesis"] = parsed.to_dict()
    return proposal


def _exploration_feedback_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if not payload:
        return {}
    if (
        payload.get("schema") != "autopilot.exploration_paper/v1"
        or payload.get("adaptive_evidence") is not True
        or payload.get("promotion_eligible") is not False
    ):
        return {}
    actionable = {
        "regime_never_fires",
        "setup_never_fires",
        "trigger_never_fires",
        "trades_exist_but_negative_expectancy",
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in (payload.get("candidate_feedback") or {}).values():
        if not isinstance(item, dict) or item.get("diagnosis") not in actionable:
            continue
        product = str(item.get("product") or "")
        hypothesis_id = str(item.get("hypothesis_id") or "")
        if not product or not hypothesis_id:
            continue
        result[(product, hypothesis_id)] = {
            key: item.get(key)
            for key in (
                "diagnosis",
                "mutation_focus_stage",
                "cycles",
                "data_ready",
                "market_bars_processed",
                "signals",
                "entries_opened",
                "completed_trades",
                "net_return_sum",
                "sized_return_sum",
                "signal_frequency",
                "failed_stages",
                "failed_predicates",
            )
            if key in item
        }
    return result


def _source_hypotheses(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for record in load_log(path):
        if not isinstance(record, dict) or not isinstance(record.get("hypothesis"), dict):
            continue
        hypothesis = Hypothesis.from_dict(record["hypothesis"])
        timestamp = str(record.get("timestamp") or "")
        prior = latest.get(hypothesis.id)
        if prior is None or timestamp > prior[0]:
            latest[hypothesis.id] = (timestamp, hypothesis.to_dict())
    return {key: value[1] for key, value in latest.items()}


def _source_key(scenario_name: Any, candidate_id: Any, reason: Any) -> tuple[str, str, str] | None:
    if not scenario_name or not candidate_id or not reason:
        return None
    return (str(scenario_name), str(candidate_id), str(reason))


def _recent_failed_mutation_sources(
    research_cycle: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    suppressed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scenario in research_cycle.get("scenarios") or []:
        if not isinstance(scenario, dict):
            continue
        if str(scenario.get("candidate_set") or "") not in SKIPPED_CANDIDATE_SETS:
            continue
        for candidate in scenario.get("incubation_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            lineage = candidate.get("mutation_lineage")
            if not isinstance(lineage, dict):
                continue
            key = _source_key(
                lineage.get("source_scenario"),
                lineage.get("source_candidate_id"),
                lineage.get("mutation_reason"),
            )
            if key is None:
                continue
            suppressed[key] = {
                "product": scenario.get("product"),
                "source_scenario": key[0],
                "source_candidate_id": key[1],
                "reason": key[2],
                "mutation_candidate_id": candidate.get("id"),
                "mutation_result_reasons": list(candidate.get("reasons") or []),
                "stage_reached": candidate.get("stage_reached"),
                "verdict": candidate.get("verdict"),
            }
    return suppressed


def _scenario_mutation_candidates(
    scenario: dict[str, Any],
    forward_feedback: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    retired_candidates: list[dict[str, Any]] = []
    for raw_candidate in scenario.get("incubation_candidates") or []:
        if not isinstance(raw_candidate, dict):
            continue
        retired_reason = _retired_reason(raw_candidate)
        if retired_reason is not None:
            retired_candidates.append(_retired_candidate(scenario, raw_candidate, retired_reason))
            continue
        candidate = dict(raw_candidate)
        feedback = forward_feedback.get(
            (str(scenario.get("product") or ""), str(candidate.get("id") or ""))
        )
        if feedback is not None:
            candidate["adaptive_mutation_reason"] = feedback["diagnosis"]
            candidate["adaptive_feedback"] = feedback
        candidates.append(candidate)
    return candidates, retired_candidates


def _select_mutation_candidates(
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
    suppressed_sources: dict[tuple[str, str, str], dict[str, Any]],
    top_per_scenario: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    ):
        reason = _primary_reason(candidate)
        key = _source_key(scenario.get("name"), candidate.get("id"), reason)
        if key is not None and key in suppressed_sources:
            suppressed.append(suppressed_sources[key])
            continue
        selected.append(candidate)
        if len(selected) >= top_per_scenario:
            break
    return selected, suppressed


def build_mutation_plan(
    research_cycle: dict[str, Any],
    *,
    top_per_scenario: int = 2,
    max_total: int = 12,
    exploration_status: dict[str, Any] | None = None,
    source_hypotheses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    skipped_scenarios: list[dict[str, Any]] = []
    suppressed_sources = _recent_failed_mutation_sources(research_cycle)
    suppressed_repeated_sources: list[dict[str, Any]] = []
    retired_candidates: list[dict[str, Any]] = []
    forward_feedback = _exploration_feedback_map(exploration_status or {})
    source_hypotheses = source_hypotheses or {}
    for scenario in research_cycle.get("scenarios") or []:
        if not isinstance(scenario, dict):
            continue
        candidate_set = str(scenario.get("candidate_set") or "")
        if candidate_set in SKIPPED_CANDIDATE_SETS:
            skipped_scenarios.append(
                {
                    "name": scenario.get("name"),
                    "candidate_set": candidate_set,
                    "reason": "mutation_depth_limit",
                    "incubation_candidates": len(scenario.get("incubation_candidates") or []),
                }
            )
            continue
        candidates, retired = _scenario_mutation_candidates(scenario, forward_feedback)
        retired_candidates.extend(retired)
        selected, suppressed = _select_mutation_candidates(
            scenario, candidates, suppressed_sources, top_per_scenario
        )
        suppressed_repeated_sources.extend(suppressed)
        proposals.extend(
            _proposal(
                scenario,
                candidate,
                source_hypothesis=source_hypotheses.get(str(candidate.get("id") or "")),
            )
            for candidate in selected
        )
    proposals = sorted(
        proposals,
        key=lambda proposal: float(proposal.get("score") or 0.0),
        reverse=True,
    )[:max_total]
    by_product = Counter(str(item.get("product") or "unknown") for item in proposals)
    by_reason = Counter(str(item.get("reason") or "unknown") for item in proposals)
    suppressed_by_product = Counter(
        str(item.get("product") or "unknown") for item in suppressed_repeated_sources
    )
    suppressed_by_reason = Counter(
        str(item.get("reason") or "unknown") for item in suppressed_repeated_sources
    )
    retired_by_product = Counter(
        str(item.get("product") or "unknown") for item in retired_candidates
    )
    retired_by_reason = Counter(str(item.get("reason") or "unknown") for item in retired_candidates)
    return {
        "ok": True,
        "generated_at": utc_now(),
        "source": {
            "research_generated_at": research_cycle.get("generated_at"),
            "keepers": (research_cycle.get("summary") or {}).get("keepers", 0),
            "exports": (research_cycle.get("summary") or {}).get("exported", 0),
            "exploration_generated_at": (exploration_status or {}).get("generated_at"),
        },
        "summary": {
            "proposals": len(proposals),
            "by_product": dict(sorted(by_product.items())),
            "by_reason": dict(by_reason.most_common()),
            "skipped_scenarios": len(skipped_scenarios),
            "suppressed_repeated_sources": len(suppressed_repeated_sources),
            "suppressed_by_product": dict(sorted(suppressed_by_product.items())),
            "suppressed_by_reason": dict(suppressed_by_reason.most_common()),
            "retired_candidates": len(retired_candidates),
            "retired_by_product": dict(sorted(retired_by_product.items())),
            "retired_by_reason": dict(retired_by_reason.most_common()),
            "executable": False,
        },
        "skipped_scenarios": skipped_scenarios,
        "suppressed_repeated_sources": suppressed_repeated_sources[:MAX_SUPPRESSED_SOURCES_REPORT],
        "retired_candidates": retired_candidates,
        "proposals": proposals,
    }


def render_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary") or {}
    lines = [
        "# Autopilot Mutation Plan",
        "",
        f"- Generated: `{plan.get('generated_at', 'unknown')}`",
        f"- Proposals: `{summary.get('proposals', 0)}`",
        f"- Skipped scenarios: `{summary.get('skipped_scenarios', 0)}`",
        f"- Suppressed repeat sources: `{summary.get('suppressed_repeated_sources', 0)}`",
        f"- Retired holdout failures: `{summary.get('retired_candidates', 0)}`",
        "- Executable: `False`",
        "- Scope: research-only; every proposal requires full validation before export.",
        "",
        "| Product | Source | Reason | Action |",
        "|---|---|---|---|",
    ]
    for proposal in plan.get("proposals") or []:
        actions = proposal.get("actions") or []
        first_action = str(actions[0]) if actions else "continue bounded rotation"
        lines.append(
            f"| {proposal.get('product', 'unknown')} | "
            f"`{proposal.get('source_candidate_id', 'unknown')}` | "
            f"`{proposal.get('reason', 'unknown')}` | {first_action} |"
        )
    if not plan.get("proposals"):
        lines.append("| none | none | none | no watchlist candidates available |")
    return "\n".join(lines) + "\n"


def run(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    markdown_path: Path | None = DEFAULT_MARKDOWN,
    top_per_scenario: int = 2,
    max_total: int = 12,
    exploration_status_path: Path = DEFAULT_EXPLORATION_STATUS,
    experiment_log_path: Path = DEFAULT_LOG,
) -> dict[str, Any]:
    research_cycle = _load_json(input_path)
    if not research_cycle:
        plan = {
            "ok": True,
            "generated_at": utc_now(),
            "status": "waiting_for_research_cycle",
            "summary": {"proposals": 0, "by_product": {}, "by_reason": {}, "executable": False},
            "proposals": [],
        }
    else:
        plan = build_mutation_plan(
            research_cycle,
            top_per_scenario=top_per_scenario,
            max_total=max_total,
            exploration_status=_load_json(exploration_status_path),
            source_hypotheses=_source_hypotheses(experiment_log_path),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, plan)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(markdown_path, render_markdown(plan))
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-executable research mutation plan.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--top-per-scenario", type=int, default=2)
    parser.add_argument("--max-total", type=int, default=12)
    parser.add_argument("--exploration-status", type=Path, default=DEFAULT_EXPLORATION_STATUS)
    parser.add_argument("--experiment-log", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = run(
        input_path=args.input,
        output_path=args.output,
        markdown_path=args.markdown_output,
        top_per_scenario=args.top_per_scenario,
        max_total=args.max_total,
        exploration_status_path=args.exploration_status,
        experiment_log_path=args.experiment_log,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
