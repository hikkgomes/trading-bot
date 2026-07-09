"""Bounded real-data research cycle for the autopilot.

The cycle is intentionally conservative:

* validate a small, fixed candidate set for each product;
* append full staged-validation records to the experiment log;
* export only strategies that already passed the existing positive-holdout gate;
* treat "no exportable strategies" as a successful research result, not a
  runtime failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from research_exploration.evaluate import EvalConfig, _needed_columns, build_aligned_frame
from research_exploration.experiment_log import DEFAULT_LOG
from research_exploration.export import run as export_strategies
from research_exploration.hypothesis_generator import (
    first_smoke_set,
    generate_batch,
    position_trading_set,
)
from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.validation import ValidationConfig, validate_batch
from src.autopilot.io import write_json_atomic
from src.autopilot.market_data import build_market_data_statuses
from src.autopilot.reporting import utc_now
from src.config import indicator_data_dir

DEFAULT_OUTPUT = Path("runtime/research_cycle.json")
DEFAULT_STATE = Path("runtime/research_cycle_state.json")
DEFAULT_MUTATION_BATCH = Path("runtime/mutation_hypotheses.json")
DEFAULT_INCUBATION_OUTPUT = Path("runtime/incubation_candidates.json")
DEFAULT_PRODUCT_STATE_FILES = {
    "active_income": Path("runtime/active_income_state.json"),
    "btc_accumulation": Path("runtime/btc_accumulation_state.json"),
}


@dataclass(frozen=True)
class ResearchScenario:
    name: str
    product: str
    base_tf: str
    pnl_unit: str
    market: str
    position: bool
    start: str
    opportunity_type: str = "research"
    end: str | None = None
    with_guards: bool = False
    candidate_set: str = "smoke"
    max_hypotheses: int | None = None


DEFAULT_SCENARIOS = (
    ResearchScenario(
        name="active_income_15m",
        product="active_income",
        base_tf="15m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2022-01-01",
        opportunity_type="swing_trading",
        candidate_set="full",
        max_hypotheses=8,
    ),
    ResearchScenario(
        name="active_income_5m_guarded",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        opportunity_type="day_trading",
        with_guards=True,
        candidate_set="full",
        max_hypotheses=8,
    ),
    ResearchScenario(
        name="active_income_1m_guarded",
        product="active_income",
        base_tf="1m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2026-01-01",
        opportunity_type="scalping",
        with_guards=True,
        candidate_set="full",
        max_hypotheses=8,
    ),
    ResearchScenario(
        name="btc_accumulation_4h",
        product="btc_accumulation",
        base_tf="4h",
        pnl_unit="btc",
        market="spot",
        position=True,
        start="2020-06-01",
        opportunity_type="btc_accumulation",
        candidate_set="position",
    ),
    ResearchScenario(
        name="btc_accumulation_4h_guarded",
        product="btc_accumulation",
        base_tf="4h",
        pnl_unit="btc",
        market="spot",
        position=True,
        start="2020-06-01",
        opportunity_type="btc_accumulation",
        with_guards=True,
        candidate_set="position",
    ),
    ResearchScenario(
        name="btc_accumulation_1h",
        product="btc_accumulation",
        base_tf="1h",
        pnl_unit="btc",
        market="spot",
        position=True,
        start="2020-06-01",
        opportunity_type="btc_accumulation",
        candidate_set="position",
    ),
    ResearchScenario(
        name="btc_accumulation_1h_guarded",
        product="btc_accumulation",
        base_tf="1h",
        pnl_unit="btc",
        market="spot",
        position=True,
        start="2020-06-01",
        opportunity_type="btc_accumulation",
        with_guards=True,
        candidate_set="position",
    ),
)

DEFAULT_EXPORTS = {
    "active_income": {
        "pnl_unit": "usdt",
        "market": "futures",
        "out": Path("outputs/active_strategies_flow.json"),
        "top_k": 3,
        "min_dsr": 0.60,
    },
    "btc_accumulation": {
        "pnl_unit": "btc",
        "market": "spot",
        "out": Path("outputs/active_strategies_position.json"),
        "top_k": 3,
    },
}
MAX_INCUBATION_CANDIDATES_PER_SCENARIO = 3
MAX_MUTATION_HYPOTHESES_PER_SCENARIO = 4


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "version": 1,
            "_state_recovered": True,
            "_state_error": f"{type(exc).__name__}: {exc}",
        }
    if isinstance(payload, dict):
        return payload
    return {
        "version": 1,
        "_state_recovered": True,
        "_state_error": f"TypeError: expected JSON object, got {type(payload).__name__}",
    }


def _market_data_skip_marker(market_data_by_market: dict[str, dict[str, Any]]) -> str:
    marker: dict[str, dict[str, Any]] = {}
    for market, status in sorted(market_data_by_market.items()):
        item = status if isinstance(status, dict) else {}
        marker[str(market)] = {
            key: item.get(key)
            for key in (
                "ok",
                "exists",
                "reason",
                "rows",
                "first_timestamp",
                "last_timestamp",
                "path",
            )
            if key in item
        }
    return json.dumps(marker, sort_keys=True)


def _hypotheses_for(scenario: ResearchScenario):
    if scenario.candidate_set == "position" or scenario.position:
        hypotheses = position_trading_set(with_guards=scenario.with_guards)
    elif scenario.candidate_set == "full":
        hypotheses = generate_batch(with_guards=scenario.with_guards)
    elif scenario.candidate_set == "smoke":
        hypotheses = first_smoke_set(with_guards=scenario.with_guards)
    else:
        raise ValueError(f"{scenario.name}: unknown candidate_set {scenario.candidate_set!r}")
    return [hyp for hyp in hypotheses if hyp.base_timeframe == scenario.base_tf]


def _select_from_hypotheses(
    scenario: ResearchScenario,
    hypotheses: list[Hypothesis],
    state: dict[str, Any],
) -> tuple[list[Hypothesis], dict[str, Any]]:
    total = len(hypotheses)
    if total == 0:
        return [], {"available": 0, "selected": 0, "offset": 0, "next_offset": 0}
    limit = total if scenario.max_hypotheses is None else min(int(scenario.max_hypotheses), total)
    offsets = state.get("scenario_offsets", {})
    offset = int(offsets.get(scenario.name, 0) or 0) % total if isinstance(offsets, dict) else 0
    indices = [(offset + idx) % total for idx in range(limit)]
    selected = [hypotheses[idx] for idx in indices]
    next_offset = (offset + limit) % total
    return selected, {
        "available": total,
        "selected": len(selected),
        "offset": offset,
        "next_offset": next_offset,
        "wrapped": next_offset <= offset and limit < total,
        "candidate_set": scenario.candidate_set,
        "max_hypotheses": scenario.max_hypotheses,
        "ids": [hyp.id for hyp in selected],
    }


def _select_hypotheses(
    scenario: ResearchScenario,
    state: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    hypotheses = _hypotheses_for(scenario)
    return _select_from_hypotheses(scenario, hypotheses, state)


def _mutation_start(product: str, base_tf: str) -> str:
    if product == "btc_accumulation":
        return "2020-06-01"
    if base_tf == "1m":
        return "2026-01-01"
    if base_tf == "5m":
        return "2023-01-01"
    return "2022-01-01"


def _load_mutation_scenarios(
    path: Path,
    *,
    max_hypotheses_per_scenario: int = MAX_MUTATION_HYPOTHESES_PER_SCENARIO,
) -> tuple[tuple[ResearchScenario, ...], dict[str, list[Hypothesis]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    if not path.exists():
        return (), {}, {}, {"status": "missing", "path": str(path), "scenarios": 0, "hypotheses": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            (),
            {},
            {},
            {
                "status": "read_error",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "scenarios": 0,
                "hypotheses": 0,
            },
        )
    if not isinstance(payload, dict):
        return (), {}, {}, {"status": "invalid", "path": str(path), "reason": "payload_not_object", "scenarios": 0}
    if not payload.get("ok") or not payload.get("research_only") or payload.get("executable"):
        return (
            (),
            {},
            {},
            {
                "status": "ignored",
                "path": str(path),
                "reason": "mutation_batch_not_research_only_or_not_ok",
                "scenarios": 0,
                "hypotheses": 0,
            },
        )
    metadata_by_id = {
        str(item.get("id")): item
        for item in payload.get("mutation_metadata") or []
        if isinstance(item, dict) and item.get("id")
    }
    grouped: dict[tuple[str, str, str, str, str, bool], list[Hypothesis]] = {}
    skipped = 0
    skipped_errors: list[str] = []
    for item in payload.get("hypotheses") or []:
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            hyp = Hypothesis.from_dict(item)
        except Exception as exc:
            skipped += 1
            if len(skipped_errors) < 5:
                skipped_errors.append(f"{type(exc).__name__}: {exc}")
            continue
        metadata = metadata_by_id.get(hyp.id, {})
        product = str(metadata.get("product") or ("btc_accumulation" if hyp.direction == "short" else "active_income"))
        market = str(metadata.get("market") or ("spot" if product == "btc_accumulation" else "futures"))
        validation_scope = metadata.get("validation_scope") if isinstance(metadata.get("validation_scope"), dict) else {}
        pnl_unit = str(validation_scope.get("pnl_unit") or ("btc" if product == "btc_accumulation" else "usdt"))
        position = product == "btc_accumulation" or pnl_unit == "btc"
        opportunity = str(metadata.get("opportunity_type") or "mutation_research")
        grouped.setdefault((product, market, pnl_unit, hyp.base_timeframe, opportunity, position), []).append(hyp)

    scenarios: list[ResearchScenario] = []
    hypotheses_by_scenario: dict[str, list[Hypothesis]] = {}
    metadata_by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for product, market, pnl_unit, base_tf, opportunity, position in sorted(grouped):
        name = f"mutation_{product}_{market}_{base_tf}_{opportunity}"
        scenarios.append(
            ResearchScenario(
                name=name,
                product=product,
                base_tf=base_tf,
                pnl_unit=pnl_unit,
                market=market,
                position=position,
                start=_mutation_start(product, base_tf),
                opportunity_type=opportunity,
                candidate_set="mutation",
                max_hypotheses=max_hypotheses_per_scenario,
            )
        )
        hypotheses = sorted(grouped[(product, market, pnl_unit, base_tf, opportunity, position)], key=lambda h: h.id)
        hypotheses_by_scenario[name] = hypotheses
        metadata_by_scenario[name] = {
            hyp.id: metadata_by_id[hyp.id]
            for hyp in hypotheses
            if hyp.id in metadata_by_id
        }
    summary = {
        "status": "loaded",
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "scenarios": len(scenarios),
        "hypotheses": sum(len(items) for items in hypotheses_by_scenario.values()),
        "skipped": skipped,
        "research_only": payload.get("research_only"),
        "executable": payload.get("executable"),
    }
    if skipped_errors:
        summary["skipped_errors"] = skipped_errors
    return (
        tuple(scenarios),
        hypotheses_by_scenario,
        metadata_by_scenario,
        summary,
    )


def _validation_config(scenario: ResearchScenario, *, n_trials: int = 1) -> ValidationConfig:
    if scenario.position:
        cfg = ValidationConfig(min_trades_train=15, min_trades_val=5, min_trades_holdout=3)
    else:
        cfg = ValidationConfig(min_trades_train=30, min_trades_val=10, min_trades_holdout=5)
    return dataclasses.replace(cfg, n_trials=max(1, int(n_trials)))


def _trial_count_for_selection(
    selection: dict[str, Any] | None,
    *,
    selected_hypotheses: int,
    supported_hypotheses: int,
) -> int:
    counts = [selected_hypotheses, supported_hypotheses]
    if isinstance(selection, dict):
        counts.extend(
            _int_count(selection.get(key))
            for key in ("available", "selected")
        )
    return max(1, *counts)


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str(result.get("verdict", "unknown")) for result in results)
    reasons = Counter(
        reason
        for result in results
        for reason in result.get("reasons", [])
    )
    keepers = [
        result
        for result in results
        if result.get("verdict") == "keep"
    ]
    return {
        "hypotheses": len(results),
        "keepers": len(keepers),
        "keeper_ids": [str(result.get("hypothesis_id")) for result in keepers],
        "verdicts": dict(sorted(verdicts.items())),
        "top_reasons": dict(reasons.most_common(8)),
    }


def _segment_summary(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    segment = result.get(name)
    if not isinstance(segment, dict):
        return None
    def finite_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    return {
        "trades": _int_count(segment.get("trades")),
        "total_return": float(segment.get("total_return") or 0.0),
        "win_rate": finite_or_none(segment.get("win_rate")),
        "sharpe": finite_or_none(segment.get("sharpe")),
    }


def _candidate_stage(result: dict[str, Any]) -> str:
    if result.get("holdout"):
        return "holdout"
    if result.get("sensitivity"):
        return "sensitivity"
    if result.get("oos"):
        return "oos"
    if result.get("validation"):
        return "validation"
    if result.get("train"):
        return "train"
    return "untested"


def _candidate_next_step(result: dict[str, Any]) -> str:
    reasons = {str(reason) for reason in result.get("reasons", [])}
    if "insufficient_train_trades" in reasons:
        return "collect_more_data_or_relax_entry_frequency"
    if "insufficient_validation_trades" in reasons or "insufficient_holdout_trades" in reasons:
        return "keep_in_rotation_for_more_samples"
    if "no_train_edge" in reasons:
        return "discard_or_mutate_entry_logic"
    if "failed_validation" in reasons:
        return "mutate_thresholds_before_retest"
    if "unstable_across_windows" in reasons:
        return "require_regime_filter_or_discard"
    if "parameter_fragile" in reasons:
        return "widen_robustness_or_discard"
    if "failed_holdout" in reasons:
        return "discard_until_new_market_regime"
    return "continue_bounded_rotation"


def _incubation_score(result: dict[str, Any]) -> float:
    # Ranking is for research attention only. It deliberately does not affect
    # export, paper execution, promotion, or live approval.
    stage_weight = {
        "holdout": 5.0,
        "sensitivity": 4.0,
        "oos": 3.0,
        "validation": 2.0,
        "train": 1.0,
        "untested": 0.0,
    }[_candidate_stage(result)]
    train = _segment_summary(result, "train") or {}
    validation = _segment_summary(result, "validation") or {}
    holdout = _segment_summary(result, "holdout") or {}
    return round(
        stage_weight
        + float(train.get("total_return") or 0.0)
        + 2.0 * float(validation.get("total_return") or 0.0)
        + 3.0 * float(holdout.get("total_return") or 0.0)
        + 0.001 * _int_count(train.get("trades")),
        6,
    )


def _incubation_candidates_from_results(
    results: list[dict[str, Any]],
    *,
    limit: int = MAX_INCUBATION_CANDIDATES_PER_SCENARIO,
    hypothesis_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    hypothesis_metadata = hypothesis_metadata or {}
    for result in results:
        if result.get("verdict") == "keep":
            continue
        train = _segment_summary(result, "train")
        if not train:
            continue
        hypothesis_id = str(result.get("hypothesis_id"))
        metadata = hypothesis_metadata.get(hypothesis_id, {})
        lineage = {
            key: metadata.get(key)
            for key in ("source_candidate_id", "source_scenario", "validation_scope")
            if metadata.get(key) is not None
        }
        if metadata.get("reason") is not None:
            lineage["mutation_reason"] = metadata.get("reason")
        candidates.append(
            {
                "id": hypothesis_id,
                "family": result.get("family"),
                "direction": result.get("direction"),
                "verdict": result.get("verdict"),
                "reasons": list(result.get("reasons") or []),
                "stage_reached": _candidate_stage(result),
                "next_step": _candidate_next_step(result),
                "score": _incubation_score(result),
                **({"mutation_lineage": lineage} if lineage else {}),
                "train": train,
                **(
                    {"validation": validation}
                    if (validation := _segment_summary(result, "validation"))
                    else {}
                ),
                **(
                    {"holdout": holdout}
                    if (holdout := _segment_summary(result, "holdout"))
                    else {}
                ),
                **(
                    {"oos_pass_rate": result["oos"]["pass_rate"]}
                    if isinstance(result.get("oos"), dict)
                    and result["oos"].get("pass_rate") is not None
                    else {}
                ),
                **(
                    {"sensitivity_pass_fraction": result["sensitivity"]["pass_fraction"]}
                    if isinstance(result.get("sensitivity"), dict)
                    and result["sensitivity"].get("pass_fraction") is not None
                    else {}
                ),
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def _int_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _top_count_key(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return str(max(counts.items(), key=lambda item: _int_count(item[1]))[0])


def _cycle_next_actions(summary: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    top_reasons = summary.get("top_reasons") or {}
    mutation_effectiveness = summary.get("mutation_effectiveness") or {}
    if summary.get("scenario_errors"):
        actions.append("repair failing research scenarios before trusting exports")
    if summary.get("unsupported_hypotheses"):
        actions.append("repair indicator coverage for unsupported hypotheses")
    if (
        mutation_effectiveness.get("evaluated_hypotheses")
        and not mutation_effectiveness.get("keepers")
        and mutation_effectiveness.get("top_reasons")
    ):
        mutation_reason = _top_count_key(mutation_effectiveness.get("top_reasons") or {})
        actions.append(f"mutation batch found no keepers; top mutation reason {mutation_reason}")
    if summary.get("hypotheses") and not summary.get("keepers"):
        if top_reasons.get("no_train_edge"):
            actions.append("continue rotating curated candidates; no positive train edge found yet")
        elif top_reasons.get("failed_validation"):
            actions.append("keep validation gate tight; candidates are losing validation")
        elif top_reasons.get("failed_holdout"):
            actions.append("keep holdout gate tight; candidates are not surviving unseen data")
        elif (
            top_reasons.get("insufficient_train_trades")
            or top_reasons.get("insufficient_validation_trades")
            or top_reasons.get("insufficient_holdout_trades")
        ):
            actions.append("favor higher-frequency or longer-window candidates to improve sample size")
        else:
            actions.append("continue bounded search; no exportable keeper found yet")
    if summary.get("keepers") and not summary.get("exported"):
        export_reasons = summary.get("export_reasons") if isinstance(summary.get("export_reasons"), dict) else {}
        if export_reasons.get("open_positions_block_export"):
            actions.append("wait for open positions to close before replacing the active paper artifact")
        else:
            actions.append("inspect kept candidates blocked during export policy checks")
    if not actions and summary.get("exported"):
        actions.append("review exported paper candidates; live still requires explicit approval")
    return actions[:4]


def _mutation_effectiveness_summary(
    scenario_reports: list[dict[str, Any]],
    mutation_batch_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    mutation_scenarios = [
        scenario
        for scenario in scenario_reports
        if scenario.get("candidate_set") == "mutation"
        or str(scenario.get("name") or "").startswith("mutation_")
    ]
    if mutation_batch_summary is None and not mutation_scenarios:
        return None

    verdicts: Counter[str] = Counter()
    top_reasons: Counter[str] = Counter()
    by_product: Counter[str] = Counter()
    evaluated_hypotheses = 0
    keepers = 0
    incubation_candidates = 0
    skipped_scenarios = 0
    scenario_errors = 0
    for scenario in mutation_scenarios:
        product = str(scenario.get("product") or "unknown")
        by_product[product] += 1
        if scenario.get("skipped"):
            skipped_scenarios += 1
        if not scenario.get("ok"):
            scenario_errors += 1
        evaluated_hypotheses += _int_count(scenario.get("hypotheses"))
        keepers += _int_count(scenario.get("keepers"))
        incubation_candidates += len(scenario.get("incubation_candidates") or [])
        for verdict, count in (scenario.get("verdicts") or {}).items():
            verdicts[str(verdict)] += _int_count(count)
        for reason, count in (scenario.get("top_reasons") or {}).items():
            top_reasons[str(reason)] += _int_count(count)

    status = None
    batch_hypotheses = None
    batch_scenarios = None
    generated_at = None
    if isinstance(mutation_batch_summary, dict):
        status = mutation_batch_summary.get("status")
        generated_at = mutation_batch_summary.get("generated_at")
        batch_hypotheses = mutation_batch_summary.get("hypotheses")
        batch_scenarios = mutation_batch_summary.get("scenarios")
    if status is None:
        status = "evaluated" if mutation_scenarios else "not_loaded"

    if scenario_errors:
        outcome = "validation_errors"
    elif keepers:
        outcome = "keeper_found"
    elif evaluated_hypotheses:
        outcome = "no_keeper"
    elif status == "loaded":
        outcome = "no_supported_mutations"
    else:
        outcome = str(status)

    return {
        "status": status,
        "generated_at": generated_at,
        "batch_hypotheses": batch_hypotheses,
        "batch_scenarios": batch_scenarios,
        "evaluated_scenarios": len(mutation_scenarios),
        "evaluated_hypotheses": evaluated_hypotheses,
        "keepers": keepers,
        "incubation_candidates": incubation_candidates,
        "skipped_scenarios": skipped_scenarios,
        "scenario_errors": scenario_errors,
        "by_product": dict(sorted(by_product.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "top_reasons": dict(top_reasons.most_common(8)),
        "outcome": outcome,
    }


def _summarize_cycle(
    scenario_reports: list[dict[str, Any]],
    export_reports: list[dict[str, Any]],
    *,
    mutation_batch_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    top_reasons: Counter[str] = Counter()
    opportunities: Counter[str] = Counter()
    opportunities_by_product: dict[str, Counter[str]] = {}
    hypotheses = 0
    selected = 0
    available = 0
    keepers = 0
    incubation_candidates = 0
    unsupported = 0
    scenario_errors = 0

    for scenario in scenario_reports:
        if not scenario.get("ok"):
            scenario_errors += 1
            top_reasons["scenario_error"] += 1
        product = str(scenario.get("product", "unknown"))
        opportunity = str(scenario.get("opportunity_type", "unknown"))
        opportunities[opportunity] += 1
        opportunities_by_product.setdefault(product, Counter())[opportunity] += 1
        hypotheses += _int_count(scenario.get("hypotheses"))
        keepers += _int_count(scenario.get("keepers"))
        incubation_candidates += len(scenario.get("incubation_candidates") or [])
        unsupported += len(scenario.get("unsupported_hypotheses") or [])
        selection = scenario.get("selection") or {}
        if isinstance(selection, dict):
            selected += _int_count(selection.get("selected"))
            available += _int_count(selection.get("available"))
        for verdict, count in (scenario.get("verdicts") or {}).items():
            verdicts[str(verdict)] += _int_count(count)
        for reason, count in (scenario.get("top_reasons") or {}).items():
            top_reasons[str(reason)] += _int_count(count)

    exported = sum(1 for item in export_reports if item.get("exported"))
    export_failures = sum(1 for item in export_reports if not item.get("ok"))
    no_exportable = sum(1 for item in export_reports if item.get("reason") == "no_exportable_strategies")
    export_reasons = Counter(
        str(item.get("reason"))
        for item in export_reports
        if not item.get("exported") and item.get("reason")
    )
    summary = {
        "scenarios": len(scenario_reports),
        "opportunity_types": dict(sorted(opportunities.items())),
        "opportunity_types_by_product": {
            product: dict(sorted(counts.items()))
            for product, counts in sorted(opportunities_by_product.items())
        },
        "scenario_errors": scenario_errors,
        "hypotheses": hypotheses,
        "selected_hypotheses": selected,
        "available_hypotheses": available,
        "keepers": keepers,
        "incubation_candidates": incubation_candidates,
        "unsupported_hypotheses": unsupported,
        "verdicts": dict(sorted(verdicts.items())),
        "top_reasons": dict(top_reasons.most_common(8)),
        "exports": len(export_reports),
        "exported": exported,
        "export_failures": export_failures,
        "no_exportable": no_exportable,
        "export_reasons": dict(export_reasons.most_common()),
    }
    mutation_effectiveness = _mutation_effectiveness_summary(
        scenario_reports,
        mutation_batch_summary,
    )
    if mutation_effectiveness is not None:
        summary["mutation_effectiveness"] = mutation_effectiveness
    summary["next_actions"] = _cycle_next_actions(summary)
    return summary


def build_incubation_review(
    scenario_reports: list[dict[str, Any]],
    *,
    generated_at: str,
    limit_per_product: int = 12,
) -> dict[str, Any]:
    """Build a durable research queue without making weak candidates executable."""
    by_product: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenario_reports:
        if scenario.get("skipped") or not scenario.get("ok"):
            continue
        product = str(scenario.get("product", "unknown"))
        for candidate in scenario.get("incubation_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            item = dict(candidate)
            item.update(
                {
                    "scenario": scenario.get("name"),
                    "product": product,
                    "market": scenario.get("market"),
                    "pnl_unit": scenario.get("pnl_unit"),
                    "opportunity_type": scenario.get("opportunity_type", "research"),
                    "base_tf": scenario.get("base_tf"),
                }
            )
            by_product.setdefault(product, []).append(item)

    selected_by_product: dict[str, list[dict[str, Any]]] = {}
    for product, candidates in sorted(by_product.items()):
        selected_by_product[product] = sorted(
            candidates,
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )[:limit_per_product]

    total = sum(len(items) for items in selected_by_product.values())
    return {
        "ok": True,
        "generated_at": generated_at,
        "schema": "autopilot.incubation_candidates/v1",
        "research_only": True,
        "executable": False,
        "paper_trade_allowed": False,
        "live_allowed": False,
        "promotion_eligible": False,
        "reason": "non_keeper_research_attention_queue",
        "summary": {
            "candidates": total,
            "by_product": {
                product: len(items)
                for product, items in selected_by_product.items()
            },
        },
        "products": selected_by_product,
    }


def _missing_columns_for_hypothesis(hypothesis, indicator_dir: Path) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for timeframe, columns in _needed_columns([hypothesis]).items():
        path = indicator_dir / f"BTCUSDT_{timeframe}_all_indicators.parquet"
        if not path.exists():
            missing[timeframe] = sorted(columns)
            continue
        schema = pq.ParquetFile(path).schema_arrow
        available = set(schema.names)
        absent = sorted(column for column in columns if column not in available)
        if absent:
            missing[timeframe] = absent
    return missing


def _partition_supported_hypotheses(
    hypotheses: list[Any],
    *,
    indicator_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    supported = []
    unsupported = []
    for hypothesis in hypotheses:
        missing = _missing_columns_for_hypothesis(hypothesis, indicator_dir)
        if missing:
            unsupported.append(
                {
                    "id": hypothesis.id,
                    "missing_columns": missing,
                }
            )
        else:
            supported.append(hypothesis)
    return supported, unsupported


def run_validation_scenario(
    scenario: ResearchScenario,
    *,
    hypotheses: list[Any] | None = None,
    selection: dict[str, Any] | None = None,
    hypothesis_metadata: dict[str, dict[str, Any]] | None = None,
    log_path: Path = DEFAULT_LOG,
) -> dict[str, Any]:
    hypotheses = _hypotheses_for(scenario) if hypotheses is None else hypotheses
    if not hypotheses:
        raise ValueError(f"{scenario.name}: no hypotheses for base timeframe {scenario.base_tf}")
    selected_indicator_dir = indicator_data_dir("BTCUSDT", scenario.market, legacy_fallback=True)
    supported_hypotheses, unsupported_hypotheses = _partition_supported_hypotheses(
        hypotheses,
        indicator_dir=selected_indicator_dir,
    )
    if not supported_hypotheses:
        return {
            "ok": True,
            "skipped": True,
            "reason": "unsupported_features",
            "name": scenario.name,
            "product": scenario.product,
            "base_tf": scenario.base_tf,
            "pnl_unit": scenario.pnl_unit,
            "market": scenario.market,
            "position": scenario.position,
            "opportunity_type": scenario.opportunity_type,
            "with_guards": scenario.with_guards,
            "candidate_set": scenario.candidate_set,
            "start": scenario.start,
            "end": scenario.end,
            "rows": 0,
            "hypotheses": 0,
            "keepers": 0,
            "selection": selection,
            "unsupported_hypotheses": unsupported_hypotheses,
            "verdicts": {},
            "top_reasons": {"unsupported_features": len(unsupported_hypotheses)},
        }
    frame = build_aligned_frame(
        supported_hypotheses,
        base_tf=scenario.base_tf,
        start=scenario.start,
        end=scenario.end,
        indicator_dir=selected_indicator_dir,
    )
    eval_cfg = EvalConfig(pnl_unit=scenario.pnl_unit, market=scenario.market)
    validation_cfg = _validation_config(
        scenario,
        n_trials=_trial_count_for_selection(
            selection,
            selected_hypotheses=len(hypotheses),
            supported_hypotheses=len(supported_hypotheses),
        ),
    )
    results = validate_batch(
        frame,
        supported_hypotheses,
        validation_cfg,
        eval_cfg=eval_cfg,
        log_path=log_path,
    )
    incubation_candidates = _incubation_candidates_from_results(
        results,
        hypothesis_metadata=hypothesis_metadata,
    )
    return {
        "ok": True,
        "name": scenario.name,
        "product": scenario.product,
        "base_tf": scenario.base_tf,
        "pnl_unit": scenario.pnl_unit,
        "market": scenario.market,
        "position": scenario.position,
        "opportunity_type": scenario.opportunity_type,
        "with_guards": scenario.with_guards,
        "candidate_set": scenario.candidate_set,
        "start": scenario.start,
        "end": scenario.end,
        "rows": int(len(frame)),
        "unsupported_hypotheses": unsupported_hypotheses,
        "trial_count": validation_cfg.n_trials,
        "selection": selection or {
            "available": len(supported_hypotheses),
            "selected": len(supported_hypotheses),
            "offset": 0,
            "next_offset": 0,
            "ids": [hyp.id for hyp in supported_hypotheses],
        },
        "incubation_candidates": incubation_candidates,
        **_summarize_results(results),
    }


def export_product(
    product: str,
    *,
    pnl_unit: str,
    market: str,
    out: Path,
    top_k: int,
    ids: list[str] | None = None,
    min_dsr: float | None = None,
    log_path: Path = DEFAULT_LOG,
    state_file: Path | None = None,
) -> dict[str, Any]:
    open_position_ids = _open_position_ids_for_export(product, state_file=state_file)
    if open_position_ids:
        return {
            "ok": True,
            "product": product,
            "pnl_unit": pnl_unit,
            "market": market,
            "exported": False,
            "reason": "open_positions_block_export",
            "detail": "active strategy artifact is left unchanged while positions are open",
            "artifact": str(out),
            "ids": ids or [],
            "open_positions": open_position_ids,
            "min_dsr": min_dsr,
        }
    try:
        path = export_strategies(
            log_path=log_path,
            output_path=out,
            top_k=top_k,
            pnl_unit=pnl_unit,
            market=market,
            ids=ids,
            min_dsr=min_dsr,
        )
    except ValueError as exc:
        if "No exportable strategies" not in str(exc):
            raise
        return {
            "ok": True,
            "product": product,
            "pnl_unit": pnl_unit,
            "market": market,
            "exported": False,
            "reason": "no_exportable_strategies",
            "detail": str(exc),
            "artifact": str(out),
            "ids": ids or [],
            "min_dsr": min_dsr,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "product": product,
        "pnl_unit": pnl_unit,
        "market": market,
        "exported": True,
        "artifact": str(path),
        "strategies": len(payload.get("strategies", [])),
        "ids": ids or [],
        "min_dsr": min_dsr,
    }


def _open_position_ids_for_export(product: str, *, state_file: Path | None = None) -> list[str]:
    path = state_file if state_file is not None else DEFAULT_PRODUCT_STATE_FILES.get(product)
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{product}: state file must contain a JSON object: {path}")
    positions = payload.get("open_positions", {})
    if positions is None:
        return []
    if not isinstance(positions, dict):
        raise ValueError(f"{product}: state open_positions must be an object: {path}")
    return sorted(str(strategy_id) for strategy_id in positions if strategy_id)


def _current_keeper_ids(
    scenario_reports: list[dict[str, Any]],
    *,
    product: str,
    market: str,
) -> list[str]:
    ids: list[str] = []
    for scenario in scenario_reports:
        if scenario.get("product") != product or scenario.get("market") != market:
            continue
        if scenario.get("skipped") or not scenario.get("ok"):
            continue
        ids.extend(str(item) for item in (scenario.get("keeper_ids") or []) if item)
    return sorted(set(ids))


def run_research_cycle(
    *,
    state_path: Path = DEFAULT_STATE,
    output_path: Path | None = None,
    incubation_output_path: Path | None = None,
    log_path: Path = DEFAULT_LOG,
    scenarios: tuple[ResearchScenario, ...] = DEFAULT_SCENARIOS,
    force: bool = False,
    include_mutations: bool = False,
    mutation_batch_path: Path = DEFAULT_MUTATION_BATCH,
) -> dict[str, Any]:
    mutation_scenarios: tuple[ResearchScenario, ...] = ()
    mutation_hypotheses: dict[str, list[Hypothesis]] = {}
    mutation_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    mutation_batch_summary: dict[str, Any] | None = None
    if include_mutations:
        (
            mutation_scenarios,
            mutation_hypotheses,
            mutation_metadata,
            mutation_batch_summary,
        ) = _load_mutation_scenarios(mutation_batch_path)
    scenarios = (*scenarios, *mutation_scenarios)
    scenario_markets = sorted({scenario.market for scenario in scenarios}) or ["futures"]
    market_data_by_market = build_market_data_statuses(scenario_markets)
    ready_markets = {
        market
        for market, status in market_data_by_market.items()
        if status.get("ok")
    }
    marker = {
        market: status.get("last_timestamp")
        for market, status in market_data_by_market.items()
        if status.get("ok")
    }
    market_marker = _market_data_skip_marker(market_data_by_market)
    report: dict[str, Any] = {
        "ok": False,
        "generated_at": utc_now(),
        "market_data": {
            "ok": all(status.get("ok") for status in market_data_by_market.values()),
            "markets": market_data_by_market,
        },
        "mutation_batch": mutation_batch_summary,
        "scenarios": [],
        "exports": [],
        "skipped": False,
    }
    if not ready_markets:
        report.update(error="market_data_not_ready")
        if output_path:
            write_json_atomic(output_path, report)
        return report

    last_timestamp = json.dumps(marker, sort_keys=True)
    mutation_marker = None
    if mutation_batch_summary is not None:
        mutation_marker = json.dumps(
            {
                "status": mutation_batch_summary.get("status"),
                "generated_at": mutation_batch_summary.get("generated_at"),
                "hypotheses": mutation_batch_summary.get("hypotheses", 0),
                "scenarios": mutation_batch_summary.get("scenarios", 0),
            },
            sort_keys=True,
        )
    state = _load_state(state_path)
    state_recovered = bool(state.get("_state_recovered"))
    if state_recovered:
        report["state_recovered"] = True
        report["state_error"] = state.get("_state_error")
    if (
        not force
        and state.get("last_market_marker") == market_marker
        and state.get("last_mutation_batch_marker") == mutation_marker
    ):
        report.update(
            ok=True,
            skipped=True,
            reason="market_data_unchanged",
            last_market_timestamp=last_timestamp,
            last_market_marker=market_marker,
            last_mutation_batch_marker=mutation_marker,
        )
        if output_path:
            write_json_atomic(output_path, report)
        return report

    scenario_reports: list[dict[str, Any]] = []
    next_offsets: dict[str, int] = {}
    for scenario in scenarios:
        try:
            if scenario.market not in ready_markets:
                scenario_reports.append(
                    {
                        "ok": True,
                        "skipped": True,
                        "reason": "market_data_not_ready",
                        "name": scenario.name,
                        "product": scenario.product,
                        "base_tf": scenario.base_tf,
                        "pnl_unit": scenario.pnl_unit,
                        "market": scenario.market,
                        "position": scenario.position,
                        "opportunity_type": scenario.opportunity_type,
                        "with_guards": scenario.with_guards,
                        "candidate_set": scenario.candidate_set,
                        "start": scenario.start,
                        "end": scenario.end,
                        "rows": 0,
                        "hypotheses": 0,
                        "keepers": 0,
                        "selection": None,
                        "unsupported_hypotheses": [],
                        "verdicts": {},
                        "top_reasons": {"market_data_not_ready": 1},
                    }
                )
                continue
            if scenario.name in mutation_hypotheses:
                hypotheses, selection = _select_from_hypotheses(
                    scenario,
                    mutation_hypotheses[scenario.name],
                    state,
                )
            else:
                hypotheses, selection = _select_hypotheses(scenario, state)
            validation_kwargs: dict[str, Any] = {
                "hypotheses": hypotheses,
                "selection": selection,
                "log_path": log_path,
            }
            if mutation_metadata.get(scenario.name):
                validation_kwargs["hypothesis_metadata"] = mutation_metadata[scenario.name]
            scenario_reports.append(run_validation_scenario(scenario, **validation_kwargs))
            next_offsets[scenario.name] = int(selection.get("next_offset", 0))
        except Exception as exc:
            scenario_reports.append(
                {
                    "ok": False,
                    "name": scenario.name,
                    "product": scenario.product,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report["scenarios"] = scenario_reports

    export_reports: list[dict[str, Any]] = []
    if all(bool(item.get("ok")) for item in scenario_reports):
        for product, export_cfg in DEFAULT_EXPORTS.items():
            product_market = str(export_cfg["market"])
            keeper_ids = _current_keeper_ids(
                scenario_reports,
                product=product,
                market=product_market,
            )
            if not keeper_ids:
                min_dsr = (
                    float(export_cfg["min_dsr"])
                    if export_cfg.get("min_dsr") is not None
                    else None
                )
                export_reports.append(
                    {
                        "ok": True,
                        "product": product,
                        "pnl_unit": str(export_cfg["pnl_unit"]),
                        "market": product_market,
                        "exported": False,
                        "reason": "no_current_cycle_keepers",
                        "artifact": str(export_cfg["out"]),
                        "ids": [],
                        "min_dsr": min_dsr,
                    }
                )
                continue
            try:
                export_reports.append(
                    export_product(
                        product,
                        pnl_unit=str(export_cfg["pnl_unit"]),
                        market=product_market,
                        out=Path(export_cfg["out"]),
                        top_k=int(export_cfg["top_k"]),
                        ids=keeper_ids,
                        min_dsr=(
                            float(export_cfg["min_dsr"])
                            if export_cfg.get("min_dsr") is not None
                            else None
                        ),
                        log_path=log_path,
                    )
                )
            except Exception as exc:
                export_reports.append(
                    {
                        "ok": False,
                        "product": product,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    report["exports"] = export_reports
    incubation_review = build_incubation_review(
        scenario_reports,
        generated_at=str(report["generated_at"]),
    )
    if incubation_output_path:
        write_json_atomic(incubation_output_path, incubation_review)
        report["incubation_review"] = {
            "path": str(incubation_output_path),
            "schema": incubation_review["schema"],
            "research_only": True,
            "executable": False,
            "paper_trade_allowed": False,
            "live_allowed": False,
            "promotion_eligible": False,
            "candidates": incubation_review["summary"]["candidates"],
            "by_product": incubation_review["summary"]["by_product"],
        }
    report["summary"] = _summarize_cycle(
        scenario_reports,
        export_reports,
        mutation_batch_summary=mutation_batch_summary,
    )
    report["ok"] = all(bool(item.get("ok")) for item in scenario_reports + export_reports)
    report["last_market_timestamp"] = last_timestamp
    report["last_market_marker"] = market_marker
    report["last_mutation_batch_marker"] = mutation_marker

    if report["ok"]:
        write_json_atomic(
            state_path,
            {
                "version": 1,
                "last_market_timestamp": last_timestamp,
                "last_market_marker": market_marker,
                "last_mutation_batch_marker": mutation_marker,
                "last_run_at": report["generated_at"],
                "scenario_offsets": next_offsets,
            },
        )
    if output_path:
        write_json_atomic(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded real-data research and gated export.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--force", action="store_true", help="Run even if market data timestamp is unchanged.")
    parser.add_argument(
        "--include-mutations",
        action="store_true",
        help="Validate research-only mutation hypotheses from --mutation-batch.",
    )
    parser.add_argument("--mutation-batch", type=Path, default=DEFAULT_MUTATION_BATCH)
    parser.add_argument(
        "--incubation-output",
        type=Path,
        default=DEFAULT_INCUBATION_OUTPUT,
        help="Write the non-executable research attention queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_research_cycle(
        state_path=args.state,
        output_path=args.output,
        log_path=args.log,
        force=args.force,
        include_mutations=args.include_mutations,
        mutation_batch_path=args.mutation_batch,
        incubation_output_path=args.incubation_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
