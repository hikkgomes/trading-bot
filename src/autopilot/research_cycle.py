"""Bounded real-data evaluation coordinator for autonomous research.

The scheduled path consumes behaviorally unique candidates from the persistent
strategy factory, checks real-history and feature contracts, records
development evidence, and durably claims a lineage's protected holdout before
it can be read.  Qualified outputs may enter isolated paper incubation, but no
candidate can become an active live strategy through this module.

Legacy curated and mutation-batch loaders remain available for old reports and
tests; production configuration uses ``--generated-only``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from research_exploration.dsr import DSR_METHOD, LIVE_MIN_DSR
from research_exploration.evaluate import EvalConfig, _needed_columns, build_aligned_frame
from research_exploration.experiment_log import DEFAULT_LOG
from research_exploration.export import run as export_strategies
from research_exploration.hypothesis_generator import (
    first_smoke_set,
    generate_batch,
    position_trading_set,
    swing_trading_set,
)
from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.strategy_grammar import validate_hypothesis_against_space
from research_exploration.validation import (
    ValidationConfig,
    _segment_bounds,
    split_frame,
    validate_batch,
    with_trial_sharpe_dispersion,
)
from src.autopilot.approvals import artifact_digest, load_artifact
from src.autopilot.candidate_activation import (
    DEFAULT_CANDIDATE_DIR,
    candidate_path_for_product,
    product_identity,
)
from src.autopilot.candidate_paper import candidate_paper_paths
from src.autopilot.config import DEFAULT_CONFIG_PATH, ProductConfig, load_config
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.experiment_memory import (
    EvaluationConflictError,
    ExperimentMemory,
    HoldoutSealBudgetError,
    canonical_strategy_hash,
)
from src.autopilot.io import write_json_atomic
from src.autopilot.market_data import build_market_data_statuses
from src.autopilot.reporting import utc_now
from src.autopilot.research_factory import (
    BATCH_SCHEMA as GENERATED_BATCH_SCHEMA,
)
from src.autopilot.research_factory import (
    DEFAULT_CONFIG as DEFAULT_RESEARCH_FACTORY_CONFIG,
)
from src.autopilot.research_factory import (
    load_factory_config,
    resolve_search_space,
    strategy_behavior_spec,
)
from src.autopilot.research_history_contract import generated_history_contract
from src.autopilot.strategy_policy import assert_loaded_strategy_artifact_allowed
from src.build_dataset import TIMEFRAME_SECONDS
from src.config import indicator_data_dir

DEFAULT_OUTPUT = Path("runtime/research_cycle.json")
DEFAULT_STATE = Path("runtime/research_cycle_state.json")
DEFAULT_MUTATION_BATCH = Path("runtime/mutation_hypotheses.json")
DEFAULT_GENERATED_BATCH = Path("runtime/research/generated_hypotheses.json")
_FILE_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}
_MAX_FILE_DIGEST_CACHE_ITEMS = 64
DEFAULT_INCUBATION_OUTPUT = Path("runtime/incubation_candidates.json")
DEFAULT_PRODUCT_STATE_FILES = {
    "active_income": Path("runtime/active_income_state.json"),
    "btc_accumulation": Path("runtime/btc_accumulation_state.json"),
}
FEATURE_DEPENDENCY_MAX_NATIVE_BARS = 240
# Every sealed protected interval permanently removes its window (plus the
# feature-dependency embargo) from adaptive research. Sealing is therefore
# rate-limited per market+symbol: a candidate that reaches the holdout gate
# sooner is deferred with a durable development outcome instead of consuming
# more chronological history. 7 days matches the paper-soak floor, so at most
# one final-evaluation window can be spent per market per soak period.
HOLDOUT_SEAL_MIN_INTERVAL_SECONDS = 7 * 24 * 3600
# ``before_holdout`` deferrals that already recorded a development outcome;
# the checkpoint must not downgrade that durable record with a second write.
HOLDOUT_GATE_DEFERRAL_REASONS = frozenset(
    {
        "holdout_already_consumed",
        "holdout_seal_budget_exhausted",
        "holdout_cohort_seal_conflict",
    }
)


class UnprotectedResearchEpochUnavailableError(EvaluationConflictError):
    """No adaptive research rows remain outside permanently protected evidence."""


def _protected_epoch_scenario_order(scenario: ResearchScenario) -> tuple[Any, ...]:
    """Give short-history/fast scenarios first choice of the newest epoch."""

    start = pd.Timestamp(scenario.start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    return (
        scenario.market,
        -int(start.value),
        int(TIMEFRAME_SECONDS.get(scenario.base_tf, 10**9)),
        scenario.name,
    )


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
    coverage_earliest: str | None = None
    coverage_max_start_delay_days: float | None = None
    coverage_max_latest_age_hours: float | None = None
    coverage_min_span_days: float | None = None
    coverage_min_rows: int | None = None
    symbol: str = "BTCUSDT"


DEFAULT_SCENARIOS = (
    ResearchScenario(
        name="active_income_1h_swing",
        product="active_income",
        base_tf="1h",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2022-01-01",
        opportunity_type="swing_trading",
        with_guards=True,
        candidate_set="swing",
        max_hypotheses=8,
        coverage_earliest="2022-01-01",
        coverage_max_start_delay_days=2,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=900,
        coverage_min_rows=20_000,
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
        coverage_earliest="2023-01-01",
        coverage_max_start_delay_days=1,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=730,
        coverage_min_rows=200_000,
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
        coverage_earliest="2026-01-01",
        coverage_max_start_delay_days=1,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=90,
        coverage_min_rows=100_000,
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
        coverage_earliest="2020-06-01",
        coverage_max_start_delay_days=2,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=1_000,
        coverage_min_rows=8_000,
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
        coverage_earliest="2020-06-01",
        coverage_max_start_delay_days=2,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=1_000,
        coverage_min_rows=8_000,
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
        coverage_earliest="2020-06-01",
        coverage_max_start_delay_days=2,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=1_000,
        coverage_min_rows=30_000,
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
        coverage_earliest="2020-06-01",
        coverage_max_start_delay_days=2,
        coverage_max_latest_age_hours=24,
        coverage_min_span_days=1_000,
        coverage_min_rows=30_000,
    ),
)

DEFAULT_EXPORTS = {
    "active_income": {
        "pnl_unit": "usdt",
        "market": "futures",
        "top_k": 3,
        "min_dsr": LIVE_MIN_DSR,
    },
    "btc_accumulation": {
        "pnl_unit": "btc",
        "market": "spot",
        "top_k": 3,
        "min_dsr": LIVE_MIN_DSR,
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


def _history_coverage_skip_marker(
    statuses: dict[str, dict[str, Any]],
) -> str | None:
    if not statuses:
        return None
    marker = {
        name: {
            "ok": status.get("ok"),
            "actual": status.get("actual"),
            "failed_checks": status.get("failed_checks"),
            "path": status.get("path"),
            "read_error": status.get("read_error"),
        }
        for name, status in sorted(statuses.items())
    }
    return json.dumps(marker, sort_keys=True)


def _hypotheses_for(scenario: ResearchScenario):
    if scenario.candidate_set == "position" or scenario.position:
        hypotheses = position_trading_set(with_guards=scenario.with_guards)
    elif scenario.candidate_set == "swing":
        hypotheses = swing_trading_set(with_guards=scenario.with_guards)
    elif scenario.candidate_set == "full":
        hypotheses = generate_batch(with_guards=scenario.with_guards)
    elif scenario.candidate_set == "smoke":
        hypotheses = first_smoke_set(with_guards=scenario.with_guards)
    else:
        raise ValueError(f"{scenario.name}: unknown candidate_set {scenario.candidate_set!r}")
    return [hyp for hyp in hypotheses if hyp.base_timeframe == scenario.base_tf]


def _consumed_holdout_ids(
    state: dict[str, Any],
    scenario_name: str,
) -> set[str]:
    registry = state.get("consumed_holdout_ids")
    if not isinstance(registry, dict):
        return set()
    values = registry.get(scenario_name)
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if isinstance(value, str) and value.strip()}


def _consumed_holdout_registry(state: dict[str, Any]) -> dict[str, set[str]]:
    registry = state.get("consumed_holdout_ids")
    if not isinstance(registry, dict):
        return {}
    return {
        str(name): {str(value) for value in values if isinstance(value, str) and value.strip()}
        for name, values in registry.items()
        if isinstance(name, str) and name and isinstance(values, list)
    }


def _serialized_holdout_registry(
    registry: dict[str, set[str]],
) -> dict[str, list[str]]:
    return {name: sorted(ids) for name, ids in sorted(registry.items()) if ids}


def _select_from_hypotheses(
    scenario: ResearchScenario,
    hypotheses: list[Hypothesis],
    state: dict[str, Any],
) -> tuple[list[Hypothesis], dict[str, Any]]:
    total = len(hypotheses)
    if total == 0:
        return [], {
            "available": 0,
            "eligible": 0,
            "consumed_holdout": 0,
            "selected": 0,
            "offset": 0,
            "next_offset": 0,
            "exhausted": False,
        }
    consumed = _consumed_holdout_ids(state, scenario.name)
    relevant_consumed = {hyp.id for hyp in hypotheses if hyp.id in consumed}
    eligible = total - len(relevant_consumed)
    limit = (
        eligible
        if scenario.max_hypotheses is None
        else min(
            int(scenario.max_hypotheses),
            eligible,
        )
    )
    offsets = state.get("scenario_offsets", {})
    offset = int(offsets.get(scenario.name, 0) or 0) % total if isinstance(offsets, dict) else 0
    indices: list[int] = []
    scanned = 0
    while scanned < total and len(indices) < limit:
        index = (offset + scanned) % total
        scanned += 1
        if hypotheses[index].id in relevant_consumed:
            continue
        indices.append(index)
    selected = [hypotheses[idx] for idx in indices]
    next_offset = (offset + scanned) % total
    return selected, {
        "available": total,
        "eligible": eligible,
        "consumed_holdout": len(relevant_consumed),
        "selected": len(selected),
        "offset": offset,
        "next_offset": next_offset,
        "wrapped": offset + scanned > total,
        "exhausted": eligible == 0,
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
) -> tuple[
    tuple[ResearchScenario, ...],
    dict[str, list[Hypothesis]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
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
        return (
            (),
            {},
            {},
            {
                "status": "invalid",
                "path": str(path),
                "reason": "payload_not_object",
                "scenarios": 0,
            },
        )
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
        product = str(
            metadata.get("product")
            or ("btc_accumulation" if hyp.direction == "short" else "active_income")
        )
        market = str(
            metadata.get("market") or ("spot" if product == "btc_accumulation" else "futures")
        )
        validation_scope = (
            metadata.get("validation_scope")
            if isinstance(metadata.get("validation_scope"), dict)
            else {}
        )
        pnl_unit = str(
            validation_scope.get("pnl_unit") or ("btc" if product == "btc_accumulation" else "usdt")
        )
        position = product == "btc_accumulation" or pnl_unit == "btc"
        opportunity = str(metadata.get("opportunity_type") or "mutation_research")
        grouped.setdefault(
            (product, market, pnl_unit, hyp.base_timeframe, opportunity, position), []
        ).append(hyp)

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
        hypotheses = sorted(
            grouped[(product, market, pnl_unit, base_tf, opportunity, position)], key=lambda h: h.id
        )
        hypotheses_by_scenario[name] = hypotheses
        metadata_by_scenario[name] = {
            hyp.id: metadata_by_id[hyp.id] for hyp in hypotheses if hyp.id in metadata_by_id
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


def _load_generated_scenarios(
    path: Path,
    *,
    factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> tuple[
    tuple[ResearchScenario, ...],
    dict[str, list[Hypothesis]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    if not path.exists():
        return (
            (),
            {},
            {},
            {
                "status": "missing",
                "path": str(path),
                "scenarios": 0,
                "hypotheses": 0,
            },
        )
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
        return (
            (),
            {},
            {},
            {
                "status": "invalid",
                "path": str(path),
                "reason": "payload_not_object",
                "scenarios": 0,
                "hypotheses": 0,
            },
        )
    safety_ok = (
        payload.get("ok") is True
        and payload.get("schema") == GENERATED_BATCH_SCHEMA
        and payload.get("research_only") is True
        and payload.get("executable") is False
        and payload.get("paper_trade_allowed") is False
        and payload.get("promotion_allowed") is False
        and payload.get("live_allowed") is False
    )
    if not safety_ok:
        return (
            (),
            {},
            {},
            {
                "status": "ignored",
                "path": str(path),
                "reason": "generated_batch_failed_safety_contract",
                "scenarios": 0,
                "hypotheses": 0,
            },
        )
    factory_config = load_factory_config(factory_config_path)
    metadata_by_id = {
        str(item.get("id")): item
        for item in payload.get("generation_metadata") or []
        if isinstance(item, dict) and item.get("id")
    }
    grouped: dict[str, list[Hypothesis]] = {}
    grouped_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    skipped_errors: list[str] = []
    for item in payload.get("hypotheses") or []:
        if not isinstance(item, dict):
            if len(skipped_errors) < 10:
                skipped_errors.append("hypothesis_not_object")
            continue
        try:
            hypothesis = Hypothesis.from_dict(item)
            metadata = metadata_by_id[hypothesis.id]
            space_name = str(metadata["search_space"])
            space = resolve_search_space(factory_config, metadata)
            expected_context = {
                "product": space.product,
                "market": space.market,
                "pnl_unit": space.pnl_unit,
                "opportunity_type": space.opportunity_type,
                "base_timeframe": space.base_timeframe,
                "symbol": space.symbol,
            }
            mismatches = [
                key
                for key, expected in expected_context.items()
                if metadata.get(key, "BTCUSDT" if key == "symbol" else None) != expected
            ]
            if mismatches:
                raise ValueError(f"metadata mismatch: {', '.join(mismatches)}")
            problems = validate_hypothesis_against_space(hypothesis, space)
            if problems:
                raise ValueError(f"grammar contract failed: {', '.join(problems)}")
            expected_hash = canonical_strategy_hash(strategy_behavior_spec(hypothesis, space))
            legacy_hash = canonical_strategy_hash(
                {
                    key: value
                    for key, value in strategy_behavior_spec(hypothesis, space).items()
                    if key != "_symbol"
                }
            )
            accepted_hashes = {expected_hash}
            if space.symbol == "BTCUSDT":
                accepted_hashes.add(legacy_hash)
            if metadata.get("strategy_hash") not in accepted_hashes:
                raise ValueError("strategy_hash does not match canonical behavior")
        except Exception as exc:
            if len(skipped_errors) < 10:
                skipped_errors.append(f"{type(exc).__name__}: {exc}")
            continue
        grouped.setdefault(space_name, []).append(hypothesis)
        grouped_metadata.setdefault(space_name, {})[hypothesis.id] = metadata

    scenarios: list[ResearchScenario] = []
    hypotheses_by_scenario: dict[str, list[Hypothesis]] = {}
    metadata_by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    cumulative_trials = _int_count((payload.get("summary") or {}).get("cumulative_trials"))
    for space_name, hypotheses in sorted(grouped.items()):
        exemplar_metadata = grouped_metadata[space_name][hypotheses[0].id]
        space = resolve_search_space(factory_config, exemplar_metadata)
        contract = generated_history_contract(space)
        name = f"generated_{space.name}"
        scenarios.append(
            ResearchScenario(
                name=name,
                product=space.product,
                base_tf=space.base_timeframe,
                pnl_unit=space.pnl_unit,
                market=space.market,
                position=space.product == "btc_accumulation",
                opportunity_type=space.opportunity_type,
                candidate_set="generated",
                max_hypotheses=len(hypotheses),
                symbol=space.symbol,
                **contract,
            )
        )
        hypotheses_by_scenario[name] = sorted(hypotheses, key=lambda item: item.id)
        metadata_by_scenario[name] = {
            hypothesis.id: {
                **grouped_metadata[space_name][hypothesis.id],
                "cumulative_trials": cumulative_trials,
            }
            for hypothesis in hypotheses
        }
    factory_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    summary = {
        "status": "loaded",
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "scenarios": len(scenarios),
        "hypotheses": sum(len(items) for items in hypotheses_by_scenario.values()),
        "skipped": len(skipped_errors),
        "skipped_errors": skipped_errors,
        "cumulative_trials": cumulative_trials,
        "new_hypotheses": _int_count(factory_summary.get("new_hypotheses")),
        "resumed_pending": _int_count(factory_summary.get("resumed_pending")),
        "revalidation_pending": _int_count(factory_summary.get("revalidation_pending")),
        "openclaw_proposals_seen": _int_count(factory_summary.get("openclaw_proposals_seen")),
        "research_only": True,
        "executable": False,
    }
    return tuple(scenarios), hypotheses_by_scenario, metadata_by_scenario, summary


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
            _int_count(selection.get(key)) for key in ("available", "selected", "cumulative_trials")
        )
    return max(1, *counts)


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str(result.get("verdict", "unknown")) for result in results)
    reasons = Counter(reason for result in results for reason in result.get("reasons", []))
    keepers = [result for result in results if result.get("verdict") == "keep"]
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
                    {"holdout": holdout} if (holdout := _segment_summary(result, "holdout")) else {}
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
    if summary.get("staged"):
        staged_candidates = summary.get("staged_candidates") or []
        staged_detail = staged_candidates[0] if staged_candidates else {}
        digest = staged_detail.get("artifact_digest") or "<missing-digest>"
        product = staged_detail.get("product") or "<unknown-product>"
        actions.append(
            f"review staged live candidate {product} digest {digest} and activate it "
            "explicitly while paused; approval and fresh live evidence are still required"
        )
    if summary.get("scenario_errors"):
        actions.append("repair failing research scenarios before trusting exports")
    if summary.get("coverage_failures"):
        failed_names = ", ".join(summary.get("coverage_failed_scenarios") or [])
        actions.append(
            "bootstrap the required direct timeframe history"
            + (f" for {failed_names}" if failed_names else "")
            + " before rerunning research"
        )
    if summary.get("unprotected_epoch_deferrals"):
        deferred_names = ", ".join(summary.get("unprotected_epoch_deferred_scenarios") or [])
        actions.append(
            "wait for additional market history to create an unprotected research epoch"
            + (f" for {deferred_names}" if deferred_names else "")
        )
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
            actions.append(
                "let the generator explore new structures; no positive train edge found yet"
                if summary.get("generative_search")
                else "continue rotating curated candidates; no positive train edge found yet"
            )
        elif top_reasons.get("failed_validation"):
            actions.append("keep validation gate tight; candidates are losing validation")
        elif top_reasons.get("failed_holdout"):
            actions.append("keep holdout gate tight; candidates are not surviving unseen data")
        elif (
            top_reasons.get("insufficient_train_trades")
            or top_reasons.get("insufficient_validation_trades")
            or top_reasons.get("insufficient_holdout_trades")
        ):
            actions.append(
                "favor higher-frequency or longer-window candidates to improve sample size"
            )
        else:
            actions.append("continue bounded search; no exportable keeper found yet")
    if summary.get("keepers") and not summary.get("exported"):
        export_reasons = (
            summary.get("export_reasons") if isinstance(summary.get("export_reasons"), dict) else {}
        )
        if export_reasons.get("open_positions_block_export"):
            if summary.get("staging_open_position_blocks"):
                actions.append(
                    "wait for open positions to close before refreshing the staged live candidate"
                )
            if summary.get("active_open_position_blocks"):
                actions.append(
                    "wait for open positions to close before replacing the active paper artifact"
                )
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


def _generated_effectiveness_summary(
    scenario_reports: list[dict[str, Any]],
    generated_batch_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    generated = [
        scenario for scenario in scenario_reports if scenario.get("candidate_set") == "generated"
    ]
    if generated_batch_summary is None and not generated:
        return None
    verdicts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_product: Counter[str] = Counter()
    evaluated = 0
    keepers = 0
    holdout_exposures = 0
    already_evaluated = 0
    for scenario in generated:
        by_product[str(scenario.get("product") or "unknown")] += 1
        evaluated += _int_count(scenario.get("hypotheses"))
        keepers += _int_count(scenario.get("keepers"))
        holdout_exposures += len(scenario.get("holdout_exposed_ids") or [])
        already_evaluated += len(scenario.get("already_evaluated_ids") or [])
        for verdict, count in (scenario.get("verdicts") or {}).items():
            verdicts[str(verdict)] += _int_count(count)
        for reason, count in (scenario.get("top_reasons") or {}).items():
            reasons[str(reason)] += _int_count(count)
    batch = generated_batch_summary or {}
    return {
        "status": batch.get("status") or ("evaluated" if generated else "not_loaded"),
        "generated_at": batch.get("generated_at"),
        "batch_hypotheses": batch.get("hypotheses"),
        "new_hypotheses": _int_count(batch.get("new_hypotheses")),
        "resumed_pending": _int_count(batch.get("resumed_pending")),
        "revalidation_pending": _int_count(batch.get("revalidation_pending")),
        "openclaw_proposals_seen": _int_count(batch.get("openclaw_proposals_seen")),
        "cumulative_trials": _int_count(batch.get("cumulative_trials")),
        "evaluated_scenarios": len(generated),
        "evaluated_hypotheses": evaluated,
        "already_evaluated": already_evaluated,
        "keepers": keepers,
        "holdout_exposures": holdout_exposures,
        "by_product": dict(sorted(by_product.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "top_development_reasons": {
            reason: count for reason, count in reasons.most_common(8) if "holdout" not in reason
        },
    }


def _summarize_cycle(
    scenario_reports: list[dict[str, Any]],
    export_reports: list[dict[str, Any]],
    *,
    mutation_batch_summary: dict[str, Any] | None = None,
    generated_batch_summary: dict[str, Any] | None = None,
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
    coverage_failed_scenarios: list[str] = []
    unprotected_epoch_deferred_scenarios: list[str] = []

    for scenario in scenario_reports:
        if not scenario.get("ok"):
            scenario_errors += 1
            top_reasons["scenario_error"] += 1
        if scenario.get("reason") == "insufficient_history_coverage":
            coverage_failed_scenarios.append(str(scenario.get("name") or "unknown"))
        if scenario.get("reason") == "unprotected_epoch_unavailable":
            unprotected_epoch_deferred_scenarios.append(str(scenario.get("name") or "unknown"))
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
    staged = sum(1 for item in export_reports if item.get("staged"))
    active_exports = sum(
        1 for item in export_reports if item.get("exported") and item.get("destination") == "active"
    )
    export_failures = sum(1 for item in export_reports if not item.get("ok"))
    no_exportable = sum(
        1 for item in export_reports if item.get("reason") == "no_exportable_strategies"
    )
    export_reasons = Counter(
        str(item.get("reason"))
        for item in export_reports
        if not item.get("exported") and item.get("reason")
    )
    staging_open_position_blocks = sum(
        1
        for item in export_reports
        if item.get("reason") == "open_positions_block_export"
        and item.get("destination") == "staging"
    )
    active_open_position_blocks = sum(
        1
        for item in export_reports
        if item.get("reason") == "open_positions_block_export"
        and item.get("destination") == "active"
    )
    staged_candidates = [
        {
            "product": item.get("product"),
            "artifact": item.get("artifact"),
            "artifact_digest": item.get("artifact_digest"),
            "active_artifact": item.get("active_artifact"),
        }
        for item in export_reports
        if item.get("staged")
    ]
    summary = {
        "scenarios": len(scenario_reports),
        "opportunity_types": dict(sorted(opportunities.items())),
        "opportunity_types_by_product": {
            product: dict(sorted(counts.items()))
            for product, counts in sorted(opportunities_by_product.items())
        },
        "scenario_errors": scenario_errors,
        "coverage_failures": len(coverage_failed_scenarios),
        "coverage_failed_scenarios": coverage_failed_scenarios,
        "unprotected_epoch_deferrals": len(unprotected_epoch_deferred_scenarios),
        "unprotected_epoch_deferred_scenarios": unprotected_epoch_deferred_scenarios,
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
        "staged": staged,
        "staged_candidates": staged_candidates,
        "active_exports": active_exports,
        "export_failures": export_failures,
        "no_exportable": no_exportable,
        "export_reasons": dict(export_reasons.most_common()),
        "staging_open_position_blocks": staging_open_position_blocks,
        "active_open_position_blocks": active_open_position_blocks,
    }
    mutation_effectiveness = _mutation_effectiveness_summary(
        scenario_reports,
        mutation_batch_summary,
    )
    if mutation_effectiveness is not None:
        summary["mutation_effectiveness"] = mutation_effectiveness
    generated_effectiveness = _generated_effectiveness_summary(
        scenario_reports,
        generated_batch_summary,
    )
    if generated_effectiveness is not None:
        summary["generative_search"] = generated_effectiveness
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
            "by_product": {product: len(items) for product, items in selected_by_product.items()},
        },
        "products": selected_by_product,
    }


def _missing_columns_for_hypothesis(
    hypothesis,
    indicator_dir: Path,
    *,
    symbol: str,
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for timeframe, columns in _needed_columns([hypothesis]).items():
        path = indicator_dir / f"{symbol}_{timeframe}_all_indicators.parquet"
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
    symbol: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    supported = []
    unsupported = []
    for hypothesis in hypotheses:
        missing = _missing_columns_for_hypothesis(
            hypothesis,
            indicator_dir,
            symbol=symbol,
        )
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


def _retire_unsupported_generated_hypotheses(
    unsupported: list[dict[str, Any]],
    *,
    hypothesis_metadata: dict[str, dict[str, Any]] | None,
    experiment_memory: ExperimentMemory | None,
) -> list[str]:
    """Retire generated work that cannot compile against the real feature inventory.

    Generated candidates are registered before evaluation so a killed process can
    resume them.  Without this terminal transition, a permanently unsupported
    candidate would remain ``pending`` forever and eventually apply backpressure
    to every future factory cycle.  Legacy hypotheses have no strategy hash and
    are deliberately left untouched.
    """

    if experiment_memory is None or not hypothesis_metadata:
        return []
    retired: list[str] = []
    for item in unsupported:
        hypothesis_id = str(item.get("id") or "")
        metadata = hypothesis_metadata.get(hypothesis_id) or {}
        behavior_hash = metadata.get("strategy_hash")
        if not isinstance(behavior_hash, str):
            continue
        missing = item.get("missing_columns")
        detail = json.dumps(missing, sort_keys=True, separators=(",", ":"))
        experiment_memory.retire_strategy(
            behavior_hash,
            reason=f"unsupported_feature_contract:{detail}"[:512],
        )
        retired.append(hypothesis_id)
    return sorted(retired)


def _scenario_coverage_requirements(
    scenario: ResearchScenario,
    *,
    now: str | pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """Return the explicit real-history contract for a scenario.

    Ad-hoc/mutation scenarios can omit the contract.  Every curated default
    scenario sets it explicitly so a recent, shallow seed can never be mistaken
    for the multi-year sample used by the validation gates.
    """
    configured = (
        scenario.coverage_earliest,
        scenario.coverage_max_start_delay_days,
        scenario.coverage_max_latest_age_hours,
        scenario.coverage_min_span_days,
        scenario.coverage_min_rows,
    )
    if all(value is None for value in configured):
        return None
    if any(value is None for value in configured):
        raise ValueError(f"{scenario.name}: incomplete history coverage contract")
    if float(scenario.coverage_max_start_delay_days or 0) < 0:
        raise ValueError(f"{scenario.name}: coverage_max_start_delay_days must be non-negative")
    if float(scenario.coverage_max_latest_age_hours or 0) <= 0:
        raise ValueError(f"{scenario.name}: coverage_max_latest_age_hours must be positive")
    if float(scenario.coverage_min_span_days or 0) <= 0:
        raise ValueError(f"{scenario.name}: coverage_min_span_days must be positive")
    if int(scenario.coverage_min_rows or 0) <= 0:
        raise ValueError(f"{scenario.name}: coverage_min_rows must be positive")

    earliest = pd.Timestamp(str(scenario.coverage_earliest))
    earliest = (
        earliest.tz_localize("UTC") if earliest.tzinfo is None else earliest.tz_convert("UTC")
    )
    reference = pd.Timestamp(now if now is not None else utc_now())
    reference = (
        reference.tz_localize("UTC") if reference.tzinfo is None else reference.tz_convert("UTC")
    )
    if scenario.end:
        end = pd.Timestamp(scenario.end)
        reference = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    latest = reference - pd.Timedelta(hours=float(scenario.coverage_max_latest_age_hours))
    return {
        "earliest_at_or_before": (
            earliest + pd.Timedelta(days=float(scenario.coverage_max_start_delay_days))
        ).isoformat(),
        "latest_at_or_after": latest.isoformat(),
        "minimum_span_days": float(scenario.coverage_min_span_days),
        "minimum_rows": int(scenario.coverage_min_rows),
    }


def _scenario_coverage_status(
    frame: pd.DataFrame,
    scenario: ResearchScenario,
    *,
    now: str | pd.Timestamp | None = None,
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> dict[str, Any] | None:
    requirements = _scenario_coverage_requirements(scenario, now=now)
    if requirements is None:
        return None

    if "timestamp" in frame.columns:
        values = frame["timestamp"]
    elif isinstance(frame.index, pd.DatetimeIndex):
        values = pd.Series(frame.index, index=frame.index)
    else:
        values = pd.Series(dtype="datetime64[ns, UTC]")
    timestamps = pd.to_datetime(values, utc=True, errors="coerce").dropna()
    earliest = timestamps.min() if not timestamps.empty else None
    latest = timestamps.max() if not timestamps.empty else None
    span_days = (
        float((latest - earliest).total_seconds() / 86_400)
        if earliest is not None and latest is not None
        else 0.0
    )
    rows = int(len(frame))
    checks = {
        "earliest": bool(
            earliest is not None and earliest <= pd.Timestamp(requirements["earliest_at_or_before"])
        ),
        "latest": bool(
            latest is not None and latest >= pd.Timestamp(requirements["latest_at_or_after"])
        ),
        "span": span_days >= float(requirements["minimum_span_days"]),
        "rows": rows >= int(requirements["minimum_rows"]),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not failed_checks,
        "requirements": requirements,
        "actual": {
            "earliest": earliest.isoformat() if earliest is not None else None,
            "latest": latest.isoformat() if latest is not None else None,
            "span_days": round(span_days, 6),
            "rows": rows,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "remediation": {
            "action": "bootstrap_research_history",
            "command": [
                ".venv/bin/python",
                "-m",
                "src.autopilot.history_bootstrap",
                "--config",
                str(research_factory_config_path),
                "--market",
                scenario.market,
                "--report",
                (
                    f"runtime/history_bootstrap_{scenario.market}.json"
                    if scenario.symbol == "BTCUSDT"
                    else f"runtime/history_bootstrap_{scenario.symbol}_{scenario.market}.json"
                ),
                *([] if scenario.symbol == "BTCUSDT" else ["--symbol", scenario.symbol]),
            ],
            "note": (
                "Fetch the direct Binance timeframe history, then rerun the research cycle. "
                "Do not weaken the coverage contract to admit a shallow sample."
            ),
        },
    }


def _unprotected_epoch_capacity_requirements(
    source_coverage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(source_coverage, dict):
        return None
    source_requirements = source_coverage.get("requirements")
    if not isinstance(source_requirements, dict):
        return None
    return {
        "minimum_span_days": float(source_requirements["minimum_span_days"]),
        "minimum_rows": int(source_requirements["minimum_rows"]),
    }


def _unprotected_epoch_capacity_status(
    frame: pd.DataFrame,
    source_coverage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Check sample capacity without reapplying full-source recency bounds."""

    requirements = _unprotected_epoch_capacity_requirements(source_coverage)
    if requirements is None:
        return None
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    earliest = timestamps.min()
    latest = timestamps.max()
    span_days = float((latest - earliest).total_seconds() / 86_400)
    actual = {
        "earliest": earliest.isoformat(),
        "latest": latest.isoformat(),
        "span_days": round(span_days, 6),
        "rows": int(len(frame)),
    }
    checks = {
        "span": span_days >= requirements["minimum_span_days"],
        "rows": len(frame) >= requirements["minimum_rows"],
    }
    return {
        "ok": all(checks.values()),
        "requirements": requirements,
        "actual": actual,
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
    }


def _scenario_indicator_coverage_status(
    scenario: ResearchScenario,
    *,
    indicator_dir: Path,
    now: str | pd.Timestamp | None = None,
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> dict[str, Any] | None:
    """Check base-timeframe depth even when every candidate lacks a feature."""
    if _scenario_coverage_requirements(scenario, now=now) is None:
        return None
    path = indicator_dir / f"{scenario.symbol}_{scenario.base_tf}_all_indicators.parquet"
    try:
        filters: list[tuple[str, str, pd.Timestamp]] = [
            ("timestamp", ">=", _utc_timestamp_for_coverage(scenario.start)),
        ]
        if scenario.end:
            filters.append(("timestamp", "<=", _utc_timestamp_for_coverage(scenario.end)))
        frame = pd.read_parquet(path, columns=["timestamp"], filters=filters)
        if "timestamp" not in frame.columns:
            frame = frame.reset_index()
    except Exception as exc:
        frame = pd.DataFrame(columns=["timestamp"])
        read_error = f"{type(exc).__name__}: {exc}"
    else:
        read_error = None
    status = _scenario_coverage_status(
        frame,
        scenario,
        now=now,
        research_factory_config_path=research_factory_config_path,
    )
    if status is None:
        raise RuntimeError(f"{scenario.name}: required indicator coverage status was not produced")
    status["path"] = str(path)
    if read_error:
        status["read_error"] = read_error
    return status


def _utc_timestamp_for_coverage(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _coverage_failure_report(
    scenario: ResearchScenario,
    coverage: dict[str, Any],
    *,
    selection: dict[str, Any] | None,
    unsupported_hypotheses: list[dict[str, Any]],
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> dict[str, Any]:
    return {
        "ok": False,
        "skipped": True,
        "reason": "insufficient_history_coverage",
        "name": scenario.name,
        "product": scenario.product,
        "base_tf": scenario.base_tf,
        "pnl_unit": scenario.pnl_unit,
        "market": scenario.market,
        "symbol": scenario.symbol,
        "position": scenario.position,
        "opportunity_type": scenario.opportunity_type,
        "with_guards": scenario.with_guards,
        "candidate_set": scenario.candidate_set,
        "start": scenario.start,
        "end": scenario.end,
        "rows": int((coverage.get("actual") or {}).get("rows") or 0),
        "hypotheses": 0,
        "keepers": 0,
        "keeper_ids": [],
        "selection": selection,
        "unsupported_hypotheses": unsupported_hypotheses,
        "holdout_exposed_ids": [],
        "verdicts": {},
        "top_reasons": {"insufficient_history_coverage": 1},
        "coverage": coverage,
        "remediation": coverage.get("remediation")
        or {
            "action": "bootstrap_research_history",
            "command": [
                ".venv/bin/python",
                "-m",
                "src.autopilot.history_bootstrap",
                "--config",
                str(research_factory_config_path),
            ],
        },
    }


def _unprotected_epoch_deferral_report(
    scenario: ResearchScenario,
    *,
    selection: dict[str, Any] | None,
    unsupported_hypotheses: list[dict[str, Any]],
    retired_unsupported_ids: list[str],
    detail: str,
    protected_epoch_selection: dict[str, Any] | None = None,
    unprotected_epoch_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": True,
        "deferred": True,
        "reason": "unprotected_epoch_unavailable",
        "name": scenario.name,
        "product": scenario.product,
        "base_tf": scenario.base_tf,
        "pnl_unit": scenario.pnl_unit,
        "market": scenario.market,
        "symbol": scenario.symbol,
        "position": scenario.position,
        "opportunity_type": scenario.opportunity_type,
        "with_guards": scenario.with_guards,
        "candidate_set": scenario.candidate_set,
        "start": scenario.start,
        "end": scenario.end,
        "rows": 0,
        "hypotheses": 0,
        "keepers": 0,
        "keeper_ids": [],
        "selection": selection,
        "unsupported_hypotheses": unsupported_hypotheses,
        "retired_unsupported_ids": retired_unsupported_ids,
        "holdout_exposed_ids": [],
        "verdicts": {},
        "top_reasons": {"unprotected_epoch_unavailable": 1},
        "detail": detail,
        **(
            {"protected_epoch_selection": protected_epoch_selection}
            if protected_epoch_selection is not None
            else {}
        ),
        **(
            {"unprotected_epoch_capacity": unprotected_epoch_capacity}
            if unprotected_epoch_capacity is not None
            else {}
        ),
        "remediation": {
            "action": "wait_for_unprotected_history",
            "note": (
                "Protected evidence was not reused. Retry the same selection automatically "
                "after additional market history arrives."
            ),
        },
    }


def _peer_coverage_gate_report(
    scenario: ResearchScenario,
    *,
    failed_scenarios: list[str],
    coverage: dict[str, Any] | None,
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> dict[str, Any]:
    return {
        "ok": False,
        "skipped": True,
        "reason": "history_coverage_gate_blocked",
        "name": scenario.name,
        "product": scenario.product,
        "base_tf": scenario.base_tf,
        "pnl_unit": scenario.pnl_unit,
        "market": scenario.market,
        "symbol": scenario.symbol,
        "position": scenario.position,
        "opportunity_type": scenario.opportunity_type,
        "with_guards": scenario.with_guards,
        "candidate_set": scenario.candidate_set,
        "start": scenario.start,
        "end": scenario.end,
        "rows": 0,
        "hypotheses": 0,
        "keepers": 0,
        "keeper_ids": [],
        "selection": None,
        "unsupported_hypotheses": [],
        "holdout_exposed_ids": [],
        "verdicts": {},
        "top_reasons": {"history_coverage_gate_blocked": 1},
        "blocked_by_scenarios": failed_scenarios,
        **({"coverage": coverage} if coverage is not None else {}),
        "remediation": {
            "action": "bootstrap_research_history",
            "command": [
                ".venv/bin/python",
                "-m",
                "src.autopilot.history_bootstrap",
                "--config",
                str(research_factory_config_path),
                "--report",
                "runtime/history_bootstrap.json",
            ],
        },
    }


def _frame_window(frame: pd.DataFrame) -> dict[str, Any]:
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.Series(pd.to_datetime(frame.index, utc=True)).dropna()
    else:
        timestamps = pd.Series(dtype="datetime64[ns, UTC]")
    return {
        "start": timestamps.min().isoformat() if not timestamps.empty else None,
        "end": timestamps.max().isoformat() if not timestamps.empty else None,
        "rows": int(len(frame)),
    }


def _content_digest(path: Path) -> str:
    """Hash immutable dataset content once per process/stat version."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"dataset input must be a non-symlink regular file: {path}")
    stat_result = path.stat()
    key = (
        str(path.resolve()),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )
    cached = _FILE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = "sha256:" + digest.hexdigest()
    if len(_FILE_DIGEST_CACHE) >= _MAX_FILE_DIGEST_CACHE_ITEMS:
        _FILE_DIGEST_CACHE.clear()
    _FILE_DIGEST_CACHE[key] = result
    return result


def _dataset_snapshot(
    scenario: ResearchScenario,
    hypotheses: list[Hypothesis],
    *,
    frame: pd.DataFrame,
    indicator_dir: Path,
) -> dict[str, Any]:
    timeframes = sorted(
        {hypothesis.base_timeframe for hypothesis in hypotheses}
        | {
            predicate.timeframe
            for hypothesis in hypotheses
            for predicate in hypothesis.all_predicates()
        }
    )
    files: list[dict[str, Any]] = []
    for timeframe in timeframes:
        path = indicator_dir / f"{scenario.symbol}_{timeframe}_all_indicators.parquet"
        stat_result = path.stat()
        parquet = pq.ParquetFile(path)
        files.append(
            {
                "timeframe": timeframe,
                "path": str(path),
                "file": path.name,
                "size_bytes": int(stat_result.st_size),
                "content_digest": _content_digest(path),
                "rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
                "schema": hashlib.sha256(str(parquet.schema_arrow).encode()).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "symbol": scenario.symbol,
        "market": scenario.market,
        "base_timeframe": scenario.base_tf,
        "scenario_window": {"start": scenario.start, "end": scenario.end},
        "loaded_window": _frame_window(frame),
        "files": files,
    }
    identity_manifest = {
        **manifest,
        "files": [{key: value for key, value in item.items() if key != "path"} for item in files],
    }
    encoded = json.dumps(identity_manifest, sort_keys=True, separators=(",", ":"))
    return {
        **manifest,
        "snapshot_id": "sha256:" + hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _development_window(frame: pd.DataFrame, config: ValidationConfig) -> dict[str, Any]:
    segments = split_frame(frame, config)
    return {
        "train": _frame_window(segments["train"]),
        "validation": _frame_window(segments["validation"]),
        "pre_holdout_rows": int(len(segments["train"]) + len(segments["validation"])),
    }


def _select_unprotected_epoch(
    frame: pd.DataFrame,
    protected_intervals: tuple[dict[str, str], ...],
    *,
    feature_timeframes: tuple[str, ...] = (),
    capacity_requirements: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Choose the largest contiguous row run that has never been protected.

    Protected observations are never recycled into adaptive research. Rather
    than letting one timeframe permanently starve every other scenario for the
    same market, later scenarios consume a disjoint chronological epoch. Ties
    prefer the newest run so regime research remains as current as the
    protection boundary permits.
    """

    if not protected_intervals or frame.empty:
        return frame, None
    if "timestamp" not in frame.columns:
        raise ValueError("protected-epoch selection requires a timestamp column")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    normalized_timeframes = tuple(sorted(set(feature_timeframes)))
    unknown_timeframes = sorted(set(normalized_timeframes) - set(TIMEFRAME_SECONDS))
    if unknown_timeframes:
        raise ValueError(
            "protected-epoch embargo has unsupported feature timeframe(s): "
            + ", ".join(unknown_timeframes)
        )
    max_timeframe = (
        max(normalized_timeframes, key=lambda item: TIMEFRAME_SECONDS[item])
        if normalized_timeframes
        else None
    )
    embargo_seconds = (
        FEATURE_DEPENDENCY_MAX_NATIVE_BARS * TIMEFRAME_SECONDS[max_timeframe]
        if max_timeframe is not None
        else 0
    )
    protected_blocked = pd.Series(False, index=frame.index)
    embargo_blocked = pd.Series(False, index=frame.index)
    relevant: list[dict[str, str]] = []
    for interval in protected_intervals:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        overlap = (timestamps >= start) & (timestamps <= end)
        embargo = pd.Series(False, index=frame.index)
        if embargo_seconds:
            embargo_end = end + pd.Timedelta(seconds=embargo_seconds)
            embargo = (timestamps > end) & (timestamps <= embargo_end)
        if bool(overlap.any()) or bool(embargo.any()):
            protected_blocked |= overlap
            embargo_blocked |= embargo
            relevant.append(interval)
    if not relevant:
        return frame, None
    # A row covered by another protected interval is classified as protected,
    # not embargo, so the returned counts remain disjoint and auditable.
    embargo_blocked &= ~protected_blocked
    blocked = protected_blocked | embargo_blocked

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for position, is_blocked in enumerate(blocked.to_numpy(dtype=bool)):
        if not is_blocked and run_start is None:
            run_start = position
        elif is_blocked and run_start is not None:
            runs.append((run_start, position))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(frame)))
    if not runs:
        raise UnprotectedResearchEpochUnavailableError(
            "no unprotected chronological research epoch remains for this market and symbol"
        )
    eligible_runs = runs
    if capacity_requirements is not None:
        minimum_rows = int(capacity_requirements["minimum_rows"])
        minimum_span_days = float(capacity_requirements["minimum_span_days"])
        qualified_runs = [
            run
            for run in runs
            if run[1] - run[0] >= minimum_rows
            and (timestamps.iloc[run[1] - 1] - timestamps.iloc[run[0]]).total_seconds() / 86_400
            >= minimum_span_days
        ]
        if qualified_runs:
            eligible_runs = qualified_runs
    start_position, end_position = max(
        eligible_runs,
        key=lambda run: (
            run[1] - run[0],
            timestamps.iloc[run[1] - 1],
        ),
    )
    selected = frame.iloc[start_position:end_position].reset_index(drop=True)
    detail: dict[str, Any] = {
        "policy": "largest_contiguous_unprotected_epoch",
        "input_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "excluded_rows": int(blocked.sum()),
        "start": str(pd.Timestamp(selected["timestamp"].iloc[0])),
        "end": str(pd.Timestamp(selected["timestamp"].iloc[-1])),
        "protected_intervals_considered": len(relevant),
    }
    if capacity_requirements is not None:
        detail["capacity_selection"] = {
            "requirements": capacity_requirements,
            "available_runs": len(runs),
            "qualified_runs": len(qualified_runs),
        }
    if normalized_timeframes:
        detail.update(
            protected_rows_excluded=int(protected_blocked.sum()),
            feature_dependency_embargo_rows_excluded=int(embargo_blocked.sum()),
            feature_dependency_embargo={
                "policy": "maximum_supported_native_rolling_dependency",
                "max_native_bars": FEATURE_DEPENDENCY_MAX_NATIVE_BARS,
                "feature_timeframes": list(normalized_timeframes),
                "max_timeframe": max_timeframe,
                "duration_seconds": embargo_seconds,
            },
        )
    return selected, detail


def _hypothesis_feature_timeframes(hypotheses: list[Hypothesis]) -> tuple[str, ...]:
    """Return every native timeframe whose values can influence the batch."""

    timeframes: set[str] = set()
    for hypothesis in hypotheses:
        timeframes.add(hypothesis.base_timeframe)
        timeframes.update(hypothesis.timeframes())
    return tuple(sorted(timeframes))


def _development_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "stage_reached": _candidate_stage(result),
        "dsr_deflated": result.get("dsr_deflated"),
        "dsr_method": result.get("dsr_method"),
        "n_trials": result.get("n_trials"),
        "sr_std_trials": result.get("sr_std_trials"),
        "trial_sharpe_count": result.get("trial_sharpe_count"),
        "trial_sharpe_observed_std": result.get("trial_sharpe_observed_std"),
        "trial_sharpe_conservative_floor": result.get("trial_sharpe_conservative_floor"),
    }
    for stage in ("train", "validation"):
        if summary := _segment_summary(result, stage):
            metrics[stage] = summary
    if isinstance(result.get("oos"), dict):
        metrics["oos_pass_rate"] = result["oos"].get("pass_rate")
    if isinstance(result.get("sensitivity"), dict):
        metrics["sensitivity_pass_fraction"] = result["sensitivity"].get("pass_fraction")
    return metrics


def _memory_protocol(
    scenario: ResearchScenario,
    validation_config: ValidationConfig,
    eval_config: EvalConfig,
) -> dict[str, Any]:
    return {
        "schema": "autopilot.staged_validation/v3",
        "research_engine_digest": execution_engine_digest(),
        "scenario": scenario.name,
        "product": scenario.product,
        "market": scenario.market,
        "symbol": scenario.symbol,
        "pnl_unit": scenario.pnl_unit,
        "base_timeframe": scenario.base_tf,
        "validation": dataclasses.asdict(validation_config),
        "evaluation": dataclasses.asdict(eval_config),
        "dsr": {
            "method": DSR_METHOD,
            "n_trials": validation_config.n_trials,
            "sr_std_trials": validation_config.sr_std_trials,
            "trial_sharpe_count": validation_config.trial_sharpe_count,
            "trial_sharpe_observed_std": validation_config.trial_sharpe_observed_std,
            "trial_sharpe_conservative_floor": (validation_config.trial_sharpe_conservative_floor),
        },
    }


def _register_legacy_memory_strategy(
    memory: ExperimentMemory,
    hypothesis: Hypothesis,
    scenario: ResearchScenario,
) -> str:
    spec = {
        **hypothesis.to_dict(),
        "_product": scenario.product,
        "_market": scenario.market,
        "_symbol": scenario.symbol,
        "_pnl_unit": scenario.pnl_unit,
        "_opportunity_type": scenario.opportunity_type,
        "_search_space": scenario.name,
    }
    behavior_hash = canonical_strategy_hash(spec)
    memory.register_strategy(
        spec,
        strategy_id=hypothesis.id,
        generation_method="legacy_seed",
        metadata={
            "family": hypothesis.family,
            "product": scenario.product,
            "market": scenario.market,
            "symbol": scenario.symbol,
            "pnl_unit": scenario.pnl_unit,
            "opportunity_type": scenario.opportunity_type,
            "search_space": scenario.name,
            "base_timeframe": scenario.base_tf,
            "lineage_depth": 0,
        },
    )
    return behavior_hash


def run_validation_scenario(
    scenario: ResearchScenario,
    *,
    hypotheses: list[Any] | None = None,
    selection: dict[str, Any] | None = None,
    hypothesis_metadata: dict[str, dict[str, Any]] | None = None,
    log_path: Path = DEFAULT_LOG,
    coverage_now: str | pd.Timestamp | None = None,
    experiment_memory: ExperimentMemory | None = None,
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
    holdout_seal_min_interval_seconds: float = HOLDOUT_SEAL_MIN_INTERVAL_SECONDS,
) -> dict[str, Any]:
    hypotheses = _hypotheses_for(scenario) if hypotheses is None else hypotheses
    if not hypotheses:
        raise ValueError(f"{scenario.name}: no hypotheses for base timeframe {scenario.base_tf}")
    selected_indicator_dir = indicator_data_dir(
        scenario.symbol, scenario.market, legacy_fallback=True
    )
    coverage = _scenario_indicator_coverage_status(
        scenario,
        indicator_dir=selected_indicator_dir,
        now=coverage_now,
        research_factory_config_path=research_factory_config_path,
    )
    if coverage is not None and not coverage["ok"]:
        return _coverage_failure_report(
            scenario,
            coverage,
            selection=selection,
            unsupported_hypotheses=[],
            research_factory_config_path=research_factory_config_path,
        )
    supported_hypotheses, unsupported_hypotheses = _partition_supported_hypotheses(
        hypotheses,
        indicator_dir=selected_indicator_dir,
        symbol=scenario.symbol,
    )
    retired_unsupported_ids = _retire_unsupported_generated_hypotheses(
        unsupported_hypotheses,
        hypothesis_metadata=hypothesis_metadata,
        experiment_memory=experiment_memory,
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
            "symbol": scenario.symbol,
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
            "retired_unsupported_ids": retired_unsupported_ids,
            "holdout_exposed_ids": [],
            "verdicts": {},
            "top_reasons": {"unsupported_features": len(unsupported_hypotheses)},
        }
    frame_kwargs: dict[str, Any] = {
        "base_tf": scenario.base_tf,
        "start": scenario.start,
        "end": scenario.end,
        "indicator_dir": selected_indicator_dir,
    }
    if scenario.symbol != "BTCUSDT":
        frame_kwargs["symbol"] = scenario.symbol
    frame = build_aligned_frame(supported_hypotheses, **frame_kwargs)
    # Freshness belongs to the complete aligned source. A protected-safe epoch
    # may intentionally end well before wall-clock recency after newer rows
    # have been sealed as final evaluation data.
    coverage = _scenario_coverage_status(
        frame,
        scenario,
        now=coverage_now,
        research_factory_config_path=research_factory_config_path,
    )
    if coverage is not None and not coverage["ok"]:
        return _coverage_failure_report(
            scenario,
            coverage,
            selection=selection,
            unsupported_hypotheses=unsupported_hypotheses,
            research_factory_config_path=research_factory_config_path,
        )
    epoch_selection = None
    if experiment_memory is not None:
        epoch_capacity_requirements = _unprotected_epoch_capacity_requirements(coverage)
        try:
            frame, epoch_selection = _select_unprotected_epoch(
                frame,
                experiment_memory.protected_intervals(
                    market=scenario.market,
                    symbol=scenario.symbol,
                ),
                feature_timeframes=_hypothesis_feature_timeframes(supported_hypotheses),
                capacity_requirements=epoch_capacity_requirements,
            )
        except UnprotectedResearchEpochUnavailableError as exc:
            return _unprotected_epoch_deferral_report(
                scenario,
                selection=selection,
                unsupported_hypotheses=unsupported_hypotheses,
                retired_unsupported_ids=retired_unsupported_ids,
                detail=str(exc),
            )
        epoch_capacity = _unprotected_epoch_capacity_status(frame, coverage)
        if epoch_selection is not None and epoch_capacity is not None and not epoch_capacity["ok"]:
            failed_checks = ", ".join(epoch_capacity["failed_checks"])
            return _unprotected_epoch_deferral_report(
                scenario,
                selection=selection,
                unsupported_hypotheses=unsupported_hypotheses,
                retired_unsupported_ids=retired_unsupported_ids,
                detail=(
                    "the available unprotected research epoch does not satisfy minimum "
                    f"sample capacity ({failed_checks})"
                ),
                protected_epoch_selection=epoch_selection,
                unprotected_epoch_capacity=epoch_capacity,
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
    validation_cfg = with_trial_sharpe_dispersion(
        frame,
        supported_hypotheses,
        validation_cfg,
        eval_cfg,
    )
    memory_hashes: dict[str, str] = {}
    dataset_snapshot: dict[str, Any] | None = None
    development_window: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    already_evaluated: list[str] = []
    if experiment_memory is not None:
        dataset_snapshot = _dataset_snapshot(
            scenario,
            supported_hypotheses,
            frame=frame,
            indicator_dir=selected_indicator_dir,
        )
        development_window = _development_window(frame, validation_cfg)
        protocol = _memory_protocol(scenario, validation_cfg, eval_cfg)
        # This check runs before validate_batch receives the frame. A protected
        # candle remains unavailable to adaptive train/validation forever,
        # even when a later snapshot grows or changes timeframe.
        experiment_memory.assert_adaptive_window_allowed(
            dataset=dataset_snapshot,
            window=development_window,
            protocol=protocol,
            phase="development",
        )
        hypothesis_metadata = hypothesis_metadata or {}
        pending_hypotheses: list[Hypothesis] = []
        for hypothesis in supported_hypotheses:
            metadata = hypothesis_metadata.get(hypothesis.id, {})
            behavior_hash = metadata.get("strategy_hash")
            if behavior_hash is None:
                behavior_hash = _register_legacy_memory_strategy(
                    experiment_memory,
                    hypothesis,
                    scenario,
                )
            else:
                behavior_hash = str(behavior_hash)
                registered = experiment_memory.get_strategy(behavior_hash)
                submitted = registered.get("submitted_spec") or {}
                if canonical_strategy_hash(submitted) != behavior_hash:
                    raise ValueError(f"{hypothesis.id}: experiment-memory identity mismatch")
            memory_hashes[hypothesis.id] = behavior_hash
            registered_strategy = experiment_memory.get_strategy(behavior_hash)
            if registered_strategy.get("holdout_exposed_at") is not None:
                already_evaluated.append(hypothesis.id)
            elif experiment_memory.is_tested(
                behavior_hash,
                dataset=dataset_snapshot,
                window=development_window,
                protocol=protocol,
                phase="development",
            ):
                already_evaluated.append(hypothesis.id)
            else:
                pending_hypotheses.append(hypothesis)
        supported_hypotheses = pending_hypotheses
        if not supported_hypotheses:
            return {
                "ok": True,
                "skipped": True,
                "reason": "already_evaluated_on_snapshot",
                "name": scenario.name,
                "product": scenario.product,
                "base_tf": scenario.base_tf,
                "pnl_unit": scenario.pnl_unit,
                "market": scenario.market,
                "symbol": scenario.symbol,
                "position": scenario.position,
                "opportunity_type": scenario.opportunity_type,
                "with_guards": scenario.with_guards,
                "candidate_set": scenario.candidate_set,
                "start": scenario.start,
                "end": scenario.end,
                "rows": int(len(frame)),
                "hypotheses": 0,
                "keepers": 0,
                "keeper_ids": [],
                "selection": selection,
                "unsupported_hypotheses": unsupported_hypotheses,
                "retired_unsupported_ids": retired_unsupported_ids,
                "holdout_exposed_ids": [],
                "already_evaluated_ids": already_evaluated,
                "verdicts": {},
                "top_reasons": {"already_evaluated_on_snapshot": len(already_evaluated)},
                "dataset_snapshot_id": dataset_snapshot["snapshot_id"],
            }

    holdout_claims: dict[str, str] = {}
    holdout_window: dict[str, Any] | None = None
    # Cohort/interval sealing is deliberately lazy. Registering it here would
    # permanently seal this frame's holdout window on every scenario pass,
    # even when no candidate ever earns a holdout read — a runaway loop can
    # then consume the entire chronological history as protected evidence.
    # The seal happens at first exposure risk instead: when the first
    # candidate passes every pre-holdout stage.
    holdout_cohort: dict[str, Any] = {"members": set(), "created": None, "scope": None}
    if experiment_memory is not None:
        if dataset_snapshot is None or protocol is None:
            raise RuntimeError(
                f"{scenario.name}: experiment-memory holdout context was not initialized"
            )
        holdout_window = _segment_bounds(split_frame(frame, validation_cfg)["holdout"])

    def _ensure_holdout_cohort_sealed() -> str | None:
        """Seal cohort+interval on first need; return a deferral reason if blocked."""

        if holdout_cohort["scope"] is not None:
            return None
        try:
            cohort = experiment_memory.register_holdout_cohort(
                [memory_hashes[hypothesis.id] for hypothesis in supported_hypotheses],
                dataset=dataset_snapshot,
                window=holdout_window,
                protocol=protocol,
                min_seconds_since_last_seal=holdout_seal_min_interval_seconds,
            )
        except HoldoutSealBudgetError:
            return "holdout_seal_budget_exhausted"
        except EvaluationConflictError:
            return "holdout_cohort_seal_conflict"
        holdout_cohort["members"] = set(cohort.member_hashes)
        holdout_cohort["created"] = cohort.created
        holdout_cohort["scope"] = cohort.scope_key
        return None

    def before_holdout(hypothesis: Hypothesis, partial_result: dict[str, Any]) -> bool | str:
        if experiment_memory is None:
            return True
        if dataset_snapshot is None or development_window is None or protocol is None:
            raise RuntimeError(
                f"{scenario.name}: experiment-memory claim context was not initialized"
            )
        behavior_hash = memory_hashes[hypothesis.id]
        experiment_memory.record_outcome(
            behavior_hash,
            dataset=dataset_snapshot,
            window=development_window,
            protocol=protocol,
            phase="development",
            outcome="pre_holdout_pass",
            metrics=_development_metrics(partial_result),
            details={"stage_reached": "sensitivity", "holdout_feedback_allowed": False},
        )
        deferral_reason = _ensure_holdout_cohort_sealed()
        if deferral_reason is not None:
            return deferral_reason
        if behavior_hash not in holdout_cohort["members"]:
            return False
        snapshot_id = str(dataset_snapshot["snapshot_id"])
        if experiment_memory.holdout_claimed(behavior_hash, snapshot_id=snapshot_id):
            return False
        try:
            claim = experiment_memory.claim_holdout(
                behavior_hash,
                snapshot_id=snapshot_id,
                dataset=dataset_snapshot,
                window=partial_result["splits"]["holdout"],
                protocol=protocol,
            )
        except EvaluationConflictError:
            return False
        if not claim.created:
            return False
        holdout_claims[hypothesis.id] = claim.evaluation_key
        return True

    def checkpoint_candidate(hypothesis: Hypothesis, result: dict[str, Any]) -> None:
        """Persist each candidate before the sequential batch advances."""

        if experiment_memory is None:
            return
        if dataset_snapshot is None or development_window is None or protocol is None:
            raise RuntimeError(
                f"{scenario.name}: experiment-memory checkpoint context was not initialized"
            )
        behavior_hash = memory_hashes[hypothesis.id]
        if result.get("holdout") is not None:
            evaluation_key = holdout_claims.get(hypothesis.id)
            if evaluation_key is None:
                raise RuntimeError(f"{hypothesis.id}: holdout was read without a durable claim")
            experiment_memory.complete_evaluation(
                evaluation_key,
                outcome=str(result["verdict"]),
                rejection_reasons=tuple(str(item) for item in result.get("reasons") or []),
                metrics={"holdout": _segment_summary(result, "holdout") or {}},
                details={"protected_feedback": True},
            )
        elif not (set(result.get("reasons") or []) & HOLDOUT_GATE_DEFERRAL_REASONS):
            experiment_memory.record_outcome(
                behavior_hash,
                dataset=dataset_snapshot,
                window=development_window,
                protocol=protocol,
                phase="development",
                outcome=str(result["verdict"]),
                rejection_reasons=tuple(str(item) for item in result.get("reasons") or []),
                metrics=_development_metrics(result),
                details={"holdout_feedback_allowed": False},
            )

    validate_kwargs: dict[str, Any] = {
        "eval_cfg": eval_cfg,
        "log_path": log_path,
    }
    if experiment_memory is not None:
        validate_kwargs["before_holdout"] = before_holdout
        validate_kwargs["after_candidate"] = checkpoint_candidate
    results = validate_batch(
        frame,
        supported_hypotheses,
        validation_cfg,
        **validate_kwargs,
    )
    incubation_candidates = _incubation_candidates_from_results(
        results,
        hypothesis_metadata=hypothesis_metadata,
    )
    holdout_exposed_ids = sorted(
        {
            str(result.get("hypothesis_id"))
            for result in results
            if result.get("hypothesis_id") and result.get("holdout") is not None
        }
    )
    return {
        "ok": True,
        "name": scenario.name,
        "product": scenario.product,
        "base_tf": scenario.base_tf,
        "pnl_unit": scenario.pnl_unit,
        "market": scenario.market,
        "symbol": scenario.symbol,
        "position": scenario.position,
        "opportunity_type": scenario.opportunity_type,
        "with_guards": scenario.with_guards,
        "candidate_set": scenario.candidate_set,
        "start": scenario.start,
        "end": scenario.end,
        "rows": int(len(frame)),
        **({"protected_epoch_selection": epoch_selection} if epoch_selection is not None else {}),
        "coverage": coverage,
        "unsupported_hypotheses": unsupported_hypotheses,
        "retired_unsupported_ids": retired_unsupported_ids,
        "already_evaluated_ids": already_evaluated,
        "holdout_exposed_ids": holdout_exposed_ids,
        **(
            {"dataset_snapshot_id": dataset_snapshot["snapshot_id"]}
            if dataset_snapshot is not None
            else {}
        ),
        **(
            {
                "holdout_cohort_scope": holdout_cohort["scope"],
                "holdout_cohort_created": holdout_cohort["created"],
                "holdout_cohort_members": len(holdout_cohort["members"]),
            }
            if holdout_cohort["scope"] is not None
            else {}
        ),
        "trial_count": validation_cfg.n_trials,
        "selection": selection
        or {
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


def stage_live_product_candidate(
    product: ProductConfig,
    *,
    pnl_unit: str,
    market: str,
    out: Path,
    top_k: int,
    ids: list[str] | None = None,
    min_dsr: float | None = None,
    log_path: Path = DEFAULT_LOG,
) -> dict[str, Any]:
    """Export, policy-check, and atomically stage a candidate for a live product."""
    if out.resolve(strict=False) == product.strategies_path.resolve(strict=False):
        raise ValueError(
            f"{product.name}: candidate staging path must be distinct from the active artifact"
        )
    open_position_ids = _open_position_ids_for_export(
        product.name,
        state_file=product.state_file,
    )
    if open_position_ids:
        return {
            "ok": True,
            "product": product.name,
            "pnl_unit": pnl_unit,
            "market": market,
            "exported": False,
            "staged": False,
            "destination": "staging",
            "activation_required": True,
            "reason": "open_positions_block_export",
            "detail": "staged candidate is left unchanged while positions are open",
            "artifact": str(out),
            "active_artifact": str(product.strategies_path),
            "ids": ids or [],
            "open_positions": open_position_ids,
            "min_dsr": min_dsr,
        }

    if out.is_symlink():
        raise ValueError(f"{product.name}: candidate staging path must not be a symlink: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.parent.is_symlink():
        raise ValueError(
            f"{product.name}: candidate staging directory must not be a symlink: {out.parent}"
        )

    existing_candidate: dict[str, Any] | None = None
    existing_digest: str | None = None
    if out.exists():
        existing_candidate = load_artifact(out)
        existing_digest = artifact_digest(existing_candidate)

    scratch_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".research.tmp",
            delete=False,
        ) as handle:
            scratch_path = Path(handle.name)
        try:
            path = export_strategies(
                log_path=log_path,
                output_path=scratch_path,
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
                "product": product.name,
                "pnl_unit": pnl_unit,
                "market": market,
                "exported": False,
                "staged": False,
                "destination": "staging",
                "activation_required": True,
                "reason": "no_exportable_strategies",
                "detail": str(exc),
                "artifact": str(out),
                "active_artifact": str(product.strategies_path),
                "ids": ids or [],
                "min_dsr": min_dsr,
            }
        payload = load_artifact(path)
        payload["product"] = product_identity(product)
        payload["candidate_staging"] = {
            "staged_at": utc_now(),
            "activation_required": True,
            "approval_granted": False,
            "active_artifact": str(product.strategies_path),
        }
        assert_loaded_strategy_artifact_allowed(
            product,
            payload,
            artifact_path=out,
            require_live_eligible=True,
        )
        if existing_candidate is not None and existing_digest is not None:
            if _candidate_paper_identity(existing_candidate) == _candidate_paper_identity(payload):
                return {
                    "ok": True,
                    "product": product.name,
                    "pnl_unit": pnl_unit,
                    "market": market,
                    "exported": False,
                    "staged": False,
                    "destination": "staging",
                    "activation_required": True,
                    "approval_granted": False,
                    "reason": "candidate_already_staged",
                    "artifact_digest": existing_digest,
                    "artifact": str(out),
                    "active_artifact": str(product.strategies_path),
                    "strategies": len(existing_candidate.get("strategies", [])),
                    "ids": ids or [],
                    "min_dsr": min_dsr,
                }
            old_state = candidate_paper_paths(
                product.name,
                existing_digest,
                candidate_dir=out.parent,
            )["state"]
            old_open_positions = _open_position_ids_for_export(
                product.name,
                state_file=old_state,
            )
            if old_open_positions:
                return {
                    "ok": True,
                    "product": product.name,
                    "pnl_unit": pnl_unit,
                    "market": market,
                    "exported": False,
                    "staged": False,
                    "destination": "staging",
                    "activation_required": True,
                    "reason": "prior_candidate_open_positions",
                    "detail": (
                        "the staged artifact is unchanged until its digest-isolated "
                        "paper positions are flat"
                    ),
                    "artifact_digest": existing_digest,
                    "artifact": str(out),
                    "active_artifact": str(product.strategies_path),
                    "open_positions": old_open_positions,
                    "state_file": str(old_state),
                    "ids": ids or [],
                    "min_dsr": min_dsr,
                }
        write_json_atomic(out, payload)
    finally:
        if scratch_path is not None and scratch_path.exists():
            scratch_path.unlink()

    return {
        "ok": True,
        "product": product.name,
        "pnl_unit": pnl_unit,
        "market": market,
        "exported": True,
        "staged": True,
        "destination": "staging",
        "activation_required": True,
        "approval_granted": False,
        "artifact_digest": artifact_digest(payload),
        "artifact": str(out),
        "active_artifact": str(product.strategies_path),
        "strategies": len(payload.get("strategies", [])),
        "ids": ids or [],
        "min_dsr": min_dsr,
    }


def _candidate_paper_identity(payload: dict[str, Any]) -> str:
    """Identity whose unchanged value must preserve accumulated paper state."""

    keys = (
        "version",
        "market",
        "symbol",
        "pnl_unit",
        "paper_trade_allowed",
        "live_allowed",
        "promotion_eligible",
        "product",
        "strategies",
    )
    return artifact_digest({key: payload.get(key) for key in keys})


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
    symbol: str,
) -> list[str]:
    ids: list[str] = []
    for scenario in scenario_reports:
        if (
            scenario.get("product") != product
            or scenario.get("market") != market
            or scenario.get("symbol", "BTCUSDT") != symbol
        ):
            continue
        if scenario.get("skipped") or not scenario.get("ok"):
            continue
        ids.extend(str(item) for item in (scenario.get("keeper_ids") or []) if item)
    return sorted(set(ids))


def _active_income_candidate_product(
    base_product: ProductConfig,
    symbol: str,
    *,
    candidate_dir: Path,
) -> ProductConfig:
    symbol = symbol.upper()
    name = f"{base_product.name}__{symbol.lower()}"
    return dataclasses.replace(
        base_product,
        name=name,
        execution_mode="live",
        symbol=symbol,
        strategies_path=candidate_path_for_product(name, candidate_dir=candidate_dir),
        state_file=candidate_dir / f"{name}_state.json",
        trade_log=candidate_dir / f"{name}_paper_trades.csv",
        preflight_report=candidate_dir / f"{name}_preflight_report.json",
        testnet_rehearsal_report=candidate_dir / f"{name}_testnet_rehearsal_report.json",
    )


def _stage_active_income_symbol_candidates(
    scenario_reports: list[dict[str, Any]],
    *,
    base_product: ProductConfig,
    export_cfg: dict[str, Any],
    candidate_dir: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    symbols = sorted(
        {
            str(scenario.get("symbol") or "BTCUSDT").upper()
            for scenario in scenario_reports
            if scenario.get("product") == "active_income"
            and scenario.get("market") == export_cfg["market"]
            and scenario.get("ok")
            and not scenario.get("skipped")
            and scenario.get("keeper_ids")
        }
        - {base_product.symbol.upper()}
    )
    reports: list[dict[str, Any]] = []
    for symbol in symbols:
        product = _active_income_candidate_product(
            base_product,
            symbol,
            candidate_dir=candidate_dir,
        )
        keeper_ids = _current_keeper_ids(
            scenario_reports,
            product="active_income",
            market=str(export_cfg["market"]),
            symbol=symbol,
        )
        try:
            report = stage_live_product_candidate(
                product,
                pnl_unit=str(export_cfg["pnl_unit"]),
                market=str(export_cfg["market"]),
                out=product.strategies_path,
                top_k=int(export_cfg["top_k"]),
                ids=keeper_ids,
                min_dsr=(
                    float(export_cfg["min_dsr"]) if export_cfg.get("min_dsr") is not None else None
                ),
                log_path=log_path,
            )
            report.update(
                research_product="active_income",
                symbol=symbol,
                configured_for_execution=False,
                activation_blocked_until_product_configured=True,
            )
        except Exception as exc:
            report = {
                "ok": False,
                "product": product.name,
                "research_product": "active_income",
                "symbol": symbol,
                "exported": False,
                "artifact": str(product.strategies_path),
                "configured_for_execution": False,
                "activation_blocked_until_product_configured": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        reports.append(report)
    return reports


def run_research_cycle(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
    state_path: Path = DEFAULT_STATE,
    output_path: Path | None = None,
    incubation_output_path: Path | None = None,
    log_path: Path = DEFAULT_LOG,
    scenarios: tuple[ResearchScenario, ...] = DEFAULT_SCENARIOS,
    force: bool = False,
    include_mutations: bool = False,
    mutation_batch_path: Path = DEFAULT_MUTATION_BATCH,
    include_generated: bool = False,
    generated_only: bool = False,
    generated_batch_path: Path = DEFAULT_GENERATED_BATCH,
    research_factory_config_path: Path = DEFAULT_RESEARCH_FACTORY_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    configured_products = {product.name: product for product in config.products}
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
    generated_scenarios: tuple[ResearchScenario, ...] = ()
    generated_hypotheses: dict[str, list[Hypothesis]] = {}
    generated_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    generated_batch_summary: dict[str, Any] | None = None
    if include_generated:
        (
            generated_scenarios,
            generated_hypotheses,
            generated_metadata,
            generated_batch_summary,
        ) = _load_generated_scenarios(
            generated_batch_path,
            factory_config_path=research_factory_config_path,
        )
    base_scenarios = () if generated_only else scenarios
    scenarios = tuple(
        sorted(
            (*base_scenarios, *generated_scenarios, *mutation_scenarios),
            key=_protected_epoch_scenario_order,
        )
    )
    scenario_markets = sorted({scenario.market for scenario in scenarios}) or ["futures"]
    market_data_by_market = build_market_data_statuses(scenario_markets)
    ready_markets = {market for market, status in market_data_by_market.items() if status.get("ok")}
    marker = {
        market: status.get("last_timestamp")
        for market, status in market_data_by_market.items()
        if status.get("ok")
    }
    market_marker = _market_data_skip_marker(market_data_by_market)
    generated_at = utc_now()
    history_coverage: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if _scenario_coverage_requirements(scenario, now=generated_at) is None:
            continue
        history_coverage[scenario.name] = (
            _scenario_indicator_coverage_status(
                scenario,
                indicator_dir=indicator_data_dir(
                    scenario.symbol, scenario.market, legacy_fallback=True
                ),
                now=generated_at,
                research_factory_config_path=research_factory_config_path,
            )
            or {}
        )
    history_failed_scenarios = [
        name for name, status in history_coverage.items() if not status.get("ok")
    ]
    history_coverage_marker = _history_coverage_skip_marker(history_coverage)
    report: dict[str, Any] = {
        "ok": False,
        "generated_at": generated_at,
        "market_data": {
            "ok": all(status.get("ok") for status in market_data_by_market.values()),
            "markets": market_data_by_market,
        },
        "mutation_batch": mutation_batch_summary,
        "generated_batch": generated_batch_summary,
        "history_coverage": {
            "ok": all(status.get("ok") for status in history_coverage.values()),
            "failure_count": len(history_failed_scenarios),
            "failed_scenarios": history_failed_scenarios,
            "scenarios": history_coverage,
        },
        "scenarios": [],
        "exports": [],
        "skipped": False,
    }
    if generated_only and not generated_scenarios:
        report.update(error="generated_batch_not_ready")
        if output_path:
            write_json_atomic(output_path, report)
        return report
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
    generated_marker = None
    if generated_batch_summary is not None:
        generated_marker = json.dumps(
            {
                "status": generated_batch_summary.get("status"),
                "generated_at": generated_batch_summary.get("generated_at"),
                "hypotheses": generated_batch_summary.get("hypotheses", 0),
                "scenarios": generated_batch_summary.get("scenarios", 0),
                "cumulative_trials": generated_batch_summary.get("cumulative_trials", 0),
            },
            sort_keys=True,
        )
    state = _load_state(state_path)
    state_recovered = bool(state.get("_state_recovered"))
    consumed_holdout_registry = _consumed_holdout_registry(state)
    initial_consumed_holdout_registry = _serialized_holdout_registry(consumed_holdout_registry)
    if state_recovered:
        report["state_recovered"] = True
        report["state_error"] = state.get("_state_error")
    if (
        not force
        and state.get("last_market_marker") == market_marker
        and state.get("last_mutation_batch_marker") == mutation_marker
        and state.get("last_generated_batch_marker") == generated_marker
        and (
            history_coverage_marker is None
            or state.get("last_history_coverage_marker") == history_coverage_marker
        )
    ):
        report.update(
            ok=True,
            skipped=True,
            reason="market_data_unchanged",
            last_market_timestamp=last_timestamp,
            last_market_marker=market_marker,
            last_mutation_batch_marker=mutation_marker,
            last_generated_batch_marker=generated_marker,
            last_history_coverage_marker=history_coverage_marker,
        )
        if output_path:
            write_json_atomic(output_path, report)
        return report

    scenario_reports: list[dict[str, Any]] = []
    next_offsets: dict[str, int] = {}
    experiment_memory = (
        ExperimentMemory(load_factory_config(research_factory_config_path).memory_path)
        if include_generated
        else None
    )
    for scenario in scenarios:
        try:
            coverage = history_coverage.get(scenario.name)
            if coverage is not None and not coverage.get("ok"):
                scenario_reports.append(
                    _coverage_failure_report(
                        scenario,
                        coverage,
                        selection=None,
                        unsupported_hypotheses=[],
                        research_factory_config_path=research_factory_config_path,
                    )
                )
                continue
            if scenario.market not in ready_markets:
                scenario_reports.append(
                    {
                        "ok": coverage is None,
                        "skipped": True,
                        "reason": "market_data_not_ready",
                        "name": scenario.name,
                        "product": scenario.product,
                        "base_tf": scenario.base_tf,
                        "pnl_unit": scenario.pnl_unit,
                        "market": scenario.market,
                        "symbol": scenario.symbol,
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
                        **({"coverage": coverage} if coverage is not None else {}),
                    }
                )
                continue
            if scenario.name in generated_hypotheses:
                hypotheses, selection = _select_from_hypotheses(
                    scenario,
                    generated_hypotheses[scenario.name],
                    state,
                )
                cumulative_trials = max(
                    (
                        _int_count(item.get("cumulative_trials"))
                        for item in generated_metadata.get(scenario.name, {}).values()
                    ),
                    default=0,
                )
                selection["cumulative_trials"] = max(
                    cumulative_trials,
                    _int_count(selection.get("available")),
                )
            elif scenario.name in mutation_hypotheses:
                hypotheses, selection = _select_from_hypotheses(
                    scenario,
                    mutation_hypotheses[scenario.name],
                    state,
                )
            else:
                hypotheses, selection = _select_hypotheses(scenario, state)
            if not hypotheses and selection.get("exhausted") is True:
                scenario_reports.append(
                    {
                        "ok": True,
                        "skipped": True,
                        "reason": "holdout_registry_exhausted",
                        "name": scenario.name,
                        "product": scenario.product,
                        "base_tf": scenario.base_tf,
                        "pnl_unit": scenario.pnl_unit,
                        "market": scenario.market,
                        "symbol": scenario.symbol,
                        "position": scenario.position,
                        "opportunity_type": scenario.opportunity_type,
                        "with_guards": scenario.with_guards,
                        "candidate_set": scenario.candidate_set,
                        "start": scenario.start,
                        "end": scenario.end,
                        "rows": 0,
                        "hypotheses": 0,
                        "keepers": 0,
                        "keeper_ids": [],
                        "holdout_exposed_ids": [],
                        "selection": selection,
                        "unsupported_hypotheses": [],
                        "verdicts": {},
                        "top_reasons": {"holdout_registry_exhausted": 1},
                    }
                )
                next_offsets[scenario.name] = int(selection.get("next_offset", 0))
                continue
            validation_kwargs: dict[str, Any] = {
                "hypotheses": hypotheses,
                "selection": selection,
                "log_path": log_path,
            }
            if (
                Path(research_factory_config_path).resolve()
                != Path(DEFAULT_RESEARCH_FACTORY_CONFIG).resolve()
            ):
                validation_kwargs["research_factory_config_path"] = research_factory_config_path
            if mutation_metadata.get(scenario.name):
                validation_kwargs["hypothesis_metadata"] = mutation_metadata[scenario.name]
            if generated_metadata.get(scenario.name):
                validation_kwargs["hypothesis_metadata"] = generated_metadata[scenario.name]
            if experiment_memory is not None:
                validation_kwargs["experiment_memory"] = experiment_memory
            scenario_report = run_validation_scenario(scenario, **validation_kwargs)
            scenario_reports.append(scenario_report)
            exposed_ids = {
                str(hypothesis_id)
                for hypothesis_id in scenario_report.get("holdout_exposed_ids") or []
                if hypothesis_id
            }
            if exposed_ids:
                consumed_holdout_registry.setdefault(scenario.name, set()).update(exposed_ids)
                state["consumed_holdout_ids"] = _serialized_holdout_registry(
                    consumed_holdout_registry
                )
            offset_key = "offset" if scenario_report.get("deferred") else "next_offset"
            next_offsets[scenario.name] = int(selection.get(offset_key, 0))
        except Exception as exc:
            scenario_reports.append(
                {
                    "ok": False,
                    "name": scenario.name,
                    "product": scenario.product,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if experiment_memory is not None:
        experiment_memory.close()
    report["scenarios"] = scenario_reports

    export_reports: list[dict[str, Any]] = []
    has_healthy_scenario = any(bool(item.get("ok")) for item in scenario_reports)
    for product, export_cfg in DEFAULT_EXPORTS.items() if has_healthy_scenario else ():
        product_config = configured_products.get(product)
        if product_config is None:
            export_reports.append(
                {
                    "ok": False,
                    "product": product,
                    "exported": False,
                    "reason": "product_not_configured",
                    "detail": f"{product}: no matching product in {config_path}",
                }
            )
            continue
        if product_config.execution_mode not in {"paper", "live"}:
            export_reports.append(
                {
                    "ok": False,
                    "product": product,
                    "exported": False,
                    "reason": "unsupported_execution_mode",
                    "detail": (
                        f"{product}: execution_mode must be paper or live, got "
                        f"{product_config.execution_mode!r}"
                    ),
                }
            )
            continue
        product_market = str(export_cfg["market"])
        is_live = product_config.execution_mode == "live"
        target_path = (
            candidate_path_for_product(product, candidate_dir=candidate_dir)
            if is_live
            else product_config.strategies_path
        )
        keeper_ids = _current_keeper_ids(
            scenario_reports,
            product=product,
            market=product_market,
            symbol=product_config.symbol,
        )
        if not keeper_ids:
            min_dsr = (
                float(export_cfg["min_dsr"]) if export_cfg.get("min_dsr") is not None else None
            )
            export_reports.append(
                {
                    "ok": True,
                    "product": product,
                    "pnl_unit": str(export_cfg["pnl_unit"]),
                    "market": product_market,
                    "exported": False,
                    "reason": "no_current_cycle_keepers",
                    "artifact": str(target_path),
                    "active_artifact": str(product_config.strategies_path),
                    "destination": "staging" if is_live else "active",
                    "staged": False,
                    "activation_required": is_live,
                    "ids": [],
                    "min_dsr": min_dsr,
                }
            )
            continue
        try:
            if is_live:
                export_report = stage_live_product_candidate(
                    product_config,
                    pnl_unit=str(export_cfg["pnl_unit"]),
                    market=product_market,
                    out=target_path,
                    top_k=int(export_cfg["top_k"]),
                    ids=keeper_ids,
                    min_dsr=(
                        float(export_cfg["min_dsr"])
                        if export_cfg.get("min_dsr") is not None
                        else None
                    ),
                    log_path=log_path,
                )
            else:
                export_report = export_product(
                    product,
                    pnl_unit=str(export_cfg["pnl_unit"]),
                    market=product_market,
                    out=target_path,
                    top_k=int(export_cfg["top_k"]),
                    ids=keeper_ids,
                    min_dsr=(
                        float(export_cfg["min_dsr"])
                        if export_cfg.get("min_dsr") is not None
                        else None
                    ),
                    log_path=log_path,
                    state_file=product_config.state_file,
                )
                export_report.update(
                    active_artifact=str(product_config.strategies_path),
                    destination="active",
                    staged=False,
                    activation_required=False,
                )
            export_reports.append(export_report)
        except Exception as exc:
            export_reports.append(
                {
                    "ok": False,
                    "product": product,
                    "exported": False,
                    "artifact": str(target_path),
                    "active_artifact": str(product_config.strategies_path),
                    "destination": "staging" if is_live else "active",
                    "activation_required": is_live,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    active_income_product = configured_products.get("active_income")
    if has_healthy_scenario and active_income_product is not None:
        export_reports.extend(
            _stage_active_income_symbol_candidates(
                scenario_reports,
                base_product=active_income_product,
                export_cfg=DEFAULT_EXPORTS["active_income"],
                candidate_dir=candidate_dir,
                log_path=log_path,
            )
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
        generated_batch_summary=generated_batch_summary,
    )
    report["ok"] = all(bool(item.get("ok")) for item in scenario_reports + export_reports)
    report["last_market_timestamp"] = last_timestamp
    report["last_market_marker"] = market_marker
    report["last_mutation_batch_marker"] = mutation_marker
    report["last_generated_batch_marker"] = generated_marker
    report["last_history_coverage_marker"] = history_coverage_marker
    serialized_holdout_registry = _serialized_holdout_registry(consumed_holdout_registry)

    if report["ok"]:
        write_json_atomic(
            state_path,
            {
                "version": 1,
                "last_market_timestamp": last_timestamp,
                "last_market_marker": market_marker,
                "last_mutation_batch_marker": mutation_marker,
                "last_generated_batch_marker": generated_marker,
                "last_history_coverage_marker": history_coverage_marker,
                "last_run_at": report["generated_at"],
                "scenario_offsets": next_offsets,
                "consumed_holdout_ids": serialized_holdout_registry,
            },
        )
    elif serialized_holdout_registry != initial_consumed_holdout_registry:
        partial_state = {
            key: state[key]
            for key in (
                "last_market_timestamp",
                "last_market_marker",
                "last_mutation_batch_marker",
                "last_generated_batch_marker",
                "last_history_coverage_marker",
                "last_run_at",
                "scenario_offsets",
            )
            if key in state
        }
        partial_state.update(
            version=1,
            consumed_holdout_ids=serialized_holdout_registry,
        )
        write_json_atomic(state_path, partial_state)
    if output_path:
        write_json_atomic(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded real-data research and gated export.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--force", action="store_true", help="Run even if market data timestamp is unchanged."
    )
    parser.add_argument(
        "--include-mutations",
        action="store_true",
        help="Validate research-only mutation hypotheses from --mutation-batch.",
    )
    parser.add_argument("--mutation-batch", type=Path, default=DEFAULT_MUTATION_BATCH)
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Validate the safe autonomous batch from --generated-batch.",
    )
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Disable the finite legacy scenarios and evaluate generated candidates only.",
    )
    parser.add_argument("--generated-batch", type=Path, default=DEFAULT_GENERATED_BATCH)
    parser.add_argument(
        "--research-factory-config",
        type=Path,
        default=DEFAULT_RESEARCH_FACTORY_CONFIG,
    )
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
        config_path=args.config,
        state_path=args.state,
        output_path=args.output,
        log_path=args.log,
        force=args.force,
        include_mutations=args.include_mutations,
        mutation_batch_path=args.mutation_batch,
        include_generated=args.include_generated,
        generated_only=args.generated_only,
        generated_batch_path=args.generated_batch,
        research_factory_config_path=args.research_factory_config,
        incubation_output_path=args.incubation_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
