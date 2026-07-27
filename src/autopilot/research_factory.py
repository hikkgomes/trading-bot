"""Resource-bounded autonomous strategy factory.

This is the native idea-production loop.  It samples a typed strategy grammar,
recursively mutates and crosses safe pre-holdout parents, incorporates optional
OpenClaw proposals as untrusted input, and persists every behavioral identity
before emitting a research batch.

It never backtests, paper trades, promotes, approves, or executes a strategy.
Those remain separate stages with independent gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import stat
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.strategy_grammar import (
    MOTIFS,
    GeneratedIdea,
    GrammarLimits,
    SearchSpace,
    build_fresh_hypothesis,
    crossover_hypotheses,
    mutate_hypothesis,
    structural_tokens,
    validate_hypothesis_against_space,
)
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.experiment_memory import (
    ExperimentMemory,
    ExperimentMemoryError,
    canonical_strategy_hash,
)
from src.autopilot.io import write_json_atomic
from src.autopilot.openclaw_bridge import ACCEPTED_SCHEMA
from src.config import PROJECT_ROOT, indicator_data_dir

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_factory.json"
DEFAULT_PROPOSAL_STATE = PROJECT_ROOT / "runtime" / "research" / "openclaw_proposal_state.json"
DEFAULT_MARKET_UNIVERSE_REPORT = PROJECT_ROOT / "runtime" / "market_universe.json"
BATCH_SCHEMA = "autopilot.generative_strategy_batch/v1"
REPORT_SCHEMA = "autopilot.research_factory_report/v1"
MAX_CONFIG_BYTES = 256 * 1024
MAX_ACCEPTED_PROPOSAL_BYTES = 64 * 1024
MAX_PROPOSAL_FILES_PER_CYCLE = 100
MAX_PROPOSAL_FILES_SCANNED = 20_000
MAX_PROPOSAL_STATE_ITEMS = 10_000
MAX_SUBMITTED_SPEC_KEYS = 32
MEMORY_COMPACTION_TRIGGER_FRACTION = 0.80
MAX_MEMORY_COMPACTION_ROWS_PER_CYCLE = 5_000
SEARCH_SPACE_FIELDS = frozenset(
    {
        "name",
        "product",
        "market",
        "pnl_unit",
        "opportunity_type",
        "base_timeframe",
        "regime_timeframe",
        "setup_timeframe",
        "trigger_timeframe",
        "directions",
        "take_profit_range",
        "stop_loss_range",
        "horizon_range",
        "risk_per_trade_range",
        "max_position_fraction",
        "max_trades_per_day",
        "symbol",
    }
)
SEARCH_SPACE_CONFIG_FIELDS = SEARCH_SPACE_FIELDS | {"symbols"}
REQUIRED_SEARCH_SPACE_FIELDS = SEARCH_SPACE_FIELDS - {"max_trades_per_day", "symbol"}

SAFETY = {
    "research_only": True,
    "executable": False,
    "paper_trade_allowed": False,
    "promotion_allowed": False,
    "live_allowed": False,
    "requires_full_validation_before_export": True,
}

SPEC_KEYS = frozenset(
    {
        "base_timeframe",
        "direction",
        "exit",
        "expected_frequency",
        "expected_holding",
        "family",
        "feature_columns",
        "id",
        "idea",
        "invalidation",
        "market_logic",
        "regime",
        "regime_timeframe",
        "risk",
        "setup",
        "setup_timeframe",
        "tags",
        "trigger",
        "trigger_timeframe",
    }
)
OPPORTUNITY_ALIASES = {
    "scalp": "scalping",
    "day": "day_trading",
    "swing": "swing_trading",
    "position": "btc_accumulation",
}


class ResearchFactoryConfigError(ValueError):
    """The autonomous research-factory configuration is unsafe or malformed."""


class UntrustedProposalCompileError(ValueError):
    """An accepted OpenClaw envelope cannot compile into the trusted grammar."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ResearchFactoryConfigError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise ResearchFactoryConfigError(f"non-standard JSON constant: {value}")


def _strict_json_file(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ResearchFactoryConfigError(f"{label} must not be a symlink: {path}")
    if not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise ResearchFactoryConfigError(f"{label} must be a regular file: {path}")
    if path.stat().st_size > maximum_bytes:
        raise ResearchFactoryConfigError(f"{label} exceeds {maximum_bytes} bytes: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchFactoryConfigError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchFactoryConfigError(f"{label} must be a JSON object: {path}")
    return payload


def _project_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ResearchFactoryConfigError(f"{label} must be a non-empty project-relative path")
    raw = Path(value)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ResearchFactoryConfigError(f"{label} must stay inside the repository") from exc
    return resolved


def _positive_int(value: Any, *, label: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise ResearchFactoryConfigError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _fraction(value: Any, *, label: str, allow_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResearchFactoryConfigError(f"{label} must be numeric")
    number = float(value)
    upper_ok = number <= 1 if allow_one else number < 1
    if not math.isfinite(number) or number < 0 or not upper_ok:
        bracket = "[0, 1]" if allow_one else "[0, 1)"
        raise ResearchFactoryConfigError(f"{label} must be in {bracket}")
    return number


@dataclass(frozen=True)
class FactoryBudgets:
    max_candidates_per_cycle: int
    max_candidates_per_space: int
    max_generation_attempts: int
    max_generation_seconds: float
    max_parent_pool: int
    max_lineage_depth: int
    max_total_predicates: int
    max_memory_bytes: int
    near_duplicate_threshold: float
    exploration_fraction: float
    mutation_fraction: float
    crossover_fraction: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FactoryBudgets:
        allowed = {
            "max_candidates_per_cycle",
            "max_candidates_per_space",
            "max_generation_attempts",
            "max_generation_seconds",
            "max_parent_pool",
            "max_lineage_depth",
            "max_total_predicates",
            "max_memory_bytes",
            "near_duplicate_threshold",
            "exploration_fraction",
            "mutation_fraction",
            "crossover_fraction",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ResearchFactoryConfigError(f"budgets has unknown fields: {', '.join(unknown)}")
        seconds = payload.get("max_generation_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise ResearchFactoryConfigError("max_generation_seconds must be numeric")
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 0 < seconds <= 300:
            raise ResearchFactoryConfigError("max_generation_seconds must be in (0, 300]")
        result = cls(
            max_candidates_per_cycle=_positive_int(
                payload.get("max_candidates_per_cycle"),
                label="max_candidates_per_cycle",
                maximum=200,
            ),
            max_candidates_per_space=_positive_int(
                payload.get("max_candidates_per_space"),
                label="max_candidates_per_space",
                maximum=50,
            ),
            max_generation_attempts=_positive_int(
                payload.get("max_generation_attempts"),
                label="max_generation_attempts",
                maximum=100_000,
            ),
            max_generation_seconds=seconds,
            max_parent_pool=_positive_int(
                payload.get("max_parent_pool"), label="max_parent_pool", maximum=500
            ),
            max_lineage_depth=_positive_int(
                payload.get("max_lineage_depth"), label="max_lineage_depth", maximum=32
            ),
            max_total_predicates=_positive_int(
                payload.get("max_total_predicates"),
                label="max_total_predicates",
                maximum=16,
            ),
            max_memory_bytes=_positive_int(
                payload.get("max_memory_bytes"),
                label="max_memory_bytes",
                maximum=2 * 1024 * 1024 * 1024,
            ),
            near_duplicate_threshold=_fraction(
                payload.get("near_duplicate_threshold"),
                label="near_duplicate_threshold",
                allow_one=False,
            ),
            exploration_fraction=_fraction(
                payload.get("exploration_fraction"), label="exploration_fraction"
            ),
            mutation_fraction=_fraction(
                payload.get("mutation_fraction"), label="mutation_fraction"
            ),
            crossover_fraction=_fraction(
                payload.get("crossover_fraction"), label="crossover_fraction"
            ),
        )
        if not math.isclose(
            result.exploration_fraction + result.mutation_fraction + result.crossover_fraction,
            1.0,
            abs_tol=1e-9,
        ):
            raise ResearchFactoryConfigError("generation fractions must sum to 1")
        if result.exploration_fraction < 0.25:
            raise ResearchFactoryConfigError(
                "exploration_fraction must preserve a 25% exploration floor"
            )
        return result


@dataclass(frozen=True)
class ResearchFactoryConfig:
    path: Path
    memory_path: Path
    generated_batch_path: Path
    openclaw_accepted_dir: Path
    proposal_state_path: Path
    budgets: FactoryBudgets
    search_spaces: tuple[SearchSpace, ...]
    dynamic_active_income_universe: bool = False


def load_factory_config(path: Path = DEFAULT_CONFIG) -> ResearchFactoryConfig:
    path = Path(path)
    payload = _strict_json_file(path, maximum_bytes=MAX_CONFIG_BYTES, label="research config")
    allowed = {
        "version",
        "memory_path",
        "generated_batch_path",
        "openclaw_accepted_dir",
        "openclaw_proposal_state_path",
        "budgets",
        "holdout_policy",
        "search_spaces",
        "dynamic_active_income_universe",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ResearchFactoryConfigError(
            f"research config has unknown fields: {', '.join(unknown)}"
        )
    if payload.get("version") != 1:
        raise ResearchFactoryConfigError("research config version must be 1")
    dynamic_universe = payload.get("dynamic_active_income_universe", False)
    if not isinstance(dynamic_universe, bool):
        raise ResearchFactoryConfigError("dynamic_active_income_universe must be a boolean")
    holdout = payload.get("holdout_policy")
    required_holdout = {
        "canonical_behavior_claims": True,
        "retire_exposed_lineages": True,
        "exclude_holdout_results_from_generation": True,
        "paper_forward_test_is_final_adaptive_free_gate": True,
    }
    if holdout != required_holdout:
        raise ResearchFactoryConfigError(
            "holdout_policy must retain every fail-closed protected-data invariant"
        )
    spaces_payload = payload.get("search_spaces")
    if not isinstance(spaces_payload, list) or not spaces_payload:
        raise ResearchFactoryConfigError("search_spaces must be a non-empty list")
    spaces_list: list[SearchSpace] = []
    for index, item in enumerate(spaces_payload):
        if not isinstance(item, Mapping):
            raise ResearchFactoryConfigError(f"search_spaces[{index}] must be an object")
        unknown_space_fields = sorted(set(item) - SEARCH_SPACE_CONFIG_FIELDS)
        if unknown_space_fields:
            raise ResearchFactoryConfigError(
                f"search_spaces[{index}] has unknown fields: {', '.join(unknown_space_fields)}"
            )
        missing_space_fields = sorted(REQUIRED_SEARCH_SPACE_FIELDS - set(item))
        if missing_space_fields:
            raise ResearchFactoryConfigError(
                f"search_spaces[{index}] is missing fields: {', '.join(missing_space_fields)}"
            )
        raw_symbols = item.get("symbols")
        if raw_symbols is not None and "symbol" in item:
            raise ResearchFactoryConfigError(
                f"search_spaces[{index}] cannot define both symbol and symbols"
            )
        if raw_symbols is None:
            symbols = [item.get("symbol", "BTCUSDT")]
        elif (
            not isinstance(raw_symbols, list)
            or not raw_symbols
            or any(not isinstance(value, str) or not value.strip() for value in raw_symbols)
        ):
            raise ResearchFactoryConfigError(
                f"search_spaces[{index}].symbols must be a non-empty list of strings"
            )
        else:
            symbols = raw_symbols
        try:
            for raw_symbol in symbols:
                expanded = {key: value for key, value in item.items() if key != "symbols"}
                expanded["symbol"] = str(raw_symbol).strip().upper()
                if len(symbols) > 1 and expanded["symbol"] != "BTCUSDT":
                    expanded["name"] = f"{item['name']}_{expanded['symbol'].lower()}"
                spaces_list.append(SearchSpace.from_dict(expanded))
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchFactoryConfigError(f"search_spaces[{index}] is invalid: {exc}") from exc
    spaces = tuple(spaces_list)
    names = [space.name for space in spaces]
    if len(set(names)) != len(names):
        raise ResearchFactoryConfigError("search-space names must be unique")
    required_products = {"active_income", "btc_accumulation"}
    products = {space.product for space in spaces}
    if products != required_products:
        raise ResearchFactoryConfigError(
            "search spaces must cover exactly active_income and btc_accumulation"
        )
    for space in spaces:
        if space.product == "active_income" and (
            space.market != "futures" or space.pnl_unit != "usdt"
        ):
            raise ResearchFactoryConfigError(
                f"{space.name}: active_income must use futures market and usdt PnL"
            )
        if space.product == "btc_accumulation" and (
            space.market != "spot"
            or space.pnl_unit != "btc"
            or space.symbol != "BTCUSDT"
            or space.opportunity_type != "btc_accumulation"
            or set(space.directions) != {"short"}
        ):
            raise ResearchFactoryConfigError(
                f"{space.name}: btc_accumulation must use spot/btc, short-only step-aside research"
            )
    required_active_horizons = {
        "scalping": "1m",
        "day_trading": "5m",
        "swing_trading": "1h",
    }
    active_spaces = [space for space in spaces if space.product == "active_income"]
    for symbol in sorted({space.symbol for space in active_spaces}):
        symbol_spaces = [space for space in active_spaces if space.symbol == symbol]
        opportunities = {space.opportunity_type for space in symbol_spaces}
        if opportunities != set(required_active_horizons):
            raise ResearchFactoryConfigError(
                f"{symbol} active-income spaces must cover exactly scalp, day, and swing research"
            )
        for opportunity, expected_base in required_active_horizons.items():
            configured_bases = {
                space.base_timeframe
                for space in symbol_spaces
                if space.opportunity_type == opportunity
            }
            if configured_bases != {expected_base}:
                raise ResearchFactoryConfigError(
                    f"{symbol} {opportunity} search spaces must use base_timeframe {expected_base}"
                )
    btc_bases = {space.base_timeframe for space in spaces if space.product == "btc_accumulation"}
    if btc_bases != {"1h", "4h"}:
        raise ResearchFactoryConfigError(
            "btc_accumulation search spaces must cover exactly 1h and 4h base timeframes"
        )
    return ResearchFactoryConfig(
        path=path.resolve(),
        memory_path=_project_path(payload.get("memory_path"), label="memory_path"),
        generated_batch_path=_project_path(
            payload.get("generated_batch_path"), label="generated_batch_path"
        ),
        openclaw_accepted_dir=_project_path(
            payload.get("openclaw_accepted_dir"), label="openclaw_accepted_dir"
        ),
        proposal_state_path=_project_path(
            payload.get(
                "openclaw_proposal_state_path",
                "runtime/research/openclaw_proposal_state.json",
            ),
            label="openclaw_proposal_state_path",
        ),
        budgets=FactoryBudgets.from_dict(payload.get("budgets") or {}),
        search_spaces=spaces,
        dynamic_active_income_universe=dynamic_universe,
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _market_universe_context(
    *,
    generated_at: str,
    report_path: Path = DEFAULT_MARKET_UNIVERSE_REPORT,
) -> dict[str, Any]:
    """Return a fresh, immutable research-universe selection or the BTC fallback."""

    context: dict[str, Any] = {
        "eligible_symbols": ["BTCUSDT"],
        "snapshot_id": None,
        "generated_at": None,
        "selection_mode": "btc_fallback",
        "fresh": False,
    }
    try:
        report = (
            _strict_json_file(
                report_path,
                maximum_bytes=2 * 1024 * 1024,
                label="market universe report",
            )
            if report_path.exists()
            else {}
        )
    except ResearchFactoryConfigError:
        report = {}
    snapshot = report.get("snapshot")
    if (
        report.get("ok") is True
        and report.get("schema") == "autopilot.market_universe/v2"
        and isinstance(snapshot, Mapping)
        and isinstance(snapshot.get("id"), str)
        and str(snapshot["id"]).startswith("sha256:")
    ):
        try:
            report_at = datetime.fromisoformat(str(report["generated_at"]).replace("Z", "+00:00"))
            cycle_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if abs((cycle_at - report_at).total_seconds()) <= 48 * 3600:
                eligible = {
                    str(symbol).upper()
                    for symbol in report.get("eligible_research_symbols") or []
                    if isinstance(symbol, str) and symbol.upper().endswith("USDT")
                }
                context = {
                    "eligible_symbols": sorted(eligible),
                    "snapshot_id": snapshot.get("id"),
                    "generated_at": report.get("generated_at"),
                    "selection_mode": report.get("selection_mode") or "screened",
                    "fresh": True,
                }
        except (KeyError, TypeError, ValueError):
            pass
    return context


def _symbol_space(template: SearchSpace, symbol: str) -> SearchSpace:
    symbol = symbol.upper()
    name = template.name if symbol == template.symbol else f"{template.name}_{symbol.lower()}"
    return dataclasses.replace(template, name=name, symbol=symbol)


def search_spaces_for_symbol(
    config: ResearchFactoryConfig,
    symbol: str,
) -> tuple[SearchSpace, ...]:
    """Resolve the configured research templates applicable to one symbol."""

    symbol = symbol.upper()
    if config.dynamic_active_income_universe and symbol != "BTCUSDT":
        return tuple(
            _symbol_space(space, symbol)
            for space in config.search_spaces
            if space.product == "active_income"
        )
    return tuple(space for space in config.search_spaces if space.symbol == symbol)


def resolve_search_space(
    config: ResearchFactoryConfig,
    metadata: Mapping[str, Any],
) -> SearchSpace:
    """Resolve a batch search space, including a dynamically expanded symbol."""

    requested_name = str(metadata.get("search_space") or "")
    exact = next((space for space in config.search_spaces if space.name == requested_name), None)
    if exact is not None:
        return exact
    if not config.dynamic_active_income_universe:
        raise KeyError(requested_name)
    symbol = str(metadata.get("symbol") or "").upper()
    matches = [
        space
        for space in config.search_spaces
        if space.product == "active_income"
        and space.opportunity_type == metadata.get("opportunity_type")
        and space.base_timeframe == metadata.get("base_timeframe")
    ]
    if len(matches) != 1:
        raise KeyError(requested_name)
    expanded = _symbol_space(matches[0], symbol)
    if expanded.name != requested_name:
        raise KeyError(requested_name)
    return expanded


def _search_spaces_for_cycle(
    config: ResearchFactoryConfig,
    *,
    generated_at: str,
    report_path: Path = DEFAULT_MARKET_UNIVERSE_REPORT,
) -> tuple[SearchSpace, ...]:
    """Gate and optionally expand active-income research from the daily screen."""

    context = _market_universe_context(generated_at=generated_at, report_path=report_path)
    eligible = set(context["eligible_symbols"])
    if config.dynamic_active_income_universe:
        templates = [space for space in config.search_spaces if space.product == "active_income"]
        expanded = [
            _symbol_space(template, symbol) for symbol in sorted(eligible) for template in templates
        ]
        return (
            *(space for space in config.search_spaces if space.product == "btc_accumulation"),
            *expanded,
        )
    eligible = eligible or {"BTCUSDT"}
    return tuple(
        space
        for space in config.search_spaces
        if space.product == "btc_accumulation" or space.symbol in eligible
    )


def _seed_for_cycle(config: ResearchFactoryConfig, now: str, explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    day = now[:10]
    digest = hashlib.sha256(f"{day}:{config.path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def strategy_behavior_spec(hypothesis: Hypothesis, space: SearchSpace) -> dict[str, Any]:
    # Context is part of behavior identity.  Keep it at top level so the
    # canonicalizer can ignore the hypothesis' display ID/prose while retaining
    # product/market/PnL semantics.
    return {
        **hypothesis.to_dict(),
        "_product": space.product,
        "_market": space.market,
        "_pnl_unit": space.pnl_unit,
        "_opportunity_type": space.opportunity_type,
        "_search_space": space.name,
        "_symbol": space.symbol,
    }


def _hypothesis_from_submitted_spec(payload: Mapping[str, Any]) -> Hypothesis:
    return Hypothesis.from_dict(
        {key: value for key, value in payload.items() if not key.startswith("_")}
    )


def _feature_inventory_for_space(space: SearchSpace) -> dict[str, set[str]] | None:
    directory = indicator_data_dir(space.symbol, space.market, legacy_fallback=True)
    inventory: dict[str, set[str]] = {}
    for timeframe in {
        space.base_timeframe,
        space.regime_timeframe,
        space.setup_timeframe,
        space.trigger_timeframe,
    }:
        path = directory / f"{space.symbol}_{timeframe}_all_indicators.parquet"
        if not path.exists() or path.is_symlink():
            return None
        try:
            inventory[timeframe] = {
                name for name in pq.ParquetFile(path).schema_arrow.names if name != "timestamp"
            }
        except Exception:
            return None
    return inventory


def _feedback_weights(feedback: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    primitives = feedback.get("primitives")
    if not isinstance(primitives, Mapping):
        return result
    for key, raw in primitives.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            continue
        experiments = int(raw.get("experiments") or 0)
        outcomes = raw.get("outcomes") if isinstance(raw.get("outcomes"), Mapping) else {}
        pre_holdout_passes = int(outcomes.get("pre_holdout_pass") or 0)
        inconclusive = int(outcomes.get("inconclusive") or 0)
        # Beta prior + small credit for surviving longer without inspecting
        # protected holdout results.  Exploration remains guaranteed elsewhere.
        posterior = (1.0 + 2.0 * pre_holdout_passes + 0.25 * inconclusive) / (
            2.0 + max(0, experiments)
        )
        result[key] = min(4.0, max(0.2, posterior))
    return result


def _proposal_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "processed": {}}
    payload = _strict_json_file(path, maximum_bytes=2 * 1024 * 1024, label="proposal state")
    processed = payload.get("processed")
    if payload.get("version") != 1 or not isinstance(processed, dict):
        raise ResearchFactoryConfigError("proposal state has invalid schema")
    return payload


def _save_proposal_state(path: Path, state: dict[str, Any]) -> None:
    processed = state.get("processed") if isinstance(state.get("processed"), dict) else {}
    if len(processed) > MAX_PROPOSAL_STATE_ITEMS:
        newest = sorted(
            processed.items(),
            key=lambda item: str(
                (item[1] if isinstance(item[1], dict) else {}).get("processed_at", "")
            ),
            reverse=True,
        )[:MAX_PROPOSAL_STATE_ITEMS]
        state["processed"] = dict(newest)
    write_json_atomic(path, state)


def _load_accepted_proposals(directory: Path, processed: set[str]) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ResearchFactoryConfigError(f"OpenClaw accepted path must be a directory: {directory}")
    proposals: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        if index >= MAX_PROPOSAL_FILES_SCANNED or len(proposals) >= MAX_PROPOSAL_FILES_PER_CYCLE:
            break
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_ACCEPTED_PROPOSAL_BYTES
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("schema") != ACCEPTED_SCHEMA:
            continue
        proposal_id = payload.get("proposal_id")
        safety = payload.get("safety")
        if not isinstance(proposal_id, str) or proposal_id in processed:
            continue
        if safety != {
            "research_only": True,
            "executable": False,
            "paper_trade_allowed": False,
            "promotion_allowed": False,
            "live_allowed": False,
            "requires_trusted_compilation": True,
            "requires_full_validation_before_export": True,
        }:
            continue
        proposals.append(payload)
    return proposals


def _purge_processed_proposals(directory: Path, processed: set[str]) -> int:
    """Remove accepted-spool files only after their disposition is durable.

    The raw OpenClaw envelope remains in the bridge archive. A crash after the
    state write but before this unlink is repaired by the next factory cycle.
    """

    if not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ResearchFactoryConfigError(f"OpenClaw accepted path must be a directory: {directory}")
    removed = 0
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        if index >= MAX_PROPOSAL_FILES_SCANNED:
            break
        try:
            before = path.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_size > MAX_ACCEPTED_PROPOSAL_BYTES
            ):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            after = path.lstat()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or not isinstance(payload, dict):
            continue
        proposal_id = payload.get("proposal_id")
        if payload.get("schema") != ACCEPTED_SCHEMA or proposal_id not in processed:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


def _space_for_proposal(
    proposal: Mapping[str, Any], spaces: Sequence[SearchSpace]
) -> SearchSpace | None:
    product = proposal.get("objective")
    opportunity = OPPORTUNITY_ALIASES.get(str(proposal.get("opportunity_type")))
    base = proposal.get("base_timeframe")
    symbol = str(proposal.get("symbol") or "BTCUSDT")
    matches = [
        space
        for space in spaces
        if space.product == product
        and space.opportunity_type == opportunity
        and space.base_timeframe == base
        and space.symbol == symbol
    ]
    return matches[0] if len(matches) == 1 else None


def _motif_for_proposal(proposal: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(proposal.get("thesis") or ""),
            *[
                str(item)
                for item in proposal.get("suggested_primitives") or []
                if isinstance(item, str)
            ],
        ]
    ).lower()
    rules = (
        ("orderflow_confirmation", ("order flow", "orderflow", "taker", "volume", "liquidity")),
        ("volatility_transition", ("volatility", "compression", "quiet", "squeeze")),
        ("range_expansion", ("breakout", "range", "expansion")),
        ("countertrend_reversion", ("reversion", "oversold", "overbought", "reclaim")),
        ("trend_following", ("trend", "continuation", "momentum")),
    )
    for motif, markers in rules:
        if any(marker in text for marker in markers):
            return motif
    return "hybrid"


def _compile_openclaw_proposal(
    proposal: Mapping[str, Any],
    space: SearchSpace,
    *,
    rng: random.Random,
    available_features: Mapping[str, Iterable[str]] | None,
    feedback_weights: Mapping[str, float],
    limits: GrammarLimits,
) -> GeneratedIdea:
    untrusted = proposal.get("untrusted_suggested_spec")
    if not isinstance(untrusted, dict):
        raise UntrustedProposalCompileError("untrusted_suggested_spec must be an object")
    if len(untrusted) > MAX_SUBMITTED_SPEC_KEYS:
        raise UntrustedProposalCompileError("suggested spec has too many top-level fields")
    if untrusted:
        unknown = sorted(set(untrusted) - SPEC_KEYS)
        required = {
            "direction",
            "base_timeframe",
            "regime_timeframe",
            "setup_timeframe",
            "trigger_timeframe",
            "regime",
            "setup",
            "trigger",
            "exit",
        }
        if unknown:
            raise UntrustedProposalCompileError(
                f"suggested spec has unknown fields: {', '.join(unknown)}"
            )
        if required.issubset(untrusted):
            trusted = dict(untrusted)
            trusted.update(
                id="OPENCLAW_PENDING_ID",
                family="generated_openclaw",
                idea=str(proposal.get("thesis") or "OpenClaw research proposal")[:1000],
                market_logic=(
                    "An optional OpenClaw research suggestion compiled into the same bounded, "
                    "non-executable strategy grammar as native ideas."
                ),
                expected_holding=space.opportunity_type.replace("_", " "),
                expected_frequency="unknown until measured on training data",
                invalidation="trusted risk/exit rules or loss of compiled entry conditions",
                tags=["autonomous_generation", "openclaw_proposal", space.name],
            )
            try:
                hypothesis = Hypothesis.from_dict(trusted)
            except ExperimentMemoryError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise UntrustedProposalCompileError(
                    f"suggested spec failed trusted schema compilation: {type(exc).__name__}: {exc}"
                ) from exc
            problems = validate_hypothesis_against_space(
                hypothesis,
                space,
                available_features=available_features,
                limits=limits,
            )
            if problems:
                raise UntrustedProposalCompileError(
                    f"suggested spec violates search-space limits: {', '.join(problems)}"
                )
            return GeneratedIdea(
                hypothesis=hypothesis,
                generation_method="openclaw_compiled_proposal",
                grammar_keys=tuple(
                    str(item)[:128]
                    for item in proposal.get("suggested_primitives") or []
                    if isinstance(item, str)
                ),
                motif=_motif_for_proposal(proposal),
            )
    # A high-level thesis is a seed, not executable input.  Native grammar owns
    # the actual structure, so OpenClaw can be absent or wrong without widening
    # the trusted language.
    idea = build_fresh_hypothesis(
        space,
        rng=rng,
        available_features=available_features,
        feedback_weights=feedback_weights,
        limits=limits,
        motif=_motif_for_proposal(proposal),
    )
    return dataclasses.replace(idea, generation_method="openclaw_seeded_grammar")


def _parent_hypothesis(parent: Mapping[str, Any]) -> Hypothesis | None:
    cached = parent.get("_parsed_hypothesis")
    if isinstance(cached, Hypothesis):
        return cached
    submitted = parent.get("submitted_spec")
    if not isinstance(submitted, Mapping):
        return None
    try:
        parsed = _hypothesis_from_submitted_spec(submitted)
        if isinstance(parent, dict):
            parent["_parsed_hypothesis"] = parsed
        return parsed
    except Exception:
        return None


def _lineage_depth(parent: Mapping[str, Any]) -> int:
    metadata = parent.get("metadata")
    if not isinstance(metadata, Mapping):
        return 0
    try:
        return max(0, int(metadata.get("lineage_depth") or 0))
    except (TypeError, ValueError):
        return 0


def _adaptive_method_score(
    method: str,
    feedback: Mapping[str, Any] | None,
) -> float:
    """Score an evolutionary operator using development-only evidence.

    ``generator_feedback`` excludes protected phases by contract.  Beta-style
    smoothing prevents a lucky first result from dominating, while duplicate
    rate and novelty keep the factory from repeatedly exploring one corner.
    """

    methods = feedback.get("generation_methods") if isinstance(feedback, Mapping) else None
    raw = methods.get(method) if isinstance(methods, Mapping) else None
    if not isinstance(raw, Mapping):
        return 1.0
    experiments = max(0, int(raw.get("experiments") or 0))
    outcomes = raw.get("outcomes") if isinstance(raw.get("outcomes"), Mapping) else {}
    passes = max(0, int(outcomes.get("pre_holdout_pass") or 0))
    success = (1.0 + passes) / (2.0 + experiments)
    proposals = max(0, int(raw.get("proposals") or 0))
    duplicates = min(proposals, max(0, int(raw.get("duplicates") or 0)))
    uniqueness = (1.0 + proposals - duplicates) / (2.0 + proposals)
    novelty = max(0.0, min(1.0, float(raw.get("mean_novelty") or 0.0)))
    return max(0.25, min(4.0, (0.5 + success) * (0.5 + uniqueness) * (0.75 + novelty)))


def _method_schedule(
    total: int,
    budgets: FactoryBudgets,
    rng: random.Random,
    *,
    feedback: Mapping[str, Any] | None = None,
) -> list[str]:
    exploration = math.ceil(total * budgets.exploration_fraction)
    remaining = max(0, total - exploration)
    mutation_weight = budgets.mutation_fraction * _adaptive_method_score(
        "recursive_mutation", feedback
    )
    crossover_weight = budgets.crossover_fraction * _adaptive_method_score("crossover", feedback)
    mutable_total = mutation_weight + crossover_weight
    mutations = round(remaining * mutation_weight / mutable_total) if mutable_total else 0
    schedule = ["grammar_sample"] * exploration
    schedule.extend(["recursive_mutation"] * mutations)
    schedule.extend(["crossover"] * (total - len(schedule)))
    rng.shuffle(schedule)
    return schedule


def _parent_weight(
    parent: Mapping[str, Any],
    parent_feedback: Mapping[str, Mapping[str, Any]],
) -> float:
    latest = parent.get("latest_evaluation")
    latest_outcome = latest.get("outcome") if isinstance(latest, Mapping) else None
    own_score = {"pre_holdout_pass": 3.0, "inconclusive": 1.0, "reject": 0.5}.get(
        str(latest_outcome),
        1.0,
    )
    raw = parent_feedback.get(str(parent.get("behavior_hash") or ""), {})
    children = max(0, int(raw.get("children") or 0))
    outcomes = raw.get("child_outcomes") if isinstance(raw.get("child_outcomes"), Mapping) else {}
    successful_children = max(0, int(outcomes.get("pre_holdout_pass") or 0))
    child_score = (1.0 + successful_children) / (2.0 + children)
    novelty = max(0.0, min(1.0, float(parent.get("novelty_score") or 0.0)))
    return max(0.1, own_score * (0.5 + child_score) * (0.75 + novelty))


def _choose_parent(
    parents: Sequence[Mapping[str, Any]],
    *,
    feedback: Mapping[str, Any],
    rng: random.Random,
) -> Mapping[str, Any]:
    feedback_by_hash = _parent_feedback_by_hash(feedback)
    weights = [_parent_weight(parent, feedback_by_hash) for parent in parents]
    return rng.choices(list(parents), weights=weights, k=1)[0]


def _parent_feedback_by_hash(
    feedback: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    parent_rows = feedback.get("parent_performance")
    return (
        {
            str(item.get("parent_hash")): item
            for item in parent_rows
            if isinstance(item, Mapping) and item.get("parent_hash")
        }
        if isinstance(parent_rows, Sequence) and not isinstance(parent_rows, str | bytes)
        else {}
    )


def _candidate_is_near_duplicate(
    hypothesis: Hypothesis,
    candidates: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    include_values: bool,
    excluded_hashes: frozenset[str] = frozenset(),
) -> tuple[bool, str | None, float]:
    nearest_hash = None
    highest = 0.0
    target_tokens = structural_tokens(hypothesis, include_values=include_values)
    cache_key = "_structural_tokens_values" if include_values else "_structural_tokens_shape"
    for candidate in candidates:
        candidate_hash = str(candidate.get("behavior_hash") or "")
        if candidate_hash in excluded_hashes:
            continue
        prior = _parent_hypothesis(candidate)
        if prior is None:
            continue
        cached_tokens = candidate.get(cache_key)
        if isinstance(cached_tokens, frozenset):
            prior_tokens = set(cached_tokens)
        else:
            prior_tokens = structural_tokens(prior, include_values=include_values)
            if isinstance(candidate, dict):
                candidate[cache_key] = frozenset(prior_tokens)
        union = target_tokens | prior_tokens
        similarity = len(target_tokens & prior_tokens) / len(union) if union else 1.0
        if similarity > highest:
            highest = similarity
            nearest_hash = candidate_hash or None
    return highest >= threshold, nearest_hash, highest


def _try_register(
    memory: ExperimentMemory,
    idea: GeneratedIdea,
    space: SearchSpace,
    *,
    parents: Sequence[Mapping[str, Any]],
    dedup_population: list[dict[str, Any]],
    budgets: FactoryBudgets,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    behavior = strategy_behavior_spec(idea.hypothesis, space)
    duplicate = memory.find_duplicate(behavior)
    if duplicate is not None:
        return None, {"reason": "exact_duplicate", "behavior_hash": duplicate}
    # Fresh roots are compared structurally against *all* prior product
    # behaviors, including retired/holdout-exposed work. Descendants may make
    # parameter-only changes to their direct parents, but are compared with
    # values against every other historical branch.
    parent_hashes = frozenset(idea.parent_hashes)
    near, nearest_hash, similarity = _candidate_is_near_duplicate(
        idea.hypothesis,
        dedup_population,
        budgets.near_duplicate_threshold,
        include_values=bool(parent_hashes),
        excluded_hashes=parent_hashes,
    )
    if near:
        return None, {
            "reason": "near_duplicate",
            "nearest_behavior_hash": nearest_hash,
            "structural_similarity": round(similarity, 6),
        }
    behavior_hash = canonical_strategy_hash(behavior)
    strategy_id = f"GEN_{space.name.upper()}_{behavior_hash[7:23]}"
    hypothesis = dataclasses.replace(idea.hypothesis, id=strategy_id)
    behavior = strategy_behavior_spec(hypothesis, space)
    parent_depths = [
        _lineage_depth(parent)
        for parent in parents
        if parent.get("behavior_hash") in set(idea.parent_hashes)
    ]
    lineage_depth = (max(parent_depths) + 1) if parent_depths else 0
    if lineage_depth > budgets.max_lineage_depth:
        return None, {"reason": "lineage_depth_limit", "lineage_depth": lineage_depth}
    metadata = {
        "family": hypothesis.family,
        "product": space.product,
        "market": space.market,
        "pnl_unit": space.pnl_unit,
        "opportunity_type": space.opportunity_type,
        "search_space": space.name,
        "symbol": space.symbol,
        "base_timeframe": space.base_timeframe,
        "generation_method": idea.generation_method,
        "grammar_keys": list(idea.grammar_keys),
        "primitives": list(idea.grammar_keys),
        "motif": idea.motif,
        "adaptation_reasons": list(idea.adaptation_reasons),
        "lineage_depth": lineage_depth,
        **dict(extra_metadata or {}),
    }
    registration = memory.register_strategy(
        behavior,
        strategy_id=strategy_id,
        generation_method=idea.generation_method,
        parent_hashes=idea.parent_hashes,
        metadata=metadata,
    )
    if registration.duplicate:
        return None, {
            "reason": "registration_duplicate",
            "behavior_hash": registration.behavior_hash,
        }
    dedup_population.append(
        {
            "behavior_hash": registration.behavior_hash,
            "submitted_spec": behavior,
            "metadata": metadata,
        }
    )
    return {
        "hypothesis": hypothesis,
        "metadata": {
            **metadata,
            "id": strategy_id,
            "strategy_hash": registration.behavior_hash,
            "parent_hashes": list(idea.parent_hashes),
            "novelty_score": round(registration.novelty_score, 8),
            "nearest_behavior_hash": registration.nearest_behavior_hash,
        },
    }, None


def _pending_for_space(
    memory: ExperimentMemory,
    space: SearchSpace,
    *,
    limit: int,
    research_engine_digest: str,
) -> list[dict[str, Any]]:
    pending_method = getattr(memory, "pending_strategies", None)
    if callable(pending_method):
        pending = list(
            pending_method(
                product=space.product,
                opportunity_type=space.opportunity_type,
                limit=limit,
                research_engine_digest=research_engine_digest,
            )
        )
        return [
            item
            for item in pending
            if (item.get("metadata") or {}).get("symbol", "BTCUSDT") == space.symbol
        ]
    candidates = memory.candidate_parents(
        product=space.product,
        opportunity_type=space.opportunity_type,
        limit=limit,
    )
    return [
        item
        for item in candidates
        if item.get("latest_evaluation") is None
        and (item.get("metadata") or {}).get("symbol", "BTCUSDT") == space.symbol
    ]


def _candidate_payload_from_pending(
    item: Mapping[str, Any], space: SearchSpace
) -> dict[str, Any] | None:
    submitted = item.get("submitted_spec")
    metadata = item.get("metadata")
    if not isinstance(submitted, Mapping) or not isinstance(metadata, Mapping):
        return None
    if (
        metadata.get("product") != space.product
        or metadata.get("opportunity_type") != space.opportunity_type
        or metadata.get("base_timeframe") != space.base_timeframe
        or submitted.get("_product") != space.product
        or submitted.get("_market") != space.market
        or submitted.get("_pnl_unit") != space.pnl_unit
        or submitted.get("_symbol", "BTCUSDT") != space.symbol
    ):
        return None
    try:
        hypothesis = _hypothesis_from_submitted_spec(submitted)
    except Exception:
        return None
    revalidation = bool(item.get("revalidation_required"))
    return {
        "hypothesis": hypothesis,
        "metadata": {
            **dict(metadata),
            # A taxonomy-only search-space rename must not strand the canonical
            # behavior. Its executable context above is still required to match.
            "search_space": space.name,
            "id": hypothesis.id,
            "strategy_hash": item.get("behavior_hash"),
            "parent_hashes": list(item.get("parent_hashes") or []),
            "novelty_score": item.get("novelty_score"),
            "nearest_behavior_hash": item.get("nearest_behavior_hash"),
            "resumed_pending": not revalidation,
            "revalidation_pending": revalidation,
        },
    }


def build_generation(
    config: ResearchFactoryConfig,
    *,
    seed: int | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    generated_at = now or _utc_now()
    cycle_seed = _seed_for_cycle(config, generated_at, seed)
    cycle_spaces = _search_spaces_for_cycle(config, generated_at=generated_at)
    universe_context = _market_universe_context(generated_at=generated_at)
    rng = random.Random(cycle_seed)
    if config.dynamic_active_income_universe:
        btc_spaces = [space for space in cycle_spaces if space.product == "btc_accumulation"]
        active_spaces = [space for space in cycle_spaces if space.product == "active_income"]
        rng.shuffle(active_spaces)
        cycle_spaces = (*btc_spaces, *active_spaces)
    budgets = config.budgets
    limits = GrammarLimits(max_total_predicates=budgets.max_total_predicates)
    deadline = time.monotonic() + budgets.max_generation_seconds
    attempts = 0
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    by_space: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    proposal_state = _proposal_state(config.proposal_state_path)
    processed = proposal_state.setdefault("processed", {})
    if not isinstance(processed, dict):
        raise ResearchFactoryConfigError("proposal state processed must be an object")
    proposals_purged = _purge_processed_proposals(
        config.openclaw_accepted_dir,
        set(processed),
    )

    with ExperimentMemory(config.memory_path) as memory:
        memory_before = config.memory_path.stat().st_size
        memory_maintenance: dict[str, Any] = {
            "triggered": False,
            "before_bytes": memory_before,
            "after_bytes": memory_before,
            "rows_compacted": 0,
        }
        if memory_before >= int(budgets.max_memory_bytes * MEMORY_COMPACTION_TRIGGER_FRACTION):
            compacted = memory.compact_storage(
                maximum_rows=MAX_MEMORY_COMPACTION_ROWS_PER_CYCLE,
                vacuum=True,
            )
            memory_maintenance = {"triggered": True, **compacted}
        memory_size = config.memory_path.stat().st_size
        if memory_size > budgets.max_memory_bytes:
            raise ResearchFactoryConfigError(
                "experiment memory remains above max_memory_bytes after bounded, "
                "integrity-checked compaction; generation is paused fail closed: "
                f"{memory_size} > {budgets.max_memory_bytes}"
            )
        research_engine_digest = execution_engine_digest()
        feedback = memory.generator_feedback(research_engine_digest=research_engine_digest)
        weights = _feedback_weights(feedback)
        features_by_space = {
            space.name: _feature_inventory_for_space(space) for space in cycle_spaces
        }
        dedup_by_space = {
            space.name: [
                item
                for item in memory.dedup_population(product=space.product)
                if (item.get("metadata") or {}).get("symbol", "BTCUSDT") == space.symbol
            ]
            for space in cycle_spaces
        }
        parents_by_space = {
            space.name: [
                item
                for item in memory.candidate_parents(
                    product=space.product,
                    opportunity_type=space.opportunity_type,
                    limit=budgets.max_parent_pool,
                    exclude_holdout_exposed=True,
                    exclude_retired=True,
                    latest_outcomes=("reject", "inconclusive", "pre_holdout_pass"),
                    research_engine_digest=research_engine_digest,
                )
                if (item.get("metadata") or {}).get("symbol", "BTCUSDT") == space.symbol
            ]
            for space in cycle_spaces
        }

        # Backpressure first: a candidate registered before a crash/restart is
        # resumed before creating more work.
        for space in cycle_spaces:
            revalidations_for_space = 0
            for item in _pending_for_space(
                memory,
                space,
                limit=budgets.max_candidates_per_space,
                research_engine_digest=research_engine_digest,
            ):
                if len(accepted) >= budgets.max_candidates_per_cycle:
                    break
                candidate = _candidate_payload_from_pending(item, space)
                if candidate is None:
                    continue
                if candidate["metadata"].get("revalidation_pending"):
                    # Keep one slot in every search space available for genuinely
                    # new exploration. A per-space budget of one therefore does
                    # not consume stale-engine revalidation work.
                    revalidation_limit = min(
                        1,
                        max(0, budgets.max_candidates_per_space - 1),
                    )
                    if revalidations_for_space >= revalidation_limit:
                        continue
                    revalidations_for_space += 1
                accepted.append(candidate)
                by_space[space.name] += 1
                by_method[
                    "revalidation_pending"
                    if candidate["metadata"].get("revalidation_pending")
                    else "resumed_pending"
                ] += 1

        # OpenClaw is optional and never blocks native generation.
        proposals = _load_accepted_proposals(config.openclaw_accepted_dir, set(processed))
        for proposal in proposals:
            proposal_id = str(proposal["proposal_id"])
            disposition: dict[str, Any] = {"processed_at": generated_at}
            space = _space_for_proposal(proposal, cycle_spaces)
            if space is None:
                disposition.update(status="rejected", reason="no_matching_search_space")
                processed[proposal_id] = disposition
                rejected.append(
                    {"reason": "openclaw_no_matching_space", "proposal_id": proposal_id}
                )
                continue
            if (
                len(accepted) >= budgets.max_candidates_per_cycle
                or by_space[space.name] >= budgets.max_candidates_per_space
            ):
                # Leave it unprocessed so the next bounded cycle can consume it.
                continue
            try:
                idea = _compile_openclaw_proposal(
                    proposal,
                    space,
                    rng=rng,
                    available_features=features_by_space[space.name],
                    feedback_weights=weights,
                    limits=limits,
                )
                candidate, rejection = _try_register(
                    memory,
                    idea,
                    space,
                    parents=parents_by_space[space.name],
                    dedup_population=dedup_by_space[space.name],
                    budgets=budgets,
                    extra_metadata={
                        "proposal_id": proposal_id,
                        "proposal_digest": proposal.get("content_digest"),
                        "proposal_source": "openclaw",
                    },
                )
            except ExperimentMemoryError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                candidate = None
                rejection = {
                    "reason": "openclaw_compile_rejected",
                    "detail": f"{type(exc).__name__}: {exc}"[:500],
                }
            if candidate is None:
                disposition.update(status="rejected", **(rejection or {"reason": "unknown"}))
                rejected.append({"proposal_id": proposal_id, **(rejection or {})})
            else:
                disposition.update(
                    status="accepted",
                    strategy_hash=candidate["metadata"]["strategy_hash"],
                )
                accepted.append(candidate)
                by_space[space.name] += 1
                by_method[idea.generation_method] += 1
            processed[proposal_id] = disposition

        schedule = _method_schedule(
            budgets.max_candidates_per_cycle,
            budgets,
            rng,
            feedback=feedback,
        )
        space_cursor = 0
        while (
            len(accepted) < budgets.max_candidates_per_cycle
            and attempts < budgets.max_generation_attempts
            and time.monotonic() < deadline
        ):
            attempts += 1
            space = cycle_spaces[space_cursor % len(cycle_spaces)]
            space_cursor += 1
            if by_space[space.name] >= budgets.max_candidates_per_space:
                continue
            method = schedule[(attempts - 1) % len(schedule)]
            parents = parents_by_space[space.name]
            usable = [
                parent
                for parent in parents
                if _parent_hypothesis(parent) is not None
                and _lineage_depth(parent) < budgets.max_lineage_depth
                and parent.get("latest_evaluation") is not None
            ]
            try:
                if method == "recursive_mutation" and usable:
                    parent = _choose_parent(usable, feedback=feedback, rng=rng)
                    parent_hypothesis = _parent_hypothesis(parent)
                    if parent_hypothesis is None:
                        raise ValueError("selected mutation parent has no valid hypothesis")
                    latest = parent.get("latest_evaluation")
                    failure_reasons = (
                        tuple(str(item) for item in latest.get("rejection_reasons") or [])
                        if isinstance(latest, Mapping)
                        else ()
                    )
                    idea = mutate_hypothesis(
                        parent_hypothesis,
                        space,
                        parent_hash=str(parent["behavior_hash"]),
                        rng=rng,
                        available_features=features_by_space[space.name],
                        feedback_weights=weights,
                        limits=limits,
                        failure_reasons=failure_reasons,
                    )
                elif method == "crossover" and len(usable) >= 2:
                    parent_feedback = _parent_feedback_by_hash(feedback)
                    compatible_pairs = [
                        (first, second)
                        for index, first in enumerate(usable)
                        for second in usable[index + 1 :]
                        if _parent_hypothesis(first).direction  # type: ignore[union-attr]
                        == _parent_hypothesis(second).direction  # type: ignore[union-attr]
                    ]
                    if not compatible_pairs:
                        raise ValueError("no compatible crossover parent pair")
                    pair_weights = [
                        _parent_weight(first, parent_feedback)
                        * _parent_weight(second, parent_feedback)
                        for first, second in compatible_pairs
                    ]
                    first, second = rng.choices(compatible_pairs, weights=pair_weights, k=1)[0]
                    first_hypothesis = _parent_hypothesis(first)
                    second_hypothesis = _parent_hypothesis(second)
                    if first_hypothesis is None or second_hypothesis is None:
                        raise ValueError("selected crossover parent has no valid hypothesis")
                    idea = crossover_hypotheses(
                        first_hypothesis,
                        second_hypothesis,
                        space,
                        parent_hashes=(
                            str(first["behavior_hash"]),
                            str(second["behavior_hash"]),
                        ),
                        rng=rng,
                        available_features=features_by_space[space.name],
                        feedback_weights=weights,
                        limits=limits,
                    )
                else:
                    idea = build_fresh_hypothesis(
                        space,
                        rng=rng,
                        available_features=features_by_space[space.name],
                        feedback_weights=weights,
                        limits=limits,
                        motif=rng.choice(MOTIFS),
                    )
                candidate, rejection = _try_register(
                    memory,
                    idea,
                    space,
                    parents=parents,
                    dedup_population=dedup_by_space[space.name],
                    budgets=budgets,
                )
            except Exception as exc:
                candidate = None
                rejection = {
                    "reason": "generation_error",
                    "method": method,
                    "space": space.name,
                    "detail": f"{type(exc).__name__}: {exc}"[:500],
                }
            if candidate is None:
                if len(rejected) < 200:
                    rejected.append({"space": space.name, "method": method, **(rejection or {})})
                continue
            accepted.append(candidate)
            by_space[space.name] += 1
            by_method[idea.generation_method] += 1
            parents_by_space[space.name].insert(
                0,
                {
                    "behavior_hash": candidate["metadata"]["strategy_hash"],
                    "submitted_spec": strategy_behavior_spec(candidate["hypothesis"], space),
                    "metadata": candidate["metadata"],
                    "parent_hashes": candidate["metadata"].get("parent_hashes", []),
                    "latest_evaluation": None,
                },
            )

        final_feedback = memory.generator_feedback(research_engine_digest=research_engine_digest)
        integrity = memory.integrity_check(deep=False)

    _save_proposal_state(config.proposal_state_path, proposal_state)
    proposals_purged += _purge_processed_proposals(
        config.openclaw_accepted_dir,
        set(proposal_state.get("processed") or {}),
    )
    hypotheses = [item["hypothesis"].to_dict() for item in accepted]
    metadata = [
        {
            **item["metadata"],
            "universe_snapshot_id": universe_context["snapshot_id"],
            "universe_generated_at": universe_context["generated_at"],
            "universe_selection_mode": universe_context["selection_mode"],
        }
        for item in accepted
    ]
    elapsed = max(0.0, budgets.max_generation_seconds - max(0.0, deadline - time.monotonic()))
    timed_out = time.monotonic() >= deadline
    return {
        "ok": True,
        "schema": BATCH_SCHEMA,
        "generated_at": generated_at,
        "seed": cycle_seed,
        **SAFETY,
        "source": {
            "config": str(config.path),
            "memory": str(config.memory_path),
            "research_engine_digest": research_engine_digest,
            "openclaw_optional": True,
            "eligible_search_spaces": [space.name for space in cycle_spaces],
            "market_universe": universe_context,
        },
        "budget": {
            "candidate_limit": budgets.max_candidates_per_cycle,
            "per_space_limit": budgets.max_candidates_per_space,
            "attempt_limit": budgets.max_generation_attempts,
            "wall_seconds_limit": budgets.max_generation_seconds,
            "memory_bytes_limit": budgets.max_memory_bytes,
            "memory_bytes": config.memory_path.stat().st_size,
            "attempts": attempts,
            "elapsed_seconds": round(elapsed, 6),
            "wall_time_exhausted": timed_out,
        },
        "summary": {
            "hypotheses": len(hypotheses),
            "new_hypotheses": sum(
                1
                for item in metadata
                if not item.get("resumed_pending") and not item.get("revalidation_pending")
            ),
            "resumed_pending": sum(1 for item in metadata if item.get("resumed_pending")),
            "revalidation_pending": sum(1 for item in metadata if item.get("revalidation_pending")),
            "rejected_attempts": len(rejected),
            "by_space": dict(sorted(by_space.items())),
            "by_product": dict(
                Counter(str(item.get("product")) for item in metadata).most_common()
            ),
            "by_method": dict(sorted(by_method.items())),
            "openclaw_proposals_seen": len(proposals),
            "openclaw_accepted_files_purged": proposals_purged,
            "openclaw_available": config.openclaw_accepted_dir.exists(),
            "cumulative_trials": int((final_feedback.get("totals") or {}).get("evaluations") or 0),
            "unique_behavioral_specs": int(
                (final_feedback.get("totals") or {}).get("strategies") or 0
            ),
        },
        "memory": {
            "integrity": integrity,
            "maintenance": memory_maintenance,
            # generator_feedback has a hard contract excluding protected
            # holdout evaluations; this summary is safe for selection/reporting.
            "feedback": final_feedback,
        },
        "generation_metadata": metadata,
        "rejected": rejected,
        "hypotheses": hypotheses,
    }


def run_factory(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = None,
    seed: int | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    try:
        config = load_factory_config(config_path)
        report = build_generation(config, seed=seed, now=now)
        destination = Path(output_path) if output_path is not None else config.generated_batch_path
        if destination.is_symlink():
            raise ResearchFactoryConfigError(
                f"generated batch must not be a symlink: {destination}"
            )
        write_json_atomic(destination, report)
        report["output"] = str(destination)
        return report
    except (
        ResearchFactoryConfigError,
        ExperimentMemoryError,
        ValueError,
        OSError,
        RuntimeError,
    ) as exc:
        return {
            "ok": False,
            "schema": REPORT_SCHEMA,
            "generated_at": now or _utc_now(),
            **SAFETY,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a bounded autonomous research batch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate", action="store_true", help="Validate config and memory only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate:
        try:
            config = load_factory_config(args.config)
            with ExperimentMemory(config.memory_path, deep_on_open=False) as memory:
                integrity = memory.integrity_check(deep=True)
            payload = {
                "ok": True,
                "schema": REPORT_SCHEMA,
                "config": str(config.path),
                "search_spaces": len(config.search_spaces),
                "memory": integrity,
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "schema": REPORT_SCHEMA,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        payload = run_factory(config_path=args.config, output_path=args.output, seed=args.seed)
    compact = {
        key: payload.get(key)
        for key in ("ok", "schema", "generated_at", "output", "error", "search_spaces")
        if key in payload
    }
    if isinstance(payload.get("summary"), dict):
        compact["summary"] = payload["summary"]
    print(json.dumps(compact, sort_keys=True))
    raise SystemExit(0 if payload.get("ok") else 1)


if __name__ == "__main__":
    main()
