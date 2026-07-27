"""Local server readiness checks for the autopilot.

This is intentionally separate from live/testnet preflight. Readiness does not
construct brokers, connect to exchanges, or place orders. It answers: "Is this
Linux box configured well enough to run the 24/7 supervisor?"
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import shutil
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.strategy_grammar import validate_hypothesis_against_space
from src.autopilot.approvals import (
    ApprovalError,
    ApprovalLedger,
    assert_artifact_live_approved,
    is_valid_approval_actor,
    is_valid_revocation_reason,
)
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.control import load_control, unknown_control_selectors
from src.autopilot.exchange_policy import (
    ACTIVE_INCOME_FUTURES_EXCHANGES,
    ACTIVE_INCOME_MAX_FUTURES_LEVERAGE,
    BTC_ACCUMULATION_SPOT_EXCHANGES,
)
from src.autopilot.experiment_memory import (
    ExperimentMemory,
    ExperimentMemoryError,
    canonical_strategy_hash,
)
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.market_data import (
    build_indicator_feature_status,
    build_indicator_feature_statuses,
    build_market_data_status,
    build_market_data_statuses,
    required_indicator_features_by_market,
)
from src.autopilot.regime_data import build_regime_data_statuses
from src.autopilot.research_factory import (
    BATCH_SCHEMA as GENERATED_BATCH_SCHEMA,
)
from src.autopilot.research_factory import (
    SAFETY as GENERATED_BATCH_SAFETY,
)
from src.autopilot.research_factory import (
    ResearchFactoryConfig,
    load_factory_config,
    resolve_search_space,
    strategy_behavior_spec,
)
from src.autopilot.runtime import (
    assert_recent_preflight,
    assert_recent_testnet_rehearsal,
    validate_config,
)
from src.autopilot.strategy_policy import StrategyPolicyError, assert_strategy_artifact_allowed
from src.config import PROJECT_ROOT
from src.envfile import parse_env_lines

LOGGER = logging.getLogger("autopilot.readiness")
MAX_READINESS_BATCH_BYTES = 8 * 1024 * 1024
_MEMORY_READINESS_CACHE_KEY: tuple[str, int, int, int] | None = None
_MEMORY_READINESS_CACHE_VALUE: dict[str, Any] | None = None


def _job_flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        return None
    return command[index + 1]


def _dynamic_universe_readiness(
    config: AutopilotConfig,
    factory: ResearchFactoryConfig | None,
) -> dict[str, Any] | None:
    if factory is None or not factory.dynamic_active_income_universe:
        return None
    job = next(
        (
            item
            for item in config.jobs
            if item.enabled and "src.autopilot.market_universe" in " ".join(item.command)
        ),
        None,
    )
    if job is None:
        return None
    output_value = _job_flag_value(job.command, "--output")
    if output_value is None:
        return {
            "ok": False,
            "reason": "market_universe_output_missing",
            "symbols": {},
        }
    report_path = Path(output_value)
    if not report_path.is_absolute():
        report_path = job.working_dir / report_path
    status: dict[str, Any] = {
        "ok": False,
        "path": str(report_path),
        "symbols": {},
    }
    if report_path.is_symlink() or not report_path.exists():
        status["reason"] = "market_universe_report_missing"
        return status
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(
            str(payload["generated_at"]).replace("Z", "+00:00")
        ).astimezone(UTC)
    except Exception as exc:
        status.update(
            reason="market_universe_report_invalid",
            error=f"{type(exc).__name__}: {exc}",
        )
        return status
    snapshot = payload.get("snapshot")
    snapshot_id = snapshot.get("id") if isinstance(snapshot, dict) else None
    snapshot_value = snapshot.get("path") if isinstance(snapshot, dict) else None
    snapshot_valid = False
    if isinstance(snapshot_value, str) and snapshot_value:
        snapshot_path = Path(snapshot_value)
        if not snapshot_path.is_absolute():
            snapshot_path = job.working_dir / snapshot_path
        try:
            snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot_valid = bool(
                not snapshot_path.is_symlink()
                and isinstance(snapshot_payload, dict)
                and (snapshot_payload.get("snapshot") or {}).get("id") == snapshot_id
                and snapshot_payload.get("generated_at") == payload.get("generated_at")
                and snapshot_payload.get("eligible_research_symbols")
                == payload.get("eligible_research_symbols")
            )
        except Exception:
            snapshot_valid = False
    symbols = [
        str(symbol).upper()
        for symbol in payload.get("eligible_research_symbols") or []
        if isinstance(symbol, str)
    ]
    age_seconds = (datetime.now(UTC) - generated_at).total_seconds()
    report_ok = bool(
        payload.get("ok") is True
        and payload.get("schema") == "autopilot.market_universe/v2"
        and isinstance(snapshot_id, str)
        and snapshot_id.startswith("sha256:")
        and snapshot.get("append_only") is True
        and snapshot_valid
        and 0 <= age_seconds <= 48 * 3600
        and symbols
    )
    required = required_indicator_features_by_market(
        ["futures"],
        jobs=config.jobs,
    )["futures"]
    symbol_statuses: dict[str, Any] = {}
    for symbol in symbols:
        market_data = build_market_data_status(
            market="futures",
            symbol=symbol,
        )
        indicators = build_indicator_feature_status(
            required,
            market="futures",
            symbol=symbol,
        )
        symbol_statuses[symbol] = {
            "ok": bool(market_data.get("ok") and indicators.get("ok")),
            "market_data": market_data,
            "indicator_features": indicators,
        }
    symbols_ready = bool(symbol_statuses) and all(item["ok"] for item in symbol_statuses.values())
    status.update(
        {
            "ok": report_ok and symbols_ready,
            "reason": (
                "ready"
                if report_ok and symbols_ready
                else (
                    "symbol_data_not_ready"
                    if report_ok
                    else "market_universe_report_not_fresh_or_snapshotted"
                )
            ),
            "generated_at": payload.get("generated_at"),
            "age_seconds": round(age_seconds, 3),
            "snapshot_id": snapshot_id,
            "snapshot_path": snapshot_value,
            "snapshot_valid": snapshot_valid,
            "eligible_symbols": symbols,
            "symbols": symbol_statuses,
        }
    )
    return status


def _check(name: str, ok: bool, *, level: str = "error", detail: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "ok": ok, "level": level}
    if detail is not None:
        payload["detail"] = detail
    return payload


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    while current != current.parent:
        if current.exists():
            return current
        current = current.parent
    return current if current.exists() else None


def _path_writable(path: Path) -> bool:
    if path.is_symlink():
        return False
    parent = _nearest_existing_parent(path)
    return bool(parent and os.access(parent, os.W_OK))


def _disk_space_status(path: Path, min_free_bytes: int) -> dict[str, Any]:
    parent = _nearest_existing_parent(path)
    if parent is None:
        return {
            "path": str(path),
            "checked_path": None,
            "free_bytes": None,
            "min_free_bytes": min_free_bytes,
            "ok": False,
            "reason": "no_existing_parent",
        }
    usage = shutil.disk_usage(parent)
    return {
        "path": str(path),
        "checked_path": str(parent),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "min_free_bytes": min_free_bytes,
        "ok": usage.free >= min_free_bytes,
    }


def _service_installer_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "readable": False,
        "non_empty": False,
        "has_shell_shebang": False,
        "required_markers": {},
        "ok": False,
    }
    if not status["exists"] or not status["is_file"]:
        return status

    status["readable"] = os.access(path, os.R_OK)
    try:
        status["non_empty"] = path.stat().st_size > 0
        content = path.read_text(encoding="utf-8")
        first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    except OSError as exc:
        status["error"] = str(exc)
        return status

    status["has_shell_shebang"] = first_line in {"#!/bin/bash", "#!/usr/bin/env bash", "#!/bin/sh"}
    required_markers = {
        "strict_shell": "set -euo pipefail" in content,
        "config_validation": "src.autopilot.runtime --config" in content
        and "--validate" in content,
        "readiness_install_gate": (
            '"$PYTHON" -m src.autopilot.readiness --config "$CONFIG"' in content
        ),
        "healthcheck_timer": "HEALTHCHECK_TIMER_NAME" in content
        and "systemctl --user enable --now" in content,
        "unit_name_validation": "validate_unit_name" in content,
        "raw_unit_value_validation": all(
            marker in content
            for marker in (
                "validate_unit_value",
                "validate_positive_integer",
                "validate_zero_or_one",
            )
        ),
    }
    status["required_markers"] = required_markers
    missing = [name for name, present in required_markers.items() if not present]
    if missing:
        status["missing_markers"] = missing
    status["ok"] = bool(
        status["readable"]
        and status["non_empty"]
        and status["has_shell_shebang"]
        and all(required_markers.values())
    )
    return status


def _strategy_smoke_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
    }
    if not path.exists():
        status["reason"] = "missing_report"
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status.update(reason="read_error", error=str(exc))
        return status
    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    failures = [
        {
            "name": scenario.get("name"),
            "error": scenario.get("error") or scenario.get("reason"),
        }
        for scenario in scenarios
        if isinstance(scenario, dict) and not scenario.get("ok")
    ]
    ok = bool(payload.get("ok")) and not failures
    status.update(
        {
            "ok": ok,
            "reason": "ready" if ok else "failed",
            "generated_at": payload.get("generated_at"),
            "scenario_count": len(scenarios),
            "failures": failures,
        }
    )
    return status


def _strategy_smoke_configured(config: AutopilotConfig) -> bool:
    return any(
        job.enabled and "src.autopilot.strategy_smoke" in " ".join(job.command)
        for job in config.jobs
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _research_factory_status(
    config: AutopilotConfig,
) -> tuple[dict[str, Any], ResearchFactoryConfig | None]:
    """Validate the autonomous search contract and its autopilot path wiring."""

    path = config.research_factory_config_file
    status: dict[str, Any] = {"path": str(path), "exists": path.exists(), "ok": False}
    try:
        factory = load_factory_config(path)
    except Exception as exc:
        status.update(reason="invalid_config", error=f"{type(exc).__name__}: {exc}")
        return status, None

    active_spaces = [space for space in factory.search_spaces if space.product == "active_income"]
    btc_spaces = [space for space in factory.search_spaces if space.product == "btc_accumulation"]
    active_opportunities = {space.opportunity_type for space in active_spaces}
    required_active = {"scalping", "day_trading", "swing_trading"}
    path_contract = {
        "memory": _same_path(factory.memory_path, config.experiment_memory_file),
        "generated_batch": _same_path(factory.generated_batch_path, config.generated_batch_file),
    }
    goal_contract = {
        "active_income_horizons": required_active.issubset(active_opportunities),
        "active_income_futures_usdt": bool(active_spaces)
        and all(space.market == "futures" and space.pnl_unit == "usdt" for space in active_spaces),
        "btc_accumulation_spot_btc": bool(btc_spaces)
        and all(space.market == "spot" and space.pnl_unit == "btc" for space in btc_spaces),
        "btc_accumulation_dodge_only": bool(btc_spaces)
        and all(set(space.directions) == {"short"} for space in btc_spaces),
        "btc_accumulation_multiple_horizons": len({space.base_timeframe for space in btc_spaces})
        >= 2,
    }
    problems = [
        *(f"path_mismatch:{name}" for name, ok in path_contract.items() if not ok),
        *(f"goal_contract:{name}" for name, ok in goal_contract.items() if not ok),
    ]
    status.update(
        ok=not problems,
        reason="ready" if not problems else "unsafe_search_contract",
        search_spaces=len(factory.search_spaces),
        products=sorted({space.product for space in factory.search_spaces}),
        active_income_opportunities=sorted(active_opportunities),
        btc_accumulation_timeframes=sorted({space.base_timeframe for space in btc_spaces}),
        path_contract=path_contract,
        goal_contract=goal_contract,
        budgets={
            "max_candidates_per_cycle": factory.budgets.max_candidates_per_cycle,
            "max_candidates_per_space": factory.budgets.max_candidates_per_space,
            "max_generation_attempts": factory.budgets.max_generation_attempts,
            "max_generation_seconds": factory.budgets.max_generation_seconds,
            "max_memory_bytes": factory.budgets.max_memory_bytes,
        },
        problems=problems,
    )
    return status, factory


def _experiment_memory_status(
    path: Path,
    *,
    backup_path: Path,
) -> dict[str, Any]:
    """Check an existing memory deeply, or prove a fresh memory can be created."""

    status: dict[str, Any] = {
        "path": str(path),
        "backup_path": str(backup_path),
        "exists": path.exists(),
        "ok": False,
    }
    if _same_path(path, backup_path):
        status.update(reason="backup_path_matches_live_memory")
        return status
    if path.is_symlink() or backup_path.is_symlink():
        status.update(reason="symlink_not_allowed")
        return status
    if backup_path.exists() and not backup_path.is_file():
        status.update(reason="backup_path_not_file")
        return status
    backup_writable = _path_writable(backup_path)
    status["backup_path_writable"] = backup_writable
    if not backup_writable:
        status.update(reason="backup_path_not_writable")
        return status
    if not path.exists():
        writable = _path_writable(path)
        status.update(
            ok=writable,
            reason="initializes_on_first_factory_run" if writable else "memory_path_not_writable",
            path_writable=writable,
            integrity=None,
        )
        return status
    if not path.is_file():
        status.update(reason="memory_path_not_file")
        return status
    try:
        initial_stat = path.stat()
        cache_key = (
            str(path.resolve()),
            initial_stat.st_ino,
            initial_stat.st_mtime_ns,
            initial_stat.st_size,
        )
    except OSError as exc:
        status.update(reason="memory_stat_failed", error=f"{type(exc).__name__}: {exc}")
        return status
    global _MEMORY_READINESS_CACHE_KEY, _MEMORY_READINESS_CACHE_VALUE
    if cache_key == _MEMORY_READINESS_CACHE_KEY and _MEMORY_READINESS_CACHE_VALUE is not None:
        cached = copy.deepcopy(_MEMORY_READINESS_CACHE_VALUE)
        cached["backup_path_writable"] = backup_writable
        return cached
    try:
        with ExperimentMemory(path, deep_on_open=False) as memory:
            integrity = memory.integrity_check(deep=True)
    except (ExperimentMemoryError, OSError, ValueError) as exc:
        status.update(reason="integrity_failed", error=f"{type(exc).__name__}: {exc}")
        return status
    status.update(ok=True, reason="ready", path_writable=_path_writable(path), integrity=integrity)
    try:
        final_stat = path.stat()
        _MEMORY_READINESS_CACHE_KEY = (
            str(path.resolve()),
            final_stat.st_ino,
            final_stat.st_mtime_ns,
            final_stat.st_size,
        )
        _MEMORY_READINESS_CACHE_VALUE = copy.deepcopy(status)
    except OSError:
        _MEMORY_READINESS_CACHE_KEY = None
        _MEMORY_READINESS_CACHE_VALUE = None
    return status


def _generated_batch_status(
    path: Path,
    *,
    factory: ResearchFactoryConfig | None,
) -> dict[str, Any]:
    """Fail closed on an unsafe batch while allowing a clean first boot."""

    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "level": "error",
    }
    if not path.exists():
        status.update(
            reason="missing; generated by the first research_factory job",
            level="warning",
            next_action="make research-generate",
        )
        return status
    if path.is_symlink() or not path.is_file():
        status.update(reason="batch_must_be_regular_file")
        return status
    if path.stat().st_size > MAX_READINESS_BATCH_BYTES:
        status.update(reason="batch_too_large", size_bytes=path.stat().st_size)
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        status.update(reason="read_error", error=f"{type(exc).__name__}: {exc}")
        return status
    if not isinstance(payload, dict):
        status.update(reason="payload_not_object")
        return status
    safety = {key: payload.get(key) for key in GENERATED_BATCH_SAFETY}
    if (
        payload.get("ok") is not True
        or payload.get("schema") != GENERATED_BATCH_SCHEMA
        or safety != GENERATED_BATCH_SAFETY
    ):
        status.update(
            reason="failed_safety_contract",
            schema=payload.get("schema"),
            safety=safety,
        )
        return status
    if factory is None:
        status.update(reason="factory_config_unavailable")
        return status
    hypotheses = payload.get("hypotheses")
    metadata_items = payload.get("generation_metadata")
    if not isinstance(hypotheses, list) or not isinstance(metadata_items, list):
        status.update(reason="hypotheses_or_metadata_not_list")
        return status
    if len(hypotheses) != len(metadata_items):
        status.update(reason="hypothesis_metadata_count_mismatch")
        return status
    if len(hypotheses) > factory.budgets.max_candidates_per_cycle:
        status.update(reason="candidate_budget_exceeded", hypotheses=len(hypotheses))
        return status

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for item in metadata_items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            status.update(reason="invalid_generation_metadata")
            return status
        strategy_id = item["id"]
        if strategy_id in metadata_by_id:
            status.update(reason="duplicate_metadata_id", strategy_id=strategy_id)
            return status
        metadata_by_id[strategy_id] = item

    hashes: set[str] = set()
    products: set[str] = set()
    try:
        for raw_hypothesis in hypotheses:
            if not isinstance(raw_hypothesis, dict):
                raise ValueError("hypothesis must be an object")
            hypothesis = Hypothesis.from_dict(raw_hypothesis)
            metadata = metadata_by_id.pop(hypothesis.id)
            space = resolve_search_space(factory, metadata)
            expected_context = {
                "product": space.product,
                "market": space.market,
                "pnl_unit": space.pnl_unit,
                "opportunity_type": space.opportunity_type,
                "base_timeframe": space.base_timeframe,
            }
            mismatches = [
                key for key, expected in expected_context.items() if metadata.get(key) != expected
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
            observed_hash = metadata.get("strategy_hash")
            if observed_hash not in accepted_hashes:
                raise ValueError("strategy behavior hash mismatch")
            if observed_hash in hashes:
                raise ValueError("duplicate strategy behavior")
            hashes.add(str(observed_hash))
            products.add(space.product)
        if metadata_by_id:
            raise ValueError("orphan generation metadata")
    except (KeyError, TypeError, ValueError) as exc:
        status.update(reason="invalid_strategy_batch", error=f"{type(exc).__name__}: {exc}")
        return status

    if not hypotheses:
        status.update(
            reason="empty_safe_batch",
            level="warning",
            hypotheses=0,
            generated_at=payload.get("generated_at"),
            next_action="make research-generate",
        )
        return status
    status.update(
        ok=True,
        level="info",
        reason="ready",
        hypotheses=len(hypotheses),
        unique_behavioral_specs=len(hashes),
        products=sorted(products),
        generated_at=payload.get("generated_at"),
    )
    return status


def _offline_rehearsal_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
    }
    if not path.exists():
        status["reason"] = "missing_report"
        status["next_action"] = "make rehearse"
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status.update(
            reason="read_error", error=f"{type(exc).__name__}: {exc}", next_action="make rehearse"
        )
        return status
    if not isinstance(payload, dict):
        status.update(
            reason="invalid_report",
            error=f"expected JSON object, got {type(payload).__name__}",
            next_action="make rehearse",
        )
        return status
    products = payload.get("products")
    product_status: dict[str, Any] = {}
    missing_products: list[str] = []
    invalid_products: list[str] = []
    for name in ("active_income", "btc_accumulation"):
        item = products.get(name) if isinstance(products, dict) else None
        if not isinstance(item, dict):
            missing_products.append(name)
            continue
        product_ok = (
            item.get("before_recommendation") == "needs_approval"
            and item.get("after_recommendation") == "already_approved"
        )
        product_status[name] = {
            "ok": product_ok,
            "artifact": item.get("artifact"),
            "trade_log": item.get("trade_log"),
            "promotion_review_json": item.get("promotion_review_json"),
            "before_recommendation": item.get("before_recommendation"),
            "after_recommendation": item.get("after_recommendation"),
        }
        if not product_ok:
            invalid_products.append(name)
    preflight_products = payload.get("preflight_products")
    if not isinstance(preflight_products, list):
        preflight_products = []
    missing_preflight_products = sorted(
        {"active_income", "btc_accumulation"} - {str(item) for item in preflight_products}
    )
    reasons: list[str] = []
    if payload.get("ok") is not True:
        reasons.append("summary_not_ok")
    if missing_products:
        reasons.append("missing_products")
    if invalid_products:
        reasons.append("invalid_product_recommendations")
    if payload.get("preflight_ok") is not True:
        reasons.append("preflight_not_ok")
    if missing_preflight_products:
        reasons.append("missing_preflight_products")
    status.update(
        {
            "ok": not reasons,
            "reason": "ready" if not reasons else "failed",
            "work_dir": payload.get("work_dir"),
            "products": product_status,
            "missing_products": missing_products,
            "invalid_products": invalid_products,
            "preflight_report": payload.get("preflight_report"),
            "preflight_ok": payload.get("preflight_ok"),
            "preflight_products": preflight_products,
            "missing_preflight_products": missing_preflight_products,
            "reasons": reasons,
        }
    )
    if reasons:
        status["next_action"] = "make rehearse"
    return status


def _approval_ledger_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": path.exists(), "ok": True}
    if not path.exists():
        status["reason"] = "missing; created on first approval"
        return status
    try:
        payload = ApprovalLedger(path).load()
    except (ApprovalError, OSError, json.JSONDecodeError) as exc:
        status.update(ok=False, reason="invalid_ledger", error=f"{type(exc).__name__}: {exc}")
        return status
    approvals = payload.get("approvals", {})
    invalid_actor_entries = []
    fingerprint_mismatch_entries = []
    invalid_revocation_entries = []
    counts: dict[str, int] = {}
    for fingerprint, raw_entry in approvals.items():
        if not isinstance(raw_entry, dict):
            counts["malformed"] = counts.get("malformed", 0) + 1
            continue
        entry_status = str(raw_entry.get("status") or "unknown")
        if entry_status == "approved" and not is_valid_approval_actor(raw_entry.get("approved_by")):
            entry_status = "invalid_actor"
            invalid_actor_entries.append(
                {
                    "fingerprint": str(fingerprint),
                    "strategy_id": raw_entry.get("strategy_id"),
                    "artifact_path": raw_entry.get("artifact_path"),
                    "product": raw_entry.get("product"),
                }
            )
        elif entry_status == "revoked":
            reasons = []
            if not is_valid_approval_actor(raw_entry.get("revoked_by")):
                reasons.append("invalid_revoked_by")
            if not is_valid_revocation_reason(raw_entry.get("revocation_reason")):
                reasons.append("missing_revocation_reason")
            if reasons:
                entry_status = "invalid_revocation_audit"
                invalid_revocation_entries.append(
                    {
                        "fingerprint": str(fingerprint),
                        "strategy_id": raw_entry.get("strategy_id"),
                        "artifact_path": raw_entry.get("artifact_path"),
                        "product": raw_entry.get("product"),
                        "reasons": reasons,
                    }
                )
        elif entry_status == "approved" and raw_entry.get("fingerprint") != str(fingerprint):
            entry_status = "fingerprint_mismatch"
            fingerprint_mismatch_entries.append(
                {
                    "fingerprint": str(fingerprint),
                    "entry_fingerprint": raw_entry.get("fingerprint"),
                    "strategy_id": raw_entry.get("strategy_id"),
                    "artifact_path": raw_entry.get("artifact_path"),
                    "product": raw_entry.get("product"),
                }
            )
        counts[entry_status] = counts.get(entry_status, 0) + 1
    status.update(
        reason="ready",
        approval_count=len(approvals),
        counts=counts,
        invalid_actor_count=len(invalid_actor_entries),
        invalid_actor_entries=invalid_actor_entries[:5],
        invalid_revocation_count=len(invalid_revocation_entries),
        invalid_revocation_entries=invalid_revocation_entries[:5],
        fingerprint_mismatch_count=len(fingerprint_mismatch_entries),
        fingerprint_mismatch_entries=fingerprint_mismatch_entries[:5],
    )
    return status


def _control_file_status(path: Path, config: AutopilotConfig) -> dict[str, Any]:
    control = load_control(path)
    unknown_selectors = unknown_control_selectors(control, config)
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": not control.get("control_error") and not unknown_selectors,
        "reason": "ready",
        "paused": control.get("paused"),
        "pause_jobs": control.get("pause_jobs"),
        "paused_products": control.get("paused_products", []),
        "paused_jobs": control.get("paused_jobs", []),
        "flatten_all": control.get("flatten_all"),
        "flatten_products": control.get("flatten_products", []),
    }
    if control.get("control_error"):
        status["reason"] = "invalid_control_file"
        status["control_error"] = control.get("control_error")
    elif unknown_selectors:
        status["reason"] = "unknown_control_selectors"
        status["unknown_selectors"] = unknown_selectors
    return status


def _market_type(product: ProductConfig) -> str:
    return "spot" if product.objective == "btc_accumulation" else "futures"


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag: 1/0, true/false, yes/no, or on/off.")


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except ValueError:
        return float("nan")


def _env_optional_str(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip()


def _env_exchange(
    env: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _read_dotenv(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        # The dedicated owner/private-mode readiness check below reports the
        # actionable failure. Do not let an unreadable credential file prevent
        # the rest of the fail-closed report from being produced.
        return {}
    return parse_env_lines(lines)


def _environment_file_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": path.exists(), "ok": False}
    if path.is_symlink():
        status["reason"] = "symlink_not_allowed"
        return status
    if not path.exists():
        status.update(ok=True, reason="missing_optional_paper_environment")
        return status
    try:
        metadata = path.stat()
    except OSError as exc:
        status.update(reason="stat_failed", error=f"{type(exc).__name__}: {exc}")
        return status
    mode = stat.S_IMODE(metadata.st_mode)
    owner_uid = os.getuid()
    status.update(
        mode=f"{mode:04o}",
        owner_uid=metadata.st_uid,
        expected_owner_uid=owner_uid,
        regular_file=stat.S_ISREG(metadata.st_mode),
        owner_matches=metadata.st_uid == owner_uid,
        group_world_permissions=mode & 0o077,
        readable=os.access(path, os.R_OK),
    )
    status["ok"] = bool(
        status["regular_file"]
        and status["owner_matches"]
        and status["group_world_permissions"] == 0
        and status["readable"]
    )
    status["reason"] = "ready" if status["ok"] else "credential_file_not_owner_private"
    return status


def _readiness_env() -> dict[str, str]:
    values = _read_dotenv(PROJECT_ROOT / ".env")
    values.update(os.environ)
    return values


def _product_readiness(
    product: ProductConfig,
    config: AutopilotConfig,
    *,
    env: Mapping[str, str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    artifact_exists = product.strategies_path.exists()
    if product.execution_mode == "live":
        checks.append(
            _check(
                f"{product.name}: strategy artifact exists",
                artifact_exists,
                detail=str(product.strategies_path),
            )
        )
    else:
        checks.append(
            _check(
                f"{product.name}: paper strategy artifact",
                artifact_exists,
                level="info",
                detail=str(product.strategies_path)
                if artifact_exists
                else "missing; product will wait for research/export",
            )
        )

    if artifact_exists:
        try:
            detail = assert_strategy_artifact_allowed(product)
            checks.append(_check(f"{product.name}: strategy policy", True, detail=detail))
        except (StrategyPolicyError, FileNotFoundError, json.JSONDecodeError) as exc:
            checks.append(
                _check(
                    f"{product.name}: strategy policy",
                    False,
                    level="error" if product.execution_mode == "live" else "warning",
                    detail=str(exc),
                )
            )

    checks.append(
        _check(
            f"{product.name}: state path writable",
            _path_writable(product.state_file),
            detail=str(product.state_file),
        )
    )
    checks.append(
        _check(
            f"{product.name}: trade-log path writable",
            _path_writable(product.trade_log),
            detail=str(product.trade_log),
        )
    )

    if product.execution_mode != "live":
        return checks

    if artifact_exists:
        try:
            assert_artifact_live_approved(
                product.strategies_path, config.approval_ledger, product=product
            )
            checks.append(_check(f"{product.name}: live approval", True))
        except (ApprovalError, FileNotFoundError, json.JSONDecodeError) as exc:
            checks.append(_check(f"{product.name}: live approval", False, detail=str(exc)))

    if product.require_preflight:
        preflight_exists = bool(product.preflight_report and product.preflight_report.exists())
        checks.append(
            _check(
                f"{product.name}: preflight report exists",
                preflight_exists,
                detail=str(product.preflight_report)
                if product.preflight_report
                else "not configured",
            )
        )
        if preflight_exists:
            try:
                detail = assert_recent_preflight(product)
                checks.append(
                    _check(f"{product.name}: preflight report current", True, detail=detail)
                )
            except (RuntimeError, OSError, json.JSONDecodeError, ValueError) as exc:
                checks.append(
                    _check(f"{product.name}: preflight report current", False, detail=str(exc))
                )

    if product.require_testnet_rehearsal:
        rehearsal_exists = bool(
            product.testnet_rehearsal_report and product.testnet_rehearsal_report.exists()
        )
        checks.append(
            _check(
                f"{product.name}: testnet rehearsal report exists",
                rehearsal_exists,
                detail=str(product.testnet_rehearsal_report)
                if product.testnet_rehearsal_report
                else "not configured",
            )
        )
        if rehearsal_exists:
            try:
                detail = assert_recent_testnet_rehearsal(product)
                checks.append(
                    _check(f"{product.name}: testnet rehearsal current", True, detail=detail)
                )
            except (RuntimeError, OSError, json.JSONDecodeError, ValueError) as exc:
                checks.append(
                    _check(f"{product.name}: testnet rehearsal current", False, detail=str(exc))
                )

    market_type = _market_type(product)
    env_errors: list[str] = []
    exchange_name = "SPOT_EXCHANGE" if market_type == "spot" else "FUTURES_EXCHANGE"
    try:
        exchange = _env_exchange(
            env,
            exchange_name,
            default="binance" if market_type == "spot" else "binanceusdm",
        )
    except ValueError as exc:
        exchange = ""
        env_errors.append(str(exc))
    quote_asset = env.get("QUOTE_ASSET", "USDT").strip().upper()
    try:
        live_enabled = _env_bool(env, "TRADING_LIVE", False)
    except ValueError as exc:
        live_enabled = False
        env_errors.append(str(exc))
    try:
        testnet_enabled = _env_bool(env, "EXCHANGE_TESTNET", True)
    except ValueError as exc:
        testnet_enabled = False
        env_errors.append(str(exc))
    max_notional = _env_float(env, "MAX_NOTIONAL_USD", 100.0)
    max_futures_leverage = _env_float(env, "MAX_FUTURES_LEVERAGE", 1.0)
    futures_margin_mode = env.get("FUTURES_MARGIN_MODE", "isolated").strip().lower()
    if env_errors:
        checks.append(
            _check(f"{product.name}: exchange environment values", False, detail=env_errors)
        )
    checks.extend(
        [
            _check(
                f"{product.name}: TRADING_LIVE=1",
                live_enabled,
                detail="required before live/testnet order submission",
            ),
            _check(
                f"{product.name}: exchange API credentials",
                bool(
                    _env_optional_str(env, "EXCHANGE_API_KEY")
                    and _env_optional_str(env, "EXCHANGE_API_SECRET")
                ),
                detail="EXCHANGE_API_KEY and EXCHANGE_API_SECRET",
            ),
            _check(
                f"{product.name}: max notional cap",
                max_notional > 0,
                detail=max_notional,
            ),
            _check(
                f"{product.name}: {market_type} exchange configured",
                bool(exchange),
                level="warning",
                detail=exchange or "using execution default",
            ),
        ]
    )
    if market_type == "futures":
        checks.extend(
            [
                _check(
                    f"{product.name}: active-income futures exchange",
                    product.objective != "active_income"
                    or exchange.strip().lower() in ACTIVE_INCOME_FUTURES_EXCHANGES,
                    detail=exchange or "using execution default",
                ),
                _check(
                    f"{product.name}: futures margin mode",
                    futures_margin_mode == "isolated",
                    detail=futures_margin_mode,
                ),
                _check(
                    f"{product.name}: max futures leverage",
                    (
                        max_futures_leverage == ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
                        if product.objective == "active_income"
                        else 1 <= max_futures_leverage <= 3
                    ),
                    detail={
                        "value": max_futures_leverage,
                        "required": ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
                        if product.objective == "active_income"
                        else "1-3",
                    },
                ),
            ]
        )
    if market_type == "spot":
        checks.append(
            _check(
                f"{product.name}: BTC accumulation spot exchange",
                product.objective != "btc_accumulation"
                or exchange.strip().lower() in BTC_ACCUMULATION_SPOT_EXCHANGES,
                detail=exchange or "using execution default",
            )
        )
    checks.append(_check(f"{product.name}: quote asset", quote_asset == "USDT", detail=quote_asset))
    checks.append(
        _check(
            f"{product.name}: production exchange routing",
            not testnet_enabled,
            detail=(
                "EXCHANGE_TESTNET=0 is required for a configured live product; "
                "run the separate testnet rehearsal while the product remains paper/paused"
            ),
        )
    )
    return checks


def build_readiness_report(
    config: AutopilotConfig,
    *,
    env: Mapping[str, str] | None = None,
    ccxt_available: bool | None = None,
    require_core_products: bool = False,
    require_core_jobs: bool = False,
    allow_data_bootstrap: bool = False,
) -> dict[str, Any]:
    env = _readiness_env() if env is None else env
    checks: list[dict[str, Any]] = []

    config_errors = validate_config(
        config,
        require_core_products=require_core_products,
        require_core_jobs=require_core_jobs,
    )
    checks.append(_check("autopilot config valid", not config_errors, detail=config_errors or None))
    research_factory, loaded_factory = _research_factory_status(config)
    checks.append(
        _check(
            "autonomous research factory config",
            bool(research_factory["ok"]),
            detail=research_factory,
        )
    )
    experiment_memory = _experiment_memory_status(
        config.experiment_memory_file,
        backup_path=config.experiment_memory_backup_file,
    )
    checks.append(
        _check(
            "experiment memory integrity and recovery path",
            bool(experiment_memory["ok"]),
            detail=experiment_memory,
        )
    )
    generated_batch = _generated_batch_status(
        config.generated_batch_file,
        factory=loaded_factory,
    )
    checks.append(
        _check(
            "generated research batch safety",
            bool(generated_batch["ok"]),
            level=str(generated_batch["level"]),
            detail=generated_batch,
        )
    )
    checks.append(
        _check(
            "runtime directory writable",
            _path_writable(config.status_file),
            detail=str(config.status_file.parent),
        )
    )
    checks.append(
        _check(
            "runtime lock path writable",
            _path_writable(config.lock_file),
            detail=str(config.lock_file),
        )
    )
    checks.append(
        _check(
            "control path writable",
            _path_writable(config.control_file),
            detail=str(config.control_file),
        )
    )
    control_file = _control_file_status(config.control_file, config)
    checks.append(_check("control file valid", bool(control_file["ok"]), detail=control_file))
    checks.append(
        _check(
            "control audit path writable",
            _path_writable(config.control_audit_file),
            detail=str(config.control_audit_file),
        )
    )
    checks.append(
        _check(
            "approval ledger path writable",
            _path_writable(config.approval_ledger),
            detail=str(config.approval_ledger),
        )
    )
    checks.append(
        _check(
            "scheduled job state path writable",
            _path_writable(config.job_state_file),
            detail=str(config.job_state_file),
        )
    )
    checks.append(
        _check(
            "alert log path writable",
            _path_writable(config.alert_file),
            detail=str(config.alert_file),
        )
    )
    checks.append(
        _check(
            "alert cooldown state path writable",
            _path_writable(config.alert_state_file),
            detail=str(config.alert_state_file),
        )
    )
    approval_ledger = _approval_ledger_status(config.approval_ledger)
    checks.append(
        _check("approval ledger readable", bool(approval_ledger["ok"]), detail=approval_ledger)
    )
    if approval_ledger.get("ok") and approval_ledger.get("invalid_actor_count"):
        checks.append(
            _check(
                "approval ledger actor audit",
                False,
                level="warning",
                detail={
                    "invalid_actor_count": approval_ledger["invalid_actor_count"],
                    "entries": approval_ledger.get("invalid_actor_entries", []),
                },
            )
        )
    if approval_ledger.get("ok") and approval_ledger.get("fingerprint_mismatch_count"):
        checks.append(
            _check(
                "approval ledger fingerprint audit",
                False,
                level="warning",
                detail={
                    "fingerprint_mismatch_count": approval_ledger["fingerprint_mismatch_count"],
                    "entries": approval_ledger.get("fingerprint_mismatch_entries", []),
                },
            )
        )
    if approval_ledger.get("ok") and approval_ledger.get("invalid_revocation_count"):
        checks.append(
            _check(
                "approval ledger revocation audit",
                False,
                level="warning",
                detail={
                    "invalid_revocation_count": approval_ledger["invalid_revocation_count"],
                    "entries": approval_ledger.get("invalid_revocation_entries", []),
                },
            )
        )
    runtime_disk = _disk_space_status(config.status_file, config.min_runtime_free_bytes)
    checks.append(
        _check(
            "runtime filesystem free space",
            bool(runtime_disk["ok"]),
            level="warning",
            detail=runtime_disk,
        )
    )
    checks.append(
        _check(
            "operator report path writable",
            _path_writable(config.operator_report_file)
            and _path_writable(config.operator_report_json_file),
            detail={
                "markdown": str(config.operator_report_file),
                "json": str(config.operator_report_json_file),
            },
        )
    )
    checks.append(
        _check(
            "readiness report path writable",
            _path_writable(config.readiness_report_file)
            and _path_writable(config.readiness_report_json_file),
            detail={
                "markdown": str(config.readiness_report_file),
                "json": str(config.readiness_report_json_file),
            },
        )
    )
    service_installer = _service_installer_status(
        PROJECT_ROOT / "scripts" / "install_autopilot_service.sh"
    )
    checks.append(
        _check("service installer usable", bool(service_installer["ok"]), detail=service_installer)
    )
    markets = sorted({product.market for product in config.products}) or ["futures"]
    market_data = build_market_data_statuses(markets)
    checks.append(
        _check(
            "market data seed and freshness",
            all(item.get("ok") for item in market_data.values()),
            level="warning" if any(item.get("exists") for item in market_data.values()) else "info",
            detail=market_data,
        )
    )
    indicator_features = build_indicator_feature_statuses(
        markets,
        required_features_by_market=required_indicator_features_by_market(
            markets, jobs=config.jobs
        ),
    )
    checks.append(
        _check(
            "indicator feature readiness",
            all(item.get("ok") for item in indicator_features.values()),
            level="warning",
            detail=indicator_features,
        )
    )
    dynamic_universe = _dynamic_universe_readiness(config, loaded_factory)
    if dynamic_universe is not None:
        dynamic_level = (
            "warning"
            if allow_data_bootstrap
            or dynamic_universe.get("reason")
            in {"market_universe_report_missing", "market_universe_output_missing"}
            else "error"
        )
        checks.append(
            _check(
                "dynamic altcoin universe data readiness",
                bool(dynamic_universe["ok"]),
                level=dynamic_level,
                detail=dynamic_universe,
            )
        )
    regime_data = build_regime_data_statuses(config.jobs)
    if regime_data:
        checks.append(
            _check(
                "regime data readiness",
                all(item.get("available") is not False for item in regime_data),
                level="warning",
                detail=regime_data,
            )
        )
    if _strategy_smoke_configured(config) or config.strategy_smoke_file.exists():
        strategy_smoke = _strategy_smoke_status(config.strategy_smoke_file)
        checks.append(
            _check(
                "strategy framework smoke",
                bool(strategy_smoke["ok"]),
                level="warning",
                detail=strategy_smoke,
            )
        )
    offline_rehearsal = _offline_rehearsal_status(
        PROJECT_ROOT / "runtime" / "rehearsal" / "rehearsal_summary.json"
    )
    checks.append(
        _check(
            "offline workflow rehearsal",
            bool(offline_rehearsal["ok"]),
            level="warning",
            detail=offline_rehearsal,
        )
    )
    env_file = PROJECT_ROOT / ".env"
    checks.append(
        _check(
            "environment file present",
            env_file.exists(),
            level="warning",
            detail=".env is optional for paper mode, required for live credentials",
        )
    )
    checks.append(
        _check("environment file not symlink", not env_file.is_symlink(), detail=str(env_file))
    )
    environment_file = _environment_file_status(env_file)
    checks.append(
        _check(
            "environment file owner-private",
            bool(environment_file["ok"]),
            detail=environment_file,
        )
    )

    any_live = any(product.execution_mode == "live" for product in config.products)
    ccxt_ok = (
        importlib.util.find_spec("ccxt") is not None if ccxt_available is None else ccxt_available
    )
    checks.append(
        _check(
            "ccxt installed for live mode",
            (not any_live) or ccxt_ok,
            detail="required only when a product is live",
        )
    )

    for product in config.products:
        checks.extend(_product_readiness(product, config, env=env))

    blocking = [check for check in checks if check["level"] == "error" and not check["ok"]]
    warnings = [check for check in checks if check["level"] == "warning" and not check["ok"]]
    return {
        "ok": not blocking,
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "checks": checks,
    }


def render_readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Autopilot Readiness",
        "",
        f"- Status: `{'ok' if report['ok'] else 'blocked'}`",
        f"- Blocking checks: `{report['blocking_count']}`",
        f"- Warnings: `{report['warning_count']}`",
        "",
        "| Check | Level | Status | Detail |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        detail = str(check.get("detail", "")).replace("|", "\\|")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(
            f"| {check['name']} | `{check['level']}` | "
            f"`{'ok' if check['ok'] else 'fail'}` | {detail} |"
        )
    return "\n".join(lines) + "\n"


def _append_blocking_check(report: dict[str, Any], name: str, detail: Any) -> None:
    report.setdefault("checks", []).append(_check(name, False, detail=detail))
    report["blocking_count"] = sum(
        1
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("level") == "error" and not check.get("ok")
    )
    report["warning_count"] = sum(
        1
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("level") == "warning" and not check.get("ok")
    )
    report["ok"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local server readiness for the autopilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("runtime/readiness_report.md"))
    parser.add_argument("--json-output", type=Path, default=Path("runtime/readiness_report.json"))
    parser.add_argument(
        "--allow-data-bootstrap",
        action="store_true",
        help=(
            "Keep missing dynamic-universe datasets nonblocking during service "
            "installation so the installed history jobs can seed them."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_failed = False
    try:
        report = build_readiness_report(
            load_config(args.config),
            require_core_products=True,
            require_core_jobs=True,
            allow_data_bootstrap=args.allow_data_bootstrap,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build readiness report")
        report = {
            "ok": False,
            "blocking_count": 1,
            "warning_count": 0,
            "checks": [
                _check(
                    "readiness build failed",
                    False,
                    detail={"config": str(args.config), "error": f"{type(exc).__name__}: {exc}"},
                )
            ],
        }
    try:
        write_text_atomic(args.output, render_readiness_markdown(report))
    except Exception as exc:
        LOGGER.exception("Failed to write readiness markdown")
        output_failed = True
        _append_blocking_check(
            report,
            "readiness markdown output writable",
            {"path": str(args.output), "error": f"{type(exc).__name__}: {exc}"},
        )
    try:
        write_json_atomic(args.json_output, report)
    except Exception as exc:
        LOGGER.exception("Failed to write readiness JSON")
        output_failed = True
        _append_blocking_check(
            report,
            "readiness JSON output writable",
            {"path": str(args.json_output), "error": f"{type(exc).__name__}: {exc}"},
        )
    if output_failed:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(str(args.output))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
