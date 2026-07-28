"""24/7 orchestration loop for the trading system."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.autopilot.approvals import (
    ApprovalError,
    artifact_digest,
    assert_loaded_artifact_live_approved,
    load_artifact,
    strategy_fingerprint,
)
from src.autopilot.config import (
    DEFAULT_CONFIG_PATH,
    AutopilotConfig,
    JobConfig,
    ProductConfig,
    canonical_product_config,
    load_config,
)
from src.autopilot.control import (
    is_job_paused,
    is_product_paused,
    load_control,
    should_flatten_product,
    unknown_control_selectors,
    update_control,
)
from src.autopilot.exchange_policy import (
    ACTIVE_INCOME_FUTURES_EXCHANGES,
    ACTIVE_INCOME_MAX_FUTURES_LEVERAGE,
    BTC_ACCUMULATION_SPOT_EXCHANGES,
    validate_exchange_policy,
    validate_product_symbol_policy,
)
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.jobs import run_due_jobs
from src.autopilot.locking import acquire_runtime_lock
from src.autopilot.notifications import (
    emit_alert,
    failure_detail,
    promotion_warning_detail,
    readiness_warning_detail,
    required_testnet_rehearsal_warning_detail,
    research_handoff_warning_detail,
    research_progress_warning_detail,
)
from src.autopilot.reporting import (
    build_operator_report,
    render_operator_markdown,
    utc_now,
    write_status,
)
from src.autopilot.strategy_policy import (
    StrategyPolicyError,
    assert_loaded_strategy_artifact_allowed,
    assert_strategy_artifact_allowed,
)
from src.autopilot.testnet_rehearsal import (
    summarize_testnet_rehearsal_report,
    testnet_rehearsal_next_action,
)
from src.build_dataset import TIMEFRAME_SECONDS
from src.execution.broker import (
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ACCOUNT_FINGERPRINT_PREFIX
from src.run_bot import PaperTradingBot, configure_logging

LOGGER = logging.getLogger("autopilot")
PREFLIGHT_REQUIRED_CHECKS = (
    "product_config",
    "execution_engine_identity",
    "strategy_artifact_exists",
    "strategy_fingerprints",
    "strategy_policy",
    "exchange_environment",
    "broker_constructed",
    "exchange_read_connectivity",
)
PREFLIGHT_CLOCK_SKEW_SECONDS = 300
SHELL_EXECUTABLES = {"bash", "csh", "fish", "ksh", "sh", "tcsh", "zsh"}
DISALLOWED_JOB_MODULES = {"src.autopilot.approvals"}
OPEN_POSITION_STALE_HORIZON_MULTIPLE = 3.0
SPOT_FLATTEN_BALANCE_FEE_TOLERANCE_FRACTION = 0.002
SPOT_FLATTEN_INTENT_KEYS = frozenset(
    {
        "version",
        "strategy_id",
        "symbol",
        "side",
        "order_type",
        "client_id",
        "broker_account_fingerprint",
        "qty",
        "quote_budget",
        "position_before",
        "created_ts",
    }
)
SPOT_FLATTEN_POSITION_EVIDENCE_KEYS = frozenset({"symbol", "qty", "avg_price"})
FUTURES_FLATTEN_INTENT_KEYS = frozenset(
    {
        "version",
        "phase",
        "strategy_id",
        "symbol",
        "side",
        "order_type",
        "reduce_only",
        "submission_kind",
        "client_id",
        "broker_account_fingerprint",
        "qty",
        "position_digest",
        "position_before",
        "quote_balance_before",
        "fill",
        "position_after",
        "quote_balance_after",
        "realized_account_delta",
        "created_ts",
        "proven_ts",
    }
)
BOT_STATUS_DURABLE_STATE_FIELDS = {
    "pending_order": (
        "version",
        "strategy_id",
        "stage",
        "symbol",
        "side",
        "qty",
        "order_type",
        "reduce_only",
        "client_id",
        "broker_account_fingerprint",
        "created_ts",
    ),
    "pending_entry_recovery": (
        "version",
        "strategy_id",
        "symbol",
        "status",
        "broker_account_fingerprint",
        "original_pending_client_id",
        "recovery_client_id",
        "recovery_side",
        "recovery_qty",
        "observed_position_qty",
        "attempt_count",
        "first_detected_at",
        "last_updated_at",
    ),
    "risk_recovery_incident": (
        "version",
        "strategy_id",
        "symbol",
        "cause",
        "status",
        "broker_account_fingerprint",
        "recovery_client_id",
        "recovery_side",
        "recovery_qty",
        "attempt_count",
        "first_detected_at",
        "last_updated_at",
    ),
    "flatten_intent": (
        "version",
        "phase",
        "strategy_id",
        "symbol",
        "side",
        "order_type",
        "client_id",
        "broker_account_fingerprint",
        "qty",
        "quote_budget",
        "reduce_only",
        "submission_kind",
        "created_ts",
    ),
}
CLIENT_ORDER_ID_SAFE_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/-"
)
JOB_OUTPUT_PATH_FLAGS = {
    "--json-output",
    "--markdown-output",
    "--output",
    "--output-json",
    "--output-md",
    "--report",
    "--state",
}
REQUIRED_CORE_PRODUCTS = {
    "btc_accumulation": "btc_accumulation",
    "active_income": "active_income",
}
REQUIRED_CORE_JOBS = (
    "market_universe_screen",
    "market_data_update_universe",
    "market_data_update_universe_1m",
    "market_data_update_futures",
    "market_data_update_futures_1m",
    "market_data_update_spot",
    "regime_tag_futures_15m",
    "research_synthetic_smoke",
    "research_factory",
    "research_cycle",
    "strategy_framework_smoke",
    "active_income_promotion_review",
    "btc_accumulation_promotion_review",
    "runtime_maintenance",
    "artifact_hygiene",
)
REQUIRED_CORE_JOB_MODULES = {
    "market_universe_screen": "src.autopilot.market_universe",
    "market_data_update_universe": "src.autopilot.universe_history",
    "market_data_update_universe_1m": "src.autopilot.universe_history",
    "market_data_update_futures": "src.autopilot.history_bootstrap",
    "market_data_update_futures_1m": "src.autopilot.history_bootstrap",
    "market_data_update_spot": "src.autopilot.history_bootstrap",
    "regime_tag_futures_15m": "src.regime",
    "research_synthetic_smoke": "src.autopilot.research_smoke",
    "research_factory": "src.autopilot.research_factory",
    "research_cycle": "src.autopilot.research_cycle",
    "strategy_framework_smoke": "src.autopilot.strategy_smoke",
    "active_income_promotion_review": "src.autopilot.promotion",
    "btc_accumulation_promotion_review": "src.autopilot.promotion",
    "runtime_maintenance": "src.autopilot.maintenance",
    "artifact_hygiene": "src.autopilot.artifact_hygiene",
}
REQUIRED_CORE_JOB_FLAG_VALUES = {
    "market_universe_screen": {
        "--config": ("config/market_universe.json",),
        "--output": ("runtime/market_universe.json",),
        "--snapshot-dir": ("runtime/market_universe_snapshots",),
    },
    "market_data_update_universe": {
        "--config": ("config/research_factory.json",),
        "--market-universe-report": ("runtime/market_universe.json",),
        "--exclude-timeframes": ("1m",),
        "--output": ("runtime/universe_history.json",),
    },
    "market_data_update_universe_1m": {
        "--config": ("config/research_factory.json",),
        "--market-universe-report": ("runtime/market_universe.json",),
        "--timeframes": ("1m",),
        "--output": ("runtime/universe_history_1m.json",),
    },
    "market_data_update_futures": {
        "--config": ("config/research_factory.json",),
        "--market": ("futures",),
        "--exclude-timeframes": ("1m",),
        "--report": ("runtime/history_bootstrap_futures.json",),
    },
    "market_data_update_futures_1m": {
        "--config": ("config/research_factory.json",),
        "--market": ("futures",),
        "--timeframes": ("1m",),
        "--report": ("runtime/history_bootstrap_futures_1m.json",),
    },
    "market_data_update_spot": {
        "--config": ("config/research_factory.json",),
        "--market": ("spot",),
        "--report": ("runtime/history_bootstrap_spot.json",),
    },
    "regime_tag_futures_15m": {
        "--market": ("futures",),
        "--timeframe": ("15m",),
        "--daily-timeframe": ("1d",),
        "--output": ("runtime/regime/futures_15m_regime.parquet",),
        "--report": ("runtime/regime_tag_futures_15m.json",),
    },
    "research_synthetic_smoke": {
        "--output": ("runtime/research_smoke.json",),
    },
    "research_factory": {
        "--config": ("config/research_factory.json",),
        "--output": ("runtime/research/generated_hypotheses.json",),
    },
    "research_cycle": {
        "--output": ("runtime/research_cycle.json",),
        "--state": ("runtime/research_cycle_state.json",),
        "--generated-batch": ("runtime/research/generated_hypotheses.json",),
        "--research-factory-config": ("config/research_factory.json",),
    },
    "strategy_framework_smoke": {
        "--output": ("runtime/strategy_framework_smoke.json",),
        "--regime-input": ("runtime/regime/futures_15m_regime.parquet",),
    },
    "active_income_promotion_review": {
        "--config": ("config/autopilot.json",),
        "--product": ("active_income",),
        "--artifact": ("outputs/active_strategies_flow.json",),
        "--trade-log": ("runtime/active_income_trades.csv",),
        "--output-json": ("runtime/active_income_promotion_review.json",),
        "--output-md": ("runtime/active_income_promotion_review.md",),
    },
    "btc_accumulation_promotion_review": {
        "--config": ("config/autopilot.json",),
        "--product": ("btc_accumulation",),
        "--artifact": ("outputs/active_strategies_position.json",),
        "--trade-log": ("runtime/btc_accumulation_trades.csv",),
        "--output-json": ("runtime/btc_accumulation_promotion_review.json",),
        "--output-md": ("runtime/btc_accumulation_promotion_review.md",),
    },
    "runtime_maintenance": {
        "--config": ("config/autopilot.json",),
        "--max-quarantine-bytes": ("268435456",),
    },
    "artifact_hygiene": {
        "--config": ("config/autopilot.json",),
        "--output": ("runtime/artifact_hygiene.json",),
    },
}
REQUIRED_CORE_JOB_PRESENCE_FLAGS = {
    "regime_tag_futures_15m": ("--compact", "--skip-if-missing"),
    "research_cycle": ("--include-generated", "--generated-only"),
}
REQUIRED_CORE_JOB_FORBIDDEN_FLAGS = {
    "market_data_update_universe": ("--timeframes",),
    "market_data_update_universe_1m": ("--exclude-timeframes",),
    "market_data_update_futures": ("--timeframes",),
    "market_data_update_futures_1m": ("--exclude-timeframes",),
    "market_data_update_spot": ("--timeframes", "--exclude-timeframes"),
}


def _is_path_command(executable: str) -> bool:
    return (
        executable.startswith(".")
        or os.sep in executable
        or bool(os.altsep and os.altsep in executable)
    )


def _resolve_job_executable(job: JobConfig) -> Path | None:
    executable = job.command[0]
    if not _is_path_command(executable):
        found = shutil.which(executable)
        return Path(found) if found else None
    path = Path(executable)
    return path if path.is_absolute() else job.working_dir / path


def _validate_python_module_job(job: JobConfig) -> list[str]:
    executable_name = Path(job.command[0]).name
    if not executable_name.startswith("python") or "-m" not in job.command:
        return []
    module_index = job.command.index("-m") + 1
    if module_index >= len(job.command) or job.command[module_index].startswith("-"):
        return [f"job {job.name}: python -m command is missing a module name"]
    module_name = job.command[module_index]
    try:
        spec = importlib.util.find_spec(module_name)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        return [f"job {job.name}: python module {module_name!r} is not importable: {exc}"]
    if spec is None:
        return [f"job {job.name}: python module {module_name!r} is not importable"]
    return []


def _validate_python_module_dependencies(job: JobConfig) -> list[str]:
    """Import a configured job with its own interpreter to catch missing deps.

    ``find_spec`` only proves that the top-level module exists.  A lean server
    can still pass that check and fail later when the module imports scipy,
    sklearn, ccxt, or another transitive dependency.  This heavier probe is
    reserved for explicit startup/readiness validation, not every 60-second
    supervision cycle.
    """
    module_name = _job_python_module(job)
    if module_name is None or not job.enabled:
        return []
    executable = _resolve_job_executable(job)
    if executable is None or not executable.exists() or not os.access(executable, os.X_OK):
        return []
    command = [
        str(executable),
        "-c",
        "import importlib,sys; importlib.import_module(sys.argv[1])",
        module_name,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=job.working_dir,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            f"job {job.name}: could not verify python module {module_name!r} dependencies: {exc}"
        ]
    if result.returncode == 0:
        return []
    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip().splitlines()
    tail = detail[-1] if detail else f"exit {result.returncode}"
    return [f"job {job.name}: python module {module_name!r} dependency import failed: {tail}"]


def _job_python_module(job: JobConfig) -> str | None:
    executable_name = Path(job.command[0]).name
    if not executable_name.startswith("python") or "-m" not in job.command:
        return None
    module_index = job.command.index("-m") + 1
    if module_index >= len(job.command) or job.command[module_index].startswith("-"):
        return None
    return job.command[module_index]


def _job_flag_values(command: list[str], flag: str) -> list[str] | None:
    values: list[str] = []
    found = False
    prefix = f"{flag}="
    for index, part in enumerate(command):
        if part.startswith(prefix):
            found = True
            value = part[len(prefix) :]
            if value:
                values.append(value)
            continue
        if part != flag:
            continue
        found = True
        value_index = index + 1
        while value_index < len(command) and not command[value_index].startswith("--"):
            values.append(command[value_index])
            value_index += 1
    return values if found else None


def _job_has_flag(command: list[str], flag: str) -> bool:
    return flag in command


def _validate_job_command(job: JobConfig) -> list[str]:
    if not job.command:
        return [f"job {job.name}: command cannot be empty"]
    errors: list[str] = []
    if not job.working_dir.exists():
        errors.append(f"job {job.name}: working_dir does not exist: {job.working_dir}")
    elif not job.working_dir.is_dir():
        errors.append(f"job {job.name}: working_dir is not a directory: {job.working_dir}")
    executable_name = Path(job.command[0]).name
    if executable_name in SHELL_EXECUTABLES:
        errors.append(
            f"job {job.name}: command must not use a shell executable ({executable_name})"
        )
    executable_path = _resolve_job_executable(job)
    if executable_path is None:
        errors.append(f"job {job.name}: executable not found on PATH: {job.command[0]}")
    elif not executable_path.exists():
        errors.append(f"job {job.name}: executable does not exist: {executable_path}")
    elif not os.access(executable_path, os.X_OK):
        errors.append(f"job {job.name}: executable is not executable: {executable_path}")
    errors.extend(_validate_python_module_job(job))
    module_name = _job_python_module(job)
    if module_name in DISALLOWED_JOB_MODULES:
        errors.append(
            f"job {job.name}: scheduled jobs must not run approval-gate module {module_name}"
        )
    return errors


def _job_output_paths(job: JobConfig) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    command = list(job.command)
    for index, part in enumerate(command):
        inline_match = next(
            (
                (flag, part[len(flag) + 1 :])
                for flag in JOB_OUTPUT_PATH_FLAGS
                if part.startswith(f"{flag}=")
            ),
            None,
        )
        if inline_match is not None:
            flag, value = inline_match
            if not value:
                continue
            path = Path(value)
            paths.append((flag, path if path.is_absolute() else job.working_dir / path))
            continue
        if part not in JOB_OUTPUT_PATH_FLAGS:
            continue
        value_index = index + 1
        if value_index >= len(command):
            continue
        value = command[value_index]
        if value.startswith("--"):
            continue
        path = Path(value)
        paths.append((part, path if path.is_absolute() else job.working_dir / path))
    return paths


def _job_output_path_errors(job: JobConfig) -> list[str]:
    errors: list[str] = []
    command = list(job.command)
    for index, part in enumerate(command):
        inline_match = next(
            (
                (flag, part[len(flag) + 1 :])
                for flag in JOB_OUTPUT_PATH_FLAGS
                if part.startswith(f"{flag}=")
            ),
            None,
        )
        if inline_match is not None:
            flag, value = inline_match
            if not value:
                errors.append(f"job {job.name}: output flag {flag} must include a path")
            continue
        if part not in JOB_OUTPUT_PATH_FLAGS:
            continue
        value_index = index + 1
        if value_index >= len(command) or command[value_index].startswith("--"):
            errors.append(f"job {job.name}: output flag {part} must include a path")
    return errors


def _protected_job_output_paths(config: AutopilotConfig) -> dict[Path, str]:
    protected: dict[Path, str] = {}
    for field_name in (
        "control_file",
        "control_audit_file",
        "status_file",
        "lock_file",
        "approval_ledger",
        "job_state_file",
        "alert_file",
        "alert_state_file",
    ):
        path = getattr(config, field_name)
        protected[path.resolve(strict=False)] = field_name
    for product in config.products:
        for field_name in (
            "strategies_path",
            "state_file",
            "trade_log",
            "preflight_report",
            "testnet_rehearsal_report",
        ):
            path = getattr(product, field_name)
            if path is None:
                continue
            if field_name == "testnet_rehearsal_report" and not product.require_testnet_rehearsal:
                continue
            protected[path.resolve(strict=False)] = f"{product.name} {field_name}"
    return protected


def _config_owned_paths(config: AutopilotConfig) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for field_name in (
        "control_file",
        "control_audit_file",
        "status_file",
        "lock_file",
        "approval_ledger",
        "job_state_file",
        "alert_file",
        "alert_state_file",
        "research_smoke_file",
        "strategy_smoke_file",
        "research_cycle_file",
        "research_factory_config_file",
        "generated_batch_file",
        "experiment_memory_file",
        "experiment_memory_backup_file",
        "incubation_candidates_file",
        "mutation_plan_file",
        "mutation_batch_file",
        "artifact_hygiene_file",
        "backup_report_file",
        "operator_report_file",
        "operator_report_json_file",
        "readiness_report_file",
        "readiness_report_json_file",
    ):
        paths.append((field_name, getattr(config, field_name)))
    for product in config.products:
        for field_name in (
            "strategies_path",
            "state_file",
            "trade_log",
            "preflight_report",
            "testnet_rehearsal_report",
        ):
            path = getattr(product, field_name)
            if path is None:
                continue
            if field_name == "testnet_rehearsal_report" and not product.require_testnet_rehearsal:
                continue
            paths.append((f"{product.name} {field_name}", path))
    return paths


def _config_path_collision_errors(config: AutopilotConfig) -> list[str]:
    errors: list[str] = []
    seen: dict[Path, str] = {}
    for owner, path in _config_owned_paths(config):
        normalized = path.resolve(strict=False)
        if previous := seen.get(normalized):
            errors.append(f"{owner} path duplicates {previous}: {path}")
        else:
            seen[normalized] = owner
    return errors


def validate_config(
    config: AutopilotConfig,
    *,
    require_core_products: bool = False,
    require_core_jobs: bool = False,
    verify_job_imports: bool = False,
    validate_jobs: bool = True,
) -> list[str]:
    errors: list[str] = []
    names: set[str] = set()
    product_paths: dict[str, dict[Path, str]] = {
        "strategies_path": {},
        "state_file": {},
        "trade_log": {},
        "preflight_report": {},
        "testnet_rehearsal_report": {},
    }
    for product in config.products:
        if product.name in names:
            errors.append(f"duplicate product name: {product.name}")
        names.add(product.name)
        for field_name, seen_paths in product_paths.items():
            path = getattr(product, field_name)
            if path is None:
                continue
            if field_name == "testnet_rehearsal_report" and not product.require_testnet_rehearsal:
                continue
            normalized = path.resolve(strict=False)
            if normalized in seen_paths:
                errors.append(
                    f"{product.name}: {field_name} duplicates {seen_paths[normalized]}: {path}"
                )
            else:
                seen_paths[normalized] = product.name
        if product.objective not in {"btc_accumulation", "active_income"}:
            errors.append(
                f"{product.name}: objective must be 'btc_accumulation' or 'active_income'"
            )
        if product.market not in {"spot", "futures"}:
            errors.append(f"{product.name}: market must be 'spot' or 'futures'")
        if product.execution_mode not in {"paper", "live"}:
            errors.append(f"{product.name}: execution_mode must be 'paper' or 'live'")
        if product.objective == "btc_accumulation" and product.base_asset.upper() != "BTC":
            errors.append(f"{product.name}: btc_accumulation must use base_asset BTC")
        if product.objective == "active_income" and product.base_asset.upper() != "USDT":
            errors.append(f"{product.name}: active_income must use base_asset USDT")
        if product.objective == "btc_accumulation" and product.market != "spot":
            errors.append(f"{product.name}: BTC accumulation must use spot market")
        if product.objective == "active_income" and product.market != "futures":
            errors.append(f"{product.name}: active income must use futures market")
        errors.extend(validate_product_symbol_policy(product))
        if product.preflight_max_age_seconds <= 0:
            errors.append(f"{product.name}: preflight_max_age_seconds must be positive")
        if product.testnet_rehearsal_max_age_seconds <= 0:
            errors.append(f"{product.name}: testnet_rehearsal_max_age_seconds must be positive")
        if product.execution_mode == "live" and not product.require_preflight:
            errors.append(f"{product.name}: live execution requires require_preflight=true")
        if product.execution_mode == "live" and product.preflight_report is None:
            errors.append(f"{product.name}: live execution requires a preflight_report path")
        if product.require_testnet_rehearsal and product.objective != "active_income":
            errors.append(
                f"{product.name}: testnet rehearsal gate is only supported for active_income futures"
            )
        if product.require_testnet_rehearsal and product.testnet_rehearsal_report is None:
            errors.append(
                f"{product.name}: testnet rehearsal gate requires a testnet_rehearsal_report path"
            )
        if (
            product.execution_mode == "live"
            and product.objective == "active_income"
            and product.market == "futures"
            and not product.require_testnet_rehearsal
        ):
            errors.append(
                f"{product.name}: active-income live execution requires require_testnet_rehearsal=true"
            )
    if config.loop_sleep_seconds <= 0:
        errors.append("loop_sleep_seconds must be positive")
    if validate_jobs and config.max_jobs_per_cycle <= 0:
        errors.append("max_jobs_per_cycle must be positive")
    if validate_jobs and config.max_consecutive_job_deferrals <= 0:
        errors.append("max_consecutive_job_deferrals must be positive")
    if config.min_runtime_free_bytes <= 0:
        errors.append("min_runtime_free_bytes must be positive")
    if validate_jobs and config.run_data_update:
        errors.append(
            "run_data_update=true is unsupported because inline downloads can block trading "
            "supervision; use the isolated market_data_update_* jobs"
        )
    # These are runtime- and product-owned paths, so supervision-only mode must
    # still reject collisions that could overwrite trading state.
    errors.extend(_config_path_collision_errors(config))
    if validate_jobs:
        job_names: set[str] = set()
        job_output_paths: dict[Path, str] = {}
        protected_output_paths = _protected_job_output_paths(config)
        for job in config.jobs:
            if job.name in job_names:
                errors.append(f"duplicate job name: {job.name}")
            job_names.add(job.name)
            if job.cadence_seconds <= 0:
                errors.append(f"job {job.name}: cadence_seconds must be positive")
            if job.timeout_seconds <= 0:
                errors.append(f"job {job.name}: timeout_seconds must be positive")
            errors.extend(_validate_job_command(job))
            errors.extend(_job_output_path_errors(job))
            for flag, path in _job_output_paths(job):
                normalized = path.resolve(strict=False)
                owner = f"{job.name} {flag}"
                if protected_owner := protected_output_paths.get(normalized):
                    errors.append(
                        f"job {job.name}: output path {path} for {flag} "
                        f"targets protected runtime file {protected_owner}"
                    )
                elif normalized in job_output_paths:
                    errors.append(
                        f"job {job.name}: output path {path} for {flag} "
                        f"duplicates {job_output_paths[normalized]}"
                    )
                else:
                    job_output_paths[normalized] = owner
    if validate_jobs and verify_job_imports:
        verified: set[tuple[Path | None, Path, str | None]] = set()
        for job in config.jobs:
            key = (
                _resolve_job_executable(job),
                job.working_dir.resolve(strict=False),
                _job_python_module(job),
            )
            if key in verified:
                continue
            verified.add(key)
            errors.extend(_validate_python_module_dependencies(job))
    if config.alert_cooldown_seconds < 0:
        errors.append("alert_cooldown_seconds must be non-negative")
    if (
        isinstance(config.active_income_max_open_positions, bool)
        or not isinstance(config.active_income_max_open_positions, int)
        or not 1 <= config.active_income_max_open_positions <= 20
    ):
        errors.append("active_income_max_open_positions must be an integer in [1, 20]")
    if require_core_products:
        errors.extend(_core_product_errors(config.products))
    if validate_jobs and require_core_jobs:
        errors.extend(_core_job_errors(config.jobs))
    return errors


def _core_product_errors(products: list[ProductConfig]) -> list[str]:
    errors: list[str] = []
    by_name = {product.name: product for product in products}
    for product_name, objective in REQUIRED_CORE_PRODUCTS.items():
        product = by_name.get(product_name)
        if product is None:
            errors.append(f"missing required product: {product_name}")
            continue
        if product.objective != objective:
            errors.append(f"{product_name}: required product must use objective {objective}")
        if not product.enabled:
            errors.append(f"{product_name}: required product must be enabled")
    return errors


def _core_job_errors(jobs: list[JobConfig]) -> list[str]:
    errors: list[str] = []
    by_name = {job.name: job for job in jobs}
    for job_name in REQUIRED_CORE_JOBS:
        job = by_name.get(job_name)
        if job is None:
            errors.append(f"missing required job: {job_name}")
            continue
        if not job.enabled:
            errors.append(f"{job_name}: required job must be enabled")
        expected_module = REQUIRED_CORE_JOB_MODULES[job_name]
        actual_module = _job_python_module(job)
        if actual_module != expected_module:
            errors.append(
                f"{job_name}: required job must run python module {expected_module}"
                f" (got {actual_module or 'none'})"
            )
        for flag in REQUIRED_CORE_JOB_PRESENCE_FLAGS.get(job_name, ()):
            if not _job_has_flag(job.command, flag):
                errors.append(f"{job_name}: required job must include {flag}")
        for flag in REQUIRED_CORE_JOB_FORBIDDEN_FLAGS.get(job_name, ()):
            if _job_flag_values(job.command, flag) is not None:
                errors.append(f"{job_name}: required job must not include {flag}")
        for flag, required_values in REQUIRED_CORE_JOB_FLAG_VALUES.get(job_name, {}).items():
            actual_values = _job_flag_values(job.command, flag)
            if actual_values is None:
                errors.append(
                    f"{job_name}: required job must include {flag} {' '.join(required_values)}"
                )
                continue
            if tuple(actual_values) != required_values:
                errors.append(
                    f"{job_name}: required job {flag} must equal {' '.join(required_values)}"
                    f" (got {' '.join(actual_values) or 'none'})"
                )
    return errors


def _product_to_status(product: ProductConfig) -> dict[str, Any]:
    payload = asdict(product)
    for key in (
        "strategies_path",
        "state_file",
        "trade_log",
        "preflight_report",
        "testnet_rehearsal_report",
    ):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload


def build_live_broker(product: ProductConfig):
    from src.execution.ccxt_broker import CcxtBroker
    from src.execution.config import ExchangeConfig

    if product.objective == "btc_accumulation":
        if product.market != "spot":
            raise RuntimeError("BTC accumulation live execution must use spot market routing.")
        cfg = ExchangeConfig.from_env(market_type="spot")
    elif product.objective == "active_income":
        if product.market != "futures":
            raise RuntimeError("Active income live execution must use futures market routing.")
        cfg = ExchangeConfig.from_env(market_type="futures")
    else:
        raise RuntimeError(f"Unsupported live product objective: {product.objective!r}.")
    errors = validate_exchange_policy(product, cfg)
    if errors:
        raise RuntimeError(
            f"{product.name}: live broker exchange policy is not ready: " + "; ".join(errors)
        )
    return CcxtBroker(cfg)


def assert_live_environment(
    product: ProductConfig,
    *,
    require_production: bool = False,
) -> dict[str, Any]:
    market_type = "spot" if product.objective == "btc_accumulation" else "futures"
    try:
        from src.execution.config import ExchangeConfig

        cfg = ExchangeConfig.from_env(market_type=market_type)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{product.name}: invalid live exchange environment: {exc}") from exc

    errors: list[str] = []
    if not cfg.live:
        errors.append("TRADING_LIVE must be 1")
    if require_production and cfg.testnet:
        errors.append(
            "EXCHANGE_TESTNET must be 0 for the live runtime; use the separate testnet rehearsal"
        )
    if cfg.max_notional_usd <= 0:
        errors.append("MAX_NOTIONAL_USD must be positive")
    if cfg.max_fill_slippage_bps <= 0:
        errors.append("MAX_FILL_SLIPPAGE_BPS must be positive")
    if cfg.market_type == "futures" and not (1 <= cfg.max_futures_leverage <= 3):
        errors.append("MAX_FUTURES_LEVERAGE must be between 1 and 3")
    if (
        product.objective == "active_income"
        and cfg.market_type == "futures"
        and cfg.max_futures_leverage != ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
    ):
        errors.append(
            f"active income futures must use MAX_FUTURES_LEVERAGE={ACTIVE_INCOME_MAX_FUTURES_LEVERAGE}"
        )
    if cfg.market_type == "futures" and cfg.futures_margin_mode != "isolated":
        errors.append("FUTURES_MARGIN_MODE must be 'isolated'")
    if not cfg.api_key or not cfg.api_secret:
        errors.append("EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required")
    errors.extend(validate_exchange_policy(product, cfg))
    if errors:
        raise RuntimeError(
            f"{product.name}: live execution environment is not ready: " + "; ".join(errors)
        )
    detail = {
        "ok": True,
        "exchange": cfg.exchange,
        "market_type": cfg.market_type,
        "testnet": cfg.testnet,
        "account_fingerprint": cfg.account_fingerprint,
        "max_notional_usd": cfg.max_notional_usd,
        "max_fill_slippage_bps": cfg.max_fill_slippage_bps,
        "quote_asset": cfg.quote_asset,
    }
    if cfg.market_type == "futures":
        detail["max_futures_leverage"] = cfg.max_futures_leverage
        detail["futures_margin_mode"] = cfg.futures_margin_mode
    return detail


def _assert_current_environment_matches_preflight(
    product: ProductConfig,
    *,
    current: dict[str, Any],
    recorded: dict[str, Any],
) -> None:
    """Require the production broker settings to equal the saved preflight."""

    keys = [
        "exchange",
        "market_type",
        "testnet",
        "account_fingerprint",
        "quote_asset",
        "max_notional_usd",
        "max_fill_slippage_bps",
    ]
    if product.market == "futures":
        keys.extend(("max_futures_leverage", "futures_margin_mode"))
    mismatches = [
        f"{key}: preflight={recorded.get(key)!r} current={current.get(key)!r}"
        for key in keys
        if recorded.get(key) != current.get(key)
    ]
    if recorded.get("testnet") is not False:
        mismatches.append(
            f"testnet: production preflight must record false, got {recorded.get('testnet')!r}"
        )
    if mismatches:
        raise RuntimeError(
            f"{product.name}: current live exchange environment does not match the production "
            "preflight; run a fresh connected preflight: " + "; ".join(mismatches)
        )


def _required_preflight_checks(product: ProductConfig) -> tuple[str, ...]:
    checks = list(PREFLIGHT_REQUIRED_CHECKS)
    if product.objective == "active_income" and product.market == "futures":
        checks.append("broker_position_mode_one_way")
        checks.append("broker_native_protective_stops")
        checks.append("broker_open_orders_empty")
        checks.append("broker_position_flat")
    if product.objective == "btc_accumulation" and product.market == "spot":
        checks.append("broker_spot_position_non_negative")
    return tuple(checks)


def _assert_preflight_checks_passed(
    product: ProductConfig, matched: dict[str, Any], *, label: str
) -> list[str]:
    checks = matched.get("checks")
    if not isinstance(checks, list):
        raise RuntimeError(f"{product.name}: {label} checks must be a list.")
    check_by_name: dict[str, dict[str, Any]] = {}
    for item in checks:
        if not isinstance(item, dict):
            raise RuntimeError(f"{product.name}: {label} checks must contain JSON objects.")
        name = item.get("name")
        if isinstance(name, str) and name:
            check_by_name[name] = item
    passed: list[str] = []
    for name in _required_preflight_checks(product):
        check = check_by_name.get(name)
        if check is None:
            raise RuntimeError(f"{product.name}: {label} missing required check {name}.")
        if check.get("ok") is not True:
            detail = check.get("error") or check.get("detail") or "not ok"
            raise RuntimeError(f"{product.name}: {label} required check {name} failed: {detail}")
        passed.append(name)
    return passed


def _assert_preflight_product_identity(
    product: ProductConfig,
    reported_product: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(reported_product, dict):
        raise RuntimeError(f"{product.name}: {label} product payload must be a JSON object.")
    expected = canonical_product_config(product)
    for field, expected_value in expected.items():
        if field not in reported_product:
            raise RuntimeError(f"{product.name}: {label} product {field} is missing.")
        actual_value = reported_product[field]
        if actual_value != expected_value:
            raise RuntimeError(
                f"{product.name}: {label} product {field} mismatch: "
                f"{actual_value!r} != {expected_value!r}."
            )
    extra_fields = sorted(set(reported_product) - set(expected))
    if extra_fields:
        raise RuntimeError(
            f"{product.name}: {label} product payload has unexpected fields: "
            f"{', '.join(extra_fields)}."
        )
    return reported_product


def _preflight_check_by_name(
    matched: dict[str, Any], name: str, *, label: str, product: ProductConfig
) -> dict[str, Any]:
    checks = matched.get("checks")
    if not isinstance(checks, list):
        raise RuntimeError(f"{product.name}: {label} checks must be a list.")
    for item in checks:
        if not isinstance(item, dict):
            raise RuntimeError(f"{product.name}: {label} checks must contain JSON objects.")
        if item.get("name") == name:
            return item
    raise RuntimeError(f"{product.name}: {label} missing required check {name}.")


def _positive_evidence_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _assert_preflight_exchange_evidence(
    product: ProductConfig,
    matched: dict[str, Any],
    *,
    label: str,
    require_testnet: bool = False,
) -> dict[str, Any]:
    check = _preflight_check_by_name(matched, "exchange_environment", label=label, product=product)
    detail = check.get("detail")
    if not isinstance(detail, dict):
        raise RuntimeError(f"{product.name}: {label} exchange environment evidence is missing.")
    if detail.get("custom_checker"):
        raise RuntimeError(
            f"{product.name}: {label} exchange environment evidence used a custom checker."
        )
    expected_market = "spot" if product.objective == "btc_accumulation" else "futures"
    actual_market = str(detail.get("market_type") or "").lower()
    if actual_market != expected_market:
        raise RuntimeError(
            f"{product.name}: {label} exchange market mismatch: "
            f"{actual_market or '<missing>'} != {expected_market}."
        )
    if str(detail.get("quote_asset") or "").upper() != "USDT":
        raise RuntimeError(f"{product.name}: {label} exchange quote asset must be USDT.")
    account_fingerprint = detail.get("account_fingerprint")
    if not isinstance(account_fingerprint, str) or not account_fingerprint.startswith(
        ACCOUNT_FINGERPRINT_PREFIX
    ):
        raise RuntimeError(f"{product.name}: {label} account fingerprint evidence is invalid.")
    fingerprint_digest = account_fingerprint.removeprefix(ACCOUNT_FINGERPRINT_PREFIX)
    if len(fingerprint_digest) != 64 or any(
        char not in "0123456789abcdef" for char in fingerprint_digest
    ):
        raise RuntimeError(f"{product.name}: {label} account fingerprint evidence is invalid.")
    exchange = str(detail.get("exchange") or "").lower()
    if product.objective == "active_income" and exchange not in ACTIVE_INCOME_FUTURES_EXCHANGES:
        allowed = ", ".join(sorted(ACTIVE_INCOME_FUTURES_EXCHANGES))
        raise RuntimeError(
            f"{product.name}: {label} exchange mismatch: {exchange or '<missing>'} not in {allowed}."
        )
    if product.objective == "btc_accumulation" and exchange not in BTC_ACCUMULATION_SPOT_EXCHANGES:
        allowed = ", ".join(sorted(BTC_ACCUMULATION_SPOT_EXCHANGES))
        raise RuntimeError(
            f"{product.name}: {label} exchange mismatch: {exchange or '<missing>'} not in {allowed}."
        )
    if require_testnet and detail.get("require_testnet") is not True:
        raise RuntimeError(f"{product.name}: {label} did not require testnet during preflight.")
    if require_testnet and detail.get("testnet") is not True:
        raise RuntimeError(f"{product.name}: {label} was not run against testnet.")
    if _positive_evidence_float(detail.get("max_notional_usd")) is None:
        raise RuntimeError(f"{product.name}: {label} max_notional_usd evidence is invalid.")
    if _positive_evidence_float(detail.get("max_fill_slippage_bps")) is None:
        raise RuntimeError(f"{product.name}: {label} max_fill_slippage_bps evidence is invalid.")
    if expected_market == "futures":
        try:
            leverage = int(detail.get("max_futures_leverage"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{product.name}: {label} max_futures_leverage evidence is invalid."
            ) from exc
        if not (1 <= leverage <= 3):
            raise RuntimeError(f"{product.name}: {label} max_futures_leverage evidence is invalid.")
        if product.objective == "active_income" and leverage != ACTIVE_INCOME_MAX_FUTURES_LEVERAGE:
            raise RuntimeError(
                f"{product.name}: {label} max_futures_leverage evidence must be "
                f"{ACTIVE_INCOME_MAX_FUTURES_LEVERAGE}."
            )
        margin_mode = str(detail.get("futures_margin_mode") or "").lower()
        if margin_mode != "isolated":
            raise RuntimeError(
                f"{product.name}: {label} futures margin mode evidence is not isolated."
            )
    return detail


def _assert_preflight_position_mode_evidence(
    product: ProductConfig,
    matched: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    if product.objective != "active_income" or product.market != "futures":
        return None
    check = _preflight_check_by_name(
        matched,
        "broker_position_mode_one_way",
        label=label,
        product=product,
    )
    detail = check.get("detail")
    if not isinstance(detail, dict):
        raise RuntimeError(f"{product.name}: {label} one-way position-mode evidence is missing.")
    if str(detail.get("symbol") or "").upper() != product.symbol.upper():
        raise RuntimeError(f"{product.name}: {label} position-mode evidence symbol is invalid.")
    if detail.get("one_way") is not True:
        raise RuntimeError(f"{product.name}: {label} did not prove one-way position mode.")
    return detail


def _evidence_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_account_fingerprint(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(ACCOUNT_FINGERPRINT_PREFIX):
        return False
    digest = value.removeprefix(ACCOUNT_FINGERPRINT_PREFIX)
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _live_broker_account_fingerprint(product: ProductConfig, broker: Any) -> str:
    fingerprint = getattr(broker, "account_fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if not _valid_account_fingerprint(fingerprint):
        raise RuntimeError(
            f"{product.name}: live broker has no valid non-secret account fingerprint."
        )
    return fingerprint


def _assert_preflight_connectivity_evidence(
    product: ProductConfig,
    matched: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    check = _preflight_check_by_name(
        matched, "exchange_read_connectivity", label=label, product=product
    )
    detail = check.get("detail")
    if not isinstance(detail, dict):
        raise RuntimeError(f"{product.name}: {label} exchange connectivity evidence is missing.")
    price = _evidence_float(detail.get("price"))
    if price is None or price <= 0:
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity price evidence is invalid."
        )
    balance = _evidence_float(detail.get("balance"))
    if balance is None or balance <= 0:
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity balance evidence is invalid."
        )
    position_qty = _evidence_float(detail.get("position_qty"))
    if position_qty is None:
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity position_qty evidence is invalid."
        )
    position_avg_price = _evidence_float(detail.get("position_avg_price"))
    if position_avg_price is None or position_avg_price < 0:
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity position_avg_price evidence is invalid."
        )
    if not isinstance(detail.get("position_is_flat"), bool):
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity position_is_flat evidence is invalid."
        )
    if (
        product.objective == "active_income"
        and product.market == "futures"
        and detail["position_is_flat"] is not True
    ):
        raise RuntimeError(f"{product.name}: {label} exchange connectivity position is not flat.")
    if product.objective == "btc_accumulation" and product.market == "spot" and position_qty < 0:
        raise RuntimeError(
            f"{product.name}: {label} exchange connectivity spot position is negative."
        )
    return detail


def _assert_preflight_open_order_evidence(
    product: ProductConfig,
    matched: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    if product.objective != "active_income" or product.market != "futures":
        return None
    check = _preflight_check_by_name(
        matched,
        "broker_open_orders_empty",
        label=label,
        product=product,
    )
    detail = check.get("detail")
    if not isinstance(detail, dict):
        raise RuntimeError(f"{product.name}: {label} open-order evidence is missing.")
    if detail.get("scope") != "whole_account":
        raise RuntimeError(f"{product.name}: {label} open-order evidence is not account-wide.")
    if str(detail.get("configured_symbol") or "").upper() != product.symbol.upper():
        raise RuntimeError(f"{product.name}: {label} open-order configured symbol is invalid.")
    for order_kind in ("regular", "conditional"):
        inventory = detail.get(order_kind)
        if not isinstance(inventory, dict):
            raise RuntimeError(
                f"{product.name}: {label} {order_kind} open-order evidence is missing."
            )
        count = inventory.get("count")
        orders = inventory.get("orders")
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise RuntimeError(
                f"{product.name}: {label} {order_kind} open-order count is not zero."
            )
        if not isinstance(orders, list) or orders:
            raise RuntimeError(
                f"{product.name}: {label} {order_kind} open-order list is not empty."
            )
    return detail


def _assert_preflight_position_inventory_evidence(
    product: ProductConfig,
    matched: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    if product.objective != "active_income" or product.market != "futures":
        return None
    check = _preflight_check_by_name(
        matched,
        "broker_position_flat",
        label=label,
        product=product,
    )
    detail = check.get("detail")
    if not isinstance(detail, dict):
        raise RuntimeError(f"{product.name}: {label} position inventory evidence is missing.")
    if detail.get("scope") != "whole_account":
        raise RuntimeError(
            f"{product.name}: {label} position inventory evidence is not account-wide."
        )
    if str(detail.get("configured_symbol") or "").upper() != product.symbol.upper():
        raise RuntimeError(
            f"{product.name}: {label} position inventory configured symbol is invalid."
        )
    count = detail.get("count")
    positions = detail.get("positions")
    if isinstance(count, bool) or not isinstance(count, int) or count != 0:
        raise RuntimeError(f"{product.name}: {label} account position count is not zero.")
    if not isinstance(positions, list) or positions:
        raise RuntimeError(f"{product.name}: {label} account position list is not empty.")
    return detail


def assert_recent_preflight(
    product: ProductConfig,
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not product.require_preflight:
        if product.execution_mode == "live":
            raise RuntimeError(f"{product.name}: live execution requires require_preflight=true.")
        return {"required": False, "ok": True}
    if product.preflight_report is None:
        raise RuntimeError(f"{product.name}: live execution requires a preflight_report path.")
    if product.preflight_report.is_symlink():
        raise RuntimeError(
            f"{product.name}: preflight report must not be a symlink: {product.preflight_report}"
        )
    if not product.preflight_report.exists():
        raise RuntimeError(
            f"{product.name}: preflight report not found: {product.preflight_report}"
        )

    try:
        report = json.loads(product.preflight_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{product.name}: preflight report read_error: {exc}") from exc
    if not isinstance(report, dict):
        raise RuntimeError(
            f"{product.name}: preflight report must be a JSON object, got {type(report).__name__}."
        )
    generated_ts = report.get("generated_ts")
    if generated_ts is None:
        raise RuntimeError(f"{product.name}: preflight report has no generated_ts.")
    try:
        generated_ts_float = float(generated_ts)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{product.name}: preflight report generated_ts is not numeric."
        ) from exc
    if not math.isfinite(generated_ts_float):
        raise RuntimeError(f"{product.name}: preflight report generated_ts is not finite.")
    age_seconds = time.time() - generated_ts_float
    if age_seconds < -PREFLIGHT_CLOCK_SKEW_SECONDS:
        raise RuntimeError(
            f"{product.name}: preflight report timestamp is in the future "
            f"({abs(age_seconds):.0f}s > {PREFLIGHT_CLOCK_SKEW_SECONDS}s clock skew)."
        )
    if age_seconds > product.preflight_max_age_seconds:
        raise RuntimeError(
            f"{product.name}: preflight report is stale "
            f"({age_seconds:.0f}s > {product.preflight_max_age_seconds}s)."
        )
    if not report.get("ok"):
        raise RuntimeError(f"{product.name}: preflight report failed.")

    products = report.get("products", [])
    if not isinstance(products, list):
        raise RuntimeError(f"{product.name}: preflight report products must be a list.")
    matched = None
    for item in products:
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{product.name}: preflight report products must contain JSON objects."
            )
        item_product = item.get("product", {})
        if not isinstance(item_product, dict):
            raise RuntimeError(
                f"{product.name}: preflight report product payload must be a JSON object."
            )
        if item_product.get("name") == product.name:
            matched = item
            break
    if matched is None:
        raise RuntimeError(f"{product.name}: preflight report does not include this product.")
    if not matched.get("ok"):
        raise RuntimeError(f"{product.name}: product preflight failed.")
    passed_checks = _assert_preflight_checks_passed(product, matched, label="preflight report")
    exchange_evidence = _assert_preflight_exchange_evidence(
        product, matched, label="preflight report"
    )
    connectivity_evidence = _assert_preflight_connectivity_evidence(
        product, matched, label="preflight report"
    )
    reported_product = _assert_preflight_product_identity(
        product,
        matched.get("product"),
        label="preflight report",
    )
    position_mode_evidence = _assert_preflight_position_mode_evidence(
        product,
        matched,
        label="preflight report",
    )
    open_order_evidence = _assert_preflight_open_order_evidence(
        product,
        matched,
        label="preflight report",
    )
    position_inventory_evidence = _assert_preflight_position_inventory_evidence(
        product,
        matched,
        label="preflight report",
    )
    report_artifact = reported_product.get("strategies_path")
    if report_artifact:
        expected = product.strategies_path.resolve()
        actual = Path(report_artifact).resolve()
        if actual != expected:
            raise RuntimeError(
                f"{product.name}: preflight artifact mismatch: {actual} != {expected}."
            )
    reported_fingerprints = matched.get("artifact_fingerprints")
    if not isinstance(reported_fingerprints, list) or not reported_fingerprints:
        raise RuntimeError(f"{product.name}: preflight report has no artifact_fingerprints.")
    artifact_snapshot = artifact if artifact is not None else load_artifact(product.strategies_path)
    current_artifact_digest = artifact_digest(artifact_snapshot)
    reported_artifact_digest = matched.get("artifact_digest")
    if not isinstance(reported_artifact_digest, str) or not reported_artifact_digest:
        raise RuntimeError(f"{product.name}: preflight report has no artifact_digest.")
    if current_artifact_digest != reported_artifact_digest:
        raise RuntimeError(
            f"{product.name}: preflight artifact digest does not match current artifact."
        )
    reported_engine_digest = matched.get("execution_engine_digest")
    current_engine_digest = execution_engine_digest()
    if not isinstance(reported_engine_digest, str) or not reported_engine_digest:
        raise RuntimeError(f"{product.name}: preflight report has no execution_engine_digest.")
    if reported_engine_digest != current_engine_digest:
        raise RuntimeError(
            f"{product.name}: preflight execution engine digest does not match current code."
        )
    current_fingerprints = [
        strategy_fingerprint(strategy) for strategy in artifact_snapshot.get("strategies", [])
    ]
    if current_fingerprints != list(reported_fingerprints):
        raise RuntimeError(
            f"{product.name}: preflight strategy fingerprints do not match current artifact."
        )
    return {
        "required": True,
        "ok": True,
        "report": str(product.preflight_report),
        "age_seconds": round(age_seconds, 3),
        "artifact_fingerprints": current_fingerprints,
        "execution_engine_digest": current_engine_digest,
        "required_checks": passed_checks,
        "exchange_environment": exchange_evidence,
        "position_mode": position_mode_evidence,
        "exchange_connectivity": connectivity_evidence,
        "open_order_inventory": open_order_evidence,
        "position_inventory": position_inventory_evidence,
    }


def assert_recent_testnet_rehearsal(
    product: ProductConfig,
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def fail(message: str) -> None:
        next_action = testnet_rehearsal_next_action()
        raise RuntimeError(
            f"{message} Next: {next_action['preflight_command']} && "
            f"{next_action['rehearsal_command']} && {next_action['status_command']}"
        )

    if not product.require_testnet_rehearsal:
        if (
            product.execution_mode == "live"
            and product.objective == "active_income"
            and product.market == "futures"
        ):
            fail(
                f"{product.name}: active-income live execution requires require_testnet_rehearsal=true."
            )
        return {"required": False, "ok": True}
    if product.objective != "active_income" or product.market != "futures":
        fail(f"{product.name}: testnet rehearsal gate is only supported for active_income futures.")
    if product.testnet_rehearsal_report is None:
        fail(f"{product.name}: live execution requires a testnet_rehearsal_report path.")
    if product.testnet_rehearsal_report.is_symlink():
        fail(
            f"{product.name}: testnet rehearsal report must not be a symlink: {product.testnet_rehearsal_report}"
        )
    status = summarize_testnet_rehearsal_report(
        product.testnet_rehearsal_report,
        max_age_seconds=product.testnet_rehearsal_max_age_seconds,
        expected_product=product,
    )
    if not status.get("exists"):
        fail(
            f"{product.name}: testnet rehearsal report not found: {product.testnet_rehearsal_report}"
        )
    if status.get("status") == "read_error":
        detail = f"; {status.get('error')}" if status.get("error") else ""
        fail(f"{product.name}: testnet rehearsal gate failed: read_error{detail}")
    if status.get("product") not in {None, product.name}:
        fail(
            f"{product.name}: testnet rehearsal product mismatch: {status.get('product')} != {product.name}."
        )
    if status.get("testnet") is not True:
        fail(f"{product.name}: testnet rehearsal report was not produced on testnet.")
    rehearsal_exchange = str(status.get("exchange") or "").lower()
    if rehearsal_exchange not in ACTIVE_INCOME_FUTURES_EXCHANGES:
        allowed = ", ".join(sorted(ACTIVE_INCOME_FUTURES_EXCHANGES))
        fail(
            f"{product.name}: testnet rehearsal exchange mismatch: "
            f"{rehearsal_exchange or '<missing>'} not in {allowed}."
        )
    if status.get("final_position_flat") is not True:
        fail(f"{product.name}: testnet rehearsal final position is not flat.")
    if not status.get("ok"):
        reason = status.get("status") or "failed"
        error = status.get("error")
        invalid_reasons = status.get("invalid_reasons")
        detail = f"; {error}" if error else ""
        if invalid_reasons:
            detail = f"{detail}; invalid_reasons={','.join(map(str, invalid_reasons))}"
        fail(f"{product.name}: testnet rehearsal gate failed: {reason}{detail}")
    report = json.loads(product.testnet_rehearsal_report.read_text(encoding="utf-8"))
    preflight = report.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError(f"{product.name}: testnet rehearsal report has no embedded preflight.")
    if preflight.get("ok") is not True:
        raise RuntimeError(f"{product.name}: testnet rehearsal embedded preflight failed.")
    products = preflight.get("products", [])
    if not isinstance(products, list):
        raise RuntimeError(f"{product.name}: testnet rehearsal preflight products must be a list.")
    matched = None
    for item in products:
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{product.name}: testnet rehearsal preflight products must contain JSON objects."
            )
        item_product = item.get("product", {})
        if not isinstance(item_product, dict):
            raise RuntimeError(
                f"{product.name}: testnet rehearsal preflight product payload must be a JSON object."
            )
        if item_product.get("name") == product.name:
            matched = item
            break
    if matched is None:
        raise RuntimeError(
            f"{product.name}: testnet rehearsal preflight does not include this product."
        )
    if matched.get("ok") is not True:
        raise RuntimeError(f"{product.name}: testnet rehearsal embedded product preflight failed.")
    passed_checks = _assert_preflight_checks_passed(
        product, matched, label="testnet rehearsal preflight"
    )
    exchange_evidence = _assert_preflight_exchange_evidence(
        product,
        matched,
        label="testnet rehearsal preflight",
        require_testnet=True,
    )
    reported_product = _assert_preflight_product_identity(
        product,
        matched.get("product"),
        label="testnet rehearsal preflight",
    )
    position_mode_evidence = _assert_preflight_position_mode_evidence(
        product,
        matched,
        label="testnet rehearsal preflight",
    )
    open_order_evidence = _assert_preflight_open_order_evidence(
        product,
        matched,
        label="testnet rehearsal preflight",
    )
    position_inventory_evidence = _assert_preflight_position_inventory_evidence(
        product,
        matched,
        label="testnet rehearsal preflight",
    )
    report_artifact = reported_product.get("strategies_path")
    if report_artifact:
        expected = product.strategies_path.resolve()
        actual = Path(report_artifact).resolve()
        if actual != expected:
            raise RuntimeError(
                f"{product.name}: testnet rehearsal preflight artifact mismatch: {actual} != {expected}."
            )
    reported_fingerprints = matched.get("artifact_fingerprints")
    if not isinstance(reported_fingerprints, list) or not reported_fingerprints:
        raise RuntimeError(
            f"{product.name}: testnet rehearsal preflight has no artifact_fingerprints."
        )
    artifact_snapshot = artifact if artifact is not None else load_artifact(product.strategies_path)
    current_artifact_digest = artifact_digest(artifact_snapshot)
    reported_artifact_digest = matched.get("artifact_digest")
    if not isinstance(reported_artifact_digest, str) or not reported_artifact_digest:
        raise RuntimeError(f"{product.name}: testnet rehearsal preflight has no artifact_digest.")
    if current_artifact_digest != reported_artifact_digest:
        raise RuntimeError(
            f"{product.name}: testnet rehearsal preflight artifact digest does not match current artifact."
        )
    reported_engine_digest = matched.get("execution_engine_digest")
    current_engine_digest = execution_engine_digest()
    if not isinstance(reported_engine_digest, str) or not reported_engine_digest:
        raise RuntimeError(
            f"{product.name}: testnet rehearsal preflight has no execution_engine_digest."
        )
    if reported_engine_digest != current_engine_digest:
        raise RuntimeError(
            f"{product.name}: testnet rehearsal execution engine digest does not match current code."
        )
    current_fingerprints = [
        strategy_fingerprint(strategy) for strategy in artifact_snapshot.get("strategies", [])
    ]
    if current_fingerprints != list(reported_fingerprints):
        raise RuntimeError(
            f"{product.name}: testnet rehearsal strategy fingerprints do not match current artifact."
        )
    return {
        "required": True,
        "ok": True,
        "report": str(product.testnet_rehearsal_report),
        "age_seconds": status.get("age_seconds"),
        "generated_at": status.get("generated_at"),
        "notional_usd": status.get("notional_usd"),
        "final_position_flat": status.get("final_position_flat"),
        "artifact_fingerprints": current_fingerprints,
        "execution_engine_digest": current_engine_digest,
        "required_preflight_checks": passed_checks,
        "preflight_exchange_environment": exchange_evidence,
        "preflight_position_mode": position_mode_evidence,
        "preflight_open_order_inventory": open_order_evidence,
        "preflight_position_inventory": position_inventory_evidence,
    }


def run_data_update() -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "src.update_candles"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def _clear_local_open_positions(product: ProductConfig) -> dict[str, Any]:
    if product.state_file.is_symlink():
        return {
            "path": str(product.state_file),
            "recovered": False,
            "cleared": False,
            "unsafe": True,
            "error": f"{product.name}: state file must not be a symlink: {product.state_file}",
        }
    recovered = False
    error = None
    if not product.state_file.exists():
        state: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(product.state_file.read_text(encoding="utf-8"))
            state = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                recovered = True
                error = "state payload was not an object"
        except (OSError, json.JSONDecodeError) as exc:
            state = {}
            recovered = True
            error = f"{type(exc).__name__}: {exc}"
    if state.get("exit_accounting_intent") is not None:
        return {
            "path": str(product.state_file),
            "recovered": recovered,
            "cleared": False,
            "unsafe": True,
            "error": (
                f"{product.name}: unresolved exit_accounting_intent must be committed "
                "before emergency flatten may rewrite local position state"
            ),
        }
    state["open_positions"] = {}
    cleared_pending_order = state.pop("pending_order", None)
    state["last_flatten"] = {
        "at": utc_now(),
        "reason": "autopilot_control",
        "recovered_state": recovered,
        "cleared_pending_order": cleared_pending_order is not None,
    }
    if isinstance(cleared_pending_order, dict):
        state["last_flatten"]["pending_client_id"] = cleared_pending_order.get("client_id")
    if error:
        state["last_flatten"]["state_error"] = error
    write_json_atomic(product.state_file, state)
    return {
        "path": str(product.state_file),
        "recovered": recovered,
        "cleared": True,
        "error": error,
    }


def _local_open_position_count(product: ProductConfig) -> int | None:
    if product.state_file.is_symlink():
        return None
    if not product.state_file.exists():
        return None
    try:
        state = json.loads(product.state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    positions = state.get("open_positions")
    if not isinstance(positions, dict):
        return None
    return len(positions)


def _active_income_portfolio_status(config: AutopilotConfig) -> dict[str, Any]:
    products: dict[str, Any] = {}
    total = 0
    ok = True
    for product in config.products:
        if not product.enabled or product.objective != "active_income":
            continue
        item: dict[str, Any] = {
            "state_file": str(product.state_file),
            "open_positions": 0,
            "ok": True,
        }
        if product.state_file.is_symlink():
            item.update(ok=False, reason="state_file_symlink")
        elif product.state_file.exists():
            count = _local_open_position_count(product)
            if count is None:
                item.update(ok=False, reason="state_file_unreadable")
            else:
                item["open_positions"] = count
                total += count
        products[product.name] = item
        ok = ok and bool(item["ok"])
    return {
        "ok": ok,
        "open_positions": total,
        "max_open_positions": config.active_income_max_open_positions,
        "entry_capacity_available": ok and total < config.active_income_max_open_positions,
        "products": products,
    }


def _local_state_requires_management(product: ProductConfig) -> bool:
    """Return true when pause must still run risk-reducing supervision."""
    if product.state_file.is_symlink():
        return True
    if not product.state_file.exists():
        return False
    try:
        payload = json.loads(product.state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(payload, dict):
        return True
    positions = payload.get("open_positions", {})
    if not isinstance(positions, dict):
        return True
    return (
        bool(positions)
        or payload.get("pending_order") is not None
        or payload.get("pending_entry_recovery") is not None
        or payload.get("risk_recovery_incident") is not None
        or payload.get("flatten_intent") is not None
        or payload.get("exit_accounting_intent") is not None
    )


def _frozen_management_artifact(product: ProductConfig) -> dict[str, Any]:
    """Rebuild a non-entry artifact from durable local management state.

    A deleted or malformed active artifact must block new entries, but it must
    not disable exits for exposure that already exists. New positions persist
    the exact executable strategy and its digest, so this fallback can recover
    only that already-approved behaviour and cannot introduce a flat strategy.
    """
    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        raise RuntimeError(
            f"{product.name}: cannot recover frozen strategy state: "
            f"{state_error or 'state reader returned no payload'}"
        )
    positions = state.get("open_positions")
    if not isinstance(positions, dict):
        raise RuntimeError(f"{product.name}: state open_positions must be an object.")

    strategies: list[dict[str, Any]] = []
    for strategy_id in sorted(positions):
        position = positions[strategy_id]
        if not isinstance(position, dict):
            raise RuntimeError(f"{product.name}: open position {strategy_id!r} must be an object.")
        snapshot = position.get("strategy_snapshot")
        fingerprint = position.get("strategy_fingerprint")
        if not isinstance(snapshot, dict) or not isinstance(fingerprint, str):
            raise RuntimeError(
                f"{product.name}: open position {strategy_id!r} has no frozen strategy snapshot."
            )
        try:
            canonical = json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{product.name}: open position {strategy_id!r} strategy snapshot is not JSON-safe."
            ) from exc
        actual_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint != actual_fingerprint or snapshot.get("id") != strategy_id:
            raise RuntimeError(
                f"{product.name}: open position {strategy_id!r} strategy snapshot failed integrity checks."
            )
        strategies.append(json.loads(canonical))

    source = "frozen_open_position_state"
    if not strategies:
        pending = state.get("pending_order")
        recovery = state.get("pending_entry_recovery")
        incident = state.get("risk_recovery_incident")
        marker = pending if isinstance(pending, dict) else None
        if marker is None and isinstance(recovery, dict):
            marker = recovery
        if marker is None and isinstance(incident, dict):
            marker = incident
        strategy_id = marker.get("strategy_id") if isinstance(marker, dict) else None
        symbol = marker.get("symbol") if isinstance(marker, dict) else None
        if not isinstance(strategy_id, str) or not strategy_id:
            raise RuntimeError(
                f"{product.name}: no frozen strategy or validated order-recovery strategy id is available."
            )
        if symbol not in {None, product.symbol}:
            raise RuntimeError(
                f"{product.name}: order-recovery symbol {symbol!r} does not match {product.symbol!r}."
            )
        pending_side = str(marker.get("side") or "").lower()
        direction = "short" if pending_side == "sell" else "long"
        if product.objective == "btc_accumulation":
            direction = "short"
        strategies.append(
            {
                "id": strategy_id,
                "base_timeframe": "5m",
                "direction": direction,
                "horizon_bars": 1,
                "take_profit": 0.01,
                "stop_loss": 0.01,
                "conditions": [
                    {
                        "feature": "tf_5m_close",
                        "kind": "value_ge",
                        "threshold": 0.0,
                        "description": "management-only placeholder; entries disabled",
                    }
                ],
                "risk": {
                    "risk_per_trade": 0.001,
                    "daily_stop_loss": -0.001,
                    "max_consecutive_losses": 1,
                    "cooldown_bars": 1,
                    "max_position_fraction": 0.001,
                    "max_trades_per_day": 1,
                },
                "fees": {"fee_bps": 0.0, "slippage_bps": 0.0},
                "pnl_unit": "btc" if product.objective == "btc_accumulation" else "usdt",
            }
        )
        source = "durable_order_recovery_state"

    return {
        "version": 1,
        "source": source,
        "market": product.market,
        "symbol": product.symbol,
        "pnl_unit": "btc" if product.objective == "btc_accumulation" else "usdt",
        "paper_trade_allowed": False,
        "live_allowed": source == "frozen_open_position_state",
        "promotion_eligible": source == "frozen_open_position_state",
        "strategies": strategies,
    }


def _read_local_state_for_flatten(
    product: ProductConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    if product.state_file.is_symlink():
        return None, f"{product.name}: state file must not be a symlink: {product.state_file}"
    if not product.state_file.exists():
        return {}, None
    try:
        loaded = json.loads(product.state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(loaded, dict):
        return None, f"state payload was not an object: {type(loaded).__name__}"
    return loaded, None


class _FrozenQuoteBalanceBroker:
    """Delegate broker reads except for the durable accounting balance.

    Exit accounting must consume the quote-balance evidence captured at the
    broker-flat boundary, not a later mutable value read after stop cleanup or
    a process restart.
    """

    def __init__(self, broker: Any, quote_balance: float):
        self._broker = broker
        self._quote_balance = float(quote_balance)

    def get_balance(self) -> float:
        return self._quote_balance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)


def _flatten_state_digest(value: Any, *, label: str) -> str:
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not canonical JSON: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten_accounting_bot(
    product: ProductConfig,
    broker: Any,
    *,
    quote_balance: float | None = None,
) -> PaperTradingBot:
    accounting_broker = (
        _FrozenQuoteBalanceBroker(broker, quote_balance) if quote_balance is not None else broker
    )
    return PaperTradingBot(
        strategies_path=product.strategies_path,
        state_file=product.state_file,
        trade_log=product.trade_log,
        starting_equity=product.starting_equity,
        regime_guard=product.regime_guard,
        regime_mayer_top=product.regime_mayer_top,
        broker=accounting_broker,
        symbol=product.symbol,
        market=product.market,
        objective=product.objective,
        base_asset=product.base_asset,
        live_gate_approved=True,
        allow_entries=False,
        artifact_payload=_frozen_management_artifact(product),
    )


def _flatten_strategy_and_position(
    bot: PaperTradingBot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = bot.state.get("open_positions")
    if not isinstance(positions, dict) or len(positions) != 1:
        count = len(positions) if isinstance(positions, dict) else "invalid"
        raise RuntimeError(
            "Emergency flatten accounting requires exactly one durable open position; "
            f"found {count}."
        )
    strategy_id, position = next(iter(positions.items()))
    if not isinstance(position, dict):
        raise RuntimeError("Emergency flatten position must be a JSON object.")
    strategy = next(
        (candidate for candidate in bot.strategies if candidate.get("id") == strategy_id),
        None,
    )
    if not isinstance(strategy, dict):
        raise RuntimeError(f"Emergency flatten strategy snapshot {strategy_id!r} is unavailable.")
    return strategy, position


def _resume_flatten_exit_accounting(
    product: ProductConfig,
    broker: Any,
) -> dict[str, Any] | None:
    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None or state.get("exit_accounting_intent") is None:
        return None
    bot = _flatten_accounting_bot(product, broker)
    resumed = bot._resume_exit_accounting_intent()
    return {
        "resumed": resumed,
        "state_file": str(product.state_file),
        "trade_log": str(product.trade_log),
        "equity": bot.state.get("equity"),
        "open_positions": len(bot.state.get("open_positions", {})),
    }


def _commit_flatten_exit_accounting(
    product: ProductConfig,
    broker: Any,
    *,
    intent: dict[str, Any],
    fill: Fill,
    quote_balance_after: float,
    spot_base_reconciliation: dict[str, float] | None = None,
    native_stop_cleanup: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bot = _flatten_accounting_bot(
        product,
        broker,
        quote_balance=quote_balance_after,
    )
    strategy, position = _flatten_strategy_and_position(bot)
    if strategy["id"] != intent.get("strategy_id"):
        raise RuntimeError("Flatten accounting strategy changed underneath its durable intent.")
    if bot.state.get("flatten_intent") != intent:
        raise RuntimeError("Flatten accounting intent changed before the keyed commit.")
    if _flatten_state_digest(position, label="Emergency flatten position") != intent.get(
        "position_digest"
    ):
        raise RuntimeError("Flatten accounting position changed underneath its durable intent.")

    exit_event_id = bot._exit_event_id(strategy["id"], position)
    bot.state.pop("flatten_intent")
    bot.state["last_flatten"] = {
        "at": utc_now(),
        "reason": "autopilot_control",
        "flatten_client_id": intent["client_id"],
        "exit_event_id": exit_event_id,
        "broker_account_fingerprint": intent["broker_account_fingerprint"],
        "submission_kind": intent["submission_kind"],
        "fill": intent["fill"],
        "position_before": intent["position_before"],
        "position_after": intent["position_after"],
        "quote_balance_before": intent["quote_balance_before"],
        "quote_balance_after": intent["quote_balance_after"],
        "realized_account_delta": intent["realized_account_delta"],
        "native_stop_cleanup": native_stop_cleanup or [],
    }
    bot._complete_position_exit(
        strategy,
        position,
        exit_time=utc_now(),
        exit_price=float(fill.price),
        exit_reason="emergency_flatten",
        broker_exit_fill=fill,
        clear_pending=False,
        spot_base_reconciliation=spot_base_reconciliation,
    )
    return {
        "state_file": str(product.state_file),
        "trade_log": str(product.trade_log),
        "exit_event_id": exit_event_id,
        "equity": bot.state.get("equity"),
        "daily_pnl": bot.state.get("daily_pnl"),
        "consecutive_losses": bot.state.get("consecutive_losses"),
        "cooldown_until_ts": bot.state.get("cooldown_until_ts"),
        "open_positions": len(bot.state.get("open_positions", {})),
        "pending_order_retained": bot.state.get("pending_order") is not None,
    }


def _futures_flatten_client_id(
    *,
    strategy_id: str,
    symbol: str,
    side: OrderSide,
    qty: float,
    position_digest: str,
    broker_account_fingerprint: str,
) -> str:
    digest = _flatten_state_digest(
        {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side.value,
            "qty": float(qty),
            "position_digest": position_digest,
            "broker_account_fingerprint": broker_account_fingerprint,
        },
        label="Futures flatten client identity",
    )
    return f"tb-ff-{digest[:28]}"


def _strict_flatten_number(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"flatten_intent.{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"flatten_intent.{field} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"flatten_intent.{field} must be finite")
    if positive and number <= 0:
        raise RuntimeError(f"flatten_intent.{field} must be positive")
    if non_negative and number < 0:
        raise RuntimeError(f"flatten_intent.{field} must be non-negative")
    return number


def _validated_futures_flatten_intent(
    product: ProductConfig,
    raw: Any,
    *,
    position: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("flatten_intent must be an object")
    missing = sorted(FUTURES_FLATTEN_INTENT_KEYS - set(raw))
    unexpected = sorted(set(raw) - FUTURES_FLATTEN_INTENT_KEYS)
    if missing:
        raise RuntimeError(f"flatten_intent is missing required key(s): {', '.join(missing)}")
    if unexpected:
        raise RuntimeError(f"flatten_intent has unexpected key(s): {', '.join(unexpected)}")
    if raw.get("version") != 1:
        raise RuntimeError("flatten_intent.version must be 1")
    phase = raw.get("phase")
    if phase not in {"prepared", "broker_flat_proven"}:
        raise RuntimeError("flatten_intent.phase is invalid")
    strategy_id = raw.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        raise RuntimeError("flatten_intent.strategy_id is invalid")
    if raw.get("symbol") != product.symbol:
        raise RuntimeError("flatten_intent.symbol does not match the product")
    if raw.get("order_type") != OrderType.MARKET.value or raw.get("reduce_only") is not True:
        raise RuntimeError("flatten_intent is not a reduce-only market close")
    submission_kind = raw.get("submission_kind")
    if submission_kind not in {"reduce_only_market", "native_stop_triggered"}:
        raise RuntimeError("flatten_intent.submission_kind is invalid")
    fingerprint = raw.get("broker_account_fingerprint")
    if not _valid_account_fingerprint(fingerprint):
        raise RuntimeError("flatten_intent.broker_account_fingerprint is invalid")
    client_id = raw.get("client_id")
    if (
        not isinstance(client_id, str)
        or not client_id
        or len(client_id) > 36
        or any(char not in CLIENT_ORDER_ID_SAFE_CHARS for char in client_id)
    ):
        raise RuntimeError("flatten_intent.client_id is unsafe")
    qty = _strict_flatten_number(raw.get("qty"), field="qty", positive=True)
    _strict_flatten_number(raw.get("created_ts"), field="created_ts", positive=True)
    position_digest = raw.get("position_digest")
    expected_digest = _flatten_state_digest(position, label="Emergency flatten position")
    if position_digest != expected_digest:
        raise RuntimeError("flatten_intent.position_digest does not match local position")
    position_before = raw.get("position_before")
    if not isinstance(position_before, dict) or set(position_before) != {
        "symbol",
        "qty",
        "avg_price",
    }:
        raise RuntimeError("flatten_intent.position_before is invalid")
    if position_before.get("symbol") != product.symbol:
        raise RuntimeError("flatten_intent.position_before symbol is invalid")
    before_qty = _strict_flatten_number(position_before.get("qty"), field="position_before.qty")
    if before_qty == 0:
        raise RuntimeError("flatten_intent.position_before.qty must be non-zero")
    _strict_flatten_number(
        position_before.get("avg_price"),
        field="position_before.avg_price",
        positive=True,
    )
    expected_side = OrderSide.SELL if before_qty > 0 else OrderSide.BUY
    if raw.get("side") != expected_side.value:
        raise RuntimeError("flatten_intent.side does not reduce position_before")
    qty_tolerance = max(qty * 1e-9, 1e-12)
    if abs(abs(before_qty) - qty) > qty_tolerance:
        raise RuntimeError("flatten_intent.qty does not match position_before")
    quote_before = _strict_flatten_number(
        raw.get("quote_balance_before"),
        field="quote_balance_before",
        non_negative=True,
    )

    fill = raw.get("fill")
    position_after = raw.get("position_after")
    quote_after = raw.get("quote_balance_after")
    account_delta = raw.get("realized_account_delta")
    proven_ts = raw.get("proven_ts")
    if phase == "prepared":
        if any(
            value is not None
            for value in (fill, position_after, quote_after, account_delta, proven_ts)
        ):
            raise RuntimeError("prepared flatten_intent contains unproven broker evidence")
    else:
        if not isinstance(fill, dict) or set(fill) != {
            "symbol",
            "side",
            "qty",
            "price",
            "fee",
            "timestamp",
        }:
            raise RuntimeError("flatten_intent.fill is invalid")
        if fill.get("symbol") != product.symbol or fill.get("side") != expected_side.value:
            raise RuntimeError("flatten_intent.fill identity is invalid")
        fill_qty = _strict_flatten_number(fill.get("qty"), field="fill.qty", positive=True)
        _strict_flatten_number(fill.get("price"), field="fill.price", positive=True)
        _strict_flatten_number(fill.get("fee"), field="fill.fee", non_negative=True)
        _strict_flatten_number(fill.get("timestamp"), field="fill.timestamp", positive=True)
        if abs(fill_qty - qty) > qty_tolerance:
            raise RuntimeError("flatten_intent.fill qty does not match its order")
        if not isinstance(position_after, dict) or set(position_after) != {
            "symbol",
            "qty",
            "avg_price",
        }:
            raise RuntimeError("flatten_intent.position_after is invalid")
        if position_after.get("symbol") != product.symbol:
            raise RuntimeError("flatten_intent.position_after symbol is invalid")
        after_qty = _strict_flatten_number(position_after.get("qty"), field="position_after.qty")
        after_avg = _strict_flatten_number(
            position_after.get("avg_price"),
            field="position_after.avg_price",
            non_negative=True,
        )
        if abs(after_qty) >= 1e-12 or after_avg != 0:
            raise RuntimeError("flatten_intent does not prove a flat broker position")
        quote_after_number = _strict_flatten_number(
            quote_after,
            field="quote_balance_after",
            non_negative=True,
        )
        delta_number = _strict_flatten_number(
            account_delta,
            field="realized_account_delta",
        )
        delta_tolerance = max(abs(quote_before) * 1e-12, 1e-9)
        if abs((quote_after_number - quote_before) - delta_number) > delta_tolerance:
            raise RuntimeError("flatten_intent realized account delta is inconsistent")
        _strict_flatten_number(proven_ts, field="proven_ts", positive=True)

    if submission_kind == "reduce_only_market":
        expected_client_id = _futures_flatten_client_id(
            strategy_id=strategy_id,
            symbol=product.symbol,
            side=expected_side,
            qty=qty,
            position_digest=position_digest,
            broker_account_fingerprint=fingerprint,
        )
        if client_id != expected_client_id:
            raise RuntimeError("flatten_intent.client_id does not match its deterministic intent")
    return dict(raw)


def _persist_flatten_intent(
    product: ProductConfig,
    state: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    if state.get("flatten_intent") is not None:
        raise RuntimeError("cannot replace an existing flatten_intent")
    updated = dict(state)
    updated["flatten_intent"] = intent
    write_json_atomic(product.state_file, updated)
    return updated


def _persist_proven_flatten_intent(
    product: ProductConfig,
    expected_intent: dict[str, Any],
    proven_intent: dict[str, Any],
) -> dict[str, Any]:
    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        raise RuntimeError(f"could not reread flatten state: {state_error}")
    if state.get("flatten_intent") != expected_intent:
        raise RuntimeError("durable flatten_intent changed before broker evidence commit")
    updated = dict(state)
    updated["flatten_intent"] = proven_intent
    write_json_atomic(product.state_file, updated)
    return updated


def _local_state_clear_failed(local_state: dict[str, Any]) -> bool:
    return bool(local_state.get("unsafe"))


def _spot_step_aside_flatten_position(
    product: ProductConfig, state: dict[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    raw_positions = state.get("open_positions", {})
    if raw_positions in (None, {}):
        return None
    if not isinstance(raw_positions, dict):
        raise RuntimeError(
            f"{product.name}: state open_positions must be an object before spot flatten."
        )
    positions = list(raw_positions.items())
    if len(positions) != 1:
        raise RuntimeError(
            f"{product.name}: spot flatten requires exactly one local step-aside position, found {len(positions)}."
        )
    strategy_id, position = positions[0]
    if not isinstance(position, dict):
        raise RuntimeError(
            f"{product.name}: spot flatten position {strategy_id!r} must be an object."
        )
    return str(strategy_id), position


def _validate_spot_step_aside_flatten_position(
    product: ProductConfig,
    strategy_id: str,
    position: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    direction = str(position.get("direction") or "").lower()
    if direction != "short":
        reasons.append("invalid_direction")
    broker_symbol = position.get("broker_symbol")
    if not isinstance(broker_symbol, str) or broker_symbol != product.symbol:
        reasons.append("invalid_broker_symbol")
    broker_side = str(position.get("broker_side") or "").lower()
    if broker_side != "sell":
        reasons.append("invalid_broker_side")
    broker_qty = _positive_evidence_float(position.get("broker_qty"))
    if broker_qty is None:
        reasons.append("invalid_broker_qty")
    broker_entry_price = _positive_evidence_float(position.get("broker_entry_price"))
    if broker_entry_price is None:
        reasons.append("invalid_broker_entry_price")
    quote_value = _positive_evidence_float(position.get("broker_entry_quote_value"))
    if quote_value is None:
        reasons.append("invalid_spot_step_aside_quote_value")
    if position.get("broker_exit_sizing") != "quote_reinvest":
        reasons.append("invalid_spot_step_aside_exit_sizing")
    broker_account_fingerprint = position.get("broker_account_fingerprint")
    if not _valid_account_fingerprint(broker_account_fingerprint):
        reasons.append("invalid_broker_account_fingerprint")
    detail = {
        "strategy_id": strategy_id,
        "direction": position.get("direction"),
        "broker_symbol": broker_symbol,
        "broker_side": position.get("broker_side"),
        "broker_qty": position.get("broker_qty"),
        "broker_entry_price": position.get("broker_entry_price"),
        "broker_entry_quote_value": position.get("broker_entry_quote_value"),
        "broker_account_fingerprint": broker_account_fingerprint,
        "broker_exit_sizing": position.get("broker_exit_sizing"),
        "reasons": reasons,
    }
    if reasons:
        raise RuntimeError(
            f"{product.name}: spot flatten state is invalid for {strategy_id}: "
            + ", ".join(reasons)
        )
    detail["quote_value"] = quote_value
    return detail


def _spot_flatten_client_id(
    *,
    strategy_id: str,
    symbol: str,
    qty: float,
    quote_budget: float,
    position_before_qty: float,
    broker_account_fingerprint: str,
) -> str:
    payload = {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "side": OrderSide.BUY.value,
        "qty": format(float(qty), ".17g"),
        "quote_budget": format(float(quote_budget), ".17g"),
        "position_before_qty": format(float(position_before_qty), ".17g"),
        "broker_account_fingerprint": broker_account_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    client_id = f"tb-sf-{digest[:28]}"
    if (
        len(client_id) > 36
        or not client_id
        or any(char not in CLIENT_ORDER_ID_SAFE_CHARS for char in client_id)
    ):  # pragma: no cover - generated invariant
        raise RuntimeError(f"Generated unsafe spot flatten client order id: {client_id!r}")
    return client_id


def _strict_spot_flatten_intent_number(
    value: Any,
    *,
    field: str,
    positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeError(f"flatten_intent.{field} must be a finite {qualifier} JSON number")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeError(f"flatten_intent.{field} must be a finite {qualifier} JSON number")
    return number


def _validated_spot_flatten_intent(
    product: ProductConfig,
    raw_intent: Any,
    *,
    strategy_id: str,
    position_detail: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_intent, dict):
        raise RuntimeError("flatten_intent must be an object")
    missing = sorted(SPOT_FLATTEN_INTENT_KEYS - set(raw_intent))
    unexpected = sorted(set(raw_intent) - SPOT_FLATTEN_INTENT_KEYS)
    if missing:
        raise RuntimeError(f"flatten_intent is missing required key(s): {', '.join(missing)}")
    if unexpected:
        raise RuntimeError(f"flatten_intent has unexpected key(s): {', '.join(unexpected)}")
    if isinstance(raw_intent.get("version"), bool) or raw_intent.get("version") != 1:
        raise RuntimeError("flatten_intent.version must be 1")
    if raw_intent.get("strategy_id") != strategy_id:
        raise RuntimeError("flatten_intent.strategy_id does not match the tracked position")
    if raw_intent.get("symbol") != product.symbol:
        raise RuntimeError("flatten_intent.symbol does not match the product")
    if raw_intent.get("side") != OrderSide.BUY.value:
        raise RuntimeError("flatten_intent.side must be buy")
    if raw_intent.get("order_type") != OrderType.MARKET.value:
        raise RuntimeError("flatten_intent.order_type must be market")
    broker_account_fingerprint = raw_intent.get("broker_account_fingerprint")
    if not _valid_account_fingerprint(broker_account_fingerprint):
        raise RuntimeError("flatten_intent.broker_account_fingerprint is invalid")
    if broker_account_fingerprint != position_detail.get("broker_account_fingerprint"):
        raise RuntimeError(
            "flatten_intent.broker_account_fingerprint does not match the tracked position"
        )

    qty = _strict_spot_flatten_intent_number(
        raw_intent.get("qty"),
        field="qty",
        positive=True,
    )
    quote_budget = _strict_spot_flatten_intent_number(
        raw_intent.get("quote_budget"),
        field="quote_budget",
        positive=True,
    )
    created_ts = _strict_spot_flatten_intent_number(
        raw_intent.get("created_ts"),
        field="created_ts",
        positive=True,
    )
    expected_quote_budget = float(position_detail["quote_value"])
    quote_tolerance = max(abs(expected_quote_budget) * 1e-9, 1e-9)
    if abs(quote_budget - expected_quote_budget) > quote_tolerance:
        raise RuntimeError("flatten_intent.quote_budget does not match the tracked position")

    position_before = raw_intent.get("position_before")
    if not isinstance(position_before, dict):
        raise RuntimeError("flatten_intent.position_before must be an object")
    missing_position = sorted(SPOT_FLATTEN_POSITION_EVIDENCE_KEYS - set(position_before))
    unexpected_position = sorted(set(position_before) - SPOT_FLATTEN_POSITION_EVIDENCE_KEYS)
    if missing_position:
        raise RuntimeError(
            "flatten_intent.position_before is missing required key(s): "
            + ", ".join(missing_position)
        )
    if unexpected_position:
        raise RuntimeError(
            "flatten_intent.position_before has unexpected key(s): "
            + ", ".join(unexpected_position)
        )
    if position_before.get("symbol") != product.symbol:
        raise RuntimeError("flatten_intent.position_before.symbol does not match the product")
    before_qty = _strict_spot_flatten_intent_number(
        position_before.get("qty"),
        field="position_before.qty",
        positive=False,
    )
    before_avg_price = _strict_spot_flatten_intent_number(
        position_before.get("avg_price"),
        field="position_before.avg_price",
        positive=False,
    )

    client_id = raw_intent.get("client_id")
    if (
        not isinstance(client_id, str)
        or not client_id
        or len(client_id) > 36
        or any(char not in CLIENT_ORDER_ID_SAFE_CHARS for char in client_id)
    ):
        raise RuntimeError("flatten_intent.client_id is unsafe")
    expected_client_id = _spot_flatten_client_id(
        strategy_id=strategy_id,
        symbol=product.symbol,
        qty=qty,
        quote_budget=quote_budget,
        position_before_qty=before_qty,
        broker_account_fingerprint=broker_account_fingerprint,
    )
    if client_id != expected_client_id:
        raise RuntimeError("flatten_intent.client_id does not match its deterministic intent")

    return {
        "version": 1,
        "strategy_id": strategy_id,
        "symbol": product.symbol,
        "side": OrderSide.BUY.value,
        "order_type": OrderType.MARKET.value,
        "client_id": client_id,
        "broker_account_fingerprint": broker_account_fingerprint,
        "qty": qty,
        "quote_budget": quote_budget,
        "position_before": {
            "symbol": product.symbol,
            "qty": before_qty,
            "avg_price": before_avg_price,
        },
        "created_ts": created_ts,
    }


def _spot_flatten_balance_evidence(
    product: ProductConfig,
    intent: dict[str, Any],
    after: Position,
    *,
    expected_filled_qty: float,
) -> dict[str, Any]:
    before = intent["position_before"]
    before_qty = float(before["qty"])
    after_qty = _evidence_float(after.qty)
    expected_qty = float(expected_filled_qty)
    tolerance = max(
        expected_qty * SPOT_FLATTEN_BALANCE_FEE_TOLERANCE_FRACTION,
        1e-9,
    )
    rounding_tolerance = max(expected_qty * 1e-9, 1e-12)
    symbol_matches = after.symbol == product.symbol
    actual_increase = after_qty - before_qty if after_qty is not None else None
    proven = (
        symbol_matches
        and after_qty is not None
        and after_qty >= 0
        and actual_increase is not None
        and expected_qty - tolerance <= actual_increase <= expected_qty + rounding_tolerance
    )
    return {
        "symbol": after.symbol,
        "symbol_matches": symbol_matches,
        "before_qty": before_qty,
        "after_qty": after_qty,
        "actual_increase": actual_increase,
        "expected_increase": expected_qty,
        "fee_tolerance": tolerance,
        "rounding_tolerance": rounding_tolerance,
        "proven": proven,
    }


def _persist_spot_flatten_intent(
    product: ProductConfig,
    state: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    if "flatten_intent" in state:
        raise RuntimeError("cannot replace an existing flatten_intent")
    updated = dict(state)
    updated["flatten_intent"] = intent
    write_json_atomic(product.state_file, updated)
    return updated


def _commit_spot_flatten_state(
    product: ProductConfig,
    broker: Any,
    intent: dict[str, Any],
    *,
    balance_evidence: dict[str, Any],
    fill: Fill,
) -> dict[str, Any]:
    if balance_evidence.get("proven") is not True:
        raise RuntimeError("spot buyback balance postcondition is not proven")
    quote_balance_after = _strict_flatten_number(
        broker.get_balance(),
        field="spot_quote_balance_after",
        non_negative=True,
    )
    bot = _flatten_accounting_bot(
        product,
        broker,
        quote_balance=quote_balance_after,
    )
    strategy, position = _flatten_strategy_and_position(bot)
    if strategy["id"] != intent.get("strategy_id"):
        raise RuntimeError("Spot flatten accounting strategy changed underneath its intent")
    if bot.state.get("flatten_intent") != intent:
        raise RuntimeError("durable flatten_intent changed before local accounting commit")
    entry_base_balance = _positive_evidence_float(position.get("broker_entry_base_qty_before"))
    after_base_balance = _evidence_float(balance_evidence.get("after_qty"))
    if entry_base_balance is None or after_base_balance is None or after_base_balance < 0:
        raise RuntimeError("spot flatten is missing durable BTC account-balance evidence")
    account_return = (after_base_balance - entry_base_balance) / entry_base_balance
    if not math.isfinite(account_return) or account_return <= -1:
        raise RuntimeError("spot flatten BTC account return is invalid")
    spot_reconciliation = {
        "entry_base_balance_before": entry_base_balance,
        "entry_base_balance_after": float(position["broker_entry_base_qty_after"]),
        "exit_base_balance_before": float(balance_evidence["before_qty"]),
        "exit_base_balance_after": after_base_balance,
        "observed_buy_qty": float(balance_evidence["actual_increase"]),
        "account_return": account_return,
    }
    exit_event_id = bot._exit_event_id(strategy["id"], position)
    bot.state.pop("flatten_intent")
    bot.state["last_flatten"] = {
        "at": utc_now(),
        "reason": "autopilot_control",
        "flatten_client_id": intent["client_id"],
        "exit_event_id": exit_event_id,
        "broker_account_fingerprint": intent["broker_account_fingerprint"],
        "submission_kind": "spot_quote_reinvestment",
        "balance_evidence": balance_evidence,
        "quote_balance_after": quote_balance_after,
        "fill": {
            "symbol": fill.symbol,
            "side": _fill_side_value(fill),
            "qty": float(fill.qty),
            "price": float(fill.price),
            "fee": float(fill.fee),
            "timestamp": float(fill.timestamp),
        },
    }
    bot._complete_position_exit(
        strategy,
        position,
        exit_time=utc_now(),
        exit_price=float(fill.price),
        exit_reason="emergency_flatten",
        broker_exit_fill=fill,
        clear_pending=False,
        spot_base_reconciliation=spot_reconciliation,
    )
    return {
        "path": str(product.state_file),
        "trade_log": str(product.trade_log),
        "exit_event_id": exit_event_id,
        "equity": bot.state.get("equity"),
        "daily_pnl": bot.state.get("daily_pnl"),
        "cooldown_until_ts": bot.state.get("cooldown_until_ts"),
        "open_positions": len(bot.state.get("open_positions", {})),
        "pending_order_retained": bot.state.get("pending_order") is not None,
    }


def _fill_side_value(fill: Fill) -> str:
    return fill.side.value if isinstance(fill.side, OrderSide) else str(fill.side)


def _assert_spot_flatten_fill_valid(strategy_id: str, order: Order, fill: Fill) -> None:
    if fill.symbol != order.symbol:
        raise RuntimeError(
            f"Spot flatten fill mismatch for {strategy_id}: expected symbol {order.symbol}, got {fill.symbol}."
        )
    if _fill_side_value(fill) != order.side.value:
        raise RuntimeError(
            f"Spot flatten fill mismatch for {strategy_id}: expected side {order.side.value}, got {_fill_side_value(fill)}."
        )
    fill_qty = _positive_evidence_float(fill.qty)
    fill_price = _positive_evidence_float(fill.price)
    if fill_qty is None:
        raise RuntimeError(
            f"Spot flatten invalid fill for {strategy_id}: fill quantity must be positive."
        )
    if fill_price is None:
        raise RuntimeError(
            f"Spot flatten invalid fill for {strategy_id}: fill price must be positive."
        )
    try:
        fee = float(fill.fee)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Spot flatten invalid fill for {strategy_id}: fill fee must be numeric."
        ) from exc
    if not math.isfinite(fee) or fee < 0:
        raise RuntimeError(
            f"Spot flatten invalid fill for {strategy_id}: fill fee must be finite and non-negative."
        )
    tolerance = max(float(order.qty) * 1e-6, 1e-9)
    if float(fill.qty) - float(order.qty) > tolerance:
        raise RuntimeError(
            f"Spot flatten overfill for {strategy_id}: requested {order.qty:g}, filled {fill.qty:g}."
        )
    if float(fill.qty) + tolerance < float(order.qty):
        raise RuntimeError(
            f"Spot flatten partial fill for {strategy_id}: requested {order.qty:g}, filled {fill.qty:g}."
        )


def _assert_futures_flatten_fill_valid(
    product: ProductConfig, before: Position, fill: Fill
) -> None:
    if fill.symbol != product.symbol:
        raise RuntimeError(
            f"{product.name}: futures flatten fill mismatch: expected symbol {product.symbol}, got {fill.symbol}."
        )
    expected_side = OrderSide.SELL if before.qty > 0 else OrderSide.BUY
    fill_side = _fill_side_value(fill)
    if fill_side != expected_side.value:
        raise RuntimeError(
            f"{product.name}: futures flatten fill mismatch: expected side {expected_side.value}, got {fill_side}."
        )
    fill_qty = _positive_evidence_float(fill.qty)
    fill_price = _positive_evidence_float(fill.price)
    if fill_qty is None:
        raise RuntimeError(f"{product.name}: futures flatten fill quantity must be positive.")
    if fill_price is None:
        raise RuntimeError(f"{product.name}: futures flatten fill price must be positive.")
    try:
        fee = float(fill.fee)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{product.name}: futures flatten fill fee must be numeric.") from exc
    if not math.isfinite(fee) or fee < 0:
        raise RuntimeError(
            f"{product.name}: futures flatten fill fee must be finite and non-negative."
        )
    expected_qty = abs(float(before.qty))
    tolerance = max(expected_qty * 1e-6, 1e-9)
    if float(fill.qty) - expected_qty > tolerance:
        raise RuntimeError(
            f"{product.name}: futures flatten overfill: requested {expected_qty:g}, filled {fill.qty:g}."
        )
    if float(fill.qty) + tolerance < expected_qty:
        raise RuntimeError(
            f"{product.name}: futures flatten partial fill: requested {expected_qty:g}, filled {fill.qty:g}."
        )


def _local_futures_protective_stops(product: ProductConfig) -> list[dict[str, Any]]:
    """Return strictly validated native-stop identities from local state.

    Panic flattening must not erase the only identifiers for an exchange
    conditional order. Account binding is proved before this helper runs;
    missing/corrupt durable state therefore blocks an arbitrary-account close,
    and invalid stop evidence blocks local-state clearing after flat proof.
    """
    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        raise RuntimeError(
            f"{product.name}: cannot inspect native stops before flatten: "
            f"{state_error or 'state reader returned no payload'}"
        )
    raw_positions = state.get("open_positions", {})
    if raw_positions in (None, {}):
        return []
    if not isinstance(raw_positions, dict):
        raise RuntimeError(
            f"{product.name}: state open_positions must be an object before futures flatten."
        )

    records: list[dict[str, Any]] = []
    required = (
        "broker_symbol",
        "broker_qty",
        "broker_stop_order_id",
        "broker_stop_client_id",
        "broker_stop_trigger_price",
        "direction",
    )
    for strategy_id, position in raw_positions.items():
        if not isinstance(position, dict):
            raise RuntimeError(
                f"{product.name}: futures flatten position {strategy_id!r} must be an object."
            )
        missing = [key for key in required if key not in position]
        if missing:
            raise RuntimeError(
                f"{product.name}: futures flatten position {strategy_id!r} is missing native-stop "
                f"metadata: {', '.join(missing)}."
            )
        symbol = str(position["broker_symbol"]).strip()
        order_id = str(position["broker_stop_order_id"]).strip()
        client_id = str(position["broker_stop_client_id"]).strip()
        direction = str(position["direction"]).lower()
        qty = _positive_evidence_float(position["broker_qty"])
        trigger_price = _positive_evidence_float(position["broker_stop_trigger_price"])
        if symbol != product.symbol:
            raise RuntimeError(
                f"{product.name}: native-stop symbol {symbol!r} does not match {product.symbol!r}."
            )
        if not order_id or not client_id:
            raise RuntimeError(
                f"{product.name}: futures flatten position {strategy_id!r} has an empty native-stop id."
            )
        if direction not in {"long", "short"} or qty is None or trigger_price is None:
            raise RuntimeError(
                f"{product.name}: futures flatten position {strategy_id!r} has invalid native-stop evidence."
            )
        records.append(
            {
                "strategy_id": str(strategy_id),
                "symbol": symbol,
                "order_id": order_id,
                "client_id": client_id,
                "qty": qty,
                "trigger_price": trigger_price,
                "side": OrderSide.SELL if direction == "long" else OrderSide.BUY,
            }
        )
    return records


def _validated_accounted_futures_flatten(
    product: ProductConfig,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    if state.get("open_positions") != {}:
        return None
    if state.get("flatten_intent") is not None or state.get("exit_accounting_intent") is not None:
        return None
    last = state.get("last_flatten")
    if not isinstance(last, dict) or last.get("reason") != "autopilot_control":
        return None
    fingerprint = last.get("broker_account_fingerprint")
    if not _valid_account_fingerprint(fingerprint):
        return None
    event_id = last.get("exit_event_id")
    if (
        not isinstance(event_id, str)
        or len(event_id) != 64
        or any(char not in "0123456789abcdef" for char in event_id)
    ):
        return None
    client_id = last.get("flatten_client_id")
    if (
        not isinstance(client_id, str)
        or not client_id.startswith("tb-ff-")
        or len(client_id) > 36
        or any(char not in CLIENT_ORDER_ID_SAFE_CHARS for char in client_id)
    ):
        return None
    if last.get("submission_kind") != "reduce_only_market":
        return None
    fill = last.get("fill")
    if not isinstance(fill, dict) or fill.get("symbol") != product.symbol:
        return None
    if str(fill.get("side")) not in {OrderSide.BUY.value, OrderSide.SELL.value}:
        return None
    if any(
        value is None
        for value in (
            _positive_evidence_float(fill.get("qty")),
            _positive_evidence_float(fill.get("price")),
            _evidence_float(fill.get("fee")),
            _positive_evidence_float(fill.get("timestamp")),
        )
    ):
        return None
    if float(fill["fee"]) < 0:
        return None
    before = last.get("position_before")
    before_qty = _evidence_float(before.get("qty")) if isinstance(before, dict) else None
    if (
        not isinstance(before, dict)
        or before.get("symbol") != product.symbol
        or before_qty in {None, 0}
        or _positive_evidence_float(before.get("avg_price")) is None
    ):
        return None
    expected_side = OrderSide.SELL.value if float(before_qty) > 0 else OrderSide.BUY.value
    if fill.get("side") != expected_side:
        return None
    after = last.get("position_after")
    if (
        not isinstance(after, dict)
        or after.get("symbol") != product.symbol
        or _evidence_float(after.get("qty")) != 0
        or _evidence_float(after.get("avg_price")) != 0
    ):
        return None
    quote_before = _evidence_float(last.get("quote_balance_before"))
    quote_after = _evidence_float(last.get("quote_balance_after"))
    delta = _evidence_float(last.get("realized_account_delta"))
    if (
        quote_before is None
        or quote_before < 0
        or quote_after is None
        or quote_after < 0
        or delta is None
    ):
        return None
    tolerance = max(abs(quote_before) * 1e-12, 1e-9)
    if abs((quote_after - quote_before) - delta) > tolerance:
        return None
    return last


def _assert_futures_flatten_account_binding(
    product: ProductConfig,
    broker: Any,
) -> str:
    """Bind panic-flatten reads and writes to the durable production account."""

    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        raise RuntimeError(
            f"{product.name}: cannot verify account identity before futures flatten: "
            f"{state_error or 'state reader returned no payload'}"
        )
    identities: list[tuple[str, Any]] = []
    positions = state.get("open_positions", {})
    if not isinstance(positions, dict):
        raise RuntimeError(
            f"{product.name}: state open_positions must be an object before futures flatten."
        )
    for strategy_id, position in positions.items():
        if not isinstance(position, dict):
            raise RuntimeError(
                f"{product.name}: futures flatten position {strategy_id!r} must be an object."
            )
        identities.append(
            (
                f"open_positions[{strategy_id!r}]",
                position.get("broker_account_fingerprint"),
            )
        )
    for state_key in (
        "pending_order",
        "pending_entry_recovery",
        "risk_recovery_incident",
    ):
        marker = state.get(state_key)
        if marker is None:
            continue
        if not isinstance(marker, dict):
            raise RuntimeError(
                f"{product.name}: {state_key} must be an object before futures flatten."
            )
        identities.append((state_key, marker.get("broker_account_fingerprint")))
    if not identities:
        accounted = _validated_accounted_futures_flatten(product, state)
        if accounted is not None:
            identities.append(("last_flatten", accounted["broker_account_fingerprint"]))
        else:
            raise RuntimeError(
                f"{product.name}: no durable broker account identity exists; refusing to read or "
                "flatten an arbitrary live account."
            )
    current = _live_broker_account_fingerprint(product, broker)
    for label, expected in identities:
        if not _valid_account_fingerprint(expected):
            raise RuntimeError(
                f"{product.name}: {label} has no valid durable broker account fingerprint."
            )
        if expected != current:
            raise RuntimeError(
                f"{product.name}: {label} belongs to a different broker account; refusing "
                "position reads, flatten orders, and local-state mutation."
            )
    return current


def _assert_flatten_protective_order(
    product: ProductConfig,
    expected: dict[str, Any],
    protective: ProtectiveOrder,
    *,
    terminal: bool,
) -> None:
    if not isinstance(protective, ProtectiveOrder):
        raise RuntimeError(
            f"{product.name}: broker returned invalid native-stop evidence during flatten."
        )
    if protective.symbol != expected["symbol"]:
        raise RuntimeError(
            f"{product.name}: native-stop symbol changed during flatten reconciliation."
        )
    if protective.order_id != expected["order_id"] or protective.client_id != expected["client_id"]:
        raise RuntimeError(
            f"{product.name}: native-stop identity changed during flatten reconciliation."
        )
    if protective.side != expected["side"]:
        raise RuntimeError(
            f"{product.name}: native-stop side changed during flatten reconciliation."
        )
    qty_tolerance = max(float(expected["qty"]) * 1e-6, 1e-9)
    trigger_tolerance = max(float(expected["trigger_price"]) * 1e-9, 1e-12)
    protective_qty = float(protective.qty)
    protective_trigger = float(protective.trigger_price)
    if (
        not math.isfinite(protective_qty)
        or abs(protective_qty - float(expected["qty"])) > qty_tolerance
    ):
        raise RuntimeError(
            f"{product.name}: native-stop quantity changed during flatten reconciliation."
        )
    if (
        not math.isfinite(protective_trigger)
        or abs(protective_trigger - float(expected["trigger_price"])) > trigger_tolerance
    ):
        raise RuntimeError(
            f"{product.name}: native-stop trigger changed during flatten reconciliation."
        )
    terminal_statuses = {
        ProtectiveOrderStatus.TRIGGERED,
        ProtectiveOrderStatus.CANCELED,
        ProtectiveOrderStatus.EXPIRED,
        ProtectiveOrderStatus.REJECTED,
    }
    if terminal and protective.status not in terminal_statuses:
        raise RuntimeError(
            f"{product.name}: native stop {protective.order_id} remains {protective.status.value}."
        )


def _finish_futures_native_stop_cleanup(
    product: ProductConfig,
    broker: Any,
    expected_stops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not expected_stops:
        return []
    capability = getattr(broker, "supports_native_protective_stops", None)
    if not callable(capability) or not capability():
        raise RuntimeError(
            f"{product.name}: broker cannot reconcile exchange-native stops after flatten."
        )

    results: list[dict[str, Any]] = []
    for expected in expected_stops:
        protective = broker.get_protective_stop(
            symbol=expected["symbol"],
            order_id=expected["order_id"],
            client_id=expected["client_id"],
        )
        _assert_flatten_protective_order(product, expected, protective, terminal=False)
        if protective.status == ProtectiveOrderStatus.OPEN:
            try:
                protective = broker.cancel_protective_stop(
                    symbol=expected["symbol"],
                    order_id=expected["order_id"],
                    client_id=expected["client_id"],
                )
            except Exception:
                # The stop can trigger between the read and cancel.  A second
                # authenticated read is the only safe way to distinguish that
                # race from an orphaned open order.
                protective = broker.get_protective_stop(
                    symbol=expected["symbol"],
                    order_id=expected["order_id"],
                    client_id=expected["client_id"],
                )
            _assert_flatten_protective_order(product, expected, protective, terminal=True)
        else:
            _assert_flatten_protective_order(product, expected, protective, terminal=True)

        confirmed = broker.get_protective_stop(
            symbol=expected["symbol"],
            order_id=expected["order_id"],
            client_id=expected["client_id"],
        )
        _assert_flatten_protective_order(product, expected, confirmed, terminal=True)
        results.append(
            {
                "strategy_id": expected["strategy_id"],
                "order_id": confirmed.order_id,
                "client_id": confirmed.client_id,
                "status": confirmed.status.value,
            }
        )
    return results


def _finalize_futures_flatten_state(
    product: ProductConfig,
    broker: Any,
    status: dict[str, Any],
) -> dict[str, Any]:
    try:
        expected_stops = _local_futures_protective_stops(product)
        status["native_stop_cleanup"] = _finish_futures_native_stop_cleanup(
            product,
            broker,
            expected_stops,
        )
    except Exception as exc:
        status.update(
            ok=False,
            reason="native_stop_cleanup_unverified",
            native_stop_cleanup_error=f"{type(exc).__name__}: {exc}",
        )
        return status

    local_state = _clear_local_open_positions(product)
    status["local_state"] = local_state
    if _local_state_clear_failed(local_state):
        status.update(ok=False, reason="unsafe_local_state", error=local_state.get("error"))
        return status
    status["ok"] = True
    return status


def _flatten_btc_spot_step_aside_product(
    product: ProductConfig, status: dict[str, Any]
) -> dict[str, Any]:
    def unresolved(
        error: str,
        *,
        intent: Any = None,
        balance_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status.update(
            ok=False,
            reason="unresolved_flatten_intent",
            error=error,
            operator_action=(
                "Keep the product paused; reconcile the deterministic client ID, spot order/fills, "
                "and BTC/USDT balances at the exchange before editing flatten_intent."
            ),
        )
        if intent is not None:
            status["flatten_intent"] = intent
        if balance_evidence is not None:
            status["balance_evidence"] = balance_evidence
        return status

    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        status.update(
            ok=False,
            reason="invalid_local_state",
            local_state={
                "path": str(product.state_file),
                "error": state_error or "state reader returned no payload",
            },
        )
        return status
    exit_accounting_intent = state.get("exit_accounting_intent")
    if exit_accounting_intent is not None:
        status.update(
            ok=False,
            reason="unresolved_exit_accounting_intent",
            error=(
                "Exit accounting must commit its keyed trade row and final local state "
                "before emergency spot flatten can prepare another broker intent."
            ),
            exit_accounting_intent=exit_accounting_intent,
        )
        return status
    has_intent = "flatten_intent" in state
    try:
        selected = _spot_step_aside_flatten_position(product, state)
    except RuntimeError as exc:
        if has_intent:
            return unresolved(str(exc), intent=state.get("flatten_intent"))
        status.update(ok=False, reason="invalid_spot_step_aside_state", error=str(exc))
        return status
    if selected is None:
        if has_intent:
            return unresolved(
                "flatten_intent exists without exactly one tracked spot step-aside position",
                intent=state.get("flatten_intent"),
            )
        status.update(
            ok=True,
            skipped=True,
            reason="no_local_spot_step_aside_position",
            local_open_positions=0,
        )
        return status
    strategy_id, position = selected
    try:
        position_detail = _validate_spot_step_aside_flatten_position(
            product,
            strategy_id,
            position,
        )
    except RuntimeError as exc:
        if has_intent:
            return unresolved(str(exc), intent=state.get("flatten_intent"))
        status.update(
            ok=False,
            reason="invalid_spot_step_aside_state",
            error=str(exc),
            position={
                key: position.get(key)
                for key in (
                    "direction",
                    "broker_symbol",
                    "broker_side",
                    "broker_qty",
                    "broker_entry_price",
                    "broker_entry_quote_value",
                    "broker_exit_sizing",
                )
                if key in position
            },
        )
        return status

    existing_intent: dict[str, Any] | None = None
    if has_intent:
        try:
            existing_intent = _validated_spot_flatten_intent(
                product,
                state.get("flatten_intent"),
                strategy_id=strategy_id,
                position_detail=position_detail,
            )
        except RuntimeError as exc:
            return unresolved(str(exc), intent=state.get("flatten_intent"))

    status["live_environment"] = assert_live_environment(
        product,
        require_production=True,
    )
    broker = build_live_broker(product)
    status["broker"] = broker.name
    current_account_fingerprint = _live_broker_account_fingerprint(product, broker)
    expected_account_fingerprint = position_detail["broker_account_fingerprint"]
    if current_account_fingerprint != expected_account_fingerprint:
        error = (
            "live broker account fingerprint does not match the tracked spot position; "
            "refusing balance reads, buyback submission, or local-state mutation"
        )
        if existing_intent is not None:
            return unresolved(error, intent=existing_intent)
        status.update(ok=False, reason="broker_account_mismatch", error=error)
        return status

    try:
        accounting_bot = _flatten_accounting_bot(product, broker)
        accounting_strategy, _accounting_position = _flatten_strategy_and_position(accounting_bot)
        if accounting_strategy["id"] != strategy_id:
            raise RuntimeError("spot flatten strategy identity changed")
        state = accounting_bot.state
    except Exception as exc:
        status.update(
            ok=False,
            reason="flatten_accounting_precondition_failed",
            error=f"{type(exc).__name__}: {exc}",
            operator_action=(
                "Keep the product paused; repair or reconcile the durable strategy/position "
                "accounting evidence before a BTC buyback is submitted."
            ),
        )
        return status

    if existing_intent is not None:
        try:
            current = broker.get_position(product.symbol)
        except Exception as exc:
            return unresolved(
                f"could not read broker BTC balance for intent reconciliation: {type(exc).__name__}: {exc}",
                intent=existing_intent,
            )
        balance_evidence = _spot_flatten_balance_evidence(
            product,
            existing_intent,
            current,
            expected_filled_qty=float(existing_intent["qty"]),
        )
        status.update(
            flatten_intent=existing_intent,
            position_current={
                "symbol": current.symbol,
                "qty": current.qty,
                "avg_price": current.avg_price,
            },
            balance_evidence=balance_evidence,
        )
        return unresolved(
            (
                "existing flatten_intent may have changed the broker BTC balance, but its "
                "fill price/fee response was not durably committed; refusing both a duplicate "
                "buyback and silent local accounting"
            ),
            intent=existing_intent,
            balance_evidence=balance_evidence,
        )

    before = broker.get_position(product.symbol)
    before_qty = _evidence_float(before.qty)
    before_avg_price = _evidence_float(before.avg_price)
    if (
        before.symbol != product.symbol
        or before_qty is None
        or before_qty < 0
        or before_avg_price is None
        or before_avg_price < 0
    ):
        status.update(
            ok=False,
            reason="invalid_spot_flatten_position_evidence",
            error="broker returned invalid pre-buyback BTC balance evidence",
        )
        return status
    price = broker.get_price(product.symbol)
    if not math.isfinite(float(price)) or float(price) <= 0:
        status.update(
            ok=False,
            reason="invalid_spot_flatten_price",
            error=f"broker price is invalid: {price!r}",
        )
        return status
    raw_requested_qty = float(position_detail["quote_value"]) / float(price)
    if not math.isfinite(raw_requested_qty) or raw_requested_qty <= 0:
        status.update(
            ok=False,
            reason="invalid_spot_flatten_qty",
            error=f"spot buyback quantity is invalid: {raw_requested_qty!r}",
        )
        return status
    normalizer = getattr(broker, "normalize_order_qty", None)
    try:
        normalized_raw = (
            normalizer(
                product.symbol,
                raw_requested_qty,
                price=float(price),
                reduce_only=False,
            )
            if callable(normalizer)
            else raw_requested_qty
        )
        requested_qty = float(normalized_raw)
    except Exception as exc:
        status.update(
            ok=False,
            reason="invalid_spot_flatten_qty",
            error=f"spot buyback quantity normalization failed: {type(exc).__name__}: {exc}",
        )
        return status
    if not math.isfinite(requested_qty) or requested_qty <= 0:
        status.update(
            ok=False,
            reason="invalid_spot_flatten_qty",
            error=f"normalized spot buyback quantity is invalid: {requested_qty!r}",
        )
        return status
    tolerance = max(abs(raw_requested_qty) * 1e-12, 1e-12)
    if requested_qty - raw_requested_qty > tolerance:
        status.update(
            ok=False,
            reason="invalid_spot_flatten_qty",
            error=(
                "spot buyback quantity normalization increased intended exposure from "
                f"{raw_requested_qty:g} to {requested_qty:g}"
            ),
        )
        return status

    client_id = _spot_flatten_client_id(
        strategy_id=strategy_id,
        symbol=product.symbol,
        qty=requested_qty,
        quote_budget=float(position_detail["quote_value"]),
        position_before_qty=before_qty,
        broker_account_fingerprint=current_account_fingerprint,
    )
    raw_intent = {
        "version": 1,
        "strategy_id": strategy_id,
        "symbol": product.symbol,
        "side": OrderSide.BUY.value,
        "order_type": OrderType.MARKET.value,
        "client_id": client_id,
        "broker_account_fingerprint": current_account_fingerprint,
        "qty": requested_qty,
        "quote_budget": float(position_detail["quote_value"]),
        "position_before": {
            "symbol": before.symbol,
            "qty": before_qty,
            "avg_price": before_avg_price,
        },
        "created_ts": time.time(),
    }
    intent = _validated_spot_flatten_intent(
        product,
        raw_intent,
        strategy_id=strategy_id,
        position_detail=position_detail,
    )
    try:
        state = _persist_spot_flatten_intent(product, state, intent)
    except Exception as exc:
        status.update(
            ok=False,
            reason="flatten_intent_persist_failed",
            error=f"could not persist flatten_intent before broker submission: {type(exc).__name__}: {exc}",
        )
        return status

    order = Order(
        symbol=product.symbol,
        side=OrderSide.BUY,
        qty=requested_qty,
        type=OrderType.MARKET,
        client_id=client_id,
    )
    status.update(
        flatten_intent=intent,
        position_before={
            "symbol": before.symbol,
            "qty": before.qty,
            "avg_price": before.avg_price,
        },
        spot_step_aside={
            **position_detail,
            "reference_price": float(price),
            "raw_requested_qty": raw_requested_qty,
            "requested_qty": requested_qty,
        },
    )
    try:
        fill = broker.place_order(order)
        _assert_spot_flatten_fill_valid(strategy_id, order, fill)
    except Exception as exc:
        return unresolved(
            f"spot buyback submission/fill is ambiguous: {type(exc).__name__}: {exc}",
            intent=intent,
        )
    fill_detail = {
        "symbol": fill.symbol,
        "side": _fill_side_value(fill),
        "qty": fill.qty,
        "price": fill.price,
        "fee": fill.fee,
        "timestamp": fill.timestamp,
    }
    status.update(flattened=True, fill=fill_detail)
    try:
        after = broker.get_position(product.symbol)
    except Exception as exc:
        return unresolved(
            f"spot buyback filled but post-fill BTC balance is unavailable: {type(exc).__name__}: {exc}",
            intent=intent,
        )
    balance_evidence = _spot_flatten_balance_evidence(
        product,
        intent,
        after,
        expected_filled_qty=float(fill.qty),
    )
    status.update(
        position_after={
            "symbol": after.symbol,
            "qty": after.qty,
            "avg_price": after.avg_price,
        },
        balance_evidence=balance_evidence,
    )
    if not balance_evidence["proven"]:
        return unresolved(
            "spot buyback fill does not match the observed BTC balance increase",
            intent=intent,
            balance_evidence=balance_evidence,
        )
    try:
        local_state = _commit_spot_flatten_state(
            product,
            broker,
            intent,
            balance_evidence=balance_evidence,
            fill=fill,
        )
    except Exception as exc:
        return unresolved(
            f"spot buyback is proven but local-state commit failed: {type(exc).__name__}: {exc}",
            intent=intent,
            balance_evidence=balance_evidence,
        )
    status.update(ok=True, local_state=local_state)
    return status


def _fill_from_flatten_intent(intent: dict[str, Any]) -> Fill:
    payload = intent.get("fill")
    if not isinstance(payload, dict):
        raise RuntimeError("Proven flatten intent has no fill payload.")
    return Fill(
        symbol=str(payload["symbol"]),
        side=OrderSide(str(payload["side"])),
        qty=float(payload["qty"]),
        price=float(payload["price"]),
        fee=float(payload["fee"]),
        timestamp=float(payload["timestamp"]),
    )


def _already_accounted_futures_flatten(
    product: ProductConfig,
    broker: Any,
) -> dict[str, Any] | None:
    state, state_error = _read_local_state_for_flatten(product)
    if state_error or state is None:
        return None
    last = _validated_accounted_futures_flatten(product, state)
    if last is None:
        return None
    for marker in ("pending_order", "pending_entry_recovery", "risk_recovery_incident"):
        if state.get(marker) is not None:
            return None
    current = broker.get_position(product.symbol)
    if current.symbol != product.symbol or not current.is_flat:
        return None
    positions = broker.list_account_futures_positions()
    regular = broker.list_account_open_orders(conditional=False)
    conditional = broker.list_account_open_orders(conditional=True)
    if not isinstance(positions, list | tuple) or positions:
        return None
    if not isinstance(regular, list | tuple) or regular:
        return None
    if not isinstance(conditional, list | tuple) or conditional:
        return None
    if product.trade_log.is_symlink() or not product.trade_log.exists():
        return None
    try:
        with product.trade_log.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    matches = [row for row in rows if row.get("exit_event_id") == last["exit_event_id"]]
    if len(matches) != 1 or matches[0].get("exit_reason") != "emergency_flatten":
        return None
    return {
        "exit_event_id": last["exit_event_id"],
        "flatten_client_id": last["flatten_client_id"],
        "trade_log": str(product.trade_log),
        "whole_account_positions": 0,
        "whole_account_regular_orders": 0,
        "whole_account_conditional_orders": 0,
    }


def _finalize_proven_futures_flatten(
    product: ProductConfig,
    broker: Any,
    status: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    try:
        current = broker.get_position(product.symbol)
        if not current.is_flat:
            raise RuntimeError(
                "Durable broker-flat evidence exists but the account is exposed again."
            )
        fill = _fill_from_flatten_intent(intent)
        expected_stops = _local_futures_protective_stops(product)
        stop_cleanup = _finish_futures_native_stop_cleanup(product, broker, expected_stops)
        accounting = _commit_flatten_exit_accounting(
            product,
            broker,
            intent=intent,
            fill=fill,
            quote_balance_after=float(intent["quote_balance_after"]),
            native_stop_cleanup=stop_cleanup,
        )
    except Exception as exc:
        status.update(
            ok=False,
            reason="flatten_accounting_unresolved",
            error=f"{type(exc).__name__}: {exc}",
            flatten_intent=intent,
            operator_action=(
                "Keep the product paused; broker exposure may be flat, but the durable stop, "
                "trade ledger, or local accounting transition is not committed."
            ),
        )
        return status
    unresolved_pending = bool(accounting["pending_order_retained"])
    status.update(
        ok=not unresolved_pending,
        flattened=True,
        reason=(
            "flatten_accounted"
            if not unresolved_pending
            else "preexisting_order_reconciliation_retained"
        ),
        fill=intent["fill"],
        position_after=intent["position_after"],
        native_stop_cleanup=stop_cleanup,
        accounting=accounting,
    )
    return status


def _flatten_futures_product(
    product: ProductConfig,
    status: dict[str, Any],
    broker: Any,
    account_fingerprint: str,
) -> dict[str, Any]:
    try:
        already_accounted = _already_accounted_futures_flatten(product, broker)
    except Exception as exc:
        status.update(
            reason="already_accounted_flatten_verification_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return status
    if already_accounted is not None:
        status.update(
            ok=True,
            skipped=True,
            flattened=False,
            reason="already_accounted_flat",
            accounting=already_accounted,
        )
        return status
    try:
        resumed = _resume_flatten_exit_accounting(product, broker)
    except Exception as exc:
        status.update(
            reason="unresolved_exit_accounting_intent",
            error=f"{type(exc).__name__}: {exc}",
        )
        return status
    if resumed is not None:
        status.update(
            ok=True,
            flattened=True,
            reason="exit_accounting_resumed",
            accounting=resumed,
        )
        return status

    try:
        accounting_bot = _flatten_accounting_bot(product, broker)
        strategy, local_position = _flatten_strategy_and_position(accounting_bot)
    except Exception as exc:
        status.update(
            reason="flatten_accounting_precondition_failed",
            error=f"{type(exc).__name__}: {exc}",
            operator_action=(
                "Keep the product paused; reconcile durable strategy/position accounting "
                "evidence before submitting a close."
            ),
        )
        return status
    state = accounting_bot.state
    existing_raw = state.get("flatten_intent")
    if existing_raw is not None:
        try:
            existing = _validated_futures_flatten_intent(
                product,
                existing_raw,
                position=local_position,
            )
        except Exception as exc:
            status.update(
                reason="unresolved_flatten_intent",
                error=f"{type(exc).__name__}: {exc}",
                flatten_intent=existing_raw,
            )
            return status
        if existing["phase"] == "broker_flat_proven":
            return _finalize_proven_futures_flatten(product, broker, status, existing)
        current = broker.get_position(product.symbol)
        status.update(
            reason="unresolved_flatten_intent",
            error=(
                "Prepared futures flatten intent has an ambiguous submission boundary; "
                "reconcile its deterministic client ID before any retry."
            ),
            flatten_intent=existing,
            position_current={
                "symbol": current.symbol,
                "qty": current.qty,
                "avg_price": current.avg_price,
            },
            operator_action=(
                "Keep the product paused. The runtime will not submit a second close order."
            ),
        )
        return status

    before = broker.get_position(product.symbol)
    status["position_before"] = {
        "symbol": before.symbol,
        "qty": before.qty,
        "avg_price": before.avg_price,
    }
    if before.is_flat:
        status.update(
            reason="unresolved_flat_without_exit_fill",
            error=(
                "Broker is flat while local accounting remains open, but no durable close "
                "fill proves price and fees. Local position state was retained."
            ),
        )
        return status

    expected_qty = _positive_evidence_float(local_position.get("broker_qty"))
    direction = str(local_position.get("direction") or "").lower()
    expected_signed = (
        expected_qty
        if direction == "long" and expected_qty is not None
        else -expected_qty
        if direction == "short" and expected_qty is not None
        else None
    )
    before_qty = _evidence_float(before.qty)
    before_price = _positive_evidence_float(before.avg_price)
    tolerance = max(abs(float(before_qty or 0.0)) * 1e-9, 1e-12)
    if (
        before.symbol != product.symbol
        or before_qty is None
        or before_price is None
        or expected_signed is None
        or abs(before_qty - expected_signed) > tolerance
    ):
        status.update(
            reason="broker_position_reconciliation_failed",
            error="Broker position does not exactly match the single durable position.",
        )
        return status

    side = OrderSide.SELL if before_qty > 0 else OrderSide.BUY
    normalizer = getattr(broker, "normalize_order_qty", None)
    try:
        normalized_qty = float(
            normalizer(
                product.symbol,
                abs(before_qty),
                price=before_price,
                reduce_only=True,
            )
            if callable(normalizer)
            else abs(before_qty)
        )
    except Exception as exc:
        status.update(reason="invalid_futures_flatten_qty", error=str(exc))
        return status
    if (
        not math.isfinite(normalized_qty)
        or normalized_qty <= 0
        or abs(normalized_qty - abs(before_qty)) > tolerance
    ):
        status.update(
            reason="invalid_futures_flatten_qty",
            error="Venue normalization did not preserve the complete open-position size.",
        )
        return status

    quote_before = _strict_flatten_number(
        broker.get_balance(),
        field="quote_balance_before",
        non_negative=True,
    )
    position_digest = _flatten_state_digest(
        local_position,
        label="Emergency flatten position",
    )
    client_id = _futures_flatten_client_id(
        strategy_id=strategy["id"],
        symbol=product.symbol,
        side=side,
        qty=normalized_qty,
        position_digest=position_digest,
        broker_account_fingerprint=account_fingerprint,
    )
    raw_intent = {
        "version": 1,
        "phase": "prepared",
        "strategy_id": strategy["id"],
        "symbol": product.symbol,
        "side": side.value,
        "order_type": OrderType.MARKET.value,
        "reduce_only": True,
        "submission_kind": "reduce_only_market",
        "client_id": client_id,
        "broker_account_fingerprint": account_fingerprint,
        "qty": normalized_qty,
        "position_digest": position_digest,
        "position_before": {
            "symbol": before.symbol,
            "qty": before_qty,
            "avg_price": before_price,
        },
        "quote_balance_before": quote_before,
        "fill": None,
        "position_after": None,
        "quote_balance_after": None,
        "realized_account_delta": None,
        "created_ts": time.time(),
        "proven_ts": None,
    }
    intent = _validated_futures_flatten_intent(
        product,
        raw_intent,
        position=local_position,
    )
    try:
        _persist_flatten_intent(product, state, intent)
    except Exception as exc:
        status.update(reason="flatten_intent_persist_failed", error=str(exc))
        return status

    order = Order(
        symbol=product.symbol,
        side=side,
        qty=normalized_qty,
        type=OrderType.MARKET,
        reduce_only=True,
        client_id=client_id,
    )
    status["flatten_intent"] = intent
    try:
        fill = broker.place_order(order)
        _assert_futures_flatten_fill_valid(product, before, fill)
    except Exception as exc:
        status.update(
            reason="unresolved_flatten_intent",
            close_error=f"{type(exc).__name__}: {exc}",
            operator_action=(
                "Keep the product paused and reconcile the deterministic client ID; no "
                "automatic retry will submit another close."
            ),
        )
        try:
            attempted = broker.get_position(product.symbol)
            status["position_after_attempt"] = {
                "symbol": attempted.symbol,
                "qty": attempted.qty,
                "avg_price": attempted.avg_price,
                "is_flat": attempted.is_flat,
            }
        except Exception as readback_exc:
            status["position_after_attempt_error"] = str(readback_exc)
        return status

    after = broker.get_position(product.symbol)
    quote_after = _strict_flatten_number(
        broker.get_balance(),
        field="quote_balance_after",
        non_negative=True,
    )
    fill_payload = {
        "symbol": fill.symbol,
        "side": _fill_side_value(fill),
        "qty": float(fill.qty),
        "price": float(fill.price),
        "fee": float(fill.fee),
        "timestamp": float(fill.timestamp),
    }
    status.update(fill=fill_payload, flattened=True)
    if not after.is_flat:
        status.update(
            reason="unresolved_flatten_intent",
            error=f"Flatten fill returned but broker position remains {after.qty:g}.",
        )
        return status
    proven_raw = {
        **intent,
        "phase": "broker_flat_proven",
        "fill": fill_payload,
        "position_after": {
            "symbol": after.symbol,
            "qty": float(after.qty),
            "avg_price": float(after.avg_price),
        },
        "quote_balance_after": quote_after,
        "realized_account_delta": quote_after - quote_before,
        "proven_ts": time.time(),
    }
    proven = _validated_futures_flatten_intent(
        product,
        proven_raw,
        position=local_position,
    )
    try:
        _persist_proven_flatten_intent(product, intent, proven)
    except Exception as exc:
        status.update(
            reason="flatten_accounting_evidence_persist_failed",
            error=str(exc),
            operator_action=(
                "Keep the product paused; broker exposure is flat but fill/account evidence "
                "did not reach durable state."
            ),
        )
        return status
    return _finalize_proven_futures_flatten(product, broker, status, proven)


def flatten_product_once(product: ProductConfig) -> dict[str, Any]:
    status: dict[str, Any] = {
        "product": _product_to_status(product),
        "started_at": utc_now(),
        "action": "flatten",
        "ok": False,
    }
    if product.execution_mode != "live":
        status.update(ok=True, skipped=True, reason="not_live")
        return status
    if product.market != "futures":
        if product.objective == "btc_accumulation" and product.market == "spot":
            return _flatten_btc_spot_step_aside_product(product, status)
        status.update(ok=True, skipped=True, reason="spot_flatten_not_supported")
        return status

    status["live_environment"] = assert_live_environment(
        product,
        require_production=True,
    )
    broker = build_live_broker(product)
    account_fingerprint = _assert_futures_flatten_account_binding(product, broker)
    status.update(
        broker=broker.name,
        broker_account_fingerprint=account_fingerprint,
    )
    return _flatten_futures_product(
        product,
        status,
        broker,
        account_fingerprint,
    )


def _bot_cycle_errors(bot: PaperTradingBot) -> list[dict[str, Any]]:
    return list(getattr(bot, "cycle_errors", []) or [])


def _open_position_details(
    bot: PaperTradingBot, open_positions: dict[str, Any]
) -> list[dict[str, Any]]:
    strategies = {
        str(strategy.get("id")): strategy
        for strategy in (getattr(bot, "strategies", []) or [])
        if isinstance(strategy, dict) and strategy.get("id") is not None
    }
    details: list[dict[str, Any]] = []
    for strategy_id, raw_position in sorted(open_positions.items()):
        if not isinstance(raw_position, dict):
            continue
        strategy = strategies.get(str(strategy_id)) or {}
        base_timeframe = strategy.get("base_timeframe")
        horizon_bars = strategy.get("horizon_bars")
        timeframe_seconds = (
            TIMEFRAME_SECONDS.get(base_timeframe) if isinstance(base_timeframe, str) else None
        )
        stale_after_seconds = None
        try:
            horizon_float = float(horizon_bars)
        except (TypeError, ValueError):
            horizon_float = 0.0
        if timeframe_seconds is not None and horizon_float > 0:
            stale_after_seconds = (
                float(timeframe_seconds) * horizon_float * OPEN_POSITION_STALE_HORIZON_MULTIPLE
            )
        details.append(
            {
                "strategy_id": str(strategy_id),
                "direction": raw_position.get("direction"),
                "entry_time": raw_position.get("entry_time"),
                "position_size": raw_position.get("position_size"),
                "entry_price": raw_position.get("entry_price"),
                "sl_price": raw_position.get("sl_price"),
                "tp_price": raw_position.get("tp_price"),
                "sl_pct": raw_position.get("sl_pct"),
                "tp_pct": raw_position.get("tp_pct"),
                "base_timeframe": base_timeframe,
                "horizon_bars": horizon_bars,
                "stale_after_seconds": stale_after_seconds,
                **{
                    key: raw_position.get(key)
                    for key in (
                        "broker_symbol",
                        "broker_qty",
                        "broker_side",
                        "broker_entry_price",
                        "broker_entry_fee",
                        "broker_entry_balance",
                        "broker_account_fingerprint",
                        "broker_entry_quote_balance_before",
                        "broker_entry_quote_balance_after",
                        "broker_entry_quote_value",
                        "broker_entry_quote_value_source",
                        "broker_requested_qty",
                        "broker_fill_ratio",
                        "broker_exit_sizing",
                        "broker_stop_order_id",
                        "broker_stop_client_id",
                        "broker_stop_trigger_price",
                    )
                    if key in raw_position
                },
            }
        )
    return details


def _bot_status_snapshot(bot: PaperTradingBot) -> dict[str, Any]:
    state = getattr(bot, "state", {}) or {}
    open_positions = state.get("open_positions", {})
    inactive_strategies = state.get("inactive_strategies", [])
    state_errors: list[dict[str, Any]] = []
    if isinstance(open_positions, dict):
        open_position_count: int | str = len(open_positions)
        open_position_details = _open_position_details(bot, open_positions)
    else:
        open_position_count = "invalid"
        open_position_details = []
        state_errors.append(
            {
                "field": "open_positions",
                "error": f"expected object, got {type(open_positions).__name__}",
            }
        )
    snapshot: dict[str, Any] = {
        "equity": state.get("equity"),
        "peak_equity": state.get("peak_equity"),
        "drawdown_fraction": state.get("drawdown_fraction"),
        "drawdown_limit_fraction": state.get("drawdown_limit_fraction"),
        "drawdown_halted": state.get("drawdown_halted"),
        "drawdown_halted_at": state.get("drawdown_halted_at"),
        "drawdown_halt_reason": state.get("drawdown_halt_reason"),
        "open_positions": open_position_count,
        "open_position_details": open_position_details,
        "inactive_strategies": list(inactive_strategies)
        if isinstance(inactive_strategies, list)
        else [],
    }
    position_events = getattr(bot, "position_events", []) or []
    if isinstance(position_events, list):
        snapshot["position_events"] = [
            dict(event) for event in position_events if isinstance(event, dict)
        ]
    exit_intent = state.get("exit_accounting_intent")
    if isinstance(exit_intent, dict):
        snapshot["exit_accounting_intent"] = {
            key: exit_intent.get(key)
            for key in (
                "version",
                "phase",
                "exit_event_id",
                "strategy_id",
                "created_at",
                "broker_flat_proven",
            )
        }
    elif exit_intent is not None:
        snapshot["exit_accounting_intent"] = exit_intent
    for state_key, visible_fields in BOT_STATUS_DURABLE_STATE_FIELDS.items():
        marker = state.get(state_key)
        if marker is None:
            continue
        if isinstance(marker, dict):
            snapshot[state_key] = {key: marker[key] for key in visible_fields if key in marker}
        else:
            # Preserve the fact that the safety marker exists without copying
            # an untrusted, potentially very large value into status reports.
            snapshot[state_key] = {"invalid_type": type(marker).__name__}
    if state_errors:
        snapshot["state_errors"] = state_errors
    return snapshot


def _bot_cycle_failure_status(
    status: dict[str, Any],
    bot: PaperTradingBot,
    exc: Exception,
    *,
    broker_name: str | None = None,
) -> dict[str, Any]:
    status.update(
        ok=False,
        error=str(exc),
        cycle_errors=[{"stage": "run_cycle", "error": str(exc), "type": type(exc).__name__}],
        **_bot_status_snapshot(bot),
    )
    if broker_name is not None:
        status["broker"] = broker_name
    return status


def _assert_live_pre_entry_gate(
    product: ProductConfig,
    artifact_snapshot: dict[str, Any],
    *,
    approval_ledger: Path,
    config: AutopilotConfig | None,
) -> dict[str, Any]:
    """Re-sample every risk-increasing gate immediately before an entry.

    A cycle can spend seconds fetching market data and building features.  The
    operator control file, approval ledger, environment, or time-bounded
    preflight evidence may change during that interval.  Exit/reconciliation
    paths deliberately do not call this gate.
    """

    if config is not None:
        control = load_control(config.control_file)
        if control.get("control_error"):
            raise RuntimeError(
                f"Live pre-entry gate found an invalid control file: {control['control_error']}"
            )
        unknown = unknown_control_selectors(control, config)
        if unknown:
            raise RuntimeError(
                "Live pre-entry gate found unknown control selectors: "
                + json.dumps(unknown, sort_keys=True)
            )
        if should_flatten_product(control, product.name):
            raise RuntimeError(f"Live pre-entry gate blocked {product.name}: flatten is requested.")
        if is_product_paused(control, product.name):
            raise RuntimeError(f"Live pre-entry gate blocked {product.name}: product is paused.")

    policy = assert_loaded_strategy_artifact_allowed(
        product,
        artifact_snapshot,
        artifact_path=product.strategies_path,
    )
    assert_loaded_artifact_live_approved(
        artifact_snapshot,
        product.strategies_path,
        approval_ledger,
        product=product,
    )
    preflight = assert_recent_preflight(product, artifact=artifact_snapshot)
    rehearsal = assert_recent_testnet_rehearsal(product, artifact=artifact_snapshot)
    current_environment = assert_live_environment(product, require_production=True)
    _assert_current_environment_matches_preflight(
        product,
        current=current_environment,
        recorded=preflight["exchange_environment"],
    )
    return {
        "strategy_policy": policy,
        "approval_gate": "approved",
        "preflight_gate": preflight,
        "testnet_rehearsal_gate": rehearsal,
        "live_environment": current_environment,
    }


def run_product_once(
    product: ProductConfig,
    *,
    approval_ledger: Path,
    allow_entries: bool = True,
    config: AutopilotConfig | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "product": _product_to_status(product),
        "started_at": utc_now(),
        "ok": False,
    }
    if not product.enabled:
        status.update(ok=True, skipped=True, reason="disabled")
        return status
    if product.execution_mode == "live":
        local_open_positions = _local_open_position_count(product)
        local_state_management = _local_state_requires_management(product)
        management_only = not allow_entries or local_state_management
        artifact_snapshot: dict[str, Any] | None = None
        status["entries_allowed"] = not management_only
        if management_only:
            status["entry_gate"] = {
                "status": "management_only",
                "reason": "paused_or_durable_management_state",
                "local_open_positions": local_open_positions,
                "local_state_requires_management": local_state_management,
            }
            # Approval/policy changes must block risk-increasing entries, never
            # reconciliation or an exit that reduces an existing exposure.
            try:
                artifact_snapshot = load_artifact(product.strategies_path)
                status["strategy_policy"] = assert_loaded_strategy_artifact_allowed(
                    product,
                    artifact_snapshot,
                    artifact_path=product.strategies_path,
                )
            except (ApprovalError, FileNotFoundError, StrategyPolicyError, ValueError) as exc:
                status["strategy_policy"] = {"ok": False, "management_warning": str(exc)}
                if artifact_snapshot is None:
                    artifact_snapshot = _frozen_management_artifact(product)
                    status["strategy_policy"]["artifact_source"] = artifact_snapshot.get("source")
            if local_state_management and not local_open_positions:
                artifact_snapshot = _frozen_management_artifact(product)
                status.setdefault("strategy_policy", {})["artifact_source"] = artifact_snapshot.get(
                    "source"
                )
            try:
                if artifact_snapshot is None:
                    raise ApprovalError("Strategy artifact snapshot is unavailable.")
                assert_loaded_artifact_live_approved(
                    artifact_snapshot,
                    product.strategies_path,
                    approval_ledger,
                    product=product,
                )
            except (ApprovalError, FileNotFoundError, StrategyPolicyError, ValueError) as exc:
                status["approval_gate"] = "management_only"
                status["approval_warning"] = str(exc)
            else:
                status["approval_gate"] = "approved"
            status["preflight_gate"] = {
                "skipped": True,
                "reason": "entry_only_gate_while_managing_exposure",
            }
            status["testnet_rehearsal_gate"] = {
                "skipped": True,
                "reason": "entry_only_gate_while_managing_exposure",
            }
        else:
            artifact_snapshot = load_artifact(product.strategies_path)
            status["strategy_policy"] = assert_loaded_strategy_artifact_allowed(
                product,
                artifact_snapshot,
                artifact_path=product.strategies_path,
            )
            assert_loaded_artifact_live_approved(
                artifact_snapshot,
                product.strategies_path,
                approval_ledger,
                product=product,
            )
            status["approval_gate"] = "approved"
            status["preflight_gate"] = assert_recent_preflight(product, artifact=artifact_snapshot)
            status["testnet_rehearsal_gate"] = assert_recent_testnet_rehearsal(
                product,
                artifact=artifact_snapshot,
            )
        status["live_environment"] = assert_live_environment(
            product,
            require_production=True,
        )
        if not management_only:
            _assert_current_environment_matches_preflight(
                product,
                current=status["live_environment"],
                recorded=status["preflight_gate"]["exchange_environment"],
            )
        broker = build_live_broker(product)
        pre_entry_gate = None
        if not management_only:

            def pre_entry_gate() -> dict[str, Any]:
                return _assert_live_pre_entry_gate(
                    product,
                    artifact_snapshot,
                    approval_ledger=approval_ledger,
                    config=config,
                )

        bot = PaperTradingBot(
            strategies_path=product.strategies_path,
            state_file=product.state_file,
            trade_log=product.trade_log,
            starting_equity=product.starting_equity,
            regime_guard=product.regime_guard,
            regime_mayer_top=product.regime_mayer_top,
            broker=broker,
            symbol=product.symbol,
            market=product.market,
            objective=product.objective,
            base_asset=product.base_asset,
            live_gate_approved=True,
            allow_entries=not management_only,
            artifact_payload=artifact_snapshot,
            pre_entry_gate=pre_entry_gate,
        )
        try:
            bot.run_cycle()
        except Exception as exc:
            return _bot_cycle_failure_status(status, bot, exc, broker_name=broker.name)
        cycle_errors = _bot_cycle_errors(bot)
        snapshot = _bot_status_snapshot(bot)
        status.update(
            ok=not cycle_errors and not snapshot.get("state_errors"),
            broker=broker.name,
            cycle_errors=cycle_errors,
            **snapshot,
        )
        return status

    if not product.strategies_path.exists():
        status.update(
            ok=True,
            skipped=True,
            reason="waiting_for_strategy_artifact",
            detail=f"Strategy artifact not found: {product.strategies_path}",
        )
        return status

    try:
        status["strategy_policy"] = assert_strategy_artifact_allowed(product)
    except StrategyPolicyError as exc:
        status.update(
            ok=True,
            skipped=True,
            reason="strategy_policy_blocked",
            detail=str(exc),
        )
        return status
    bot = PaperTradingBot(
        strategies_path=product.strategies_path,
        state_file=product.state_file,
        trade_log=product.trade_log,
        starting_equity=product.starting_equity,
        regime_guard=product.regime_guard,
        regime_mayer_top=product.regime_mayer_top,
        symbol=product.symbol,
        market=product.market,
        objective=product.objective,
        base_asset=product.base_asset,
        allow_entries=allow_entries,
    )
    try:
        bot.run_cycle()
    except Exception as exc:
        return _bot_cycle_failure_status(status, bot, exc)
    cycle_errors = _bot_cycle_errors(bot)
    snapshot = _bot_status_snapshot(bot)
    status.update(
        ok=not cycle_errors and not snapshot.get("state_errors"),
        cycle_errors=cycle_errors,
        **snapshot,
    )
    return status


def _report_output_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "written": False}


def _report_error(stage: str, exc: Exception, *, path: Path | None = None) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
    }
    if path is not None:
        detail["path"] = str(path)
    return detail


def _write_report_output(
    summary: dict[str, Any],
    key: str,
    path: Path,
    writer: Callable[[Path, Any], None],
    payload: Any,
) -> None:
    try:
        writer(path, payload)
        summary["outputs"][key] = {"path": str(path), "written": True}
    except Exception as exc:
        LOGGER.exception("Failed to write %s", key)
        summary["errors"].append(_report_error(f"{key}_write_failed", exc, path=path))


def write_cycle_reports(config: AutopilotConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": True,
        "outputs": {
            "operator_report": _report_output_status(config.operator_report_file),
            "operator_report_json": _report_output_status(config.operator_report_json_file),
            "readiness_report": _report_output_status(config.readiness_report_file),
            "readiness_report_json": _report_output_status(config.readiness_report_json_file),
        },
        "errors": [],
    }

    try:
        operator_report = build_operator_report(config)
    except Exception as exc:
        LOGGER.exception("Failed to build operator report")
        summary["errors"].append(_report_error("operator_report_build_failed", exc))
    else:
        try:
            operator_markdown = render_operator_markdown(operator_report)
        except Exception as exc:
            LOGGER.exception("Failed to render operator markdown")
            summary["errors"].append(_report_error("operator_report_render_failed", exc))
        else:
            _write_report_output(
                summary,
                "operator_report",
                config.operator_report_file,
                write_text_atomic,
                operator_markdown,
            )
        _write_report_output(
            summary,
            "operator_report_json",
            config.operator_report_json_file,
            write_json_atomic,
            operator_report,
        )

    from src.autopilot.readiness import build_readiness_report, render_readiness_markdown

    try:
        readiness_report = build_readiness_report(config)
    except Exception as exc:
        LOGGER.exception("Failed to build readiness report")
        summary["errors"].append(_report_error("readiness_report_build_failed", exc))
    else:
        try:
            readiness_markdown = render_readiness_markdown(readiness_report)
        except Exception as exc:
            LOGGER.exception("Failed to render readiness markdown")
            summary["errors"].append(_report_error("readiness_report_render_failed", exc))
        else:
            _write_report_output(
                summary,
                "readiness_report",
                config.readiness_report_file,
                write_text_atomic,
                readiness_markdown,
            )
        _write_report_output(
            summary,
            "readiness_report_json",
            config.readiness_report_json_file,
            write_json_atomic,
            readiness_report,
        )

    summary["ok"] = not summary["errors"]
    summary["operator_report"] = str(config.operator_report_file)
    summary["operator_report_json"] = str(config.operator_report_json_file)
    summary["readiness_report"] = str(config.readiness_report_file)
    summary["readiness_report_json"] = str(config.readiness_report_json_file)
    return summary


def _report_json_available(reporting: dict[str, Any], key: str, path: Path) -> bool:
    outputs = reporting.get("outputs")
    if isinstance(outputs, dict):
        output = outputs.get(key)
        return bool(isinstance(output, dict) and output.get("written"))
    return path.exists()


def _emit_runtime_alert(
    config: AutopilotConfig,
    *,
    severity: str,
    title: str,
    detail: dict[str, Any],
    cooldown_seconds: int | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    try:
        return emit_alert(
            alert_file=config.alert_file,
            state_file=config.alert_state_file,
            severity=severity,
            title=title,
            detail=detail,
            cooldown_seconds=(
                config.alert_cooldown_seconds if cooldown_seconds is None else cooldown_seconds
            ),
            dedupe_key=dedupe_key,
            webhook_url_env=config.webhook_url_env,
        )
    except Exception as exc:  # alerting must never crash trading supervision
        LOGGER.exception("Failed to emit autopilot alert: %s", title)
        return {"sent": False, "error": str(exc)}


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cycle_failure_dedupe_key(report: dict[str, Any]) -> str:
    failed_products = sorted(
        str((item.get("product") or {}).get("name") or "unknown")
        for item in report.get("products", [])
        if isinstance(item, dict) and not item.get("ok")
    )
    failed_jobs = sorted(
        str(item.get("name") or "unknown")
        for item in report.get("jobs", [])
        if isinstance(item, dict) and not item.get("ok")
    )
    signature = {
        "control_error": bool(report.get("control_error")),
        "job_config_errors": bool(report.get("job_config_errors")),
        "data_update_failed": isinstance(report.get("data_update"), dict)
        and not report["data_update"].get("ok"),
        "products": failed_products,
        "jobs": failed_jobs,
    }
    return f"cycle-failed:{_stable_digest(signature)}"


def _previous_status(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _emit_position_change_alerts(
    config: AutopilotConfig,
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for product_status in products:
        if not isinstance(product_status, dict):
            continue
        product = product_status.get("product")
        product = product if isinstance(product, dict) else {}
        events = product_status.get("position_events")
        if not isinstance(events, list):
            continue
        for raw_event in events:
            if not isinstance(raw_event, dict):
                continue
            event_id = raw_event.get("event_id")
            event_type = raw_event.get("event_type")
            if not isinstance(event_id, str) or not event_id:
                continue
            if event_type not in {"opened", "closed"}:
                continue
            detail = {
                **raw_event,
                "product": product.get("name"),
                "objective": product.get("objective"),
                "configured_mode": product.get("execution_mode"),
                "autonomous": True,
                "operator_action_required": False,
            }
            result = _emit_runtime_alert(
                config,
                severity="info",
                title=f"autonomous position {event_type}",
                detail=detail,
                cooldown_seconds=7 * 24 * 60 * 60,
                dedupe_key=f"position:{event_type}:{event_id}",
            )
            results.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "product": product.get("name"),
                    **result,
                }
            )
    return results


def _daily_digest_detail(operator_report: dict[str, Any]) -> dict[str, Any]:
    products = []
    for item in operator_report.get("products", []):
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        products.append(
            {
                key: item.get(key)
                for key in (
                    "name",
                    "objective",
                    "market",
                    "mode",
                    "cycle_ok",
                    "equity",
                    "drawdown_fraction",
                    "drawdown_halted",
                    "open_positions",
                    "reason",
                )
            }
        )
    jobs = [
        item
        for item in operator_report.get("scheduled_jobs", [])
        if isinstance(item, dict) and item.get("enabled")
    ]
    research_cycle = operator_report.get("research_cycle")
    research_cycle = research_cycle if isinstance(research_cycle, dict) else {}
    generated_batch = operator_report.get("generated_batch")
    generated_batch = generated_batch if isinstance(generated_batch, dict) else {}
    candidate_paper = operator_report.get("candidate_paper")
    candidate_paper = candidate_paper if isinstance(candidate_paper, dict) else {}
    return {
        "autonomous": True,
        "operator_action_required": False,
        "system_ok": operator_report.get("ok"),
        "products": products,
        "scheduled_jobs": {
            "enabled": len(jobs),
            "failing": sorted(
                str(item.get("name") or "unknown")
                for item in jobs
                if item.get("status") == "failed" or item.get("ok") is False
            ),
            "overdue": sorted(
                str(item.get("name") or "unknown") for item in jobs if item.get("overdue") is True
            ),
        },
        "research": {
            "cycle_ok": research_cycle.get("ok"),
            "cycle_generated_at": research_cycle.get("generated_at"),
            "cycle_summary": research_cycle.get("summary"),
            "batch_ok": generated_batch.get("ok"),
            "batch_generated_at": generated_batch.get("generated_at"),
            "batch_summary": generated_batch.get("summary"),
        },
        "candidate_paper": {
            key: candidate_paper.get(key)
            for key in (
                "status",
                "ok",
                "fresh",
                "products",
                "open_positions",
                "activation_ready_products",
                "drawdown_halted_products",
            )
        },
    }


def _auto_clear_successful_flatten_requests(
    config: AutopilotConfig,
    control: dict[str, Any],
    flatten_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not flatten_results:
        return []
    results: list[dict[str, Any]] = []
    reason = "auto-cleared after successful runtime flatten"
    expected_control = dict(control)
    if control.get("flatten_all"):
        targets = [
            {
                key: item.get(key)
                for key in ("product_name", "ok", "skipped", "reason", "error")
                if item.get(key) is not None
            }
            for item in flatten_results
        ]
        if not all(bool(item.get("ok")) for item in flatten_results):
            return [
                {
                    "command": "clear-flatten",
                    "name": None,
                    "ok": True,
                    "skipped": True,
                    "reason": "flatten_all_has_failures",
                    "targets": targets,
                }
            ]
        try:
            payload = update_control(
                config.control_file,
                "clear-flatten",
                reason=reason,
                audit_path=config.control_audit_file,
                actor="autopilot",
                enforce_flatten_pause=True,
                expected_control=expected_control,
            )
        except Exception as exc:
            results.append(
                {
                    "command": "clear-flatten",
                    "name": None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "targets": targets,
                }
            )
        else:
            results.append(
                {
                    "command": "clear-flatten",
                    "name": None,
                    "ok": True,
                    "paused": payload.get("paused"),
                    "flatten_all": payload.get("flatten_all"),
                    "flatten_products": payload.get("flatten_products", []),
                    "targets": targets,
                }
            )
        return results

    requested_products = set(control.get("flatten_products", []))
    for item in flatten_results:
        name = item.get("product_name")
        if name not in requested_products or not item.get("ok"):
            continue
        try:
            payload = update_control(
                config.control_file,
                "clear-flatten",
                name=name,
                reason=reason,
                audit_path=config.control_audit_file,
                actor="autopilot",
                enforce_flatten_pause=True,
                expected_control=expected_control,
            )
        except Exception as exc:
            results.append(
                {
                    "command": "clear-flatten",
                    "name": name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            expected_control = payload
            results.append(
                {
                    "command": "clear-flatten",
                    "name": name,
                    "ok": True,
                    "paused_products": payload.get("paused_products", []),
                    "flatten_all": payload.get("flatten_all"),
                    "flatten_products": payload.get("flatten_products", []),
                }
            )
    return results


def run_once(config: AutopilotConfig, *, run_jobs: bool = True) -> dict[str, Any]:
    errors = validate_config(config, validate_jobs=run_jobs)
    if errors:
        return {"ok": False, "errors": errors, "products": []}

    previous_status = _previous_status(config.status_file)
    control = load_control(config.control_file)
    report: dict[str, Any] = {
        "ok": True,
        "control": control,
        "job_config_errors": list(config.job_config_errors),
        "data_update": None,
        "jobs": [],
        "products": [],
        "active_income_portfolio": _active_income_portfolio_status(config),
    }
    if config.job_config_errors:
        # Continue product management, but make the isolated job failure loud
        # in the supervisor heartbeat and alert stream.
        report["ok"] = False
    if control.get("control_error"):
        report["ok"] = False
        report["control_error"] = control["control_error"]
    elif unknown_selectors := unknown_control_selectors(control, config):
        control["paused"] = True
        control["pause_jobs"] = True
        control["reason"] = "unknown_control_selector"
        control["control_error"] = "unknown control selectors: " + json.dumps(
            unknown_selectors, sort_keys=True
        )
        report["ok"] = False
        report["control_error"] = control["control_error"]
        report["unknown_control_selectors"] = unknown_selectors

    flatten_results: list[dict[str, Any]] = []
    for product in config.products:
        if should_flatten_product(control, product.name):
            try:
                product_status = flatten_product_once(product)
            except (ValueError, RuntimeError, OSError) as exc:
                LOGGER.exception("Product flatten failed: %s", product.name)
                product_status = {
                    "product": _product_to_status(product),
                    "action": "flatten",
                    "ok": False,
                    "error": str(exc),
                }
            report["products"].append(product_status)
            report["ok"] = report["ok"] and bool(product_status.get("ok"))
            flatten_results.append(
                {
                    "product_name": product.name,
                    "ok": bool(product_status.get("ok")),
                    "skipped": product_status.get("skipped"),
                    "reason": product_status.get("reason"),
                    "error": product_status.get("error") or product_status.get("close_error"),
                }
            )
            continue
        if is_product_paused(control, product.name):
            if not _local_state_requires_management(product):
                report["products"].append(
                    {
                        "product": _product_to_status(product),
                        "ok": True,
                        "skipped": True,
                        "reason": "paused",
                    }
                )
                continue
            try:
                management_kwargs: dict[str, Any] = {
                    "approval_ledger": config.approval_ledger,
                    "allow_entries": False,
                }
                if product.execution_mode == "live":
                    management_kwargs["config"] = config
                product_status = run_product_once(product, **management_kwargs)
                product_status["paused"] = True
            except (
                ApprovalError,
                FileNotFoundError,
                StrategyPolicyError,
                ValueError,
                RuntimeError,
            ) as exc:
                LOGGER.exception("Paused product risk-management cycle failed: %s", product.name)
                product_status = {
                    "product": _product_to_status(product),
                    "ok": False,
                    "paused": True,
                    "entries_allowed": False,
                    "error": str(exc),
                }
            report["products"].append(product_status)
            report["ok"] = report["ok"] and bool(product_status.get("ok"))
            continue
        try:
            product_kwargs: dict[str, Any] = {
                "approval_ledger": config.approval_ledger,
            }
            portfolio_gate: dict[str, Any] | None = None
            if product.objective == "active_income":
                portfolio_gate = _active_income_portfolio_status(config)
                if not portfolio_gate["entry_capacity_available"]:
                    product_kwargs["allow_entries"] = False
            if product.execution_mode == "live":
                product_kwargs["config"] = config
            product_status = run_product_once(product, **product_kwargs)
            if portfolio_gate is not None:
                product_status["portfolio_entry_gate"] = portfolio_gate
        except (
            ApprovalError,
            FileNotFoundError,
            StrategyPolicyError,
            ValueError,
            RuntimeError,
        ) as exc:
            LOGGER.exception("Product cycle failed: %s", product.name)
            product_status = {
                "product": _product_to_status(product),
                "ok": False,
                "error": str(exc),
            }
        report["products"].append(product_status)
        report["ok"] = report["ok"] and bool(product_status.get("ok"))
        if product.objective == "active_income":
            report["active_income_portfolio"] = _active_income_portfolio_status(config)

    if config.alerts_enabled and config.position_change_alerts_enabled:
        position_alerts = _emit_position_change_alerts(config, report["products"])
        if position_alerts:
            report["position_alerts"] = position_alerts

    # Product supervision and emergency flattening always run before bounded
    # maintenance/research work.  A slow data job must never postpone the first
    # risk-management pass of a cycle.
    if run_jobs and config.run_data_update and not control.get("paused"):
        report["data_update"] = run_data_update()
        report["ok"] = report["ok"] and bool(report["data_update"]["ok"])

    if run_jobs and not control.get("paused"):
        try:
            paused_jobs = {job.name for job in config.jobs if is_job_paused(control, job.name)}
            report["jobs"] = run_due_jobs(
                config.jobs,
                config.job_state_file,
                paused_jobs=paused_jobs,
                max_jobs_per_cycle=config.max_jobs_per_cycle,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.exception("Autopilot jobs failed")
            report["jobs"] = [
                {
                    "name": "scheduler",
                    "ok": False,
                    "error": str(exc),
                    "state_file": str(config.job_state_file),
                }
            ]
            report["ok"] = False
        for job_result in report["jobs"]:
            report["ok"] = report["ok"] and bool(job_result.get("ok"))

    control_clear = _auto_clear_successful_flatten_requests(config, control, flatten_results)
    if control_clear:
        report["control_clear"] = control_clear
        report["ok"] = report["ok"] and all(bool(item.get("ok")) for item in control_clear)

    if config.alerts_enabled and not report["ok"]:
        incident_key = _cycle_failure_dedupe_key(report)
        previous_incident_key = (
            _cycle_failure_dedupe_key(previous_status)
            if previous_status.get("ok") is False
            else None
        )
        if previous_incident_key == incident_key:
            report["alert"] = {
                "sent": False,
                "reason": "unchanged_incident",
                "incident_key": incident_key,
            }
        else:
            report["alert"] = _emit_runtime_alert(
                config,
                severity="error",
                title="autopilot cycle failed",
                detail=failure_detail(report),
                cooldown_seconds=0,
                dedupe_key=incident_key,
            )
    elif config.alerts_enabled and previous_status.get("ok") is False:
        previous_incident_key = _cycle_failure_dedupe_key(previous_status)
        report["recovery_alert"] = _emit_runtime_alert(
            config,
            severity="info",
            title="autopilot cycle recovered",
            detail={
                "cleared_incident": previous_incident_key,
                "operator_action_required": False,
            },
            cooldown_seconds=0,
            dedupe_key=f"cycle-recovered:{previous_incident_key}:{utc_now()}",
        )
    write_status(config.status_file, report)
    if config.auto_report_enabled:
        try:
            report["reporting"] = write_cycle_reports(config)
            reports_need_refresh = False
            if (
                config.alerts_enabled
                and config.advisory_alerts_enabled
                and _report_json_available(
                    report.get("reporting", {}),
                    "readiness_report_json",
                    config.readiness_report_json_file,
                )
            ):
                readiness_report = json.loads(
                    config.readiness_report_json_file.read_text(encoding="utf-8")
                )
                readiness_detail = readiness_warning_detail(readiness_report)
                if readiness_detail["warnings"]:
                    report["readiness_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot readiness warnings",
                        detail=readiness_detail,
                    )
                    reports_need_refresh = True
            if config.alerts_enabled and _report_json_available(
                report.get("reporting", {}),
                "operator_report_json",
                config.operator_report_json_file,
            ):
                operator_report = json.loads(
                    config.operator_report_json_file.read_text(encoding="utf-8")
                )
                research_handoff_detail = research_handoff_warning_detail(operator_report)
                if config.advisory_alerts_enabled and research_handoff_detail["warnings"]:
                    report["research_handoff_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot research handoff warnings",
                        detail=research_handoff_detail,
                    )
                    reports_need_refresh = True
                research_progress_detail = research_progress_warning_detail(operator_report)
                if config.advisory_alerts_enabled and research_progress_detail["warnings"]:
                    report["research_progress_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot research progress warnings",
                        detail=research_progress_detail,
                    )
                    reports_need_refresh = True
                testnet_rehearsal_detail = required_testnet_rehearsal_warning_detail(
                    operator_report
                )
                if config.advisory_alerts_enabled and testnet_rehearsal_detail["warnings"]:
                    report["testnet_rehearsal_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot testnet rehearsal warnings",
                        detail=testnet_rehearsal_detail,
                    )
                    reports_need_refresh = True
                promotion_detail = promotion_warning_detail(operator_report)
                if config.advisory_alerts_enabled and promotion_detail["warnings"]:
                    report["promotion_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot promotion review warnings",
                        detail=promotion_detail,
                    )
                    reports_need_refresh = True
                if config.daily_digest_enabled:
                    digest_period = int(time.time() // config.daily_digest_cadence_seconds)
                    digest_alert = _emit_runtime_alert(
                        config,
                        severity="info",
                        title="autopilot daily digest",
                        detail=_daily_digest_detail(operator_report),
                        cooldown_seconds=config.daily_digest_cadence_seconds * 2,
                        dedupe_key=f"daily-digest:{digest_period}",
                    )
                    report["daily_digest_alert"] = digest_alert
                    if digest_alert.get("sent"):
                        reports_need_refresh = True
            if reports_need_refresh:
                write_status(config.status_file, report)
                report["reporting"] = write_cycle_reports(config)
        except Exception as exc:  # reporting must never crash trading supervision
            LOGGER.exception("Failed to write cycle reports")
            report["reporting"] = {"ok": False, "error": str(exc)}
        write_status(config.status_file, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight trading autopilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--once", action="store_true", help="Run one orchestration cycle and exit.")
    parser.add_argument("--validate", action="store_true", help="Validate config and exit.")
    parser.add_argument(
        "--skip-jobs",
        action="store_true",
        help="Run trading supervision only; use src.autopilot.job_worker for scheduled jobs.",
    )
    parser.add_argument("--sleep", type=int, default=None, help="Override loop sleep seconds.")
    return parser.parse_args()


def _effective_sleep_seconds(config: AutopilotConfig, override: int | None) -> int:
    sleep_seconds = override if override is not None else config.loop_sleep_seconds
    if sleep_seconds <= 0:
        raise ValueError("sleep seconds must be positive")
    return sleep_seconds


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config, strict_jobs=not args.skip_jobs)
    errors = validate_config(
        config,
        require_core_products=True,
        require_core_jobs=not args.skip_jobs,
        verify_job_imports=not args.skip_jobs,
        validate_jobs=not args.skip_jobs,
    )
    if args.validate:
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"valid: {args.config}")
        return
    if errors:
        raise SystemExit("\n".join(errors))
    try:
        sleep_seconds = _effective_sleep_seconds(config, args.sleep)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        with acquire_runtime_lock(config.lock_file):
            while True:
                report = run_once(config, run_jobs=not args.skip_jobs)
                print(
                    json.dumps(
                        {"ok": report["ok"], "status_file": str(config.status_file)}, sort_keys=True
                    )
                )
                if args.once:
                    if not report["ok"]:
                        raise SystemExit(1)
                    return
                time.sleep(sleep_seconds)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
