"""24/7 orchestration loop for the trading system."""

from __future__ import annotations

import argparse
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
PREFLIGHT_PRODUCT_KEYS = (
    "objective",
    "base_asset",
    "market",
    "symbol",
    "execution_mode",
    "starting_equity",
    "regime_guard",
    "regime_mayer_top",
)
PREFLIGHT_REQUIRED_CHECKS = (
    "product_config",
    "execution_engine_identity",
    "strategy_artifact_exists",
    "strategy_fingerprints",
    "strategy_policy",
    "approval_gate",
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
        "strategy_id",
        "symbol",
        "side",
        "order_type",
        "client_id",
        "broker_account_fingerprint",
        "qty",
        "quote_budget",
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
    "market_data_update_futures",
    "market_data_update_futures_1m",
    "market_data_update_spot",
    "regime_tag_futures_15m",
    "research_synthetic_smoke",
    "research_factory",
    "research_cycle",
    "candidate_paper_cycle",
    "strategy_framework_smoke",
    "active_income_promotion_review",
    "btc_accumulation_promotion_review",
    "runtime_maintenance",
    "runtime_backup",
    "artifact_hygiene",
)
REQUIRED_CORE_JOB_MODULES = {
    "market_data_update_futures": "src.autopilot.history_bootstrap",
    "market_data_update_futures_1m": "src.autopilot.history_bootstrap",
    "market_data_update_spot": "src.autopilot.history_bootstrap",
    "regime_tag_futures_15m": "src.regime",
    "research_synthetic_smoke": "src.autopilot.research_smoke",
    "research_factory": "src.autopilot.research_factory",
    "research_cycle": "src.autopilot.research_cycle",
    "candidate_paper_cycle": "src.autopilot.candidate_paper",
    "strategy_framework_smoke": "src.autopilot.strategy_smoke",
    "active_income_promotion_review": "src.autopilot.promotion",
    "btc_accumulation_promotion_review": "src.autopilot.promotion",
    "runtime_maintenance": "src.autopilot.maintenance",
    "runtime_backup": "src.autopilot.backup",
    "artifact_hygiene": "src.autopilot.artifact_hygiene",
}
REQUIRED_CORE_JOB_FLAG_VALUES = {
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
    "candidate_paper_cycle": {
        "--config": ("config/autopilot.json",),
        "--output": ("runtime/candidate_paper_status.json",),
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
    "runtime_backup": {
        "--config": ("config/autopilot.json",),
        "--report": ("runtime/backup_report.json",),
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
    if str(detail.get("symbol") or "").upper() != product.symbol.upper():
        raise RuntimeError(f"{product.name}: {label} open-order evidence symbol is invalid.")
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
    reported_product = matched.get("product") or {}
    if not isinstance(reported_product, dict):
        raise RuntimeError(
            f"{product.name}: preflight report product payload must be a JSON object."
        )
    for key in PREFLIGHT_PRODUCT_KEYS:
        expected = str(getattr(product, key))
        actual = str(reported_product.get(key, ""))
        if key in {"base_asset", "symbol"}:
            expected = expected.upper()
            actual = actual.upper()
        if actual != expected:
            raise RuntimeError(
                f"{product.name}: preflight product {key} mismatch: {actual or '<missing>'} != {expected}."
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
    reported_product = matched.get("product") or {}
    if not isinstance(reported_product, dict):
        raise RuntimeError(
            f"{product.name}: testnet rehearsal preflight product payload must be a JSON object."
        )
    for key in PREFLIGHT_PRODUCT_KEYS:
        expected = str(getattr(product, key))
        actual = str(reported_product.get(key, ""))
        if key in {"base_asset", "symbol"}:
            expected = expected.upper()
            actual = actual.upper()
        if actual != expected:
            raise RuntimeError(
                f"{product.name}: testnet rehearsal preflight product {key} mismatch: "
                f"{actual or '<missing>'} != {expected}."
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
    if state_error:
        raise RuntimeError(f"{product.name}: cannot recover frozen strategy state: {state_error}")
    assert state is not None
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
    state: dict[str, Any],
    intent: dict[str, Any],
    *,
    balance_evidence: dict[str, Any],
    auto_finalized: bool,
    fill: Fill | None,
) -> dict[str, Any]:
    if state.get("flatten_intent") != intent:
        raise RuntimeError("durable flatten_intent changed before local-state commit")
    if balance_evidence.get("proven") is not True:
        raise RuntimeError("spot buyback balance postcondition is not proven")
    updated = dict(state)
    updated["open_positions"] = {}
    updated.pop("flatten_intent", None)
    cleared_pending_order = updated.pop("pending_order", None)
    last_flatten: dict[str, Any] = {
        "at": utc_now(),
        "reason": "autopilot_control",
        "recovered_state": False,
        "cleared_pending_order": cleared_pending_order is not None,
        "flatten_client_id": intent["client_id"],
        "auto_finalized": auto_finalized,
        "balance_evidence": balance_evidence,
    }
    if isinstance(cleared_pending_order, dict):
        last_flatten["pending_client_id"] = cleared_pending_order.get("client_id")
    if fill is not None:
        last_flatten["fill"] = {
            "symbol": fill.symbol,
            "side": _fill_side_value(fill),
            "qty": fill.qty,
            "price": fill.price,
            "fee": fill.fee,
            "timestamp": fill.timestamp,
        }
    updated["last_flatten"] = last_flatten
    write_json_atomic(product.state_file, updated)
    return {
        "path": str(product.state_file),
        "recovered": False,
        "cleared": True,
        "error": None,
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
    if state_error:
        raise RuntimeError(
            f"{product.name}: cannot inspect native stops before flatten: {state_error}"
        )
    assert state is not None
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


def _assert_futures_flatten_account_binding(
    product: ProductConfig,
    broker: Any,
) -> str:
    """Bind panic-flatten reads and writes to the durable production account."""

    state, state_error = _read_local_state_for_flatten(product)
    if state_error:
        raise RuntimeError(
            f"{product.name}: cannot verify account identity before futures flatten: {state_error}"
        )
    assert state is not None
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
    if state_error:
        status.update(
            ok=False,
            reason="invalid_local_state",
            local_state={"path": str(product.state_file), "error": state_error},
        )
        return status
    assert state is not None
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
        if not balance_evidence["proven"]:
            return unresolved(
                "existing flatten_intent cannot be proven filled from the broker BTC balance; "
                "refusing duplicate buyback",
                intent=existing_intent,
                balance_evidence=balance_evidence,
            )
        try:
            local_state = _commit_spot_flatten_state(
                product,
                state,
                existing_intent,
                balance_evidence=balance_evidence,
                auto_finalized=True,
                fill=None,
            )
        except Exception as exc:
            return unresolved(
                f"buyback is proven but local-state commit failed: {type(exc).__name__}: {exc}",
                intent=existing_intent,
                balance_evidence=balance_evidence,
            )
        status.update(
            ok=True,
            flattened=True,
            auto_finalized=True,
            reason="flatten_intent_auto_finalized",
            local_state=local_state,
            position_after={
                "symbol": current.symbol,
                "qty": current.qty,
                "avg_price": current.avg_price,
            },
        )
        return status

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
            state,
            intent,
            balance_evidence=balance_evidence,
            auto_finalized=False,
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
    status["broker_account_fingerprint"] = account_fingerprint
    before = broker.get_position(product.symbol)
    status.update(
        broker=broker.name,
        position_before={"symbol": before.symbol, "qty": before.qty, "avg_price": before.avg_price},
    )
    if before.is_flat:
        status.update(flattened=False, reason="already_flat")
        return _finalize_futures_flatten_state(product, broker, status)

    try:
        fill = broker.close_position(product.symbol)
        if fill is not None:
            _assert_futures_flatten_fill_valid(product, before, fill)
    except Exception as exc:
        status["close_error"] = f"{type(exc).__name__}: {exc}"
        try:
            after = broker.get_position(product.symbol)
            status["position_after_attempt"] = {
                "symbol": after.symbol,
                "qty": after.qty,
                "avg_price": after.avg_price,
                "is_flat": after.is_flat,
            }
        except Exception as readback_exc:
            status["position_after_attempt_error"] = (
                f"{type(readback_exc).__name__}: {readback_exc}"
            )
        status["ok"] = False
        return status
    after = broker.get_position(product.symbol)
    status.update(
        flattened=fill is not None,
        fill=None
        if fill is None
        else {
            "symbol": fill.symbol,
            "side": fill.side.value,
            "qty": fill.qty,
            "price": fill.price,
            "fee": fill.fee,
            "timestamp": fill.timestamp,
        },
        position_after={"symbol": after.symbol, "qty": after.qty, "avg_price": after.avg_price},
    )
    if not after.is_flat:
        status["ok"] = False
        status["error"] = (
            f"{product.name}: flatten order sent but broker position is not flat: {after.qty}"
        )
        return status
    return _finalize_futures_flatten_state(product, broker, status)


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
) -> dict[str, Any]:
    try:
        return emit_alert(
            alert_file=config.alert_file,
            state_file=config.alert_state_file,
            severity=severity,
            title=title,
            detail=detail,
            cooldown_seconds=config.alert_cooldown_seconds,
            webhook_url_env=config.webhook_url_env,
        )
    except Exception as exc:  # alerting must never crash trading supervision
        LOGGER.exception("Failed to emit autopilot alert: %s", title)
        return {"sent": False, "error": str(exc)}


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

    control = load_control(config.control_file)
    report: dict[str, Any] = {
        "ok": True,
        "control": control,
        "job_config_errors": list(config.job_config_errors),
        "data_update": None,
        "jobs": [],
        "products": [],
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
            if product.execution_mode == "live":
                product_kwargs["config"] = config
            product_status = run_product_once(product, **product_kwargs)
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
        report["alert"] = _emit_runtime_alert(
            config,
            severity="error",
            title="autopilot cycle failed",
            detail=failure_detail(report),
        )
    write_status(config.status_file, report)
    if config.auto_report_enabled:
        try:
            report["reporting"] = write_cycle_reports(config)
            reports_need_refresh = False
            if config.alerts_enabled and _report_json_available(
                report.get("reporting", {}),
                "readiness_report_json",
                config.readiness_report_json_file,
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
                if research_handoff_detail["warnings"]:
                    report["research_handoff_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot research handoff warnings",
                        detail=research_handoff_detail,
                    )
                    reports_need_refresh = True
                research_progress_detail = research_progress_warning_detail(operator_report)
                if research_progress_detail["warnings"]:
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
                if testnet_rehearsal_detail["warnings"]:
                    report["testnet_rehearsal_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot testnet rehearsal warnings",
                        detail=testnet_rehearsal_detail,
                    )
                    reports_need_refresh = True
                promotion_detail = promotion_warning_detail(operator_report)
                if promotion_detail["warnings"]:
                    report["promotion_alert"] = _emit_runtime_alert(
                        config,
                        severity="warning",
                        title="autopilot promotion review warnings",
                        detail=promotion_detail,
                    )
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
