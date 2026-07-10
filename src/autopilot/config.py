"""JSON configuration for the lightweight 24/7 autopilot."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


class DuplicateConfigKeyError(ValueError):
    """Raised when a JSON object repeats a key while loading autopilot config."""


class NonStandardConfigConstantError(ValueError):
    """Raised when config JSON uses non-standard numeric constants."""


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "autopilot.json"
DEFAULT_CONTROL_FILE = PROJECT_ROOT / "runtime" / "control.json"
DEFAULT_CONTROL_AUDIT_FILE = PROJECT_ROOT / "runtime" / "control_audit.jsonl"
DEFAULT_STATUS_FILE = PROJECT_ROOT / "runtime" / "status.json"
DEFAULT_LOCK_FILE = PROJECT_ROOT / "runtime" / "autopilot.lock"
DEFAULT_APPROVAL_LEDGER = PROJECT_ROOT / "runtime" / "approvals.json"
DEFAULT_JOB_STATE_FILE = PROJECT_ROOT / "runtime" / "job_state.json"
DEFAULT_ALERT_FILE = PROJECT_ROOT / "runtime" / "alerts.jsonl"
DEFAULT_ALERT_STATE_FILE = PROJECT_ROOT / "runtime" / "alert_state.json"
DEFAULT_RESEARCH_SMOKE_FILE = PROJECT_ROOT / "runtime" / "research_smoke.json"
DEFAULT_STRATEGY_SMOKE_FILE = PROJECT_ROOT / "runtime" / "strategy_framework_smoke.json"
DEFAULT_RESEARCH_CYCLE_FILE = PROJECT_ROOT / "runtime" / "research_cycle.json"
DEFAULT_RESEARCH_FACTORY_CONFIG_FILE = PROJECT_ROOT / "config" / "research_factory.json"
DEFAULT_GENERATED_BATCH_FILE = PROJECT_ROOT / "runtime" / "research" / "generated_hypotheses.json"
DEFAULT_EXPERIMENT_MEMORY_FILE = PROJECT_ROOT / "runtime" / "research" / "experiment_memory.sqlite3"
DEFAULT_EXPERIMENT_MEMORY_BACKUP_FILE = (
    PROJECT_ROOT / "runtime" / "research" / "experiment_memory.backup.sqlite3"
)
DEFAULT_INCUBATION_CANDIDATES_FILE = PROJECT_ROOT / "runtime" / "incubation_candidates.json"
DEFAULT_MUTATION_PLAN_FILE = PROJECT_ROOT / "runtime" / "mutation_plan.json"
DEFAULT_MUTATION_BATCH_FILE = PROJECT_ROOT / "runtime" / "mutation_hypotheses.json"
DEFAULT_ARTIFACT_HYGIENE_FILE = PROJECT_ROOT / "runtime" / "artifact_hygiene.json"
DEFAULT_BACKUP_REPORT_FILE = PROJECT_ROOT / "runtime" / "backup_report.json"
DEFAULT_OPERATOR_REPORT_FILE = PROJECT_ROOT / "runtime" / "operator_report.md"
DEFAULT_OPERATOR_REPORT_JSON_FILE = PROJECT_ROOT / "runtime" / "operator_report.json"
DEFAULT_READINESS_REPORT_FILE = PROJECT_ROOT / "runtime" / "readiness_report.md"
DEFAULT_READINESS_REPORT_JSON_FILE = PROJECT_ROOT / "runtime" / "readiness_report.json"
DEFAULT_MIN_RUNTIME_FREE_BYTES = 512 * 1024 * 1024
JOB_CONFIG_KEYS = {
    "cadence_seconds",
    "command",
    "enabled",
    "name",
    "timeout_seconds",
    "working_dir",
}
PRODUCT_CONFIG_KEYS = {
    "base_asset",
    "enabled",
    "execution_mode",
    "market",
    "name",
    "objective",
    "preflight_max_age_seconds",
    "preflight_report",
    "regime_guard",
    "regime_mayer_top",
    "require_preflight",
    "require_testnet_rehearsal",
    "starting_equity",
    "state_file",
    "strategies_path",
    "symbol",
    "testnet_rehearsal_max_age_seconds",
    "testnet_rehearsal_report",
    "trade_log",
}
AUTOPILOT_CONFIG_KEYS = {
    "alert_cooldown_seconds",
    "alert_file",
    "alert_state_file",
    "alerts_enabled",
    "approval_ledger",
    "artifact_hygiene_file",
    "auto_report_enabled",
    "backup_report_file",
    "control_audit_file",
    "control_file",
    "experiment_memory_backup_file",
    "experiment_memory_file",
    "generated_batch_file",
    "incubation_candidates_file",
    "job_state_file",
    "jobs",
    "lock_file",
    "loop_sleep_seconds",
    "max_consecutive_job_deferrals",
    "max_jobs_per_cycle",
    "min_runtime_free_bytes",
    "mutation_batch_file",
    "mutation_plan_file",
    "operator_report_file",
    "operator_report_json_file",
    "products",
    "readiness_report_file",
    "readiness_report_json_file",
    "research_cycle_file",
    "research_factory_config_file",
    "research_smoke_file",
    "run_data_update",
    "status_file",
    "strategy_smoke_file",
    "webhook_url_env",
}


@dataclass
class JobConfig:
    name: str
    enabled: bool
    command: list[str]
    cadence_seconds: int
    timeout_seconds: int = 600
    working_dir: Path = PROJECT_ROOT

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobConfig:
        name = _required_non_empty_str(payload, "name", label="job")
        _reject_unknown_keys(payload, allowed=JOB_CONFIG_KEYS, label=f"job {name}")
        command = payload.get("command", [])
        if isinstance(command, str):
            raise ValueError(f"job {name}: command must be a list, not a string")
        if not isinstance(command, list):
            raise ValueError(f"job {name}: command must be a list")
        if not command:
            raise ValueError(f"job {name}: command cannot be empty")
        command_parts = [_non_empty_str(part, f"job {name}: command[{index}]") for index, part in enumerate(command)]
        cadence_seconds = _positive_int(payload.get("cadence_seconds"), f"job {name}: cadence_seconds")
        timeout_seconds = _positive_int(payload.get("timeout_seconds", 600), f"job {name}: timeout_seconds")
        return cls(
            name=name,
            enabled=_json_bool(payload, "enabled", default=True, field=f"job {name}: enabled"),
            command=command_parts,
            cadence_seconds=cadence_seconds,
            timeout_seconds=timeout_seconds,
            working_dir=_optional_project_path(
                payload,
                "working_dir",
                default=PROJECT_ROOT,
                field=f"job {name}: working_dir",
            ),
        )


@dataclass
class ProductConfig:
    name: str
    enabled: bool
    objective: str
    base_asset: str
    market: str
    execution_mode: str
    symbol: str
    strategies_path: Path
    state_file: Path
    trade_log: Path
    starting_equity: float
    regime_guard: bool = False
    regime_mayer_top: float = 2.4
    require_preflight: bool = True
    preflight_report: Path | None = None
    preflight_max_age_seconds: int = 3600
    require_testnet_rehearsal: bool = False
    testnet_rehearsal_report: Path | None = None
    testnet_rehearsal_max_age_seconds: int = 30 * 24 * 60 * 60

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProductConfig:
        name = _required_non_empty_str(payload, "name", label="product")
        _reject_unknown_keys(payload, allowed=PRODUCT_CONFIG_KEYS, label=f"product {name}")
        return cls(
            name=name,
            enabled=_json_bool(payload, "enabled", default=True, field=f"product {name}: enabled"),
            objective=_required_non_empty_str(payload, "objective", label=f"product {name}"),
            base_asset=_required_non_empty_str(payload, "base_asset", label=f"product {name}"),
            market=_required_non_empty_str(payload, "market", label=f"product {name}"),
            execution_mode=_optional_non_empty_str(
                payload,
                "execution_mode",
                default="paper",
                field=f"product {name}: execution_mode",
            ),
            symbol=_optional_non_empty_str(payload, "symbol", default="BTCUSDT", field=f"product {name}: symbol"),
            strategies_path=_required_project_path(payload, "strategies_path", label=f"product {name}"),
            state_file=_required_project_path(payload, "state_file", label=f"product {name}"),
            trade_log=_required_project_path(payload, "trade_log", label=f"product {name}"),
            starting_equity=_positive_float(
                payload.get("starting_equity", 1000.0),
                f"product {name}: starting_equity",
            ),
            regime_guard=_json_bool(payload, "regime_guard", default=False, field=f"product {name}: regime_guard"),
            regime_mayer_top=_positive_float(
                payload.get("regime_mayer_top", 2.4),
                f"product {name}: regime_mayer_top",
            ),
            require_preflight=_json_bool(
                payload,
                "require_preflight",
                default=True,
                field=f"product {name}: require_preflight",
            ),
            preflight_report=_optional_project_path(
                payload,
                "preflight_report",
                default=f"runtime/{name}_preflight_report.json",
                field=f"product {name}: preflight_report",
            ),
            preflight_max_age_seconds=_positive_int(
                payload.get("preflight_max_age_seconds", 3600),
                f"product {name}: preflight_max_age_seconds",
            ),
            require_testnet_rehearsal=_json_bool(
                payload,
                "require_testnet_rehearsal",
                default=False,
                field=f"product {name}: require_testnet_rehearsal",
            ),
            testnet_rehearsal_report=_optional_project_path(
                payload,
                "testnet_rehearsal_report",
                default="runtime/testnet_rehearsal_report.json",
                field=f"product {name}: testnet_rehearsal_report",
            ),
            testnet_rehearsal_max_age_seconds=_positive_int(
                payload.get("testnet_rehearsal_max_age_seconds", 30 * 24 * 60 * 60),
                f"product {name}: testnet_rehearsal_max_age_seconds",
            ),
        )


@dataclass
class AutopilotConfig:
    control_file: Path = DEFAULT_CONTROL_FILE
    control_audit_file: Path = DEFAULT_CONTROL_AUDIT_FILE
    status_file: Path = DEFAULT_STATUS_FILE
    lock_file: Path = DEFAULT_LOCK_FILE
    approval_ledger: Path = DEFAULT_APPROVAL_LEDGER
    job_state_file: Path = DEFAULT_JOB_STATE_FILE
    alert_file: Path = DEFAULT_ALERT_FILE
    alert_state_file: Path = DEFAULT_ALERT_STATE_FILE
    research_smoke_file: Path = DEFAULT_RESEARCH_SMOKE_FILE
    strategy_smoke_file: Path = DEFAULT_STRATEGY_SMOKE_FILE
    research_cycle_file: Path = DEFAULT_RESEARCH_CYCLE_FILE
    research_factory_config_file: Path = DEFAULT_RESEARCH_FACTORY_CONFIG_FILE
    generated_batch_file: Path = DEFAULT_GENERATED_BATCH_FILE
    experiment_memory_file: Path = DEFAULT_EXPERIMENT_MEMORY_FILE
    experiment_memory_backup_file: Path = DEFAULT_EXPERIMENT_MEMORY_BACKUP_FILE
    incubation_candidates_file: Path = DEFAULT_INCUBATION_CANDIDATES_FILE
    mutation_plan_file: Path = DEFAULT_MUTATION_PLAN_FILE
    mutation_batch_file: Path = DEFAULT_MUTATION_BATCH_FILE
    artifact_hygiene_file: Path = DEFAULT_ARTIFACT_HYGIENE_FILE
    backup_report_file: Path = DEFAULT_BACKUP_REPORT_FILE
    operator_report_file: Path = DEFAULT_OPERATOR_REPORT_FILE
    operator_report_json_file: Path = DEFAULT_OPERATOR_REPORT_JSON_FILE
    readiness_report_file: Path = DEFAULT_READINESS_REPORT_FILE
    readiness_report_json_file: Path = DEFAULT_READINESS_REPORT_JSON_FILE
    auto_report_enabled: bool = False
    alerts_enabled: bool = True
    alert_cooldown_seconds: int = 900
    webhook_url_env: str = "AUTOPILOT_WEBHOOK_URL"
    min_runtime_free_bytes: int = DEFAULT_MIN_RUNTIME_FREE_BYTES
    loop_sleep_seconds: int = 60
    max_jobs_per_cycle: int = 1
    max_consecutive_job_deferrals: int = 3
    run_data_update: bool = False
    jobs: list[JobConfig] = field(default_factory=list)
    job_config_errors: list[str] = field(default_factory=list)
    products: list[ProductConfig] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        strict_jobs: bool = True,
    ) -> AutopilotConfig:
        _reject_unknown_keys(payload, allowed=AUTOPILOT_CONFIG_KEYS, label="autopilot config")
        raw_jobs = payload.get("jobs", [])
        jobs: list[JobConfig] = []
        job_config_errors: list[str] = []
        if strict_jobs:
            if not isinstance(raw_jobs, list):
                raise ValueError("autopilot config jobs must be a list")
            for index, job in enumerate(raw_jobs):
                if not isinstance(job, dict):
                    raise ValueError(f"autopilot config jobs[{index}] must be a JSON object")
            _reject_duplicate_names(raw_jobs, label="job")
            jobs = [JobConfig.from_dict(job) for job in raw_jobs]
        elif isinstance(raw_jobs, list):
            # Supervision never executes jobs, but retaining every independently
            # valid definition keeps normal status/report output complete. Bad
            # entries and duplicates remain fatal in the dedicated job worker.
            seen_job_names: set[str] = set()
            for index, raw_job in enumerate(raw_jobs):
                if not isinstance(raw_job, dict):
                    job_config_errors.append(
                        f"autopilot config jobs[{index}] must be a JSON object"
                    )
                    continue
                try:
                    job = JobConfig.from_dict(raw_job)
                except (TypeError, ValueError) as exc:
                    job_config_errors.append(f"jobs[{index}]: {exc}")
                    continue
                if job.name in seen_job_names:
                    job_config_errors.append(f"duplicate job name: {job.name}")
                    continue
                seen_job_names.add(job.name)
                jobs.append(job)
        else:
            job_config_errors.append("autopilot config jobs must be a list")
        products_payload = payload.get("products", [])
        if not isinstance(products_payload, list):
            raise ValueError("autopilot config products must be a list")
        for index, product in enumerate(products_payload):
            if not isinstance(product, dict):
                raise ValueError(f"autopilot config products[{index}] must be a JSON object")
        _reject_duplicate_names(products_payload, label="product")
        alert_cooldown_seconds = _non_negative_int(
            payload.get("alert_cooldown_seconds", 900),
            "alert_cooldown_seconds",
        )
        min_runtime_free_bytes = _positive_int(
            payload.get("min_runtime_free_bytes", DEFAULT_MIN_RUNTIME_FREE_BYTES),
            "min_runtime_free_bytes",
        )
        loop_sleep_seconds = _positive_int(payload.get("loop_sleep_seconds", 60), "loop_sleep_seconds")
        try:
            max_jobs_per_cycle = _positive_int(
                payload.get("max_jobs_per_cycle", 1),
                "max_jobs_per_cycle",
            )
        except ValueError:
            if strict_jobs:
                raise
            job_config_errors.append("max_jobs_per_cycle must be a positive JSON integer")
            max_jobs_per_cycle = 1
        try:
            max_consecutive_job_deferrals = _positive_int(
                payload.get("max_consecutive_job_deferrals", 3),
                "max_consecutive_job_deferrals",
            )
        except ValueError:
            if strict_jobs:
                raise
            job_config_errors.append(
                "max_consecutive_job_deferrals must be a positive JSON integer"
            )
            max_consecutive_job_deferrals = 3
        try:
            run_data_update = _json_bool(
                payload,
                "run_data_update",
                default=False,
                field="run_data_update",
            )
        except ValueError:
            if strict_jobs:
                raise
            job_config_errors.append("run_data_update must be a JSON boolean")
            run_data_update = False
        return cls(
            control_file=_optional_project_path(payload, "control_file", default=DEFAULT_CONTROL_FILE, field="control_file"),
            control_audit_file=_optional_project_path(
                payload,
                "control_audit_file",
                default=DEFAULT_CONTROL_AUDIT_FILE,
                field="control_audit_file",
            ),
            status_file=_optional_project_path(payload, "status_file", default=DEFAULT_STATUS_FILE, field="status_file"),
            lock_file=_optional_project_path(payload, "lock_file", default=DEFAULT_LOCK_FILE, field="lock_file"),
            approval_ledger=_optional_project_path(
                payload,
                "approval_ledger",
                default=DEFAULT_APPROVAL_LEDGER,
                field="approval_ledger",
            ),
            job_state_file=_optional_project_path(
                payload,
                "job_state_file",
                default=DEFAULT_JOB_STATE_FILE,
                field="job_state_file",
            ),
            alert_file=_optional_project_path(payload, "alert_file", default=DEFAULT_ALERT_FILE, field="alert_file"),
            alert_state_file=_optional_project_path(
                payload,
                "alert_state_file",
                default=DEFAULT_ALERT_STATE_FILE,
                field="alert_state_file",
            ),
            research_smoke_file=_optional_project_path(
                payload,
                "research_smoke_file",
                default=DEFAULT_RESEARCH_SMOKE_FILE,
                field="research_smoke_file",
            ),
            strategy_smoke_file=_optional_project_path(
                payload,
                "strategy_smoke_file",
                default=DEFAULT_STRATEGY_SMOKE_FILE,
                field="strategy_smoke_file",
            ),
            research_cycle_file=_optional_project_path(
                payload,
                "research_cycle_file",
                default=DEFAULT_RESEARCH_CYCLE_FILE,
                field="research_cycle_file",
            ),
            research_factory_config_file=_optional_project_path(
                payload,
                "research_factory_config_file",
                default=DEFAULT_RESEARCH_FACTORY_CONFIG_FILE,
                field="research_factory_config_file",
            ),
            generated_batch_file=_optional_project_path(
                payload,
                "generated_batch_file",
                default=DEFAULT_GENERATED_BATCH_FILE,
                field="generated_batch_file",
            ),
            experiment_memory_file=_optional_project_path(
                payload,
                "experiment_memory_file",
                default=DEFAULT_EXPERIMENT_MEMORY_FILE,
                field="experiment_memory_file",
            ),
            experiment_memory_backup_file=_optional_project_path(
                payload,
                "experiment_memory_backup_file",
                default=DEFAULT_EXPERIMENT_MEMORY_BACKUP_FILE,
                field="experiment_memory_backup_file",
            ),
            incubation_candidates_file=_optional_project_path(
                payload,
                "incubation_candidates_file",
                default=DEFAULT_INCUBATION_CANDIDATES_FILE,
                field="incubation_candidates_file",
            ),
            mutation_plan_file=_optional_project_path(
                payload,
                "mutation_plan_file",
                default=DEFAULT_MUTATION_PLAN_FILE,
                field="mutation_plan_file",
            ),
            mutation_batch_file=_optional_project_path(
                payload,
                "mutation_batch_file",
                default=DEFAULT_MUTATION_BATCH_FILE,
                field="mutation_batch_file",
            ),
            artifact_hygiene_file=_optional_project_path(
                payload,
                "artifact_hygiene_file",
                default=DEFAULT_ARTIFACT_HYGIENE_FILE,
                field="artifact_hygiene_file",
            ),
            backup_report_file=_optional_project_path(
                payload,
                "backup_report_file",
                default=DEFAULT_BACKUP_REPORT_FILE,
                field="backup_report_file",
            ),
            operator_report_file=_optional_project_path(
                payload,
                "operator_report_file",
                default=DEFAULT_OPERATOR_REPORT_FILE,
                field="operator_report_file",
            ),
            operator_report_json_file=_optional_project_path(
                payload,
                "operator_report_json_file",
                default=DEFAULT_OPERATOR_REPORT_JSON_FILE,
                field="operator_report_json_file",
            ),
            readiness_report_file=_optional_project_path(
                payload,
                "readiness_report_file",
                default=DEFAULT_READINESS_REPORT_FILE,
                field="readiness_report_file",
            ),
            readiness_report_json_file=_optional_project_path(
                payload,
                "readiness_report_json_file",
                default=DEFAULT_READINESS_REPORT_JSON_FILE,
                field="readiness_report_json_file",
            ),
            auto_report_enabled=_json_bool(
                payload,
                "auto_report_enabled",
                default=False,
                field="auto_report_enabled",
            ),
            alerts_enabled=_json_bool(payload, "alerts_enabled", default=True, field="alerts_enabled"),
            alert_cooldown_seconds=alert_cooldown_seconds,
            webhook_url_env=_optional_non_empty_str(
                payload,
                "webhook_url_env",
                default="AUTOPILOT_WEBHOOK_URL",
                field="webhook_url_env",
            ),
            min_runtime_free_bytes=min_runtime_free_bytes,
            loop_sleep_seconds=loop_sleep_seconds,
            max_jobs_per_cycle=max_jobs_per_cycle,
            max_consecutive_job_deferrals=max_consecutive_job_deferrals,
            run_data_update=run_data_update,
            jobs=jobs,
            job_config_errors=job_config_errors,
            products=[ProductConfig.from_dict(product) for product in products_payload],
        )


def _json_bool(payload: dict[str, Any], key: str, *, default: bool, field: str) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a JSON boolean")


def _non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _required_non_empty_str(payload: dict[str, Any], field: str, *, label: str) -> str:
    if field not in payload:
        raise ValueError(f"{label} must include {field}")
    return _non_empty_str(payload[field], f"{label} {field}")


def _optional_non_empty_str(payload: dict[str, Any], key: str, *, default: str, field: str) -> str:
    return _non_empty_str(payload.get(key, default), field)


def _required_project_path(payload: dict[str, Any], key: str, *, label: str) -> Path:
    if key not in payload:
        raise ValueError(f"{label} must include {key}")
    return _project_path(payload[key], f"{label}: {key}")


def _optional_project_path(payload: dict[str, Any], key: str, *, default: str | Path, field: str) -> Path:
    return _project_path(payload.get(key, default), field)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    parsed = value
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    parsed = value
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _reject_duplicate_names(items: list[dict[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    for item in items:
        name = item.get("name")
        if not isinstance(name, str):
            continue
        normalized = name.strip()
        if not normalized:
            continue
        if normalized in seen:
            raise ValueError(f"duplicate {label} name: {normalized}")
        seen.add(normalized)


def _reject_unknown_keys(payload: dict[str, Any], *, allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if not unknown:
        return
    if len(unknown) == 1:
        raise ValueError(f"{label} has unknown field: {unknown[0]}")
    raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _project_path(value: Any, field: str) -> Path:
    if isinstance(value, Path):
        path = value
    else:
        path = Path(_non_empty_str(value, field))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise DuplicateConfigKeyError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_non_standard_json_constant(value: str) -> None:
    raise NonStandardConfigConstantError(f"invalid JSON constant: {value}")


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    strict_jobs: bool = True,
) -> AutopilotConfig:
    """Load strict product/runtime config, optionally isolating scheduled jobs.

    The trading supervisor passes ``strict_jobs=False``. A malformed optional
    research command must stop the separate job worker and healthcheck, but it
    must not prevent management of existing exchange exposure.
    """
    if path.is_symlink():
        raise ValueError(f"autopilot config must not be a symlink: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Autopilot config not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_standard_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"autopilot config must be valid JSON: {path}: {exc}") from exc
    except DuplicateConfigKeyError as exc:
        raise ValueError(f"autopilot config must not contain duplicate JSON keys: {path}: {exc}") from exc
    except NonStandardConfigConstantError as exc:
        raise ValueError(f"autopilot config must be strict JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"autopilot config must be a JSON object: {path}")
    return AutopilotConfig.from_dict(payload, strict_jobs=strict_jobs)
