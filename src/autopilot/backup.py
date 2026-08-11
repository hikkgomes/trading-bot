"""Create small recovery backups of autopilot runtime state."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path
from posixpath import normpath
from typing import Any

from src.autopilot.candidate_activation import candidate_path_for_product
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, load_config
from src.autopilot.experiment_memory import ExperimentMemory
from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT

DEFAULT_BACKUP_DIR = PROJECT_ROOT / "runtime" / "backups"
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
BACKUP_ARCHIVE_PATTERN = "autopilot_state_*.zip"
SUPPORTED_MANIFEST_VERSION = 1
JOB_PATH_FLAGS = (
    "--artifact",
    "--input",
    "--json-output",
    "--journal",
    "--markdown-output",
    "--mutation-batch",
    "--output",
    "--output-json",
    "--output-md",
    "--report",
    "--state",
)


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _add_path(paths: list[Path], seen: set[Path], path: Path | None) -> None:
    if path is None:
        return
    normalized = path.resolve(strict=False)
    if normalized in seen:
        return
    seen.add(normalized)
    paths.append(path)


def _job_command_path(job, flag: str) -> Path | None:
    command = list(job.command)
    try:
        index = command.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    value = str(command[value_index])
    if not value or value.startswith("--"):
        return None
    path = Path(value)
    return path if path.is_absolute() else job.working_dir / path


def _uses_project_artifacts(config_path: Path) -> bool:
    try:
        config_path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return False
    return True


def configured_backup_paths(config: AutopilotConfig, *, config_path: Path) -> list[Path]:
    """Return the small files needed to recover/review autopilot state."""

    paths: list[Path] = []
    seen: set[Path] = set()
    include_project_artifacts = _uses_project_artifacts(config_path)
    for path in (
        config_path,
        config.approval_ledger,
        config.control_file,
        config.control_audit_file,
        config.status_file,
        config.trade_starvation_history_file,
        config.trade_starvation_report_file,
        config.job_state_file,
        config.alert_file,
        config.alert_state_file,
        config.research_smoke_file,
        config.strategy_smoke_file,
        config.research_cycle_file,
        config.research_factory_config_file,
        config.generated_batch_file,
        config.experiment_memory_backup_file,
        config.incubation_candidates_file,
        config.mutation_plan_file,
        config.mutation_batch_file,
        config.artifact_hygiene_file,
        config.backup_report_file,
        config.operator_report_file,
        config.operator_report_json_file,
        config.readiness_report_file,
        config.readiness_report_json_file,
        PROJECT_ROOT / "config" / "event_capture.json" if include_project_artifacts else None,
        config.event_capture_status_file,
        PROJECT_ROOT / "config" / "microstructure_research.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "microstructure.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "config" / "ml_research.json" if include_project_artifacts else None,
        PROJECT_ROOT / "config" / "portfolio_risk.json" if include_project_artifacts else None,
        config.portfolio_risk_file,
        PROJECT_ROOT / "config" / "relative_value.json" if include_project_artifacts else None,
        PROJECT_ROOT / "runtime" / "research" / "relative_value.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "relative_value_paper.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "relative_value_paper_state.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "ml_research.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "ml_research_state.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper_state.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "exploration_paper" / "manifest.json"
        if include_project_artifacts
        else None,
        PROJECT_ROOT / "runtime" / "exploration_paper" / "status.json"
        if include_project_artifacts
        else None,
    ):
        _add_path(paths, seen, path)
    if include_project_artifacts:
        for path in sorted(
            (PROJECT_ROOT / "runtime" / "research" / "ml_candidates").glob("*.json")
        ):
            _add_path(paths, seen, path)
    if config.candidate_paper_enabled:
        _add_path(paths, seen, config.candidate_paper_status_file)
    for product in config.products:
        candidate_path = candidate_path_for_product(product.name)
        for path in (
            product.strategies_path,
            candidate_path,
            candidate_path.parent / f"{product.name}_paper_trades.csv",
            candidate_path.parent / f"{product.name}_promotion_review.json",
            candidate_path.parent / f"{product.name}_promotion_review.md",
            product.state_file,
            product.trade_log,
            product.preflight_report,
            product.testnet_rehearsal_report,
        ):
            _add_path(paths, seen, path)
        for state_path in sorted(candidate_path.parent.glob(f"{product.name}_paper_state_*.json")):
            _add_path(paths, seen, state_path)
    for job in config.jobs:
        for flag in JOB_PATH_FLAGS:
            _add_path(paths, seen, _job_command_path(job, flag))
    return paths


def _configured_backup_roles(
    config: AutopilotConfig,
    *,
    config_path: Path,
) -> dict[Path, str]:
    roles: dict[Path, str] = {}
    include_project_artifacts = _uses_project_artifacts(config_path)

    def add(path: Path | None, role: str) -> None:
        if path is not None:
            roles.setdefault(path.resolve(strict=False), role)

    for path, role in (
        (config_path, "autopilot_config"),
        (config.trade_starvation_history_file, "trade_starvation_history"),
        (config.trade_starvation_report_file, "trade_starvation_report"),
        (config.approval_ledger, "approval_ledger"),
        (config.control_file, "operator_control"),
        (config.control_audit_file, "operator_control_audit"),
        (config.status_file, "runtime_status"),
        (config.job_state_file, "scheduled_job_state"),
        (config.alert_file, "alert_log"),
        (config.alert_state_file, "alert_cooldown_state"),
        (config.research_smoke_file, "research_smoke"),
        (config.strategy_smoke_file, "strategy_smoke"),
        (config.research_cycle_file, "research_cycle"),
        (config.research_factory_config_file, "research_factory_config"),
        (config.generated_batch_file, "generated_research_batch"),
        (config.experiment_memory_backup_file, "experiment_memory_snapshot"),
        (config.incubation_candidates_file, "incubation_candidates"),
        (config.mutation_plan_file, "mutation_plan"),
        (config.mutation_batch_file, "mutation_batch"),
        (config.artifact_hygiene_file, "artifact_hygiene"),
        (config.backup_report_file, "previous_backup_report"),
        (config.operator_report_file, "operator_report_markdown"),
        (config.operator_report_json_file, "operator_report_json"),
        (config.readiness_report_file, "readiness_report_markdown"),
        (config.readiness_report_json_file, "readiness_report_json"),
        (
            PROJECT_ROOT / "config" / "event_capture.json" if include_project_artifacts else None,
            "event_capture_config",
        ),
        (config.event_capture_status_file, "event_capture_status"),
        (
            PROJECT_ROOT / "config" / "microstructure_research.json"
            if include_project_artifacts
            else None,
            "microstructure_research_config",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "microstructure.json"
            if include_project_artifacts
            else None,
            "microstructure_research_report",
        ),
        (
            PROJECT_ROOT / "config" / "ml_research.json" if include_project_artifacts else None,
            "ml_research_config",
        ),
        (
            PROJECT_ROOT / "config" / "portfolio_risk.json" if include_project_artifacts else None,
            "portfolio_risk_config",
        ),
        (config.portfolio_risk_file, "portfolio_risk_model"),
        (
            PROJECT_ROOT / "config" / "relative_value.json" if include_project_artifacts else None,
            "relative_value_config",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "relative_value.json"
            if include_project_artifacts
            else None,
            "relative_value_report",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "relative_value_paper.json"
            if include_project_artifacts
            else None,
            "relative_value_paper_report",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "relative_value_paper_state.json"
            if include_project_artifacts
            else None,
            "relative_value_paper_state",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "ml_research.json"
            if include_project_artifacts
            else None,
            "ml_research_report",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "ml_research_state.json"
            if include_project_artifacts
            else None,
            "ml_research_state",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper.json"
            if include_project_artifacts
            else None,
            "ml_forward_paper_report",
        ),
        (
            PROJECT_ROOT / "runtime" / "research" / "ml_forward_paper_state.json"
            if include_project_artifacts
            else None,
            "ml_forward_paper_state",
        ),
        (
            PROJECT_ROOT / "runtime" / "exploration_paper" / "manifest.json"
            if include_project_artifacts
            else None,
            "exploration_paper_manifest",
        ),
        (
            PROJECT_ROOT / "runtime" / "exploration_paper" / "status.json"
            if include_project_artifacts
            else None,
            "exploration_paper_status",
        ),
    ):
        add(path, role)
    if include_project_artifacts:
        for path in sorted(
            (PROJECT_ROOT / "runtime" / "research" / "ml_candidates").glob("*.json")
        ):
            add(path, "ml_reviewable_candidate")
    if config.candidate_paper_enabled:
        add(config.candidate_paper_status_file, "candidate_paper_status")
    for product in config.products:
        candidate_path = candidate_path_for_product(product.name)
        for path, field in (
            (product.strategies_path, "strategy_artifact"),
            (candidate_path, "staged_candidate"),
            (candidate_path.parent / f"{product.name}_paper_trades.csv", "candidate_paper_log"),
            (
                candidate_path.parent / f"{product.name}_promotion_review.json",
                "candidate_promotion_review_json",
            ),
            (
                candidate_path.parent / f"{product.name}_promotion_review.md",
                "candidate_promotion_review_markdown",
            ),
            (product.state_file, "product_state"),
            (product.trade_log, "product_trade_log"),
            (product.preflight_report, "preflight_report"),
            (product.testnet_rehearsal_report, "testnet_rehearsal_report"),
        ):
            add(path, f"product:{product.name}:{field}")
        for state_path in sorted(candidate_path.parent.glob(f"{product.name}_paper_state_*.json")):
            add(state_path, f"product:{product.name}:candidate_paper_state")
    for job in config.jobs:
        for flag in JOB_PATH_FLAGS:
            add(_job_command_path(job, flag), f"job:{job.name}:{flag.removeprefix('--')}")
    return roles


def _arcname(path: Path, root: Path) -> str:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        digest = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        return f"external/{digest}/{resolved_path.name}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_private_archive(path: Path):
    """Open a zip destination without exposing it through the caller's umask."""

    flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"backup output must be a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w+b")
    except BaseException:
        os.close(descriptor)
        raise


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"restore directory must not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"restore directory must be a directory: {path}")
    path.chmod(0o700)


def _write_private_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"restore target must be a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short restore write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot_experiment_memory(config: AutopilotConfig) -> dict[str, Any]:
    """Create and deeply validate a stable SQLite recovery image.

    The live database is never added to the zip directly. SQLite's online
    backup API captures one transactionally consistent image even while the
    research worker is active.
    """

    source = config.experiment_memory_file
    destination = config.experiment_memory_backup_file
    status: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "source_exists": source.exists(),
        "existing_snapshot": destination.exists(),
        "refreshed": False,
    }
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("experiment memory and its backup must not be symlinks")
    if source.resolve(strict=False) == destination.resolve(strict=False):
        raise ValueError("experiment memory backup path must differ from the live database")
    if not source.exists():
        if destination.exists():
            with ExperimentMemory(destination, deep_on_open=False) as snapshot:
                status["snapshot_integrity"] = snapshot.integrity_check(deep=True)
            status.update(
                reason="live_memory_missing_existing_snapshot_retained",
                size_bytes=destination.stat().st_size,
                sha256=_sha256_file(destination),
            )
        else:
            status["reason"] = "live_memory_not_initialized"
        return status
    with ExperimentMemory(source, deep_on_open=False) as memory:
        status["source_integrity"] = memory.integrity_check(deep=True)
        memory.backup_to(destination)
    with ExperimentMemory(destination, deep_on_open=False) as snapshot:
        status["snapshot_integrity"] = snapshot.integrity_check(deep=True)
    status.update(
        refreshed=True,
        existing_snapshot=True,
        size_bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
    )
    return status


def build_backup_archive(
    *,
    config_path: Path,
    output: Path | None = None,
    extra_paths: list[Path] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    config = load_config(config_path)
    memory_snapshot = _snapshot_experiment_memory(config)
    output = output or (DEFAULT_BACKUP_DIR / f"autopilot_state_{_utc_stamp()}.zip")
    if output.is_symlink():
        raise ValueError(f"backup output must not be a symlink: {output}")
    paths = configured_backup_paths(config, config_path=config_path)
    configured_roles = _configured_backup_roles(config, config_path=config_path)
    configured_path_count = len(paths)
    seen_paths = {path.resolve(strict=False) for path in paths}
    for path in extra_paths or []:
        _add_path(paths, seen_paths, path)

    manifest_entries: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with _open_private_archive(output) as archive_handle:
        with zipfile.ZipFile(
            archive_handle,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path_index, path in enumerate(paths):
                configured_recovery_file = path_index < configured_path_count
                observed_exists = path.exists() or path.is_symlink()
                entry: dict[str, Any] = {
                    "path": str(path),
                    "arcname": _arcname(path, root),
                    "exists": observed_exists,
                    "included": False,
                    "role": configured_roles.get(
                        path.resolve(strict=False),
                        "operator_extra"
                        if not configured_recovery_file
                        else "configured_recovery_file",
                    ),
                    "required_if_present": configured_recovery_file,
                }
                if path.is_symlink():
                    entry["reason"] = "symlink"
                elif not path.exists():
                    entry["reason"] = "missing"
                    entry["optional_missing"] = True
                elif not path.is_file():
                    entry["reason"] = "not_file"
                else:
                    size = path.stat().st_size
                    entry["size_bytes"] = size
                    if size > max_file_bytes:
                        entry["reason"] = "too_large"
                        entry["max_file_bytes"] = max_file_bytes
                    else:
                        entry["sha256"] = _sha256_file(path)
                        archive.write(path, entry["arcname"])
                        entry["included"] = True
                manifest_entries.append(entry)
            manifest = {
                "version": 1,
                "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                "config": str(config_path),
                "max_file_bytes": max_file_bytes,
                "experiment_memory_snapshot": memory_snapshot,
                "files": manifest_entries,
                "required_recovery_files": sum(
                    1 for item in manifest_entries if item.get("required_if_present") is True
                ),
                "required_recovery_roles": sorted(
                    str(item["role"])
                    for item in manifest_entries
                    if item.get("required_if_present") is True
                ),
                "included_files": sum(1 for item in manifest_entries if item.get("included")),
                "missing_files": sum(
                    1 for item in manifest_entries if item.get("reason") == "missing"
                ),
                "skipped_files": sum(
                    1
                    for item in manifest_entries
                    if item.get("exists") and not item.get("included")
                ),
                "optional_missing_files": sum(
                    1 for item in manifest_entries if item.get("optional_missing") is True
                ),
                "critical_skipped_files": sum(
                    1
                    for item in manifest_entries
                    if item.get("required_if_present") is True
                    and item.get("exists") is True
                    and item.get("included") is not True
                ),
            }
            archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive_handle.flush()
        os.fsync(archive_handle.fileno())
    report = {
        "ok": False,
        "output": str(output),
        "manifest": manifest,
        "archive_size_bytes": output.stat().st_size,
    }
    report["verification"] = verify_backup_archive(output)
    report["ok"] = report["verification"].get("ok") is True
    return report


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_backup_archive(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "checked_files": 0,
        "issues": issues,
    }
    if not path.exists():
        issues.append({"code": "missing_archive", "message": "backup archive does not exist"})
        return report
    if not path.is_file():
        issues.append({"code": "not_file", "message": "backup archive path is not a file"})
        return report
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                issues.append({"code": "zip_crc_failed", "member": bad_member})
            archive_names = archive.namelist()
            names = set(archive_names)
            duplicate_archive_members = sorted(
                name for name in names if archive_names.count(name) > 1
            )
            for name in duplicate_archive_members:
                issues.append({"code": "duplicate_archive_member", "arcname": name})
            if "MANIFEST.json" not in names:
                issues.append({"code": "missing_manifest", "message": "MANIFEST.json is missing"})
                report["ok"] = False
                return report
            try:
                manifest = json.loads(archive.read("MANIFEST.json"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                issues.append({"code": "invalid_manifest", "message": str(exc)})
                return report
            files = manifest.get("files")
            if not isinstance(files, list):
                issues.append(
                    {"code": "invalid_manifest_files", "message": "manifest files must be a list"}
                )
                return report
            version = manifest.get("version")
            if version != SUPPORTED_MANIFEST_VERSION:
                issues.append(
                    {
                        "code": "unsupported_manifest_version",
                        "version": version,
                        "supported_version": SUPPORTED_MANIFEST_VERSION,
                    }
                )
            included_entries = [
                item for item in files if isinstance(item, dict) and item.get("included")
            ]
            required_entries = [
                item
                for item in files
                if isinstance(item, dict) and item.get("required_if_present") is True
            ]
            for item in files:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("required_if_present") is True
                    and item.get("exists") is True
                    and item.get("included") is not True
                ):
                    issues.append(
                        {
                            "code": "required_recovery_file_skipped",
                            "path": item.get("path"),
                            "role": item.get("role"),
                            "reason": item.get("reason"),
                        }
                    )
            memory_snapshot = manifest.get("experiment_memory_snapshot")
            if isinstance(memory_snapshot, dict):
                report["experiment_memory_snapshot"] = memory_snapshot
                snapshot_entries = [
                    item
                    for item in files
                    if isinstance(item, dict) and item.get("role") == "experiment_memory_snapshot"
                ]
                if len(snapshot_entries) > 1:
                    issues.append(
                        {
                            "code": "duplicate_memory_snapshot_entries",
                            "count": len(snapshot_entries),
                        }
                    )
                if memory_snapshot.get("source_exists") is True:
                    if memory_snapshot.get("refreshed") is not True:
                        issues.append({"code": "memory_snapshot_not_refreshed"})
                if (
                    memory_snapshot.get("source_exists") is True
                    or memory_snapshot.get("existing_snapshot") is True
                ):
                    integrity = memory_snapshot.get("snapshot_integrity")
                    if not isinstance(integrity, dict) or integrity.get("ok") is not True:
                        issues.append({"code": "memory_snapshot_integrity_missing"})
                    included_snapshots = [
                        item for item in snapshot_entries if item.get("included") is True
                    ]
                    if len(included_snapshots) != 1:
                        issues.append(
                            {
                                "code": "required_memory_snapshot_missing",
                                "included": len(included_snapshots),
                            }
                        )
                    elif included_snapshots[0].get("sha256") != memory_snapshot.get("sha256"):
                        issues.append({"code": "memory_snapshot_manifest_hash_mismatch"})
            report["manifest"] = {
                "version": manifest.get("version"),
                "generated_at": manifest.get("generated_at"),
                "included_files": manifest.get("included_files"),
                "missing_files": manifest.get("missing_files"),
                "skipped_files": manifest.get("skipped_files"),
                "optional_missing_files": manifest.get("optional_missing_files"),
                "critical_skipped_files": manifest.get("critical_skipped_files"),
                "required_recovery_files": manifest.get("required_recovery_files"),
                "required_recovery_roles": manifest.get("required_recovery_roles"),
            }
            if manifest.get("included_files") != len(included_entries):
                issues.append(
                    {
                        "code": "included_count_mismatch",
                        "manifest_included_files": manifest.get("included_files"),
                        "actual_included_files": len(included_entries),
                    }
                )
            actual_counts = {
                "missing_files": sum(
                    1
                    for item in files
                    if isinstance(item, dict) and item.get("reason") == "missing"
                ),
                "skipped_files": sum(
                    1
                    for item in files
                    if isinstance(item, dict)
                    and item.get("exists") is True
                    and item.get("included") is not True
                ),
                "optional_missing_files": sum(
                    1
                    for item in files
                    if isinstance(item, dict) and item.get("optional_missing") is True
                ),
                "critical_skipped_files": sum(
                    1
                    for item in required_entries
                    if item.get("exists") is True and item.get("included") is not True
                ),
            }
            for field, actual_count in actual_counts.items():
                if field in manifest and manifest.get(field) != actual_count:
                    issues.append(
                        {
                            "code": f"{field.removesuffix('_files')}_count_mismatch",
                            "manifest_count": manifest.get(field),
                            "actual_count": actual_count,
                        }
                    )
            declared_required_count = manifest.get("required_recovery_files")
            if declared_required_count is not None and declared_required_count != len(
                required_entries
            ):
                issues.append(
                    {
                        "code": "required_recovery_count_mismatch",
                        "manifest_count": declared_required_count,
                        "actual_count": len(required_entries),
                    }
                )
            declared_required_roles = manifest.get("required_recovery_roles")
            if declared_required_roles is not None:
                if not isinstance(declared_required_roles, list) or any(
                    not isinstance(role, str) or not role for role in declared_required_roles
                ):
                    issues.append({"code": "invalid_required_recovery_roles"})
                else:
                    actual_required_roles = [
                        str(item.get("role") or "") for item in required_entries
                    ]
                    duplicate_declared_roles = sorted(
                        role
                        for role in set(declared_required_roles)
                        if declared_required_roles.count(role) > 1
                    )
                    duplicate_actual_roles = sorted(
                        role
                        for role in set(actual_required_roles)
                        if actual_required_roles.count(role) > 1
                    )
                    if duplicate_declared_roles or duplicate_actual_roles:
                        issues.append(
                            {
                                "code": "duplicate_required_recovery_roles",
                                "declared": duplicate_declared_roles,
                                "actual": duplicate_actual_roles,
                            }
                        )
                    missing_roles = sorted(
                        set(declared_required_roles) - set(actual_required_roles)
                    )
                    unexpected_roles = sorted(
                        set(actual_required_roles) - set(declared_required_roles)
                    )
                    if missing_roles or unexpected_roles:
                        issues.append(
                            {
                                "code": "required_recovery_roles_mismatch",
                                "missing_roles": missing_roles,
                                "unexpected_roles": unexpected_roles,
                            }
                        )
            expected_archive_members = {
                item.get("arcname")
                for item in included_entries
                if isinstance(item.get("arcname"), str) and item.get("arcname")
            }
            expected_archive_members.add("MANIFEST.json")
            for name in sorted(names - expected_archive_members):
                issues.append({"code": "unexpected_member", "arcname": name})
            seen_arcnames: set[str] = set()
            for item in included_entries:
                arcname = item.get("arcname")
                if not isinstance(arcname, str) or not arcname:
                    issues.append({"code": "missing_arcname", "path": item.get("path")})
                    continue
                try:
                    _safe_member_path(Path("."), arcname)
                except ValueError:
                    issues.append({"code": "unsafe_arcname", "arcname": arcname})
                    continue
                if arcname in seen_arcnames:
                    issues.append({"code": "duplicate_manifest_arcname", "arcname": arcname})
                    continue
                seen_arcnames.add(arcname)
                if arcname not in names:
                    issues.append({"code": "missing_member", "arcname": arcname})
                    continue
                payload = archive.read(arcname)
                report["checked_files"] += 1
                expected_sha = item.get("sha256")
                actual_sha = _sha256_bytes(payload)
                if expected_sha != actual_sha:
                    issues.append(
                        {
                            "code": "sha256_mismatch",
                            "arcname": arcname,
                            "expected": expected_sha,
                            "actual": actual_sha,
                        }
                    )
                expected_size = item.get("size_bytes")
                if expected_size is not None and expected_size != len(payload):
                    issues.append(
                        {
                            "code": "size_mismatch",
                            "arcname": arcname,
                            "expected": expected_size,
                            "actual": len(payload),
                        }
                    )
    except zipfile.BadZipFile as exc:
        issues.append({"code": "bad_zip", "message": str(exc)})
    report["ok"] = not issues
    return report


def _safe_member_path(restore_dir: Path, arcname: str) -> Path:
    normalized = normpath(arcname)
    if normalized in {"", ".", ".."} or normalized.startswith("../") or normalized.startswith("/"):
        raise ValueError(f"unsafe archive member path: {arcname}")
    return restore_dir / normalized


def _safe_restore_target(restore_dir: Path, arcname: str) -> Path:
    target = _safe_member_path(restore_dir, arcname)
    resolved_root = restore_dir.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"restore target escapes restore dir: {arcname}") from exc
    return target


def restore_backup_archive(
    path: Path, restore_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    verification = verify_backup_archive(path)
    if not verification.get("ok"):
        raise ValueError(f"backup verification failed: {path}")
    if restore_dir.exists() and restore_dir.is_symlink():
        raise ValueError(f"restore_dir must not be a symlink: {restore_dir}")
    _ensure_private_directory(restore_dir)
    restored: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        included_entries = [
            item
            for item in manifest.get("files", [])
            if isinstance(item, dict) and item.get("included")
        ]
        planned: list[tuple[dict[str, Any], Path]] = []
        for item in included_entries:
            arcname = str(item.get("arcname") or "")
            target = _safe_restore_target(restore_dir, arcname)
            if target.is_symlink():
                raise ValueError(f"restore target is a symlink: {arcname}")
            planned.append((item, target))
            if target.exists() and not overwrite:
                conflicts.append({"arcname": arcname, "target": str(target)})
        if conflicts:
            return {
                "ok": False,
                "archive": str(path),
                "restore_dir": str(restore_dir),
                "verification": verification,
                "restored_files": 0,
                "conflicts": conflicts,
                "reason": "target_exists",
            }
        for item, target in planned:
            _ensure_private_directory(target.parent)
            payload = archive.read(str(item["arcname"]))
            _write_private_file(target, payload)
            restored.append(
                {
                    "arcname": item["arcname"],
                    "target": str(target),
                    "size_bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
    report = {
        "ok": True,
        "archive": str(path),
        "restore_dir": str(restore_dir),
        "verification": verification,
        "restored_files": len(restored),
        "files": restored,
        "overwrite": overwrite,
    }
    write_json_atomic(restore_dir / "RESTORE_REPORT.json", report)
    (restore_dir / "RESTORE_REPORT.json").chmod(0o600)
    return report


def prune_backup_archives(backup_dir: Path, *, keep: int, dry_run: bool = False) -> dict[str, Any]:
    if keep <= 0:
        raise ValueError("keep must be positive")
    archives = sorted(
        (path for path in backup_dir.glob(BACKUP_ARCHIVE_PATTERN) if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    deleted = []
    for path in archives[keep:]:
        item = {"path": str(path), "size_bytes": path.stat().st_size}
        deleted.append(item)
        if not dry_run:
            path.unlink()
    return {
        "path": str(backup_dir),
        "keep": keep,
        "archives": len(archives),
        "deleted_archives": len(deleted),
        "deleted_bytes": sum(int(item["size_bytes"]) for item in deleted),
        "deleted": deleted,
        "dry_run": dry_run,
        "changed": bool(deleted) and not dry_run,
    }


def backup_output_summary(report: dict[str, Any]) -> dict[str, Any]:
    manifest = report.get("manifest") or {}
    return {
        "ok": bool(report.get("ok")),
        "output": report.get("output"),
        "archive_size_bytes": report.get("archive_size_bytes"),
        "included_files": manifest.get("included_files"),
        "missing_files": manifest.get("missing_files"),
        "skipped_files": manifest.get("skipped_files"),
        "optional_missing_files": manifest.get("optional_missing_files"),
        "critical_skipped_files": manifest.get("critical_skipped_files"),
        "required_recovery_files": manifest.get("required_recovery_files"),
        "experiment_memory_snapshot": manifest.get("experiment_memory_snapshot"),
        "retention": report.get("retention"),
        "verification": {
            "ok": (report.get("verification") or {}).get("ok"),
            "checked_files": (report.get("verification") or {}).get("checked_files"),
            "issues": len((report.get("verification") or {}).get("issues") or []),
        }
        if report.get("verification")
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small autopilot recovery backup zip.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, help="Output zip path. Defaults to runtime/backups timestamped zip."
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--verify", type=Path, help="Verify an existing backup zip instead of creating one."
    )
    parser.add_argument("--restore", type=Path, help="Verify and extract an existing backup zip.")
    parser.add_argument(
        "--restore-dir", type=Path, help="Directory where --restore should extract files."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow --restore to overwrite existing files."
    )
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument(
        "--max-backups", type=int, help="Keep only the newest N generated backup zip files."
    )
    parser.add_argument(
        "--extra", type=Path, action="append", default=[], help="Additional small file to include."
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print the full manifest instead of a compact summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        report = verify_backup_archive(args.verify)
        if args.report:
            write_json_atomic(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)
    if args.restore:
        if args.restore_dir is None:
            raise SystemExit("--restore-dir is required with --restore")
        try:
            report = restore_backup_archive(
                args.restore, args.restore_dir, overwrite=args.overwrite
            )
        except Exception as exc:
            report = {
                "ok": False,
                "archive": str(args.restore),
                "restore_dir": str(args.restore_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
        if args.report:
            write_json_atomic(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["ok"] else 1)
    report = build_backup_archive(
        config_path=args.config,
        output=args.output,
        extra_paths=list(args.extra),
        max_file_bytes=args.max_file_bytes,
    )
    if args.max_backups is not None and report["ok"]:
        report["retention"] = prune_backup_archives(
            Path(report["output"]).parent, keep=args.max_backups
        )
    if args.report:
        write_json_atomic(args.report, report)
    output = report if args.full_output else backup_output_summary(report)
    print(json.dumps(output, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
