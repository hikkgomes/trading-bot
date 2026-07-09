"""Small runtime maintenance tasks for the autopilot."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, load_config
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.config import PROJECT_ROOT

DEFAULT_EXPERIMENT_LOG = PROJECT_ROOT / "outputs" / "research_exploration" / "experiment_log.jsonl"
DEFAULT_EXPERIMENT_ARCHIVE_DIR = PROJECT_ROOT / "outputs" / "research_exploration" / "archive"
DEFAULT_CONTROL_AUDIT_ARCHIVE_DIR = PROJECT_ROOT / "runtime" / "archive"
DEFAULT_QUARANTINE_DIR = PROJECT_ROOT / "runtime" / "quarantine"


def _reject_symlink_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")


def compact_jsonl(path: Path, max_lines: int, dry_run: bool = False) -> dict[str, Any]:
    """Keep only the most recent JSONL records once a file exceeds max_lines."""
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    _reject_symlink_file(path, "jsonl path")
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "max_lines": max_lines,
        "dry_run": dry_run,
        "line_count": 0,
        "kept_lines": 0,
        "trimmed_lines": 0,
        "would_trim": False,
        "changed": False,
    }
    if not path.exists():
        return report

    lines = path.read_text(encoding="utf-8").splitlines()
    line_count = len(lines)
    kept = lines[-max_lines:]
    trimmed = max(0, line_count - len(kept))
    report.update(
        {
            "line_count": line_count,
            "kept_lines": len(kept),
            "trimmed_lines": trimmed,
            "would_trim": trimmed > 0,
            "changed": trimmed > 0 and not dry_run,
        }
    )
    if trimmed > 0 and not dry_run:
        write_text_atomic(path, "\n".join(kept) + "\n")
    return report


def rotate_jsonl(
    path: Path,
    *,
    max_lines: int,
    archive_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive older JSONL records to gzip and keep the hot file bounded."""
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    _reject_symlink_file(path, "jsonl path")
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "max_lines": max_lines,
        "archive_dir": str(archive_dir),
        "dry_run": dry_run,
        "line_count": 0,
        "kept_lines": 0,
        "archived_lines": 0,
        "archive_path": None,
        "would_rotate": False,
        "changed": False,
    }
    if not path.exists():
        return report

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        report.update(line_count=len(lines), kept_lines=len(lines))
        return report

    archived = lines[:-max_lines]
    kept = lines[-max_lines:]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_path = archive_dir / f"{path.stem}.{stamp}.jsonl.gz"
    suffix = 1
    while archive_path.exists() or archive_path.is_symlink():
        archive_path = archive_dir / f"{path.stem}.{stamp}.{suffix}.jsonl.gz"
        suffix += 1
    report.update(
        {
            "line_count": len(lines),
            "kept_lines": len(kept),
            "archived_lines": len(archived),
            "archive_path": str(archive_path),
            "would_rotate": True,
            "changed": not dry_run,
        }
    )
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive_path, "xt", encoding="utf-8") as handle:
            handle.write("\n".join(archived) + "\n")
        write_text_atomic(path, "\n".join(kept) + "\n")
    return report


def compact_alert_state(path: Path, max_fingerprints: int, dry_run: bool = False) -> dict[str, Any]:
    """Keep the newest alert cooldown fingerprints in the state JSON."""
    if max_fingerprints <= 0:
        raise ValueError("max_fingerprints must be positive")
    _reject_symlink_file(path, "alert state path")
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "max_fingerprints": max_fingerprints,
        "dry_run": dry_run,
        "alert_count": 0,
        "valid_alerts": 0,
        "invalid_alerts": 0,
        "kept_alerts": 0,
        "pruned_alerts": 0,
        "would_prune": False,
        "changed": False,
    }
    if not path.exists():
        return report

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"alert state must be a JSON object: {path}")
    alerts = payload.get("alerts")
    if not isinstance(alerts, dict):
        raise ValueError(f"alert state alerts must be an object: {path}")
    rows: list[tuple[str, dict[str, Any], float]] = []
    invalid_alerts = 0
    now = time.time()
    for fingerprint, raw_entry in alerts.items():
        if not isinstance(raw_entry, dict):
            invalid_alerts += 1
            continue
        try:
            last_sent_ts = float(raw_entry.get("last_sent_ts") or 0.0)
        except (TypeError, ValueError):
            invalid_alerts += 1
            continue
        if not math.isfinite(last_sent_ts) or last_sent_ts < 0 or last_sent_ts > now:
            invalid_alerts += 1
            continue
        rows.append((str(fingerprint), raw_entry, last_sent_ts))
    rows = sorted(
        rows,
        key=lambda item: item[2],
        reverse=True,
    )
    kept = {fingerprint: entry for fingerprint, entry, _last_sent_ts in rows[:max_fingerprints]}
    pruned = max(0, len(rows) - len(kept))
    changed = (pruned > 0 or invalid_alerts > 0) and not dry_run
    report.update(
        {
            "alert_count": len(alerts),
            "valid_alerts": len(rows),
            "invalid_alerts": invalid_alerts,
            "kept_alerts": len(kept),
            "pruned_alerts": pruned,
            "would_prune": pruned > 0 or invalid_alerts > 0,
            "changed": changed,
        }
    )
    if changed:
        compacted = dict(payload)
        compacted["alerts"] = kept
        write_json_atomic(path, compacted)
    return report


def _remove_empty_dirs(root: Path) -> int:
    removed = 0
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def prune_directory_by_size(
    root: Path,
    *,
    max_bytes: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete oldest files inside a generated directory until it fits max_bytes."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    report: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "max_bytes": max_bytes,
        "dry_run": dry_run,
        "initial_bytes": 0,
        "final_bytes": 0,
        "files": 0,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "deleted": [],
        "changed": False,
    }
    if not root.exists():
        return report
    if root.is_symlink():
        raise ValueError(f"root must not be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"root must be a directory: {root}")
    files = [path for path in root.rglob("*") if not path.is_symlink() and path.is_file()]
    rows = sorted(
        (
            {
                "path": path,
                "size_bytes": path.stat().st_size,
                "modified_ts": path.stat().st_mtime,
            }
            for path in files
        ),
        key=lambda item: (float(item["modified_ts"]), str(item["path"])),
    )
    total = sum(int(item["size_bytes"]) for item in rows)
    report.update(initial_bytes=total, final_bytes=total, files=len(rows))
    for item in rows:
        if total <= max_bytes:
            break
        path = Path(item["path"])
        size = int(item["size_bytes"])
        report["deleted"].append({"path": str(path), "size_bytes": size})
        report["deleted_files"] += 1
        report["deleted_bytes"] += size
        total -= size
        if not dry_run:
            path.unlink()
    report["final_bytes"] = total
    report["changed"] = bool(report["deleted_files"]) and not dry_run
    if report["changed"]:
        report["removed_empty_dirs"] = _remove_empty_dirs(root)
    return report


def _maintenance_error(task: str, exc: Exception) -> dict[str, Any]:
    return {
        "task": task,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _run_maintenance_task(
    report: dict[str, Any],
    task: str,
    func,
    *args,
    **kwargs,
) -> None:
    try:
        report[task] = func(*args, **kwargs)
    except Exception as exc:
        error = _maintenance_error(task, exc)
        report["ok"] = False
        report.setdefault("errors", []).append(error)
        report[task] = {"ok": False, **error}


def run_maintenance(
    config_path: Path,
    *,
    max_alert_lines: int,
    max_alert_fingerprints: int,
    max_experiment_lines: int,
    max_control_audit_lines: int,
    experiment_log: Path = DEFAULT_EXPERIMENT_LOG,
    experiment_archive_dir: Path = DEFAULT_EXPERIMENT_ARCHIVE_DIR,
    control_audit_archive_dir: Path = DEFAULT_CONTROL_AUDIT_ARCHIVE_DIR,
    max_quarantine_bytes: int | None = None,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    report: dict[str, Any] = {
        "ok": True,
        "config": str(config_path),
        "errors": [],
    }
    _run_maintenance_task(report, "alerts", compact_jsonl, config.alert_file, max_alert_lines, dry_run=dry_run)
    _run_maintenance_task(
        report,
        "alert_state",
        compact_alert_state,
        config.alert_state_file,
        max_alert_fingerprints,
        dry_run=dry_run,
    )
    _run_maintenance_task(
        report,
        "experiment_log",
        rotate_jsonl,
        experiment_log,
        max_lines=max_experiment_lines,
        archive_dir=experiment_archive_dir,
        dry_run=dry_run,
    )
    _run_maintenance_task(
        report,
        "control_audit",
        rotate_jsonl,
        config.control_audit_file,
        max_lines=max_control_audit_lines,
        archive_dir=control_audit_archive_dir,
        dry_run=dry_run,
    )
    if max_quarantine_bytes is not None:
        _run_maintenance_task(
            report,
            "quarantine",
            prune_directory_by_size,
            quarantine_dir,
            max_bytes=max_quarantine_bytes,
            dry_run=dry_run,
        )
    if not report["errors"]:
        report.pop("errors", None)
    return report


def run_alert_maintenance(config_path: Path, max_alert_lines: int, dry_run: bool = False) -> dict[str, Any]:
    return run_maintenance(
        config_path,
        max_alert_lines=max_alert_lines,
        max_alert_fingerprints=1000,
        max_experiment_lines=5000,
        max_control_audit_lines=5000,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded autopilot runtime maintenance tasks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-alert-lines", type=int, default=1000)
    parser.add_argument("--max-alert-fingerprints", type=int, default=1000)
    parser.add_argument("--max-experiment-lines", type=int, default=5000)
    parser.add_argument("--max-control-audit-lines", type=int, default=5000)
    parser.add_argument("--experiment-log", type=Path, default=DEFAULT_EXPERIMENT_LOG)
    parser.add_argument("--experiment-archive-dir", type=Path, default=DEFAULT_EXPERIMENT_ARCHIVE_DIR)
    parser.add_argument("--control-audit-archive-dir", type=Path, default=DEFAULT_CONTROL_AUDIT_ARCHIVE_DIR)
    parser.add_argument("--max-quarantine-bytes", type=int)
    parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_maintenance(
        args.config,
        max_alert_lines=args.max_alert_lines,
        max_alert_fingerprints=args.max_alert_fingerprints,
        max_experiment_lines=args.max_experiment_lines,
        max_control_audit_lines=args.max_control_audit_lines,
        experiment_log=args.experiment_log,
        experiment_archive_dir=args.experiment_archive_dir,
        control_audit_archive_dir=args.control_audit_archive_dir,
        max_quarantine_bytes=args.max_quarantine_bytes,
        quarantine_dir=args.quarantine_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
