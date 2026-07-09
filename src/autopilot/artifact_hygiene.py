"""Report and optionally quarantine stale strategy artifacts.

This is intentionally conservative. The default mode is a dry-run report that
identifies invalid configured product artifacts and unreferenced
``outputs/active_strategies*.json`` files. ``--apply`` moves policy-blocked
configured artifacts into ``runtime/quarantine``. Add
``--quarantine-unreferenced-active`` to also move unreferenced active-strategy
artifacts. It never deletes files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now
from src.autopilot.strategy_policy import StrategyPolicyError, assert_strategy_artifact_allowed
from src.config import PROJECT_ROOT

DEFAULT_QUARANTINE_DIR = PROJECT_ROOT / "runtime" / "quarantine"


def _artifact_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary
    stat = path.stat()
    summary.update(
        size_bytes=stat.st_size,
        modified_ts=stat.st_mtime,
        modified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
        summary.update(
            readable=True,
            version=payload.get("version") if isinstance(payload, dict) else None,
            strategies=len(strategies) if isinstance(strategies, list) else None,
            generated_at=payload.get("generated_at") if isinstance(payload, dict) else None,
        )
    except Exception as exc:
        summary.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    return summary


def _quarantine_path(path: Path, quarantine_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return quarantine_dir / f"{path.stem}.{stamp}{path.suffix}"


def _move_to_quarantine(path: Path, quarantine_dir: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"refusing to quarantine symlink source: {path}")
    if quarantine_dir.exists() and quarantine_dir.is_symlink():
        raise ValueError(f"quarantine_dir must not be a symlink: {quarantine_dir}")
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = _quarantine_path(path, quarantine_dir)
    while os.path.lexists(target):
        target = target.with_name(f"{target.stem}.dup{target.suffix}")
    shutil.move(str(path), str(target))
    return target


def _hygiene_error(scope: str, path: Path | None, exc: Exception) -> dict[str, Any]:
    detail = {
        "scope": scope,
        "error": f"{type(exc).__name__}: {exc}",
    }
    if path is not None:
        detail["path"] = str(path)
    return detail


def inspect_product_artifact(
    product: ProductConfig,
    *,
    apply: bool = False,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
) -> dict[str, Any]:
    summary = _artifact_summary(product.strategies_path)
    report: dict[str, Any] = {
        "product": product.name,
        "objective": product.objective,
        "market": product.market,
        "mode": product.execution_mode,
        "artifact": summary,
        "ok": True,
        "action": "none",
    }
    if not product.strategies_path.exists():
        report.update(status="missing", reason="waiting_for_research_export")
        return report
    try:
        policy = assert_strategy_artifact_allowed(product)
    except (StrategyPolicyError, FileNotFoundError, json.JSONDecodeError) as exc:
        report.update(
            ok=False,
            status="policy_blocked",
            reason=str(exc),
            quarantine_candidate=True,
        )
        if apply and product.execution_mode != "live":
            target = _move_to_quarantine(product.strategies_path, quarantine_dir)
            report.update(action="quarantined", quarantined_to=str(target))
        elif apply and product.execution_mode == "live":
            report.update(action="not_quarantined_live_product")
        return report
    report.update(status="valid", policy=policy, quarantine_candidate=False)
    return report


def find_unreferenced_active_artifacts(config: AutopilotConfig, outputs_dir: Path) -> list[dict[str, Any]]:
    referenced = {product.strategies_path.resolve() for product in config.products}
    if not outputs_dir.exists():
        return []
    artifacts = []
    for path in sorted(outputs_dir.glob("active_strategies*.json")):
        if path.resolve() in referenced:
            continue
        artifacts.append(
            {
                **_artifact_summary(path),
                "status": "unreferenced_active_artifact",
                "action": "none",
            }
        )
    return artifacts


def quarantine_unreferenced_active_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
) -> list[dict[str, Any]]:
    updated = []
    for artifact in artifacts:
        row = dict(artifact)
        path = Path(str(row.get("path", "")))
        if path.exists():
            try:
                target = _move_to_quarantine(path, quarantine_dir)
                row.update(action="quarantined", quarantined_to=str(target))
            except Exception as exc:
                row.update(
                    ok=False,
                    action="error",
                    error=_hygiene_error("unreferenced_active_artifact", path, exc)["error"],
                )
        else:
            row.update(action="missing")
        updated.append(row)
    return updated


def find_search_outputs(outputs_dir: Path) -> list[dict[str, Any]]:
    if not outputs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_dir.iterdir()):
        if not path.is_dir():
            continue
        if not (
            path.name.startswith("search")
            or path.name.startswith("strategy_search")
            or path.name.endswith("_search")
            or path.name == "day_trade_search"
        ):
            continue
        files = [child for child in path.iterdir() if child.is_file()]
        latest_mtime = max((child.stat().st_mtime for child in files), default=path.stat().st_mtime)
        rows.append(
            {
                "path": str(path),
                "files": len(files),
                "modified_ts": latest_mtime,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(latest_mtime)),
                "status": "historical_search_output",
                "action": "none",
            }
        )
    return rows


def quarantine_historical_search_outputs(
    artifacts: list[dict[str, Any]],
    *,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
) -> list[dict[str, Any]]:
    updated = []
    for artifact in artifacts:
        row = dict(artifact)
        path = Path(str(row.get("path", "")))
        if path.exists():
            try:
                target = _move_to_quarantine(path, quarantine_dir)
                row.update(action="quarantined", quarantined_to=str(target))
            except Exception as exc:
                row.update(
                    ok=False,
                    action="error",
                    error=_hygiene_error("historical_search_output", path, exc)["error"],
                )
        else:
            row.update(action="missing")
        updated.append(row)
    return updated


def build_artifact_hygiene_report(
    config: AutopilotConfig,
    *,
    apply: bool = False,
    quarantine_unreferenced_active: bool = False,
    quarantine_historical_search: bool = False,
    outputs_dir: Path = PROJECT_ROOT / "outputs",
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    products = []
    for product in config.products:
        try:
            products.append(inspect_product_artifact(product, apply=apply, quarantine_dir=quarantine_dir))
        except Exception as exc:
            error = _hygiene_error("configured_product", product.strategies_path, exc)
            errors.append({**error, "product": product.name})
            products.append(
                {
                    "product": product.name,
                    "objective": product.objective,
                    "market": product.market,
                    "mode": product.execution_mode,
                    "artifact": _artifact_summary(product.strategies_path),
                    "ok": False,
                    "status": "error",
                    "action": "error",
                    "error": error["error"],
                }
            )
    unreferenced = find_unreferenced_active_artifacts(config, outputs_dir)
    if apply and quarantine_unreferenced_active:
        unreferenced = quarantine_unreferenced_active_artifacts(
            unreferenced,
            quarantine_dir=quarantine_dir,
        )
        errors.extend(
            {
                "scope": "unreferenced_active_artifact",
                "path": str(item.get("path", "")),
                "error": str(item["error"]),
            }
            for item in unreferenced
            if item.get("action") == "error" and item.get("error")
        )
    historical = find_search_outputs(outputs_dir)
    if apply and quarantine_historical_search:
        historical = quarantine_historical_search_outputs(
            historical,
            quarantine_dir=quarantine_dir,
        )
        errors.extend(
            {
                "scope": "historical_search_output",
                "path": str(item.get("path", "")),
                "error": str(item["error"]),
            }
            for item in historical
            if item.get("action") == "error" and item.get("error")
        )
    quarantine_candidates = [
        item for item in products if item.get("quarantine_candidate")
    ]
    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "dry_run": not apply,
        "outputs_dir": str(outputs_dir),
        "quarantine_dir": str(quarantine_dir),
        **({"errors": errors} if errors else {}),
        "configured_products": products,
        "unreferenced_active_artifacts": unreferenced,
        "historical_search_outputs": historical,
        "summary": {
            "configured_products": len(products),
            "quarantine_candidates": len(quarantine_candidates),
            "unreferenced_active_artifacts": len(unreferenced),
            "historical_search_outputs": len(historical),
            "errors": len(errors),
            "quarantined": sum(1 for item in products if item.get("action") == "quarantined")
            + sum(1 for item in unreferenced if item.get("action") == "quarantined")
            + sum(1 for item in historical if item.get("action") == "quarantined"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report or quarantine stale strategy artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("runtime/artifact_hygiene.json"))
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR)
    parser.add_argument("--apply", action="store_true", help="Move policy-blocked paper artifacts to quarantine.")
    parser.add_argument(
        "--quarantine-unreferenced-active",
        action="store_true",
        help="With --apply, also move unreferenced outputs/active_strategies*.json files to quarantine.",
    )
    parser.add_argument(
        "--quarantine-historical-search",
        action="store_true",
        help="With --apply, also move historical outputs/search* and outputs/*_search directories to quarantine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_artifact_hygiene_report(
        load_config(args.config),
        apply=args.apply,
        quarantine_unreferenced_active=args.quarantine_unreferenced_active,
        quarantine_historical_search=args.quarantine_historical_search,
        outputs_dir=args.outputs_dir,
        quarantine_dir=args.quarantine_dir,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
