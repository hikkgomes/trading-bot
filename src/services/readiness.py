"""Fail-closed readiness checks for the PostgreSQL-authoritative platform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select

from src.data.database import (
    PlatformDatabase,
    dataset_snapshot,
    experiment,
    feature_manifest,
    universe_snapshot,
)
from src.services.config import load_platform_config, load_split_configuration


def _check(name: str, ok: bool, *, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **({"detail": detail} if detail is not None else {})}


def _regular_directory(path: Path) -> tuple[bool, str]:
    if path.is_symlink():
        return False, "path_is_symlink"
    if path.exists() and not path.is_dir():
        return False, "path_is_not_directory"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"directory_unavailable:{type(exc).__name__}"
    return True, "ready"


def build_readiness(config_path: Path = Path("config/platform.json")) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        config = load_platform_config(config_path)
        split = load_split_configuration(config_path.parent)
        checks.append(_check("platform_configuration", True))
        checks.append(
            _check(
                "products_paper_only",
                all(
                    product.get("execution_mode") == "paper"
                    for product in split["products"]["products"]
                ),
                detail={
                    str(product["product_id"]): product.get("execution_mode")
                    for product in split["products"]["products"]
                },
            )
        )
        checks.append(
            _check(
                "automatic_live_canary_disabled",
                all(
                    policy.get("automatic_live_canary_promotion") is False
                    for policy in split["promotion"]["policies"]
                ),
                detail={
                    str(policy["policy_id"]): policy.get("automatic_live_canary_promotion")
                    for policy in split["promotion"]["policies"]
                },
            )
        )
    except Exception as exc:
        return {
            "schema": "platform.readiness/v1",
            "ok": False,
            "checks": [
                _check("platform_configuration", False, detail=f"{type(exc).__name__}: {exc}")
            ],
        }

    paths: dict[str, dict[str, Any]] = {}
    for name, raw_path in config.paths.items():
        path = Path(raw_path)
        ok, reason = _regular_directory(path)
        parquet_count = sum(1 for item in path.rglob("*.parquet") if item.is_file()) if ok else 0
        paths[name] = {
            "path": str(path),
            "ok": ok,
            "reason": reason,
            "parquet_files": parquet_count,
        }
        checks.append(_check(f"path:{name}", ok, detail=paths[name]))

    database = None
    try:
        database = PlatformDatabase(config.database_url())
        checks.append(_check("postgresql_authority", database.is_postgresql))
        if database.is_postgresql:
            database.assert_migrated()
        else:
            raise RuntimeError("platform readiness requires PostgreSQL")
        with database.engine.connect() as connection:
            table_names = set(inspect(database.engine).get_table_names())
            counts = {
                "universe_snapshots": int(
                    connection.execute(
                        select(func.count()).select_from(universe_snapshot)
                    ).scalar_one()
                ),
                "dataset_snapshots": int(
                    connection.execute(
                        select(func.count()).select_from(dataset_snapshot)
                    ).scalar_one()
                ),
                "feature_manifests": int(
                    connection.execute(
                        select(func.count()).select_from(feature_manifest)
                    ).scalar_one()
                ),
                "experiments": int(
                    connection.execute(select(func.count()).select_from(experiment)).scalar_one()
                ),
            }
        checks.append(
            _check("canonical_tables", True, detail={"count": len(table_names), "rows": counts})
        )
    except Exception as exc:
        checks.append(_check("postgresql_authority", False, detail=f"{type(exc).__name__}: {exc}"))
    finally:
        if database is not None:
            database.dispose()

    return {
        "schema": "platform.readiness/v1",
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "paths": paths,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PostgreSQL-authoritative platform readiness."
    )
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_readiness(args.config)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.is_symlink():
            raise SystemExit("readiness output must not be a symlink")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
