"""Fail-closed readiness checks for the PostgreSQL-authoritative platform."""

from __future__ import annotations

import argparse
import datetime as dt
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
from src.domain._codec import timestamp
from src.risk.engine import SqlRiskSnapshotStore
from src.services.config import load_platform_config, load_split_configuration
from src.services.portfolio_state import DatabasePortfolioStateWorker


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


def build_readiness(
    config_path: Path = Path("config/platform.json"), *, live: bool = False, now: str | None = None
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        config = load_platform_config(config_path)
        split = load_split_configuration(config_path.parent)
        checks.append(_check("platform_configuration", True))
        checks.append(
            _check(
                "products_paper_only" if not live else "products_execution_configured",
                all(
                    product.get("execution_mode") == "paper"
                    if not live
                    else product.get("execution_mode") in {"paper", "live"}
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
    current = timestamp(now or dt.datetime.now(dt.UTC), field="now")
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
        source_store = SqlRiskSnapshotStore(database.engine)
        state_details: dict[str, Any] = {}
        for product in split["products"]["products"]:
            product_id = str(product["product_id"])
            try:
                state_id, state = source_store.latest(
                    kind="canonical_portfolio_risk_state", product_id=product_id, at=current
                )
                observed_at = timestamp(str(state["observed_at"]), field="state.observed_at")
                age = (
                    dt.datetime.fromisoformat(current) - dt.datetime.fromisoformat(observed_at)
                ).total_seconds()
                if age < 0:
                    raise ValueError("canonical portfolio state timestamp is in the future")
                maximum_age = float(state["maximum_state_age_seconds"])
                source_ids = state.get("source_snapshot_ids")
                source_ages: dict[str, float] = {}
                source_observed_at: list[str] = []
                if (
                    not isinstance(source_ids, dict)
                    or set(source_ids) != DatabasePortfolioStateWorker.REQUIRED_SOURCES
                ):
                    raise ValueError("canonical state source identities are incomplete")
                for source, source_id in source_ids.items():
                    if (
                        not isinstance(source_id, str)
                        or not source_id.startswith("sha256:")
                        or len(source_id) != 71
                    ):
                        raise ValueError(f"{source} source identity is invalid")
                    source_payload = source_store.get(str(source_id))
                    if source_payload.get("product_id") != product_id:
                        raise ValueError(f"{source} source belongs to another product")
                    if source_payload.get("kind") not in {source, f"{source}_snapshot"}:
                        raise ValueError(f"{source} source has the wrong kind")
                    source_observed = timestamp(
                        str(source_payload.get("observed_at", source_payload.get("created_at"))),
                        field=f"{source}.observed_at",
                    )
                    source_age = (
                        dt.datetime.fromisoformat(current)
                        - dt.datetime.fromisoformat(source_observed)
                    ).total_seconds()
                    if source_age < 0:
                        raise ValueError(f"{source} source timestamp is in the future")
                    source_ages[source] = source_age
                    source_observed_at.append(source_observed)
                    if source_age > maximum_age:
                        raise ValueError(f"{source} source is stale")
                if source_observed_at and observed_at != max(source_observed_at):
                    raise ValueError(
                        "canonical portfolio state is not at the latest source timestamp"
                    )
                policy_ids = state.get("risk_policy_ids")
                policy_hash = state.get("risk_policy_hash")
                if not isinstance(policy_ids, list | tuple) or not policy_ids:
                    raise ValueError("risk policy identities are missing")
                if (
                    not isinstance(policy_hash, str)
                    or not policy_hash.startswith("sha256:")
                    or len(policy_hash) != 71
                ):
                    raise ValueError("risk policy hash is missing")
                state_details[product_id] = {
                    "state_id": state_id,
                    "age_seconds": age,
                    "maximum_age_seconds": maximum_age,
                    "source_ages_seconds": source_ages,
                    "risk_policy_ids": list(policy_ids),
                    "risk_policy_hash": policy_hash,
                }
                if age > maximum_age:
                    raise ValueError("canonical portfolio state is stale")
            except Exception as exc:
                state_details[product_id] = {"error": f"{type(exc).__name__}: {exc}"}
        checks.append(
            _check(
                "canonical_portfolio_state_authority",
                all("error" not in detail for detail in state_details.values()),
                detail=state_details,
            )
        )
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
        "mode": "live" if live else "paper",
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
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_readiness(args.config, live=args.live)
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
