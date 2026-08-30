"""Fail-closed readiness checks for the PostgreSQL-authoritative platform."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select

from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    cost_model_manifest,
    dataset_bundle,
    dataset_snapshot,
    experiment,
    feature_manifest,
    platform_rehearsal_report,
    platform_schedule,
    production_preflight,
    service_heartbeat,
    strategy_approval,
    strategy_artefact,
    strategy_definition,
    strategy_version,
    universe_snapshot,
    validation_stage,
)
from src.data.database import (
    instrument as instrument_table,
)
from src.domain._codec import canonical_hash, timestamp
from src.execution.config import ACCOUNT_FINGERPRINT_PREFIX
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.risk.engine import SqlRiskSnapshotStore
from src.services.config import load_platform_config, load_split_configuration
from src.services.live_execution import (
    _exchange_config,
    execution_engine_identity,
    live_authority_configuration_hash,
)
from src.services.portfolio_state import DatabasePortfolioStateWorker, portfolio_state_policies
from src.services.scheduler import AUTONOMOUS_SCHEDULES


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


def _execution_engine_identity() -> str:
    return execution_engine_identity()


def _live_product_checks(
    *,
    connection,
    product: dict[str, Any],
    accounts: dict[str, dict[str, Any]],
    promotion_policies: dict[str, dict[str, Any]],
    risk_configuration: Mapping[str, Any],
    now: str,
) -> dict[str, Any]:
    product_id = str(product["product_id"])
    account_id = str(product["account_id"])
    account_config = accounts.get(account_id)
    assignment = SqlActiveStrategyAssignmentRepository(connection.engine).active(
        product_id,
        execution_mode="live",
        at=now,
    )
    result: dict[str, Any] = {
        "product_configured_live": product.get("execution_mode") == "live",
        "assignment": assignment is not None,
        "approval": False,
        "preflight": False,
        "account_snapshot": False,
        "connected_testnet_rehearsal": False,
        "account_fingerprint": False,
    }
    if assignment is None:
        result["ok"] = False
        return result
    assignment = dict(assignment)
    artifact_row = (
        connection.execute(
            select(strategy_artefact.c.payload, strategy_artefact.c.created_at).where(
                strategy_artefact.c.id == str(assignment["artefact_hash"])
            )
        )
        .mappings()
        .first()
    )
    artifact = artifact_row["payload"] if artifact_row is not None else None
    if not isinstance(artifact, Mapping):
        result["ok"] = False
        result["artifact"] = False
        return result
    content = dict(artifact)
    declared = content.pop("artefact_hash", None)
    result["artifact"] = (
        declared == assignment["artefact_hash"]
        and canonical_hash(content) == declared
        and artifact.get("product_id") == product_id
        and artifact.get("account_id") == account_id
        and artifact.get("portfolio_id") == product.get("portfolio_id")
        and artifact.get("created_at") is not None
        and timestamp(str(artifact["created_at"]), field="artifact.created_at")
        == timestamp(str(artifact_row["created_at"]), field="artifact.created_at")
    )
    instrument_payload: dict[str, Any] | None = None
    assignment_instrument_id = str(assignment.get("instrument_id") or "")
    if assignment_instrument_id:
        persisted_instrument = connection.execute(
            select(instrument_table.c.payload).where(
                instrument_table.c.id == assignment_instrument_id
            )
        ).scalar_one_or_none()
        if isinstance(persisted_instrument, Mapping):
            instrument_payload = dict(persisted_instrument)
            instrument_payload["instrument_id"] = assignment_instrument_id
    approval = (
        connection.execute(
            select(strategy_approval)
            .where(
                strategy_approval.c.strategy_version_id == assignment["strategy_version_id"],
                strategy_approval.c.product_id == product_id,
                strategy_approval.c.account_id == account_id,
                strategy_approval.c.approved_at <= now,
            )
            .order_by(strategy_approval.c.approved_at.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    preflight = (
        connection.execute(
            select(production_preflight)
            .where(
                production_preflight.c.strategy_version_id == assignment["strategy_version_id"],
                production_preflight.c.product_id == product_id,
                production_preflight.c.account_id == account_id,
            )
            .order_by(production_preflight.c.checked_at.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    approval_payload = approval["payload"] if approval else None
    preflight_payload = preflight["payload"] if preflight else None
    result["approval"] = bool(
        approval
        and approval["status"] == "approved"
        and approval["artefact_hash"] == assignment["artefact_hash"]
        and approval["source_commit_hash"] == artifact.get("source_commit_hash")
        and approval["engine_version"] == artifact.get("engine_version")
    )
    preflight_age = (
        (
            dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(str(preflight["checked_at"]))
        ).total_seconds()
        if preflight
        else None
    )
    result["preflight"] = bool(
        preflight
        and preflight["accepted"]
        and preflight["artefact_hash"] == assignment["artefact_hash"]
        and preflight["source_commit_hash"] == artifact.get("source_commit_hash")
        and preflight["engine_version"] == artifact.get("engine_version")
        and preflight_age is not None
        and 0 <= preflight_age <= int(product.get("preflight_max_age_seconds", 3600))
    )
    snapshot = None
    for candidate in connection.execute(
        select(account_snapshot)
        .where(
            account_snapshot.c.account_id == account_id,
            account_snapshot.c.observed_at <= now,
        )
        .order_by(account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc())
    ).mappings():
        if (
            isinstance(candidate["payload"], Mapping)
            and candidate["payload"].get("product_id") == product_id
        ):
            snapshot = candidate
            break
    snapshot_payload = snapshot["payload"] if snapshot else None
    expected_fingerprint = ""
    fingerprint_error = ""
    if account_config is not None:
        try:
            market = "spot" if account_config.get("market") == "spot" else "futures"
            expected_fingerprint = _exchange_config(
                account_config, market=market
            ).account_fingerprint
        except Exception as exc:
            fingerprint_error = f"{type(exc).__name__}: {exc}"
    actual_fingerprint = (
        str(snapshot_payload.get("account_fingerprint") or "")
        if isinstance(snapshot_payload, Mapping)
        else ""
    )
    result["account_fingerprint"] = bool(
        expected_fingerprint and actual_fingerprint == expected_fingerprint
    )
    engine_identity = _execution_engine_identity()
    configuration_hash = ""
    if account_config is not None and instrument_payload is not None:
        configuration_hash = live_authority_configuration_hash(
            product=product,
            account=account_config,
            instrument_payload=instrument_payload,
            artefact=artifact,
            sleeve_id=str(assignment["sleeve_id"]),
            promotion_policy=promotion_policies[str(product["promotion_policy_id"])],
            risk_configuration=risk_configuration,
        )
    result["approval"] = bool(
        result["approval"]
        and isinstance(approval_payload, Mapping)
        and approval_payload.get("schema") == "platform.strategy-approval/v1"
        and approval_payload.get("preflight_id") == (preflight["id"] if preflight else None)
        and approval_payload.get("instrument_id") == assignment_instrument_id
        and approval_payload.get("sleeve_id") == assignment["sleeve_id"]
        and approval_payload.get("environment")
        == (account_config.get("environment") if account_config else None)
        and approval_payload.get("account_fingerprint") == expected_fingerprint
        and approval_payload.get("execution_engine_identity") == engine_identity
        and approval_payload.get("configuration_hash") == configuration_hash
    )
    result["preflight"] = bool(
        result["preflight"]
        and isinstance(preflight_payload, Mapping)
        and preflight_payload.get("schema") == "platform.production-preflight/v1"
        and preflight_payload.get("instrument_id") == assignment_instrument_id
        and preflight_payload.get("sleeve_id") == assignment["sleeve_id"]
        and preflight_payload.get("environment")
        == (account_config.get("environment") if account_config else None)
        and preflight_payload.get("account_fingerprint") == expected_fingerprint
        and preflight_payload.get("execution_engine_identity") == engine_identity
        and preflight_payload.get("configuration_hash") == configuration_hash
        and preflight.get("content_hash") == canonical_hash(dict(preflight_payload))
    )
    snapshot_age = (
        (
            dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(str(snapshot["observed_at"]))
        ).total_seconds()
        if snapshot is not None
        else None
    )
    result["account_fingerprint_detail"] = {
        "expected": expected_fingerprint,
        "actual": actual_fingerprint,
        **({"error": fingerprint_error} if fingerprint_error else {}),
    }
    required_account_fields = {
        "balances",
        "free_balances",
        "positions",
        "regular_orders",
        "conditional_orders",
        "used_margin",
        "maintenance_margin",
        "used_margin_fraction",
        "liquidation_buffer_fraction",
        "account_mode",
        "unknown_exposure",
        "account_state_known",
        "account_state_authority",
        "account_fingerprint",
        "observed_at",
    }

    def finite_number(value: object) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(str(value)))
        except (TypeError, ValueError):
            return False

    account_values: Mapping[str, Any] = (
        snapshot_payload if isinstance(snapshot_payload, Mapping) else {}
    )
    account_shape = bool(
        bool(account_values)
        and required_account_fields.issubset(account_values)
        and isinstance(account_values.get("balances"), Mapping)
        and isinstance(account_values.get("free_balances"), Mapping)
        and isinstance(account_values.get("positions"), Mapping)
        and isinstance(account_values.get("regular_orders"), list)
        and isinstance(account_values.get("conditional_orders"), list)
        and isinstance(account_values.get("unknown_exposure"), Mapping)
        and isinstance(account_values.get("account_mode"), str)
        and bool(str(account_values.get("account_mode") or "").strip())
        and all(
            finite_number(account_values.get(field))
            for field in (
                "used_margin",
                "maintenance_margin",
                "used_margin_fraction",
                "liquidation_buffer_fraction",
            )
        )
    )
    account_content_hash = snapshot.get("content_hash") if snapshot is not None else None
    account_content_valid = bool(
        account_shape and account_content_hash == canonical_hash(dict(account_values))
    )
    account_identity_valid = bool(
        account_shape
        and snapshot is not None
        and account_values.get("account_id") == account_id
        and account_values.get("product_id") == product_id
        and account_values.get("observed_at") == snapshot.get("observed_at")
    )
    result["account_snapshot_detail"] = {
        "shape": account_shape,
        "content_hash": account_content_valid,
        "identity": account_identity_valid,
        "unknown_exposure": account_values.get("unknown_exposure") if account_shape else None,
    }
    result["account_snapshot"] = bool(
        account_content_valid
        and account_identity_valid
        and account_values.get("account_state_known") is True
        and account_values.get("account_state_authority")
        in {"authenticated_rest", "authenticated_reconciled"}
        and snapshot is not None
        and snapshot.get("source") in {"authenticated_rest", "authenticated_reconciled"}
        and account_values.get("unknown_exposure") == {}
        and account_values.get("account_fingerprint")
        and snapshot_age is not None
        and 0 <= snapshot_age <= int(product.get("account_snapshot_max_age_seconds", 60))
    )
    report = (
        connection.execute(
            select(platform_rehearsal_report)
            .where(
                platform_rehearsal_report.c.product_id == product_id,
                platform_rehearsal_report.c.created_at <= now,
            )
            .order_by(platform_rehearsal_report.c.created_at.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    report_payload = report["payload"] if report else None
    report_integrity = False
    report_age: float | None = None
    if report is not None and isinstance(report_payload, Mapping):
        unsigned_report = dict(report_payload)
        declared_report_hash = unsigned_report.pop("report_hash", None)
        signature = unsigned_report.pop("signature", None)
        calculated_report_hash = canonical_hash(unsigned_report)
        signing_key = os.environ.get("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY", "")
        signature_valid = bool(
            signing_key
            and isinstance(signature, str)
            and signature
            and isinstance(declared_report_hash, str)
        )
        if signature_valid:
            expected_signature = hmac.new(
                signing_key.encode(), declared_report_hash.encode(), hashlib.sha256
            ).hexdigest()
            signature_valid = hmac.compare_digest(signature, expected_signature)
        report_age = (
            dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(str(report["created_at"]))
        ).total_seconds()
        required_scopes = ["strategy", "instrument", "sleeve", "product", "account", "global"]
        report_integrity = (
            bool(report["accepted"])
            and report["content_hash"] == declared_report_hash == calculated_report_hash
            and report["id"] == declared_report_hash
            and signature_valid
            and report_age is not None
            and 0 <= report_age <= int(product.get("connected_testnet_max_age_seconds", 86_400))
            and report_payload.get("environment") == "testnet"
            and report_payload.get("real_exchange") is True
            and report_payload.get("product_id") == product_id
            and report_payload.get("account_id") == account_id
            and report_payload.get("assignment_id") == assignment["id"]
            and report_payload.get("artefact_hash") == assignment["artefact_hash"]
            and report_payload.get("execution_engine_identity") == engine_identity
            and report_payload.get("open_acknowledged") is True
            and report_payload.get("close_acknowledged") is True
            and report_payload.get("user_stream_fill") is True
            and report_payload.get("accounting_reconciled") is True
            and report_payload.get("flat_reconciliation") is True
            and report_payload.get("risk_accepted") is True
            and report_payload.get("risk_scopes") == required_scopes
            and all(
                isinstance(report_payload.get(field), str)
                and report_payload[field].startswith("sha256:")
                for field in ("forecast_id", "target_position_snapshot_id", "risk_assessment_id")
            )
            and isinstance(report_payload.get("recovery_identifiers"), Mapping)
            and report_payload.get("recovery_lookup") is True
            and isinstance(report_payload.get("account_fingerprint"), str)
            and report_payload["account_fingerprint"].startswith(ACCOUNT_FINGERPRINT_PREFIX)
            and len(report_payload["account_fingerprint"].removeprefix(ACCOUNT_FINGERPRINT_PREFIX))
            == 64
            and all(
                character in "0123456789abcdef"
                for character in report_payload["account_fingerprint"].removeprefix(
                    ACCOUNT_FINGERPRINT_PREFIX
                )
            )
        )
    result["connected_testnet_rehearsal"] = report_integrity
    result["connected_testnet_rehearsal_detail"] = {
        "report_id": report["id"] if report is not None else None,
        "age_seconds": report_age,
        "integrity": report_integrity,
    }
    result["ok"] = all(
        result.get(name) is True
        for name in (
            "product_configured_live",
            "artifact",
            "assignment",
            "approval",
            "preflight",
            "account_snapshot",
            "account_fingerprint",
            "connected_testnet_rehearsal",
        )
    )
    return result


def build_readiness(
    config_path: Path = Path("config/platform.json"),
    *,
    live: bool = False,
    product_id: str | None = None,
    now: str | None = None,
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
                    else product.get("execution_mode") == "live"
                    for product in split["products"]["products"]
                    if product_id is None or str(product["product_id"]) == product_id
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
                "dataset_bundles": int(
                    connection.execute(
                        select(func.count()).select_from(dataset_bundle)
                    ).scalar_one()
                ),
                "feature_manifests": int(
                    connection.execute(
                        select(func.count()).select_from(feature_manifest)
                    ).scalar_one()
                ),
                "cost_model_manifests": int(
                    connection.execute(
                        select(func.count()).select_from(cost_model_manifest)
                    ).scalar_one()
                ),
                "experiments": int(
                    connection.execute(select(func.count()).select_from(experiment)).scalar_one()
                ),
                "schedules": int(
                    connection.execute(
                        select(func.count()).select_from(platform_schedule)
                    ).scalar_one()
                ),
                "account_snapshots": int(
                    connection.execute(
                        select(func.count()).select_from(account_snapshot)
                    ).scalar_one()
                ),
            }
            bundle_rows = connection.execute(select(dataset_bundle.c.payload)).scalars().all()
            ready_roles_by_product: dict[str, set[str]] = {}
            for raw_bundle in bundle_rows:
                if not isinstance(raw_bundle, Mapping):
                    continue
                if raw_bundle.get("lifecycle_state") != "ready":
                    continue
                product_key = str(raw_bundle.get("product_id") or "")
                stages = raw_bundle.get("stage_snapshot_ids")
                if product_key and isinstance(stages, Mapping):
                    ready_roles_by_product.setdefault(product_key, set()).update(
                        str(role) for role in stages
                    )
            schedule_rows = connection.execute(select(platform_schedule)).mappings().all()
            activity_rows = connection.execute(
                select(strategy_definition.c.product_id, experiment.c.submitted_at)
                .select_from(
                    experiment.join(
                        strategy_version,
                        experiment.c.strategy_version_id == strategy_version.c.id,
                    ).join(
                        strategy_definition,
                        strategy_version.c.definition_id == strategy_definition.c.id,
                    )
                )
            ).all()
            stage_activity_rows = connection.execute(
                select(strategy_definition.c.product_id, validation_stage.c.evaluated_at)
                .select_from(
                    validation_stage.join(
                        experiment,
                        validation_stage.c.experiment_id == experiment.c.id,
                    ).join(
                        strategy_version,
                        experiment.c.strategy_version_id == strategy_version.c.id,
                    ).join(
                        strategy_definition,
                        strategy_version.c.definition_id == strategy_definition.c.id,
                    )
                )
            ).all()
        source_store = SqlRiskSnapshotStore(database.engine)
        state_details: dict[str, Any] = {}
        products_by_id = {
            str(product["product_id"]): dict(product) for product in split["products"]["products"]
        }
        expected_policies = portfolio_state_policies({"risk": split["risk"]}, products_by_id)
        products_to_check = [
            product
            for product in split["products"]["products"]
            if product_id is None or str(product["product_id"]) == product_id
        ]
        selected_product_id = product_id
        for product in products_to_check:
            state_product_id = str(product["product_id"])
            try:
                state_id, state = source_store.latest(
                    kind="canonical_portfolio_risk_state", product_id=state_product_id, at=current
                )
                observed_at = timestamp(str(state["observed_at"]), field="state.observed_at")
                age = (
                    dt.datetime.fromisoformat(current) - dt.datetime.fromisoformat(observed_at)
                ).total_seconds()
                if age < 0:
                    raise ValueError("canonical portfolio state timestamp is in the future")
                maximum_age = float(state["maximum_state_age_seconds"])
                readiness_maximum_age = maximum_age if live else max(maximum_age, 600.0)
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
                    if source_payload.get("product_id") != state_product_id:
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
                    if (
                        source in {"account", "balances", "market"}
                        and source_age > readiness_maximum_age
                    ):
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
                expected_policy = expected_policies[state_product_id]
                if policy_hash != canonical_hash(expected_policy):
                    raise ValueError("canonical state risk policy identity is invalid")
                expected_policy_ids = {
                    str(item) for item in expected_policy.get("risk_policy_ids", ())
                }
                if {str(item) for item in policy_ids} != expected_policy_ids:
                    raise ValueError("canonical state risk policy IDs are invalid")
                state_details[state_product_id] = {
                    "state_id": state_id,
                    "age_seconds": age,
                    "maximum_age_seconds": maximum_age,
                    "readiness_maximum_age_seconds": readiness_maximum_age,
                    "source_ages_seconds": source_ages,
                    "risk_policy_ids": list(policy_ids),
                    "risk_policy_hash": policy_hash,
                }
                if age > readiness_maximum_age:
                    raise ValueError("canonical portfolio state is stale")
            except Exception as exc:
                state_details[state_product_id] = {"error": f"{type(exc).__name__}: {exc}"}
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
        required_schedule_names = {spec.name for spec in AUTONOMOUS_SCHEDULES}
        schedule_details: dict[str, dict[str, Any]] = {}
        schedule_fresh = True
        maximum_heartbeat_age = float(config.metrics.get("stale_after_seconds", 60))
        for row in schedule_rows:
            name = str(row["job_name"])
            updated_at = timestamp(str(row["updated_at"]), field=f"schedule.{name}.updated_at")
            age = (
                dt.datetime.fromisoformat(current)
                - dt.datetime.fromisoformat(updated_at)
            ).total_seconds()
            maximum_age = max(
                maximum_heartbeat_age * 2.0,
                float(row["interval_seconds"]) * 2.0,
            )
            fresh = 0 <= age <= maximum_age
            schedule_fresh = schedule_fresh and fresh
            schedule_details[name] = {
                "state": str(row["state"]),
                "updated_at": updated_at,
                "age_seconds": age,
                "maximum_age_seconds": maximum_age,
                "fresh": fresh,
            }
        schedule_authority_ok = (
            {str(row["job_name"]) for row in schedule_rows} == required_schedule_names
            and schedule_fresh
        )
        checks.append(
            _check(
                "autonomous_scheduler_authority",
                schedule_authority_ok,
                detail={
                    "schedule_rows": counts["schedules"],
                    "required_schedule_names": sorted(required_schedule_names),
                    "schedule_details": schedule_details,
                },
            )
        )
        required_products = {
            str(product["product_id"])
            for product in split["products"]["products"]
            if product_id is None or str(product["product_id"]) == product_id
        }
        required_roles = {
            "screening",
            "development",
            "robustness",
            "protected_holdout",
            "forward_observation",
        }
        dataset_details = {
            product_key: {
                "roles": sorted(ready_roles_by_product.get(product_key, set())),
                "missing_roles": sorted(
                    required_roles - ready_roles_by_product.get(product_key, set())
                ),
            }
            for product_key in sorted(required_products)
        }
        checks.append(
            _check(
                "canonical_dataset_role_authority",
                all(not detail["missing_roles"] for detail in dataset_details.values()),
                detail={"dataset_bundles": counts["dataset_bundles"], "products": dataset_details},
            )
        )
        latest_activity: dict[str, str] = {}
        for raw_product, observed in (*activity_rows, *stage_activity_rows):
            product_key = str(raw_product)
            observed_at = timestamp(str(observed), field="research_activity.observed_at")
            if observed_at > latest_activity.get(product_key, ""):
                latest_activity[product_key] = observed_at
        progress_details: dict[str, dict[str, Any]] = {}
        progress_ok = True
        maximum_progress_age = 86_400.0
        for product_key in sorted(required_products):
            observed_at = latest_activity.get(product_key)
            age = (
                (
                    dt.datetime.fromisoformat(current)
                    - dt.datetime.fromisoformat(observed_at)
                ).total_seconds()
                if observed_at is not None
                else None
            )
            progressed = age is not None and 0 <= age <= maximum_progress_age
            progress_ok = progress_ok and progressed
            progress_details[product_key] = {
                "latest_activity_at": observed_at,
                "age_seconds": age,
                "maximum_age_seconds": maximum_progress_age,
                "progressed": progressed,
            }
        checks.append(
            _check(
                "research_progress_authority",
                progress_ok,
                detail={"products": progress_details},
            )
        )
        checks.append(
            _check(
                "account_snapshot_authority",
                counts["account_snapshots"] >= len(split["products"]["products"]),
                detail={
                    "account_snapshot_rows": counts["account_snapshots"],
                    "required": len(split["products"]["products"]),
                },
            )
        )
        required_manifests = len(split["products"]["products"])
        checks.append(
            _check(
                "bootstrap_manifest_authority",
                counts["feature_manifests"] >= required_manifests
                and counts["cost_model_manifests"] >= required_manifests,
                detail={
                    "feature_manifests": counts["feature_manifests"],
                    "cost_model_manifests": counts["cost_model_manifests"],
                    "required": required_manifests,
                },
            )
        )
        with database.engine.connect() as connection:
            heartbeat_rows = connection.execute(
                select(
                    service_heartbeat.c.service_name,
                    service_heartbeat.c.observed_at,
                    service_heartbeat.c.healthy,
                )
                .where(
                    service_heartbeat.c.service_name.in_(
                        ("platform-scheduler", "account-reconciliation")
                    ),
                    service_heartbeat.c.observed_at <= current,
                )
                .order_by(service_heartbeat.c.observed_at.desc())
            ).mappings()
        latest_heartbeats: dict[str, dict[str, Any]] = {}
        for heartbeat in heartbeat_rows:
            service_name = str(heartbeat["service_name"])
            latest_heartbeats.setdefault(service_name, dict(heartbeat))
        heartbeat_ok = True
        heartbeat_details: dict[str, Any] = {}
        for service_name in ("platform-scheduler", "account-reconciliation"):
            heartbeat = latest_heartbeats.get(service_name)
            if heartbeat is None:
                heartbeat_ok = False
                heartbeat_details[service_name] = "missing"
                continue
            age = (
                dt.datetime.fromisoformat(current)
                - dt.datetime.fromisoformat(str(heartbeat["observed_at"]))
            ).total_seconds()
            healthy = bool(heartbeat["healthy"]) and 0 <= age <= float(
                config.metrics.get("stale_after_seconds", 60)
            )
            heartbeat_ok = heartbeat_ok and healthy
            heartbeat_details[service_name] = {"age_seconds": age, "healthy": healthy}
        checks.append(
            _check(
                "scheduler_and_reconciliation_heartbeats", heartbeat_ok, detail=heartbeat_details
            )
        )
        if live:
            live_products = [
                dict(product)
                for product in split["products"]["products"]
                if selected_product_id is None or str(product["product_id"]) == selected_product_id
            ]
            live_details: dict[str, Any] = {}
            with database.engine.connect() as live_connection:
                for product in live_products:
                    live_details[str(product["product_id"])] = _live_product_checks(
                        connection=live_connection,
                        product=product,
                        accounts={
                            str(item["account_id"]): dict(item)
                            for item in split["accounts"]["accounts"]
                        },
                        promotion_policies={
                            str(item["policy_id"]): dict(item)
                            for item in split["promotion"]["policies"]
                        },
                        risk_configuration=split["risk"],
                        now=current,
                    )
            checks.append(
                _check(
                    "live_execution_authority",
                    all(detail.get("ok") is True for detail in live_details.values())
                    and bool(live_details),
                    detail=live_details,
                )
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
    parser.add_argument("--product")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_readiness(args.config, live=args.live, product_id=args.product)
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
