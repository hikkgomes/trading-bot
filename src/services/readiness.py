"""Fail-closed readiness checks for the PostgreSQL-authoritative platform."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select

from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    active_strategy_assignment,
    alert_delivery_attempt,
    cost_model_manifest,
    dataset_bundle,
    dataset_snapshot,
    experiment,
    feature_manifest,
    historical_bootstrap_progress,
    job,
    job_quarantine,
    platform_rehearsal_report,
    platform_schedule,
    production_preflight,
    promotion_event,
    service_heartbeat,
    strategy_approval,
    strategy_artefact,
    strategy_definition,
    strategy_version,
    universe_member,
    universe_snapshot,
    validation_stage,
)
from src.data.database import (
    instrument as instrument_table,
)
from src.domain._codec import canonical_hash, timestamp
from src.execution.config import ACCOUNT_FINGERPRINT_PREFIX
from src.research.canonical import SqlActiveStrategyAssignmentRepository
from src.research.datasets import dataset_payload_is_non_promotable
from src.risk.engine import SqlRiskSnapshotStore
from src.services.config import (
    PLATFORM_PROCESS_SERVICES,
    load_platform_config,
    load_split_configuration,
)
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
    result = _live_initial_result(product, assignment)
    if assignment is None:
        result["ok"] = False
        return result
    assignment = dict(assignment)
    artifact_row = _live_artifact_row(connection, assignment)
    artifact = artifact_row["payload"] if artifact_row is not None else None
    if not isinstance(artifact, Mapping):
        result["ok"] = False
        result["artifact"] = False
        return result
    result["artifact"] = _live_artifact_integrity(
        artifact,
        artifact_row=artifact_row,
        assignment=assignment,
        product_id=product_id,
        account_id=account_id,
        portfolio_id=str(product.get("portfolio_id") or ""),
    )
    assignment_instrument_id, instrument_payload = _live_assignment_instrument(
        connection, assignment
    )
    authority = _live_authority_checks(
        connection=connection,
        product=product,
        account=account_config,
        assignment=assignment,
        artifact=artifact,
        product_id=product_id,
        account_id=account_id,
        instrument_id=assignment_instrument_id,
        instrument_payload=instrument_payload,
        promotion_policies=promotion_policies,
        risk_configuration=risk_configuration,
        now=now,
    )
    result.update(authority["checks"])
    expected_fingerprint = authority["expected_fingerprint"]
    actual_fingerprint = authority["actual_fingerprint"]
    fingerprint_error = authority["fingerprint_error"]
    engine_identity = authority["engine_identity"]
    account_checks = _live_account_snapshot_checks(
        connection=connection,
        product=product,
        product_id=product_id,
        account_id=account_id,
        now=now,
        expected_fingerprint=expected_fingerprint,
        actual_fingerprint=actual_fingerprint,
        fingerprint_error=fingerprint_error,
    )
    result.update(account_checks)
    rehearsal = _live_rehearsal_checks(
        connection=connection,
        product=product,
        product_id=product_id,
        account_id=account_id,
        assignment=assignment,
        now=now,
        engine_identity=engine_identity,
    )
    result.update(rehearsal)
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


def _live_initial_result(product: Mapping[str, Any], assignment: Any) -> dict[str, Any]:
    return {
        "product_configured_live": product.get("execution_mode") == "live",
        "assignment": assignment is not None,
        "approval": False,
        "preflight": False,
        "account_snapshot": False,
        "connected_testnet_rehearsal": False,
        "account_fingerprint": False,
    }


def _live_artifact_row(connection: Any, assignment: Mapping[str, Any]) -> Any:
    return (
        connection.execute(
            select(strategy_artefact.c.payload, strategy_artefact.c.created_at).where(
                strategy_artefact.c.id == str(assignment["artefact_hash"])
            )
        )
        .mappings()
        .first()
    )


def _live_artifact_integrity(
    artifact: Mapping[str, Any],
    *,
    artifact_row: Mapping[str, Any],
    assignment: Mapping[str, Any],
    product_id: str,
    account_id: str,
    portfolio_id: str,
) -> bool:
    content = dict(artifact)
    declared = content.pop("artefact_hash", None)
    return (
        declared == assignment["artefact_hash"]
        and canonical_hash(content) == declared
        and artifact.get("product_id") == product_id
        and artifact.get("account_id") == account_id
        and artifact.get("portfolio_id") == portfolio_id
        and artifact.get("created_at") is not None
        and timestamp(str(artifact["created_at"]), field="artifact.created_at")
        == timestamp(str(artifact_row["created_at"]), field="artifact.created_at")
    )


def _live_assignment_instrument(
    connection: Any, assignment: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    instrument_id = str(assignment.get("instrument_id") or "")
    if not instrument_id:
        return instrument_id, None
    persisted = connection.execute(
        select(instrument_table.c.payload).where(instrument_table.c.id == instrument_id)
    ).scalar_one_or_none()
    if not isinstance(persisted, Mapping):
        return instrument_id, None
    payload = dict(persisted)
    payload["instrument_id"] = instrument_id
    return instrument_id, payload


def _live_authority_checks(
    *,
    connection: Any,
    product: Mapping[str, Any],
    account: Mapping[str, Any] | None,
    assignment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    product_id: str,
    account_id: str,
    instrument_id: str,
    instrument_payload: Mapping[str, Any] | None,
    promotion_policies: Mapping[str, Mapping[str, Any]],
    risk_configuration: Mapping[str, Any],
    now: str,
) -> dict[str, Any]:
    approval = _latest_live_record(
        connection,
        strategy_approval,
        strategy_version_id=assignment["strategy_version_id"],
        product_id=product_id,
        account_id=account_id,
        time_column=strategy_approval.c.approved_at,
        now=now,
    )
    preflight = _latest_live_record(
        connection,
        production_preflight,
        strategy_version_id=assignment["strategy_version_id"],
        product_id=product_id,
        account_id=account_id,
        time_column=production_preflight.c.checked_at,
        now=now,
    )
    snapshot = _latest_live_account_snapshot(connection, product_id, account_id, now)
    snapshot_payload = snapshot["payload"] if snapshot else None
    expected_fingerprint, fingerprint_error = _expected_live_fingerprint(account)
    actual_fingerprint = (
        str(snapshot_payload.get("account_fingerprint") or "")
        if isinstance(snapshot_payload, Mapping)
        else ""
    )
    engine_identity = _execution_engine_identity()
    configuration_hash = _live_configuration_hash(
        product=product,
        account=account,
        instrument_payload=instrument_payload,
        artifact=artifact,
        assignment=assignment,
        promotion_policies=promotion_policies,
        risk_configuration=risk_configuration,
    )
    checks = {
        "approval": _live_approval_valid(
            approval,
            assignment=assignment,
            artifact=artifact,
            preflight=preflight,
            account=account,
            instrument_id=instrument_id,
            expected_fingerprint=expected_fingerprint,
            engine_identity=engine_identity,
            configuration_hash=configuration_hash,
        ),
        "preflight": _live_preflight_valid(
            preflight,
            assignment=assignment,
            artifact=artifact,
            product=product,
            account=account,
            instrument_id=instrument_id,
            expected_fingerprint=expected_fingerprint,
            engine_identity=engine_identity,
            configuration_hash=configuration_hash,
            now=now,
        ),
        "account_fingerprint": bool(
            expected_fingerprint and actual_fingerprint == expected_fingerprint
        ),
    }
    return {
        "checks": checks,
        "approval": approval,
        "preflight": preflight,
        "snapshot": snapshot,
        "snapshot_payload": snapshot_payload,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "fingerprint_error": fingerprint_error,
        "engine_identity": engine_identity,
    }


def _latest_live_record(
    connection: Any,
    table: Any,
    *,
    strategy_version_id: str,
    product_id: str,
    account_id: str,
    time_column: Any,
    now: str,
) -> Mapping[str, Any] | None:
    return (
        connection.execute(
            select(table)
            .where(
                table.c.strategy_version_id == strategy_version_id,
                table.c.product_id == product_id,
                table.c.account_id == account_id,
                time_column <= now,
            )
            .order_by(time_column.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )


def _latest_live_account_snapshot(
    connection: Any, product_id: str, account_id: str, now: str
) -> Mapping[str, Any] | None:
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
            return candidate
    return None


def _expected_live_fingerprint(
    account: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if account is None:
        return "", "account_configuration_missing"
    try:
        market = "spot" if account.get("market") == "spot" else "futures"
        return _exchange_config(account, market=market).account_fingerprint, ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _live_configuration_hash(
    *,
    product: Mapping[str, Any],
    account: Mapping[str, Any] | None,
    instrument_payload: Mapping[str, Any] | None,
    artifact: Mapping[str, Any],
    assignment: Mapping[str, Any],
    promotion_policies: Mapping[str, Mapping[str, Any]],
    risk_configuration: Mapping[str, Any],
) -> str:
    if account is None or instrument_payload is None:
        return ""
    return live_authority_configuration_hash(
        product=product,
        account=account,
        instrument_payload=instrument_payload,
        artefact=artifact,
        sleeve_id=str(assignment["sleeve_id"]),
        promotion_policy=promotion_policies[str(product["promotion_policy_id"])],
        risk_configuration=risk_configuration,
    )


def _live_approval_valid(
    approval: Mapping[str, Any] | None,
    *,
    assignment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    account: Mapping[str, Any] | None,
    instrument_id: str,
    expected_fingerprint: str,
    engine_identity: str,
    configuration_hash: str,
) -> bool:
    payload = approval.get("payload") if approval else None
    return bool(
        approval
        and approval["status"] == "approved"
        and approval["artefact_hash"] == assignment["artefact_hash"]
        and approval["source_commit_hash"] == artifact.get("source_commit_hash")
        and approval["engine_version"] == artifact.get("engine_version")
        and isinstance(payload, Mapping)
        and payload.get("schema") == "platform.strategy-approval/v1"
        and payload.get("preflight_id") == (preflight["id"] if preflight else None)
        and payload.get("instrument_id") == instrument_id
        and payload.get("sleeve_id") == assignment["sleeve_id"]
        and payload.get("environment") == (account.get("environment") if account else None)
        and payload.get("account_fingerprint") == expected_fingerprint
        and payload.get("execution_engine_identity") == engine_identity
        and payload.get("configuration_hash") == configuration_hash
    )


def _live_preflight_valid(
    preflight: Mapping[str, Any] | None,
    *,
    assignment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    product: Mapping[str, Any],
    account: Mapping[str, Any] | None,
    instrument_id: str,
    expected_fingerprint: str,
    engine_identity: str,
    configuration_hash: str,
    now: str,
) -> bool:
    if preflight is None:
        return False
    age = _record_age(now, str(preflight["checked_at"]))
    payload = preflight.get("payload")
    return bool(
        preflight["accepted"]
        and preflight["artefact_hash"] == assignment["artefact_hash"]
        and preflight["source_commit_hash"] == artifact.get("source_commit_hash")
        and preflight["engine_version"] == artifact.get("engine_version")
        and age is not None
        and 0 <= age <= int(product.get("preflight_max_age_seconds", 3600))
        and isinstance(payload, Mapping)
        and payload.get("schema") == "platform.production-preflight/v1"
        and payload.get("instrument_id") == instrument_id
        and payload.get("sleeve_id") == assignment["sleeve_id"]
        and payload.get("environment") == (account.get("environment") if account else None)
        and payload.get("account_fingerprint") == expected_fingerprint
        and payload.get("execution_engine_identity") == engine_identity
        and payload.get("configuration_hash") == configuration_hash
        and preflight.get("content_hash") == canonical_hash(dict(payload))
    )


def _record_age(now: str, observed_at: str) -> float | None:
    try:
        return (
            dt.datetime.fromisoformat(now) - dt.datetime.fromisoformat(observed_at)
        ).total_seconds()
    except ValueError:
        return None


def _live_account_snapshot_checks(
    *,
    connection: Any,
    product: Mapping[str, Any],
    product_id: str,
    account_id: str,
    now: str,
    expected_fingerprint: str,
    actual_fingerprint: str,
    fingerprint_error: str,
) -> dict[str, Any]:
    snapshot = _latest_live_account_snapshot(connection, product_id, account_id, now)
    payload = snapshot["payload"] if snapshot else None
    values = payload if isinstance(payload, Mapping) else {}
    shape = _account_shape(values)
    content_valid = bool(
        shape and snapshot and snapshot.get("content_hash") == canonical_hash(dict(values))
    )
    identity_valid = bool(
        shape
        and snapshot
        and values.get("account_id") == account_id
        and values.get("product_id") == product_id
        and values.get("observed_at") == snapshot.get("observed_at")
    )
    age = _record_age(now, str(snapshot["observed_at"])) if snapshot else None
    snapshot_valid = bool(
        content_valid
        and identity_valid
        and values.get("account_state_known") is True
        and values.get("account_state_authority")
        in {"authenticated_rest", "authenticated_reconciled"}
        and snapshot
        and snapshot.get("source") in {"authenticated_rest", "authenticated_reconciled"}
        and values.get("unknown_exposure") == {}
        and values.get("account_fingerprint")
        and age is not None
        and 0 <= age <= int(product.get("account_snapshot_max_age_seconds", 60))
    )
    return {
        "account_fingerprint_detail": {
            "expected": expected_fingerprint,
            "actual": actual_fingerprint,
            **({"error": fingerprint_error} if fingerprint_error else {}),
        },
        "account_snapshot_detail": {
            "shape": shape,
            "content_hash": content_valid,
            "identity": identity_valid,
            "unknown_exposure": values.get("unknown_exposure") if shape else None,
        },
        "account_snapshot": snapshot_valid,
    }


def _account_shape(values: Mapping[str, Any]) -> bool:
    required = {
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
    return bool(
        values
        and required.issubset(values)
        and isinstance(values.get("balances"), Mapping)
        and isinstance(values.get("free_balances"), Mapping)
        and isinstance(values.get("positions"), Mapping)
        and isinstance(values.get("regular_orders"), list)
        and isinstance(values.get("conditional_orders"), list)
        and isinstance(values.get("unknown_exposure"), Mapping)
        and isinstance(values.get("account_mode"), str)
        and bool(str(values.get("account_mode") or "").strip())
        and all(
            _finite_number(values.get(field))
            for field in (
                "used_margin",
                "maintenance_margin",
                "used_margin_fraction",
                "liquidation_buffer_fraction",
            )
        )
    )


def _finite_number(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(str(value)))
    except (TypeError, ValueError):
        return False


def _live_rehearsal_checks(
    *,
    connection: Any,
    product: Mapping[str, Any],
    product_id: str,
    account_id: str,
    assignment: Mapping[str, Any],
    now: str,
    engine_identity: str,
) -> dict[str, Any]:
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
    rehearsal_assignment = assignment
    if (
        isinstance(report_payload, Mapping)
        and report_payload.get("assignment_id") != assignment["id"]
        and assignment.get("lifecycle_state") == "live"
    ):
        assignment_payload = assignment.get("payload")
        canary_evidence_id = (
            assignment_payload.get("canary_evidence_id")
            if isinstance(assignment_payload, Mapping)
            else None
        )
        if canary_evidence_id:
            canary_report = (
                connection.execute(
                    select(platform_rehearsal_report).where(
                        platform_rehearsal_report.c.id == str(canary_evidence_id)
                    )
                )
                .mappings()
                .first()
            )
            if canary_report is not None:
                report = canary_report
                report_payload = report["payload"]
                canary_assignment_id = (
                    report_payload.get("assignment_id")
                    if isinstance(report_payload, Mapping)
                    else None
                )
                canary_assignment = (
                    connection.execute(
                        select(active_strategy_assignment).where(
                            active_strategy_assignment.c.id == str(canary_assignment_id)
                        )
                    )
                    .mappings()
                    .first()
                    if canary_assignment_id
                    else None
                )
                if canary_assignment is not None:
                    rehearsal_assignment = dict(canary_assignment)
    report_age = _record_age(now, str(report["created_at"])) if report else None
    integrity = _rehearsal_report_integrity(
        report=report,
        payload=report_payload,
        product=product,
        product_id=product_id,
        account_id=account_id,
        assignment=rehearsal_assignment,
        now=now,
        engine_identity=engine_identity,
        report_age=report_age,
    )
    return {
        "connected_testnet_rehearsal": integrity,
        "connected_testnet_rehearsal_detail": {
            "report_id": report["id"] if report else None,
            "age_seconds": report_age,
            "integrity": integrity,
        },
    }


def _rehearsal_report_integrity(
    *,
    report: Mapping[str, Any] | None,
    payload: Any,
    product: Mapping[str, Any],
    product_id: str,
    account_id: str,
    assignment: Mapping[str, Any],
    now: str,
    engine_identity: str,
    report_age: float | None,
) -> bool:
    if report is None or not isinstance(payload, Mapping):
        return False
    unsigned = dict(payload)
    declared_hash = unsigned.pop("report_hash", None)
    signature = unsigned.pop("signature", None)
    signature_valid = _rehearsal_signature_valid(declared_hash, signature)
    protective_stop_applicable = payload.get(
        "protective_stop_applicable", product_id == "active_income"
    )
    stop_lifecycle = _rehearsal_stop_lifecycle(payload, applicable=protective_stop_applicable)
    return bool(
        report["accepted"]
        and report["content_hash"] == declared_hash == canonical_hash(unsigned)
        and report["id"] == declared_hash
        and signature_valid
        and report_age is not None
        and 0 <= report_age <= int(product.get("connected_testnet_max_age_seconds", 86_400))
        and payload.get("environment") == "testnet"
        and payload.get("real_exchange") is True
        and payload.get("product_id") == product_id
        and payload.get("account_id") == account_id
        and payload.get("assignment_id") == assignment["id"]
        and payload.get("artefact_hash") == assignment["artefact_hash"]
        and payload.get("execution_engine_identity") == engine_identity
        and payload.get("open_acknowledged") is True
        and payload.get("close_acknowledged") is True
        and payload.get("user_stream_fill") is True
        and payload.get("accounting_reconciled") is True
        and payload.get("flat_reconciliation") is True
        and payload.get("risk_accepted") is True
        and payload.get("risk_scopes")
        == ["strategy", "instrument", "sleeve", "product", "account", "global"]
        and all(
            isinstance(payload.get(field), str) and payload[field].startswith("sha256:")
            for field in ("forecast_id", "target_position_snapshot_id", "risk_assessment_id")
        )
        and isinstance(payload.get("recovery_identifiers"), Mapping)
        and payload.get("recovery_lookup") is True
        and stop_lifecycle
        and payload.get("emergency_reduction_tested") is True
        and _valid_account_fingerprint(payload.get("account_fingerprint"))
    )


def _rehearsal_stop_lifecycle(payload: Mapping[str, Any], *, applicable: Any) -> bool:
    if applicable is not True:
        return payload.get("protective_stop_not_applicable") is True
    return all(
        payload.get(field) is True
        for field in (
            "native_stop_created",
            "native_stop_readback_matched",
            "algo_update_processed",
            "partial_fill_quantity_handled",
            "stop_resize_or_replacement_handled",
            "native_stop_cancelled",
        )
    )


def _rehearsal_signature_valid(declared_hash: Any, signature: Any) -> bool:
    signing_key = os.environ.get("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY", "")
    if not signing_key or not isinstance(signature, str) or not signature:
        return False
    if not isinstance(declared_hash, str):
        return False
    expected = hmac.new(signing_key.encode(), declared_hash.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def _valid_account_fingerprint(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(ACCOUNT_FINGERPRINT_PREFIX):
        return False
    suffix = value.removeprefix(ACCOUNT_FINGERPRINT_PREFIX)
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)


def _readiness_configuration(
    *, config_path: Path, live: bool, product_id: str | None
) -> tuple[Any, Mapping[str, Any], list[dict[str, Any]]]:
    config = load_platform_config(config_path)
    split = load_split_configuration(config_path.parent)
    selected_products = [
        product
        for product in split["products"]["products"]
        if product_id is None or str(product["product_id"]) == product_id
    ]
    expected_mode = "live" if live else "paper"
    mode_name = "products_execution_configured" if live else "products_paper_only"
    checks = [
        _check("platform_configuration", True),
        _check(
            mode_name,
            all(product.get("execution_mode") == expected_mode for product in selected_products),
            detail={
                str(product["product_id"]): product.get("execution_mode")
                for product in split["products"]["products"]
            },
        ),
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
        ),
    ]
    if live:
        alerting = config.alerting
        sink = str(alerting.get("sink") or "").strip().lower()
        webhook_configured = bool(os.environ.get("TRADING_PLATFORM_ALERT_WEBHOOK_URL", "").strip())
        checks.append(
            _check(
                "external_alert_delivery_configured",
                not bool(alerting.get("enabled", True)) or sink != "webhook" or webhook_configured,
                detail={
                    "enabled": bool(alerting.get("enabled", True)),
                    "sink": sink,
                    "webhook_configured": webhook_configured,
                },
            )
        )
    return config, split, checks


def _readiness_paths(config: Any, checks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = {}
    for name, raw_path in config.paths.items():
        path = Path(raw_path)
        ok, reason = _regular_directory(path)
        paths[name] = {
            "path": str(path),
            "ok": ok,
            "reason": reason,
        }
        checks.append(_check(f"path:{name}", ok, detail=paths[name]))
    return paths


def _database_readiness_checks(
    *,
    database: PlatformDatabase,
    config: Any,
    split: Mapping[str, Any],
    live: bool,
    product_id: str | None,
    current: str,
) -> list[dict[str, Any]]:
    if not database.is_postgresql:
        raise RuntimeError("platform readiness requires PostgreSQL")
    database.assert_migrated()
    checks: list[dict[str, Any]] = [_check("postgresql_authority", True)]
    products = {
        str(product["product_id"]): dict(product) for product in split["products"]["products"]
    }
    policies = portfolio_state_policies({"risk": split["risk"]}, products)
    selected_products = [
        product
        for product in split["products"]["products"]
        if product_id is None or str(product["product_id"]) == product_id
    ]
    required_products = {str(product["product_id"]) for product in selected_products}
    inventory = _readiness_inventory(
        database,
        current=current,
        maximum_heartbeat_age=float(config.metrics.get("stale_after_seconds", 60)),
        required_products=required_products,
    )
    state_details = _state_authority_details(
        database, selected_products, policies, live=live, current=current
    )
    checks.append(
        _check(
            "canonical_portfolio_state_authority",
            all("error" not in detail for detail in state_details.values()),
            detail=state_details,
        )
    )
    checks.append(_check("canonical_tables", True, detail=inventory["tables"]))
    checks.append(
        _check(
            "autonomous_scheduler_authority",
            inventory["scheduler"]["ok"],
            detail=inventory["scheduler"],
        )
    )
    checks.append(
        _check(
            "canonical_dataset_role_authority",
            inventory["datasets"]["ok"],
            detail=inventory["datasets"],
        )
    )
    checks.append(
        _check(
            "research_progress_authority",
            inventory["progress"]["ok"],
            detail=inventory["progress"],
        )
    )
    account_readiness = _account_snapshot_readiness(
        database=database,
        products=selected_products,
        live=live,
        current=current,
    )
    checks.append(
        _check("account_snapshot_authority", account_readiness["ok"], detail=account_readiness)
    )
    required_manifests = len(split["products"]["products"])
    checks.append(
        _check(
            "bootstrap_manifest_authority",
            inventory["counts"]["feature_manifests"] >= required_manifests
            and inventory["counts"]["cost_model_manifests"] >= required_manifests,
            detail={
                "feature_manifests": inventory["counts"]["feature_manifests"],
                "cost_model_manifests": inventory["counts"]["cost_model_manifests"],
                "required": required_manifests,
            },
        )
    )
    checks.append(
        _check(
            "scheduler_and_reconciliation_heartbeats",
            inventory["heartbeats"]["ok"],
            detail=inventory["heartbeats"],
        )
    )
    operational = _operational_readiness(
        database=database,
        config=config,
        required_products=required_products,
        products=products,
        current=current,
    )
    checks.extend(_check(name, detail["ok"], detail=detail) for name, detail in operational.items())
    if live:
        checks.append(_live_readiness_check(database, split, selected_products, current))
    return checks


def _readiness_inventory(
    database: PlatformDatabase,
    *,
    current: str,
    maximum_heartbeat_age: float,
    required_products: set[str],
) -> dict[str, Any]:
    with database.engine.connect() as connection:
        table_names = set(inspect(database.engine).get_table_names())
        count_tables = (
            ("universe_snapshots", universe_snapshot),
            ("dataset_snapshots", dataset_snapshot),
            ("dataset_bundles", dataset_bundle),
            ("feature_manifests", feature_manifest),
            ("cost_model_manifests", cost_model_manifest),
            ("experiments", experiment),
            ("schedules", platform_schedule),
            ("account_snapshots", account_snapshot),
            ("historical_bootstrap_progress", historical_bootstrap_progress),
            ("alert_delivery_attempts", alert_delivery_attempt),
            ("quarantined_jobs", job_quarantine),
            ("jobs", job),
        )
        counts = {
            name: int(connection.execute(select(func.count()).select_from(table)).scalar_one())
            for name, table in count_tables
        }
        bundle_rows = connection.execute(select(dataset_bundle.c.payload)).scalars().all()
        snapshot_rows = (
            connection.execute(select(dataset_snapshot.c.id, dataset_snapshot.c.payload))
            .mappings()
            .all()
        )
        schedule_rows = connection.execute(select(platform_schedule)).mappings().all()
        activity_rows = connection.execute(
            select(strategy_definition.c.product_id, experiment.c.submitted_at).select_from(
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
            select(strategy_definition.c.product_id, validation_stage.c.evaluated_at).select_from(
                validation_stage.join(
                    experiment,
                    validation_stage.c.experiment_id == experiment.c.id,
                )
                .join(
                    strategy_version,
                    experiment.c.strategy_version_id == strategy_version.c.id,
                )
                .join(
                    strategy_definition,
                    strategy_version.c.definition_id == strategy_definition.c.id,
                )
            )
        ).all()
        heartbeat_rows = (
            connection.execute(
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
            )
            .mappings()
            .all()
        )
    ready_roles = _ready_dataset_roles(bundle_rows, snapshot_rows)
    return {
        "tables": {"count": len(table_names), "rows": counts},
        "counts": counts,
        "scheduler": _scheduler_readiness(
            schedule_rows, current=current, maximum_heartbeat_age=maximum_heartbeat_age
        ),
        "datasets": _dataset_readiness(ready_roles, counts, required_products),
        "progress": _progress_readiness(
            activity_rows, stage_activity_rows, required_products, current=current
        ),
        "heartbeats": _heartbeat_readiness(
            heartbeat_rows, current=current, maximum_age=maximum_heartbeat_age
        ),
    }


def _operational_readiness(
    *,
    database: PlatformDatabase,
    config: Any,
    required_products: set[str],
    products: Mapping[str, Mapping[str, Any]],
    current: str,
) -> dict[str, dict[str, Any]]:
    with database.engine.connect() as connection:
        jobs = (
            connection.execute(select(job.c.name, job.c.state, job.c.available_at)).mappings().all()
        )
        bootstrap = (
            connection.execute(
                select(
                    historical_bootstrap_progress.c.product_id,
                    historical_bootstrap_progress.c.market,
                    historical_bootstrap_progress.c.symbol,
                    historical_bootstrap_progress.c.timeframe,
                    historical_bootstrap_progress.c.range_start,
                    historical_bootstrap_progress.c.range_end,
                    historical_bootstrap_progress.c.status,
                    historical_bootstrap_progress.c.updated_at,
                )
            )
            .mappings()
            .all()
        )
        delivery_attempt_rows = (
            connection.execute(
                select(
                    alert_delivery_attempt.c.alert_id,
                    alert_delivery_attempt.c.sink,
                    alert_delivery_attempt.c.attempt_no,
                    alert_delivery_attempt.c.status,
                    alert_delivery_attempt.c.acknowledged_at,
                ).order_by(
                    alert_delivery_attempt.c.attempt_no,
                    alert_delivery_attempt.c.id,
                )
            )
            .mappings()
            .all()
        )
    latest_delivery: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in delivery_attempt_rows:
        latest_delivery[(str(row["alert_id"]), str(row["sink"]))] = row
    failed_deliveries = sum(
        row["status"] in {"failed", "dead_letter"} and row["acknowledged_at"] is None
        for row in latest_delivery.values()
    )
    data_root = Path(config.paths["parquet"])
    spool = data_root.parent / "runtime" / "events" / "spool"
    current_epoch = dt.datetime.fromisoformat(current).timestamp()
    stale_spool = (
        tuple(
            str(path)
            for path in spool.rglob("segment-*.jsonl")
            if path.is_file() and path.stat().st_mtime < current_epoch - 7 * 86_400
        )
        if spool.is_dir()
        else ()
    )
    storage_files = (
        sum(
            1
            for kind in ("raw", "bars", "features", "funding", "metadata")
            for path in (data_root / kind).rglob("*.parquet")
            if path.is_file()
        )
        if data_root.is_dir()
        else 0
    )
    progress_by_product: dict[str, list[Mapping[str, Any]]] = {}
    for row in bootstrap:
        progress_by_product.setdefault(str(row["product_id"]), []).append(row)
    with database.engine.connect() as connection:
        expected_symbols_by_product = {
            product_id: _expected_bootstrap_symbols(
                connection,
                product=products.get(product_id, {}),
                current=current,
            )
            for product_id in required_products
        }
    completed_products = {
        product_id
        for product_id in required_products
        if _history_progress_complete(
            progress_by_product.get(product_id, ()),
            product=products.get(product_id, {}),
            expected_symbols=expected_symbols_by_product[product_id],
        )
    }
    research_names = {
        "dataset_snapshot_validate",
        "register_strategy_catalogue",
        "generate_hypotheses",
        "register_candidate",
        "evaluate_candidate",
        "bounded_backtest",
        "event_replay",
        "register_ml_candidate",
        "train_ml_experiment",
        "historical_feature_calculation",
        "historical_bootstrap",
    }
    pending_research = [
        str(row["name"])
        for row in jobs
        if row["state"] in {"pending", "running"} and str(row["name"]) in research_names
    ]
    dead_research = [
        str(row["name"])
        for row in jobs
        if row["state"] == "dead_letter" and str(row["name"]) in research_names
    ]
    maximum_queue_age = int(config.resource_limits.get("maximum_queue_age_seconds", 86_400))
    current_datetime = dt.datetime.fromisoformat(current)
    stale_jobs = []
    for row in jobs:
        if row["state"] not in {"pending", "running"}:
            continue
        available_at = dt.datetime.fromisoformat(str(row["available_at"]))
        age_seconds = (current_datetime - available_at).total_seconds()
        if available_at <= current_datetime and age_seconds > maximum_queue_age:
            stale_jobs.append(
                {
                    "job_id": str(row["id"]),
                    "name": str(row["name"]),
                    "state": str(row["state"]),
                    "age_seconds": age_seconds,
                }
            )
    emergency = [
        str(row["name"])
        for row in jobs
        if row["state"] in {"pending", "running"}
        and str(row["name"])
        in {
            "emergency_flatten",
            "emergency_reduction",
            "cancel_entry_orders",
        }
    ]
    configured_topology_ok = all(
        "@" not in process and all("@" not in service for service in services)
        for process, services in PLATFORM_PROCESS_SERVICES.items()
    ) and {"market-gateway", "data-writer", "trading-runtime", "scheduler-operations"}.issubset(
        PLATFORM_PROCESS_SERVICES
    )
    systemd = _systemd_topology()
    topology_ok = configured_topology_ok and systemd["ok"]
    storage = _storage_metrics(data_root, spool)
    metadata_files = (
        sum(
            1
            for path in (data_root / "metadata").rglob("metadata-*.json")
            if path.is_file() and not path.is_symlink()
        )
        if (data_root / "metadata").is_dir()
        else 0
    )
    metadata_products = {
        product_id
        for product_id, symbols in expected_symbols_by_product.items()
        if symbols
        and all(
            any(
                path.is_file() and not path.is_symlink()
                for path in (data_root / "metadata" / "binance" / market / symbol).glob(
                    "metadata-*.json"
                )
            )
            for market, symbol in (
                (
                    "spot"
                    if str(products[product_id].get("product_id")) == "btc_accumulation"
                    else "futures",
                    symbol,
                )
                for symbol in symbols
            )
        )
    }
    backup_root = Path(config.paths["backups"])
    parquet_backups = (
        tuple(
            path
            for path in backup_root.glob("*/parquet.tar.gz")
            if path.is_file() and not path.is_symlink()
        )
        if backup_root.is_dir()
        else ()
    )
    newest_parquet_backup_age = (
        max(0.0, current_epoch - max(path.stat().st_mtime for path in parquet_backups))
        if parquet_backups
        else None
    )
    compact_backup = newest_parquet_backup_age is not None and newest_parquet_backup_age <= int(
        config.backup["maximum_age_seconds"]
    )
    return {
        "service_topology": {
            "ok": topology_ok,
            "processes": sorted(PLATFORM_PROCESS_SERVICES),
            "systemd": systemd,
        },
        "storage_health": {
            "ok": not stale_spool and storage["stale_spool_files"] == 0,
            "active_parquet_files": storage_files,
            "stale_spool": list(stale_spool),
            **storage,
        },
        "research_queue_health": {
            "ok": not dead_research,
            "pending_research_jobs": len(pending_research),
            "dead_letter_research_jobs": dead_research,
        },
        "queue_health": {
            "ok": not stale_jobs,
            "maximum_age_seconds": maximum_queue_age,
            "stale_jobs": stale_jobs,
        },
        "historical_bootstrap": {
            "ok": required_products.issubset(completed_products)
            and required_products.issubset(metadata_products),
            "required_products": sorted(required_products),
            "completed_products": sorted(completed_products),
            "metadata_files": metadata_files,
            "metadata_products": sorted(metadata_products),
        },
        "compact_backup": {
            "ok": compact_backup,
            "backup_root": str(backup_root),
            "latest_age_seconds": newest_parquet_backup_age,
            "maximum_age_seconds": int(config.backup["maximum_age_seconds"]),
        },
        "alert_delivery": {"ok": failed_deliveries == 0, "failed_attempts": failed_deliveries},
        "emergency_actions": {"ok": not emergency, "pending_jobs": emergency},
        "lifecycle_authority": _lifecycle_readiness(
            database,
            required_products=required_products,
            current=current,
        ),
    }


def _lifecycle_readiness(
    database: PlatformDatabase,
    *,
    required_products: set[str],
    current: str,
) -> dict[str, Any]:
    """Ensure assignments and promotion events expose one lifecycle state."""

    repository = SqlActiveStrategyAssignmentRepository(database.engine)
    assignments = tuple(
        row
        for product_id in sorted(required_products)
        for row in repository.active_assignments(product_id, at=current)
    )
    with database.engine.connect() as connection:
        events = {
            str(row["id"]): row["payload"]
            for row in connection.execute(select(promotion_event)).mappings()
            if isinstance(row["payload"], Mapping)
        }
        assignment_event_ids = {
            str(payload.get("promotion_event_id"))
            for row in connection.execute(select(active_strategy_assignment.c.payload)).scalars()
            if isinstance(payload := row, Mapping) and payload.get("promotion_event_id")
        }
    failures: list[dict[str, Any]] = []
    checked = 0
    for row in assignments:
        payload = row.get("payload")
        if isinstance(payload, Mapping) and payload.get("diagnostic") is True:
            continue
        lifecycle = str(row.get("lifecycle_state") or "")
        if lifecycle not in {"forward_paper", "live_ready", "live_canary", "live"}:
            continue
        checked += 1
        event_id = (
            str(payload.get("promotion_event_id") or "") if isinstance(payload, Mapping) else ""
        )
        event = events.get(event_id)
        if not event or (
            str(event.get("strategy_version_id")) != str(row["strategy_version_id"])
            or str(event.get("next_state")) != lifecycle
            or event.get("accepted") is not True
        ):
            failures.append(
                {
                    "assignment_id": str(row["id"]),
                    "lifecycle_state": lifecycle,
                    "promotion_event_id": event_id or None,
                    "reason": "assignment_and_promotion_event_diverged",
                }
            )
    for event_id, event in events.items():
        if (
            event.get("accepted") is True
            and str(event.get("prior_state") or "") != str(event.get("next_state") or "")
            and str(event.get("next_state") or "")
            in {"forward_paper", "live_ready", "live_canary", "live"}
            and str(event.get("product_id") or "") in required_products
            and event_id not in assignment_event_ids
        ):
            failures.append(
                {
                    "promotion_event_id": event_id,
                    "lifecycle_state": str(event.get("next_state")),
                    "reason": "promotion_event_has_no_assignment",
                }
            )
    return {
        "ok": not failures,
        "checked_assignments": checked,
        "divergences": failures,
    }


def _history_progress_complete(
    rows: Iterable[Mapping[str, Any]],
    *,
    product: Mapping[str, Any],
    expected_symbols: set[str],
) -> bool:
    latest_by_range: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        identity = (
            str(row.get("market")),
            str(row.get("symbol")),
            str(row.get("timeframe")),
            str(row.get("range_start")),
            str(row.get("range_end")),
        )
        previous = latest_by_range.get(identity)
        if previous is None or str(row.get("updated_at")) > str(previous.get("updated_at")):
            latest_by_range[identity] = row
    materialised = tuple(latest_by_range.values())
    if not materialised or any(row.get("status") != "completed" for row in materialised):
        return False
    try:
        minimum_history = dt.timedelta(days=365)
        if any(
            dt.datetime.fromisoformat(str(row["range_end"]))
            - dt.datetime.fromisoformat(str(row["range_start"]))
            < minimum_history
            for row in materialised
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    bars = {
        str(row.get("symbol")) for row in materialised if str(row.get("timeframe")) != "funding"
    }
    funding = {
        str(row.get("symbol")) for row in materialised if str(row.get("timeframe")) == "funding"
    }
    has_bars = bool(expected_symbols) and expected_symbols.issubset(bars)
    account_market = str(product.get("market") or "")
    requires_funding = product.get("product_id") == "active_income" or account_market in {
        "futures",
        "usdt_futures",
    }
    return has_bars and (not requires_funding or expected_symbols.issubset(funding))


def _expected_bootstrap_symbols(
    connection: Any, *, product: Mapping[str, Any], current: str
) -> set[str]:
    product_id = str(product.get("product_id") or "")
    if product_id == "btc_accumulation":
        return {"BTCUSDT"}
    universe_id = str(product.get("universe_id") or "")
    if not universe_id:
        return set()
    snapshot_id = connection.execute(
        select(universe_snapshot.c.id)
        .where(
            universe_snapshot.c.universe_id == universe_id,
            universe_snapshot.c.observed_at <= current,
        )
        .order_by(universe_snapshot.c.observed_at.desc(), universe_snapshot.c.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if snapshot_id is None:
        return set()
    return {
        str(value)
        for value in connection.execute(
            select(instrument_table.c.exchange_symbol)
            .select_from(
                universe_member.join(
                    instrument_table,
                    universe_member.c.instrument_id == instrument_table.c.id,
                )
            )
            .where(
                universe_member.c.snapshot_id == snapshot_id,
                universe_member.c.eligible.is_(True),
            )
        ).scalars()
        if str(value).strip()
    }


def _account_snapshot_readiness(
    *,
    database: PlatformDatabase,
    products: list[Mapping[str, Any]],
    live: bool,
    current: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for product in products:
        product_id = str(product["product_id"])
        account_id = str(product["account_id"])
        with database.engine.connect() as connection:
            row = (
                connection.execute(
                    select(account_snapshot)
                    .where(
                        account_snapshot.c.account_id == account_id,
                        account_snapshot.c.observed_at <= current,
                    )
                    .order_by(account_snapshot.c.observed_at.desc(), account_snapshot.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        payload = row["payload"] if row is not None else None
        expected_source = "authenticated_rest" if live else "paper_ledger"
        age = _record_age(current, str(row["observed_at"])) if row is not None else None
        fresh = age is not None and 0 <= age <= float(
            product.get("account_snapshot_max_age_seconds", 60)
        )
        valid = (
            row is not None
            and isinstance(payload, Mapping)
            and payload.get("product_id") == product_id
            and payload.get("account_state_known") is True
            and payload.get("account_state_authority") == expected_source
            and fresh
        )
        details[product_id] = {
            "account_id": account_id,
            "source": payload.get("account_state_authority")
            if isinstance(payload, Mapping)
            else None,
            "age_seconds": age,
            "maximum_age_seconds": product.get("account_snapshot_max_age_seconds", 60),
            "fresh": fresh,
            "valid": valid,
        }
    return {
        "ok": bool(details) and all(item["valid"] for item in details.values()),
        "products": details,
    }


def _storage_metrics(data_root: Path, spool: Path) -> dict[str, Any]:
    feature_spool = spool.parent / "feature-spool"
    quarantine_roots = tuple(
        path
        for path in (data_root / "quarantine", spool.parent.parent / "quarantine" / "storage")
        if path.is_dir()
    )
    kinds = ("raw", "bars", "features", "funding", "metadata")
    active_files = {
        kind: sum(
            1
            for path in (data_root / kind).rglob(
                "metadata-*.json" if kind == "metadata" else "*.parquet"
            )
            if path.is_file() and not path.is_symlink()
        )
        if (data_root / kind).is_dir()
        else 0
        for kind in kinds
    }
    spool_by_state = {
        state: sum(
            1
            for path in (
                tuple((spool / state).glob("segment-*.jsonl"))
                + tuple((feature_spool / state).glob("segment-*.jsonl"))
            )
            if path.is_file() and not path.is_symlink()
        )
        if (spool / state).is_dir() or (feature_spool / state).is_dir()
        else 0
        for state in ("pending", "claimed", "committed", "deleted")
    }
    spool_files = [
        path
        for state in spool_by_state
        for path in (
            tuple((spool / state).glob("segment-*.jsonl"))
            + tuple((feature_spool / state).glob("segment-*.jsonl"))
        )
        if path.is_file() and not path.is_symlink()
    ]
    quarantine_files = [
        path
        for root in quarantine_roots
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    spool_ages = [time.time() - path.stat().st_mtime for path in spool_files]
    return {
        "active_files_by_kind": active_files,
        "active_inode_count": sum(active_files.values()),
        "active_file_count": sum(active_files.values()),
        "spool_files_by_state": spool_by_state,
        "spool_inode_count": len(spool_files),
        "spool_bytes": sum(path.stat().st_size for path in spool_files),
        "spool_oldest_age_seconds": max(spool_ages, default=0.0),
        "stale_spool_files": sum(
            path.stat().st_mtime < dt.datetime.now(dt.UTC).timestamp() - 7 * 86_400
            for path in spool_files
        ),
        "quarantine_file_count": len(quarantine_files),
        "quarantine_bytes": sum(path.stat().st_size for path in quarantine_files),
    }


def _systemd_topology() -> dict[str, Any]:
    if os.name != "posix" or not Path("/run/systemd/system").exists():
        return {"ok": True, "available": False, "reason": "systemd_not_active"}
    legacy_patterns = (
        "trading-platform@*.service",
        "trading-platform-research@*.service",
        "trading-bot-*.service",
        "trading-bot-*.timer",
    )
    unit_directories = (Path("/etc/systemd/system"), Path("/run/systemd/system"))
    installed = tuple(
        sorted(
            str(path)
            for directory in unit_directories
            for pattern in legacy_patterns
            for path in directory.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    )
    try:
        result = subprocess.run(
            ["systemctl", "list-units", "--all", "--no-legend", "--plain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "ok": not installed,
            "available": False,
            "installed_legacy_units": list(installed),
            "active_legacy_units": [],
        }
    active = tuple(
        sorted(
            line.split()[0]
            for line in result.stdout.splitlines()
            if line.split()
            and any(_unit_matches(line.split()[0], pattern) for pattern in legacy_patterns)
        )
    )
    return {
        "ok": not installed and not active,
        "available": result.returncode == 0,
        "installed_legacy_units": list(installed),
        "active_legacy_units": list(active),
    }


def _unit_matches(unit: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatchcase(unit, pattern)


def _ready_dataset_roles(
    rows: list[Any], snapshot_rows: list[Mapping[str, Any]]
) -> dict[str, set[str]]:
    snapshots = {
        str(row["id"]): row["payload"]
        for row in snapshot_rows
        if isinstance(row, Mapping) and isinstance(row.get("payload"), Mapping)
    }
    result: dict[str, set[str]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("lifecycle_state") != "ready":
            continue
        product_key = str(raw.get("product_id") or "")
        stages = raw.get("stage_snapshot_ids")
        if not product_key or not isinstance(stages, Mapping):
            continue
        stage_payloads = [snapshots.get(str(snapshot_id)) for snapshot_id in stages.values()]
        if len(stage_payloads) != len(stages) or any(
            dataset_payload_is_non_promotable(payload) for payload in stage_payloads
        ):
            continue
        result.setdefault(product_key, set()).update(str(role) for role in stages)
    return result


def _scheduler_readiness(
    rows: list[Mapping[str, Any]], *, current: str, maximum_heartbeat_age: float
) -> dict[str, Any]:
    required = {spec.name for spec in AUTONOMOUS_SCHEDULES}
    details: dict[str, Any] = {}
    fresh = True
    for row in rows:
        name = str(row["job_name"])
        updated_at = timestamp(str(row["updated_at"]), field=f"schedule.{name}.updated_at")
        age = _record_age(current, updated_at)
        maximum_age = max(maximum_heartbeat_age * 2.0, float(row["interval_seconds"]) * 2.0)
        is_fresh = age is not None and 0 <= age <= maximum_age
        fresh = fresh and is_fresh
        details[name] = {
            "state": str(row["state"]),
            "updated_at": updated_at,
            "age_seconds": age,
            "maximum_age_seconds": maximum_age,
            "fresh": is_fresh,
        }
    return {
        "ok": {str(row["job_name"]) for row in rows} == required and fresh,
        "schedule_rows": len(rows),
        "required_schedule_names": sorted(required),
        "schedule_details": details,
    }


def _dataset_readiness(
    ready_roles: Mapping[str, set[str]],
    counts: Mapping[str, int],
    required_products: set[str],
) -> dict[str, Any]:
    required_roles = {
        "screening",
        "development",
        "robustness",
        "protected_holdout",
    }
    details = {
        product: {
            "roles": sorted(ready_roles.get(product, set())),
            "missing_roles": sorted(required_roles - ready_roles.get(product, set())),
        }
        for product in sorted(required_products)
    }
    return {
        "ok": all(not detail["missing_roles"] for detail in details.values()),
        "dataset_bundles": counts["dataset_bundles"],
        "products": details,
    }


def _progress_readiness(
    activity_rows: list[Any],
    stage_rows: list[Any],
    required_products: set[str],
    *,
    current: str,
) -> dict[str, Any]:
    latest: dict[str, str] = {}
    for raw_product, observed in (*activity_rows, *stage_rows):
        product = str(raw_product)
        observed_at = timestamp(str(observed), field="research_activity.observed_at")
        if observed_at > latest.get(product, ""):
            latest[product] = observed_at
    details: dict[str, dict[str, Any]] = {}
    for product in sorted(required_products):
        observed = latest.get(product)
        details[product] = {
            "latest_activity_at": observed,
            "age_seconds": _record_age(current, observed) if observed is not None else None,
            "maximum_age_seconds": 86_400.0,
            "progressed": True,
        }
    for detail in details.values():
        detail["progressed"] = (
            detail["age_seconds"] is not None and 0 <= detail["age_seconds"] <= 86_400
        )
    return {
        "ok": bool(details) and all(item["progressed"] for item in details.values()),
        "products": details,
    }


def _heartbeat_readiness(
    rows: list[Mapping[str, Any]], *, current: str, maximum_age: float
) -> dict[str, Any]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if timestamp(str(row["observed_at"]), field="heartbeat.observed_at") > current:
            continue
        latest.setdefault(str(row["service_name"]), row)
    details: dict[str, Any] = {}
    ok = True
    for name in ("platform-scheduler", "account-reconciliation"):
        candidate = latest.get(name)
        if candidate is None:
            ok = False
            details[name] = "missing"
            continue
        age = _record_age(current, str(candidate["observed_at"]))
        healthy = bool(candidate["healthy"]) and age is not None and 0 <= age <= maximum_age
        ok = ok and healthy
        details[name] = {"age_seconds": age, "healthy": healthy}
    return {"ok": ok, "details": details}


def _state_authority_details(
    database: PlatformDatabase,
    products: list[Mapping[str, Any]],
    expected_policies: Mapping[str, Mapping[str, Any]],
    *,
    live: bool,
    current: str,
) -> dict[str, dict[str, Any]]:
    store = SqlRiskSnapshotStore(database.engine)
    result: dict[str, dict[str, Any]] = {}
    for product in products:
        product_id = str(product["product_id"])
        try:
            result[product_id] = _state_authority_detail(
                store, product_id, expected_policies[product_id], live=live, current=current
            )
        except Exception as exc:
            result[product_id] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _state_authority_detail(
    store: SqlRiskSnapshotStore,
    product_id: str,
    expected_policy: Mapping[str, Any],
    *,
    live: bool,
    current: str,
) -> dict[str, Any]:
    state_id, state = store.latest(
        kind="canonical_portfolio_risk_state", product_id=product_id, at=current
    )
    observed_at = timestamp(str(state["observed_at"]), field="state.observed_at")
    age = _record_age(current, observed_at)
    if age is None or age < 0:
        raise ValueError("canonical portfolio state timestamp is in the future")
    maximum_age = float(state["maximum_state_age_seconds"])
    readiness_age = maximum_age if live else max(maximum_age, 600.0)
    source_ages = _state_source_ages(store, state, product_id, readiness_age, current)
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
    if policy_hash != canonical_hash(expected_policy):
        raise ValueError("canonical state risk policy identity is invalid")
    if {str(item) for item in policy_ids} != {
        str(item) for item in expected_policy.get("risk_policy_ids", ())
    }:
        raise ValueError("canonical state risk policy IDs are invalid")
    if age > readiness_age:
        raise ValueError("canonical portfolio state is stale")
    return {
        "state_id": state_id,
        "age_seconds": age,
        "maximum_age_seconds": maximum_age,
        "readiness_maximum_age_seconds": readiness_age,
        "source_ages_seconds": source_ages,
        "risk_policy_ids": list(policy_ids),
        "risk_policy_hash": policy_hash,
    }


def _state_source_ages(
    store: SqlRiskSnapshotStore,
    state: Mapping[str, Any],
    product_id: str,
    readiness_age: float,
    current: str,
) -> dict[str, float]:
    source_ids = state.get("source_snapshot_ids")
    if (
        not isinstance(source_ids, dict)
        or set(source_ids) != DatabasePortfolioStateWorker.REQUIRED_SOURCES
    ):
        raise ValueError("canonical state source identities are incomplete")
    ages: dict[str, float] = {}
    observed: list[str] = []
    for source, source_id in source_ids.items():
        if (
            not isinstance(source_id, str)
            or not source_id.startswith("sha256:")
            or len(source_id) != 71
        ):
            raise ValueError(f"{source} source identity is invalid")
        payload = store.get(source_id)
        if payload.get("product_id") != product_id:
            raise ValueError(f"{source} source belongs to another product")
        if payload.get("kind") not in {source, f"{source}_snapshot"}:
            raise ValueError(f"{source} source has the wrong kind")
        observed_at = timestamp(
            str(payload.get("observed_at", payload.get("created_at"))),
            field=f"{source}.observed_at",
        )
        age = _record_age(current, observed_at)
        if age is None or age < 0:
            raise ValueError(f"{source} source timestamp is in the future")
        ages[source] = age
        observed.append(observed_at)
        if source in {"account", "balances", "market"} and age > readiness_age:
            raise ValueError(f"{source} source is stale")
    state_observed = timestamp(str(state["observed_at"]), field="state.observed_at")
    if observed and state_observed != max(observed):
        raise ValueError("canonical portfolio state is not at the latest source timestamp")
    return ages


def _live_readiness_check(
    database: PlatformDatabase,
    split: Mapping[str, Any],
    products: list[Mapping[str, Any]],
    current: str,
) -> dict[str, Any]:
    accounts = {str(item["account_id"]): dict(item) for item in split["accounts"]["accounts"]}
    policies = {str(item["policy_id"]): dict(item) for item in split["promotion"]["policies"]}
    with database.engine.connect() as connection:
        details = {
            str(product["product_id"]): _live_product_checks(
                connection=connection,
                product=dict(product),
                accounts=accounts,
                promotion_policies=policies,
                risk_configuration=split["risk"],
                now=current,
            )
            for product in products
        }
    return _check(
        "live_execution_authority",
        all(detail.get("ok") is True for detail in details.values()) and bool(details),
        detail=details,
    )


def build_readiness(
    config_path: Path = Path("config/platform.json"),
    *,
    live: bool = False,
    product_id: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    try:
        config, split, checks = _readiness_configuration(
            config_path=config_path,
            live=live,
            product_id=product_id,
        )
    except Exception as exc:
        return {
            "schema": "platform.readiness/v1",
            "ok": False,
            "checks": [
                _check("platform_configuration", False, detail=f"{type(exc).__name__}: {exc}")
            ],
        }

    paths = _readiness_paths(config, checks)

    current = timestamp(now or dt.datetime.now(dt.UTC), field="now")
    database = None
    try:
        database = PlatformDatabase(config.database_url())
        checks.extend(
            _database_readiness_checks(
                database=database,
                config=config,
                split=split,
                live=live,
                product_id=product_id,
                current=current,
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
