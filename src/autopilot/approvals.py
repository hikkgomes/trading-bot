"""Strategy approval ledger.

The live-trading gate is intentionally independent from environment variables.
``TRADING_LIVE=1`` can enable an exchange adapter, but a strategy artifact still
cannot be used for live trading until every strategy fingerprint is approved in
this ledger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.autopilot.config import (
    DEFAULT_CONFIG_PATH,
    ProductConfig,
    canonical_product_config,
    load_config,
)
from src.autopilot.exchange_policy import (
    ACTIVE_INCOME_FUTURES_EXCHANGES,
    BTC_ACCUMULATION_SPOT_EXCHANGES,
)
from src.autopilot.execution_identity import execution_engine_digest
from src.autopilot.io import write_json_atomic
from src.autopilot.locking import acquire_file_update_lock
from src.autopilot.strategy_policy import StrategyPolicyError, validate_strategy_artifact
from src.config import PROJECT_ROOT
from src.execution.config import ACCOUNT_FINGERPRINT_PREFIX

DEFAULT_APPROVAL_LEDGER = PROJECT_ROOT / "runtime" / "approvals.json"

BEHAVIOR_KEYS = (
    "id",
    "market",
    "symbol",
    "entry_type",
    "base_timeframe",
    "direction",
    "leverage",
    "margin_mode",
    "horizon_bars",
    "take_profit",
    "stop_loss",
    "use_atr_tp_sl",
    "pnl_unit",
    "conditions",
    "hypothesis",
    "risk",
    "fees",
    "baseline_win_rate",
)
AUTOMATION_APPROVAL_ACTOR_KEYS = {
    "agent",
    "automation",
    "autopilot",
    "bot",
    "ci",
    "codex",
    "cron",
    "github-actions",
    "github-actions[bot]",
    "robot",
    "scheduler",
    "service",
    "system",
    "trading-bot",
}
PRODUCTION_PREFLIGHT_CLOCK_SKEW_SECONDS = 300
PRODUCTION_PREFLIGHT_REQUIRED_CHECKS = (
    "product_config",
    "execution_engine_identity",
    "strategy_artifact_exists",
    "strategy_fingerprints",
    "strategy_policy",
    "exchange_environment",
    "broker_constructed",
    "exchange_read_connectivity",
)


class ApprovalError(RuntimeError):
    """Raised when a strategy artifact is not approved for live trading."""


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_approval_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ApprovalError(f"{field} must be a non-empty operator identifier.")
    actor = value.strip()
    if not actor:
        raise ApprovalError(f"{field} must be a non-empty operator identifier.")
    if _is_automation_approval_actor(actor):
        raise ApprovalError(
            f"{field} must identify a human operator, not automation identity {actor!r}."
        )
    return actor


def normalize_revocation_reason(value: str) -> str:
    if not isinstance(value, str):
        raise ApprovalError("revocation reason must be non-empty.")
    reason = value.strip()
    if not reason:
        raise ApprovalError("revocation reason must be non-empty.")
    return reason


def is_valid_approval_actor(value: Any) -> bool:
    return (
        isinstance(value, str) and bool(value.strip()) and not _is_automation_approval_actor(value)
    )


def is_valid_revocation_reason(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_automation_approval_actor(value: str) -> bool:
    key = value.strip().casefold().replace("_", "-").replace(" ", "-")
    return key in AUTOMATION_APPROVAL_ACTOR_KEYS or key.endswith("-bot") or key.endswith("[bot]")


def _revocation_audit_reasons(entry: dict[str, Any]) -> list[str]:
    reasons = []
    if not is_valid_approval_actor(entry.get("revoked_by")):
        reasons.append("invalid_revoked_by")
    if not is_valid_revocation_reason(entry.get("revocation_reason")):
        reasons.append("missing_revocation_reason")
    return reasons


def _display_status(fingerprint: str, entry: dict[str, Any]) -> str:
    status = str(entry.get("status") or "unknown")
    if status == "approved" and not is_valid_approval_actor(entry.get("approved_by")):
        return "invalid_actor"
    if status == "approved" and entry.get("fingerprint") != str(fingerprint):
        return "fingerprint_mismatch"
    if status == "revoked" and _revocation_audit_reasons(entry):
        return "invalid_revocation_audit"
    return status


def _list_entry_line(fingerprint: str, entry: dict[str, Any]) -> str:
    status = _display_status(fingerprint, entry)
    raw_status = str(entry.get("status") or "unknown")
    actor = entry.get("revoked_by") if raw_status == "revoked" else entry.get("approved_by")
    line = f"{status:24} {entry.get('strategy_id', '<unknown>')} {fingerprint} by={actor or '-'}"
    if raw_status == "revoked":
        reason = entry.get("revocation_reason") or "-"
        line += f" reason={reason}"
        audit_reasons = _revocation_audit_reasons(entry)
        if audit_reasons:
            line += f" audit={','.join(audit_reasons)}"
    return line


def strategy_behavior(strategy: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields that affect live trading behavior."""
    return {key: strategy[key] for key in BEHAVIOR_KEYS if key in strategy}


def strategy_fingerprint(strategy: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json(strategy_behavior(strategy)).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def artifact_digest(artifact: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ApprovalError(f"Strategy artifact must not be a symlink: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Strategy artifact not found: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApprovalError(f"{path} must be valid JSON: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ApprovalError(f"{path} must be a JSON object.")
    strategies = artifact.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ApprovalError(f"{path} has no strategies to approve.")
    bad_indexes = [
        index for index, strategy in enumerate(strategies) if not isinstance(strategy, dict)
    ]
    if bad_indexes:
        indexes = ", ".join(str(index) for index in bad_indexes)
        raise ApprovalError(f"{path} strategies must be JSON objects; invalid indexes: {indexes}.")
    return artifact


def _same_path(left: str | Path | None, right: Path) -> bool:
    if left is None:
        return False
    try:
        return Path(left).resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return False


def preflight_report_digest(path: Path) -> str:
    """Digest the exact reviewed preflight report bytes."""

    if path.is_symlink():
        raise ApprovalError(f"Production preflight report must not be a symlink: {path}")
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ApprovalError(f"Could not read production preflight report {path}: {exc}") from exc


def _positive_json_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _read_production_preflight(path: Path, product: ProductConfig) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ApprovalError(f"Production preflight report must not be a symlink: {path}")
    try:
        report_bytes = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(report_bytes).hexdigest()
        report = json.loads(report_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApprovalError(
            f"Production preflight report must be valid JSON: {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ApprovalError(f"Could not read production preflight report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("ok") is not True:
        raise ApprovalError(f"{product.name}: production preflight report is not successful.")
    return report, digest


def _validate_preflight_freshness(
    report: dict[str, Any], product: ProductConfig, *, require_fresh: bool
) -> Any:
    generated_ts = report.get("generated_ts")
    if not _positive_json_number(generated_ts):
        raise ApprovalError(f"{product.name}: production preflight generated_ts is invalid.")
    age_seconds = time.time() - float(generated_ts)
    if require_fresh and age_seconds < -PRODUCTION_PREFLIGHT_CLOCK_SKEW_SECONDS:
        raise ApprovalError(f"{product.name}: production preflight timestamp is in the future.")
    if require_fresh and age_seconds > product.preflight_max_age_seconds:
        raise ApprovalError(f"{product.name}: production preflight report is stale.")
    return generated_ts


def _matching_preflight_product(
    report: dict[str, Any], product: ProductConfig
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    products = report.get("products")
    if not isinstance(products, list):
        raise ApprovalError(f"{product.name}: production preflight products must be a list.")
    matches = [
        item
        for item in products
        if isinstance(item, dict)
        and isinstance(item.get("product"), dict)
        and item["product"].get("name") == product.name
    ]
    if len(matches) != 1:
        raise ApprovalError(
            f"{product.name}: production preflight must contain exactly one matching product."
        )
    matched = matches[0]
    if matched.get("ok") is not True:
        raise ApprovalError(f"{product.name}: product production preflight is not successful.")
    if matched.get("product") != canonical_product_config(product):
        raise ApprovalError(
            f"{product.name}: production preflight product identity does not match current config."
        )
    checks = matched.get("checks")
    if not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks):
        raise ApprovalError(f"{product.name}: production preflight checks are invalid.")
    checks_by_name = {
        item.get("name"): item
        for item in checks
        if isinstance(item.get("name"), str) and item.get("name")
    }
    return matched, checks_by_name


def _required_preflight_checks(product: ProductConfig) -> list[str]:
    required_checks = list(PRODUCTION_PREFLIGHT_REQUIRED_CHECKS)
    if product.objective == "active_income" and product.market == "futures":
        required_checks.extend(
            (
                "broker_position_mode_one_way",
                "broker_native_protective_stops",
                "broker_open_orders_empty",
                "broker_position_flat",
            )
        )
    if product.objective == "btc_accumulation" and product.market == "spot":
        required_checks.append("broker_spot_position_non_negative")
    return required_checks


def _validate_preflight_gates(
    matched: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
    product: ProductConfig,
    *,
    expected_artifact_digest: str | None,
) -> list[str]:
    failed = [
        name
        for name in _required_preflight_checks(product)
        if checks_by_name.get(name, {}).get("ok") is not True
    ]
    if failed:
        raise ApprovalError(
            f"{product.name}: production preflight missing successful checks: {', '.join(failed)}."
        )
    if matched.get("execution_engine_digest") != execution_engine_digest():
        raise ApprovalError(
            f"{product.name}: production preflight execution engine digest is not current."
        )
    if (
        expected_artifact_digest is not None
        and matched.get("artifact_digest") != expected_artifact_digest
    ):
        raise ApprovalError(
            f"{product.name}: production preflight artifact digest does not match the reviewed artifact."
        )
    fingerprints = matched.get("artifact_fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints:
        raise ApprovalError(f"{product.name}: production preflight has no artifact fingerprints.")
    return fingerprints


def _validate_preflight_exchange(
    checks_by_name: dict[str, dict[str, Any]], product: ProductConfig
) -> dict[str, Any]:
    exchange_check = checks_by_name["exchange_environment"]
    exchange = exchange_check.get("detail")
    if not isinstance(exchange, dict) or exchange.get("custom_checker"):
        raise ApprovalError(f"{product.name}: production preflight exchange evidence is invalid.")
    expected_market = "spot" if product.objective == "btc_accumulation" else "futures"
    allowed_exchanges = (
        BTC_ACCUMULATION_SPOT_EXCHANGES
        if expected_market == "spot"
        else ACTIVE_INCOME_FUTURES_EXCHANGES
    )
    account_fingerprint = exchange.get("account_fingerprint")
    fingerprint_digest = (
        account_fingerprint.removeprefix(ACCOUNT_FINGERPRINT_PREFIX)
        if isinstance(account_fingerprint, str)
        and account_fingerprint.startswith(ACCOUNT_FINGERPRINT_PREFIX)
        else ""
    )
    invalid_exchange = (
        str(exchange.get("exchange") or "").lower() not in allowed_exchanges
        or str(exchange.get("market_type") or "").lower() != expected_market
        or str(exchange.get("quote_asset") or "").upper() != "USDT"
        or exchange.get("testnet") is not False
        or exchange.get("require_testnet") is not False
        or len(fingerprint_digest) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint_digest)
        or not _positive_json_number(exchange.get("max_notional_usd"))
        or not _positive_json_number(exchange.get("max_fill_slippage_bps"))
    )
    if expected_market == "futures":
        leverage = exchange.get("max_futures_leverage")
        invalid_exchange = invalid_exchange or (
            isinstance(leverage, bool)
            or not isinstance(leverage, int)
            or leverage != 1
            or str(exchange.get("futures_margin_mode") or "").lower() != "isolated"
        )
    if invalid_exchange:
        raise ApprovalError(
            f"{product.name}: production preflight exchange evidence is incomplete or unsafe."
        )
    return exchange


def load_production_preflight_evidence(
    product: ProductConfig,
    *,
    expected_artifact_digest: str | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Validate and capture the exact connected production preflight for approval."""

    if product.execution_mode != "live":
        raise ApprovalError(f"{product.name}: final live approval requires execution_mode='live'.")
    path = product.preflight_report
    if path is None:
        raise ApprovalError(f"{product.name}: final live approval requires preflight_report.")
    report, digest = _read_production_preflight(path, product)
    generated_ts = _validate_preflight_freshness(report, product, require_fresh=require_fresh)
    matched, checks_by_name = _matching_preflight_product(report, product)
    _validate_preflight_gates(
        matched,
        checks_by_name,
        product,
        expected_artifact_digest=expected_artifact_digest,
    )
    exchange = _validate_preflight_exchange(checks_by_name, product)
    manifest = {"exchange_environment": exchange}
    manifest_digest = (
        "sha256:" + hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    )
    return {
        "manifest": manifest,
        "manifest_digest": manifest_digest,
        # Retained as audit provenance for the human review. These rotating
        # report fields are deliberately not part of approval identity.
        "source_report_path": str(path.resolve(strict=False)),
        "source_report_digest": digest,
        "source_generated_at": report.get("generated_at"),
        "source_generated_ts": generated_ts,
    }


def _current_production_preflight(
    product: ProductConfig | None, artifact_digest_value: str | None
) -> dict[str, Any] | None:
    if product is None or product.execution_mode != "live":
        return None
    try:
        return load_production_preflight_evidence(
            product,
            expected_artifact_digest=artifact_digest_value,
            require_fresh=False,
        )
    except ApprovalError as exc:
        raise ApprovalError(
            f"Live trading blocked; production preflight approval evidence is invalid: {exc}"
        ) from exc


def _basic_approval_failure(entry: Any, label: str) -> tuple[str, str] | None:
    if entry is None:
        return "missing", label
    if not isinstance(entry, dict):
        return "malformed", label
    if entry.get("status") != "approved":
        return "revoked", label
    if not is_valid_approval_actor(entry.get("approved_by")):
        return "invalid_actor", label
    return None


def _identity_approval_failure(
    entry: dict[str, Any],
    fingerprint: str,
    label: str,
    *,
    artifact_path: Path | None,
    artifact_digest_value: str | None,
    current_execution_engine_digest: str,
    product: ProductConfig | None,
    current_production_preflight: dict[str, Any] | None,
) -> tuple[str, str] | None:
    if entry.get("fingerprint") != fingerprint:
        return "fingerprint_mismatch", label
    if artifact_path is not None and not _same_path(entry.get("artifact_path"), artifact_path):
        return "artifact_mismatch", label
    if artifact_digest_value is not None and entry.get("artifact_digest") != artifact_digest_value:
        return (
            "artifact_content_mismatch",
            f"{label} approved={entry.get('artifact_digest') or '<missing>'} current={artifact_digest_value}",
        )
    if entry.get("execution_engine_digest") != current_execution_engine_digest:
        return (
            "execution_engine_mismatch",
            f"{label} approved={entry.get('execution_engine_digest') or '<missing>'} "
            f"current={current_execution_engine_digest}",
        )
    if not _entry_matches_product(entry, product):
        return "product_mismatch", label
    if current_production_preflight is not None and not _entry_matches_production_preflight(
        entry, current_production_preflight
    ):
        return "production_preflight_mismatch", label
    return None


def _raise_basic_approval_failures(failures: dict[str, list[str]]) -> None:
    if not any(failures[key] for key in ("missing", "revoked", "malformed", "invalid_actor")):
        return
    parts = []
    if failures["missing"]:
        parts.append("missing approval: " + ", ".join(failures["missing"]))
    if failures["revoked"]:
        parts.append("revoked/not approved: " + ", ".join(failures["revoked"]))
    if failures["malformed"]:
        parts.append("malformed approval: " + ", ".join(failures["malformed"]))
    if failures["invalid_actor"]:
        parts.append("invalid approval actor: " + ", ".join(failures["invalid_actor"]))
    raise ApprovalError("Live trading blocked; " + "; ".join(parts))


def _raise_identity_approval_failures(failures: dict[str, list[str]]) -> None:
    keys = (
        "artifact_mismatch",
        "artifact_content_mismatch",
        "product_mismatch",
        "fingerprint_mismatch",
        "execution_engine_mismatch",
        "production_preflight_mismatch",
    )
    if not any(failures[key] for key in keys):
        return
    labels = {
        "fingerprint_mismatch": "approval fingerprint mismatch",
        "artifact_mismatch": "approval artifact mismatch",
        "artifact_content_mismatch": "approval artifact content mismatch",
        "product_mismatch": "approval product mismatch",
        "execution_engine_mismatch": "approval execution engine mismatch",
        "production_preflight_mismatch": "approval production preflight mismatch",
    }
    parts = [
        f"{labels[key]}: {', '.join(failures[key])}"
        for key in (
            "fingerprint_mismatch",
            "artifact_mismatch",
            "artifact_content_mismatch",
            "product_mismatch",
            "execution_engine_mismatch",
            "production_preflight_mismatch",
        )
        if failures[key]
    ]
    raise ApprovalError("Live trading blocked; " + "; ".join(parts))


def _entry_matches_production_preflight(
    entry: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    approved = entry.get("production_preflight")
    if not isinstance(approved, dict):
        return False
    return approved.get("manifest") == current.get("manifest") and approved.get(
        "manifest_digest"
    ) == current.get("manifest_digest")


def _product_approval_payload(product: ProductConfig | None) -> dict[str, Any] | None:
    if product is None:
        return None
    return canonical_product_config(product)


def _entry_matches_product(entry: dict[str, Any], product: ProductConfig | None) -> bool:
    if product is None:
        return True
    approved_product = entry.get("product")
    if not isinstance(approved_product, dict):
        return False
    return approved_product == canonical_product_config(product)


def _entry_event(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry.get("revoked_at"):
        event = "revoked"
        event_at = entry.get("revoked_at")
        actor = entry.get("revoked_by")
    elif entry.get("approved_at"):
        event = "approved"
        event_at = entry.get("approved_at")
        actor = entry.get("approved_by")
    else:
        return None
    return {
        "event": event,
        "event_at": event_at,
        "actor": actor,
        "status": entry.get("status"),
        "strategy_id": entry.get("strategy_id"),
        "artifact_path": entry.get("artifact_path"),
        "artifact_digest": entry.get("artifact_digest"),
        "execution_engine_digest": entry.get("execution_engine_digest"),
        "product": entry.get("product"),
        "production_preflight": entry.get("production_preflight"),
        "revocation_reason": entry.get("revocation_reason"),
    }


def _entry_history(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(entry, dict):
        return []
    history = [item for item in entry.get("history", []) if isinstance(item, dict)]
    event = _entry_event(entry)
    if event is not None:
        history.append(event)
    return history


@dataclass
class ApprovalLedger:
    path: Path = DEFAULT_APPROVAL_LEDGER

    def _reject_symlink(self) -> None:
        if self.path.is_symlink():
            raise ApprovalError(f"Approval ledger must not be a symlink: {self.path}")

    def load(self) -> dict[str, Any]:
        self._reject_symlink()
        if not self.path.exists():
            return {"version": 1, "approvals": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ApprovalError(f"Approval ledger must be valid JSON: {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ApprovalError(f"Approval ledger must be a JSON object: {self.path}")
        payload.setdefault("version", 1)
        payload.setdefault("approvals", {})
        if not isinstance(payload["approvals"], dict):
            raise ApprovalError(f"Approval ledger approvals must be a JSON object: {self.path}")
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self._reject_symlink()
        write_json_atomic(self.path, payload)

    def approve(
        self,
        strategy: dict[str, Any],
        *,
        artifact_path: Path,
        approved_by: str,
        product: ProductConfig | None = None,
        artifact: dict[str, Any] | None = None,
        notes: str = "",
    ) -> str:
        return self.approve_many(
            [strategy],
            artifact_path=artifact_path,
            approved_by=approved_by,
            product=product,
            artifact=artifact,
            notes=notes,
        )[0]

    def approve_many(
        self,
        strategies: Iterable[dict[str, Any]],
        *,
        artifact_path: Path,
        approved_by: str,
        product: ProductConfig | None = None,
        artifact: dict[str, Any] | None = None,
        notes: str = "",
    ) -> list[str]:
        if artifact_path.is_symlink():
            raise ApprovalError(f"Strategy artifact must not be a symlink: {artifact_path}")
        actor = normalize_approval_actor(approved_by, field="approved_by")
        selected = list(strategies)
        if not selected:
            raise ApprovalError("at least one strategy is required for approval")
        current_artifact = artifact
        if current_artifact is None and artifact_path.exists():
            current_artifact = load_artifact(artifact_path)
        product_payload = _product_approval_payload(product)
        current_artifact_digest = (
            artifact_digest(current_artifact) if current_artifact is not None else None
        )
        production_preflight = None
        if product is not None and product.execution_mode == "live":
            production_preflight = load_production_preflight_evidence(
                product,
                expected_artifact_digest=current_artifact_digest,
            )
        engine_digest = execution_engine_digest()
        approved_at = utc_now()
        fingerprints: list[str] = []
        with acquire_file_update_lock(self.path, label="approval ledger"):
            payload = self.load()
            for strategy in selected:
                fingerprint = strategy_fingerprint(strategy)
                previous_entry = payload["approvals"].get(fingerprint)
                entry = {
                    "status": "approved",
                    "fingerprint": fingerprint,
                    "approved_at": approved_at,
                    "approved_by": actor,
                    "strategy_id": strategy.get("id"),
                    "artifact_path": str(artifact_path),
                    "execution_engine_digest": engine_digest,
                    "behavior": strategy_behavior(strategy),
                    "notes": notes,
                }
                history = _entry_history(previous_entry)
                if history:
                    entry["history"] = history
                if current_artifact is not None:
                    entry["artifact_digest"] = current_artifact_digest
                if product_payload is not None:
                    entry["product"] = product_payload
                if production_preflight is not None:
                    entry["production_preflight"] = production_preflight
                payload["approvals"][fingerprint] = entry
                fingerprints.append(fingerprint)
            self.save(payload)
        return fingerprints

    def revoke(self, fingerprint: str, *, revoked_by: str, reason: str = "") -> None:
        actor = normalize_approval_actor(revoked_by, field="revoked_by")
        revocation_reason = normalize_revocation_reason(reason)
        with acquire_file_update_lock(self.path, label="approval ledger"):
            payload = self.load()
            entry = payload["approvals"].get(fingerprint)
            if entry is None:
                raise KeyError(f"Unknown approval fingerprint: {fingerprint}")
            if not isinstance(entry, dict):
                raise ApprovalError(f"Malformed approval entry for fingerprint: {fingerprint}")
            history = _entry_history(entry)
            if history:
                entry["history"] = history
            entry["status"] = "revoked"
            entry["revoked_at"] = utc_now()
            entry["revoked_by"] = actor
            entry["revocation_reason"] = revocation_reason
            self.save(payload)

    def assert_approved(
        self,
        strategies: Iterable[dict[str, Any]],
        *,
        artifact_path: Path | None = None,
        artifact_digest_value: str | None = None,
        product: ProductConfig | None = None,
    ) -> None:
        payload = self.load()
        approvals = payload.get("approvals", {})
        current_execution_engine_digest = execution_engine_digest()
        current_production_preflight = _current_production_preflight(product, artifact_digest_value)
        failure_keys = (
            "missing",
            "revoked",
            "malformed",
            "invalid_actor",
            "artifact_mismatch",
            "artifact_content_mismatch",
            "product_mismatch",
            "fingerprint_mismatch",
            "execution_engine_mismatch",
            "production_preflight_mismatch",
        )
        failures = {key: [] for key in failure_keys}
        for strategy in strategies:
            fingerprint = strategy_fingerprint(strategy)
            entry = approvals.get(fingerprint)
            label = f"{strategy.get('id', '<unknown>')} ({fingerprint})"
            basic_failure = _basic_approval_failure(entry, label)
            if basic_failure is not None:
                failures[basic_failure[0]].append(basic_failure[1])
                continue
            identity_failure = _identity_approval_failure(
                entry,
                fingerprint,
                label,
                artifact_path=artifact_path,
                artifact_digest_value=artifact_digest_value,
                current_execution_engine_digest=current_execution_engine_digest,
                product=product,
                current_production_preflight=current_production_preflight,
            )
            if identity_failure is not None:
                failures[identity_failure[0]].append(identity_failure[1])
        _raise_basic_approval_failures(failures)
        _raise_identity_approval_failures(failures)


def assert_artifact_live_approved(
    artifact_path: Path,
    ledger_path: Path = DEFAULT_APPROVAL_LEDGER,
    *,
    product: ProductConfig | None = None,
) -> None:
    artifact = load_artifact(artifact_path)
    assert_loaded_artifact_live_approved(
        artifact,
        artifact_path,
        ledger_path,
        product=product,
    )


def assert_loaded_artifact_live_approved(
    artifact: dict[str, Any],
    artifact_path: Path,
    ledger_path: Path = DEFAULT_APPROVAL_LEDGER,
    *,
    product: ProductConfig | None = None,
) -> None:
    """Validate approval and policy for one already-loaded artifact snapshot.

    Callers that will execute the artifact should load it exactly once and pass
    that same payload through every gate and into the bot.  This prevents a
    path replacement between approval and execution from changing behaviour.
    """
    if not isinstance(artifact, dict):
        raise ApprovalError("Strategy artifact must be a JSON object.")
    strategies = artifact.get("strategies")
    if not isinstance(strategies, list):
        raise ApprovalError("Strategy artifact strategies must be a list.")
    ApprovalLedger(ledger_path).assert_approved(
        strategies,
        artifact_path=artifact_path,
        artifact_digest_value=artifact_digest(artifact),
        product=product,
    )
    try:
        assert_artifact_policy_allowed_for_product(artifact, product)
    except StrategyPolicyError as exc:
        raise ApprovalError(str(exc)) from exc


def find_product_for_artifact(
    config_path: Path,
    product_name: str | None,
    artifact_path: Path,
) -> ProductConfig | None:
    if product_name is None and not config_path.exists():
        return None
    config = load_config(config_path)
    if product_name:
        for product in config.products:
            if product.name == product_name:
                if not _same_path(product.strategies_path, artifact_path):
                    raise ApprovalError(
                        f"Artifact {artifact_path} does not match product {product.name} "
                        f"strategies_path {product.strategies_path}."
                    )
                return product
        raise ApprovalError(f"No product named {product_name!r} in {config_path}.")

    expected = artifact_path.resolve()
    matches = [
        product for product in config.products if product.strategies_path.resolve() == expected
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ApprovalError(f"Multiple products use artifact {artifact_path}; pass --product.")
    return None


def assert_artifact_policy_allowed_for_product(
    artifact: dict[str, Any],
    product: ProductConfig | None,
) -> None:
    if product is None:
        return
    errors = validate_strategy_artifact(product, artifact)
    if errors:
        raise StrategyPolicyError(
            f"{product.name}: strategy artifact violates policy: " + "; ".join(errors)
        )


def require_product_for_cli(
    product: ProductConfig | None, *, artifact_path: Path, config_path: Path
) -> ProductConfig:
    if product is None:
        raise ApprovalError(
            f"Approval CLI requires a product context for {artifact_path}; "
            f"pass --config {config_path} with a matching product or pass --product."
        )
    return product


def _select_strategies(
    artifact: dict[str, Any], strategy_id: str | None, all_strategies: bool
) -> list[dict[str, Any]]:
    strategies = list(artifact.get("strategies", []))
    if all_strategies:
        return strategies
    if strategy_id:
        selected = [strategy for strategy in strategies if strategy.get("id") == strategy_id]
        if not selected:
            raise SystemExit(f"No strategy with id {strategy_id!r} in artifact.")
        return selected
    raise SystemExit("Pass --strategy-id <id> or --all.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve or check strategy artifacts for live trading."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_APPROVAL_LEDGER)
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser("approve", help="Approve one or all strategies from an artifact.")
    approve.add_argument("--artifact", type=Path, required=True)
    approve.add_argument(
        "--expected-artifact-digest",
        required=True,
        help="Digest printed by the reviewed promotion packet; blocks path-replacement races.",
    )
    approve.add_argument(
        "--expected-preflight-digest",
        help=(
            "Exact sha256 digest of the reviewed successful connected production-preflight "
            "report. Required for final live approval."
        ),
    )
    approve.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    approve.add_argument("--product", help="Product name for product-aware policy validation.")
    approve_scope = approve.add_mutually_exclusive_group(required=True)
    approve_scope.add_argument("--strategy-id")
    approve_scope.add_argument("--all", action="store_true")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that this approval can unlock live trading after other gates pass.",
    )
    approve.add_argument("--notes", default="")

    check = sub.add_parser("check", help="Check that all strategies in an artifact are approved.")
    check.add_argument("--artifact", type=Path, required=True)
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    check.add_argument("--product", help="Product name for product-aware policy validation.")

    list_cmd = sub.add_parser("list", help="List approvals in the ledger.")
    list_cmd.add_argument("--json", action="store_true")

    revoke = sub.add_parser("revoke", help="Revoke an approved strategy fingerprint.")
    revoke.add_argument("--fingerprint", required=True)
    revoke.add_argument("--revoked-by", required=True)
    revoke.add_argument("--reason", required=True)
    return parser.parse_args()


def _approval_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], ProductConfig, str, dict[str, Any]]:
    if not args.confirm_live:
        raise SystemExit("Approval requires --confirm-live because this can unlock live trading.")
    artifact = load_artifact(args.artifact)
    current_digest = artifact_digest(artifact)
    if args.expected_artifact_digest != current_digest:
        raise SystemExit(
            "Approval refused: reviewed artifact digest does not match the current file "
            f"({args.expected_artifact_digest} != {current_digest})."
        )
    product = require_product_for_cli(
        find_product_for_artifact(args.config, args.product, args.artifact),
        artifact_path=args.artifact,
        config_path=args.config,
    )
    try:
        assert_artifact_policy_allowed_for_product(artifact, product)
    except StrategyPolicyError as exc:
        raise SystemExit(str(exc)) from exc
    if product.execution_mode != "live":
        raise SystemExit(
            f"Approval refused: product {product.name} must be configured live while paused "
            "before final approval."
        )
    try:
        production_preflight = load_production_preflight_evidence(
            product,
            expected_artifact_digest=current_digest,
        )
    except ApprovalError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.expected_preflight_digest:
        raise SystemExit(
            "Approval refused: --expected-preflight-digest is required for final live approval."
        )
    if args.expected_preflight_digest != production_preflight["source_report_digest"]:
        raise SystemExit(
            "Approval refused: reviewed production-preflight digest does not match the "
            f"current report ({args.expected_preflight_digest} != "
            f"{production_preflight['source_report_digest']})."
        )
    return artifact, product, current_digest, production_preflight


def _approve_command(args: argparse.Namespace, ledger: ApprovalLedger) -> None:
    artifact, product, _, _ = _approval_context(args)
    selected = _select_strategies(artifact, args.strategy_id, args.all)
    fingerprints = ledger.approve_many(
        selected,
        artifact_path=args.artifact,
        approved_by=args.approved_by,
        product=product,
        artifact=artifact,
        notes=args.notes,
    )
    for strategy, fingerprint in zip(selected, fingerprints, strict=True):
        print(f"approved {strategy.get('id')} {fingerprint}")


def _check_command(args: argparse.Namespace) -> None:
    artifact = load_artifact(args.artifact)
    product = require_product_for_cli(
        find_product_for_artifact(args.config, args.product, args.artifact),
        artifact_path=args.artifact,
        config_path=args.config,
    )
    try:
        assert_artifact_policy_allowed_for_product(artifact, product)
    except StrategyPolicyError as exc:
        raise SystemExit(str(exc)) from exc
    assert_artifact_live_approved(args.artifact, args.ledger, product=product)
    print(f"approved for live: {args.artifact}")


def _list_command(args: argparse.Namespace, ledger: ApprovalLedger) -> None:
    payload = ledger.load()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    approvals = payload.get("approvals", {})
    if not isinstance(approvals, dict):
        raise SystemExit(f"Approval ledger approvals must be a JSON object: {ledger.path}")
    for fingerprint, entry in sorted(approvals.items()):
        if not isinstance(entry, dict):
            print(f"malformed <invalid> {fingerprint} by=-")
            continue
        print(_list_entry_line(str(fingerprint), entry))


def _revoke_command(args: argparse.Namespace, ledger: ApprovalLedger) -> None:
    ledger.revoke(args.fingerprint, revoked_by=args.revoked_by, reason=args.reason)
    print(f"revoked {args.fingerprint}")


def main() -> None:
    args = parse_args()
    ledger = ApprovalLedger(args.ledger)
    if args.command == "approve":
        _approve_command(args, ledger)
    elif args.command == "check":
        _check_command(args)
    elif args.command == "list":
        _list_command(args, ledger)
    elif args.command == "revoke":
        _revoke_command(args, ledger)


if __name__ == "__main__":
    main()
