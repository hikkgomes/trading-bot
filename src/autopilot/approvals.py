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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, ProductConfig, load_config
from src.autopilot.io import write_json_atomic
from src.autopilot.strategy_policy import StrategyPolicyError, validate_strategy_artifact
from src.config import PROJECT_ROOT

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


class ApprovalError(RuntimeError):
    """Raised when a strategy artifact is not approved for live trading."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_approval_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ApprovalError(f"{field} must be a non-empty operator identifier.")
    actor = value.strip()
    if not actor:
        raise ApprovalError(f"{field} must be a non-empty operator identifier.")
    if _is_automation_approval_actor(actor):
        raise ApprovalError(f"{field} must identify a human operator, not automation identity {actor!r}.")
    return actor


def normalize_revocation_reason(value: str) -> str:
    if not isinstance(value, str):
        raise ApprovalError("revocation reason must be non-empty.")
    reason = value.strip()
    if not reason:
        raise ApprovalError("revocation reason must be non-empty.")
    return reason


def is_valid_approval_actor(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not _is_automation_approval_actor(value)


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
    digest = hashlib.sha256(_canonical_json(strategy_behavior(strategy)).encode("utf-8")).hexdigest()
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
    bad_indexes = [index for index, strategy in enumerate(strategies) if not isinstance(strategy, dict)]
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


def _product_approval_payload(product: ProductConfig | None) -> dict[str, Any] | None:
    if product is None:
        return None
    return {
        "name": product.name,
        "objective": product.objective,
        "market": product.market,
        "symbol": product.symbol,
        "base_asset": product.base_asset,
    }


def _entry_matches_product(entry: dict[str, Any], product: ProductConfig | None) -> bool:
    if product is None:
        return True
    approved_product = entry.get("product")
    if not isinstance(approved_product, dict):
        return False
    return (
        approved_product.get("name") == product.name
        and approved_product.get("objective") == product.objective
        and approved_product.get("market") == product.market
        and str(approved_product.get("symbol", "")).upper() == product.symbol.upper()
        and str(approved_product.get("base_asset", "")).upper() == product.base_asset.upper()
    )


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
        "product": entry.get("product"),
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
        if artifact_path.is_symlink():
            raise ApprovalError(f"Strategy artifact must not be a symlink: {artifact_path}")
        payload = self.load()
        previous_entry = payload["approvals"].get(strategy_fingerprint(strategy))
        fingerprint = strategy_fingerprint(strategy)
        actor = normalize_approval_actor(approved_by, field="approved_by")
        current_artifact = artifact
        if current_artifact is None and artifact_path.exists():
            current_artifact = load_artifact(artifact_path)
        entry = {
            "status": "approved",
            "fingerprint": fingerprint,
            "approved_at": utc_now(),
            "approved_by": actor,
            "strategy_id": strategy.get("id"),
            "artifact_path": str(artifact_path),
            "behavior": strategy_behavior(strategy),
            "notes": notes,
        }
        history = _entry_history(previous_entry)
        if history:
            entry["history"] = history
        if current_artifact is not None:
            entry["artifact_digest"] = artifact_digest(current_artifact)
        product_payload = _product_approval_payload(product)
        if product_payload is not None:
            entry["product"] = product_payload
        payload["approvals"][fingerprint] = entry
        self.save(payload)
        return fingerprint

    def revoke(self, fingerprint: str, *, revoked_by: str, reason: str = "") -> None:
        actor = normalize_approval_actor(revoked_by, field="revoked_by")
        revocation_reason = normalize_revocation_reason(reason)
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
        missing: list[str] = []
        revoked: list[str] = []
        artifact_mismatch: list[str] = []
        artifact_content_mismatch: list[str] = []
        product_mismatch: list[str] = []
        fingerprint_mismatch: list[str] = []
        malformed: list[str] = []
        invalid_actor: list[str] = []
        for strategy in strategies:
            fingerprint = strategy_fingerprint(strategy)
            entry = approvals.get(fingerprint)
            label = f"{strategy.get('id', '<unknown>')} ({fingerprint})"
            if entry is None:
                missing.append(label)
            elif not isinstance(entry, dict):
                malformed.append(label)
            elif entry.get("status") != "approved":
                revoked.append(label)
            elif not is_valid_approval_actor(entry.get("approved_by")):
                invalid_actor.append(label)
            elif entry.get("fingerprint") != fingerprint:
                fingerprint_mismatch.append(label)
            elif artifact_path is not None and not _same_path(entry.get("artifact_path"), artifact_path):
                artifact_mismatch.append(label)
            elif artifact_digest_value is not None and entry.get("artifact_digest") != artifact_digest_value:
                artifact_content_mismatch.append(
                    f"{label} approved={entry.get('artifact_digest') or '<missing>'} current={artifact_digest_value}"
                )
            elif not _entry_matches_product(entry, product):
                product_mismatch.append(label)
        if missing or revoked or malformed or invalid_actor:
            parts = []
            if missing:
                parts.append("missing approval: " + ", ".join(missing))
            if revoked:
                parts.append("revoked/not approved: " + ", ".join(revoked))
            if malformed:
                parts.append("malformed approval: " + ", ".join(malformed))
            if invalid_actor:
                parts.append("invalid approval actor: " + ", ".join(invalid_actor))
            raise ApprovalError("Live trading blocked; " + "; ".join(parts))
        if artifact_mismatch or artifact_content_mismatch or product_mismatch or fingerprint_mismatch:
            parts = []
            if fingerprint_mismatch:
                parts.append("approval fingerprint mismatch: " + ", ".join(fingerprint_mismatch))
            if artifact_mismatch:
                parts.append("approval artifact mismatch: " + ", ".join(artifact_mismatch))
            if artifact_content_mismatch:
                parts.append("approval artifact content mismatch: " + ", ".join(artifact_content_mismatch))
            if product_mismatch:
                parts.append("approval product mismatch: " + ", ".join(product_mismatch))
            raise ApprovalError("Live trading blocked; " + "; ".join(parts))


def assert_artifact_live_approved(
    artifact_path: Path,
    ledger_path: Path = DEFAULT_APPROVAL_LEDGER,
    *,
    product: ProductConfig | None = None,
) -> None:
    artifact = load_artifact(artifact_path)
    ApprovalLedger(ledger_path).assert_approved(
        artifact["strategies"],
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
    matches = [product for product in config.products if product.strategies_path.resolve() == expected]
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


def require_product_for_cli(product: ProductConfig | None, *, artifact_path: Path, config_path: Path) -> ProductConfig:
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
    parser = argparse.ArgumentParser(description="Approve or check strategy artifacts for live trading.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_APPROVAL_LEDGER)
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser("approve", help="Approve one or all strategies from an artifact.")
    approve.add_argument("--artifact", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    ledger = ApprovalLedger(args.ledger)
    if args.command == "approve":
        if not args.confirm_live:
            raise SystemExit("Approval requires --confirm-live because this can unlock live trading.")
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
        for strategy in _select_strategies(artifact, args.strategy_id, args.all):
            fingerprint = ledger.approve(
                strategy,
                artifact_path=args.artifact,
                approved_by=args.approved_by,
                product=product,
                artifact=artifact,
                notes=args.notes,
            )
            print(f"approved {strategy.get('id')} {fingerprint}")
    elif args.command == "check":
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
    elif args.command == "list":
        payload = ledger.load()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            approvals = payload.get("approvals", {})
            if not isinstance(approvals, dict):
                raise SystemExit(f"Approval ledger approvals must be a JSON object: {ledger.path}")
            for fingerprint, entry in sorted(approvals.items()):
                if not isinstance(entry, dict):
                    print(f"malformed <invalid> {fingerprint} by=-")
                    continue
                print(_list_entry_line(str(fingerprint), entry))
    elif args.command == "revoke":
        ledger.revoke(args.fingerprint, revoked_by=args.revoked_by, reason=args.reason)
        print(f"revoked {args.fingerprint}")


if __name__ == "__main__":
    main()
