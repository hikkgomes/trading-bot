"""Explicit operator handoff from a staged live candidate to the active path.

Research is allowed to refresh ``runtime/candidates/<product>.json`` for a
live product, but it must never replace the artifact the executor is using.
This module is the deliberately manual bridge between those two paths.  It
does not approve the candidate and it does not refresh preflight or testnet
evidence; all of those gates remain bound to the newly activated artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from src.autopilot.approvals import artifact_digest
from src.autopilot.config import (
    DEFAULT_CONFIG_PATH,
    AutopilotConfig,
    ProductConfig,
    load_config,
)
from src.autopilot.control import (
    control_update_lock,
    is_product_paused,
    load_control,
    should_flatten_product,
)
from src.autopilot.io import append_json_line, write_json_atomic
from src.autopilot.locking import acquire_runtime_lock
from src.autopilot.promotion import PromotionThresholds, build_promotion_review
from src.autopilot.reporting import utc_now
from src.autopilot.strategy_policy import (
    StrategyPolicyError,
    assert_loaded_strategy_artifact_allowed,
)
from src.config import PROJECT_ROOT

DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "runtime" / "candidates"
SAFE_PRODUCT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ARTIFACT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class CandidateActivationError(RuntimeError):
    """Raised when a staged candidate cannot safely become active."""


def product_identity(product: ProductConfig) -> dict[str, Any]:
    """Identity stamped by research and checked again during activation."""
    return {
        "name": product.name,
        "objective": product.objective,
        "base_asset": product.base_asset.upper(),
        "market": product.market,
        "symbol": product.symbol.upper(),
        "starting_equity": product.starting_equity,
        "regime_guard": product.regime_guard,
        "regime_mayer_top": product.regime_mayer_top,
    }


def candidate_path_for_product(
    product_name: str,
    *,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
) -> Path:
    if not SAFE_PRODUCT_NAME_RE.fullmatch(product_name):
        raise CandidateActivationError(
            "product name is unsafe for a deterministic candidate path: "
            f"{product_name!r}"
        )
    return candidate_dir / f"{product_name}.json"


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise CandidateActivationError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        raise CandidateActivationError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise CandidateActivationError(f"{label} must be a regular file: {path}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise CandidateActivationError(
                    f"{label} must not contain duplicate JSON key {key!r}: {path}"
                )
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise CandidateActivationError(
            f"{label} must use strict JSON; invalid constant {value!r}: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except CandidateActivationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateActivationError(
            f"{label} must be readable valid JSON: {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CandidateActivationError(f"{label} must contain a JSON object: {path}")
    return payload


def _assert_path_shape(path: Path, *, label: str, suffix: str) -> None:
    if not path.is_absolute():
        raise CandidateActivationError(f"{label} must be an absolute path: {path}")
    if ".." in path.parts:
        raise CandidateActivationError(f"{label} must not contain parent traversal: {path}")
    if path.suffix != suffix:
        raise CandidateActivationError(f"{label} must end in {suffix}: {path}")

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise CandidateActivationError(
                f"{label} must not contain a symlink component: {current}"
            )
    if path.exists() and not path.is_file():
        raise CandidateActivationError(f"{label} must be a regular file: {path}")


def _assert_distinct_paths(paths: dict[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        normalized = path.resolve(strict=False)
        previous = seen.get(normalized)
        if previous is not None:
            raise CandidateActivationError(
                f"{label} path must be distinct from {previous}: {path}"
            )
        seen[normalized] = label


def _load_flat_state(path: Path, *, product_name: str) -> dict[str, Any]:
    state = _strict_json_object(path, label=f"{product_name} state file")
    positions = state.get("open_positions", {})
    if not isinstance(positions, dict):
        raise CandidateActivationError(
            f"{product_name} state open_positions must be a JSON object"
        )
    if positions:
        ids = ", ".join(sorted(str(value) for value in positions))
        raise CandidateActivationError(
            f"{product_name} must have no open positions before activation: {ids}"
        )
    for field in (
        "pending_order",
        "flatten_intent",
        "pending_entry_recovery",
        "risk_recovery_incident",
        "exit_accounting_intent",
    ):
        if state.get(field) is not None:
            raise CandidateActivationError(
                f"{product_name} state still has {field}; reconcile it before activation"
            )
    return state


def _validate_audit_log(path: Path) -> None:
    _assert_path_shape(path, label="activation audit", suffix=".jsonl")
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CandidateActivationError(
            f"activation audit must be readable: {path}: {type(exc).__name__}: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateActivationError(
                f"activation audit contains malformed JSON on line {line_number}: {path}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise CandidateActivationError(
                f"activation audit line {line_number} must be a JSON object: {path}"
            )


def _append_activation_event(path: Path, event: dict[str, Any]) -> None:
    """Append/fsync the event and its directory entry before continuing."""
    append_json_line(path, event)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _operator(value: str | None) -> str:
    actor = (value or os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown").strip()
    if not actor:
        raise CandidateActivationError("operator must be a non-empty identifier")
    return actor


def _absolute_lexical(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _configured_product(
    config_path: Path,
    product_name: str,
) -> tuple[AutopilotConfig, ProductConfig]:
    config = load_config(config_path)
    matches = [product for product in config.products if product.name == product_name]
    if len(matches) != 1:
        available = ", ".join(sorted(product.name for product in config.products)) or "<none>"
        raise CandidateActivationError(
            f"configured product {product_name!r} was not found; available: {available}"
        )
    return config, matches[0]


def activate_candidate(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    product_name: str,
    confirm: bool,
    expected_candidate_digest: str,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
    operator: str | None = None,
) -> dict[str, Any]:
    """Activate one staged live candidate without granting live approval."""
    if confirm is not True:
        raise CandidateActivationError("candidate activation requires --confirm")
    if not isinstance(expected_candidate_digest, str) or not ARTIFACT_DIGEST_RE.fullmatch(
        expected_candidate_digest
    ):
        raise CandidateActivationError(
            "candidate activation requires --expected-candidate-digest sha256:<64 lowercase hex>"
        )

    config_path = _absolute_lexical(config_path)
    candidate_dir = _absolute_lexical(candidate_dir)
    config, product = _configured_product(config_path, product_name)
    if product.execution_mode != "live":
        raise CandidateActivationError(
            f"{product.name} must be configured in live mode to activate a staged candidate"
        )

    lock_path = _absolute_lexical(config.lock_file)
    _assert_path_shape(lock_path, label="runtime lock", suffix=".lock")
    try:
        with acquire_runtime_lock(lock_path):
            with control_update_lock(config.control_file):
                return _activate_candidate_locked(
                    config=config,
                    product=product,
                    config_path=config_path,
                    candidate_dir=candidate_dir,
                    expected_candidate_digest=expected_candidate_digest,
                    operator=operator,
                )
    except CandidateActivationError:
        raise
    except RuntimeError as exc:
        raise CandidateActivationError(
            "candidate activation locking failed; stop the autopilot supervisor "
            f"for this maintenance window and retry: {exc}"
        ) from exc


def _activate_candidate_locked(
    *,
    config: AutopilotConfig,
    product: ProductConfig,
    config_path: Path,
    candidate_dir: Path,
    expected_candidate_digest: str,
    operator: str | None,
) -> dict[str, Any]:
    """Run every mutable activation check while holding the runtime lock."""

    candidate_path = candidate_path_for_product(product.name, candidate_dir=candidate_dir)
    active_path = product.strategies_path
    audit_path = config.control_audit_file
    path_specs = {
        "config": (config_path, ".json"),
        "candidate": (candidate_path, ".json"),
        "active artifact": (active_path, ".json"),
        "product state": (product.state_file, ".json"),
        "control": (config.control_file, ".json"),
        "activation audit": (audit_path, ".jsonl"),
        "runtime lock": (config.lock_file, ".lock"),
    }
    for label, (path, suffix) in path_specs.items():
        _assert_path_shape(path, label=label, suffix=suffix)
    _assert_distinct_paths({label: path for label, (path, _) in path_specs.items()})

    control = load_control(config.control_file)
    if control.get("control_error"):
        raise CandidateActivationError(
            f"control file is malformed; repair it before activation: {control['control_error']}"
        )
    if not is_product_paused(control, product.name):
        raise CandidateActivationError(
            f"{product.name} must be paused before candidate activation"
        )
    if should_flatten_product(control, product.name):
        raise CandidateActivationError(
            f"{product.name} still has a flatten request; reconcile it before activation"
        )

    _load_flat_state(product.state_file, product_name=product.name)
    candidate = _strict_json_object(candidate_path, label=f"{product.name} candidate")
    candidate_digest = artifact_digest(candidate)
    if candidate_digest != expected_candidate_digest:
        raise CandidateActivationError(
            f"{product.name} candidate changed after review: expected "
            f"{expected_candidate_digest}, current {candidate_digest}"
        )
    identity = candidate.get("product")
    expected_identity = product_identity(product)
    if identity != expected_identity:
        raise CandidateActivationError(
            f"{product.name} candidate product identity mismatch: "
            f"expected {expected_identity!r}, got {identity!r}"
        )
    try:
        assert_loaded_strategy_artifact_allowed(
            product,
            candidate,
            artifact_path=candidate_path,
            require_live_eligible=True,
        )
    except StrategyPolicyError as exc:
        raise CandidateActivationError(str(exc)) from exc

    candidate_trade_log = candidate_path.parent / f"{product.name}_paper_trades.csv"
    forward_review = build_promotion_review(
        artifact_path=candidate_path,
        trade_log=candidate_trade_log,
        ledger_path=config.approval_ledger,
        thresholds=PromotionThresholds(),
        product=product,
        config_path=config_path,
    )
    not_ready = [
        {
            "id": item.get("id"),
            "recommendation": item.get("recommendation"),
            "reasons": item.get("reasons") or [],
        }
        for item in forward_review.get("strategies", [])
        if item.get("recommendation") not in {"needs_approval", "already_approved"}
    ]
    if not forward_review.get("strategies") or not_ready:
        raise CandidateActivationError(
            f"{product.name} candidate exact-fingerprint forward-paper evidence is not ready; "
            f"review {candidate_path.parent / f'{product.name}_promotion_review.md'}: "
            f"{not_ready or 'no strategy evidence'}"
        )

    old_digest: str | None = None
    if active_path.exists():
        active = _strict_json_object(active_path, label=f"{product.name} active artifact")
        old_digest = artifact_digest(active)
        prior_activation = active.get("candidate_activation")
        if (
            isinstance(prior_activation, dict)
            and prior_activation.get("candidate_artifact_digest") == candidate_digest
        ):
            raise CandidateActivationError(
                f"{product.name} candidate is already the source of the active artifact"
            )

    _validate_audit_log(audit_path)
    actor = _operator(operator)
    activated_at = utc_now()
    activation_id = f"{product.name}:{uuid.uuid4()}"
    activated = dict(candidate)
    activated["candidate_activation"] = {
        "activation_id": activation_id,
        "activated_at": activated_at,
        "activated_by": actor,
        "candidate_path": str(candidate_path),
        "candidate_artifact_digest": candidate_digest,
        "approval_granted": False,
    }
    new_digest = artifact_digest(activated)
    if old_digest == new_digest:
        raise CandidateActivationError(
            f"{product.name} candidate is already the active artifact ({new_digest})"
        )

    common_event = {
        "at": activated_at,
        "actor": actor,
        "command": "activate-candidate",
        "activation_id": activation_id,
        "product": expected_identity,
        "candidate_path": str(candidate_path),
        "active_artifact": str(active_path),
        "previous_artifact_digest": old_digest,
        "candidate_artifact_digest": candidate_digest,
        "activated_artifact_digest": new_digest,
        "approval_granted": False,
    }
    # Write-ahead evidence means a replacement can always be reconciled by
    # digest even if the second audit append fails after the atomic rename.
    _append_activation_event(
        audit_path,
        {**common_event, "status": "activation_intent"},
    )
    write_json_atomic(active_path, activated)
    _append_activation_event(
        audit_path,
        {
            **common_event,
            "at": utc_now(),
            "status": "activated",
        },
    )

    return {
        "ok": True,
        "product": product.name,
        "candidate": str(candidate_path),
        "active_artifact": str(active_path),
        "artifact_digest": new_digest,
        "candidate_artifact_digest": candidate_digest,
        "previous_artifact_digest": old_digest,
        "audit": str(audit_path),
        "approval_granted": False,
        "live_ready": False,
        "forward_paper_trade_log": str(candidate_trade_log),
        "forward_paper_evidence": [
            {
                "id": item.get("id"),
                "fingerprint": item.get("fingerprint"),
                "paper": item.get("paper"),
            }
            for item in forward_review.get("strategies", [])
        ],
        "next_actions": [
            "rebuild the promotion packet against the activated artifact and its candidate paper log, then record explicit human approval",
            "run fresh connected preflight for the activated artifact",
            "run a fresh testnet rehearsal when required by the product",
            "resume the product only after every live gate passes",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicitly activate a staged candidate for one paused live product."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--expected-candidate-digest", required=True)
    parser.add_argument("--operator", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = activate_candidate(
            config_path=args.config,
            product_name=args.product,
            confirm=args.confirm,
            expected_candidate_digest=args.expected_candidate_digest,
            operator=args.operator,
        )
    except (CandidateActivationError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
