"""Credential-free research workspace between OpenClaw and trusted research code.

OpenClaw runs as a separate service/user.  This module never launches it and
never passes environment variables to it. The outward workspace is allowlisted,
includes operational and development-research feedback, and excludes secrets
and final-holdout outcomes. Inbound actions are untrusted inert records; they
are never directly executed or promoted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import heapq
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now

INBOX_ROOT = Path("runtime/research_inbox/openclaw")
DEFAULT_INCOMING = INBOX_ROOT / "incoming"
DEFAULT_ACCEPTED = INBOX_ROOT / "accepted"
DEFAULT_REJECTED = INBOX_ROOT / "rejected"
DEFAULT_ARCHIVE = INBOX_ROOT / "archive"
DEFAULT_INDEX = INBOX_ROOT / "index.json"
DEFAULT_INGEST_STATUS = INBOX_ROOT / "ingest_status.json"
DEFAULT_CONTEXT = Path("runtime/openclaw/research_context.json")
DEFAULT_REVIEW_AUDIT = Path("runtime/openclaw/review_audit.jsonl")
DEFAULT_EVENT_STATE = Path("runtime/openclaw/event_state.json")
DEFAULT_RESEARCH_CYCLE = Path("runtime/research_cycle.json")
DEFAULT_GENERATED_BATCH = Path("runtime/research/generated_hypotheses.json")
DEFAULT_MARKET_UNIVERSE = Path("runtime/market_universe.json")
DEFAULT_OPERATOR_REPORT = Path("runtime/operator_report.json")
DEFAULT_EXPERIMENT_MEMORY = Path("runtime/research/experiment_memory.sqlite3")
DEFAULT_PROPOSAL_STATE = Path("runtime/research/openclaw_proposal_state.json")
PROPOSAL_SCHEMA = "research_proposal/v1"
ACTION_SCHEMA = "research_action/v1"
ACCEPTED_SCHEMA = "autopilot.openclaw_research_action/v2"
LEGACY_ACCEPTED_SCHEMA = "autopilot.openclaw_research_proposal/v1"
CONTEXT_SCHEMA = "autopilot.openclaw_research_workspace/v2"
ACTION_TYPES = frozenset({"new", "revise", "retry", "retire", "request_test"})
OBJECTIVES = frozenset({"active_income", "btc_accumulation"})
OPPORTUNITY_TYPES = frozenset({"day", "position", "scalp", "swing"})
TIMEFRAMES = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
)
INPUT_KEYS = frozenset(
    {
        "base_timeframe",
        "action",
        "changes",
        "constraints",
        "created_at",
        "expected_outcome",
        "falsification_criteria",
        "objective",
        "opportunity_type",
        "parent_hypothesis_id",
        "provenance",
        "reasoning",
        "schema",
        "source",
        "source_proposal_id",
        "suggested_primitives",
        "suggested_spec",
        "symbol",
        "thesis",
    }
)
PROVENANCE_KEYS = frozenset({"agent", "model", "reference", "session_id", "version"})
SAFETY = {
    "research_only": True,
    "executable": False,
    "paper_trade_allowed": False,
    "promotion_allowed": False,
    "live_allowed": False,
    "requires_trusted_compilation": True,
    "requires_full_validation_before_export": True,
}
MAX_PROPOSAL_BYTES = 64 * 1024
MAX_SPEC_BYTES = 32 * 1024
MAX_BATCH = 100
MAX_WORKSPACE_HYPOTHESES = 2_000
MAX_WORKSPACE_EVALUATIONS_PER_HYPOTHESIS = 8
MAX_WORKSPACE_ACTION_HISTORY = 200
MAX_REVIEW_HISTORY = 40
EVENT_WAKE_COOLDOWN_SECONDS = 15 * 60
MAX_ACCEPTED_SCAN = 20_000
MAX_ACCEPTED_FILES = 2_000
MAX_ACCEPTED_BYTES = 128 * 1024 * 1024
MAX_DEDUP_INDEX_ITEMS = 50_000
MAX_DEDUP_INDEX_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_FILES = 2_000
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_REJECTED_FILES = 5_000
MAX_REJECTED_BYTES = 32 * 1024 * 1024
MAX_PRIVATE_PRUNE_PER_CYCLE = 5_000
MAX_INCOMING_FILES = 2_000
MAX_INCOMING_BYTES = 128 * 1024 * 1024
MAX_INCOMING_SCAN = 20_000
MAX_INCOMING_PRUNE_PER_CYCLE = 5_000
STALE_INCOMING_TEMP_SECONDS = 60 * 60
SAFE_PROGRESS_KEYS = (
    "hypotheses",
    "opportunity_types",
    "opportunity_types_by_product",
    "scenarios",
    "selected_hypotheses",
)
SAFE_NOVELTY_KEYS = (
    "duplicate_proposals",
    "duplicate_specs",
    "novel_proposals",
    "novel_specs",
    "unique_lineages",
    "unique_proposals",
    "unique_specs",
)
HOLDOUT_MARKERS = ("holdout", "final_test", "locked_test", "test_set")
SECRET_MARKERS = (
    "api_key",
    "apikey",
    "approval",
    "authorization",
    "credential",
    "exchange_account",
    "password",
    "private_key",
    "secret",
    "token",
    "webhook",
)
FORBIDDEN_SPEC_CONTROL_KEYS = frozenset(
    {
        "approved",
        "executable",
        "execution_mode",
        "live_allowed",
        "paper_trade_allowed",
        "promotion_allowed",
    }
)


class ProposalValidationError(ValueError):
    """Raised when an untrusted OpenClaw proposal violates the inbox contract."""


class DuplicateJsonKeyError(ValueError):
    """Raised when an untrusted JSON object repeats a key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProposalValidationError(f"non-standard JSON constant: {value}")


def _bounded_source_bytes(path: Path) -> tuple[bytes, os.stat_result, bool]:
    """Read at most one byte beyond the proposal limit without following links."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProposalValidationError(f"cannot safely open proposal file: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProposalValidationError("proposal path must be a regular file")
        data = bytearray()
        while len(data) <= MAX_PROPOSAL_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_PROPOSAL_BYTES + 1 - len(data)),
            )
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ProposalValidationError("proposal file changed while being read")
        oversized = before.st_size > MAX_PROPOSAL_BYTES or len(data) > MAX_PROPOSAL_BYTES
        return bytes(data), before, oversized
    finally:
        os.close(descriptor)


def _load_untrusted_json(path: Path) -> dict[str, Any]:
    data, _source_stat, oversized = _bounded_source_bytes(path)
    if oversized:
        raise ProposalValidationError(f"proposal exceeds {MAX_PROPOSAL_BYTES} bytes")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ProposalValidationError("proposal must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ProposalValidationError("proposal must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProposalValidationError("proposal must be a JSON object")
    return payload


def _required_string(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ProposalValidationError(f"{key} must be a string")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise ProposalValidationError(f"{key} length must be between {minimum} and {maximum}")
    return value


def _bounded_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProposalValidationError(f"{key} must be a list of strings")
    if len(value) > 64:
        raise ProposalValidationError(f"{key} cannot contain more than 64 items")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip() or len(item.strip()) > 240:
            raise ProposalValidationError(
                f"{key}[{index}] must be a non-empty string up to 240 characters"
            )
        result.append(item.strip())
    return result


def _validate_created_at(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ProposalValidationError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProposalValidationError("created_at must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _is_forbidden_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in FORBIDDEN_SPEC_CONTROL_KEYS or any(
        marker in normalized for marker in SECRET_MARKERS
    )


def _validate_bounded_json(value: Any, *, path: str = "suggested_spec", depth: int = 0) -> Any:
    if depth > 8:
        raise ProposalValidationError(f"{path} exceeds maximum nesting depth")
    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str) and len(value) > 2000:
            raise ProposalValidationError(f"{path} contains a string longer than 2000 characters")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProposalValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        if len(value) > 100:
            raise ProposalValidationError(f"{path} contains more than 100 list items")
        return [
            _validate_bounded_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if len(value) > 100:
            raise ProposalValidationError(f"{path} contains more than 100 object fields")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 120:
                raise ProposalValidationError(f"{path} contains an invalid field name")
            if _is_forbidden_key(key):
                raise ProposalValidationError(
                    f"{path} contains forbidden security/control field: {key}"
                )
            result[key] = _validate_bounded_json(item, path=f"{path}.{key}", depth=depth + 1)
        return result
    raise ProposalValidationError(f"{path} contains unsupported type {type(value).__name__}")


def _validate_provenance(payload: dict[str, Any]) -> dict[str, str]:
    value = payload.get("provenance", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProposalValidationError("provenance must be a JSON object")
    unknown = sorted(set(value) - PROVENANCE_KEYS)
    if unknown:
        raise ProposalValidationError(f"provenance has unknown fields: {', '.join(unknown)}")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or not item.strip() or len(item.strip()) > 240:
            raise ProposalValidationError(
                f"provenance.{key} must be a non-empty string up to 240 characters"
            )
        result[key] = item.strip()
    return result


def validate_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - INPUT_KEYS)
    if unknown:
        raise ProposalValidationError(f"proposal has unknown fields: {', '.join(unknown)}")
    schema = payload.get("schema")
    if schema not in {PROPOSAL_SCHEMA, ACTION_SCHEMA}:
        raise ProposalValidationError(f"schema must be {PROPOSAL_SCHEMA!r} or {ACTION_SCHEMA!r}")
    if payload.get("source") != "openclaw":
        raise ProposalValidationError("source must be 'openclaw'")
    objective = _required_string(payload, "objective", maximum=40)
    if objective not in OBJECTIVES:
        raise ProposalValidationError(f"objective must be one of {sorted(OBJECTIVES)}")
    opportunity_type = _required_string(payload, "opportunity_type", maximum=20)
    if opportunity_type not in OPPORTUNITY_TYPES:
        raise ProposalValidationError(
            f"opportunity_type must be one of {sorted(OPPORTUNITY_TYPES)}"
        )
    base_timeframe = _required_string(payload, "base_timeframe", maximum=8)
    if base_timeframe not in TIMEFRAMES:
        raise ProposalValidationError(f"base_timeframe must be one of {sorted(TIMEFRAMES)}")
    created_at = _validate_created_at(_required_string(payload, "created_at", maximum=80))
    thesis = _required_string(payload, "thesis", minimum=20, maximum=4000)
    symbol = str(payload.get("symbol") or "BTCUSDT").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,20}USDT", symbol):
        raise ProposalValidationError("symbol must be an uppercase USDT pair")
    if objective == "btc_accumulation" and symbol != "BTCUSDT":
        raise ProposalValidationError("btc_accumulation proposals must target BTCUSDT")
    source_proposal_id = payload.get("source_proposal_id")
    if source_proposal_id is not None:
        if not isinstance(source_proposal_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", source_proposal_id
        ):
            raise ProposalValidationError("source_proposal_id contains unsupported characters")
    raw_action = payload.get("action", "new")
    if not isinstance(raw_action, str):
        raise ProposalValidationError("action must be a string")
    action = raw_action.strip().lower()
    if action not in ACTION_TYPES:
        raise ProposalValidationError(f"action must be one of {sorted(ACTION_TYPES)}")
    parent_hypothesis_id = payload.get("parent_hypothesis_id")
    if action == "new":
        if parent_hypothesis_id is not None:
            raise ProposalValidationError("new actions cannot specify parent_hypothesis_id")
    else:
        if not isinstance(parent_hypothesis_id, str) or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", parent_hypothesis_id
        ):
            raise ProposalValidationError(
                f"{action} actions require parent_hypothesis_id as a sha256 behavior hash"
            )
    raw_reasoning = payload.get("reasoning", thesis)
    if not isinstance(raw_reasoning, str):
        raise ProposalValidationError("reasoning must be a string")
    reasoning = raw_reasoning.strip()
    if not 20 <= len(reasoning) <= 4000:
        raise ProposalValidationError("reasoning length must be between 20 and 4000")
    raw_expected_outcome = payload.get("expected_outcome", "")
    raw_falsification_criteria = payload.get("falsification_criteria", "")
    if not isinstance(raw_expected_outcome, str):
        raise ProposalValidationError("expected_outcome must be a string")
    if not isinstance(raw_falsification_criteria, str):
        raise ProposalValidationError("falsification_criteria must be a string")
    expected_outcome = raw_expected_outcome.strip()
    falsification_criteria = raw_falsification_criteria.strip()
    if (
        schema == ACTION_SCHEMA
        and action in {"new", "revise"}
        and (len(expected_outcome) < 10 or len(falsification_criteria) < 10)
    ):
        raise ProposalValidationError(
            "new/revise actions require expected_outcome and falsification_criteria"
        )
    if len(expected_outcome) > 2000 or len(falsification_criteria) > 2000:
        raise ProposalValidationError(
            "expected_outcome and falsification_criteria must be at most 2000 characters"
        )
    changes = _bounded_string_list(payload, "changes")
    suggested_spec = payload.get("suggested_spec", {})
    if suggested_spec is None:
        suggested_spec = {}
    if not isinstance(suggested_spec, dict):
        raise ProposalValidationError("suggested_spec must be a JSON object")
    suggested_spec = _validate_bounded_json(suggested_spec)
    if len(json.dumps(suggested_spec, sort_keys=True, separators=(",", ":"))) > MAX_SPEC_BYTES:
        raise ProposalValidationError(f"suggested_spec exceeds {MAX_SPEC_BYTES} serialized bytes")
    normalized = {
        "source": "openclaw",
        "source_created_at": created_at,
        "action": action,
        "objective": objective,
        "opportunity_type": opportunity_type,
        "base_timeframe": base_timeframe,
        "thesis": thesis,
        "reasoning": reasoning,
        "expected_outcome": expected_outcome,
        "falsification_criteria": falsification_criteria,
        "changes": changes,
        "symbol": symbol,
        "suggested_primitives": _bounded_string_list(payload, "suggested_primitives"),
        "constraints": _bounded_string_list(payload, "constraints"),
        "untrusted_suggested_spec": suggested_spec,
        "provenance": _validate_provenance(payload),
    }
    if source_proposal_id is not None:
        normalized["source_proposal_id"] = source_proposal_id
    if parent_hypothesis_id is not None:
        normalized["parent_hypothesis_id"] = parent_hypothesis_id
    return normalized


def canonical_proposal_digest(proposal: dict[str, Any]) -> str:
    semantic = {
        key: proposal.get(key)
        for key in (
            "base_timeframe",
            "action",
            "changes",
            "constraints",
            "expected_outcome",
            "falsification_criteria",
            "objective",
            "opportunity_type",
            "parent_hypothesis_id",
            "reasoning",
            "suggested_primitives",
            "symbol",
            "thesis",
            "untrusted_suggested_spec",
        )
    }
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_accepted_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_proposal(payload)
    digest = canonical_proposal_digest(normalized)
    return {
        "schema": ACCEPTED_SCHEMA,
        "proposal_id": f"openclaw-action-{digest.removeprefix('sha256:')[:20]}",
        "content_digest": digest,
        "received_at": utc_now(),
        **normalized,
        "safety": dict(SAFETY),
    }


def _raw_digest(path: Path) -> str:
    try:
        data, source_stat, oversized = _bounded_source_bytes(path)
    except ProposalValidationError:
        data = str(path).encode("utf-8")
        source_stat = None
        oversized = False
    digest = hashlib.sha256()
    if oversized and source_stat is not None:
        digest.update(f"oversized:{source_stat.st_size}:".encode("ascii"))
    digest.update(data)
    return "sha256:" + digest.hexdigest()


def _shared_group_enabled() -> bool:
    return os.environ.get("OPENCLAW_SHARED_GROUP") == "1"


def _chmod_if_needed(path: Path, mode: int) -> None:
    """Enforce an owned directory mode without chmodding a bind-mount root."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.geteuid():
            raise ProposalValidationError(
                f"directory must be real and owned by the current process user: {path}"
            )
        if stat.S_IMODE(observed.st_mode) != mode:
            os.fchmod(descriptor, mode)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(final.st_mode)
            or final.st_uid != os.geteuid()
            or stat.S_IMODE(final.st_mode) != mode
        ):
            raise ProposalValidationError(f"directory mode enforcement failed: {path}")
    finally:
        os.close(descriptor)


def _safe_directory(
    path: Path,
    *,
    shared_incoming: bool = False,
    shared_traverse: bool = False,
) -> None:
    if path.is_symlink():
        raise ProposalValidationError(f"inbox directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ProposalValidationError(f"inbox path must be a directory: {path}")
    if _shared_group_enabled() and shared_incoming:
        mode = 0o2770
    elif _shared_group_enabled() and shared_traverse:
        mode = 0o2710
    else:
        mode = 0o700
    _chmod_if_needed(path, mode)


@contextmanager
def _inbox_lock(root: Path):
    _safe_directory(root, shared_traverse=True)
    lock_path = root / ".ingest.lock"
    if lock_path.is_symlink():
        raise ProposalValidationError(f"inbox lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProposalValidationError("inbox lock must be a regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _existing_accepted(
    accepted_dir: Path,
    *,
    index_path: Path | None = None,
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    digests: set[str] = set()
    if index_path is not None and index_path.exists():
        if index_path.is_symlink() or not index_path.is_file():
            raise ProposalValidationError(f"inbox index must be a regular file: {index_path}")
        if index_path.stat().st_size > MAX_DEDUP_INDEX_BYTES:
            raise ProposalValidationError("inbox deduplication index exceeds its safe size limit")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProposalValidationError(f"cannot read inbox deduplication index: {exc}") from exc
        if (
            not isinstance(index, dict)
            or index.get("schema") != "autopilot.openclaw_inbox_index/v1"
        ):
            raise ProposalValidationError("inbox deduplication index has an invalid schema")
        prior_ids = index.get("proposal_ids")
        prior_digests = index.get("content_digests")
        if not isinstance(prior_ids, list) or not isinstance(prior_digests, list):
            raise ProposalValidationError("inbox deduplication index has invalid identity lists")
        if len(prior_ids) > MAX_DEDUP_INDEX_ITEMS or len(prior_digests) > MAX_DEDUP_INDEX_ITEMS:
            raise ProposalValidationError("inbox deduplication index exceeds its item limit")
        if any(not isinstance(item, str) for item in (*prior_ids, *prior_digests)):
            raise ProposalValidationError("inbox deduplication index contains invalid identities")
        ids.update(prior_ids)
        digests.update(prior_digests)
    for index, path in enumerate(sorted(accepted_dir.glob("*.json"))):
        if index >= MAX_ACCEPTED_SCAN or path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("schema") != ACCEPTED_SCHEMA:
            continue
        if isinstance(payload.get("proposal_id"), str):
            ids.add(payload["proposal_id"])
        if isinstance(payload.get("content_digest"), str):
            digests.add(payload["content_digest"])
    return ids, digests


def _accepted_spool_usage(accepted_dir: Path) -> dict[str, Any]:
    """Measure the durable, unprocessed hand-off spool with a hard scan bound."""

    files = 0
    bytes_used = 0
    scan_truncated = False
    with os.scandir(accepted_dir) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_ACCEPTED_SCAN:
                scan_truncated = True
                break
            path = Path(entry.path)
            try:
                item = path.lstat()
            except OSError as exc:
                raise ProposalValidationError(
                    f"cannot inspect accepted proposal spool path {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
                raise ProposalValidationError(
                    f"accepted proposal spool contains an unsafe path: {path}"
                )
            files += 1
            bytes_used += item.st_size
    return {
        "file_limit": MAX_ACCEPTED_FILES,
        "byte_limit": MAX_ACCEPTED_BYTES,
        "scan_limit": MAX_ACCEPTED_SCAN,
        "scan_truncated": scan_truncated,
        "files": files,
        "bytes": bytes_used,
        "limits_satisfied": (
            not scan_truncated and files <= MAX_ACCEPTED_FILES and bytes_used <= MAX_ACCEPTED_BYTES
        ),
    }


def _unique_archive_path(archive_dir: Path, raw_digest: str) -> Path:
    stem = raw_digest.removeprefix("sha256:")[:16]
    while True:
        candidate = archive_dir / f"{time.time_ns()}-{stem}.json"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate


def _unlink_source_if_unchanged(path: Path, source_stat: os.stat_result) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise ProposalValidationError(
            f"proposal path changed before cleanup: {path}: {exc}"
        ) from exc
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) != (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise ProposalValidationError(f"proposal path changed before cleanup: {path}")
    try:
        path.unlink()
    except OSError as exc:
        raise ProposalValidationError(f"cannot remove consumed proposal {path}: {exc}") from exc
    _fsync_directory(path.parent)


def _archive_untrusted_source(
    source_path: Path,
    archive_path: Path,
    *,
    expected_digest: str,
) -> bool:
    """Copy bounded untrusted bytes into a new owner-created private record."""

    data, source_stat, oversized = _bounded_source_bytes(source_path)
    if oversized:
        _unlink_source_if_unchanged(source_path, source_stat)
        return False
    actual_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ProposalValidationError("proposal changed between validation and archival")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=archive_path.parent,
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while archiving proposal")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, archive_path)
        _fsync_directory(archive_path.parent)
        _unlink_source_if_unchanged(source_path, source_stat)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return True


def _write_rejection(
    rejected_dir: Path,
    *,
    source_name: str,
    raw_digest: str,
    reason: str,
    duplicate_of: str | None = None,
) -> Path:
    name = f"{time.time_ns()}-{raw_digest.removeprefix('sha256:')[:16]}.json"
    path = rejected_dir / name
    payload: dict[str, Any] = {
        "schema": "autopilot.openclaw_rejection/v1",
        "rejected_at": utc_now(),
        "source_name": source_name[:240],
        "raw_digest": raw_digest,
        "reason": reason[:500],
    }
    if duplicate_of is not None:
        payload["duplicate_of"] = duplicate_of
    write_json_atomic(path, payload)
    path.chmod(0o600)
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_retention(
    directory: Path,
    *,
    file_limit: int,
    byte_limit: int,
) -> dict[str, Any]:
    """Prune oldest private records without following or ignoring unsafe paths."""

    directory_stat = directory.lstat()
    if (
        stat.S_ISLNK(directory_stat.st_mode)
        or not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.geteuid()
        or directory_stat.st_mode & 0o077
    ):
        raise ProposalValidationError(
            f"retention directory must be owner-private and non-symlink: {directory}"
        )
    total_files = 0
    total_bytes = 0

    def records():
        nonlocal total_files, total_bytes
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    item = path.lstat()
                except OSError as exc:
                    raise ProposalValidationError(
                        f"cannot inspect private retention path {path}: {exc}"
                    ) from exc
                if (
                    stat.S_ISLNK(item.st_mode)
                    or not stat.S_ISREG(item.st_mode)
                    or item.st_uid != os.geteuid()
                    or item.st_mode & 0o077
                ):
                    raise ProposalValidationError(f"private retention path is unsafe: {path}")
                total_files += 1
                total_bytes += item.st_size
                yield (
                    item.st_mtime_ns,
                    entry.name,
                    item.st_size,
                    item.st_dev,
                    item.st_ino,
                    path,
                )

    oldest = heapq.nsmallest(MAX_PRIVATE_PRUNE_PER_CYCLE, records())
    initial_files = total_files
    initial_bytes = total_bytes
    pruned_files = 0
    pruned_bytes = 0
    for modified_ns, _name, size, device, inode, path in oldest:
        if total_files <= file_limit and total_bytes <= byte_limit:
            break
        try:
            current = path.lstat()
        except OSError as exc:
            raise ProposalValidationError(
                f"private retention path changed during pruning: {path}: {exc}"
            ) from exc
        if (
            current.st_mtime_ns,
            current.st_size,
            current.st_dev,
            current.st_ino,
        ) != (modified_ns, size, device, inode):
            raise ProposalValidationError(f"private retention path changed during pruning: {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise ProposalValidationError(
                f"cannot prune private retention path {path}: {exc}"
            ) from exc
        total_files -= 1
        total_bytes -= size
        pruned_files += 1
        pruned_bytes += size
    if pruned_files:
        _fsync_directory(directory)
    return {
        "file_limit": file_limit,
        "byte_limit": byte_limit,
        "initial_files": initial_files,
        "initial_bytes": initial_bytes,
        "pruned_files": pruned_files,
        "pruned_bytes": pruned_bytes,
        "retained_files": total_files,
        "retained_bytes": total_bytes,
        "limits_satisfied": total_files <= file_limit and total_bytes <= byte_limit,
    }


def _incoming_entries(directory: Path) -> tuple[list[tuple[int, str, int, Path, bool]], bool]:
    """Return a bounded, oldest-first view of the untrusted incoming spool."""

    observed: list[tuple[int, str, int, Path, bool]] = []
    truncated = False
    with os.scandir(directory) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_INCOMING_SCAN:
                truncated = True
                break
            path = Path(entry.path)
            try:
                item = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(item.st_mode):
                observed.append((item.st_mtime_ns, entry.name, item.st_size, path, True))
                continue
            if not stat.S_ISREG(item.st_mode):
                raise ProposalValidationError(f"incoming spool contains an unsafe path: {path}")
            observed.append((item.st_mtime_ns, entry.name, item.st_size, path, False))
    observed.sort(key=lambda item: (item[0], item[1]))
    return observed, truncated


def _incoming_hygiene(directory: Path) -> dict[str, Any]:
    """Bound completed backlog and stale temporary files without reading them."""

    entries, scan_truncated = _incoming_entries(directory)
    now_ns = time.time_ns()
    retained_files = len(entries)
    retained_json_files = sum(1 for item in entries if item[1].endswith(".json"))
    retained_bytes = sum(item[2] for item in entries)
    pruned_files = 0
    pruned_bytes = 0
    stale_temp_files = 0
    for modified_ns, name, size, path, _is_symlink in entries:
        stale_temp = (
            not name.endswith(".json")
            and now_ns - modified_ns >= STALE_INCOMING_TEMP_SECONDS * 1_000_000_000
        )
        over_limit = retained_files > MAX_INCOMING_FILES or retained_bytes > MAX_INCOMING_BYTES
        if not stale_temp and not over_limit:
            continue
        if pruned_files >= MAX_INCOMING_PRUNE_PER_CYCLE:
            break
        try:
            current = path.lstat()
        except FileNotFoundError:
            retained_files -= 1
            retained_json_files -= int(name.endswith(".json"))
            retained_bytes -= size
            continue
        except OSError:
            continue
        if (current.st_mtime_ns, current.st_size) != (modified_ns, size):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        retained_files -= 1
        retained_json_files -= int(name.endswith(".json"))
        retained_bytes -= size
        pruned_files += 1
        pruned_bytes += size
        stale_temp_files += int(stale_temp)
    if pruned_files:
        _fsync_directory(directory)
    limits_satisfied = (
        not scan_truncated
        and retained_files <= MAX_INCOMING_FILES
        and retained_bytes <= MAX_INCOMING_BYTES
    )
    return {
        "file_limit": MAX_INCOMING_FILES,
        "byte_limit": MAX_INCOMING_BYTES,
        "scan_limit": MAX_INCOMING_SCAN,
        "scan_truncated": scan_truncated,
        "observed_files": len(entries),
        "observed_bytes": sum(item[2] for item in entries),
        "pruned_files": pruned_files,
        "pruned_bytes": pruned_bytes,
        "stale_temp_files_pruned": stale_temp_files,
        "retained_observed_files": retained_files,
        "retained_observed_json_files": retained_json_files,
        "retained_observed_bytes": retained_bytes,
        "limits_satisfied": limits_satisfied,
    }


def ingest_inbox(
    *,
    incoming_dir: Path = DEFAULT_INCOMING,
    accepted_dir: Path = DEFAULT_ACCEPTED,
    rejected_dir: Path = DEFAULT_REJECTED,
    archive_dir: Path = DEFAULT_ARCHIVE,
    index_path: Path = DEFAULT_INDEX,
    max_batch: int = MAX_BATCH,
) -> dict[str, Any]:
    if max_batch <= 0 or max_batch > MAX_BATCH:
        raise ValueError(f"max_batch must be between 1 and {MAX_BATCH}")
    root = incoming_dir.parent
    paths = (incoming_dir, accepted_dir, rejected_dir, archive_dir)
    if any(path.parent != root for path in paths):
        raise ValueError("incoming, accepted, rejected, and archive must share one inbox root")
    if index_path.parent != root:
        raise ValueError("index path must be inside the inbox root")
    if index_path.is_symlink():
        raise ProposalValidationError(f"inbox index must not be a symlink: {index_path}")
    with _inbox_lock(root):
        for path in paths:
            _safe_directory(path, shared_incoming=path == incoming_dir)
        known_ids, known_digests = _existing_accepted(
            accepted_dir,
            index_path=index_path,
        )
        accepted_spool = _accepted_spool_usage(accepted_dir)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        archived = 0
        oversized_discarded = 0
        dedup_capacity_rejections = 0
        accepted_spool_capacity_rejections = 0
        # Producers write a temporary non-.json file and atomically rename it
        # when complete. Only completed proposal names enter the ingest batch.
        incoming_snapshot, candidate_scan_truncated = _incoming_entries(incoming_dir)
        candidates = [
            path
            for _modified_ns, name, _size, path, _is_symlink in incoming_snapshot
            if name.endswith(".json") and not name.startswith(".")
        ][:max_batch]
        for source_path in candidates:
            if source_path.name.startswith("."):
                continue
            if source_path.is_symlink() or not source_path.is_file():
                raw_digest = (
                    "sha256:"
                    + hashlib.sha256(f"unsafe-path:{source_path.name}".encode()).hexdigest()
                )
                reason = "proposal path must be a non-symlink regular file"
                _write_rejection(
                    rejected_dir,
                    source_name=source_path.name,
                    raw_digest=raw_digest,
                    reason=reason,
                )
                rejected.append({"source_name": source_path.name, "reason": reason})
                if source_path.is_symlink():
                    source_path.unlink()
                continue
            raw_digest = _raw_digest(source_path)
            archive_path = _unique_archive_path(archive_dir, raw_digest)
            try:
                raw = _load_untrusted_json(source_path)
                proposal = build_accepted_proposal(raw)
                proposal_id = proposal["proposal_id"]
                content_digest = proposal["content_digest"]
                proposal_size = len(json.dumps(proposal, indent=2, sort_keys=True).encode("utf-8"))
                if proposal_id in known_ids or content_digest in known_digests:
                    _write_rejection(
                        rejected_dir,
                        source_name=source_path.name,
                        raw_digest=raw_digest,
                        reason="duplicate_proposal",
                        duplicate_of=proposal_id,
                    )
                    rejected.append(
                        {
                            "source_name": source_path.name,
                            "reason": "duplicate_proposal",
                            "duplicate_of": proposal_id,
                        }
                    )
                elif (
                    accepted_spool["scan_truncated"]
                    or accepted_spool["files"] >= MAX_ACCEPTED_FILES
                    or accepted_spool["bytes"] + proposal_size > MAX_ACCEPTED_BYTES
                ):
                    accepted_spool_capacity_rejections += 1
                    _write_rejection(
                        rejected_dir,
                        source_name=source_path.name,
                        raw_digest=raw_digest,
                        reason="accepted_spool_capacity",
                    )
                    rejected.append(
                        {
                            "source_name": source_path.name,
                            "reason": "accepted_spool_capacity",
                        }
                    )
                elif (
                    len(known_ids) >= MAX_DEDUP_INDEX_ITEMS
                    or len(known_digests) >= MAX_DEDUP_INDEX_ITEMS
                ):
                    dedup_capacity_rejections += 1
                    _write_rejection(
                        rejected_dir,
                        source_name=source_path.name,
                        raw_digest=raw_digest,
                        reason="dedup_index_capacity",
                    )
                    rejected.append(
                        {
                            "source_name": source_path.name,
                            "reason": "dedup_index_capacity",
                        }
                    )
                else:
                    destination = accepted_dir / f"{proposal_id}.json"
                    write_json_atomic(destination, proposal)
                    destination.chmod(0o600)
                    known_ids.add(proposal_id)
                    known_digests.add(content_digest)
                    accepted_spool["files"] += 1
                    accepted_spool["bytes"] += proposal_size
                    accepted.append(
                        {
                            "source_name": source_path.name,
                            "proposal_id": proposal_id,
                            "content_digest": content_digest,
                        }
                    )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {str(exc)[:420]}"
                _write_rejection(
                    rejected_dir,
                    source_name=source_path.name,
                    raw_digest=raw_digest,
                    reason=reason,
                )
                rejected.append({"source_name": source_path.name, "reason": reason})
            try:
                if _archive_untrusted_source(
                    source_path,
                    archive_path,
                    expected_digest=raw_digest,
                ):
                    archived += 1
                else:
                    oversized_discarded += 1
            except (OSError, ProposalValidationError) as exc:
                rejected.append(
                    {
                        "source_name": source_path.name,
                        "reason": f"archive_failed: {type(exc).__name__}",
                    }
                )
        index = {
            "schema": "autopilot.openclaw_inbox_index/v1",
            "updated_at": utc_now(),
            "accepted": len(known_ids),
            "proposal_ids": sorted(known_ids),
            "content_digests": sorted(known_digests),
        }
        write_json_atomic(index_path, index)
        index_path.chmod(0o600)
        # Retention happens only after the durable dedup index update. Removing
        # rolling audit/spool records therefore cannot make an old proposal new.
        retention = {
            "accepted": {
                **accepted_spool,
                "capacity_rejections": accepted_spool_capacity_rejections,
                "limits_satisfied": (
                    not accepted_spool["scan_truncated"]
                    and accepted_spool["files"] <= MAX_ACCEPTED_FILES
                    and accepted_spool["bytes"] <= MAX_ACCEPTED_BYTES
                ),
            },
            "archive": _private_retention(
                archive_dir,
                file_limit=MAX_ARCHIVE_FILES,
                byte_limit=MAX_ARCHIVE_BYTES,
            ),
            "rejected": _private_retention(
                rejected_dir,
                file_limit=MAX_REJECTED_FILES,
                byte_limit=MAX_REJECTED_BYTES,
            ),
            "incoming": _incoming_hygiene(incoming_dir),
        }
        dedup_at_capacity = (
            len(known_ids) >= MAX_DEDUP_INDEX_ITEMS or len(known_digests) >= MAX_DEDUP_INDEX_ITEMS
        )
        degraded_reasons: list[str] = []
        operational_rejections = [
            item
            for item in rejected
            if str(item.get("reason") or "").startswith(
                (
                    "archive_failed:",
                    "ProposalValidationError: cannot safely open proposal file:",
                    "ProposalValidationError: cannot remove consumed proposal",
                )
            )
        ]
        if operational_rejections:
            degraded_reasons.append("inbox_io_error")
        if dedup_at_capacity or dedup_capacity_rejections:
            degraded_reasons.append("dedup_index_capacity")
        if accepted_spool_capacity_rejections or not retention["accepted"]["limits_satisfied"]:
            degraded_reasons.append("accepted_spool_capacity")
        if not retention["archive"]["limits_satisfied"]:
            degraded_reasons.append("archive_retention_backlog")
        if not retention["rejected"]["limits_satisfied"]:
            degraded_reasons.append("rejected_retention_backlog")
        if candidate_scan_truncated or retention["incoming"]["scan_truncated"]:
            degraded_reasons.append("incoming_scan_truncated")
        if not retention["incoming"]["limits_satisfied"]:
            degraded_reasons.append("incoming_retention_backlog")
    return {
        "ok": not operational_rejections,
        "degraded": bool(degraded_reasons),
        "degraded_reasons": list(dict.fromkeys(degraded_reasons)),
        "native_generation_unaffected": True,
        "generated_at": utc_now(),
        "accepted": accepted,
        "rejected": rejected,
        "archived": archived,
        "oversized_discarded": oversized_discarded,
        "remaining": retention["incoming"]["retained_observed_json_files"],
        "remaining_scan_truncated": retention["incoming"]["scan_truncated"],
        "dedup_index": {
            "items": max(len(known_ids), len(known_digests)),
            "item_limit": MAX_DEDUP_INDEX_ITEMS,
            "at_capacity": dedup_at_capacity,
            "capacity_rejections": dedup_capacity_rejections,
        },
        "retention": retention,
        "safety": dict(SAFETY),
    }


def _read_json_object(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.exists() or not path.is_file():
        return {}
    try:
        if path.stat().st_size > max_bytes:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_scalar(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return [_safe_scalar(item, depth=depth + 1) for item in value[:40]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _safe_scalar(item, depth=depth + 1)
            for key, item in list(value.items())[:60]
            if not _is_forbidden_key(key) and not _is_holdout_key(key)
        }
    return None


def _is_holdout_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(marker in normalized for marker in HOLDOUT_MARKERS)


def _development_reasons(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        if _is_holdout_key(key):
            continue
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[str(key)[:100]] = count
    return result


def _safe_trade_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _safe_scalar(value.get(key))
        for key in (
            "trades",
            "wins",
            "win_rate",
            "net_return_sum",
            "sized_return_sum",
            "last_exit_time",
            "invalid_rows",
        )
        if key in value
    }


def _safe_operational_snapshot(path: Path) -> dict[str, Any]:
    report = _read_json_object(path)
    products: list[dict[str, Any]] = []
    for item in report.get("products") or []:
        if not isinstance(item, dict):
            continue
        products.append(
            {
                key: _safe_scalar(item.get(key))
                for key in (
                    "name",
                    "objective",
                    "market",
                    "mode",
                    "enabled",
                    "cycle_ok",
                    "skipped",
                    "reason",
                    "error",
                    "equity",
                    "peak_equity",
                    "drawdown_fraction",
                    "drawdown_limit_fraction",
                    "drawdown_halted",
                    "drawdown_halt_reason",
                    "open_positions",
                    "open_position_details",
                )
                if key in item
            }
            | {"trade_summary": _safe_trade_summary(item.get("trade_summary"))}
        )
    scheduled_jobs: list[dict[str, Any]] = []
    for item in report.get("scheduled_jobs") or []:
        if not isinstance(item, dict):
            continue
        scheduled_jobs.append(
            {
                key: _safe_scalar(item.get(key))
                for key in (
                    "name",
                    "enabled",
                    "status",
                    "due",
                    "last_ok",
                    "last_started_at",
                    "last_duration_seconds",
                    "last_reason",
                    "last_error",
                    "consecutive_failures",
                    "consecutive_deferrals",
                    "cadence_seconds",
                )
                if key in item
            }
        )
    control = report.get("control") if isinstance(report.get("control"), dict) else {}
    candidate_paper = (
        report.get("candidate_paper") if isinstance(report.get("candidate_paper"), dict) else {}
    )
    job_worker = report.get("job_worker") if isinstance(report.get("job_worker"), dict) else {}
    return {
        "source_generated_at": report.get("generated_at"),
        "ok": _safe_scalar(report.get("ok")),
        "runtime_ok": _safe_scalar(report.get("runtime_ok")),
        "control": {
            key: _safe_scalar(control.get(key))
            for key in ("paused", "pause_jobs", "paused_products", "paused_jobs", "reason")
            if key in control
        },
        "products": products,
        "job_worker": {
            key: _safe_scalar(job_worker.get(key))
            for key in (
                "ok",
                "fresh",
                "last_cycle_ok",
                "last_cycle_reason",
                "generated_at",
            )
            if key in job_worker
        },
        "scheduled_jobs": scheduled_jobs,
        "candidate_paper": {
            key: _safe_scalar(candidate_paper.get(key))
            for key in (
                "ok",
                "status",
                "reason",
                "generated_at",
                "open_positions",
                "activation_ready_products",
                "drawdown_halted_products",
                "products",
            )
            if key in candidate_paper
        },
        "operational_issues": _safe_scalar(report.get("operational_issues", [])),
        "position_alerts": _safe_scalar(report.get("position_alerts", [])),
    }


def _safe_strategy_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _safe_scalar(value.get(key))
        for key in (
            "direction",
            "base_timeframe",
            "regime_timeframe",
            "setup_timeframe",
            "trigger_timeframe",
            "regime",
            "setup",
            "trigger",
            "exit",
            "risk",
            "_product",
            "_market",
            "_pnl_unit",
            "_symbol",
        )
        if key in value
    }


def _stored_json_or_default(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed


def _hypothesis_workspace(memory_path: Path) -> dict[str, Any]:
    if memory_path.is_symlink() or not memory_path.is_file():
        return {"available": False, "hypotheses": [], "count": 0, "truncated": False}
    connection: sqlite3.Connection | None = None
    try:
        uri = f"file:{memory_path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        rows = list(
            connection.execute(
                """
                SELECT * FROM strategies
                WHERE retired_at IS NULL AND holdout_exposed_at IS NULL
                ORDER BY created_at DESC, behavior_hash
                LIMIT ?
                """,
                (MAX_WORKSPACE_HYPOTHESES + 1,),
            )
        )
        truncated = len(rows) > MAX_WORKSPACE_HYPOTHESES
        rows = rows[:MAX_WORKSPACE_HYPOTHESES]
        hypotheses: list[dict[str, Any]] = []
        for row in rows:
            behavior_hash = str(row["behavior_hash"])
            parent_rows = connection.execute(
                """
                SELECT parent_hash FROM lineage_edges
                WHERE child_hash = ? ORDER BY parent_ordinal, parent_hash
                """,
                (behavior_hash,),
            ).fetchall()
            evaluations: list[dict[str, Any]] = []
            for evaluation in connection.execute(
                """
                SELECT * FROM evaluations
                WHERE behavior_hash = ?
                ORDER BY COALESCE(completed_at, claimed_at) DESC, evaluation_key DESC
                LIMIT ?
                """,
                (behavior_hash, MAX_WORKSPACE_EVALUATIONS_PER_HYPOTHESIS * 3),
            ):
                phase = str(evaluation["phase"] or "")
                if _is_holdout_key(phase):
                    continue
                evaluations.append(
                    {
                        "evaluation_key": evaluation["evaluation_key"],
                        "phase": phase,
                        "status": evaluation["status"],
                        "claimed_at": evaluation["claimed_at"],
                        "completed_at": evaluation["completed_at"],
                        "outcome": evaluation["outcome"],
                        "rejection_reasons": _safe_scalar(
                            _stored_json_or_default(evaluation["rejection_reasons_json"], [])
                        ),
                        "metrics": _safe_scalar(
                            _stored_json_or_default(evaluation["metrics_json"], {})
                        ),
                        "details": _safe_scalar(
                            _stored_json_or_default(evaluation["details_json"], {})
                        ),
                    }
                )
                if len(evaluations) >= MAX_WORKSPACE_EVALUATIONS_PER_HYPOTHESIS:
                    break
            metadata = _stored_json_or_default(row["metadata_json"], {})
            hypotheses.append(
                {
                    "hypothesis_id": behavior_hash,
                    "strategy_id": row["primary_strategy_id"],
                    "created_at": row["created_at"],
                    "generation_method": row["generation_method"],
                    "parent_hypothesis_ids": [str(parent["parent_hash"]) for parent in parent_rows],
                    "lineage_depth": _safe_scalar(
                        metadata.get("lineage_depth") if isinstance(metadata, dict) else 0
                    ),
                    "product": _safe_scalar(
                        metadata.get("product") if isinstance(metadata, dict) else row["product"]
                    ),
                    "opportunity_type": _safe_scalar(
                        metadata.get("opportunity_type")
                        if isinstance(metadata, dict)
                        else row["opportunity_type"]
                    ),
                    "symbol": _safe_scalar(
                        metadata.get("symbol", "BTCUSDT")
                        if isinstance(metadata, dict)
                        else "BTCUSDT"
                    ),
                    "base_timeframe": _safe_scalar(
                        metadata.get("base_timeframe") if isinstance(metadata, dict) else None
                    ),
                    "proposal_id": _safe_scalar(
                        metadata.get("proposal_id") if isinstance(metadata, dict) else None
                    ),
                    "novelty_score": _safe_scalar(row["novelty_score"]),
                    "spec": _safe_strategy_spec(
                        _stored_json_or_default(row["primary_spec_json"], {})
                    ),
                    "latest_evaluation": evaluations[0] if evaluations else None,
                    "evaluation_history": evaluations,
                }
            )
        return {
            "available": True,
            "count": len(hypotheses),
            "truncated": truncated,
            "hypotheses": hypotheses,
        }
    except (OSError, sqlite3.DatabaseError):
        return {"available": False, "hypotheses": [], "count": 0, "truncated": False}
    finally:
        if connection is not None:
            connection.close()


def _action_history(path: Path, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = _read_json_object(path, max_bytes=16 * 1024 * 1024)
    processed = state.get("processed") if isinstance(state.get("processed"), dict) else {}
    latest_by_hash = {
        item["hypothesis_id"]: item.get("latest_evaluation")
        for item in hypotheses
        if isinstance(item, dict) and isinstance(item.get("hypothesis_id"), str)
    }
    items: list[dict[str, Any]] = []
    for proposal_id, disposition in processed.items():
        if not isinstance(disposition, dict):
            continue
        strategy_hash = disposition.get("strategy_hash") or disposition.get("parent_hypothesis_id")
        latest_result = latest_by_hash.get(strategy_hash)
        if isinstance(latest_result, dict):
            result_at = latest_result.get("completed_at") or latest_result.get("claimed_at")
            processed_at = disposition.get("processed_at")
            try:
                result_time = datetime.fromisoformat(str(result_at).replace("Z", "+00:00"))
                action_time = datetime.fromisoformat(str(processed_at).replace("Z", "+00:00"))
            except ValueError:
                latest_result = None
            else:
                if result_time <= action_time:
                    latest_result = None
        items.append(
            {
                key: _safe_scalar(disposition.get(key))
                for key in (
                    "processed_at",
                    "status",
                    "reason",
                    "action",
                    "parent_hypothesis_id",
                    "strategy_hash",
                    "objective",
                    "opportunity_type",
                    "symbol",
                    "thesis",
                    "reasoning",
                    "changes",
                    "expected_outcome",
                    "falsification_criteria",
                )
                if key in disposition
            }
            | {
                "proposal_id": str(proposal_id)[:160],
                "latest_result": latest_result,
            }
        )
    items.sort(key=lambda item: str(item.get("processed_at") or ""), reverse=True)
    return items[:MAX_WORKSPACE_ACTION_HISTORY]


def _review_history(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 512 * 1024))
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    items: list[dict[str, Any]] = []
    for line in lines[-MAX_REVIEW_HISTORY:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(
                {
                    key: _safe_scalar(item.get(key))
                    for key in (
                        "recorded_at",
                        "run_id",
                        "model",
                        "summary",
                        "proposal_count",
                        "action_counts",
                    )
                    if key in item
                }
            )
    return items


def build_research_context(
    *,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    generated_batch_path: Path = DEFAULT_GENERATED_BATCH,
    market_universe_path: Path = DEFAULT_MARKET_UNIVERSE,
    operator_report_path: Path = DEFAULT_OPERATOR_REPORT,
    experiment_memory_path: Path = DEFAULT_EXPERIMENT_MEMORY,
    proposal_state_path: Path = DEFAULT_PROPOSAL_STATE,
    review_audit_path: Path = DEFAULT_REVIEW_AUDIT,
) -> dict[str, Any]:
    """Build the allowlisted operational and research workspace for Alfred."""

    research = _read_json_object(research_cycle_path)
    batch = _read_json_object(generated_batch_path)
    universe = _read_json_object(market_universe_path)
    research_summary = research.get("summary") if isinstance(research.get("summary"), dict) else {}
    batch_summary = batch.get("summary") if isinstance(batch.get("summary"), dict) else {}
    memory = batch.get("memory") if isinstance(batch.get("memory"), dict) else {}
    feedback = memory.get("feedback") if isinstance(memory.get("feedback"), dict) else {}
    progress = {
        key: _safe_scalar(research_summary.get(key))
        for key in SAFE_PROGRESS_KEYS
        if key in research_summary
    }
    progress["development_failure_reasons"] = _development_reasons(
        research_summary.get("top_reasons")
    )
    novelty = {
        key: _safe_scalar(research_summary.get(key))
        for key in SAFE_NOVELTY_KEYS
        if key in research_summary
    }
    feedback_reasons = _development_reasons(feedback.get("rejection_reasons"))
    feedback_outcomes = _development_reasons(feedback.get("outcomes"))
    feedback_totals = feedback.get("totals") if isinstance(feedback.get("totals"), dict) else {}
    safe_totals = {
        key: _safe_scalar(feedback_totals.get(key))
        for key in (
            "strategies",
            "identities",
            "duplicate_identities",
            "evaluations",
            "claimed",
            "completed",
            "retired",
        )
        if key in feedback_totals
    }
    hypothesis_workspace = _hypothesis_workspace(experiment_memory_path)
    hypotheses = hypothesis_workspace.get("hypotheses") or []
    return {
        "schema": CONTEXT_SCHEMA,
        "generated_at": utc_now(),
        "purpose": "alfred_autonomous_research_supervision",
        "operational_snapshot": _safe_operational_snapshot(operator_report_path),
        "objectives": [
            {
                "name": "btc_accumulation",
                "base_asset": "BTC",
                "market": "spot",
                "constraints": ["no leverage", "measure performance in BTC"],
            },
            {
                "name": "active_income",
                "base_asset": "USDT",
                "market": "futures",
                "constraints": ["strict risk controls", "scalp/day/swing opportunities"],
            },
        ],
        "active_income_universe": {
            "source_generated_at": universe.get("generated_at"),
            "research_symbols": _safe_scalar(universe.get("research_symbols", [])),
            "eligible_research_symbols": _safe_scalar(
                universe.get("eligible_research_symbols", [])
            ),
            "criteria": _safe_scalar(universe.get("criteria", {})),
        },
        "research_progress": {
            "source_generated_at": research.get("generated_at"),
            **progress,
            "novelty": novelty,
        },
        "experiment_memory": {
            "totals": safe_totals,
            "development_outcomes": feedback_outcomes,
            "development_failure_reasons": feedback_reasons,
            "generation_methods": _safe_scalar(feedback.get("generation_methods", {})),
            "families": _safe_scalar(feedback.get("families", {})),
            "primitives": _safe_scalar(feedback.get("primitives", {})),
        },
        "generated_batch": {
            "source_generated_at": batch.get("generated_at"),
            "hypotheses": _safe_scalar(batch_summary.get("hypotheses", batch.get("count", 0))),
            "by_product": _safe_scalar(batch_summary.get("by_product", {})),
            "by_space": _safe_scalar(batch_summary.get("by_space", {})),
            "by_method": _safe_scalar(batch_summary.get("by_method", {})),
            "new_hypotheses": _safe_scalar(batch_summary.get("new_hypotheses", 0)),
            "resumed_pending": _safe_scalar(batch_summary.get("resumed_pending", 0)),
            "unique_behavioral_specs": _safe_scalar(
                batch_summary.get("unique_behavioral_specs", 0)
            ),
            "cumulative_trials": _safe_scalar(batch_summary.get("cumulative_trials", 0)),
        },
        "hypothesis_workspace": hypothesis_workspace,
        "action_history": _action_history(proposal_state_path, hypotheses),
        "review_history": _review_history(review_audit_path),
        "proposal_contract": {
            "drop_directory": str(DEFAULT_INCOMING),
            "schema": ACTION_SCHEMA,
            "source": "openclaw",
            "actions": sorted(ACTION_TYPES),
            "required_fields": [
                "schema",
                "source",
                "created_at",
                "action",
                "objective",
                "opportunity_type",
                "base_timeframe",
                "thesis",
                "reasoning",
                "expected_outcome",
                "falsification_criteria",
            ],
            "optional_fields": [
                "parent_hypothesis_id",
                "changes",
                "suggested_primitives",
                "constraints",
                "suggested_spec",
                "symbol",
                "provenance",
                "source_proposal_id",
            ],
        },
        "boundary": {
            "untrusted_input": True,
            "research_only": True,
            "credentials_excluded": True,
            "approvals_excluded": True,
            "operational_state_allowlisted": True,
            "live_controls_read_only_in_workspace": True,
            "final_holdout_feedback_excluded": True,
            "direct_strategy_import_forbidden": True,
            "autonomous_live_promotion_forbidden": True,
            "autonomous_order_placement_forbidden": True,
            "autonomous_risk_increase_forbidden": True,
        },
    }


def _event_semantics(context: dict[str, Any]) -> dict[str, Any]:
    operational = context.get("operational_snapshot")
    operational = operational if isinstance(operational, dict) else {}
    products = operational.get("products") if isinstance(operational.get("products"), list) else []
    hypotheses = context.get("hypothesis_workspace")
    hypotheses = hypotheses if isinstance(hypotheses, dict) else {}
    hypothesis_items = (
        hypotheses.get("hypotheses") if isinstance(hypotheses.get("hypotheses"), list) else []
    )
    actions = context.get("action_history")
    actions = actions if isinstance(actions, list) else []
    return {
        "operational_ok": operational.get("ok"),
        "runtime_ok": operational.get("runtime_ok"),
        "products": [
            {
                key: item.get(key)
                for key in (
                    "name",
                    "cycle_ok",
                    "error",
                    "drawdown_halted",
                    "drawdown_fraction",
                    "open_positions",
                    "equity",
                    "trade_summary",
                )
            }
            for item in products
            if isinstance(item, dict)
        ],
        "job_worker": {
            key: (operational.get("job_worker") or {}).get(key)
            for key in ("ok", "fresh", "last_cycle_ok", "last_cycle_reason")
            if isinstance(operational.get("job_worker"), dict)
        },
        "scheduled_jobs": [
            {
                key: item.get(key)
                for key in (
                    "name",
                    "status",
                    "last_ok",
                    "last_error",
                    "consecutive_failures",
                )
            }
            for item in operational.get("scheduled_jobs") or []
            if isinstance(item, dict)
        ],
        "operational_issues": operational.get("operational_issues"),
        "position_alerts": operational.get("position_alerts"),
        "hypotheses": [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "latest_evaluation": item.get("latest_evaluation"),
            }
            for item in hypothesis_items
            if isinstance(item, dict)
        ],
        "actions": [
            {
                "proposal_id": item.get("proposal_id"),
                "status": item.get("status"),
                "strategy_hash": item.get("strategy_hash"),
                "latest_result": item.get("latest_result"),
            }
            for item in actions[:50]
            if isinstance(item, dict)
        ],
    }


def _material_event_reasons(before: Any, after: dict[str, Any]) -> list[str]:
    if not isinstance(before, dict):
        return []
    reasons: list[str] = []
    if before.get("operational_ok") != after.get("operational_ok"):
        reasons.append("operational_health_changed")
    if before.get("runtime_ok") != after.get("runtime_ok"):
        reasons.append("runtime_health_changed")
    before_products = {
        item.get("name"): item
        for item in before.get("products") or []
        if isinstance(item, dict) and item.get("name")
    }
    for item in after.get("products") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        prior = before_products.get(item["name"])
        if prior is None:
            continue
        if prior.get("cycle_ok") != item.get("cycle_ok") or prior.get("error") != item.get("error"):
            reasons.append(f"product_health_changed:{item['name']}")
        if prior.get("drawdown_halted") != item.get("drawdown_halted"):
            reasons.append(f"drawdown_halt_changed:{item['name']}")
        if prior.get("open_positions") != item.get("open_positions"):
            reasons.append(f"position_count_changed:{item['name']}")
        prior_trades = (
            prior.get("trade_summary", {}).get("trades")
            if isinstance(prior.get("trade_summary"), dict)
            else None
        )
        current_trades = (
            item.get("trade_summary", {}).get("trades")
            if isinstance(item.get("trade_summary"), dict)
            else None
        )
        if prior_trades != current_trades:
            reasons.append(f"paper_trade_result_changed:{item['name']}")
        try:
            drawdown_delta = abs(
                float(item.get("drawdown_fraction") or 0)
                - float(prior.get("drawdown_fraction") or 0)
            )
        except (TypeError, ValueError):
            drawdown_delta = 0
        if drawdown_delta >= 0.01:
            reasons.append(f"material_drawdown_change:{item['name']}")
        try:
            prior_equity = float(prior.get("equity"))
            current_equity = float(item.get("equity"))
            equity_change = abs(current_equity - prior_equity) / max(abs(prior_equity), 1e-12)
        except (TypeError, ValueError):
            equity_change = 0
        if equity_change >= 0.01:
            reasons.append(f"material_equity_change:{item['name']}")
    before_hypotheses = {
        item.get("hypothesis_id"): item.get("latest_evaluation")
        for item in before.get("hypotheses") or []
        if isinstance(item, dict) and item.get("hypothesis_id")
    }
    for item in after.get("hypotheses") or []:
        if not isinstance(item, dict) or not item.get("hypothesis_id"):
            continue
        prior = before_hypotheses.get(item["hypothesis_id"])
        latest = item.get("latest_evaluation")
        if prior != latest and latest is not None:
            reasons.append(f"research_result_completed:{item['hypothesis_id']}")
    if before.get("actions") != after.get("actions"):
        reasons.append("research_action_disposition_changed")
    if before.get("job_worker") != after.get("job_worker"):
        reasons.append("job_worker_state_changed")
    if before.get("scheduled_jobs") != after.get("scheduled_jobs"):
        reasons.append("scheduled_job_state_changed")
    if before.get("operational_issues") != after.get("operational_issues"):
        reasons.append("operational_issue_changed")
    if before.get("position_alerts") != after.get("position_alerts"):
        reasons.append("position_or_risk_alert_changed")
    return list(dict.fromkeys(reasons))[:40]


def update_event_state(
    context: dict[str, Any],
    *,
    event_path: Path = DEFAULT_EVENT_STATE,
) -> dict[str, Any]:
    semantics = _event_semantics(context)
    previous = _read_json_object(event_path)
    reasons = _material_event_reasons(previous.get("semantics"), semantics)
    payload = {
        "schema": "autopilot.openclaw_event_state/v1",
        "updated_at": utc_now(),
        "pending": bool(previous.get("pending")) or bool(reasons),
        "reasons": list(dict.fromkeys([*(previous.get("reasons") or []), *reasons]))[:80],
        "first_pending_at": (
            previous.get("first_pending_at")
            if previous.get("pending")
            else (utc_now() if reasons else None)
        ),
        "last_wake_at": previous.get("last_wake_at"),
        "last_acknowledged_at": previous.get("last_acknowledged_at"),
        "semantics": semantics,
    }
    write_json_atomic(event_path, payload)
    event_path.chmod(0o600)
    return payload


def claim_review_event(
    *,
    event_path: Path = DEFAULT_EVENT_STATE,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = _read_json_object(event_path)
    if not payload.get("pending"):
        return {"claimed": False, "reason": "no_pending_event"}
    current = now or datetime.now(UTC)
    last_wake_at = payload.get("last_wake_at")
    if isinstance(last_wake_at, str):
        try:
            last_wake = datetime.fromisoformat(last_wake_at.replace("Z", "+00:00"))
        except ValueError:
            last_wake = None
        if (
            last_wake is not None
            and (current - last_wake.astimezone(UTC)).total_seconds() < EVENT_WAKE_COOLDOWN_SECONDS
        ):
            return {"claimed": False, "reason": "wake_cooldown"}
    payload["last_wake_at"] = current.astimezone(UTC).isoformat()
    write_json_atomic(event_path, payload)
    event_path.chmod(0o600)
    return {
        "claimed": True,
        "reasons": _safe_scalar(payload.get("reasons", [])),
        "first_pending_at": payload.get("first_pending_at"),
    }


def acknowledge_review_event(
    *,
    event_path: Path = DEFAULT_EVENT_STATE,
) -> None:
    payload = _read_json_object(event_path)
    if not payload:
        return
    payload.update(
        pending=False,
        reasons=[],
        first_pending_at=None,
        last_acknowledged_at=utc_now(),
    )
    write_json_atomic(event_path, payload)
    event_path.chmod(0o600)


def export_research_context(
    output_path: Path = DEFAULT_CONTEXT,
    *,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    generated_batch_path: Path = DEFAULT_GENERATED_BATCH,
    market_universe_path: Path = DEFAULT_MARKET_UNIVERSE,
    operator_report_path: Path = DEFAULT_OPERATOR_REPORT,
    experiment_memory_path: Path = DEFAULT_EXPERIMENT_MEMORY,
    proposal_state_path: Path = DEFAULT_PROPOSAL_STATE,
    review_audit_path: Path = DEFAULT_REVIEW_AUDIT,
    event_path: Path | None = None,
) -> dict[str, Any]:
    context = build_research_context(
        research_cycle_path=research_cycle_path,
        generated_batch_path=generated_batch_path,
        market_universe_path=market_universe_path,
        operator_report_path=operator_report_path,
        experiment_memory_path=experiment_memory_path,
        proposal_state_path=proposal_state_path,
        review_audit_path=review_audit_path,
    )
    if output_path.is_symlink():
        raise ValueError(f"OpenClaw context output must not be a symlink: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod_if_needed(output_path.parent, 0o2750 if _shared_group_enabled() else 0o700)
    write_json_atomic(output_path, context)
    output_path.chmod(0o640 if _shared_group_enabled() else 0o600)
    update_event_state(
        context,
        event_path=event_path or output_path.with_name("event_state.json"),
    )
    return context


def record_review(
    *,
    audit_path: Path = DEFAULT_REVIEW_AUDIT,
    context_path: Path = DEFAULT_CONTEXT,
    run_id: str,
    model: str,
    summary: str,
    proposal_count: int,
    action_counts: dict[str, int] | None = None,
    event_path: Path | None = None,
) -> dict[str, Any]:
    """Append a bounded receipt for every Alfred review, including no-op reviews."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ProposalValidationError("run_id contains unsupported characters")
    if not 1 <= len(model.strip()) <= 120:
        raise ProposalValidationError("model must be 1-120 characters")
    if not 1 <= len(summary.strip()) <= 1000:
        raise ProposalValidationError("summary must be 1-1000 characters")
    if not 0 <= proposal_count <= MAX_BATCH:
        raise ProposalValidationError(f"proposal_count must be between 0 and {MAX_BATCH}")
    action_counts = action_counts or {}
    if (
        not isinstance(action_counts, dict)
        or set(action_counts) - ACTION_TYPES
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > MAX_BATCH
            for value in action_counts.values()
        )
        or sum(action_counts.values()) != proposal_count
    ):
        raise ProposalValidationError(
            "action_counts must contain known actions with non-negative integer counts "
            "summing to proposal_count"
        )
    if context_path.is_symlink() or not context_path.is_file():
        raise ProposalValidationError("research context must be a regular file")
    context_digest = "sha256:" + hashlib.sha256(context_path.read_bytes()).hexdigest()
    receipt = {
        "schema": "autopilot.alfred_research_review/v2",
        "recorded_at": utc_now(),
        "run_id": run_id,
        "model": model.strip(),
        "summary": summary.strip(),
        "proposal_count": proposal_count,
        "action_counts": dict(sorted(action_counts.items())),
        "context_digest": context_digest,
        "research_only": True,
        "live_allowed": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n"
    with audit_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    acknowledge_review_event(event_path=event_path or context_path.with_name("event_state.json"))
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sanitized OpenClaw context or ingest inert proposals."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Write sanitized research context for OpenClaw.")
    export.add_argument("--output", type=Path, default=DEFAULT_CONTEXT)
    export.add_argument("--research-cycle", type=Path, default=DEFAULT_RESEARCH_CYCLE)
    export.add_argument("--generated-batch", type=Path, default=DEFAULT_GENERATED_BATCH)
    export.add_argument("--market-universe", type=Path, default=DEFAULT_MARKET_UNIVERSE)
    export.add_argument("--operator-report", type=Path, default=DEFAULT_OPERATOR_REPORT)
    export.add_argument("--experiment-memory", type=Path, default=DEFAULT_EXPERIMENT_MEMORY)
    export.add_argument("--proposal-state", type=Path, default=DEFAULT_PROPOSAL_STATE)
    export.add_argument("--review-audit", type=Path, default=DEFAULT_REVIEW_AUDIT)
    export.add_argument("--event-state", type=Path, default=DEFAULT_EVENT_STATE)
    ingest = subparsers.add_parser("ingest", help="Validate and archive untrusted proposal files.")
    ingest.add_argument("--incoming", type=Path, default=DEFAULT_INCOMING)
    ingest.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    ingest.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    ingest.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ingest.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest.add_argument("--status", type=Path, default=DEFAULT_INGEST_STATUS)
    ingest.add_argument("--max-batch", type=int, default=MAX_BATCH)
    review = subparsers.add_parser(
        "record-review", help="Append an auditable receipt for an OpenClaw daily review."
    )
    review.add_argument("--audit", type=Path, default=DEFAULT_REVIEW_AUDIT)
    review.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    review.add_argument("--run-id", required=True)
    review.add_argument("--model", required=True)
    review.add_argument("--summary", required=True)
    review.add_argument("--proposal-count", type=int, required=True)
    review.add_argument(
        "--action-counts-json",
        default="{}",
        help='Action counts as JSON, for example {"new":1,"revise":1}.',
    )
    review.add_argument("--event-state", type=Path, default=DEFAULT_EVENT_STATE)
    event = subparsers.add_parser(
        "claim-event",
        help="Claim a pending material-event wake; exits 1 when no wake is due.",
    )
    event.add_argument("--event-state", type=Path, default=DEFAULT_EVENT_STATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "export":
        payload = export_research_context(
            args.output,
            research_cycle_path=args.research_cycle,
            generated_batch_path=args.generated_batch,
            market_universe_path=args.market_universe,
            operator_report_path=args.operator_report,
            experiment_memory_path=args.experiment_memory,
            proposal_state_path=args.proposal_state,
            review_audit_path=args.review_audit,
            event_path=args.event_state,
        )
    elif args.command == "ingest":
        payload = ingest_inbox(
            incoming_dir=args.incoming,
            accepted_dir=args.accepted,
            rejected_dir=args.rejected,
            archive_dir=args.archive,
            index_path=args.index,
            max_batch=args.max_batch,
        )
        write_json_atomic(args.status, payload)
        args.status.chmod(0o600)
    elif args.command == "record-review":
        try:
            action_counts = json.loads(args.action_counts_json)
        except json.JSONDecodeError as exc:
            raise ProposalValidationError("action-counts-json must be valid JSON") from exc
        payload = record_review(
            audit_path=args.audit,
            context_path=args.context,
            run_id=args.run_id,
            model=args.model,
            summary=args.summary,
            proposal_count=args.proposal_count,
            action_counts=action_counts,
            event_path=args.event_state,
        )
    else:
        payload = claim_review_event(event_path=args.event_state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command == "ingest" and not payload.get("ok"):
        raise SystemExit(1)
    if args.command == "claim-event" and not payload.get("claimed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
