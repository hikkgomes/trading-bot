"""One-way, credential-free bridge between OpenClaw and trusted research code.

OpenClaw runs as a separate service/user.  This module never launches it and
never passes environment variables to it.  The outward context is allowlisted
and excludes final-holdout outcomes.  Inbound proposals are untrusted inert
records; they are not strategy artifacts and cannot be executed or promoted.
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
DEFAULT_CONTEXT = Path("runtime/openclaw/research_context.json")
DEFAULT_REVIEW_AUDIT = Path("runtime/openclaw/review_audit.jsonl")
DEFAULT_RESEARCH_CYCLE = Path("runtime/research_cycle.json")
DEFAULT_GENERATED_BATCH = Path("runtime/research/generated_hypotheses.json")
DEFAULT_MARKET_UNIVERSE = Path("runtime/market_universe.json")
PROPOSAL_SCHEMA = "research_proposal/v1"
ACCEPTED_SCHEMA = "autopilot.openclaw_research_proposal/v1"
CONTEXT_SCHEMA = "autopilot.openclaw_research_context/v1"
OBJECTIVES = frozenset({"active_income", "btc_accumulation"})
OPPORTUNITY_TYPES = frozenset({"day", "position", "scalp", "swing"})
TIMEFRAMES = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
)
INPUT_KEYS = frozenset(
    {
        "base_timeframe",
        "constraints",
        "created_at",
        "objective",
        "opportunity_type",
        "provenance",
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
    if payload.get("schema") != PROPOSAL_SCHEMA:
        raise ProposalValidationError(f"schema must be {PROPOSAL_SCHEMA!r}")
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
        "objective": objective,
        "opportunity_type": opportunity_type,
        "base_timeframe": base_timeframe,
        "thesis": thesis,
        "symbol": symbol,
        "suggested_primitives": _bounded_string_list(payload, "suggested_primitives"),
        "constraints": _bounded_string_list(payload, "constraints"),
        "untrusted_suggested_spec": suggested_spec,
        "provenance": _validate_provenance(payload),
    }
    if source_proposal_id is not None:
        normalized["source_proposal_id"] = source_proposal_id
    return normalized


def canonical_proposal_digest(proposal: dict[str, Any]) -> str:
    semantic = {
        key: proposal.get(key)
        for key in (
            "base_timeframe",
            "constraints",
            "objective",
            "opportunity_type",
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
        "proposal_id": f"openclaw-{digest.removeprefix('sha256:')[:20]}",
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
        "ok": not any(item.get("reason", "").startswith("archive_failed") for item in rejected),
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


def build_research_context(
    *,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    generated_batch_path: Path = DEFAULT_GENERATED_BATCH,
    market_universe_path: Path = DEFAULT_MARKET_UNIVERSE,
) -> dict[str, Any]:
    """Build allowlisted research feedback with no final-holdout information."""

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
    return {
        "schema": CONTEXT_SCHEMA,
        "generated_at": utc_now(),
        "purpose": "sanitized_research_proposal_context",
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
        "proposal_contract": {
            "drop_directory": str(DEFAULT_INCOMING),
            "schema": PROPOSAL_SCHEMA,
            "source": "openclaw",
            "required_fields": [
                "schema",
                "source",
                "created_at",
                "objective",
                "opportunity_type",
                "base_timeframe",
                "thesis",
            ],
            "optional_fields": [
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
            "execution_state_excluded": True,
            "live_controls_excluded": True,
            "final_holdout_feedback_excluded": True,
            "direct_strategy_import_forbidden": True,
        },
    }


def export_research_context(
    output_path: Path = DEFAULT_CONTEXT,
    *,
    research_cycle_path: Path = DEFAULT_RESEARCH_CYCLE,
    generated_batch_path: Path = DEFAULT_GENERATED_BATCH,
    market_universe_path: Path = DEFAULT_MARKET_UNIVERSE,
) -> dict[str, Any]:
    context = build_research_context(
        research_cycle_path=research_cycle_path,
        generated_batch_path=generated_batch_path,
        market_universe_path=market_universe_path,
    )
    if output_path.is_symlink():
        raise ValueError(f"OpenClaw context output must not be a symlink: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod_if_needed(output_path.parent, 0o2750 if _shared_group_enabled() else 0o700)
    write_json_atomic(output_path, context)
    output_path.chmod(0o640 if _shared_group_enabled() else 0o600)
    return context


def record_review(
    *,
    audit_path: Path = DEFAULT_REVIEW_AUDIT,
    context_path: Path = DEFAULT_CONTEXT,
    run_id: str,
    model: str,
    summary: str,
    proposal_count: int,
) -> dict[str, Any]:
    """Append a bounded receipt for every OpenClaw review, including no-op reviews."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ProposalValidationError("run_id contains unsupported characters")
    if not 1 <= len(model.strip()) <= 120:
        raise ProposalValidationError("model must be 1-120 characters")
    if not 1 <= len(summary.strip()) <= 1000:
        raise ProposalValidationError("summary must be 1-1000 characters")
    if not 0 <= proposal_count <= MAX_BATCH:
        raise ProposalValidationError(f"proposal_count must be between 0 and {MAX_BATCH}")
    if context_path.is_symlink() or not context_path.is_file():
        raise ProposalValidationError("research context must be a regular file")
    context_digest = "sha256:" + hashlib.sha256(context_path.read_bytes()).hexdigest()
    receipt = {
        "schema": "autopilot.openclaw_daily_review/v1",
        "recorded_at": utc_now(),
        "run_id": run_id,
        "model": model.strip(),
        "summary": summary.strip(),
        "proposal_count": proposal_count,
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
    ingest = subparsers.add_parser("ingest", help="Validate and archive untrusted proposal files.")
    ingest.add_argument("--incoming", type=Path, default=DEFAULT_INCOMING)
    ingest.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    ingest.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    ingest.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ingest.add_argument("--index", type=Path, default=DEFAULT_INDEX)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "export":
        payload = export_research_context(
            args.output,
            research_cycle_path=args.research_cycle,
            generated_batch_path=args.generated_batch,
            market_universe_path=args.market_universe,
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
    else:
        payload = record_review(
            audit_path=args.audit,
            context_path=args.context,
            run_id=args.run_id,
            model=args.model,
            summary=args.summary,
            proposal_count=args.proposal_count,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
