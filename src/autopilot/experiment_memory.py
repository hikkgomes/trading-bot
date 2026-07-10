"""Durable behavioral memory for autonomous strategy research.

The research engine needs a different identity from the human-facing hypothesis
ID used in reports.  Two hypotheses with different prose (or IDs) but the same
executable rules are the same experiment.  This module gives the generator a
small, stdlib-only SQLite database that records that behavioral identity,
lineage, immutable data/protocol exposure, and results.

There is deliberately no execution or promotion API here.  Entries in this
store are research evidence only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
import threading
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MEMORY_FORMAT = "autonomous_strategy_experiment_memory"
DEFAULT_MEMORY_PATH = Path("runtime/experiment_memory.sqlite3")
MAX_PARENTS = 16
MAX_QUERY_LIMIT = 500
MAX_COMPACTION_ROWS = 5_000
MAX_DECOMPRESSED_JSON_BYTES = 64 * 1024 * 1024
COMPACT_JSON_PREFIX = "zlib-json-v1:"
ENGINE_SCOPE_BACKFILL_META = "evaluation_engine_scopes_backfill_v1"
PROTECTED_PHASES = frozenset({"holdout", "final_holdout", "final"})

# These fields describe or identify an idea, but do not change its executable
# behavior.  They are ignored only at the top level; a nested ``id`` may be a
# real DSL component identifier and therefore remains significant.
_TOP_LEVEL_NON_BEHAVIORAL = frozenset(
    {
        "id",
        "strategy_id",
        "name",
        "title",
        "description",
        "idea",
        "market_logic",
        "rationale",
        "expected_holding",
        "expected_frequency",
        "invalidation",
        "feature_columns",
        "family",
        "tags",
        "note",
        "notes",
        "lineage",
        "mutation_lineage",
        "parent_hashes",
        "parents",
        "generation_method",
        "generator",
        "operator",
        "seed",
        "provenance",
        "created_at",
        "updated_at",
        "generated_at",
        "metrics",
        "validation",
        "result",
        "results",
        "safety",
        # Search-space names and horizon labels are research taxonomy, not
        # executable semantics. Product/market/PnL context remains behavioral.
        "_search_space",
        "_opportunity_type",
    }
)
_NESTED_NON_BEHAVIORAL = frozenset(
    {
        "note",
        "notes",
        "description",
        "label",
        "display_name",
        "provenance",
        "lineage",
        "created_at",
        "updated_at",
    }
)
_COMMUTATIVE_LIST_FIELDS = frozenset(
    {"regime", "setup", "trigger", "conditions", "filters", "all", "any", "and", "or"}
)
_PRIMITIVE_VALUE_FIELDS = frozenset(
    {
        "op",
        "operator",
        "feature",
        "feature_b",
        "indicator",
        "primitive",
        "component",
        "signal",
        "entry_type",
        "exit_type",
        "sizing_type",
    }
)


class ExperimentMemoryError(RuntimeError):
    """Base class for durable experiment-memory failures."""


class ExperimentMemoryCorruptionError(ExperimentMemoryError):
    """The database or its logical contents failed integrity validation."""


class ExperimentMemoryBusyError(ExperimentMemoryError):
    """Another writer held the database longer than the configured timeout."""


class ExperimentMemoryClosedError(ExperimentMemoryError):
    """The memory was used after its context/connection was closed."""


class StrategyIdentityConflictError(ExperimentMemoryError):
    """One external strategy ID was reused for different behavior."""


class EvaluationConflictError(ExperimentMemoryError):
    """An immutable evaluation was completed with conflicting evidence."""


@dataclass(frozen=True)
class StrategyRegistration:
    strategy_id: str
    behavior_hash: str
    created: bool
    identity_created: bool
    duplicate: bool
    novelty_score: float
    nearest_behavior_hash: str | None


@dataclass(frozen=True)
class EvaluationClaim:
    evaluation_key: str
    behavior_hash: str
    created: bool
    status: str
    holdout_consumed: bool


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_key: str
    behavior_hash: str
    created: bool
    completed: bool
    was_claimed: bool


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _validate_json(value: Any, *, label: str) -> Any:
    """Return a detached, strict-JSON value or raise a useful ValueError."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values: {exc}") from exc


def _normalise_behavior(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("strategy spec contains a non-finite number")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        ignored = _TOP_LEVEL_NON_BEHAVIORAL if depth == 0 else _NESTED_NON_BEHAVIORAL
        for child_key in sorted(value):
            if not isinstance(child_key, str):
                raise ValueError("strategy spec object keys must be strings")
            if child_key in ignored:
                continue
            normalised[child_key] = _normalise_behavior(
                value[child_key], key=child_key, depth=depth + 1
            )
        return normalised
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = [_normalise_behavior(item, depth=depth + 1) for item in value]
        if key in _COMMUTATIVE_LIST_FIELDS:
            items.sort(key=_canonical_json)
        return items
    raise ValueError(f"strategy spec contains unsupported value {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stored_json(value: str, *, label: str) -> Any:
    """Decode plain or compacted JSON with a strict decompression ceiling."""

    if not isinstance(value, str):
        raise ExperimentMemoryCorruptionError(f"{label} is not text")
    try:
        if value.startswith(COMPACT_JSON_PREFIX):
            encoded = value.removeprefix(COMPACT_JSON_PREFIX)
            compressed = base64.b85decode(encoded.encode("ascii"))
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(compressed, MAX_DECOMPRESSED_JSON_BYTES + 1)
            if (
                len(raw) > MAX_DECOMPRESSED_JSON_BYTES
                or decompressor.unconsumed_tail
                or not decompressor.eof
                or decompressor.unused_data
            ):
                raise ValueError("compressed JSON exceeds its safe decoding limit")
            text = raw.decode("utf-8")
        else:
            text = value
        return json.loads(text)
    except (UnicodeError, ValueError, zlib.error) as exc:
        raise ExperimentMemoryCorruptionError(f"invalid {label} in experiment memory") from exc


def _compact_json_storage(value: str, *, label: str) -> str:
    """Return a self-describing compressed representation of stored JSON."""

    if value.startswith(COMPACT_JSON_PREFIX):
        _stored_json(value, label=label)
        return value
    parsed = _stored_json(value, label=label)
    canonical = _canonical_json(parsed).encode("utf-8")
    compressed = zlib.compress(canonical, level=9)
    return COMPACT_JSON_PREFIX + base64.b85encode(compressed).decode("ascii")


def _research_engine_digest(protocol: Mapping[str, Any]) -> str | None:
    if not isinstance(protocol, Mapping):
        raise ExperimentMemoryCorruptionError("evaluation protocol JSON is not an object")
    value = protocol.get("research_engine_digest")
    if value is None:
        return None
    return _validate_hash(value, label="research_engine_digest")


def canonical_strategy_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable portion of a strategy in stable JSON form."""

    if not isinstance(spec, Mapping):
        raise ValueError("strategy spec must be a JSON object")
    normalised = _normalise_behavior(spec)
    if not isinstance(normalised, dict) or not normalised:
        raise ValueError("strategy spec has no behavioral fields after canonicalization")
    # Round-trip once so callers cannot mutate a nested object retained by us.
    return json.loads(_canonical_json(normalised))


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()}"


def canonical_strategy_hash(spec: Mapping[str, Any]) -> str:
    """Hash executable semantics while ignoring IDs, prose, and provenance."""

    return _sha256_json(canonical_strategy_spec(spec))


def _validate_hash(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256: hash")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256: hash") from exc
    return value


def _normalise_context(
    dataset: Mapping[str, Any],
    window: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be a JSON object")
    dataset_json = _validate_json(dict(dataset), label="dataset")
    snapshot_id = dataset_json.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("dataset.snapshot_id must identify immutable input data")
    if len(snapshot_id) > 512:
        raise ValueError("dataset.snapshot_id is too long")
    if not isinstance(window, Mapping) or not window:
        raise ValueError("window must be a non-empty JSON object")
    window_json = _validate_json(dict(window), label="window")
    if protocol is None:
        protocol = {}
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a JSON object")
    protocol_json = _validate_json(dict(protocol), label="protocol")
    phase = _validate_text(phase, label="phase", maximum=64)
    protocol_hash = _sha256_json({"phase": phase, "protocol": protocol_json, "window": window_json})
    return dataset_json, window_json, protocol_json, snapshot_id.strip(), protocol_hash


def canonical_test_hash(
    behavior_hash: str,
    *,
    dataset: Mapping[str, Any],
    window: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
    phase: str = "validation",
) -> str:
    """Hash behavior + immutable data snapshot + complete evaluation protocol."""

    behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
    _, _, _, snapshot_id, protocol_hash = _normalise_context(dataset, window, protocol, phase)
    return _sha256_json(
        {
            "behavior_hash": behavior_hash,
            "data_snapshot_id": snapshot_id,
            "protocol_hash": protocol_hash,
        }
    )


def _validate_text(value: str, *, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = value.strip()
    if len(value) > maximum:
        raise ValueError(f"{label} is too long (maximum {maximum})")
    return value


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit > MAX_QUERY_LIMIT:
        raise ValueError(f"limit cannot exceed {MAX_QUERY_LIMIT}")
    return limit


def _extract_primitives(spec: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    explicit = metadata.get("primitives")
    values: set[str] = set()
    if isinstance(explicit, Sequence) and not isinstance(explicit, str | bytes | bytearray):
        for item in explicit:
            if isinstance(item, str) and item.strip():
                values.add(item.strip()[:128])

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in _PRIMITIVE_VALUE_FIELDS and isinstance(child, str) and child.strip():
                    values.add(f"{key}:{child.strip()}"[:128])
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            for item in value:
                visit(item)

    visit(spec)
    return sorted(values)[:256]


def _novelty_tokens(value: Any, *, path: str = "$") -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            tokens.add(child_path)
            tokens.update(_novelty_tokens(child, path=child_path))
    elif isinstance(value, list):
        tokens.add(f"{path}[]")
        for child in value:
            tokens.update(_novelty_tokens(child, path=f"{path}[]"))
    else:
        tokens.add(f"{path}={_canonical_json(value)}")
    return tokens


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


class ExperimentMemory:
    """Transactional strategy identity, lineage, and experiment evidence store.

    One instance owns one SQLite connection.  It is safe for threads through an
    internal lock, while SQLite serializes independent processes.  Instantiate
    after forking; inherited connections fail closed.
    """

    def __init__(
        self,
        path: Path = DEFAULT_MEMORY_PATH,
        *,
        timeout_seconds: float = 15.0,
        deep_on_open: bool = True,
    ):
        self.path = Path(path)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(deep_on_open, bool):
            raise ValueError("deep_on_open must be boolean")
        self._timeout_seconds = float(timeout_seconds)
        self._deep_on_open = deep_on_open
        self._mutex = threading.RLock()
        self._closed = False
        self._owner_pid = os.getpid()
        self._connection: sqlite3.Connection | None = None
        self._open()

    def __enter__(self) -> ExperimentMemory:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._closed = True

    def _ensure_open(self) -> sqlite3.Connection:
        if self._closed or self._connection is None:
            raise ExperimentMemoryClosedError(f"experiment memory is closed: {self.path}")
        if os.getpid() != self._owner_pid:
            raise ExperimentMemoryClosedError(
                "experiment memory connection was inherited across a process fork; reopen it"
            )
        return self._connection

    def _open(self) -> None:
        if self.path.is_symlink():
            raise ExperimentMemoryCorruptionError(
                f"experiment memory must not be a symlink: {self.path}"
            )
        if self.path.exists() and not stat.S_ISREG(self.path.stat().st_mode):
            raise ExperimentMemoryCorruptionError(
                f"experiment memory must be a regular file: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self._timeout_seconds,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA journal_mode = DELETE")
            self._connection = connection
            self._initialise_schema()
            self.integrity_check(deep=self._deep_on_open)
            os.chmod(self.path, 0o600)
        except Exception as exc:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._closed = True
            if isinstance(exc, ExperimentMemoryError):
                raise
            if isinstance(exc, sqlite3.DatabaseError):
                raise ExperimentMemoryCorruptionError(
                    f"cannot open experiment memory {self.path}: {exc}"
                ) from exc
            raise

    @contextmanager
    def _database(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            connection = self._ensure_open()
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
                connection.execute("COMMIT")
            except BaseException as exc:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
                    raise ExperimentMemoryBusyError(
                        f"experiment memory remained locked: {self.path}"
                    ) from exc
                if isinstance(exc, sqlite3.DatabaseError):
                    raise ExperimentMemoryCorruptionError(
                        f"experiment memory database failure at {self.path}: {exc}"
                    ) from exc
                raise

    def _initialise_schema(self) -> None:
        connection = self._ensure_open()
        try:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if existing_tables and "memory_meta" not in existing_tables:
                raise ExperimentMemoryCorruptionError(
                    f"refusing to use unrelated SQLite database as experiment memory: {self.path}"
                )
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategies (
                    behavior_hash TEXT PRIMARY KEY,
                    canonical_spec_json TEXT NOT NULL,
                    primary_spec_json TEXT NOT NULL,
                    primary_strategy_id TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    primitive_tokens_json TEXT NOT NULL,
                    novelty_tokens_json TEXT NOT NULL,
                    novelty_score REAL NOT NULL,
                    nearest_behavior_hash TEXT,
                    product TEXT,
                    opportunity_type TEXT,
                    created_at TEXT NOT NULL,
                    holdout_exposed_at TEXT,
                    retired_at TEXT,
                    retirement_reason TEXT,
                    FOREIGN KEY(nearest_behavior_hash) REFERENCES strategies(behavior_hash)
                );
                CREATE TABLE IF NOT EXISTS strategy_identities (
                    strategy_id TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    submitted_spec_json TEXT NOT NULL,
                    generation_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    parent_hashes_json TEXT NOT NULL,
                    is_duplicate INTEGER NOT NULL CHECK(is_duplicate IN (0, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    child_hash TEXT NOT NULL,
                    parent_hash TEXT NOT NULL,
                    parent_ordinal INTEGER NOT NULL,
                    generation_method TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(child_hash, parent_hash),
                    FOREIGN KEY(child_hash) REFERENCES strategies(behavior_hash),
                    FOREIGN KEY(parent_hash) REFERENCES strategies(behavior_hash),
                    CHECK(child_hash <> parent_hash)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_key TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    dataset_json TEXT NOT NULL,
                    window_json TEXT NOT NULL,
                    protocol_json TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('claimed', 'completed')),
                    claimed_at TEXT NOT NULL,
                    completed_at TEXT,
                    outcome TEXT,
                    rejection_reasons_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
                CREATE TABLE IF NOT EXISTS holdout_claim_scopes (
                    lineage_root_hash TEXT NOT NULL,
                    data_snapshot_id TEXT NOT NULL,
                    evaluation_key TEXT NOT NULL,
                    behavior_hash TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    PRIMARY KEY(lineage_root_hash, data_snapshot_id),
                    FOREIGN KEY(lineage_root_hash) REFERENCES strategies(behavior_hash),
                    FOREIGN KEY(evaluation_key) REFERENCES evaluations(evaluation_key),
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
                CREATE TABLE IF NOT EXISTS evaluation_engine_scopes (
                    evaluation_key TEXT PRIMARY KEY,
                    behavior_hash TEXT NOT NULL,
                    research_engine_digest TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    FOREIGN KEY(evaluation_key) REFERENCES evaluations(evaluation_key)
                        ON DELETE CASCADE,
                    FOREIGN KEY(behavior_hash) REFERENCES strategies(behavior_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_evaluations_strategy
                    ON evaluations(behavior_hash, claimed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_evaluations_snapshot
                    ON evaluations(data_snapshot_id, protocol_hash);
                CREATE INDEX IF NOT EXISTS idx_lineage_parent
                    ON lineage_edges(parent_hash, child_hash);
                CREATE INDEX IF NOT EXISTS idx_strategies_candidate
                    ON strategies(product, opportunity_type, retired_at, holdout_exposed_at);
                CREATE INDEX IF NOT EXISTS idx_engine_scopes_candidate
                    ON evaluation_engine_scopes(
                        behavior_hash, research_engine_digest, phase, claimed_at
                    );
                INSERT OR IGNORE INTO memory_meta(key, value)
                    VALUES ('schema_version', '1');
                INSERT OR IGNORE INTO memory_meta(key, value)
                    VALUES ('format', 'autonomous_strategy_experiment_memory');
                COMMIT;
                """
            )
            row = connection.execute(
                "SELECT value FROM memory_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None or row["value"] != str(SCHEMA_VERSION):
                raise ExperimentMemoryCorruptionError(
                    f"unsupported experiment memory schema version: {None if row is None else row['value']}"
                )
            format_row = connection.execute(
                "SELECT value FROM memory_meta WHERE key = 'format'"
            ).fetchone()
            if format_row is None or format_row["value"] != MEMORY_FORMAT:
                raise ExperimentMemoryCorruptionError(
                    "experiment memory format marker is missing or invalid"
                )
            self._backfill_engine_scopes(connection)
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _backfill_engine_scopes(connection: sqlite3.Connection) -> None:
        """Create the indexed engine view for databases written before it existed."""

        marker = connection.execute(
            "SELECT value FROM memory_meta WHERE key = ?",
            (ENGINE_SCOPE_BACKFILL_META,),
        ).fetchone()
        if marker is not None and marker["value"] == "complete":
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            for row in connection.execute(
                """
                SELECT evaluation_key, behavior_hash, protocol_json, phase, claimed_at
                FROM evaluations
                WHERE NOT EXISTS (
                    SELECT 1 FROM evaluation_engine_scopes scope
                    WHERE scope.evaluation_key = evaluations.evaluation_key
                )
                """
            ):
                protocol = _stored_json(row["protocol_json"], label="evaluation protocol JSON")
                if not isinstance(protocol, Mapping):
                    raise ExperimentMemoryCorruptionError(
                        f"evaluation protocol is not an object: {row['evaluation_key']}"
                    )
                digest = _research_engine_digest(protocol)
                if digest is None:
                    continue
                connection.execute(
                    """
                    INSERT INTO evaluation_engine_scopes(
                        evaluation_key, behavior_hash, research_engine_digest, phase, claimed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["evaluation_key"],
                        row["behavior_hash"],
                        digest,
                        row["phase"],
                        row["claimed_at"],
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO memory_meta(key, value) VALUES (?, 'complete')",
                (ENGINE_SCOPE_BACKFILL_META,),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def integrity_check(self, *, deep: bool = True) -> dict[str, Any]:
        """Validate physical SQLite state and, optionally, all logical hashes."""

        with self._database(write=False) as connection:
            quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
            if quick != ["ok"]:
                raise ExperimentMemoryCorruptionError(
                    f"SQLite quick_check failed for {self.path}: {quick[:3]}"
                )
            foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
            if foreign_keys:
                raise ExperimentMemoryCorruptionError(
                    f"foreign-key integrity failed for {self.path}: {foreign_keys[:3]}"
                )
            if deep:
                self._deep_integrity_check(connection)
        return {"ok": True, "path": str(self.path), "deep": deep, "schema_version": SCHEMA_VERSION}

    def _deep_integrity_check(self, connection: sqlite3.Connection) -> None:
        strategy_hashes: set[str] = set()
        for row in connection.execute(
            "SELECT behavior_hash, canonical_spec_json, primary_spec_json FROM strategies"
        ):
            canonical = _stored_json(row["canonical_spec_json"], label="strategy canonical JSON")
            primary = _stored_json(row["primary_spec_json"], label="strategy primary JSON")
            expected = _sha256_json(canonical_strategy_spec(primary))
            if expected != row["behavior_hash"] or canonical_strategy_spec(primary) != canonical:
                raise ExperimentMemoryCorruptionError(
                    f"strategy behavioral hash mismatch: {row['behavior_hash']}"
                )
            strategy_hashes.add(row["behavior_hash"])

        for row in connection.execute(
            "SELECT strategy_id, behavior_hash, submitted_spec_json, parent_hashes_json FROM strategy_identities"
        ):
            submitted = _stored_json(row["submitted_spec_json"], label="identity submitted JSON")
            parents = _stored_json(row["parent_hashes_json"], label="identity parent JSON")
            if canonical_strategy_hash(submitted) != row["behavior_hash"]:
                raise ExperimentMemoryCorruptionError(
                    f"identity behavioral hash mismatch: {row['strategy_id']}"
                )
            if not isinstance(parents, list) or any(
                parent not in strategy_hashes for parent in parents
            ):
                raise ExperimentMemoryCorruptionError(
                    f"identity has invalid parents: {row['strategy_id']}"
                )

        edges: dict[str, set[str]] = defaultdict(set)
        for row in connection.execute("SELECT child_hash, parent_hash FROM lineage_edges"):
            edges[row["child_hash"]].add(row["parent_hash"])
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ExperimentMemoryCorruptionError("strategy lineage contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for parent in edges.get(node, set()):
                visit(parent)
            visiting.remove(node)
            visited.add(node)

        for behavior_hash in strategy_hashes:
            visit(behavior_hash)

        protected_hashes: set[str] = set()
        expected_engine_scopes: dict[str, tuple[str, str, str, str]] = {}
        for row in connection.execute("SELECT * FROM evaluations"):
            dataset = _stored_json(row["dataset_json"], label="evaluation dataset JSON")
            window = _stored_json(row["window_json"], label="evaluation window JSON")
            protocol = _stored_json(row["protocol_json"], label="evaluation protocol JSON")
            reasons = _stored_json(
                row["rejection_reasons_json"], label="evaluation rejection-reason JSON"
            )
            _stored_json(row["metrics_json"], label="evaluation metrics JSON")
            _stored_json(row["details_json"], label="evaluation details JSON")
            expected_key = canonical_test_hash(
                row["behavior_hash"],
                dataset=dataset,
                window=window,
                protocol=protocol,
                phase=row["phase"],
            )
            if expected_key != row["evaluation_key"]:
                raise ExperimentMemoryCorruptionError(
                    f"evaluation key mismatch: {row['evaluation_key']}"
                )
            _, _, _, snapshot_id, protocol_hash = _normalise_context(
                dataset, window, protocol, row["phase"]
            )
            if snapshot_id != row["data_snapshot_id"] or protocol_hash != row["protocol_hash"]:
                raise ExperimentMemoryCorruptionError(
                    f"evaluation context mismatch: {row['evaluation_key']}"
                )
            if row["status"] == "completed":
                if not row["completed_at"] or not row["outcome"]:
                    raise ExperimentMemoryCorruptionError(
                        f"completed evaluation is incomplete: {row['evaluation_key']}"
                    )
            elif row["completed_at"] or row["outcome"] or reasons:
                raise ExperimentMemoryCorruptionError(
                    f"claimed evaluation contains premature outcome: {row['evaluation_key']}"
                )
            if row["phase"] in PROTECTED_PHASES:
                protected_hashes.add(row["behavior_hash"])
            digest = _research_engine_digest(protocol)
            if digest is not None:
                expected_engine_scopes[row["evaluation_key"]] = (
                    row["behavior_hash"],
                    digest,
                    row["phase"],
                    row["claimed_at"],
                )

        actual_engine_scopes = {
            row["evaluation_key"]: (
                row["behavior_hash"],
                row["research_engine_digest"],
                row["phase"],
                row["claimed_at"],
            )
            for row in connection.execute("SELECT * FROM evaluation_engine_scopes")
        }
        if actual_engine_scopes != expected_engine_scopes:
            raise ExperimentMemoryCorruptionError(
                "evaluation engine-scope index does not match immutable protocols"
            )

        exposed = {
            row["behavior_hash"]
            for row in connection.execute(
                "SELECT behavior_hash FROM strategies WHERE holdout_exposed_at IS NOT NULL"
            )
        }
        if not protected_hashes.issubset(exposed):
            raise ExperimentMemoryCorruptionError(
                "protected evaluation exists without durable holdout exposure"
            )
        for row in connection.execute(
            """
            SELECT evaluation_key, behavior_hash, data_snapshot_id
            FROM evaluations WHERE phase IN ('holdout', 'final_holdout', 'final')
            """
        ):
            roots = self._lineage_roots(connection, row["behavior_hash"])
            claims = {
                claim["lineage_root_hash"]
                for claim in connection.execute(
                    """
                    SELECT lineage_root_hash FROM holdout_claim_scopes
                    WHERE evaluation_key = ? AND data_snapshot_id = ?
                    """,
                    (row["evaluation_key"], row["data_snapshot_id"]),
                )
            }
            if not roots.issubset(claims):
                raise ExperimentMemoryCorruptionError(
                    f"protected evaluation lacks lineage holdout claim: {row['evaluation_key']}"
                )

    def register_strategy(
        self,
        spec: Mapping[str, Any],
        *,
        strategy_id: str,
        generation_method: str,
        parent_hashes: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyRegistration:
        """Persist a generated identity and return exact-dedup/novelty evidence."""

        strategy_id = _validate_text(strategy_id, label="strategy_id")
        generation_method = _validate_text(
            generation_method, label="generation_method", maximum=128
        )
        submitted_spec = _validate_json(dict(spec), label="strategy spec")
        canonical_spec = canonical_strategy_spec(submitted_spec)
        behavior_hash = _sha256_json(canonical_spec)
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        metadata_json = _validate_json(dict(metadata), label="metadata")
        for key in ("family", "product", "opportunity_type"):
            if key not in metadata_json and isinstance(submitted_spec.get(key), str):
                metadata_json[key] = submitted_spec[key]
        primitives = _extract_primitives(submitted_spec, metadata_json)
        parents = list(dict.fromkeys(parent_hashes))
        if len(parents) > MAX_PARENTS:
            raise ValueError(f"a strategy cannot have more than {MAX_PARENTS} parents")
        for parent in parents:
            _validate_hash(parent, label="parent_hash")
        now = _utc_now()

        with self._database(write=True) as connection:
            existing_identity = connection.execute(
                "SELECT behavior_hash FROM strategy_identities WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
            if existing_identity is not None:
                if existing_identity["behavior_hash"] != behavior_hash:
                    raise StrategyIdentityConflictError(
                        f"strategy_id {strategy_id!r} already identifies different behavior"
                    )
                row = self._strategy_row(connection, behavior_hash)
                return StrategyRegistration(
                    strategy_id=strategy_id,
                    behavior_hash=behavior_hash,
                    created=False,
                    identity_created=False,
                    duplicate=bool(
                        connection.execute(
                            "SELECT is_duplicate FROM strategy_identities WHERE strategy_id = ?",
                            (strategy_id,),
                        ).fetchone()[0]
                    ),
                    novelty_score=float(row["novelty_score"]),
                    nearest_behavior_hash=row["nearest_behavior_hash"],
                )

            missing = [
                parent
                for parent in parents
                if connection.execute(
                    "SELECT 1 FROM strategies WHERE behavior_hash = ?", (parent,)
                ).fetchone()
                is None
            ]
            if missing:
                raise ValueError(f"unknown parent strategy hash(es): {', '.join(missing)}")

            existing = connection.execute(
                "SELECT * FROM strategies WHERE behavior_hash = ?", (behavior_hash,)
            ).fetchone()
            created = existing is None
            if created:
                novelty_tokens = _novelty_tokens(canonical_spec)
                novelty_score, nearest_hash = self._nearest_strategy(connection, novelty_tokens)
                product = metadata_json.get("product")
                opportunity_type = metadata_json.get("opportunity_type")
                connection.execute(
                    """
                    INSERT INTO strategies(
                        behavior_hash, canonical_spec_json, primary_spec_json,
                        primary_strategy_id, generation_method, metadata_json,
                        primitive_tokens_json, novelty_tokens_json, novelty_score,
                        nearest_behavior_hash, product, opportunity_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        behavior_hash,
                        _canonical_json(canonical_spec),
                        _canonical_json(submitted_spec),
                        strategy_id,
                        generation_method,
                        _canonical_json(metadata_json),
                        _canonical_json(primitives),
                        _canonical_json(sorted(novelty_tokens)),
                        novelty_score,
                        nearest_hash,
                        product if isinstance(product, str) else None,
                        opportunity_type if isinstance(opportunity_type, str) else None,
                        now,
                    ),
                )
                for ordinal, parent in enumerate(parents):
                    if parent == behavior_hash:
                        continue
                    connection.execute(
                        """
                        INSERT INTO lineage_edges(
                            child_hash, parent_hash, parent_ordinal, generation_method, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (behavior_hash, parent, ordinal, generation_method, now),
                    )
            else:
                if (
                    _stored_json(
                        existing["canonical_spec_json"], label="strategy canonical JSON"
                    )
                    != canonical_spec
                ):
                    raise ExperimentMemoryCorruptionError(
                        f"behavioral hash collision detected: {behavior_hash}"
                    )
                novelty_score = float(existing["novelty_score"])
                nearest_hash = existing["nearest_behavior_hash"]

            connection.execute(
                """
                INSERT INTO strategy_identities(
                    strategy_id, behavior_hash, submitted_spec_json, generation_method,
                    metadata_json, parent_hashes_json, is_duplicate, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    behavior_hash,
                    _canonical_json(submitted_spec),
                    generation_method,
                    _canonical_json(metadata_json),
                    _canonical_json(parents),
                    0 if created else 1,
                    now,
                ),
            )
            return StrategyRegistration(
                strategy_id=strategy_id,
                behavior_hash=behavior_hash,
                created=created,
                identity_created=True,
                duplicate=not created,
                novelty_score=novelty_score,
                nearest_behavior_hash=nearest_hash,
            )

    def _nearest_strategy(
        self, connection: sqlite3.Connection, tokens: set[str]
    ) -> tuple[float, str | None]:
        nearest_score = 1.0
        nearest_hash: str | None = None
        found = False
        for row in connection.execute("SELECT behavior_hash, novelty_tokens_json FROM strategies"):
            prior_tokens = set(
                _stored_json(row["novelty_tokens_json"], label="strategy novelty-token JSON")
            )
            distance = _jaccard_distance(tokens, prior_tokens)
            if (
                not found
                or distance < nearest_score
                or (distance == nearest_score and row["behavior_hash"] < (nearest_hash or ""))
            ):
                found = True
                nearest_score = distance
                nearest_hash = row["behavior_hash"]
        return (nearest_score if found else 1.0), nearest_hash

    @staticmethod
    def _strategy_row(connection: sqlite3.Connection, behavior_hash: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM strategies WHERE behavior_hash = ?", (behavior_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown strategy behavior hash: {behavior_hash}")
        return row

    def find_duplicate(self, spec: Mapping[str, Any]) -> str | None:
        behavior_hash = canonical_strategy_hash(spec)
        with self._database(write=False) as connection:
            row = connection.execute(
                "SELECT behavior_hash FROM strategies WHERE behavior_hash = ?", (behavior_hash,)
            ).fetchone()
        return behavior_hash if row is not None else None

    def get_strategy(self, behavior_hash: str) -> dict[str, Any]:
        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        with self._database(write=False) as connection:
            row = self._strategy_row(connection, behavior_hash)
            return self._strategy_payload(connection, row)

    def _strategy_payload(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        research_engine_digest: str | None = None,
    ) -> dict[str, Any]:
        parents = [
            edge["parent_hash"]
            for edge in connection.execute(
                """
                SELECT parent_hash FROM lineage_edges
                WHERE child_hash = ? ORDER BY parent_ordinal, parent_hash
                """,
                (row["behavior_hash"],),
            )
        ]
        if research_engine_digest is None:
            latest = connection.execute(
                """
                SELECT status, outcome, phase, claimed_at, completed_at,
                       rejection_reasons_json
                FROM evaluations WHERE behavior_hash = ?
                ORDER BY COALESCE(completed_at, claimed_at) DESC, evaluation_key DESC LIMIT 1
                """,
                (row["behavior_hash"],),
            ).fetchone()
        else:
            latest = connection.execute(
                """
                SELECT evaluation.status, evaluation.outcome, evaluation.phase,
                       evaluation.claimed_at, evaluation.completed_at,
                       evaluation.rejection_reasons_json
                FROM evaluations evaluation
                JOIN evaluation_engine_scopes scope
                  ON scope.evaluation_key = evaluation.evaluation_key
                WHERE evaluation.behavior_hash = ?
                  AND scope.research_engine_digest = ?
                ORDER BY COALESCE(evaluation.completed_at, evaluation.claimed_at) DESC,
                         evaluation.evaluation_key DESC
                LIMIT 1
                """,
                (row["behavior_hash"], research_engine_digest),
            ).fetchone()
        return {
            "behavior_hash": row["behavior_hash"],
            "strategy_id": row["primary_strategy_id"],
            "spec": _stored_json(row["canonical_spec_json"], label="strategy canonical JSON"),
            "submitted_spec": _stored_json(
                row["primary_spec_json"], label="strategy primary JSON"
            ),
            "generation_method": row["generation_method"],
            "parent_hashes": parents,
            "metadata": _stored_json(row["metadata_json"], label="strategy metadata JSON"),
            "primitives": _stored_json(
                row["primitive_tokens_json"], label="strategy primitive-token JSON"
            ),
            "novelty_score": float(row["novelty_score"]),
            "nearest_behavior_hash": row["nearest_behavior_hash"],
            "created_at": row["created_at"],
            "holdout_exposed_at": row["holdout_exposed_at"],
            "retired_at": row["retired_at"],
            "retirement_reason": row["retirement_reason"],
            "latest_evaluation": None
            if latest is None
            else {
                "status": latest["status"],
                "outcome": latest["outcome"],
                "phase": latest["phase"],
                "claimed_at": latest["claimed_at"],
                "completed_at": latest["completed_at"],
                "rejection_reasons": _stored_json(
                    latest["rejection_reasons_json"],
                    label="evaluation rejection-reason JSON",
                ),
            },
        }

    @staticmethod
    def _insert_engine_scope(
        connection: sqlite3.Connection,
        *,
        evaluation_key: str,
        behavior_hash: str,
        protocol_json: Mapping[str, Any],
        phase: str,
        claimed_at: str,
    ) -> None:
        digest = _research_engine_digest(protocol_json)
        if digest is None:
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO evaluation_engine_scopes(
                evaluation_key, behavior_hash, research_engine_digest, phase, claimed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (evaluation_key, behavior_hash, digest, phase, claimed_at),
        )

    def claim_evaluation(
        self,
        behavior_hash: str,
        *,
        dataset: Mapping[str, Any],
        window: Mapping[str, Any],
        protocol: Mapping[str, Any] | None = None,
        phase: str = "validation",
    ) -> EvaluationClaim:
        """Durably consume an evaluation context before data is inspected.

        A crash after this commit leaves the context ``claimed`` and therefore
        tested.  In particular, protected holdouts can never silently be reused.
        """

        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        dataset_json, window_json, protocol_json, snapshot_id, protocol_hash = _normalise_context(
            dataset, window, protocol, phase
        )
        phase = phase.strip()
        evaluation_key = canonical_test_hash(
            behavior_hash,
            dataset=dataset_json,
            window=window_json,
            protocol=protocol_json,
            phase=phase,
        )
        now = _utc_now()
        with self._database(write=True) as connection:
            self._strategy_row(connection, behavior_hash)
            existing = connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
            if existing is not None:
                self._assert_evaluation_context(
                    existing,
                    behavior_hash=behavior_hash,
                    dataset_json=dataset_json,
                    window_json=window_json,
                    protocol_json=protocol_json,
                    phase=phase,
                )
                if phase in PROTECTED_PHASES:
                    self._claim_holdout_scopes(
                        connection,
                        behavior_hash=behavior_hash,
                        data_snapshot_id=snapshot_id,
                        evaluation_key=evaluation_key,
                        claimed_at=existing["claimed_at"],
                    )
                return EvaluationClaim(
                    evaluation_key=evaluation_key,
                    behavior_hash=behavior_hash,
                    created=False,
                    status=existing["status"],
                    holdout_consumed=phase in PROTECTED_PHASES,
                )
            connection.execute(
                """
                INSERT INTO evaluations(
                    evaluation_key, behavior_hash, data_snapshot_id, dataset_json,
                    window_json, protocol_json, protocol_hash, phase, status,
                    claimed_at, rejection_reasons_json, metrics_json, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'claimed', ?, '[]', '{}', '{}')
                """,
                (
                    evaluation_key,
                    behavior_hash,
                    snapshot_id,
                    _canonical_json(dataset_json),
                    _canonical_json(window_json),
                    _canonical_json(protocol_json),
                    protocol_hash,
                    phase,
                    now,
                ),
            )
            self._insert_engine_scope(
                connection,
                evaluation_key=evaluation_key,
                behavior_hash=behavior_hash,
                protocol_json=protocol_json,
                phase=phase,
                claimed_at=now,
            )
            if phase in PROTECTED_PHASES:
                self._claim_holdout_scopes(
                    connection,
                    behavior_hash=behavior_hash,
                    data_snapshot_id=snapshot_id,
                    evaluation_key=evaluation_key,
                    claimed_at=now,
                )
                connection.execute(
                    """
                    UPDATE strategies SET holdout_exposed_at = COALESCE(holdout_exposed_at, ?)
                    WHERE behavior_hash = ?
                    """,
                    (now, behavior_hash),
                )
            return EvaluationClaim(
                evaluation_key=evaluation_key,
                behavior_hash=behavior_hash,
                created=True,
                status="claimed",
                holdout_consumed=phase in PROTECTED_PHASES,
            )

    def claim_holdout(
        self,
        behavior_hash: str,
        *,
        snapshot_id: str,
        window: Mapping[str, Any],
        protocol: Mapping[str, Any] | None = None,
        dataset: Mapping[str, Any] | None = None,
        phase: str = "holdout",
    ) -> EvaluationClaim:
        """Convenience wrapper that durably claims protected data for a lineage."""

        if phase not in PROTECTED_PHASES:
            raise ValueError(f"holdout phase must be one of {sorted(PROTECTED_PHASES)}")
        dataset_payload = dict(dataset or {})
        existing_snapshot = dataset_payload.get("snapshot_id")
        if existing_snapshot is not None and existing_snapshot != snapshot_id:
            raise ValueError("dataset.snapshot_id conflicts with snapshot_id")
        dataset_payload["snapshot_id"] = snapshot_id
        return self.claim_evaluation(
            behavior_hash,
            dataset=dataset_payload,
            window=window,
            protocol=protocol,
            phase=phase,
        )

    def holdout_claimed(self, behavior_hash: str, *, snapshot_id: str) -> bool:
        """Return whether this strategy's lineage has consumed a data snapshot."""

        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        snapshot_id = _validate_text(snapshot_id, label="snapshot_id", maximum=512)
        with self._database(write=False) as connection:
            self._strategy_row(connection, behavior_hash)
            roots = self._lineage_roots(connection, behavior_hash)
            placeholders = ",".join("?" for _ in roots)
            row = connection.execute(
                f"""
                SELECT 1 FROM holdout_claim_scopes
                WHERE data_snapshot_id = ? AND lineage_root_hash IN ({placeholders})
                LIMIT 1
                """,
                (snapshot_id, *sorted(roots)),
            ).fetchone()
        return row is not None

    def _claim_holdout_scopes(
        self,
        connection: sqlite3.Connection,
        *,
        behavior_hash: str,
        data_snapshot_id: str,
        evaluation_key: str,
        claimed_at: str,
    ) -> None:
        for root_hash in sorted(self._lineage_roots(connection, behavior_hash)):
            existing = connection.execute(
                """
                SELECT evaluation_key, behavior_hash FROM holdout_claim_scopes
                WHERE lineage_root_hash = ? AND data_snapshot_id = ?
                """,
                (root_hash, data_snapshot_id),
            ).fetchone()
            if existing is not None:
                if existing["evaluation_key"] != evaluation_key:
                    raise EvaluationConflictError(
                        "holdout snapshot was already consumed by this strategy lineage: "
                        f"root={root_hash} snapshot={data_snapshot_id} "
                        f"behavior={existing['behavior_hash']}"
                    )
                continue
            connection.execute(
                """
                INSERT INTO holdout_claim_scopes(
                    lineage_root_hash, data_snapshot_id, evaluation_key,
                    behavior_hash, claimed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (root_hash, data_snapshot_id, evaluation_key, behavior_hash, claimed_at),
            )

    @staticmethod
    def _lineage_roots(connection: sqlite3.Connection, behavior_hash: str) -> set[str]:
        return {
            row["behavior_hash"]
            for row in connection.execute(
                """
                WITH RECURSIVE ancestry(behavior_hash) AS (
                    SELECT ?
                    UNION
                    SELECT edge.parent_hash FROM lineage_edges edge
                    JOIN ancestry current ON edge.child_hash = current.behavior_hash
                )
                SELECT node.behavior_hash FROM ancestry node
                WHERE NOT EXISTS (
                    SELECT 1 FROM lineage_edges edge WHERE edge.child_hash = node.behavior_hash
                )
                """,
                (behavior_hash,),
            )
        }

    @staticmethod
    def _assert_evaluation_context(
        row: sqlite3.Row,
        *,
        behavior_hash: str,
        dataset_json: Mapping[str, Any],
        window_json: Mapping[str, Any],
        protocol_json: Mapping[str, Any],
        phase: str,
    ) -> None:
        expected = (
            behavior_hash,
            dict(dataset_json),
            dict(window_json),
            dict(protocol_json),
            phase,
        )
        actual = (
            row["behavior_hash"],
            _stored_json(row["dataset_json"], label="evaluation dataset JSON"),
            _stored_json(row["window_json"], label="evaluation window JSON"),
            _stored_json(row["protocol_json"], label="evaluation protocol JSON"),
            row["phase"],
        )
        if actual != expected:
            raise ExperimentMemoryCorruptionError(
                f"evaluation-key collision or context corruption: {row['evaluation_key']}"
            )

    def complete_evaluation(
        self,
        evaluation_key: str,
        *,
        outcome: str,
        rejection_reasons: Sequence[str] = (),
        metrics: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Complete a previously claimed evaluation without overwriting history."""

        evaluation_key = _validate_hash(evaluation_key, label="evaluation_key")
        outcome, reasons, metrics_json, details_json = self._normalise_outcome(
            outcome, rejection_reasons, metrics, details
        )
        with self._database(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
            if row is None:
                raise KeyError(f"evaluation was not claimed: {evaluation_key}")
            if row["status"] == "completed":
                expected = (
                    outcome,
                    reasons,
                    metrics_json,
                    details_json,
                )
                actual = (
                    row["outcome"],
                    _stored_json(
                        row["rejection_reasons_json"],
                        label="evaluation rejection-reason JSON",
                    ),
                    _stored_json(row["metrics_json"], label="evaluation metrics JSON"),
                    _stored_json(row["details_json"], label="evaluation details JSON"),
                )
                if actual != expected:
                    raise EvaluationConflictError(
                        f"evaluation already has a different outcome: {evaluation_key}"
                    )
                return EvaluationResult(
                    evaluation_key=evaluation_key,
                    behavior_hash=row["behavior_hash"],
                    created=False,
                    completed=True,
                    was_claimed=True,
                )
            connection.execute(
                """
                UPDATE evaluations
                SET status = 'completed', completed_at = ?, outcome = ?,
                    rejection_reasons_json = ?, metrics_json = ?, details_json = ?
                WHERE evaluation_key = ?
                """,
                (
                    _utc_now(),
                    outcome,
                    _canonical_json(reasons),
                    _canonical_json(metrics_json),
                    _canonical_json(details_json),
                    evaluation_key,
                ),
            )
            return EvaluationResult(
                evaluation_key=evaluation_key,
                behavior_hash=row["behavior_hash"],
                created=True,
                completed=True,
                was_claimed=True,
            )

    @staticmethod
    def _normalise_outcome(
        outcome: str,
        rejection_reasons: Sequence[str],
        metrics: Mapping[str, Any] | None,
        details: Mapping[str, Any] | None,
    ) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
        outcome = _validate_text(outcome, label="outcome", maximum=64)
        if isinstance(rejection_reasons, str | bytes | bytearray):
            raise ValueError("rejection_reasons must be a sequence of strings")
        reasons = list(
            dict.fromkeys(
                _validate_text(reason, label="rejection reason") for reason in rejection_reasons
            )
        )
        if outcome == "reject" and not reasons:
            raise ValueError("a rejected evaluation must record at least one rejection reason")
        if metrics is None:
            metrics = {}
        if details is None:
            details = {}
        if not isinstance(metrics, Mapping) or not isinstance(details, Mapping):
            raise ValueError("metrics and details must be JSON objects")
        metrics_json = _validate_json(dict(metrics), label="metrics")
        details_json = _validate_json(dict(details), label="details")
        return outcome, reasons, metrics_json, details_json

    def record_outcome(
        self,
        behavior_hash: str,
        *,
        dataset: Mapping[str, Any],
        window: Mapping[str, Any],
        outcome: str,
        rejection_reasons: Sequence[str] = (),
        metrics: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        protocol: Mapping[str, Any] | None = None,
        phase: str = "validation",
    ) -> EvaluationResult:
        """Atomically record a non-protected evaluation.

        Protected holdouts intentionally require ``claim_evaluation`` followed
        by ``complete_evaluation`` so exposure is committed before data access.
        """

        if phase.strip() in PROTECTED_PHASES:
            raise ValueError(
                "protected evaluations must be claimed before reading data; "
                "use claim_evaluation then complete_evaluation"
            )
        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        dataset_json, window_json, protocol_json, snapshot_id, protocol_hash = _normalise_context(
            dataset, window, protocol, phase
        )
        phase = phase.strip()
        outcome, reasons, metrics_json, details_json = self._normalise_outcome(
            outcome, rejection_reasons, metrics, details
        )
        evaluation_key = canonical_test_hash(
            behavior_hash,
            dataset=dataset_json,
            window=window_json,
            protocol=protocol_json,
            phase=phase,
        )
        now = _utc_now()
        with self._database(write=True) as connection:
            self._strategy_row(connection, behavior_hash)
            existing = connection.execute(
                "SELECT * FROM evaluations WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
            if existing is not None:
                self._assert_evaluation_context(
                    existing,
                    behavior_hash=behavior_hash,
                    dataset_json=dataset_json,
                    window_json=window_json,
                    protocol_json=protocol_json,
                    phase=phase,
                )
                if existing["status"] == "claimed":
                    # This can happen when a caller explicitly claimed a
                    # validation context. Complete that durable claim.
                    connection.execute(
                        """
                        UPDATE evaluations SET status = 'completed', completed_at = ?, outcome = ?,
                            rejection_reasons_json = ?, metrics_json = ?, details_json = ?
                        WHERE evaluation_key = ?
                        """,
                        (
                            now,
                            outcome,
                            _canonical_json(reasons),
                            _canonical_json(metrics_json),
                            _canonical_json(details_json),
                            evaluation_key,
                        ),
                    )
                    return EvaluationResult(
                        evaluation_key=evaluation_key,
                        behavior_hash=behavior_hash,
                        created=True,
                        completed=True,
                        was_claimed=True,
                    )
                expected = (
                    outcome,
                    reasons,
                    metrics_json,
                    details_json,
                )
                actual = (
                    existing["outcome"],
                    _stored_json(
                        existing["rejection_reasons_json"],
                        label="evaluation rejection-reason JSON",
                    ),
                    _stored_json(existing["metrics_json"], label="evaluation metrics JSON"),
                    _stored_json(existing["details_json"], label="evaluation details JSON"),
                )
                if actual != expected:
                    raise EvaluationConflictError(
                        f"evaluation already has different evidence: {evaluation_key}"
                    )
                return EvaluationResult(
                    evaluation_key=evaluation_key,
                    behavior_hash=behavior_hash,
                    created=False,
                    completed=True,
                    was_claimed=False,
                )
            connection.execute(
                """
                INSERT INTO evaluations(
                    evaluation_key, behavior_hash, data_snapshot_id, dataset_json,
                    window_json, protocol_json, protocol_hash, phase, status,
                    claimed_at, completed_at, outcome, rejection_reasons_json,
                    metrics_json, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_key,
                    behavior_hash,
                    snapshot_id,
                    _canonical_json(dataset_json),
                    _canonical_json(window_json),
                    _canonical_json(protocol_json),
                    protocol_hash,
                    phase,
                    now,
                    now,
                    outcome,
                    _canonical_json(reasons),
                    _canonical_json(metrics_json),
                    _canonical_json(details_json),
                ),
            )
            self._insert_engine_scope(
                connection,
                evaluation_key=evaluation_key,
                behavior_hash=behavior_hash,
                protocol_json=protocol_json,
                phase=phase,
                claimed_at=now,
            )
            return EvaluationResult(
                evaluation_key=evaluation_key,
                behavior_hash=behavior_hash,
                created=True,
                completed=True,
                was_claimed=False,
            )

    def is_tested(
        self,
        behavior_hash: str,
        *,
        dataset: Mapping[str, Any],
        window: Mapping[str, Any],
        protocol: Mapping[str, Any] | None = None,
        phase: str = "validation",
    ) -> bool:
        evaluation_key = canonical_test_hash(
            behavior_hash,
            dataset=dataset,
            window=window,
            protocol=protocol,
            phase=phase,
        )
        with self._database(write=False) as connection:
            row = connection.execute(
                "SELECT 1 FROM evaluations WHERE evaluation_key = ?", (evaluation_key,)
            ).fetchone()
        return row is not None

    def list_evaluations(
        self,
        *,
        behavior_hash: str | None = None,
        status: str | None = None,
        phase: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return a bounded evidence history for reporting and recovery."""

        limit = _validate_limit(limit)
        where: list[str] = []
        params: list[Any] = []
        if behavior_hash is not None:
            behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
            where.append("behavior_hash = ?")
            params.append(behavior_hash)
        if status is not None:
            if status not in {"claimed", "completed"}:
                raise ValueError("status must be 'claimed' or 'completed'")
            where.append("status = ?")
            params.append(status)
        if phase is not None:
            phase = _validate_text(phase, label="phase", maximum=64)
            where.append("phase = ?")
            params.append(phase)
        query = "SELECT * FROM evaluations"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY claimed_at DESC, evaluation_key DESC LIMIT ?"
        params.append(limit)
        with self._database(write=False) as connection:
            rows = list(connection.execute(query, params))
        return [
            {
                "evaluation_key": row["evaluation_key"],
                "behavior_hash": row["behavior_hash"],
                "data_snapshot_id": row["data_snapshot_id"],
                "dataset": _stored_json(row["dataset_json"], label="evaluation dataset JSON"),
                "window": _stored_json(row["window_json"], label="evaluation window JSON"),
                "protocol": _stored_json(row["protocol_json"], label="evaluation protocol JSON"),
                "protocol_hash": row["protocol_hash"],
                "phase": row["phase"],
                "status": row["status"],
                "claimed_at": row["claimed_at"],
                "completed_at": row["completed_at"],
                "outcome": row["outcome"],
                "rejection_reasons": _stored_json(
                    row["rejection_reasons_json"], label="evaluation rejection-reason JSON"
                ),
                "metrics": _stored_json(row["metrics_json"], label="evaluation metrics JSON"),
                "details": _stored_json(row["details_json"], label="evaluation details JSON"),
            }
            for row in rows
        ]

    def retire_strategy(self, behavior_hash: str, *, reason: str) -> None:
        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        reason = _validate_text(reason, label="retirement reason", maximum=512)
        with self._database(write=True) as connection:
            self._strategy_row(connection, behavior_hash)
            connection.execute(
                """
                UPDATE strategies SET retired_at = COALESCE(retired_at, ?),
                    retirement_reason = COALESCE(retirement_reason, ?)
                WHERE behavior_hash = ?
                """,
                (_utc_now(), reason, behavior_hash),
            )

    def candidate_parents(
        self,
        *,
        product: str | None = None,
        opportunity_type: str | None = None,
        limit: int = 50,
        exclude_holdout_exposed: bool = True,
        exclude_retired: bool = True,
        latest_outcomes: Sequence[str] | None = None,
        research_engine_digest: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return bounded generator parents, excluding tainted ancestry by default."""

        limit = _validate_limit(limit)
        if product is not None:
            product = _validate_text(product, label="product", maximum=128)
        if opportunity_type is not None:
            opportunity_type = _validate_text(
                opportunity_type, label="opportunity_type", maximum=128
            )
        allowed_outcomes = None if latest_outcomes is None else set(latest_outcomes)
        if research_engine_digest is not None:
            research_engine_digest = _validate_hash(
                research_engine_digest,
                label="research_engine_digest",
            )
        params: list[Any] = []
        where: list[str] = []
        if product is not None:
            where.append("product = ?")
            params.append(product)
        if opportunity_type is not None:
            where.append("opportunity_type = ?")
            params.append(opportunity_type)
        query = "SELECT * FROM strategies"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC, behavior_hash"
        results: list[dict[str, Any]] = []
        with self._database(write=False) as connection:
            for row in connection.execute(query, params):
                if self._lineage_is_tainted(
                    connection,
                    row["behavior_hash"],
                    holdout=exclude_holdout_exposed,
                    retired=exclude_retired,
                ):
                    continue
                payload = self._strategy_payload(
                    connection,
                    row,
                    research_engine_digest=research_engine_digest,
                )
                latest = payload["latest_evaluation"]
                if allowed_outcomes is not None and (
                    latest is None or latest.get("outcome") not in allowed_outcomes
                ):
                    continue
                results.append(payload)
                if len(results) >= limit:
                    break
        return results

    def pending_strategies(
        self,
        *,
        product: str | None = None,
        opportunity_type: str | None = None,
        limit: int = 50,
        research_engine_digest: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return untested work, or safe work needing current-engine revalidation.

        Revalidation returns the existing canonical strategy instead of trying
        to register a duplicate. Any strategy whose ancestry has touched a
        protected holdout (or has been retired) remains ineligible.
        """

        limit = _validate_limit(limit)
        if product is not None:
            product = _validate_text(product, label="product", maximum=128)
        if opportunity_type is not None:
            opportunity_type = _validate_text(
                opportunity_type, label="opportunity_type", maximum=128
            )
        if research_engine_digest is not None:
            research_engine_digest = _validate_hash(
                research_engine_digest,
                label="research_engine_digest",
            )
        params: list[Any] = []
        where: list[str] = []
        if research_engine_digest is None:
            where.append(
                "NOT EXISTS (SELECT 1 FROM evaluations e "
                "WHERE e.behavior_hash = strategies.behavior_hash)"
            )
        else:
            where.append(
                "NOT EXISTS (SELECT 1 FROM evaluation_engine_scopes scope "
                "WHERE scope.behavior_hash = strategies.behavior_hash "
                "AND scope.research_engine_digest = ? "
                f"AND scope.phase NOT IN ({','.join('?' for _ in PROTECTED_PHASES)}))"
            )
            params.extend((research_engine_digest, *sorted(PROTECTED_PHASES)))
        if product is not None:
            where.append("product = ?")
            params.append(product)
        if opportunity_type is not None:
            where.append("opportunity_type = ?")
            params.append(opportunity_type)
        results: list[dict[str, Any]] = []
        with self._database(write=False) as connection:
            for row in connection.execute(
                "SELECT * FROM strategies WHERE "
                + " AND ".join(where)
                + " ORDER BY CASE WHEN EXISTS ("
                "SELECT 1 FROM evaluations prior "
                "WHERE prior.behavior_hash = strategies.behavior_hash"
                ") THEN 1 ELSE 0 END, created_at, behavior_hash",
                params,
            ):
                if self._lineage_is_tainted(
                    connection,
                    row["behavior_hash"],
                    holdout=True,
                    retired=True,
                ):
                    continue
                prior_evaluation = connection.execute(
                    "SELECT 1 FROM evaluations WHERE behavior_hash = ? LIMIT 1",
                    (row["behavior_hash"],),
                ).fetchone()
                payload = self._strategy_payload(
                    connection,
                    row,
                    research_engine_digest=research_engine_digest,
                )
                payload["revalidation_required"] = prior_evaluation is not None
                results.append(payload)
                if len(results) >= limit:
                    break
        return results

    def dedup_population(
        self,
        *,
        product: str,
        maximum: int = 50_000,
    ) -> list[dict[str, Any]]:
        """Return every prior behavior for product-wide semantic deduplication.

        Holdout-exposed and retired strategies are intentionally included:
        neither a taxonomy rename nor retiring an old branch may let a fresh
        root probe substantially the same historical holdout again.  The hard
        ceiling fails closed instead of silently ignoring older memory.
        """

        product = _validate_text(product, label="product", maximum=128)
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 1 <= maximum <= 100_000
        ):
            raise ValueError("maximum must be an integer between 1 and 100000")
        with self._database(write=False) as connection:
            rows = list(
                connection.execute(
                    """
                    SELECT behavior_hash, primary_spec_json, metadata_json
                    FROM strategies WHERE product = ?
                    ORDER BY created_at, behavior_hash LIMIT ?
                    """,
                    (product, maximum + 1),
                )
            )
        if len(rows) > maximum:
            raise ExperimentMemoryError(
                f"dedup population for {product} exceeds safe comparison limit {maximum}"
            )
        return [
            {
                "behavior_hash": row["behavior_hash"],
                "submitted_spec": _stored_json(
                    row["primary_spec_json"], label="strategy primary JSON"
                ),
                "metadata": _stored_json(
                    row["metadata_json"], label="strategy metadata JSON"
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _lineage_is_tainted(
        connection: sqlite3.Connection,
        behavior_hash: str,
        *,
        holdout: bool,
        retired: bool,
    ) -> bool:
        if not holdout and not retired:
            return False
        clauses: list[str] = []
        if holdout:
            clauses.append("holdout_exposed_at IS NOT NULL")
        if retired:
            clauses.append("retired_at IS NOT NULL")
        row = connection.execute(
            f"""
            WITH RECURSIVE ancestry(behavior_hash) AS (
                SELECT ?
                UNION
                SELECT edge.parent_hash
                FROM lineage_edges edge
                JOIN ancestry current ON edge.child_hash = current.behavior_hash
            )
            SELECT 1 FROM strategies
            WHERE behavior_hash IN (SELECT behavior_hash FROM ancestry)
              AND ({" OR ".join(clauses)})
            LIMIT 1
            """,
            (behavior_hash,),
        ).fetchone()
        if row is not None:
            return True
        if holdout:
            roots = ExperimentMemory._lineage_roots(connection, behavior_hash)
            placeholders = ",".join("?" for _ in roots)
            claimed = connection.execute(
                f"""
                SELECT 1 FROM holdout_claim_scopes
                WHERE lineage_root_hash IN ({placeholders}) LIMIT 1
                """,
                tuple(sorted(roots)),
            ).fetchone()
            if claimed is not None:
                return True
        return False

    def ancestry(
        self,
        behavior_hash: str,
        *,
        max_depth: int = 8,
        limit: int = 100,
    ) -> dict[str, Any]:
        behavior_hash = _validate_hash(behavior_hash, label="behavior_hash")
        limit = _validate_limit(limit)
        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or not 0 <= max_depth <= 64
        ):
            raise ValueError("max_depth must be between 0 and 64")
        with self._database(write=False) as connection:
            self._strategy_row(connection, behavior_hash)
            rows = list(
                connection.execute(
                    """
                    WITH RECURSIVE tree(child_hash, parent_hash, depth) AS (
                        SELECT child_hash, parent_hash, 1 FROM lineage_edges
                        WHERE child_hash = ?
                        UNION ALL
                        SELECT edge.child_hash, edge.parent_hash, tree.depth + 1
                        FROM lineage_edges edge JOIN tree ON edge.child_hash = tree.parent_hash
                        WHERE tree.depth < ?
                    )
                    SELECT child_hash, parent_hash, depth FROM tree
                    ORDER BY depth, child_hash, parent_hash LIMIT ?
                    """,
                    (behavior_hash, max_depth, limit + 1),
                )
            )
        truncated = len(rows) > limit
        rows = rows[:limit]
        return {
            "behavior_hash": behavior_hash,
            "max_depth": max_depth,
            "truncated": truncated,
            "edges": [dict(row) for row in rows],
        }

    def generator_feedback(
        self,
        *,
        category_limit: int = 100,
        research_engine_digest: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded adaptive evidence for search-policy selection."""

        category_limit = _validate_limit(category_limit)
        if research_engine_digest is not None:
            research_engine_digest = _validate_hash(
                research_engine_digest,
                label="research_engine_digest",
            )
        with self._database(write=False) as connection:
            totals = {
                "strategies": connection.execute("SELECT COUNT(*) FROM strategies").fetchone()[0],
                "identities": connection.execute(
                    "SELECT COUNT(*) FROM strategy_identities"
                ).fetchone()[0],
                "duplicate_identities": connection.execute(
                    "SELECT COUNT(*) FROM strategy_identities WHERE is_duplicate = 1"
                ).fetchone()[0],
                "evaluations": connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
                "claimed": connection.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE status = 'claimed'"
                ).fetchone()[0],
                "completed": connection.execute(
                    "SELECT COUNT(*) FROM evaluations WHERE status = 'completed'"
                ).fetchone()[0],
                "holdout_exposed": connection.execute(
                    "SELECT COUNT(*) FROM strategies WHERE holdout_exposed_at IS NOT NULL"
                ).fetchone()[0],
                "retired": connection.execute(
                    "SELECT COUNT(*) FROM strategies WHERE retired_at IS NOT NULL"
                ).fetchone()[0],
            }
            outcome_counts: Counter[str] = Counter()
            reason_counts: Counter[str] = Counter()
            methods: dict[str, dict[str, Any]] = defaultdict(self._feedback_bucket)
            families: dict[str, dict[str, Any]] = defaultdict(self._feedback_bucket)
            primitives: dict[str, dict[str, Any]] = defaultdict(self._feedback_bucket)
            strategy_info: dict[str, tuple[str, str, list[str], float]] = {}

            for row in connection.execute(
                """
                SELECT behavior_hash, generation_method, metadata_json,
                       primitive_tokens_json, novelty_score FROM strategies
                """
            ):
                metadata = _stored_json(row["metadata_json"], label="strategy metadata JSON")
                family = str(metadata.get("family") or "unknown")
                primitive_list = _stored_json(
                    row["primitive_tokens_json"], label="strategy primitive-token JSON"
                )
                method = row["generation_method"]
                novelty = float(row["novelty_score"])
                strategy_info[row["behavior_hash"]] = (method, family, primitive_list, novelty)
                for bucket in [
                    methods[method],
                    families[family],
                    *(primitives[p] for p in primitive_list),
                ]:
                    bucket["unique_strategies"] += 1
                    bucket["novelty_sum"] += novelty

            for row in connection.execute(
                """
                SELECT identity.generation_method, identity.is_duplicate,
                       identity.metadata_json, strategy.metadata_json AS strategy_metadata_json,
                       strategy.primitive_tokens_json
                FROM strategy_identities identity
                JOIN strategies strategy ON strategy.behavior_hash = identity.behavior_hash
                """
            ):
                bucket = methods[row["generation_method"]]
                bucket["proposals"] += 1
                if row["is_duplicate"]:
                    bucket["duplicates"] += 1
                identity_metadata = _stored_json(
                    row["metadata_json"], label="identity metadata JSON"
                )
                strategy_metadata = _stored_json(
                    row["strategy_metadata_json"], label="strategy metadata JSON"
                )
                family = str(
                    identity_metadata.get("family") or strategy_metadata.get("family") or "unknown"
                )
                proposal_buckets = [
                    families[family],
                    *(
                        primitives[primitive]
                        for primitive in _stored_json(
                            row["primitive_tokens_json"],
                            label="strategy primitive-token JSON",
                        )
                    ),
                ]
                for proposal_bucket in proposal_buckets:
                    proposal_bucket["proposals"] += 1
                    if row["is_duplicate"]:
                        proposal_bucket["duplicates"] += 1

            protected_placeholders = ",".join("?" for _ in PROTECTED_PHASES)
            protected_phases = tuple(sorted(PROTECTED_PHASES))
            for row in connection.execute(
                f"""
                SELECT behavior_hash, outcome, rejection_reasons_json, protocol_json
                FROM evaluations
                WHERE status = 'completed' AND phase NOT IN ({protected_placeholders})
                """,
                protected_phases,
            ):
                if research_engine_digest is not None:
                    protocol = _stored_json(
                        row["protocol_json"], label="evaluation protocol JSON"
                    )
                    if protocol.get("research_engine_digest") != research_engine_digest:
                        continue
                outcome = row["outcome"]
                reasons = _stored_json(
                    row["rejection_reasons_json"], label="evaluation rejection-reason JSON"
                )
                outcome_counts[outcome] += 1
                reason_counts.update(reasons)
                info = strategy_info.get(row["behavior_hash"])
                if info is None:
                    raise ExperimentMemoryCorruptionError("evaluation references unknown strategy")
                method, family, primitive_list, _ = info
                for bucket in [
                    methods[method],
                    families[family],
                    *(primitives[p] for p in primitive_list),
                ]:
                    bucket["experiments"] += 1
                    bucket["outcomes"][outcome] += 1
                    bucket["rejection_reasons"].update(reasons)

            parent_performance: dict[str, dict[str, Any]] = {}
            for row in connection.execute(
                "SELECT parent_hash, child_hash FROM lineage_edges ORDER BY parent_hash, child_hash"
            ):
                parent = parent_performance.setdefault(
                    row["parent_hash"],
                    {
                        "parent_hash": row["parent_hash"],
                        "children": 0,
                        "child_outcomes": Counter(),
                        "child_rejection_reasons": Counter(),
                    },
                )
                parent["children"] += 1
                for evaluation in connection.execute(
                    f"""
                    SELECT outcome, rejection_reasons_json, protocol_json FROM evaluations
                    WHERE behavior_hash = ? AND status = 'completed'
                      AND phase NOT IN ({protected_placeholders})
                    """,
                    (row["child_hash"], *protected_phases),
                ):
                    if research_engine_digest is not None:
                        protocol = _stored_json(
                            evaluation["protocol_json"], label="evaluation protocol JSON"
                        )
                        if protocol.get("research_engine_digest") != research_engine_digest:
                            continue
                    parent["child_outcomes"][evaluation["outcome"]] += 1
                    parent["child_rejection_reasons"].update(
                        _stored_json(
                            evaluation["rejection_reasons_json"],
                            label="evaluation rejection-reason JSON",
                        )
                    )

        return {
            "schema_version": SCHEMA_VERSION,
            "research_engine_digest": research_engine_digest,
            "adaptive_evaluations": sum(outcome_counts.values()),
            "totals": totals,
            "outcomes": dict(outcome_counts.most_common(category_limit)),
            "rejection_reasons": dict(reason_counts.most_common(category_limit)),
            "generation_methods": self._render_feedback_groups(methods, category_limit),
            "families": self._render_feedback_groups(families, category_limit),
            "primitives": self._render_feedback_groups(primitives, category_limit),
            "parent_performance": [
                {
                    **{k: v for k, v in item.items() if not isinstance(v, Counter)},
                    "child_outcomes": dict(item["child_outcomes"].most_common(category_limit)),
                    "child_rejection_reasons": dict(
                        item["child_rejection_reasons"].most_common(category_limit)
                    ),
                }
                for item in sorted(
                    parent_performance.values(),
                    key=lambda value: (-value["children"], value["parent_hash"]),
                )[:category_limit]
            ],
        }

    @staticmethod
    def _feedback_bucket() -> dict[str, Any]:
        return {
            "proposals": 0,
            "duplicates": 0,
            "unique_strategies": 0,
            "experiments": 0,
            "novelty_sum": 0.0,
            "outcomes": Counter(),
            "rejection_reasons": Counter(),
        }

    @staticmethod
    def _render_feedback_groups(
        groups: Mapping[str, dict[str, Any]], limit: int
    ) -> dict[str, dict[str, Any]]:
        ordered = sorted(
            groups.items(),
            key=lambda item: (
                -item[1]["experiments"],
                -item[1]["unique_strategies"],
                item[0],
            ),
        )[:limit]
        rendered: dict[str, dict[str, Any]] = {}
        for name, bucket in ordered:
            unique = bucket["unique_strategies"]
            proposals = bucket["proposals"]
            rendered[name] = {
                "proposals": proposals,
                "duplicates": bucket["duplicates"],
                "duplicate_rate": bucket["duplicates"] / proposals if proposals else 0.0,
                "unique_strategies": unique,
                "experiments": bucket["experiments"],
                "mean_novelty": bucket["novelty_sum"] / unique if unique else 0.0,
                "outcomes": dict(bucket["outcomes"].most_common(limit)),
                "rejection_reasons": dict(bucket["rejection_reasons"].most_common(limit)),
            }
        return rendered

    def compact_storage(
        self,
        *,
        maximum_rows: int = MAX_COMPACTION_ROWS,
        vacuum: bool = True,
    ) -> dict[str, Any]:
        """Compact bulky immutable JSON without discarding research evidence.

        The work is row-bounded. Every rewritten value remains self-describing
        and deep-integrity checks decode and re-hash the exact original
        semantics. Lineage, strategy IDs, evaluation keys, engine scopes, and
        protected-holdout claims are never deleted or rewritten.
        """

        if (
            not isinstance(maximum_rows, int)
            or isinstance(maximum_rows, bool)
            or not 1 <= maximum_rows <= MAX_COMPACTION_ROWS
        ):
            raise ValueError(
                f"maximum_rows must be an integer between 1 and {MAX_COMPACTION_ROWS}"
            )
        if not isinstance(vacuum, bool):
            raise ValueError("vacuum must be boolean")

        before_bytes = self.path.stat().st_size
        table_specs = (
            (
                "strategies",
                "behavior_hash",
                (
                    "canonical_spec_json",
                    "primary_spec_json",
                    "metadata_json",
                    "primitive_tokens_json",
                    "novelty_tokens_json",
                ),
            ),
            (
                "strategy_identities",
                "strategy_id",
                ("submitted_spec_json", "metadata_json", "parent_hashes_json"),
            ),
            (
                "evaluations",
                "evaluation_key",
                (
                    "dataset_json",
                    "window_json",
                    "protocol_json",
                    "rejection_reasons_json",
                    "metrics_json",
                    "details_json",
                ),
            ),
        )
        compacted_by_table: Counter[str] = Counter()
        with self._database(write=True) as connection:
            remaining = maximum_rows
            for table, key_column, json_columns in table_specs:
                if remaining <= 0:
                    break
                needs_compaction = " OR ".join(
                    f"{column} NOT LIKE ?" for column in json_columns
                )
                rows = list(
                    connection.execute(
                        f"SELECT {key_column}, {', '.join(json_columns)} FROM {table} "
                        f"WHERE {needs_compaction} ORDER BY rowid LIMIT ?",
                        (*([f"{COMPACT_JSON_PREFIX}%"] * len(json_columns)), remaining),
                    )
                )
                assignments = ", ".join(f"{column} = ?" for column in json_columns)
                for row in rows:
                    encoded = [
                        _compact_json_storage(
                            row[column],
                            label=f"{table}.{column}",
                        )
                        for column in json_columns
                    ]
                    connection.execute(
                        f"UPDATE {table} SET {assignments} WHERE {key_column} = ?",
                        (*encoded, row[key_column]),
                    )
                    compacted_by_table[table] += 1
                remaining -= len(rows)
            self._deep_integrity_check(connection)

        rows_compacted = sum(compacted_by_table.values())
        did_vacuum = vacuum and rows_compacted > 0
        if did_vacuum:
            with self._mutex:
                connection = self._ensure_open()
                try:
                    connection.execute("VACUUM")
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                        raise ExperimentMemoryBusyError(
                            f"experiment memory remained locked during compaction: {self.path}"
                        ) from exc
                    raise ExperimentMemoryCorruptionError(
                        f"experiment memory compaction failed at {self.path}: {exc}"
                    ) from exc
        integrity = self.integrity_check(deep=True)
        after_bytes = self.path.stat().st_size
        return {
            "ok": True,
            "rows_compacted": rows_compacted,
            "by_table": dict(sorted(compacted_by_table.items())),
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "saved_bytes": max(0, before_bytes - after_bytes),
            "vacuumed": did_vacuum,
            "integrity": integrity,
        }

    def backup_to(self, destination: Path) -> Path:
        """Create an atomic, integrity-checked online backup of the memory DB."""

        destination = Path(destination)
        if destination.is_symlink():
            raise ValueError(f"backup destination must not be a symlink: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        with self._mutex:
            source = self._ensure_open()
            try:
                descriptor, name = tempfile.mkstemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                )
                os.close(descriptor)
                temporary = Path(name)
                target = sqlite3.connect(temporary)
                try:
                    source.backup(target)
                    quick = [row[0] for row in target.execute("PRAGMA quick_check")]
                    if quick != ["ok"]:
                        raise ExperimentMemoryCorruptionError(
                            f"experiment-memory backup failed integrity check: {quick[:3]}"
                        )
                    target.commit()
                finally:
                    target.close()
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
                temporary = None
                directory_fd = os.open(
                    destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return destination
