"""Frozen, read-only SQLite archive decoding and identity rules; never for new strategies."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1

MEMORY_FORMAT = "autonomous_strategy_experiment_memory"

MAX_DECOMPRESSED_JSON_BYTES = 64 * 1024 * 1024

COMPACT_JSON_PREFIX = "zlib-json-v1:"

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


class ExperimentMemoryError(RuntimeError):
    """Base class for durable experiment-memory failures."""


class ExperimentMemoryCorruptionError(ExperimentMemoryError):
    """The database or its logical contents failed integrity validation."""


def _validate_json(value: Any, *, label: str) -> Any:
    """Return a detached, strict-JSON value or raise a useful ValueError."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values: {exc}") from exc


def _normalise_behavior_mapping(value: Mapping[str, Any], *, depth: int) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    ignored = _TOP_LEVEL_NON_BEHAVIORAL if depth == 0 else _NESTED_NON_BEHAVIORAL
    for child_key in sorted(value):
        if not isinstance(child_key, str):
            raise ValueError("strategy spec object keys must be strings")
        if child_key not in ignored:
            normalised[child_key] = _normalise_behavior(
                value[child_key], key=child_key, depth=depth + 1
            )
    return normalised


def _normalise_behavior_sequence(value: Sequence[Any], *, key: str | None, depth: int) -> list[Any]:
    items = [_normalise_behavior(item, depth=depth + 1) for item in value]
    if key in _COMMUTATIVE_LIST_FIELDS:
        items.sort(key=_canonical_json)
    return items


def _normalise_behavior(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("strategy spec contains a non-finite number")
        if value == 0:
            return 0
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        return _normalise_behavior_mapping(value, depth=depth)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return _normalise_behavior_sequence(value, key=key, depth=depth)
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
