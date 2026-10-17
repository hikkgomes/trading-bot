"""Small validation and canonical-encoding helpers for domain contracts."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import sys
from decimal import Decimal
from enum import Enum
from typing import Any


def finite(value: float, *, field: str, minimum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum:g}")
    return value


def non_empty(value: str, *, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field} cannot be empty")
    return value


def timestamp(value: str | dt.datetime, *, field: str, preserve_subseconds: bool = False) -> str:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = non_empty(value, field=field)
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    parsed = parsed.astimezone(dt.UTC)
    return (parsed if preserve_subseconds else parsed.replace(microsecond=0)).isoformat()


def json_value(value: Any, *, field: str) -> Any:
    try:
        return _copy_json_value(value, set())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-safe") from exc


def _copy_json_value(value: Any, ancestors: set[int]) -> Any:
    """Detach a JSON tree without allocating its entire encoded text as well."""
    if isinstance(value, dict | list | tuple):
        return _copy_json_container(value, ancestors)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        integer = int.__int__(value)
        if integer.bit_length() > 3 * sys.int_info.str_digits_check_threshold:
            str(integer)  # Preserve the JSON encoder's integer digit limit.
        return integer
    if isinstance(value, float):
        number = float.__float__(value)
        if not math.isfinite(number):
            raise ValueError("non-finite JSON number")
        return number
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _copy_json_container(value: Any, ancestors: set[int]) -> Any:
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Circular reference detected")
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            return {
                _json_key(key): _copy_json_value(item, ancestors)
                for key, item in sorted(value.items())
            }
        return [_copy_json_value(item, ancestors) for item in value]
    finally:
        ancestors.remove(identity)


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return str.__str__(key)
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(int.__int__(key))
    if isinstance(key, float) and math.isfinite(key):
        return repr(float.__float__(key))
    raise TypeError("JSON object keys must be finite numbers, strings, booleans or null")


def canonical_hash(value: Any) -> str:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        value = value.value
    encoder = json.JSONEncoder(
        default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode())
    return f"sha256:{digest.hexdigest()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def to_primitive(value: Any) -> Any:
    """Return a JSON-safe representation of an enum/dataclass contract."""
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3_600, "d": 86_400, "w": 604_800}
    value = str(timeframe).strip()
    if len(value) < 2 or value[-1] not in units:
        raise ValueError(f"unsupported declared timeframe: {timeframe}")
    try:
        count = int(value[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported declared timeframe: {timeframe}") from exc
    if count <= 0:
        raise ValueError(f"unsupported declared timeframe: {timeframe}")
    return count * units[value[-1]]
