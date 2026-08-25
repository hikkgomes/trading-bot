"""Small validation and canonical-encoding helpers for domain contracts."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
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


def timestamp(value: str | dt.datetime, *, field: str) -> str:
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
    return parsed.astimezone(dt.UTC).replace(microsecond=0).isoformat()


def json_value(value: Any, *, field: str) -> Any:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-safe") from exc
    return json.loads(raw)


def canonical_hash(value: Any) -> str:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        value = value.value
    raw = json.dumps(
        value, default=_json_default, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()}"


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
