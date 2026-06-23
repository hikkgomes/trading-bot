"""Strategy registry.

A tiny name -> Strategy-class map so the CLI, backtester and (eventually) the
executor can discover any strategy regardless of which paradigm it belongs to
(simple rule, multi-timeframe, ML, condition-grid, ...). Register with the
``@register`` decorator; look up with ``get`` / ``available``.
"""

from __future__ import annotations

from typing import Dict, List, Type

_REGISTRY: Dict[str, Type] = {}


def register(cls: Type) -> Type:
    """Class decorator: register a Strategy subclass under its ``name``."""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} must set a unique class-level `name`.")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Duplicate strategy name {name!r} ({cls.__name__}).")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> Type:
    """Return the Strategy class registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {', '.join(available()) or '(none)'}"
        )
    return _REGISTRY[name]


def available() -> List[str]:
    """Sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def describe() -> Dict[str, str]:
    """Map of name -> one-line description for every registered strategy."""
    return {name: getattr(cls, "description", "") for name, cls in sorted(_REGISTRY.items())}
