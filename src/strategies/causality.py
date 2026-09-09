"""Check whether later observations alter already-issued strategy decisions."""

from collections.abc import Callable, Sequence
from typing import Any


def prefix_invariant(function: Callable[[Any], Sequence[Any]], rows: Sequence[Any]) -> bool:
    complete = tuple(function(rows))
    if len(complete) != len(rows):
        raise ValueError("causality check requires timestamp-aligned decisions")
    cuts = sorted({len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4} - {0})
    return bool(cuts) and all(tuple(function(rows[:cut])) == complete[:cut] for cut in cuts)
