"""Deterministic negative controls derived from immutable research inputs."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from src.domain._codec import canonical_hash


class NegativeControlError(ValueError):
    """A negative-control input cannot be evaluated safely."""


def derive_control_signals(
    signals: Sequence[float], *, seed_material: object
) -> dict[str, list[float]]:
    """Perturb entry timing while preserving execution costs and holding rules."""
    values = [float(value) for value in signals]
    if len(values) < 3:
        return {}
    if not all(math.isfinite(value) for value in values):
        raise NegativeControlError("control entry pulses must be finite")
    noise = _synthetic_autocorrelated(values, seed_material=seed_material)
    order = sorted(range(len(noise)), key=lambda index: noise[index])
    null = [0.0] * len(values)
    shorts = sum(value < 0 for value in values)
    longs = sum(value > 0 for value in values)
    for index in order[:shorts]:
        null[index] = -1.0
    for index in order[len(order) - longs :]:
        null[index] = 1.0
    return {
        "placebo_event_times": _rotate(values, max(1, len(values) // 3)),
        "block_permutation": _block_permutation(values, seed_material=seed_material),
        "synthetic_autocorrelated_null": null,
    }


def control_identity(
    name: str,
    values: Sequence[float],
    *,
    seed_material: object,
    method: str,
) -> str:
    """Hash the full derived control input for the audit receipt."""

    return canonical_hash(
        {
            "control": name,
            "method": method,
            "seed_material": seed_material,
            "returns": list(values),
        }
    )


def _rotate(values: Sequence[float], offset: int) -> list[float]:
    offset %= len(values)
    return [*values[offset:], *values[:offset]]


def _block_permutation(values: Sequence[float], *, seed_material: object) -> list[float]:
    block_size = max(2, int(math.sqrt(len(values))))
    blocks = [
        list(values[index : index + block_size]) for index in range(0, len(values), block_size)
    ]
    randomiser = random.Random(int(canonical_hash(seed_material)[7:23], 16))
    randomiser.shuffle(blocks)
    return [value for block in blocks for value in block]


def _synthetic_autocorrelated(values: Sequence[float], *, seed_material: object) -> list[float]:
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    if deviation == 0.0:
        return [mean] * len(values)
    lagged = _correlation(values[:-1], values[1:])
    rho = max(-0.95, min(0.95, lagged or 0.0))
    randomiser = random.Random(int(canonical_hash({"null": seed_material})[7:23], 16))
    innovations = [randomiser.gauss(0.0, deviation) for _ in values]
    result = [mean + innovations[0]]
    for innovation in innovations[1:]:
        result.append(mean + rho * (result[-1] - mean) + innovation * math.sqrt(1.0 - rho * rho))
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=False))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None
