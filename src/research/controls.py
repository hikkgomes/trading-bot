"""Deterministic negative controls derived from immutable research inputs."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from src.domain._codec import canonical_hash


class NegativeControlError(ValueError):
    """A negative-control input cannot be evaluated safely."""


def derive_control_returns(
    name: str,
    signals: Sequence[float],
    returns: Sequence[float],
    *,
    seed_material: object,
    instrument_scope: Sequence[str] = (),
) -> tuple[list[float], str] | None:
    """Return a control PnL series and method, or ``None`` when inapplicable.

    Controls are calculated from the same immutable signal and market-return
    rows as the candidate.  They are not fabricated zero series, and their
    deterministic seed is part of the evidence identity.
    """

    values = _aligned_values(signals, returns)
    if values is None:
        return None
    signal_values, return_values = values
    if name in {"cross_instrument", "predeclared_universe_holdout"}:
        return None
    if name == "feature_ablation":
        return [0.0] * len(return_values), "feature_ablation_v1"
    if name == "parameter_neighbourhood":
        shifted = _rotate(signal_values, 1)
        return _pnl(shifted, return_values), "parameter_neighbourhood_shift_v1"
    if name == "placebo_event_times":
        shifted = _rotate(signal_values, max(1, len(signal_values) // 3))
        return _pnl(shifted, return_values), "placebo_event_times_v1"
    if name == "block_permutation":
        permuted = _block_permutation(return_values, seed_material=seed_material)
        return _pnl(signal_values, permuted), "block_permutation_v1"
    if name == "synthetic_autocorrelated_null":
        synthetic = _synthetic_autocorrelated(return_values, seed_material=seed_material)
        return _pnl(signal_values, synthetic), "synthetic_autocorrelated_null_v1"
    return None


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


def _aligned_values(
    signals: Sequence[float], returns: Sequence[float]
) -> tuple[list[float], list[float]] | None:
    aligned = min(len(signals), len(returns))
    if aligned < 3:
        return None
    signal_values = [float(value) for value in signals[:aligned]]
    return_values = [float(value) for value in returns[:aligned]]
    if not all(math.isfinite(value) for value in (*signal_values, *return_values)):
        raise NegativeControlError("negative-control inputs must be finite")
    return signal_values, return_values


def _pnl(signals: Sequence[float], returns: Sequence[float]) -> list[float]:
    return [float(signal) * float(value) for signal, value in zip(signals, returns, strict=True)]


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
