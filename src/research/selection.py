"""Selection-adjusted block bootstrap for a sealed cohort of net return streams."""

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.domain._codec import canonical_hash, timestamp

METHOD = "selection_adjusted_block_bootstrap/v1"


@dataclass(frozen=True)
class FrozenCohort:
    members: tuple[str, ...]
    instruments: tuple[str, ...]
    frozen_at: str
    development_hash: str
    selection_rule: str = "maximum_studentised_mean"

    def __post_init__(self) -> None:
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("cohort must contain every unique finalist")
        if not self.instruments or len(set(self.instruments)) != len(self.instruments):
            raise ValueError("cohort instruments must be unique")
        object.__setattr__(self, "frozen_at", timestamp(self.frozen_at, field="cohort frozen_at"))
        if self.selection_rule != "maximum_studentised_mean":
            raise ValueError("unsupported frozen cohort selection rule")

    @property
    def manifest_id(self) -> str:
        return canonical_hash(self)


def development_block_length(returns: np.ndarray) -> int:
    values = np.asarray(returns, dtype=float)
    if values.ndim == 3:
        values = values.sum(axis=2)
    if values.ndim != 2 or len(values) < 40 or not np.isfinite(values).all():
        raise ValueError("block selection requires finite development-only observations")
    centred = values - values.mean(axis=0)
    variance = np.mean(centred * centred, axis=0)
    autocovariance = np.mean(centred[:-1] * centred[1:], axis=0)
    rho = np.divide(autocovariance, variance, out=np.zeros_like(variance), where=variance > 0)
    dependence = float(np.clip(np.max(rho), 0, 0.9))
    proposed = int(np.ceil(len(values) ** (1 / 3) * (1 + dependence)))
    return max(2, min(len(values) // 16, proposed))


def selection_adjusted_block_bootstrap(
    cohort: FrozenCohort,
    *,
    timestamps: Sequence[str],
    net_returns: np.ndarray,
    benchmark_returns: Sequence[float],
    development_returns: np.ndarray,
    opened_at: str,
    seed: int,
    draws: int = 2000,
) -> dict[str, Any]:
    if draws < 2000:
        raise ValueError("selection inference requires at least 2000 draws")
    if timestamp(opened_at, field="protected opened_at") <= cohort.frozen_at:
        raise ValueError("cohort must be frozen before protected evidence is opened")
    values = _aligned_returns(cohort, timestamps, net_returns, benchmark_returns)
    development = np.asarray(development_returns, dtype=float)
    if canonical_hash(development.tolist()) != cohort.development_hash:
        raise ValueError("development returns do not match the frozen cohort manifest")
    base = development_block_length(development)
    lengths = tuple(sorted({max(1, base // 2), base, base * 2}))
    results = [
        _bootstrap_length(values, length, seed + index, draws)
        for index, length in enumerate(lengths)
    ]
    receipt = {
        "method": METHOD,
        "cohort_id": cohort.manifest_id,
        "members": list(cohort.members),
        "instruments": list(cohort.instruments),
        "sample_unit": "calendar_interval",
        "timestamps": list(timestamps),
        "seed": seed,
        "draws": draws,
        "block_lengths": lengths,
        "studentizer": "frozen_Bartlett_HAC_per_block_length",
        "input_hash": canonical_hash(
            {
                "net_returns": np.asarray(net_returns).tolist(),
                "benchmark_returns": list(benchmark_returns),
            }
        ),
        "qualification_authority": False,
    }
    if any(result is None for result in results):
        return {
            **receipt,
            "status": "inconclusive",
            "reason": "sample_variance_or_dependence_unsupported",
        }
    valid = [result for result in results if result is not None]
    pvalues = np.max(np.stack([result["pvalues"] for result in valid]), axis=0)
    effective = np.min(np.stack([result["effective_samples"] for result in valid]), axis=0)
    return {
        **receipt,
        "status": "measured",
        "adjusted_pvalues": dict(zip(cohort.members, pvalues.tolist(), strict=True)),
        "effective_samples": dict(zip(cohort.members, effective.tolist(), strict=True)),
    }


def _aligned_returns(
    cohort: FrozenCohort, times: Sequence[str], returns: np.ndarray, benchmark: Sequence[float]
) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    reference = np.asarray(benchmark, dtype=float)
    expected = (len(times), len(cohort.members), len(cohort.instruments))
    if values.shape not in {expected, expected[:2]} or reference.shape != (len(times),):
        raise ValueError("cohort return streams must share one calendar index and instrument axis")
    normalised = [timestamp(value, field="return timestamp") for value in times]
    if any(first >= second for first, second in zip(normalised, normalised[1:], strict=False)):
        raise ValueError("return timestamps must be unique and chronological")
    parsed = [dt.datetime.fromisoformat(value) for value in normalised]
    durations = {second - first for first, second in zip(parsed, parsed[1:], strict=False)}
    if len(durations) > 1:
        raise ValueError("return calendar intervals must have equal duration")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("net and benchmark returns must be finite")
    # Instrument contributions are combined only after alignment. Every bootstrap
    # draw uses the same time indices for every instrument and configuration.
    portfolio_returns = values.sum(axis=2) if values.ndim == 3 else values
    return portfolio_returns - reference[:, None]


def _hac_variance(values: np.ndarray, block: int) -> np.ndarray:
    centred = values - values.mean(axis=0)
    variance = np.mean(centred * centred, axis=0)
    for lag in range(1, block):
        covariance = np.mean(centred[:-lag] * centred[lag:], axis=0)
        variance += 2 * (1 - lag / block) * covariance
    return variance


def _bootstrap_length(
    values: np.ndarray, block: int, seed: int, draws: int
) -> Mapping[str, np.ndarray] | None:
    count = len(values)
    if count < max(40, block * 8):
        return None
    variance = _hac_variance(values, block)
    marginal = np.var(values, axis=0)
    if np.any(variance <= 1e-16) or np.any(marginal <= 1e-16):
        return None
    effective = np.minimum(count, count * marginal / variance)
    if np.any(effective < 20):
        return None
    scale = np.sqrt(variance / count)
    observed = values.mean(axis=0) / scale
    centred = values - values.mean(axis=0)
    exceedances = _maximum_statistic_counts(centred, scale, observed, block, seed, draws)
    return {"pvalues": (exceedances + 1) / (draws + 1), "effective_samples": effective}


def _maximum_statistic_counts(
    centred: np.ndarray, scale: np.ndarray, observed: np.ndarray, block: int, seed: int, draws: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = len(centred)
    blocks = (count + block - 1) // block
    exceedances: np.ndarray = np.zeros(len(scale), dtype=int)
    # Bound peak memory independently of the requested number of draws.
    batch_size = max(1, min(64, 1_000_000 // max(1, centred.size)))
    for offset in range(0, draws, batch_size):
        size = min(batch_size, draws - offset)
        starts = rng.integers(0, count, size=(size, blocks, 1))
        indices = ((starts + np.arange(block)) % count).reshape(size, -1)[:, :count]
        statistics = centred[indices].mean(axis=1) / scale
        maximum = statistics.max(axis=1)
        exceedances += (maximum[:, None] >= observed).sum(axis=0)
    return exceedances
