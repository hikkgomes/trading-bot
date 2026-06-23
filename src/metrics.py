from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _as_returns(returns: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(returns), dtype=float)
    return arr[np.isfinite(arr)]


def sharpe_ratio(returns: Iterable[float]) -> float:
    arr = _as_returns(returns)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std)


def expected_max_sharpe(n_trials: int, sr_std_trials: float) -> float:
    """Bailey/Lopez de Prado expected maximum SR among n_trials null strategies.

    Scaled by the cross-trial SR dispersion so it lives in the same units as
    the per-trade Sharpe being tested. Returns 0 when deflation is impossible
    (single trial or degenerate dispersion).
    """
    n_trials = max(int(n_trials), 1)
    if n_trials <= 1 or not np.isfinite(sr_std_trials) or sr_std_trials <= 0:
        return 0.0
    return float(
        sr_std_trials
        * (
            (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    in_sample_sr: float,
    n_trials: int,
    skew: float,
    kurt: float,
    n_obs: int,
    sr_std_trials: float = 0.0,
) -> float:
    """P(true SR > 0) after deflating for selection across n_trials candidates.

    With sr_std_trials=0 (or n_trials<=1) this degrades to the Probabilistic
    Sharpe Ratio against a zero benchmark (no deflation).
    """
    if n_obs <= 1 or not np.isfinite(in_sample_sr):
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sr_std_trials)
    variance = max(1e-12, 1.0 - skew * in_sample_sr + ((kurt - 1.0) / 4.0) * in_sample_sr**2)
    z_score = (in_sample_sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(variance)
    return float(norm.cdf(z_score))


def bootstrap_sharpe_ci(
    returns: Iterable[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    block_size: int | None = None,
    random_state: int = 42,
) -> tuple[float, float]:
    arr = _as_returns(returns)
    if arr.size < 2:
        return (0.0, 0.0)
    block_size = int(block_size or max(1, round(np.sqrt(arr.size))))
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(arr.size / block_size))
    starts = rng.integers(0, arr.size, size=(int(n_boot), n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % arr.size
    samples = arr[indices.reshape(int(n_boot), -1)[:, : arr.size]]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    estimates = np.divide(means, stds, out=np.zeros_like(means), where=stds > 0)
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(low), float(high))


def probability_backtest_overfitting(
    per_window_returns_matrix: Sequence[Sequence[float]],
    n_splits: int = 16,
) -> float:
    matrix = np.asarray(per_window_returns_matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return 0.0
    n_strategies, n_windows = matrix.shape
    s = min(max(2, int(n_splits)), n_windows)
    if s % 2:
        s -= 1
    if s < 2:
        return 0.0
    split_returns = np.array([part.mean(axis=1) for part in np.array_split(matrix, s, axis=1)]).T
    logits = []
    split_ids = np.arange(s)
    for train_splits in combinations(range(s), s // 2):
        train_idx = np.array(train_splits, dtype=int)
        test_idx = np.setdiff1d(split_ids, train_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_scores = np.array([sharpe_ratio(row[train_idx]) for row in split_returns])
        test_scores = np.array([sharpe_ratio(row[test_idx]) for row in split_returns])
        winner = int(np.nanargmax(train_scores))
        rank = float((test_scores < test_scores[winner]).sum() + 1) / float(n_strategies + 1)
        rank = min(max(rank, 1e-12), 1.0 - 1e-12)
        logits.append(np.log(rank / (1.0 - rank)))
    if not logits:
        return 0.0
    return float(np.mean(np.asarray(logits) < 0.0))


def cluster_strategies_by_overlap(
    masks_dict: Mapping[str, Iterable[bool]],
    jaccard_threshold: float = 0.8,
) -> Dict[int, List[str]]:
    names = list(masks_dict.keys())
    masks = {name: np.asarray(list(masks_dict[name]), dtype=bool) for name in names}
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = masks[left], masks[right]
            denom = np.logical_or(a, b).sum()
            score = 1.0 if denom == 0 else float(np.logical_and(a, b).sum() / denom)
            if score >= jaccard_threshold:
                union(left, right)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        clusters[find(name)].append(name)
    return {idx: sorted(values) for idx, values in enumerate(clusters.values())}
