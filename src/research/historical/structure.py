"""Causal market structure; anchors become usable only when confirmed."""

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.domain._codec import timestamp


@dataclass(frozen=True)
class MarketBar:
    """A closed bar whose timestamp is its availability time."""

    timestamp: str
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            timestamp(self.timestamp, field="market timestamp", preserve_subseconds=True),
        )
        if not all(
            math.isfinite(value) and value > 0 for value in (self.high, self.low, self.close)
        ):
            raise ValueError("historical market prices must be finite and positive")
        if not self.low <= self.close <= self.high:
            raise ValueError("historical market bar has inconsistent prices")


@dataclass(frozen=True)
class Pivot:
    anchor_at: str
    available_at: str
    price: float
    kind: str


@dataclass(frozen=True)
class StructureObservation:
    timestamp: str
    trend: float | None
    volatility: float | None
    structural_break: int
    retracement: float | None
    high_anchor: Pivot | None
    low_anchor: Pivot | None


def structure_features(
    bars: Sequence[MarketBar], *, pivot_width: int = 3, lookback: int = 20
) -> tuple[StructureObservation, ...]:
    if pivot_width < 1 or lookback < 3:
        raise ValueError("structure windows are invalid")
    if any(
        first.timestamp >= second.timestamp for first, second in zip(bars, bars[1:], strict=False)
    ):
        raise ValueError("market bars must have unique increasing timestamps")
    high = low = None
    result = []
    for index, bar in enumerate(bars):
        high = _confirmed_pivot(bars, index, pivot_width, "high") or high
        low = _confirmed_pivot(bars, index, pivot_width, "low") or low
        history = bars[max(0, index - lookback + 1) : index + 1]
        trend, volatility = _trailing_metrics(history, lookback)
        previous = bars[index - 1].close if index else bar.close
        result.append(
            StructureObservation(
                bar.timestamp,
                trend,
                volatility,
                _break(previous, bar.close, high, low),
                _retracement(bar.close, high, low),
                high,
                low,
            )
        )
    return tuple(result)


def _confirmed_pivot(bars: Sequence[MarketBar], index: int, width: int, kind: str) -> Pivot | None:
    if index < 2 * width:
        return None
    window = bars[index - 2 * width : index + 1]
    values = [getattr(bar, kind) for bar in window]
    value = max(values) if kind == "high" else min(values)
    if values[width] != value or values.count(value) != 1:
        return None
    return Pivot(window[width].timestamp, bars[index].timestamp, value, kind)


def _trailing_metrics(
    history: Sequence[MarketBar], lookback: int
) -> tuple[float | None, float | None]:
    if len(history) < lookback:
        return None, None
    closes = [bar.close for bar in history]
    returns = [math.log(second / first) for first, second in zip(closes, closes[1:], strict=False)]
    return closes[-1] / statistics.mean(closes) - 1, statistics.stdev(returns)


def _break(previous: float, current: float, high: Pivot | None, low: Pivot | None) -> int:
    if high is not None and previous <= high.price < current:
        return 1
    if low is not None and previous >= low.price > current:
        return -1
    return 0


def _retracement(price: float, high: Pivot | None, low: Pivot | None) -> float | None:
    if high is None or low is None or high.price <= low.price:
        return None
    distance = high.price - low.price
    return (
        (high.price - price) if high.anchor_at > low.anchor_at else (price - low.price)
    ) / distance


def matched_pseudo_entries(
    observations: Sequence[StructureObservation],
    entries: Sequence[int],
    *,
    exposure_states: Sequence[str | None],
    seed: int,
    horizon: int,
    embargo: int,
    controls_per_entry: int = 10,
    entry_exposure_states: Mapping[int, str | None] | None = None,
    excluded_entries: Sequence[int] = (),
) -> dict[int, tuple[int, ...]]:
    if horizon < 1 or embargo < 0 or controls_per_entry < 1:
        raise ValueError("pseudo-entry study bounds are invalid")
    if len(exposure_states) != len(observations) or any(
        value not in {None, "flat", "long", "short", "hedged"} for value in exposure_states
    ):
        raise ValueError("one known or explicitly unresolved exposure state is required per bar")
    overrides = dict(entry_exposure_states or {})
    if any(
        key not in entries or value not in {None, "flat", "long", "short", "hedged"}
        for key, value in overrides.items()
    ):
        raise ValueError("entry exposure overrides must identify declared entries and states")
    excluded = {
        index
        for entry in {*entries, *excluded_entries}
        for index in range(entry - embargo, entry + embargo + 1)
    }
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations[:-horizon]):
        if (
            index not in excluded
            and observation.volatility is not None
            and exposure_states[index] is not None
        ):
            buckets[(_match_key(observation), exposure_states[index])].append(index)
    rng = random.Random(seed)
    return {
        entry: tuple(rng.sample(candidates, min(controls_per_entry, len(candidates))))
        for entry in entries
        if 0 <= entry < len(observations) - horizon
        for candidates in (
            buckets.get(
                (_match_key(observations[entry]), overrides.get(entry, exposure_states[entry])), []
            ),
        )
    }


def _match_key(observation: StructureObservation) -> tuple:
    # Fixed covariate bins are part of the declared study specification.
    volatility = observation.volatility
    bucket = (
        None if volatility is None else 0 if volatility < 0.005 else 1 if volatility < 0.02 else 2
    )
    direction = None if observation.trend is None else observation.trend > 0
    return observation.timestamp[11:13], direction, bucket
