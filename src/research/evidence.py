"""Pure evidence rules shared by research and promotion policy code."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    measured = float(value)
    return measured if math.isfinite(measured) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


_PROFILE_DIMENSIONS = ("product_id", "family", "horizon", "stage")
_PROFILE_FINITE_FIELDS = (
    "minimum_cost_adjusted_return",
    "minimum_deflated_sharpe",
    "minimum_walk_forward_pass_fraction",
    "maximum_backtest_overfitting_probability",
    "minimum_calendar_days",
    "maximum_drawdown",
    "maximum_tail_loss",
    "maximum_parameter_degradation",
    "minimum_positive_symbol_fraction",
    "minimum_cross_symbol_median_return",
    "minimum_cross_symbol_pooled_return",
    "minimum_cross_symbol_lower_quantile_return",
    "allowed_holdout_degradation",
)
_PROFILE_INTEGER_FIELDS = (
    "minimum_walk_forward_windows",
    "minimum_bootstrap_observations",
    "minimum_closed_trades",
    "minimum_effective_episodes",
    "minimum_trading_days",
    "minimum_cycles",
)
_PROFILE_UNIT_FIELDS = (
    "minimum_walk_forward_pass_fraction",
    "maximum_backtest_overfitting_probability",
    "minimum_positive_symbol_fraction",
    "maximum_parameter_degradation",
    "allowed_holdout_degradation",
)


def _normalise_profile_dimensions(profile: EvidenceProfile) -> None:
    for name in _PROFILE_DIMENSIONS:
        value = str(getattr(profile, name)).strip()
        if not value:
            raise ValueError(f"evidence profile {name} cannot be empty")
        object.__setattr__(profile, name, value)


def _validate_profile_finite_fields(profile: EvidenceProfile) -> None:
    for name in _PROFILE_FINITE_FIELDS:
        value = getattr(profile, name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"evidence profile {name} must be finite")


def _validate_profile_integer_fields(profile: EvidenceProfile) -> None:
    for name in _PROFILE_INTEGER_FIELDS:
        value = getattr(profile, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"evidence profile {name} must be a non-negative integer")


def _validate_profile_unit_fields(profile: EvidenceProfile) -> None:
    for name in _PROFILE_UNIT_FIELDS:
        value = getattr(profile, name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evidence profile {name} must be between zero and one")


def _validate_profile_thresholds(profile: EvidenceProfile) -> None:
    if profile.minimum_calendar_days < 0:
        raise ValueError("evidence profile minimum_calendar_days must be non-negative")
    if (
        profile.minimum_cost_adjusted_return is not None
        and profile.minimum_cost_adjusted_return < 0
    ):
        raise ValueError("evidence profile minimum_cost_adjusted_return must be non-negative")
    if profile.minimum_deflated_sharpe is not None and profile.minimum_deflated_sharpe < 0:
        raise ValueError("evidence profile minimum_deflated_sharpe must be non-negative")
    if (
        profile.minimum_bootstrap_observations is not None
        and profile.minimum_bootstrap_observations < 1
    ):
        raise ValueError("evidence profile minimum_bootstrap_observations must be positive")
    for name in ("maximum_drawdown", "maximum_tail_loss"):
        value = getattr(profile, name)
        if value is not None and value < 0:
            raise ValueError(f"evidence profile {name} must be non-negative")


@dataclass(frozen=True)
class EvidenceProfile:
    """One product, family, and horizon evidence policy.

    A ``*`` dimension is a deliberate wildcard.  Profiles are selected by
    specificity, so an exact product/family/horizon profile always wins over
    a broader default.
    """

    product_id: str = "*"
    family: str = "*"
    horizon: str = "*"
    stage: str = "*"
    minimum_cost_adjusted_return: float | None = None
    minimum_deflated_sharpe: float | None = None
    minimum_walk_forward_windows: int | None = None
    minimum_walk_forward_pass_fraction: float | None = None
    maximum_backtest_overfitting_probability: float | None = None
    minimum_bootstrap_observations: int | None = None
    minimum_closed_trades: int = 0
    minimum_effective_episodes: int = 0
    minimum_trading_days: int = 0
    minimum_cycles: int = 0
    minimum_calendar_days: float = 0.0
    maximum_drawdown: float | None = None
    maximum_tail_loss: float | None = None
    maximum_parameter_degradation: float = 0.5
    minimum_positive_symbol_fraction: float = 0.5
    minimum_cross_symbol_median_return: float = 0.0
    minimum_cross_symbol_pooled_return: float = 0.0
    minimum_cross_symbol_lower_quantile_return: float | None = None
    allowed_holdout_degradation: float = 0.5

    def __post_init__(self) -> None:
        _normalise_profile_dimensions(self)
        _validate_profile_finite_fields(self)
        _validate_profile_integer_fields(self)
        _validate_profile_unit_fields(self)
        _validate_profile_thresholds(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvidenceProfile:
        if not isinstance(value, Mapping):
            raise ValueError("evidence profile must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("evidence profile contains unsupported fields: " + ", ".join(unknown))
        return cls(**{str(key): value[key] for key in value})

    def to_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def matches(
        self, *, stage: str, product_id: str | None, family: str | None, horizon: str | None
    ) -> bool:
        return all(
            expected == "*" or expected == actual
            for expected, actual in (
                (self.stage, stage),
                (self.product_id, product_id),
                (self.family, family),
                (self.horizon, horizon),
            )
        )

    @property
    def specificity(self) -> int:
        return sum(
            value != "*" for value in (self.stage, self.product_id, self.family, self.horizon)
        )


def initial_forward_profile(evidence_type: str | None) -> EvidenceProfile:
    """Return the initial live-readiness evidence policy for one horizon family."""

    values: dict[str, tuple[float, int, int, int]] = {
        "scalping": (14.0, 200, 200, 10),
        "intraday": (30.0, 60, 60, 0),
        "swing": (60.0, 20, 20, 0),
        "btc_allocation": (90.0, 0, 0, 8),
    }
    days, trades, episodes, cycles = values.get(str(evidence_type or ""), (0.0, 0, 0, 0))
    return EvidenceProfile(
        stage="forward",
        family=str(evidence_type or "*"),
        minimum_calendar_days=days,
        minimum_closed_trades=trades,
        minimum_effective_episodes=episodes,
        minimum_trading_days=10 if evidence_type == "scalping" else 0,
        minimum_cycles=cycles,
    )


def select_profile(
    profiles: Sequence[EvidenceProfile],
    *,
    stage: str,
    product_id: str | None,
    family: str | None,
    horizon: str | None,
) -> EvidenceProfile | None:
    matches = [
        profile
        for profile in profiles
        if profile.matches(stage=stage, product_id=product_id, family=family, horizon=horizon)
    ]
    return max(
        matches,
        key=lambda profile: (profile.specificity, profile.to_payload().__repr__()),
        default=None,
    )


def sample_evidence_passes(value: object, profile: EvidenceProfile) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    observations = _integer(value.get("observations"))
    closed_trades = _integer(
        value.get("closed_trades", value.get("trades", value.get("effective_trades", observations)))
    )
    episodes = _integer(
        value.get("effective_independent_episodes", value.get("episodes", closed_trades))
    )
    trading_days = _integer(value.get("trading_days", value.get("evidence_days", 0)))
    return (
        observations is not None
        and observations > 0
        and closed_trades is not None
        and closed_trades >= profile.minimum_closed_trades
        and episodes is not None
        and episodes >= profile.minimum_effective_episodes
        and trading_days is not None
        and trading_days >= profile.minimum_trading_days
    )


def parameter_stability_passes(value: object, profile: EvidenceProfile) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("status") == "not_applicable":
        return True
    if value.get("cliff_detected") is True or value.get("degradation_shape") == "cliff":
        return False
    parameter_data = _parameter_data(value)
    if parameter_data is None:
        return False
    results, returns = parameter_data
    base = _finite(value.get("base_return", value.get("base_cost_adjusted_return")))
    median = _finite(value.get("median_return"))
    worst = _finite(value.get("worst_return"))
    if median is None:
        median = statistics.median(item for item in returns if item is not None)
    if worst is None:
        worst = min(item for item in returns if item is not None)
    if base is not None:
        degradation = max(0.0, (base - worst) / max(abs(base), 1e-12))
        allowed = _finite(value.get("maximum_degradation"))
        if allowed is None:
            allowed = profile.maximum_parameter_degradation
        if degradation > allowed + 1e-12:
            return False
    passed_count = sum(
        1 for item in results if isinstance(item, Mapping) and item.get("passed") is True
    )
    return (
        median
        >= (base * (1.0 - profile.maximum_parameter_degradation) if base is not None else 0.0)
        and passed_count >= (len(results) + 1) // 2
    )


def _parameter_data(
    value: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[float]] | None:
    results = value.get("results")
    tested = _integer(value.get("neighbours_tested"))
    if (
        not isinstance(results, list | tuple)
        or tested is None
        or tested < 2
        or len(results) != tested
    ):
        return None
    result_items = [item for item in results if isinstance(item, Mapping)]
    returns = [_finite(item.get("return")) for item in result_items]
    if len(returns) != len(results) or any(item is None for item in returns):
        return None
    if not all(
        isinstance(item, Mapping)
        and isinstance(item.get("observations"), int)
        and int(item["observations"]) > 0
        and _valid_hash(item.get("run_id"))
        and _valid_hash(item.get("input_hash"))
        for item in results
    ):
        return None
    return result_items, [item for item in returns if item is not None]


def cross_symbol_stability_passes(value: object, profile: EvidenceProfile) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("status") == "not_applicable":
        return True
    returns = _cross_symbol_returns(value)
    if returns is None:
        return False
    median = _finite(value.get("median_return"))
    pooled = _finite(value.get("pooled_return"))
    lower = _finite(value.get("lower_quantile_return"))
    if median is None:
        median = statistics.median(returns)
    if pooled is None:
        pooled = sum(returns) / len(returns)
    if lower is None:
        lower = _quantile(returns, 0.1)
    positive_fraction = sum(item >= 0 for item in returns) / len(returns)
    declared_fraction = _finite(value.get("positive_symbol_fraction"))
    if declared_fraction is not None:
        positive_fraction = declared_fraction
    minimum_lower = profile.minimum_cross_symbol_lower_quantile_return
    return (
        positive_fraction >= profile.minimum_positive_symbol_fraction
        and median >= profile.minimum_cross_symbol_median_return
        and pooled >= profile.minimum_cross_symbol_pooled_return
        and (minimum_lower is None or lower >= minimum_lower)
    )


def _cross_symbol_returns(value: Mapping[str, Any]) -> list[float] | None:
    per_symbol = value.get("per_symbol")
    symbols = _integer(value.get("symbols"))
    if (
        not isinstance(per_symbol, Mapping)
        or symbols is None
        or symbols <= 0
        or len(per_symbol) != symbols
    ):
        return None
    returns: list[float] = []
    for item in per_symbol.values():
        if not isinstance(item, Mapping):
            return None
        measured = _finite(item.get("return"))
        if (
            measured is None
            or not isinstance(item.get("observations"), int)
            or int(item["observations"]) < 2
        ):
            return None
        if not _valid_hash(item.get("run_id")) or not _valid_hash(item.get("input_hash")):
            return None
        returns.append(measured)
    return returns


def drawdown_passes(value: object, profile: EvidenceProfile) -> bool:
    if not isinstance(value, Mapping):
        return False
    maximum = _finite(value.get("maximum_drawdown"))
    if maximum is None or maximum < 0:
        return False
    limit = profile.maximum_drawdown
    if limit is not None and maximum > limit + 1e-12:
        return False
    tail = _finite(value.get("tail_loss"))
    tail_limit = profile.maximum_tail_loss
    if tail_limit is not None and (tail is None or tail > tail_limit + 1e-12):
        return False
    return value.get("passed") is True


def monte_carlo_passes(value: object, profile: EvidenceProfile) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    iterations = _integer(value.get("iterations"))
    maximum = _finite(value.get("maximum_drawdown"))
    tail = _finite(value.get("tail_loss"))
    if iterations is None or iterations < 1 or maximum is None:
        return False
    if profile.maximum_drawdown is not None and maximum > profile.maximum_drawdown + 1e-12:
        return False
    return profile.maximum_tail_loss is None or (
        tail is not None and tail <= profile.maximum_tail_loss + 1e-12
    )


def holdout_degradation_passes(
    development: Mapping[str, Any], protected: Mapping[str, Any], profile: EvidenceProfile
) -> bool:
    development_value = _objective_or_return(development)
    protected_value = _objective_or_return(protected)
    if development_value is None or protected_value is None:
        return False
    degradation = max(
        0.0,
        (development_value - protected_value) / max(abs(development_value), 1e-12),
    )
    return degradation <= profile.allowed_holdout_degradation + 1e-12


def data_integrity_passes(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    snapshots = value.get("dataset_snapshot_ids")
    return (
        isinstance(snapshots, list | tuple)
        and bool(snapshots)
        and all(_valid_hash(item) for item in snapshots)
        and _valid_hash(value.get("input_hash"))
    )


def semantic_parity_passes(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    source_identity = value.get("behaviour_hash")
    receipt = value.get("parity_receipt")
    if not _valid_hash(source_identity) or not isinstance(receipt, Mapping):
        return False
    saved_hash = receipt.get("receipt_hash")
    content = dict(receipt)
    content.pop("receipt_hash", None)
    input_hash = receipt.get("input_hash")
    receipt_behaviour = receipt.get("behaviour_hash")
    return (
        _valid_hash(saved_hash)
        and canonical_hash(content) == saved_hash
        and _valid_hash(input_hash)
        and (receipt_behaviour is None or receipt_behaviour == source_identity)
    )


def realistic_costs_passes(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    fee = _finite(value.get("fee_bps"))
    slippage = _finite(value.get("slippage_bps"))
    funding = _finite(value.get("funding_rate"))
    return (
        fee is not None
        and fee >= 0.0
        and slippage is not None
        and slippage >= 0.0
        and funding is not None
    )


def family_evidence_passes(value: object, family: str | None = None) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    actual = str(value.get("family") or "")
    return bool(actual) and (family is None or actual == family)


def regime_breakdown_passes(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    regimes = value.get("regimes")
    return (
        isinstance(regimes, Mapping)
        and bool(regimes)
        and all(_finite(measured) is not None for measured in regimes.values())
    )


def _objective_or_return(value: Mapping[str, Any]) -> float | None:
    for name in (
        "objective_excess_fraction",
        "objective_excess",
        "cost_adjusted_return",
        "net_pnl",
    ):
        measured = _finite(value.get(name))
        if measured is not None:
            return measured
    return None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _valid_hash(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True
