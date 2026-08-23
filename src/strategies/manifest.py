"""Complete named strategy manifest for the unified research catalogue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyManifestEntry:
    name: str
    family: str
    evidence_type: str
    execution_contract: str = "forecast_or_target_position"

    @property
    def catalogue(self) -> str:
        if self.family == "meta_strategy":
            return "portfolio_meta"
        if self.family == "execution":
            return "execution_policy"
        return "predictive_alpha"

    @property
    def input_contract(self) -> str:
        return {
            "cross_sectional": "point_in_time_instrument_panel",
            "relative_value": "linked_instrument_panel",
            "microstructure": "event_and_order_book_state",
            "machine_learning": "frozen_model_feature_manifest",
            "meta_strategy": "alpha_forecast_collection",
            "execution": "target_delta_and_market_state",
        }.get(self.family, "single_instrument_time_series")

    @property
    def output_contract(self) -> str:
        return {
            "cross_sectional": "ranked_forecasts_or_portfolio_targets",
            "relative_value": "hedged_multi_leg_targets",
            "meta_strategy": "aggregated_forecasts",
            "execution": "order_intents",
        }.get(self.family, "alpha_forecast_or_target_position")


_FAMILIES = {
    "time_series": (
        "sma_cross",
        "macd_trend",
        "supertrend",
        "adx_trend",
        "momentum_roc",
        "donchian_breakout",
        "keltner_breakout",
        "atr_channel_breakout",
        "bollinger_squeeze",
        "multi_tf_trend",
        "regression_channel",
        "swing_structure",
        "btc_cycle_guard",
    ),
    "mean_reversion": (
        "rsi_reversion",
        "bollinger_reversion",
        "zscore_reversion",
        "stochastic_reversion",
        "candlestick_reversal",
        "rsi_divergence",
        "fear_greed_contrarian",
        "condition_grid",
        "regime_filter",
    ),
    "cross_sectional": (
        "relative_momentum",
        "funding_adjusted_ranking",
    ),
    "relative_value": (
        "spot_perpetual_basis",
        "beta_neutral_spreads",
    ),
    "microstructure": (
        "bid_ask_depth_imbalance",
        "microprice_displacement",
    ),
    "machine_learning": ("frozen_linear_model",),
    "meta_strategy": ("correlation_aware_ensemble",),
    "execution": ("market_execution",),
}

_EVIDENCE = {
    "time_series": "swing",
    "mean_reversion": "intraday",
    "cross_sectional": "cross_sectional",
    "relative_value": "pairs",
    "microstructure": "scalping",
    "machine_learning": "ml",
    "meta_strategy": "intraday",
    "execution": "market_making",
}

REQUIRED_STRATEGY_UNIVERSE = tuple(name for family in _FAMILIES.values() for name in family)


def strategy_manifest() -> tuple[StrategyManifestEntry, ...]:
    return tuple(
        StrategyManifestEntry(name, family, _EVIDENCE[family])
        for family, names in _FAMILIES.items()
        for name in names
    )


def predictive_alpha_manifest() -> tuple[StrategyManifestEntry, ...]:
    return tuple(entry for entry in strategy_manifest() if entry.catalogue == "predictive_alpha")


def portfolio_meta_manifest() -> tuple[StrategyManifestEntry, ...]:
    return tuple(entry for entry in strategy_manifest() if entry.catalogue == "portfolio_meta")


def execution_policy_manifest() -> tuple[StrategyManifestEntry, ...]:
    return tuple(entry for entry in strategy_manifest() if entry.catalogue == "execution_policy")


def manifest_by_name() -> dict[str, StrategyManifestEntry]:
    return {entry.name: entry for entry in strategy_manifest()}


def assert_manifest_complete() -> None:
    entries = strategy_manifest()
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("strategy manifest contains duplicate names")
    if set(names) != set(REQUIRED_STRATEGY_UNIVERSE):
        raise ValueError("strategy manifest does not cover the declared strategy universe")
    from src.strategies.frozen_model import FrozenLinearModel
    from src.strategies.registry import available
    from src.strategies.semantic import SEMANTIC_STRATEGIES

    ordinary = set(_FAMILIES["time_series"]) | set(_FAMILIES["mean_reversion"])
    missing_ordinary = ordinary - set(available())
    if missing_ordinary:
        raise ValueError(
            "declared strategies have no concrete implementation: "
            + ", ".join(sorted(missing_ordinary))
        )
    for family in ("cross_sectional", "relative_value", "microstructure", "meta_strategy", "execution"):
        for name in _FAMILIES[family]:
            SEMANTIC_STRATEGIES.get(name)
    if not callable(FrozenLinearModel.load):
        raise ValueError("frozen machine-learning implementation is absent")


def manifest_description(name: str) -> str:
    entry = manifest_by_name().get(name)
    if entry is None:
        return "Registered strategy outside the target universe."
    return f"{entry.family} strategy using the {entry.execution_contract} contract."


def manifest_source_type(name: str) -> str:
    family = manifest_by_name().get(name)
    if family is None:
        return "registered_python"
    return {
        "cross_sectional": "cross_sectional",
        "relative_value": "relative_value",
        "microstructure": "microstructure",
        "machine_learning": "machine_learning",
        "meta_strategy": "ensemble",
        "execution": "registered_python",
    }.get(family.family, "registered_python")
