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
        if self.family == "market_making":
            return "market_making"
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
            "market_making": "quote_state",
        }.get(self.family, "single_instrument_time_series")

    @property
    def output_contract(self) -> str:
        return {
            "cross_sectional": "ranked_forecasts_or_portfolio_targets",
            "relative_value": "hedged_multi_leg_targets",
            "meta_strategy": "aggregated_forecasts",
            "execution": "order_intents",
            "market_making": "quote_targets",
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
        "pairs_trading",
    ),
    "microstructure": (
        "bid_ask_depth_imbalance",
        "microprice_displacement",
        "aggressor_flow_scalping",
        "liquidation_burst_scalping",
        "liquidity_vacuum_scalping",
    ),
    "machine_learning": (
        "frozen_linear_model",
        "frozen_logistic_model",
        "lightgbm_classifier",
        "lightgbm_regressor",
        "cross_sectional_ranker",
        "triple_barrier_classifier",
        "return_regressor",
        "meta_labelling",
        "probability_calibration",
        "regime_classification",
    ),
    "advanced_alpha": (
        "seasonality",
        "volatility_forecast",
        "statistical_pattern_recognition",
        "point_in_time_sentiment",
    ),
    "meta_strategy": (
        "correlation_aware_ensemble",
        "regime_routing",
        "performance_decay_weighting",
        "forecast_conflict_suppression",
        "drift_based_capital_reduction",
        "sleeve_reallocation",
    ),
    "execution": (
        "market_execution",
        "limit_execution",
        "post_only_execution",
        "twap",
        "vwap",
        "percentage_of_volume",
        "adaptive_urgency",
        "spread_aware_selection",
        "depth_aware_sizing",
        "cancel_replace",
        "binance_spot_sor",
    ),
    "market_making": ("inventory_aware_market_making",),
}

_EVIDENCE = {
    "time_series": "swing",
    "mean_reversion": "intraday",
    "cross_sectional": "cross_sectional",
    "relative_value": "pairs",
    "microstructure": "scalping",
    "machine_learning": "ml",
    "advanced_alpha": "swing",
    "meta_strategy": "intraday",
    "execution": "market_making",
    "market_making": "market_making",
}

REQUIRED_STRATEGY_UNIVERSE = tuple(name for family in _FAMILIES.values() for name in family)

_REGISTERED_FEATURE_NODES: dict[str, tuple[str, ...]] = {
    "sma_cross": ("sma_fast", "sma_slow"),
    "macd_trend": ("macd",),
    "supertrend": ("supertrend",),
    "adx_trend": ("adx",),
    "momentum_roc": ("bar_return",),
    "donchian_breakout": ("breakout",),
    "keltner_breakout": ("keltner",),
    "atr_channel_breakout": ("supertrend",),
    "bollinger_squeeze": ("bollinger",),
    "multi_tf_trend": ("multi_timeframe",),
    "regression_channel": ("bar_return",),
    "swing_structure": ("breakout",),
    "btc_cycle_guard": ("realised_volatility",),
    "rsi_reversion": ("rsi",),
    "bollinger_reversion": ("bollinger",),
    "zscore_reversion": ("bollinger",),
    "stochastic_reversion": ("rsi",),
    "candlestick_reversal": ("bar_return",),
    "rsi_divergence": ("rsi",),
    "fear_greed_contrarian": ("sentiment",),
    "condition_grid": ("rsi", "adx"),
    "regime_filter": ("bar_return",),
}


def registered_feature_contract(name: str) -> tuple[tuple[str, ...], dict[str, object]]:
    try:
        nodes = _REGISTERED_FEATURE_NODES[name]
    except KeyError as exc:
        raise ValueError(f"registered strategy has no live feature contract: {name}") from exc
    return nodes, {"kind": "registered_strategy/v1", "registered_strategy": name}


def registered_live_contract(name: str) -> tuple[tuple[str, ...], dict[str, object]]:
    """Compatibility name for the feature contract, without a second rule engine."""

    return registered_feature_contract(name)


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


def market_making_manifest() -> tuple[StrategyManifestEntry, ...]:
    return tuple(entry for entry in strategy_manifest() if entry.catalogue == "market_making")


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
    missing_live_contracts = ordinary - set(_REGISTERED_FEATURE_NODES)
    if missing_live_contracts:
        raise ValueError(
            "declared registered strategies have no live contract: "
            + ", ".join(sorted(missing_live_contracts))
        )
    semantic_names = {
        "relative_momentum",
        "funding_adjusted_ranking",
        "spot_perpetual_basis",
        "beta_neutral_spreads",
        "bid_ask_depth_imbalance",
        "microprice_displacement",
        "correlation_aware_ensemble",
        "market_execution",
    }
    for family in (
        "cross_sectional",
        "relative_value",
        "microstructure",
        "meta_strategy",
        "execution",
    ):
        for name in _FAMILIES[family]:
            if name in semantic_names:
                SEMANTIC_STRATEGIES.get(name)
    from src.strategies.advanced import ADVANCED_IMPLEMENTATIONS

    missing_advanced = (
        set(names)
        - ordinary
        - semantic_names
        - set(_FAMILIES["machine_learning"])
        - set(ADVANCED_IMPLEMENTATIONS)
    )
    if missing_advanced:
        raise ValueError(
            "declared advanced strategies have no implementation: "
            + ", ".join(sorted(missing_advanced))
        )
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
        "advanced_alpha": "registered_python",
        "market_making": "microstructure",
    }.get(family.family, "registered_python")
