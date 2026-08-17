"""Complete named strategy manifest for the unified research catalogue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyManifestEntry:
    name: str
    family: str
    evidence_type: str
    execution_contract: str = "forecast_or_target_position"


_FAMILIES = {
    "time_series": (
        "moving_average_trend",
        "macd_trend",
        "supertrend",
        "adx_trend",
        "time_series_momentum",
        "rate_of_change_momentum",
        "donchian_breakout",
        "keltner_breakout",
        "atr_breakout",
        "bollinger_squeeze",
        "volatility_breakout",
        "range_breakout",
        "channel_trading",
        "regression_channels",
        "market_structure_breakout",
        "trend_pullback",
        "volatility_scaled_trend",
    ),
    "mean_reversion": (
        "rsi_reversion",
        "bollinger_reversion",
        "z_score_reversion",
        "stochastic_reversion",
        "vwap_reversion",
        "residual_reversion",
        "session_reversion",
        "volatility_conditioned_reversion",
        "order_flow_exhaustion",
        "post_liquidation_reversion",
    ),
    "cross_sectional": (
        "relative_momentum",
        "relative_reversal",
        "volatility_adjusted_ranking",
        "liquidity_adjusted_ranking",
        "funding_adjusted_ranking",
        "btc_beta_neutral_residual_momentum",
        "sector_neutral_ranking",
        "long_short_baskets",
        "risk_parity_basket",
        "cross_sectional_ml_ranking",
    ),
    "relative_value": (
        "cointegrated_pairs",
        "rolling_hedge_ratio_pairs",
        "pca_residual_baskets",
        "beta_neutral_spreads",
        "spot_perpetual_basis",
        "perpetual_funding_carry",
        "calendar_basis",
        "cross_symbol_relative_volatility",
        "index_constituent_residuals",
        "synthetic_basket_spreads",
    ),
    "microstructure": (
        "bid_ask_depth_imbalance",
        "microprice_displacement",
        "aggressor_trade_imbalance",
        "cancel_add_pressure",
        "spread_compression_expansion",
        "liquidity_vacuum",
        "short_horizon_continuation",
        "short_horizon_reversal",
        "liquidation_clustering",
        "mark_price_divergence",
        "order_book_resilience",
        "event_time_volatility",
    ),
    "machine_learning": (
        "logistic_regression",
        "elastic_net_classification",
        "linear_regression",
        "gradient_boosting",
        "lightgbm",
        "random_forest",
        "calibrated_classifier",
        "pairwise_ranking",
        "cross_sectional_ranking_model",
        "triple_barrier_classifier",
        "return_regressor",
        "meta_labelling",
        "regime_classification",
        "volatility_prediction",
        "dynamic_position_sizing",
        "online_learning",
        "shallow_sequence_model",
        "temporal_convolutional_network",
    ),
    "meta_strategy": (
        "regime_routing",
        "strategy_weighting",
        "bayesian_model_averaging",
        "performance_decay_weighting",
        "correlation_aware_ensemble",
        "conflict_suppression",
        "confidence_calibration",
        "dynamic_sleeve_allocation",
        "volatility_targeting",
        "drawdown_based_deallocation",
    ),
    "execution": (
        "market_execution",
        "passive_limit_execution",
        "post_only_execution",
        "time_sliced_execution",
        "volume_weighted_execution",
        "spread_aware_order_selection",
        "depth_aware_sizing",
        "cancel_replace",
        "multi_leg_hedge_execution",
        "emergency_unwind",
    ),
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


def manifest_by_name() -> dict[str, StrategyManifestEntry]:
    return {entry.name: entry for entry in strategy_manifest()}


def assert_manifest_complete() -> None:
    entries = strategy_manifest()
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("strategy manifest contains duplicate names")
    if set(names) != set(REQUIRED_STRATEGY_UNIVERSE):
        raise ValueError("strategy manifest does not cover the declared strategy universe")


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
