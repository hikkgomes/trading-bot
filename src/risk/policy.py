"""Canonical dimensions for configured risk limits; values are never guessed or rescaled."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.domain._codec import canonical_hash, non_empty

RISK_LIMIT_UNITS = {
    "global": {
        "maximum_drawdown": "equity_fraction",
        "maximum_database_staleness_seconds": "seconds",
        "maximum_market_data_staleness_seconds": "seconds",
        "maximum_clock_skew_seconds": "seconds",
    },
    "accounts": {
        "maximum_drawdown": "equity_fraction",
        "minimum_liquidation_buffer": "equity_fraction",
        "maximum_margin_fraction": "equity_fraction",
        "reject_unknown_exposure": "boolean",
    },
    "products": {
        "maximum_daily_loss": "equity_fraction",
        "maximum_exposure": "equity_fraction",
        "maximum_gross": "equity_multiple",
        "maximum_net": "equity_multiple",
        "maximum_drawdown": "equity_fraction",
        "maximum_margin_fraction": "equity_fraction",
    },
    "instrument": {
        "maximum_fraction": "equity_fraction",
        "maximum_slippage_bps": "basis_points",
        "minimum_daily_quote_volume": "quote_notional_per_day",
        "maximum_position_notional": "quote_notional",
        "maximum_order_notional": "quote_notional",
        "maximum_visible_depth_fraction": "visible_depth_fraction",
        "maximum_volatility": "return_volatility_ratio",
    },
    "strategy": {
        "maximum_fraction": "equity_fraction",
        "maximum_drawdown": "equity_fraction",
        "maximum_turnover_fraction": "equity_turnover_multiple_per_decision",
        "maximum_trades_per_day": "count_per_day",
        "maximum_slippage_bps": "basis_points",
        "maximum_funding_cost_fraction": "equity_cost_fraction_per_settlement",
    },
    "sleeve": {
        "maximum_fraction": "equity_fraction",
        "maximum_correlation": "absolute_correlation",
        "maximum_drawdown": "equity_fraction",
        "maximum_abs_beta": "absolute_beta",
        "maximum_turnover_fraction": "equity_turnover_multiple_per_decision",
    },
}


def convert_legacy_risk_policy(
    policy: Mapping[str, Any], *, notional_currency: str
) -> dict[str, Any]:
    if policy.get("schema") != "risk/v1" or set(policy) != {"schema", *RISK_LIMIT_UNITS}:
        raise ValueError("legacy risk policy shape is unknown; explicit review is required")
    converted = deepcopy(dict(policy))
    converted.update(
        schema="risk/v2", notional_currency=notional_currency, units=deepcopy(RISK_LIMIT_UNITS)
    )
    validate_risk_policy(converted)
    return converted


def validate_risk_policy(policy: Mapping[str, Any]) -> None:
    expected = {"schema", "notional_currency", "units", *RISK_LIMIT_UNITS}
    if set(policy) != expected or policy.get("schema") != "risk/v2":
        raise ValueError("risk policy requires the explicit risk/v2 unit contract")
    if policy.get("notional_currency") != "USDT":
        raise ValueError("risk policy requires the configured USDT notional currency")
    if policy.get("units") != RISK_LIMIT_UNITS:
        raise ValueError("risk policy unit declarations do not match the canonical dimensions")
    for scope, units in RISK_LIMIT_UNITS.items():
        values = policy[scope]
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"risk.{scope} must contain its declared limits")
        records = values.items() if scope in {"accounts", "products"} else ((scope, values),)
        for identity, limits in records:
            _validate_limits(scope, str(identity), limits, units)


def risk_policy_identity(
    logical_id: str, configuration: Mapping[str, Any], *, account_id: str
) -> str:
    validate_risk_policy(configuration)
    return canonical_hash(
        {
            "schema": "platform.risk_policy/v2",
            "compiler": "risk_policy_compiler/v2",
            "logical_policy_id": non_empty(logical_id, field="logical risk policy id"),
            "account_id": non_empty(account_id, field="risk account id"),
            "configuration": dict(configuration),
        }
    )


def _validate_limits(scope: str, identity: str, limits: Any, units: Mapping[str, str]) -> None:
    if not isinstance(limits, Mapping) or not limits or set(limits) - set(units):
        raise ValueError(f"risk.{scope}.{identity} has missing or unsupported limits")
    if scope == "products":
        required = {"maximum_daily_loss", "maximum_drawdown"}
        exposure_keys = set(limits) & {"maximum_gross", "maximum_exposure"}
        if not required.issubset(limits) or len(exposure_keys) != 1:
            raise ValueError(
                "product risk requires one gross exposure limit and loss/drawdown limits"
            )
    elif set(limits) != set(units):
        raise ValueError(f"risk.{scope}.{identity} has incomplete limits")
    for field, value in limits.items():
        _validate_value(value, units[field], field=f"risk.{scope}.{identity}.{field}")


def _validate_value(value: Any, unit: str, *, field: str) -> None:
    if unit == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field} must be a finite non-negative {unit} value")
    if unit in {"equity_fraction", "visible_depth_fraction", "absolute_correlation"} and value > 1:
        raise ValueError(f"{field} must not exceed one")
    if unit == "count_per_day" and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer count")
