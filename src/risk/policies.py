"""Compile split configuration into immutable six-scope risk-policy records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.risk.engine import SqlRiskPolicyStore
from src.risk.policy import risk_policy_identity


def install_product_risk_policies(
    store: SqlRiskPolicyStore,
    *,
    risk_configuration: Mapping[str, Any],
    products: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    installed: dict[str, str] = {}
    for product_id, product in products.items():
        policy_id = str(product["risk_policy_id"])
        account_id = str(product["account_id"])
        record_id = risk_policy_identity(policy_id, risk_configuration, account_id=account_id)
        store.save(
            record_id,
            compile_risk_limits(risk_configuration, policy_id, account_id),
            contract={
                "schema": "platform.risk_policy/v2",
                "logical_policy_id": policy_id,
                "account_id": account_id,
                "configuration": dict(risk_configuration),
            },
        )
        installed[product_id] = record_id
    return installed


def compile_risk_limits(
    risk_configuration: Mapping[str, Any], policy_id: str, account_id: str
) -> dict[str, dict]:
    instrument = risk_configuration["instrument"]
    strategy = risk_configuration["strategy"]
    sleeve = risk_configuration["sleeve"]
    global_limits = risk_configuration["global"]
    product_limits = risk_configuration["products"][policy_id]
    account = risk_configuration["accounts"][account_id]
    exposure = float(product_limits.get("maximum_exposure", 1.0))
    return {
        "strategy": {
            "max_position_fraction": float(strategy["maximum_fraction"]),
            "max_turnover_fraction": float(strategy["maximum_turnover_fraction"]),
            "max_trades_per_day": int(strategy["maximum_trades_per_day"]),
            "max_slippage_bps": float(strategy["maximum_slippage_bps"]),
            "max_funding_cost_fraction": float(strategy["maximum_funding_cost_fraction"]),
        },
        "instrument": {
            "max_position_notional": float(instrument["maximum_position_notional"]),
            "max_order_notional": float(instrument["maximum_order_notional"]),
            "max_visible_depth_fraction": float(instrument["maximum_visible_depth_fraction"]),
            "max_spread_bps": float(instrument["maximum_slippage_bps"]),
            "max_volatility": float(instrument["maximum_volatility"]),
            "max_concentration_fraction": float(instrument["maximum_fraction"]),
        },
        "sleeve": {
            "max_capital_fraction": float(sleeve["maximum_fraction"]),
            "max_drawdown_fraction": float(sleeve["maximum_drawdown"]),
            "max_correlation": float(sleeve["maximum_correlation"]),
            "max_abs_beta": float(sleeve["maximum_abs_beta"]),
            "max_turnover_fraction": float(sleeve["maximum_turnover_fraction"]),
        },
        "product": {
            "max_gross_fraction": float(product_limits.get("maximum_gross", exposure)),
            "max_net_fraction": float(product_limits.get("maximum_net", exposure)),
            "max_drawdown_fraction": float(product_limits["maximum_drawdown"]),
            "max_margin_fraction": float(product_limits.get("maximum_margin_fraction", 1.0)),
            "max_daily_loss_fraction": float(product_limits["maximum_daily_loss"]),
        },
        "account": {
            "max_used_margin_fraction": float(account["maximum_margin_fraction"]),
            "min_liquidation_buffer_fraction": float(account["minimum_liquidation_buffer"]),
            "reject_unknown_exposure": bool(account["reject_unknown_exposure"]),
        },
        "global": {
            "max_drawdown_fraction": float(global_limits["maximum_drawdown"]),
            "max_data_age_seconds": float(global_limits["maximum_market_data_staleness_seconds"]),
            "max_clock_skew_seconds": float(global_limits["maximum_clock_skew_seconds"]),
        },
    }
