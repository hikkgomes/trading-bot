"""Generate paper artifacts for execution-path diagnostics.

These artifacts are not research results. They exist to let a fresh server
validate artifact loading and, for legacy open positions, exercise exit
management while the research loop searches for validated edges. The runtime
blocks them from opening new positions and their metadata blocks promotion/live
trading.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.io import write_json_atomic

SCHEMA_VERSION = 1


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _fees() -> dict[str, float]:
    return {"fee_bps": 5.0, "slippage_bps": 2.0}


def _paper_only_payload(product: ProductConfig, strategies: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "schema": "autopilot.paper_bootstrap/v1",
        "generated_at": _now(),
        "source": "paper_bootstrap",
        "entry_policy": "management_only",
        "executable": False,
        "paper_trade_allowed": False,
        "live_allowed": False,
        "promotion_eligible": False,
        "market": product.market,
        "symbol": product.symbol,
        "pnl_unit": product.base_asset.lower(),
        "strategies": strategies,
    }


def _btc_accumulation_strategies(product: ProductConfig) -> list[dict[str, Any]]:
    return [
        {
            "id": "btc_bootstrap_step_aside_rsi_4h",
            "rank": 1,
            "market": product.market,
            "symbol": product.symbol,
            "base_timeframe": "4h",
            "direction": "short",
            "horizon_bars": 6,
            "take_profit": 0.015,
            "stop_loss": 0.01,
            "use_atr_tp_sl": False,
            "pnl_unit": "btc",
            "conditions": [
                {
                    "feature": "tf_4h_rsi_14",
                    "kind": "value_le",
                    "threshold": 45.0,
                    "description": "tf_4h_rsi_14 <= 45.0",
                }
            ],
            "rule": "paper-only BTC step-aside probe when 4h RSI is weak",
            "risk": {
                "risk_per_trade": 0.001,
                "max_position_fraction": 0.10,
                "daily_stop_loss": -0.005,
                "max_consecutive_losses": 2,
                "cooldown_bars": 48,
                "max_trades_per_day": 1,
            },
            "fees": _fees(),
            "metrics": {},
        }
    ]


def _active_income_strategies(product: ProductConfig) -> list[dict[str, Any]]:
    common_risk = {
        "risk_per_trade": 0.001,
        "max_position_fraction": 0.05,
        "daily_stop_loss": -0.01,
        "max_consecutive_losses": 2,
        "cooldown_bars": 24,
        "max_trades_per_day": 3,
    }
    return [
        {
            "id": "active_bootstrap_long_rsi_5m",
            "rank": 1,
            "market": product.market,
            "symbol": product.symbol,
            "base_timeframe": "5m",
            "direction": "long",
            "horizon_bars": 6,
            "take_profit": 0.008,
            "stop_loss": 0.004,
            "use_atr_tp_sl": False,
            "pnl_unit": "usdt",
            "conditions": [
                {
                    "feature": "tf_5m_rsi_14",
                    "kind": "value_ge",
                    "threshold": 55.0,
                    "description": "tf_5m_rsi_14 >= 55.0",
                }
            ],
            "rule": "paper-only long momentum probe when 5m RSI is firm",
            "risk": dict(common_risk),
            "fees": _fees(),
            "metrics": {},
        },
        {
            "id": "active_bootstrap_short_rsi_5m",
            "rank": 2,
            "market": product.market,
            "symbol": product.symbol,
            "base_timeframe": "5m",
            "direction": "short",
            "horizon_bars": 6,
            "take_profit": 0.008,
            "stop_loss": 0.004,
            "use_atr_tp_sl": False,
            "pnl_unit": "usdt",
            "conditions": [
                {
                    "feature": "tf_5m_rsi_14",
                    "kind": "value_le",
                    "threshold": 45.0,
                    "description": "tf_5m_rsi_14 <= 45.0",
                }
            ],
            "rule": "paper-only short momentum probe when 5m RSI is weak",
            "risk": dict(common_risk),
            "fees": _fees(),
            "metrics": {},
        },
    ]


def build_bootstrap_artifact(product: ProductConfig) -> dict[str, Any]:
    if product.objective == "btc_accumulation":
        strategies = _btc_accumulation_strategies(product)
    elif product.objective == "active_income":
        strategies = _active_income_strategies(product)
    else:
        raise ValueError(f"{product.name}: unsupported objective {product.objective!r}")
    return _paper_only_payload(product, strategies)


def write_bootstrap_artifacts(
    config: AutopilotConfig,
    *,
    product_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    products = [
        product
        for product in config.products
        if product.enabled and product.execution_mode == "paper"
    ]
    if product_name:
        products = [product for product in products if product.name == product_name]
    if product_name and not products:
        raise ValueError(f"No enabled paper product named {product_name!r}.")

    rows: list[dict[str, Any]] = []
    for product in products:
        path = product.strategies_path
        if path.exists() and not overwrite:
            rows.append(
                {
                    "product": product.name,
                    "path": str(path),
                    "ok": True,
                    "action": "skipped_existing",
                }
            )
            continue
        artifact = build_bootstrap_artifact(product)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, artifact)
        rows.append(
            {
                "product": product.name,
                "path": str(path),
                "ok": True,
                "action": "written",
                "entry_policy": "management_only",
                "strategies": len(artifact["strategies"]),
            }
        )
    return {"generated_at": _now(), "ok": all(row["ok"] for row in rows), "artifacts": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper-only bootstrap strategy artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--product", help="Optional product name. Defaults to all enabled paper products."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing configured strategy artifacts."
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_bootstrap_artifacts(
        load_config(args.config),
        product_name=args.product,
        overwrite=args.overwrite,
    )
    if args.report:
        write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
