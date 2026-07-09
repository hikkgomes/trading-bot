"""Live/testnet readiness checks.

This module never places orders. It validates the product configuration,
strategy approval, exchange environment, broker construction, and optionally
read-only exchange connectivity before a product is promoted or exercised on
testnet.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from src.autopilot.approvals import (
    ApprovalError,
    artifact_digest,
    assert_artifact_live_approved,
    load_artifact,
    strategy_fingerprint,
)
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.exchange_policy import (
    ACTIVE_INCOME_MAX_FUTURES_LEVERAGE,
    validate_exchange_policy,
)
from src.autopilot.io import write_json_atomic
from src.autopilot.reporting import utc_now
from src.autopilot.runtime import build_live_broker, validate_config
from src.autopilot.strategy_policy import StrategyPolicyError, assert_strategy_artifact_allowed
from src.execution.config import ExchangeConfig

LOGGER = logging.getLogger("autopilot.preflight")


def _market_type(product: ProductConfig) -> str:
    return "spot" if product.objective == "btc_accumulation" else "futures"


def _exchange_env_detail(cfg: ExchangeConfig, *, require_testnet: bool) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "exchange": cfg.exchange,
        "market_type": cfg.market_type,
        "testnet": cfg.testnet,
        "require_testnet": require_testnet,
        "quote_asset": cfg.quote_asset,
        "max_notional_usd": cfg.max_notional_usd,
        "max_fill_slippage_bps": cfg.max_fill_slippage_bps,
    }
    if cfg.market_type == "futures":
        detail["max_futures_leverage"] = cfg.max_futures_leverage
        detail["futures_margin_mode"] = cfg.futures_margin_mode
    return detail


def _check_exchange_env(product: ProductConfig, *, require_testnet: bool = False) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    try:
        cfg = ExchangeConfig.from_env(market_type=_market_type(product))
    except (OSError, ValueError) as exc:
        return [f"invalid exchange environment: {exc}"], None
    detail = _exchange_env_detail(cfg, require_testnet=require_testnet)
    if not cfg.live:
        errors.append("TRADING_LIVE must be 1 for a live/testnet execution rehearsal.")
    if cfg.max_notional_usd <= 0:
        errors.append("MAX_NOTIONAL_USD must be positive.")
    if cfg.max_fill_slippage_bps <= 0:
        errors.append("MAX_FILL_SLIPPAGE_BPS must be positive.")
    if cfg.market_type == "futures" and not (1 <= cfg.max_futures_leverage <= 3):
        errors.append("MAX_FUTURES_LEVERAGE must be between 1 and 3.")
    if (
        product.objective == "active_income"
        and cfg.market_type == "futures"
        and cfg.max_futures_leverage != ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
    ):
        errors.append(
            f"active income futures must use MAX_FUTURES_LEVERAGE={ACTIVE_INCOME_MAX_FUTURES_LEVERAGE}."
        )
    if cfg.market_type == "futures" and cfg.futures_margin_mode != "isolated":
        errors.append("FUTURES_MARGIN_MODE must be 'isolated'.")
    if require_testnet and not cfg.testnet:
        errors.append("EXCHANGE_TESTNET must be 1 for this preflight.")
    if not cfg.api_key or not cfg.api_secret:
        errors.append("EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required for balance/position checks.")
    errors.extend(validate_exchange_policy(product, cfg))
    return errors, detail


def _product_status(product: ProductConfig) -> dict[str, Any]:
    return {
        "name": product.name,
        "enabled": product.enabled,
        "objective": product.objective,
        "base_asset": product.base_asset,
        "market": product.market,
        "symbol": product.symbol,
        "execution_mode": product.execution_mode,
        "strategies_path": str(product.strategies_path),
        "require_preflight": product.require_preflight,
        "preflight_report": str(product.preflight_report) if product.preflight_report is not None else None,
        "preflight_max_age_seconds": product.preflight_max_age_seconds,
        "require_testnet_rehearsal": product.require_testnet_rehearsal,
        "testnet_rehearsal_report": (
            str(product.testnet_rehearsal_report) if product.testnet_rehearsal_report is not None else None
        ),
        "testnet_rehearsal_max_age_seconds": product.testnet_rehearsal_max_age_seconds,
    }


def _artifact_fingerprints(path: Path) -> list[str]:
    artifact = load_artifact(path)
    return [strategy_fingerprint(strategy) for strategy in artifact.get("strategies", [])]


def _artifact_digest(path: Path) -> str:
    return artifact_digest(load_artifact(path))


def _requires_flat_connect_position(product: ProductConfig) -> bool:
    return product.objective == "active_income" and product.market == "futures"


def _requires_non_negative_spot_position(product: ProductConfig) -> bool:
    return product.objective == "btc_accumulation" and product.market == "spot"


def preflight_product(
    product: ProductConfig,
    config: AutopilotConfig,
    *,
    assume_live: bool = False,
    connect: bool = False,
    require_testnet: bool = False,
    exchange_env_checker: Callable[[ProductConfig], list[str]] | None = None,
    broker_builder: Callable[[ProductConfig], Any] | None = None,
) -> dict[str, Any]:
    broker_builder = broker_builder or build_live_broker
    live_product = replace(product, execution_mode="live") if assume_live else product
    result: dict[str, Any] = {
        "ok": True,
        "product": _product_status(live_product),
        "checks": [],
        "errors": [],
    }

    def check(name: str, ok: bool, detail: dict[str, Any] | None = None, error: str | None = None) -> None:
        entry = {"name": name, "ok": ok}
        if detail is not None:
            entry["detail"] = detail
        if error is not None:
            entry["error"] = error
            result["errors"].append(error)
        result["checks"].append(entry)
        result["ok"] = result["ok"] and ok

    cfg_errors = validate_config(AutopilotConfig(products=[live_product]))
    check("product_config", not cfg_errors, error="; ".join(cfg_errors) if cfg_errors else None)

    artifact_exists = live_product.strategies_path.exists()
    check("strategy_artifact_exists", artifact_exists, {"path": str(live_product.strategies_path)} if artifact_exists else None,
          None if artifact_exists else f"Strategy artifact not found: {live_product.strategies_path}")

    if artifact_exists:
        try:
            fingerprints = _artifact_fingerprints(live_product.strategies_path)
            result["artifact_fingerprints"] = fingerprints
            result["artifact_digest"] = _artifact_digest(live_product.strategies_path)
            check("strategy_fingerprints", bool(fingerprints), {"fingerprints": fingerprints},
                  None if fingerprints else f"No strategy fingerprints found: {live_product.strategies_path}")
        except (ApprovalError, FileNotFoundError, json.JSONDecodeError) as exc:
            check("strategy_fingerprints", False, error=str(exc))
        try:
            detail = assert_strategy_artifact_allowed(live_product)
            check("strategy_policy", True, detail)
        except (StrategyPolicyError, FileNotFoundError) as exc:
            check("strategy_policy", False, error=str(exc))

    try:
        assert_artifact_live_approved(live_product.strategies_path, config.approval_ledger, product=live_product)
        check("approval_gate", True)
    except (ApprovalError, FileNotFoundError) as exc:
        check("approval_gate", False, error=str(exc))

    if exchange_env_checker is None:
        env_errors, env_detail = _check_exchange_env(live_product, require_testnet=require_testnet)
    else:
        env_errors = exchange_env_checker(live_product)
        env_detail = {"custom_checker": True, "require_testnet": require_testnet}
        if require_testnet:
            try:
                exchange_cfg = ExchangeConfig.from_env(market_type=_market_type(live_product))
            except (OSError, ValueError) as exc:
                env_errors.append(f"invalid exchange environment: {exc}")
            else:
                if not exchange_cfg.testnet:
                    env_errors.append("EXCHANGE_TESTNET must be 1 for this preflight.")
    check(
        "exchange_environment",
        not env_errors,
        detail=env_detail,
        error="; ".join(env_errors) if env_errors else None,
    )

    if not result["ok"]:
        check(
            "broker_constructed",
            False,
            detail={"skipped": True, "reason": "prerequisite_checks_failed"},
        )
        return result

    try:
        broker = broker_builder(live_product)
        check("broker_constructed", True, {"broker": broker.name})
    except Exception as exc:
        check("broker_constructed", False, error=str(exc))
        return result

    if connect:
        try:
            price = broker.get_price(live_product.symbol)
            balance = broker.get_balance()
            position = broker.get_position(live_product.symbol)
            check(
                "exchange_read_connectivity",
                True,
                {
                    "price": price,
                    "balance": balance,
                    "position_qty": position.qty,
                    "position_avg_price": position.avg_price,
                    "position_is_flat": position.is_flat,
                },
            )
            if _requires_flat_connect_position(live_product):
                check(
                    "broker_position_flat",
                    position.is_flat,
                    {
                        "symbol": live_product.symbol,
                        "position_qty": position.qty,
                        "position_avg_price": position.avg_price,
                    },
                    None
                    if position.is_flat
                    else (
                        f"{live_product.name}: broker position must be flat before "
                        f"active-income live/testnet enablement; got qty {position.qty:g}."
                    ),
                )
            if _requires_non_negative_spot_position(live_product):
                check(
                    "broker_spot_position_non_negative",
                    position.qty >= 0,
                    {
                        "symbol": live_product.symbol,
                        "position_qty": position.qty,
                        "position_avg_price": position.avg_price,
                    },
                    None
                    if position.qty >= 0
                    else (
                        f"{live_product.name}: spot BTC position must be non-negative; "
                        f"got qty {position.qty:g}."
                    ),
                )
        except Exception as exc:
            check("exchange_read_connectivity", False, error=str(exc))

    return result


def run_preflight(
    config: AutopilotConfig,
    *,
    product_name: str | None = None,
    assume_live: bool = False,
    connect: bool = False,
    require_testnet: bool = False,
    exchange_env_checker: Callable[[ProductConfig], list[str]] | None = None,
    broker_builder: Callable[[ProductConfig], Any] | None = None,
) -> dict[str, Any]:
    products = config.products
    if product_name:
        products = [product for product in products if product.name == product_name]
    else:
        products = [product for product in products if product.execution_mode == "live" or assume_live]
    if not products:
        return {
            "ok": False,
            "errors": [f"No products selected for preflight: {product_name or 'live products'}"],
            "products": [],
        }
    product_results = [
        preflight_product(
            product,
            config,
            assume_live=assume_live,
            connect=connect,
            require_testnet=require_testnet,
            exchange_env_checker=exchange_env_checker,
            broker_builder=broker_builder,
        )
        for product in products
    ]
    return {
        "generated_at": utc_now(),
        "generated_ts": time.time(),
        "ok": all(item["ok"] for item in product_results),
        "products": product_results,
    }


def _preflight_failure_report(name: str, detail: dict[str, Any]) -> dict[str, Any]:
    error = str(detail.get("error", "unknown error"))
    return {
        "generated_at": utc_now(),
        "generated_ts": time.time(),
        "ok": False,
        "errors": [f"{name}: {error}"],
        "checks": [{"name": name, "ok": False, "detail": detail}],
        "products": [],
    }


def _append_preflight_error(report: dict[str, Any], name: str, detail: dict[str, Any]) -> None:
    error = str(detail.get("error", "unknown error"))
    report["ok"] = False
    report.setdefault("errors", []).append(f"{name}: {error}")
    report.setdefault("checks", []).append({"name": name, "ok": False, "detail": detail})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live/testnet readiness checks without placing orders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", help="Product name to check. Defaults to live products.")
    parser.add_argument("--assume-live", action="store_true", help="Check selected paper products as if promoted live.")
    parser.add_argument("--connect", action="store_true", help="Fetch ticker, balance, and position. No orders are sent.")
    parser.add_argument(
        "--require-testnet",
        action="store_true",
        help="Fail unless EXCHANGE_TESTNET=1. Use before testnet order rehearsals.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON preflight report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = run_preflight(
            load_config(args.config),
            product_name=args.product,
            assume_live=args.assume_live,
            connect=args.connect,
            require_testnet=args.require_testnet,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build preflight report")
        report = _preflight_failure_report(
            "preflight_build_failed",
            {"config": str(args.config), "error": f"{type(exc).__name__}: {exc}"},
        )
    if args.output:
        try:
            write_json_atomic(args.output, report)
        except Exception as exc:
            LOGGER.exception("Failed to write preflight report")
            _append_preflight_error(
                report,
                "preflight_output_write_failed",
                {"path": str(args.output), "error": f"{type(exc).__name__}: {exc}"},
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
