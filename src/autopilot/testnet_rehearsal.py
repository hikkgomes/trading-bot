"""Guarded active-income futures testnet order rehearsal.

Preflight is intentionally read-only. This module is the explicit next step: it
requires the normal live approval/preflight gates plus EXCHANGE_TESTNET=1, then
places a tiny futures testnet market entry and immediately closes it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from src.autopilot.approvals import ApprovalError, artifact_digest, load_artifact
from src.autopilot.config import DEFAULT_CONFIG_PATH, AutopilotConfig, ProductConfig, load_config
from src.autopilot.exchange_policy import ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT
from src.execution.broker import Broker, Fill, Order, OrderSide, OrderType
from src.execution.config import ExchangeConfig

DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "testnet_rehearsal_report.json"
DEFAULT_MAX_REPORT_AGE_SECONDS = 30 * 24 * 60 * 60
TESTNET_REHEARSAL_CLOCK_SKEW_SECONDS = 300
LOGGER = logging.getLogger("autopilot.testnet_rehearsal")
EMBEDDED_PREFLIGHT_PRODUCT_KEYS = ("objective", "base_asset", "market", "symbol")


def testnet_rehearsal_next_action() -> dict[str, Any]:
    return {
        "preflight_command": "make preflight PRODUCT=active_income REQUIRE_TESTNET=1",
        "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5",
        "status_command": "make testnet-status",
        "required_env": [
            "TRADING_LIVE=1",
            "EXCHANGE_TESTNET=1",
            "FUTURES_EXCHANGE=binanceusdm",
            "EXCHANGE_API_KEY",
            "EXCHANGE_API_SECRET",
            "MAX_FUTURES_LEVERAGE=1",
            "FUTURES_MARGIN_MODE=isolated",
        ],
        "note": "A successful fresh rehearsal report is required before active_income live execution.",
    }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _fill_payload(fill: Fill | None) -> dict[str, Any] | None:
    if fill is None:
        return None
    return {
        "symbol": fill.symbol,
        "side": fill.side.value,
        "qty": fill.qty,
        "price": fill.price,
        "fee": fill.fee,
        "timestamp": fill.timestamp,
    }


def _fill_side_value(fill: Fill) -> str:
    return fill.side.value if isinstance(fill.side, OrderSide) else str(fill.side)


def _assert_fill_numeric_evidence(fill: Fill, *, label: str) -> None:
    qty = _finite_float(fill.qty)
    price = _finite_float(fill.price)
    fee = _finite_float(fill.fee)
    timestamp = _finite_float(fill.timestamp)
    if qty is None or qty <= 0:
        raise RuntimeError(f"{label} fill quantity must be positive.")
    if price is None or price <= 0:
        raise RuntimeError(f"{label} fill price must be positive.")
    if fee is None or fee < 0:
        raise RuntimeError(f"{label} fill fee must be finite and non-negative.")
    if timestamp is None or timestamp <= 0:
        raise RuntimeError(f"{label} fill timestamp must be positive.")


def _assert_rehearsal_fill_valid(
    product: ProductConfig,
    fill: Fill | None,
    *,
    label: str,
    expected_side: OrderSide,
    expected_qty: float,
) -> None:
    if fill is None:
        raise RuntimeError(f"missing {label} fill.")
    if fill.symbol != product.symbol:
        raise RuntimeError(f"{label} fill mismatch: expected symbol {product.symbol}, got {fill.symbol}.")
    side = _fill_side_value(fill)
    if side != expected_side.value:
        raise RuntimeError(f"{label} fill mismatch: expected side {expected_side.value}, got {side}.")
    _assert_fill_numeric_evidence(fill, label=label)
    tolerance = max(float(expected_qty) * 1e-6, 1e-9)
    if abs(float(fill.qty) - float(expected_qty)) > tolerance:
        raise RuntimeError(f"{label} fill mismatch: expected qty {expected_qty:g}, got {fill.qty:g}.")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fill_invalid_reasons(
    fill: dict[str, Any],
    *,
    label: str,
    expected_product: ProductConfig | None = None,
) -> list[str]:
    reasons: list[str] = []
    qty = _finite_float(fill.get("qty"))
    price = _finite_float(fill.get("price"))
    fee = _finite_float(fill.get("fee"))
    timestamp = _finite_float(fill.get("timestamp"))
    if expected_product is not None:
        symbol = str(fill.get("symbol") or "").strip().upper()
        if symbol != expected_product.symbol.upper():
            reasons.append(f"{label}_fill_symbol_mismatch")
    if qty is None or qty <= 0:
        reasons.append(f"{label}_fill_invalid_qty")
    if price is None or price <= 0:
        reasons.append(f"{label}_fill_invalid_price")
    if fee is None or fee < 0:
        reasons.append(f"{label}_fill_invalid_fee")
    if timestamp is None or timestamp <= 0:
        reasons.append(f"{label}_fill_invalid_timestamp")
    return reasons


def _fill_qty_mismatch_reasons(
    entry_fill: dict[str, Any],
    close_fill: dict[str, Any],
    *,
    order_qty: float | None,
    expected_product: ProductConfig | None,
) -> list[str]:
    if expected_product is None:
        return []
    if order_qty is None or order_qty <= 0:
        return ["invalid_order_qty"]
    reasons: list[str] = []
    tolerance = max(abs(order_qty) * 1e-6, 1e-9)
    entry_qty = _finite_float(entry_fill.get("qty")) if entry_fill else None
    close_qty = _finite_float(close_fill.get("qty")) if close_fill else None
    if entry_qty is not None and entry_qty > 0 and abs(entry_qty - order_qty) > tolerance:
        reasons.append("entry_fill_qty_mismatch")
    if close_qty is not None and close_qty > 0 and abs(close_qty - order_qty) > tolerance:
        reasons.append("close_fill_qty_mismatch")
    return reasons


def _expected_product_payload(product: ProductConfig) -> dict[str, Any]:
    return {
        "name": product.name,
        "objective": product.objective,
        "base_asset": product.base_asset,
        "market": product.market,
        "symbol": product.symbol,
    }


def _normalize_product_value(field: str, value: Any) -> str:
    text = str(value or "").strip()
    if field == "symbol":
        return text.upper()
    return text.lower()


def _product_invalid_reasons(report_product: Any, expected_product: ProductConfig | None) -> list[str]:
    if expected_product is None:
        return []
    if not isinstance(report_product, dict):
        return ["missing_product"]
    reasons = []
    expected = _expected_product_payload(expected_product)
    for field, expected_value in expected.items():
        if _normalize_product_value(field, report_product.get(field)) != _normalize_product_value(field, expected_value):
            reasons.append(f"product_{field}_mismatch")
    return reasons


def _risk_controls_payload(exchange_cfg: ExchangeConfig) -> dict[str, Any]:
    return {
        "max_futures_leverage": exchange_cfg.max_futures_leverage,
        "futures_margin_mode": exchange_cfg.futures_margin_mode,
        "max_notional_usd": exchange_cfg.max_notional_usd,
        "max_fill_slippage_bps": exchange_cfg.max_fill_slippage_bps,
    }


def _risk_control_invalid_reasons(risk_controls: Any, expected_product: ProductConfig | None) -> list[str]:
    if expected_product is None:
        return []
    if expected_product.objective != "active_income" or expected_product.market != "futures":
        return []
    if not isinstance(risk_controls, dict):
        return ["missing_risk_controls"]
    reasons = []
    try:
        leverage = int(risk_controls.get("max_futures_leverage"))
    except (TypeError, ValueError):
        leverage = 0
    if leverage != ACTIVE_INCOME_MAX_FUTURES_LEVERAGE:
        reasons.append("max_futures_leverage_invalid")
    margin_mode = str(risk_controls.get("futures_margin_mode") or "").strip().lower()
    if margin_mode != "isolated":
        reasons.append("futures_margin_mode_not_isolated")
    max_notional = _finite_float(risk_controls.get("max_notional_usd"))
    if max_notional is None or max_notional <= 0:
        reasons.append("max_notional_usd_invalid")
    slippage = _finite_float(risk_controls.get("max_fill_slippage_bps"))
    if slippage is None or slippage <= 0:
        reasons.append("max_fill_slippage_bps_invalid")
    return reasons


def _embedded_preflight_invalid_reasons(payload: dict[str, Any], expected_product: ProductConfig | None) -> list[str]:
    if expected_product is None or "preflight" not in payload:
        return []
    preflight = payload.get("preflight")
    if not isinstance(preflight, dict):
        return ["embedded_preflight_invalid"]
    reasons: list[str] = []
    if preflight.get("ok") is not True:
        reasons.append("embedded_preflight_failed")
    products = preflight.get("products")
    if not isinstance(products, list):
        return [*reasons, "embedded_preflight_products_invalid"]
    matched = None
    for item in products:
        if not isinstance(item, dict):
            return [*reasons, "embedded_preflight_products_invalid"]
        item_product = item.get("product")
        if not isinstance(item_product, dict):
            return [*reasons, "embedded_preflight_product_invalid"]
        if item_product.get("name") == expected_product.name:
            matched = item
            break
    if matched is None:
        return [*reasons, "embedded_preflight_missing_product"]
    if matched.get("ok") is not True:
        reasons.append("embedded_preflight_product_failed")
    reported_product = matched.get("product")
    if not isinstance(reported_product, dict):
        return [*reasons, "embedded_preflight_product_invalid"]
    for field in EMBEDDED_PREFLIGHT_PRODUCT_KEYS:
        if _normalize_product_value(field, reported_product.get(field)) != _normalize_product_value(
            field, getattr(expected_product, field)
        ):
            reasons.append(f"embedded_preflight_product_{field}_mismatch")
    report_artifact = reported_product.get("strategies_path")
    if report_artifact:
        try:
            if Path(report_artifact).resolve() != expected_product.strategies_path.resolve():
                reasons.append("embedded_preflight_artifact_path_mismatch")
        except OSError:
            reasons.append("embedded_preflight_artifact_path_invalid")
    reported_digest = matched.get("artifact_digest")
    if not isinstance(reported_digest, str) or not reported_digest:
        reasons.append("embedded_preflight_missing_artifact_digest")
    else:
        try:
            current_digest = artifact_digest(load_artifact(expected_product.strategies_path))
        except (ApprovalError, FileNotFoundError, json.JSONDecodeError, OSError):
            reasons.append("embedded_preflight_artifact_read_error")
        else:
            if current_digest != reported_digest:
                reasons.append("embedded_preflight_artifact_digest_mismatch")
    return reasons


def summarize_testnet_rehearsal_report(
    path: Path = DEFAULT_OUTPUT,
    *,
    max_age_seconds: int = DEFAULT_MAX_REPORT_AGE_SECONDS,
    now_ts: float | None = None,
    expected_product: ProductConfig | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "status": "missing",
    }
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    if path.is_symlink():
        status.update(
            exists=True,
            status="read_error",
            error=f"testnet rehearsal report must not be a symlink: {path}",
            next_action=testnet_rehearsal_next_action(),
        )
        return status
    if not path.exists():
        status["next_action"] = testnet_rehearsal_next_action()
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status.update(status="read_error", error=str(exc), next_action=testnet_rehearsal_next_action())
        return status
    if not isinstance(payload, dict):
        status.update(
            status="read_error",
            error=f"TypeError: expected JSON object, got {type(payload).__name__}",
            next_action=testnet_rehearsal_next_action(),
        )
        return status
    generated_ts = payload.get("generated_ts")
    age_seconds = None
    fresh = None
    generated_ts_valid = False
    generated_ts_future = False
    if generated_ts is not None:
        try:
            generated_ts_float = float(generated_ts)
        except (TypeError, ValueError):
            generated_ts_float = float("nan")
        if math.isfinite(generated_ts_float):
            generated_ts_valid = True
            now_ts = now_ts if now_ts is not None else time.time()
            age_seconds = now_ts - generated_ts_float
            generated_ts_future = age_seconds < -TESTNET_REHEARSAL_CLOCK_SKEW_SECONDS
            fresh = age_seconds <= max_age_seconds
    report_product = payload.get("product") if isinstance(payload.get("product"), dict) else None
    expected_product_payload = (
        _expected_product_payload(expected_product) if expected_product is not None else None
    )
    risk_controls = payload.get("risk_controls") if isinstance(payload.get("risk_controls"), dict) else None
    entry_fill = payload.get("entry_fill") if isinstance(payload.get("entry_fill"), dict) else {}
    close_fill = payload.get("close_fill") if isinstance(payload.get("close_fill"), dict) else {}
    entry_side = str(entry_fill.get("side") or "").lower() if entry_fill else None
    close_side = str(close_fill.get("side") or "").lower() if close_fill else None
    try:
        notional_usd = float(payload.get("notional_usd"))
    except (TypeError, ValueError):
        notional_usd = float("nan")
    order_qty = _finite_float(payload.get("order_qty"))
    final_position_qty = payload.get("final_position_qty")
    try:
        final_position_flat = abs(float(final_position_qty)) < 1e-12
    except (TypeError, ValueError):
        final_position_flat = False
    payload_ok = bool(payload.get("ok"))
    invalid_reasons: list[str] = []
    if not generated_ts_valid:
        invalid_reasons.append("missing_generated_ts" if generated_ts is None else "invalid_generated_ts")
    elif generated_ts_future:
        invalid_reasons.append("future_generated_ts")
    if not math.isfinite(notional_usd) or notional_usd <= 0:
        invalid_reasons.append("invalid_notional_usd")
    if payload.get("testnet") is not True:
        invalid_reasons.append("not_testnet")
    invalid_reasons.extend(_product_invalid_reasons(report_product, expected_product))
    invalid_reasons.extend(_risk_control_invalid_reasons(risk_controls, expected_product))
    invalid_reasons.extend(_embedded_preflight_invalid_reasons(payload, expected_product))
    if not entry_fill:
        invalid_reasons.append("missing_entry_fill")
    elif entry_side != "buy":
        invalid_reasons.append("entry_fill_side_not_buy")
    if entry_fill:
        invalid_reasons.extend(_fill_invalid_reasons(entry_fill, label="entry", expected_product=expected_product))
    if not close_fill:
        invalid_reasons.append("missing_close_fill")
    elif close_side != "sell":
        invalid_reasons.append("close_fill_side_not_sell")
    if close_fill:
        invalid_reasons.extend(_fill_invalid_reasons(close_fill, label="close", expected_product=expected_product))
    invalid_reasons.extend(
        _fill_qty_mismatch_reasons(
            entry_fill,
            close_fill,
            order_qty=order_qty,
            expected_product=expected_product,
        )
    )
    if not final_position_flat:
        invalid_reasons.append("final_position_not_flat")
    structurally_ok = not invalid_reasons
    if payload_ok and fresh is False:
        report_status = "stale"
    elif payload_ok and structurally_ok:
        report_status = "ok"
    else:
        report_status = "failed"
    status.update(
        {
            "ok": payload_ok and structurally_ok and fresh is not False,
            "status": report_status,
            "fresh": fresh,
            "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            "max_age_seconds": max_age_seconds,
            "clock_skew_seconds": TESTNET_REHEARSAL_CLOCK_SKEW_SECONDS,
            "generated_at": payload.get("generated_at"),
            "product": report_product.get("name") if report_product else None,
            "report_product": report_product,
            "expected_product": expected_product_payload,
            "risk_controls": risk_controls,
            "exchange": payload.get("exchange"),
            "testnet": payload.get("testnet"),
            "notional_usd": payload.get("notional_usd"),
            "order_qty": payload.get("order_qty"),
            "entry_side": entry_side,
            "close_side": close_side,
            "final_position_qty": final_position_qty,
            "final_position_flat": final_position_flat,
            "error": payload.get("error"),
            "invalid_reasons": invalid_reasons,
        }
    )
    if not status["ok"]:
        status["next_action"] = testnet_rehearsal_next_action()
    return status


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


def _fail(product: ProductConfig | None, error: str, **extra: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "generated_ts": time.time(),
        "ok": False,
        "error": error,
    }
    if product is not None:
        report["product"] = _product_status(product)
    report.update(extra)
    return report


def _append_rehearsal_error(report: dict[str, Any], name: str, detail: dict[str, Any]) -> None:
    report["ok"] = False
    if "error" not in report:
        report["error"] = name
    report.setdefault("errors", []).append({"name": name, "detail": detail})


def _write_rehearsal_report(path: Path, report: dict[str, Any]) -> None:
    try:
        write_json_atomic(path, report)
    except Exception as exc:
        LOGGER.exception("Failed to write testnet rehearsal report")
        _append_rehearsal_error(
            report,
            "testnet_rehearsal_output_write_failed",
            {"path": str(path), "error": f"{type(exc).__name__}: {exc}"},
        )


def _selected_product(config: AutopilotConfig, product_name: str) -> ProductConfig | None:
    for product in config.products:
        if product.name == product_name:
            return product
    return None


def _summarize_for_config(config: AutopilotConfig, product_name: str, path: Path) -> dict[str, Any]:
    product = _selected_product(config, product_name)
    if product is None:
        return {
            "path": str(path),
            "exists": path.exists(),
            "ok": False,
            "status": "product_not_found",
            "error": f"product not found: {product_name}",
            "next_action": testnet_rehearsal_next_action(),
        }
    return summarize_testnet_rehearsal_report(
        path,
        max_age_seconds=product.testnet_rehearsal_max_age_seconds,
        expected_product=product,
    )


def _validate_product(product: ProductConfig) -> str | None:
    if product.objective != "active_income" or product.market != "futures":
        return "testnet order rehearsal is currently restricted to active_income futures products"
    if product.symbol.upper() != "BTCUSDT":
        return "testnet order rehearsal is currently restricted to BTCUSDT"
    return None


def _validate_rehearsal_env(notional_usd: float) -> tuple[ExchangeConfig | None, str | None]:
    try:
        exchange_cfg = ExchangeConfig.from_env(market_type="futures")
    except (OSError, ValueError) as exc:
        return None, f"invalid exchange environment: {exc}"
    if not exchange_cfg.live:
        return exchange_cfg, "TRADING_LIVE must be 1 for testnet order rehearsal"
    if not exchange_cfg.testnet:
        return exchange_cfg, "EXCHANGE_TESTNET must be 1; refusing to rehearse on a live venue"
    if exchange_cfg.market_type != "futures":
        return exchange_cfg, "testnet order rehearsal requires futures market routing"
    if not math.isfinite(float(notional_usd)) or notional_usd <= 0:
        return exchange_cfg, "notional_usd must be finite and positive"
    if notional_usd > exchange_cfg.max_notional_usd:
        return exchange_cfg, (
            f"notional_usd {notional_usd:g} exceeds MAX_NOTIONAL_USD "
            f"{exchange_cfg.max_notional_usd:g}"
        )
    return exchange_cfg, None


def _order_qty(notional_usd: float, price: float) -> float:
    if not math.isfinite(float(price)) or price <= 0:
        raise ValueError(f"reference price must be finite and positive, got {price!r}")
    qty = float(notional_usd) / float(price)
    if not math.isfinite(qty) or qty <= 0:
        raise ValueError("computed order quantity must be finite and positive")
    return qty


def run_testnet_rehearsal(
    config: AutopilotConfig,
    *,
    product_name: str = "active_income",
    notional_usd: float = 5.0,
    confirm: bool = False,
    output_path: Path | None = None,
    broker_builder: Callable[[ProductConfig], Broker] | None = None,
) -> dict[str, Any]:
    def finish(report: dict[str, Any]) -> dict[str, Any]:
        if output_path is not None:
            _write_rehearsal_report(output_path, report)
        return report

    product = _selected_product(config, product_name)
    if product is None:
        return finish(_fail(None, f"product not found: {product_name}"))
    product_error = _validate_product(product)
    if product_error:
        return finish(_fail(product, product_error))
    if not confirm:
        return finish(
            _fail(
                product,
                "explicit confirmation is required because this places testnet orders",
                required_flag="--confirm",
            )
        )

    exchange_cfg, env_error = _validate_rehearsal_env(notional_usd)
    if env_error:
        return finish(_fail(product, env_error))

    live_product = replace(product, execution_mode="live")
    if broker_builder is None:
        from src.autopilot.runtime import build_live_broker

        builder = build_live_broker
    else:
        builder = broker_builder
    broker_cache: dict[str, Broker] = {}

    def cached_builder(selected: ProductConfig) -> Broker:
        if "broker" not in broker_cache:
            broker_cache["broker"] = builder(selected)
        return broker_cache["broker"]

    from src.autopilot.preflight import run_preflight

    preflight = run_preflight(
        config,
        product_name=product.name,
        assume_live=True,
        connect=True,
        require_testnet=True,
        broker_builder=cached_builder,
    )
    if not preflight.get("ok"):
        return finish(_fail(product, "preflight_failed", preflight=preflight))

    broker = cached_builder(live_product)
    initial_position = broker.get_position(live_product.symbol)
    if not initial_position.is_flat:
        return finish(
            _fail(
                product,
                "initial_position_not_flat",
                initial_position_qty=initial_position.qty,
                initial_position_avg_price=initial_position.avg_price,
                preflight=preflight,
            )
        )

    entry_fill: Fill | None = None
    close_fill: Fill | None = None
    recovery_close_fill: Fill | None = None
    final_position = None
    try:
        price = broker.get_price(live_product.symbol)
        balance = broker.get_balance()
        qty = _order_qty(notional_usd, price)
        entry_fill = broker.place_order(
            Order(
                symbol=live_product.symbol,
                side=OrderSide.BUY,
                qty=qty,
                type=OrderType.MARKET,
                client_id=f"testnet-rehearsal-{int(time.time())}",
            )
        )
        _assert_rehearsal_fill_valid(
            live_product,
            entry_fill,
            label="entry",
            expected_side=OrderSide.BUY,
            expected_qty=qty,
        )
        close_fill = broker.close_position(live_product.symbol)
        _assert_rehearsal_fill_valid(
            live_product,
            close_fill,
            label="close",
            expected_side=OrderSide.SELL,
            expected_qty=qty,
        )
        final_position = broker.get_position(live_product.symbol)
    except Exception as exc:
        recovery_error = None
        if entry_fill is not None:
            try:
                current_position = broker.get_position(live_product.symbol)
                if not current_position.is_flat:
                    recovery_close_fill = broker.close_position(live_product.symbol)
                final_position = broker.get_position(live_product.symbol)
            except Exception as recovery_exc:
                recovery_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
        recovery: dict[str, Any] = {
            "attempted": entry_fill is not None,
            "close_fill": _fill_payload(recovery_close_fill),
        }
        if final_position is not None:
            recovery.update(
                {
                    "final_position_qty": final_position.qty,
                    "final_position_avg_price": final_position.avg_price,
                    "final_position_flat": final_position.is_flat,
                }
            )
        if recovery_error is not None:
            recovery["error"] = recovery_error
        return finish(
            _fail(
                product,
                f"order_rehearsal_failed: {exc}",
                preflight=preflight,
                entry_fill=_fill_payload(entry_fill),
                close_fill=_fill_payload(close_fill),
                recovery=recovery,
            )
        )

    ok = close_fill is not None and final_position.is_flat
    report = {
        "generated_at": utc_now(),
        "generated_ts": time.time(),
        "ok": ok,
        "product": _product_status(live_product),
        "exchange": exchange_cfg.exchange if exchange_cfg else None,
        "testnet": exchange_cfg.testnet if exchange_cfg else None,
        "risk_controls": _risk_controls_payload(exchange_cfg),
        "notional_usd": float(notional_usd),
        "reference_price": price,
        "balance_before_entry": balance,
        "order_qty": qty,
        "entry_fill": _fill_payload(entry_fill),
        "close_fill": _fill_payload(close_fill),
        "final_position_qty": final_position.qty,
        "preflight": preflight,
    }
    if not ok:
        report["error"] = "final_position_not_flat_after_close"
    return finish(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place and immediately close a tiny active-income futures testnet order.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", default="active_income")
    parser.add_argument("--notional-usd", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", action="store_true", help="Summarize the saved rehearsal report without placing orders.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required. Confirms this command may place testnet orders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.status:
            report = _summarize_for_config(config, args.product, args.output)
        else:
            report = run_testnet_rehearsal(
                config,
                product_name=args.product,
                notional_usd=args.notional_usd,
                confirm=args.confirm,
                output_path=args.output,
            )
    except Exception as exc:
        LOGGER.exception("Failed to run testnet rehearsal")
        if args.status:
            report = {
                "path": str(args.output),
                "exists": args.output.exists(),
                "ok": False,
                "status": "status_failed",
                "error": "testnet_rehearsal_status_failed",
                "config": str(args.config),
                "exception": f"{type(exc).__name__}: {exc}",
                "next_action": testnet_rehearsal_next_action(),
            }
        else:
            report = _fail(
                None,
                "testnet_rehearsal_failed",
                config=str(args.config),
                exception=f"{type(exc).__name__}: {exc}",
            )
            _write_rehearsal_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
