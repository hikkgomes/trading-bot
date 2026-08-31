"""Guarded active-income futures testnet order rehearsal.

Preflight is intentionally read-only. This module is the explicit next step: it
requires the normal live approval/preflight gates plus EXCHANGE_TESTNET=1, then
places a tiny futures testnet market entry, proves a native reduce-only stop is
open, closes the position, and proves the stop is canceled or otherwise terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.autopilot.approvals import (
    ApprovalError,
    artifact_digest,
    assert_loaded_artifact_live_approved,
    load_artifact,
)
from src.autopilot.config import (
    DEFAULT_CONFIG_PATH,
    AutopilotConfig,
    ProductConfig,
    canonical_product_config,
    load_config,
)
from src.autopilot.exchange_policy import ACTIVE_INCOME_MAX_FUTURES_LEVERAGE
from src.autopilot.io import write_json_atomic
from src.config import PROJECT_ROOT
from src.execution.broker import (
    Broker,
    Fill,
    Order,
    OrderSide,
    OrderType,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ExchangeConfig

DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "testnet_rehearsal_report.json"
DEFAULT_MAX_REPORT_AGE_SECONDS = 30 * 24 * 60 * 60
TESTNET_REHEARSAL_CLOCK_SKEW_SECONDS = 300
TESTNET_PROTECTIVE_STOP_DISTANCE_FRACTION = 0.05
LOGGER = logging.getLogger("autopilot.testnet_rehearsal")


def testnet_rehearsal_next_action() -> dict[str, Any]:
    return {
        "preflight_command": "make preflight PRODUCT=active_income REQUIRE_TESTNET=1",
        "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100",
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
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


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


def _protective_order_payload(order: ProtectiveOrder | None) -> dict[str, Any] | None:
    if order is None:
        return None
    side = order.side.value if isinstance(order.side, OrderSide) else str(order.side)
    status = (
        order.status.value if isinstance(order.status, ProtectiveOrderStatus) else str(order.status)
    )
    return {
        "symbol": order.symbol,
        "side": side,
        "qty": order.qty,
        "trigger_price": order.trigger_price,
        "status": status,
        "order_id": order.order_id,
        "client_id": order.client_id,
        "filled_qty": order.filled_qty,
        "average_price": order.average_price,
        "fee": order.fee,
    }


def _assert_rehearsal_protective_identity(
    product: ProductConfig,
    order: ProtectiveOrder,
    *,
    label: str,
) -> None:
    if not isinstance(order, ProtectiveOrder):
        raise RuntimeError(f"{label} protective-stop evidence must be a ProtectiveOrder.")
    if order.symbol != product.symbol:
        raise RuntimeError(
            f"{label} protective-stop symbol mismatch: expected {product.symbol}, got {order.symbol}."
        )
    side = order.side.value if isinstance(order.side, OrderSide) else str(order.side)
    if side != OrderSide.SELL.value:
        raise RuntimeError(f"{label} protective-stop side mismatch: expected sell, got {side}.")


def _assert_rehearsal_protective_prices(
    order: ProtectiveOrder,
    *,
    label: str,
    expected_qty: float,
    expected_trigger_price: float,
) -> None:
    qty = _finite_float(order.qty)
    qty_tolerance = max(abs(expected_qty) * 1e-6, 1e-9)
    if qty is None or qty <= 0 or abs(qty - expected_qty) > qty_tolerance:
        raise RuntimeError(
            f"{label} protective-stop quantity mismatch: expected {expected_qty:g}, got {order.qty!r}."
        )
    trigger_price = _finite_float(order.trigger_price)
    trigger_tolerance = max(abs(expected_trigger_price) * 1e-8, 1e-8)
    if (
        trigger_price is None
        or trigger_price <= 0
        or abs(trigger_price - expected_trigger_price) > trigger_tolerance
    ):
        raise RuntimeError(
            f"{label} protective-stop trigger mismatch: expected {expected_trigger_price:g}, "
            f"got {order.trigger_price!r}."
        )


def _assert_rehearsal_protective_identity_fields(
    order: ProtectiveOrder,
    *,
    label: str,
    expected_client_id: str,
    expected_status: ProtectiveOrderStatus,
    expected_order_id: str | None,
    allowed_statuses: frozenset[ProtectiveOrderStatus] | None,
) -> None:
    if not isinstance(order.order_id, str) or not order.order_id.strip():
        raise RuntimeError(f"{label} protective-stop order id is missing.")
    if expected_order_id is not None and order.order_id != expected_order_id:
        raise RuntimeError(
            f"{label} protective-stop order id mismatch: expected {expected_order_id}, got {order.order_id}."
        )
    if order.client_id != expected_client_id:
        raise RuntimeError(
            f"{label} protective-stop client id mismatch: expected {expected_client_id}, "
            f"got {order.client_id}."
        )
    accepted_statuses = allowed_statuses or frozenset({expected_status})
    if order.status not in accepted_statuses:
        status = (
            order.status.value
            if isinstance(order.status, ProtectiveOrderStatus)
            else str(order.status)
        )
        expected = ", ".join(sorted(item.value for item in accepted_statuses))
        raise RuntimeError(
            f"{label} protective-stop status mismatch: expected one of {expected}, got {status}."
        )


def _assert_rehearsal_protective_unfilled(
    order: ProtectiveOrder,
    *,
    label: str,
) -> None:
    filled_qty = _finite_float(order.filled_qty)
    fee = _finite_float(order.fee)
    if filled_qty is None or filled_qty != 0:
        raise RuntimeError(f"{label} protective stop unexpectedly reports a fill.")
    if fee is None or fee < 0:
        raise RuntimeError(f"{label} protective-stop fee must be finite and non-negative.")


def _assert_rehearsal_protective_order_valid(
    product: ProductConfig,
    order: ProtectiveOrder,
    *,
    label: str,
    expected_qty: float,
    expected_trigger_price: float,
    expected_client_id: str,
    expected_status: ProtectiveOrderStatus,
    expected_order_id: str | None = None,
    allowed_statuses: frozenset[ProtectiveOrderStatus] | None = None,
) -> None:
    _assert_rehearsal_protective_identity(product, order, label=label)
    _assert_rehearsal_protective_prices(
        order,
        label=label,
        expected_qty=expected_qty,
        expected_trigger_price=expected_trigger_price,
    )
    _assert_rehearsal_protective_identity_fields(
        order,
        label=label,
        expected_client_id=expected_client_id,
        expected_status=expected_status,
        expected_order_id=expected_order_id,
        allowed_statuses=allowed_statuses,
    )
    _assert_rehearsal_protective_unfilled(order, label=label)


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
        raise RuntimeError(
            f"{label} fill mismatch: expected symbol {product.symbol}, got {fill.symbol}."
        )
    side = _fill_side_value(fill)
    if side != expected_side.value:
        raise RuntimeError(
            f"{label} fill mismatch: expected side {expected_side.value}, got {side}."
        )
    _assert_fill_numeric_evidence(fill, label=label)
    tolerance = max(float(expected_qty) * 1e-6, 1e-9)
    if abs(float(fill.qty) - float(expected_qty)) > tolerance:
        raise RuntimeError(
            f"{label} fill mismatch: expected qty {expected_qty:g}, got {fill.qty:g}."
        )


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


def _native_stop_flag_invalid_reasons(evidence: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key, reason in (
        ("capability_supported", "native_stop_capability_not_supported"),
        ("native", "native_stop_not_native"),
        ("reduce_only", "native_stop_not_reduce_only"),
        ("open_verified", "native_stop_open_not_verified"),
        ("canceled_verified", "native_stop_cancel_not_verified"),
    ):
        if evidence.get(key) is not True:
            reasons.append(reason)
    return reasons


def _native_stop_trigger_invalid_reasons(evidence: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    distance = _finite_float(evidence.get("trigger_distance_fraction"))
    if distance is None or not 0 < distance < 1:
        reasons.append("native_stop_invalid_trigger_distance")
    trigger_reference = _finite_float(evidence.get("trigger_reference_price"))
    if trigger_reference is None or trigger_reference <= 0:
        reasons.append("native_stop_invalid_trigger_reference")
    raw_trigger = _finite_float(evidence.get("raw_trigger_price"))
    if raw_trigger is None or raw_trigger <= 0:
        reasons.append("native_stop_invalid_raw_trigger")
    normalized_trigger = _finite_float(evidence.get("normalized_trigger_price"))
    if normalized_trigger is None or normalized_trigger <= 0:
        reasons.append("native_stop_invalid_normalized_trigger")
    return reasons


def _native_stop_snapshot_invalid_reasons(
    snapshot: Any,
    *,
    label: str,
    expected_statuses: frozenset[str],
    expected_label: str,
    order_qty: float | None,
    expected_product: ProductConfig | None,
    canonical_order_id: str | None,
    canonical_client_id: str | None,
    canonical_trigger: float | None,
) -> list[str]:
    if not isinstance(snapshot, dict):
        return [f"native_stop_missing_{label}"]
    reasons = _native_stop_snapshot_basic_reasons(
        snapshot,
        label=label,
        expected_statuses=expected_statuses,
        expected_label=expected_label,
        order_qty=order_qty,
        expected_product=expected_product,
    )
    reasons.extend(
        _native_stop_snapshot_consistency_reasons(
            snapshot,
            label=label,
            canonical_order_id=canonical_order_id,
            canonical_client_id=canonical_client_id,
            canonical_trigger=canonical_trigger,
        )
    )
    return reasons


def _native_stop_snapshot_basic_reasons(
    snapshot: dict[str, Any],
    *,
    label: str,
    expected_statuses: frozenset[str],
    expected_label: str,
    order_qty: float | None,
    expected_product: ProductConfig | None,
) -> list[str]:
    reasons = _native_stop_snapshot_shape_reasons(
        snapshot,
        label=label,
        expected_product=expected_product,
    )
    reasons.extend(
        _native_stop_snapshot_quantity_reasons(snapshot, label=label, order_qty=order_qty)
    )
    reasons.extend(
        _native_stop_snapshot_state_reasons(
            snapshot,
            label=label,
            expected_statuses=expected_statuses,
            expected_label=expected_label,
        )
    )
    return reasons


def _native_stop_snapshot_shape_reasons(
    snapshot: dict[str, Any],
    *,
    label: str,
    expected_product: ProductConfig | None,
) -> list[str]:
    reasons: list[str] = []
    symbol = str(snapshot.get("symbol") or "").strip().upper()
    if expected_product is not None and symbol != expected_product.symbol.upper():
        reasons.append(f"native_stop_{label}_symbol_mismatch")
    if str(snapshot.get("side") or "").strip().lower() != "sell":
        reasons.append(f"native_stop_{label}_side_not_sell")
    return reasons


def _native_stop_snapshot_quantity_reasons(
    snapshot: dict[str, Any],
    *,
    label: str,
    order_qty: float | None,
) -> list[str]:
    reasons: list[str] = []
    qty = _finite_float(snapshot.get("qty"))
    if qty is None or qty <= 0:
        reasons.append(f"native_stop_{label}_invalid_qty")
    elif order_qty is not None:
        tolerance = max(abs(order_qty) * 1e-6, 1e-9)
        if abs(qty - order_qty) > tolerance:
            reasons.append(f"native_stop_{label}_qty_mismatch")
    return reasons


def _native_stop_snapshot_state_reasons(
    snapshot: dict[str, Any],
    *,
    label: str,
    expected_statuses: frozenset[str],
    expected_label: str,
) -> list[str]:
    reasons: list[str] = []
    trigger = _finite_float(snapshot.get("trigger_price"))
    if trigger is None or trigger <= 0:
        reasons.append(f"native_stop_{label}_invalid_trigger")
    status = str(snapshot.get("status") or "").strip().lower()
    if status not in expected_statuses:
        reasons.append(f"native_stop_{label}_status_not_{expected_label}")
    order_id = str(snapshot.get("order_id") or "").strip()
    client_id = str(snapshot.get("client_id") or "").strip()
    if not order_id:
        reasons.append(f"native_stop_{label}_missing_order_id")
    if not client_id:
        reasons.append(f"native_stop_{label}_missing_client_id")
    filled_qty = _finite_float(snapshot.get("filled_qty"))
    if filled_qty is None or filled_qty != 0:
        reasons.append(f"native_stop_{label}_unexpected_fill")
    fee = _finite_float(snapshot.get("fee"))
    if fee is None or fee < 0:
        reasons.append(f"native_stop_{label}_invalid_fee")
    return reasons


def _native_stop_snapshot_consistency_reasons(
    snapshot: dict[str, Any],
    *,
    label: str,
    canonical_order_id: str | None,
    canonical_client_id: str | None,
    canonical_trigger: float | None,
) -> list[str]:
    if label == "placed":
        return []
    reasons: list[str] = []
    if label != "placed":
        order_id = str(snapshot.get("order_id") or "").strip()
        client_id = str(snapshot.get("client_id") or "").strip()
        trigger = _finite_float(snapshot.get("trigger_price"))
        if canonical_order_id is not None and order_id != canonical_order_id:
            reasons.append(f"native_stop_{label}_order_id_mismatch")
        if canonical_client_id is not None and client_id != canonical_client_id:
            reasons.append(f"native_stop_{label}_client_id_mismatch")
        if canonical_trigger is not None and trigger is not None:
            tolerance = max(abs(canonical_trigger) * 1e-8, 1e-8)
            if abs(trigger - canonical_trigger) > tolerance:
                reasons.append(f"native_stop_{label}_trigger_mismatch")
    return reasons


def _native_stop_relationship_invalid_reasons(
    *,
    canonical_trigger: float | None,
    trigger_reference: float | None,
    distance: float | None,
    raw_trigger: float | None,
    normalized_trigger: float | None,
    entry_fill: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    entry_price = _finite_float(entry_fill.get("price")) if entry_fill else None
    if canonical_trigger is not None and entry_price is not None:
        if canonical_trigger >= entry_price:
            reasons.append("native_stop_trigger_not_below_long_entry")
    if (
        trigger_reference is not None
        and distance is not None
        and 0 < distance < 1
        and raw_trigger is not None
    ):
        expected_raw_trigger = trigger_reference * (1.0 - distance)
        tolerance = max(abs(expected_raw_trigger) * 1e-8, 1e-8)
        if abs(raw_trigger - expected_raw_trigger) > tolerance:
            reasons.append("native_stop_raw_trigger_distance_mismatch")
    if canonical_trigger is not None and normalized_trigger is not None:
        tolerance = max(abs(canonical_trigger) * 1e-8, 1e-8)
        if abs(normalized_trigger - canonical_trigger) > tolerance:
            reasons.append("native_stop_normalized_trigger_mismatch")
    return reasons


def _native_protective_stop_invalid_reasons(
    evidence: Any,
    *,
    order_qty: float | None,
    entry_fill: dict[str, Any],
    expected_product: ProductConfig | None,
) -> list[str]:
    if not isinstance(evidence, dict):
        return ["missing_native_protective_stop"]

    reasons = _native_stop_flag_invalid_reasons(evidence)
    distance = _finite_float(evidence.get("trigger_distance_fraction"))
    trigger_reference = _finite_float(evidence.get("trigger_reference_price"))
    raw_trigger = _finite_float(evidence.get("raw_trigger_price"))
    normalized_trigger = _finite_float(evidence.get("normalized_trigger_price"))
    reasons.extend(_native_stop_trigger_invalid_reasons(evidence))

    snapshots = (
        ("placed", frozenset({"open"}), "open"),
        ("fetched_open", frozenset({"open"}), "open"),
        ("cancel_result", frozenset({"canceled"}), "canceled"),
        (
            "fetched_terminal",
            frozenset({"canceled", "expired", "rejected"}),
            "terminal",
        ),
    )
    canonical_order_id: str | None = None
    canonical_client_id: str | None = None
    canonical_trigger: float | None = None
    for label, expected_statuses, expected_label in snapshots:
        snapshot = evidence.get(label)
        reasons.extend(
            _native_stop_snapshot_invalid_reasons(
                snapshot,
                label=label,
                expected_statuses=expected_statuses,
                expected_label=expected_label,
                order_qty=order_qty,
                expected_product=expected_product,
                canonical_order_id=canonical_order_id,
                canonical_client_id=canonical_client_id,
                canonical_trigger=canonical_trigger,
            )
        )
        if label == "placed" and isinstance(snapshot, dict):
            order_id = str(snapshot.get("order_id") or "").strip()
            client_id = str(snapshot.get("client_id") or "").strip()
            trigger = _finite_float(snapshot.get("trigger_price"))
            canonical_order_id = order_id or None
            canonical_client_id = client_id or None
            canonical_trigger = trigger
    reasons.extend(
        _native_stop_relationship_invalid_reasons(
            canonical_trigger=canonical_trigger,
            trigger_reference=trigger_reference,
            distance=distance,
            raw_trigger=raw_trigger,
            normalized_trigger=normalized_trigger,
            entry_fill=entry_fill,
        )
    )
    return reasons


def _expected_product_payload(product: ProductConfig) -> dict[str, Any]:
    return canonical_product_config(product)


def _product_invalid_reasons(
    report_product: Any, expected_product: ProductConfig | None
) -> list[str]:
    if expected_product is None:
        return []
    if not isinstance(report_product, dict):
        return ["missing_product"]
    reasons = []
    expected = _expected_product_payload(expected_product)
    for field, expected_value in expected.items():
        if field not in report_product or report_product[field] != expected_value:
            reasons.append(f"product_{field}_mismatch")
    if set(report_product) - set(expected):
        reasons.append("product_unexpected_fields")
    return reasons


def _risk_controls_payload(exchange_cfg: ExchangeConfig) -> dict[str, Any]:
    return {
        "max_futures_leverage": exchange_cfg.max_futures_leverage,
        "futures_margin_mode": exchange_cfg.futures_margin_mode,
        "max_notional_usd": exchange_cfg.max_notional_usd,
        "max_fill_slippage_bps": exchange_cfg.max_fill_slippage_bps,
    }


def _risk_control_invalid_reasons(
    risk_controls: Any, expected_product: ProductConfig | None
) -> list[str]:
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


def _embedded_preflight_invalid_reasons(
    payload: dict[str, Any], expected_product: ProductConfig | None
) -> list[str]:
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
    preflight_product = replace(expected_product, execution_mode="live")
    expected_payload = canonical_product_config(preflight_product)
    for field, expected_value in expected_payload.items():
        if field not in reported_product or reported_product[field] != expected_value:
            reasons.append(f"embedded_preflight_product_{field}_mismatch")
    if set(reported_product) - set(expected_payload):
        reasons.append("embedded_preflight_product_unexpected_fields")
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
    if reasons:
        return reasons
    if expected_product.objective == "active_income" and expected_product.market == "futures":
        checks = matched.get("checks")
        if not isinstance(checks, list):
            return ["embedded_preflight_checks_invalid"]
        capability_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "broker_native_protective_stops"
            ),
            None,
        )
        if capability_check is None:
            return ["embedded_preflight_missing_native_stop_capability"]
        if capability_check.get("ok") is not True:
            return ["embedded_preflight_native_stop_capability_failed"]
        detail = capability_check.get("detail")
        if not isinstance(detail, dict) or detail.get("supported") is not True:
            return ["embedded_preflight_native_stop_capability_invalid"]
        position_mode_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "broker_position_mode_one_way"
            ),
            None,
        )
        if position_mode_check is None:
            return ["embedded_preflight_missing_one_way_position_mode"]
        if position_mode_check.get("ok") is not True:
            return ["embedded_preflight_one_way_position_mode_failed"]
        position_mode_detail = position_mode_check.get("detail")
        if (
            not isinstance(position_mode_detail, dict)
            or position_mode_detail.get("one_way") is not True
            or str(position_mode_detail.get("symbol") or "").upper()
            != expected_product.symbol.upper()
        ):
            return ["embedded_preflight_one_way_position_mode_invalid"]
        open_orders_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "broker_open_orders_empty"
            ),
            None,
        )
        if open_orders_check is None:
            return ["embedded_preflight_missing_open_order_inventory"]
        if open_orders_check.get("ok") is not True:
            return ["embedded_preflight_open_order_inventory_failed"]
        inventory_detail = open_orders_check.get("detail")
        if not isinstance(inventory_detail, dict):
            return ["embedded_preflight_open_order_inventory_invalid"]
        if (
            inventory_detail.get("scope") != "whole_account"
            or str(inventory_detail.get("configured_symbol") or "").upper()
            != expected_product.symbol.upper()
        ):
            return ["embedded_preflight_open_order_inventory_invalid"]
        for order_kind in ("regular", "conditional"):
            inventory = inventory_detail.get(order_kind)
            if not isinstance(inventory, dict):
                return ["embedded_preflight_open_order_inventory_invalid"]
            count = inventory.get("count")
            orders = inventory.get("orders")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count != 0
                or not isinstance(orders, list)
                or orders
            ):
                return ["embedded_preflight_open_order_inventory_invalid"]
        position_inventory_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "broker_position_flat"
            ),
            None,
        )
        if position_inventory_check is None:
            return ["embedded_preflight_missing_position_inventory"]
        if position_inventory_check.get("ok") is not True:
            return ["embedded_preflight_position_inventory_failed"]
        position_inventory = position_inventory_check.get("detail")
        if (
            not isinstance(position_inventory, dict)
            or position_inventory.get("scope") != "whole_account"
            or str(position_inventory.get("configured_symbol") or "").upper()
            != expected_product.symbol.upper()
            or position_inventory.get("count") != 0
            or position_inventory.get("positions") != []
        ):
            return ["embedded_preflight_position_inventory_invalid"]
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
        status.update(
            status="read_error", error=str(exc), next_action=testnet_rehearsal_next_action()
        )
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
    risk_controls = (
        payload.get("risk_controls") if isinstance(payload.get("risk_controls"), dict) else None
    )
    entry_fill = payload.get("entry_fill") if isinstance(payload.get("entry_fill"), dict) else {}
    close_fill = payload.get("close_fill") if isinstance(payload.get("close_fill"), dict) else {}
    native_protective_stop = payload.get("native_protective_stop")
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
        invalid_reasons.append(
            "missing_generated_ts" if generated_ts is None else "invalid_generated_ts"
        )
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
        invalid_reasons.extend(
            _fill_invalid_reasons(entry_fill, label="entry", expected_product=expected_product)
        )
    if not close_fill:
        invalid_reasons.append("missing_close_fill")
    elif close_side != "sell":
        invalid_reasons.append("close_fill_side_not_sell")
    if close_fill:
        invalid_reasons.extend(
            _fill_invalid_reasons(close_fill, label="close", expected_product=expected_product)
        )
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
    if "native_protective_stop" in payload or not invalid_reasons:
        invalid_reasons.extend(
            _native_protective_stop_invalid_reasons(
                native_protective_stop,
                order_qty=order_qty,
                entry_fill=entry_fill,
                expected_product=expected_product,
            )
        )
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
            "native_protective_stop": (
                native_protective_stop if isinstance(native_protective_stop, dict) else None
            ),
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
    return canonical_product_config(product)


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
    notional_usd: float = 100.0,
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

    try:
        approved_artifact = load_artifact(product.strategies_path)
        assert_loaded_artifact_live_approved(
            approved_artifact,
            product.strategies_path,
            config.approval_ledger,
            product=product,
        )
    except (ApprovalError, FileNotFoundError, json.JSONDecodeError) as exc:
        return finish(_fail(product, f"approval_failed: {exc}"))

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
    placed_stop: ProtectiveOrder | None = None
    fetched_open_stop: ProtectiveOrder | None = None
    cancel_result: ProtectiveOrder | None = None
    fetched_terminal_stop: ProtectiveOrder | None = None
    stop_client_id: str | None = None
    stop_trigger_price: float | None = None
    native_stop_evidence: dict[str, Any] = {
        "capability_supported": False,
        "native": True,
        "reduce_only": True,
        "trigger_distance_fraction": TESTNET_PROTECTIVE_STOP_DISTANCE_FRACTION,
        "open_verified": False,
        "canceled_verified": False,
        "placed": None,
        "fetched_open": None,
        "cancel_result": None,
        "fetched_terminal": None,
    }
    final_position = None
    try:
        capability_supported = broker.supports_native_protective_stops()
        native_stop_evidence["capability_supported"] = capability_supported
        if capability_supported is not True:
            raise RuntimeError("connected broker does not support exchange-native protective stops")
        price = broker.get_price(live_product.symbol)
        balance = broker.get_balance()
        raw_stop_trigger_price = float(price) * (1.0 - TESTNET_PROTECTIVE_STOP_DISTANCE_FRACTION)
        stop_trigger_price = _finite_float(
            broker.normalize_order_price(
                live_product.symbol,
                raw_stop_trigger_price,
            )
        )
        if (
            stop_trigger_price is None
            or stop_trigger_price <= 0
            or stop_trigger_price >= float(price)
        ):
            raise RuntimeError(
                "broker returned an invalid normalized protective-stop trigger price"
            )
        raw_qty = _order_qty(notional_usd, price)
        normalized_qty = _finite_float(
            broker.normalize_order_qty(
                live_product.symbol,
                raw_qty,
                price=price,
            )
        )
        if normalized_qty is None or normalized_qty <= 0:
            raise RuntimeError("broker returned an invalid normalized order quantity")
        qty = normalized_qty
        native_stop_evidence["trigger_reference_price"] = float(price)
        native_stop_evidence["raw_trigger_price"] = raw_stop_trigger_price
        native_stop_evidence["normalized_trigger_price"] = stop_trigger_price
        nonce = int(time.time() * 1000)
        entry_fill = broker.place_order(
            Order(
                symbol=live_product.symbol,
                side=OrderSide.BUY,
                qty=qty,
                type=OrderType.MARKET,
                client_id=f"testnet-entry-{nonce}",
            )
        )
        _assert_rehearsal_fill_valid(
            live_product,
            entry_fill,
            label="entry",
            expected_side=OrderSide.BUY,
            expected_qty=qty,
        )
        entry_qty = float(entry_fill.qty)
        if stop_trigger_price >= float(entry_fill.price):
            raise RuntimeError(
                "normalized protective-stop trigger is not below the actual entry fill"
            )
        stop_client_id = f"testnet-stop-{nonce}"
        placed_stop = broker.place_protective_stop(
            symbol=live_product.symbol,
            side=OrderSide.SELL,
            qty=entry_qty,
            trigger_price=stop_trigger_price,
            client_id=stop_client_id,
        )
        native_stop_evidence["placed"] = _protective_order_payload(placed_stop)
        _assert_rehearsal_protective_order_valid(
            live_product,
            placed_stop,
            label="placed",
            expected_qty=entry_qty,
            expected_trigger_price=stop_trigger_price,
            expected_client_id=stop_client_id,
            expected_status=ProtectiveOrderStatus.OPEN,
        )
        fetched_open_stop = broker.get_protective_stop(
            symbol=live_product.symbol,
            order_id=placed_stop.order_id,
            client_id=stop_client_id,
        )
        native_stop_evidence["fetched_open"] = _protective_order_payload(fetched_open_stop)
        _assert_rehearsal_protective_order_valid(
            live_product,
            fetched_open_stop,
            label="fetched-open",
            expected_qty=entry_qty,
            expected_trigger_price=stop_trigger_price,
            expected_client_id=stop_client_id,
            expected_status=ProtectiveOrderStatus.OPEN,
            expected_order_id=placed_stop.order_id,
        )
        native_stop_evidence["open_verified"] = True

        close_fill = broker.close_position(live_product.symbol)
        _assert_rehearsal_fill_valid(
            live_product,
            close_fill,
            label="close",
            expected_side=OrderSide.SELL,
            expected_qty=entry_qty,
        )
        final_position = broker.get_position(live_product.symbol)
        if not final_position.is_flat:
            raise RuntimeError(
                f"position is not flat after reduce-only close: qty {final_position.qty:g}"
            )

        cancel_result = broker.cancel_protective_stop(
            symbol=live_product.symbol,
            order_id=placed_stop.order_id,
            client_id=stop_client_id,
        )
        native_stop_evidence["cancel_result"] = _protective_order_payload(cancel_result)
        _assert_rehearsal_protective_order_valid(
            live_product,
            cancel_result,
            label="cancel-result",
            expected_qty=entry_qty,
            expected_trigger_price=stop_trigger_price,
            expected_client_id=stop_client_id,
            expected_status=ProtectiveOrderStatus.CANCELED,
            expected_order_id=placed_stop.order_id,
        )
        fetched_terminal_stop = broker.get_protective_stop(
            symbol=live_product.symbol,
            order_id=placed_stop.order_id,
            client_id=stop_client_id,
        )
        native_stop_evidence["fetched_terminal"] = _protective_order_payload(fetched_terminal_stop)
        _assert_rehearsal_protective_order_valid(
            live_product,
            fetched_terminal_stop,
            label="fetched-terminal",
            expected_qty=entry_qty,
            expected_trigger_price=stop_trigger_price,
            expected_client_id=stop_client_id,
            expected_status=ProtectiveOrderStatus.CANCELED,
            expected_order_id=placed_stop.order_id,
            allowed_statuses=frozenset(
                {
                    ProtectiveOrderStatus.CANCELED,
                    ProtectiveOrderStatus.EXPIRED,
                    ProtectiveOrderStatus.REJECTED,
                }
            ),
        )
        native_stop_evidence["canceled_verified"] = True
        final_position = broker.get_position(live_product.symbol)
        if not final_position.is_flat:
            raise RuntimeError(
                f"position changed after protective-stop cancellation: qty {final_position.qty:g}"
            )
    except Exception as exc:
        recovery_errors: dict[str, str] = {}
        if entry_fill is not None:
            try:
                current_position = broker.get_position(live_product.symbol)
                if not current_position.is_flat:
                    recovery_close_fill = broker.close_position(live_product.symbol)
                final_position = broker.get_position(live_product.symbol)
            except Exception as recovery_exc:
                recovery_errors["close"] = f"{type(recovery_exc).__name__}: {recovery_exc}"
                try:
                    final_position = broker.get_position(live_product.symbol)
                except Exception as position_exc:
                    recovery_errors["position_reconciliation"] = (
                        f"{type(position_exc).__name__}: {position_exc}"
                    )

        if (
            final_position is not None
            and final_position.is_flat
            and stop_client_id is not None
            and native_stop_evidence["canceled_verified"] is not True
        ):
            try:
                recovery_cancel = broker.cancel_protective_stop(
                    symbol=live_product.symbol,
                    order_id=placed_stop.order_id if placed_stop is not None else None,
                    client_id=stop_client_id,
                )
                native_stop_evidence["recovery_cancel_result"] = _protective_order_payload(
                    recovery_cancel
                )
                if stop_trigger_price is not None:
                    _assert_rehearsal_protective_order_valid(
                        live_product,
                        recovery_cancel,
                        label="recovery-cancel",
                        expected_qty=float(entry_fill.qty),
                        expected_trigger_price=stop_trigger_price,
                        expected_client_id=stop_client_id,
                        expected_status=ProtectiveOrderStatus.CANCELED,
                        expected_order_id=(
                            placed_stop.order_id if placed_stop is not None else None
                        ),
                    )
                    recovery_fetched = broker.get_protective_stop(
                        symbol=live_product.symbol,
                        order_id=recovery_cancel.order_id,
                        client_id=stop_client_id,
                    )
                    native_stop_evidence["recovery_fetched_terminal"] = _protective_order_payload(
                        recovery_fetched
                    )
                    _assert_rehearsal_protective_order_valid(
                        live_product,
                        recovery_fetched,
                        label="recovery-fetched-terminal",
                        expected_qty=float(entry_fill.qty),
                        expected_trigger_price=stop_trigger_price,
                        expected_client_id=stop_client_id,
                        expected_status=ProtectiveOrderStatus.CANCELED,
                        expected_order_id=recovery_cancel.order_id,
                        allowed_statuses=frozenset(
                            {
                                ProtectiveOrderStatus.CANCELED,
                                ProtectiveOrderStatus.EXPIRED,
                                ProtectiveOrderStatus.REJECTED,
                            }
                        ),
                    )
                    native_stop_evidence["canceled_verified"] = True
            except Exception as recovery_exc:
                recovery_errors["protective_stop_cancel"] = (
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        elif placed_stop is not None and final_position is not None and not final_position.is_flat:
            native_stop_evidence["left_open_to_protect_non_flat_position"] = True

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
        if recovery_errors:
            recovery["errors"] = recovery_errors
        return finish(
            _fail(
                product,
                f"order_rehearsal_failed: {exc}",
                preflight=preflight,
                entry_fill=_fill_payload(entry_fill),
                close_fill=_fill_payload(close_fill),
                native_protective_stop=native_stop_evidence,
                recovery=recovery,
            )
        )

    ok = (
        close_fill is not None
        and final_position.is_flat
        and native_stop_evidence["open_verified"] is True
        and native_stop_evidence["canceled_verified"] is True
    )
    report = {
        "generated_at": utc_now(),
        "generated_ts": time.time(),
        "ok": ok,
        "product": _product_status(product),
        "exchange": exchange_cfg.exchange if exchange_cfg else None,
        "testnet": exchange_cfg.testnet if exchange_cfg else None,
        "risk_controls": _risk_controls_payload(exchange_cfg),
        "notional_usd": float(notional_usd),
        "reference_price": price,
        "balance_before_entry": balance,
        "raw_order_qty": raw_qty,
        "order_qty": qty,
        "entry_fill": _fill_payload(entry_fill),
        "close_fill": _fill_payload(close_fill),
        "native_protective_stop": native_stop_evidence,
        "final_position_qty": final_position.qty,
        "preflight": preflight,
    }
    if not ok:
        report["error"] = "final_position_not_flat_after_close"
    return finish(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place and immediately close a tiny active-income futures testnet order."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--product", default="active_income")
    parser.add_argument("--notional-usd", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Summarize the saved rehearsal report without placing orders.",
    )
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
