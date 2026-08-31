"""Fail-closed exposure accounting for live order admission.

The live authority is evaluated immediately before an exchange side effect.  It
therefore has to debit durable positions and still-open intents, not only the
new order being considered.  This module keeps that calculation independent of
the exchange adapter so it can be tested without credentials or network calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.domain.orders import OrderIntent, OrderSide
from src.domain.positions import Position


class ExposureBudgetError(PermissionError):
    """A live order would exceed a canonical exposure budget."""


@dataclass(frozen=True)
class ExposureBudgetAssessment:
    """Auditable values used to accept or reject one live order."""

    product_id: str
    portfolio_id: str
    instrument_id: str
    equity: float
    reference_price: float
    current_gross_notional: float
    pending_gross_notional: float
    proposed_notional: float
    gross_notional: float
    net_notional: float
    limits: Mapping[str, float]

    def as_payload(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "portfolio_id": self.portfolio_id,
            "instrument_id": self.instrument_id,
            "equity": self.equity,
            "reference_price": self.reference_price,
            "current_gross_notional": self.current_gross_notional,
            "pending_gross_notional": self.pending_gross_notional,
            "proposed_notional": self.proposed_notional,
            "gross_notional": self.gross_notional,
            "net_notional": self.net_notional,
            "limits": dict(self.limits),
        }


class ExposureBudgetGuard:
    """Check assignment, sleeve, product, instrument, and account budgets."""

    def assess(
        self,
        *,
        product_id: str,
        product: Mapping[str, Any],
        account: Mapping[str, Any],
        assignment: Mapping[str, Any],
        risk_configuration: Mapping[str, Any],
        account_payload: Mapping[str, Any],
        order: OrderIntent,
        positions: Iterable[Position],
        orders: Iterable[OrderIntent],
    ) -> ExposureBudgetAssessment:
        if order.portfolio_id != str(product.get("portfolio_id") or ""):
            raise ExposureBudgetError("live order portfolio does not match product")
        position_rows = tuple(item for item in positions if item.portfolio_id == order.portfolio_id)
        order_rows = tuple(orders)
        price = _order_reference_price(order)
        equity = _account_equity(
            product_id=product_id,
            account=account,
            account_payload=account_payload,
            price=price,
        )
        price_by_instrument = _known_prices(
            order=order, account_payload=account_payload, orders=order_rows
        )
        current_notional: dict[str, float] = {}
        current_signed: dict[str, float] = {}
        for position in position_rows:
            if abs(position.quantity) <= 1e-12:
                continue
            mark = price_by_instrument.get(position.instrument_id)
            if mark is None:
                raise ExposureBudgetError(
                    f"live exposure has no current price for {position.instrument_id}"
                )
            notional = abs(position.quantity) * mark
            current_notional[position.instrument_id] = (
                current_notional.get(position.instrument_id, 0.0) + notional
            )
            current_signed[position.instrument_id] = (
                current_signed.get(position.instrument_id, 0.0) + position.quantity * mark
            )

        pending_gross = 0.0
        pending_net = 0.0
        for existing in order_rows:
            if (
                existing.order_id == order.order_id
                or existing.portfolio_id != order.portfolio_id
                or existing.is_terminal
                or existing.reduce_only
            ):
                continue
            mark = price_by_instrument.get(existing.instrument_id)
            if mark is None:
                raise ExposureBudgetError(
                    f"pending live exposure has no current price for {existing.instrument_id}"
                )
            notional = existing.remaining_quantity * mark
            pending_gross += notional
            pending_net += _signed_notional(existing.side, notional)

        proposed = 0.0 if order.reduce_only else order.quantity * price
        proposed_net = 0.0 if order.reduce_only else _signed_notional(order.side, proposed)
        gross = sum(current_notional.values()) + pending_gross + proposed
        net = sum(current_signed.values()) + pending_net + proposed_net
        limits = _limits(
            product_id=product_id,
            product=product,
            account=account,
            assignment=assignment,
            risk_configuration=risk_configuration,
            equity=equity,
        )
        assessment = ExposureBudgetAssessment(
            product_id=product_id,
            portfolio_id=order.portfolio_id,
            instrument_id=order.instrument_id,
            equity=equity,
            reference_price=price,
            current_gross_notional=sum(current_notional.values()),
            pending_gross_notional=pending_gross,
            proposed_notional=proposed,
            gross_notional=gross,
            net_notional=net,
            limits=limits,
        )
        if order.reduce_only:
            _assert_reduce_only(order, current_signed.get(order.instrument_id, 0.0), price)
            return assessment
        for name, limit in limits.items():
            value = abs(net) if name == "net_notional" else gross
            if value > limit + max(1e-9, abs(limit) * 1e-12):
                raise ExposureBudgetError(f"live order exceeds {name}: {value:g} > {limit:g}")
        return assessment

    def enforce(self, **kwargs: Any) -> ExposureBudgetAssessment:
        """Return the auditable assessment or raise a permission error."""

        return self.assess(**kwargs)


def _account_equity(
    *,
    product_id: str,
    account: Mapping[str, Any],
    account_payload: Mapping[str, Any],
    price: float,
) -> float:
    balances = account_payload.get("balances")
    if not isinstance(balances, Mapping):
        raise ExposureBudgetError("live account snapshot has no balances")
    clean = {
        str(key).upper(): _finite(value, field=f"balance {key}") for key, value in balances.items()
    }
    if product_id == "btc_accumulation" or str(account.get("market")) == "spot":
        equity = clean.get("USDT", 0.0) + clean.get("BTC", 0.0) * price
    else:
        equity = clean.get("USDT", 0.0)
    if equity <= 0:
        raise ExposureBudgetError("live account equity must be positive")
    return equity


def _known_prices(
    *,
    order: OrderIntent,
    account_payload: Mapping[str, Any],
    orders: Iterable[OrderIntent],
) -> dict[str, float]:
    prices: dict[str, float] = {order.instrument_id: _order_reference_price(order)}
    raw_prices = account_payload.get("market_prices")
    if isinstance(raw_prices, Mapping):
        for instrument_id, raw_price in raw_prices.items():
            prices[str(instrument_id)] = _finite(raw_price, field=f"market price {instrument_id}")
    raw_positions = account_payload.get("positions")
    if isinstance(raw_positions, Mapping):
        for instrument_id, raw_position in raw_positions.items():
            if not isinstance(raw_position, Mapping):
                continue
            for key in ("mark_price", "price", "reference_price"):
                if raw_position.get(key) is not None:
                    prices.setdefault(
                        str(instrument_id),
                        _finite(raw_position[key], field=f"position {instrument_id} price"),
                    )
                    break
    for existing in orders:
        if existing.instrument_id in prices:
            continue
        try:
            prices[existing.instrument_id] = _order_reference_price(existing)
        except ExposureBudgetError:
            continue
    return prices


def _order_reference_price(order: OrderIntent) -> float:
    candidates = (
        order.metadata.get("reference_price"),
        order.metadata.get("target_metadata", {}).get("reference_price")
        if isinstance(order.metadata.get("target_metadata"), Mapping)
        else None,
        order.limit_price,
    )
    for candidate in candidates:
        if candidate is not None:
            return _finite(
                candidate, field=f"order {order.order_id} reference price", positive=True
            )
    raise ExposureBudgetError(f"live order {order.order_id} has no reference price")


def _limits(
    *,
    product_id: str,
    product: Mapping[str, Any],
    account: Mapping[str, Any],
    assignment: Mapping[str, Any],
    risk_configuration: Mapping[str, Any],
    equity: float,
) -> dict[str, float]:
    risk_id = str(product.get("risk_policy_id") or "")
    product_limits = _mapping(risk_configuration.get("products"), f"risk products {risk_id}").get(
        risk_id
    )
    if not isinstance(product_limits, Mapping):
        raise ExposureBudgetError(f"risk policy is missing for {product_id}")
    account_limits = _mapping(risk_configuration.get("accounts"), "risk accounts").get(
        str(account.get("account_id") or "")
    )
    if not isinstance(account_limits, Mapping):
        raise ExposureBudgetError("risk policy is missing for the live account")
    sleeve_limits = _mapping(risk_configuration.get("sleeve"), "risk sleeve")
    assignment_cap = _positive_cap(assignment.get("capital_limit"), "assignment capital_limit")
    assignment_risk = _positive_cap(assignment.get("risk_budget"), "assignment risk_budget")
    sleeve_cap = equity * _fraction_cap(
        sleeve_limits.get("maximum_fraction"), "sleeve maximum_fraction"
    )
    product_gross_raw = product_limits.get("maximum_gross", product_limits.get("maximum_exposure"))
    product_gross = equity * _fraction_cap(product_gross_raw, "product gross exposure")
    product_net_raw = product_limits.get("maximum_net")
    instrument_limits = _mapping(risk_configuration.get("instrument"), "risk instrument")
    instrument_cap = equity * _fraction_cap(
        instrument_limits.get("maximum_fraction"), "instrument maximum_fraction"
    )
    leverage = _positive_cap(account.get("maximum_leverage"), "account maximum_leverage")
    account_margin_fraction = _fraction_cap(
        account_limits.get("maximum_margin_fraction"), "account maximum_margin_fraction"
    )
    account_cap = equity * leverage * account_margin_fraction
    limits = {
        "assignment_capital_notional": assignment_cap,
        "assignment_risk_notional": assignment_risk,
        "sleeve_notional": sleeve_cap,
        "product_gross_notional": product_gross,
        "instrument_notional": instrument_cap,
        "account_margin_notional": account_cap,
    }
    if product_net_raw is not None:
        limits["product_net_notional"] = equity * _fraction_cap(
            product_net_raw, "product net exposure"
        )
    return limits


def _assert_reduce_only(order: OrderIntent, current_signed_notional: float, price: float) -> None:
    if abs(current_signed_notional) <= 1e-12:
        raise ExposureBudgetError("reduce-only order has no current position")
    current_side = OrderSide.BUY if current_signed_notional > 0 else OrderSide.SELL
    if order.side is current_side:
        raise ExposureBudgetError("reduce-only order would increase the current position")
    if order.quantity * price > abs(current_signed_notional) + max(
        1e-9, abs(current_signed_notional) * 1e-12
    ):
        raise ExposureBudgetError("reduce-only order exceeds the current position")


def _signed_notional(side: OrderSide, notional: float) -> float:
    return notional if side is OrderSide.BUY else -notional


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExposureBudgetError(f"{field} must be an object")
    return value


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ExposureBudgetError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExposureBudgetError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "finite and non-negative"
        raise ExposureBudgetError(f"{field} must be {qualifier}")
    return result


def _positive_cap(value: object, field: str) -> float:
    return _finite(value, field=field, positive=True)


def _fraction_cap(value: object, field: str) -> float:
    result = _finite(value, field=field)
    if result > 1.0:
        raise ExposureBudgetError(f"{field} must not exceed 1")
    return result
