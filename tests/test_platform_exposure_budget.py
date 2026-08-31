from __future__ import annotations

import pytest

from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType
from src.domain.positions import Position, PositionStatus
from src.services.exposure_budget import ExposureBudgetError, ExposureBudgetGuard

NOW = "2026-08-31T10:00:00+00:00"


def _order(
    *,
    order_id: str = "new",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 2.0,
    reduce_only: bool = False,
    price: float = 100.0,
) -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        portfolio_id="active-income-portfolio",
        instrument_id="binance:futures:BTCUSDT:USDT",
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        created_at=NOW,
        reduce_only=reduce_only,
        status=OrderStatus.PERSISTED,
        metadata={"reference_price": price, "target_metadata": {"sleeve": "directional"}},
    )


def _position(quantity: float) -> Position:
    return Position(
        portfolio_id="active-income-portfolio",
        instrument_id="binance:futures:BTCUSDT:USDT",
        quantity=quantity,
        average_entry_price=100.0,
        status=PositionStatus.OPEN,
        updated_at=NOW,
    )


def _configuration(
    *,
    assignment_cap: float = 1_000.0,
    risk_budget: float = 1_000.0,
    sleeve_fraction: float = 1.0,
    product_gross: float = 1.0,
    instrument_fraction: float = 1.0,
    account_margin: float = 1.0,
) -> tuple[dict, dict, dict, dict, dict]:
    product = {
        "product_id": "active_income",
        "portfolio_id": "active-income-portfolio",
        "risk_policy_id": "active-income",
    }
    account = {
        "account_id": "futures-account",
        "market": "usdt_futures",
        "maximum_leverage": 1.0,
    }
    assignment = {
        "capital_limit": assignment_cap,
        "risk_budget": risk_budget,
        "sleeve_id": "directional",
    }
    risk = {
        "products": {"active-income": {"maximum_gross": product_gross, "maximum_net": 1.0}},
        "accounts": {"futures-account": {"maximum_margin_fraction": account_margin}},
        "sleeve": {"maximum_fraction": sleeve_fraction},
        "instrument": {"maximum_fraction": instrument_fraction},
    }
    snapshot = {"balances": {"USDT": 1_000.0}, "positions": {}}
    return product, account, assignment, risk, snapshot


def _guard(
    *,
    assignment_cap: float = 1_000.0,
    risk_budget: float = 1_000.0,
    sleeve_fraction: float = 1.0,
    product_gross: float = 1.0,
    instrument_fraction: float = 1.0,
    account_margin: float = 1.0,
) -> tuple[ExposureBudgetGuard, dict, dict, dict, dict, dict]:
    return (ExposureBudgetGuard(),) + _configuration(
        assignment_cap=assignment_cap,
        risk_budget=risk_budget,
        sleeve_fraction=sleeve_fraction,
        product_gross=product_gross,
        instrument_fraction=instrument_fraction,
        account_margin=account_margin,
    )


def _assess(guard: ExposureBudgetGuard, product, account, assignment, risk, snapshot, order):
    return guard.assess(
        product_id="active_income",
        product=product,
        account=account,
        assignment=assignment,
        risk_configuration=risk,
        account_payload=snapshot,
        order=order,
        positions=(_position(2.0),),
        orders=(_order(order_id="pending", quantity=3.0), order),
    )


@pytest.mark.parametrize(
    "keyword",
    (
        "assignment_capital_notional",
        "assignment_risk_notional",
        "sleeve_notional",
        "product_gross_notional",
        "instrument_notional",
        "account_margin_notional",
    ),
)
def test_live_budget_debits_open_and_pending_exposure(keyword: str) -> None:
    limits = {
        "assignment_cap": 1_000.0,
        "risk_budget": 1_000.0,
        "sleeve_fraction": 1.0,
        "product_gross": 1.0,
        "instrument_fraction": 1.0,
        "account_margin": 1.0,
    }
    if keyword == "assignment_capital_notional":
        limits["assignment_cap"] = 600.0
    elif keyword == "assignment_risk_notional":
        limits["risk_budget"] = 600.0
    elif keyword == "sleeve_notional":
        limits["sleeve_fraction"] = 0.6
    elif keyword == "product_gross_notional":
        limits["product_gross"] = 0.6
    elif keyword == "instrument_notional":
        limits["instrument_fraction"] = 0.6
    else:
        limits["account_margin"] = 0.6
    guard, product, account, assignment, risk, snapshot = _guard(**limits)
    with pytest.raises(ExposureBudgetError, match=keyword):
        _assess(guard, product, account, assignment, risk, snapshot, _order())


def test_live_budget_uses_order_reference_price() -> None:
    guard, product, account, assignment, risk, snapshot = _guard()
    assessment = _assess(
        guard,
        product,
        account,
        assignment,
        risk,
        snapshot,
        _order(quantity=1.0, price=125.0),
    )
    assert assessment.reference_price == pytest.approx(125.0)
    assert assessment.proposed_notional == pytest.approx(125.0)


def test_reduce_only_is_allowed_within_current_position_but_cannot_add() -> None:
    guard, product, account, assignment, risk, snapshot = _guard(
        assignment_cap=1.0, risk_budget=1.0, sleeve_fraction=0.01, product_gross=0.01
    )
    allowed = _assess(
        guard,
        product,
        account,
        assignment,
        risk,
        snapshot,
        _order(side=OrderSide.SELL, quantity=1.0, reduce_only=True),
    )
    assert allowed.proposed_notional == 0.0
    with pytest.raises(ExposureBudgetError, match="increase"):
        _assess(
            guard,
            product,
            account,
            assignment,
            risk,
            snapshot,
            _order(side=OrderSide.BUY, quantity=1.0, reduce_only=True),
        )
    with pytest.raises(ExposureBudgetError, match="exceeds the current position"):
        _assess(
            guard,
            product,
            account,
            assignment,
            risk,
            snapshot,
            _order(side=OrderSide.SELL, quantity=3.0, reduce_only=True),
        )
