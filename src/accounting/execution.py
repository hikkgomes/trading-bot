"""Financial postings for an executed fill, independent of order transport."""

from decimal import Decimal

from src.accounting.fees import FeeConversionError, convert_fee, instrument_asset
from src.accounting.ledger import Ledger
from src.domain._codec import canonical_hash
from src.domain.orders import Fill, OrderIntent
from src.domain.positions import Position


def record_execution_costs(
    ledger: Ledger, order: OrderIntent, fill: Fill, *, previous_position: Position
) -> None:
    target_metadata = order.metadata.get("target_metadata")
    target_metadata = target_metadata if isinstance(target_metadata, dict) else {}
    fee_metadata = {
        **dict(target_metadata),
        **dict(order.metadata),
        **dict(fill.metadata),
    }
    try:
        fee_conversion = convert_fee(
            amount=fill.fee,
            fee_asset=fill.fee_asset,
            accounting_asset=ledger.accounting_asset,
            trade_price=fill.price,
            base_asset=instrument_asset(order.instrument_id, "base"),
            quote_asset=instrument_asset(order.instrument_id, "quote"),
            metadata=fee_metadata,
        )
    except FeeConversionError as exc:
        raise ValueError(str(exc)) from exc
    allocations = _strategy_allocations(order.strategy_contributions)
    multi_attribution = len(allocations) > 1
    fill_sign = 1.0 if fill.side.value == "buy" else -1.0
    opposing = previous_position.quantity != 0 and (previous_position.quantity > 0) != (
        fill_sign > 0
    )
    if opposing:
        closing_quantity = min(abs(previous_position.quantity), fill.quantity)
        direction = Decimal("1") if previous_position.quantity > 0 else Decimal("-1")
        reference_price = fill.price
        reference_entry_price = previous_position.average_entry_price
        quote_pnl = (
            (Decimal(str(reference_price)) - Decimal(str(reference_entry_price)))
            * Decimal(str(closing_quantity))
            * direction
        )
        realised_pnl = (
            quote_pnl / Decimal(str(fill.price)) if ledger.accounting_asset == "BTC" else quote_pnl
        )
    else:
        realised_pnl = Decimal("0")
    if ledger.accounting_asset == "BTC" and ":spot:" in order.instrument_id:
        realised_pnl = Decimal("0")
    for strategy, fraction in allocations:
        suffix = (
            ""
            if not multi_attribution
            else ":" + canonical_hash({"strategy": strategy}).removeprefix("sha256:")[:16]
        )
        attribution = {
            "product": ledger.product_id,
            "symbol": order.instrument_id,
            "instrument_id": order.instrument_id,
            "order_id": order.order_id,
            "strategy": strategy,
            "strategy_version_id": strategy if strategy != "ensemble" else None,
            "strategy_contribution_fraction": fraction,
            "sleeve": target_metadata.get("sleeve") or "unassigned",
            "assignment_id": target_metadata.get("assignment_id"),
            **fee_conversion.to_payload(),
        }
        fee_amount = (
            Decimal(str(fee_conversion.accounting_amount))
            if not multi_attribution
            else Decimal(str(fee_conversion.accounting_amount * fraction))
        )
        ledger.record_fee(
            entry_id=f"{fill.fill_id}:fee{suffix}",
            amount=fee_amount,
            occurred_at=fill.occurred_at,
            attribution=attribution,
        )
        slippage = Decimal(str(fill.metadata.get("slippage_cost") or 0))
        if multi_attribution:
            slippage *= Decimal(str(fraction))
        if slippage:
            ledger.append(
                entry_id=f"{fill.fill_id}:slippage{suffix}",
                postings={"memo:execution_shortfall": Decimal(0), "memo:offset": Decimal(0)},
                occurred_at=fill.occurred_at,
                metadata={
                    **attribution,
                    "kind": "slippage",
                    "cost": str(slippage),
                    "pnl_effect": "0",
                    "included_in_fill_price": True,
                },
            )
        allocated_pnl = (
            realised_pnl if not multi_attribution else realised_pnl * Decimal(str(fraction))
        )
        if allocated_pnl:
            ledger.record_realised_pnl(
                entry_id=f"{fill.fill_id}:realised_pnl{suffix}",
                amount=allocated_pnl,
                occurred_at=fill.occurred_at,
                attribution=attribution,
            )


def _strategy_allocations(contributions) -> tuple[tuple[str, float], ...]:
    weights = tuple(
        (str(strategy), abs(float(weight)))
        for strategy, weight in contributions.items()
        if abs(float(weight)) > 0.0
    )
    total = sum(weight for _strategy, weight in weights)
    if total <= 0.0:
        return (("ensemble", 1.0),)
    return tuple((strategy, weight / total) for strategy, weight in sorted(weights))
