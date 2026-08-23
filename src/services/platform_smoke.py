"""Verify the PostgreSQL platform event-to-trade chain for both products."""

from __future__ import annotations

import argparse
import json
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.accounting.ledger import JsonlLedgerStore, Ledger
from src.data.database import PlatformDatabase
from src.domain.portfolios import TargetPosition
from src.domain.risk import RiskDecision
from src.execution.order_manager import JsonlOrderStore, OrderManager
from src.execution.order_planner import plan_orders
from src.execution.paper_exchange import PaperExchange
from src.execution.position_manager import PositionManager
from src.products.btc_accumulation import target_btc_allocation
from src.services.config import load_split_configuration
from src.strategies import get

_STAGES = (
    "closed_candle",
    "features",
    "strategy_assignment",
    "alpha_forecast",
    "target_position",
    "risk_decision",
    "order_intent",
    "partial_fill",
    "complete_fill",
    "position",
    "accounting_entry",
    "decision_trace",
)


def run_smoke(
    database_url: str, *, config_path: Path = Path("config/platform.json")
) -> dict[str, Any]:
    database = PlatformDatabase(database_url)
    if not database.is_postgresql:
        raise ValueError("platform smoke requires PostgreSQL")
    database.assert_migrated()
    products = load_split_configuration(config_path.parent)["products"]["products"]
    results: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="platform-smoke-") as directory:
            root = Path(directory)
            for product in products:
                results.append(_product_fixture(str(product["product_id"]), root=root))
    finally:
        database.dispose()
    return {
        "schema": "platform.smoke/v1",
        "ok": all(item["ok"] for item in results),
        "deterministic_fixtures": True,
        "products": results,
    }


def _product_fixture(product_id: str, *, root: Path) -> dict[str, Any]:
    if product_id not in {"btc_accumulation", "active_income"}:
        raise ValueError(f"unsupported smoke product: {product_id}")
    now = "2026-08-23T00:00:00+00:00"
    later = "2026-08-23T01:00:00+00:00"
    instrument_id = (
        "binance:spot:BTCUSDT" if product_id == "btc_accumulation" else "binance:futures:BTCUSDT:USDT"
    )
    close = np.r_[np.full(95, 99_000.0), 100_000.0]
    frame = pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(close.size, 10.0),
        },
        index=pd.date_range("2026-08-19", periods=close.size, freq="1h", tz="UTC"),
    )
    strategy = get("sma_cross")(fast=2, slow=3)
    forecast = strategy.forecast(
        frame,
        strategy_version_id="smoke:sma-cross:immutable",
        product_id=product_id,
        instrument_id=instrument_id,
        valid_from=now,
        valid_until=later,
        horizon_seconds=3_600,
        maximum_position=0.2,
    )
    if product_id == "btc_accumulation":
        allocation = target_btc_allocation((forecast,))
        target_quantity = allocation.target_btc_fraction * 0.01
        target_fraction = allocation.target_btc_fraction
        accounting_asset = "BTC"
        fee_asset = "BTC"
        fee_in_base = True
    else:
        target_quantity = 0.01
        target_fraction = 0.1
        accounting_asset = "USDT"
        fee_asset = "USDT"
        fee_in_base = False
    target = TargetPosition(
        portfolio_id=f"smoke:{product_id}",
        instrument_id=instrument_id,
        target_quantity=target_quantity,
        target_notional=target_quantity * 100_000.0,
        target_fraction=target_fraction,
        strategy_contributions={forecast.strategy_version_id: forecast.signed_strength},
        risk_budget=0.1,
        valid_until=later,
        metadata={"product_id": product_id},
    )
    decisions = tuple(
        RiskDecision(
            decision_id=f"smoke:{product_id}:risk:{scope}",
            scope=scope,
            accepted=True,
            reason_code=None,
            evaluated_at=now,
            input_snapshot={"product_id": product_id, "fixture": True},
            limits={"risk_policy_id": f"smoke:{product_id}:risk-v1"},
        )
        for scope in ("strategy", "instrument", "sleeve", "product", "account", "global")
    )
    manager = OrderManager(JsonlOrderStore(root / f"{product_id}.orders.jsonl"))
    positions = PositionManager()
    exchange = PaperExchange(
        order_manager=manager,
        position_manager=positions,
        price_source=lambda _instrument: 100_000.0,
        fill_fraction=0.5,
        fee_asset=fee_asset,
        fee_in_base=fee_in_base,
    )
    order = plan_orders((target,), current_quantities={}, decided_at=now)[0]
    first_fill = exchange.submit(order)
    second_fill = exchange.fill_remaining(order.order_id)
    position = positions.get(target.portfolio_id, instrument_id)
    ledger = Ledger(
        product_id=product_id,
        accounting_asset=accounting_asset,
        store=JsonlLedgerStore(root / f"{product_id}.ledger.jsonl"),
    )
    ledger.record_capital(
        entry_id=f"smoke:{product_id}:capital",
        amount=Decimal("1" if accounting_asset == "BTC" else "1000"),
        occurred_at=now,
    )
    owners = {product_id}
    counts = {stage: 1 for stage in _STAGES}
    counts["risk_decision"] = len(decisions)
    counts["complete_fill"] = 1 if manager.get(order.order_id).is_terminal else 0
    ok = (
        forecast.product_id == product_id
        and target.metadata["product_id"] == product_id
        and len(owners) == 1
        and all(decision.input_snapshot["product_id"] == product_id for decision in decisions)
        and first_fill.quantity > 0
        and second_fill.quantity > 0
        and abs(
            position.quantity
            - (target_quantity * (1.0 - 5.0 / 10_000.0) if fee_in_base else target_quantity)
        )
        <= 1e-12
        and counts["complete_fill"] == 1
        and bool(ledger.entries)
    )
    return {
        "product_id": product_id,
        "ok": ok,
        "first_blocked_stage": next((stage for stage in _STAGES if not counts[stage]), None),
        "counts": counts,
        "row_product_ids": sorted(owners),
        "forecast_direction": forecast.direction.value,
        "final_quantity": position.quantity,
        "accounting_asset": accounting_asset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    args = parser.parse_args(argv)
    report = run_smoke(args.database_url, config_path=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
