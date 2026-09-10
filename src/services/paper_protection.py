"""Paper resting brackets using the shared order, protection and financial ledgers."""

import datetime as dt
from dataclasses import replace
from typing import Any

from sqlalchemy import func, select

from src.data.database import protective_stop, risk_snapshot
from src.domain._codec import canonical_hash, timeframe_seconds, to_primitive
from src.domain.orders import OrderIntent, OrderSide, OrderStatus, OrderType, ProtectionKind
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.paper_exchange import paper_fill
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.stops import ProtectiveStop, SqlStopStore, StopManager, StopStatus
from src.strategies.position_state import PositionRules


def has_paper_protection(engine, instrument_id: str, product_id: str) -> bool:
    latest = (
        select(protective_stop.c.stop_id, func.max(protective_stop.c.sequence).label("sequence"))
        .group_by(protective_stop.c.stop_id)
        .subquery()
    )
    with engine.connect() as connection:
        return (
            connection.execute(
                select(protective_stop.c.id)
                .join(
                    latest,
                    (latest.c.stop_id == protective_stop.c.stop_id)
                    & (latest.c.sequence == protective_stop.c.sequence),
                )
                .where(
                    protective_stop.c.payload["execution_mode"].as_string() == "paper",
                    protective_stop.c.payload["instrument_id"].as_string() == instrument_id,
                    protective_stop.c.payload["paper_parameters"]["scope"]["product_id"].as_string()
                    == product_id,
                    protective_stop.c.status.not_in(["cancelled", "reconciled"]),
                )
                .limit(1)
            ).first()
            is not None
        )


def sync_paper_protection(engine, connection, scope, order, fill, position, request):
    if ":futures:" not in order.instrument_id:
        return
    manager = StopManager(SqlStopStore(engine, connection=connection))
    if order.reduce_only:
        _reduce_paper_protection(engine, connection, manager, order, position, fill.occurred_at)
        return
    metadata = order.metadata.get("target_metadata", order.metadata)
    if not metadata.get("position_rules"):
        return
    rules = PositionRules(**metadata["position_rules"])
    if not request or not request.get("execution_costs"):
        raise ValueError("paper bracket requires its explicit execution cost policy")
    reference = float(fill.metadata["reference_price"])
    sign = 1 if order.side is OrderSide.BUY else -1
    prices = {ProtectionKind.STOP: reference * (1 - sign * rules.stop_loss)}
    if rules.take_profit is not None:
        prices[ProtectionKind.TAKE_PROFIT] = reference * (1 + sign * rules.take_profit)
    for kind, price in prices.items():
        _save_paper_leg(manager, scope, order, fill, request, kind, price)


def _save_paper_leg(manager, scope, order, fill, request, kind, price):
    identity = canonical_hash({"paper_bracket": order.order_id, "kind": kind.value})
    existing = next((stop for stop in manager.all() if stop.stop_id == identity), None)
    if existing is None:
        manager.create(
            ProtectiveStop(
                stop_id=identity,
                portfolio_id=order.portfolio_id,
                instrument_id=order.instrument_id,
                exit_side=OrderSide.SELL if order.side is OrderSide.BUY else OrderSide.BUY,
                quantity=order.filled_quantity,
                protected_quantity=order.filled_quantity,
                trigger_price=price,
                created_at=fill.occurred_at,
                entry_order_id=order.order_id,
                entry_quantity=order.quantity,
                kind=kind,
                execution_mode="paper",
                paper_parameters={
                    "scope": to_primitive(scope),
                    "execution_costs": request["execution_costs"],
                    "fee_in_base": request["fee_in_base"],
                    "fill_fraction": request.get("fill_fraction", 1.0),
                },
            )
        )
    else:
        manager.store.append(
            replace(
                existing, quantity=order.filled_quantity, protected_quantity=order.filled_quantity
            )
        )


def _paper_plans(manager, portfolio_id, instrument_id):
    return sorted(
        (
            stop
            for stop in manager.all()
            if stop.execution_mode == "paper"
            and stop.portfolio_id == portfolio_id
            and stop.instrument_id == instrument_id
            and stop.status not in {StopStatus.CANCELLED, StopStatus.RECONCILED}
        ),
        key=lambda stop: (stop.kind is not ProtectionKind.STOP, stop.created_at, stop.stop_id),
    )


def _reduce_paper_protection(engine, connection, manager, order, position, at):
    remaining = {kind: abs(position.quantity) for kind in ProtectionKind}
    for stop in _paper_plans(manager, order.portfolio_id, order.instrument_id):
        initiating = order.metadata.get("protective_stop_id") == stop.stop_id
        if initiating and stop.entry_order_id:
            _cancel_paper_order(engine, connection, stop.entry_order_id)
        desired = min(stop.quantity, remaining[stop.kind])
        if initiating or stop.status is StopStatus.TRIGGERED:
            state = (
                StopStatus.RECONCILED
                if not position.quantity or (initiating and order.status is OrderStatus.FILLED)
                else StopStatus.TRIGGERED
            )
            updated = replace(
                stop, status=state, triggered_at=stop.triggered_at or at, protected_quantity=0
            )
        elif desired <= 1e-12:
            updated = replace(stop, status=StopStatus.CANCELLED, protected_quantity=0)
        else:
            updated = replace(stop, quantity=desired, protected_quantity=desired)
            remaining[stop.kind] -= desired
        manager.store.append(updated)
        if not position.quantity:
            _cancel_paper_child(engine, connection, stop, at)


def _cancel_paper_child(engine, connection, stop, at):
    _cancel_paper_order(engine, connection, _child_identity(stop))


def _cancel_paper_order(engine, connection, identity):
    orders = OrderManager(SqlOrderStore(engine, connection=connection))
    child = next((order for order in orders.all() if order.order_id == identity), None)
    if child is not None and not child.is_terminal:
        orders.request_cancel(identity)
        orders.cancelled(identity)


def _child_identity(stop):
    return canonical_hash({"paper_protective_child": stop.stop_id})


def _snapshot_bar(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot.get("source_candle_closed") is not True:
        raise ValueError("paper execution requires a closed native market bar")
    values = snapshot["values"]
    bar: dict[str, Any] = {key: float(values[key]) for key in ("open", "high", "low", "close")}
    bar.update(
        open_time_ms=snapshot["source_open_time_ms"],
        close_timestamp=dt.datetime.fromtimestamp(
            snapshot["source_close_time_ms"] / 1000, dt.UTC
        ).isoformat(),
        instrument_id=snapshot["instrument_id"],
        timeframe=snapshot["source_timeframe"],
    )
    return bar


def next_execution_bar(engine, order, *, now: str):
    metadata = order.metadata.get("target_metadata", order.metadata)
    timeframe = metadata.get("timeframe")
    if not metadata.get("position_rules") or not timeframe:
        return None
    earliest = dt.datetime.fromisoformat(order.created_at).timestamp() * 1000
    fills = OrderManager(SqlOrderStore(engine)).fills_for(order.order_id)
    if fills:
        earliest = max(
            earliest,
            max(
                dt.datetime.fromisoformat(fill.occurred_at).timestamp() * 1000 + 1 for fill in fills
            ),
        )
    with engine.connect() as connection:
        rows = connection.execute(
            select(risk_snapshot.c.id, risk_snapshot.c.payload)
            .where(
                risk_snapshot.c.payload["kind"].as_string() == "market_data_input",
                risk_snapshot.c.payload["instrument_id"].as_string() == order.instrument_id,
                risk_snapshot.c.payload["source_candle_closed"].as_boolean().is_(True),
                risk_snapshot.c.created_at <= now,
                risk_snapshot.c.payload["source_open_time_ms"].as_float() >= earliest,
            )
            .order_by(risk_snapshot.c.created_at, risk_snapshot.c.id)
        ).mappings()
        for row in rows:
            if timeframe_seconds(row["payload"]["source_timeframe"]) == timeframe_seconds(
                timeframe
            ):
                return row["id"], _snapshot_bar(row["payload"])
    return None


def barrier_fill_price(stop, bar):
    opened = float(bar["open"])
    if stop.status is StopStatus.TRIGGERED:
        return opened
    adverse = bar["low"] if stop.exit_side is OrderSide.SELL else bar["high"]
    favourable = bar["high"] if stop.exit_side is OrderSide.SELL else bar["low"]
    tested = adverse if stop.kind is ProtectionKind.STOP else favourable
    if not stop.is_triggered_by(float(tested)):
        return None
    return opened if stop.is_triggered_by(opened) else stop.trigger_price


class PaperProtectionRunner:
    def __init__(self, engine, accounts, *, queue_scope=None):
        self.engine, self.accounts, self.queue_scope = engine, accounts, queue_scope

    def execute(self, snapshot_id: str, *, now: str):
        with self.engine.connect() as connection:
            snapshot = connection.execute(
                select(risk_snapshot.c.payload).where(risk_snapshot.c.id == snapshot_id)
            ).scalar_one()
        bar = _snapshot_bar(snapshot)
        manager = StopManager(SqlStopStore(self.engine))
        plans = sorted(
            (
                stop
                for stop in manager.all()
                if stop.execution_mode == "paper"
                and stop.paper_parameters["scope"]["product_id"] == snapshot["product_id"]
                and stop.instrument_id == bar["instrument_id"]
                and stop.created_at <= bar["close_timestamp"]
                and stop.status not in {StopStatus.CANCELLED, StopStatus.RECONCILED}
            ),
            key=lambda stop: (stop.kind is not ProtectionKind.STOP, stop.stop_id),
        )
        fills = [self._trigger(stop, bar, snapshot_id, now) for stop in plans]
        return {
            "reason_code": "paper_protection_evaluated",
            "fills": [fill for fill in fills if fill],
        }

    def _trigger(self, old_stop, bar, snapshot_id, now):
        from src.portfolio.books import paper_account_for_order
        from src.services.financial_application import FinancialApplication, FinancialScope

        scope = FinancialScope(**old_stop.paper_parameters["scope"])
        with self.engine.begin() as connection:
            application = FinancialApplication(
                self.engine, scope, connection=connection, queue_scope=self.queue_scope
            )
            application._lock(connection)
            manager = StopManager(SqlStopStore(self.engine, connection=connection))
            stop = manager.get(old_stop.stop_id)
            position = PositionManager(SqlPositionStore(self.engine, connection=connection)).get(
                stop.portfolio_id, stop.instrument_id
            )
            if (
                stop.status in {StopStatus.CANCELLED, StopStatus.RECONCILED}
                or not position.quantity
            ):
                return None
            price = barrier_fill_price(stop, bar)
            if price is None:
                return None
            orders = OrderManager(SqlOrderStore(self.engine, connection=connection))
            child = self._child(orders, stop, price, abs(position.quantity), bar["close_timestamp"])
            costs = stop.paper_parameters["execution_costs"]

            def build(current):
                fraction = min(1.0, abs(position.quantity) / current.remaining_quantity) * (
                    1.0
                    if current.status is OrderStatus.PARTIALLY_FILLED
                    else stop.paper_parameters["fill_fraction"]
                )
                filled = paper_fill(
                    current,
                    price=price,
                    fee_bps=costs["fee_bps"],
                    slippage_bps=costs["slippage_bps"],
                    fill_fraction=fraction,
                    fee_asset=scope.accounting_asset,
                    fee_in_base=False,
                    now=bar["close_timestamp"],
                )
                return replace(
                    filled,
                    metadata={
                        **filled.metadata,
                        "execution_time_resolution": "bar",
                        "trigger_kind": stop.kind.value,
                        "source_bar_id": snapshot_id,
                    },
                )

            result = application.apply_paper(
                order_id=child.order_id,
                instrument_id=stop.instrument_id,
                request_id=f"paper-bracket:{stop.stop_id}:{snapshot_id}",
                request_hash=canonical_hash({"stop": stop.stop_id, "bar": snapshot_id}),
                build_fill=build,
                account=paper_account_for_order(
                    self.engine,
                    child,
                    self.accounts,
                    product_id=scope.product_id,
                    connection=connection,
                ),
                request_payload={},
            )
            application.acknowledge(result.application_id, at=now)
            return result.fill.fill_id if result.applied else None

    @staticmethod
    def _child(orders, stop, price, quantity, now):
        from src.portfolio.book_plans import protective_book_metadata

        identity = _child_identity(stop)
        existing = next((order for order in orders.all() if order.order_id == identity), None)
        if existing is not None:
            return existing
        parent = orders.get(stop.entry_order_id)
        return orders.create(
            OrderIntent(
                order_id=identity,
                portfolio_id=stop.portfolio_id,
                instrument_id=stop.instrument_id,
                side=stop.exit_side,
                quantity=min(stop.quantity, quantity),
                order_type=OrderType.MARKET,
                created_at=now,
                reduce_only=True,
                strategy_contributions=parent.strategy_contributions,
                metadata={
                    **parent.metadata,
                    "target_metadata": protective_book_metadata(
                        parent.metadata.get("target_metadata", {}),
                        exit_sign=1 if stop.exit_side is OrderSide.BUY else -1,
                    ),
                    "protective_stop_id": stop.stop_id,
                    "reference_price": price,
                    "source": "paper_protective_child",
                },
            )
        )
