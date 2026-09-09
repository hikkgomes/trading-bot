"""One transaction for a unique fill and every local financial projection."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from sqlalchemy import exists, insert, select, text, update

from src.accounting.execution import record_execution_costs
from src.accounting.fees import instrument_asset
from src.accounting.ledger import Ledger, SqlLedgerStore
from src.accounting.paper_account import PaperAccountLedger
from src.data.database import fill as fill_table
from src.data.database import financial_application, financial_outbox, job
from src.domain._codec import canonical_hash, to_primitive
from src.domain.orders import Fill, OrderIntent, OrderSide, OrderStatus
from src.domain.positions import Position
from src.execution.order_manager import OrderManager, SqlOrderStore
from src.execution.position_manager import PositionManager, SqlPositionStore


@dataclass(frozen=True)
class FinancialScope:
    product_id: str
    account_id: str
    environment: str
    venue: str
    accounting_asset: str


@dataclass(frozen=True)
class FinancialResult:
    application_id: str
    fill: Fill
    order: OrderIntent
    position: Position
    applied: bool
    continuation_job_id: str | None = None


class FinancialApplication:
    def __init__(
        self,
        engine,
        scope: FinancialScope,
        *,
        fault: Callable[[str], None] | None = None,
        queue_scope: str | None = None,
    ):
        if not all((scope.product_id, scope.account_id, scope.venue, scope.accounting_asset)):
            raise ValueError("financial application requires an explicit account scope")
        if scope.environment not in {"paper", "testnet", "production"}:
            raise ValueError("financial environment is unknown")
        self.engine, self.scope, self.fault = engine, scope, fault
        self.queue_scope = queue_scope

    def apply_fill(self, fill: Fill) -> FinancialResult:
        trade = fill.metadata.get("trade_id", fill.metadata.get("exchange_trade_id"))
        if trade is None and self.scope.environment != "paper":
            raise ValueError("exchange fill requires its exact trade identity")
        identity = self._identity(fill.instrument_id, str(trade or fill.fill_id))
        return self._apply(
            identity,
            fill.order_id,
            lambda order: self._execution_reference(order, fill),
            request_hash=_fill_business_hash(fill),
            source_trade_id=str(trade or fill.fill_id),
        )

    def _execution_reference(self, order: OrderIntent, fill: Fill) -> Fill:
        reference = order.metadata.get("reference_price")
        shortfall = None
        if reference is not None:
            reference = float(reference)
            if not isfinite(reference) or reference <= 0:
                raise ValueError("persisted execution reference must be finite and positive")
            sign = 1 if fill.side is OrderSide.BUY else -1
            quote_shortfall = sign * (fill.price - reference) * fill.quantity
            if self.scope.accounting_asset == instrument_asset(fill.instrument_id, "quote"):
                shortfall = quote_shortfall
            elif self.scope.accounting_asset == instrument_asset(fill.instrument_id, "base"):
                shortfall = quote_shortfall / fill.price
        return replace(
            fill,
            metadata={
                **fill.metadata,
                "reference_price": reference,
                "reference_source": "persisted_order" if reference is not None else "unavailable",
                "slippage_cost": shortfall,
                "slippage_asset": self.scope.accounting_asset,
            },
        )

    def assert_order_reconciled(self, order_id: str) -> None:
        with self.engine.connect() as connection:
            _assert_no_legacy_fill_gap(connection, order_id)

    def apply_paper(
        self,
        *,
        order_id: str,
        instrument_id: str,
        request_id: str,
        request_hash: str,
        build_fill: Callable[[OrderIntent], Fill],
        account: PaperAccountLedger | None,
        request_payload: Mapping[str, Any],
    ) -> FinancialResult:
        if self.scope.environment != "paper":
            raise ValueError("paper execution cannot use an exchange environment")
        return self._apply(
            self._identity(instrument_id, request_id),
            order_id,
            build_fill,
            request_hash=request_hash,
            paper_account=account,
            source_trade_id=request_id,
            request_payload=request_payload,
        )

    def _identity(self, instrument_id: str, trade_id: str) -> str:
        return canonical_hash(
            {
                "environment": self.scope.environment,
                "account_id": self.scope.account_id,
                "venue": self.scope.venue,
                "instrument_id": instrument_id,
                "trade_id": trade_id,
            }
        )

    def _apply(
        self,
        identity,
        order_id,
        build_fill,
        *,
        request_hash,
        source_trade_id,
        paper_account=None,
        request_payload=None,
    ) -> FinancialResult:
        with self.engine.begin() as connection:
            self._lock(connection)
            orders = OrderManager(SqlOrderStore(self.engine, connection=connection))
            positions = PositionManager(SqlPositionStore(self.engine, connection=connection))
            order = orders.get(order_id)
            self._assert_order_scope(order)
            existing = connection.execute(
                select(financial_application.c.payload).where(
                    financial_application.c.id == identity,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return self._existing(existing, request_hash, order, positions)
            _assert_no_legacy_fill_gap(connection, order_id)
            incoming = build_fill(order)
            fill = replace(
                incoming,
                fill_id=identity,
                metadata={
                    **incoming.metadata,
                    "source_fill_id": incoming.fill_id,
                    "account_id": self.scope.account_id,
                    "environment": self.scope.environment,
                },
            )
            self._prepare_order(orders, order)
            previous = positions.get(order.portfolio_id, order.instrument_id)
            updated = orders.apply_fill(fill)
            self._checkpoint("order_and_fill")
            target_metadata = order.metadata.get("target_metadata", {})
            position = positions.apply_fill(
                order.portfolio_id,
                fill,
                contributions=dict(order.strategy_contributions),
                assignment_id=target_metadata.get("assignment_id"),
            )
            self._checkpoint("position")
            self._account(connection, order, fill, previous, position, paper_account)
            self._checkpoint("accounting")
            continuation_id = self._schedule_followup(
                connection, identity, updated, fill, request_payload
            )
            payload = {
                "scope": to_primitive(self.scope),
                "request_hash": request_hash,
                "fill": to_primitive(fill),
                "position": to_primitive(position),
                "order_id": order_id,
                "continuation_job_id": continuation_id,
            }
            connection.execute(
                insert(financial_application).values(
                    id=identity,
                    created_at=fill.occurred_at,
                    account_id=self.scope.account_id,
                    environment=self.scope.environment,
                    venue=self.scope.venue,
                    instrument_id=fill.instrument_id,
                    exchange_trade_id=source_trade_id,
                    payload=payload,
                )
            )
            connection.execute(
                insert(financial_outbox).values(
                    id=identity,
                    created_at=fill.occurred_at,
                    processed_at=None,
                    payload={"product_id": self.scope.product_id, "order_id": order_id},
                )
            )
            self._checkpoint("outbox")
        return FinancialResult(identity, fill, updated, position, True, continuation_id)

    def _lock(self, connection) -> None:
        if connection.dialect.name == "postgresql":
            for key in sorted(
                (
                    f"financial:{self.scope.environment}:{self.scope.account_id}",
                    self.scope.product_id,
                )
            ):
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": key}
                )

    def _assert_order_scope(self, order: OrderIntent) -> None:
        metadata = order.metadata.get("target_metadata", order.metadata)
        if metadata.get("account_id") not in {None, self.scope.account_id}:
            raise ValueError("fill account does not match its order")
        if metadata.get("execution_environment") not in {None, self.scope.environment}:
            raise ValueError("fill environment does not match its order")

    def _prepare_order(self, orders: OrderManager, order: OrderIntent) -> None:
        if self.scope.environment == "paper" and order.status in {
            OrderStatus.CREATED,
            OrderStatus.VALIDATED,
            OrderStatus.PERSISTED,
        }:
            orders.persist_for_submission(order.order_id)
        if orders.get(order.order_id).status is OrderStatus.PERSISTED:
            orders.submitted(order.order_id)

    def _account(self, connection, order, fill, previous, position, paper_account) -> None:
        ledger = Ledger(
            product_id=self.scope.product_id,
            accounting_asset=self.scope.accounting_asset,
            store=SqlLedgerStore(
                self.engine, product_id=self.scope.product_id, connection=connection
            ),
        )
        record_execution_costs(ledger, order, fill, previous_position=previous)
        if paper_account is not None:
            account = PaperAccountLedger(
                engine=self.engine,
                account_id=self.scope.account_id,
                product_id=self.scope.product_id,
                accounting_asset=self.scope.accounting_asset,
                starting_balances=paper_account.starting_balances,
                starting_positions=paper_account.starting_positions,
                connection=connection,
            )
            account.record_fill(
                order=order, fill=fill, position=position, previous_position=previous
            )

    def _existing(self, payload, request_hash, order, positions) -> FinancialResult:
        if payload["scope"] != to_primitive(self.scope) or payload["request_hash"] != request_hash:
            raise ValueError("financial event identity collision")
        values = dict(payload["fill"])
        values["side"] = OrderSide(values["side"])
        fill = Fill(**values)
        return FinancialResult(
            fill.fill_id,
            fill,
            order,
            positions.get(order.portfolio_id, order.instrument_id),
            False,
            payload.get("continuation_job_id"),
        )

    def _schedule_followup(self, connection, identity, order, fill, request_payload):
        from src.services.scheduler import DatabaseJobQueue

        queue = DatabaseJobQueue(self.engine, claim_scope=self.queue_scope)
        if self.scope.environment != "paper":
            queue.enqueue_if_absent(
                job_id=f"financial-followup:{identity}",
                name=financial_followup_job_name(self.queue_scope),
                payload={"application_id": identity},
                available_at=fill.occurred_at,
                priority=100,
                connection=connection,
            )
        if request_payload is None or order.status is not OrderStatus.PARTIALLY_FILLED:
            return None
        job_id = "paper-order-continue:" + canonical_hash(
            {"order_id": order.order_id, "filled_quantity": order.filled_quantity}
        ).removeprefix("sha256:")
        queue.enqueue_if_absent(
            job_id=job_id,
            name="paper_order_continue",
            payload={**request_payload, "fill_fraction": 1.0},
            available_at=fill.occurred_at,
            priority=32,
            connection=connection,
        )
        return job_id

    def acknowledge(self, application_id: str, *, at: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(financial_outbox)
                .where(financial_outbox.c.id == application_id)
                .values(processed_at=at)
            )
            connection.execute(
                update(job)
                .where(
                    job.c.id == f"financial-followup:{application_id}",
                    job.c.state == "pending",
                    job.c.attempts == 0,
                )
                .values(state="completed", terminal_reason="financial_followup_delivered_inline")
            )

    def _checkpoint(self, name: str) -> None:
        if self.fault is not None:
            self.fault(name)


def financial_followup_job_name(scope: str | None) -> str:
    return (
        "financial_live_followup"
        if scope is None
        else "financial_live_followup:" + canonical_hash(scope)[7:23]
    )


def _assert_no_legacy_fill_gap(connection, order_id: str) -> None:
    missing = connection.execute(
        select(fill_table.c.id)
        .where(
            fill_table.c.order_id == order_id,
            ~exists(
                select(financial_application.c.id).where(
                    financial_application.c.id == fill_table.c.id
                )
            ),
        )
        .limit(1)
    ).first()
    if missing is not None:
        raise RuntimeError("legacy financial projections require reconciliation before new fills")


def _fill_business_hash(fill: Fill) -> str:
    return canonical_hash(
        {
            key: value
            for key, value in to_primitive(fill).items()
            if key not in {"fill_id", "metadata"}
        }
    )
