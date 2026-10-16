"""One immutable account target from every active strategy book at a cutoff."""

from dataclasses import replace

from sqlalchemy import insert, select, text

from src.data.database import account_portfolio_plan, book_event, job, strategy_book
from src.domain._codec import canonical_hash, to_primitive
from src.domain.portfolios import TargetPosition
from src.execution.position_manager import PositionManager, SqlPositionStore
from src.execution.stops import live_protected_instruments
from src.portfolio.book_plans import net_book_changes
from src.portfolio.books import book_balances, load_book
from src.products.btc_accumulation import BtcAllocationPolicy
from src.research.canonical import SqlActiveStrategyAssignmentRepository


def live_book_targets(
    builder, *, product_id, product, forecasts, clean, current_positions, prices, state_id, at
):
    engine = builder.repository.engine
    assignments = tuple(
        row
        for row in SqlActiveStrategyAssignmentRepository(engine).active_assignments(
            product_id, at=at
        )
        if row["execution_mode"] == "live" and row["portfolio_id"] == product["portfolio_id"]
    )
    if not assignments or any(not row["payload"].get("book_id") for row in assignments):
        raise ValueError("live execution requires reconciled strategy books")
    with engine.begin() as connection:
        _lock(connection, str(product["account_id"]))
        existing = connection.execute(
            select(account_portfolio_plan.c.payload).where(
                account_portfolio_plan.c.product_id == product_id,
                account_portfolio_plan.c.account_id == product["account_id"],
                account_portfolio_plan.c.decision_at == at,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return (
                tuple(TargetPosition(**row) for row in existing["targets"]),
                existing["equity"],
                current_positions,
            )
        _assert_forecasts_settled(connection, assignments, at)
        plan = _calculate(
            builder,
            connection,
            assignments,
            product,
            forecasts,
            clean,
            current_positions,
            prices,
            state_id,
            at,
        )
        identity = canonical_hash(plan)
        targets = tuple(
            replace(target, metadata={**target.metadata, "account_plan_id": identity})
            for target in plan.pop("target_objects")
        )
        payload = {
            **plan,
            "plan_id": identity,
            "targets": [to_primitive(target) for target in targets],
        }
        connection.execute(
            insert(account_portfolio_plan).values(
                id=identity,
                product_id=product_id,
                account_id=product["account_id"],
                decision_at=at,
                payload=to_primitive(payload),
            )
        )
        return targets, plan["equity"], current_positions


def _lock(connection, account_id):
    if connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": "account-plan:" + account_id},
        )


def _assert_forecasts_settled(connection, assignments, at):
    from src.services.portfolio_engine import PortfolioInputsPending

    pending = connection.execute(
        select(job.c.id)
        .where(
            job.c.name == "strategy_evaluation",
            job.c.state.in_(["pending", "running"]),
            job.c.available_at <= at,
            job.c.payload["assignment_id"].as_string().in_([row["id"] for row in assignments]),
        )
        .limit(1)
    ).first()
    if pending is not None:
        raise PortfolioInputsPending("account waits for due book forecasts")


def _calculate(
    builder,
    connection,
    assignments,
    product,
    forecasts,
    clean,
    current_positions,
    prices,
    state_id,
    at,
):
    from src.portfolio.books import book_foreign_fee_cost
    from src.services.portfolio_engine import _marked_income_equity

    engine = builder.repository.engine
    equity = (
        float(clean["balances"].get("BTC", 0)) * prices["binance:spot:BTCUSDT"]
        + float(clean["balances"].get("USDT", 0))
        if product["base_accounting_asset"] == "BTC"
        else _marked_income_equity(clean, clean["balances"], current_positions)
    )
    clean = {**clean, "marked_equity": equity, "marked_equity_currency": "USDT"}
    if clean["open_orders"]:
        from src.services.portfolio_engine import PortfolioInputsPending

        raise PortfolioInputsPending("account execution pending")
    positions = PositionManager(SqlPositionStore(engine, connection=connection))
    books = {
        row["payload"]["book_id"]: load_book(
            engine, row["payload"]["book_id"], connection=connection
        )
        for row in assignments
    }
    balances = {
        identity: book_balances(engine, identity, at=at, connection=connection)
        for identity in books
    }
    before = {identity: positions.current_quantities(identity) for identity in books}
    capital = _capital_proposals(
        engine, connection, assignments, clean, at, reserve_fraction=book_reserve_fraction(product)
    )
    capital_records = {
        identity: connection.execute(
            select(book_event.c.payload).where(book_event.c.id == "live-capital:" + identity)
        ).scalar_one_or_none()
        for identity in books
    }
    capital_records.update({row["book_id"]: row for row in capital})
    projected = {identity: dict(values) for identity, values in balances.items()}
    for allocation in capital:
        for asset, value in allocation["balance_changes"].items():
            values = projected[allocation["book_id"]]
            values[asset] = values.get(asset, 0) + value
    held, desired, originals, owners = {}, {}, {}, {}
    for assignment in assignments:
        book_id = assignment["payload"]["book_id"]
        held[book_id] = dict(before[book_id])
        if product["product_id"] == "btc_accumulation":
            held[book_id]["binance:spot:BTCUSDT"] = projected[book_id].get("BTC", 0)
        selected = tuple(f for f in forecasts if f.metadata.get("book_id") == book_id)
        owners[book_id] = _owner(assignment)
        if not selected:
            desired[book_id] = None
            continue
        targets = _book_targets(
            builder,
            product,
            assignment,
            selected,
            held[book_id],
            projected[book_id],
            clean,
            prices,
            state_id,
            positions,
            capital_records[book_id],
            foreign_fee_cost=book_foreign_fee_cost(engine, book_id, at=at, connection=connection),
        )
        desired[book_id] = {target.instrument_id: target.target_quantity for target in targets}
        originals.update({(book_id, target.instrument_id): target for target in targets})
    deferred = _sequence_bracket_entries(
        product,
        held,
        desired,
        originals,
        live_protected_instruments(engine, product["portfolio_id"], connection=connection),
    )
    net = net_book_changes(desired, held, prices)
    unallocated = _unallocated_positions(product, current_positions, held)
    targets = _account_targets(
        net, owners, originals, unallocated, current_positions, prices, product, forecasts, equity
    )
    targets = tuple(
        replace(
            target,
            metadata={
                **target.metadata,
                "account_id": product["account_id"],
                "book_id": None,
                "execution_environment": builder.account_configuration[product["account_id"]][
                    "environment"
                ],
            },
        )
        for target in targets
    )
    return {
        "schema": "account_portfolio_plan/v1",
        "product_id": product["product_id"],
        "account_id": product["account_id"],
        "decision_at": at,
        "equity": equity,
        "canonical_state_id": state_id,
        "books": owners,
        "forecast_ids": [canonical_hash(to_primitive(f)) for f in forecasts],
        "book_positions": held,
        "book_positions_before": before,
        "account_positions": current_positions,
        "book_balances_before": balances,
        "book_resources": {
            book: {"balances": projected[book], "capital": capital_records[book]} for book in books
        },
        "initial_capital": capital,
        "prices": prices,
        "book_targets": net.desired_positions,
        "internal_transfers": list(net.internal_transfers),
        "entry_deferrals": deferred,
        "target_objects": targets,
    }


def _sequence_bracket_entries(product, held, desired, originals, protected_symbols):
    if product["product_id"] != "active_income":
        return []
    deferred = []
    for (book, symbol), target in originals.items():
        if not target.target_quantity or held[book].get(symbol):
            continue
        incumbents = [
            owner for owner, positions in held.items() if owner != book and positions.get(symbol)
        ]
        protected = (
            symbol in protected_symbols
            or target.metadata.get("position_rules")
            or any(
                originals.get((owner, symbol))
                and originals[owner, symbol].metadata.get("position_rules")
                for owner in incumbents
            )
        )
        if incumbents and protected:
            desired[book][symbol] = 0.0
            deferred.append(
                {
                    "book_id": book,
                    "instrument_id": symbol,
                    "reason_code": "prior_bracket_owner_must_close",
                }
            )
    return deferred


def _owner(assignment):
    return {
        key: assignment[key]
        for key in (
            "id",
            "strategy_version_id",
            "artefact_hash",
            "capital_limit",
            "risk_budget",
            "sleeve_id",
        )
    } | {
        "assignment_id": assignment["id"],
        "book_id": assignment["payload"]["book_id"],
        "capital_limit": float(assignment["capital_limit"]),
        "risk_budget": float(assignment["risk_budget"]),
    }


def book_reserve_fraction(product):
    return BtcAllocationPolicy(
        core_btc_fraction=float(product.get("btc_core_fraction", 1.0)),
        max_tactical_fraction=float(product.get("btc_max_tactical_fraction", 0.0)),
    ).minimum_btc_fraction


def _capital_proposals(engine, connection, assignments, clean, at, *, reserve_fraction=None):
    """Read-only allocation of settled account assets, including inactive books."""
    account_id = load_book(engine, assignments[0]["payload"]["book_id"], connection=connection)[
        "account_id"
    ]
    identities = (
        connection.execute(
            select(strategy_book.c.id).where(
                strategy_book.c.account_id == account_id, strategy_book.c.execution_mode == "live"
            )
        )
        .scalars()
        .all()
    )
    balances = [
        book_balances(engine, identity, at=at, connection=connection) for identity in identities
    ]
    available = {
        asset: float(value) - sum(row.get(asset, 0) for row in balances)
        for asset, value in clean["balances"].items()
    }
    allocations = []
    for assignment in assignments:
        book_id = assignment["payload"]["book_id"]
        identity = "live-capital:" + book_id
        if connection.execute(select(book_event.c.id).where(book_event.c.id == identity)).first():
            continue
        amount = float(assignment["capital_limit"])
        if amount <= 0:
            raise ValueError("live book capital must be positive")
        changes = {"USDT": amount}
        baseline = None
        if assignment["product_id"] == "btc_accumulation":
            if reserve_fraction is None or not 0 <= reserve_fraction <= 1:
                raise ValueError("BTC book allocation requires an explicit core reserve")
            price = float(clean["market"]["binance:spot:BTCUSDT"]["price"])
            quantity = min(max(0.0, available.get("BTC", 0)), amount / price)
            changes = {"BTC": quantity, "USDT": amount - quantity * price}
            baseline = {
                "initial_btc": quantity,
                "initial_quote": changes["USDT"],
                "initial_price": price,
                "initial_btc_equivalent": amount / price,
                "reserve_fraction": reserve_fraction,
                "core_btc_quantity": amount / price * reserve_fraction,
            }
        for asset, value in changes.items():
            if value > available.get(asset, 0) + 1e-9:
                raise ValueError("live book capital exceeds unallocated settled assets")
            available[asset] = available.get(asset, 0) - value
        allocations.append(
            {
                "kind": "approved_initial_capital",
                "book_id": book_id,
                "assignment_id": assignment["id"],
                "balance_changes": changes,
                "capital_limit": amount,
                "source_state_id": canonical_hash(clean),
                "btc_capital_baseline": baseline,
            }
        )
    return allocations


def assert_plan_state(engine, connection, plan, positions, at):
    for book_id, expected in plan["book_positions_before"].items():
        if positions.current_quantities(book_id) != expected:
            raise ValueError("book position changed after the account decision cutoff")
        if (
            book_balances(engine, book_id, at=at, connection=connection)
            != plan["book_balances_before"][book_id]
        ):
            raise ValueError("book balances changed after the account decision cutoff")


def apply_initial_capital(connection, plan, positions):
    """Called only inside the authorised account-plan transaction."""
    for allocation in plan["initial_capital"]:
        book_id = allocation["book_id"]
        connection.execute(
            insert(book_event).values(
                id="live-capital:" + book_id,
                created_at=plan["decision_at"],
                payload=allocation,
            )
        )
        quantity = allocation["balance_changes"].get("BTC", 0)
        if quantity:
            positions.reconcile_position(
                portfolio_id=book_id,
                instrument_id="binance:spot:BTCUSDT",
                quantity=quantity,
                average_entry_price=plan["prices"]["binance:spot:BTCUSDT"],
                updated_at=plan["decision_at"],
                contributions={plan["books"][book_id]["strategy_version_id"]: 1.0},
            )


def _book_targets(
    builder,
    product,
    assignment,
    forecasts,
    current,
    balances,
    clean,
    prices,
    state_id,
    positions,
    capital_record,
    *,
    foreign_fee_cost=0,
):
    book_id = assignment["payload"]["book_id"]
    scoped = {**product, "portfolio_id": book_id, "book_id": book_id}
    if product["product_id"] == "btc_accumulation":
        if not capital_record or not capital_record.get("btc_capital_baseline"):
            raise ValueError("live BTC book requires its own approved initial capital baseline")
        result = builder._btc_targets(
            product_id=product["product_id"],
            product=scoped,
            seed=forecasts[0],
            forecasts=forecasts,
            clean=clean,
            balances=balances,
            current_positions=dict(current),
            prices=prices,
            state_id=state_id,
            capital=capital_record["btc_capital_baseline"],
        )
    else:
        entries = {
            p.instrument_id: p.average_entry_price
            for p in positions.all()
            if p.portfolio_id == book_id
        }
        equity = (
            balances.get("USDT", 0)
            - foreign_fee_cost
            + sum(q * (prices[symbol] - entries[symbol]) for symbol, q in current.items() if q)
        )
        result = builder._active_income_targets(
            product=scoped,
            forecasts=forecasts,
            clean={**clean, "marked_equity": equity, "marked_equity_currency": "USDT"},
            balances=balances,
            current_positions=current,
            market=clean["market"],
            prices=prices,
        )
    return result[0]


def _unallocated_positions(product, actual, books):
    total = {}
    for positions in books.values():
        for symbol, quantity in positions.items():
            total[symbol] = total.get(symbol, 0.0) + quantity
    remainder = {
        symbol: actual.get(symbol, 0) - total.get(symbol, 0) for symbol in set(actual) | set(total)
    }
    if product["product_id"] != "btc_accumulation" and any(
        abs(value) > 1e-9 for value in remainder.values()
    ):
        raise ValueError("live book positions do not reconcile to the exchange account")
    return remainder


def _account_targets(
    net, owners, originals, unallocated, actual, prices, product, forecasts, equity
):
    symbols = set(actual) | {
        symbol for positions in net.desired_positions.values() for symbol in positions
    }
    targets = []
    for symbol in sorted(symbols):
        contributors = [
            {**owners[row["book_id"]], **row} for row in net.contributions.get(symbol, ())
        ]
        active = [
            book for book, positions in net.desired_positions.items() if positions.get(symbol)
        ]
        _assert_bracket_scope(product, symbol, active, originals)
        quantity = unallocated.get(symbol, 0.0) + sum(
            positions.get(symbol, 0.0) for positions in net.desired_positions.values()
        )
        source = next(
            (originals.get((book, symbol)) for book in active if originals.get((book, symbol))),
            None,
        )
        weights = {row["strategy_version_id"]: abs(row["signed_quantity"]) for row in contributors}
        targets.append(
            TargetPosition(
                portfolio_id=product["portfolio_id"],
                instrument_id=symbol,
                target_quantity=quantity,
                target_notional=quantity * prices[symbol],
                target_fraction=quantity * prices[symbol] / equity,
                strategy_contributions=weights,
                risk_budget=sum(float(row["risk_budget"]) for row in owners.values()),
                valid_until=min(f.valid_until for f in forecasts),
                metadata={
                    **(dict(source.metadata) if source else {}),
                    "book_contributions": contributors,
                    "book_authorities": list(owners.values()),
                    "product_id": product["product_id"],
                    **(
                        {"stablecoin_fraction": max(0.0, 1 - quantity * prices[symbol] / equity)}
                        if product["product_id"] == "btc_accumulation"
                        else {}
                    ),
                },
            )
        )
    return tuple(targets)


def _assert_bracket_scope(product, symbol, active, originals):
    if product["product_id"] != "active_income" or len(active) < 2:
        return
    if any(
        originals.get((book, symbol)) and originals[book, symbol].metadata.get("position_rules")
        for book in active
    ):
        raise ValueError(
            "independent native brackets require distinct instruments or separately approved accounts"
        )


def prepare_book_flatten(engine, scope, intent):
    """Persist owned safety reduction and internal netting before venue I/O."""
    from src.accounting.book_allocations import apply_internal_transfers
    from src.execution.order_manager import OrderManager, SqlOrderStore
    from src.execution.position_manager import PositionManager, SqlPositionStore
    from src.services.financial_application import FinancialApplication

    with engine.begin() as connection:
        FinancialApplication(engine, scope)._lock(connection)
        orders = OrderManager(SqlOrderStore(engine, connection=connection))
        existing = next((row for row in orders.all() if row.order_id == intent.order_id), None)
        if existing is not None:
            return existing
        books = tuple(
            connection.execute(
                select(strategy_book.c.payload).where(
                    strategy_book.c.account_id == scope.account_id,
                    strategy_book.c.product_id == scope.product_id,
                    strategy_book.c.execution_mode == "live",
                )
            ).scalars()
        )
        positions = PositionManager(SqlPositionStore(engine, connection=connection))
        held = {
            book["book_id"]: {
                intent.instrument_id: positions.current_quantities(book["book_id"]).get(
                    intent.instrument_id, 0
                )
            }
            for book in books
            if intent.instrument_id in book["instrument_ids"]
        }
        if not held:
            return replace(
                intent,
                metadata={
                    **intent.metadata,
                    "book_attribution_status": "unallocated_legacy_recovery",
                },
            )
        net = net_book_changes(
            {book: {} for book in held},
            held,
            {intent.instrument_id: intent.metadata["reference_price"]},
        )
        signed = intent.quantity * (1 if intent.side.value == "buy" else -1)
        if abs(net.external_changes.get(intent.instrument_id, 0) - signed) > 1e-9:
            raise ValueError("emergency book ownership does not reconcile to the account position")
        owners = {
            book["book_id"]: {
                "book_id": book["book_id"],
                "strategy_version_id": book["strategy_version_id"],
                "assignment_id": None,
                "sleeve_id": book["sleeve_id"],
            }
            for book in books
        }
        contributions = [
            {**owners[row["book_id"]], **row} for row in net.contributions[intent.instrument_id]
        ]
        allocated = replace(
            intent,
            strategy_contributions={
                row["strategy_version_id"]: abs(row["signed_quantity"]) for row in contributions
            },
            metadata={
                **intent.metadata,
                "target_metadata": {
                    "book_contributions": contributions,
                    "account_id": scope.account_id,
                    "execution_environment": scope.environment,
                    "reduction_authority": "emergency_flatten",
                },
            },
        )
        apply_internal_transfers(
            engine,
            connection,
            {
                "plan_id": intent.order_id,
                "decision_at": intent.created_at,
                "books": owners,
                "internal_transfers": net.internal_transfers,
            },
            positions,
        )
        orders.create(allocated)
        return allocated
