from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import insert, select

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.database import (
    PlatformDatabase,
    account_snapshot,
    market_input_buffer,
    nav_snapshot,
    position,
    risk_snapshot,
)
from src.domain._codec import canonical_hash
from src.risk.engine import SqlRiskSnapshotStore
from src.services.portfolio_engine import _validate_state_health
from src.services.risk_state import PortfolioRiskCalculator


def test_market_input_buffer_expires_unpinned_updates_and_keeps_pins(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'market-buffer.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    start = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    identities = []
    payloads = []
    for minute in range(21):
        observed = start + dt.timedelta(minutes=minute)
        payload = {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": "binance:futures:BTCUSDT:USDT",
            "source_event_type": "best_bid_ask",
            "source_event_time": observed.isoformat(),
            "availability_time": observed.isoformat(),
            "values": {"close": 100.0 + minute},
        }
        payloads.append(payload)
        identities.append(
            store.save_market_input(
                payload,
                created_at=observed.isoformat(),
                retention_at=observed.isoformat(),
            )
        )

    with database.engine.connect() as connection:
        recent = set(connection.execute(select(market_input_buffer.c.id)).scalars())
    assert recent == set(identities[-16:])

    pinned = store.get(identities[-1])
    assert pinned == payloads[-1]
    later = start + dt.timedelta(minutes=36)
    latest_payload = {
        **payloads[-1],
        "source_event_time": later.isoformat(),
        "availability_time": later.isoformat(),
        "values": {"close": 200.0},
    }
    latest_id = store.save_market_input(
        latest_payload,
        created_at=later.isoformat(),
        retention_at=later.isoformat(),
    )
    with database.engine.connect() as connection:
        recent = set(connection.execute(select(market_input_buffer.c.id)).scalars())
        durable = set(connection.execute(select(risk_snapshot.c.id)).scalars())
    assert recent == {latest_id}
    assert identities[-1] in durable
    assert store.get(identities[-1]) == payloads[-1]
    with pytest.raises(KeyError, match="risk snapshot does not exist"):
        store.get(identities[-2])
    database.dispose()


def test_portfolio_market_uses_buffer_and_rejects_its_expired_quote(tmp_path) -> None:
    from src.services.portfolio_state import DatabasePortfolioSourceService

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'buffer-market-source.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    observed = dt.datetime(2026, 9, 2, tzinfo=dt.UTC)
    payload = {
        "kind": "market_data_input",
        "product_id": "btc_accumulation",
        "instrument_id": "binance:spot:BTCUSDT",
        "source_event_type": "best_bid_ask",
        "source_event_time": observed.isoformat(),
        "availability_time": observed.isoformat(),
        "values": {
            "close": 100.0,
            "spread_bps": 1.0,
            "visible_depth_quote": 10_000.0,
            "volatility": 0.01,
        },
    }
    identity = store.save_market_input(
        payload,
        created_at=observed.isoformat(),
        retention_at=observed.isoformat(),
    )
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={"btc_accumulation": {"account_id": "spot", "dataset_timeframe": "1m"}},
        accounts={"spot": {"market": "spot"}},
    )

    market, _ = source._market("btc_accumulation", observed.isoformat())
    assert market["binance:spot:BTCUSDT"]["price"] == 100.0
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(risk_snapshot.c.id).where(risk_snapshot.c.id == identity)
            ).scalar_one()
            == identity
        )
        assert (
            connection.execute(
                select(market_input_buffer.c.id).where(market_input_buffer.c.id == identity)
            ).scalar_one_or_none()
            is None
        )

    later = observed + dt.timedelta(minutes=16)
    assert source._market("btc_accumulation", later.isoformat())[0] == {}
    database.dispose()


def test_market_input_expiry_deletes_a_bounded_batch(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'bounded-market-cleanup.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    observed = dt.datetime(2026, 9, 3, tzinfo=dt.UTC)
    rows = []
    for index in range(1_001):
        payload = {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": "binance:futures:BTCUSDT:USDT",
            "source_event_type": "trade",
            "source_event_id": str(index),
            "source_event_time": observed.isoformat(),
            "availability_time": observed.isoformat(),
            "values": {"close": 100.0 + index},
        }
        rows.append(
            {
                "id": canonical_hash(payload),
                "product_id": "active_income",
                "instrument_id": "binance:futures:BTCUSDT:USDT",
                "event_type": "trade",
                "created_at": observed.isoformat(),
                "payload": payload,
            }
        )
    with database.engine.begin() as connection:
        connection.execute(insert(market_input_buffer), rows)

    expiry_time = (observed + dt.timedelta(minutes=16)).isoformat()
    assert store.expire_market_inputs(at=expiry_time) == 1_000
    with database.engine.connect() as connection:
        remaining = connection.execute(select(market_input_buffer.c.id)).scalars().all()
    assert len(remaining) == 1
    assert store.expire_market_inputs(at=expiry_time) == 1
    database.dispose()


def test_live_entry_guard_pins_a_fresh_buffered_book(tmp_path) -> None:
    from src.domain.orders import OrderIntent, OrderSide, OrderType
    from src.services.live_execution import ApprovedLiveExecution

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'buffered-live-guard.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    observed_at = "2026-09-04T00:00:00+00:00"
    instrument_id = "binance:futures:BTCUSDT:USDT"
    payload = {
        "kind": "market_data_input",
        "product_id": "active_income",
        "instrument_id": instrument_id,
        "source_event_type": "best_bid_ask",
        "source_event_time": observed_at,
        "availability_time": observed_at,
        "values": {
            "close": 100.0,
            "spread_bps": 1.0,
            "bid_depth_quote": 2_000.0,
            "ask_depth_quote": 2_000.0,
        },
    }
    identity = store.save_market_input(
        payload,
        created_at=observed_at,
        retention_at=observed_at,
    )
    guard = ApprovedLiveExecution.__new__(ApprovedLiveExecution)
    guard.engine = database.engine
    guard.products = {"active_income": {"account_id": "futures"}}
    guard.risk_configuration = {
        "global": {"maximum_market_data_staleness_seconds": 5},
        "instrument": {"maximum_visible_depth_fraction": 0.1, "maximum_slippage_bps": 25},
    }
    intent = OrderIntent(
        "entry",
        "portfolio",
        instrument_id,
        OrderSide.BUY,
        1,
        OrderType.MARKET,
        observed_at,
        valid_until="2026-09-04T00:01:00+00:00",
    )

    guard._assert_market_admission({"product_id": "active_income"}, intent)
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(risk_snapshot.c.id).where(risk_snapshot.c.id == identity)
            ).scalar_one()
            == identity
        )
        assert (
            connection.execute(
                select(market_input_buffer.c.id).where(market_input_buffer.c.id == identity)
            ).scalar_one_or_none()
            is None
        )
    database.dispose()


def test_forward_opening_mark_pins_one_fresh_buffered_input(tmp_path) -> None:
    from src.services.forward_metrics import ForwardEvidenceCollector

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'buffered-forward-mark.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    observed_at = "2026-09-05T00:00:00+00:00"
    instrument_id = "binance:futures:BTCUSDT:USDT"
    identity = store.save_market_input(
        {
            "kind": "market_data_input",
            "product_id": "active_income",
            "instrument_id": instrument_id,
            "source_event_type": "mark_price",
            "source_event_time": observed_at,
            "availability_time": observed_at,
            "values": {"close": 101.0},
        },
        created_at=observed_at,
        retention_at=observed_at,
    )

    assert (
        ForwardEvidenceCollector(database.engine)._opening_market_price(
            "active_income", instrument_id, observed_at
        )
        == 101.0
    )
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                select(risk_snapshot.c.id).where(risk_snapshot.c.id == identity)
            ).scalar_one()
            == identity
        )
        assert (
            connection.execute(
                select(market_input_buffer.c.id).where(market_input_buffer.c.id == identity)
            ).scalar_one_or_none()
            is None
        )
    database.dispose()


def test_unavailable_factor_measurements_block_new_targets() -> None:
    with pytest.raises(ValueError, match="unavailable factor measurements"):
        _validate_state_health(
            {
                "risk_data_available": False,
                "risk_data_missing": ["beta:binance:futures:ETHUSDT:USDT"],
                "unknown_exposure": {},
                "exchange_connected": True,
                "database_healthy": True,
                "execution_drift": False,
                "model_drift": False,
            }
        )


def test_risk_measurements_use_ledger_marks_and_pending_orders(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-state.sqlite3'}")
    database.create_schema()
    account_payload = {
        "account_id": "account-usdt",
        "product_id": "active_income",
        "observed_at": "2026-08-30T00:00:00+00:00",
        "balances": {"USDT": 10_000.0},
    }
    with database.engine.begin() as connection:
        connection.execute(
            account_snapshot.insert().values(
                id=canonical_hash(account_payload),
                account_id="account-usdt",
                observed_at=account_payload["observed_at"],
                source="paper_config",
                content_hash=canonical_hash(account_payload),
                payload=account_payload,
            )
        )
    snapshots = SqlRiskSnapshotStore(database.engine)
    for index, price in enumerate((100.0, 110.0, 99.0)):
        observed_at = f"2026-08-30T00:0{index + 1}:00+00:00"
        snapshots.save(
            {
                "kind": "market_data_input",
                "product_id": "active_income",
                "instrument_id": "BTCUSDT",
                "source_event_type": "candle",
                "source_candle_closed": True,
                "source_timeframe": "1m",
                "source_close_time_ms": 1788048000000 + (index + 1) * 60000 - 1,
                "availability_time": observed_at,
                "values": {
                    "close": price,
                    "spread_bps": 1.0,
                    "visible_depth_quote": 100_000.0,
                    "volatility": 0.1,
                    "funding": 0.0,
                },
            },
            created_at=observed_at,
        )
    ledger = Ledger(
        product_id="active_income",
        accounting_asset="USDT",
        store=SqlLedgerStore(database.engine, product_id="active_income"),
    )
    ledger.record_capital(
        entry_id="capital",
        amount=10_000,
        occurred_at="2026-08-30T00:00:00+00:00",
    )
    ledger.record_realised_pnl(
        entry_id="profit",
        amount=100,
        occurred_at="2026-08-30T00:00:01+00:00",
    )
    ledger.record_realised_pnl(
        entry_id="loss",
        amount=-300,
        occurred_at="2026-08-30T00:00:02+00:00",
    )

    measurements = PortfolioRiskCalculator(database.engine).calculate(
        product_id="active_income",
        account_id="account-usdt",
        product={"portfolio_id": "portfolio-active-income"},
        account={"paper_starting_balances": {"USDT": 10_000.0}},
        balances={"USDT": 9_800.0},
        positions={"BTCUSDT": 1.0},
        open_orders=({"instrument_id": "BTCUSDT", "quantity": 1.0, "price": 99.0},),
        market={
            "BTCUSDT": {
                "price": 99.0,
                "spread_bps": 1.0,
                "visible_depth_quote": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T00:03:00+00:00",
    )

    assert measurements.product_drawdown_fraction > 0.0
    assert measurements.daily_pnl_fraction == -0.02
    assert measurements.global_drawdown_fraction > 0.0
    assert measurements.open_exposure_fraction > 0.0
    assert measurements.pending_exposure_fraction > 0.0
    assert not measurements.risk_data_available
    assert "position_entry:BTCUSDT" in measurements.risk_data_missing
    assert measurements.correlations == {"BTCUSDT": {"BTCUSDT": 1.0}}
    assert measurements.beta == {"BTCUSDT": 1.0}
    assert measurements.clusters == {"BTCUSDT": "base:BTC"}
    reduced = PortfolioRiskCalculator(database.engine).calculate(
        product_id="active_income",
        account_id="account-usdt",
        product={"portfolio_id": "portfolio-active-income"},
        account={"paper_starting_balances": {"USDT": 10_000.0}},
        balances={"USDT": 9_800.0},
        positions={"BTCUSDT": 1.0},
        open_orders=(
            {
                "instrument_id": "BTCUSDT",
                "quantity": 100.0,
                "reduce_only": True,
            },
        ),
        market={
            "BTCUSDT": {
                "price": 99.0,
                "spread_bps": 1.0,
                "visible_depth_quote": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T00:03:00+00:00",
    )
    assert reduced.pending_exposure_fraction == 0.0
    database.dispose()


def test_risk_measurements_resolve_prefixed_btc_benchmark(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-beta.sqlite3'}")
    database.create_schema()
    snapshots = SqlRiskSnapshotStore(database.engine)
    for index, (btc, eth) in enumerate(((100.0, 10.0), (110.0, 11.0), (99.0, 9.9))):
        observed_at = f"2026-08-30T00:0{index + 1}:00+00:00"
        for instrument_id, close in (
            ("binance:futures:BTCUSDT:USDT", btc),
            ("binance:futures:ETHUSDT:USDT", eth),
        ):
            snapshots.save(
                {
                    "kind": "market_data_input",
                    "product_id": "active_income",
                    "instrument_id": instrument_id,
                    "source_event_type": "candle",
                    "source_candle_closed": True,
                    "source_timeframe": "1m",
                    "source_close_time_ms": 1788048000000 + (index + 1) * 60000 - 1,
                    "availability_time": observed_at,
                    "values": {"close": close},
                },
                created_at=observed_at,
            )

    calculator = PortfolioRiskCalculator(database.engine)

    _correlations, beta, missing, receipt = calculator._factor_measurements(
        product_id="active_income",
        at="2026-08-30T00:03:00+00:00",
        instrument_ids=("binance:futures:BTCUSDT:USDT", "binance:futures:ETHUSDT:USDT"),
    )
    assert missing == ()
    assert receipt["estimates"]["beta:binance:futures:ETHUSDT:USDT"]["sample_count"] == 2

    assert beta["binance:futures:BTCUSDT:USDT"] == 1.0
    assert beta["binance:futures:ETHUSDT:USDT"] == pytest.approx(1.0)
    database.dispose()


def test_risk_measurements_mark_missing_factor_history_unavailable(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-missing.sqlite3'}")
    database.create_schema()
    calculator = PortfolioRiskCalculator(database.engine)

    _correlations, _beta_values, missing, _receipt = calculator._factor_measurements(
        product_id="active_income",
        at="2026-08-30T00:03:00+00:00",
        instrument_ids=(
            "binance:futures:BTCUSDT:USDT",
            "binance:futures:ETHUSDT:USDT",
        ),
    )

    assert missing == (
        "beta:binance:futures:ETHUSDT:USDT",
        "correlation:binance:futures:BTCUSDT:USDT:binance:futures:ETHUSDT:USDT",
    )
    database.dispose()


@pytest.mark.parametrize("book", [False, True])
def test_risk_measurements_use_canonical_nav_for_open_position_loss(tmp_path, book) -> None:
    from decimal import Decimal

    from src.accounting.ledger import LedgerEntry, SqlLedgerStore

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-nav.sqlite3'}")
    database.create_schema()
    for observed_at, value in (
        ("2026-08-30T00:00:00+00:00", 1_000.0),
        ("2026-08-30T01:00:00+00:00", 950.0),
    ):
        payload = {
            "product_id": "active_income",
            "accounting_asset": "USDT",
            "nav": value,
            "observed_at": observed_at,
            "components": {},
            **({"portfolio_id": "portfolio-active-income"} if book else {}),
        }
        with database.engine.begin() as connection:
            connection.execute(
                nav_snapshot.insert().values(
                    id=canonical_hash(payload),
                    created_at=observed_at,
                    payload=payload,
                )
            )

    foreign = {**payload, "portfolio_id": "other-book", "nav": 1}
    with database.engine.begin() as connection:
        connection.execute(
            nav_snapshot.insert().values(
                id=canonical_hash(foreign), created_at=foreign["observed_at"], payload=foreign
            )
        )
    SqlLedgerStore(database.engine, product_id="active_income").append(
        LedgerEntry(
            entry_id="other-book-loss",
            product_id="active_income",
            accounting_asset="USDT",
            occurred_at=foreign["observed_at"],
            postings={"expense:fees": Decimal(500), "cash": Decimal(-500)},
            metadata={"pnl_effect": -500, "account_id": "other-account", "book_id": "other-book"},
        )
    )

    measurements = PortfolioRiskCalculator(database.engine).calculate(
        product_id="active_income",
        account_id="account-usdt",
        product={
            "portfolio_id": "portfolio-active-income",
            **({"book_id": "this-book"} if book else {}),
        },
        account={"paper_starting_balances": {"USDT": 1_000.0}},
        balances={"USDT": 950.0},
        positions={},
        open_orders=(),
        market={
            "BTCUSDT": {
                "price": 100.0,
                "spread_bps": 1.0,
                "visible_depth_quote": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T01:00:00+00:00",
    )

    assert measurements.product_drawdown_fraction == pytest.approx(0.05)
    assert measurements.daily_pnl_fraction == pytest.approx(-0.05)
    assert measurements.global_drawdown_fraction == 0
    database.dispose()


def test_latest_flat_position_does_not_reuse_an_older_entry_price(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-position-history.sqlite3'}")
    database.create_schema()
    with database.engine.begin() as connection:
        connection.execute(
            position.insert().values(
                id="position-open",
                created_at="2026-08-30T00:00:00+00:00",
                payload={
                    "portfolio_id": "portfolio-active-income",
                    "instrument_id": "BTCUSDT",
                    "quantity": 1.0,
                    "average_entry_price": 100.0,
                },
            )
        )
        connection.execute(
            position.insert().values(
                id="position-flat",
                created_at="2026-08-30T01:00:00+00:00",
                payload={
                    "portfolio_id": "portfolio-active-income",
                    "instrument_id": "BTCUSDT",
                    "quantity": 0.0,
                    "average_entry_price": 100.0,
                },
            )
        )

    assert (
        PortfolioRiskCalculator(database.engine)._position_terms(
            portfolio_id="portfolio-active-income", at="2026-08-30T01:00:00+00:00"
        )
        == {}
    )
    database.dispose()


@pytest.mark.parametrize("product_id", ["btc_accumulation", "btc-diagnostic"])
def test_btc_risk_exposure_uses_quote_equity_for_quote_notionals(tmp_path, product_id) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'btc-risk-units.sqlite3'}")
    database.create_schema()
    measurements = PortfolioRiskCalculator(database.engine).calculate(
        product_id=product_id,
        account_id="spot-account",
        product={"portfolio_id": "btc-portfolio", "product_family": "btc_accumulation"},
        account={"paper_starting_balances": {"BTC": 0.5, "USDT": 50}},
        balances={"BTC": 0.5, "USDT": 50},
        positions={"binance:spot:BTCUSDT": 0.5},
        open_orders=({"instrument_id": "binance:spot:BTCUSDT", "remaining_quantity": 0.1},),
        market={"binance:spot:BTCUSDT": {"price": 100}},
        at="2026-08-30T00:00:00+00:00",
    )
    # 0.5 BTC + 50 USDT = 100 USDT; exposure is 50 USDT and the order is 10 USDT.
    assert measurements.open_exposure_fraction == pytest.approx(0.5)
    assert measurements.pending_exposure_fraction == pytest.approx(0.1)
    assert measurements.risk_data_available
    database.dispose()


def test_flat_instrument_without_entry_basis_does_not_break_marked_equity(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'flat-risk.sqlite3'}")
    calculator = PortfolioRiskCalculator(database.engine)
    assert (
        calculator._current_equity(
            product_id="active_income",
            balances={"USDT": 1000},
            positions={"BTCUSDT": 1, "ETHUSDT": 0},
            position_terms={"BTCUSDT": (1, 100)},
            market={"BTCUSDT": {"price": 90}},
        )
        == 990
    )
    database.dispose()


@pytest.mark.parametrize(
    "case",
    [
        "aligned",
        "shifted",
        "stale",
        "late",
        "wrong_timeframe",
        "duplicate",
        "flat",
        "conflict",
        "late_subsecond",
    ],
)
def test_factor_estimates_require_matching_available_candle_intervals(tmp_path, case):
    import datetime as dt

    from src.risk.factors import estimate_factors

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'factor-intervals.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    start = dt.datetime(2026, 8, 30, tzinfo=dt.UTC)
    source_ids = set()
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for index, price in enumerate((100, 110, 99), 1):
            minute = index + (case == "shifted" and symbol == "ETHUSDT")
            close_at = start + dt.timedelta(minutes=minute)
            availability = close_at + dt.timedelta(
                minutes=10 if case == "late" and symbol == "ETHUSDT" else 0
            )
            if case == "late_subsecond" and symbol == "ETHUSDT" and index == 3:
                availability = start + dt.timedelta(minutes=4, milliseconds=500)
            payload = {
                "kind": "market_data_input",
                "product_id": "income",
                "instrument_id": symbol,
                "source_event_type": "candle",
                "source_candle_closed": True,
                "source_timeframe": "5m"
                if case == "wrong_timeframe" and symbol == "ETHUSDT"
                else "1m",
                "source_close_time_ms": int(close_at.timestamp() * 1000) - 1,
                "availability_time": availability.isoformat(),
                "values": {"close": 100 if case == "flat" else price},
            }
            source_ids.add(store.save(payload, created_at=close_at.isoformat()))
            if case in {"duplicate", "conflict"}:
                store.save(
                    {
                        **payload,
                        "source_event_id": "duplicate",
                        "values": {"close": price + (case == "conflict")},
                    },
                    created_at=(close_at + dt.timedelta(seconds=1)).isoformat(),
                )
        # An unrelated trade must never replace a candle or create another return.
        store.save(
            {
                "kind": "market_data_input",
                "product_id": "income",
                "instrument_id": symbol,
                "source_event_type": "trade",
                "values": {"close": 10000},
            },
            created_at=start.isoformat(),
        )
    at = start + dt.timedelta(
        minutes=10 if case == "stale" else 4, milliseconds=250 if case == "late_subsecond" else 0
    )
    if case == "conflict":
        with pytest.raises(ValueError, match="conflicting risk candle"):
            estimate_factors(
                database.engine,
                product_id="income",
                instrument_ids=("BTCUSDT", "ETHUSDT"),
                timeframe="1m",
                at=at.isoformat(),
            )
        database.dispose()
        return
    correlations, beta, missing, receipt = estimate_factors(
        database.engine,
        product_id="income",
        instrument_ids=("BTCUSDT", "ETHUSDT"),
        timeframe="1m",
        at=at.isoformat(),
    )
    assert beta["BTCUSDT"] == 1
    if case in {"aligned", "duplicate"}:
        assert missing == ()
        assert beta["ETHUSDT"] == pytest.approx(1)
        assert correlations["BTCUSDT"]["ETHUSDT"] == pytest.approx(1)
        measured = receipt["estimates"]["correlation:BTCUSDT:ETHUSDT"]
        assert measured["sample_count"] == 2
        assert set(measured["source_snapshot_ids"]) == source_ids
        assert receipt["estimates"]["volatility:BTCUSDT"]["value"] == pytest.approx(0.1)
    else:
        assert "beta:ETHUSDT" in missing
        assert "correlation:BTCUSDT:ETHUSDT" in missing
        assert "ETHUSDT" not in beta
    if case == "stale":
        assert "volatility:BTCUSDT" not in receipt["estimates"]
    database.dispose()


def test_market_volatility_uses_closed_bars_and_does_not_refresh_old_estimate(tmp_path):
    import datetime as dt

    from src.services.portfolio_state import DatabasePortfolioSourceService

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'volatility-source.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    start = dt.datetime(2026, 8, 30, tzinfo=dt.UTC)
    instrument_id = "binance:spot:BTCUSDT"
    quote = {
        "kind": "market_data_input",
        "product_id": "btc",
        "instrument_id": instrument_id,
        "values": {"price": 100, "spread_bps": 1, "visible_depth_quote": 1000, "volatility": 0.2},
    }
    supplied_id = store.save(quote, created_at=start.isoformat())
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={"btc": {"account_id": "spot", "dataset_timeframe": "1m"}},
        accounts={"spot": {"market": "spot"}},
    )
    market, _ = source._market("btc", start.isoformat())
    assert market[instrument_id]["volatility"] == 0.2
    assert market[instrument_id]["volatility_estimate"]["source_snapshot_ids"] == [supplied_id]
    later = start + dt.timedelta(minutes=3)
    store.save(
        {**quote, "values": {"price": 120, "spread_bps": 1, "visible_depth_quote": 1000}},
        created_at=later.isoformat(),
    )
    assert source._market("btc", later.isoformat())[0] == {}
    bar_ids = set()
    for index, price in enumerate((100, 110, 99), 1):
        available = start + dt.timedelta(minutes=index)
        bar_ids.add(
            store.save(
                {
                    "kind": "market_data_input",
                    "product_id": "btc",
                    "instrument_id": instrument_id,
                    "source_event_type": "candle",
                    "source_candle_closed": True,
                    "source_timeframe": "1m",
                    "source_close_time_ms": int(available.timestamp() * 1000) - 1,
                    "availability_time": available.isoformat(),
                    "values": {"close": price},
                },
                created_at=available.isoformat(),
            )
        )
    market, _ = source._market("btc", later.isoformat())
    assert market[instrument_id]["volatility"] == pytest.approx(0.1)
    assert set(market[instrument_id]["volatility_estimate"]["source_snapshot_ids"]) == bar_ids
    database.dispose()


def test_new_mark_cannot_refresh_stale_book_fields(tmp_path):
    from src.services.portfolio_state import DatabasePortfolioSourceService

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'stale-book.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    base = {
        "kind": "market_data_input",
        "product_id": "btc",
        "instrument_id": "binance:spot:BTCUSDT",
    }
    old = "2026-09-01T00:00:00+00:00"
    now = "2026-09-01T00:01:00+00:00"
    store.save(
        {
            **base,
            "values": {
                "price": 100,
                "spread_bps": 1,
                "visible_depth_quote": 1000,
                "volatility": 0.01,
            },
        },
        created_at=old,
    )
    store.save({**base, "values": {"price": 101}}, created_at=now)
    source = DatabasePortfolioSourceService(
        engine=database.engine,
        store=store,
        products={"btc": {"account_id": "spot", "dataset_timeframe": "1h"}},
        accounts={"spot": {"market": "spot"}},
    )
    assert source._market("btc", now)[0] == {}
    store.save(
        {
            **base,
            "source_event_time": old,
            "values": {"price": 101, "spread_bps": 1, "visible_depth_quote": 1000},
        },
        created_at=now,
    )
    assert source._market("btc", now)[0] == {}
    store.save(
        {**base, "values": {"price": 101, "spread_bps": 1, "visible_depth_quote": 1000}},
        created_at=now,
    )
    assert source._market("btc", now)[0][base["instrument_id"]]["price"] == 101
    database.dispose()
