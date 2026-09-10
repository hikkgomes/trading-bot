import pytest

from src.accounting.capital import btc_capital_baseline
from src.data.database import PlatformDatabase
from src.domain.strategies import StrategySourceType
from src.research.accounting import BtcResearchAccounting, FuturesResearchAccounting
from src.research.economic_series import account_return_series, btc_nav_series, futures_nav_series
from src.research.executors import (
    _append_semantic_futures_events,
    _stress_report,
    execute_generated_dsl,
)
from src.research.position_execution import PositionTimingUnavailable, managed_futures_events
from src.research.providers import provider_candidate
from src.strategies.position_state import PositionAction
from src.strategies.semantic import RankedTargets


def test_forward_passive_btc_benchmark_stays_in_btc_when_the_quote_price_doubles(tmp_path):
    from src.accounting.paper_account import PaperAccountLedger
    from src.services.forward_metrics import ForwardEvidenceCollector

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'btc-benchmark.db'}")
    database.create_schema()
    at, end = "2026-09-07T00:00:00+00:00", "2026-09-07T01:00:00+00:00"
    PaperAccountLedger(
        engine=database.engine,
        account_id="btc-paper",
        product_id="btc_accumulation",
        accounting_asset="BTC",
        starting_balances={"BTC": 1, "USDT": 0},
    ).ensure_seed(observed_at=at)
    result = ForwardEvidenceCollector(database.engine)._objective_metrics(
        product_id="btc_accumulation",
        assignment={"account_id": "btc-paper"},
        artefact_created_at=at,
        evaluation_time=end,
        ledger_rows=(),
        nav_rows=(),
        market_rows=tuple(
            {"id": str(price), "payload": {"values": {"close": price}}} for price in (100, 200)
        ),
    )
    assert result[:5] == ("BTC", 1, 1, 0, 0)
    database.dispose()


def test_completed_futures_campaigns_ignore_marks_and_partial_reductions():
    import datetime as dt

    from src.research.economic_series import completed_campaigns

    start = dt.datetime(2026, 9, 7, tzinfo=dt.UTC)
    events = [
        {
            "type": "fill",
            "side": "buy",
            "symbol": "BTCUSDT",
            "quantity": 2,
            "price": 100,
            "timestamp": start.isoformat(),
        }
    ]
    events.extend(
        {
            "type": "mark",
            "symbol": "BTCUSDT",
            "mark_price": 100,
            "timestamp": (start + dt.timedelta(minutes=index)).isoformat(),
        }
        for index in range(1, 101)
    )
    for index in (101, 102):
        events.append(
            {
                "type": "fill",
                "side": "sell",
                "symbol": "BTCUSDT",
                "quantity": 1,
                "price": 100,
                "timestamp": (start + dt.timedelta(minutes=index)).isoformat(),
            }
        )
        report = FuturesResearchAccounting().evaluate(initial_cash=1000, events=events)
        count = completed_campaigns(
            {"objective_unit": "USDT", "event_receipts": report.event_receipts}
        )
        assert count == (0 if index == 101 else 1)
    assert completed_campaigns(None) is None


@pytest.mark.parametrize("asset,amount", [("BTC", 0.25), ("USDT", 200)])
def test_btc_partial_reentry_is_not_a_cycle_and_external_flows_are_not_cycle_profit(asset, amount):
    trades = [
        {"timestamp": "2026-09-07T00:00:00+00:00", "side": "sell", "quantity": 0.5, "price": 100},
        {"timestamp": "2026-09-07T01:00:00+00:00", "side": "buy", "quantity": 0.2, "price": 80},
        {"timestamp": "2026-09-07T02:00:00+00:00", "side": "buy", "quantity": 0.3, "price": 80},
    ]
    flows = [
        {
            "type": "deposit",
            "timestamp": "2026-09-07T00:30:00+00:00",
            "asset": asset,
            "amount": amount,
        }
    ]
    partial = BtcResearchAccounting().evaluate(
        initial_btc=1, initial_price=100, trade_events=trades[:2], external_events=flows
    )
    complete = BtcResearchAccounting().evaluate(
        initial_btc=1, initial_price=100, trade_events=trades, external_events=flows
    )
    assert partial.cycles == 0
    assert complete.cycles == 1
    assert complete.round_trip_btc_gain == pytest.approx(0.125)


def test_failed_btc_reentry_closes_when_proceeds_are_spent_and_accounts_for_the_full_loss():
    report = BtcResearchAccounting().evaluate(
        initial_btc=1,
        initial_price=100,
        trade_events=[
            {
                "timestamp": "2026-09-07T00:00:00+00:00",
                "side": "sell",
                "quantity": 0.5,
                "price": 100,
            },
            {
                "timestamp": "2026-09-07T01:00:00+00:00",
                "side": "buy",
                "quantity": 50 / 120,
                "price": 120,
            },
        ],
    )
    assert report.cycles == 1
    assert report.round_trip_btc_gain == pytest.approx(50 / 120 - 0.5)


def test_native_open_entry_is_funded_only_after_entry_and_intrabar_ordering_is_not_invented():
    opened = "2026-09-07T08:00:00+00:00"
    closed = "2026-09-07T09:00:00+00:00"
    row = {
        "open_timestamp": opened,
        "open": 100,
        "close": 100,
        "funding_events": [
            {
                "type": "funding",
                "timestamp": time,
                "symbol": "BTCUSDT",
                "mark_price": 100,
                "funding_rate": 0.01,
            }
            for time in (opened, "2026-09-07T08:30:00+00:00")
        ],
    }
    events = managed_futures_events(
        [row],
        [closed],
        [PositionAction(0, "entry", 1, 100, "entry_pulse")],
        notional=100,
        fee_rate=0,
        slippage_rate=0,
    )
    report = FuturesResearchAccounting().evaluate(initial_cash=1000, events=events)
    assert report.funding_pnl == pytest.approx(-1)
    series = futures_nav_series(report)
    assert [point["timestamp"] for point in series] == [closed]
    assert series[0]["nav"] == pytest.approx(999)
    assert next(event for event in events if event["type"] == "fill")["timestamp"] == opened
    with pytest.raises(PositionTimingUnavailable, match="finer execution data"):
        managed_futures_events(
            [row],
            [closed],
            [PositionAction(0, "exit", 1, 95, "stop")],
            notional=100,
            fee_rate=0,
            slippage_rate=0,
        )


def test_btc_cost_stress_uses_affordable_btc_nav_instead_of_signed_price_returns():
    context = {
        "product_id": "btc_accumulation",
        "initial_btc": 0,
        "initial_stablecoin": 100,
        "initial_price": 100,
        "reserve_fraction": 0,
        "max_tactical_fraction": 1,
        "signals": [1, 1],
        "fee_bps": 100,
        "slippage_bps": 0,
        "market_frame": [
            {"timestamp": f"2026-09-07T0{index}:00:00+00:00", "close": 100} for index in range(2)
        ],
    }
    report = _stress_report(context, [1, 1], [0], fee=2)
    assert report.net_pnl == pytest.approx(100 / 102 - 1)
    assert context["fee_bps"] == 100


@pytest.mark.parametrize("fee_field", ["fee", "fee_amount", "commission"])
@pytest.mark.parametrize(
    "asset,amount,expected_btc_fee", [("BTC", 0.01, 0.02), ("USDT", 1.0, 2 / 101)]
)
def test_cost_stress_keeps_fee_currency_and_accounting_amount_fields(
    fee_field, asset, amount, expected_btc_fee
):
    from src.research.stress import cost_stress_context

    event = {
        "type": "trade",
        "side": "buy",
        "quantity_btc": 1,
        "price": 101,
        "reference_price": 100,
        "fee_asset": asset,
        fee_field: amount,
        "timestamp": "2026-09-07T00:00:00+00:00",
    }
    stressed = cost_stress_context({"btc_trade_events": [event]}, fee=2, slippage=2)
    trade = stressed["btc_trade_events"][0]
    report = BtcResearchAccounting().evaluate(
        initial_btc=0,
        initial_stablecoin=200,
        initial_price=100,
        reserve_fraction=0,
        max_tactical_fraction=1,
        trade_events=[trade],
        marks=[{"timestamp": "2026-09-07T00:01:00+00:00", "price": 102}],
    )
    assert trade["price"] == 102
    assert report.fees_btc == pytest.approx(expected_btc_fee)
    assert event[fee_field] == amount


@pytest.mark.parametrize(
    "side,price,stressed_price",
    [("buy", 101, 102), ("buy", 99, 100), ("sell", 99, 98), ("sell", 101, 100)],
)
def test_adverse_fill_stress_never_rewards_price_improvement(side, price, stressed_price):
    from src.research.stress import cost_stress_context

    context = {
        "trade_events": [
            {"type": "fill", "side": side, "price": price, "reference_price": 100, "fee": 0}
        ]
    }
    assert cost_stress_context(context)["trade_events"][0]["price"] == price
    assert cost_stress_context(context, slippage=2)["trade_events"][0]["price"] == stressed_price


def test_funding_stress_replays_native_payments_once_at_actual_quantity():
    rows = [
        {"timestamp": "2026-09-07T00:00:00+00:00", "close": 100},
        {
            "timestamp": "2026-09-07T01:00:00+00:00",
            "close": 100,
            "funding_events": [
                {
                    "type": "funding",
                    "timestamp": "2026-09-07T01:00:00+00:00",
                    "symbol": "BTCUSDT",
                    "mark_price": 100,
                    "funding_rate": 0.01,
                }
            ],
        },
    ]
    context = {
        "product_id": "active_income",
        "initial_cash": 1000,
        "maximum_position": 0.1,
        "fee_bps": 0,
        "slippage_bps": 0,
        "signals": [1, 1],
        "market_frame": rows,
        "bars": rows,
    }
    report = _stress_report(context, [1, 1], [0], funding=2)
    assert report.funding_pnl == pytest.approx(-2 / 1000)
    assert report.net_pnl == pytest.approx(-2 / 1000)
    assert rows[1]["funding_events"][0]["funding_rate"] == 0.01


def test_dsl_decisions_reach_fee_inclusive_btc_accounting():
    snapshot = "sha256:" + "a" * 64
    candidate = provider_candidate(
        identity="btc-allocation-fixture",
        version="1",
        family="trend",
        product="btc_accumulation",
        thesis_id="sha256:" + "b" * 64,
        lineage_id="sha256:" + "c" * 64,
        provider="generated_dsl",
        source_type=StrategySourceType.GENERATED_DSL,
        universe={"type": "fixed", "symbols": ["BTCUSDT"]},
        source_payload={"rule": {"feature": "close", "operator": "gt", "threshold": 0}},
        dataset_snapshot_hashes=(snapshot,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    rows = [
        {"timestamp": f"2026-09-07T00:0{index}:00+00:00", "close": price}
        for index, price in enumerate((100.0, 110.0, 121.0))
    ]
    result = execute_generated_dsl(
        candidate,
        {
            "product_id": "btc_accumulation",
            "dataset_snapshot_ids": [snapshot],
            "feature_rows": rows,
            "market_frame": rows,
            "returns": [0.1, 0.1],
            "initial_btc": 0.0,
            "initial_stablecoin": 100.0,
            "initial_price": 100.0,
            "reserve_fraction": 0.0,
            "max_tactical_fraction": 1.0,
            "fee_bps": 100.0,
            "slippage_bps": 0.0,
            "bootstrap_iterations": 10,
            "closed_trades": 1000,
            "completed_episodes": 1000,
        },
    )
    accounting = result.evidence["product_accounting"]
    assert result.evidence["sample_evidence"]["closed_trades"] == 0
    assert result.evidence["sample_evidence"]["completed_episodes"] == 0
    assert accounting["objective_value"] == pytest.approx(100 / 101)
    assert accounting["benchmark_value"] == pytest.approx(100 / 101)
    assert accounting["objective_excess"] == pytest.approx(0)
    assert result.evidence["return_ledger"]["net_pnl"] == pytest.approx(100 / 101 - 1)
    assert result.evidence["delay_stress"]["cost_adjusted_return"] == pytest.approx(
        100 / (110 * 1.01) - 1
    )


def test_missing_decision_does_not_remove_market_loss_or_native_funding():
    import datetime as dt
    from types import SimpleNamespace

    from src.research.executors import _temporal_stress

    instrument = "binance:futures:BTCUSDT:USDT"
    start = dt.datetime(2026, 9, 7, tzinfo=dt.UTC)
    rows = [
        {
            "instrument_id": instrument,
            "timestamp": (start + dt.timedelta(minutes=index)).isoformat(),
            "close": 100 if index < 19 else 50,
            "funding_events": [],
            "funding_available": True,
        }
        for index in range(20)
    ]
    rows[-1]["funding_events"] = [
        {
            "type": "funding",
            "symbol": instrument,
            "timestamp": rows[-1]["timestamp"],
            "funding_rate": 0.01,
            "mark_price": 50,
            "event_id": "settlement",
            "mark_source": "fixture",
        }
    ]
    candidate = SimpleNamespace(
        candidate_id="sha256:" + "a" * 64, definition=SimpleNamespace(signal_model={})
    )
    result = _temporal_stress(
        candidate,
        {
            "product_id": "active_income",
            "market_frame": rows,
            "initial_cash": 1000,
            "maximum_position": 0.1,
            "leverage": 1,
            "funding_available": True,
            "fee_bps": 0,
            "slippage_bps": 0,
        },
        [1] * 19 + [0],
        0,
        delay=False,
    )
    assert result["unavailable_decision_indices"] == [19]
    assert result["market_observations"] == 20
    assert result["cost_adjusted_return"] == pytest.approx(-0.0505)


@pytest.mark.parametrize("side,close_side,direction", [("buy", "sell", 1), ("sell", "buy", -1)])
def test_partial_close_preserves_cost_basis_and_funding_is_unique(side, close_side, direction):
    funding = {
        "type": "funding",
        "timestamp": "2026-09-07T00:01:00+00:00",
        "symbol": "BTCUSDT",
        "mark_price": 110,
        "rate": 0.01,
    }
    report = FuturesResearchAccounting().evaluate(
        initial_cash=1000,
        events=[
            {
                "type": "fill",
                "timestamp": "2026-09-07T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "side": side,
                "quantity": 2,
                "price": 100,
                "fee": 1,
            },
            funding,
            funding,
            {
                "type": "fill",
                "timestamp": "2026-09-07T00:02:00+00:00",
                "symbol": "BTCUSDT",
                "side": close_side,
                "quantity": 1,
                "price": 110,
                "fee": 1,
            },
            {
                "type": "mark",
                "timestamp": "2026-09-07T00:03:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 120,
            },
        ],
    )
    assert report.realised_pnl == pytest.approx(direction * 10)
    assert report.unrealised_pnl == pytest.approx(direction * 20)
    assert report.funding_pnl == pytest.approx(-direction * 2.2)
    assert report.final_equity == pytest.approx(998 + direction * 27.8)


def test_execution_price_slippage_is_not_deducted_twice():
    report = FuturesResearchAccounting().evaluate(
        initial_cash=1000,
        events=[
            {
                "type": "fill",
                "timestamp": "2026-09-07T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 1,
                "price": 101,
                "reference_price": 100,
                "fee": 1,
            },
            {
                "type": "mark",
                "timestamp": "2026-09-07T00:01:00+00:00",
                "symbol": "BTCUSDT",
                "mark_price": 100,
            },
        ],
    )
    assert report.final_equity == pytest.approx(998)
    assert report.implementation_shortfall == pytest.approx(2)


def test_short_first_leg_and_removed_member_reconcile_to_cash():
    import datetime as dt

    events, positions = [], {}
    start = "2026-09-07T00:00:00+00:00"
    end = "2026-09-07T01:00:00+00:00"
    for time, targets, prices in (
        (start, {"A": -0.5, "B": 0.5}, {"A": 100, "B": 50}),
        (end, {"B": 0.0}, {"A": 80, "B": 55}),
    ):
        _append_semantic_futures_events(
            events,
            positions,
            RankedTargets(time, targets, targets),
            {"information_time": time, "cross_sectional_values": prices},
            initial_cash=1000,
            execution_bars={
                symbol: {
                    "open": price,
                    "open_timestamp": (
                        dt.datetime.fromisoformat(time) + dt.timedelta(minutes=1)
                    ).isoformat(),
                }
                for symbol, price in prices.items()
            },
        )
    report = FuturesResearchAccounting().evaluate(initial_cash=1000, events=events)
    assert positions == {"A": 0.0, "B": 0.0}
    assert report.realised_pnl == pytest.approx(150)
    assert report.unrealised_pnl == 0
    assert report.final_equity == pytest.approx(1150)


def test_market_move_does_not_rebase_or_sell_btc_core():
    report = BtcResearchAccounting().evaluate(
        initial_btc=1,
        reserve_fraction=0.7,
        max_tactical_fraction=0.3,
        trade_events=[
            {
                "timestamp": "2026-09-07T00:00:00+00:00",
                "side": "sell",
                "quantity_btc": 0.3,
                "price": 100,
            }
        ],
        marks=[
            {"timestamp": "2026-09-07T00:00:00+00:00", "price": 100},
            {"timestamp": "2026-09-07T01:00:00+00:00", "price": 50},
        ],
    )
    assert report.core_btc_quantity == pytest.approx(0.7)
    assert report.nav_series[-1].btc_balance == pytest.approx(0.7)
    assert report.final_btc_nav == pytest.approx(1.3)


def test_deposit_is_not_a_strategy_return():
    report = BtcResearchAccounting().evaluate(
        initial_btc=1,
        trade_events=[],
        external_events=[
            {
                "timestamp": "2026-09-07T01:00:00+00:00",
                "type": "deposit",
                "asset": "BTC",
                "amount": 1,
            }
        ],
        marks=[
            {"timestamp": "2026-09-07T00:00:00+00:00", "price": 100},
            {"timestamp": "2026-09-07T01:00:00+00:00", "price": 100},
        ],
    )
    assert report.final_btc_nav == 2
    assert report.return_fraction == 0
    returns = account_return_series(
        {
            "initial_value": report.initial_btc_nav,
            "nav_series": btc_nav_series(report),
            "fees": 0,
        }
    )
    assert returns.net_pnl == 0


def test_capital_reserve_survives_restart_and_price_change(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'capital.sqlite3'}"
    database = PlatformDatabase(url)
    database.create_schema()
    inputs = {
        "product_id": "btc_accumulation",
        "account_id": "paper",
        "mode": "paper",
        "state_id": "initial-state",
        "observed_at": "2026-09-07T00:00:00+00:00",
        "btc": 1.0,
        "quote": 0.0,
        "price": 100.0,
        "reserve_fraction": 0.7,
    }
    initial = btc_capital_baseline(database.engine, **inputs)
    database.dispose()
    database = PlatformDatabase(url)
    later = btc_capital_baseline(database.engine, **{**inputs, "price": 50.0, "btc": 2.0})
    assert later == initial
    assert later["core_btc_quantity"] == pytest.approx(0.7)
    database.dispose()


def test_live_btc_books_use_their_own_fixed_capital_reserves(tmp_path):
    from types import SimpleNamespace

    from sqlalchemy import insert, select
    from test_platform_execution_recovery import _allocation_books

    from src.data.database import book_event, capital_baseline
    from src.domain._codec import canonical_hash
    from src.domain.forecasts import AlphaForecast, ForecastDirection
    from src.execution.position_manager import PositionManager, SqlPositionStore
    from src.portfolio.account_plans import _book_targets, _capital_proposals, apply_initial_capital
    from src.services.portfolio_engine import DatabasePortfolioTargetBuilder

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'btc-books.db'}")
    database.create_schema()
    books = _allocation_books(database, "btc-account", "btc-core", product="btc_accumulation")
    at = "2026-09-07T00:00:00+00:00"
    symbol = "binance:spot:BTCUSDT"
    assignments = [
        {
            "id": book,
            "product_id": "btc_accumulation",
            "capital_limit": 100,
            "payload": {"book_id": book},
        }
        for book in books
    ]
    with database.engine.begin() as connection:
        connection.execute(
            insert(capital_baseline).values(
                id=canonical_hash(
                    {"product_id": "btc_accumulation", "account_id": "btc-account", "mode": "live"}
                ),
                created_at=at,
                payload={"core_btc_quantity": 1.4},
            )
        )
        allocations = _capital_proposals(
            database.engine,
            connection,
            assignments,
            {"balances": {"BTC": 2, "USDT": 0}, "market": {symbol: {"price": 100}}},
            at,
            reserve_fraction=0.7,
        )
        assert connection.execute(select(book_event.c.id)).first() is None
        positions = PositionManager(SqlPositionStore(database.engine, connection=connection))
        apply_initial_capital(
            connection,
            {
                "initial_capital": allocations,
                "decision_at": at,
                "prices": {symbol: 100},
                "books": {book: {"strategy_version_id": book} for book in books},
            },
            positions,
        )
    builder = DatabasePortfolioTargetBuilder(
        repository=SimpleNamespace(engine=database.engine),
        snapshot_store=None,
        positions=PositionManager(SqlPositionStore(database.engine)),
        product_configuration={},
        account_configuration={},
    )
    for assignment in assignments:
        with database.engine.connect() as connection:
            capital = connection.execute(
                select(book_event.c.payload).where(
                    book_event.c.id == "live-capital:" + assignment["id"]
                )
            ).scalar_one()
        assert capital["btc_capital_baseline"]["core_btc_quantity"] == pytest.approx(0.7)
        forecast = AlphaForecast(
            assignment["id"],
            "btc_accumulation",
            symbol,
            ForecastDirection.SHORT,
            1,
            None,
            None,
            60,
            at,
            "2026-09-07T00:01:00+00:00",
            0.1,
            0.3,
        )
        targets = _book_targets(
            builder,
            {
                "product_id": "btc_accumulation",
                "account_id": "btc-account",
                "execution_mode": "live",
                "btc_core_fraction": 1,
                "btc_max_tactical_fraction": 0.3,
            },
            assignment,
            (forecast,),
            {symbol: 1},
            {"BTC": 1, "USDT": 0},
            {"portfolio_risk_budget": 0.3},
            {symbol: 200},
            "later-mark",
            builder.positions,
            capital,
        )
        assert targets[0].portfolio_id == assignment["id"]
        assert targets[0].target_quantity == pytest.approx(0.7)
        assert targets[0].metadata["core_btc_quantity"] == pytest.approx(0.7)
    database.dispose()


@pytest.mark.parametrize(
    "side,price,expected", [("buy", 110, 0.1), ("buy", 90, -0.1), ("sell", 90, 0.1)]
)
def test_forward_shortfall_uses_quote_prices_and_preserves_improvement(side, price, expected):
    from src.services.forward_metrics import ForwardEvidenceCollector

    rows = (
        {
            "payload": {
                "side": side,
                "quantity": 2,
                "price": price,
                "metadata": {
                    "reference_price": 100,
                    "slippage_cost": 0.001,
                    "slippage_asset": "BTC",
                },
            }
        },
    )
    assert ForwardEvidenceCollector._slippage(rows) == pytest.approx(expected)


def test_missing_execution_reference_cannot_be_hidden_by_a_measured_interval():
    from src.research.canonical import _forward_slippage
    from src.research.outcomes import OutcomeCategory, forward_outcome
    from src.services.forward_metrics import ForwardEvidenceCollector

    rows = ({"payload": {"side": "buy", "quantity": 1, "price": 100, "metadata": {}}},)
    assert ForwardEvidenceCollector._slippage(rows) is None
    assert ForwardEvidenceCollector._slippage(()) is None
    assert (
        _forward_slippage(
            [
                {"fill_ids": ["measured"], "slippage": 0.01},
                {"fill_ids": ["missing"], "slippage": None},
            ]
        )
        is None
    )
    assert (
        forward_outcome(False, "forward_execution_reference_missing")
        is OutcomeCategory.BLOCKED_DATA
    )


@pytest.mark.parametrize("quantity,mark,side", [(2.0, 110.0, "sell"), (-2.0, 90.0, "buy")])
def test_forward_carried_position_does_not_recount_opening_unrealised_profit(quantity, mark, side):
    from src.services.forward_metrics import _futures_assignment_equity

    result = _futures_assignment_equity(
        initial_equity=1000,
        ledger_pnl=9,
        fills=({"payload": {"side": side, "quantity": 1, "price": mark}},),
        market_rows=({"payload": {"values": {"price": mark}}},),
        opening_quantity=quantity,
        opening_entry_price=100,
        opening_mark_price=mark,
    )
    assert result == pytest.approx(999)


def test_forward_equity_retains_losses_beyond_the_starting_allocation():
    from src.services.forward_metrics import _futures_assignment_equity

    result = _futures_assignment_equity(
        initial_equity=10,
        ledger_pnl=0,
        fills=(),
        market_rows=({"payload": {"values": {"price": 80}}},),
        opening_quantity=1,
        opening_entry_price=100,
        opening_mark_price=100,
    )
    assert result == pytest.approx(-10)


@pytest.mark.parametrize("product_id", ["btc_accumulation", "btc-diagnostic"])
def test_paper_btc_fee_denomination_does_not_depend_on_product_name(product_id, tmp_path):
    from decimal import Decimal

    from src.accounting.ledger import Ledger, SqlLedgerStore
    from src.accounting.paper_account import PaperAccountLedger
    from src.data.database import PlatformDatabase
    from src.domain.orders import OrderIntent, OrderSide, OrderType
    from src.execution.order_manager import OrderManager, SqlOrderStore
    from src.execution.paper_exchange import paper_fill
    from src.execution.position_manager import PositionManager, SqlPositionStore
    from src.observability.decision_trace import SqlDecisionTraceStore
    from src.services.order_execution import DatabaseExecutionWorker, DatabasePaperExecutionWorker
    from src.services.scheduler import DatabaseJobQueue

    now = "2026-09-07T00:00:00+00:00"
    order = OrderIntent(
        "btc-fee",
        "portfolio",
        "binance:spot:BTCUSDT",
        OrderSide.BUY,
        0.01,
        OrderType.MARKET,
        created_at=now,
    )
    payload = DatabaseExecutionWorker._order_payload(
        order,
        {"event_id": "event"},
        product_id,
        {"base_accounting_asset": "BTC", "execution_costs": {"fee_bps": 10, "slippage_bps": 0}},
        {order.instrument_id: 50000},
    )
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'fee-units.sqlite3'}")
    database.create_schema()
    queue = DatabaseJobQueue(database.engine)
    queue.register_worker(
        worker_id="paper",
        node_id="fixture",
        role="paper",
        capabilities=("paper_order_submit",),
        observed_at=now,
    )
    orders = OrderManager(SqlOrderStore(database.engine))
    orders.create(order)
    account = PaperAccountLedger(
        engine=database.engine,
        account_id="paper-account",
        product_id=product_id,
        accounting_asset="BTC",
        starting_balances={"BTC": 0, "USDT": 1000},
    )
    ledger = Ledger(
        product_id=product_id,
        accounting_asset="BTC",
        store=SqlLedgerStore(database.engine, product_id=product_id),
    )
    worker = DatabasePaperExecutionWorker(
        queue=queue,
        worker_id="paper",
        order_manager=orders,
        positions=PositionManager(SqlPositionStore(database.engine)),
        ledgers={product_id: ledger},
        trace_store=SqlDecisionTraceStore(database.engine),
        paper_accounts={product_id: account},
    )
    queue.enqueue(job_id="fee-fill", name="paper_order_submit", payload=payload, available_at=now)
    assert worker.run_once(now=now)["reason_code"] == "paper_order_filled"
    filled = orders.fills_for(order.order_id)[0]
    assert filled.fee_asset == "USDT"
    assert filled.fee == pytest.approx(0.5)
    assert account.state().balances == {"BTC": 0.01, "USDT": 499.5}
    ledger.reload()
    assert sum(item.postings.get("expense:fees", 0) for item in ledger.entries) == Decimal(
        "0.00001"
    )
    explicit_base_fee = paper_fill(
        order,
        price=payload["price"],
        fee_bps=payload["execution_costs"]["fee_bps"],
        slippage_bps=0,
        fill_fraction=1,
        fee_asset="BTC",
        fee_in_base=True,
        now=now,
    )
    assert explicit_base_fee.fee_asset == "BTC"
    assert explicit_base_fee.fee == pytest.approx(0.00001)
    assert explicit_base_fee.metadata["base_fee_quantity"] == pytest.approx(0.00001)
    database.dispose()


def test_panel_execution_applies_signed_funding_once_and_distinguishes_leg_capacity():
    import datetime as dt

    from src.research.executors import _semantic_panel_series_context
    from src.research.funding import FundingUnavailable

    opens = [dt.datetime(2026, 9, 7, hour, tzinfo=dt.UTC) for hour in range(3)]
    times = [(time + dt.timedelta(hours=1, milliseconds=-1)).isoformat() for time in opens]
    values = {"A": {"momentum": 1, "close": 100}, "B": {"momentum": -1, "close": 100}}
    funding = [
        {
            "type": "funding",
            "symbol": symbol,
            "timestamp": "2026-09-07T01:30:00+00:00",
            "event_id": f"{symbol}-settlement",
            "mark_price": 100,
            "funding_rate": rate,
        }
        for symbol, rate in (("A", 0.001), ("B", 0.002))
    ]
    rows = [
        {
            "instrument_id": symbol,
            "timestamp": time,
            "open_timestamp": opens[bar_index].isoformat(),
            "open": 100,
            "close": 100,
            "funding_events": [funding[index], funding[index]] if time == times[1] else [],
        }
        for bar_index, time in enumerate(times)
        for index, symbol in enumerate(("A", "B"))
    ]
    context = {
        "initial_cash": 1000,
        "funding_available": True,
        "market_frame": rows,
        "cross_sectional_history": [
            {
                "information_time": time,
                "cross_sectional_values": values,
                "forward_returns": {"A": 0, "B": 0},
            }
            for time in times
        ],
    }
    series, outputs = _semantic_panel_series_context(
        name="relative_momentum", model={}, context=context, instrument_id="A"
    )
    assert (
        min(event["timestamp"] for event in series["futures_events"] if event["type"] == "fill")
        == opens[1].isoformat()
    )
    assert outputs[0].target_fractions == {"A": 0.5, "B": -0.5}
    report = FuturesResearchAccounting().evaluate(
        initial_cash=1000, events=series["futures_events"]
    )
    assert report.funding_pnl == pytest.approx(0.5)
    assert report.final_equity == pytest.approx(1000.5)
    assert report.capacity_violations == 0
    assert sum(event["type"] == "funding" for event in series["futures_events"]) == 2
    with pytest.raises(FundingUnavailable, match="coverage"):
        _semantic_panel_series_context(
            name="relative_momentum",
            model={},
            context={**context, "funding_available": False},
            instrument_id="A",
        )
    empty, _ = _semantic_panel_series_context(
        name="relative_momentum",
        model={},
        context={**context, "market_frame": [{**row, "funding_events": []} for row in rows]},
        instrument_id="A",
    )
    no_settlement = FuturesResearchAccounting().evaluate(
        initial_cash=1000, events=empty["futures_events"]
    )
    assert no_settlement.funding_pnl == 0


def test_panel_decisions_fill_at_next_open_and_final_signal_cannot_trade():
    import copy

    from src.research.executors import DataUnavailableError, _semantic_panel_series_context

    times = ["2026-09-07T00:59:59.999000+00:00", "2026-09-07T01:59:59.999000+00:00"]
    history = [
        {
            "information_time": time,
            "cross_sectional_values": {
                "A": {"momentum": 1, "close": a},
                "B": {"momentum": -1, "close": b},
            },
            "forward_returns": {"A": 0.8, "B": -0.4},
        }
        for time, a, b in ((times[0], 100, 100), (times[1], 180, 60))
    ]
    rows = [
        {
            "instrument_id": symbol,
            "timestamp": times[index],
            "open_timestamp": f"2026-09-07T0{index}:00:00+00:00",
            "open": opened,
            "close": closed,
            "funding_events": [],
        }
        for index, symbol, opened, closed in (
            (0, "A", 100, 100),
            (0, "B", 100, 100),
            (1, "A", 120, 180),
            (1, "B", 80, 60),
        )
    ]
    context = {
        "initial_cash": 1000,
        "funding_available": True,
        "market_frame": rows,
        "cross_sectional_history": history,
    }
    series, _ = _semantic_panel_series_context(
        name="relative_momentum", model={}, context=context, instrument_id="A"
    )
    fills = [event for event in series["futures_events"] if event["type"] == "fill"]
    assert [(event["symbol"], event["price"]) for event in fills] == [("A", 120), ("B", 80)]
    assert [event["quantity"] for event in fills] == pytest.approx([500 / 120, 500 / 80])
    assert {event["timestamp"] for event in fills} == {"2026-09-07T01:00:00+00:00"}
    assert FuturesResearchAccounting().evaluate(
        initial_cash=1000, events=series["futures_events"]
    ).final_equity == pytest.approx(1375)
    changed = copy.deepcopy(context)
    changed["cross_sectional_history"][-1]["cross_sectional_values"]["A"]["momentum"] = -2
    changed_series, _ = _semantic_panel_series_context(
        name="relative_momentum", model={}, context=changed, instrument_id="A"
    )
    assert changed_series["futures_events"] == series["futures_events"]
    for row in changed["market_frame"]:
        row.pop("open_timestamp")
    with pytest.raises(DataUnavailableError, match="opening price and time"):
        _semantic_panel_series_context(
            name="relative_momentum", model={}, context=changed, instrument_id="A"
        )
