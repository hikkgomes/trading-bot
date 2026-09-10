from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.domain.forecasts import AlphaForecast, ForecastDirection
from src.domain.orders import OrderIntent, OrderSide, OrderType
from src.domain.portfolios import TargetPosition
from src.domain.positions import Position, PositionStatus
from src.portfolio.optimiser import PortfolioConstraints, optimise_targets
from src.services.exposure_budget import ExposureBudgetGuard

NOW = "2026-09-07T00:00:00+00:00"


def test_persisted_risk_policy_rejects_changed_or_missing_unit_contract(tmp_path):
    from sqlalchemy import select, update

    from src.data.database import PlatformDatabase, risk_policy
    from src.risk.engine import SqlRiskPolicyStore
    from src.risk.policies import install_product_risk_policies
    from src.services.config import load_split_configuration

    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-contract.sqlite3'}")
    database.create_schema()
    store = SqlRiskPolicyStore(database.engine)
    products = {item["product_id"]: item for item in configuration["products"]["products"]}
    identities = install_product_risk_policies(
        store, risk_configuration=configuration["risk"], products=products
    )
    identity = identities["active_income"]
    assert store.resolve((identity,))["product"]["max_gross_fraction"] == 1.5
    with database.engine.connect() as connection:
        original = connection.execute(
            select(risk_policy.c.payload).where(risk_policy.c.id == identity)
        ).scalar_one()
    for altered in (None, {**original["contract"], "schema": "platform.risk_policy/v1"}):
        damaged = {**original, "contract": altered}
        with database.engine.begin() as connection:
            connection.execute(
                update(risk_policy).where(risk_policy.c.id == identity).values(payload=damaged)
            )
        with pytest.raises(ValueError, match="unit contract"):
            store.resolve((identity,))
    database.dispose()


def test_risk_policy_validates_dimensions_without_capping_exposure_multiples():
    from src.risk.policy import convert_legacy_risk_policy, validate_risk_policy
    from src.services.config import load_split_configuration

    policy = load_split_configuration(Path(__file__).resolve().parents[1] / "config")["risk"]
    assert policy["products"]["active-income"]["maximum_gross"] == 1.5
    validate_risk_policy(policy)
    legacy = {
        key: deepcopy(value)
        for key, value in policy.items()
        if key not in {"units", "notional_currency"}
    }
    legacy["schema"] = "risk/v1"
    converted = convert_legacy_risk_policy(legacy, notional_currency="USDT")
    assert converted == policy and legacy["schema"] == "risk/v1"
    from src.services.live_execution import live_authority_configuration_hash

    binding = {
        "product": {},
        "account": {},
        "instrument_payload": {},
        "sleeve_id": "fixture",
        "promotion_policy": {},
        "artefact": {
            "artefact_hash": "fixture",
            "source_commit_hash": "fixture",
            "engine_version": "fixture",
        },
    }
    assert live_authority_configuration_hash(
        **binding, risk_configuration=legacy
    ) != live_authority_configuration_hash(**binding, risk_configuration=converted)
    with pytest.raises(ValueError, match="USDT"):
        convert_legacy_risk_policy(legacy, notional_currency="BTC")
    invalid = deepcopy(policy)
    invalid["instrument"]["maximum_fraction"] = 1.5
    with pytest.raises(ValueError, match="must not exceed one"):
        validate_risk_policy(invalid)
    invalid["units"]["instrument"]["maximum_fraction"] = "equity_multiple"
    with pytest.raises(ValueError, match="canonical dimensions"):
        validate_risk_policy(invalid)
    invalid = deepcopy(policy)
    invalid["strategy"]["maximum_trades_per_day"] = 2.5
    with pytest.raises(ValueError, match="integer count"):
        validate_risk_policy(invalid)
    invalid = deepcopy(policy)
    invalid["products"]["active-income"]["maximum_exposure"] = 1
    with pytest.raises(ValueError, match="one gross exposure"):
        validate_risk_policy(invalid)
    invalid = deepcopy(policy)
    invalid["schema"] = "risk/v1"
    with pytest.raises(ValueError, match="explicit risk/v2"):
        validate_risk_policy(invalid)


def test_compiled_risk_policy_keeps_legacy_history_and_versions_changed_limits(tmp_path):
    from sqlalchemy import select

    from src.data.database import PlatformDatabase, risk_policy
    from src.risk.engine import SqlRiskPolicyStore
    from src.risk.policies import install_product_risk_policies
    from src.risk.policy import risk_policy_identity
    from src.services.config import load_split_configuration

    configuration = load_split_configuration(Path(__file__).resolve().parents[1] / "config")
    products = {row["product_id"]: row for row in configuration["products"]["products"]}
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'policy-versions.sqlite3'}")
    database.create_schema()
    store = SqlRiskPolicyStore(database.engine)
    store.save("legacy", {"product": {"max_gross_fraction": 1.0}})
    installed = install_product_risk_policies(
        store, risk_configuration=configuration["risk"], products=products
    )
    assert (
        install_product_risk_policies(
            store, risk_configuration=configuration["risk"], products=products
        )
        == installed
    )
    original = installed["active_income"]
    assert store.resolve((original,))["product"]["max_gross_fraction"] == 1.5
    assert store.resolve((original,), account_id="binance-futures-main")
    with pytest.raises(ValueError, match="assessed account"):
        store.resolve((original,), account_id="binance-spot-main")
    changed = deepcopy(configuration["risk"])
    changed["products"]["active-income"]["maximum_gross"] = 1.6
    updated = install_product_risk_policies(store, risk_configuration=changed, products=products)[
        "active_income"
    ]
    assert updated != original
    assert store.resolve((updated,))["product"]["max_gross_fraction"] == 1.6
    assert store.resolve((original,))["product"]["max_gross_fraction"] == 1.5
    with pytest.raises(ValueError, match="unit contract"):
        store.resolve(("legacy",))
    assert store.resolve(("legacy",), historical=True)["product"]["max_gross_fraction"] == 1
    with database.engine.connect() as connection:
        record = connection.execute(
            select(risk_policy.c.payload).where(risk_policy.c.id == original)
        ).scalar_one()
    altered = deepcopy(record["limits"])
    altered["product"]["max_gross_fraction"] = 20
    with pytest.raises(ValueError, match="do not match"):
        store.save(original, altered, contract=record["contract"])
    assert (
        risk_policy_identity("active-income", configuration["risk"], account_id="binance-spot-main")
        != original
    )
    database.dispose()


def test_risk_inputs_use_notional_exposure_order_deltas_and_paying_funding_legs():
    from src.services.portfolio_engine import _target_risk_scopes

    targets = tuple(
        TargetPosition(
            portfolio_id="portfolio",
            instrument_id=symbol,
            target_quantity=quantity,
            target_notional=quantity * 100,
            target_fraction=fraction,
            strategy_contributions={"strategy": fraction},
            risk_budget=0.25,
            valid_until=NOW,
        )
        for symbol, quantity, fraction in (("A", 9, 0.25), ("B", -9, -0.25))
    )
    market = {
        symbol: {
            "price": 100,
            "spread_bps": 1,
            "volatility": 0.01,
            "visible_depth": 10000,
            "funding": 0.03,
        }
        for symbol in ("A", "B")
    }
    clean = {
        "trades_today": 0,
        "product_drawdown_fraction": 0,
        "correlations": {},
        "beta": {"A": 1, "B": 1},
        "used_margin_fraction": 0,
        "daily_pnl_fraction": 0,
    }
    arguments = {
        "targets": targets,
        "target_ids": ("A", "B"),
        "current_positions": {"A": 8, "B": -8},
        "equity": 1000,
        "market": market,
        "clean": clean,
    }
    measured = _target_risk_scopes(**arguments)
    assert measured["product"]["inputs"]["gross_fraction"] == pytest.approx(1.8)
    assert measured["product"]["inputs"]["net_fraction"] == 0
    assert measured["instrument"]["inputs"]["order_notional"] == 100
    assert measured["instrument"]["inputs"]["visible_depth_fraction"] == pytest.approx(0.01)
    assert measured["strategy"]["inputs"]["expected_funding_cost_fraction"] == pytest.approx(0.027)
    market["B"]["funding"] = -0.03
    assert _target_risk_scopes(**arguments)["strategy"]["inputs"][
        "expected_funding_cost_fraction"
    ] == pytest.approx(0.054)


def test_fixed_entry_quantity_still_obeys_a_tighter_portfolio_cap():
    forecast = AlphaForecast(
        strategy_version_id="strategy",
        product_id="active_income",
        instrument_id="BTC",
        direction=ForecastDirection.LONG,
        score=1,
        expected_return=None,
        confidence=None,
        horizon_seconds=3600,
        valid_from=NOW,
        valid_until="2026-09-07T01:00:00+00:00",
        target_volatility=0.1,
        maximum_position=0.1,
        metadata={"preserve_entry_quantity": True},
    )
    (target,) = optimise_targets(
        (forecast,),
        prices={"BTC": 100},
        valid_until=forecast.valid_until,
        constraints=PortfolioConstraints(
            portfolio_id="portfolio", equity=1000, max_symbol_fraction=0.05
        ),
        current_quantities={"BTC": 1},
    )
    assert target.target_quantity == pytest.approx(0.5)


def _inputs():
    order = OrderIntent(
        "entry",
        "portfolio",
        "A",
        OrderSide.BUY,
        0.1,
        OrderType.MARKET,
        created_at=NOW,
        metadata={
            "reference_price": 100,
            "target_metadata": {"assignment_id": "assignment-a", "sleeve": "directional"},
        },
    )
    positions = (
        Position(
            "portfolio",
            "A",
            6,
            100,
            PositionStatus.OPEN,
            NOW,
            metadata={"assignment_id": "assignment-a", "sleeve": "directional"},
        ),
        Position(
            "portfolio",
            "B",
            -6,
            100,
            PositionStatus.OPEN,
            NOW,
            metadata={"assignment_id": "assignment-b", "sleeve": "relative-value"},
        ),
    )
    return {
        "product_id": "active_income",
        "product": {"portfolio_id": "portfolio", "risk_policy_id": "income"},
        "account": {"account_id": "account", "market": "usdt_futures", "maximum_leverage": 3},
        "assignment": {
            "id": "assignment-a",
            "sleeve_id": "directional",
            "capital_limit": 1000,
            "risk_budget": 1000,
            "payload": {"capital_limit_currency": "USDT", "risk_budget_unit": "quote_notional"},
        },
        "risk_configuration": {
            "products": {"income": {"maximum_gross": 1.5, "maximum_net": 0.5}},
            "accounts": {"account": {"maximum_margin_fraction": 1}},
            "instrument": {"maximum_fraction": 0.7},
            "sleeve": {"maximum_fraction": 1},
        },
        "account_payload": {"balances": {"USDT": 1000}, "market_prices": {"A": 100, "B": 100}},
        "order": order,
        "positions": positions,
        "orders": (),
    }


def test_gross_multiple_net_and_scoped_caps_use_distinct_exposures():
    result = ExposureBudgetGuard().assess(**_inputs())
    assert result.limits["product_gross_notional"] == 1500
    assert result.gross_notional == 1210
    assert result.net_notional == 10
    assert result.scoped_exposure["instrument"] == 610
    assert result.scoped_exposure["assignment"] == 610
    assert result.scoped_exposure["sleeve"] == 610


def test_assignment_risk_budget_converts_only_its_declared_unit():
    inputs = _inputs()
    inputs["assignment"]["risk_budget"] = 0.7
    inputs["assignment"]["payload"]["risk_budget_unit"] = "equity_fraction"
    fraction = ExposureBudgetGuard().assess(**inputs)
    assert fraction.limits["assignment_risk_notional"] == 700
    inputs["assignment"]["risk_budget"] = 700
    inputs["assignment"]["payload"]["risk_budget_unit"] = "quote_notional"
    amount = ExposureBudgetGuard().assess(**inputs)
    assert amount.limits == fraction.limits
    inputs["assignment"]["payload"].pop("risk_budget_unit")
    with pytest.raises(PermissionError, match="unit is missing"):
        ExposureBudgetGuard().assess(**inputs)
    inputs["assignment"]["payload"]["risk_budget_unit"] = "quote_notional"
    inputs["assignment"]["payload"]["capital_limit_currency"] = "BTC"
    with pytest.raises(PermissionError, match="explicit USDT"):
        ExposureBudgetGuard().assess(**inputs)


def test_unrealised_loss_reduces_the_equity_denominator():
    inputs = _inputs()
    inputs["account_payload"]["market_prices"]["A"] = 90
    inputs["order"] = replace(inputs["order"], metadata={"reference_price": 90})
    result = ExposureBudgetGuard().assess(**inputs)
    assert result.equity == 940
    assert result.limits["product_gross_notional"] == pytest.approx(1410)


def test_malformed_optional_caps_cannot_prevent_verified_reduction():
    inputs = _inputs()
    inputs["risk_configuration"] = {}
    inputs["assignment"]["payload"] = {}
    inputs["order"] = replace(inputs["order"], side=OrderSide.SELL, quantity=1, reduce_only=True)
    result = ExposureBudgetGuard().assess(**inputs)
    assert result.proposed_notional == 0
    assert result.limits == {}
    assert result.equity is None


def test_derived_risk_snapshot_does_not_replace_authoritative_source(tmp_path):
    from src.data.database import PlatformDatabase
    from src.risk.engine import SqlRiskSnapshotStore

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'source-envelope.sqlite3'}")
    database.create_schema()
    store = SqlRiskSnapshotStore(database.engine)
    source = {
        "kind": "positions",
        "product_id": "active_income",
        "observed_at": "2026-09-07T00:00:00+00:00",
        "values": {"positions": {}},
    }
    source_id = store.save(source, created_at=source["observed_at"])
    derived = {
        "kind": "positions",
        "product_id": "active_income",
        "positions": {},
        "canonical_state_id": "fixture-derived-state",
    }
    at = "2026-09-07T00:00:01+00:00"
    derived_id = store.save(derived, created_at=at)
    assert store.latest(kind="positions", product_id="active_income", at=at)[0] == derived_id
    assert store.latest(
        kind="positions", product_id="active_income", at=at, require_source_envelope=True
    ) == (source_id, source)
    assert store.get(derived_id) == derived
    database.dispose()


@pytest.mark.parametrize("correlation", [1.0, -1.0, 0.8])
def test_perfect_cross_instrument_correlation_is_not_discarded(correlation):
    from src.services.portfolio_engine import _maximum_correlation

    assert _maximum_correlation(
        {"A": {"A": 1, "B": correlation}, "B": {"A": correlation, "B": 1}}
    ) == abs(correlation)
    assert _maximum_correlation({"A": {"A": 1}}) == 0


@pytest.mark.parametrize(
    "field, maximum, reason",
    [
        ("maximum_position_notional", 605, "instrument_absolute_notional"),
        ("maximum_order_notional", 9, "order_notional"),
    ],
)
def test_pretrade_guard_enforces_configured_absolute_notional_caps(field, maximum, reason):
    inputs = _inputs()
    inputs["risk_configuration"]["instrument"][field] = maximum
    with pytest.raises(PermissionError, match=reason):
        ExposureBudgetGuard().assess(**inputs)


def test_absolute_instrument_cap_excludes_other_symbols_and_counts_pending_orders():
    inputs = _inputs()
    inputs["risk_configuration"]["instrument"]["maximum_position_notional"] = 650
    inputs["risk_configuration"]["instrument"]["maximum_order_notional"] = 50
    result = ExposureBudgetGuard().assess(**inputs)
    assert result.scoped_exposure["instrument"] == 610
    assert result.scoped_exposure["order"] == 10
    assert result.limits["instrument_absolute_notional"] == 650
    inputs["orders"] = (replace(inputs["order"], order_id="pending", quantity=0.5),)
    with pytest.raises(PermissionError, match="instrument_absolute_notional"):
        ExposureBudgetGuard().assess(**inputs)


@pytest.mark.parametrize("mark, cap, limit", [(90, 9.5, None), (110, 10.5, None), (100, 12.5, 130)])
def test_order_cap_uses_the_larger_reference_market_or_limit_price(mark, cap, limit):
    inputs = _inputs()
    inputs["account_payload"]["market_prices"]["A"] = mark
    inputs["risk_configuration"]["instrument"]["maximum_order_notional"] = cap
    if limit is not None:
        inputs["order"] = replace(inputs["order"], order_type=OrderType.LIMIT, limit_price=limit)
    with pytest.raises(PermissionError, match="order_notional"):
        ExposureBudgetGuard().assess(**inputs)


@pytest.mark.parametrize("mark", [0, -1])
def test_nonpositive_account_marks_cannot_hide_existing_exposure(mark):
    inputs = _inputs()
    inputs["account_payload"]["market_prices"]["A"] = mark
    with pytest.raises(PermissionError, match="positive"):
        ExposureBudgetGuard().assess(**inputs)


def test_unfilled_opposite_orders_cannot_offset_reserved_net_risk():
    inputs = _inputs()
    inputs["positions"] = ()
    inputs["account_payload"]["market_prices"]["C"] = 100
    inputs["orders"] = tuple(
        replace(
            inputs["order"],
            order_id=f"pending-{symbol}",
            instrument_id=symbol,
            side=side,
            quantity=5,
            metadata={
                "reference_price": 100,
                "target_metadata": {"assignment_id": symbol, "sleeve": symbol},
            },
        )
        for symbol, side in (("B", OrderSide.BUY), ("C", OrderSide.SELL))
    )
    # Netting both unfilled orders would show only 10 USDT. A fill of the buy
    # before the sell can expose 510 USDT against the 500 USDT net limit.
    with pytest.raises(PermissionError, match="product_net_notional"):
        ExposureBudgetGuard().assess(**inputs)
    inputs["risk_configuration"]["products"]["income"]["maximum_net"] = 0.6
    assessment = ExposureBudgetGuard().assess(**inputs)
    assert assessment.net_notional == 10
    assert assessment.scoped_exposure["product_net"] == 510
    assert assessment.gross_notional == 1010


def test_book_netting_preserves_absent_forecasts_and_explicit_close_ownership():
    from src.portfolio.book_plans import net_book_changes

    held = {"A": {"SOL": 2}, "B": {}}
    plan = net_book_changes({"A": None, "B": {"BTC": 1}}, held, {"BTC": 100})
    assert plan.desired_positions["A"] == {"SOL": 2}
    assert plan.external_changes == {"BTC": 1}
    assert plan.contributions["BTC"] == ({"book_id": "B", "signed_quantity": 1},)
    closing = net_book_changes({"A": {}, "B": None}, held, {"SOL": 50})
    assert closing.external_changes == {"SOL": -2}
    assert closing.contributions["SOL"] == ({"book_id": "A", "signed_quantity": -2},)


@pytest.mark.parametrize("quantity", [2, 0.0049504950495049506])
def test_fractional_book_allocations_conserve_beyond_default_decimal_context(quantity):
    from decimal import Decimal, localcontext

    from src.portfolio.book_plans import allocate_book_fill

    allocations = allocate_book_fill(
        quantity,
        0.25005,
        [
            {"book_id": "A", "signed_quantity": 1},
            {"book_id": "B", "signed_quantity": 2},
        ],
    )
    with localcontext() as context:
        context.prec = 60
        assert sum(item["signed_quantity"] for item in allocations) == Decimal(str(quantity))
        assert sum(item["fee"] for item in allocations) == Decimal("0.25005")


def test_internal_netting_is_separate_from_conserved_partial_external_fills():
    from decimal import Decimal

    from src.portfolio.book_plans import allocate_book_fill, net_book_changes

    plan = net_book_changes({"A": {"BTC": 10}, "B": {"BTC": -5}, "C": {"BTC": 5}}, {}, {"BTC": 100})
    assert plan.external_changes == {"BTC": 10}
    assert plan.internal_transfers == (
        {
            "kind": "internal_netting",
            "instrument_id": "BTC",
            "from_book": "B",
            "to_book": "A",
            "quantity": 5,
            "price": 100,
        },
    )
    allocations = allocate_book_fill(3, 0.17, plan.contributions["BTC"])
    assert sum(row["signed_quantity"] for row in allocations) == Decimal("3")
    assert sum(row["fee"] for row in allocations) == Decimal("0.17")
    assert {row["book_id"]: row["signed_quantity"] for row in allocations} == {
        "A": Decimal("1.5"),
        "C": Decimal("1.5"),
    }
    with pytest.raises(ValueError, match="opposing intent"):
        allocate_book_fill(
            3,
            0.17,
            [{"book_id": "A", "signed_quantity": 4}, {"book_id": "B", "signed_quantity": -1}],
        )
