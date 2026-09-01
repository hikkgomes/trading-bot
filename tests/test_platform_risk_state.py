from __future__ import annotations

import pytest

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.database import PlatformDatabase, account_snapshot, nav_snapshot
from src.domain._codec import canonical_hash
from src.risk.engine import SqlRiskSnapshotStore
from src.services.portfolio_engine import _validate_state_health
from src.services.risk_state import PortfolioRiskCalculator


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
        observed_at = f"2026-08-30T00:00:0{index}+00:00"
        snapshots.save(
            {
                "kind": "market_data_input",
                "product_id": "active_income",
                "instrument_id": "BTCUSDT",
                "values": {
                    "close": price,
                    "spread_bps": 1.0,
                    "visible_depth": 100_000.0,
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
                "visible_depth": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T00:00:03+00:00",
    )

    assert measurements.product_drawdown_fraction > 0.0
    assert measurements.daily_pnl_fraction == -0.02
    assert measurements.global_drawdown_fraction > 0.0
    assert measurements.open_exposure_fraction > 0.0
    assert measurements.pending_exposure_fraction > 0.0
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
                "visible_depth": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T00:00:03+00:00",
    )
    assert reduced.pending_exposure_fraction == 0.0
    database.dispose()


def test_risk_measurements_resolve_prefixed_btc_benchmark(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-beta.sqlite3'}")
    database.create_schema()
    snapshots = SqlRiskSnapshotStore(database.engine)
    for index, (btc, eth) in enumerate(((100.0, 10.0), (110.0, 11.0), (99.0, 9.9))):
        observed_at = f"2026-08-30T00:00:0{index}+00:00"
        for instrument_id, close in (
            ("binance:futures:BTCUSDT:USDT", btc),
            ("binance:futures:ETHUSDT:USDT", eth),
        ):
            snapshots.save(
                {
                    "kind": "market_data_input",
                    "product_id": "active_income",
                    "instrument_id": instrument_id,
                    "values": {"close": close},
                },
                created_at=observed_at,
            )

    calculator = PortfolioRiskCalculator(database.engine)

    beta = calculator._beta(product_id="active_income", at="2026-08-30T00:00:03+00:00")

    assert beta["binance:futures:BTCUSDT:USDT"] == 1.0
    assert beta["binance:futures:ETHUSDT:USDT"] == pytest.approx(1.0)
    database.dispose()


def test_risk_measurements_mark_missing_factor_history_unavailable(tmp_path) -> None:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-missing.sqlite3'}")
    database.create_schema()
    calculator = PortfolioRiskCalculator(database.engine)

    _correlations, _beta_values, missing = calculator._factor_measurements(
        product_id="active_income",
        at="2026-08-30T00:00:03+00:00",
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


def test_risk_measurements_use_canonical_nav_for_open_position_loss(tmp_path) -> None:
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
        }
        with database.engine.begin() as connection:
            connection.execute(
                nav_snapshot.insert().values(
                    id=canonical_hash(payload),
                    created_at=observed_at,
                    payload=payload,
                )
            )

    measurements = PortfolioRiskCalculator(database.engine).calculate(
        product_id="active_income",
        account_id="account-usdt",
        product={"portfolio_id": "portfolio-active-income"},
        account={"paper_starting_balances": {"USDT": 1_000.0}},
        balances={"USDT": 950.0},
        positions={},
        open_orders=(),
        market={
            "BTCUSDT": {
                "price": 100.0,
                "spread_bps": 1.0,
                "visible_depth": 100_000.0,
                "volatility": 0.1,
                "funding": 0.0,
            }
        },
        at="2026-08-30T01:00:00+00:00",
    )

    assert measurements.product_drawdown_fraction == pytest.approx(0.05)
    assert measurements.daily_pnl_fraction == pytest.approx(-0.05)
    database.dispose()
