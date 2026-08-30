from __future__ import annotations

from src.accounting.ledger import Ledger, SqlLedgerStore
from src.data.database import PlatformDatabase, account_snapshot
from src.domain._codec import canonical_hash
from src.risk.engine import SqlRiskSnapshotStore
from src.services.risk_state import PortfolioRiskCalculator


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
        open_orders=(
            {"instrument_id": "BTCUSDT", "quantity": 1.0, "price": 99.0},
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

    assert measurements.product_drawdown_fraction > 0.0
    assert measurements.daily_pnl_fraction == -0.02
    assert measurements.global_drawdown_fraction > 0.0
    assert measurements.open_exposure_fraction > 0.0
    assert measurements.pending_exposure_fraction > 0.0
    assert measurements.correlations == {"BTCUSDT": {"BTCUSDT": 1.0}}
    assert measurements.beta == {"BTCUSDT": 1.0}
    assert measurements.clusters == {"BTCUSDT": "base:BTC"}
    database.dispose()
