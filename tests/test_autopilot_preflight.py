import json
import sys

import pytest

from src.autopilot.approvals import ApprovalLedger, artifact_digest
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.preflight import main, run_preflight
from src.execution.broker import OpenOrderIdentity, Position
from src.execution.config import ExchangeConfig


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "active.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "preflight_report": tmp_path / "preflight.json",
        "testnet_rehearsal_report": tmp_path / "testnet.json",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    if (
        "require_testnet_rehearsal" not in overrides
        and payload["objective"] == "active_income"
        and payload["market"] == "futures"
    ):
        payload["require_testnet_rehearsal"] = True
    return ProductConfig(**payload)


def strategy_artifact(path):
    strategy = {
        "id": "preflight_r1",
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 4,
        "take_profit": 0.02,
        "stop_loss": 0.01,
        "pnl_unit": "usdt",
        "conditions": [
            {
                "feature": "tf_5m_rsi_14",
                "kind": "value_ge",
                "threshold": 50.0,
                "description": "tf_5m_rsi_14 >= 50.0",
            }
        ],
        "risk": {
            "risk_per_trade": 0.003,
            "max_position_fraction": 0.25,
            "daily_stop_loss": -0.02,
            "max_consecutive_losses": 3,
            "cooldown_bars": 24,
            "max_trades_per_day": 4,
        },
        "fees": {"fee_bps": 5.0, "slippage_bps": 2.0},
        "metrics": {"holdout_total_return": 0.03, "dsr_deflated": 0.72},
    }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "futures",
                "symbol": "BTCUSDT",
                "pnl_unit": "usdt",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    return strategy


def btc_strategy_artifact(path):
    strategy = strategy_artifact(path)
    strategy["market"] = "spot"
    strategy["direction"] = "short"
    strategy["pnl_unit"] = "btc"
    strategy["risk"]["daily_stop_loss"] = -0.005
    strategy["risk"]["max_trades_per_day"] = 1
    strategy["metrics"] = {
        "holdout_total_return": 0.03,
        "holdout_excess_return_vs_buy_hold": 0.01,
    }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "market": "spot",
                "symbol": "BTCUSDT",
                "pnl_unit": "btc",
                "paper_trade_allowed": True,
                "live_allowed": True,
                "promotion_eligible": True,
                "strategies": [strategy],
            }
        ),
        encoding="utf-8",
    )
    return strategy


def set_live_env(monkeypatch):
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("FUTURES_EXCHANGE", "binanceusdm")
    monkeypatch.setenv("EXCHANGE_API_KEY", "key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")


class FakeBroker:
    name = "fake"

    def __init__(self, position_qty=0.0):
        self.position_qty = position_qty

    def get_price(self, symbol):
        return 100.0

    def get_balance(self):
        return 1000.0

    def get_position(self, symbol):
        return Position(
            symbol=symbol, qty=self.position_qty, avg_price=90.0 if self.position_qty else 0.0
        )

    def supports_native_protective_stops(self):
        return True

    def verify_one_way_position_mode(self, symbol):
        return True

    def list_open_orders(self, symbol, *, conditional):
        return ()


def test_preflight_no_products_selected(tmp_path):
    report = run_preflight(AutopilotConfig(products=[]))

    assert report["ok"] is False
    assert "No products selected" in report["errors"][0]


def test_preflight_requires_approval(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    strategy_artifact(tmp_path / "active.json")
    cfg = AutopilotConfig(approval_ledger=tmp_path / "approvals.json", products=[product(tmp_path)])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when approval is missing")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    assert report["ok"] is False
    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert checks["approval_gate"]["ok"] is False
    assert checks["broker_constructed"]["detail"] == {
        "skipped": True,
        "reason": "prerequisite_checks_failed",
    }


def test_preflight_rejects_malformed_strategy_artifact_before_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    artifact.write_text("[]", encoding="utf-8")
    cfg = AutopilotConfig(approval_ledger=tmp_path / "approvals.json", products=[product(tmp_path)])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when strategy artifact is malformed")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    assert report["ok"] is False
    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert checks["strategy_fingerprints"]["ok"] is False
    assert "must be a JSON object" in checks["strategy_fingerprints"]["error"]
    assert checks["approval_gate"]["ok"] is False
    assert "must be a JSON object" in checks["approval_gate"]["error"]
    assert checks["broker_constructed"]["detail"] == {
        "skipped": True,
        "reason": "prerequisite_checks_failed",
    }


def test_preflight_rejects_invalid_json_strategy_artifact_before_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    artifact.write_text('{"version": 1,', encoding="utf-8")
    cfg = AutopilotConfig(approval_ledger=tmp_path / "approvals.json", products=[product(tmp_path)])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when strategy artifact JSON is invalid")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    assert report["ok"] is False
    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert checks["strategy_fingerprints"]["ok"] is False
    assert "must be valid JSON" in checks["strategy_fingerprints"]["error"]
    assert checks["strategy_policy"]["ok"] is False
    assert "must be valid JSON" in checks["strategy_policy"]["error"]
    assert checks["approval_gate"]["ok"] is False
    assert "must be valid JSON" in checks["approval_gate"]["error"]
    assert checks["broker_constructed"]["detail"] == {
        "skipped": True,
        "reason": "prerequisite_checks_failed",
    }


def test_preflight_passes_with_approval_and_env(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", lambda product: FakeBroker())

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    assert report["ok"] is True
    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    product_payload = report["products"][0]["product"]
    assert product_payload["enabled"] is True
    assert product_payload["execution_mode"] == "live"
    assert product_payload["strategies_path"] == str(artifact)
    assert product_payload["require_preflight"] is True
    assert product_payload["preflight_report"] == str(active_product.preflight_report)
    assert product_payload["preflight_max_age_seconds"] == active_product.preflight_max_age_seconds
    assert product_payload["require_testnet_rehearsal"] is True
    assert product_payload["testnet_rehearsal_report"] == str(
        active_product.testnet_rehearsal_report
    )
    assert (
        product_payload["testnet_rehearsal_max_age_seconds"]
        == active_product.testnet_rehearsal_max_age_seconds
    )
    assert checks["exchange_environment"]["ok"] is True
    assert checks["exchange_environment"]["detail"] == {
        "exchange": "binanceusdm",
        "market_type": "futures",
        "testnet": False,
        "require_testnet": False,
        "quote_asset": "USDT",
        "account_fingerprint": ExchangeConfig(
            exchange="binanceusdm",
            market_type="futures",
            api_key="key",
            testnet=False,
        ).account_fingerprint,
        "max_notional_usd": 100.0,
        "max_fill_slippage_bps": 100.0,
        "max_futures_leverage": 1,
        "futures_margin_mode": "isolated",
    }
    assert checks["broker_constructed"]["detail"]["broker"] == "fake"
    assert checks["strategy_fingerprints"]["ok"] is True
    assert len(report["products"][0]["artifact_fingerprints"]) == 1
    assert report["products"][0]["artifact_digest"] == artifact_digest(
        json.loads(artifact.read_text(encoding="utf-8"))
    )


def test_preflight_rejects_unsafe_futures_leverage(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "10")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when exchange env is invalid")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert "MAX_FUTURES_LEVERAGE" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_rejects_active_income_leverage_above_one(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "2")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when active-income leverage is above 1")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert (
        "active income futures must use MAX_FUTURES_LEVERAGE=1"
        in checks["exchange_environment"]["error"]
    )
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_rejects_non_positive_fill_slippage(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "0")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when exchange env is invalid")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert "MAX_FILL_SLIPPAGE_BPS" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_rejects_malformed_exchange_env_without_building_broker(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("MAX_NOTIONAL_USD", "nan")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when exchange env is malformed")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert "invalid exchange environment" in checks["exchange_environment"]["error"]
    assert "MAX_NOTIONAL_USD" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_require_testnet_rejects_mainnet_env(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when testnet is required but disabled")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(
        cfg, product_name="active_income", assume_live=True, require_testnet=True
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert checks["exchange_environment"]["detail"]["require_testnet"] is True
    assert checks["exchange_environment"]["detail"]["testnet"] is False
    assert "EXCHANGE_TESTNET must be 1" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_production_preflight_rejects_testnet_environment(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_TESTNET", "1")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy,
        artifact_path=artifact,
        approved_by="test",
        product=active_product,
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built for a sandbox production preflight")

    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        connect=True,
        broker_builder=fail_broker,
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert checks["exchange_environment"]["detail"]["testnet"] is True
    assert (
        "EXCHANGE_TESTNET must be 0 for a production preflight"
        in checks["exchange_environment"]["error"]
    )
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_rejects_non_isolated_futures_margin(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "cross")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when exchange env is invalid")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert "FUTURES_MARGIN_MODE" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_rejects_non_binance_active_income_futures(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("FUTURES_EXCHANGE", "okx")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when futures exchange is invalid")

    monkeypatch.setattr("src.autopilot.preflight.build_live_broker", fail_broker)

    report = run_preflight(cfg, product_name="active_income", assume_live=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_environment"]["ok"] is False
    assert "Binance USDT futures" in checks["exchange_environment"]["error"]
    assert checks["broker_constructed"]["detail"]["skipped"] is True


def test_preflight_broker_policy_rejects_non_binance_even_with_custom_env_checker(
    monkeypatch, tmp_path
):
    set_live_env(monkeypatch)
    monkeypatch.setenv("FUTURES_EXCHANGE", "okx")
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])

    class FailBroker:
        def __init__(self, config):
            raise AssertionError("broker should not be constructed when exchange policy fails")

    monkeypatch.setattr("src.execution.ccxt_broker.CcxtBroker", FailBroker)

    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        exchange_env_checker=lambda product: [],
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert checks["exchange_environment"]["ok"] is True
    assert checks["broker_constructed"]["ok"] is False
    assert "Binance USDT futures" in checks["broker_constructed"]["error"]


def test_preflight_connect_reads_exchange(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    monkeypatch.setattr(
        "src.autopilot.preflight.build_live_broker", lambda product: FakeBroker(position_qty=0.0)
    )

    report = run_preflight(cfg, product_name="active_income", assume_live=True, connect=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert checks["exchange_read_connectivity"]["ok"] is True
    assert checks["exchange_read_connectivity"]["detail"]["price"] == 100.0
    assert checks["exchange_read_connectivity"]["detail"]["position_qty"] == 0.0
    assert checks["exchange_read_connectivity"]["detail"]["position_is_flat"] is True
    assert checks["broker_native_protective_stops"] == {
        "name": "broker_native_protective_stops",
        "ok": True,
        "detail": {"supported": True},
    }
    assert checks["broker_position_mode_one_way"] == {
        "name": "broker_position_mode_one_way",
        "ok": True,
        "detail": {"symbol": "BTCUSDT", "one_way": True},
    }
    assert checks["broker_open_orders_empty"] == {
        "name": "broker_open_orders_empty",
        "ok": True,
        "detail": {
            "symbol": "BTCUSDT",
            "regular": {"count": 0, "orders": []},
            "conditional": {"count": 0, "orders": []},
        },
    }
    assert checks["broker_position_flat"]["ok"] is True


def test_preflight_connect_rejects_hedge_position_mode(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy,
        artifact_path=artifact,
        approved_by="test",
        product=active_product,
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    broker = FakeBroker()
    broker.verify_one_way_position_mode = lambda symbol: False

    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        connect=True,
        broker_builder=lambda product: broker,
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["broker_position_mode_one_way"]["ok"] is False
    assert checks["broker_position_mode_one_way"]["detail"] == {
        "symbol": "BTCUSDT",
        "one_way": False,
    }
    assert "hedge mode" in checks["broker_position_mode_one_way"]["error"]


@pytest.mark.parametrize("conditional", [False, True])
def test_preflight_connect_rejects_orphaned_open_orders(
    monkeypatch,
    tmp_path,
    conditional,
):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy,
        artifact_path=artifact,
        approved_by="test",
        product=active_product,
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    broker = FakeBroker()
    orphan_kind = conditional

    def inventory(symbol, *, conditional):
        if conditional != orphan_kind:
            return ()
        return (
            OpenOrderIdentity(
                symbol=symbol,
                order_id="12345",
                client_id="manual-order-1",
                status="open",
                conditional=conditional,
            ),
        )

    broker.list_open_orders = inventory
    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        connect=True,
        broker_builder=lambda product: broker,
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    detail = checks["broker_open_orders_empty"]["detail"]
    assert report["ok"] is False
    assert checks["broker_open_orders_empty"]["ok"] is False
    assert detail["conditional" if conditional else "regular"]["count"] == 1
    assert detail["conditional" if conditional else "regular"]["orders"][0] == {
        "symbol": "BTCUSDT",
        "order_id": "12345",
        "client_id": "manual-order-1",
        "status": "open",
        "conditional": conditional,
    }
    assert "manual reconciliation" in checks["broker_open_orders_empty"]["error"]


def test_preflight_connect_fails_closed_when_open_order_query_is_unavailable(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy,
        artifact_path=artifact,
        approved_by="test",
        product=active_product,
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    broker = FakeBroker()

    def unavailable(symbol, *, conditional):
        raise RuntimeError("inventory endpoint unavailable")

    broker.list_open_orders = unavailable
    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        connect=True,
        broker_builder=lambda product: broker,
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["broker_open_orders_empty"]["ok"] is False
    assert "inventory endpoint unavailable" in checks["broker_open_orders_empty"]["error"]


def test_preflight_connect_rejects_active_income_broker_without_native_stops(
    monkeypatch,
    tmp_path,
):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy,
        artifact_path=artifact,
        approved_by="test",
        product=active_product,
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    broker = FakeBroker()
    broker.supports_native_protective_stops = lambda: False

    report = run_preflight(
        cfg,
        product_name="active_income",
        assume_live=True,
        connect=True,
        broker_builder=lambda product: broker,
    )

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["broker_native_protective_stops"]["ok"] is False
    assert checks["broker_native_protective_stops"]["detail"] == {"supported": False}
    assert "exchange-native reduce-only" in checks["broker_native_protective_stops"]["error"]


def test_preflight_connect_rejects_non_flat_active_income_position(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path)
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=active_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[active_product])
    monkeypatch.setattr(
        "src.autopilot.preflight.build_live_broker", lambda product: FakeBroker(position_qty=0.5)
    )

    report = run_preflight(cfg, product_name="active_income", assume_live=True, connect=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_read_connectivity"]["ok"] is True
    assert checks["broker_position_flat"]["ok"] is False
    assert checks["broker_position_flat"]["detail"]["position_qty"] == pytest.approx(0.5)
    assert "must be flat" in checks["broker_position_flat"]["error"]


def test_preflight_connect_allows_existing_btc_accumulation_spot_position(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("SPOT_EXCHANGE", "binance")
    artifact = tmp_path / "btc.json"
    strategy = btc_strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        strategies_path=artifact,
    )
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=btc_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[btc_product])
    monkeypatch.setattr(
        "src.autopilot.preflight.build_live_broker", lambda product: FakeBroker(position_qty=0.5)
    )

    report = run_preflight(cfg, product_name="btc_accumulation", assume_live=True, connect=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is True
    assert checks["exchange_read_connectivity"]["ok"] is True
    assert checks["exchange_read_connectivity"]["detail"]["position_qty"] == pytest.approx(0.5)
    assert "broker_position_flat" not in checks
    assert checks["broker_spot_position_non_negative"]["ok"] is True


def test_preflight_connect_rejects_negative_btc_accumulation_spot_position(monkeypatch, tmp_path):
    set_live_env(monkeypatch)
    monkeypatch.setenv("SPOT_EXCHANGE", "binance")
    artifact = tmp_path / "btc.json"
    strategy = btc_strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        strategies_path=artifact,
    )
    ApprovalLedger(ledger).approve(
        strategy, artifact_path=artifact, approved_by="test", product=btc_product
    )
    cfg = AutopilotConfig(approval_ledger=ledger, products=[btc_product])
    monkeypatch.setattr(
        "src.autopilot.preflight.build_live_broker", lambda product: FakeBroker(position_qty=-0.01)
    )

    report = run_preflight(cfg, product_name="btc_accumulation", assume_live=True, connect=True)

    checks = {item["name"]: item for item in report["products"][0]["checks"]}
    assert report["ok"] is False
    assert checks["exchange_read_connectivity"]["ok"] is True
    assert checks["broker_spot_position_non_negative"]["ok"] is False
    assert checks["broker_spot_position_non_negative"]["detail"]["position_qty"] == pytest.approx(
        -0.01
    )
    assert "must be non-negative" in checks["broker_spot_position_non_negative"]["error"]


def test_preflight_cli_writes_output(monkeypatch, tmp_path):
    output = tmp_path / "reports" / "preflight.json"
    report = {"ok": True, "generated_ts": 1.0, "products": []}
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight",
            "--config",
            str(tmp_path / "config.json"),
            "--connect",
            "--require-testnet",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr("src.autopilot.preflight.load_config", lambda path: AutopilotConfig())

    def fake_run_preflight(*args, **kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr("src.autopilot.preflight.run_preflight", fake_run_preflight)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert captured["connect"] is True
    assert captured["require_testnet"] is True


def test_preflight_cli_writes_failure_report_when_config_load_fails(monkeypatch, tmp_path, capsys):
    output = tmp_path / "reports" / "preflight.json"
    config_path = tmp_path / "bad_config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    def fail_load(path):
        raise ValueError("bad config")

    monkeypatch.setattr("src.autopilot.preflight.load_config", fail_load)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == payload
    assert payload["ok"] is False
    assert payload["products"] == []
    assert payload["checks"] == [
        {
            "name": "preflight_build_failed",
            "ok": False,
            "detail": {"config": str(config_path), "error": "ValueError: bad config"},
        }
    ]
    assert payload["errors"] == ["preflight_build_failed: ValueError: bad config"]


def test_preflight_cli_prints_json_when_output_write_fails(monkeypatch, tmp_path, capsys):
    output = tmp_path / "reports" / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight",
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(output),
        ],
    )
    report = {"ok": True, "generated_ts": 1.0, "products": []}
    monkeypatch.setattr("src.autopilot.preflight.load_config", lambda path: AutopilotConfig())
    monkeypatch.setattr(
        "src.autopilot.preflight.run_preflight", lambda *args, **kwargs: report.copy()
    )

    def fail_write(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr("src.autopilot.preflight.write_json_atomic", fail_write)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["products"] == []
    assert printed["checks"] == [
        {
            "name": "preflight_output_write_failed",
            "ok": False,
            "detail": {"path": str(output), "error": "OSError: disk full"},
        }
    ]
    assert printed["errors"] == ["preflight_output_write_failed: OSError: disk full"]
