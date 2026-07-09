import json

import pytest

from src.autopilot.approvals import ApprovalLedger, artifact_digest
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.testnet_rehearsal import (
    main,
    run_testnet_rehearsal,
    summarize_testnet_rehearsal_report,
)
from src.execution.broker import Fill, OrderSide, Position


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
        "id": "testnet_r1",
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


def approved_config(tmp_path):
    artifact = tmp_path / "active.json"
    strategy = strategy_artifact(artifact)
    ledger = tmp_path / "approvals.json"
    active_product = product(tmp_path, strategies_path=artifact)
    ApprovalLedger(ledger).approve(strategy, artifact_path=artifact, approved_by="test", product=active_product)
    return AutopilotConfig(approval_ledger=ledger, products=[active_product])


def set_testnet_env(monkeypatch):
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "1")
    monkeypatch.setenv("FUTURES_EXCHANGE", "binanceusdm")
    monkeypatch.setenv("EXCHANGE_API_KEY", "key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "10")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "isolated")


class FakeTestnetBroker:
    name = "fake-testnet"

    def __init__(self, *, initial_qty=0.0):
        self.position = Position(symbol="BTCUSDT", qty=initial_qty, avg_price=100.0 if initial_qty else 0.0)
        self.orders = []

    def get_price(self, symbol):
        return 100.0

    def get_balance(self):
        return 1000.0

    def get_position(self, symbol):
        return Position(symbol=symbol, qty=self.position.qty, avg_price=self.position.avg_price)

    def place_order(self, order):
        self.orders.append(order)
        signed = order.qty if order.side == OrderSide.BUY else -order.qty
        if order.reduce_only:
            self.position = Position(symbol=order.symbol)
        else:
            self.position = Position(symbol=order.symbol, qty=self.position.qty + signed, avg_price=100.0)
        return Fill(symbol=order.symbol, side=order.side, qty=order.qty, price=100.0, fee=0.01)

    def close_position(self, symbol):
        if self.position.is_flat:
            return None
        side = OrderSide.SELL if self.position.qty > 0 else OrderSide.BUY
        class RehearsalOrder:
            def __init__(self, symbol, side, qty):
                self.symbol = symbol
                self.side = side
                self.qty = qty
                self.reduce_only = True

        return self.place_order(RehearsalOrder(symbol, side, abs(self.position.qty)))


class CloseFailsOnceBroker(FakeTestnetBroker):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close_position(self, symbol):
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("close timeout")
        return super().close_position(symbol)


class WrongCloseFillSideBroker(FakeTestnetBroker):
    def close_position(self, symbol):
        if self.position.is_flat:
            return None
        qty = abs(self.position.qty)
        self.position = Position(symbol=symbol, qty=0.0, avg_price=0.0)
        return Fill(symbol=symbol, side=OrderSide.BUY, qty=qty, price=100.0, fee=0.01)


def test_testnet_rehearsal_requires_explicit_confirmation(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    cfg = approved_config(tmp_path)
    output = tmp_path / "reports" / "failed.json"

    report = run_testnet_rehearsal(
        cfg,
        confirm=False,
        output_path=output,
        broker_builder=lambda product: FakeTestnetBroker(),
    )

    assert report["ok"] is False
    assert report["required_flag"] == "--confirm"
    assert json.loads(output.read_text(encoding="utf-8"))["required_flag"] == "--confirm"


def test_testnet_rehearsal_refuses_non_testnet_env(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    cfg = approved_config(tmp_path)

    report = run_testnet_rehearsal(cfg, confirm=True, broker_builder=lambda product: FakeTestnetBroker())

    assert report["ok"] is False
    assert "EXCHANGE_TESTNET must be 1" in report["error"]


def test_testnet_rehearsal_requires_approval_before_broker_use(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    cfg = AutopilotConfig(approval_ledger=tmp_path / "approvals.json", products=[product(tmp_path, strategies_path=artifact)])

    def fail_broker(_product):
        raise AssertionError("broker should not be built when approval is missing")

    report = run_testnet_rehearsal(cfg, confirm=True, broker_builder=fail_broker)

    assert report["ok"] is False
    assert report["error"] == "preflight_failed"
    checks = {item["name"]: item for item in report["preflight"]["products"][0]["checks"]}
    assert checks["approval_gate"]["ok"] is False


def test_testnet_rehearsal_requires_flat_starting_position(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    cfg = approved_config(tmp_path)

    report = run_testnet_rehearsal(
        cfg,
        confirm=True,
        broker_builder=lambda product: FakeTestnetBroker(initial_qty=0.1),
    )

    assert report["ok"] is False
    assert report["error"] == "preflight_failed"
    checks = {item["name"]: item for item in report["preflight"]["products"][0]["checks"]}
    assert checks["broker_position_flat"]["ok"] is False
    assert checks["broker_position_flat"]["detail"]["position_qty"] == pytest.approx(0.1)


def test_testnet_rehearsal_places_tiny_entry_and_reduce_only_close(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    cfg = approved_config(tmp_path)
    broker = FakeTestnetBroker()
    output = tmp_path / "reports" / "testnet.json"

    report = run_testnet_rehearsal(
        cfg,
        confirm=True,
        notional_usd=5.0,
        output_path=output,
        broker_builder=lambda product: broker,
    )

    assert report["ok"] is True
    assert report["testnet"] is True
    active_product = cfg.products[0]
    product_payload = report["product"]
    assert product_payload["enabled"] is True
    assert product_payload["execution_mode"] == "live"
    assert product_payload["strategies_path"] == str(active_product.strategies_path)
    assert product_payload["require_preflight"] is True
    assert product_payload["preflight_report"] == str(active_product.preflight_report)
    assert product_payload["preflight_max_age_seconds"] == active_product.preflight_max_age_seconds
    assert product_payload["require_testnet_rehearsal"] is True
    assert product_payload["testnet_rehearsal_report"] == str(active_product.testnet_rehearsal_report)
    assert (
        product_payload["testnet_rehearsal_max_age_seconds"]
        == active_product.testnet_rehearsal_max_age_seconds
    )
    assert report["risk_controls"] == {
        "max_futures_leverage": 1,
        "futures_margin_mode": "isolated",
        "max_notional_usd": 10.0,
        "max_fill_slippage_bps": 100.0,
    }
    assert report["order_qty"] == pytest.approx(0.05)
    assert report["entry_fill"]["side"] == "buy"
    assert report["close_fill"]["side"] == "sell"
    assert report["final_position_qty"] == pytest.approx(0.0)
    assert len(broker.orders) == 2
    assert broker.orders[0].reduce_only is False
    assert broker.orders[1].reduce_only is True
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_testnet_rehearsal_attempts_recovery_close_after_entry_close_failure(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    cfg = approved_config(tmp_path)
    broker = CloseFailsOnceBroker()
    output = tmp_path / "reports" / "testnet_failed.json"

    report = run_testnet_rehearsal(
        cfg,
        confirm=True,
        notional_usd=5.0,
        output_path=output,
        broker_builder=lambda product: broker,
    )

    assert report["ok"] is False
    assert report["error"] == "order_rehearsal_failed: close timeout"
    assert report["entry_fill"]["side"] == "buy"
    assert report["close_fill"] is None
    assert report["recovery"]["attempted"] is True
    assert report["recovery"]["close_fill"]["side"] == "sell"
    assert report["recovery"]["final_position_qty"] == pytest.approx(0.0)
    assert report["recovery"]["final_position_flat"] is True
    assert len(broker.orders) == 2
    assert broker.orders[1].reduce_only is True
    assert json.loads(output.read_text(encoding="utf-8"))["recovery"]["final_position_flat"] is True


def test_testnet_rehearsal_rejects_invalid_close_fill_evidence(monkeypatch, tmp_path):
    set_testnet_env(monkeypatch)
    cfg = approved_config(tmp_path)
    broker = WrongCloseFillSideBroker()
    output = tmp_path / "reports" / "testnet_invalid_fill.json"

    report = run_testnet_rehearsal(
        cfg,
        confirm=True,
        notional_usd=5.0,
        output_path=output,
        broker_builder=lambda product: broker,
    )

    assert report["ok"] is False
    assert "close fill mismatch: expected side sell, got buy" in report["error"]
    assert report["entry_fill"]["side"] == "buy"
    assert report["close_fill"]["side"] == "buy"
    assert report["recovery"]["attempted"] is True
    assert report["recovery"]["final_position_flat"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is False


def test_testnet_rehearsal_cli_writes_output(monkeypatch, tmp_path):
    output = tmp_path / "testnet.json"
    report = {"ok": True}
    monkeypatch.setattr(
        "sys.argv",
        [
            "testnet-rehearsal",
            "--config",
            str(tmp_path / "config.json"),
            "--confirm",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr("src.autopilot.testnet_rehearsal.load_config", lambda path: AutopilotConfig())
    monkeypatch.setattr(
        "src.autopilot.testnet_rehearsal.run_testnet_rehearsal",
        lambda *args, **kwargs: (output.write_text(json.dumps(report), encoding="utf-8"), report)[1],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_testnet_rehearsal_cli_status_summarizes_without_confirm(monkeypatch, tmp_path, capsys):
    output = tmp_path / "testnet.json"
    active_product = product(tmp_path, testnet_rehearsal_max_age_seconds=123)
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "testnet-rehearsal",
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(output),
            "--status",
        ],
    )
    monkeypatch.setattr(
        "src.autopilot.testnet_rehearsal.load_config",
        lambda path: AutopilotConfig(products=[active_product]),
    )

    def fake_summary(path, *, max_age_seconds, expected_product):
        captured["path"] = path
        captured["max_age_seconds"] = max_age_seconds
        captured["expected_product"] = expected_product
        return {"ok": True, "status": "ok", "product": expected_product.name}

    def fail_rehearsal(*args, **kwargs):
        raise AssertionError("status mode must not run the order rehearsal")

    monkeypatch.setattr("src.autopilot.testnet_rehearsal.summarize_testnet_rehearsal_report", fake_summary)
    monkeypatch.setattr("src.autopilot.testnet_rehearsal.run_testnet_rehearsal", fail_rehearsal)

    with pytest.raises(SystemExit) as exc:
        main()

    printed = json.loads(capsys.readouterr().out)
    assert exc.value.code == 0
    assert printed == {"ok": True, "status": "ok", "product": "active_income"}
    assert captured == {
        "path": output,
        "max_age_seconds": 123,
        "expected_product": active_product,
    }


def test_testnet_rehearsal_marks_output_write_failure(monkeypatch, tmp_path):
    cfg = AutopilotConfig(products=[])
    output = tmp_path / "reports" / "testnet.json"

    def fail_write(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr("src.autopilot.testnet_rehearsal.write_json_atomic", fail_write)

    report = run_testnet_rehearsal(cfg, output_path=output)

    assert report["ok"] is False
    assert report["error"] == "product not found: active_income"
    assert report["errors"] == [
        {
            "name": "testnet_rehearsal_output_write_failed",
            "detail": {"path": str(output), "error": "OSError: disk full"},
        }
    ]


def test_testnet_rehearsal_cli_writes_failure_report_when_config_load_fails(monkeypatch, tmp_path, capsys):
    output = tmp_path / "reports" / "testnet.json"
    config_path = tmp_path / "bad_config.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "testnet-rehearsal",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
    )

    def fail_load(path):
        raise ValueError("bad config")

    monkeypatch.setattr("src.autopilot.testnet_rehearsal.load_config", fail_load)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == payload
    assert payload["ok"] is False
    assert payload["error"] == "testnet_rehearsal_failed"
    assert payload["config"] == str(config_path)
    assert payload["exception"] == "ValueError: bad config"


def test_testnet_rehearsal_summary_reports_missing_and_stale(tmp_path):
    missing = summarize_testnet_rehearsal_report(tmp_path / "missing.json", now_ts=1000.0)

    assert missing["exists"] is False
    assert missing["status"] == "missing"
    assert missing["ok"] is False
    assert missing["next_action"]["preflight_command"] == "make preflight PRODUCT=active_income REQUIRE_TESTNET=1"
    assert missing["next_action"]["rehearsal_command"] == "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5"
    assert missing["next_action"]["status_command"] == "make testnet-status"
    assert "EXCHANGE_TESTNET=1" in missing["next_action"]["required_env"]

    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "1970-01-01T00:00:00+00:00",
                "generated_ts": 0.0,
                "product": {"name": "active_income"},
                "exchange": "binanceusdm",
                "testnet": True,
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    stale = summarize_testnet_rehearsal_report(report_path, max_age_seconds=10, now_ts=1000.0)

    assert stale["exists"] is True
    assert stale["status"] == "stale"
    assert stale["ok"] is False
    assert stale["fresh"] is False
    assert stale["product"] == "active_income"
    assert stale["final_position_flat"] is True


def test_testnet_rehearsal_summary_rejects_symlink_report(tmp_path):
    target = tmp_path / "target_testnet.json"
    report_path = tmp_path / "testnet.json"
    target.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    report_path.symlink_to(target)

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0)

    assert status["exists"] is True
    assert status["ok"] is False
    assert status["status"] == "read_error"
    assert "must not be a symlink" in status["error"]
    assert status["next_action"]["status_command"] == "make testnet-status"


def test_testnet_rehearsal_summary_rejects_unmatched_expected_product(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "ETHUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["product_symbol_mismatch"]
    assert status["report_product"]["symbol"] == "ETHUSDT"
    assert status["expected_product"]["symbol"] == "BTCUSDT"
    assert status["next_action"]["rehearsal_command"] == "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5"
    assert status["next_action"]["status_command"] == "make testnet-status"


def test_testnet_rehearsal_summary_rejects_unmatched_fill_symbols(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {
                    "side": "buy",
                    "qty": 0.05,
                    "price": 100.0,
                    "fee": 0.01,
                    "timestamp": 1000.0,
                },
                "close_fill": {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "qty": 0.05,
                    "price": 100.0,
                    "fee": 0.01,
                    "timestamp": 1001.0,
                },
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == [
        "entry_fill_symbol_mismatch",
        "close_fill_symbol_mismatch",
    ]


def test_testnet_rehearsal_summary_rejects_fill_qty_not_matching_order_qty(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.04,
                "entry_fill": {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "qty": 0.05,
                    "price": 100.0,
                    "fee": 0.01,
                    "timestamp": 1000.0,
                },
                "close_fill": {
                    "symbol": "BTCUSDT",
                    "side": "sell",
                    "qty": 0.03,
                    "price": 100.0,
                    "fee": 0.01,
                    "timestamp": 1001.0,
                },
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == [
        "entry_fill_qty_mismatch",
        "close_fill_qty_mismatch",
    ]


def test_testnet_rehearsal_summary_rejects_missing_risk_controls_for_expected_product(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["missing_risk_controls"]


def test_testnet_rehearsal_summary_rejects_unsafe_risk_controls(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 10,
                    "futures_margin_mode": "cross",
                    "max_notional_usd": 0,
                    "max_fill_slippage_bps": -1,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == [
        "max_futures_leverage_invalid",
        "futures_margin_mode_not_isolated",
        "max_notional_usd_invalid",
        "max_fill_slippage_bps_invalid",
    ]


def test_testnet_rehearsal_summary_rejects_active_income_leverage_above_one(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 2,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=product(tmp_path))

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["max_futures_leverage_invalid"]


def test_testnet_rehearsal_summary_rejects_embedded_preflight_missing_artifact_digest(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    active_product = product(tmp_path, strategies_path=artifact)
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
                "preflight": {
                    "ok": True,
                    "products": [
                        {
                            "ok": True,
                            "product": {
                                "name": active_product.name,
                                "objective": active_product.objective,
                                "base_asset": active_product.base_asset,
                                "market": active_product.market,
                                "symbol": active_product.symbol,
                                "execution_mode": "live",
                                "strategies_path": str(artifact),
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=active_product)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["embedded_preflight_missing_artifact_digest"]


def test_testnet_rehearsal_summary_rejects_embedded_preflight_artifact_digest_mismatch(tmp_path):
    artifact = tmp_path / "active.json"
    strategy_artifact(artifact)
    active_product = product(tmp_path, strategies_path=artifact)
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
                "preflight": {
                    "ok": True,
                    "products": [
                        {
                            "ok": True,
                            "artifact_digest": artifact_digest({"stale": True}),
                            "product": {
                                "name": active_product.name,
                                "objective": active_product.objective,
                                "base_asset": active_product.base_asset,
                                "market": active_product.market,
                                "symbol": active_product.symbol,
                                "execution_mode": "live",
                                "strategies_path": str(artifact),
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0, expected_product=active_product)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["embedded_preflight_artifact_digest_mismatch"]


def test_testnet_rehearsal_summary_rejects_future_timestamp(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1401.0,
                "product": {"name": "active_income"},
                "exchange": "binanceusdm",
                "testnet": True,
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1000.0)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["fresh"] is True
    assert status["age_seconds"] == pytest.approx(-401.0)
    assert status["clock_skew_seconds"] == 300
    assert status["invalid_reasons"] == ["future_generated_ts"]
    assert status["next_action"]["rehearsal_command"] == "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5"
    assert status["next_action"]["status_command"] == "make testnet-status"


def test_testnet_rehearsal_summary_reports_non_object_json(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text("[]", encoding="utf-8")

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1000.0)

    assert status["exists"] is True
    assert status["ok"] is False
    assert status["status"] == "read_error"
    assert status["error"] == "TypeError: expected JSON object, got list"
    assert status["next_action"]["rehearsal_command"] == "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5"
    assert status["next_action"]["status_command"] == "make testnet-status"


def test_testnet_rehearsal_summary_fails_unsafe_ok_reports(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {"name": "active_income"},
                "exchange": "binanceusdm",
                "testnet": False,
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.01,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == ["not_testnet", "final_position_not_flat"]
    assert status["final_position_flat"] is False
    assert status["next_action"]["preflight_command"] == "make preflight PRODUCT=active_income REQUIRE_TESTNET=1"
    assert status["next_action"]["status_command"] == "make testnet-status"


def test_testnet_rehearsal_summary_fails_incomplete_ok_reports(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "product": {"name": "active_income"},
                "exchange": "binanceusdm",
                "testnet": True,
                "notional_usd": 0,
                "entry_fill": {"side": "sell"},
                "close_fill": {"side": "buy"},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == [
        "missing_generated_ts",
        "invalid_notional_usd",
        "entry_fill_side_not_buy",
        "entry_fill_invalid_qty",
        "entry_fill_invalid_price",
        "entry_fill_invalid_fee",
        "entry_fill_invalid_timestamp",
        "close_fill_side_not_sell",
        "close_fill_invalid_qty",
        "close_fill_invalid_price",
        "close_fill_invalid_fee",
        "close_fill_invalid_timestamp",
    ]
    assert status["entry_side"] == "sell"
    assert status["close_side"] == "buy"


def test_testnet_rehearsal_summary_fails_invalid_fill_evidence(tmp_path):
    report_path = tmp_path / "testnet.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "generated_ts": 1000.0,
                "product": {"name": "active_income"},
                "exchange": "binanceusdm",
                "testnet": True,
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"side": "buy", "qty": "nan", "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"side": "sell", "qty": 0.05, "price": 0.0, "fee": -0.01, "timestamp": None},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )

    status = summarize_testnet_rehearsal_report(report_path, now_ts=1001.0)

    assert status["ok"] is False
    assert status["status"] == "failed"
    assert status["invalid_reasons"] == [
        "entry_fill_invalid_qty",
        "close_fill_invalid_price",
        "close_fill_invalid_fee",
        "close_fill_invalid_timestamp",
    ]
    assert status["next_action"]["rehearsal_command"] == "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5"
