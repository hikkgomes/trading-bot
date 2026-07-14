import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.runtime import flatten_product_once, run_once
from src.execution.broker import (
    Fill,
    OrderSide,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)

ACCOUNT_FINGERPRINT = f"account-v1:{'a' * 64}"


def _canonical_digest(payload):
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _strategy():
    return {
        "id": "live_futures_s1",
        "market": "futures",
        "symbol": "BTCUSDT",
        "base_timeframe": "5m",
        "direction": "long",
        "horizon_bars": 12,
        "take_profit": 0.02,
        "stop_loss": 0.05,
        "conditions": [
            {
                "feature": "tf_5m_close",
                "kind": "value_ge",
                "threshold": 0.0,
                "description": "test",
            }
        ],
        "risk": {
            "risk_per_trade": 0.01,
            "daily_stop_loss": -0.03,
            "max_consecutive_losses": 2,
            "cooldown_bars": 3,
            "max_position_fraction": 0.5,
            "max_trades_per_day": 3,
        },
        "fees": {"fee_bps": 4.0, "slippage_bps": 2.0},
        "pnl_unit": "usdt",
        "metrics": {"holdout_total_return": 0.1, "dsr": 0.9},
    }


def _state(strategy):
    signal_time = pd.Timestamp("2026-01-01T00:00:00Z")
    entry_time = signal_time + pd.Timedelta(minutes=5)
    position = {
        "signal_time": signal_time.isoformat(),
        "entry_time": entry_time.isoformat(),
        "direction": "long",
        "entry_price": 100.0,
        "sl_pct": 0.05,
        "tp_pct": 0.02,
        "sl_price": 95.0,
        "tp_price": 102.0,
        "position_size": 0.2,
        "strategy_snapshot": strategy,
        "strategy_fingerprint": _canonical_digest(strategy),
        "approval_strategy_fingerprint": "b" * 64,
        "artifact_digest": "c" * 64,
        "broker_symbol": "BTCUSDT",
        "broker_side": "buy",
        "broker_requested_qty": 0.5,
        "broker_fill_ratio": 1.0,
        "broker_qty": 0.5,
        "broker_entry_price": 100.0,
        "broker_entry_fee": 0.1,
        "broker_entry_balance": 1000.0,
        "broker_account_fingerprint": ACCOUNT_FINGERPRINT,
        "broker_stop_order_id": "stop-1",
        "broker_stop_client_id": "tb-sl-stop-1",
        "broker_stop_trigger_price": 95.0,
    }
    return {
        "equity": 1000.0,
        "peak_equity": 1000.0,
        "drawdown_fraction": 0.0,
        "drawdown_limit_fraction": 0.1,
        "drawdown_halted": False,
        "drawdown_halted_at": None,
        "drawdown_halt_reason": None,
        "open_positions": {strategy["id"]: position},
        "inactive_strategies": [],
        "consecutive_losses": 0,
        "cooldown_until_ts": 0.0,
        "daily_pnl": 0.0,
        "daily_trades_by_strategy": {strategy["id"]: 1},
        "last_entry_decision_bar_by_strategy": {},
        "last_pnl_reset_date": "2026-01-01",
    }


def _product(tmp_path, state_file, trade_log):
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "unused.json",
        state_file=state_file,
        trade_log=trade_log,
        preflight_report=tmp_path / "preflight.json",
        testnet_rehearsal_report=tmp_path / "testnet.json",
        require_testnet_rehearsal=True,
        starting_equity=1000.0,
    )


def _live_env(monkeypatch):
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("FUTURES_EXCHANGE", "binanceusdm")
    monkeypatch.setenv("EXCHANGE_API_KEY", "key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")


class AccountingBroker:
    name = "accounting-live"
    account_fingerprint = ACCOUNT_FINGERPRINT
    config = SimpleNamespace(live=True, market_type="futures")

    def __init__(self, *, ambiguous=False):
        self.position = Position("BTCUSDT", qty=0.5, avg_price=100.0)
        self.balance = 1000.0
        self.orders = []
        self.stop_status = ProtectiveOrderStatus.OPEN
        self.ambiguous = ambiguous

    def get_position(self, symbol):
        return self.position

    def get_balance(self):
        return self.balance

    def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
        return qty

    def place_order(self, order):
        self.orders.append(order)
        self.position = Position(order.symbol)
        self.balance = 1004.0
        if self.ambiguous:
            raise RuntimeError("response lost after fill")
        return Fill(order.symbol, order.side, order.qty, 101.0, 0.1)

    def supports_native_protective_stops(self):
        return True

    def list_account_futures_positions(self):
        return () if self.position.is_flat else (self.position,)

    def list_account_open_orders(self, *, conditional):
        return ()

    def _stop(self):
        return ProtectiveOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            qty=0.5,
            trigger_price=95.0,
            status=self.stop_status,
            order_id="stop-1",
            client_id="tb-sl-stop-1",
        )

    def get_protective_stop(self, **kwargs):
        return self._stop()

    def cancel_protective_stop(self, **kwargs):
        self.stop_status = ProtectiveOrderStatus.CANCELED
        return self._stop()


def _setup(tmp_path):
    strategy = _strategy()
    state_file = tmp_path / "state.json"
    trade_log = tmp_path / "trades.csv"
    state_file.write_text(json.dumps(_state(strategy)), encoding="utf-8")
    return _product(tmp_path, state_file, trade_log), state_file, trade_log


def test_futures_flatten_persists_wal_and_commits_accounting_once(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    product, state_file, trade_log = _setup(tmp_path)
    broker = AccountingBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    result = flatten_product_once(product)

    assert result["ok"] is True
    assert result["reason"] == "flatten_accounted"
    assert len(broker.orders) == 1
    assert broker.orders[0].reduce_only is True
    assert broker.orders[0].client_id.startswith("tb-ff-")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["open_positions"] == {}
    assert "flatten_intent" not in state
    assert state["last_flatten"]["realized_account_delta"] == pytest.approx(4.0)
    assert state["last_flatten"]["fill"]["fee"] == pytest.approx(0.1)
    trades = pd.read_csv(trade_log)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "emergency_flatten"
    assert trades.iloc[0]["broker_exit_balance"] == pytest.approx(1004.0)


def test_futures_flatten_ambiguous_submission_never_duplicates(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    product, state_file, trade_log = _setup(tmp_path)
    broker = AccountingBroker(ambiguous=True)
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    first = flatten_product_once(product)
    durable = json.loads(state_file.read_text(encoding="utf-8"))
    second = flatten_product_once(product)

    assert first["reason"] == "unresolved_flatten_intent"
    assert second["reason"] == "unresolved_flatten_intent"
    assert durable["flatten_intent"]["phase"] == "prepared"
    assert len(broker.orders) == 1
    assert not trade_log.exists()
    assert json.loads(state_file.read_text(encoding="utf-8"))["open_positions"]


def test_futures_flatten_resumes_proven_accounting_without_second_order(
    monkeypatch,
    tmp_path,
):
    _live_env(monkeypatch)
    product, state_file, trade_log = _setup(tmp_path)
    broker = AccountingBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)
    from src.autopilot import runtime

    real_commit = runtime._commit_flatten_exit_accounting
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("crash before accounting intent")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(runtime, "_commit_flatten_exit_accounting", fail_once)

    first = flatten_product_once(product)
    assert first["reason"] == "flatten_accounting_unresolved"
    assert json.loads(state_file.read_text(encoding="utf-8"))["flatten_intent"]["phase"] == (
        "broker_flat_proven"
    )
    second = flatten_product_once(product)

    assert second["ok"] is True
    assert second["reason"] == "flatten_accounted"
    assert len(broker.orders) == 1
    assert len(pd.read_csv(trade_log)) == 1


def test_run_once_auto_clears_control_after_accounted_flatten_restart(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    product, state_file, trade_log = _setup(tmp_path)
    broker = AccountingBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)
    control_file = tmp_path / "control.json"
    control_file.write_text(
        json.dumps({"flatten_products": [product.name], "reason": "panic"}),
        encoding="utf-8",
    )

    first = flatten_product_once(product)
    assert first["ok"] is True
    assert json.loads(control_file.read_text(encoding="utf-8"))["flatten_products"] == [
        product.name
    ]
    config = AutopilotConfig(
        control_file=control_file,
        status_file=tmp_path / "status.json",
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=tmp_path / "jobs.json",
        products=[product],
    )

    report = run_once(config)

    assert report["products"], report
    assert report["products"][0]["reason"] == "already_accounted_flat"
    assert len(broker.orders) == 1
    assert len(pd.read_csv(trade_log)) == 1
    cleared = json.loads(control_file.read_text(encoding="utf-8"))
    assert cleared.get("flatten_products", []) == []
    assert json.loads(state_file.read_text(encoding="utf-8"))["open_positions"] == {}


def test_spot_flatten_commits_observed_btc_accounting_and_trade_log(monkeypatch, tmp_path):
    _live_env(monkeypatch)
    monkeypatch.setenv("SPOT_EXCHANGE", "binance")
    strategy = {
        **_strategy(),
        "id": "btc_step_aside",
        "market": "spot",
        "direction": "short",
        "pnl_unit": "btc",
        "metrics": {"holdout_excess_return_vs_buy_hold": 0.02},
    }
    position = _state(strategy)["open_positions"][strategy["id"]]
    position.update(
        direction="short",
        broker_side="sell",
        broker_entry_base_qty_before=1.0,
        broker_entry_base_qty_after=0.5,
        broker_entry_quote_balance_before=0.0,
        broker_entry_quote_balance_after=49.9,
        broker_entry_quote_value=49.9,
        broker_entry_quote_value_source="observed_free_quote_delta",
        broker_exit_sizing="quote_reinvest",
    )
    for key in (
        "broker_entry_balance",
        "broker_stop_order_id",
        "broker_stop_client_id",
        "broker_stop_trigger_price",
    ):
        position.pop(key, None)
    state = _state(strategy)
    state["equity"] = 1.0
    state["peak_equity"] = 1.0
    state["drawdown_limit_fraction"] = 0.05
    state["open_positions"] = {strategy["id"]: position}
    state_file = tmp_path / "spot-state.json"
    trade_log = tmp_path / "spot-trades.csv"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    product = ProductConfig(
        name="btc_accumulation",
        enabled=True,
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "unused-spot.json",
        state_file=state_file,
        trade_log=trade_log,
        preflight_report=tmp_path / "spot-preflight.json",
        testnet_rehearsal_report=tmp_path / "spot-testnet.json",
        starting_equity=1.0,
    )

    class SpotBroker:
        name = "spot-accounting-live"
        account_fingerprint = ACCOUNT_FINGERPRINT
        config = SimpleNamespace(live=True, market_type="spot")

        def __init__(self):
            self.position = Position("BTCUSDT", qty=0.5)
            self.quote_balance = 49.9
            self.orders = []

        def get_position(self, symbol):
            return self.position

        def get_balance(self):
            return self.quote_balance

        def get_price(self, symbol):
            return 100.0

        def normalize_order_qty(self, symbol, qty, *, price=None, reduce_only=False):
            return qty

        def place_order(self, order):
            durable = json.loads(state_file.read_text(encoding="utf-8"))
            assert durable["flatten_intent"]["client_id"] == order.client_id
            self.orders.append(order)
            self.position = Position(order.symbol, qty=self.position.qty + order.qty)
            self.quote_balance = 0.0
            return Fill(order.symbol, order.side, order.qty, 100.0, 0.0)

    broker = SpotBroker()
    monkeypatch.setattr("src.autopilot.runtime.build_live_broker", lambda product: broker)

    result = flatten_product_once(product)

    assert result["ok"] is True
    assert len(broker.orders) == 1
    final_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert final_state["open_positions"] == {}
    assert "flatten_intent" not in final_state
    assert final_state["equity"] == pytest.approx(0.999)
    assert final_state["last_flatten"]["balance_evidence"]["proven"] is True
    trades = pd.read_csv(trade_log)
    assert len(trades) == 1
    assert trades.iloc[0]["accounting_return_source"] == "observed_btc_balance"
