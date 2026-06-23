"""Tests for the futures execution layer (paper broker, config, ccxt guards)."""

from __future__ import annotations

import pytest

from src.execution import ExchangeConfig, Order, OrderSide, PaperBroker, Position


class _Px:
    """Mutable price source."""

    def __init__(self, price):
        self.price = price

    def __call__(self, symbol):
        return self.price


def test_paper_broker_long_profit_cycle():
    px = _Px(100.0)
    b = PaperBroker(price_source=px, starting_balance=10_000, fee_bps=4, slippage_bps=2)
    b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))
    pos = b.get_position("BTCUSDT")
    assert pos.qty == pytest.approx(1.0)
    assert pos.side == OrderSide.BUY

    px.price = 110.0  # mark up
    assert b.equity() > 10_000  # unrealised gain

    b.close_position("BTCUSDT")
    assert b.get_position("BTCUSDT").is_flat
    assert b.get_balance() > 10_000  # realised the gain (net of fees)


def test_paper_broker_short_loss_on_rally():
    px = _Px(100.0)
    b = PaperBroker(price_source=px, starting_balance=10_000, fee_bps=4, slippage_bps=2)
    b.place_order(Order("BTCUSDT", OrderSide.SELL, qty=1.0))
    assert b.get_position("BTCUSDT").qty == pytest.approx(-1.0)
    px.price = 120.0  # rally hurts the short
    assert b.equity() < 10_000
    b.close_position("BTCUSDT")
    assert b.get_balance() < 10_000


def test_paper_broker_flip_through_zero():
    px = _Px(100.0)
    b = PaperBroker(price_source=px, starting_balance=10_000)
    b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))
    b.place_order(Order("BTCUSDT", OrderSide.SELL, qty=3.0))  # flip to net short 2
    pos = b.get_position("BTCUSDT")
    assert pos.qty == pytest.approx(-2.0)
    assert pos.avg_price == pytest.approx(b.fills[-1].price)  # re-entry at fill price


def test_paper_broker_rejects_nonpositive_qty():
    b = PaperBroker(price_source=_Px(100.0))
    with pytest.raises(ValueError):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.0))


def test_empty_position_is_flat():
    b = PaperBroker(price_source=_Px(100.0))
    assert b.get_position("ETHUSDT") == Position(symbol="ETHUSDT")
    assert b.get_position("ETHUSDT").is_flat


def test_exchange_config_from_env(monkeypatch):
    monkeypatch.setenv("EXCHANGE", "bybit")
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "250")
    cfg = ExchangeConfig.from_env(load_file=False)
    assert cfg.exchange == "bybit"
    assert cfg.live is True
    assert cfg.testnet is False
    assert cfg.max_notional_usd == 250.0


def test_exchange_config_defaults_safe(monkeypatch):
    for var in ["EXCHANGE", "TRADING_LIVE", "EXCHANGE_TESTNET", "MAX_NOTIONAL_USD"]:
        monkeypatch.delenv(var, raising=False)
    cfg = ExchangeConfig.from_env(load_file=False)
    assert cfg.live is False  # never trade live by default
    assert cfg.testnet is True


def test_load_dotenv(tmp_path, monkeypatch):
    from src.execution.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text('EXCHANGE="okx"\n# comment\nMAX_NOTIONAL_USD=42\n')
    monkeypatch.delenv("EXCHANGE", raising=False)
    monkeypatch.delenv("MAX_NOTIONAL_USD", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["EXCHANGE"] == "okx"
    assert os.environ["MAX_NOTIONAL_USD"] == "42"


def test_ccxt_broker_without_ccxt_raises():
    pytest.importorskip  # noqa: B018 - sentinel; explicit branch below
    try:
        import ccxt  # noqa: F401
    except ImportError:
        from src.execution.ccxt_broker import CcxtBroker

        with pytest.raises(ImportError):
            CcxtBroker(ExchangeConfig())
    else:
        pytest.skip("ccxt installed; ImportError path not exercised.")
