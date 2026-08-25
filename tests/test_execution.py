"""Tests for the futures execution layer (paper broker, config, ccxt guards)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.execution import (
    ExchangeConfig,
    FuturesPositionIdentity,
    OpenOrderIdentity,
    Order,
    OrderSide,
    OrderType,
    PaperBroker,
    Position,
    ProtectiveOrderStatus,
)


class _Px:
    """Mutable price source."""

    def __init__(self, price):
        self.price = price

    def __call__(self, symbol):
        return self.price


def _futures_settings_position(
    *,
    leverage=1,
    margin_mode="isolated",
    contracts=0.0,
    side=None,
    entry_price=None,
):
    return {
        "symbol": "BTC/USDT:USDT",
        "contracts": contracts,
        "side": side,
        "entryPrice": entry_price,
        "marginMode": margin_mode,
        "leverage": leverage,
    }


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"starting_balance": float("nan")}, "starting_balance must be finite and non-negative"),
        ({"starting_balance": -1.0}, "starting_balance must be finite and non-negative"),
        ({"fee_bps": float("inf")}, "fee_bps must be finite and non-negative"),
        ({"fee_bps": -0.1}, "fee_bps must be finite and non-negative"),
        ({"slippage_bps": float("nan")}, "slippage_bps must be finite and non-negative"),
        ({"slippage_bps": -0.1}, "slippage_bps must be finite and non-negative"),
        ({"slippage_bps": 10_000.0}, "slippage_bps must be less than 10000"),
    ],
)
def test_paper_broker_rejects_invalid_simulation_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PaperBroker(price_source=_Px(100.0), **kwargs)


def test_paper_broker_reduce_only_reduces_open_long():
    b = PaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)
    b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    fill = b.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.4, reduce_only=True))

    assert fill.qty == pytest.approx(0.4)
    assert b.get_position("BTCUSDT").qty == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("starting_order", "reduce_order", "message"),
    [
        (
            Order("BTCUSDT", OrderSide.BUY, qty=1.0),
            Order("BTCUSDT", OrderSide.BUY, qty=0.1, reduce_only=True),
            "side must reduce the current long position",
        ),
        (
            Order("BTCUSDT", OrderSide.BUY, qty=1.0),
            Order("BTCUSDT", OrderSide.SELL, qty=1.1, reduce_only=True),
            "quantity 1.1 exceeds open position 1",
        ),
        (
            Order("BTCUSDT", OrderSide.SELL, qty=1.0),
            Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True),
            "side must reduce the current short position",
        ),
    ],
)
def test_paper_broker_rejects_reduce_only_orders_that_do_not_reduce(
    starting_order, reduce_order, message
):
    b = PaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)
    b.place_order(starting_order)

    with pytest.raises(ValueError, match=message):
        b.place_order(reduce_order)


def test_paper_broker_reduce_only_requires_open_position():
    b = PaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)

    with pytest.raises(ValueError, match="requires an open position"):
        b.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))


def test_broker_open_order_inventory_default_fails_closed():
    broker = PaperBroker(price_source=_Px(100.0))

    with pytest.raises(NotImplementedError, match="cannot verify regular open orders"):
        broker.list_open_orders("BTCUSDT", conditional=False)


def test_spot_paper_broker_keeps_step_aside_sell_then_buy_semantics():
    class SpotPaperBroker(PaperBroker):
        class Config:
            market_type = "spot"

        config = Config()

    b = SpotPaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)
    b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    fill = b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.25, reduce_only=True))

    assert fill.qty == pytest.approx(0.25)
    assert b.get_position("BTCUSDT").qty == pytest.approx(1.25)


def test_paper_broker_rejects_nonpositive_qty():
    b = PaperBroker(price_source=_Px(100.0))
    with pytest.raises(ValueError):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.0))


@pytest.mark.parametrize("qty", [float("nan"), float("inf")])
def test_paper_broker_rejects_nonfinite_qty(qty):
    b = PaperBroker(price_source=_Px(100.0))
    with pytest.raises(ValueError, match="Order qty must be positive"):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=qty))


@pytest.mark.parametrize("price", [0.0, float("nan"), float("inf")])
def test_paper_broker_rejects_invalid_price_source(price):
    b = PaperBroker(price_source=_Px(price))
    with pytest.raises(ValueError, match="Paper price"):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


def test_paper_limit_order_requires_price():
    b = PaperBroker(price_source=_Px(100.0))
    with pytest.raises(ValueError, match="Limit order price is required"):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0, type=OrderType.LIMIT))


@pytest.mark.parametrize("price", [0.0, float("nan"), float("inf")])
def test_paper_limit_order_rejects_invalid_price(price):
    b = PaperBroker(price_source=_Px(100.0))
    with pytest.raises(ValueError, match="Order price must be finite and positive"):
        b.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0, type=OrderType.LIMIT, price=price))


def test_empty_position_is_flat():
    b = PaperBroker(price_source=_Px(100.0))
    assert b.get_position("ETHUSDT") == Position(symbol="ETHUSDT")
    assert b.get_position("ETHUSDT").is_flat


def test_exchange_config_from_env(monkeypatch):
    monkeypatch.setenv("EXCHANGE", "bybit")
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "0")
    monkeypatch.setenv("MAX_NOTIONAL_USD", "250")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "75")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "2")
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "isolated")
    cfg = ExchangeConfig.from_env(load_file=False)
    assert cfg.exchange == "bybit"
    assert cfg.market_type == "futures"
    assert cfg.live is True
    assert cfg.testnet is False
    assert cfg.max_notional_usd == 250.0
    assert cfg.max_fill_slippage_bps == 75.0
    assert cfg.max_futures_leverage == 2
    assert cfg.futures_margin_mode == "isolated"


def test_exchange_config_spot_defaults_to_binance(monkeypatch):
    for var in ["EXCHANGE", "FUTURES_EXCHANGE", "SPOT_EXCHANGE", "EXCHANGE_MARKET_TYPE"]:
        monkeypatch.delenv(var, raising=False)

    cfg = ExchangeConfig.from_env(load_file=False, market_type="spot")

    assert cfg.exchange == "binance"
    assert cfg.market_type == "spot"


def test_exchange_config_market_type_from_env(monkeypatch):
    monkeypatch.setenv("EXCHANGE_MARKET_TYPE", "spot")
    monkeypatch.setenv("SPOT_EXCHANGE", " kraken ")

    cfg = ExchangeConfig.from_env(load_file=False)

    assert cfg.exchange == "kraken"
    assert cfg.market_type == "spot"


def test_exchange_config_defaults_safe(monkeypatch):
    for var in [
        "EXCHANGE",
        "TRADING_LIVE",
        "EXCHANGE_TESTNET",
        "MAX_NOTIONAL_USD",
        "MAX_FILL_SLIPPAGE_BPS",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = ExchangeConfig.from_env(load_file=False)
    assert cfg.live is False  # never trade live by default
    assert cfg.testnet is True
    assert cfg.max_fill_slippage_bps == 100.0


@pytest.mark.parametrize(
    ("live_value", "testnet_value", "expected_live", "expected_testnet"),
    [
        ("yes", "no", True, False),
        ("on", "off", True, False),
        ("true", "false", True, False),
        ("1", "0", True, False),
    ],
)
def test_exchange_config_accepts_explicit_boolean_aliases(
    monkeypatch,
    live_value,
    testnet_value,
    expected_live,
    expected_testnet,
):
    monkeypatch.setenv("TRADING_LIVE", live_value)
    monkeypatch.setenv("EXCHANGE_TESTNET", testnet_value)

    cfg = ExchangeConfig.from_env(load_file=False)

    assert cfg.live is expected_live
    assert cfg.testnet is expected_testnet


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("TRADING_LIVE", "treu", "TRADING_LIVE must be a boolean flag"),
        ("EXCHANGE_TESTNET", "treu", "EXCHANGE_TESTNET must be a boolean flag"),
        ("FUTURES_EXCHANGE", " ", "FUTURES_EXCHANGE must be non-empty"),
        ("MAX_NOTIONAL_USD", "nan", "MAX_NOTIONAL_USD must be finite and positive"),
        ("MAX_NOTIONAL_USD", "0", "MAX_NOTIONAL_USD must be finite and positive"),
        ("MAX_FILL_SLIPPAGE_BPS", "inf", "MAX_FILL_SLIPPAGE_BPS must be finite and positive"),
        ("MAX_FILL_SLIPPAGE_BPS", "abc", "MAX_FILL_SLIPPAGE_BPS must be numeric"),
        ("MAX_FUTURES_LEVERAGE", "0", "MAX_FUTURES_LEVERAGE must be between 1 and 3"),
        ("MAX_FUTURES_LEVERAGE", "1.5", "MAX_FUTURES_LEVERAGE must be an integer"),
        ("QUOTE_ASSET", " ", "QUOTE_ASSET must be non-empty"),
    ],
)
def test_exchange_config_rejects_invalid_env_values(monkeypatch, name, value, message):
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "isolated")
    monkeypatch.setenv("QUOTE_ASSET", "USDT")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        ExchangeConfig.from_env(load_file=False)


def test_exchange_config_rejects_blank_spot_exchange(monkeypatch):
    monkeypatch.setenv("SPOT_EXCHANGE", " ")

    with pytest.raises(ValueError, match="SPOT_EXCHANGE must be non-empty"):
        ExchangeConfig.from_env(load_file=False, market_type="spot")


def test_exchange_config_trims_credentials(monkeypatch):
    monkeypatch.setenv("EXCHANGE_API_KEY", " key ")
    monkeypatch.setenv("EXCHANGE_API_SECRET", " secret ")
    monkeypatch.setenv("EXCHANGE_API_PASSWORD", " password ")

    cfg = ExchangeConfig.from_env(load_file=False)

    assert cfg.api_key == "key"
    assert cfg.api_secret == "secret"
    assert cfg.api_password == "password"


def test_exchange_config_account_fingerprint_binds_key_and_routing_not_secret():
    base = ExchangeConfig(
        exchange="BINANCEUSDM",
        market_type="FUTURES",
        api_key="key-a",
        api_secret="secret-a",
        api_password="password-a",
        testnet=False,
    )
    same_identity = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="key-a",
        api_secret="rotated-secret",
        api_password="rotated-password",
        testnet=False,
    )

    assert base.account_fingerprint == same_identity.account_fingerprint
    assert base.account_fingerprint.startswith("account-v1:")
    assert "key-a" not in base.account_fingerprint
    assert "secret-a" not in base.account_fingerprint
    assert (
        ExchangeConfig(
            exchange="binanceusdm",
            market_type="futures",
            api_key="key-b",
            testnet=False,
        ).account_fingerprint
        != base.account_fingerprint
    )
    assert (
        ExchangeConfig(
            exchange="binanceusdm",
            market_type="futures",
            api_key="key-a",
            testnet=True,
        ).account_fingerprint
        != base.account_fingerprint
    )


def test_exchange_config_rejects_non_isolated_futures_margin_mode(monkeypatch):
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")
    monkeypatch.setenv("QUOTE_ASSET", "USDT")
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "cross")

    with pytest.raises(ValueError, match="FUTURES_MARGIN_MODE must be 'isolated'"):
        ExchangeConfig.from_env(load_file=False, market_type="futures")


def test_exchange_config_ignores_futures_margin_mode_for_spot(monkeypatch):
    monkeypatch.setenv("MAX_NOTIONAL_USD", "100")
    monkeypatch.setenv("MAX_FILL_SLIPPAGE_BPS", "100")
    monkeypatch.setenv("MAX_FUTURES_LEVERAGE", "1")
    monkeypatch.setenv("QUOTE_ASSET", "USDT")
    monkeypatch.setenv("FUTURES_MARGIN_MODE", "cross")

    cfg = ExchangeConfig.from_env(load_file=False, market_type="spot")

    assert cfg.market_type == "spot"
    assert cfg.futures_margin_mode == "cross"


def test_load_dotenv(tmp_path, monkeypatch):
    from src.execution.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text('EXCHANGE="okx"\n# comment\nMAX_NOTIONAL_USD=42\n')
    monkeypatch.delenv("EXCHANGE", raising=False)
    monkeypatch.delenv("MAX_NOTIONAL_USD", raising=False)
    load_dotenv(env)
    assert os.environ["EXCHANGE"] == "okx"
    assert os.environ["MAX_NOTIONAL_USD"] == "42"


def test_load_dotenv_handles_inline_comments_and_quoted_hashes(tmp_path, monkeypatch):
    from src.execution.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "TRADING_LIVE=1  # enable live routing",
                'WEBHOOK_URL="https://example.test/hook#alerts" # destination',
                "TOKEN=abc#123",
                "SPACED='  keep # literal  ' # trailing comment",
                "EMPTY= # intentionally blank",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("TRADING_LIVE", "WEBHOOK_URL", "TOKEN", "SPACED", "EMPTY"):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(env)

    assert os.environ["TRADING_LIVE"] == "1"
    assert os.environ["WEBHOOK_URL"] == "https://example.test/hook#alerts"
    assert os.environ["TOKEN"] == "abc#123"
    assert os.environ["SPACED"] == "  keep # literal  "
    assert os.environ["EMPTY"] == ""


def test_shipped_dotenv_example_builds_execution_config(monkeypatch):
    from src.execution.config import load_dotenv

    keys = (
        "FUTURES_EXCHANGE",
        "EXCHANGE_MARKET_TYPE",
        "EXCHANGE_TESTNET",
        "TRADING_LIVE",
        "MAX_NOTIONAL_USD",
        "MAX_FILL_SLIPPAGE_BPS",
        "QUOTE_ASSET",
        "FUTURES_MARGIN_MODE",
        "MAX_FUTURES_LEVERAGE",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    load_dotenv(Path(".env.example"))
    cfg = ExchangeConfig.from_env(load_file=False, market_type="futures")

    assert cfg.exchange == "binanceusdm"
    assert cfg.testnet is True
    assert cfg.live is False
    assert cfg.max_notional_usd == 250.0
    assert cfg.max_fill_slippage_bps == 100.0
    assert cfg.futures_margin_mode == "isolated"
    assert cfg.max_futures_leverage == 1
    assert cfg.quote_asset == "USDT"


def test_load_dotenv_rejects_symlink_without_loading_target(tmp_path, monkeypatch):
    from src.execution.config import load_dotenv

    env = tmp_path / ".env"
    target = tmp_path / "external.env"
    target.write_text("TRADING_LIVE=1\nEXCHANGE=okx\n", encoding="utf-8")
    env.symlink_to(target)
    monkeypatch.delenv("TRADING_LIVE", raising=False)
    monkeypatch.delenv("EXCHANGE", raising=False)

    with pytest.raises(ValueError, match=r"\.env must not be a symlink"):
        load_dotenv(env)

    assert env.is_symlink()
    assert "TRADING_LIVE" not in os.environ
    assert "EXCHANGE" not in os.environ


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


def test_ccxt_spot_position_and_reduce_only_params():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = None
            self.tickers = []

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"BTC": 0.25, "USDT": 500.0}}

        def fetch_ticker(self, symbol):
            self.tickers.append(symbol)
            return {"last": 100.0}

        def create_order(self, **kwargs):
            self.created = kwargs
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    assert broker.get_balance() == 500.0
    assert broker.get_position("BTCUSDT").qty == pytest.approx(0.25)
    fill = broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))

    assert fill.qty == pytest.approx(0.1)
    assert broker._client.tickers == ["BTC/USDT"]
    assert broker._client.created["symbol"] == "BTC/USDT"
    assert broker._client.created["params"] == {}


def test_ccxt_quote_balance_uses_free_funds_and_never_total_fallback():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"free": {}, "total": {"USDT": 500.0}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=1000,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    assert broker.get_balance() == 0.0


def test_ccxt_quote_balance_rejects_missing_free_balance_object():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"total": {"USDT": 500.0}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=1000,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="free-balance object"):
        broker.get_balance()


def test_ccxt_binance_order_passes_safe_client_id_to_exchange_params():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = None

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_order(self, **kwargs):
            self.created = kwargs
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()
    client_id = "tb-en-0123456789abcdef0123456789ab"

    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.5, client_id=client_id))

    assert broker._client.created["params"] == {"newClientOrderId": client_id}


def test_ccxt_futures_read_calls_use_ccxt_symbol_but_return_requested_symbol():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.tickers = []
            self.position_symbols = []

        def fetch_ticker(self, symbol):
            self.tickers.append(symbol)
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            self.position_symbols.append(symbols)
            return [{"contracts": 0.25, "side": "long", "entryPrice": 90.0}]

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    assert broker.get_price("BTCUSDT") == pytest.approx(100.0)
    position = broker.get_position("BTCUSDT")

    assert position.symbol == "BTCUSDT"
    assert position.qty == pytest.approx(0.25)
    assert broker._client.tickers == ["BTC/USDT:USDT"]
    assert broker._client.position_symbols == [["BTC/USDT:USDT"]]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "Ticker price must be finite"),
        (float("inf"), "Ticker price must be finite"),
        (0.0, "Ticker price must be positive"),
        (-1.0, "Ticker price must be positive"),
    ],
)
def test_ccxt_get_price_rejects_invalid_ticker_values(value, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": value}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.get_price("BTCUSDT")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "Quote balance must be finite"),
        (float("inf"), "Quote balance must be finite"),
        (-1.0, "Quote balance must be non-negative"),
        ("", "Quote balance must be numeric"),
    ],
)
def test_ccxt_get_balance_rejects_invalid_quote_balance(value, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"free": {"USDT": value}, "total": {"USDT": 100.0}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.get_balance()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "Spot BTC position quantity must be finite"),
        (float("inf"), "Spot BTC position quantity must be finite"),
        (-0.1, "Spot BTC position quantity must be non-negative"),
        ("", "Spot BTC position quantity must be numeric"),
    ],
)
def test_ccxt_spot_get_position_rejects_invalid_base_quantity(value, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"total": {"BTC": value}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.get_position("BTCUSDT")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"contracts": float("nan"), "side": "long", "entryPrice": 90.0},
            "Futures position contracts must be finite",
        ),
        (
            {"contracts": -1.0, "side": "long", "entryPrice": 90.0},
            "Futures position contracts must be non-negative",
        ),
        (
            {"contracts": "", "side": "long", "entryPrice": 90.0},
            "Futures position contracts must be numeric",
        ),
        (
            {"contracts": 0.1, "side": "flat", "entryPrice": 90.0},
            "Futures position side must be long or short",
        ),
        (
            {"contracts": 0.1, "side": "long", "entryPrice": 0.0},
            "Futures position entry price must be positive",
        ),
        (
            {"contracts": 0.1, "side": "long", "entryPrice": float("nan")},
            "Futures position entry price must be finite",
        ),
        (
            {"contracts": 0.1, "side": "long", "entryPrice": ""},
            "Futures position entry price must be numeric",
        ),
    ],
)
def test_ccxt_futures_get_position_rejects_invalid_position_payload(payload, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_positions(self, symbols):
            return [payload]

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.get_position("BTCUSDT")


def test_ccxt_spot_sell_rejects_quantity_above_base_balance_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"BTC": 0.25, "USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_order(self, **kwargs):
            raise AssertionError("spot oversell should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Refusing to short spot"):
        broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.3))


def test_ccxt_spot_buy_allows_notional_within_quote_balance():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = None

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            return [{"contracts": 0.1, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            self.created = kwargs
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.5))

    assert fill.qty == pytest.approx(0.5)
    assert broker._client.created["symbol"] == "BTC/USDT"


def test_ccxt_spot_buy_rejects_notional_above_quote_balance_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_balance(self):
            return {"free": {"USDT": 50.0}, "total": {"USDT": 50.0}}

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_order(self, **kwargs):
            raise AssertionError("spot overspend should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Refusing to overspend spot quote balance"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.6))


def test_ccxt_order_rejects_requested_notional_above_cap_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 200.0}

        def create_order(self, **kwargs):
            raise AssertionError("oversize order should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order notional"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


@pytest.mark.parametrize("max_notional", [0.0, -1.0, float("nan"), float("inf")])
def test_ccxt_order_rejects_invalid_notional_cap_before_balance_or_exchange_call(max_notional):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            raise AssertionError("invalid notional cap should fail before balance lookup")

        def create_order(self, **kwargs):
            raise AssertionError("invalid notional cap should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=max_notional,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="MAX_NOTIONAL_USD must be finite and positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


def test_ccxt_order_rejects_nonpositive_quantity_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            raise AssertionError("nonpositive qty should fail before price lookup")

        def create_order(self, **kwargs):
            raise AssertionError("nonpositive qty should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order quantity must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.0))


@pytest.mark.parametrize("qty", [float("nan"), float("inf")])
def test_ccxt_order_rejects_nonfinite_quantity_before_exchange_call(qty):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            raise AssertionError("nonfinite qty should fail before price lookup")

        def create_order(self, **kwargs):
            raise AssertionError("nonfinite qty should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order quantity must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=qty))


def test_ccxt_order_rejects_nonpositive_explicit_price_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            raise AssertionError("explicit invalid price should fail before price lookup")

        def create_order(self, **kwargs):
            raise AssertionError("explicit invalid price should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order price must be positive"):
        broker.place_order(
            Order("BTCUSDT", OrderSide.BUY, qty=1.0, type=OrderType.LIMIT, price=0.0)
        )


def test_ccxt_order_rejects_nonpositive_reference_price_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 0.0}

        def create_order(self, **kwargs):
            raise AssertionError("invalid reference price should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Ticker price must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


@pytest.mark.parametrize("price", [float("nan"), float("inf")])
def test_ccxt_order_rejects_nonfinite_reference_price_before_exchange_call(price):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": price}

        def create_order(self, **kwargs):
            raise AssertionError("invalid reference price should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Ticker price must be finite"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


def test_ccxt_limit_order_requires_price_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            raise AssertionError("missing limit price should fail before price lookup")

        def create_order(self, **kwargs):
            raise AssertionError("missing limit price should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=100
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Limit order price is required"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0, type=OrderType.LIMIT))


def test_ccxt_order_rejects_filled_notional_above_cap_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 200.0, "filled": 1.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=150,
        max_fill_slippage_bps=20_000,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Filled order notional"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_order_accepts_fill_within_slippage_cap():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            return {"average": 100.5, "filled": 1.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=150,
        max_fill_slippage_bps=51,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert fill.price == pytest.approx(100.5)


def test_ccxt_order_rejects_fill_slippage_above_cap_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 101.01, "filled": 1.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance",
        market_type="spot",
        live=True,
        max_notional_usd=150,
        max_fill_slippage_bps=100,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Filled order slippage"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_order_rejects_nonpositive_filled_price_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 0.0, "filled": 1.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Filled order price must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_order_rejects_nonpositive_filled_quantity_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": 0.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Filled order quantity must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_order_rejects_partial_fill_quantity_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": 0.5, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Refusing to accept a partial fill"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_order_rejects_overfilled_quantity_after_exchange_response():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": 1.01, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="exceeds requested quantity"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"filled": 1.0, "fee": {"cost": 0.1}}, "Filled order price missing"),
        ({"average": 100.0, "fee": {"cost": 0.1}}, "Filled order quantity missing"),
    ],
)
def test_ccxt_order_rejects_missing_fill_fields_after_exchange_response(result, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return dict(result)

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


@pytest.mark.parametrize("status", ["closed", "filled", None])
def test_ccxt_order_accepts_closed_or_missing_status_with_valid_fill(status):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            result = {"average": 100.0, "filled": 1.0, "fee": {"cost": 0.1}}
            if status is not None:
                result["status"] = status
            return result

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert fill.qty == pytest.approx(1.0)


def test_ccxt_order_accepts_matching_response_side_and_symbol():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            return {
                "symbol": "BTC/USDT",
                "side": "buy",
                "average": 100.0,
                "filled": 1.0,
                "fee": {"cost": 0.1},
            }

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert fill.symbol == "BTCUSDT"
    assert fill.side == OrderSide.BUY


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"symbol": "ETH/USDT", "side": "buy"}, "Exchange order symbol"),
        ({"symbol": "BTC/USDT", "side": "sell"}, "Exchange order side"),
    ],
)
def test_ccxt_order_rejects_response_symbol_or_side_mismatch(response, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            result = {"average": 100.0, "filled": 1.0, "fee": {"cost": 0.1}}
            result.update(response)
            return result

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


@pytest.mark.parametrize(
    "status", ["open", "canceled", "cancelled", "rejected", "expired", "partial"]
)
def test_ccxt_order_rejects_unfinished_or_failed_status_after_exchange_response(status):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"status": status, "average": 100.0, "filled": 1.0, "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="is not closed/filled"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("average", float("nan"), "Filled order price must be positive"),
        ("average", float("inf"), "Filled order price must be positive"),
        ("filled", float("nan"), "Filled order quantity must be positive"),
        ("filled", float("inf"), "Filled order quantity must be positive"),
    ],
)
def test_ccxt_order_rejects_nonfinite_fill_values_after_exchange_response(field, value, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            result = {"average": 100.0, "filled": 1.0, "fee": {"cost": 0.1}}
            result[field] = value
            return result

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


@pytest.mark.parametrize("fee", [-0.01, float("nan"), float("inf")])
def test_ccxt_order_rejects_invalid_fill_fee_after_exchange_response(fee):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": 1.0, "fee": {"cost": fee}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binance", market_type="spot", live=True, max_notional_usd=150
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Filled order fee"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))

    assert len(broker._client.created) == 1


def test_ccxt_futures_reduce_only_param():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = None

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            return [{"contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            self.created = kwargs
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))

    assert fill.symbol == "BTCUSDT"
    assert broker._client.created["symbol"] == "BTC/USDT:USDT"
    assert broker._client.created["params"] == {"reduceOnly": True}


def test_ccxt_futures_reduce_only_close_can_exceed_notional_cap():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = None

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            return [{"contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            self.created = kwargs
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=50
    )
    broker.name = "fake"
    broker._client = FakeClient()

    fill = broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=1.0, reduce_only=True))

    assert fill.qty == pytest.approx(1.0)
    assert broker._client.created["params"] == {"reduceOnly": True}


def test_ccxt_futures_reduce_only_close_rejects_quantity_above_position():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            return [{"contracts": 0.5, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            raise AssertionError("oversized reduce-only order should not be sent")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Reduce-only quantity"):
        broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.6, reduce_only=True))


def test_ccxt_futures_reduce_only_close_rejects_wrong_position_side():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_positions(self, symbols):
            return [{"contracts": 0.5, "side": "short", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            raise AssertionError("wrong-side reduce-only order should not be sent")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="requires an existing long"):
        broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))


def test_ccxt_futures_open_order_still_rejects_notional_above_cap():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            raise AssertionError("margin mode should not be set after notional rejection")

        def set_leverage(self, leverage, symbol):
            raise AssertionError("leverage should not be set after notional rejection")

        def create_order(self, **kwargs):
            raise AssertionError("order should not be created after notional rejection")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=50
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order notional"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


def test_ccxt_futures_entry_reapplies_and_reads_back_risk_settings_every_time():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []
            self.leverage_calls = []
            self.margin_calls = []
            self.position_mode_calls = []
            self.open_order_calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_leverage(self, leverage, symbol):
            self.leverage_calls.append((leverage, symbol))

        def set_margin_mode(self, margin_mode, symbol):
            self.margin_calls.append((margin_mode, symbol))

        def set_position_mode(self, hedged, symbol):
            self.position_mode_calls.append((hedged, symbol))

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_open_orders(self, symbol, params):
            self.open_order_calls.append((symbol, params))
            return []

        def fetch_positions(self, symbols):
            return [_futures_settings_position(leverage=2)]

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=2,
    )
    broker.name = "fake"
    broker._client = FakeClient()
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert broker._client.margin_calls == [
        ("isolated", "BTC/USDT:USDT"),
        ("isolated", "BTC/USDT:USDT"),
    ]
    assert broker._client.leverage_calls == [
        (2, "BTC/USDT:USDT"),
        (2, "BTC/USDT:USDT"),
    ]
    assert broker._client.position_mode_calls == [
        (False, "BTC/USDT:USDT"),
        (False, "BTC/USDT:USDT"),
    ]
    assert broker._client.open_order_calls == [
        (None, {}),
        (None, {"trigger": True}),
        (None, {}),
        (None, {"trigger": True}),
    ]
    assert broker._client.created[0]["symbol"] == "BTC/USDT:USDT"
    assert broker._client.created[0]["params"] == {}


def test_ccxt_futures_open_order_refuses_when_margin_mode_cannot_be_set():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(RuntimeError, match="cannot set isolated margin mode"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))


def test_ccxt_futures_open_order_allows_already_isolated_margin_message():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []
            self.leverage_calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            raise RuntimeError("No need to change margin type.")

        def set_leverage(self, leverage, symbol):
            self.leverage_calls.append((leverage, symbol))

        def set_position_mode(self, hedged, symbol):
            return None

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_open_orders(self, symbol, params):
            return []

        def fetch_positions(self, symbols):
            return [_futures_settings_position()]

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {"average": 100.0, "filled": kwargs["amount"], "fee": {"cost": 0.1}}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
    )
    broker.name = "fake"
    broker._client = FakeClient()
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert broker._client.leverage_calls == [(1, "BTC/USDT:USDT")]
    assert broker._client.created[0]["symbol"] == "BTC/USDT:USDT"
    assert len(broker._client.created) == 1


def test_ccxt_futures_margin_mode_rejects_unrelated_already_message():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def set_margin_mode(self, margin_mode, symbol):
            raise RuntimeError("Account is already locked for maintenance")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures")
    broker._client = FakeClient()

    with pytest.raises(RuntimeError, match="could not set isolated margin mode"):
        broker._ensure_futures_margin_mode("BTCUSDT")


def test_ccxt_futures_position_mode_accepts_exact_binance_already_set_code():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def set_position_mode(self, hedged, symbol):
            raise RuntimeError(
                'binanceusdm {"code":-4059,"msg":"No need to change position side."}'
            )

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures")
    broker._client = FakeClient()

    broker._ensure_futures_position_mode("BTCUSDT")


@pytest.mark.parametrize(
    ("position_payload", "message"),
    [
        (_futures_settings_position(margin_mode="cross"), "isolated margin mode"),
        (_futures_settings_position(leverage=2), "does not match configured leverage"),
        ([], "exactly one matching position-settings record"),
    ],
)
def test_ccxt_futures_entry_rejects_unconfirmed_risk_settings(position_payload, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

        def set_leverage(self, leverage, symbol):
            return None

        def set_position_mode(self, hedged, symbol):
            return None

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_positions(self, symbols):
            return position_payload if isinstance(position_payload, list) else [position_payload]

        def create_order(self, **kwargs):
            raise AssertionError("entry must not be submitted without verified risk settings")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(RuntimeError, match=message):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))


@pytest.mark.parametrize(
    ("position_side", "expected_signed_qty"),
    [("long", "0.25"), ("short", "-0.25")],
)
def test_ccxt_futures_entry_rechecks_flatness_immediately_before_create_order(
    position_side,
    expected_signed_qty,
):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

        def set_leverage(self, leverage, symbol):
            return None

        def set_position_mode(self, hedged, symbol):
            return None

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_open_orders(self, symbol, params):
            return []

        def fetch_positions(self, symbols):
            return [
                _futures_settings_position(
                    contracts=0.25,
                    side=position_side,
                    entry_price=100.0,
                )
            ]

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            raise AssertionError("non-flat futures entry reached create_order")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
    )
    broker.name = "fake"
    broker._client = FakeClient()
    with pytest.raises(RuntimeError, match=rf"signed qty {expected_signed_qty}"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert broker._client.created == []


def test_ccxt_futures_open_order_refuses_when_leverage_cannot_be_set():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(RuntimeError, match="cannot set leverage"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))


def test_ccxt_futures_open_order_rejects_unsafe_leverage_value():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

        def set_leverage(self, leverage, symbol):
            raise AssertionError("unsafe leverage should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=10,
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="MAX_FUTURES_LEVERAGE"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))


def test_ccxt_futures_open_order_rejects_non_isolated_margin_mode_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            raise AssertionError("cross margin should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1000,
        max_futures_leverage=1,
        futures_margin_mode="cross",
    )
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="FUTURES_MARGIN_MODE"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))


def _protective_payload(
    *,
    status="open",
    filled=0.0,
    average=None,
    client_id="tb-sl-0123456789abcdef0123456789ab",
):
    return {
        "id": "987654321",
        "clientOrderId": client_id,
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "amount": 1.0,
        "triggerPrice": 95.0,
        "status": status,
        "filled": filled,
        "average": average,
        "fee": {"cost": 0.0},
        "reduceOnly": True,
        "info": {"positionSide": "BOTH"},
    }


def _ccxt_protective_broker(client):
    from src.execution.ccxt_broker import CcxtBroker

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        quote_asset="USDT",
        futures_margin_mode="isolated",
    )
    broker.name = "fake"
    broker._client = client
    return broker


def test_ccxt_open_order_inventory_queries_regular_and_conditional_paths():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def fetch_open_orders(self, symbol, params):
            self.calls.append((symbol, params))
            if params == {"trigger": True}:
                return [
                    {
                        "info": {
                            "algoId": 456,
                            "clientAlgoId": "manual-stop-1",
                            "symbol": "BTCUSDT",
                            "algoStatus": "NEW",
                        }
                    }
                ]
            return [
                {
                    "id": 123,
                    "clientOrderId": "manual-order-1",
                    "symbol": "BTC/USDT:USDT",
                    "status": "partially_filled",
                }
            ]

    client = FakeClient()
    broker = _ccxt_protective_broker(client)

    regular = broker.list_open_orders("BTCUSDT", conditional=False)
    conditional = broker.list_open_orders("BTCUSDT", conditional=True)

    assert regular == (
        OpenOrderIdentity(
            symbol="BTCUSDT",
            order_id="123",
            client_id="manual-order-1",
            status="partially_filled",
            conditional=False,
        ),
    )
    assert conditional == (
        OpenOrderIdentity(
            symbol="BTCUSDT",
            order_id="456",
            client_id="manual-stop-1",
            status="open",
            conditional=True,
        ),
    )
    assert client.calls == [
        ("BTC/USDT:USDT", {}),
        ("BTC/USDT:USDT", {"trigger": True}),
    ]


def test_ccxt_account_inventory_returns_other_symbol_orders_and_positions():
    class FakeClient:
        def fetch_open_orders(self, symbol, params):
            assert symbol is None
            return [
                {
                    "id": "eth-order-1",
                    "clientOrderId": "manual-eth-1",
                    "symbol": "ETH/USDT:USDT",
                    "status": "open",
                }
            ]

        def fetch_positions(self, symbols):
            assert symbols is None
            return [
                _futures_settings_position(),
                {
                    "symbol": "ETH/USDT:USDT",
                    "contracts": 2.0,
                    "side": "short",
                    "entryPrice": 2500.0,
                },
            ]

    broker = _ccxt_protective_broker(FakeClient())

    assert broker.list_account_open_orders(conditional=False) == (
        OpenOrderIdentity(
            symbol="ETH/USDT:USDT",
            order_id="eth-order-1",
            client_id="manual-eth-1",
            status="open",
            conditional=False,
        ),
    )
    assert broker.list_account_futures_positions() == (
        FuturesPositionIdentity(
            symbol="ETH/USDT:USDT",
            qty=-2.0,
            avg_price=2500.0,
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [None],
        [{"symbol": "ETHUSDT", "contracts": "nan", "side": "long", "entryPrice": 1}],
        [{"symbol": "ETHUSDT", "contracts": 1, "side": "mystery", "entryPrice": 1}],
    ],
)
def test_ccxt_account_position_inventory_fails_closed_on_malformed_payload(payload):
    class FakeClient:
        def fetch_positions(self, symbols):
            return payload

    broker = _ccxt_protective_broker(FakeClient())

    with pytest.raises(ValueError):
        broker.list_account_futures_positions()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "must be a list"),
        ([None], "item 0 must be an object"),
        ([{"info": []}], "info must be an object"),
        (
            [
                {
                    "id": "1",
                    "clientOrderId": "manual-1",
                    "symbol": "ETH/USDT:USDT",
                    "status": "open",
                }
            ],
            "does not match",
        ),
        (
            [{"id": "1", "symbol": "BTCUSDT", "status": "open"}],
            "client id is missing or invalid",
        ),
        (
            [
                {
                    "id": "1",
                    "clientOrderId": "manual-1",
                    "symbol": "BTCUSDT",
                    "status": "closed",
                }
            ],
            "is not an active status",
        ),
    ],
)
def test_ccxt_open_order_inventory_rejects_malformed_or_unsafe_responses(
    payload,
    message,
):
    class FakeClient:
        def fetch_open_orders(self, symbol, params):
            return payload

    broker = _ccxt_protective_broker(FakeClient())

    with pytest.raises(ValueError, match=message):
        broker.list_open_orders("BTCUSDT", conditional=False)


def test_ccxt_futures_entry_refuses_outstanding_orders_immediately_before_submit():
    class FakeClient:
        def __init__(self):
            self.created = []
            self.open_order_calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

        def set_leverage(self, leverage, symbol):
            return None

        def set_position_mode(self, hedged, symbol):
            return None

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_open_orders(self, symbol, params):
            self.open_order_calls.append((symbol, params))
            if params:
                return []
            return [
                {
                    "id": "123",
                    "clientOrderId": "manual-order-1",
                    "symbol": "ETH/USDT:USDT",
                    "status": "open",
                }
            ]

        def fetch_positions(self, symbols):
            return [_futures_settings_position()]

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            raise AssertionError("entry must not be submitted with outstanding orders")

    client = FakeClient()
    broker = _ccxt_protective_broker(client)
    broker.config.max_notional_usd = 1000

    with pytest.raises(RuntimeError, match=r"regular=1, conditional=0"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert client.created == []
    assert client.open_order_calls == [
        (None, {}),
        (None, {"trigger": True}),
    ]


def test_ccxt_multi_symbol_mode_allows_an_unrelated_position_and_order():
    class FakeClient:
        def __init__(self):
            self.created = []
            self.open_order_calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_margin_mode(self, margin_mode, symbol):
            return None

        def set_leverage(self, leverage, symbol):
            return None

        def set_position_mode(self, hedged, symbol):
            return None

        def fetch_position_mode(self, symbol):
            return {"hedged": False}

        def fetch_open_orders(self, symbol, params):
            self.open_order_calls.append((symbol, params))
            return []

        def fetch_positions(self, symbols):
            if symbols == ["BTC/USDT:USDT"]:
                return [_futures_settings_position()]
            raise AssertionError("multi-symbol order must not require account flatness")

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {
                "status": "closed",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "type": "market",
                "amount": 0.1,
                "filled": 0.1,
                "average": 100.0,
                "fee": {"cost": 0.01},
            }

    from src.execution.ccxt_broker import CcxtBroker

    client = FakeClient()
    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
        max_notional_usd=1_000,
        max_futures_leverage=1,
        allow_multi_symbol_positions=True,
    )
    broker.name = "fake"
    broker._client = client

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert fill.qty == pytest.approx(0.1)
    assert client.open_order_calls == [("BTC/USDT:USDT", {})]
    assert len(client.created) == 1


def test_ccxt_places_binance_usdm_reduce_only_stop_market_and_validates_ack():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def __init__(self):
            self.created = None

        def fetch_positions(self, symbols):
            return [{"contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            self.created = kwargs
            return _protective_payload(client_id=client_id)

    client = FakeClient()
    broker = _ccxt_protective_broker(client)

    protective = broker.place_protective_stop(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        qty=1.0,
        trigger_price=95.0,
        client_id=client_id,
    )

    assert broker.supports_native_protective_stops() is True
    assert protective.status == ProtectiveOrderStatus.OPEN
    assert protective.order_id == "987654321"
    assert client.created == {
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "sell",
        "amount": 1.0,
        "price": None,
        "params": {
            "stopLossPrice": 95.0,
            "reduceOnly": True,
            "positionSide": "BOTH",
            "newClientOrderId": client_id,
        },
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"id": None}, "order id missing"),
        ({"clientOrderId": "wrong-client"}, "client id mismatch"),
        ({"status": None}, "status .* missing or unsupported"),
        ({"status": "rejected"}, "unsafe status"),
        ({"reduceOnly": False}, "does not prove reduceOnly=true"),
        ({"side": "buy"}, "side mismatch"),
        ({"amount": 0.5}, "quantity mismatch"),
        ({"triggerPrice": 96.0}, "trigger price mismatch"),
        ({"info": {"positionSide": "LONG"}}, "not one-way BOTH"),
    ],
)
def test_ccxt_rejects_unproven_or_mismatched_protective_stop_ack(updates, message):
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def fetch_positions(self, symbols):
            return [{"contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            payload = _protective_payload(client_id=client_id)
            payload.update(updates)
            return payload

    broker = _ccxt_protective_broker(FakeClient())

    with pytest.raises(ValueError, match=message):
        broker.place_protective_stop(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            qty=1.0,
            trigger_price=95.0,
            client_id=client_id,
        )


def test_ccxt_fetches_and_adopts_only_complete_trigger_stop_fill():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def __init__(self):
            self.fetch_args = None

        def fetch_order(self, *args):
            self.fetch_args = args
            return _protective_payload(
                status="closed",
                filled=1.0,
                average=94.5,
                client_id=client_id,
            )

    client = FakeClient()
    broker = _ccxt_protective_broker(client)

    protective = broker.get_protective_stop(
        symbol="BTCUSDT",
        order_id="987654321",
        client_id=client_id,
    )

    assert protective.status == ProtectiveOrderStatus.TRIGGERED
    assert protective.filled_qty == pytest.approx(1.0)
    assert protective.average_price == pytest.approx(94.5)
    assert client.fetch_args == ("987654321", "BTC/USDT:USDT", {"trigger": True})


def test_ccxt_rejects_partial_triggered_stop_as_unsafe_to_adopt():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def fetch_order(self, *args):
            return _protective_payload(
                status="closed",
                filled=0.5,
                average=94.5,
                client_id=client_id,
            )

    broker = _ccxt_protective_broker(FakeClient())

    with pytest.raises(ValueError, match="not fully filled"):
        broker.get_protective_stop(
            symbol="BTCUSDT",
            order_id="987654321",
            client_id=client_id,
        )


def test_ccxt_cancels_trigger_order_then_reads_back_terminal_status():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def __init__(self):
            self.cancel_args = None
            self.fetch_args = None

        def cancel_order(self, *args):
            self.cancel_args = args
            return {"id": "987654321"}

        def fetch_order(self, *args):
            self.fetch_args = args
            return _protective_payload(status="canceled", client_id=client_id)

    client = FakeClient()
    broker = _ccxt_protective_broker(client)

    protective = broker.cancel_protective_stop(
        symbol="BTCUSDT",
        order_id="987654321",
        client_id=client_id,
    )

    assert protective.status == ProtectiveOrderStatus.CANCELED
    expected = ("987654321", "BTC/USDT:USDT", {"trigger": True})
    assert client.cancel_args == expected
    assert client.fetch_args == expected


def test_ccxt_rejects_cancel_ack_when_protective_stop_remains_open():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def cancel_order(self, *args):
            return {}

        def fetch_order(self, *args):
            return _protective_payload(status="open", client_id=client_id)

    broker = _ccxt_protective_broker(FakeClient())

    with pytest.raises(RuntimeError, match="still open after cancellation"):
        broker.cancel_protective_stop(
            symbol="BTCUSDT",
            order_id="987654321",
            client_id=client_id,
        )


def test_ccxt_can_cancel_ambiguous_stop_placement_by_deterministic_client_id():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class FakeClient:
        def __init__(self):
            self.cancel_args = None

        def cancel_order(self, *args):
            self.cancel_args = args
            return {}

        def fetch_order(self, *args):
            return _protective_payload(status="canceled", client_id=client_id)

    client = FakeClient()
    broker = _ccxt_protective_broker(client)

    protective = broker.cancel_protective_stop(
        symbol="BTCUSDT",
        order_id=None,
        client_id=client_id,
    )

    assert protective.status == ProtectiveOrderStatus.CANCELED
    assert client.cancel_args == (
        client_id,
        "BTC/USDT:USDT",
        {"trigger": True, "clientAlgoId": client_id},
    )


def test_ccxt_live_binance_usdm_rejects_unvalidated_installed_version(monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.execution.ccxt_broker import CcxtBroker

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        live=True,
    )
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(__version__="4.5.63"))

    with pytest.raises(RuntimeError, match="requires ccxt==4.5.64"):
        broker._build_client()


class _FullPrecisionClient:
    def __init__(
        self,
        *,
        symbol="BTC/USDT:USDT",
        amount_output="0.123",
        price_output="95.1",
        limits=None,
    ):
        self.symbol = symbol
        self.amount_output = amount_output
        self.price_output = price_output
        self.market_payload = {
            "symbol": symbol,
            "limits": limits
            or {
                "amount": {"min": 0.001, "max": 1000.0},
                "market": {"min": 0.001, "max": 500.0},
                "price": {"min": 0.1, "max": 1_000_000.0},
                "cost": {"min": 5.0, "max": 1_000_000.0},
            },
        }
        self.load_calls = 0
        self.amount_calls = []
        self.price_calls = []
        self.created = []

    def load_markets(self):
        self.load_calls += 1
        return {self.symbol: self.market_payload}

    def market(self, symbol):
        assert symbol == self.symbol
        return self.market_payload

    def amount_to_precision(self, symbol, amount):
        self.amount_calls.append((symbol, amount))
        return self.amount_output

    def price_to_precision(self, symbol, price):
        self.price_calls.append((symbol, price))
        return self.price_output


def _precision_broker(client, *, market_type="futures"):
    from src.execution.ccxt_broker import CcxtBroker

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm" if market_type == "futures" else "binance",
        market_type=market_type,
        live=True,
        max_notional_usd=1000,
    )
    broker.name = "precision-fake"
    broker._client = client
    broker._precision_markets = {}
    return broker


def test_ccxt_precision_hooks_load_market_once_and_return_exchange_values():
    client = _FullPrecisionClient()
    broker = _precision_broker(client)

    qty = broker.normalize_order_qty("BTCUSDT", 0.1239, price=100.0)
    trigger = broker.normalize_order_price("BTCUSDT", 95.19)

    assert qty == pytest.approx(0.123)
    assert trigger == pytest.approx(95.1)
    assert client.load_calls == 1
    assert client.amount_calls == [("BTC/USDT:USDT", 0.1239)]
    assert client.price_calls == [("BTC/USDT:USDT", 95.19)]


@pytest.mark.parametrize(
    ("amount_output", "limits", "kwargs", "message"),
    [
        (
            "0",
            {"amount": {"min": 0.001, "max": 10.0}},
            {},
            "Precision-normalized order quantity must be positive",
        ),
        (
            "0.0005",
            {"amount": {"min": 0.001, "max": 10.0}},
            {},
            "below exchange minimum",
        ),
        (
            "0.005",
            {
                "amount": {"min": 0.001, "max": 10.0},
                "market": {"min": 0.01, "max": 10.0},
            },
            {},
            "Market order quantity .* below exchange minimum",
        ),
        (
            "0.01",
            {
                "amount": {"min": 0.001, "max": 10.0},
                "cost": {"min": 5.0, "max": 1000.0},
            },
            {"price": 100.0},
            "Order notional .* below exchange minimum",
        ),
        (
            "0.124",
            {"amount": {"min": 0.001, "max": 10.0}},
            {},
            "increased intended quantity",
        ),
    ],
)
def test_ccxt_quantity_normalization_rejects_impossible_or_below_minimum_orders(
    amount_output,
    limits,
    kwargs,
    message,
):
    client = _FullPrecisionClient(amount_output=amount_output, limits=limits)
    broker = _precision_broker(client)
    raw_qty = 0.1235 if amount_output == "0.124" else 0.01

    with pytest.raises(ValueError, match=message):
        broker.normalize_order_qty("BTCUSDT", raw_qty, **kwargs)

    assert client.created == []


def test_ccxt_reduce_only_normalization_exempts_minimum_notional_but_not_amount_filters():
    client = _FullPrecisionClient(
        amount_output="0.01",
        limits={
            "amount": {"min": 0.001, "max": 10.0},
            "cost": {"min": 5.0, "max": 1000.0},
        },
    )
    broker = _precision_broker(client)

    assert broker.normalize_order_qty(
        "BTCUSDT",
        0.01,
        price=100.0,
        reduce_only=True,
    ) == pytest.approx(0.01)


def test_ccxt_price_normalization_rejects_exchange_price_minimum():
    client = _FullPrecisionClient(
        price_output="0.05",
        limits={"price": {"min": 0.1, "max": 1000.0}},
    )
    broker = _precision_broker(client)

    with pytest.raises(ValueError, match="Order price .* below exchange minimum"):
        broker.normalize_order_price("BTCUSDT", 0.051)


def test_ccxt_place_order_submits_and_validates_normalized_quantity():
    class Client(_FullPrecisionClient):
        def __init__(self):
            super().__init__(symbol="BTC/USDT", amount_output="0.123")

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}, "total": {"USDT": 500.0}}

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return {
                "average": 100.0,
                "filled": kwargs["amount"],
                "fee": {"cost": 0.0},
                "status": "closed",
            }

    client = Client()
    broker = _precision_broker(client, market_type="spot")

    fill = broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1239))

    assert fill.qty == pytest.approx(0.123)
    assert client.created[0]["amount"] == pytest.approx(0.123)


def test_ccxt_place_order_rejects_min_notional_before_exchange_submission():
    class Client(_FullPrecisionClient):
        def __init__(self):
            super().__init__(symbol="BTC/USDT", amount_output="0.01")

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_order(self, **kwargs):
            raise AssertionError("below-minimum order must not be submitted")

    client = Client()
    broker = _precision_broker(client, market_type="spot")

    with pytest.raises(ValueError, match="Order notional .* below exchange minimum"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.01))

    assert client.created == []


def test_ccxt_protective_stop_uses_normalized_quantity_and_trigger():
    client_id = "tb-sl-0123456789abcdef0123456789ab"

    class Client(_FullPrecisionClient):
        def __init__(self):
            super().__init__(amount_output="1.0", price_output="95.1")

        def fetch_positions(self, symbols):
            return [{"contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        def create_order(self, **kwargs):
            self.created.append(kwargs)
            return _protective_payload(client_id=client_id) | {
                "triggerPrice": 95.1,
            }

    client = Client()
    broker = _precision_broker(client)

    protective = broker.place_protective_stop(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        qty=1.0004,
        trigger_price=95.19,
        client_id=client_id,
    )

    assert protective.qty == pytest.approx(1.0)
    assert protective.trigger_price == pytest.approx(95.1)
    assert client.created[0]["amount"] == pytest.approx(1.0)
    assert client.created[0]["params"]["stopLossPrice"] == pytest.approx(95.1)


def test_ccxt_futures_entry_refuses_unverified_hedge_mode_before_order():
    from src.execution.ccxt_broker import CcxtBroker

    class Client:
        def set_position_mode(self, hedged, symbol):
            assert hedged is False

        def fetch_position_mode(self, symbol):
            return {"hedged": True}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True)
    broker._client = Client()

    with pytest.raises(RuntimeError, match="did not confirm one-way"):
        broker._ensure_futures_position_mode("BTCUSDT")


def test_ccxt_position_mode_verification_is_read_only():
    from src.execution.ccxt_broker import CcxtBroker

    class Client:
        def __init__(self):
            self.fetch_calls = []

        def fetch_position_mode(self, symbol):
            self.fetch_calls.append(symbol)
            return {"hedged": False}

        def set_position_mode(self, hedged, symbol):
            raise AssertionError("read-only verification must not mutate position mode")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="key",
        testnet=False,
    )
    broker._client = Client()

    assert broker.verify_one_way_position_mode("BTCUSDT") is True
    assert broker._client.fetch_calls == ["BTC/USDT:USDT"]
    assert broker.account_fingerprint == broker.config.account_fingerprint


def test_ccxt_position_mode_verification_reports_hedge_mode():
    from src.execution.ccxt_broker import CcxtBroker

    class Client:
        def fetch_position_mode(self, symbol):
            return {"hedged": True}

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures")
    broker._client = Client()

    assert broker.verify_one_way_position_mode("BTCUSDT") is False


def test_ccxt_futures_entry_refuses_client_without_position_mode_verification():
    from src.execution.ccxt_broker import CcxtBroker

    class Client:
        def set_position_mode(self, hedged, symbol):
            return None

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True)
    broker._client = Client()

    with pytest.raises(RuntimeError, match="cannot verify one-way"):
        broker._ensure_futures_position_mode("BTCUSDT")
