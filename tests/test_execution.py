"""Tests for the futures execution layer (paper broker, config, ccxt guards)."""

from __future__ import annotations

import pytest

from src.execution import ExchangeConfig, Order, OrderSide, OrderType, PaperBroker, Position


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
def test_paper_broker_rejects_reduce_only_orders_that_do_not_reduce(starting_order, reduce_order, message):
    b = PaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)
    b.place_order(starting_order)

    with pytest.raises(ValueError, match=message):
        b.place_order(reduce_order)


def test_paper_broker_reduce_only_requires_open_position():
    b = PaperBroker(price_source=_Px(100.0), fee_bps=0, slippage_bps=0)

    with pytest.raises(ValueError, match="requires an open position"):
        b.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))


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
    for var in ["EXCHANGE", "TRADING_LIVE", "EXCHANGE_TESTNET", "MAX_NOTIONAL_USD", "MAX_FILL_SLIPPAGE_BPS"]:
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
    import os

    assert os.environ["EXCHANGE"] == "okx"
    assert os.environ["MAX_NOTIONAL_USD"] == "42"


def test_load_dotenv_rejects_symlink_without_loading_target(tmp_path, monkeypatch):
    import os

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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
    broker.name = "fake"
    broker._client = FakeClient()

    assert broker.get_balance() == 500.0
    assert broker.get_position("BTCUSDT").qty == pytest.approx(0.25)
    fill = broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.1, reduce_only=True))

    assert fill.qty == pytest.approx(0.1)
    assert broker._client.tickers == ["BTC/USDT"]
    assert broker._client.created["symbol"] == "BTC/USDT"
    assert broker._client.created["params"] == {}


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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.get_position("BTCUSDT")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"contracts": float("nan"), "side": "long", "entryPrice": 90.0}, "Futures position contracts must be finite"),
        ({"contracts": -1.0, "side": "long", "entryPrice": 90.0}, "Futures position contracts must be non-negative"),
        ({"contracts": "", "side": "long", "entryPrice": 90.0}, "Futures position contracts must be numeric"),
        ({"contracts": 0.1, "side": "flat", "entryPrice": 90.0}, "Futures position side must be long or short"),
        ({"contracts": 0.1, "side": "long", "entryPrice": 0.0}, "Futures position entry price must be positive"),
        ({"contracts": 0.1, "side": "long", "entryPrice": float("nan")}, "Futures position entry price must be finite"),
        ({"contracts": 0.1, "side": "long", "entryPrice": ""}, "Futures position entry price must be numeric"),
    ],
)
def test_ccxt_futures_get_position_rejects_invalid_position_payload(payload, message):
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_positions(self, symbols):
            return [payload]

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match="Order price must be positive"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0, type=OrderType.LIMIT, price=0.0))


def test_ccxt_order_rejects_nonpositive_reference_price_before_exchange_call():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def fetch_ticker(self, symbol):
            return {"last": 0.0}

        def create_order(self, **kwargs):
            raise AssertionError("invalid reference price should fail before exchange call")

    broker = CcxtBroker.__new__(CcxtBroker)
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=100)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
    broker.name = "fake"
    broker._client = FakeClient()

    with pytest.raises(ValueError, match=message):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


@pytest.mark.parametrize("status", ["open", "canceled", "cancelled", "rejected", "expired", "partial"])
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binance", market_type="spot", live=True, max_notional_usd=150)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=50)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=1000)
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
    broker.config = ExchangeConfig(exchange="binanceusdm", market_type="futures", live=True, max_notional_usd=50)
    broker.name = "fake"
    broker._client = FakeClient()
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

    with pytest.raises(ValueError, match="Order notional"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=1.0))


def test_ccxt_futures_open_order_sets_configured_leverage_once():
    from src.execution.ccxt_broker import CcxtBroker

    class FakeClient:
        def __init__(self):
            self.created = []
            self.leverage_calls = []
            self.margin_calls = []

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def set_leverage(self, leverage, symbol):
            self.leverage_calls.append((leverage, symbol))

        def set_margin_mode(self, margin_mode, symbol):
            self.margin_calls.append((margin_mode, symbol))

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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert broker._client.margin_calls == [("isolated", "BTC/USDT:USDT")]
    assert broker._client.leverage_calls == [(2, "BTC/USDT:USDT")]
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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))

    assert broker._client.leverage_calls == [(1, "BTC/USDT:USDT")]
    assert broker._client.created[0]["symbol"] == "BTC/USDT:USDT"
    assert len(broker._client.created) == 1


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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

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
    broker._leverage_set_symbols = set()
    broker._margin_mode_set_symbols = set()

    with pytest.raises(ValueError, match="FUTURES_MARGIN_MODE"):
        broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.1))
