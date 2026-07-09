from src.autopilot.config import ProductConfig
from src.autopilot.exchange_policy import (
    split_symbol,
    validate_exchange_policy,
    validate_product_symbol_policy,
)
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
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def test_split_symbol_handles_binance_and_ccxt_forms():
    assert split_symbol("BTCUSDT") == ("BTC", "USDT", None)
    assert split_symbol("BTC/USDT") == ("BTC", "USDT", None)
    assert split_symbol("BTC/USDT:USDT") == ("BTC", "USDT", "USDT")


def test_product_symbol_policy_allows_active_income_usdt_margined_btc_futures(tmp_path):
    active_product = product(tmp_path, symbol="BTC/USDT:USDT")

    assert validate_product_symbol_policy(active_product) == []


def test_product_symbol_policy_rejects_wrong_active_income_quote(tmp_path):
    active_product = product(tmp_path, symbol="BTCUSDC")

    errors = validate_product_symbol_policy(active_product)

    assert errors == ["active_income: active income symbol must be BTC/USDT; got 'BTCUSDC'."]


def test_product_symbol_policy_rejects_spot_btc_accumulation_settlement(tmp_path):
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
        symbol="BTC/USDT:USDT",
    )

    errors = validate_product_symbol_policy(btc_product)

    assert errors == ["btc_accumulation: BTC accumulation spot symbol must not include a settlement asset."]


def test_exchange_policy_rejects_non_binance_btc_accumulation_spot(tmp_path):
    btc_product = product(
        tmp_path,
        name="btc_accumulation",
        objective="btc_accumulation",
        base_asset="BTC",
        market="spot",
        starting_equity=1.0,
        symbol="BTCUSDT",
    )
    cfg = ExchangeConfig(exchange="kraken", market_type="spot", quote_asset="USDT")

    errors = validate_exchange_policy(btc_product, cfg)

    assert errors == [
        "btc_accumulation: BTC accumulation live spot must use Binance spot "
        "(SPOT_EXCHANGE=binance); got 'kraken'."
    ]


def test_exchange_policy_rejects_active_income_non_binance_futures(tmp_path):
    active_product = product(tmp_path)
    cfg = ExchangeConfig(exchange="bybit", market_type="futures", quote_asset="USDT")

    errors = validate_exchange_policy(active_product, cfg)

    assert errors == [
        "active_income: active income live futures must use Binance USDT futures "
        "(FUTURES_EXCHANGE=binanceusdm); got 'bybit'."
    ]


def test_exchange_policy_rejects_active_income_leverage_above_one(tmp_path):
    active_product = product(tmp_path)
    cfg = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        quote_asset="USDT",
        max_futures_leverage=2,
    )

    errors = validate_exchange_policy(active_product, cfg)

    assert errors == ["active_income: active income futures must use MAX_FUTURES_LEVERAGE=1."]


def test_exchange_policy_rejects_active_income_cross_margin(tmp_path):
    active_product = product(tmp_path)
    cfg = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        quote_asset="USDT",
        futures_margin_mode="cross",
    )

    errors = validate_exchange_policy(active_product, cfg)

    assert errors == ["active_income: active income futures must use isolated margin."]
