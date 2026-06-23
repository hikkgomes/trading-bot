# Execution Layer

A broker abstraction (`src/execution`) so the algo-trading system can trade any
futures venue with one interface, paper by default and live behind explicit
safety switches.

## Brokers

| Broker | Use | Notes |
|---|---|---|
| `PaperBroker` | development, tests, paper cron | Simulated market fills with fees + slippage; injectable price source; tracks signed positions, realised/unrealised PnL and equity. Dependency-free. |
| `CcxtBroker` | live / testnet | Wraps [ccxt](https://github.com/ccxt/ccxt) — Binance USDM, Bybit, OKX, … Requires `pip install ccxt`. |

```python
from src.execution import PaperBroker, Order, OrderSide

broker = PaperBroker(price_source=lambda s: 30_000.0, starting_balance=1_000)
broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=0.01))
print(broker.get_position("BTCUSDT"), broker.equity())
```

Positions are **signed**: `qty > 0` long, `qty < 0` short, `0` flat. Balances are
in the quote currency (USDT).

## Going live — safety rails

Live trading is opt-in. Copy `.env.example` → `.env` (gitignored) and fill in
credentials. A real order is sent **only when all** of these hold:

* `TRADING_LIVE=1`
* order notional ≤ `MAX_NOTIONAL_USD` (client-side hard cap)

Otherwise `place_order` raises, so a misconfigured run cannot trade real size.
Set `EXCHANGE_TESTNET=1` to route everything to the exchange sandbox first.

```bash
EXCHANGE=binanceusdm
EXCHANGE_TESTNET=1
TRADING_LIVE=0          # flip to 1 only when you mean it
MAX_NOTIONAL_USD=100
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

```python
from src.execution.ccxt_broker import CcxtBroker
broker = CcxtBroker()   # reads .env; refuses live orders until TRADING_LIVE=1
```

## Where this plugs in

`src/run_bot.py` is the current paper executor that reads `active_strategies*.json`
and evaluates closed candles. The next integration step (see
[ROADMAP.md](ROADMAP.md)) is to route its fills through a `Broker` so the same
strategy code can drive paper or live execution by swapping the broker.
