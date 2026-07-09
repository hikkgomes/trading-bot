# Execution Layer

A broker abstraction (`src/execution`) so the algo-trading system can trade spot
or futures venues with one interface, paper by default and live behind explicit
safety switches.

## Brokers

| Broker | Use | Notes |
|---|---|---|
| `PaperBroker` | development, tests, paper autopilot cycles | Simulated market fills with fees + slippage; injectable price source; tracks signed positions, realised/unrealised PnL and equity. Dependency-free. |
| `CcxtBroker` | live / testnet | Wraps [ccxt](https://github.com/ccxt/ccxt) for spot or futures. Requires `pip install ccxt`. |

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
* order quantity > 0, checked before price lookup or exchange submission
* explicit order prices and fetched reference prices > 0; limit orders require
  an explicit price
* entry/increase order notional ≤ `MAX_NOTIONAL_USD` (client-side hard cap)
* futures reduce-only closes may exceed `MAX_NOTIONAL_USD` so emergency flatten
  can reduce existing exposure
* fill price deviation from the pre-submit reference price ≤
  `MAX_FILL_SLIPPAGE_BPS` (default 100 bps)
* exchange-reported filled quantity and fill price must both be present and
  positive before a fill is accepted into local state; broker-routed fills must
  match the requested quantity within tolerance, so partial or overfilled
  exchange responses are refused, and missing fill data is not inferred from
  requested quantity or reference price
* when the exchange reports an order status, it must be `closed`/`filled`;
  `open`, canceled, rejected, expired, or unknown statuses are refused
* when the exchange reports an order symbol or side, it must match the requested
  order after symbol normalization
* spot sell quantity ≤ current base-asset balance, checked before order
  submission so spot BTC accumulation cannot open a margin/short position
* exchange ticker, quote-balance, and position reads must be numeric and finite;
  prices and open-position entry prices must be positive, while balances and
  base/contract quantities must be non-negative
* spot buy notional ≤ current free quote balance, checked before order
  submission so BTC accumulation cannot overspend available USDT
* futures margin mode is `isolated`, and the broker successfully sets it before
  opening futures orders
* futures leverage is configured with `MAX_FUTURES_LEVERAGE` in the conservative
  range 1-3, and the broker successfully sets it before opening futures orders

Otherwise `place_order` raises, so a misconfigured run cannot trade real size.
Set `EXCHANGE_TESTNET=1` to route everything to the exchange sandbox first.
`src.autopilot.preflight` is read-only; the explicit active-income testnet order
rehearsal is `make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5`. It requires the
same approval/preflight/env gates, refuses non-testnet venues, starts only from a
flat futures position, places one tiny BTCUSDT futures market entry, and closes
it immediately through the broker close path.

```bash
FUTURES_EXCHANGE=binanceusdm
SPOT_EXCHANGE=binance
EXCHANGE_MARKET_TYPE=futures
EXCHANGE_TESTNET=1
TRADING_LIVE=0          # flip to 1 only when you mean it
MAX_NOTIONAL_USD=100
MAX_FILL_SLIPPAGE_BPS=100
QUOTE_ASSET=USDT
FUTURES_MARGIN_MODE=isolated
MAX_FUTURES_LEVERAGE=1
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

`active_income` live execution is deliberately restricted to Binance USDT-M
futures (`FUTURES_EXCHANGE=binanceusdm`), `QUOTE_ASSET=USDT`, and a BTC/USDT
product symbol, matching the research/data source and the product objective.
`btc_accumulation` is likewise restricted to spot BTC/USDT with no settlement
asset. BTC accumulation strategy artifacts must be spot step-aside shorts:
they may sell a bounded slice of existing BTC and later rebuy with the original
quote proceeds, but they cannot be long-style quote-funded BTC buys. A different
venue, quote, market, product symbol, or BTC strategy shape fails validation,
readiness, preflight, or runtime live checks before order placement.

```python
from src.execution.ccxt_broker import CcxtBroker
broker = CcxtBroker()   # reads .env; refuses live orders until TRADING_LIVE=1
```

## Where this plugs in

`src/run_bot.py` reads `active_strategies*.json` and evaluates closed candles.
By default it still runs internal paper accounting. The autopilot passes each
product's configured symbol and market into the bot, so `active_income` evaluates
futures candles while `btc_accumulation` evaluates spot candles. The bot compacts
ccxt-style symbols such as `BTC/USDT:USDT` to `BTCUSDT` for Binance public kline
REST calls, while the live ccxt adapter normalizes compact symbols back to
exchange-native ccxt symbols before ticker, position, margin, leverage, and
order calls. Local state and fills keep the configured/order symbol so
reconciliation stays stable. Closed public candle rows are validated before
feature construction: OHLC prices must be finite, positive, and internally
consistent (`high >= open/close >= low`), timestamps must be valid and strictly
increasing, and volume/trade-count fields must be finite and non-negative. When a `Broker` is injected, entries place broker
market orders sized from broker balance and the
strategy risk fraction. Broker-sourced balances, prices, requested quantities,
fill quantities/prices, and fees must be finite before they can affect an order
or local state; balances, prices, and quantities must also be positive, while
fees must be non-negative. The live adapter also validates read-only ticker,
balance, and position responses before preflight, sizing, reconciliation, or
flatten logic can use them. If a broker entry partially fills, local position
state is not opened; the operator must reconcile the broker/account state before
continuing. Broker entry fills above the requested order quantity are also
rejected. Entry fills with a mismatched symbol/side or invalid quantity/price are
rejected before local state opens. Futures exits send reduce-only market orders
for the strategy's stored fill quantity.
Spot BTC step-aside exits reinvest the quote
proceeds from the sell, so a successful sell-lower/buy-lower round trip can
increase BTC holdings instead of leaving gains idle in USDT. Each cycle
reconciles local open state against the broker position before managing exits.
If persisted broker state is missing required broker metadata, contains a
non-positive or non-numeric `broker_qty`, a `broker_qty` that does not match
`broker_requested_qty`, a negative or non-numeric `broker_entry_fee`, or a fill
ratio other than `1`, startup fails before broker reconciliation or exit order
placement. When a broker is attached, paper-style open-position records with no
broker metadata at all are rejected for the same reason.
If a broker exit reports a mismatched, invalid, partial, or overfilled fill, the
bot raises before deleting local open-position state or writing a trade row,
leaving the position visible for operator reconciliation.
When a broker exit is accepted, the trade row records broker exit quantity,
price, and fee; operator reporting validates those audit fields as finite,
positive quantity/price and non-negative fee, and healthcheck fails live products
with corrupted trade-log audit fields. Every trade-log row must also contain
finite `net_return` and `sized_return` values; missing or malformed returns are
reported instead of being treated as zero.
Runtime state enforces the exported risk block: per-trade risk,
`max_position_fraction`, daily stop, per-strategy `max_trades_per_day`,
consecutive-loss cooldown, and drift deactivation. `max_position_fraction`
caps notional sizing after stop-distance sizing, so a tight stop cannot expand
into full-equity exposure unless the artifact explicitly allows it. When an
artifact contains multiple strategies, the account-level daily stop uses the
strictest `daily_stop_loss` across the artifact. The bot also allows only one
open position per product/symbol at a time, so multi-strategy artifacts cannot
stack simultaneous BTCUSDT exposure. Strategy risk blocks are validated and
normalized at load time; malformed, non-finite, non-integer, missing-cap, or
unsafe risk, TP/SL, fee, holdout metric, drift baseline, horizon, condition, or
hypothesis-entry values fail before any cycle can evaluate entries. The one
exception is an explicit paper-only bootstrap artifact generated with
`make bootstrap-strategies`: it may omit holdout/DSR evidence only while
`live_allowed: false` and `promotion_eligible: false`, and approval/live policy
checks reject it. Promotion/live policy requires `paper_trade_allowed`,
`live_allowed`, and `promotion_eligible` to be explicitly `true`; missing
eligibility flags fail closed. New exports stamp `market` and `symbol` at both
the artifact and strategy level. If an
artifact or strategy declares either field,
it must match the bot's configured `--market` and `--symbol`; legacy artifacts
without declared fields are still accepted for compatibility. These basic
executable-artifact checks also run when `src.run_bot` is invoked directly, so
manual paper rehearsals cannot bypass them. Direct `src.run_bot` CLI runs do not
construct a live broker; live broker injection is accepted only from the
autopilot path after approval, preflight, and testnet rehearsal gates have
already passed. Direct manual product runs
should pass `--objective active_income --base-asset USDT` or
`--objective btc_accumulation --base-asset BTC` so product-specific DSR,
holdout, and step-aside-only guards are enabled outside the autopilot wrapper.
When the BTC accumulation regime guard is enabled, a failed daily macro-regime
refresh blocks new entries for that cycle instead of assuming entries are safe.
Persisted bot state is also checked at startup: equity must be finite and
positive, cooldown timestamps and loss/trade counters must be finite and
non-negative, daily trade counters must be integers, and any open-position
record must have a known strategy id, valid timestamp, direction, positive
prices/percentages, bounded position size, and sane broker metadata. Corrupt
state fails the cycle before sizing or order logic can run.

The autopilot wires approved `active_income` live products to a futures ccxt
broker and approved `btc_accumulation` live products to a spot ccxt broker.
For spot BTC accumulation, logical sell/step-aside orders are sized from the
existing base-asset balance rather than quote balance, so the product cannot
sell more BTC than the account holds. The ccxt spot adapter repeats that check
immediately before order submission. The runtime records BTC before/after the sell
and the quote budget available for buyback. BTC accumulation never routes through
the futures broker, so it cannot use leverage.

Emergency control is file-based. Adding a live futures product to
`flatten_products` in `runtime/control.json` asks the runtime to close the broker
position with a reduce-only market order and clear local open-position state
after the broker reports flat. If the state file is corrupt at that point, the
runtime writes a minimal recovered state with a `last_flatten` audit marker.
Spot BTC accumulation is skipped by this command.

Before a live/testnet run, use the preflight. It does not place orders:

```bash
python -m src.autopilot.preflight \
  --config config/autopilot.json \
  --product active_income \
  --assume-live \
  --connect \
  --require-testnet \
  --output runtime/active_income_preflight_report.json
```

The check validates strategy approval, env safety switches, broker construction,
and read-only ticker/balance/position access. By default, live runtime also
checks that the saved report is successful, has a finite timestamp that is not
stale or far in the future, and is tied to the exact strategy artifact and
strategy fingerprints about to trade. It also requires the saved report to include
successful check entries for config, strategy artifact, policy, approval,
exchange environment, broker construction, read-only connectivity, and the
product-specific position check. The live runtime also enforces
`TRADING_LIVE=1`, exchange credentials, a positive
`MAX_NOTIONAL_USD`, a positive `MAX_FILL_SLIPPAGE_BPS`,
`FUTURES_MARGIN_MODE=isolated`, and `MAX_FUTURES_LEVERAGE=1` for the
active-income futures product before broker construction, so disabling preflight
does not skip the basic safety checks. If the preflight command itself cannot
build or write the report, it exits nonzero with structured JSON describing the
failed gate.

For the active-income testnet rehearsal, prefer the Makefile wrapper:

```bash
make preflight PRODUCT=active_income REQUIRE_TESTNET=1
```

The wrapper always performs read-only connectivity checks. `REQUIRE_TESTNET=1`
makes the preflight refuse `EXCHANGE_TESTNET=0`, preventing a mainnet-connected
preflight report from being mistaken for the sandbox rehearsal setup.

To exercise the actual Binance USDT-M testnet order path after approval and a
green connected preflight:

```bash
make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5
make testnet-status
```

The report is written to `runtime/testnet_rehearsal_report.json`. Do not run it
against mainnet; the command requires `EXCHANGE_TESTNET=1` and exits otherwise.
`make testnet-status` reads the saved report without placing orders, and
`make report` surfaces the latest rehearsal as missing, failed, stale, or ok.
A usable report must have a finite timestamp, positive notional, positive order
quantity, product-symbol-matched buy entry fill, product-symbol-matched sell
close fill, entry/close quantities matching the order quantity, testnet routing,
and a flat final position. The default active-income product config requires
that report to be recent and
successful before live execution can construct a broker. CLI failures, including
config-load and output-write failures, are emitted as structured JSON. If the
entry fills but close/readback fails, the command attempts one best-effort cleanup
close and records the recovery result, while still treating the rehearsal as
failed until it can be rerun cleanly.
