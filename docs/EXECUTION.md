# Execution Layer

A broker abstraction (`src/execution`) so the algo-trading system can trade spot
or futures venues with one interface, paper by default and live behind explicit
safety switches.

## Brokers

| Broker | Use | Notes |
|---|---|---|
| `PaperBroker` | development, tests, paper autopilot cycles | Simulated market fills with fees + slippage; injectable price source; tracks signed positions, realised/unrealised PnL and equity. Dependency-free. |
| `CcxtBroker` | live / testnet | Wraps the pinned `ccxt==4.5.64` from `requirements-bot.txt`; Binance native conditional-stop routing is version-sensitive. |

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
* the environment parser accepts futures leverage in the conservative range
  1–3, but `active_income` policy requires `MAX_FUTURES_LEVERAGE=1`; the broker
  must successfully set it before an entry
* Binance USD-M must report one-way position mode (`positionSide=BOTH`); hedge
  mode is refused
* immediately before a non-reduce futures `create_order`, the broker performs
  unfiltered USD-M account reads and requires every position plus every regular
  and conditional order across all symbols to be empty; concurrent or manual
  exposure cannot be hidden outside the configured BTC symbol

Otherwise `place_order` raises, so a misconfigured run cannot trade real size.
Set `EXCHANGE_TESTNET=1` only for the exchange sandbox rehearsal. Production
runtime and production preflight require `EXCHANGE_TESTNET=0`.
`src.autopilot.preflight` is read-only; the explicit active-income testnet order
rehearsal is `make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100`. It requires the
same approval/preflight/env gates, refuses non-testnet venues, starts only from a
flat futures position, places one tiny BTCUSDT futures market entry, and closes
it immediately through the broker close path.

```bash
FUTURES_EXCHANGE=binanceusdm
SPOT_EXCHANGE=binance
EXCHANGE_MARKET_TYPE=futures
EXCHANGE_TESTNET=1
TRADING_LIVE=0          # flip to 1 only when you mean it
MAX_NOTIONAL_USD=250
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

Feature construction can take long enough for an operator action or approval to
change. Therefore the autopilot live-entry path reloads and rechecks control
(panic/pause/flatten), approval, saved preflight, and current environment after
signal evaluation and immediately before the durable entry intent. A stale
earlier gate can never authorize the eventual order.
Spot BTC step-aside exits reinvest the quote
proceeds from the sell, so a successful sell-lower/buy-lower round trip can
increase BTC holdings instead of leaving gains idle in USDT. Each cycle
reconciles local open state against the broker position before managing exits.
For a live spot sell, the bot does not infer that buyback budget from the
fill's numeric fee because the fill does not identify whether commission was
paid in USDT, BTC, or BNB. It reads free USDT immediately before and after the
sell, requires that positive delta to remain within 1% of the filled quote
notional (and never above it beyond numeric tolerance), and persists the two
balances, their observed delta, and the source marker. The later buyback uses
that observed delta exactly. A missing, non-positive, unexpectedly small, or
inflated delta leaves the durable entry intent unresolved and blocks trading
for reconciliation. Use a dedicated spot account/subaccount: transfers or
manual trades during this window deliberately trip this protection.
Every live position and recovery/flatten intent also persists the broker's
non-secret account fingerprint. Normal management, recovery, and emergency
flatten compare it with the currently configured broker before placing an order;
an API-key/venue/routing change cannot accidentally close a different account.
If credentials must rotate while exposure exists, keep the product paused and
reconcile/restore the intended account binding before management resumes.
If persisted broker state is missing required broker metadata, contains a
non-positive or non-numeric `broker_qty`, a `broker_qty` that does not match
`broker_requested_qty`, a negative or non-numeric `broker_entry_fee`, or a fill
ratio other than `1`, startup fails before broker reconciliation or exit order
placement. When a broker is attached, paper-style open-position records with no
broker metadata at all are rejected for the same reason.
For every live futures entry, the market-order intent is persisted first. After
an accepted full fill, the bot places a native reduce-only stop-market order,
fetches it back, and persists its order ID, client ID, trigger, quantity, and
status with the position before clearing the entry intent. A process/network
failure therefore leaves durable restart evidence; it never turns an
unprotected fill into an ordinary open local position.
Live futures reconciliation requires the broker's signed quantity to equal the
persisted signed quantity within fill tolerance. Extra same-direction contracts,
opposite exposure, or a partial external close means the exchange-native stop no
longer covers the actual account. The bot then persists a deterministic
reduce-only recovery intent, closes the full current broker quantity, proves the
account flat, cancels any still-open tracked stop, and latches
`risk_recovery_incident`. A tracked native stop that is unexpectedly canceled,
expired, rejected, or only partially fills while residual exposure remains uses
the same full-actual-quantity recovery close. The
incident remains blocking after a successful close; reconcile its fill,
accounting, stop state, and cause before removing it during reviewed maintenance.
Runtime status and the operator report expose the native-stop fields plus
`pending_order`, `pending_entry_recovery`, `risk_recovery_incident`,
`flatten_intent`, and `exit_accounting_intent`; healthcheck treats any of these
durable states on a live product as blocking.
When the stop lookup is malformed or only a partial fill can be inferred, the
residual recovery fill is not booked as though it closed the original position.
The bot proves the residual exposure flat but retains local position/accounting
state and the sticky incident so the stop and recovery fills can be reconciled
together before a trade row is committed.

Every live pending order, open position, and recovery marker stores a non-secret
`broker_account_fingerprint` bound to exchange, market, testnet flag, and API-key
identity (never the secret). Startup, reconciliation, ordinary exits, and
emergency flatten compare it with the current broker before any account read or
order. Credential rotation to another account, missing legacy identity, or
mixed recovery identities fail closed without touching that account.

Live futures entry intents also survive an ambiguous order response. If the
entry was accepted but no protected local position was committed, the bot checks
the actual broker position immediately and again on restart, persists
`pending_entry_recovery`, and submits one deterministic reduce-only close for the
exact actual quantity. A broker-flat readback never clears the original
`pending_order`; it only proves risk reduction, then blocks for operator review.
Missing, stale, changed, or revoked strategy/approval/preflight evidence cannot
block this management-only close, and can never authorize another entry. Spot
orders and pending exits are excluded from this automatic pending-entry path.
If a broker exit reports a mismatched, invalid, partial, or overfilled fill, the
bot raises before deleting local open-position state or writing a trade row,
leaving the position visible for operator reconciliation.
When a broker exit is accepted, the trade row records broker exit quantity,
price, and fee; operator reporting validates those audit fields as finite,
positive quantity/price and non-negative fee, and healthcheck fails live products
with corrupted trade-log audit fields. Every trade-log row must also contain
finite `net_return` and `sized_return` values; missing or malformed returns are
reported instead of being treated as zero.
For live futures, each newly opened position also persists the flat quote-balance
baseline read immediately before entry. After an ordinary or native-stop exit
has been proved flat, the bot reads the balance again and compares the observed
account return with fill-price/commission accounting. It books the worse result
into local equity, `daily_pnl`, the consecutive-loss cooldown, and the trade log.
This downside-only reconciliation captures funding and broker-booked debits,
while a deposit or other positive credit cannot improve the modeled result.
Trade rows expose `broker_entry_balance`, `broker_exit_balance`,
`broker_balance_return`, `accounting_adjustment_fraction`, and
`accounting_return_source`; operator reporting rejects malformed balance
evidence. Missing or invalid baseline/readback data fails before local close
accounting is finalized. Across multiple closes in one day, the tracker keeps
the worse of additive and compounded cumulative returns, so a gain followed by
an equal percentage loss cannot be misreported as flat and consecutive losses
do not receive a favorable compounding offset. The daily window resets on the
UTC calendar date, independent of the server's local timezone.

The broker-flat exit and its local accounting commit are crash-consistent. The
bot first writes `exit_accounting_intent` with a deterministic exit event ID,
integrity digests, the exact trade row, and pre/post state. It then atomically
inserts or verifies that keyed CSV row and commits the target state. A restart
revalidates broker flat and resumes these steps without placing another exit;
the event ID prevents duplicate trade rows. Until commit completes, runtime
status/operator reporting show the intent and healthcheck blocks a live product.
Do not delete the WAL or hand-edit a replacement trade row.

Use a dedicated futures account or subaccount and do not deposit, withdraw, or
manually trade in it while the bot has exposure. A positive transfer can mask a
simultaneous funding debit even though it cannot turn a modeled trading loss into
a gain. Live fill records carry a numeric fee but not a fee currency, so this
accounting assumes futures commissions are charged in the USDT settlement asset.
Do not enable payment from an alternative fee asset; if the account does use one,
its balance and USDT value require separate monitoring and the modeled fee floor
is only an estimate. The daily stop is realized at the proved-flat exit boundary;
it does not continuously mark unrealized PnL or intraday funding while a position
remains open. The one-position limit, approved stop distance, and exchange-native
stop bound that interim exposure.
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
the artifact and strategy level. If an artifact or strategy declares either field,
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

### Persistent peak-equity circuit breaker

Daily limits reset on UTC boundaries, so they cannot by themselves prevent a
strategy from losing a small amount repeatedly across many days. Every product
therefore also maintains an account-level, peak-to-current equity circuit
breaker in its atomic bot state:

- active-income futures halt new entries at a 10% drawdown from peak local USDT
  equity;
- BTC accumulation halts new entries at a 5% drawdown from peak local BTC
  equity;
- direct runs that omit `--objective` receive the safer 5% limit.

The BTC envelope is tighter because that product's purpose is conservative base
asset preservation. The active-income envelope is wider than its 3% default
daily stop so ordinary daily protection fires first, while the 10% ceiling
still bounds cumulative damage across UTC resets. `peak_equity` only moves
upward. At or beyond the fixed limit the bot atomically latches
`drawdown_halted`, `drawdown_halted_at`, and `drawdown_halt_reason`, continues to
manage and close existing exposure, and refuses every new entry. A new UTC day,
later profit, restart, artifact change, or strategy approval does not clear the
halt. Operator reports show current/peak equity and drawdown, and healthcheck
treats a live halt as blocking (paper halt as a warning).

Old valid state is migrated without losing safety fields: its initial peak is
the greater of current local equity and configured starting equity, and an
already-breached envelope latches immediately. Do not lower `peak_equity` to
make the percentage look smaller.

Recovery is deliberately a reviewed maintenance action, not an automatic or
daily reset. Keep the product paused; reconcile exchange balances, funding,
fills, the local trade ledger, pending-order state, and the cause of loss; make
sure all exposure is either deliberately managed or flat; back up the state;
then review the proposed recovery with the owner. Only after current local
equity is genuinely inside the unchanged peak/limit envelope may the stopped
service's state be atomically changed to `drawdown_halted: false` with
`drawdown_halted_at` and `drawdown_halt_reason` set to `null`. If the reconciled
drawdown is still at the limit, startup immediately relatches. There is no
one-command auto-clear path.

The autopilot wires approved `active_income` live products to a futures ccxt
broker and approved `btc_accumulation` live products to a spot ccxt broker.
For spot BTC accumulation, logical sell/step-aside orders are sized from the
existing base-asset balance rather than quote balance, so the product cannot
sell more BTC than the account holds. The ccxt spot adapter repeats that check
immediately before order submission. The runtime records BTC before/after the sell
and the observed free-USDT delta available for buyback. BTC accumulation never routes through
the futures broker, so it cannot use leverage.

Emergency control is file-based. Adding a live futures product to
`flatten_products` in `runtime/operator-control/control.json` asks the runtime to close the broker
position with a deterministic-client-ID reduce-only market order. Before
submission it writes `flatten_intent`; after the fill it durably records the
fill, before/after broker position, before/after quote balance, and realized
account delta. It clears local open-position state only after the broker reports
flat, every tracked native stop is proved terminal, and the keyed trade/equity/
daily-PnL/cooldown accounting transition commits exactly once. A BTC-accumulation flatten buys back exactly one fully tracked spot
step-aside position; it never sells the base BTC stack. Its normalized buyback
is protected by a durable `flatten_intent`, so an ambiguous restart cannot send
the same buy twice. Invalid or insufficient state fails closed.

An emergency runtime flatten is an operator recovery path, not a normal bot exit.
The runtime atomically clears a successful flatten request while leaving the
affected product (or all products for `flatten_all`) paused. A crash after the
accounting commit but before that control update is idempotent: the next pass
proves the same account is flat account-wide, verifies the keyed trade row and
`last_flatten` evidence, sends no order, and clears the stale request. Any
ambiguous submit, missing fill/fee evidence, malformed inventory, or incomplete
accounting keeps the request and product paused for reconciliation.

Preflight never places orders. Sandbox and production evidence are distinct.
For a testnet rehearsal use:

```bash
python -m src.autopilot.preflight \
  --config config/autopilot.json \
  --product active_income \
  --assume-live \
  --connect \
  --require-testnet \
  --output runtime/active_income_preflight_report.json
```

For the saved production gate, switch to production credentials and
`EXCHANGE_TESTNET=0`, then run `make preflight PRODUCT=active_income` (or
`PRODUCT=btc_accumulation`). Production preflight refuses testnet routing.

The read-only check validates artifact/product identity, the execution-engine
digest (Python, pinned installed versions, and source), environment, broker
construction, and read-only ticker/balance/position access. Futures also proves
one-way position mode, native-stop capability, empty regular/conditional order
inventories, and a flat position. Live runtime requires a fresh saved report
bound to the exact artifact digest/fingerprints, engine, and a non-secret account
fingerprint derived from API key plus venue/market/testnet routing. Current
exchange, market, account, testnet flag, quote asset, notional/slippage caps,
leverage, and margin mode must exactly match the production report; any change
requires new connected evidence. Final human approval pins that stable
account/venue/risk-cap manifest plus every canonical product field/path. A fresh
equivalent preflight renews the runtime age gate without requiring reapproval;
manifest drift does require it. Product settings and execution-capable
code/dependency changes also invalidate prior evidence.

The live runtime independently enforces
`TRADING_LIVE=1`, exchange credentials, a positive
`MAX_NOTIONAL_USD`, a positive `MAX_FILL_SLIPPAGE_BPS`,
`FUTURES_MARGIN_MODE=isolated`, and `MAX_FUTURES_LEVERAGE=1` for the
active-income futures product before broker construction. If preflight cannot
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
make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100
make testnet-status
```

The report is written to `runtime/testnet_rehearsal_report.json`. Do not run it
against mainnet; the command requires `EXCHANGE_TESTNET=1` and exits otherwise.
`make testnet-status` reads the saved report without placing orders, and
`make report` surfaces the latest rehearsal as missing, failed, stale, or ok.
A usable report must have a finite timestamp, positive notional, positive order
quantity, matched entry/close fills, a verified native stop
place/read/cancel/terminal lifecycle, testnet routing, empty preflight order
inventories, and a flat final position. The default active-income product config requires
that report to be recent and
successful before live execution can construct a broker. CLI failures, including
config-load and output-write failures, are emitted as structured JSON. If the
entry fills but close/readback fails, the command attempts one best-effort cleanup
close and records the recovery result, while still treating the rehearsal as
failed until it can be rerun cleanly.

After a passing rehearsal, run the fresh `EXCHANGE_TESTNET=0` production
preflight described above. A testnet report never substitutes for production
account/environment evidence.
