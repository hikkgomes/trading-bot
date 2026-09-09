# Execution and safety

The PostgreSQL platform is the only execution authority. `execution-engine`
can submit orders. Runtime state, orders, fills, positions, controls, risk,
accounting, and recovery are durable database records.

## Closed-candle path

```text
market event
  -> immutable Parquet and market snapshot
  -> causal feature snapshot
  -> shared strategy behaviour
  -> alpha forecast
  -> product target
  -> six-scope risk decision
  -> durable order intent
  -> paper or authorised live venue
  -> user stream and REST reconciliation
  -> fill, position, accounting, attribution, trace
```

The same sealed behaviour and feature contract run in research, paper,
connected testnet, and live execution. The artefact identity includes the
executable behaviour, parameters, feature graph, position and exit logic,
cost assumptions, and engine version. Deployment provenance is recorded
separately.

## Live admission

Before an entry, the platform requires all of the following:

- product, account, instrument, sleeve, artefact, approval, and preflight identity match;
- current authenticated account snapshot and account fingerprint;
- current portfolio, market, and risk snapshots with no unknown exposure;
- current control state permits new risk;
- exposure and pending-order budgets remain within assignment and product caps;
- exchange precision, balance, margin, leverage, position mode, and notional checks pass;
- the order is persisted before the exchange side effect.

BTC accumulation is Binance BTCUSDT spot only. It is non-leveraged, cannot
short, and may use only its bounded tactical BTC step-aside sleeve. Active
income is Binance USDT-margined futures with explicit configured symbols,
isolated margin, bounded leverage, signed funding, and product risk limits.

## Protective stops

For a futures entry the required lifecycle is:

```text
entry fill
  -> native reduce-only protective stop intent
  -> exchange stop confirmation
  -> durable protected position
```

Partial fills resize the stop. Restart and account refresh reconcile the stop
against actual exchange exposure. `ALGO_UPDATE` and ordinary order updates are
both processed. A missing, cancelled, expired, rejected, or undersized stop
creates a recovery incident and blocks further exposure.

Stop quantities cover the position once across its entries. A scale-in receives
only the uncovered quantity. Replacement persists a new client ID and retains
the previous exchange identity. It confirms the new reduce-only stop, reads
the exchange position again, and then cancels the previous stop. Both identities
remain available until cancellation is confirmed. A changed position or trigger
during replacement requires reconciliation and blocks new entries. Temporary
overlap is restricted to reduce-only protection during that replacement.

An attempted submission is durable before the API call. After an ambiguous
response, recovery queries that client ID and never blindly repeats placement.
Failure to confirm protection immediately queues risk reduction; there is no
grace period in which another entry may add risk.

`ALGO_UPDATE` identifies the conditional order separately from its matching-engine
child. A trigger does not prove a fill. The child is read back by its exact
exchange ID and adopted as a reduce-only order with the original strategy and
account attribution. User-stream trades and REST recovery then use the same
financial transaction as ordinary fills. Partial children stay active; duplicate
trade IDs have one economic effect. Unknown child identity remains a recovery
dependency, and a terminal child with uncovered residual exposure queues reduction.

REST income recovery compares commissions and realised PnL with their exact
account, instrument and trade records. These comparisons create audit receipts,
not second cash postings. Signed funding is recorded once by scoped income ID.
Changed amounts, unmatched trades and unclassified income types block recovery.
Previously recorded unscoped REST income needs explicit reconciliation.

Income history advances in bounded daily windows. A durable ledger checkpoint is
written only after every returned record reconciles. Restart resumes that window
with a one-minute overlap; replayed records cannot change PnL. Pending windows
remain deferred work rather than a completed account-history check.

Each order also records a REST trade checkpoint after its returned quantities
match the exchange order and all financial effects are committed. A restart
uses that checkpoint and the existing fills together; websocket arrival order
alone cannot advance the REST checkpoint.

Emergency reconciliation never replaces financial positions with a balance
snapshot. It recovers the actual closing fills and verifies the resulting
position and PnL. Connected rehearsal evidence uses real fills and received
algo events; it cannot claim a partial fill by passing a smaller invented
position to the protection service.

## Controls and recovery

The database control plane supports run, block-new-risk, management-only,
suspended, and emergency-flatten modes. Pausing new risk does not disable
reconciliation or risk-reducing actions. Emergency reduction and flatten
commands are idempotent and use durable action identities.

Ambiguous submissions, unknown orders, missed fills, fee conversions, and
stream gaps enter recovery. REST reconciliation backfills order, fill,
commission, funding, and stop state. Recovery cannot repeat a known exchange
side effect and cannot clear an unresolved mismatch by assumption.

## Verification

```bash
make platform-permissions-test
make platform-smoke
make platform-testnet-rehearsal
make platform-testnet-connected PRODUCT=active_income NOTIONAL_USD=10 CONFIRM=1
```

The connected command requires a dedicated flat testnet account and explicit
confirmation. It must never receive production credentials. A testnet report
does not approve live capital.

The obsolete file-based trading controller and search entry points have been
removed. `src.research.legacy_codec` is a frozen, read-only decoder for migration
archives; it cannot schedule research or authorise execution.
