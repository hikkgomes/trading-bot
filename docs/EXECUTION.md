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

The legacy `src.autopilot` and `src/run_bot.py` paths are retained only as
offline migration or research libraries. They are not used by the production
execution path.
