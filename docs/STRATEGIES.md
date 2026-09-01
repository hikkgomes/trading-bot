# Strategy catalogue and behaviour contract

The platform accepts named registered strategies, typed generated DSL,
machine-learning artefacts, semantic research families, and bounded agent
proposals. Every source is compiled into a content-addressed strategy
definition and evaluated through the same PostgreSQL research queue.

## Shared execution contract

The strategy definition binds:

- product and point-in-time instrument universe;
- feature graph and data requirements;
- signal and position logic, including exits;
- parameters and cost assumptions;
- risk and evidence policy;
- executable source identity and engine version.

The sealed behaviour is reused in screening, development, robustness,
protected holdout, forward paper, paper execution, connected testnet, and live
execution. Parity receipts compare signals, positions, and order intents over
the same canonical input frame.

## Research families

The configured campaigns cover:

- BTC tactical accumulation;
- directional futures trend, breakout, momentum, and mean reversion;
- cross-sectional ranking;
- funding carry and spot-perpetual basis;
- pairs and other relative value;
- event and order-book microstructure;
- ensembles and frozen machine-learning models.

Family labels select applicable evidence. A family is not considered tested
because it appears in the manifest. It must have an executable candidate,
appropriate data, and evidence in the PostgreSQL funnel.

## Product constraints

`btc_accumulation` is restricted to Binance BTCUSDT spot. Its primary metric is
BTC NAV excess versus passive BTC holding after fees and re-entry costs. It
cannot use leverage, futures, short borrowing, or other symbols.

`active_income` uses Binance USDT-margined futures. Its metrics include signed
funding, net USDT PnL, effective decisions, drawdown, capacity, margin,
liquidation buffer, and portfolio exposure.

## Adding a strategy

1. Add the executable strategy or semantic provider under the allowed source roots.
2. Register its feature contract and manifest entry.
3. Bind it to a typed thesis, product, family, and point-in-time universe.
4. Add parity, accounting, and failure-case tests.
5. Submit it through the PostgreSQL research queue.

Do not add a hand-written production proxy for a strategy. Do not put metrics,
approval flags, or raw data in a queue command. A candidate must pass the
normal dataset, evidence, protected holdout, and forward-paper lifecycle.

Use [`docs/RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) for the funnel and
[`docs/EXECUTION.md`](EXECUTION.md) for runtime behaviour. Legacy strategy
modules remain available for offline migration or research experiments only.
