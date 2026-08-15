# Complete Target Specification

This replaces the previous staged answer.

## 1. Repository classification

The repository is already an **autonomous quantitative research platform with a limited execution engine**.

It already defines the two required products:

| Product            | Objective                | Market               | Accounting unit |
| ------------------ | ------------------------ | -------------------- | --------------- |
| `btc_accumulation` | Increase BTC holdings    | Binance spot         | BTC             |
| `active_income`    | Generate trading returns | Binance USDT futures | USDT            |

Both products are currently configured for paper execution.

The repository is not limited to momentum. It has 24 registered strategies across:

* Trend

* Momentum

* Breakout

* Mean reversion

* Volatility

* Multi-timeframe trading

* Candlestick and market structure

* Sentiment

* Regime filters

* BTC cycle models

* Machine-learning classifiers

* Machine-learning regressors

* Generated condition grids

It also has separate research systems for:

* Generative strategy grammar

* Mutation and crossover

* Relative value

* Statistical pairs

* Spot-perpetual basis

* Cross-sectional trading

* Market microstructure

* Order-flow signals

* Gradient-boosting models

* Portfolio correlation and beta controls

The principal limitation is execution. The current live path is still centred on:

* One configured symbol per product
* BTCUSDT for active income
* One open position per product and symbol
* A futures account that must be fully flat before a new entry
* No partial-fill workflow
* No production multi-leg execution
* No portfolio target-position engine

Relative-value and multi-leg strategies are therefore research-only.

The final product must turn the existing research platform into a multi-symbol portfolio and execution platform.

---

# 2. Why the current system has no trades

There are four structural causes in the repository.

## 2.1 Bootstrap strategies cannot enter trades

The bootstrap artefacts contain:

* `entry_policy: management_only`
* `executable: false`
* `paper_trade_allowed: false`
* `live_allowed: false`
* `promotion_eligible: false`

They exist to validate artefact loading and manage old positions. They cannot test a complete entry and exit cycle.

If these are still the active artefacts, zero trades are the intended result.

## 2.2 The autonomous cycle does not use the full strategy catalogue

The named strategy library is available for manual backtests and sweeps. The production research cycle runs with `--generated-only`.

This means that the autonomous pipeline mainly evaluates generated predicate strategies. It does not continuously evaluate the complete library of SMA, Donchian, RSI, Bollinger, Supertrend, and other registered strategies.

The strategy catalogue and the autonomous research factory are effectively separate systems.

## 2.3 Generated strategies are frequently too restrictive

Each generated strategy has:

* A regime stage
* A setup stage
* A trigger stage
* At least one predicate in each stage
* Up to seven total predicates

A generated entry can therefore require:

```text
regime condition
AND setup condition
AND trigger condition
AND optional extra conditions
```

This creates sparse signals. Sparse signals then fail the minimum trade-count gates.

## 2.4 Every candidate enters an expensive validation process

The current validation includes:

* Positive training edge

* Positive validation edge

* Chronological window stability

* Parameter sensitivity

* Higher transaction costs

* Delayed entries

* Adverse fills

* Funding charges

* Missing-bar stress

* Final protected holdout

* Deflated Sharpe requirements

These are suitable promotion gates. They are inefficient as the first complete test for every generated candidate.

## 2.5 Scheduled work competes through one bounded worker

The configuration has many data, ML, research, portfolio, microstructure, mutation, reporting, and maintenance jobs. It permits one scheduled job to start per worker cycle.

This causes research latency and job deferrals when history or ML tasks take a long time.

## Required final fix

The final system must record a full decision trace for every processed market event:

```text
data available
feature available
strategy evaluated
regime passed
setup passed
trigger passed
signal produced
portfolio accepted
risk accepted
order planned
order submitted
order filled
position opened
position closed
```

Each failed step must have a machine-readable reason.

There must also be an isolated **execution diagnostic strategy** that:

* Is permitted to trade in paper mode
* Is never eligible for live promotion
* Produces a deterministic signal
* Opens and closes a small synthetic paper position
* Tests the complete execution, accounting, and logging path

This separates an execution fault from a lack of trading edge.

---

# 3. Final system architecture

The platform is one distributed system with two compute nodes.

## Linux OptiPlex

The Linux machine is the production authority.

It runs:

| Service              | Responsibility                                           |
| -------------------- | -------------------------------------------------------- |
| `market-gateway`     | Binance REST, WebSocket, user stream, clocks, reconnects |
| `data-writer`        | Raw event and bar persistence                            |
| `feature-service`    | Incremental live features                                |
| `portfolio-engine`   | Alpha aggregation and target positions                   |
| `risk-engine`        | Product, symbol, portfolio, and account controls         |
| `execution-engine`   | Order planning, submission, recovery, and reconciliation |
| `paper-engine`       | Production-equivalent simulated execution                |
| `product-supervisor` | BTC accumulation and active-income processes             |
| `accounting-service` | Balances, NAV, funding, fees, and PnL attribution        |
| `promotion-engine`   | Strategy lifecycle and automatic policy decisions        |
| `control-api`        | Status, pause, configuration, and reports                |
| PostgreSQL           | Operational state and research metadata                  |
| Parquet store        | Market data, features, events, and immutable snapshots   |

Only the Linux machine can submit exchange orders.

## 2017 MacBook Pro

The Mac is a dedicated research node.

It runs:

| Service                | Responsibility                                 |
| ---------------------- | ---------------------------------------------- |
| `research-worker`      | Rule, cross-sectional, and portfolio backtests |
| `ml-worker`            | Scikit-learn and LightGBM training             |
| `event-replay-worker`  | Order-book and trade replay                    |
| `agent-sandbox`        | OpenClaw code generation and tests             |
| `feature-build-worker` | Historical feature calculation                 |
| `report-worker`        | Research and attribution reports               |

It claims research jobs from PostgreSQL. It returns immutable result artefacts and model artefacts.

## Communication

The two machines use:

* Tailscale or WireGuard networking
* PostgreSQL as the shared control plane
* A PostgreSQL job queue with worker leases
* SSH or `rsync` for Parquet and model artefacts
* Content hashes for every transferred file
* Git for source code
* Separate operating-system service accounts

Execution and position management do not depend on the Mac.

---

# 4. Canonical domain model

The current repository has several partially overlapping strategy and artefact formats. The final platform uses one canonical model.

## 4.1 Instrument

```text
Instrument
  venue
  market_type
  base_asset
  quote_asset
  settlement_asset
  exchange_symbol
  price_precision
  quantity_precision
  minimum_quantity
  minimum_notional
  contract_size
  status
```

This removes the BTCUSDT hard-coding.

## 4.2 Market event

```text
MarketEvent
  instrument_id
  event_type
  exchange_timestamp
  receive_timestamp
  sequence
  payload
```

Supported event types:

* Candle
* Trade
* Aggregate trade
* Best bid and ask
* Depth update
* Liquidation
* Mark price
* Funding rate
* Open interest
* Account balance
* Order update
* Fill update

## 4.3 Alpha forecast

Every strategy returns the same contract:

```text
AlphaForecast
  strategy_version_id
  product_id
  instrument_id
  direction
  score
  expected_return
  confidence
  horizon
  valid_from
  valid_until
  target_volatility
  maximum_position
  metadata
```

The existing `AlphaForecast` and portfolio structures form the base for this contract.

## 4.4 Target position

Strategies do not place orders.

They produce forecasts. The portfolio engine produces:

```text
TargetPosition
  portfolio_id
  instrument_id
  target_quantity
  target_notional
  target_fraction
  strategy_contributions
  risk_budget
  valid_until
```

The execution engine converts the difference between the current position and the target position into orders.

## 4.5 Strategy definition

```text
StrategyDefinition
  identity
  version
  family
  product
  universe
  data_requirements
  feature_graph
  signal_model
  position_model
  execution_preferences
  risk_policy
  validation_policy
  source_type
  source_hash
```

`source_type` can be:

* Registered Python strategy
* Generated DSL strategy
* Machine-learning model
* Cross-sectional strategy
* Relative-value strategy
* Microstructure strategy
* Ensemble strategy
* Agent-generated Python strategy

## 4.6 Strategy artefact

A deployable artefact contains:

* Frozen strategy definition
* Exact source hash
* Exact dependency hash
* Dataset snapshot hashes
* Feature set version
* Cost model version
* Validation evidence
* Holdout claim
* Forward-paper evidence
* Promotion policy
* Position and risk limits
* Model parameters
* Model file hashes
* Supported products and instruments

The active artefact is immutable. A new version creates a new artefact.

---

# 5. Product A: BTC accumulation

The current short-signal convention is replaced by a BTC allocation model.

## Objective

Maximise BTC holdings relative to a passive BTC benchmark.

The product reports:

* BTC balance
* BTC NAV
* BTC gained or lost against hold
* Time outside BTC
* Stablecoin exposure
* Missed BTC appreciation
* BTC-denominated drawdown
* Fees paid in BTC terms
* Performance by market regime

## Portfolio model

The product controls a configurable tactical share of BTC.

```text
BTC portfolio
  core BTC allocation
  tactical BTC allocation
  stablecoin allocation
```

The strategy output is:

```text
target_btc_fraction between 0 and 1
```

It is not a synthetic short signal.

A value of:

* `1.0` means fully allocated to BTC
* `0.7` means 70% BTC and 30% stablecoin
* `0.0` means fully in stablecoin

## Supported strategy families

* Long-term trend allocation
* Moving-average regime allocation
* Donchian risk-off allocation
* Volatility targeting
* Drawdown-based reduction
* Cycle overheat reduction
* Momentum regime
* Mean-reversion re-entry
* Staged sell and staged re-entry
* Dynamic DCA
* Stablecoin liquidity selection
* Ensemble allocation
* Regime-conditioned allocation
* Machine-learning allocation
* On-chain or sentiment overlays through optional data adapters

## Execution

The BTC product supports:

* BTC against eligible stablecoins
* Market and limit orders
* Staged execution
* Exact quote-proceeds tracking
* Stablecoin balance reconciliation
* Partial fills
* Fee-currency accounting
* Re-entry from the actual realised stablecoin proceeds
* Multiple strategy sleeves merged into one BTC target

## Accounting

BTC is the primary accounting unit.

USDT or another stablecoin value is secondary reporting only.

---

# 6. Product B: Active income

The active-income product becomes a multi-symbol futures portfolio.

## Universe

The complete Binance futures instrument list enters the universe service.

The universe service stores point-in-time snapshots and filters instruments by:

* Listing status
* Listing age
* Quote volume
* Trade count
* Spread
* Open interest
* Funding
* Realised volatility
* Depth
* Data completeness

The repository already screens all trading USDT perpetuals and retains up to 25 symbols.

The final system removes the fixed maximum as a structural restriction. Each research and execution strategy receives a dynamic universe based on its own liquidity and data requirements.

## Portfolio sleeves

The active-income portfolio contains independent sleeves:

| Sleeve           | Strategies                                            |
| ---------------- | ----------------------------------------------------- |
| Directional      | Trend, momentum, breakout, volatility expansion       |
| Mean reversion   | RSI, Bollinger, z-score, short-term reversal          |
| Cross-sectional  | Relative momentum, reversal, residual momentum        |
| Relative value   | Pairs, baskets, PCA residuals, cointegration          |
| Carry            | Funding selection, basis, spot-perpetual carry        |
| Microstructure   | Order-book imbalance, microprice, aggressor flow      |
| Event-driven     | Liquidation flow, spread shocks, volatility shocks    |
| Machine learning | Classification, regression, ranking, meta-labelling   |
| Ensemble         | Forecast aggregation, model weighting, regime routing |

Each sleeve has:

* Capital budget
* Risk budget
* Instrument universe
* Maximum turnover
* Maximum holding period
* Cost model
* Allocation policy

## Portfolio output

The portfolio engine calculates simultaneous target positions across symbols.

It applies:

* Signal confidence
* Expected return
* Volatility
* Correlation
* BTC beta
* Liquidity
* Funding cost
* Existing positions
* Gross exposure
* Net exposure
* Cluster exposure
* Drawdown
* Margin use

The current `src.autopilot.portfolio` module already contains forecast aggregation, position caps, correlation controls, and BTC-beta controls. It becomes the central portfolio optimiser rather than an entry gate.

---

# 7. Complete strategy universe

## 7.1 Time-series strategies

* Moving-average trend
* MACD trend
* Supertrend
* ADX trend
* Time-series momentum
* Rate-of-change momentum
* Donchian breakout
* Keltner breakout
* ATR breakout
* Bollinger squeeze
* Volatility breakout
* Range breakout
* Channel trading
* Regression channels
* Market-structure breakout
* Trend pullback
* Volatility-scaled trend

## 7.2 Mean-reversion strategies

* RSI reversion
* Bollinger reversion
* Z-score reversion
* Stochastic reversion
* VWAP reversion
* Residual reversion
* Overnight or session reversion
* Volatility-conditioned reversion
* Order-flow exhaustion
* Post-liquidation reversion

## 7.3 Cross-sectional strategies

* Relative momentum
* Relative reversal
* Volatility-adjusted ranking
* Liquidity-adjusted ranking
* Funding-adjusted ranking
* Residual momentum after BTC and market beta removal
* Sector-neutral ranking
* Long-short top and bottom baskets
* Risk-parity basket allocation
* Cross-sectional machine-learning ranking

## 7.4 Relative-value strategies

* Cointegrated pairs
* Rolling hedge-ratio pairs
* PCA residual baskets
* Beta-neutral spreads
* Spot-perpetual basis
* Perpetual funding carry
* Calendar basis where matching contracts exist
* Cross-symbol relative volatility
* Index versus constituent residuals
* Synthetic basket spreads

## 7.5 Microstructure strategies

* Bid and ask depth imbalance
* Microprice displacement
* Aggressor trade imbalance
* Cancel and add pressure
* Spread compression and expansion
* Liquidity-vacuum detection
* Short-horizon continuation
* Short-horizon reversal
* Liquidation clustering
* Mark-price divergence
* Order-book resilience
* Event-time volatility

The existing event capture already supports trades, aggregate trades, top of book, 100 ms depth, mark price, and liquidations.

## 7.6 Machine-learning strategies

* Logistic regression
* Elastic-net classification
* Linear regression
* Gradient boosting
* LightGBM
* Random forest
* Calibrated classifiers
* Pairwise ranking
* Cross-sectional ranking
* Triple-barrier classifiers
* Return regressors
* Meta-labelling
* Regime classification
* Volatility prediction
* Dynamic position sizing
* Online learning
* Shallow sequence models
* Small temporal convolutional networks

The final system keeps model training CPU-bound and single-worker on the Mac.

## 7.7 Meta-strategies

* Regime routing
* Strategy weighting
* Bayesian model averaging
* Performance-decay weighting
* Correlation-aware ensemble
* Conflict suppression
* Confidence calibration
* Dynamic sleeve allocation
* Volatility targeting
* Drawdown-based deallocation

## 7.8 Execution strategies

* Market execution
* Passive limit execution
* Post-only execution
* Time-sliced execution
* Volume-weighted execution
* Spread-aware order selection
* Depth-aware sizing
* Cancel and replace
* Multi-leg hedge execution
* Emergency unwind

---

# 8. Unified research engine

The current repository has separate backtesting systems for named strategies, grid search, generated hypotheses, ML, relative value, and microstructure.

The final system uses one research coordinator and two simulation engines.

## Bar simulation engine

Used for:

* Trend
* Momentum
* Mean reversion
* Cross-sectional
* Allocation
* ML
* Most relative-value systems

It supports:

* Multiple simultaneous symbols
* Multiple simultaneous positions
* Portfolio cash and margin
* Funding
* Fees
* Dynamic position sizes
* Target positions
* Portfolio rebalancing
* Partial exits
* Staged entries
* Overlapping strategies

The current strategy backtester models non-overlapping trades with fixed TP, SL, and time exits. That remains available as a fast evaluator, but it is not the final portfolio simulator.

## Event simulation engine

Used for:

* Microstructure
* Liquidation signals
* Limit-order strategies
* Short-horizon execution
* Multi-leg execution tests

It replays:

* Trades
* Order-book updates
* Best bid and ask
* Liquidations
* Mark prices
* Funding events

It models:

* Queue position estimate
* Partial fills
* Order expiry
* Cancel latency
* Spread
* Visible depth
* Market impact
* Adverse selection
* Connection gaps

## Strategy generators

The research coordinator accepts candidates from:

* Registered strategy catalogue
* Parameter search
* Generated DSL
* Mutation
* Crossover
* OpenClaw proposals
* OpenClaw code patches
* ML search
* Cross-sectional search
* Relative-value search
* Microstructure search
* Ensemble construction

The registered library and the generative factory therefore enter the same pipeline.

## Research funnel

Each candidate passes through:

### Screening

* Compile strategy
* Validate features
* Validate causality
* Measure signal frequency
* Estimate turnover
* Reject invalid or empty behaviour

### Development evaluation

* Chronological train and validation
* Cost-adjusted return
* Regime breakdown
* Parameter stability
* Trade or observation sample checks
* Cross-symbol stability
* Portfolio overlap checks

### Robustness evaluation

* Walk-forward analysis
* Purging and embargo
* Cost stress
* Delay stress
* Missing-data stress
* Funding stress
* Monte Carlo trade-order resampling
* Bootstrap confidence ranges
* Probability of backtest overfitting
* Deflated Sharpe
* Drawdown stability

### Protected evaluation

* Frozen cohort
* Final holdout claim
* No adaptive feedback from protected results
* Exact data and code hashes

### Forward evaluation

* Production-equivalent paper engine
* Real-time market data
* Exact strategy fingerprint
* Exact cost and execution model
* Minimum observation duration
* Minimum trades or rebalance events
* Drift checks

## Strategy-specific evidence

The final system does not use one trade-count rule for every strategy type.

Evidence units are:

| Strategy type   | Evidence unit                                        |
| --------------- | ---------------------------------------------------- |
| Scalping        | Independent trades and event windows                 |
| Intraday        | Trades and trading days                              |
| Swing           | Trades, months, and regimes                          |
| BTC allocation  | Exposure days and market regimes                     |
| Cross-sectional | Rebalance dates and portfolio returns                |
| Funding carry   | Funding intervals                                    |
| Pairs           | Independent spread excursions                        |
| Market making   | Orders, fills, and inventory cycles                  |
| ML              | Purged prediction windows and calibrated probability |

---

# 9. Agentic research and implementation

OpenClaw becomes a complete research and implementation worker.

The existing bridge already supports bounded research actions and excludes credentials, approvals, raw market data, and protected holdouts.

## Agent roles

The agent runs four logical roles:

| Role        | Responsibility                                     |
| ----------- | -------------------------------------------------- |
| Researcher  | Generates a thesis and strategy design             |
| Critic      | Checks economic logic and likely failure modes     |
| Implementer | Produces DSL or Python code                        |
| Reviewer    | Checks tests, causality, results, and resource use |

These can run through one OpenClaw instance with separate prompts and state.

## Agent actions

OpenClaw can:

* Create a new strategy thesis
* Revise an existing strategy
* Create a generated DSL specification
* Create a Python strategy module
* Create a feature
* Create a data adapter
* Create a parameter space
* Create an ML experiment
* Create an ensemble
* Request a new research test
* Retire a failed lineage
* Create tests
* Run bounded research jobs
* Produce a Git branch
* Produce a merge request
* Produce an evidence report

## Agent code workflow

Agent-generated code runs on the Mac in an isolated worktree.

The system performs:

* Formatting
* Static type checks
* Unit tests
* Property tests
* Determinism tests
* Look-ahead tests
* Data-access tests
* Resource-limit tests
* Synthetic signal tests
* Historical backtests
* Strategy contract validation

Code that passes can merge automatically into the research branch.

Live execution never imports unreviewed source from an agent workspace. The promotion engine deploys only frozen artefacts built from committed source.

## Agent data access

OpenClaw receives:

* Strategy catalogue
* Feature catalogue
* Instrument universe
* Development results
* Failure reasons
* Signal-frequency reports
* Cost model
* Resource budget
* Current research queue
* Existing strategy lineage
* Public market summaries

It does not receive protected holdout details before the strategy lifecycle permits them.

## Autonomous promotion

The promotion engine supports these modes:

* Automatic research registration
* Automatic development testing
* Automatic forward-paper deployment
* Automatic bounded live canary deployment
* Automatic canary suspension
* Automatic retirement
* Automatic capital reduction
* Automatic strategy replacement

The agent proposes and implements. The deterministic promotion policy grants lifecycle transitions.

The promotion policy uses:

* Frozen artefact hash
* Validation evidence
* Protected holdout result
* Forward-paper result
* Strategy drift
* Execution drift
* Drawdown
* Cost drift
* Portfolio capacity
* Current risk budget

---

# 10. Execution engine

The current `CcxtBroker` becomes one exchange adapter behind a full order-management system.

## Order lifecycle

```text
CREATED
VALIDATED
PERSISTED
SUBMITTED
ACKNOWLEDGED
PARTIALLY_FILLED
FILLED
CANCEL_PENDING
CANCELLED
REJECTED
EXPIRED
RECOVERY_REQUIRED
RECONCILED
```

Every transition is persisted before the next external action.

## Position lifecycle

```text
FLAT
ENTRY_PENDING
OPEN
REDUCE_PENDING
EXIT_PENDING
RECOVERY_PENDING
FLAT_CONFIRMED
```

## Core components

| Module                 | Responsibility                        |
| ---------------------- | ------------------------------------- |
| `order_planner`        | Converts target deltas to orders      |
| `order_manager`        | Owns order state                      |
| `exchange_adapter`     | Binance and CCXT calls                |
| `user_stream`          | Receives account and order events     |
| `reconciler`           | Compares exchange and local state     |
| `position_manager`     | Maintains positions and stops         |
| `multi_leg_manager`    | Coordinates hedged strategies         |
| `recovery_manager`     | Handles ambiguous and partial results |
| `execution_cost_model` | Estimates spread, impact, and fees    |
| `paper_exchange`       | Production-equivalent simulation      |

## Multi-symbol execution

The account no longer needs to be flat before each entry.

The reconciler maintains:

* Known positions
* Known orders
* Reserved margin
* Strategy ownership
* Portfolio ownership
* Unknown account exposure
* Exchange balances
* Exchange margin
* Current leverage
* Current liquidation distance

A new order is permitted when the target delta and full account state agree.

## Partial fills

Partial fills create a valid intermediate state.

The system records:

* Filled quantity
* Remaining quantity
* Average fill
* Fees
* Position delta
* Hedge requirement
* Timeout
* Cancel decision
* Replacement decision

## Multi-leg execution

Multi-leg strategies use an order-group state machine:

```text
PLANNED
PRIMARY_SUBMITTED
PRIMARY_PARTIAL
HEDGE_SUBMITTED
HEDGED
ACTIVE
EXITING
FLAT
RECOVERY
```

This supports:

* Pairs
* Spot-perpetual basis
* Funding carry
* Baskets
* Beta-neutral spreads

## Stops and exits

The engine supports:

* Native exchange stop orders
* Local strategy exits
* Time exits
* Portfolio exits
* Drawdown exits
* Risk-engine exits
* Emergency flatten
* Partial take-profit
* Trailing stop
* Multi-leg exit coordination

---

# 11. Risk engine

Risk is enforced at six levels.

## Strategy level

* Risk per trade
* Maximum position
* Maximum turnover
* Maximum trades
* Cooldown
* Maximum holding period
* Maximum slippage
* Maximum funding cost

## Instrument level

* Maximum notional
* Maximum order size
* Maximum share of visible depth
* Maximum spread
* Maximum volatility
* Maximum position concentration

## Sleeve level

* Capital budget
* Risk budget
* Maximum drawdown
* Maximum correlation
* Maximum beta
* Maximum turnover

## Product level

* BTC accumulation exposure limits
* Active-income gross and net exposure
* Product drawdown
* Product margin use
* Product daily loss

## Account level

* Available balance
* Used margin
* Maintenance margin
* Liquidation buffer
* Unknown positions
* Unknown orders
* Account drawdown

## Global level

* Total capital
* Total drawdown
* Exchange connectivity
* Data freshness
* Clock condition
* Database condition
* Execution drift
* Model drift

Risk decisions are deterministic and are stored with the input snapshot that produced them.

---

# 12. Data platform

## Storage classes

| Data                              | Storage                               |
| --------------------------------- | ------------------------------------- |
| Orders, fills, positions, jobs    | PostgreSQL                            |
| Strategy metadata and evidence    | PostgreSQL                            |
| Agent actions and promotion state | PostgreSQL                            |
| Bars and funding                  | Parquet                               |
| Raw trades and depth              | Parquet                               |
| Derived features                  | Parquet                               |
| Backtest equity and trades        | Parquet                               |
| Models and immutable artefacts    | Content-addressed filesystem          |
| Reports                           | Generated from PostgreSQL and Parquet |

## Historical query engine

DuckDB queries Parquet directly.

The strategy API continues to expose pandas-compatible frames where required. Large scans and joins move to DuckDB and Arrow.

## Partition layout

```text
data/
  raw/
    venue/
      market/
        event_type/
          symbol/
            date/
  bars/
    venue/
      market/
        symbol/
          timeframe/
  features/
    feature_set/
      venue/
        market/
          symbol/
            timeframe/
  snapshots/
  models/
  artefacts/
  reports/
```

## Time semantics

Every data record can contain:

* Exchange event time
* Local receive time
* Bar open time
* Bar close time
* Feature availability time
* Strategy evaluation time
* Order decision time
* Order submission time
* Fill time

Higher-timeframe features become available only after their source candle closes.

## Universe history

Each universe change creates:

* Snapshot ID
* Timestamp
* Membership
* Inclusion reason
* Exclusion reason
* Liquidity metrics
* Data availability
* Strategy eligibility

Backtests use the universe snapshot that was available at that historical time.

---

# 13. PostgreSQL schema

The final database contains these core tables.

## Market and data

* `instrument`
* `instrument_status`
* `universe`
* `universe_snapshot`
* `universe_member`
* `dataset_snapshot`
* `feature_set`
* `feature_manifest`

## Research

* `strategy_definition`
* `strategy_version`
* `strategy_lineage`
* `experiment`
* `experiment_run`
* `experiment_metric`
* `validation_result`
* `holdout_claim`
* `model_artifact`
* `strategy_artefact`
* `forward_evidence`

## Agent

* `agent_action`
* `agent_proposal`
* `agent_patch`
* `agent_review`
* `agent_disposition`

## Portfolio

* `portfolio`
* `portfolio_sleeve`
* `portfolio_strategy`
* `alpha_forecast`
* `target_position`
* `risk_snapshot`
* `risk_decision`

## Execution

* `account`
* `balance_snapshot`
* `order_intent`
* `exchange_order`
* `order_group`
* `fill`
* `position`
* `position_event`
* `reconciliation_event`

## Accounting

* `nav_snapshot`
* `accounting_entry`
* `trade_attribution`
* `funding_entry`
* `fee_entry`

## Operations

* `job`
* `job_attempt`
* `worker`
* `worker_lease`
* `service_heartbeat`
* `alert`
* `control_event`
* `promotion_event`

---

# 14. Final repository structure

```text
src/
  domain/
    instruments.py
    market_events.py
    strategies.py
    forecasts.py
    portfolios.py
    orders.py
    positions.py
    risk.py

  data/
    binance_market.py
    binance_user_stream.py
    catalogue.py
    parquet_store.py
    feature_store.py
    snapshots.py
    replay.py

  research/
    coordinator.py
    catalogue.py
    generators/
    validators/
    backtest/
      bar_engine.py
      event_engine.py
      portfolio_engine.py
    ml/
    cross_sectional/
    relative_value/
    microstructure/
    artefacts.py
    promotion.py

  strategies/
    library/
    generated/
    agent/
    indicators.py
    features.py
    registry.py

  portfolio/
    aggregation.py
    optimiser.py
    allocator.py
    sleeves.py
    exposure.py

  products/
    btc_accumulation.py
    active_income.py

  execution/
    adapters/
      binance_spot.py
      binance_futures.py
    order_planner.py
    order_manager.py
    order_groups.py
    position_manager.py
    reconciler.py
    recovery.py
    stops.py
    paper_exchange.py
    cost_model.py

  risk/
    strategy.py
    instrument.py
    sleeve.py
    product.py
    account.py
    global_risk.py

  accounting/
    ledger.py
    nav.py
    attribution.py
    reconciliation.py

  agents/
    openclaw_bridge.py
    context.py
    proposals.py
    code_worker.py
    reviewer.py
    sandbox.py

  services/
    market_gateway.py
    execution_service.py
    research_worker.py
    feature_worker.py
    supervisor.py
    scheduler.py
    control_api.py

  observability/
    metrics.py
    health.py
    decision_trace.py
    reports.py
```

---

# 15. Existing module changes

| Current component                            | Final change                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| `src/run_bot.py`                             | Split into product, portfolio, position, and execution services                 |
| `src/strategies/backtester.py`               | Retain as fast single-strategy evaluator                                        |
| `src/day_trade_search.py`                    | Replace with the unified research coordinator                                   |
| `research_exploration/`                      | Move into `src/research/`                                                       |
| `src/autopilot/research_factory.py`          | Keep as one candidate generator                                                 |
| `src/autopilot/ml_research.py`               | Convert into a general ML experiment provider                                   |
| `src/autopilot/relative_value_research.py`   | Promote to first-class strategy provider                                        |
| `src/autopilot/microstructure_research.py`   | Connect to the event backtester and portfolio forecasts                         |
| `src/autopilot/portfolio.py`                 | Expand into target-position optimisation                                        |
| `src/autopilot/runtime.py`                   | Reduce to supervision and service coordination                                  |
| `src/execution/ccxt_broker.py`               | Keep as an adapter, remove business-state ownership                             |
| `outputs/active_strategies_*.json`           | Replace with versioned immutable strategy artefacts                             |
| `runtime/*.json` state                       | Move authoritative state into PostgreSQL                                        |
| `runtime/research/experiment_memory.sqlite3` | Migrate to PostgreSQL experiment tables                                         |
| `config/autopilot.json`                      | Split into platform, accounts, products, portfolios, research, and risk configs |
| `bootstrap_strategies.py`                    | Replace with a real isolated execution-diagnostic product                       |
| OpenClaw proposal bridge                     | Add code patch, test, review, and research execution support                    |

JSON files can remain as generated audit exports. They are no longer the authoritative state.

---

# 16. Configuration model

## Platform configuration

Contains:

* Node identities
* Service assignments
* PostgreSQL connection
* Data paths
* Worker limits
* Network endpoints
* Logging
* Metrics
* Alerting
* Backup policy

## Account configuration

Contains:

* Account ID
* Venue
* Market
* Product access
* Quote and settlement assets
* Testnet or production
* Leverage limits
* Margin mode
* API environment variable names

Secrets remain outside version control.

## Product configuration

Contains:

* Product ID
* Objective
* Base accounting asset
* Account
* Universe
* Portfolio sleeves
* Strategy eligibility
* Promotion policy
* Product risk
* Accounting policy

## Research configuration

Contains:

* Strategy families
* Search budgets
* Data requirements
* Validation policies
* Holdout policies
* Model families
* Agent permissions
* Worker assignment
* Resource limits

## Promotion configuration

Contains:

* Automatic paper promotion
* Automatic live canary promotion
* Canary capital limit
* Required forward evidence
* Maximum drawdown
* Maximum execution drift
* Maximum model drift
* Retirement conditions
* Capital-increase policy

---

# 17. Observability and control

The control API and dashboard show:

## Trading

* Positions
* Orders
* Fills
* Account balances
* Margin
* Liquidation distance
* Stops
* Target positions
* Current alpha forecasts

## Research

* Candidate queue
* Active experiments
* Rejection reasons
* Signal frequency
* Validation results
* Holdout states
* Forward-paper states
* Strategy lineage
* Agent activity

## Products

* BTC accumulated
* BTC versus passive hold
* Active-income NAV
* Drawdown
* PnL by strategy
* PnL by sleeve
* PnL by symbol
* PnL by regime
* Fees
* Funding
* Slippage

## Operations

* Data freshness
* WebSocket health
* Job queue
* Worker utilisation
* Disk use
* Memory use
* Service heartbeats
* Backup state
* Decision-funnel status

Every no-trade state must identify the first blocked stage.

---

# 18. Hardware operating scope

With the Linux OptiPlex and the 2017 MacBook Pro, the final system supports:

* All Binance futures symbols at bar-data level
* Dynamic multi-symbol portfolio research
* 1-minute to weekly strategies
* Multiple simultaneous futures positions
* Cross-sectional strategies
* Pairs and basis research
* Funding strategies
* CPU-based ML
* Continuous autonomous research
* Continuous agent code generation
* Order-flow research for a selected symbol subset
* Second-level event-driven execution
* Full paper and small-to-moderate live portfolio execution

The event-data budget is assigned dynamically:

| Data type                 | Scope                                |
| ------------------------- | ------------------------------------ |
| Candles                   | Full eligible universe               |
| Funding and open interest | Full eligible universe               |
| Trades and best bid/ask   | Active research and trading universe |
| 100 ms depth              | Small rotating liquid subset         |
| Full event replay         | Mac research worker                  |

This is a complete systematic and event-driven trading platform. The hardware boundary is exchange-network latency, not strategy breadth.

---

# 19. Final acceptance criteria

The product is complete when all of these conditions hold.

## Data

* The universe is point-in-time correct.
* Market data has availability timestamps.
* Feature calculations are deterministic.
* Missing or stale data blocks only the affected instrument.
* Historical and live feature values match.

## Research

* Every strategy family uses the same core contracts.
* Named, generated, ML, relative-value, and agent strategies enter one research queue.
* Results include costs and portfolio overlap.
* Protected holdout data cannot enter adaptive feedback.
* Strategy evidence is reproducible from exact hashes.
* Rejection reasons are machine-readable.

## Agent

* OpenClaw can create DSL strategies.
* OpenClaw can create Python strategies.
* OpenClaw can create tests and features.
* Generated code runs in an isolated Mac workspace.
* Passing code can enter the research branch automatically.
* The agent can trigger bounded research jobs.
* The agent cannot directly send exchange orders.

## BTC accumulation

* Strategies produce target BTC allocations.
* The system can sell and rebuy tactical BTC.
* Accounting is correct in BTC.
* Passive BTC hold is the benchmark.
* Multiple strategies can contribute to one BTC target.

## Active income

* The portfolio supports multiple symbols.
* Multiple positions can exist at the same time.
* Forecasts are aggregated at portfolio level.
* Correlation, beta, liquidity, funding, and margin affect allocation.
* Relative-value and multi-leg positions use order groups.
* Partial fills and recovery are supported.

## Execution

* Every order has a durable intent.
* Partial fills are represented correctly.
* Exchange and local states reconcile after restart.
* Unknown exchange states create recovery workflows.
* Stops survive process restarts.
* Multi-leg exposure has a deterministic hedge or unwind path.
* Paper and live engines use the same order and position contracts.

## Accounting

* Fees, funding, slippage, and realised PnL reconcile.
* BTC and USDT products have separate ledgers.
* NAV can be reconstructed from immutable entries.
* PnL attribution exists by strategy, symbol, sleeve, and product.

## Operations

* Linux can continue trading when the Mac is offline.
* Research jobs resume after interruption.
* Execution does not wait for research.
* All services expose heartbeats.
* Backups restore operational and research state.
* A diagnostic strategy proves the complete paper trading path.
* Every no-trade condition has an exact recorded cause.

This is the final target: one distributed quantitative trading platform, two independent portfolios, one shared strategy contract, one research system, one portfolio engine, one execution engine, and a bounded autonomous agent that can create, implement, test, and promote strategies.
