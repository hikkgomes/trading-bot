Use multi-timeframe market structure.

Examples:

* 1d / 4h defines macro or regime context
* 1h / 30m defines setup
* 15m / 5m defines confirmation
* 1m defines trigger or execution timing

Example hypothesis:

```text id="k9nh2s"
When BTC is in a 4h bullish trend,
and the 30m timeframe shows a pullback,
and the 5m or 1m timeframe shows momentum recovery,
there may be a profitable long day-trade setup.
```

Another example:

```text id="do40th"
When 4h volatility is compressed,
30m range tightens,
and 5m/1m breaks range with volume expansion,
there may be a profitable breakout setup.
```

Another example:

```text id="tzv7mk"
When BTC sells off sharply on 15m/30m,
but 4h trend is still intact,
and 1m/5m reclaim happens,
there may be a mean-reversion scalp/day-trade setup.
```

Your job is to help me build a research system that explores these kinds of hypotheses.

Do not start by reading old strategy results unless needed only to understand what not to repeat.

## First principle

Do not ask “which current strategy can be improved?”

Ask:

```text id="wbzldu"
What market behaviours can be expressed as structured multi-timeframe hypotheses,
and how can we test them systematically?
```

## Required workflow

Create a new exploratory workflow, separate from the old failed experiments.

Suggested folder:

```text id="tjo4f3"
research_exploration/
```

or another clean name.

The workflow should have these parts:

## 1. Data and feature inventory

First, inspect what data and columns exist across all available parquets/timeframes.

Do not load huge files fully unless necessary.

Create a script that inventories:

* timeframe
* column names
* indicator families
* OHLCV columns
* trend indicators
* momentum indicators
* volatility indicators
* volume indicators
* order-flow features if available
* target/future columns to exclude
* columns that look potentially leaky
* missing/invalid columns where cheap to check

The point is to make the agent aware of the full feature universe across:

```text id="uany52"
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
```

Do not default to only 15m or 5m.

## 2. Multi-timeframe hypothesis grammar

Create a grammar for strategy hypotheses.

A candidate strategy must have this structure:

```text id="jop1ac"
REGIME_CONTEXT + SETUP + TRIGGER + EXIT + RISK
```

Where:

```text id="9v7owr"
REGIME_CONTEXT = higher timeframe condition
SETUP = mid timeframe condition
TRIGGER = lower timeframe entry condition
EXIT = TP / SL / trailing / time / invalidation
RISK = sizing, max loss, cooldown, volatility filter
```

Example:

```text id="lt8zcd"
REGIME_CONTEXT:
4h EMA trend up and 1d close above long moving average

SETUP:
30m RSI pullback or price near VWAP/band support

TRIGGER:
5m or 1m close above short EMA with volume expansion

EXIT:
ATR stop, fixed R multiple, time exit

RISK:
skip if volatility too low/high, max trades per day, max daily loss
```

## 3. Strategy families to explore

Create hypothesis generators for at least these families:

### A. Trend continuation

Higher timeframe trend + mid timeframe pullback + low timeframe re-entry trigger.

### B. Volatility breakout

Higher/mid timeframe compression + low timeframe breakout + volume confirmation.

### C. Mean reversion

Price stretched too far + reclaim trigger + strict invalidation.

### D. Momentum continuation

Strong directional pressure + shallow pullback + fast continuation entry.

### E. Liquidity sweep / failed breakdown

Price breaks previous low/high, then reclaims, with higher timeframe context.

### F. Regime avoidance

No-trade conditions where strategies usually fail:

* chop
* volatility too low
* volatility too high
* bad risk/reward after fees
* too many recent false breaks

Do not generate random indicator combinations. Generate candidates that belong to a named market behaviour.

## 4. Candidate generation

Create a candidate generator that produces structured hypothesis objects.

Each object should include:

* hypothesis ID
* family
* human-readable idea
* direction: long/short/both
* regime timeframe
* setup timeframe
* trigger timeframe
* exact feature columns used
* entry rules
* exit rules
* risk rules
* expected holding period
* expected trade frequency
* reason this behaviour could exist
* what would invalidate it

Example object:

```json
{
  "id": "TREND_PULLBACK_LONG_001",
  "family": "trend_continuation",
  "idea": "In a 4h bullish regime, 30m pullbacks followed by 5m momentum recovery may produce positive expectancy long trades.",
  "direction": "long",
  "regime_timeframe": "4h",
  "setup_timeframe": "30m",
  "trigger_timeframe": "5m",
  "market_logic": "Trend followers and dip buyers may re-enter after controlled pullbacks in a higher-timeframe uptrend.",
  "entry": [
    "4h trend condition",
    "30m pullback condition",
    "5m recovery trigger"
  ],
  "exit": [
    "ATR stop",
    "take-profit",
    "time exit"
  ],
  "risk": [
    "max daily loss",
    "max trades per day",
    "skip extreme volatility"
  ]
}
```

## 5. Testing approach

The system should be able to test candidates using existing backtest infrastructure where possible, but do not force the old condition-grid engine if it does not fit the hypothesis grammar.

If needed, write an adapter.

Testing must include:

* fees
* slippage
* realistic next-bar entry
* no lookahead bias
* train/validation/holdout split
* enough trade count
* yearly/monthly breakdown
* long and short separately
* different market regimes
* parameter sensitivity checks

But do not begin with a massive full search.

First build a small, controlled smoke test.

## 6. Exploration before optimisation

The first output should not be “the best strategy”.

The first output should be a report answering:

```text id="xnfkzi"
Which multi-timeframe hypothesis families can this repo currently express?
Which data/features are available for each?
Which families need extra data?
Which first 20–50 hypotheses should we test?
Which existing code can evaluate them?
Which code is missing?
```

## 7. Do not overfit

Avoid brute-forcing thousands of arbitrary combinations.

The research flow should be:

```text id="j7mm8o"
market idea → precise hypothesis → controlled test → reject/keep → log result
```

not:

```text id="pkttxr"
all indicators × all thresholds × all timeframes × all exits
```

## 8. Use all timeframes intentionally

Candidates should explicitly combine timeframes.

Examples:

```text id="lwh6t0"
1d + 4h + 30m + 5m
4h + 1h + 15m + 1m
1h + 30m + 5m
30m + 5m + 1m
```

Do not silently collapse everything to 15m.

Do not use a single timeframe unless the hypothesis specifically requires it.

## 9. Deliverables for this Claude session

Build or draft:

1. `research_exploration/feature_inventory.py`
2. `research_exploration/hypothesis_schema.py`
3. `research_exploration/hypothesis_generator.py`
4. `research_exploration/strategy_families.py`
5. `research_exploration/experiment_log.py`
6. A markdown report:
   `outputs/research_exploration/initial_research_map.md`

The report should include:

* available timeframes
* usable feature families
* missing feature types
* proposed hypothesis families
* first batch of candidate hypotheses
* which existing repo modules can be reused
* which new adapters are needed
* suggested next command for me to run

Do not run expensive searches.

Run only lightweight schema inspection, unit tests, or synthetic/small-row smoke tests unless I approve otherwise.

## Final instruction

Your job is not to prove a strategy works today.

Your job is to build the system that lets us explore the BTC/USDT market properly, across all available timeframes, without getting trapped in previous failed experiments.
