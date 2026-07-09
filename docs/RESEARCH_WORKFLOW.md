# Research Workflow — from market idea to a strategy the bot can run

This is the end-to-end runbook for finding strategies and handing them to the
paper bots. It covers both products:

1. **Position / BTC accumulation** (`pnl_unit: btc`) — hold BTC, occasionally
   short pullbacks so you end with *more BTC than buy-and-hold*. Long-term
   savings.
2. **Day-trade / swing on futures** (`pnl_unit: usdt`) — repeatable intraday /
   multi-day setups for steady USDT income.

The loop is always:

```
market idea -> precise hypothesis -> staged validation -> keep/reject (logged)
            -> export kept -> paper bot -> monitor -> (eventually) live
```

Two research engines feed the same execution contract:

| engine | style | admission | modules |
|---|---|---|---|
| **Hypothesis exploration** (preferred) | curated multi-timeframe hypotheses from named market behaviours | 5-stage validation, **holdout gates** | `research_exploration/` |
| Condition-grid search (legacy) | combinatorial condition mining | walk-forward + DSR; holdout **now gates at export** | `src/strategy_search`, `src/day_trade_search` |

Everything below runs from the project root with the venv active. Steps marked
**HEAVY** read multi-GB parquets or run long — on this repo's convention they
need your own go-ahead; nothing here launches them automatically.

---

## 0. One-time / periodic: refresh data

```bash
python -m src.update_candles --market futures --bootstrap-days 90 --skip-if-missing --timeframes 5m 15m 30m 1h 4h 1d
python -m src.update_candles --market spot --bootstrap-days 365 --skip-if-missing --timeframes 1h 4h 1d 1w
```

(Full rebuild — `python build_binance_indicator_dataset.py` — is **HEAVY** and
rarely needed.)

## 1. Generate hypothesis batches (cheap)

```bash
# Scenario 2 (day/swing): 40 candidates, 5 families x tf stacks x long/short
python -m research_exploration.hypothesis_generator --print

# Stricter variant with Family-F no-trade guards (anti-chop + vol band)
python -m research_exploration.hypothesis_generator --with-guards

# Scenario 1 (BTC accumulation): 15 short-side swing candidates on coarse stacks
python -m research_exploration.hypothesis_generator --position
```

Batches land in `outputs/research_exploration/hypotheses_*.json`. Every
candidate is a `REGIME + SETUP + TRIGGER + EXIT + RISK` object from a *named*
market behaviour — never a blind indicator grid.

## 2. Prove wiring with synthetic data (cheap, run anytime)

```bash
python -m research_exploration.evaluate --synthetic
python -m research_exploration.validation --synthetic
python -m research_exploration.validation --synthetic --position --pnl-unit btc
python -m pytest -q
```

## 3. Staged validation on real data (**HEAVY**, bounded window)

Autonomous bounded loop:

```bash
make research-cycle
```

That command runs the current active-income and BTC-accumulation candidate
batches against local real data, logs every verdict, and attempts export through
the same positive-holdout gate. Active-income export also requires
`dsr_deflated >= 0.60`, so DSR-near-zero flow candidates stay research-only even
if they look positive once. It skips repeated validation when the latest local
market-data timestamp has not changed. A run with no keepers is not a failure;
it records `no_exportable_strategies`.

If a product currently has an open paper/live position, the cycle does not replace
that product's active strategy artifact even when new keepers exist. It records
`open_positions_block_export` and waits for the existing position to close first,
so the executor can continue managing exits with the strategy definition that
opened the position.

The autonomous cycle is bounded by design. Active-income scenarios validate
deterministic slices of the full curated batch, including a recent 1m scalping
slice, and advance a cursor in `runtime/research_cycle_state.json` only after a
successful run. Candidates whose required features are missing from the current
local indicator parquets are reported as `unsupported_hypotheses` while the
supported candidates in the slice still run. DSR is deflated by the full
available universe for the scenario, not just the current slice, so rotating
small slices does not dilute the multiple-testing penalty.
If `runtime/research_cycle_state.json` is corrupt or not a JSON object, the cycle
records `state_recovered`, reruns instead of skipping on stale markers, and
rewrites a clean state file after a successful run. A corrupt mutation batch is
reported as `mutation_batch.status=read_error` and ignored, so curated baseline
research still runs while unsafe generated inputs stay out of validation.
Operator reporting and healthcheck warnings surface both conditions so they are
visible to a lightweight watchdog without turning baseline research into a hard
failure.

The cycle also writes `runtime/incubation_candidates.json`, a ranked
research-attention queue built from rejected non-keepers. It is marked
`research_only`, `executable: false`, `paper_trade_allowed: false`,
`live_allowed: false`, and `promotion_eligible: false`; it exists to guide
mutation and review work, not to feed paper trading or promotion gates. Missing
eligibility flags are also treated as not live-eligible by promotion/live
policy.

Manual equivalents:

```bash
# Scenario 2 — day/swing batch, per base timeframe:
python -m research_exploration.validation --real --base-tf 15m \
    --start 2022-01-01 --end 2025-12-31
python -m research_exploration.validation --real --base-tf 5m --with-guards \
    --start 2023-01-01 --end 2025-12-31

# Scenario 1 — position batch, scored in BTC (use the FULL history: these
# candidates trade weekly-to-monthly and need every year of data):
python -m research_exploration.validation --real --position --pnl-unit btc \
    --base-tf 4h --start 2020-06-01 --end 2025-12-31
python -m research_exploration.validation --real --position --pnl-unit btc \
    --base-tf 1h --start 2020-06-01 --end 2025-12-31
```

A batch spans several base timeframes; run once per `--base-tf` (candidates on
other bases are skipped in that run). Every verdict is appended to
`outputs/research_exploration/experiment_log.jsonl`.

Stage minimum trade counts default to day-trade cadence (30 train / 10
validation / 5 holdout) and automatically drop to 15/5/3 with `--position`
(swing cadence can't produce day-trade counts); override with
`--min-trades-train/--min-trades-val/--min-trades-holdout`.

A hypothesis is only **kept** if it survives, in order:

1. **TRAIN** (first 60%) — shows an edge at all;
2. **VALIDATION** (next 20%) — the edge repeats out-of-sample;
3. **OOS WINDOWS** — the edge is spread across time, not one lucky cluster;
4. **SENSITIVITY** — survives ±25% exits and jittered entry thresholds;
5. **HOLDOUT** (final 20%) — untouched until stages 1–4 pass, and it **gates**:
   a negative holdout rejects.

DSR is deflated by the batch size, and every hypothesis gets per-regime
(bull/bear/range) and yearly/monthly breakdowns.

Diagnostics for candidates that never fire (or fire and lose — the funnel
shows which predicate is the bottleneck):

```bash
python -m research_exploration.predicate_funnel --base-tfs 5m,15m,30m \
    --start 2024-01-01 --end 2024-07-01
python -m research_exploration.predicate_funnel --position --base-tfs 4h,1h \
    --start 2020-06-01 --end 2025-12-31          # scenario-1 batch
python -m research_exploration.experiment_log --summary
```

## 4. Export kept hypotheses to the bot contract (cheap)

```bash
# Scenario 2 -> day-trade bot artifact
python -m research_exploration.export --pnl-unit usdt \
    --out outputs/active_strategies_research.json

# Scenario 1 -> position bot artifact
python -m research_exploration.export --pnl-unit btc \
    --out outputs/active_strategies_position.json

# Optional extra bars: --min-dsr 0.9  --top-k 3  --ids <ID ...>
```

The exporter only accepts `keep` verdicts, re-checks the holdout was positive
and populated, refuses to mix pnl units in one artifact, and embeds the full
hypothesis + validation metrics + git provenance. Entries carry
`entry_type: "hypothesis"`.

## 5. Run the paper bot on the artifact

```bash
python -m src.run_bot --strategies outputs/active_strategies_research.json \
    --state-file runtime/active_income_state.json \
    --trade-log runtime/active_income_trades.csv \
    --starting-equity 1000 --market futures \
    --objective active_income --base-asset USDT

python -m src.run_bot --strategies outputs/active_strategies_position.json \
    --state-file runtime/btc_accumulation_state.json \
    --trade-log runtime/btc_accumulation_trades.csv \
    --starting-equity 1 --market spot \
    --objective btc_accumulation --base-asset BTC \
    --regime-guard
```

`run_bot` evaluates hypothesis entries with
`research_exploration.predicates.entry_mask` — the *same code* that validated
them — on closed candles only, sizing fetches so every rolling window is
defined. Existing safety rails apply: `max_position_fraction`, daily stop,
per-strategy `max_trades_per_day`, consecutive-loss cooldown, and win-rate drift
kill switch. Stop-distance sizing is capped by `max_position_fraction`; generated
hypothesis artifacts default to 25% notional exposure and a bounded daily trade
cap (`4` entries per active-income strategy, `1` entry per BTC-accumulation
strategy). For multi-strategy artifacts, the account daily stop uses the
strictest `daily_stop_loss` across included strategies, and execution allows only
one open position per product/symbol at a time. Malformed risk or fee blocks fail
at bot load time before any entry can be evaluated. Generated artifacts should
omit leverage and margin metadata unless it is explicitly `leverage: 1`; spot
BTC-accumulation artifacts must not include margin metadata, and futures artifacts
may only declare `margin_mode: isolated`. In normal operation,
`src.autopilot.runtime` schedules data updates and product cycles from
`config/autopilot.json`; direct `run_bot` invocations are for manual paper
checks.

Give a new artifact **weeks of paper trading** and compare its live win rate /
expectancy to the exported `metrics` before considering real money (see
`docs/EXECUTION.md` for the broker layer and its `TRADING_LIVE=1` +
`MAX_NOTIONAL_USD` rails). Promotion review requires every counted paper-trade
row to have finite `net_return` and `sized_return` values; malformed paper
returns block approval recommendations instead of being treated as zero. Invalid
manual review thresholds also fail closed, so an accidentally permissive
promotion command cannot emit an approval command.

## Legacy path: condition-grid searches

Still available; the same export-time discipline now applies:

```bash
python -m src.strategy_search --walk-forward --holdout-fraction 0.2 ...   # HEAVY
python -m src.export_strategies --search-dir outputs/<dir> [--min-dsr 0.9]
```

`export_strategies` now **rejects strategies that lost on their own holdout**
(the old report-only behaviour was the documented flaw that promoted losing
strategies). When a search writes `ranked_strategies_clustered.csv`, export uses
those low-overlap representatives by default; pass `--raw-ranked` only for
deliberate inspection runs. Escape hatch: `--no-holdout-gate` /
`--min-holdout-return`. The current `active_strategies.json` /
`active_strategies_flow.json` were exported before this gate and remain
unvalidated — replace them the next time a search or hypothesis batch produces
genuine keeps.

## Honesty rules (why the gates exist)

- Chronological splits only; the holdout is never touched until everything else
  passes, and it always gates.
- Quantile thresholds are causal (rolling windows), never fit on the full
  sample; higher timeframes join via closed-candle `merge_asof` — no lookahead.
- Fees + slippage on every simulated trade; next-bar-open entries; stop checked
  before target intrabar.
- DSR is deflated by the number of ideas tested together — testing 40 ideas
  costs each of them evidence.
- Every experiment is logged (`experiment_log.jsonl`) so nothing is re-tested
  blind and no negative result is lost.
- A "keep" is a candidate for **paper trading**, not proof. The market gets the
  final vote.
