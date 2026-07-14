# Autonomous Crypto Trading System

Safety-first research, paper trading, and execution framework for a light Linux
server. The repo has two separate products:

1. **BTC accumulation**: conservative, BTC-denominated, spot-only. It may trade
   BTC/USDT to increase total BTC holdings by stepping aside from a bounded
   slice of existing BTC and rebuying with the same quote proceeds. No leverage,
   no futures, and no quote-funded long BTC strategies.
2. **Active income**: USDT-denominated day/swing/scalp research and paper trading
   for Binance USDT futures. Risk is capped per trade and per day.

The mandatory production rule is enforced in code:

**No new or changed strategy artifact can enter live mode unless every strategy
fingerprint is present in `runtime/approvals.json`.**

Environment flags such as `TRADING_LIVE=1` are not enough. The approval ledger is
checked separately by `src.autopilot.approvals`. Live products also require a
fresh automated preflight report by default, so approval alone cannot bypass
environment and broker readiness checks.

## Architecture

| Layer | Modules | Purpose |
|---|---|---|
| Data | `src/autopilot/history_bootstrap.py`, `src/load_data.py`, `src/build_dataset.py` | Resumable native-timeframe Binance history and derived features. |
| Research | `src/autopilot/research_factory.py`, `research_exploration/strategy_grammar.py`, `src/autopilot/experiment_memory.py`, `src/autopilot/research_cycle.py` | Continuously generate typed hypotheses, remember/deduplicate lineages, adapt from development evidence, and run crash-safe staged validation. |
| Strategy contract | `src/export_strategies.py`, `outputs/active_strategies*.json` | Converts validated research results into execution artifacts. Holdout is a hard gate by default. |
| Runtime | `src/autopilot/` | Lightweight 24/7 orchestration, file-based pause control, status reporting, and live approval enforcement. |
| Execution | `src/run_bot.py`, `src/execution/` | Closed-candle paper execution by default; approved active-income live mode can route orders through the ccxt broker behind safety rails. |
| Communications | `src/autopilot/telegram_edge.py`, `src/autopilot/openclaw_bridge.py` | Optional deterministic Telegram alerts/status/pause-only control and an isolated, research-only OpenClaw proposal boundary. |

Generated datasets, search results, runtime state, and trade logs are ignored by
git. The repo should stay small; regenerate artifacts when needed.

## Setup

For a fresh 24/7 Linux server, use the authoritative
[deployment and operations runbook](docs/DEPLOYMENT.md).
Optional Telegram and OpenClaw setup, including their security boundaries, is
documented in [the communications runbook](docs/COMMUNICATIONS.md).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

For a small Linux autopilot server, use Python 3.11+ and the fully pinned runtime
lock:

```bash
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
python3 -m venv .venv
.venv/bin/pip install -r requirements-bot.txt
.venv/bin/pip check
```

`requirements-bot.txt` pins the lean execution/job-worker set, including its
transitive networking and crypto packages and the validated `ccxt==4.5.64`
native-stop path. It covers every enabled default job; optional heavy research
tooling remains off the server. Python and the complete installed package
environment are part of the execution-engine identity used by the live gate,
so do not upgrade or add packages underneath a running approval.

Copy `.env.example` to `.env` for exchange credentials and run `chmod 600 .env`.
Readiness and the Linux installer reject symlinked, non-regular, wrong-owner, or
group/world-accessible environment files. Keep `TRADING_LIVE=0` until the
approval ledger and live adapter path are deliberately enabled.
Copy `config/alerts.env.example` to `runtime/alerts.env` and make it mode `0600`
when webhook or Telegram alerts are wanted; it must never contain exchange
credentials.

## Autopilot

Default config is `config/autopilot.json`. It defines:

| Product | Base asset | Market | Default mode |
|---|---:|---|---|
| `btc_accumulation` | BTC | spot | paper |
| `active_income` | USDT | futures | paper |

Config validation fails closed if BTC accumulation is not BTC/spot, if active
income is not USDT/futures, if a product uses an unknown objective or market, or
if products share runtime-owned strategy, state, trade-log, preflight, or required
testnet-rehearsal evidence paths. Scheduled jobs are command lists, never shell
strings, and validation rejects duplicate job output/report paths so autonomous
jobs cannot silently overwrite each other's evidence. Config booleans must be
JSON booleans (`true` or `false`), and numeric fields such as cadences,
freshness windows, disk limits, and starting equity must be JSON numbers, not
quoted strings or boolean stand-ins. Product routing, runtime paths, and
environment-variable names must be non-empty strings, so hand-edited safety gates
fail closed on typos. Unknown top-level, product, or scheduled-job config keys
are rejected instead of being silently ignored, and duplicate JSON keys are
rejected instead of letting the last value silently win. Non-standard JSON
constants such as `NaN` and `Infinity` are also rejected.
Production runtime and readiness validation also require both core products,
`btc_accumulation` and `active_income`, to be present and enabled, and require
the core autonomous jobs for data updates, regime tagging, generative research,
promotion review, maintenance, and artifact hygiene to be present, enabled,
pointed at their expected Python modules, and wired with the expected
market/product/reporting arguments. Candidate paper testing and verified backup
use separately configured, dedicated systemd timers rather than the generic job
worker.
The Linux deployment isolates product supervision and emergency flattening from
scheduled work: the trading service runs the runtime with `--skip-jobs`, while a
separate job-worker service executes
at most `max_jobs_per_cycle` (`1` by default). The scheduler rotates the starting
job after each execution, so a backlog drains over multiple cycles instead of
always favoring the first configured job. Deferred due jobs remain due for a
later cycle, are persisted in job state as `cycle_job_limit`, and are surfaced as
healthcheck warnings. A job that reaches `max_consecutive_job_deferrals` (`16` by
default) becomes a healthcheck failure so scheduler starvation cannot stay silent.

Commands:

```bash
make autopilot-validate
make bootstrap-strategies
make readiness
make autopilot-once
make jobs-once
make report
make healthcheck
make backup
make backup-verify
make backup-restore RESTORE_DIR=/tmp/trading-bot-restore
make maintenance
make data-update
make data-update-1m-flow
make regime-tag-futures
make strategy-smoke
make research-smoke
make research-factory-validate
make research-generate
make research-cycle
make research-once
python -m src.autopilot.runtime --config config/autopilot.json --skip-jobs
```

`make autopilot-once` runs one supervision/position-management cycle and never
runs scheduled work. `make jobs-once` drains at most one due scheduled job under
the independent job-worker lock. This is the same isolation used by systemd.

For 24/7 operation on Linux with user-level systemd:

```bash
make service-dry-run
DRY_RUN=1 bash scripts/install_autopilot_service.sh
bash scripts/install_autopilot_service.sh
```

The installer runs `src.autopilot.runtime --validate` and
`src.autopilot.readiness` before initially enabling the units. On every trading
supervisor start, systemd repeats strict config validation but does not make
full operational readiness a start gate: the supervisor must still come up to
manage existing exposure and report stale-data or research failures. New live
entries retain their independent approval, preflight, rehearsal, environment,
and broker gates. It installs a separate
`trading-bot-autopilot-jobs.service` for scheduled jobs plus a companion
healthcheck service/timer, which runs the machine-readable healthcheck every five
minutes. Blocking healthcheck issues emit the configured
critical alert through the same JSONL/webhook channel as runtime alerts, with
cooldown applied. Only the trading supervisor loads `$REPO/.env`; the watchdog
reads the strictly allowlisted operations-only `runtime/alerts.env`, and scheduled research,
candidate-paper, and backup units load neither. All units write stdout/stderr to
the user journal, restart or rerun according to their service type, use bounded
timers, and apply a restrictive umask plus lightweight sandboxing options. They
also cap common BLAS/OpenMP/joblib thread pools to
`AUTOPILOT_THREADS=2` by default; override that environment variable when
installing if the server has more or fewer cores available. The generated
units also set conservative cgroup limits by default:
`AUTOPILOT_MEMORY_MAX=1G`, `AUTOPILOT_CPU_QUOTA=75%`, and
`AUTOPILOT_TASKS_MAX=128`; override them during install if your server budget is
different. The generated units also enable additional systemd sandboxing
(`PrivateDevices`, kernel/control-group protections, SUID/SGID restrictions, and
locked personality) while still allowing network access and runtime file writes.
Custom service and timer names supplied through environment variables are
validated as plain `.service` or `.timer` unit names before any unit files are
written, and raw resource/timer overrides are rejected if they are empty or
contain control characters. `DRY_RUN` must be exactly `0` or `1`, so ambiguous
values such as `true` fail before the installer can touch live user units.
`make service-dry-run` writes the generated unit files under
`runtime/systemd-dry-run` for inspection instead of touching the live user
systemd directory.
Operate it with:

```bash
systemctl --user status trading-bot-autopilot.service --no-pager
systemctl --user status trading-bot-autopilot-jobs.service --no-pager
systemctl --user list-timers trading-bot-autopilot-healthcheck.timer --no-pager
journalctl --user -u trading-bot-autopilot.service -f
journalctl --user -u trading-bot-autopilot-jobs.service -f
journalctl --user -u trading-bot-autopilot-healthcheck.service -f
systemctl --user restart trading-bot-autopilot.service trading-bot-autopilot-jobs.service
```

The runtime writes status to `runtime/status.json` and takes an exclusive
`runtime/autopilot.lock` while running so accidental duplicate supervisors cannot
evaluate or trade the same products concurrently. Duplicate startup attempts do
not release the existing holder's lock; the file content is only holder metadata
for inspection. Pause control is a tiny JSON file at
`runtime/operator-control/control.json`, with an atomic CLI wrapper for safer
operation:

```bash
make control ARGS="status"
make control ARGS="pause --reason maintenance"
make control ARGS="resume --reason done"
```

Mutating default control commands append durable JSONL audit events to
`runtime/operator-control/control_audit.jsonl`, including the command, reason, operator, and
before/after control payload. For alternate control files, pass
`--audit path/to/control_audit.jsonl` before the command. If the audit append
fails after the control file is written, the command still prints the applied
control state with `audit_error` so emergency pause/flatten requests are not
lost.

If the control file is malformed, not a JSON object, uses invalid selector
lists, or has typoed boolean controls such as `paused`, `pause_jobs`, or
`flatten_all`, the runtime fails closed: scheduled jobs and new product entries
are paused, while tracked open positions continue management-only supervision.
The cycle is marked failed, and the control parse error is written into
status/reporting. The CLI can repair a bad file with
`make control ARGS="clear --reason repaired"`; when a mutating control command
recovers from malformed existing state, its JSON output and audit event include a
transient `recovered_control_error` while the repaired control file itself stays
clean. The default
`make control` path validates product/job names against `config/autopilot.json`
before writing, and `make control ARGS="status"` prints selector validation for
the current file. Unknown product or job names in manually edited
`paused_products`, `paused_jobs`, or `flatten_products` still fail closed at
runtime so typoed emergency controls are not ignored silently. `make readiness`
also parses the current control file and blocks service preflight when it is
malformed or references unknown products/jobs.

Pause one product:

```bash
make control ARGS="pause-product active_income --reason 'risk review'"
make control ARGS="resume-product active_income --reason done"
```

A product pause blocks new entries. If local state contains a tracked open
position, the runtime continues management-only cycles so risk-reducing exits
remain available.

Pause one autonomous job without pausing trading supervision:

```bash
make control ARGS="pause-job market_data_update_futures --reason 'network maintenance'"
make control ARGS="resume-job market_data_update_futures --reason done"
```

or pause all scheduled jobs while products continue:

```bash
make control ARGS="pause-jobs --reason 'server load'"
make control ARGS="resume-jobs --reason done"
```

To emergency-close live exposure for a product, request flatten. Futures flatten
runs even while paused, writes a deterministic reduce-only order intent before
submission, and clears local open-position state only after the broker is flat,
every tracked native stop is proved terminal, and the keyed trade/equity/daily-
PnL/cooldown accounting commit succeeds. Unreadable or corrupt futures state is not reconstructed:
the request fails closed for exchange/state reconciliation. BTC spot
accumulation is never flattened by selling the BTC base stack; when local state
contains one tracked spot step-aside position, flatten buys BTC back with the
recorded quote-reinvest budget and clears state only after the full fill and BTC
balance increase are proved:

```bash
make control ARGS="panic --reason 'exchange incident'"
make control ARGS="pause --reason 'emergency close'"
make control ARGS="flatten active_income --reason 'emergency close'"
```

`panic` is the one-command fail-safe: it pauses all products, pauses scheduled
jobs, and requests flatten for every live product in one audited atomic
control-file write. BTC spot flatten fails closed without placing an order when
local state is unreadable, has multiple open positions, lacks broker metadata, or
does not contain a quote-reinvest step-aside position. A spot product with no
local step-aside position is reported as skipped and left untouched.

After a successful flatten the runtime atomically clears the corresponding
flatten request; the affected product remains paused. If a crash occurs between
accounting and that control update, the next pass verifies account-wide flatness,
the durable `last_flatten` identity, and the unique trade row without sending
another order, then clears the stale request. Do not manually
`clear-flatten` while a close or reconciliation is unresolved. Reconcile the
exchange, local state, native-stop inventory, fees, funding, and accounting,
then explicitly resume.

Manual JSON edits are still supported for dependency-free recovery; the CLI
writes the same schema atomically.

If a paper product has no exported strategy artifact yet, the runtime reports it
as `waiting_for_strategy_artifact` instead of failing the whole service. Live
products still fail closed until their strategy artifact exists, is explicitly
approved, and has a fresh matching preflight report.

For a fresh server, `make bootstrap-strategies` writes deterministic paper-only
artifacts for missing paper products. These probes are marked
`paper_trade_allowed: true`, `live_allowed: false`, and
`promotion_eligible: false`; they can exercise data, sizing, state, and paper
order flow while research looks for validated edges, but the approval and live
policy paths reject them.

Product symbols are policy-checked at startup. The current system is scoped to
BTC/USDT data and execution: `btc_accumulation` must be spot BTC/USDT, while
`active_income` must be BTC/USDT on USDT-margined futures. Binance compact
symbols (`BTCUSDT`) and ccxt forms (`BTC/USDT`, `BTC/USDT:USDT`) are accepted
where they describe the same instrument; data fetches use compact Binance REST
symbols and ccxt broker calls are normalized internally.

Before any artifact is executed, the runtime also applies a product-aware
strategy policy: positive finite holdout evidence, bounded finite risk per
trade, `max_position_fraction`, daily stop, per-strategy daily trade cap,
consecutive-loss limit, cooldown, sane finite TP/SL and fee inputs, positive
integer horizons, valid condition or hypothesis-entry payloads, and the correct
BTC/USDT PnL unit for the product. Explicit paper-only bootstrap artifacts may
omit performance metrics only while `live_allowed` and `promotion_eligible` are
both `false`; ordinary paper, promotion, and live-eligible artifacts still need
the same positive evidence gates. Promotion/live review also requires
`paper_trade_allowed`, `live_allowed`, and `promotion_eligible` to be
explicitly `true`; missing flags are treated as not live-eligible. Runtime
execution caps stop-distance sizing
by `max_position_fraction` and also prevents
multi-strategy artifacts from stacking simultaneous positions on the same
product/symbol. Active-income artifacts must show finite deflated-Sharpe
evidence (`dsr_deflated` or legacy `dsr`) of at least `0.60`; DSR-near-zero flow
candidates are treated as false positives and blocked before paper/live
execution. BTC-accumulation artifacts must also show positive
finite `holdout_excess_return_vs_buy_hold`, so a strategy is judged by extra BTC
accumulated versus simply holding BTC. Active-income artifacts may risk at most
`0.5%` per trade, `25%` notional exposure, `3%` daily loss, and `8` entries per
strategy per day; BTC-accumulation artifacts may risk at most `0.3%` per trade,
`35%` step-aside exposure, `1%` daily BTC drawdown, and `2` entries per strategy
per day. Older ignored artifacts that no longer pass these rules are blocked
even in paper mode. The same basic executable artifact checks run inside
`src.run_bot` for direct manual invocations. If an artifact or individual
strategy declares `market` or `symbol`, direct bot runs require those fields to
match the configured bot market and symbol. New exports stamp both fields at the
artifact and strategy level. Direct manual product runs
should pass `--objective active_income --base-asset USDT` or
`--objective btc_accumulation --base-asset BTC` so product-specific DSR,
holdout, and step-aside-only guards are enabled outside the autopilot wrapper.
When the BTC accumulation regime guard is enabled, daily macro-regime data
failures block new entries for that cycle instead of assuming risk-on.
Persisted bot state is validated on load as well: equity must be finite and
positive, cooldown timestamps and loss/trade counters must be finite and
non-negative, daily trade counters must be integers, and open-position records
must reference known strategies with valid timestamps, directions, prices,
position sizes, and broker metadata.
Malformed strategy artifacts, including non-object JSON or non-object strategy
entries, are blocked by policy, approval, preflight, and live gates.

Autonomous data, maintenance, promotion-review, hygiene, and research jobs live
under `jobs`. They are command arrays, not shell strings, and each has a cadence
and timeout. Candidate papering and backup have separate top-level cadence and
timeout settings because they run in isolated units. The native history
bootstrap loads the same strict
`config/research_factory.json` used by generation, downloads only the Binance
timeframes declared by its search-space roles, builds the grammar's feature
inventory, writes atomic parquet replacements, and resumes interrupted pagination
from checkpoints. It does not retain legacy timeframes such as 30m unless a
configured search space actually declares them, and it does not rebuild years of
coarse candles from 1m data. Initial and manual refresh commands are:

```bash
make research-history-plan
make research-history-bootstrap
make research-history-bootstrap MARKET=futures
make research-history-bootstrap MARKET=spot
```

The three scheduled `src.autopilot.history_bootstrap` jobs pass that same config.
They partition futures 1m from every configured non-1m futures timeframe and
derive the spot set directly, so a valid role change cannot fall between static
timeframe lists. They keep history current without truncating multi-year files. The
research cycle checks minimum start/end/span/row coverage before consuming a
holdout. Candle timestamp/cadence, OHLCV, duplicate, and future-timestamp checks
fail closed before data replaces the last good parquet. Routine jobs also cover
regime tagging, bounded generation/validation, promotion review, maintenance,
and artifact hygiene. The independent timers cover forward candidate papering
and verified backup while the job worker continues research.

Job state is persisted in `runtime/job_state.json` and recent stdout/stderr tails
are included in `runtime/status.json`. Malformed job state is surfaced in
operator reports and healthcheck as a runtime file issue, and non-object
job-state payloads fail the scheduler closed with a clear error; malformed or
future per-job timestamps are treated as due, surfaced as healthcheck issues for
enabled jobs, and repaired by the next successful run. Malformed job counters and
durations are sanitized in reports and shown as healthcheck job-state issues.
Jobs may log progress to stdout and then print a final JSON object; the scheduler
captures that final structured status for reports, failure reasons, and backoff
state, including pretty-printed JSON objects after progress logs. Large
structured reports are summarized before being stored in runtime status, but list
counts and the first few `errors` are retained so failed maintenance or hygiene
jobs keep actionable failure detail.
`make autopilot-validate` verifies that job working directories and executables
exist, rejects shell-wrapper jobs, and checks configured Python `-m` modules are
importable before the service starts.
Failed jobs retry sooner than their normal cadence, then back off exponentially
up to six hours as consecutive failures accumulate, so a broken data or research
command does not spin a light server.
`make research-smoke` runs the typed compositional grammar and cheap synthetic
validation across every active-income and BTC-accumulation search space and writes
`runtime/research_smoke.json`. It is a wiring check, not evidence of a live edge.
`make strategy-smoke` runs lightweight representative `src.sweep` checks over
synthetic data and, when available, the latest regime-tagged futures parquet; it
writes `runtime/strategy_framework_smoke.json` and is also scheduled by the
autopilot as `strategy_framework_smoke`.
`make research-once` runs one complete bounded real-data iteration: the factory
creates or resumes behaviorally unique active-income and BTC-accumulation ideas,
then the research cycle validates them, checkpoints every candidate in SQLite,
and exports only candidates that passed the protected holdout gate.
Paper products continue to receive successful exports at their configured
`strategies_path`. Live products are different by design: research writes only
to `runtime/candidates/<product>.json` and never replaces the configured active
artifact, so an approved strategy keeps running while a new candidate waits for
review. If no
candidate qualifies, the cycle exits successfully and records
`no_exportable_strategies` in `runtime/research_cycle.json`. Fresh grammar
samples, recursive failure-aware mutation, and crossover continue over
successive cycles with a mandatory exploration floor. Canonical behavior,
lineage, dataset/protocol exposure, development results, and pre-read holdout
claims persist in `runtime/research/experiment_memory.sqlite3`; protected
outcomes never weight future generation. Deflated Sharpe accounting pays for
the cumulative tested population. If a candidate needs a feature missing from
the current local parquets, it is listed under `unsupported_hypotheses` and
retired so it cannot clog the pending queue. Rejected-but-informative candidates are also written to
`runtime/incubation_candidates.json` as a non-executable research queue; that
file is explicitly blocked from paper trading, promotion, and live execution.
The research report and operator report count active paper exports separately
from staged live candidates and print the manual activation step when one is
waiting.
The lightweight strategy framework can also triage ML ideas with
triple-barrier/capped-return targets and Spearman feature screening through
`ml_classifier` and `ml_regressor`; heavier meta-labeling and purged walk-forward
model-signal tooling remain under `src/meta_labeling.py` and
`src/model_signals.py`.
For regime-specific variants, tag a dataset with `python -m src.regime` and
sweep `regime_filter` across `strategy` and `regime_ids`; it masks child
strategy signals outside the selected market states while keeping the child
strategy's TP/SL/horizon defaults. The autopilot also runs
`regime_tag_futures_15m`, which writes `runtime/regime/futures_15m_regime.parquet`
and `runtime/regime_tag_futures_15m.json` as a compact artifact containing
timestamp, OHLCV, and daily regime ID from the market-aware 15m and 1d indicator
files when they exist.
The normal `make data-update` target skips 1m indicator rebuilds because they
are the slow path. If reports show missing 1m scalping flow features such as
`volume_z_20`, run `make data-update-1m-flow` during a maintenance window.

Alerts are enabled by default and written to `runtime/alerts.jsonl` with a
fingerprint cooldown stored in `runtime/alert_state.json`. Set
`AUTOPILOT_WEBHOOK_URL` to send the sanitized payload to an HTTPS webhook
(plain HTTP is accepted only for an explicit loopback test endpoint). The local
alert and cooldown are made durable first; webhook and Telegram delivery then
run on a bounded background queue, so a slow network endpoint cannot stall
trading supervision. Delivery outcomes are appended as
`autopilot.alert_delivery/v1` records with the same fingerprint. Delivery is
best-effort: failures are recorded in the local alert JSONL but do not crash
trading supervision or bypass the alert cooldown. If local alert
file or cooldown writes fail after readiness has passed, the runtime records that
alert error in status and keeps supervising products. If the JSONL alert is
written but cooldown-state persistence fails, the alert result still reports
`sent: true` with `state_error`, and operator reports plus healthcheck warning
details surface that field, so operators know the alert was durable but may
repeat until state storage is fixed. Runtime alerts cover failed
cycles, market-data/feature/disk-space readiness warnings, research cycles that
keep generating hypotheses without exportable candidates while paper products
wait for artifacts, missing/stale required testnet rehearsals, and promotion
reviews where an already-approved strategy later fails the configured
paper-review thresholds.
Failed-cycle alerts include product mode, market, cycle errors, and local
`state_errors` so a paging payload can distinguish paper faults from live faults.
If the cooldown state file is malformed, alerting recovers to a fresh cooldown
map and records the recovery reason in the emitted alert payload.
Alert cooldown fingerprints ignore volatile fields such as timestamps and
free-byte counters, while the full detail is still written to
`runtime/alerts.jsonl`.
`make maintenance` runs bounded housekeeping: it compacts `runtime/alerts.jsonl`
to the most recent 1000 records, prunes `runtime/alert_state.json` to the most
recent 1000 cooldown fingerprints, and rotates older
`outputs/research_exploration/experiment_log.jsonl` and
`runtime/operator-control/control_audit.jsonl` lines into compressed archives while keeping each
hot log at 5000 records. It does not touch trade logs, approvals, strategy
artifacts, or `data/`, and it refuses symlinked log/state inputs instead of
following them during unattended compaction. Independent maintenance tasks keep
running after one task fails; the command prints `ok: false` with per-task
`errors` and exits nonzero so the scheduler records the failed housekeeping
evidence without losing successful task results. The default scheduled
maintenance command also keeps generated files under `runtime/quarantine` within
256 MiB (`268435456` bytes) by deleting the oldest first. For a deliberate manual
override, pass a byte budget such as
`make maintenance QUARANTINE_BYTES=1073741824`.
State backups include each deterministic staged-candidate path and the configured
control audit, which also contains candidate activation intent/completion events.
`make artifact-hygiene` writes a dry-run report of configured artifacts,
policy-blocked active artifacts, unreferenced `active_strategies*.json` files,
and historical search-output directories. Add `APPLY=1` only when you want to
move policy-blocked paper artifacts into `runtime/quarantine`; add
`UNREFERENCED=1` as well to also move unreferenced active-strategy artifacts.
It never deletes files and it will not quarantine live-product artifacts. If one
artifact move fails, the report keeps inspecting the rest, records per-artifact
`errors`, writes `ok: false`, and exits nonzero so the scheduled job preserves
the failure evidence. Operator reports summarize the first hygiene error, and
healthcheck surfaces failed hygiene reports as warnings.
`make healthcheck` writes `runtime/healthcheck.json` and exits nonzero if the
last status heartbeat is stale, missing, or timestamped in the future, the latest
cycle failed, market-data status is missing, stale, invalid, or timestamped in
the future, readiness has blocking failures, an enabled scheduled job is still
marked failing, an enabled job is more than two effective cadences overdue, an
enabled scheduled job reports invalid scheduler state, or the latest verified
backup is stale. Live open positions also fail healthcheck when their
counts/detail rows, risk metadata, or stale-position monitoring metadata are
missing, invalid, or timestamped in the future.
Paper open-position visibility, risk, and monitoring metadata gaps are
warning-level so paper execution issues remain visible without failing the
watchdog.
Invalid trade-log numeric audit fields are surfaced the same way: live product
trade-log corruption is a healthcheck issue, while paper product trade-log
corruption is a warning.
Warning-level readiness checks, such as optional paper-mode `.env` absence or
approval-ledger actor audit warnings, are also copied into healthcheck warnings
without changing the exit code.
Runtime report-refresh failures are also warning-level in healthcheck: they
include the failed report path and any successfully refreshed outputs, but do not
fail the trading watchdog unless the cycle itself failed.
Malformed product state snapshots fail the product cycle, are preserved in
operator reports under `state_errors`, shown in the product table issue column,
and are surfaced directly by healthcheck: live state errors are issues, while
paper state errors are warnings. Malformed open-position state is also
classified by the live/paper visibility checks above.
Durable broker recovery/accounting state is visible too:
`pending_order`, `pending_entry_recovery`, `risk_recovery_incident`,
`flatten_intent`, and `exit_accounting_intent` appear in runtime status and the
operator report. Healthcheck treats any of them on a live product as blocking
(paper as warning). Native futures stop IDs, trigger/quantity evidence, and
staleness metadata remain attached to each open-position detail.
Backup staleness defaults to twice the enabled dedicated-backup cadence, and verified
backup reports timestamped in the future also fail healthcheck instead of being
treated as fresh. Use `--max-backup-age-hours` for stricter external watchdogs. Use
`--ignore-job-overdue` only when an external scheduler intentionally owns job
cadence. Enabled scheduled jobs that are due but have never run are healthcheck
issues too; this catches a supervisor that starts but never launches its
background data, research, or maintenance jobs. Warning-level alerts such as
readiness warnings, stale/unsafe/failed research handoffs, promotion-review
warnings, enabled paper products waiting for an exported strategy artifact,
research cycles that found no exportable candidates, or a missing/stale required
testnet rehearsal for a paper product are also included in the JSON under
`warnings` without changing the exit code. A
missing/stale rehearsal required by a live product is a healthcheck issue and
exits nonzero. Blocking healthcheck issues also emit a cooldown-controlled
`autopilot healthcheck failed` alert unless you pass `--no-alert`; if that alert
write fails, the error is kept in `healthcheck_alert` so the original healthcheck
JSON still prints and writes. Config/report build failures print
`healthcheck_build_failed`; if the output file itself cannot be written, the
command still prints JSON with `healthcheck_output_write_failed` and exits
nonzero. It is intended for a cron, systemd timer, or external uptime monitor
that can alert when the main supervisor has stopped or a background workflow is
stuck.
`make backup` creates a small timestamped zip under `runtime/backups/` and writes
`runtime/backup_report.json`. It includes the autopilot config, approvals,
control/audit files, product state, trade logs, active strategy artifacts, and
runtime reports, including staged candidates, candidate-paper state/log/reviews,
and the non-executable incubation queue, with a per-file size cap so market
datasets and large research outputs are not pulled into recovery bundles.
Creation immediately verifies the manifest/archive; a verification failure exits
nonzero and skips retention pruning. Symlink sources are skipped instead of
followed, so a linked
runtime path cannot pull unintended target contents into the archive, and
symlink backup output paths are refused before writing. It does not include
`.env` or API credentials. A separate credential-free daily backup timer can
read the approval ledger only through a read-only mount while writing bounded
runtime backup outputs; generic research jobs cannot read it. It keeps the
latest 30 generated backup zips. Backup archives are always mode `0600`, and
staged restore directories/files are forced to `0700`/`0600` regardless of the
caller's umask (including overwrite restores).
Manifest entries distinguish optional files that have not been created yet from
existing recovery files that could not be archived. Missing optional state is
counted as `optional_missing_files` and documented without failing verification;
any existing configured recovery file skipped because it is too large, a
symlink, or not a regular file increments `critical_skipped_files`, fails backup
verification, and makes healthcheck fail.
`backup_enabled`, `backup_cadence_seconds`, and `backup_timeout_seconds` in the
autopilot config keep operator-report/health freshness expectations aligned with
the dedicated timer (24 hours by default).
`make backup-verify` validates the latest backup zip against its manifest; pass
`BACKUP=runtime/backups/name.zip` to verify a specific archive. Verification
fails closed for unsupported manifest versions, so restore will not accept a
backup written with an unknown archive schema.
`make backup-restore RESTORE_DIR=/tmp/trading-bot-restore` verifies a backup and
extracts it into a separate directory without overwriting existing files, so you
can rehearse recovery before copying state back into a live runtime directory.
Restore refuses archive path traversal, symlink restore roots, and symlink
escapes from the restore directory, including when overwrite mode is enabled.
The operator report summarizes the latest backup report and staged-candidate
paper status, including exact candidate digests, freshness, open-position count,
errors, drawdown halts, and activation readiness. `make healthcheck` fails if an
existing backup report says creation or verification failed, if an existing
recovery file was skipped, if the last verified backup is older than the
configured freshness window, or if an enabled candidate-paper status is stale,
invalid, or failed.
When `auto_report_enabled` is true, every runtime cycle refreshes
`runtime/operator_report.md/json` and `runtime/readiness_report.md/json` after
writing the latest status. Report-rendering errors are recorded in status but do
not stop trading supervision; partial runtime report output-write failures are
recorded with the affected path and successful outputs continue to refresh.
Scheduled job subprocess output is captured through temporary files with bounded
in-memory reads; runtime status keeps short stdout/stderr tails and truncation
metadata instead of loading unbounded logs into the supervisor. Truncation flags
and byte counts are also persisted in `runtime/job_state.json` and shown in the
operator report; `make healthcheck` surfaces them as warning-level watchdog
events.
`make readiness` and `make report` run the same renderers manually. If readiness
cannot load config, build its report, or write one of its output files, it exits
nonzero with a readiness-shaped failure report.
If the operator report CLI cannot load config or build the report, it exits
nonzero with a compact operator failure report; if an output write fails, it
prints the structured report JSON to stdout.
Readiness checks local server paths including alert log and cooldown-state
writability, config, artifacts, approval/preflight requirements for live
products, environment switches, 1m candle freshness, required indicator feature
columns, and runtime filesystem free space using
`min_runtime_free_bytes` from `config/autopilot.json`. Existing
approval ledgers must be readable JSON objects; malformed ledgers block
readiness because they would also block live execution. When
`src.regime` jobs are configured, readiness also warns if regime-tagged research
data is missing or skipped. It also warns when the strategy-framework smoke
report is missing or failing. The service-installer readiness check verifies the
script still has strict shell mode, config validation, the initial readiness gate,
healthcheck timer installation, and fail-closed unit override validation. The
operator report summarizes status heartbeat age, control state, market-data
freshness, indicator feature readiness, regime-data rows/counts,
strategy-framework smoke status, synthetic and real-data research status,
approval ledger health including revoked fingerprints and the latest approval
event, configured scheduled-job state, latest-cycle job output, and product
trade logs.

## Staged Live Candidate Activation

An autonomous research cycle cannot change a live product's active strategy.
For a product already configured `live`, research stages
`runtime/candidates/<product>.json`; the dedicated 45-second candidate-paper
timer runs that candidate through a separate paper bot. State is isolated by candidate
digest, while the trade log and review use stable paths:

- `runtime/candidates/<product>_paper_state_<digest-prefix>.json`
- `runtime/candidates/<product>_paper_trades.csv`
- `runtime/candidates/<product>_promotion_review.md`

Every unseen closed base-timeframe bar is consumed once from a durable cursor,
ordered by bar close/information availability with deterministic shorter-
timeframe ties. Only a fresh latest signal observed within two timer cadences
can enter promotable paper evidence, using a credential-free public quote and
its post-response observation timestamp. The partially elapsed entry bar is
excluded from OHLC exit checks. Bounded downtime replay still advances state
and manages positions, but historical next-open entries and any trade touched
by catch-up are explicitly quarantined from promotion. Only genuine-forward
rows matching the exact strategy fingerprint, candidate artifact digest,
observation schema, and current candidate-paper engine digest count. By default
every strategy needs at least 20 such trades spanning at least seven days,
positive sized return, and the configured drawdown/loss-streak limits. The
45-second timer adds public market data, CPU, and disk activity; the dedicated
unit retains configured CPU, memory, task, timeout, non-overlap-lock, and
bounded catch-up limits.
Check `candidate_activation_ready: true` before maintenance; activation
recomputes these gates and rejects stale or different-fingerprint evidence.

Pause the product and all jobs, verify flat/reconciled state, and stop both
long-running services before activating the exact reviewed digest:

```bash
make candidate-paper-once
jq '.products[] | select(.product == "active_income")' runtime/candidate_paper_status.json
make control ARGS="pause-product active_income --reason 'candidate activation review'"
make control ARGS="pause-jobs --reason 'candidate activation review'"
systemctl --user stop trading-bot-autopilot.service trading-bot-autopilot-jobs.service \
  trading-bot-candidate-paper.timer trading-bot-candidate-paper.service
CANDIDATE_DIGEST=$(jq -r '.products[] | select(.product == "active_income" and .candidate_activation_ready == true) | .candidate_digest' runtime/candidate_paper_status.json)
make activate-candidate PRODUCT=active_income CANDIDATE_DIGEST="$CANDIDATE_DIGEST" CONFIRM=1 OPERATOR=henrique
```

Activation requires the configured product to be in `live` mode and paused. It
also acquires the same exclusive lock as the supervisor, so stop the supervisor
for this short maintenance window; a held lock fails closed and concurrent
activations cannot race. Inside it, activation takes the shared control
transaction lock and re-reads pause/flatten intent, so a concurrent resume or
panic command cannot race the replacement. It fails closed if the control or
product state is malformed, if any open position,
`pending_order`, `flatten_intent`, `pending_entry_recovery`,
`exit_accounting_intent`, `risk_recovery_incident`, or flatten request remains,
if a relevant path is a symlink/invalid type, or if candidate identity and
product strategy policy do not match. `CANDIDATE_DIGEST` is checked against the
candidate loaded inside both locks; if research changed the file after review,
activation fails instead of switching to unreviewed bytes. The active artifact
replacement is atomic. Write-ahead intent and completion events bound to the old
and new digests are fsynced to the control audit.

Activation is not approval. Its result deliberately reports
`approval_granted: false` and `live_ready: false`; the new artifact digest does
not match old approval, preflight, or rehearsal evidence. A unique activation
record is included in the active artifact specifically so even historical
approval for identical candidate bytes cannot carry over. Keep the product
and jobs paused. Rebuild the promotion packet against the activated artifact and
candidate paper log; its printed approval command is bound to the new active
digest:

```bash
make promotion-review PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/candidates/active_income_paper_trades.csv
```

With the product configured `live` but still paused, run the connected production
preflight first. Final human approval pins its stable account/venue/risk-cap
manifest. Run the required testnet rehearsal using the separate testnet-preflight
output, then switch back to production credentials and refresh the production
preflight. An equivalent refresh keeps approval valid; account, venue, routing,
notional/slippage, leverage, or margin drift invalidates it. Start both services
while still paused, confirm reports and exchange state, resume the product, and
finally `resume-jobs`. A staged candidate remains available for audit.

## Live Approval Gate

Approve strategies only after reviewing the exported artifact and validation
report. Use a non-empty human operator identifier for `--approved-by` and
`--revoked-by`, include `--confirm-live` when approving, and include a non-empty
`--reason` when revoking. Blank approval actors are rejected and existing
blank-actor approvals do not satisfy the live gate. Obvious automation
identities such as `autopilot`, `system`, `cron`, and `github-actions[bot]` are
also rejected for approval/revocation actor fields and are treated as invalid if
found in an existing ledger. Revocation reasons are free-form required text:

```bash
ARTIFACT_DIGEST=$(jq -r '.artifact_digest' runtime/promotion_review.json)
make preflight PRODUCT=active_income
PREFLIGHT_DIGEST="sha256:$(sha256sum runtime/active_income_preflight_report.json | awk '{print $1}')"
python -m src.autopilot.approvals approve \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json \
  --expected-artifact-digest "$ARTIFACT_DIGEST" \
  --expected-preflight-digest "$PREFLIGHT_DIGEST" \
  --all \
  --approved-by henrique \
  --confirm-live \
  --notes "Reviewed holdout, paper stats, and risk limits"

python -m src.autopilot.approvals check \
  --config config/autopilot.json \
  --product active_income \
  --artifact outputs/active_strategies_flow.json

python -m src.autopilot.approvals list

python -m src.autopilot.approvals revoke \
  --fingerprint sha256:<fingerprint> \
  --revoked-by henrique \
  --reason "paper drawdown breached"
```

`list` shows the relevant actor for the current state: `approved_by` for active
approvals and `revoked_by` plus the revocation reason for revoked fingerprints.

Changing a strategy’s behavior changes its fingerprint and invalidates the old
approval, including entry/exit logic, risk, fees, market/symbol routing, and the
drift baseline used by the kill switch. The required reviewed artifact digest
also closes the review-to-approval replacement window. Strategy-level `leverage` and
`margin_mode`, when present, are also behavior fields; policy rejects leverage
above `1`, spot margin metadata, and non-isolated futures margin. Metrics-only
changes do not affect the fingerprint. Live checks also require the approval to
match the entire current artifact digest/path and every canonical `ProductConfig`
field, including execution mode, all state/evidence paths, gate switches and age
limits, starting equity, and regime settings, and then re-apply product policy.
Approval pins the stable production account/venue/risk-cap manifest from the
reviewed successful connected preflight. A timestamp-only refresh does not
require human reapproval, but operational manifest drift does. Approval also
records an execution-engine
digest over Python, pinned installed dependency versions, and execution-capable
source. A code, Python, or dependency change invalidates approval, preflight,
and rehearsal evidence until the operator reviews and repeats the sequence.
Thus an approval for one product, copied artifact, or now-policy-failing engine
cannot be reused silently. Revoked
fingerprints are retained in the ledger for auditability and block live
execution. Each approval entry must also carry the same fingerprint as its ledger
key; missing or mismatched embedded fingerprints are treated as invalid approval
evidence. A malformed approval ledger or malformed approval entry also blocks
live execution rather than being ignored.

## Live/Testnet Preflight

Before enabling a product, run a connected preflight. It never places orders.
Production preflight requires `TRADING_LIVE=1` and `EXCHANGE_TESTNET=0`; the
separate sandbox preflight uses `REQUIRE_TESTNET=1` and
`EXCHANGE_TESTNET=1`. It checks product config, artifact and engine identity,
environment, broker construction, and read-only exchange access. Because it is
read-only, production preflight intentionally precedes final human approval.
Config validation and normal live execution both require `require_preflight=true`,
a configured `preflight_report`, and a fresh passing report that points at the
current product and strategy artifact. The saved report must include successful
check entries for config, strategy artifact, policy, exchange
environment, broker construction, read-only connectivity, and the product-specific
position check (`broker_position_flat` for active-income futures or
`broker_spot_position_non_negative` for BTC accumulation spot).
Futures preflight additionally proves one-way mode, native protective-stop
capability, and empty regular/conditional open-order inventories.
The live gate also validates the saved read-only connectivity evidence itself:
price and balance must be finite and positive, position quantity must be finite,
position average price must be finite and non-negative, and active-income
connectivity evidence must show a flat futures position.
Preflight reports are bound to the artifact/fingerprints, execution-engine
digest, product identity, and a non-secret account fingerprint derived from API
key plus venue/market/testnet routing. At live entry, exchange, market, account,
testnet flag, quote, notional/slippage caps, leverage, and margin mode must
exactly match the saved production preflight. Changing any of them requires a
fresh connected preflight. Active-income live execution also always requires
`require_testnet_rehearsal=true` and a recent successful testnet rehearsal.
Malformed preflight reports block readiness and live execution instead of being
treated as missing approval evidence.
If the preflight CLI cannot load config, build the report, or write the output
file, it exits nonzero with structured JSON so automation can record the failed
gate instead of losing the reason in a traceback.
Separately, every live cycle still checks `TRADING_LIVE=1`, exchange
credentials, a positive `MAX_NOTIONAL_USD`, a positive
`MAX_FILL_SLIPPAGE_BPS`, and `MAX_FUTURES_LEVERAGE=1` for the active-income
futures product before constructing a broker.

```bash
make preflight PRODUCT=active_income

python -m src.autopilot.preflight \
  --config config/autopilot.json \
  --product btc_accumulation \
  --assume-live \
  --connect \
  --output runtime/btc_accumulation_preflight_report.json
```

The Makefile preflight target always passes `--connect`; it performs read-only
ticker, balance, position, position-mode, native-stop capability, and open-order
inventory checks. For
`active_income` futures, a connected preflight also requires the broker position
to be flat before live/testnet enablement; use the flatten control or reconcile
manually if it is not. BTC accumulation spot preflight allows existing BTC
holdings because the product starts from the BTC base stack. Use testnet keys
first, with `TRADING_LIVE=1`, `EXCHANGE_TESTNET=1`, and a tiny
`MAX_NOTIONAL_USD`. Keep `MAX_FILL_SLIPPAGE_BPS` tight enough to catch bad
fills, and keep `MAX_FUTURES_LEVERAGE=1`; active-income autopilot gates reject
higher leverage. `REQUIRE_TESTNET=1` writes a separate
`runtime/<product>_testnet_preflight_report.json` and cannot overwrite the
approved production report.

After an `active_income` artifact has been explicitly approved and its
`REQUIRE_TESTNET=1` sandbox preflight is green, the next exchange-facing step is
a tiny futures testnet rehearsal:

```bash
make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100
make testnet-status
```

This command is intentionally separate from preflight because it places testnet
orders. It requires `TRADING_LIVE=1`, `EXCHANGE_TESTNET=1`, Binance USDT-M
futures routing, approval, successful preflight checks, a flat starting futures
position, and `NOTIONAL_USD <= MAX_NOTIONAL_USD`. It buys a tiny BTCUSDT futures
quantity and immediately closes it with the broker close path, then writes
`runtime/testnet_rehearsal_report.json`. `make testnet-status` reads that saved
artifact without placing orders and prints the same structural status used by the
live gate. The operator report summarizes the file as `Testnet rehearsal`,
including missing, malformed, failed, stale, or successful status. A usable
report must have a finite timestamp, positive notional, positive order quantity,
product-symbol-matched buy entry fill, product-symbol-matched sell close fill,
entry/close quantities matching the order quantity, testnet routing, and a flat
final position. If an entry fills but the normal close/readback step fails, the command
attempts one best-effort cleanup close and records the recovery result, but the
rehearsal still fails and must be rerun cleanly before live execution. Rehearsal
CLI failures are reported as structured JSON, including
config-load or output-write failures. Missing, malformed, stale, or failed
rehearsal warnings include the next preflight/rehearsal commands and required
testnet environment checklist, plus the read-only `make testnet-status`
inspection command. The default active-income config also
requires a recent successful
testnet rehearsal report before any live cycle can construct a broker. The live
gate compares the rehearsal's embedded preflight product metadata, artifact path,
fingerprints, engine/account identity, and recorded testnet exchange with the current product,
strategy artifact, and Binance USDT-M policy, so a changed strategy or routing
config must be approved, preflighted, and rehearsed again before it can reach live
execution. Malformed embedded preflight payloads also invalidate the rehearsal.
After the rehearsal, switch to production credentials and
`EXCHANGE_TESTNET=0`, then run `make preflight PRODUCT=active_income` again. The
production report, not the sandbox report, is the saved live-entry preflight.

## Promotion Review

The system can prepare review packets, but it never approves strategies by
itself. A promotion review combines the strategy artifact, validation metrics,
paper-trade results, paper drawdown/loss-streak evidence, approval status, and
the exact approval command:

```bash
make promotion-review \
  PRODUCT=active_income \
  ARTIFACT=outputs/active_strategies_flow.json \
  TRADE_LOG=runtime/active_income_trades.csv
```

This writes `runtime/promotion_review.json` and `runtime/promotion_review.md`.
Only run the printed approval command after reviewing the packet. When a product
is supplied or inferred from `config/autopilot.json`, the packet also applies
the same product-aware strategy policy as runtime and suppresses approval
commands for policy-failing artifacts. It also withholds approval commands when
exact-fingerprint paper evidence fails the default minimum of 20 valid trades
spanning seven days, positive sized return, drawdown, consecutive-loss, or
trade-log validity gates. Rows from a previous strategy with the same ID do not
count. Invalid review
thresholds, such as negative paper-return requirements or zero required paper
trades, are reported as threshold failures and also suppress approval commands.
If an
already-approved strategy later fails
those paper-review thresholds, the packet marks it `approved_review_failed` so
you can revoke the fingerprint before any further live use. If the approval
ledger is malformed, the packet records the ledger error and emits no approval
commands until the ledger is repaired.
Operator reports and healthcheck warnings treat stale, missing-timestamp, and
future-dated promotion review packets as not fresh, so approval review evidence
cannot be made to look current by a bad clock.

Per-product promotion-review jobs are enabled in `config/autopilot.json` so the
operator report stays current as paper evidence accumulates. The manual target is
still useful for ad hoc review packets.

## Offline Rehearsal

Run a deterministic, no-network rehearsal of the full safety workflow:

```bash
make rehearse
```

It creates two distinct bounded generations around synthetic development
feedback, exercises the typed validator across both products, creates synthetic
strategy and paper-trade artifacts under `runtime/rehearsal`, produces promotion
reviews, approves only a local rehearsal ledger, and runs preflight through a
fake read-only broker. This does not touch real approvals, credentials, market
data, or exchange APIs. `make readiness` reports a
warning until this offline workflow rehearsal has produced a passing
`runtime/rehearsal/rehearsal_summary.json` for both products.

## Research Workflow

Run the autonomous research machinery with outputs kept out of git:

```bash
make research-factory-validate
make research-history-plan
make research-smoke
make research-generate
make research-cycle
# or: make research-once
```

Validated exports explicitly stamp `paper_trade_allowed: true`,
`live_allowed: true`, and `promotion_eligible: true`; bootstrap probes and
research-only generated/incubation artifacts stamp the opposite and are rejected
by promotion/live gates. Live review fails closed if any of the three
eligibility flags is missing instead of explicitly `true`. The generated batch
itself is non-executable, cannot enter paper or promotion directly, and is
strictly revalidated by the research cycle.

For hypothesis-driven research, use `docs/RESEARCH_WORKFLOW.md`.
The autonomous path is `make research-once`; scheduled operation runs the same
factory and evaluation stages as separate bounded jobs. It uses staged validation
and export gates and skips exact behavior/data/protocol repetitions. Changes in spot/futures
freshness, row counts, timestamps, or missing-data reasons trigger a fresh cycle
so recovered or degraded data feeds are not hidden by an old timestamp.

## Tests And Quality

```bash
make test
make lint
make lint-autopilot
```

Focused smoke checks:

```bash
python -m pytest tests/test_autopilot_approvals.py tests/test_autopilot_runtime.py
python -m pytest tests/test_execution.py tests/test_run_bot.py
```

## Current Safety Posture

Paper trading and research orchestration are the default runtime path. Approved
`active_income` live mode injects a futures ccxt broker into `src.run_bot`.
Approved `btc_accumulation` live mode injects a spot ccxt broker, so it cannot
route through futures or leverage. `src.run_bot` refuses a live broker unless
the autopilot path has already passed approval, preflight, and testnet rehearsal
gates. The bot also uses each product's configured
market for closed-candle data, so BTC accumulation evaluates spot candles while
active income evaluates futures candles.

Broker-routed execution sizes orders from broker balances, records broker fill
details, and reconciles local open positions against broker positions before
managing exits. BTC spot step-aside exits reinvest the original sell proceeds
into the buyback, so profitable sell-lower/buy-lower cycles can increase BTC
instead of leaving gains idle in USDT. The budget is the exact observed increase
in free USDT across the sell—not an assumed fill notional—and must pass a tight
plausibility check. Use a dedicated spot account with no concurrent transfers or
manual trades. Connected evidence is bound to a non-secret fingerprint of the
API key, venue, market, and testnet routing. Live positions and durable
recovery/flatten intents retain it; management refuses a different configured
account. The ccxt adapter still requires
`TRADING_LIVE=1` and enforces `MAX_NOTIONAL_USD` plus
`MAX_FILL_SLIPPAGE_BPS` for entries and position increases. Futures reduce-only
closes may exceed `MAX_NOTIONAL_USD` so an emergency flatten can reduce existing
exposure. For futures, it also sets `MAX_FUTURES_LEVERAGE` before opening
entries and refuses to open if the exchange adapter cannot set leverage.

After market-data/feature evaluation, every live entry rechecks current control
(including panic/pause/flatten), approval, preflight, and environment immediately
before persisting its order intent. The futures broker then fetches the signed
position again immediately before `create_order` and refuses a non-flat account,
so external exposure created during signal computation cannot be stacked or
silently offset.

Live futures positions persist the flat USDT balance immediately before entry
and read it again after a proved-flat normal or native-stop exit. Risk accounting
uses the worse of that observed account return and fill-price/commission PnL, so
funding or unexplained debits tighten `daily_pnl` and cooldown controls while
positive credits never increase modeled performance. Run active income in a
dedicated subaccount with no manual trades or transfers during exposure. Daily
loss is realized at exit rather than continuously marked while a trade is open;
the daily tracker conservatively keeps the worse of additive and compounded
cumulative closed-trade returns.
After a proved-flat normal/native-stop exit, the bot first persists a keyed
`exit_accounting_intent`. Trade-log insertion is idempotent and atomic, and a
restart resumes the accounting/state commit without submitting another close.
An ambiguous futures entry is instead reduced using a deterministic recovery
order; an invalid/missing native stop or signed broker/local quantity mismatch
closes the full actual broker quantity. The durable recovery incident remains
latched after flat proof for human reconciliation and blocks new entries.
Emergency control flatten is a recovery path and currently does not book its
fill/funding delta into those counters; keep the product paused and reconcile the
exchange balance and local trade ledger before resuming. A successful flatten
request clears automatically; the pause remains.
