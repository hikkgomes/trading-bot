# PostgreSQL Platform Deployment

This is the only production deployment path. PostgreSQL owns lifecycle state,
strategy artefacts, assignments, approvals, orders, fills, positions, risk,
accounting, controls, and reports. Frozen Parquet is immutable research
data. Legacy SQLite data is read only for one-time migration.

## 1. Install the host

Use a Debian or Ubuntu Linux host with PostgreSQL 17 or newer, ACL support, and Python 3.11
or later.
Quote writes use PostgreSQL 17 transaction deadlines. Use the official PostgreSQL
APT repository if the distribution provides an older major version.

```bash
sudo apt-get update
sudo apt-get install -y acl bubblewrap git jq make postgresql-17 postgresql-client-17 python3 python3-venv rsync restic rclone
sudo systemctl enable --now postgresql
git clone "$REPOSITORY_URL" /home/alfred/trading-bot
cd /home/alfred/trading-bot
python3 -m venv .venv-runtime
python3 -m venv .venv-research
python3 -m venv .venv-agent
.venv-runtime/bin/pip install -r requirements-runtime.txt
.venv-research/bin/pip install -r requirements-research-linux.txt -r requirements-dev.txt
.venv-agent/bin/pip install -r requirements-agent.txt
ln -s .venv-runtime .venv
make platform-validate
make platform-install-dry-run
sudo REPO=/home/alfred/trading-bot NODE=linux-optiplex bash scripts/install_platform_services.sh
```

The installer creates separate service users, private environment files,
process-aligned units, exact writable paths, and backup timers. It disables
known legacy trading and autopilot units. It does not enable live trading.
Disabled legacy unit files remain available for recovery until the new services
pass the permission checks, restart test and paper-mode soak. After that
handover, archive the retired unit files and drop-ins outside the checkout,
remove those exact retired files from `/etc/systemd/system`, and run
`sudo systemctl daemon-reload`. Do not remove them before replacement startup
has been verified.

## 2. Create and migrate PostgreSQL

Do not use `trading_platform` as a database login. The migration service runs
with the owner role, never against `trading_platform` as a worker login. Do not
use SQLite for the active platform. Schema changes must run through the
migration service.

```bash
sudo -u postgres createuser --createdb --createrole --pwprompt trading_platform_migrator
sudo -u postgres createdb -O trading_platform_migrator trading_platform
sudo -u postgres createdb -O trading_platform_migrator trading_platform_smoke
sudo systemctl start trading-platform-migration.service
```

Set `/etc/trading-platform/migration.env` before starting the service:

```text
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_platform_migrator:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0
EXCHANGE_TESTNET=1
```

The migration service applies Alembic revisions and runs the idempotent
PostgreSQL bootstrap. Verify migrations with:

```bash
make db-migration-check
```

For an upgrade of an existing database, run `make platform-upgrade` after the
backup credentials are configured. It creates a PostgreSQL backup, then
quarantines legacy pending and dead research jobs. The repaired scheduler does
not import that queue state.

The upgrade first refuses enabled or active legacy user services, before
stopping system services or modifying the database. Back up their files and
state, then retire those exact user units before retrying. To run that check
alone, use `sudo bash scripts/install_platform_services.sh --check-authorities`.

## 3. Configure isolated environments

Create root-owned mode `0640` files under `/etc/trading-platform/`.

```text
# runtime.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_runtime_service:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0
EXCHANGE_TESTNET=1
TRADING_CONTROL_TOKEN=<RANDOM_64_HEX_CONTROL_TOKEN>
TRADING_PLATFORM_HEALTHCHECK_URL=<PRIVATE_OFF_HOST_HEARTBEAT_URL>

# research.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_research_worker:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0

# agent.env
TRADING_LIVE=0

# evaluator.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_evaluation_worker:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0

# backup.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_backup_service:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
```

Only the runtime environment may contain exchange credentials. Research and
agent processes cannot submit exchange orders. Never include environment-file
contents in an audit archive.

Each database login must inherit only its service role. In particular, research
must not inherit `trading_runtime`; startup rejects mixed service roles. The
evaluator uses `trading_evaluator` to read protected samples and publish sealed
evidence. The adaptive research worker cannot read those samples or write sealed
artefacts. The agent sandbox has no database credential.

Frozen evaluation and execution use the same `.venv-runtime` interpreter and
lock. Their OS users and database roles remain separate. Run inference calibration,
policy approval and live-authority commands with that installed interpreter.
Adaptive searches and code checks retain the separate research environment.

Create missing service logins with `sudo -u postgres createuser --pwprompt NAME`,
then apply the following as the PostgreSQL administrator. For an existing host,
remove any other inherited service or owner roles before starting workers.

```sql
GRANT trading_runtime TO trading_runtime_service;
REVOKE trading_runtime, trading_agent, trading_evaluator, trading_platform_owner
    FROM trading_research_worker;
GRANT trading_research TO trading_research_worker;
GRANT trading_evaluator TO trading_evaluation_worker;
ALTER ROLE trading_agent_worker NOLOGIN;
REVOKE trading_runtime FROM trading_agent_worker, trading_backup_service;
GRANT pg_read_all_data TO trading_backup_service;
ALTER ROLE trading_backup_service BYPASSRLS;
```

The backup login has read access to all data, including protected evidence, but
no trading writes. `BYPASSRLS` is needed because `pg_dump` refuses a partial
row-security-filtered backup. Keep its credential out of research and agent
environments; only the PostgreSQL administrator may grant this attribute.

Configure an independent monitor to accept heartbeat POSTs and notify you when
they stop. Put its private URL only in `runtime.env`. The scheduler sends a ping
only while all critical runtime, gateway and data-writer heartbeats are healthy
and at most 60 seconds old. Each request has a two-second timeout; failed pings
are not replayed later. The scheduler heartbeat records delivery status. Choose
a missed-ping deadline longer than the scheduler interval and freshness window,
then prove an alert arrives by stopping pings during the paper-mode soak. This
monitor receives no database, exchange or control token and has no order authority.

Use separate API keys for the spot and futures production accounts. Connected
testnet rehearsals must use dedicated testnet keys and a testnet account
configuration. Testnet keys must never be copied into production account files.

## 4. Start the platform

The code remains modular, but units follow privilege and failure domains.

```bash
sudo systemctl start trading-platform-runtime.service
sudo systemctl start trading-platform-market-gateway.service
sudo systemctl start trading-platform-data-writer.service
sudo systemctl start trading-platform-operations.service
sudo systemctl start trading-platform-research.service
sudo systemctl start trading-platform-evaluator.service
sudo systemctl start trading-platform-agent.service
sudo systemctl start trading-platform-control.service
sudo systemctl enable --now trading-platform-backup-postgresql.timer
sudo systemctl enable --now trading-platform-backup-offsite.timer
sudo systemctl enable --now trading-platform-backup-verify.timer
```

The market gateway owns persistent exchange streams and the durable spool. The
data-writer process owns raw, bar, and feature persistence. The runtime process
owns strategy evaluation, portfolio, risk, execution, paper execution,
accounting, and reconciliation. The operations process owns scheduling,
universe refresh, promotion, and forward summaries. The research process owns
adaptive generation and diagnostic research. The evaluator owns dataset sealing
and frozen staged evaluation. The control API and agent sandbox use
separate processes. `migration-service` is a one-shot process.

Check the process and timer state:

```bash
sudo systemctl list-units --all 'trading-platform*' --no-pager
sudo systemctl list-timers --all 'trading-platform*' --no-pager
sudo systemctl --failed --no-pager
```

## 5. Verify paper operation

Queue consumers drain completed jobs promptly and poll idle live inputs at most
once per second. Service heartbeats remain bounded to the configured interval;
health changes publish immediately. Historical downloads run separately from
live batch writes, with renewable leases and a hard research job deadline.

The default feed captures closed candles, current funding marks and book quotes
for the configured scope. It coalesces quotes to the latest exchange observation
per flush. Full trade and depth replay collection is not enabled by default on
this host. The capture policy is recorded on sampled quotes.

Runtime processes share the Binance IP request budget in
`runtime/exchange-limits`. Public data work leaves capacity for execution and
recovery. HTTP 418/429 backoff applies across processes. Do not place this
directory on an unrelated or non-shared filesystem for one trading host.

Quote depth is USDT notional, with separate bid and ask values. Admission uses
the execution side and exchange observation time. A newly received old quote
cannot refresh market authority.

Both products must stay in paper mode until the complete evidence chain has
been reviewed.

```bash
make platform-validate
make platform-readiness
make platform-smoke
make platform-permissions-test
curl --fail http://127.0.0.1:8088/status
curl --fail http://127.0.0.1:9108/metrics
```

Readiness must show canonical dataset roles, current account authority, no
unknown exposure, no unresolved recovery, and no active live assignment.

## 6. Legacy import and archival boundary

Legacy SQLite memory, JSONL logs, and file artefacts may be imported once with
the explicit migration command. The importer stores immutable provenance and
is idempotent. It must not be scheduled and it must not write the old runtime
authority.

```bash
make sqlite-import SOURCE=/path/to/legacy/experiment_memory.sqlite3
```

After import, archive the source outside the active repository. Do not enable
`src.autopilot`, `src/run_bot.py`, `runtime/approvals.json`, or
`outputs/active_strategies*.json` as production authorities. The normal
operator interface uses PostgreSQL commands only.

## 7. Live authority sequence

Live mode is disabled in checked-in configuration. Enable it only after a
reviewed artefact has completed:

```text
screening -> development -> robustness -> protected holdout
-> forward paper -> accepted forward summary -> live_ready
-> human approval -> fresh preflight -> live_canary
```

The live authority command checks the exact product, account, instrument,
artefact, behaviour hash, forward summary, approval, preflight, configuration,
engine identity, account fingerprint, capital cap, and current risk state.
BTC accumulation accepts only Binance BTCUSDT spot artefacts. Active income
uses only the selected instrument from the current eligible universe. No agent,
scheduler, or automatic promotion can grant live authority.

Use the command only with an explicit human actor and confirmation:

```bash
make platform-live-authority ARGS="inspect --product active_income --artefact-hash <ARTEFACT_HASH> --instrument-id <INSTRUMENT_ID> --sleeve-id directional"
```

After the connected canary has fresh accepted evidence, promote that exact
assignment with a separate human confirmation:

```bash
make platform-live-authority ARGS="canary-to-live --product active_income --artefact-hash <ARTEFACT_HASH> --instrument-id <INSTRUMENT_ID> --sleeve-id directional --expected-preflight-id <PREFLIGHT_ID> --expected-approval-id <APPROVAL_ID> --canary-evidence-id <CANARY_REPORT_ID> --capital-limit <CAPITAL> --risk-budget <RISK_BUDGET> --assigned-by <HUMAN_OPERATOR> --confirm"
```

Do not run connected testnet orders automatically. Use a dedicated flat testnet
account and an explicit confirmation:

```bash
make platform-testnet-connected PRODUCT=active_income NOTIONAL_USD=10 CONFIRM=1
```

The connected report must match the reviewed commit, artefact, configuration,
account fingerprint, and execution-engine identity before any canary decision.

## 8. Backups and recovery

Keep hourly backups for `hourly_retention_hours` (6 in the supplied configuration), then the newest
backup of each type per UTC day for `retention_days` (7). The newest verified
backup of each type is always retained, even when stale. A replacement must
verify before older recovery points are removed. This preserves the recovery
window without retaining hundreds of full database copies on the server.

Archive creation streams tar members; verification streams headers and contents
without extracting a second temporary data store. A restore publishes its
private destination only after every member and the gzip checksum validate.
Backup maintenance is serialised. Retention runs before creation and preserves the newest verified recovery point of each type. Archives are built on the backup filesystem and moved into their published bundle without another full copy. Hidden incomplete builds are excluded from verification. Each committed file is streamed from one open inode; writer temporary files and locks are excluded.

Weekly verification restores only the newest PostgreSQL dump into a temporary
local cluster with TCP disabled and a private Unix socket. It checks the schema
revision and core table counts, then writes a separate hash-bound restore
receipt. The rehearsal does not connect to or change the production database.

Daily offsite backups use restic through a private rclone Google Drive remote.
They capture a fresh PostgreSQL dump, all canonical data and public configuration
in one encrypted incremental snapshot. The database capture is pinned with a
hard link, so normal hourly retention cannot remove an in-progress input.
The old daily full-tar timer is retired; manual local archives remain available.

Install the maintained `restic` and `rclone` packages. Configure only `drive.file`
on the dedicated Google OAuth client. Store its configuration at
`/etc/trading-platform/offsite/rclone.conf` and the restic password at
`/etc/trading-platform/offsite/restic-password`, readable only by `trading-runtime`.
The directory must allow that user to save refreshed OAuth tokens. Initialise
the repository once; the scheduled job never silently creates a replacement.
The root-owned `/etc/trading-platform/offsite.env` supplies `RCLONE_CONFIG`,
`RESTIC_PASSWORD_FILE`, `RESTIC_REPOSITORY` and `RESTIC_CACHE_DIR`.
Keep the encryption password outside the server. The installed recovery copy
is in Henrique's Mac login Keychain, service `Trading Platform encrypted backups`,
account `hikkgomes/trading-bot/optiplex`.

Every offsite run checks the repository and restores the captured database dump
and a canonical Parquet file into a private temporary directory. Both hashes
must match before retention runs. The first run and each subsequent week read
all encrypted repository data; intervening runs rotate through seven subsets.
`runtime/backups/offsite-latest.json` records the verified snapshot and checks.
Readiness rejects a missing, corrupt, future-dated or stale offsite receipt.
The offsite recovery point is daily; hourly local database backups remain.

Create and verify local backups, or trigger the encrypted offsite service:

```bash
make platform-backup-postgresql
make platform-backup-parquet
make platform-backup-verify
sudo systemctl start trading-platform-backup-offsite.service
```

Restore into a new target. Never restore over the active database or data
directory.

```bash
make platform-backup-restore-postgresql BACKUP_ID=<BACKUP_ID> TARGET_DATABASE_URL=<NEW_DATABASE_URL>
make platform-backup-restore-parquet BACKUP_ID=<BACKUP_ID> DESTINATION=/tmp/platform-restore
```

After a restore, run migration checks, readiness, and the platform smoke. A
failed backup or restore verification is an operational failure and must stop
live enablement.

## 9. Controls and incidents

The control API persists global and product control events. Supported actions
include block-new-risk, strategy suspension, management-only operation,
cancel-entry-orders, emergency reduction, emergency flatten, and explicit
resume. Reduce-only execution, reconciliation, and account authority checks
remain active during a pause.

On an incident:

1. Block new risk through the control API.
2. Inspect unresolved recovery, protective stops, account authority, and open orders.
3. Use product or global flatten when required.
4. Wait for authenticated reconciliation to prove the resulting account state.
5. Resolve the incident and resume only with explicit operator confirmation.

The recovery worker queries ambiguous orders by exchange and client identity,
adopts known state, applies missing fills, cancels confirmed open orders, and
places deterministic emergency reductions. Unresolved state remains blocked
and visible in the operator report.

If the host or database is unavailable, use an independent device and the
exchange's authenticated interface. Do not wait for the local control API:

1. Verify the exact account, sub-account and production/testnet environment.
2. Disable trading permission for the dedicated bot API key, or revoke that key,
   so a recovering host cannot submit another entry.
3. Inspect actual positions and both ordinary and conditional orders. Cancel
   pending entries. Keep protective orders until the corresponding exposure
   is closed or replacement protection is confirmed.
4. For Active Income, close the actual futures exposure with reduce-only orders.
   Verify fills and remaining quantity after each attempt; an accepted order is
   not a flat position. Resolve rejected orders or residual quantities through
   the exchange interface or exchange support. Do not switch position mode.
5. Confirm that positions are flat and no ordinary or conditional orders remain.
   BTC accumulation is a spot holding: do not liquidate its protected core as
   part of a futures recovery action.
6. Save fills, funding, fees and account statements privately. Keep the bot key
   disabled until restored records reconcile and the operator explicitly
   authorises recovery. New credentials or changed code require fresh authority.

The exchange order contracts are documented in the
[official Binance USD-M interface examples](https://github.com/binance/binance-cli/blob/master/examples/derivatives-trading-usds-futures.md).

## 10. Quality gate

Run the repository gate before deployment:

```bash
make platform-ci
```

This runs formatting, Ruff including C90 complexity, platform type checking,
database migration checks, PostgreSQL smoke, rehearsal tests, and the full
test suite. Connected exchange tests remain a separate manual deployment gate.

### Bounded market storage

High-rate market observations use a 15-minute PostgreSQL buffer. Any observation
used in a durable decision or required feature job is verified and copied to the
immutable risk evidence store before its reference is published. Expiry removes
only unpinned buffer rows, in bounded batches. Closed candles remain permanent.

Live candle, feature and native funding Parquet stores publish a complete daily
`current.parquet` through an atomic replacement under a partition writer lock.
Readers use one published generation and hash the same opened file they read.
Frozen datasets, backtests, models and artefacts remain immutable. Automatic
whole-tree compaction is removed; it could block capture and exhaust memory.
The original source files are retained during deployment for recovery.

New public market batch jobs carry an explicit seven-day retention tag. A
bounded database function removes completed tagged transport jobs and their
attempts after seven days. It preserves financial, research, user-stream,
quarantined and untagged historical jobs. Runtime has no direct job-delete
permission. Operator queue totals describe the retained history.
