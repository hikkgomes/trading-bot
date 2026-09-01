# PostgreSQL Platform Deployment

This is the only production deployment path. PostgreSQL owns lifecycle state,
strategy artefacts, assignments, approvals, orders, fills, positions, risk,
accounting, controls, and reports. Parquet is immutable market and research
data. The legacy autopilot is an offline migration source only.

## 1. Install the host

Use a Debian or Ubuntu Linux host with PostgreSQL, ACL support, and Python 3.11
or later.

```bash
sudo apt-get update
sudo apt-get install -y acl git jq make postgresql postgresql-client python3 python3-venv rsync
sudo systemctl enable --now postgresql
git clone "$REPOSITORY_URL" /home/alfred/trading-bot
cd /home/alfred/trading-bot
python3 -m venv .venv-runtime
python3 -m venv .venv-research
python3 -m venv .venv-agent
.venv-runtime/bin/pip install -r requirements-runtime.txt
.venv-research/bin/pip install -r requirements-research-linux.txt -r requirements-dev.txt
.venv-agent/bin/pip install -r requirements-agent.txt
make platform-validate
make platform-install-dry-run
sudo REPO=/home/alfred/trading-bot NODE=linux-optiplex bash scripts/install_platform_services.sh
```

The installer creates separate service users, private environment files,
process-aligned units, exact writable paths, and backup timers. It disables
known legacy trading and autopilot units. It does not enable live trading.

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

## 3. Configure isolated environments

Create root-owned mode `0640` files under `/etc/trading-platform/`.

```text
# runtime.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_runtime_service:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0
EXCHANGE_TESTNET=1
TRADING_CONTROL_TOKEN=<RANDOM_64_HEX_CONTROL_TOKEN>

# research.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_research_worker:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0

# agent.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_agent_worker:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
TRADING_LIVE=0

# backup.env
TRADING_PLATFORM_DATABASE_URL=postgresql+psycopg://trading_backup_service:<URL_ENCODED_PASSWORD>@127.0.0.1:5432/trading_platform?sslmode=require
```

Only the runtime environment may contain exchange credentials. Research and
agent processes cannot submit exchange orders. Never include environment-file
contents in an audit archive.

Use separate API keys for the spot and futures production accounts. Connected
testnet rehearsals must use dedicated testnet keys and a testnet account
configuration. Testnet keys must never be copied into production account files.

## 4. Start the platform

The code remains modular, but units follow privilege and failure domains.

```bash
sudo systemctl start trading-platform-runtime.service
sudo systemctl start trading-platform-research.service
sudo systemctl start trading-platform-agent.service
sudo systemctl start trading-platform-control.service
sudo systemctl enable --now trading-platform-backup-postgresql.timer
sudo systemctl enable --now trading-platform-backup-parquet.timer
sudo systemctl enable --now trading-platform-backup-verify.timer
```

The runtime process owns market data, features, strategy evaluation, portfolio,
risk, execution, paper execution, accounting, and reconciliation. The
research process owns scheduling, staged evaluation, forward summaries,
promotion, and reporting. The control API and agent sandbox use separate
processes. `migration-service` is a one-shot process.

Check the process and timer state:

```bash
sudo systemctl list-units --all 'trading-platform*' --no-pager
sudo systemctl list-timers --all 'trading-platform*' --no-pager
sudo systemctl --failed --no-pager
```

## 5. Verify paper operation

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
uses only the explicitly configured live symbol scope. No agent, scheduler, or
automatic promotion can grant live authority.

Use the command only with an explicit human actor and confirmation:

```bash
make platform-live-authority ARGS="inspect --product active_income --artefact <ARTEFACT_HASH> --instrument <INSTRUMENT_ID> --sleeve directional"
```

Do not run connected testnet orders automatically. Use a dedicated flat testnet
account and an explicit confirmation:

```bash
make platform-testnet-connected PRODUCT=active_income NOTIONAL_USD=10 CONFIRM=1
```

The connected report must match the reviewed commit, artefact, configuration,
account fingerprint, and execution-engine identity before any canary decision.

## 8. Backups and recovery

Create and verify both PostgreSQL and Parquet backups:

```bash
make platform-backup-postgresql
make platform-backup-parquet
make platform-backup-verify
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

## 10. Quality gate

Run the repository gate before deployment:

```bash
make platform-ci
```

This runs formatting, Ruff including C90 complexity, platform type checking,
database migration checks, PostgreSQL smoke, rehearsal tests, and the full
test suite. Connected exchange tests remain a separate manual deployment gate.
