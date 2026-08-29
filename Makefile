# Trading-bot developer tasks.
# Heavy targets (data, search, train) are guarded behind CONFIRM=1 because they
# cost hours of compute — see CLAUDE.md "Important Constraints".

VENV ?= .venv
PY ?= $(VENV)/bin/python
BOOTSTRAP_PY ?= python3
PLATFORM_TYPECHECK_FILES = \
	src/agents/openclaw_bridge.py \
	src/data/database.py src/data/migrate.py \
	src/data/parquet_store.py src/data/snapshots.py src/data/universe.py \
	src/domain/_codec.py src/domain/market_events.py src/domain/strategies.py \
	src/observability/reports.py \
	src/research/artefacts.py src/research/canonical.py src/research/catalogue.py \
	src/research/coordinator.py src/research/evaluation.py src/research/sqlite_import.py \
	src/research/store.py \
	src/services/agent_worker.py src/services/backup_service.py src/services/backups.py \
	src/services/config.py src/services/data_writer.py src/services/live_execution.py \
	src/services/account_reconciliation.py src/services/platform_bootstrap.py \
	src/services/forward_observation.py src/services/paper_diagnostic.py \
	src/services/platform_live_authority.py \
	src/services/platform_testnet_connected.py src/services/platform_testnet_rehearsal.py \
	src/services/scheduler.py \
	src/services/promotion.py src/services/readiness.py src/services/research_jobs.py \
	src/services/supervisor.py \
	src/strategies/base.py src/strategies/manifest.py
PREFLIGHT_OUTPUT ?= $(if $(REQUIRE_TESTNET),runtime/$(PRODUCT)_testnet_preflight_report.json,$(if $(PRODUCT),runtime/$(PRODUCT)_preflight_report.json,runtime/preflight_report.json))
BACKUP ?= $(shell ls -t runtime/backups/autopilot_state_*.zip 2>/dev/null | head -1)
RESTORE_DIR ?= runtime/restore_rehearsal

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: setup
setup:  ## Create venv and install research + dev deps
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt

.PHONY: setup-bot
setup-bot:  ## Install the minimal execution-only deps (for the server)
	$(PY) -m pip install -r requirements-bot.txt

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run the full test suite
	$(PY) -m pytest

.PHONY: test-fast
test-fast:  ## Run tests excluding ones marked slow
	$(PY) -m pytest -m "not slow"

.PHONY: lint
lint:  ## Lint with ruff
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

.PHONY: typecheck-platform
typecheck-platform:  ## Type-check the PostgreSQL-authoritative platform boundary
	$(PY) -m mypy --ignore-missing-imports --follow-imports=skip $(PLATFORM_TYPECHECK_FILES)

.PHONY: lint-autopilot
lint-autopilot:  ## Lint autonomous runtime + execution surface
	$(PY) -m ruff check src/autopilot src/run_bot.py src/execution \
		tests/test_autopilot_*.py tests/test_run_bot.py tests/test_execution.py

.PHONY: format
format:  ## Auto-format + autofix with ruff
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

.PHONY: clean
clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

# ---------------------------------------------------------------------------
# Backtesting (cheap) — the new strategy framework
# ---------------------------------------------------------------------------
.PHONY: strategies
strategies:  ## List all registered strategies
	$(PY) -m src.run_backtest --list

.PHONY: backtest
backtest:  ## Backtest a strategy. Usage: make backtest STRAT=sma_cross INPUT=data/processed/train_15m_indicators.parquet
	$(PY) -m src.run_backtest --strategy $(STRAT) --input $(INPUT)

.PHONY: sweep
sweep:  ## Compare all strategies on a holdout + buy-and-hold. Usage: make sweep INPUT=... [BASE_TF=15m]
	$(PY) -m src.sweep --all --input $(INPUT) $(if $(BASE_TF),--base-tf $(BASE_TF),)

.PHONY: sweep-synth
sweep-synth:  ## Quick sweep of every strategy on synthetic data (no dataset needed)
	$(PY) -m src.sweep --all --synthetic 8000

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
.PHONY: bot
bot:  ## Run one BTC-accumulation paper cycle using runtime state
	$(PY) -m src.run_bot --strategies outputs/active_strategies_position.json \
		--state-file runtime/btc_accumulation_state.json \
		--trade-log runtime/btc_accumulation_trades.csv \
		--starting-equity 1 --market spot --symbol BTCUSDT --regime-guard

.PHONY: bot-flow
bot-flow:  ## Run one active-income paper cycle using runtime state
	$(PY) -m src.run_bot --strategies outputs/active_strategies_flow.json \
		--state-file runtime/active_income_state.json \
		--trade-log runtime/active_income_trades.csv \
		--starting-equity 1000 --market futures --symbol BTCUSDT

.PHONY: autopilot-validate
autopilot-validate:  ## Validate the 24/7 autopilot config
	$(PY) -m src.autopilot.runtime --config config/autopilot.json --validate

.PHONY: platform-validate
platform-validate:  ## Validate split platform configuration and node assignments
	$(PY) -m src.services.supervisor --config config/platform.json \
		--node linux-optiplex --service product-supervisor --validate

.PHONY: platform-install-dry-run
platform-install-dry-run:  ## Validate the Linux installer without changing the host
	DRY_RUN=1 REPO="$(CURDIR)" bash scripts/install_platform_services.sh

.PHONY: platform-readiness
platform-readiness:  ## Check PostgreSQL, canonical tables, Parquet, and paper-only state
	$(PY) -m src.services.readiness --config config/platform.json

.PHONY: platform-readiness-live
platform-readiness-live:  ## Check live platform readiness with PostgreSQL schema verification
	$(PY) -m src.services.readiness --config config/platform.json --live $(if $(PRODUCT),--product $(PRODUCT),)

.PHONY: platform-smoke
platform-smoke:  ## Run the PostgreSQL closed-event platform smoke for both products
	$(PY) -m src.services.platform_smoke --database-url "$(TRADING_PLATFORM_DATABASE_URL)"

.PHONY: platform-testnet-rehearsal
platform-testnet-rehearsal:  ## Verify the platform live/user-stream/accounting/recovery rehearsal path
	$(PY) -m pytest -q tests/test_platform_testnet_rehearsal.py tests/test_platform_testnet_rehearsal_integration.py

.PHONY: platform-testnet-connected
platform-testnet-connected:  ## Run a confirmed connected new-platform Binance testnet open/close rehearsal
	@test "$(CONFIRM)" = "1" || (echo "Set CONFIRM=1 to place real testnet orders." >&2; exit 1)
	$(PY) -m src.services.platform_testnet_connected --config config/platform.json --product $(or $(PRODUCT),active_income) --notional-usd $(or $(NOTIONAL_USD),10) --confirm

.PHONY: platform-live-authority
platform-live-authority:  ## Inspect or record exact manual live authority
	$(PY) -m src.services.platform_live_authority --config config/platform.json $(ARGS)

.PHONY: platform-permissions-test
platform-permissions-test:  ## Verify installed service users, writable paths, and one cycle per domain
	REPO="$(CURDIR)" bash scripts/verify_platform_service_install.sh

.PHONY: platform-ci
platform-ci:  ## Run the platform configuration, lint, migration, smoke, and test checks
	$(MAKE) platform-validate
	$(MAKE) lint-autopilot
	$(MAKE) lint
	$(MAKE) db-alembic
	$(MAKE) db-migration-check
	$(MAKE) platform-smoke
	$(MAKE) test

.PHONY: db-migrate
db-migrate:  ## Apply versioned PostgreSQL migrations
	$(PY) -m src.data.migrate --database-url "$(TRADING_PLATFORM_DATABASE_URL)"

.PHONY: db-alembic
db-alembic:  ## Apply immutable Alembic revisions as the database owner
	TRADING_PLATFORM_DATABASE_URL="$(TRADING_PLATFORM_DATABASE_URL)" $(PY) -m alembic upgrade head

.PHONY: db-migration-check
db-migration-check:  ## Verify the complete PostgreSQL schema is migrated
	$(PY) -m src.data.migrate --database-url "$(TRADING_PLATFORM_DATABASE_URL)" --check

.PHONY: strategy-manifest-validate
strategy-manifest-validate:  ## Validate the complete named strategy manifest
	$(PY) -c "from src.strategies.manifest import assert_manifest_complete, strategy_manifest; assert_manifest_complete(); print(len(strategy_manifest()))"

.PHONY: sqlite-import
sqlite-import:  ## Import a verified legacy SQLite memory into PostgreSQL
	$(PY) -m src.research.sqlite_import --source "$(SOURCE)" \
		--database-url "$(TRADING_PLATFORM_DATABASE_URL)" $(if $(ARCHIVE),--archive "$(ARCHIVE)",)

.PHONY: platform-backup-postgresql
platform-backup-postgresql:  ## Create a verified PostgreSQL platform backup
	$(PY) -m src.services.backup_service --config config/platform.json --mode postgresql

.PHONY: platform-backup-parquet
platform-backup-parquet:  ## Create a verified Parquet platform backup
	$(PY) -m src.services.backup_service --config config/platform.json --mode parquet

.PHONY: platform-backup-verify
platform-backup-verify:  ## Verify every PostgreSQL and Parquet platform backup
	$(PY) -m src.services.backup_service --config config/platform.json --mode verify

.PHONY: platform-backup-restore-parquet
platform-backup-restore-parquet:  ## Restore a Parquet backup into a new directory
	$(PY) -m src.services.backup_service --config config/platform.json \
		--mode restore-parquet --backup-id "$(BACKUP_ID)" --destination "$(DESTINATION)" \
		--confirm-restore

.PHONY: platform-backup-restore-postgresql
platform-backup-restore-postgresql:  ## Restore a PostgreSQL backup into an explicit target database
	$(PY) -m src.services.backup_service --config config/platform.json \
		--mode restore-postgresql --backup-id "$(BACKUP_ID)" \
		$(if $(TARGET_DATABASE_URL),--target-database-url "$(TARGET_DATABASE_URL)",) \
		--confirm-restore

.PHONY: bootstrap-strategies
bootstrap-strategies:  ## Write paper-only bootstrap strategy artifacts for missing paper products
	$(PY) -m src.autopilot.bootstrap_strategies --config config/autopilot.json \
		--report runtime/bootstrap_strategies.json $(if $(OVERWRITE),--overwrite,)

.PHONY: execution-diagnostic
execution-diagnostic:  ## Run the isolated paper order, fill, and position diagnostic
	$(PY) -m src.products.execution_diagnostic \
		--journal runtime/execution_diagnostic_orders.jsonl \
		--output runtime/execution_diagnostic.json

.PHONY: readiness
readiness:  ## Check local server readiness for autopilot operation
	$(PY) -m src.autopilot.readiness --config config/autopilot.json \
		--output runtime/readiness_report.md --json-output runtime/readiness_report.json

.PHONY: service-dry-run
service-dry-run:  ## Rehearse user-systemd unit generation without touching live units
	UNIT_DIR="$(CURDIR)/runtime/systemd-dry-run" REPO="$(CURDIR)" \
		PYTHON="$(abspath $(PY))" CONFIG="$(CURDIR)/config/autopilot.json" \
		DRY_RUN=1 bash scripts/install_autopilot_service.sh

.PHONY: autopilot-once
autopilot-once:  ## Run one trading-supervision cycle without scheduled jobs
	$(PY) -m src.autopilot.runtime --config config/autopilot.json --once --skip-jobs

.PHONY: jobs-once
jobs-once:  ## Run one separately locked scheduled-job cycle
	$(PY) -m src.autopilot.job_worker --config config/autopilot.json --once

.PHONY: approvals
approvals:  ## List strategy live-trading approvals
	$(PY) -m src.autopilot.approvals list

.PHONY: control
control:  ## Read/update runtime control. Usage: make control ARGS="pause --reason maintenance"
	$(PY) -m src.autopilot.control --config config/autopilot.json $(ARGS)

.PHONY: preflight
preflight:  ## Run connected live/testnet readiness checks. Usage: make preflight PRODUCT=active_income
	$(PY) -m src.autopilot.preflight --config config/autopilot.json \
		$(if $(PRODUCT),--product $(PRODUCT),) --assume-live \
		--connect $(if $(REQUIRE_TESTNET),--require-testnet,) \
		--output $(PREFLIGHT_OUTPUT)

.PHONY: testnet-rehearsal
testnet-rehearsal:  ## Place+close a tiny active-income futures testnet order. Requires CONFIRM=1
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing: this target can place testnet orders."; \
		echo "Re-run with CONFIRM=1 after approval, preflight, and sandbox env are ready."; exit 1; fi
	$(PY) -m src.autopilot.testnet_rehearsal --config config/autopilot.json \
		--product active_income --notional-usd $(or $(NOTIONAL_USD),100) \
		--output runtime/testnet_rehearsal_report.json --confirm

.PHONY: testnet-status
testnet-status:  ## Summarize the saved active-income testnet rehearsal report without placing orders
	$(PY) -m src.autopilot.testnet_rehearsal --config config/autopilot.json \
		--product active_income --output runtime/testnet_rehearsal_report.json --status

.PHONY: promotion-review
promotion-review:  ## Build human review packet. Usage: make promotion-review ARTIFACT=... TRADE_LOG=...
	$(PY) -m src.autopilot.promotion --config config/autopilot.json \
		$(if $(PRODUCT),--product $(PRODUCT),) --artifact $(ARTIFACT) --trade-log $(TRADE_LOG)

.PHONY: report
report:  ## Build compact operator report
	$(PY) -m src.autopilot.reporting --config config/autopilot.json \
		--output runtime/operator_report.md --json-output runtime/operator_report.json

.PHONY: telegram-status
telegram-status:  ## Print the same sanitized read-only status exposed through Telegram
	$(PY) -m src.autopilot.telegram_edge --config config/autopilot.json --status

.PHONY: telegram-send-status
telegram-send-status:  ## Send one sanitized status message using runtime/telegram.env
	$(PY) -m src.autopilot.telegram_edge --config config/autopilot.json --send-status

.PHONY: openclaw-context
openclaw-context:  ## Export credential- and final-holdout-free research context for OpenClaw
	$(PY) -m src.autopilot.openclaw_bridge export

.PHONY: openclaw-ingest
openclaw-ingest:  ## Validate/archive inert OpenClaw proposals from the research inbox
	$(PY) -m src.autopilot.openclaw_bridge ingest

.PHONY: healthcheck
healthcheck:  ## Machine-readable watchdog check; exits nonzero on stale/failed autopilot state
	$(PY) -m src.autopilot.healthcheck --config config/autopilot.json \
		--output runtime/healthcheck.json $(HEALTHCHECK_ARGS)

.PHONY: backup
backup:  ## Create a small recovery backup zip of autopilot runtime state
	$(PY) -m src.autopilot.backup --config config/autopilot.json \
		--report runtime/backup_report.json --max-file-bytes 52428800 \
		--max-backups 30

.PHONY: backup-verify
backup-verify:  ## Verify a backup zip. Usage: make backup-verify [BACKUP=runtime/backups/...zip]
	@if [ -z "$(BACKUP)" ]; then \
		echo "No backup zip found. Run 'make backup' first or pass BACKUP=runtime/backups/...zip."; exit 1; fi
	$(PY) -m src.autopilot.backup --verify $(BACKUP)

.PHONY: backup-restore
backup-restore:  ## Extract a verified backup into RESTORE_DIR without overwriting existing files
	@if [ -z "$(BACKUP)" ]; then \
		echo "No backup zip found. Run 'make backup' first or pass BACKUP=runtime/backups/...zip."; exit 1; fi
	$(PY) -m src.autopilot.backup --restore $(BACKUP) --restore-dir $(RESTORE_DIR)

.PHONY: maintenance
maintenance:  ## Compact bounded autopilot runtime logs
	$(PY) -m src.autopilot.maintenance --config config/autopilot.json \
		--max-alert-lines 1000 --max-alert-fingerprints 1000 \
		--max-experiment-lines 5000 \
		--max-control-audit-lines 5000 \
		$(if $(QUARANTINE_BYTES),--max-quarantine-bytes $(QUARANTINE_BYTES),)

.PHONY: artifact-hygiene
artifact-hygiene:  ## Report stale/invalid artifacts; add APPLY=1 [UNREFERENCED=1] [HISTORICAL=1] to quarantine
	$(PY) -m src.autopilot.artifact_hygiene --config config/autopilot.json \
		--output runtime/artifact_hygiene.json $(if $(APPLY),--apply,) \
		$(if $(UNREFERENCED),--quarantine-unreferenced-active,) \
		$(if $(HISTORICAL),--quarantine-historical-search,)

.PHONY: data-update
data-update: data-update-futures data-update-spot  ## Incrementally update all seeded market datasets

.PHONY: regime-tag-futures
regime-tag-futures:  ## Build a futures 15m regime-tagged research parquet
	$(PY) -m src.regime --market futures --timeframe 15m --daily-timeframe 1d \
		--output runtime/regime/futures_15m_regime.parquet \
		--report runtime/regime_tag_futures_15m.json --compact --skip-if-missing

.PHONY: data-update-futures
data-update-futures:  ## Incrementally update seeded futures candles/features
	$(PY) -m src.autopilot.history_bootstrap --config config/research_factory.json \
		--market futures --exclude-timeframes 1m --report runtime/history_bootstrap_futures.json

.PHONY: data-update-1m-flow
data-update-1m-flow:  ## Rebuild 1m indicators for scalping flow features
	$(PY) -m src.autopilot.history_bootstrap --config config/research_factory.json --market futures \
		--timeframes 1m --report runtime/history_bootstrap_futures_1m.json

.PHONY: data-update-spot
data-update-spot:  ## Incrementally update seeded spot candles/features for BTC accumulation
	$(PY) -m src.autopilot.history_bootstrap --config config/research_factory.json \
		--market spot --report runtime/history_bootstrap_spot.json

.PHONY: research-smoke
research-smoke:  ## Run cheap synthetic research wiring checks for both products
	$(PY) -m src.autopilot.research_smoke --output runtime/research_smoke.json

.PHONY: research-factory-validate
research-factory-validate:  ## Validate autonomous grammar, budgets, and experiment memory
	$(PY) -m src.autopilot.research_factory --config config/research_factory.json --validate

.PHONY: research-generate
research-generate:  ## Generate the next bounded, deduplicated research population
	$(PY) -m src.autopilot.research_factory --config config/research_factory.json \
		--output runtime/research/generated_hypotheses.json

.PHONY: research-history-plan
research-history-plan:  ## Show exact native-timeframe history required by autonomous research
	$(PY) -m src.autopilot.history_bootstrap --config config/research_factory.json \
		--plan $(if $(MARKET),--market $(MARKET),)

.PHONY: research-history-bootstrap
research-history-bootstrap:  ## Bootstrap/update resumable lightweight native-timeframe research history
	$(PY) -m src.autopilot.history_bootstrap --config config/research_factory.json \
		$(if $(MARKET),--market $(MARKET),) \
		--report runtime/history_bootstrap$(if $(MARKET),_$(MARKET),).json

.PHONY: strategy-smoke
strategy-smoke:  ## Run lightweight strategy-framework sweeps on synthetic + regime data
	$(PY) -m src.autopilot.strategy_smoke \
		--output runtime/strategy_framework_smoke.json \
		--regime-input runtime/regime/futures_15m_regime.parquet

.PHONY: research-cycle
research-cycle: mutation-batch  ## Validate the autonomous population and gate paper handoff
	$(PY) -m src.autopilot.research_cycle \
		--config config/autopilot.json \
		--output runtime/research_cycle.json \
		--state runtime/research_cycle_state.json \
		--include-generated --include-mutations \
		--generated-batch runtime/research/generated_hypotheses.json \
		--mutation-batch runtime/mutation_hypotheses.json \
		--research-factory-config config/research_factory.json

.PHONY: mutation-plan
mutation-plan:  ## Convert exploration/predicate feedback into research-only mutation instructions
	$(PY) -m src.autopilot.mutation_plan \
		--input runtime/research_cycle.json \
		--exploration-status runtime/exploration_paper/status.json \
		--experiment-log outputs/research_exploration/experiment_log.jsonl \
		--output runtime/mutation_plan.json \
		--markdown-output runtime/mutation_plan.md --max-total 12

.PHONY: mutation-batch
mutation-batch: mutation-plan  ## Compile bounded research-only stage-directed mutations
	$(PY) -m src.autopilot.mutation_batch \
		--input runtime/mutation_plan.json \
		--output runtime/mutation_hypotheses.json --max-total 12

.PHONY: research-once
research-once: research-generate research-cycle  ## Generate, mutate, and validate one bounded population

.PHONY: ml-research-validate
ml-research-validate:  ## Validate the bounded chronological ML research grid
	$(PY) -m src.autopilot.ml_research --config config/ml_research.json --validate

.PHONY: ml-research-once
ml-research-once:  ## Run at most two pre-holdout ML trials (resource bounded)
	$(PY) -m src.autopilot.ml_research --config config/ml_research.json \
		--output runtime/research/ml_research.json \
		--state runtime/research/ml_research_state.json

.PHONY: ml-forward-paper-once
ml-forward-paper-once:  ## Run isolated non-promotable forward paper for protected ML candidates
	$(PY) -m src.autopilot.ml_forward_paper --config config/ml_research.json \
		--candidates runtime/research/ml_research.json \
		--output runtime/research/ml_forward_paper.json \
		--state runtime/research/ml_forward_paper_state.json

.PHONY: ml-stage-candidate
ml-stage-candidate:  ## Stage an exact reviewed ML artifact for candidate paper (PRODUCT=... ARTIFACT=... DIGEST=...)
	$(PY) -m src.autopilot.ml_candidate_artifact --config config/autopilot.json \
		--product "$(PRODUCT)" --artifact "$(ARTIFACT)" --expected-digest "$(DIGEST)" \
		$(if $(REPLACE),--replace,)

.PHONY: portfolio-risk-validate
portfolio-risk-validate:  ## Validate rolling correlation and benchmark-beta configuration
	$(PY) -m src.autopilot.portfolio_risk --config config/portfolio_risk.json --validate

.PHONY: portfolio-risk
portfolio-risk:  ## Build the rolling cross-symbol risk model
	$(PY) -m src.autopilot.portfolio_risk --config config/portfolio_risk.json \
		--output runtime/portfolio_risk.json

.PHONY: relative-value-validate
relative-value-validate:  ## Validate bounded relative-value research configuration
	$(PY) -m src.autopilot.relative_value_research --config config/relative_value.json --validate

.PHONY: relative-value
relative-value:  ## Build research-only basis, cross-sectional, and pairs forecasts
	$(PY) -m src.autopilot.relative_value_research --config config/relative_value.json \
		--output runtime/research/relative_value.json

.PHONY: relative-value-paper-once
relative-value-paper-once:  ## Run zero-money non-promotable relative-value forward paper
	$(PY) -m src.autopilot.relative_value_paper \
		--input runtime/research/relative_value.json \
		--output runtime/research/relative_value_paper.json \
		--state runtime/research/relative_value_paper_state.json --timeframe 1h

.PHONY: activate-candidate
activate-candidate:  ## Activate reviewed live candidate. Usage: make activate-candidate PRODUCT=active_income CANDIDATE_DIGEST=sha256:... CONFIRM=1
	@if [ -z "$(PRODUCT)" ]; then \
		echo "Refusing: pass PRODUCT=<configured-live-product>."; exit 1; fi
	@if [ -z "$(CANDIDATE_DIGEST)" ]; then \
		echo "Refusing: pass CANDIDATE_DIGEST=sha256:<reviewed-candidate-digest>."; exit 1; fi
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing: candidate activation requires CONFIRM=1."; exit 1; fi
	$(PY) -m src.autopilot.candidate_activation \
		--config config/autopilot.json --product "$(PRODUCT)" \
		--expected-candidate-digest "$(CANDIDATE_DIGEST)" --confirm $(if $(OPERATOR),--operator "$(OPERATOR)")

.PHONY: candidate-paper-once
candidate-paper-once:  ## Run promotion-paper candidates and adaptive exploration paper once
	$(PY) -m src.autopilot.candidate_paper --config config/autopilot.json \
		--output runtime/candidate_paper_status.json

.PHONY: exploration-paper-build
exploration-paper-build:  ## Compile the incubation queue into non-promotable paper artifacts
	$(PY) -m src.autopilot.exploration_paper --config config/autopilot.json --build

.PHONY: exploration-paper-once
exploration-paper-once:  ## Run one adaptive, permanently non-promotable paper cycle
	$(PY) -m src.autopilot.exploration_paper --config config/autopilot.json

.PHONY: event-capture-validate
event-capture-validate:  ## Validate bounded public market-event capture configuration
	$(PY) -m src.autopilot.event_capture --config config/event_capture.json --validate

.PHONY: event-capture-smoke
event-capture-smoke:  ## Capture at most 100 public events for an operational smoke test
	$(PY) -m src.autopilot.event_capture --config config/event_capture.json \
		--status runtime/event_capture_status.json --max-events 100 --max-seconds 30

.PHONY: event-replay
event-replay:  ## Replay captured events. Usage: make event-replay PATHS='runtime/events/*.jsonl' SYMBOL=BTCUSDT
	@if [ -z "$(PATHS)" ] || [ -z "$(SYMBOL)" ]; then \
		echo "Refusing: pass PATHS='<event files>' and SYMBOL=<symbol>."; exit 1; fi
	$(PY) -m src.autopilot.event_replay $(PATHS) --symbol "$(SYMBOL)"

.PHONY: microstructure-research-validate
microstructure-research-validate:  ## Validate bounded short-horizon replay configuration
	$(PY) -m src.autopilot.microstructure_research \
		--config config/microstructure_research.json --validate

.PHONY: microstructure-research
microstructure-research:  ## Replay recent events through research-only microstructure alpha
	$(PY) -m src.autopilot.microstructure_research \
		--config config/microstructure_research.json \
		--output runtime/research/microstructure.json

.PHONY: accounting
accounting:  ## Reconcile trade logs and update the hash-chained attribution journal
	$(PY) -m src.autopilot.accounting --config config/autopilot.json \
		--journal runtime/accounting/journal.jsonl \
		--output runtime/accounting/report.json

.PHONY: rehearse
rehearse:  ## Run offline end-to-end autopilot workflow rehearsal
	$(PY) -m src.autopilot.rehearsal

# ---------------------------------------------------------------------------
# Heavy / guarded (require CONFIRM=1) — never run casually
# ---------------------------------------------------------------------------
.PHONY: guard
guard:
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing: this target is expensive (hours of compute)."; \
		echo "Re-run with CONFIRM=1 if you really mean it."; exit 1; fi

.PHONY: data
data: guard  ## [HEAVY] Rebuild the full Binance indicator dataset
	$(PY) build_binance_indicator_dataset.py

.PHONY: search-btc
search-btc: guard  ## [HEAVY] Run the BTC-position walk-forward search
	$(PY) -m src.strategy_search --walk-forward --n-jobs 7 --resume --output-dir outputs/search_btc_manual

.PHONY: search-flow
search-flow: guard  ## [HEAVY] Run the flow/day-trade walk-forward search
	$(PY) -m src.day_trade_search --base-tf 15m --walk-forward --n-jobs 7 --resume --output-dir outputs/search_flow_manual
