# Trading-bot developer tasks.
# Heavy targets (data, search, train) are guarded behind CONFIRM=1 because they
# cost hours of compute — see CLAUDE.md "Important Constraints".

VENV ?= .venv
PY ?= $(VENV)/bin/python
BOOTSTRAP_PY ?= python3
PREFLIGHT_OUTPUT ?= $(if $(PRODUCT),runtime/$(PRODUCT)_preflight_report.json,runtime/preflight_report.json)
BACKUP ?= $(shell ls -t runtime/backups/autopilot_state_*.zip 2>/dev/null | head -1)
RESTORE_DIR ?= runtime/restore_rehearsal

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
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
	$(PY) -m ruff check src tests

.PHONY: lint-autopilot
lint-autopilot:  ## Lint autonomous runtime + execution surface
	$(PY) -m ruff check src/autopilot src/run_bot.py src/execution \
		tests/test_autopilot_*.py tests/test_run_bot.py tests/test_execution.py

.PHONY: format
format:  ## Auto-format + autofix with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

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

.PHONY: bootstrap-strategies
bootstrap-strategies:  ## Write paper-only bootstrap strategy artifacts for missing paper products
	$(PY) -m src.autopilot.bootstrap_strategies --config config/autopilot.json \
		--report runtime/bootstrap_strategies.json $(if $(OVERWRITE),--overwrite,)

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
autopilot-once:  ## Run one 24/7 autopilot orchestration cycle
	$(PY) -m src.autopilot.runtime --config config/autopilot.json --once

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
		--product active_income --notional-usd $(or $(NOTIONAL_USD),5) \
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

.PHONY: healthcheck
healthcheck:  ## Machine-readable watchdog check; exits nonzero on stale/failed autopilot state
	$(PY) -m src.autopilot.healthcheck --config config/autopilot.json \
		--output runtime/healthcheck.json $(HEALTHCHECK_ARGS)

.PHONY: backup
backup:  ## Create a small recovery backup zip of autopilot runtime state
	$(PY) -m src.autopilot.backup --config config/autopilot.json \
		--report runtime/backup_report.json --max-file-bytes 10485760 \
		--max-backups 30

.PHONY: backup-verify
backup-verify:  ## Verify a backup zip. Usage: make backup-verify [BACKUP=runtime/backups/...zip]
	$(PY) -m src.autopilot.backup --verify $(BACKUP)

.PHONY: backup-restore
backup-restore:  ## Extract a verified backup into RESTORE_DIR without overwriting existing files
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
		--report runtime/regime_tag_futures_15m.json --skip-if-missing

.PHONY: data-update-futures
data-update-futures:  ## Incrementally update seeded futures candles/features
	$(PY) -m src.update_candles --market futures --bootstrap-days 90 --skip-if-missing --timeframes 5m 15m 30m 1h 4h 1d

.PHONY: data-update-1m-flow
data-update-1m-flow:  ## Rebuild 1m indicators for scalping flow features
	$(PY) -m src.update_candles --market futures --bootstrap-days 90 --skip-if-missing --timeframes 1m

.PHONY: data-update-spot
data-update-spot:  ## Incrementally update seeded spot candles/features for BTC accumulation
	$(PY) -m src.update_candles --market spot --bootstrap-days 365 --skip-if-missing --timeframes 1h 4h 1d 1w

.PHONY: research-smoke
research-smoke:  ## Run cheap synthetic research wiring checks for both products
	$(PY) -m src.autopilot.research_smoke --output runtime/research_smoke.json

.PHONY: strategy-smoke
strategy-smoke:  ## Run lightweight strategy-framework sweeps on synthetic + regime data
	$(PY) -m src.autopilot.strategy_smoke \
		--output runtime/strategy_framework_smoke.json \
		--regime-input runtime/regime/futures_15m_regime.parquet

.PHONY: research-cycle
research-cycle:  ## Run bounded real-data validation + gated strategy export
	$(PY) -m src.autopilot.research_cycle \
		--output runtime/research_cycle.json \
		--state runtime/research_cycle_state.json \
		--include-mutations \
		--mutation-batch runtime/mutation_hypotheses.json

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
