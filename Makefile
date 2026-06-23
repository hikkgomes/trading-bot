# Trading-bot developer tasks.
# Heavy targets (data, search, train) are guarded behind CONFIRM=1 because they
# cost hours of compute — see CLAUDE.md "Important Constraints".

PY ?= python
VENV ?= .venv

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
	$(PY) -m venv $(VENV)
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
bot:  ## Run one position-bot paper cycle
	$(PY) -m src.run_bot

.PHONY: bot-flow
bot-flow:  ## Run one flow/day-trade-bot paper cycle
	$(PY) -m src.run_bot --strategies outputs/active_strategies_flow.json \
		--state-file outputs/bot_state_flow.json --starting-equity 1000

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
	$(PY) -m src.strategy_search --walk-forward --n-jobs 7 --resume --output-dir outputs/search_v4_btc

.PHONY: search-flow
search-flow: guard  ## [HEAVY] Run the flow/day-trade walk-forward search
	$(PY) -m src.day_trade_search --base-tf 15m --walk-forward --n-jobs 7 --resume --output-dir outputs/search_flow
