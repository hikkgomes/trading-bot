# PostgreSQL-authoritative platform developer and operator commands.

VENV ?= .venv
PY ?= $(VENV)/bin/python
PLATFORM_CONFIG ?= config/platform.json
PREFLIGHT_OUTPUT ?= runtime/preflight_report.json
BACKUP_ID ?=
DESTINATION ?=
PROCESS ?= trading-runtime

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show platform commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: setup setup-bot
setup:  ## Create the local development environment
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt

setup-bot:  ## Install the lean Linux execution environment
	$(PY) -m pip install -r requirements-runtime.txt

.PHONY: test test-fast lint lint-complexity typecheck-platform format clean
test:  ## Run the complete test suite
	$(PY) -m pytest

test-fast:  ## Run tests excluding slow tests
	$(PY) -m pytest -m "not slow"

lint:  ## Run Ruff lint and formatting checks
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

lint-complexity:  ## Enforce maximum cyclomatic complexity of 10
	$(PY) -m ruff check . --select C90

typecheck-platform:  ## Type-check the PostgreSQL platform boundary
	$(PY) -m mypy --ignore-missing-imports --follow-imports=skip \
		src/data src/domain src/observability src/research src/services src/strategies

format:  ## Apply Ruff formatting and safe fixes
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

clean:  ## Remove local caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

.PHONY: platform-validate platform-install-dry-run platform-readiness platform-readiness-live
platform-validate:  ## Validate the Linux platform and process topology
	$(PY) -m src.services.supervisor --config $(PLATFORM_CONFIG) \
		--node linux-optiplex --service product-supervisor --validate

platform-install-dry-run:  ## Validate the Linux installer without changing the host
	DRY_RUN=1 REPO="$(CURDIR)" bash scripts/install_platform_services.sh

platform-readiness:  ## Check PostgreSQL schema, data, paper state, and progress
	$(PY) -m src.services.readiness --config $(PLATFORM_CONFIG)

platform-readiness-live:  ## Check live readiness for one configured product
	$(PY) -m src.services.readiness --config $(PLATFORM_CONFIG) --live \
		$(if $(PRODUCT),--product $(PRODUCT),)

.PHONY: platform-process-once
platform-process-once:  ## Run one grouped platform process cycle
	@test -n "$(PROCESS)" || (echo "PROCESS is required" >&2; exit 1)
	$(PY) -m src.services.process --config $(PLATFORM_CONFIG) \
		--node linux-optiplex --process $(PROCESS) --once

.PHONY: platform-smoke platform-testnet-rehearsal platform-testnet-connected
platform-smoke:  ## Run the closed-event PostgreSQL smoke for both products
	$(PY) -m src.services.platform_smoke --database-url "$(TRADING_PLATFORM_DATABASE_URL)"

platform-testnet-rehearsal:  ## Run the deterministic live/user-stream rehearsal
	$(PY) -m pytest -q tests/test_platform_testnet_rehearsal.py \
		tests/test_platform_testnet_rehearsal_integration.py

platform-testnet-connected:  ## Run a confirmed connected Binance testnet rehearsal
	@test "$(CONFIRM)" = "1" || (echo "Set CONFIRM=1 to place real testnet orders." >&2; exit 1)
	$(PY) -m src.services.platform_testnet_connected --config $(PLATFORM_CONFIG) \
		--product $(or $(PRODUCT),active_income) --notional-usd $(or $(NOTIONAL_USD),10) --confirm

.PHONY: platform-live-authority platform-permissions-test platform-report control
platform-live-authority:  ## Inspect or record exact manual live authority
	$(PY) -m src.services.platform_live_authority --config $(PLATFORM_CONFIG) $(ARGS)

platform-permissions-test:  ## Verify installed service users and one domain cycle
	REPO="$(CURDIR)" bash scripts/verify_platform_service_install.sh

platform-report:  ## Materialise the PostgreSQL operator report
	$(PY) -m src.services.supervisor --config $(PLATFORM_CONFIG) \
		--node linux-optiplex --service report-worker --once

control:  ## Operate the PostgreSQL control plane, for example ARGS="status"
	$(PY) -m src.services.control_cli --config $(PLATFORM_CONFIG) $(ARGS)

.PHONY: db-alembic db-migrate db-migration-check sqlite-import
db-alembic:  ## Apply immutable Alembic migrations
	TRADING_PLATFORM_DATABASE_URL="$(TRADING_PLATFORM_DATABASE_URL)" \
		$(PY) -m alembic upgrade head

db-migrate:  ## Apply the platform migration service command
	$(PY) -m src.data.migrate --database-url "$(TRADING_PLATFORM_DATABASE_URL)"

db-migration-check:  ## Verify the complete PostgreSQL schema is migrated
	$(PY) -m src.data.migrate --database-url "$(TRADING_PLATFORM_DATABASE_URL)" --check

sqlite-import:  ## Import legacy SQLite memory with immutable provenance
	@test -n "$(SOURCE)" || (echo "SOURCE is required" >&2; exit 1)
	$(PY) -m src.research.sqlite_import --source "$(SOURCE)" \
		--database-url "$(TRADING_PLATFORM_DATABASE_URL)" $(if $(ARCHIVE),--archive "$(ARCHIVE)",)

.PHONY: platform-backup-postgresql platform-backup-parquet platform-backup-verify
platform-backup-postgresql:  ## Create and verify a PostgreSQL backup
	$(PY) -m src.services.backup_service --config $(PLATFORM_CONFIG) --mode postgresql

platform-backup-parquet:  ## Create and verify a Parquet backup
	$(PY) -m src.services.backup_service --config $(PLATFORM_CONFIG) --mode parquet

platform-backup-verify:  ## Verify every platform backup
	$(PY) -m src.services.backup_service --config $(PLATFORM_CONFIG) --mode verify

.PHONY: platform-backup-restore-postgresql platform-backup-restore-parquet
platform-backup-restore-postgresql:  ## Restore a PostgreSQL backup into a new database
	@test -n "$(BACKUP_ID)" || (echo "BACKUP_ID is required" >&2; exit 1)
	@test -n "$(TARGET_DATABASE_URL)" || (echo "TARGET_DATABASE_URL is required" >&2; exit 1)
	$(PY) -m src.services.backup_service --config $(PLATFORM_CONFIG) \
		--mode restore-postgresql --backup-id "$(BACKUP_ID)" \
		--target-database-url "$(TARGET_DATABASE_URL)" --confirm-restore

platform-backup-restore-parquet:  ## Restore a Parquet backup into a new directory
	@test -n "$(BACKUP_ID)" || (echo "BACKUP_ID is required" >&2; exit 1)
	@test -n "$(DESTINATION)" || (echo "DESTINATION is required" >&2; exit 1)
	$(PY) -m src.services.backup_service --config $(PLATFORM_CONFIG) \
		--mode restore-parquet --backup-id "$(BACKUP_ID)" \
		--destination "$(DESTINATION)" --confirm-restore

.PHONY: strategy-manifest-validate research-policy-check platform-ci
strategy-manifest-validate:  ## Validate named strategy and feature contracts
	$(PY) -c "from src.strategies.manifest import assert_manifest_complete; assert_manifest_complete()"

research-policy-check:  ## Run canonical research policy tests
	$(PY) -m pytest -q tests/test_research_evidence_integrity.py \
		tests/test_research_job_authority.py tests/test_research_generation.py \
		tests/test_strategy_evaluation_pipeline.py

platform-ci:  ## Run the complete platform quality gate
	$(MAKE) platform-validate
	$(MAKE) lint
	$(MAKE) lint-complexity
	$(MAKE) typecheck-platform
	$(MAKE) research-policy-check
	$(MAKE) db-alembic
	$(MAKE) db-migration-check
	$(MAKE) platform-smoke
	$(MAKE) platform-testnet-rehearsal
	$(MAKE) test

# Compatibility names retained only as PostgreSQL command aliases. They do not
# invoke the archived file or SQLite authority.
.PHONY: autopilot-validate lint-autopilot readiness autopilot-once report artifact-hygiene
autopilot-validate: platform-validate
lint-autopilot: lint
readiness: platform-readiness
autopilot-once:  ## Alias for one canonical trading-runtime process cycle
	$(MAKE) platform-process-once PROCESS=trading-runtime
report: platform-report
artifact-hygiene: platform-readiness
