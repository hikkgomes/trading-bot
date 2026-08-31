from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _makefile() -> str:
    return (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")


def test_makefile_exposes_postgresql_as_the_operational_authority():
    makefile = _makefile()

    assert makefile.startswith("# PostgreSQL-authoritative platform")
    assert "PLATFORM_CONFIG ?= config/platform.json" in makefile
    assert "src.services.control_cli --config $(PLATFORM_CONFIG)" in makefile
    assert "src.autopilot.control --config config/autopilot.json" not in makefile


def test_live_release_commands_are_explicit_and_readiness_bound():
    makefile = _makefile()

    assert "platform-readiness-live" in makefile
    assert "src.services.readiness --config $(PLATFORM_CONFIG) --live" in makefile
    assert "platform-testnet-connected" in makefile
    assert 'test "$(CONFIRM)" = "1"' in makefile
    assert "src.services.platform_testnet_connected" in makefile


def test_platform_installation_uses_the_grouped_linux_installer():
    makefile = _makefile()

    assert "platform-install-dry-run" in makefile
    assert 'DRY_RUN=1 REPO="$(CURDIR)" bash scripts/install_platform_services.sh' in makefile
    assert "install_autopilot_service.sh" not in makefile


def test_legacy_operations_are_only_explicit_migration_or_aliases():
    makefile = _makefile()

    assert "sqlite-import:" in makefile
    assert "src.research.sqlite_import" in makefile
    assert "autopilot-validate: platform-validate" in makefile
    assert "autopilot-once:" in makefile
    assert "activate-candidate" not in makefile
    assert "research-factory-validate" not in makefile


def test_platform_ci_contains_all_required_quality_gates():
    makefile = _makefile()

    assert "lint-complexity:" in makefile
    assert "$(PY) -m ruff check . --select C90" in makefile
    assert "$(MAKE) typecheck-platform" in makefile
    assert "$(MAKE) research-policy-check" in makefile
    assert "$(MAKE) db-alembic" in makefile
    assert "$(MAKE) db-migration-check" in makefile
    assert "$(MAKE) platform-smoke" in makefile
    assert "$(MAKE) platform-testnet-rehearsal" in makefile
    assert "$(MAKE) test" in makefile


def test_platform_ci_workflow_provisions_postgresql_without_live_credentials():
    workflow = (PROJECT_ROOT / ".github/workflows/platform.yml").read_text(encoding="utf-8")

    assert "image: postgres:16" in workflow
    assert "TRADING_PLATFORM_DATABASE_URL:" in workflow
    assert "TRADING_PLATFORM_TESTNET_DATABASE_URL:" in workflow
    assert "make platform-ci" in workflow
    assert "BINANCE_API_KEY" not in workflow
    assert "BINANCE_API_SECRET" not in workflow
