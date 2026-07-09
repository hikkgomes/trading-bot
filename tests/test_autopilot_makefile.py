from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_preflight_make_target_always_runs_connected_checks():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "--connect $(if $(REQUIRE_TESTNET),--require-testnet,)" in makefile
    assert "$(if $(CONNECT),--connect,)" not in makefile


def test_control_make_target_uses_config_for_selector_validation():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "$(PY) -m src.autopilot.control --config config/autopilot.json $(ARGS)" in makefile


def test_testnet_status_make_target_is_read_only():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert ".PHONY: testnet-status" in makefile
    assert "--output runtime/testnet_rehearsal_report.json --status" in makefile


def test_service_dry_run_make_target_uses_local_unit_dir():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert ".PHONY: service-dry-run" in makefile
    assert 'UNIT_DIR="$(CURDIR)/runtime/systemd-dry-run"' in makefile
    assert 'DRY_RUN=1 bash scripts/install_autopilot_service.sh' in makefile
