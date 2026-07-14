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
    assert "DRY_RUN=1 bash scripts/install_autopilot_service.sh" in makefile


def test_candidate_activation_make_target_requires_explicit_confirmation():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert ".PHONY: activate-candidate" in makefile
    assert 'if [ "$(CONFIRM)" != "1" ]' in makefile
    assert 'if [ -z "$(CANDIDATE_DIGEST)" ]' in makefile
    assert "-m src.autopilot.candidate_activation" in makefile
    assert '--expected-candidate-digest "$(CANDIDATE_DIGEST)" --confirm' in makefile


def test_autonomous_research_targets_generate_then_validate_typed_batch():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert ".PHONY: research-factory-validate" in makefile
    assert (
        "-m src.autopilot.research_factory --config config/research_factory.json --validate"
        in makefile
    )
    assert ".PHONY: research-generate" in makefile
    assert "--output runtime/research/generated_hypotheses.json" in makefile
    assert "--include-generated --generated-only" in makefile
    assert "research-once: research-generate research-cycle" in makefile
