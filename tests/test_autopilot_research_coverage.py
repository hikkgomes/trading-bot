import json
from types import SimpleNamespace

import pandas as pd

from research_exploration.hypothesis_generator import generate_batch
from src.autopilot import research_cycle as rc
from src.autopilot.healthcheck import evaluate_health
from src.autopilot.reporting import render_operator_markdown


def coverage_scenario(**overrides):
    values = {
        "name": "coverage_test",
        "product": "active_income",
        "base_tf": "5m",
        "pnl_unit": "usdt",
        "market": "futures",
        "position": False,
        "start": "2026-01-01",
        "candidate_set": "full",
        "max_hypotheses": 1,
        "coverage_earliest": "2026-01-01",
        "coverage_max_start_delay_days": 1,
        "coverage_max_latest_age_hours": 24,
        "coverage_min_span_days": 4,
        "coverage_min_rows": 5,
    }
    values.update(overrides)
    return rc.ResearchScenario(**values)


def test_every_default_scenario_has_explicit_history_contract():
    for scenario in rc.DEFAULT_SCENARIOS:
        assert scenario.coverage_earliest
        assert scenario.coverage_max_start_delay_days is not None
        assert scenario.coverage_max_latest_age_hours
        assert scenario.coverage_min_span_days
        assert scenario.coverage_min_rows


def test_history_coverage_reports_earliest_latest_span_and_rows():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-03", periods=5, freq="1D", tz="UTC")})

    status = rc._scenario_coverage_status(
        frame,
        coverage_scenario(),
        now="2026-01-08T00:00:00Z",
    )

    assert status["ok"] is False
    assert status["failed_checks"] == ["earliest"]
    assert status["checks"] == {
        "earliest": False,
        "latest": True,
        "span": True,
        "rows": True,
    }
    assert status["requirements"] == {
        "earliest_at_or_before": "2026-01-02T00:00:00+00:00",
        "latest_at_or_after": "2026-01-07T00:00:00+00:00",
        "minimum_span_days": 4.0,
        "minimum_rows": 5,
    }
    assert status["actual"]["earliest"] == "2026-01-03T00:00:00+00:00"
    assert status["remediation"]["action"] == "bootstrap_research_history"


def test_validation_skips_before_holdout_when_history_is_shallow(monkeypatch, tmp_path):
    scenario = coverage_scenario(coverage_min_rows=100)
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    monkeypatch.setattr(
        rc,
        "_scenario_indicator_coverage_status",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(rc, "_missing_columns_for_hypothesis", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rc,
        "build_aligned_frame",
        lambda *args, **kwargs: pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")}
        ),
    )

    def fail_validation(*args, **kwargs):
        raise AssertionError("holdout validation must not run on shallow history")

    monkeypatch.setattr(rc, "validate_batch", fail_validation)

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=[hypothesis],
        coverage_now="2026-01-06T00:00:00Z",
        log_path=None,
        research_factory_config_path=tmp_path / "custom_factory.json",
    )

    assert report["ok"] is False
    assert report["skipped"] is True
    assert report["reason"] == "insufficient_history_coverage"
    assert report["coverage"]["failed_checks"] == ["rows"]
    assert report["holdout_exposed_ids"] == []
    assert report["remediation"]["command"][2:] == [
        "src.autopilot.history_bootstrap",
        "--config",
        str(tmp_path / "custom_factory.json"),
        "--market",
        "futures",
        "--report",
        "runtime/history_bootstrap_futures.json",
    ]


def test_protected_epoch_may_end_before_full_source_freshness_threshold(monkeypatch):
    scenario = coverage_scenario()
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    full_frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=10, freq="1D", tz="UTC")}
    )
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": "2026-01-08T00:00:00+00:00",
            "end": "2026-01-10T00:00:00+00:00",
        },
    )
    captured = {}

    class FakeExperimentMemory:
        def __init__(self):
            self.strategies = {}

        def protected_intervals(self, **_kwargs):
            return protected

        def register_strategy(self, spec, **_kwargs):
            self.strategies[rc.canonical_strategy_hash(spec)] = spec

        def get_strategy(self, behavior_hash):
            return {
                "submitted_spec": self.strategies[behavior_hash],
                "holdout_exposed_at": None,
            }

        def is_tested(self, *_args, **_kwargs):
            return False

        def assert_adaptive_window_allowed(self, **kwargs):
            captured["adaptive_window"] = kwargs["window"]

        def register_holdout_cohort(self, behavior_hashes, **_kwargs):
            return SimpleNamespace(
                member_hashes=tuple(behavior_hashes),
                created=True,
                scope_key="protected-test-cohort",
            )

    monkeypatch.setattr(
        rc,
        "_scenario_indicator_coverage_status",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(rc, "_missing_columns_for_hypothesis", lambda *args, **kwargs: {})
    monkeypatch.setattr(rc, "build_aligned_frame", lambda *args, **kwargs: full_frame.copy())
    monkeypatch.setattr(
        rc,
        "with_trial_sharpe_dispersion",
        lambda _frame, _hypotheses, cfg, _eval_cfg: cfg,
    )
    monkeypatch.setattr(
        rc,
        "_dataset_snapshot",
        lambda *args, **kwargs: {"snapshot_id": "sha256:" + "a" * 64},
    )

    def fake_validate(frame, hypotheses, _config, **_kwargs):
        captured["validation_latest"] = pd.Timestamp(frame["timestamp"].iloc[-1])
        return [
            {
                "hypothesis_id": hypotheses[0].id,
                "family": hypotheses[0].family,
                "direction": hypotheses[0].direction,
                "verdict": "reject",
                "reasons": ["no_train_edge"],
                "train": {"trades": 0, "total_return": 0.0},
            }
        ]

    monkeypatch.setattr(rc, "validate_batch", fake_validate)

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=[hypothesis],
        coverage_now="2026-01-10T00:00:00Z",
        experiment_memory=FakeExperimentMemory(),
        log_path=None,
    )

    assert report["ok"] is True
    assert report["coverage"]["ok"] is True
    assert report["coverage"]["actual"]["latest"] == "2026-01-10T00:00:00+00:00"
    assert report["rows"] == 7
    assert report["protected_epoch_selection"]["end"] == "2026-01-07 00:00:00+00:00"
    assert captured["validation_latest"] == pd.Timestamp("2026-01-07T00:00:00Z")
    assert pd.Timestamp(captured["adaptive_window"]["validation"]["end"]) < pd.Timestamp(
        "2026-01-08T00:00:00Z"
    )


def test_small_unprotected_epoch_defers_until_minimum_sample_capacity(monkeypatch):
    scenario = coverage_scenario(end="2026-01-06")
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    full_frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6, freq="1D", tz="UTC")}
    )

    class CapacityOnlyMemory:
        def protected_intervals(self, **_kwargs):
            return (
                {
                    "interval_key": "sealed",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                    "start": str(full_frame["timestamp"].iloc[0]),
                    "end": str(full_frame["timestamp"].iloc[-2]),
                },
            )

        def __getattr__(self, name):
            raise AssertionError(
                f"experiment memory must not be used after capacity deferral: {name}"
            )

    monkeypatch.setattr(
        rc,
        "_scenario_indicator_coverage_status",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(rc, "_missing_columns_for_hypothesis", lambda *args, **kwargs: {})
    monkeypatch.setattr(rc, "build_aligned_frame", lambda *args, **kwargs: full_frame.copy())
    monkeypatch.setattr(rc, "_hypothesis_feature_timeframes", lambda _hypotheses: ("5m",))

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=[hypothesis],
        selection={"offset": 4, "next_offset": 5, "selected": 1},
        coverage_now="2026-01-06T00:00:00Z",
        experiment_memory=CapacityOnlyMemory(),
        log_path=None,
    )

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["deferred"] is True
    assert report["reason"] == "unprotected_epoch_unavailable"
    assert report["selection"]["offset"] == 4
    assert report["hypotheses"] == 0
    assert report["holdout_exposed_ids"] == []
    assert report["unprotected_epoch_capacity"] == {
        "ok": False,
        "requirements": {"minimum_span_days": 4.0, "minimum_rows": 5},
        "actual": {
            "earliest": "2026-01-06T00:00:00+00:00",
            "latest": "2026-01-06T00:00:00+00:00",
            "span_days": 0.0,
            "rows": 1,
        },
        "checks": {"span": False, "rows": False},
        "failed_checks": ["span", "rows"],
    }


def test_stale_full_aligned_source_fails_before_protected_epoch_selection(monkeypatch):
    scenario = coverage_scenario()
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    stale_frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=8, freq="1D", tz="UTC")}
    )

    class SelectionMustNotRun:
        def protected_intervals(self, **_kwargs):
            raise AssertionError("protected epoch selection must follow full-source coverage")

    monkeypatch.setattr(
        rc,
        "_scenario_indicator_coverage_status",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(rc, "_missing_columns_for_hypothesis", lambda *args, **kwargs: {})
    monkeypatch.setattr(rc, "build_aligned_frame", lambda *args, **kwargs: stale_frame.copy())
    monkeypatch.setattr(
        rc,
        "validate_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stale aligned history must not reach validation")
        ),
    )

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=[hypothesis],
        coverage_now="2026-01-10T00:00:00Z",
        experiment_memory=SelectionMustNotRun(),
        log_path=None,
    )

    assert report["ok"] is False
    assert report["reason"] == "insufficient_history_coverage"
    assert report["coverage"]["failed_checks"] == ["latest"]
    assert report["coverage"]["actual"]["latest"] == "2026-01-08T00:00:00+00:00"


def test_coverage_failure_blocks_exports_and_is_visible_in_summary():
    scenario_report = {
        "ok": False,
        "skipped": True,
        "reason": "insufficient_history_coverage",
        "name": "active_income_5m_guarded",
        "product": "active_income",
        "opportunity_type": "day_trading",
        "hypotheses": 0,
        "keepers": 0,
        "top_reasons": {"insufficient_history_coverage": 1},
    }
    summary = rc._summarize_cycle([scenario_report], [])

    assert summary["scenario_errors"] == 1
    assert summary["coverage_failures"] == 1
    assert summary["coverage_failed_scenarios"] == ["active_income_5m_guarded"]
    assert any(
        "bootstrap the required direct timeframe history" in item
        for item in summary["next_actions"]
    )


def test_operator_and_health_reports_surface_history_blocker():
    summary = {
        "scenarios": 1,
        "scenario_errors": 1,
        "coverage_failures": 1,
        "coverage_failed_scenarios": ["active_income_5m_guarded"],
        "top_reasons": {"insufficient_history_coverage": 1},
        "next_actions": ["bootstrap the required direct timeframe history"],
    }
    report = {
        "generated_at": "2026-01-08T00:00:01Z",
        "ok": True,
        "status_heartbeat": {
            "fresh": True,
            "generated_at": "2026-01-08T00:00:00Z",
            "age_seconds": 1,
            "limit_seconds": 300,
        },
        "scheduled_jobs": [],
        "products": [],
        "jobs": [],
        "research_cycle": {
            "ok": False,
            "generated_at": "2026-01-08T00:00:00Z",
            "summary": summary,
        },
    }

    markdown = render_operator_markdown(report)
    health = evaluate_health(report, now_ts=1_767_830_400)

    assert "history blockers 1 (active_income_5m_guarded)" in markdown
    warning = next(
        item
        for item in health["warnings"]
        if item["code"] == "research_history_coverage_insufficient"
    )
    assert warning["detail"]["scenarios"] == ["active_income_5m_guarded"]


def test_history_coverage_marker_prevents_legacy_timestamp_skip(tmp_path, monkeypatch):
    scenario = coverage_scenario(coverage_min_rows=5)
    market_statuses = {
        "futures": {
            "ok": True,
            "last_timestamp": "2026-01-08T00:00:00Z",
            "rows": 10,
            "path": "BTCUSDT_1m.parquet",
        }
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "last_market_marker": rc._market_data_skip_marker(market_statuses),
                "last_mutation_batch_marker": None,
            }
        ),
        encoding="utf-8",
    )
    coverage = {
        "ok": True,
        "actual": {
            "earliest": "2026-01-01T00:00:00Z",
            "latest": "2026-01-08T00:00:00Z",
            "span_days": 7,
            "rows": 100,
        },
        "failed_checks": [],
        "path": "BTCUSDT_5m_all_indicators.parquet",
    }
    calls = []
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: market_statuses)
    monkeypatch.setattr(rc, "_scenario_indicator_coverage_status", lambda *args, **kwargs: coverage)
    monkeypatch.setattr(
        rc,
        "run_validation_scenario",
        lambda selected, **kwargs: calls.append(selected.name)
        or {
            "ok": True,
            "name": selected.name,
            "product": selected.product,
            "market": selected.market,
            "opportunity_type": selected.opportunity_type,
            "hypotheses": 1,
            "keepers": 0,
            "selection": kwargs.get("selection"),
            "verdicts": {"reject": 1},
        },
    )

    first = rc.run_research_cycle(
        scenarios=(scenario,),
        state_path=state_path,
        output_path=tmp_path / "cycle.json",
        log_path=tmp_path / "log.jsonl",
    )
    persisted = json.loads(state_path.read_text())
    second = rc.run_research_cycle(
        scenarios=(scenario,),
        state_path=state_path,
        output_path=tmp_path / "cycle.json",
        log_path=tmp_path / "log.jsonl",
    )

    assert first["skipped"] is False
    assert calls == [scenario.name]
    assert persisted["last_history_coverage_marker"] == rc._history_coverage_skip_marker(
        {scenario.name: coverage}
    )
    assert second["skipped"] is True


def test_one_history_failure_blocks_every_scenario_before_holdout(tmp_path, monkeypatch):
    futures = coverage_scenario(name="futures_ready")
    spot = coverage_scenario(
        name="spot_shallow",
        product="btc_accumulation",
        market="spot",
        pnl_unit="btc",
        position=True,
        base_tf="1h",
        candidate_set="position",
    )
    statuses = {
        "futures": {"ok": True, "last_timestamp": "2026-01-08T00:00:00Z", "rows": 10},
        "spot": {"ok": True, "last_timestamp": "2026-01-08T00:00:00Z", "rows": 10},
    }
    passing = {
        "ok": True,
        "actual": {
            "earliest": "2026-01-01T00:00:00Z",
            "latest": "2026-01-08T00:00:00Z",
            "span_days": 7,
            "rows": 10,
        },
        "failed_checks": [],
    }
    failing = {
        "ok": False,
        "actual": {
            "earliest": "2026-01-07T00:00:00Z",
            "latest": "2026-01-08T00:00:00Z",
            "span_days": 1,
            "rows": 2,
        },
        "failed_checks": ["earliest", "span", "rows"],
        "remediation": {"action": "bootstrap_research_history", "command": ["bootstrap"]},
    }
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: statuses)
    monkeypatch.setattr(
        rc,
        "_scenario_indicator_coverage_status",
        lambda scenario, **kwargs: failing if scenario.name == spot.name else passing,
    )
    monkeypatch.setattr(
        rc,
        "run_validation_scenario",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no scenario may spend holdout while the history gate is blocked")
        ),
    )

    report = rc.run_research_cycle(
        scenarios=(futures, spot),
        state_path=tmp_path / "state.json",
        output_path=tmp_path / "cycle.json",
        log_path=tmp_path / "log.jsonl",
    )

    by_name = {item["name"]: item for item in report["scenarios"]}
    assert report["ok"] is False
    assert report["exports"] == []
    assert by_name["spot_shallow"]["reason"] == "insufficient_history_coverage"
    assert by_name["futures_ready"]["reason"] == "history_coverage_gate_blocked"
    assert by_name["futures_ready"]["blocked_by_scenarios"] == ["spot_shallow"]
