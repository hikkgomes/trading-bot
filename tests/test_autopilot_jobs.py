import json
import sys

import pytest

import src.autopilot.jobs as jobs_module
from src.autopilot.config import JobConfig
from src.autopilot.jobs import (
    effective_job_cadence_seconds,
    job_definition_fingerprint,
    job_due,
    load_job_state,
    parse_structured_stdout,
    run_due_jobs,
    save_job_state,
)


def job(tmp_path, **overrides):
    payload = {
        "name": "smoke",
        "enabled": True,
        "command": [sys.executable, "-c", "print('ok')"],
        "cadence_seconds": 60,
        "timeout_seconds": 5,
        "working_dir": tmp_path,
    }
    payload.update(overrides)
    return JobConfig(**payload)


def test_job_due_first_run_and_cadence(tmp_path):
    state = {"version": 1, "jobs": {}}
    cfg = job(tmp_path, cadence_seconds=60)

    assert job_due(cfg, state, now=100.0)

    state["jobs"]["smoke"] = {"last_started_ts": 90.0}
    assert not job_due(cfg, state, now=100.0)
    assert job_due(cfg, state, now=151.0)


def test_failed_job_retries_with_bounded_shorter_cadence(tmp_path):
    cfg = job(tmp_path, cadence_seconds=6 * 60 * 60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": 100.0, "last_ok": False}}}

    assert not job_due(cfg, state, now=900.0)
    assert job_due(cfg, state, now=1000.0)


def test_failed_job_retry_never_exceeds_configured_short_cadence(tmp_path):
    cfg = job(tmp_path, cadence_seconds=60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": 100.0, "last_ok": False}}}

    assert not job_due(cfg, state, now=159.0)
    assert job_due(cfg, state, now=160.0)


def test_repeated_failed_job_retries_with_exponential_backoff(tmp_path):
    cfg = job(tmp_path, cadence_seconds=6 * 60 * 60)
    entry = {"last_started_ts": 100.0, "last_ok": False, "consecutive_failures": 3}
    state = {"version": 1, "jobs": {"smoke": entry}}

    assert effective_job_cadence_seconds(cfg, entry) == 60 * 60
    assert not job_due(cfg, state, now=100.0 + 59 * 60)
    assert job_due(cfg, state, now=100.0 + 60 * 60)


def test_repeated_failed_job_backoff_is_capped_by_job_cadence(tmp_path):
    cfg = job(tmp_path, cadence_seconds=30 * 60)
    entry = {"last_started_ts": 100.0, "last_ok": False, "consecutive_failures": 5}

    assert effective_job_cadence_seconds(cfg, entry) == 30 * 60


def test_job_due_when_persisted_definition_changes(tmp_path):
    original = job(tmp_path, command=[sys.executable, "-c", "print('old')"])
    changed = job(tmp_path, command=[sys.executable, "-c", "print('new')"])
    state = {
        "version": 1,
        "jobs": {
            "smoke": {
                "last_started_ts": 100.0,
                "last_ok": True,
                "definition_fingerprint": job_definition_fingerprint(original),
            }
        },
    }

    assert job_due(changed, state, now=120.0)


def test_missing_definition_fingerprint_keeps_existing_cadence_for_migration(tmp_path):
    cfg = job(tmp_path, cadence_seconds=60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": 100.0, "last_ok": True}}}

    assert not job_due(cfg, state, now=120.0)


def test_job_due_when_last_started_timestamp_is_malformed(tmp_path):
    cfg = job(tmp_path, cadence_seconds=60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": "not-a-number", "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


def test_job_due_when_last_started_timestamp_is_future(tmp_path):
    cfg = job(tmp_path, cadence_seconds=60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": 9999.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


@pytest.mark.parametrize("last_started_ts", [float("nan"), float("inf"), float("-inf")])
def test_job_due_when_last_started_timestamp_is_nonfinite(tmp_path, last_started_ts):
    cfg = job(tmp_path, cadence_seconds=60)
    state = {"version": 1, "jobs": {"smoke": {"last_started_ts": last_started_ts, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


@pytest.mark.parametrize(
    ("now", "message"),
    [
        ("bad", "job scheduler now timestamp must be numeric"),
        (float("nan"), "job scheduler now timestamp must be finite and non-negative"),
        (float("inf"), "job scheduler now timestamp must be finite and non-negative"),
        (-1.0, "job scheduler now timestamp must be finite and non-negative"),
    ],
)
def test_job_due_rejects_invalid_now_timestamp(tmp_path, now, message):
    with pytest.raises(ValueError, match=message):
        job_due(job(tmp_path), {"version": 1, "jobs": {}}, now=now)


def test_disabled_job_is_not_due(tmp_path):
    assert not job_due(job(tmp_path, enabled=False), {"version": 1, "jobs": {}}, now=100.0)


def test_missing_seed_makes_bootstrap_market_data_job_due(tmp_path, monkeypatch):
    seed_path = tmp_path / "missing" / "BTCUSDT_1m.parquet"
    monkeypatch.setattr("src.autopilot.jobs.default_1m_candle_path", lambda market: seed_path)
    cfg = job(
        tmp_path,
        name="market_data_update_spot",
        command=[
            sys.executable,
            "-m",
            "src.update_candles",
            "--market",
            "spot",
            "--bootstrap-days",
            "365",
        ],
        cadence_seconds=6 * 60 * 60,
    )
    state = {
        "version": 1,
        "jobs": {
            "market_data_update_spot": {
                "last_started_ts": 100.0,
                "last_ok": True,
            }
        },
    }

    assert job_due(cfg, state, now=120.0)


def test_missing_seed_makes_native_history_job_due(tmp_path, monkeypatch):
    seed_path = tmp_path / "missing" / "BTCUSDT_1m.parquet"
    monkeypatch.setattr("src.autopilot.jobs.default_1m_candle_path", lambda market: seed_path)
    cfg = job(
        tmp_path,
        name="market_data_update_spot",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.history_bootstrap",
            "--market",
            "spot",
            "--timeframes",
            "1m",
            "1h",
        ],
        cadence_seconds=6 * 60 * 60,
    )
    state = {
        "version": 1,
        "jobs": {
            "market_data_update_spot": {
                "last_started_ts": 100.0,
                "last_ok": True,
            }
        },
    }

    assert job_due(cfg, state, now=120.0)


def test_missing_seed_does_not_bypass_failed_job_retry(tmp_path, monkeypatch):
    seed_path = tmp_path / "missing" / "BTCUSDT_1m.parquet"
    monkeypatch.setattr("src.autopilot.jobs.default_1m_candle_path", lambda market: seed_path)
    cfg = job(
        tmp_path,
        name="market_data_update_spot",
        command=[
            sys.executable,
            "-m",
            "src.update_candles",
            "--market",
            "spot",
            "--bootstrap-days",
            "365",
        ],
        cadence_seconds=6 * 60 * 60,
    )
    state = {
        "version": 1,
        "jobs": {
            "market_data_update_spot": {
                "last_started_ts": 100.0,
                "last_ok": False,
            }
        },
    }

    assert not job_due(cfg, state, now=120.0)
    assert job_due(cfg, state, now=1000.0)


def test_missing_seed_does_not_make_non_bootstrap_job_due(tmp_path, monkeypatch):
    seed_path = tmp_path / "missing" / "BTCUSDT_1m.parquet"
    monkeypatch.setattr("src.autopilot.jobs.default_1m_candle_path", lambda market: seed_path)
    cfg = job(
        tmp_path,
        name="market_data_update_spot",
        command=[
            sys.executable,
            "-m",
            "src.update_candles",
            "--market",
            "spot",
            "--skip-if-missing",
        ],
        cadence_seconds=6 * 60 * 60,
    )
    state = {
        "version": 1,
        "jobs": {
            "market_data_update_spot": {
                "last_started_ts": 100.0,
                "last_ok": True,
            }
        },
    }

    assert not job_due(cfg, state, now=120.0)


def test_mutation_plan_due_when_research_cycle_source_is_newer(tmp_path):
    research_cycle = tmp_path / "research_cycle.json"
    mutation_plan = tmp_path / "mutation_plan.json"
    research_cycle.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:05:00+00:00"}),
        encoding="utf-8",
    )
    mutation_plan.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="mutation_plan",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.mutation_plan",
            "--input",
            str(research_cycle),
            "--output",
            str(mutation_plan),
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"mutation_plan": {"last_started_ts": 100.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


def test_mutation_plan_due_with_inline_input_output_flags(tmp_path):
    research_cycle = tmp_path / "research_cycle.json"
    mutation_plan = tmp_path / "mutation_plan.json"
    research_cycle.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:05:00+00:00"}),
        encoding="utf-8",
    )
    mutation_plan.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="mutation_plan",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.mutation_plan",
            f"--input={research_cycle}",
            f"--output={mutation_plan}",
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"mutation_plan": {"last_started_ts": 100.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


def test_mutation_plan_not_due_when_research_cycle_source_is_current(tmp_path):
    research_cycle = tmp_path / "research_cycle.json"
    mutation_plan = tmp_path / "mutation_plan.json"
    research_cycle.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:05:00+00:00"}),
        encoding="utf-8",
    )
    mutation_plan.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "source": {"research_generated_at": "2026-01-01T01:05:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="mutation_plan",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.mutation_plan",
            "--input",
            str(research_cycle),
            "--output",
            str(mutation_plan),
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"mutation_plan": {"last_started_ts": 100.0, "last_ok": True}}}

    assert not job_due(cfg, state, now=120.0)


def test_mutation_batch_due_when_plan_source_is_newer_or_output_missing(tmp_path):
    mutation_plan = tmp_path / "mutation_plan.json"
    mutation_batch = tmp_path / "mutation_hypotheses.json"
    mutation_plan.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:06:00+00:00"}),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="mutation_batch",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.mutation_batch",
            "--input",
            str(mutation_plan),
            "--output",
            str(mutation_batch),
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"mutation_batch": {"last_started_ts": 100.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)

    mutation_batch.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:07:00+00:00",
                "source": {"plan_generated_at": "2026-01-01T00:56:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    assert job_due(cfg, state, now=120.0)

    mutation_batch.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:07:00+00:00",
                "source": {"plan_generated_at": "2026-01-01T01:06:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    assert not job_due(cfg, state, now=120.0)


def test_research_cycle_due_when_mutation_batch_has_not_been_evaluated(tmp_path):
    research_state = tmp_path / "research_cycle_state.json"
    mutation_batch = tmp_path / "mutation_hypotheses.json"
    mutation_batch.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:07:00+00:00",
                "count": 12,
                "status": None,
                "summary": {"hypotheses": 12},
            }
        ),
        encoding="utf-8",
    )
    research_state.write_text(
        json.dumps(
            {
                "version": 1,
                "last_mutation_batch_marker": json.dumps(
                    {
                        "status": "loaded",
                        "generated_at": "2026-01-01T00:57:00+00:00",
                        "hypotheses": 12,
                        "scenarios": 5,
                    },
                    sort_keys=True,
                ),
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--state",
            str(research_state),
            "--include-mutations",
            "--mutation-batch",
            str(mutation_batch),
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"research_cycle": {"last_started_ts": 100.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)

    research_state.write_text(
        json.dumps(
            {
                "version": 1,
                "last_mutation_batch_marker": json.dumps(
                    {
                        "status": None,
                        "generated_at": "2026-01-01T01:07:00+00:00",
                        "hypotheses": 12,
                        "scenarios": 5,
                    },
                    sort_keys=True,
                ),
            }
        ),
        encoding="utf-8",
    )

    assert not job_due(cfg, state, now=120.0)


def test_research_cycle_mutation_batch_due_requires_include_mutations(tmp_path):
    research_state = tmp_path / "research_cycle_state.json"
    mutation_batch = tmp_path / "mutation_hypotheses.json"
    research_state.write_text(json.dumps({"version": 1}), encoding="utf-8")
    mutation_batch.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:07:00+00:00", "count": 2}),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--state",
            str(research_state),
            "--mutation-batch",
            str(mutation_batch),
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"research_cycle": {"last_started_ts": 100.0, "last_ok": True}}}

    assert not job_due(cfg, state, now=120.0)


def test_research_cycle_due_exactly_once_for_new_generated_population(tmp_path):
    research_state = tmp_path / "research_cycle_state.json"
    generated_batch = tmp_path / "generated_hypotheses.json"
    generated_batch.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-07-10T00:00:00+00:00",
                "summary": {
                    "hypotheses": 12,
                    "by_space": {"active_day": 6, "btc_position": 6},
                    "cumulative_trials": 240,
                },
                "hypotheses": [{}] * 12,
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--state",
            str(research_state),
            "--include-generated",
            "--generated-only",
            "--generated-batch",
            str(generated_batch),
        ],
        cadence_seconds=86400,
    )
    state = {
        "version": 1,
        "jobs": {"research_cycle": {"last_started_ts": 100.0, "last_ok": True}},
    }

    assert job_due(cfg, state, now=120.0)

    research_state.write_text(
        json.dumps(
            {
                "last_generated_batch_marker": json.dumps(
                    {
                        "status": "loaded",
                        "generated_at": "2026-07-10T00:00:00+00:00",
                        "hypotheses": 12,
                        "scenarios": 2,
                        "cumulative_trials": 240,
                    },
                    sort_keys=True,
                )
            }
        ),
        encoding="utf-8",
    )

    assert not job_due(cfg, state, now=120.0)


def test_failed_research_cycle_respects_backoff_for_pending_generated_population(tmp_path):
    research_state = tmp_path / "research_cycle_state.json"
    generated_batch = tmp_path / "generated_hypotheses.json"
    generated_batch.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-07-10T00:00:00+00:00",
                "summary": {
                    "hypotheses": 12,
                    "by_space": {"active_day": 12},
                    "cumulative_trials": 240,
                },
                "hypotheses": [{}] * 12,
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--state",
            str(research_state),
            "--include-generated",
            "--generated-only",
            "--generated-batch",
            str(generated_batch),
        ],
        cadence_seconds=86400,
    )
    state = {
        "version": 1,
        "jobs": {
            "research_cycle": {
                "last_started_ts": 100.0,
                "last_ok": False,
                "consecutive_failures": 1,
            }
        },
    }

    assert not job_due(cfg, state, now=120.0)
    assert job_due(cfg, state, now=1000.0)


def test_universe_history_due_when_market_snapshot_has_not_been_bootstrapped(tmp_path):
    universe = tmp_path / "market_universe.json"
    output = tmp_path / "universe_history.json"
    universe.write_text(
        json.dumps({"snapshot": {"id": "sha256:" + "1" * 64}}),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="market_data_update_universe",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.universe_history",
            "--market-universe-report",
            str(universe),
            "--output",
            str(output),
        ],
        cadence_seconds=86400,
    )
    state = {
        "version": 1,
        "jobs": {
            cfg.name: {
                "last_started_ts": 100.0,
                "last_ok": True,
            }
        },
    }

    assert job_due(cfg, state, now=120.0)

    output.write_text(
        json.dumps(
            {
                "ok": True,
                "market_universe": {"snapshot_id": "sha256:" + "1" * 64},
            }
        ),
        encoding="utf-8",
    )
    assert not job_due(cfg, state, now=120.0)

    universe.write_text(
        json.dumps({"snapshot": {"id": "sha256:" + "2" * 64}}),
        encoding="utf-8",
    )
    assert job_due(cfg, state, now=120.0)


def test_research_cycle_waits_when_scheduler_created_next_mutation_batch(tmp_path):
    research_state = tmp_path / "research_cycle_state.json"
    mutation_batch = tmp_path / "mutation_hypotheses.json"
    mutation_batch.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:07:00+00:00", "count": 12}),
        encoding="utf-8",
    )
    research_state.write_text(
        json.dumps(
            {
                "version": 1,
                "last_mutation_batch_marker": json.dumps(
                    {
                        "status": None,
                        "generated_at": "2026-01-01T00:57:00+00:00",
                        "hypotheses": 12,
                    },
                    sort_keys=True,
                ),
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--state",
            str(research_state),
            "--include-mutations",
            "--mutation-batch",
            str(mutation_batch),
        ],
        cadence_seconds=86400,
    )
    state = {
        "version": 1,
        "jobs": {
            "research_cycle": {"last_started_ts": 100.0, "last_ok": True},
            "mutation_batch": {"last_started_ts": 110.0, "last_ok": True},
        },
    }

    assert not job_due(cfg, state, now=120.0)


def test_research_cycle_due_when_open_position_export_block_clears(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "research_cycle.json").write_text(
        json.dumps(
            {
                "ok": True,
                "exports": [
                    {
                        "product": "active_income",
                        "exported": False,
                        "reason": "open_positions_block_export",
                        "open_positions": ["active_bootstrap_short_rsi_5m"],
                    }
                ],
                "summary": {"export_reasons": {"open_positions_block_export": 1}},
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "active_income_state.json").write_text(
        json.dumps({"version": 1, "open_positions": {}}),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--output",
            "runtime/research_cycle.json",
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"research_cycle": {"last_started_ts": 100.0, "last_ok": True}}}

    assert job_due(cfg, state, now=120.0)


def test_research_cycle_waits_when_open_position_export_remains_blocked(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "research_cycle.json").write_text(
        json.dumps(
            {
                "ok": True,
                "exports": [
                    {
                        "product": "active_income",
                        "exported": False,
                        "reason": "open_positions_block_export",
                        "open_positions": ["active_bootstrap_short_rsi_5m"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "active_income_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "open_positions": {
                    "active_bootstrap_short_rsi_5m": {"side": "short", "qty": 0.001}
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = job(
        tmp_path,
        name="research_cycle",
        command=[
            sys.executable,
            "-m",
            "src.autopilot.research_cycle",
            "--output",
            "runtime/research_cycle.json",
        ],
        cadence_seconds=86400,
    )
    state = {"version": 1, "jobs": {"research_cycle": {"last_started_ts": 100.0, "last_ok": True}}}

    assert not job_due(cfg, state, now=120.0)


def test_run_due_jobs_persists_success(tmp_path):
    state_path = tmp_path / "jobs.json"

    results = run_due_jobs([job(tmp_path)], state_path, now=100.0)

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert "ok" in results[0]["stdout_tail"]
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is True
    assert state["jobs"]["smoke"]["definition_fingerprint"] == job_definition_fingerprint(
        job(tmp_path)
    )


def test_load_job_state_rejects_non_object_payload(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text("[]", encoding="utf-8")

    try:
        load_job_state(state_path)
    except ValueError as exc:
        assert str(exc) == f"job state must be a JSON object: {state_path}"
    else:
        raise AssertionError("expected ValueError")


def test_load_job_state_rejects_non_object_jobs_map(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text(json.dumps({"version": 1, "jobs": []}), encoding="utf-8")

    try:
        load_job_state(state_path)
    except ValueError as exc:
        assert str(exc) == f"job state jobs must be a JSON object: {state_path}"
    else:
        raise AssertionError("expected ValueError")


def test_load_job_state_rejects_symlink_without_trusting_target(tmp_path):
    state_path = tmp_path / "jobs.json"
    target = tmp_path / "external_jobs.json"
    target.write_text(json.dumps({"version": 1, "jobs": {"external": {}}}), encoding="utf-8")
    state_path.symlink_to(target)

    with pytest.raises(ValueError, match="job state must not be a symlink"):
        load_job_state(state_path)

    assert state_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["jobs"] == {"external": {}}


def test_save_job_state_rejects_symlink_without_touching_target(tmp_path):
    state_path = tmp_path / "jobs.json"
    target = tmp_path / "external_jobs.json"
    target.write_text(json.dumps({"version": 1, "jobs": {"external": {}}}), encoding="utf-8")
    state_path.symlink_to(target)

    with pytest.raises(ValueError, match="job state must not be a symlink"):
        save_job_state(state_path, {"version": 1, "jobs": {"new": {}}})

    assert state_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["jobs"] == {"external": {}}


def test_run_due_jobs_skips_until_cadence(tmp_path):
    state_path = tmp_path / "jobs.json"
    cfg = job(tmp_path, cadence_seconds=60)

    assert len(run_due_jobs([cfg], state_path, now=100.0)) == 1
    assert run_due_jobs([cfg], state_path, now=120.0) == []


def test_run_due_jobs_defers_due_jobs_after_cycle_limit(tmp_path):
    state_path = tmp_path / "jobs.json"
    first = job(tmp_path, name="first")
    second = job(tmp_path, name="second")

    results = run_due_jobs([first, second], state_path, now=100.0, max_jobs_per_cycle=1)

    assert [item["name"] for item in results] == ["first", "second"]
    assert results[0]["ok"] is True
    assert results[0].get("skipped") is not True
    assert results[1] == {
        "name": "second",
        "ok": True,
        "skipped": True,
        "reason": "cycle_job_limit",
        "started_at": results[1]["started_at"],
        "started_ts": 100.0,
    }
    state = load_job_state(state_path)
    assert set(state["jobs"]) == {"first", "second"}
    assert state["jobs"]["second"]["last_deferred_reason"] == "cycle_job_limit"
    assert state["jobs"]["second"]["last_deferred_ts"] == 100.0
    assert state["jobs"]["second"]["consecutive_deferrals"] == 1
    assert "last_started_ts" not in state["jobs"]["second"]
    assert state["scheduler"]["next_index"] == 1


def test_run_due_jobs_rotates_after_cycle_limited_execution(tmp_path):
    state_path = tmp_path / "jobs.json"
    first = job(tmp_path, name="first")
    second = job(tmp_path, name="second")

    first_results = run_due_jobs([first, second], state_path, now=100.0, max_jobs_per_cycle=1)
    second_results = run_due_jobs([first, second], state_path, now=101.0, max_jobs_per_cycle=1)

    assert [item["name"] for item in first_results] == ["first", "second"]
    assert first_results[0].get("skipped") is not True
    assert first_results[1]["reason"] == "cycle_job_limit"
    assert [item["name"] for item in second_results] == ["second"]
    assert second_results[0].get("skipped") is not True
    state = load_job_state(state_path)
    assert state["jobs"]["second"]["last_started_ts"] == 101.0
    assert state["scheduler"]["next_index"] == 0


def test_run_due_jobs_repairs_invalid_scheduler_cursor(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text(
        json.dumps({"version": 1, "jobs": {}, "scheduler": {"next_index": "bad"}}), encoding="utf-8"
    )

    results = run_due_jobs(
        [job(tmp_path, name="first"), job(tmp_path, name="second")],
        state_path,
        now=100.0,
        max_jobs_per_cycle=1,
    )

    assert [item["name"] for item in results] == ["first", "second"]
    assert load_job_state(state_path)["scheduler"]["next_index"] == 1


def test_run_due_jobs_paused_jobs_do_not_consume_cycle_limit(tmp_path):
    state_path = tmp_path / "jobs.json"
    paused = job(tmp_path, name="paused")
    active = job(tmp_path, name="active")

    results = run_due_jobs(
        [paused, active],
        state_path,
        now=100.0,
        paused_jobs={"paused"},
        max_jobs_per_cycle=1,
    )

    assert [item["name"] for item in results] == ["paused", "active"]
    assert results[0]["reason"] == "paused"
    assert results[1]["ok"] is True
    assert results[1].get("skipped") is not True
    assert set(load_job_state(state_path)["jobs"]) == {"active"}


def test_run_due_jobs_rejects_non_positive_cycle_limit(tmp_path):
    with pytest.raises(ValueError, match="max_jobs_per_cycle must be positive"):
        run_due_jobs([job(tmp_path)], tmp_path / "jobs.json", max_jobs_per_cycle=0)


def test_run_due_jobs_reports_failure(tmp_path):
    cfg = job(tmp_path, command=[sys.executable, "-c", "raise SystemExit(7)"])

    results = run_due_jobs([cfg], tmp_path / "jobs.json", now=100.0)

    assert results[0]["ok"] is False
    assert results[0]["returncode"] == 7


def test_run_due_jobs_increments_and_resets_consecutive_failures(tmp_path):
    state_path = tmp_path / "jobs.json"
    failing = job(tmp_path, command=[sys.executable, "-c", "raise SystemExit(7)"])
    passing = job(tmp_path, command=[sys.executable, "-c", "print('ok')"])

    run_due_jobs([failing], state_path, now=100.0)
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["consecutive_failures"] == 1

    state["jobs"]["smoke"]["last_started_ts"] = -10_000.0
    state["jobs"]["smoke"]["last_ok"] = False
    state["jobs"]["smoke"]["consecutive_failures"] = 4
    state_path.write_text(json.dumps(state), encoding="utf-8")

    run_due_jobs([passing], state_path, now=100.0)
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is True
    assert state["jobs"]["smoke"]["consecutive_failures"] == 0


def test_run_due_jobs_repairs_malformed_job_entry(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "smoke": {
                        "last_started_ts": "not-a-number",
                        "last_ok": False,
                        "consecutive_failures": "bad",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = run_due_jobs([job(tmp_path)], state_path, now=100.0)

    assert len(results) == 1
    assert results[0]["ok"] is True
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_started_ts"] == results[0]["started_ts"]
    assert state["jobs"]["smoke"]["consecutive_failures"] == 0


def test_parse_structured_stdout_reads_json_objects_only():
    assert parse_structured_stdout('{"ok": false, "reason": "empty_seed_dataset"}') == {
        "ok": False,
        "reason": "empty_seed_dataset",
    }
    assert parse_structured_stdout(
        '2026-01-01 00:00:00 INFO starting\nprogress 50%\n{"ok": true, "reason": "refreshed"}\n'
    ) == {"ok": True, "reason": "refreshed"}
    assert parse_structured_stdout(
        "2026-01-01 00:00:00 INFO starting\n"
        "progress 50%\n"
        "{\n"
        '  "ok": true,\n'
        '  "reason": "refreshed",\n'
        '  "summary": {\n'
        '    "rows": 12\n'
        "  }\n"
        "}\n"
    ) == {"ok": True, "reason": "refreshed", "summary": {"rows": 12}}
    assert parse_structured_stdout("plain output") is None
    assert parse_structured_stdout("[1, 2, 3]") is None
    assert parse_structured_stdout("log line\n[1, 2, 3]") is None


def test_run_due_jobs_treats_structured_failed_report_as_failure(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {"ok": False, "reason": "empty_seed_dataset"}
    cfg = job(
        tmp_path,
        command=[sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is False
    assert results[0]["returncode"] == 0
    assert results[0]["structured_report"] == payload
    assert results[0]["error"] == "empty_seed_dataset"
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is False
    assert state["jobs"]["smoke"]["last_returncode"] == 0
    assert state["jobs"]["smoke"]["last_reason"] == "empty_seed_dataset"
    assert state["jobs"]["smoke"]["last_error"] == "empty_seed_dataset"


def test_run_due_jobs_persists_structured_success_report(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {"ok": True, "reason": "missing_seed_dataset", "skipped": True}
    cfg = job(
        tmp_path,
        command=[sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is True
    assert results[0]["structured_report"] == payload
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is True
    assert state["jobs"]["smoke"]["last_reason"] == "missing_seed_dataset"
    assert "last_error" not in state["jobs"]["smoke"]


def test_run_due_jobs_parses_structured_report_after_log_lines(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {"ok": True, "reason": "refreshed", "summary": {"rows": 12}}
    cfg = job(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            (
                "import json; "
                "print('starting update'); "
                "print('downloaded 12 rows'); "
                f"print(json.dumps({payload!r}))"
            ),
        ],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is True
    assert results[0]["structured_report"] == payload
    assert "starting update" in results[0]["stdout_tail"]
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_reason"] == "refreshed"


def test_run_due_jobs_parses_pretty_structured_report_after_log_lines(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {"ok": True, "reason": "refreshed", "summary": {"rows": 12}}
    cfg = job(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            (
                "import json; "
                "print('starting update'); "
                "print('downloaded 12 rows'); "
                f"print(json.dumps({payload!r}, indent=2, sort_keys=True))"
            ),
        ],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is True
    assert results[0]["structured_report"] == payload
    assert "downloaded 12 rows" in results[0]["stdout_tail"]
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_reason"] == "refreshed"


def test_run_due_jobs_summarizes_large_structured_report_for_status(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {
        "ok": True,
        "reason": "large_report",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"proposals": 300},
        "source": {"research_generated_at": "2026-01-01T00:00:00+00:00"},
        "proposals": [{"id": f"p{i}", "details": "x" * 200} for i in range(60)],
    }
    cfg = job(
        tmp_path,
        command=[sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert "structured_report" not in results[0]
    assert results[0]["structured_report_truncated"] is True
    assert results[0]["structured_report_bytes"] > 4000
    assert results[0]["structured_report_summary"] == {
        "ok": True,
        "reason": "large_report",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"proposals": 300},
        "source": {"research_generated_at": "2026-01-01T00:00:00+00:00"},
        "proposals_count": 60,
    }
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_reason"] == "large_report"


def test_run_due_jobs_summarizes_large_structured_report_errors_for_status(tmp_path):
    state_path = tmp_path / "jobs.json"
    payload = {
        "ok": False,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"errors": 20},
        "errors": [
            {"task": "alert_state", "error": "ValueError: alert state path must not be a symlink"},
            {"task": "control_audit", "error": "OSError: disk full"},
            {"task": "experiment_log", "error": "OSError: permission denied"},
            {"task": "quarantine", "error": "OSError: quota exceeded"},
        ],
        "artifacts": [{"path": f"outputs/search_{idx}", "details": "x" * 200} for idx in range(60)],
    }
    cfg = job(
        tmp_path,
        command=[sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is False
    assert results[0]["returncode"] == 0
    assert results[0]["error"] == "alert_state: ValueError: alert state path must not be a symlink"
    assert "structured_report" not in results[0]
    assert results[0]["structured_report_truncated"] is True
    assert results[0]["structured_report_summary"] == {
        "ok": False,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"errors": 20},
        "errors_count": 4,
        "errors": [
            {"task": "alert_state", "error": "ValueError: alert state path must not be a symlink"},
            {"task": "control_audit", "error": "OSError: disk full"},
            {"task": "experiment_log", "error": "OSError: permission denied"},
        ],
    }
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is False
    assert (
        state["jobs"]["smoke"]["last_error"]
        == "alert_state: ValueError: alert state path must not be a symlink"
    )
    assert state["jobs"]["smoke"]["last_structured_errors_count"] == 4
    assert state["jobs"]["smoke"]["last_structured_errors"] == [
        {"task": "alert_state", "error": "ValueError: alert state path must not be a symlink"},
        {"task": "control_audit", "error": "OSError: disk full"},
        {"task": "experiment_log", "error": "OSError: permission denied"},
    ]


def test_run_due_jobs_bounds_large_stdout_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_module, "MAX_JOB_OUTPUT_CAPTURE_BYTES", 128)
    monkeypatch.setattr(jobs_module, "JOB_OUTPUT_TAIL_BYTES", 32)
    state_path = tmp_path / "jobs.json"
    cfg = job(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 300 + 'TAIL')",
        ],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is True
    assert results[0]["stdout_truncated"] is True
    assert results[0]["stdout_bytes"] == 304
    assert results[0]["stdout_tail"].endswith("TAIL")
    assert len(results[0]["stdout_tail"]) == 32
    assert "structured_report" not in results[0]
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is True
    assert state["jobs"]["smoke"]["last_stdout_truncated"] is True
    assert state["jobs"]["smoke"]["last_stdout_bytes"] == 304


def test_run_due_jobs_bounds_large_stderr_capture_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_module, "MAX_JOB_OUTPUT_CAPTURE_BYTES", 128)
    monkeypatch.setattr(jobs_module, "JOB_OUTPUT_TAIL_BYTES", 32)
    state_path = tmp_path / "jobs.json"
    cfg = job(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('e' * 300 + 'ERRTAIL'); raise SystemExit(7)",
        ],
    )

    results = run_due_jobs([cfg], state_path, now=100.0)

    assert results[0]["ok"] is False
    assert results[0]["returncode"] == 7
    assert results[0]["stderr_truncated"] is True
    assert results[0]["stderr_bytes"] == 307
    assert results[0]["stderr_tail"].endswith("ERRTAIL")
    assert len(results[0]["stderr_tail"]) == 32
    state = load_job_state(state_path)
    assert state["jobs"]["smoke"]["last_ok"] is False
    assert state["jobs"]["smoke"]["last_stderr_truncated"] is True
    assert state["jobs"]["smoke"]["last_stderr_bytes"] == 307


def test_run_due_jobs_reports_paused_without_advancing_state(tmp_path):
    state_path = tmp_path / "jobs.json"
    cfg = job(tmp_path, command=[sys.executable, "-c", "raise SystemExit(9)"])

    results = run_due_jobs([cfg], state_path, now=100.0, paused_jobs={"smoke"})

    assert results == [
        {
            "name": "smoke",
            "ok": True,
            "skipped": True,
            "reason": "paused",
            "started_at": results[0]["started_at"],
            "started_ts": 100.0,
        }
    ]
    assert load_job_state(state_path)["jobs"] == {}


def test_run_due_jobs_does_not_report_disabled_paused_jobs(tmp_path):
    results = run_due_jobs(
        [job(tmp_path, enabled=False)],
        tmp_path / "jobs.json",
        now=100.0,
        paused_jobs={"smoke"},
    )

    assert results == []
