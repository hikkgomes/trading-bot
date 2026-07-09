import json

import pytest

from src.autopilot.notifications import (
    alert_fingerprint,
    emit_alert,
    failure_detail,
    promotion_warning_detail,
    readiness_warning_detail,
    required_testnet_rehearsal_warning_detail,
    research_handoff_warning_detail,
    research_progress_warning_detail,
)


def test_emit_alert_writes_jsonl_and_cooldown(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}

    first = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=120.0,
    )
    third = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=161.0,
    )

    assert first["sent"] is True
    assert second == {
        "sent": False,
        "reason": "cooldown",
        "fingerprint": first["fingerprint"],
    }
    assert third["sent"] is True
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["detail"] == detail


def test_emit_alert_recovers_from_malformed_alert_state(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    state_file.write_text("{", encoding="utf-8")

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    alert = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["alert_state_recovered"]["path"] == str(state_file)
    assert "JSONDecodeError" in alert["alert_state_recovered"]["error"]
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][result["fingerprint"]]["last_sent_ts"] == 100.0


def test_emit_alert_ignores_malformed_cooldown_entry(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}
    fingerprint = alert_fingerprint("error", "cycle failed", detail)
    state_file.write_text(
        json.dumps({"version": 1, "alerts": {fingerprint: {"last_sent_ts": "not-a-number"}}}),
        encoding="utf-8",
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )

    assert result == {"sent": True, "fingerprint": fingerprint}
    assert len(alert_file.read_text(encoding="utf-8").splitlines()) == 1
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][fingerprint]["last_sent_ts"] == 100.0


def test_emit_alert_ignores_future_cooldown_entry_and_records_recovery(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    detail = {"products": [{"name": "active_income", "error": "failed"}]}
    fingerprint = alert_fingerprint("error", "cycle failed", detail)
    state_file.write_text(
        json.dumps({"version": 1, "alerts": {fingerprint: {"last_sent_ts": 9999.0}}}),
        encoding="utf-8",
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail=detail,
        cooldown_seconds=60,
        now=100.0,
    )

    assert result == {"sent": True, "fingerprint": fingerprint}
    alert = json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["alert_state_entry_recovered"] == {
        "fingerprint": fingerprint,
        "field": "last_sent_ts",
        "value": 9999.0,
        "reason": "future_timestamp",
    }
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["alerts"][fingerprint]["last_sent_ts"] == 100.0


@pytest.mark.parametrize(
    ("now", "cooldown_seconds", "message"),
    [
        (float("nan"), 60, "alert now timestamp must be finite and non-negative"),
        (-1.0, 60, "alert now timestamp must be finite and non-negative"),
        ("bad", 60, "alert now timestamp must be numeric"),
        (100.0, float("inf"), "alert cooldown_seconds must be finite and non-negative"),
        (100.0, -1, "alert cooldown_seconds must be finite and non-negative"),
        (100.0, "bad", "alert cooldown_seconds must be numeric"),
    ],
)
def test_emit_alert_rejects_invalid_timing_inputs(tmp_path, now, cooldown_seconds, message):
    with pytest.raises(ValueError, match=message):
        emit_alert(
            alert_file=tmp_path / "alerts.jsonl",
            state_file=tmp_path / "alert_state.json",
            severity="error",
            title="cycle failed",
            detail={"products": [{"name": "active_income", "error": "failed"}]},
            cooldown_seconds=cooldown_seconds,
            now=now,
        )


def test_alert_fingerprint_ignores_volatile_operational_fields():
    first_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 300,
                "total_bytes": 2000,
                "used_bytes": 1700,
                "min_free_bytes": 500,
            },
            {
                "product": "active_income",
                "approved_review_failed": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
            },
        ]
    }
    second_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 280,
                "total_bytes": 2000,
                "used_bytes": 1720,
                "min_free_bytes": 500,
            },
            {
                "product": "active_income",
                "approved_review_failed": 1,
                "generated_at": "2026-01-01T00:01:00+00:00",
            },
        ]
    }
    changed_threshold = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 280,
                "min_free_bytes": 1000,
            }
        ]
    }

    first = alert_fingerprint("warning", "autopilot readiness warnings", first_detail)

    assert alert_fingerprint("warning", "autopilot readiness warnings", second_detail) == first
    assert alert_fingerprint("warning", "autopilot readiness warnings", changed_threshold) != first


def test_emit_alert_cooldown_uses_stable_fingerprint_but_persists_full_detail(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    first_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "free_bytes": 300,
                "min_free_bytes": 500,
            }
        ]
    }
    second_detail = {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "free_bytes": 280,
                "min_free_bytes": 500,
            }
        ]
    }

    first = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="autopilot readiness warnings",
        detail=first_detail,
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="autopilot readiness warnings",
        detail=second_detail,
        cooldown_seconds=60,
        now=120.0,
    )

    assert second == {"sent": False, "reason": "cooldown", "fingerprint": first["fingerprint"]}
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["detail"] == first_detail


def test_emit_alert_records_webhook_failure_without_raising(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")

    def fail_webhook(url, payload):
        raise RuntimeError(f"cannot reach {url}")

    monkeypatch.setattr("src.autopilot.notifications._post_webhook", fail_webhook)

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )
    second = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=120.0,
    )

    assert result["sent"] is True
    assert result["webhook"]["ok"] is False
    assert "cannot reach" in result["webhook"]["error"]
    assert second["reason"] == "cooldown"
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["webhook"]["ok"] is False
    assert json.loads(state_file.read_text(encoding="utf-8"))["alerts"][result["fingerprint"]]["last_sent_ts"] == 100.0


def test_emit_alert_returns_webhook_status(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setenv("AUTOPILOT_WEBHOOK_URL", "https://alerts.invalid/hook")
    monkeypatch.setattr(
        "src.autopilot.notifications._post_webhook",
        lambda url, payload: {"status_code": 503, "ok": False},
    )

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="warning",
        title="readiness warning",
        detail={"warnings": [{"name": "market data"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    assert result["webhook"] == {"status_code": 503, "ok": False}
    assert json.loads(alert_file.read_text(encoding="utf-8").splitlines()[0])["webhook"] == {
        "status_code": 503,
        "ok": False,
    }


def test_emit_alert_reports_state_write_failure_after_alert_is_written(tmp_path, monkeypatch):
    alert_file = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"

    def fail_save(path, payload):
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr("src.autopilot.notifications.write_json_atomic", fail_save)

    result = emit_alert(
        alert_file=alert_file,
        state_file=state_file,
        severity="error",
        title="cycle failed",
        detail={"products": [{"name": "active_income", "error": "failed"}]},
        cooldown_seconds=60,
        now=100.0,
    )

    assert result["sent"] is True
    assert result["state_error"] == f"OSError: cannot write {state_file}"
    lines = alert_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["fingerprint"] == result["fingerprint"]


def test_emit_alert_rejects_symlink_alert_log_without_saving_state(tmp_path):
    alert_file = tmp_path / "alerts.jsonl"
    target = tmp_path / "external_alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    alert_file.symlink_to(target)

    with pytest.raises(ValueError, match="jsonl path must not be a symlink"):
        emit_alert(
            alert_file=alert_file,
            state_file=state_file,
            severity="critical",
            title="autopilot failed",
            detail={"ok": False},
            cooldown_seconds=0,
            now=100.0,
        )

    assert alert_file.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert not state_file.exists()


def test_failure_detail_summarizes_failed_products_and_jobs():
    report = {
        "control_error": "unknown control selectors",
        "unknown_control_selectors": {"paused_jobs": ["typo"]},
        "control": {
            "reason": "unknown_control_selector",
            "paused": True,
            "pause_jobs": True,
        },
        "products": [
            {
                "ok": False,
                "product": {"name": "active_income", "execution_mode": "paper", "market": "futures"},
                "error": "missing artifact",
            },
            {"ok": True, "product": {"name": "btc_accumulation"}},
            {
                "ok": False,
                "product": {"name": "scalp", "execution_mode": "live", "market": "futures"},
                "action": "flatten",
                "broker": "fake-live",
                "close_error": "RuntimeError: exchange timeout",
                "fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.4, "price": 125.0, "fee": 0.02},
                "spot_step_aside": {
                    "strategy_id": "btc_step_aside",
                    "quote_value": 50.0,
                    "requested_qty": 0.4,
                },
                "local_state": {"path": "runtime/btc_state.json", "recovered": False},
                "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                "position_after_error": "RuntimeError: readback timeout",
                "position_after_attempt": {"symbol": "BTCUSDT", "qty": 0.5, "is_flat": False},
                "cycle_errors": [{"strategy_id": "s1", "error": "network down"}],
                "state_errors": [{"field": "open_positions", "error": "expected object"}],
            },
        ],
        "jobs": [
            {"ok": False, "name": "sweep", "returncode": 2, "stderr_tail": "bad"},
            {"ok": True, "name": "smoke"},
        ],
        "data_update": {"ok": True},
    }

    assert failure_detail(report) == {
        "control": {
            "error": "unknown control selectors",
            "reason": "unknown_control_selector",
            "paused": True,
            "pause_jobs": True,
            "unknown_selectors": {"paused_jobs": ["typo"]},
        },
        "products": [
            {
                "name": "active_income",
                "mode": "paper",
                "market": "futures",
                "action": None,
                "error": "missing artifact",
                "cycle_errors": [],
                "state_errors": [],
            },
            {
                "name": "scalp",
                "mode": "live",
                "market": "futures",
                "action": "flatten",
                "error": "RuntimeError: exchange timeout",
                "cycle_errors": [{"strategy_id": "s1", "error": "network down"}],
                "state_errors": [{"field": "open_positions", "error": "expected object"}],
                "broker": "fake-live",
                "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                "position_after_error": "RuntimeError: readback timeout",
                "position_after_attempt": {"symbol": "BTCUSDT", "qty": 0.5, "is_flat": False},
                "fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.4, "price": 125.0, "fee": 0.02},
                "spot_step_aside": {
                    "strategy_id": "btc_step_aside",
                    "quote_value": 50.0,
                    "requested_qty": 0.4,
                },
                "local_state": {"path": "runtime/btc_state.json", "recovered": False},
            },
        ],
        "jobs": [{"name": "sweep", "error": "bad"}],
        "data_update": {"ok": True},
    }


def test_failure_detail_ignores_malformed_sections():
    assert failure_detail({"products": {"bad": "shape"}, "jobs": None, "data_update": {"ok": False}}) == {
        "products": [],
        "jobs": [],
        "data_update": {"ok": False},
    }
    assert failure_detail(
        {
            "products": [
                "bad-product",
                {"ok": False, "product": {"name": "active_income"}, "error": "failed"},
            ],
            "jobs": ["bad-job", {"ok": False, "name": "sweep", "returncode": 2}],
        }
    ) == {
        "products": [
            {
                "name": "active_income",
                "mode": None,
                "market": None,
                "action": None,
                "error": "failed",
                "cycle_errors": [],
                "state_errors": [],
            }
        ],
        "jobs": [{"name": "sweep", "error": 2}],
        "data_update": None,
    }


def test_readiness_warning_detail_summarizes_readiness_warnings():
    report = {
        "checks": [
            {
                "name": "market data seed and freshness",
                "level": "warning",
                "ok": False,
                "detail": {
                    "futures": {"ok": True, "reason": "fresh", "path": "futures.parquet"},
                    "spot": {"ok": False, "reason": "missing_seed_dataset", "path": "spot.parquet"},
                },
            },
            {
                "name": "indicator feature readiness",
                "level": "warning",
                "ok": False,
                "detail": {
                    "spot": {
                        "ok": False,
                        "timeframes": {
                            "1h": {
                                "ok": False,
                                "reason": "missing_indicator_dataset",
                                "missing_features": ["volume_z_20"],
                            },
                            "4h": {"ok": True, "missing_features": []},
                        },
                    },
                },
            },
            {
                "name": "environment file present",
                "level": "warning",
                "ok": False,
                "detail": ".env is optional for paper mode",
            },
            {
                "name": "runtime filesystem free space",
                "level": "warning",
                "ok": False,
                "detail": {
                    "path": "runtime/status.json",
                    "checked_path": "runtime",
                    "free_bytes": 300,
                    "min_free_bytes": 500,
                },
            },
            {
                "name": "strategy framework smoke",
                "level": "warning",
                "ok": False,
                "detail": {
                    "reason": "missing_report",
                    "path": "runtime/strategy_framework_smoke.json",
                    "scenario_count": None,
                    "failures": [],
                },
            },
            {
                "name": "approval ledger revocation audit",
                "level": "warning",
                "ok": False,
                "detail": {
                    "invalid_revocation_count": 1,
                    "entries": [
                        {
                            "fingerprint": "sha256:bad",
                            "strategy_id": "s1",
                            "reasons": ["missing_revocation_reason"],
                        }
                    ],
                },
            },
        ]
    }

    assert readiness_warning_detail(report) == {
        "warnings": [
            {
                "name": "market data seed and freshness",
                "markets": {
                    "spot": {
                        "reason": "missing_seed_dataset",
                        "path": "spot.parquet",
                    }
                },
            },
            {
                "name": "indicator feature readiness",
                "missing": {
                    "spot": {
                        "1h": {
                            "reason": "missing_indicator_dataset",
                            "missing_features": ["volume_z_20"],
                        }
                    }
                },
            },
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": "runtime",
                "free_bytes": 300,
                "min_free_bytes": 500,
                "reason": None,
            },
            {
                "name": "strategy framework smoke",
                "reason": "missing_report",
                "path": "runtime/strategy_framework_smoke.json",
                "scenario_count": None,
                "failures": [],
            },
            {
                "name": "approval ledger revocation audit",
                "entries": [
                    {
                        "fingerprint": "sha256:bad",
                        "strategy_id": "s1",
                        "reasons": ["missing_revocation_reason"],
                    }
                ],
                "invalid_revocation_count": 1,
            },
        ]
    }


@pytest.mark.parametrize(
    ("name", "count_key"),
    [
        ("approval ledger actor audit", "invalid_actor_count"),
        ("approval ledger fingerprint audit", "fingerprint_mismatch_count"),
        ("approval ledger revocation audit", "invalid_revocation_count"),
    ],
)
def test_readiness_warning_detail_summarizes_approval_ledger_audits(name, count_key):
    report = {
        "checks": [
            {
                "name": name,
                "level": "warning",
                "ok": False,
                "detail": {
                    count_key: 1,
                    "entries": [{"fingerprint": "sha256:bad", "strategy_id": "s1"}],
                },
            }
        ]
    }

    assert readiness_warning_detail(report) == {
        "warnings": [
            {
                "name": name,
                count_key: 1,
                "entries": [{"fingerprint": "sha256:bad", "strategy_id": "s1"}],
            }
        ]
    }


def test_readiness_warning_detail_ignores_malformed_checks():
    assert readiness_warning_detail({"checks": {"bad": "shape"}}) == {"warnings": []}
    assert readiness_warning_detail(
        {
            "checks": [
                "bad-check",
                {
                    "name": "market data seed and freshness",
                    "level": "warning",
                    "ok": False,
                    "detail": "not-an-object",
                },
                {
                    "name": "runtime filesystem free space",
                    "level": "warning",
                    "ok": False,
                    "detail": {"path": "runtime/status.json"},
                },
            ]
        }
    ) == {
        "warnings": [
            {
                "name": "runtime filesystem free space",
                "path": "runtime/status.json",
                "checked_path": None,
                "free_bytes": None,
                "min_free_bytes": None,
                "reason": None,
            }
        ]
    }


def test_promotion_warning_detail_summarizes_approved_review_failures_only():
    report = {
        "promotion_reviews": [
            {
                "product": "active_income",
                "status": "ready",
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "recommendations": {"approved_review_failed": 1, "needs_approval": 2},
            },
            {
                "product": "btc_accumulation",
                "status": "ready",
                "path": "runtime/btc_accumulation_promotion_review.json",
                "recommendations": {"needs_approval": 1},
            },
        ]
    }

    assert promotion_warning_detail(report) == {
        "warnings": [
            {
                "name": "approved_review_failed",
                "product": "active_income",
                "status": "ready",
                "approved_review_failed": 1,
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    }


def test_promotion_warning_detail_ignores_malformed_reviews():
    assert promotion_warning_detail({"promotion_reviews": {"bad": "shape"}}) == {"warnings": []}
    assert promotion_warning_detail(
        {
            "promotion_reviews": [
                "bad-review",
                {"product": "active_income", "recommendations": {"approved_review_failed": 1}},
            ]
        }
    ) == {
        "warnings": [
            {
                "name": "approved_review_failed",
                "product": "active_income",
                "status": "unknown",
                "approved_review_failed": 1,
                "path": None,
                "generated_at": None,
            }
        ]
    }


def test_promotion_warning_detail_summarizes_stale_review_packets():
    report = {
        "promotion_reviews": [
            {
                "product": "active_income",
                "status": "ready",
                "enabled": True,
                "exists": True,
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "fresh": False,
                "age_seconds": 200000.0,
                "max_age_seconds": 172800.0,
                "needs_approval": 1,
                "recommendations": {"needs_approval": 1},
            },
            {
                "product": "btc_accumulation",
                "enabled": True,
                "exists": False,
                "fresh": False,
            },
            {
                "product": "disabled",
                "enabled": False,
                "exists": True,
                "fresh": False,
            },
        ]
    }

    assert promotion_warning_detail(report) == {
        "warnings": [
            {
                "name": "promotion_review_stale",
                "product": "active_income",
                "status": "ready",
                "path": "runtime/active_income_promotion_review.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "fresh": False,
                "age_seconds": 200000.0,
                "max_age_seconds": 172800.0,
                "needs_approval": 1,
            }
        ]
    }


def test_research_handoff_warning_detail_summarizes_stale_sources():
    report = {
        "research_cycle": {"generated_at": "2026-01-01T01:05:00+00:00"},
        "mutation_plan": {
            "generated_at": "2026-01-01T01:06:00+00:00",
            "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
        },
        "mutation_batch": {
            "generated_at": "2026-01-01T01:07:00+00:00",
            "source": {"plan_generated_at": "2026-01-01T00:56:00+00:00"},
        },
    }

    assert research_handoff_warning_detail(report) == {
        "warnings": [
            {
                "name": "mutation_plan_stale_source",
                "research_generated_at": "2026-01-01T01:05:00+00:00",
                "mutation_plan_source_research_generated_at": "2026-01-01T00:55:00+00:00",
                "mutation_plan_generated_at": "2026-01-01T01:06:00+00:00",
            },
            {
                "name": "mutation_batch_stale_source",
                "mutation_plan_generated_at": "2026-01-01T01:06:00+00:00",
                "mutation_batch_source_plan_generated_at": "2026-01-01T00:56:00+00:00",
                "mutation_batch_generated_at": "2026-01-01T01:07:00+00:00",
            },
        ]
    }


def test_research_handoff_warning_detail_summarizes_unsafe_mutation_artifacts():
    report = {
        "mutation_plan": {
            "ok": True,
            "generated_at": "2026-01-01T01:06:00+00:00",
            "path": "runtime/mutation_plan.json",
            "summary": {"executable": True},
        },
        "mutation_batch": {
            "ok": False,
            "status": "unsafe_mutation_plan",
            "generated_at": "2026-01-01T01:07:00+00:00",
            "path": "runtime/mutation_hypotheses.json",
            "summary": {
                "skipped": 0,
                "unsafe_flags": ["summary.executable"],
                "executable": False,
            },
        },
    }

    assert research_handoff_warning_detail(report) == {
        "warnings": [
            {
                "name": "mutation_plan_unhealthy",
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "path": "runtime/mutation_plan.json",
                "unsafe_flags": ["summary.executable"],
            },
            {
                "name": "mutation_batch_unhealthy",
                "ok": False,
                "status": "unsafe_mutation_plan",
                "generated_at": "2026-01-01T01:07:00+00:00",
                "path": "runtime/mutation_hypotheses.json",
                "unsafe_flags": ["summary.executable"],
                "skipped": 0,
            },
        ]
    }


def test_research_handoff_warning_detail_ignores_current_sources():
    report = {
        "research_cycle": {"generated_at": "2026-01-01T01:05:00+00:00"},
        "mutation_plan": {
            "generated_at": "2026-01-01T01:06:00+00:00",
            "source": {"research_generated_at": "2026-01-01T01:05:00+00:00"},
        },
        "mutation_batch": {
            "generated_at": "2026-01-01T01:07:00+00:00",
            "source": {"plan_generated_at": "2026-01-01T01:06:00+00:00"},
        },
    }

    assert research_handoff_warning_detail(report) == {"warnings": []}


def test_research_progress_warning_detail_summarizes_no_exportable_research():
    report = {
        "research_cycle": {
            "ok": True,
            "generated_at": "2026-01-01T01:05:00+00:00",
            "summary": {
                "hypotheses": 12,
                "keepers": 0,
                "exported": 0,
                "top_reasons": {"no_train_edge": 8},
                "next_actions": ["continue bounded search"],
                "mutation_effectiveness": {"evaluated_hypotheses": 4, "outcome": "no_keeper"},
            },
        },
        "products": [
            {
                "name": "active_income",
                "objective": "active_income",
                "market": "futures",
                "enabled": True,
                "mode": "paper",
                "reason": "waiting_for_strategy_artifact",
            },
            {
                "name": "disabled",
                "enabled": False,
                "mode": "paper",
                "reason": "waiting_for_strategy_artifact",
            },
        ],
    }

    assert research_progress_warning_detail(report) == {
        "warnings": [
            {
                "name": "research_cycle_no_exportable_strategies",
                "generated_at": "2026-01-01T01:05:00+00:00",
                "hypotheses": 12,
                "top_reasons": {"no_train_edge": 8},
                "next_actions": ["continue bounded search"],
                "waiting_products": [
                    {"name": "active_income", "objective": "active_income", "market": "futures"}
                ],
                "mutation_effectiveness": {"evaluated_hypotheses": 4, "outcome": "no_keeper"},
            }
        ]
    }


def test_research_progress_warning_detail_summarizes_open_position_export_block():
    report = {
        "research_cycle": {
            "ok": True,
            "generated_at": "2026-01-01T01:05:00+00:00",
            "summary": {
                "hypotheses": 12,
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1, "no_current_cycle_keepers": 1},
                "next_actions": ["wait for open positions to close before replacing the active paper artifact"],
            },
        },
        "products": [
            {
                "name": "active_income",
                "objective": "active_income",
                "market": "futures",
                "enabled": True,
                "mode": "paper",
                "open_positions": 1,
            },
            {
                "name": "btc_accumulation",
                "objective": "btc_accumulation",
                "market": "spot",
                "enabled": True,
                "mode": "paper",
                "open_positions": 0,
            },
        ],
    }

    assert research_progress_warning_detail(report) == {
        "warnings": [
            {
                "name": "research_export_blocked_open_positions",
                "generated_at": "2026-01-01T01:05:00+00:00",
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1, "no_current_cycle_keepers": 1},
                "next_actions": ["wait for open positions to close before replacing the active paper artifact"],
                "open_products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 1,
                    }
                ],
            }
        ]
    }


def test_research_progress_warning_detail_ignores_keeper_or_missing_waiting_products():
    assert research_progress_warning_detail(
        {
            "research_cycle": {"ok": True, "summary": {"hypotheses": 12, "keepers": 1, "exported": 0}},
            "products": [
                {"name": "active_income", "enabled": True, "mode": "paper", "reason": "waiting_for_strategy_artifact"}
            ],
        }
    ) == {"warnings": []}


def test_research_progress_warning_detail_ignores_malformed_products():
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {"hypotheses": 12, "keepers": 0, "exported": 0},
            },
            "products": {"bad": "shape"},
        }
    ) == {"warnings": []}
    assert research_progress_warning_detail(
        {
            "research_cycle": {
                "ok": True,
                "summary": {
                    "hypotheses": 12,
                    "keepers": 1,
                    "exported": 0,
                    "export_reasons": {"open_positions_block_export": 1},
                },
            },
            "products": ["bad-product", {"name": "active_income", "enabled": True, "open_positions": 1}],
        }
    ) == {
        "warnings": [
            {
                "name": "research_export_blocked_open_positions",
                "generated_at": None,
                "keepers": 1,
                "exported": 0,
                "export_reasons": {"open_positions_block_export": 1},
                "next_actions": [],
                "open_products": [
                    {
                        "name": "active_income",
                        "objective": None,
                        "market": None,
                        "open_positions": 1,
                    }
                ],
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_summarizes_required_missing_rehearsal():
    assert required_testnet_rehearsal_warning_detail(
        {
            "testnet_rehearsal": {
                "required": True,
                "required_by": ["active_income"],
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": False,
                "product": "active_income",
                "next_action": {
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5",
                },
            }
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "product": "active_income",
                "next_action": {
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=5",
                },
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_surfaces_product_mismatch():
    assert required_testnet_rehearsal_warning_detail(
        {
            "testnet_rehearsal": {
                "required": True,
                "required_by": ["active_income"],
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": False,
                "product": "active_income",
                "invalid_reasons": ["product_symbol_mismatch"],
                "report_product": {"name": "active_income", "symbol": "ETHUSDT"},
                "expected_product": {"name": "active_income", "symbol": "BTCUSDT"},
            }
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "product": "active_income",
                "invalid_reasons": ["product_symbol_mismatch"],
                "report_product": {"name": "active_income", "symbol": "ETHUSDT"},
                "expected_product": {"name": "active_income", "symbol": "BTCUSDT"},
            }
        ]
    }


def test_testnet_rehearsal_warning_detail_ignores_optional_or_ready_rehearsal():
    assert required_testnet_rehearsal_warning_detail(
        {"testnet_rehearsal": {"required": False, "status": "missing", "ok": False}}
    ) == {"warnings": []}
    assert required_testnet_rehearsal_warning_detail(
        {
            "products": [{"name": "active_income", "enabled": True, "require_testnet_rehearsal": True}],
            "testnet_rehearsal": {"status": "ok", "ok": True, "required": True},
        }
    ) == {"warnings": []}
    assert research_progress_warning_detail(
        {
            "research_cycle": {"ok": True, "summary": {"hypotheses": 12, "keepers": 0, "exported": 0}},
            "products": [],
        }
    ) == {"warnings": []}


def test_testnet_rehearsal_warning_detail_ignores_malformed_products():
    assert required_testnet_rehearsal_warning_detail({"products": {"bad": "shape"}}) == {"warnings": []}
    assert required_testnet_rehearsal_warning_detail(
        {
            "products": ["bad-product", {"name": "active_income", "require_testnet_rehearsal": True}],
            "testnet_rehearsal": {"status": "missing", "ok": False},
        }
    ) == {
        "warnings": [
            {
                "name": "required_testnet_rehearsal_not_ready",
                "status": "missing",
                "required_by": [],
            }
        ]
    }
