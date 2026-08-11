import json

import pytest

from src.autopilot.config import AutopilotConfig
from src.autopilot.healthcheck import (
    REMOTE_ALERT_DRAIN_SECONDS,
    _drain_oneshot_remote_alert,
    build_healthcheck,
    evaluate_health,
    main,
)


def operator_report(**overrides):
    payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "status_generated_at": "2026-01-01T00:00:00+00:00",
        "ok": True,
        "status_heartbeat": {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "fresh": True,
            "age_seconds": 10.0,
            "limit_seconds": 300.0,
        },
        "scheduled_jobs": [],
    }
    payload.update(overrides)
    return payload


def test_healthcheck_passes_for_fresh_ok_report():
    health = evaluate_health(operator_report(), readiness_report={"ok": True, "checks": []})

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == []
    assert health["readiness_ok"] is True


def test_healthcheck_warns_when_enabled_product_entries_are_disabled():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "market": "futures",
                    "mode": "paper",
                    "entries_allowed": False,
                    "entry_gate": {
                        "status": "management_only",
                        "reason": "unvalidated_bootstrap_artifact",
                    },
                    "decision_trace": {"summary": {"outcomes": {"entry_disabled": 2}}},
                }
            ]
        )
    )

    assert health["ok"] is True
    warning = next(
        item for item in health["warnings"] if item["code"] == "product_entries_disabled"
    )
    assert warning["detail"]["products"] == [
        {
            "name": "active_income",
            "objective": "active_income",
            "market": "futures",
            "mode": "paper",
            "gate_status": "management_only",
            "gate_reason": "unvalidated_bootstrap_artifact",
            "decision_outcomes": {"entry_disabled": 2},
        }
    ]


def test_healthcheck_fails_for_unhealthy_candidate_paper_status_and_returns_summary():
    candidate_paper = {
        "configured": True,
        "enabled": True,
        "job": "candidate_paper_cycle",
        "path": "runtime/candidate_paper_status.json",
        "exists": True,
        "status": "error",
        "ok": False,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "age_seconds": 901.0,
        "max_age_seconds": 600.0,
        "fresh": False,
        "reason": "invalid_candidate_paper_status",
        "errors": [
            {
                "scope": "product",
                "product": "active_income",
                "field": "candidate_digest",
                "reason": "invalid_sha256_digest",
            }
        ],
        "open_positions": 1,
        "activation_ready_products": [],
        "drawdown_halted_products": [],
    }

    health = evaluate_health(operator_report(candidate_paper=candidate_paper))

    assert health["ok"] is False
    assert health["candidate_paper"] == candidate_paper
    issue = next(item for item in health["issues"] if item["code"] == "candidate_paper_unhealthy")
    assert issue["detail"]["fresh"] is False
    assert issue["detail"]["open_positions"] == 1
    assert issue["detail"]["errors"][0]["reason"] == "invalid_sha256_digest"


def test_healthcheck_warns_when_candidate_paper_drawdown_is_halted():
    health = evaluate_health(
        operator_report(
            candidate_paper={
                "configured": True,
                "enabled": True,
                "job": "candidate_paper_cycle",
                "path": "runtime/candidate_paper_status.json",
                "exists": True,
                "status": "ready",
                "ok": True,
                "fresh": True,
                "open_positions": 0,
                "activation_ready_products": [],
                "drawdown_halted_products": ["active_income"],
                "errors": [],
            }
        )
    )

    assert health["ok"] is True
    assert health["warnings"] == [
        {
            "code": "candidate_paper_drawdown_halted",
            "message": "one or more staged candidates are halted by paper drawdown controls",
            "detail": {"products": ["active_income"]},
        }
    ]


def test_healthcheck_fails_for_stale_status():
    health = evaluate_health(
        operator_report(
            status_heartbeat={
                "generated_at": "2026-01-01T00:00:00+00:00",
                "fresh": False,
                "age_seconds": 901.0,
                "limit_seconds": 300.0,
                "reason": "stale",
            }
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "stale_status"
    assert health["issues"][0]["detail"]["age_seconds"] == 901.0
    assert health["issues"][0]["detail"]["reason"] == "stale"


def test_healthcheck_fails_for_future_status_heartbeat_reason():
    health = evaluate_health(
        operator_report(
            status_heartbeat={
                "generated_at": "2026-01-01T00:20:00+00:00",
                "fresh": False,
                "age_seconds": None,
                "limit_seconds": 300.0,
                "reason": "future_generated_at",
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "stale_status"
    assert health["issues"][0]["detail"] == {
        "generated_at": "2026-01-01T00:20:00+00:00",
        "age_seconds": None,
        "limit_seconds": 300.0,
        "reason": "future_generated_at",
    }


def test_healthcheck_fails_for_failed_cycle():
    health = evaluate_health(operator_report(ok=False))

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "cycle_failed"


def test_healthcheck_fails_for_malformed_operator_report_sections():
    health = evaluate_health(
        operator_report(
            products={"active_income": {"mode": "paper"}},
            jobs=["bad-job-entry"],
            scheduled_jobs=[
                "bad-scheduled-entry",
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "fail",
                    "consecutive_failures": 2,
                },
            ],
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "operator_report_malformed",
        "message": "operator report has malformed collection sections",
        "detail": {
            "sections": [
                {"section": "products", "error": "expected list, got dict"},
                {"section": "jobs", "invalid_entries": [{"index": 0, "type": "str"}]},
                {"section": "scheduled_jobs", "invalid_entries": [{"index": 0, "type": "str"}]},
            ]
        },
    }
    assert health["issues"][1]["code"] == "scheduled_job_failed"
    assert health["issues"][1]["detail"]["jobs"] == [
        {
            "name": "research_cycle",
            "status": "fail",
            "consecutive_failures": 2,
            "last_error": None,
            "last_reason": None,
        }
    ]


def test_healthcheck_fails_for_malformed_readiness_checks():
    health = evaluate_health(
        operator_report(),
        readiness_report={
            "ok": True,
            "checks": [
                "bad-check-entry",
                {
                    "name": "environment file present",
                    "level": "warning",
                    "ok": False,
                    "detail": ".env is optional for paper mode",
                },
            ],
        },
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "readiness_report_malformed",
            "message": "readiness report has malformed check entries",
            "detail": {"section": "checks", "invalid_entries": [{"index": 0, "type": "str"}]},
        }
    ]
    assert health["warnings"] == [
        {
            "code": "readiness_warning",
            "message": "autopilot readiness has warning-level failures",
            "detail": {
                "warning_checks": [
                    {
                        "name": "environment file present",
                        "level": "warning",
                        "detail": ".env is optional for paper mode",
                    }
                ]
            },
        }
    ]


def test_healthcheck_cycle_failed_surfaces_control_error_details():
    unknown = {"paused_products": ["active-incme"], "paused_jobs": ["network-jb"]}
    health = evaluate_health(
        operator_report(
            ok=False,
            control_error="unknown control selectors",
            unknown_control_selectors=unknown,
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "cycle_failed",
            "message": "latest autopilot cycle did not complete successfully",
            "detail": {
                "ok": False,
                "products": [],
                "jobs": [],
                "data_update": None,
                "control_error": "unknown control selectors",
                "unknown_control_selectors": unknown,
            },
        }
    ]


def test_healthcheck_cycle_failed_surfaces_control_clear_failure():
    control_clear = [
        {
            "command": "clear-flatten",
            "name": "active_income",
            "ok": False,
            "error": "OSError: disk full",
        }
    ]
    health = evaluate_health(operator_report(ok=False, control_clear=control_clear))

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "cycle_failed",
            "message": "latest autopilot cycle did not complete successfully",
            "detail": {
                "ok": False,
                "products": [],
                "jobs": [],
                "data_update": None,
                "control_clear": control_clear,
            },
        }
    ]


def test_healthcheck_warns_when_operator_control_is_active():
    health = evaluate_health(
        operator_report(
            control={
                "paused": True,
                "pause_jobs": True,
                "paused_products": [],
                "paused_jobs": [],
                "flatten_products": [],
                "flatten_all": False,
                "reason": "maintenance",
            }
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "operator_control_active",
            "message": "operator control is actively pausing or flattening the autopilot",
            "detail": {
                "paused": True,
                "pause_jobs": True,
                "paused_products": [],
                "paused_jobs": [],
                "flatten_all": False,
                "flatten_products": [],
                "reason": "maintenance",
            },
        }
    ]


def test_healthcheck_warns_for_selected_operator_controls():
    health = evaluate_health(
        operator_report(
            control={
                "paused": False,
                "pause_jobs": False,
                "paused_products": ["btc_accumulation"],
                "paused_jobs": ["research_cycle"],
                "flatten_products": ["active_income"],
                "flatten_all": False,
            }
        )
    )

    assert health["ok"] is True
    assert health["warnings"][0] == {
        "code": "operator_control_active",
        "message": "operator control is actively pausing or flattening the autopilot",
        "detail": {
            "paused": False,
            "pause_jobs": False,
            "paused_products": ["btc_accumulation"],
            "paused_jobs": ["research_cycle"],
            "flatten_all": False,
            "flatten_products": ["active_income"],
        },
    }


def test_healthcheck_ignores_clear_operator_control():
    health = evaluate_health(
        operator_report(
            control={
                "paused": False,
                "pause_jobs": False,
                "paused_products": [],
                "paused_jobs": [],
                "flatten_products": [],
                "flatten_all": False,
                "reason": "",
            }
        )
    )

    assert health["ok"] is True
    assert health["warnings"] == []


def test_healthcheck_cycle_failed_surfaces_failed_product_details():
    health = evaluate_health(
        operator_report(
            ok=True,
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "cycle_ok": False,
                    "action": "flatten",
                    "broker": "fake-live",
                    "close_error": "RuntimeError: exchange timeout",
                    "fill": {
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "qty": 0.4,
                        "price": 125.0,
                        "fee": 0.02,
                    },
                    "spot_step_aside": {
                        "strategy_id": "btc_step_aside",
                        "quote_value": 50.0,
                        "requested_qty": 0.4,
                    },
                    "local_state": {"path": "runtime/btc_state.json", "recovered": False},
                    "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                    "position_after_error": "RuntimeError: readback timeout",
                    "position_after_attempt": {"symbol": "BTCUSDT", "qty": 0.5, "is_flat": False},
                }
            ],
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "cycle_failed",
            "message": "latest autopilot cycle did not complete successfully",
            "detail": {
                "ok": True,
                "products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "live",
                        "action": "flatten",
                        "close_error": "RuntimeError: exchange timeout",
                        "broker": "fake-live",
                        "fill": {
                            "symbol": "BTCUSDT",
                            "side": "buy",
                            "qty": 0.4,
                            "price": 125.0,
                            "fee": 0.02,
                        },
                        "spot_step_aside": {
                            "strategy_id": "btc_step_aside",
                            "quote_value": 50.0,
                            "requested_qty": 0.4,
                        },
                        "local_state": {"path": "runtime/btc_state.json", "recovered": False},
                        "position_before": {"symbol": "BTCUSDT", "qty": 0.5},
                        "position_after_error": "RuntimeError: readback timeout",
                        "position_after_attempt": {
                            "symbol": "BTCUSDT",
                            "qty": 0.5,
                            "is_flat": False,
                        },
                    }
                ],
                "jobs": [],
                "data_update": None,
            },
        }
    ]


def test_healthcheck_cycle_failed_surfaces_failed_job_and_data_update():
    health = evaluate_health(
        operator_report(
            ok=True,
            jobs=[
                {
                    "name": "research_cycle",
                    "ok": False,
                    "returncode": 2,
                    "stderr_tail": "failed research",
                }
            ],
            data_update={
                "ok": False,
                "returncode": 1,
                "stderr_tail": "download failed",
            },
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "cycle_failed",
            "message": "latest autopilot cycle did not complete successfully",
            "detail": {
                "ok": True,
                "products": [],
                "jobs": [
                    {
                        "name": "research_cycle",
                        "returncode": 2,
                        "stderr_tail": "failed research",
                    }
                ],
                "data_update": {
                    "ok": False,
                    "returncode": 1,
                    "stderr_tail": "download failed",
                },
            },
        }
    ]


def test_healthcheck_fails_for_live_product_state_errors():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "state_errors": [
                        {"field": "open_positions", "error": "expected object, got list"}
                    ],
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_product_state_invalid",
            "message": "one or more live products reported invalid local state",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "live",
                        "state_errors": [
                            {"field": "open_positions", "error": "expected object, got list"}
                        ],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_for_paper_product_state_errors():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "state_errors": [
                        {"field": "open_positions", "error": "expected object, got list"}
                    ],
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_product_state_invalid",
            "message": "one or more paper products reported invalid local state",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "paper",
                        "state_errors": [
                            {"field": "open_positions", "error": "expected object, got list"}
                        ],
                    }
                ]
            },
        }
    ]


@pytest.mark.parametrize(
    ("mode", "bucket", "code", "expected_ok"),
    [
        ("live", "issues", "live_product_recovery_pending", False),
        ("paper", "warnings", "paper_product_recovery_pending", True),
    ],
)
def test_healthcheck_surfaces_unresolved_product_recovery_state(
    mode,
    bucket,
    code,
    expected_ok,
):
    states = {
        "pending_order": {"stage": "entry", "client_id": "tb-en-1"},
        "pending_entry_recovery": {
            "status": "recovery_close_failed_position_remains",
            "recovery_client_id": "tb-rc-1",
        },
        "risk_recovery_incident": {
            "cause": "broker_position_quantity_mismatch",
            "status": "recovery_close_filled_and_flat",
        },
        "flatten_intent": {"client_id": "tb-sf-1"},
    }
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": mode,
                    "market": "futures",
                    **states,
                }
            ]
        )
    )

    assert health["ok"] is expected_ok
    assert health[bucket] == [
        {
            "code": code,
            "message": (
                f"one or more {mode} products have unresolved broker intents "
                "or safety-recovery incidents"
            ),
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": mode,
                        "states": states,
                    }
                ]
            },
        }
    ]
    assert health["warnings" if bucket == "issues" else "issues"] == []


@pytest.mark.parametrize(
    ("mode", "bucket", "code", "expected_ok"),
    [
        ("live", "issues", "live_exit_accounting_pending", False),
        ("paper", "warnings", "paper_exit_accounting_pending", True),
    ],
)
def test_healthcheck_surfaces_unresolved_exit_accounting_intent(
    mode,
    bucket,
    code,
    expected_ok,
):
    intent = {
        "version": 1,
        "phase": "ready_to_commit",
        "exit_event_id": "a" * 64,
        "strategy_id": "live_r1",
        "broker_flat_proven": True,
    }
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": mode,
                    "market": "futures",
                    "exit_accounting_intent": intent,
                }
            ]
        )
    )

    assert health["ok"] is expected_ok
    assert health[bucket] == [
        {
            "code": code,
            "message": (
                f"one or more {mode} products have an unresolved idempotent exit accounting intent"
            ),
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": mode,
                        "intent": intent,
                    }
                ]
            },
        }
    ]
    assert health["warnings" if bucket == "issues" else "issues"] == []


def test_healthcheck_fails_for_live_trade_log_issues():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "trade_summary": {
                        "path": "runtime/active_income_trades.csv",
                        "invalid_rows": 1,
                        "issue": "trade log has 1 row(s) with invalid numeric fields",
                        "numeric_errors": [
                            {"line": 2, "field": "broker_exit_fee", "value": "-0.01"}
                        ],
                    },
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_trade_log_invalid",
            "message": "one or more live products have invalid trade-log audit fields",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "live",
                        "path": "runtime/active_income_trades.csv",
                        "invalid_rows": 1,
                        "issue": "trade log has 1 row(s) with invalid numeric fields",
                        "numeric_errors": [
                            {"line": 2, "field": "broker_exit_fee", "value": "-0.01"}
                        ],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_live_drawdown_circuit_breaker():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "equity": 900.0,
                    "peak_equity": 1000.0,
                    "drawdown_fraction": 0.10,
                    "drawdown_limit_fraction": 0.10,
                    "drawdown_halted": True,
                    "drawdown_halted_at": "2026-07-09T12:00:00+00:00",
                    "drawdown_halt_reason": "equity_drawdown_limit_reached objective=active_income",
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_product_drawdown_halted",
            "message": "one or more live products hit the sticky peak-equity drawdown circuit breaker",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "live",
                        "equity": 900.0,
                        "peak_equity": 1000.0,
                        "drawdown_fraction": 0.10,
                        "drawdown_limit_fraction": 0.10,
                        "drawdown_halted_at": "2026-07-09T12:00:00+00:00",
                        "drawdown_halt_reason": "equity_drawdown_limit_reached objective=active_income",
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_for_paper_drawdown_circuit_breaker():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "btc_accumulation",
                    "enabled": True,
                    "objective": "btc_accumulation",
                    "mode": "paper",
                    "market": "spot",
                    "equity": 0.95,
                    "peak_equity": 1.0,
                    "drawdown_fraction": 0.05,
                    "drawdown_limit_fraction": 0.05,
                    "drawdown_halted": True,
                    "drawdown_halted_at": "2026-07-09T12:00:00+00:00",
                    "drawdown_halt_reason": "equity_drawdown_limit_reached objective=btc_accumulation",
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"][0]["code"] == "paper_product_drawdown_halted"
    assert health["warnings"][0]["detail"]["products"][0]["product"] == "btc_accumulation"


def test_healthcheck_warns_for_paper_trade_log_issues():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "trade_summary": {
                        "path": "runtime/active_income_paper_trades.csv",
                        "invalid_rows": 1,
                        "issue": "trade log has 1 row(s) with invalid numeric fields",
                        "numeric_errors": [{"line": 2, "field": "net_return", "value": "bad"}],
                    },
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_trade_log_invalid",
            "message": "one or more paper products have invalid trade-log audit fields",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "paper",
                        "path": "runtime/active_income_paper_trades.csv",
                        "invalid_rows": 1,
                        "issue": "trade log has 1 row(s) with invalid numeric fields",
                        "numeric_errors": [{"line": 2, "field": "net_return", "value": "bad"}],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_for_stale_open_position():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "mode": "paper",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_r1",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767234600.0,
    )

    assert health["ok"] is True
    assert health["warnings"] == [
        {
            "code": "open_position_stale",
            "message": "one or more open positions are older than their expected strategy horizon",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "mode": "paper",
                        "market": "futures",
                        "strategy_id": "live_r1",
                        "direction": "short",
                        "entry_time": "2026-01-01T00:00:00+00:00",
                        "age_seconds": 9000.0,
                        "stale_after_seconds": 5400.0,
                        "base_timeframe": "5m",
                        "horizon_bars": 6,
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_stale_live_open_position():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_r1",
                            "direction": "long",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 98.0,
                            "tp_price": 104.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767234600.0,
    )

    assert health["ok"] is False
    assert health["warnings"] == []
    assert health["issues"] == [
        {
            "code": "live_open_position_stale",
            "message": "one or more live open positions are older than their expected strategy horizon",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "mode": "live",
                        "market": "futures",
                        "strategy_id": "live_r1",
                        "direction": "long",
                        "entry_time": "2026-01-01T00:00:00+00:00",
                        "age_seconds": 9000.0,
                        "stale_after_seconds": 5400.0,
                        "base_timeframe": "5m",
                        "horizon_bars": 6,
                        "objective": "active_income",
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_live_open_position_missing_risk_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_r1",
                            "direction": "long",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "tp_price": 104.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_risk_invalid",
            "message": "one or more live open positions have missing or invalid risk metadata",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "strategy_id": "live_r1",
                        "direction": "long",
                        "position_size": 0.1,
                        "entry_price": 100.0,
                        "sl_price": None,
                        "tp_price": 104.0,
                        "reasons": ["invalid_stop_price"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_live_open_position_invalid_stop_target_order():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 98.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "live_open_position_risk_invalid"
    assert health["issues"][0]["detail"]["positions"][0]["reasons"] == [
        "invalid_short_stop_target_order"
    ]


def test_healthcheck_warns_for_paper_open_position_missing_risk_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "paper_r1",
                            "direction": "long",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "tp_price": 104.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_open_position_risk_invalid",
            "message": "one or more paper open positions have missing or invalid risk metadata",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "strategy_id": "paper_r1",
                        "direction": "long",
                        "position_size": 0.1,
                        "entry_price": 100.0,
                        "sl_price": None,
                        "tp_price": 104.0,
                        "reasons": ["invalid_stop_price"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_allows_live_open_position_with_valid_risk_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == []


def test_healthcheck_fails_for_live_open_position_missing_broker_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    broker_issue = next(
        item for item in health["issues"] if item["code"] == "live_open_position_broker_invalid"
    )
    assert broker_issue == {
        "code": "live_open_position_broker_invalid",
        "message": "one or more live open positions have missing or invalid broker metadata",
        "detail": {
            "positions": [
                {
                    "product": "active_income",
                    "objective": "active_income",
                    "market": "futures",
                    "strategy_id": "live_short",
                    "direction": "short",
                    "broker_symbol": None,
                    "broker_side": None,
                    "broker_qty": None,
                    "broker_requested_qty": None,
                    "broker_fill_ratio": None,
                    "broker_entry_fee": None,
                    "broker_entry_balance": None,
                    "reasons": [
                        "invalid_broker_symbol",
                        "invalid_broker_side",
                        "broker_side_direction_mismatch",
                        "invalid_broker_qty",
                        "invalid_broker_entry_price",
                        "invalid_broker_entry_fee",
                        "invalid_broker_requested_qty",
                        "invalid_broker_fill_ratio",
                        "invalid_broker_entry_balance",
                        "invalid_broker_stop_order_id",
                        "invalid_broker_stop_client_id",
                        "invalid_broker_stop_trigger_price",
                    ],
                    "broker_stop_order_id": None,
                    "broker_stop_client_id": None,
                    "broker_stop_trigger_price": None,
                }
            ]
        },
    }


def test_healthcheck_allows_live_open_position_with_valid_broker_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.5,
                            "broker_entry_price": 100.0,
                            "broker_entry_fee": 0.0,
                            "broker_requested_qty": 0.5,
                            "broker_fill_ratio": 1.0,
                            "broker_entry_balance": 10_000.0,
                            "broker_stop_order_id": "stop-1",
                            "broker_stop_client_id": "tb-sl-stop-1",
                            "broker_stop_trigger_price": 102.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == []


def test_healthcheck_fails_for_live_open_position_invalid_broker_entry_fee():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.5,
                            "broker_entry_price": 100.0,
                            "broker_entry_fee": -0.01,
                            "broker_requested_qty": 0.5,
                            "broker_fill_ratio": 1.0,
                            "broker_entry_balance": 10_000.0,
                            "broker_stop_order_id": "stop-1",
                            "broker_stop_client_id": "tb-sl-stop-1",
                            "broker_stop_trigger_price": 102.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    broker_issue = next(
        item for item in health["issues"] if item["code"] == "live_open_position_broker_invalid"
    )
    assert broker_issue["detail"]["positions"][0]["broker_entry_fee"] == -0.01
    assert broker_issue["detail"]["positions"][0]["reasons"] == ["invalid_broker_entry_fee"]


def test_healthcheck_fails_for_live_open_position_partial_broker_fill_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.25,
                            "broker_entry_price": 100.0,
                            "broker_entry_fee": 0.0,
                            "broker_requested_qty": 0.5,
                            "broker_fill_ratio": 0.5,
                            "broker_entry_balance": 10_000.0,
                            "broker_stop_order_id": "stop-1",
                            "broker_stop_client_id": "tb-sl-stop-1",
                            "broker_stop_trigger_price": 102.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    broker_issue = next(
        item for item in health["issues"] if item["code"] == "live_open_position_broker_invalid"
    )
    assert broker_issue["detail"]["positions"][0]["reasons"] == [
        "broker_fill_ratio_not_complete",
        "broker_qty_mismatch_requested",
    ]


def test_healthcheck_fails_for_live_futures_position_with_mismatched_native_stop():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.5,
                            "broker_entry_price": 100.0,
                            "broker_entry_fee": 0.0,
                            "broker_requested_qty": 0.5,
                            "broker_fill_ratio": 1.0,
                            "broker_entry_balance": 10_000.0,
                            "broker_stop_order_id": "stop-1",
                            "broker_stop_client_id": "tb-sl-stop-1",
                            "broker_stop_trigger_price": 101.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    broker_issue = next(
        item for item in health["issues"] if item["code"] == "live_open_position_broker_invalid"
    )
    assert broker_issue["detail"]["positions"][0]["reasons"] == [
        "broker_stop_trigger_mismatch_strategy_stop"
    ]


def test_healthcheck_fails_for_live_btc_step_aside_missing_quote_budget_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "btc_accumulation",
                    "enabled": True,
                    "objective": "btc_accumulation",
                    "mode": "live",
                    "market": "spot",
                    "open_positions": 1,
                    "open_position_details": [
                        {
                            "strategy_id": "btc_step_aside",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "4h",
                            "horizon_bars": 6,
                            "stale_after_seconds": 259200.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.1,
                            "broker_entry_price": 100.0,
                            "broker_entry_fee": 0.0,
                            "broker_requested_qty": 0.1,
                            "broker_fill_ratio": 1.0,
                            "broker_exit_sizing": "quote_reinvest",
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    broker_issue = next(
        item for item in health["issues"] if item["code"] == "live_open_position_broker_invalid"
    )
    assert broker_issue["detail"]["positions"][0]["product"] == "btc_accumulation"
    assert broker_issue["detail"]["positions"][0]["reasons"] == [
        "invalid_spot_step_aside_quote_value"
    ]


def test_healthcheck_fails_for_live_open_position_missing_monitoring_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                        }
                    ],
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_monitoring_invalid",
            "message": "one or more live open positions have missing or invalid monitoring metadata",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "strategy_id": "live_short",
                        "direction": "short",
                        "entry_time": None,
                        "base_timeframe": "5m",
                        "horizon_bars": 6,
                        "stale_after_seconds": None,
                        "reasons": ["invalid_entry_time", "invalid_stale_after_seconds"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_for_paper_open_position_missing_monitoring_metadata():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "paper_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                        }
                    ],
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_open_position_monitoring_invalid",
            "message": "one or more paper open positions have missing or invalid monitoring metadata",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "strategy_id": "paper_short",
                        "direction": "short",
                        "entry_time": None,
                        "base_timeframe": "5m",
                        "horizon_bars": 6,
                        "stale_after_seconds": None,
                        "reasons": ["invalid_entry_time", "invalid_stale_after_seconds"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_live_open_position_future_entry_time():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:10:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 1800.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767225600.0,  # 2026-01-01T00:00:00+00:00
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_monitoring_invalid",
            "message": "one or more live open positions have missing or invalid monitoring metadata",
            "detail": {
                "positions": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "strategy_id": "live_short",
                        "direction": "short",
                        "entry_time": "2026-01-01T00:10:00+00:00",
                        "base_timeframe": "5m",
                        "horizon_bars": 6,
                        "stale_after_seconds": 1800.0,
                        "reasons": ["future_entry_time"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_for_paper_open_position_future_entry_time():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_position_details": [
                        {
                            "strategy_id": "paper_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:10:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 1800.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767225600.0,  # 2026-01-01T00:00:00+00:00
    )

    assert health["ok"] is True
    assert health["issues"] == []
    warning = next(
        item
        for item in health["warnings"]
        if item["code"] == "paper_open_position_monitoring_invalid"
    )
    assert warning["detail"]["positions"] == [
        {
            "product": "active_income",
            "objective": "active_income",
            "market": "futures",
            "strategy_id": "paper_short",
            "direction": "short",
            "entry_time": "2026-01-01T00:10:00+00:00",
            "base_timeframe": "5m",
            "horizon_bars": 6,
            "stale_after_seconds": 1800.0,
            "reasons": ["future_entry_time"],
        }
    ]


def test_healthcheck_fails_when_live_open_position_details_are_missing():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [],
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_visibility_invalid",
            "message": "one or more live products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 1,
                        "open_position_details_count": 0,
                        "reasons": ["missing_open_position_details"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_does_not_fail_when_paper_open_position_details_are_missing():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_positions": 1,
                    "open_position_details": [],
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_open_position_visibility_invalid",
            "message": "one or more paper products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 1,
                        "open_position_details_count": 0,
                        "reasons": ["missing_open_position_details"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_when_live_open_position_count_is_invalid():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": -1,
                    "open_position_details": [],
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_visibility_invalid",
            "message": "one or more live products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": -1,
                        "open_position_details_count": 0,
                        "reasons": ["invalid_open_position_count"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_when_paper_open_position_count_is_invalid():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_positions": "invalid",
                    "open_position_details": [],
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_open_position_visibility_invalid",
            "message": "one or more paper products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": "invalid",
                        "open_position_details_count": 0,
                        "reasons": ["invalid_open_position_count"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_when_live_open_position_details_payload_is_invalid():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 0,
                    "open_position_details": {},
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_visibility_invalid",
            "message": "one or more live products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 0,
                        "open_position_details_count": None,
                        "reasons": ["invalid_open_position_details"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_warns_when_paper_open_position_details_payload_is_invalid():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "open_positions": 0,
                    "open_position_details": {},
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"][0]["code"] == "paper_open_position_visibility_invalid"
    assert health["warnings"][0]["detail"]["products"][0] == {
        "product": "active_income",
        "objective": "active_income",
        "market": "futures",
        "open_positions": 0,
        "open_position_details_count": None,
        "reasons": ["invalid_open_position_details"],
    }


def test_healthcheck_fails_when_live_open_position_details_exceed_count():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "open_positions": 0,
                    "open_position_details": [
                        {
                            "strategy_id": "live_short",
                            "direction": "short",
                            "position_size": 0.1,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ]
        ),
        now_ts=1767229200.0,
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "live_open_position_visibility_invalid",
            "message": "one or more live products report open positions without matching position details",
            "detail": {
                "products": [
                    {
                        "product": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "open_positions": 0,
                        "open_position_details_count": 1,
                        "reasons": ["open_position_count_detail_mismatch"],
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_runtime_load_errors():
    load_error = {
        "name": "status",
        "path": "runtime/status.json",
        "error": "JSONDecodeError: bad json",
    }
    health = evaluate_health(operator_report(runtime_load_errors=[load_error]))

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "runtime_file_unreadable"
    assert health["issues"][0]["detail"] == {"files": [load_error]}


def test_healthcheck_fails_for_runtime_shape_errors():
    shape_error = {
        "name": "status",
        "path": "runtime/status.json",
        "field": "products",
        "error": "expected list, got dict",
    }
    health = evaluate_health(operator_report(runtime_shape_errors=[shape_error]))

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "runtime_file_shape_invalid",
        "message": "one or more runtime JSON files have malformed fields",
        "detail": {"files": [shape_error]},
    }


def test_healthcheck_warns_for_runtime_reporting_errors():
    health = evaluate_health(
        operator_report(
            reporting={
                "ok": False,
                "outputs": {
                    "operator_report_json": {
                        "path": "runtime/operator_report.json",
                        "written": False,
                    },
                    "readiness_report_json": {
                        "path": "runtime/readiness_report.json",
                        "written": True,
                    },
                },
                "errors": [
                    {
                        "stage": "operator_report_json_write_failed",
                        "path": "runtime/operator_report.json",
                        "error": "OSError: disk full",
                    }
                ],
            }
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "runtime_reporting_failed",
            "message": "latest autopilot cycle could not refresh all operator reports",
            "detail": {
                "errors": [
                    {
                        "stage": "operator_report_json_write_failed",
                        "path": "runtime/operator_report.json",
                        "error": "OSError: disk full",
                    }
                ],
                "outputs": {
                    "operator_report_json": {
                        "path": "runtime/operator_report.json",
                        "written": False,
                    },
                    "readiness_report_json": {
                        "path": "runtime/readiness_report.json",
                        "written": True,
                    },
                },
            },
        }
    ]


def test_healthcheck_fails_when_market_data_status_is_not_ok():
    health = evaluate_health(
        operator_report(
            market_data={
                "ok": False,
                "markets": {
                    "futures": {
                        "ok": False,
                        "market": "futures",
                        "path": "data/candles/BTCUSDT/BTCUSDT_1m.parquet",
                        "exists": True,
                        "reason": "future_timestamp",
                        "rows": 10,
                        "first_timestamp": "2026-01-01T00:00:00+00:00",
                        "last_timestamp": "2026-01-01T00:10:00+00:00",
                        "age_seconds": -600.0,
                        "max_age_seconds": 86400,
                    },
                    "spot": {
                        "ok": True,
                        "market": "spot",
                        "reason": "fresh",
                    },
                },
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "market_data_unhealthy",
        "message": "one or more market data feeds are missing, stale, invalid, or timestamped in the future",
        "detail": {
            "markets": [
                {
                    "market": "futures",
                    "path": "data/candles/BTCUSDT/BTCUSDT_1m.parquet",
                    "exists": True,
                    "reason": "future_timestamp",
                    "rows": 10,
                    "first_timestamp": "2026-01-01T00:00:00+00:00",
                    "last_timestamp": "2026-01-01T00:10:00+00:00",
                    "age_seconds": -600.0,
                    "max_age_seconds": 86400,
                }
            ]
        },
    }


def test_healthcheck_fails_for_legacy_single_market_data_status():
    health = evaluate_health(
        operator_report(
            market_data={
                "ok": False,
                "market": "futures",
                "path": "data/candles/BTCUSDT/BTCUSDT_1m.parquet",
                "exists": False,
                "reason": "missing_seed_dataset",
                "remediation": {"action": "bootstrap_market_data"},
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "market_data_unhealthy",
        "message": "one or more market data feeds are missing, stale, invalid, or timestamped in the future",
        "detail": {
            "markets": [
                {
                    "market": "futures",
                    "path": "data/candles/BTCUSDT/BTCUSDT_1m.parquet",
                    "exists": False,
                    "reason": "missing_seed_dataset",
                    "remediation": {"action": "bootstrap_market_data"},
                }
            ]
        },
    }


def test_healthcheck_fails_for_readiness_blockers():
    health = evaluate_health(
        operator_report(),
        readiness_report={
            "ok": False,
            "checks": [
                {
                    "name": "runtime directory writable",
                    "level": "error",
                    "ok": False,
                    "detail": "runtime",
                },
                {
                    "name": "environment file present",
                    "level": "warning",
                    "ok": False,
                    "detail": ".env",
                },
            ],
        },
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "readiness_blocked"
    assert health["issues"][0]["detail"]["blocking_checks"] == [
        {"name": "runtime directory writable", "level": "error", "detail": "runtime"}
    ]


def test_healthcheck_surfaces_readiness_warnings_without_failing():
    health = evaluate_health(
        operator_report(),
        readiness_report={
            "ok": True,
            "checks": [
                {
                    "name": "approval ledger actor audit",
                    "level": "warning",
                    "ok": False,
                    "detail": {"invalid_actor_count": 1},
                },
                {
                    "name": "runtime directory writable",
                    "level": "error",
                    "ok": True,
                    "detail": "runtime",
                },
            ],
        },
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "readiness_warning",
            "message": "autopilot readiness has warning-level failures",
            "detail": {
                "warning_checks": [
                    {
                        "name": "approval ledger actor audit",
                        "level": "warning",
                        "detail": {"invalid_actor_count": 1},
                    }
                ]
            },
        }
    ]


def test_healthcheck_fails_for_enabled_failed_scheduled_job():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "fail",
                    "consecutive_failures": 3,
                },
                {
                    "name": "market_data_update",
                    "enabled": True,
                    "status": "recovered",
                    "consecutive_failures": 1,
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "fail",
                    "consecutive_failures": 1,
                },
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "scheduled_job_failed"
    assert health["issues"][0]["detail"]["jobs"] == [
        {
            "name": "research_cycle",
            "status": "fail",
            "consecutive_failures": 3,
            "last_error": None,
            "last_reason": None,
        }
    ]


def test_healthcheck_failed_scheduled_job_includes_structured_errors():
    errors = [
        {"task": "alert_state", "error": "ValueError: alert state path must not be a symlink"},
        {"task": "control_audit", "error": "OSError: disk full"},
    ]
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "maintenance",
                    "enabled": True,
                    "status": "fail",
                    "consecutive_failures": 2,
                    "last_error": "alert_state: ValueError: alert state path must not be a symlink",
                    "last_structured_errors_count": 4,
                    "last_structured_errors": errors,
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "scheduled_job_failed"
    assert health["issues"][0]["detail"]["jobs"] == [
        {
            "name": "maintenance",
            "status": "fail",
            "consecutive_failures": 2,
            "last_error": "alert_state: ValueError: alert state path must not be a symlink",
            "last_reason": None,
            "last_structured_errors": errors,
            "last_structured_errors_count": 4,
        }
    ]


def test_healthcheck_allows_enabled_job_that_is_due_but_not_overdue():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "ok",
                    "due": True,
                    "age_seconds": 90_000.0,
                    "effective_cadence_seconds": 86_400.0,
                    "timeout_seconds": 900,
                    "last_started_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []


def test_healthcheck_fails_when_independent_job_worker_is_stale():
    health = evaluate_health(
        operator_report(
            job_worker={
                "configured": True,
                "ok": False,
                "reason": "stale",
                "path": "runtime/job_worker_status.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "age_seconds": 600.0,
                "limit_seconds": 300.0,
                "last_cycle_ok": True,
                "last_cycle_reason": None,
                "enabled_jobs": ["research_factory", "research_cycle"],
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "scheduled_job_worker_unhealthy",
            "message": "the independent scheduled-job worker is missing, stale, or failing",
            "detail": {
                "reason": "stale",
                "path": "runtime/job_worker_status.json",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "age_seconds": 600.0,
                "limit_seconds": 300.0,
                "last_cycle_ok": True,
                "last_cycle_reason": None,
                "enabled_jobs": ["research_factory", "research_cycle"],
            },
        }
    ]


def test_healthcheck_does_not_mislabel_operational_failure_as_runtime_cycle_failure():
    health = evaluate_health(
        operator_report(
            ok=False,
            runtime_ok=True,
            job_worker={
                "configured": True,
                "ok": False,
                "reason": "missing",
                "path": "runtime/job_worker_status.json",
                "enabled_jobs": ["research_cycle"],
            },
        )
    )

    assert health["ok"] is False
    assert [issue["code"] for issue in health["issues"]] == ["scheduled_job_worker_unhealthy"]


def test_healthcheck_fails_for_enabled_overdue_scheduled_job():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "ok",
                    "due": True,
                    "age_seconds": 180_000.0,
                    "effective_cadence_seconds": 86_400.0,
                    "timeout_seconds": 900,
                    "last_started_at": "2026-01-01T00:00:00+00:00",
                    "last_reason": "no_exportable_strategies",
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "ok",
                    "due": True,
                    "age_seconds": 999_999.0,
                    "effective_cadence_seconds": 60.0,
                },
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "scheduled_job_overdue"
    assert health["issues"][0]["detail"]["jobs"] == [
        {
            "name": "research_cycle",
            "status": "ok",
            "due": True,
            "age_seconds": 180000.0,
            "limit_seconds": 172800.0,
            "effective_cadence_seconds": 86400.0,
            "last_started_at": "2026-01-01T00:00:00+00:00",
            "last_reason": "no_exportable_strategies",
        }
    ]


def test_healthcheck_fails_for_enabled_due_job_that_never_ran():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_synthetic_smoke",
                    "enabled": True,
                    "status": "never_run",
                    "due": True,
                    "cadence_seconds": 86400.0,
                    "effective_cadence_seconds": 86400.0,
                    "timeout_seconds": 300,
                    "last_started_at": None,
                    "age_seconds": None,
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "never_run",
                    "due": True,
                    "cadence_seconds": 60.0,
                },
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "scheduled_job_never_ran",
        "message": "one or more enabled scheduled jobs are due but have never run",
        "detail": {
            "jobs": [
                {
                    "name": "research_synthetic_smoke",
                    "status": "never_run",
                    "due": True,
                    "cadence_seconds": 86400.0,
                    "effective_cadence_seconds": 86400.0,
                    "timeout_seconds": 300,
                }
            ]
        },
    }


def test_healthcheck_fails_for_enabled_scheduled_job_with_invalid_state_marker():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "ok",
                    "due": True,
                    "age_seconds": None,
                    "last_started_at": "bad",
                    "last_reason": "invalid job state last_started_ts: 'bad'",
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "ok",
                    "due": True,
                    "age_seconds": None,
                    "last_started_at": "bad",
                    "last_reason": "invalid job state last_started_ts: 'bad'",
                },
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "scheduled_job_state_invalid",
        "message": "one or more enabled scheduled jobs reported invalid scheduler state",
        "detail": {
            "jobs": [
                {
                    "name": "research_cycle",
                    "status": "ok",
                    "due": True,
                    "last_started_at": "bad",
                    "age_seconds": None,
                    "last_reason": "invalid job state last_started_ts: 'bad'",
                }
            ]
        },
    }


def test_healthcheck_warns_for_truncated_scheduled_job_output():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "ok",
                    "last_started_at": "2026-01-01T00:00:00+00:00",
                    "last_stdout_truncated": True,
                    "last_stdout_bytes": 1234567,
                    "last_stderr_truncated": False,
                    "last_stderr_bytes": 0,
                },
                {
                    "name": "mutation_batch",
                    "enabled": True,
                    "status": "fail",
                    "last_started_at": "2026-01-01T00:05:00+00:00",
                    "last_stdout_truncated": False,
                    "last_stdout_bytes": 0,
                    "last_stderr_truncated": True,
                    "last_stderr_bytes": 2345678,
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "ok",
                    "last_stdout_truncated": True,
                    "last_stdout_bytes": 999,
                },
            ]
        ),
        fail_on_job_failures=False,
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"][0] == {
        "code": "scheduled_job_output_truncated",
        "message": "one or more enabled scheduled jobs produced truncated output",
        "detail": {
            "jobs": [
                {
                    "name": "research_cycle",
                    "status": "ok",
                    "last_started_at": "2026-01-01T00:00:00+00:00",
                    "stdout_truncated": True,
                    "stdout_bytes": 1234567,
                    "stderr_truncated": False,
                    "stderr_bytes": 0,
                },
                {
                    "name": "mutation_batch",
                    "status": "fail",
                    "last_started_at": "2026-01-01T00:05:00+00:00",
                    "stdout_truncated": False,
                    "stdout_bytes": 0,
                    "stderr_truncated": True,
                    "stderr_bytes": 2345678,
                },
            ]
        },
    }


def test_healthcheck_warns_for_cycle_limited_scheduled_job():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "deferred",
                    "due": True,
                    "last_deferred_at": "2026-01-01T00:00:00+00:00",
                    "last_deferred_reason": "cycle_job_limit",
                    "consecutive_deferrals": 2,
                },
                {
                    "name": "disabled_job",
                    "enabled": False,
                    "status": "deferred",
                    "last_deferred_reason": "cycle_job_limit",
                },
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"][0] == {
        "code": "scheduled_job_deferred",
        "message": "one or more enabled scheduled jobs were deferred by the per-cycle job limit",
        "detail": {
            "jobs": [
                {
                    "name": "research_cycle",
                    "status": "deferred",
                    "due": True,
                    "last_deferred_at": "2026-01-01T00:00:00+00:00",
                    "last_deferred_reason": "cycle_job_limit",
                    "consecutive_deferrals": 2,
                }
            ]
        },
    }


def test_healthcheck_fails_for_excessive_cycle_limited_scheduled_job_deferrals():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "deferred",
                    "due": True,
                    "last_deferred_at": "2026-01-01T00:00:00+00:00",
                    "last_deferred_reason": "cycle_job_limit",
                    "consecutive_deferrals": 3,
                },
                {
                    "name": "mutation_batch",
                    "enabled": True,
                    "status": "deferred",
                    "due": True,
                    "last_deferred_reason": "cycle_job_limit",
                    "consecutive_deferrals": 2,
                },
            ]
        ),
        max_consecutive_job_deferrals=3,
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "scheduled_job_deferral_limit",
        "message": "one or more enabled scheduled jobs exceeded the consecutive deferral limit",
        "detail": {
            "jobs": [
                {
                    "name": "research_cycle",
                    "status": "deferred",
                    "due": True,
                    "last_deferred_at": "2026-01-01T00:00:00+00:00",
                    "last_deferred_reason": "cycle_job_limit",
                    "consecutive_deferrals": 3,
                    "max_consecutive_job_deferrals": 3,
                }
            ]
        },
    }
    assert health["warnings"][0]["code"] == "scheduled_job_deferred"


def test_healthcheck_can_ignore_overdue_scheduled_jobs():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "ok",
                    "due": True,
                    "age_seconds": 180_000.0,
                    "effective_cadence_seconds": 86_400.0,
                    "timeout_seconds": 900,
                }
            ]
        ),
        fail_on_job_overdue=False,
    )

    assert health["ok"] is True
    assert health["issues"] == []


def test_healthcheck_can_ignore_never_run_scheduled_jobs_with_overdue_switch():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_synthetic_smoke",
                    "enabled": True,
                    "status": "never_run",
                    "due": True,
                    "cadence_seconds": 86400.0,
                    "effective_cadence_seconds": 86400.0,
                    "timeout_seconds": 300,
                }
            ]
        ),
        fail_on_job_overdue=False,
    )

    assert health["ok"] is True
    assert health["issues"] == []


def test_healthcheck_warns_for_stale_promotion_review_packet():
    health = evaluate_health(
        operator_report(
            promotion_reviews=[
                {
                    "job": "active_income_promotion_review",
                    "product": "active_income",
                    "enabled": True,
                    "exists": True,
                    "path": "runtime/active_income_promotion_review.json",
                    "status": "ready",
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "age_seconds": 200000.0,
                    "max_age_seconds": 172800.0,
                    "fresh": False,
                    "needs_approval": 1,
                }
            ]
        )
    )

    assert health["ok"] is True
    warning = next(item for item in health["warnings"] if item["code"] == "promotion_review_stale")
    assert warning["detail"]["reviews"] == [
        {
            "job": "active_income_promotion_review",
            "product": "active_income",
            "path": "runtime/active_income_promotion_review.json",
            "status": "ready",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "age_seconds": 200000.0,
            "max_age_seconds": 172800.0,
            "fresh": False,
            "needs_approval": 1,
        }
    ]


def test_healthcheck_warns_for_unsafe_mutation_handoff_artifacts():
    health = evaluate_health(
        operator_report(
            mutation_plan={
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "path": "runtime/mutation_plan.json",
                "summary": {"executable": True},
            },
            mutation_batch={
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
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    warning = next(
        item for item in health["warnings"] if item["code"] == "research_handoff_warning"
    )
    assert warning == {
        "code": "research_handoff_warning",
        "message": "research handoff artifacts are stale, unsafe, or failed",
        "detail": {
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
        },
    }


def test_healthcheck_warns_for_failed_artifact_hygiene_report():
    health = evaluate_health(
        operator_report(
            artifact_hygiene={
                "ok": False,
                "summary": {
                    "quarantine_candidates": 1,
                    "unreferenced_active_artifacts": 2,
                    "historical_search_outputs": 3,
                    "errors": 1,
                    "quarantined": 0,
                },
                "errors": [
                    {
                        "scope": "unreferenced_active_artifact",
                        "path": "outputs/active_strategies_old.json",
                        "error": "ValueError: refusing to quarantine symlink source",
                    }
                ],
            }
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    warning = next(
        item for item in health["warnings"] if item["code"] == "artifact_hygiene_unhealthy"
    )
    assert warning == {
        "code": "artifact_hygiene_unhealthy",
        "message": "artifact hygiene reported cleanup or inspection failures",
        "detail": {
            "summary": {
                "quarantine_candidates": 1,
                "unreferenced_active_artifacts": 2,
                "historical_search_outputs": 3,
                "errors": 1,
                "quarantined": 0,
            },
            "errors": [
                {
                    "scope": "unreferenced_active_artifact",
                    "path": "outputs/active_strategies_old.json",
                    "error": "ValueError: refusing to quarantine symlink source",
                }
            ],
        },
    }


def test_healthcheck_warns_for_future_promotion_review_packet():
    health = evaluate_health(
        operator_report(
            promotion_reviews=[
                {
                    "job": "active_income_promotion_review",
                    "product": "active_income",
                    "enabled": True,
                    "exists": True,
                    "path": "runtime/active_income_promotion_review.json",
                    "status": "ready",
                    "generated_at": "2026-01-01T00:20:00+00:00",
                    "age_seconds": None,
                    "max_age_seconds": 172800.0,
                    "fresh": False,
                    "reason": "future_generated_at",
                    "needs_approval": 1,
                }
            ]
        )
    )

    assert health["ok"] is True
    warning = next(item for item in health["warnings"] if item["code"] == "promotion_review_stale")
    assert warning["detail"]["reviews"] == [
        {
            "job": "active_income_promotion_review",
            "product": "active_income",
            "path": "runtime/active_income_promotion_review.json",
            "status": "ready",
            "generated_at": "2026-01-01T00:20:00+00:00",
            "age_seconds": None,
            "max_age_seconds": 172800.0,
            "fresh": False,
            "reason": "future_generated_at",
            "needs_approval": 1,
        }
    ]


def test_healthcheck_fails_for_failed_backup_verification():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/bad.zip",
                "verification": {
                    "ok": False,
                    "issues": [{"code": "sha256_mismatch", "arcname": "runtime/status.json"}],
                },
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "backup_unhealthy"
    assert health["issues"][0]["detail"] == {
        "ok": True,
        "output": "runtime/backups/bad.zip",
        "verification_ok": False,
        "verification_issues": [{"code": "sha256_mismatch", "arcname": "runtime/status.json"}],
    }


def test_healthcheck_fails_when_existing_recovery_file_was_skipped_from_backup():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/incomplete.zip",
                "manifest": {
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "critical_skipped_files": 1,
                    "files": [
                        {
                            "path": "runtime/active_income_state.json",
                            "role": "product:active_income:product_state",
                            "reason": "too_large",
                            "exists": True,
                            "included": False,
                            "required_if_present": True,
                        }
                    ],
                },
                "verification": {"ok": True, "issues": []},
            }
        ),
        now_ts=1767225600.0,
    )

    assert health["ok"] is False
    assert health["issues"] == [
        {
            "code": "backup_incomplete",
            "message": "latest backup omitted one or more existing recovery files",
            "detail": {
                "output": "runtime/backups/incomplete.zip",
                "critical_skipped_files": 1,
                "files": [
                    {
                        "path": "runtime/active_income_state.json",
                        "role": "product:active_income:product_state",
                        "reason": "too_large",
                    }
                ],
            },
        }
    ]


def test_healthcheck_fails_when_enabled_backup_job_has_no_report():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": True,
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                }
            ]
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "backup_report_missing",
        "message": "backup job is enabled but no backup report is available",
        "detail": {
            "scheduled_jobs": [
                {
                    "name": "runtime_backup",
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                    "cadence_seconds": None,
                }
            ]
        },
    }


def test_healthcheck_tracks_dedicated_backup_timer_without_generic_job():
    health = evaluate_health(
        operator_report(
            backup_schedule={
                "enabled": True,
                "name": "runtime_backup_timer",
                "cadence_seconds": 86400,
                "timeout_seconds": 60,
            }
        )
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "backup_report_missing",
        "message": "backup job is enabled but no backup report is available",
        "detail": {
            "scheduled_jobs": [
                {
                    "name": "runtime_backup_timer",
                    "status": "dedicated_timer",
                    "effective_cadence_seconds": 86400,
                    "cadence_seconds": 86400,
                }
            ]
        },
    }


def test_healthcheck_allows_missing_backup_report_when_no_backup_job_is_enabled():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": False,
                    "status": "disabled",
                    "effective_cadence_seconds": 86400,
                }
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []


def test_healthcheck_passes_for_recent_verified_backup():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/good.zip",
                "manifest": {"generated_at": "2026-01-01T00:00:00+00:00"},
                "verification": {"ok": True, "issues": []},
            },
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": True,
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                }
            ],
        ),
        now_ts=1767312000.0,  # 2026-01-02T00:00:00+00:00
    )

    assert health["ok"] is True
    assert health["issues"] == []


def test_healthcheck_fails_for_stale_verified_backup():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/old.zip",
                "manifest": {"generated_at": "2026-01-01T00:00:00+00:00"},
                "verification": {"ok": True, "issues": []},
            },
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": True,
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                }
            ],
        ),
        now_ts=1767484801.0,  # 2026-01-04T00:00:01+00:00
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "backup_stale"
    assert health["issues"][0]["detail"] == {
        "output": "runtime/backups/old.zip",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "age_seconds": 259201.0,
        "limit_seconds": 172800.0,
    }


def test_healthcheck_fails_for_future_verified_backup_timestamp():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/future.zip",
                "manifest": {"generated_at": "2026-01-02T00:00:01+00:00"},
                "verification": {"ok": True, "issues": []},
            },
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": True,
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                }
            ],
        ),
        now_ts=1767312000.0,  # 2026-01-02T00:00:00+00:00
    )

    assert health["ok"] is False
    assert health["issues"][0] == {
        "code": "backup_timestamp_future",
        "message": "latest verified backup is timestamped in the future",
        "detail": {
            "output": "runtime/backups/future.zip",
            "generated_at": "2026-01-02T00:00:01+00:00",
        },
    }


def test_healthcheck_backup_age_override_can_be_stricter_than_job_cadence():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/good.zip",
                "manifest": {"generated_at": "2026-01-01T00:00:00+00:00"},
                "verification": {"ok": True, "issues": []},
            },
            scheduled_jobs=[
                {
                    "name": "runtime_backup",
                    "enabled": True,
                    "status": "ok",
                    "effective_cadence_seconds": 86400,
                }
            ],
        ),
        now_ts=1767312000.0,  # 2026-01-02T00:00:00+00:00
        max_backup_age_seconds=12 * 60 * 60,
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "backup_stale"
    assert health["issues"][0]["detail"]["limit_seconds"] == 43200.0


def test_healthcheck_fails_when_backup_timestamp_is_missing():
    health = evaluate_health(
        operator_report(
            backup_report={
                "ok": True,
                "output": "runtime/backups/unknown.zip",
                "verification": {"ok": True, "issues": []},
            }
        ),
        now_ts=1767312000.0,
    )

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "backup_timestamp_missing"
    assert health["issues"][0]["detail"] == {
        "output": "runtime/backups/unknown.zip",
        "generated_at": None,
    }


def test_healthcheck_can_ignore_scheduled_job_failures():
    health = evaluate_health(
        operator_report(
            scheduled_jobs=[
                {
                    "name": "research_cycle",
                    "enabled": True,
                    "status": "fail",
                    "consecutive_failures": 3,
                }
            ]
        ),
        fail_on_job_failures=False,
    )

    assert health["ok"] is True
    assert health["warnings"] == []


def test_healthcheck_warns_when_paper_product_waits_for_strategy_artifact():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_trading_income",
                    "market": "futures",
                    "mode": "paper",
                    "reason": "waiting_for_strategy_artifact",
                    "strategy_artifact": "outputs/active_strategies_flow.json",
                    "detail": "Strategy artifact not found: outputs/active_strategies_flow.json",
                },
                {
                    "name": "btc_accumulation",
                    "enabled": True,
                    "market": "spot",
                    "mode": "paper",
                    "reason": "cycle_ok",
                },
                {
                    "name": "disabled_research",
                    "enabled": False,
                    "market": "futures",
                    "mode": "paper",
                    "reason": "waiting_for_strategy_artifact",
                },
            ]
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "paper_product_waiting_for_strategy_artifact",
            "message": "paper product is waiting for an exported strategy artifact",
            "detail": {
                "name": "active_income",
                "objective": "active_trading_income",
                "market": "futures",
                "strategy_artifact": "outputs/active_strategies_flow.json",
                "detail": "Strategy artifact not found: outputs/active_strategies_flow.json",
            },
        }
    ]


def test_healthcheck_warns_when_research_found_no_exportable_strategies():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "objective": "active_income",
                    "market": "futures",
                    "mode": "paper",
                    "reason": "waiting_for_strategy_artifact",
                }
            ],
            research_cycle={
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "summary": {
                    "hypotheses": 12,
                    "keepers": 0,
                    "exported": 0,
                    "top_reasons": {"no_train_edge": 8, "insufficient_train_trades": 4},
                    "next_actions": ["continue rotating curated candidates"],
                    "mutation_effectiveness": {
                        "evaluated_hypotheses": 4,
                        "keepers": 0,
                        "outcome": "no_keeper",
                    },
                },
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"][1] == {
        "code": "research_cycle_no_exportable_strategies",
        "message": "research has run but found no exportable strategy candidates",
        "detail": {
            "generated_at": "2026-01-01T01:05:00+00:00",
            "hypotheses": 12,
            "top_reasons": {"no_train_edge": 8, "insufficient_train_trades": 4},
            "next_actions": ["continue rotating curated candidates"],
            "waiting_products": [
                {
                    "name": "active_income",
                    "objective": "active_income",
                    "market": "futures",
                }
            ],
            "mutation_effectiveness": {
                "evaluated_hypotheses": 4,
                "keepers": 0,
                "outcome": "no_keeper",
            },
        },
    }


def test_healthcheck_warns_when_research_cycle_recovered_state():
    health = evaluate_health(
        operator_report(
            research_cycle={
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "state_recovered": True,
                "state_error": "JSONDecodeError: bad state",
                "summary": {"hypotheses": 0, "keepers": 0, "exported": 0},
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "research_cycle_state_recovered",
            "message": "research cycle recovered a corrupt or invalid state file",
            "detail": {
                "generated_at": "2026-01-01T01:05:00+00:00",
                "state_error": "JSONDecodeError: bad state",
            },
        }
    ]


def test_healthcheck_warns_when_research_cycle_ignored_mutation_batch_read_error():
    health = evaluate_health(
        operator_report(
            research_cycle={
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "mutation_batch": {
                    "status": "read_error",
                    "path": "runtime/mutation_hypotheses.json",
                    "error": "JSONDecodeError: bad mutation batch",
                },
                "summary": {"hypotheses": 0, "keepers": 0, "exported": 0},
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "research_cycle_mutation_batch_read_error",
            "message": "research cycle ignored a mutation batch it could not read",
            "detail": {
                "generated_at": "2026-01-01T01:05:00+00:00",
                "path": "runtime/mutation_hypotheses.json",
                "error": "JSONDecodeError: bad mutation batch",
            },
        }
    ]


def test_healthcheck_surfaces_generated_batch_and_experiment_memory_health():
    health = evaluate_health(
        operator_report(
            generated_batch={
                "ok": True,
                "schema": "autopilot.generative_strategy_batch/v1",
                "generated_at": "2026-07-10T00:01:00+00:00",
                "research_only": True,
                "executable": False,
                "paper_trade_allowed": False,
                "promotion_allowed": False,
                "live_allowed": False,
                "requires_full_validation_before_export": True,
                "hypotheses_count": 3,
                "summary": {"hypotheses": 3},
            },
            experiment_memory={
                "ok": True,
                "status": "ready",
                "path": "runtime/research/experiment_memory.sqlite3",
                "integrity": {"ok": True},
                "adaptive_evidence_scope": "development_only",
                "protected_holdout_results_excluded": True,
                "feedback": {
                    "totals": {"strategies": 9, "evaluations": 12},
                    "outcomes": {"reject": 8, "pre_holdout_pass": 4},
                },
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == []
    assert health["generative_research"] == {
        "batch_ok": True,
        "batch_generated_at": "2026-07-10T00:01:00+00:00",
        "batch_hypotheses": 3,
        "unique_behavioral_specs": 9,
        "recorded_evaluations": 12,
        "memory_status": "ready",
        "memory_integrity_ok": True,
        "adaptive_evidence_scope": "development_only",
        "protected_holdout_results_excluded": True,
    }


def test_healthcheck_warns_when_generated_research_pipeline_is_unhealthy():
    health = evaluate_health(
        operator_report(
            research_cycle={
                "ok": False,
                "generated_at": "2026-07-10T00:02:00+00:00",
                "generated_batch": {
                    "status": "ignored",
                    "path": "runtime/research/generated_hypotheses.json",
                    "reason": "generated_batch_failed_safety_contract",
                },
            },
            generated_batch={
                "ok": True,
                "schema": "autopilot.generative_strategy_batch/v1",
                "generated_at": "2026-07-10T00:01:00+00:00",
                "research_only": True,
                "executable": True,
                "paper_trade_allowed": False,
                "promotion_allowed": False,
                "live_allowed": False,
                "requires_full_validation_before_export": True,
                "hypotheses_count": 0,
                "summary": {"hypotheses": 0, "rejected_attempts": 100},
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert [warning["code"] for warning in health["warnings"]] == [
        "research_cycle_generated_batch_unhealthy",
        "generated_batch_unhealthy",
        "generative_search_empty",
    ]
    assert health["warnings"][0]["detail"]["status"] == "ignored"
    assert health["warnings"][1]["detail"]["unsafe_flags"] == ["executable"]


def test_healthcheck_fails_closed_for_bad_experiment_memory_or_feedback_scope():
    health = evaluate_health(
        operator_report(
            experiment_memory={
                "ok": False,
                "status": "error",
                "path": "runtime/research/experiment_memory.sqlite3",
                "error": "ExperimentMemoryCorruptionError: quick_check failed",
                "protected_holdout_results_excluded": False,
            }
        )
    )

    assert health["ok"] is False
    assert [issue["code"] for issue in health["issues"]] == [
        "experiment_memory_unhealthy",
        "experiment_memory_feedback_scope_invalid",
    ]
    assert health["issues"][0]["detail"]["status"] == "error"


def test_healthcheck_does_not_warn_no_exportable_before_research_runs():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "market": "futures",
                    "mode": "paper",
                    "reason": "waiting_for_strategy_artifact",
                }
            ],
            research_cycle={"ok": True, "summary": {"hypotheses": 0, "keepers": 0, "exported": 0}},
        )
    )

    assert health["ok"] is True
    assert [warning["code"] for warning in health["warnings"]] == [
        "paper_product_waiting_for_strategy_artifact"
    ]


def test_healthcheck_warns_when_required_testnet_rehearsal_is_missing():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "market": "futures",
                    "mode": "paper",
                    "require_testnet_rehearsal": True,
                }
            ],
            testnet_rehearsal={
                "required": True,
                "required_by": ["active_income"],
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "exists": False,
                "ok": False,
                "next_action": {
                    "preflight_command": "make preflight PRODUCT=active_income REQUIRE_TESTNET=1",
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100",
                },
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "required_testnet_rehearsal_not_ready",
            "message": "required exchange testnet rehearsal is missing, stale, or failed",
            "detail": {
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "next_action": {
                    "preflight_command": "make preflight PRODUCT=active_income REQUIRE_TESTNET=1",
                    "rehearsal_command": "make testnet-rehearsal CONFIRM=1 NOTIONAL_USD=100",
                },
            },
        }
    ]


def test_healthcheck_fails_when_live_required_testnet_rehearsal_is_missing():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "objective": "active_income",
                    "enabled": True,
                    "market": "futures",
                    "mode": "live",
                    "require_testnet_rehearsal": True,
                }
            ],
            testnet_rehearsal={
                "required": True,
                "required_by": ["active_income"],
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "exists": False,
                "ok": False,
            },
        )
    )

    assert health["ok"] is False
    assert health["warnings"] == []
    assert health["issues"] == [
        {
            "code": "live_required_testnet_rehearsal_not_ready",
            "message": "live product requires a successful recent exchange testnet rehearsal",
            "detail": {
                "status": "missing",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "live_products": [
                    {
                        "name": "active_income",
                        "objective": "active_income",
                        "market": "futures",
                        "mode": "live",
                    }
                ],
            },
        }
    ]


def test_healthcheck_surfaces_invalid_testnet_rehearsal_reasons():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "market": "futures",
                    "mode": "paper",
                    "require_testnet_rehearsal": True,
                }
            ],
            testnet_rehearsal={
                "required": True,
                "required_by": ["active_income"],
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": False,
                "testnet": False,
                "final_position_flat": False,
                "invalid_reasons": ["not_testnet", "final_position_not_flat"],
            },
        )
    )

    assert health["ok"] is True
    assert health["warnings"] == [
        {
            "code": "required_testnet_rehearsal_not_ready",
            "message": "required exchange testnet rehearsal is missing, stale, or failed",
            "detail": {
                "status": "failed",
                "path": "runtime/testnet_rehearsal_report.json",
                "required_by": ["active_income"],
                "testnet": False,
                "final_position_flat": False,
                "invalid_reasons": ["not_testnet", "final_position_not_flat"],
            },
        }
    ]


def test_healthcheck_does_not_warn_when_required_testnet_rehearsal_is_ok():
    health = evaluate_health(
        operator_report(
            products=[
                {
                    "name": "active_income",
                    "enabled": True,
                    "market": "futures",
                    "mode": "paper",
                    "require_testnet_rehearsal": True,
                }
            ],
            testnet_rehearsal={
                "required": True,
                "required_by": ["active_income"],
                "status": "ok",
                "path": "runtime/testnet_rehearsal_report.json",
                "ok": True,
                "fresh": True,
                "testnet": True,
                "final_position_flat": True,
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == []


def test_healthcheck_surfaces_warning_alerts_without_failing():
    health = evaluate_health(
        operator_report(
            readiness_alert={"sent": True, "fingerprint": "ready123"},
            research_handoff_alert={
                "sent": False,
                "reason": "cooldown",
                "fingerprint": "research123",
            },
            testnet_rehearsal_alert={"sent": True, "fingerprint": "testnet123"},
            promotion_alert={
                "sent": True,
                "fingerprint": "promo123",
                "webhook": {"ok": False},
                "state_error": "OSError: cannot write alert_state.json",
            },
        )
    )

    assert health["ok"] is True
    assert health["issues"] == []
    assert health["warnings"] == [
        {
            "code": "readiness_warning_alert",
            "message": "autopilot has active readiness warnings",
            "detail": {"sent": True, "fingerprint": "ready123"},
        },
        {
            "code": "research_handoff_warning_alert",
            "message": "autopilot has active research handoff warnings",
            "detail": {"sent": False, "reason": "cooldown", "fingerprint": "research123"},
        },
        {
            "code": "testnet_rehearsal_warning_alert",
            "message": "autopilot has active testnet rehearsal warnings",
            "detail": {"sent": True, "fingerprint": "testnet123"},
        },
        {
            "code": "promotion_warning_alert",
            "message": "autopilot has active promotion warnings",
            "detail": {
                "sent": True,
                "fingerprint": "promo123",
                "webhook": {"ok": False},
                "state_error": "OSError: cannot write alert_state.json",
            },
        },
    ]


def test_build_healthcheck_emits_alert_for_blocking_issues(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: operator_report(ok=False),
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    health = build_healthcheck(cfg, emit_failure_alert=True)

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "cycle_failed"
    assert health["healthcheck_alert"]["sent"] is True
    alert = json.loads(cfg.alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["severity"] == "critical"
    assert alert["title"] == "autopilot healthcheck failed"
    assert alert["detail"]["issues"][0]["code"] == "cycle_failed"


def test_build_healthcheck_does_not_repeat_unchanged_incident(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: operator_report(ok=False),
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    first = build_healthcheck(cfg, emit_failure_alert=True)
    second = build_healthcheck(
        cfg,
        emit_failure_alert=True,
        previous_health=first,
    )

    assert first["healthcheck_alert"]["sent"] is True
    assert second["healthcheck_alert"] == {
        "sent": False,
        "reason": "unchanged_incident",
        "incident_signature": first["incident_signature"],
    }
    assert len(cfg.alert_file.read_text(encoding="utf-8").splitlines()) == 1


def test_build_healthcheck_does_not_realert_when_incident_shrinks_or_flaps(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )

    def report_for(*names):
        return operator_report(
            scheduled_jobs=[
                {
                    "name": name,
                    "enabled": True,
                    "status": "fail",
                    "due": False,
                    "consecutive_failures": 1,
                }
                for name in names
            ]
        )

    current = {"report": report_for("research_factory", "ml_research")}
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: current["report"],
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    first = build_healthcheck(cfg, emit_failure_alert=True)
    current["report"] = report_for("research_factory")
    second = build_healthcheck(cfg, emit_failure_alert=True, previous_health=first)
    current["report"] = report_for("research_factory", "ml_research")
    third = build_healthcheck(cfg, emit_failure_alert=True, previous_health=second)

    assert first["healthcheck_alert"]["sent"] is True
    assert second["healthcheck_alert"]["reason"] == "unchanged_incident"
    assert third["healthcheck_alert"]["reason"] == "unchanged_incident"
    assert len(cfg.alert_file.read_text(encoding="utf-8").splitlines()) == 1


def test_build_healthcheck_alerts_for_new_identity_during_existing_incident(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    current = {
        "report": operator_report(
            scheduled_jobs=[
                {
                    "name": "research_factory",
                    "enabled": True,
                    "status": "fail",
                    "due": False,
                    "consecutive_failures": 1,
                }
            ]
        )
    }
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: current["report"],
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    first = build_healthcheck(cfg, emit_failure_alert=True)
    current["report"] = operator_report(
        scheduled_jobs=[
            {
                "name": name,
                "enabled": True,
                "status": "fail",
                "due": False,
                "consecutive_failures": 1,
            }
            for name in ("research_factory", "portfolio_risk")
        ]
    )
    second = build_healthcheck(cfg, emit_failure_alert=True, previous_health=first)

    assert second["healthcheck_alert"]["sent"] is True
    assert len(cfg.alert_file.read_text(encoding="utf-8").splitlines()) == 2


def test_build_healthcheck_records_alert_failure_without_crashing(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: operator_report(ok=False),
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    def fail_alert(**_kwargs):
        raise OSError("alert path unavailable")

    monkeypatch.setattr("src.autopilot.healthcheck.emit_alert", fail_alert)

    health = build_healthcheck(cfg, emit_failure_alert=True)

    assert health["ok"] is False
    assert health["issues"][0]["code"] == "cycle_failed"
    assert health["healthcheck_alert"] == {"sent": False, "error": "alert path unavailable"}
    assert not cfg.alert_file.exists()


def test_oneshot_healthcheck_drains_queued_remote_alert_before_exit(monkeypatch):
    waits = []
    monkeypatch.setattr(
        "src.autopilot.healthcheck.wait_for_remote_alerts",
        lambda timeout: waits.append(timeout) or True,
    )
    health = {
        "healthcheck_alert": {
            "sent": True,
            "remote_delivery": {"status": "queued"},
        }
    }

    _drain_oneshot_remote_alert(health)

    assert waits == [REMOTE_ALERT_DRAIN_SECONDS]
    assert health["healthcheck_alert"]["remote_delivery"] == {
        "status": "queued",
        "drained": True,
    }


def test_oneshot_healthcheck_records_remote_alert_drain_timeout(monkeypatch):
    monkeypatch.setattr(
        "src.autopilot.healthcheck.wait_for_remote_alerts",
        lambda _timeout: False,
    )
    health = {
        "healthcheck_alert": {
            "sent": True,
            "remote_delivery": {"status": "queued"},
        }
    }

    _drain_oneshot_remote_alert(health)

    remote = health["healthcheck_alert"]["remote_delivery"]
    assert remote["drained"] is False
    assert "did not drain" in remote["drain_error"]


def test_build_healthcheck_does_not_alert_when_ok(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: operator_report(),
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    health = build_healthcheck(cfg, emit_failure_alert=True)

    assert health["ok"] is True
    assert "healthcheck_alert" not in health
    assert not cfg.alert_file.exists()


def test_build_healthcheck_emits_one_recovery_after_previous_failure(monkeypatch, tmp_path):
    cfg = AutopilotConfig(
        alert_file=tmp_path / "alerts.jsonl",
        alert_state_file=tmp_path / "alert_state.json",
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_operator_report",
        lambda config: operator_report(),
    )
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_readiness_report",
        lambda config, **_kwargs: {"ok": True, "checks": []},
    )

    health = build_healthcheck(
        cfg,
        emit_failure_alert=True,
        previous_health={
            "ok": False,
            "issues": [{"code": "scheduled_job_failed", "message": "research failed"}],
        },
    )

    assert health["ok"] is True
    assert health["healthcheck_recovery_alert"]["sent"] is True
    alert = json.loads(cfg.alert_file.read_text(encoding="utf-8").splitlines()[0])
    assert alert["title"] == "autopilot healthcheck recovered"
    assert alert["detail"]["cleared_issue_codes"] == ["scheduled_job_failed"]
    assert alert["detail"]["operator_action_required"] is False


def test_healthcheck_cli_prints_json_when_output_write_fails(monkeypatch, tmp_path, capsys):
    output = tmp_path / "healthcheck.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "healthcheck",
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(output),
            "--no-alert",
        ],
    )
    monkeypatch.setattr("src.autopilot.healthcheck.load_config", lambda path: AutopilotConfig())
    monkeypatch.setattr(
        "src.autopilot.healthcheck.build_healthcheck",
        lambda *args, **kwargs: {
            "ok": True,
            "issues": [],
            "warnings": [],
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "operator_report_generated_at": "2026-01-01T00:00:00+00:00",
            "readiness_ok": True,
        },
    )

    def fail_write(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr("src.autopilot.healthcheck.write_json_atomic", fail_write)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["issues"] == [
        {
            "code": "healthcheck_output_write_failed",
            "message": "healthcheck could not write its JSON output file",
            "detail": {"path": str(output), "error": "OSError: disk full"},
        }
    ]
    assert printed["output_error"] == {"path": str(output), "error": "disk full"}
    assert printed["config"] == str(tmp_path / "config.json")


def test_healthcheck_cli_prints_json_when_config_load_fails(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "bad_config.json"
    monkeypatch.setattr(
        "sys.argv",
        ["healthcheck", "--config", str(config_path), "--no-alert"],
    )

    def fail_load(path):
        raise ValueError("invalid config")

    monkeypatch.setattr("src.autopilot.healthcheck.load_config", fail_load)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "ok": False,
        "issues": [
            {
                "code": "healthcheck_build_failed",
                "message": "healthcheck could not load config or build its report",
                "detail": {"config": str(config_path), "error": "ValueError: invalid config"},
            }
        ],
        "warnings": [],
        "config": str(config_path),
    }


def test_healthcheck_cli_prints_json_when_report_build_fails(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        "sys.argv",
        ["healthcheck", "--config", str(config_path), "--no-alert"],
    )
    monkeypatch.setattr("src.autopilot.healthcheck.load_config", lambda path: AutopilotConfig())

    def fail_build(*args, **kwargs):
        raise RuntimeError("operator report unavailable")

    monkeypatch.setattr("src.autopilot.healthcheck.build_healthcheck", fail_build)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["issues"] == [
        {
            "code": "healthcheck_build_failed",
            "message": "healthcheck could not load config or build its report",
            "detail": {
                "config": str(config_path),
                "error": "RuntimeError: operator report unavailable",
            },
        }
    ]
    assert printed["warnings"] == []
    assert printed["config"] == str(config_path)
