import csv
import json

import pytest

from src.autopilot.config import AutopilotConfig, JobConfig, ProductConfig
from src.autopilot.jobs import job_definition_fingerprint
from src.autopilot.reporting import build_operator_report, main, render_operator_markdown


def product(tmp_path, **overrides):
    payload = {
        "name": "active_income",
        "enabled": True,
        "objective": "active_income",
        "base_asset": "USDT",
        "market": "futures",
        "execution_mode": "paper",
        "symbol": "BTCUSDT",
        "strategies_path": tmp_path / "active.json",
        "state_file": tmp_path / "state.json",
        "trade_log": tmp_path / "trades.csv",
        "starting_equity": 1000.0,
    }
    payload.update(overrides)
    return ProductConfig(**payload)


def write_trades(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exit_time", "net_return", "sized_return"])
        writer.writeheader()
        writer.writerow({"exit_time": "2026-01-01T00:00:00Z", "net_return": "0.02", "sized_return": "0.002"})
        writer.writerow({"exit_time": "2026-01-01T01:00:00Z", "net_return": "-0.01", "sized_return": "-0.001"})


def test_operator_report_summarizes_status_approvals_and_trades(tmp_path):
    trade_log = tmp_path / "trades.csv"
    write_trades(trade_log)
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                    "generated_at": "1970-01-01T00:02:00+00:00",
                    "control": {
                        "paused": False,
                        "pause_jobs": True,
                        "paused_jobs": ["market_data_update_futures"],
                        "flatten_products": ["active_income"],
                    },
                "products": [
                    {
                        "ok": True,
                        "product": {"name": "active_income"},
                        "equity": 1001.0,
                        "open_positions": 1,
                        "state_errors": [
                            {"field": "paper_state", "error": "example state warning"}
                        ],
                        "open_position_details": [
                            {
                                "strategy_id": "live_r1",
                                "direction": "short",
                                "position_size": 0.25,
                                "entry_price": 100.0,
                                "sl_price": 101.0,
                                "tp_price": 98.0,
                                "entry_time": "2026-01-01T00:00:00+00:00",
                                "base_timeframe": "5m",
                                "horizon_bars": 6,
                                "stale_after_seconds": 3600.0,
                            }
                        ],
                    }
                ],
                "jobs": [{"name": "smoke", "ok": True, "stdout_tail": "ok"}],
                "reporting": {
                    "ok": False,
                    "errors": [
                        {
                            "stage": "operator_report_json_write_failed",
                            "path": "runtime/operator_report.json",
                            "error": "OSError: disk full",
                        }
                    ],
                },
                "promotion_alert": {"sent": True},
            }
        ),
        encoding="utf-8",
    )
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(json.dumps({"approvals": {"abc": {}, "def": {}}}), encoding="utf-8")
    research_smoke_file = tmp_path / "research_smoke.json"
    research_smoke_file.write_text(
        json.dumps({"ok": True, "generated_at": "2026-01-01T01:00:00+00:00", "scenarios": [{}, {}]}),
        encoding="utf-8",
    )
    research_cycle_file = tmp_path / "research_cycle.json"
    research_cycle_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "scenarios": [{}],
                "exports": [{"exported": False}],
                "summary": {
                    "scenarios": 1,
                    "opportunity_types_by_product": {
                        "active_income": {
                            "scalping": 1,
                            "day_trading": 1,
                            "swing_trading": 1,
                        }
                    },
                    "keepers": 0,
                    "incubation_candidates": 3,
                    "exported": 0,
                    "top_reasons": {"failed_validation": 1, "no_train_edge": 4},
                    "mutation_effectiveness": {
                        "status": "loaded",
                        "evaluated_hypotheses": 2,
                        "keepers": 0,
                        "outcome": "no_keeper",
                        "top_reasons": {"no_train_edge": 2},
                    },
                    "next_actions": [
                        "continue rotating curated candidates; no positive train edge found yet"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    incubation_candidates_file = tmp_path / "incubation_candidates.json"
    incubation_candidates_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:05:30+00:00",
                "research_only": True,
                "executable": False,
                "paper_trade_allowed": False,
                "live_allowed": False,
                "promotion_eligible": False,
                "summary": {
                    "candidates": 3,
                    "by_product": {"active_income": 2, "btc_accumulation": 1},
                },
                "products": {},
            }
        ),
        encoding="utf-8",
    )
    mutation_plan_file = tmp_path / "mutation_plan.json"
    mutation_plan_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "source": {"research_generated_at": "2026-01-01T01:05:00+00:00"},
                "summary": {
                    "proposals": 4,
                    "by_product": {"active_income": 3, "btc_accumulation": 1},
                    "skipped_scenarios": 2,
                    "suppressed_repeated_sources": 1,
                    "suppressed_by_product": {"active_income": 1},
                    "suppressed_by_reason": {"no_train_edge": 1},
                    "executable": False,
                },
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    mutation_batch_file = tmp_path / "mutation_hypotheses.json"
    mutation_batch_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:07:00+00:00",
                "source": {"plan_generated_at": "2026-01-01T01:06:00+00:00"},
                "research_only": True,
                "executable": False,
                "count": 4,
                "summary": {
                    "hypotheses": 4,
                    "skipped": 1,
                    "by_product": {"active_income": 3, "btc_accumulation": 1},
                    "executable": False,
                },
                "hypotheses": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        approval_ledger=approval_ledger,
        research_smoke_file=research_smoke_file,
        research_cycle_file=research_cycle_file,
        incubation_candidates_file=incubation_candidates_file,
        mutation_plan_file=mutation_plan_file,
        mutation_batch_file=mutation_batch_file,
        job_state_file=tmp_path / "job_state.json",
        jobs=[
            JobConfig(
                name="market_data_update_futures",
                enabled=True,
                command=["python", "-m", "src.update_candles"],
                cadence_seconds=60,
                timeout_seconds=10,
                working_dir=tmp_path,
            ),
            JobConfig(
                name="disabled_heavy_search",
                enabled=False,
                command=["python", "-m", "src.sweep"],
                cadence_seconds=3600,
                timeout_seconds=10,
                working_dir=tmp_path,
            ),
        ],
        products=[product(tmp_path, trade_log=trade_log)],
    )
    cfg.job_state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "market_data_update_futures": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": True,
                        "last_returncode": 0,
                        "last_duration_seconds": 2.5,
                        "last_reason": "fresh",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["ok"] is True
    assert report["status_heartbeat"]["fresh"] is True
    assert report["status_heartbeat"]["age_seconds"] == 80.0
    assert report["status_heartbeat"]["limit_seconds"] == 300.0
    assert report["approval_count"] == 2
    assert report["research_smoke"]["ok"] is True
    assert report["research_cycle"]["compacted"] is True
    assert report["research_cycle"]["scenarios_count"] == 1
    assert "scenarios" not in report["research_cycle"]
    assert report["incubation_candidates"]["compacted"] is True
    assert "products" not in report["incubation_candidates"]
    assert report["mutation_plan"]["compacted"] is True
    assert report["mutation_plan"]["proposals_count"] == 0
    assert "proposals" not in report["mutation_plan"]
    assert report["mutation_batch"]["compacted"] is True
    assert report["mutation_batch"]["hypotheses_count"] == 0
    assert "hypotheses" not in report["mutation_batch"]
    assert report["scheduled_jobs"][0]["status"] == "ok"
    assert report["scheduled_jobs"][0]["due"] is True
    assert report["scheduled_jobs"][1]["status"] == "disabled"
    assert report["products"][0]["state_errors"] == [
        {"field": "paper_state", "error": "example state warning"}
    ]
    assert report["products"][0]["open_position_details"][0]["strategy_id"] == "live_r1"
    assert report["products"][0]["open_position_details"][0]["sl_price"] == 101.0
    assert report["products"][0]["trade_summary"]["trades"] == 2
    assert report["products"][0]["trade_summary"]["win_rate"] == 0.5
    assert "active_income" in markdown
    assert "state error paper_state: example state warning" in markdown
    assert "## Open Positions" in markdown
    assert "| active_income | paper | futures | live_r1 | short |  | 0.25 | 100.0 | 101.0 | 98.0 |" in markdown
    assert "Flatten products: `active_income`" in markdown
    assert "Pause jobs: `True`" in markdown
    assert "Paused jobs: `market_data_update_futures`" in markdown
    assert "Status heartbeat: `ok` (age 80s, limit 300s)" in markdown
    assert "Error alert: `none`" in markdown
    assert "Readiness alert: `none`" in markdown
    assert "Research handoff alert: `none`" in markdown
    assert "mutations 2 tested/0 keepers (no_keeper)" in markdown
    assert report["promotion_alert"]["sent"] is True
    assert "Promotion alert: `sent`" in markdown
    assert "Research smoke: `ok` (2 synthetic scenarios" in markdown
    assert "Research cycle: `ok` (1 real scenarios, keepers 0, watchlist 3, exports 0, top reason no_train_edge" in markdown
    assert (
        "Incubation queue: `ok` (3 research-only candidates, active_income 2, btc_accumulation 1, "
        "research_only `True`, executable `False`, paper `False`, live `False`"
    ) in markdown
    assert (
        "Mutation plan: `ok` (4 research-only proposals, active_income 3, btc_accumulation 1, "
        "skipped scenarios 2, suppressed repeats 1 (active_income 1), "
        "suppressed reasons no_train_edge 1, source current"
    ) in markdown
    assert (
        "Mutation batch: `ok` (4 research-only hypotheses, active_income 3, btc_accumulation 1, "
        "skipped 1, executable `False`, source current"
    ) in markdown
    assert "executable `False`" in markdown
    assert "Promotion reviews: `none configured`" in markdown
    assert "next continue rotating curated candidates" in markdown
    assert "coverage active_income: scalping, day_trading, swing_trading" in markdown
    assert report["scheduled_jobs"][0]["last_reason"] == "fresh"
    assert report["reporting"] == {
        "ok": False,
        "errors": [
            {
                "stage": "operator_report_json_write_failed",
                "path": "runtime/operator_report.json",
                "error": "OSError: disk full",
            }
        ],
    }
    assert "| market_data_update_futures | `True` | `ok` | `True` | 2026-01-01T00:00:00+00:00 | fresh |" in markdown
    assert "| disabled_heavy_search | `False` | `disabled` | `False` | never |  |" in markdown
    assert (
        "| active_income | paper | futures | ok | cycle | 1 | 1001.0000 | 2 | 50.00% | "
        "0.10% | state error paper_state: example state warning |"
    ) in markdown


def test_operator_report_surfaces_flatten_failure_diagnostics(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": False,
                "generated_at": "1970-01-01T00:00:00+00:00",
                "products": [
                    {
                        "ok": False,
                        "action": "flatten",
                        "product": {"name": "active_income"},
                        "broker": "fake-live",
                        "close_error": "RuntimeError: exchange timeout",
                        "spot_step_aside": {
                            "strategy_id": "btc_step_aside",
                            "quote_value": 50.0,
                            "requested_qty": 0.4,
                        },
                        "fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.4, "price": 125.0, "fee": 0.02},
                        "local_state": {"path": str(tmp_path / "state.json"), "recovered": False},
                        "position_before": {"symbol": "BTCUSDT", "qty": 0.5, "avg_price": 100.0},
                        "position_after_error": "RuntimeError: readback timeout",
                        "position_after_attempt": {
                            "symbol": "BTCUSDT",
                            "qty": 0.5,
                            "avg_price": 100.0,
                            "is_flat": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        products=[product(tmp_path, execution_mode="live")],
    )

    report = build_operator_report(cfg, now_ts=120.0)
    markdown = render_operator_markdown(report)

    product_report = report["products"][0]
    assert product_report["cycle_ok"] is False
    assert product_report["action"] == "flatten"
    assert product_report["broker"] == "fake-live"
    assert product_report["close_error"] == "RuntimeError: exchange timeout"
    assert product_report["spot_step_aside"]["quote_value"] == 50.0
    assert product_report["fill"]["side"] == "buy"
    assert product_report["local_state"]["recovered"] is False
    assert product_report["position_after_error"] == "RuntimeError: readback timeout"
    assert product_report["position_after_attempt"]["qty"] == 0.5
    assert "RuntimeError: exchange timeout; after attempt qty 0.5" in markdown


def test_operator_markdown_renders_open_position_age_and_stale_status():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T02:00:00+00:00",
            "status_generated_at": "2026-01-01T02:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "runtime_load_errors": [],
            "status_heartbeat": {"fresh": True, "age_seconds": 1, "limit_seconds": 300},
            "market_data": {},
            "indicator_features": {},
            "regime_data": {},
            "research_smoke": {},
            "research_cycle": {},
            "incubation_candidates": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "backup_report": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [
                {
                    "name": "active_income",
                    "mode": "live",
                    "market": "futures",
                    "cycle_ok": True,
                    "action": "cycle",
                    "open_positions": 1,
                    "trade_summary": {"trades": 0, "win_rate": None, "sized_return_sum": 0.0},
                    "open_position_details": [
                        {
                            "strategy_id": "active_short",
                            "direction": "short",
                            "position_size": 0.5,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T00:00:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.1,
                            "broker_requested_qty": 0.1,
                            "broker_fill_ratio": 1.0,
                        }
                    ],
                }
            ],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "## Open Positions" in markdown
    assert (
        "| active_income | live | futures | active_short | short | "
        "BTCUSDT, sell, qty 0.1/0.1, fill 100.00% | 0.5 | 100.0 | 102.0 | 96.0 | "
        "2026-01-01T00:00:00+00:00 | 7200s | 5m x 6 | 5400s | yes |"
    ) in markdown


def test_operator_markdown_renders_future_open_position_age():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T02:00:00+00:00",
            "status_generated_at": "2026-01-01T02:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "runtime_load_errors": [],
            "status_heartbeat": {"fresh": True, "age_seconds": 1, "limit_seconds": 300},
            "market_data": {},
            "indicator_features": {},
            "regime_data": {},
            "research_smoke": {},
            "research_cycle": {},
            "incubation_candidates": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "backup_report": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [
                {
                    "name": "active_income",
                    "mode": "paper",
                    "market": "futures",
                    "cycle_ok": True,
                    "action": "cycle",
                    "open_positions": 1,
                    "trade_summary": {"trades": 0, "win_rate": None, "sized_return_sum": 0.0},
                    "open_position_details": [
                        {
                            "strategy_id": "paper_short",
                            "direction": "short",
                            "position_size": 0.5,
                            "entry_price": 100.0,
                            "sl_price": 102.0,
                            "tp_price": 96.0,
                            "entry_time": "2026-01-01T02:10:00+00:00",
                            "base_timeframe": "5m",
                            "horizon_bars": 6,
                            "stale_after_seconds": 5400.0,
                        }
                    ],
                }
            ],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert (
        "| active_income | paper | futures | paper_short | short |  | 0.5 | 100.0 | 102.0 | 96.0 | "
        "2026-01-01T02:10:00+00:00 | -600s | 5m x 6 | 5400s | no |"
    ) in markdown


def test_operator_markdown_renders_btc_step_aside_broker_quote_metadata():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T02:00:00+00:00",
            "status_generated_at": "2026-01-01T02:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "runtime_load_errors": [],
            "status_heartbeat": {"fresh": True, "age_seconds": 1, "limit_seconds": 300},
            "market_data": {},
            "indicator_features": {},
            "regime_data": {},
            "research_smoke": {},
            "research_cycle": {},
            "incubation_candidates": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "backup_report": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [
                {
                    "name": "btc_accumulation",
                    "mode": "live",
                    "market": "spot",
                    "cycle_ok": True,
                    "action": "cycle",
                    "open_positions": 1,
                    "trade_summary": {"trades": 0, "win_rate": None, "sized_return_sum": 0.0},
                    "open_position_details": [
                        {
                            "strategy_id": "btc_step_aside",
                            "direction": "short",
                            "position_size": 0.35,
                            "entry_price": 100.0,
                            "sl_price": 101.0,
                            "tp_price": 98.0,
                            "entry_time": "2026-01-01T01:00:00+00:00",
                            "base_timeframe": "4h",
                            "horizon_bars": 6,
                            "stale_after_seconds": 259200.0,
                            "broker_symbol": "BTCUSDT",
                            "broker_side": "sell",
                            "broker_qty": 0.5,
                            "broker_requested_qty": 0.5,
                            "broker_fill_ratio": 1.0,
                            "broker_entry_quote_value": 50.0,
                            "broker_exit_sizing": "quote_reinvest",
                        }
                    ],
                }
            ],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert (
        "| btc_accumulation | live | spot | btc_step_aside | short | "
        "BTCUSDT, sell, qty 0.5/0.5, fill 100.00%, quote 50.0, quote_reinvest |"
    ) in markdown


def test_operator_report_marks_stale_research_handoffs():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T01:10:00+00:00",
            "ok": True,
            "status_heartbeat": {"fresh": True, "age_seconds": 1, "limit_seconds": 300},
            "approval_count": 0,
            "runtime_load_errors": [],
            "market_data": {},
            "indicator_features_by_market": {},
            "regime_data": {},
            "research_cycle": {
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "summary": {"scenarios": 1, "keepers": 0, "incubation_candidates": 1, "exported": 0},
                "scenarios": [{}],
                "exports": [],
            },
            "incubation_candidates": {},
            "mutation_plan": {
                "ok": True,
                "generated_at": "2026-01-01T01:06:00+00:00",
                "source": {"research_generated_at": "2026-01-01T00:55:00+00:00"},
                "summary": {"proposals": 1, "by_product": {}, "skipped_scenarios": 0},
            },
            "mutation_batch": {
                "ok": True,
                "generated_at": "2026-01-01T01:07:00+00:00",
                "source": {"plan_generated_at": "2026-01-01T00:56:00+00:00"},
                "summary": {"hypotheses": 1, "by_product": {}, "skipped": 0},
            },
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "backup_report": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
        }
    )

    assert (
        "source stale (research 2026-01-01T01:05:00+00:00, "
        "plan source 2026-01-01T00:55:00+00:00)"
    ) in markdown
    assert (
        "source stale (plan 2026-01-01T01:06:00+00:00, "
        "batch source 2026-01-01T00:56:00+00:00)"
    ) in markdown


def test_operator_report_surfaces_invalid_trade_log_rows(tmp_path):
    trade_log = tmp_path / "trades.csv"
    trade_log.parent.mkdir(parents=True, exist_ok=True)
    with trade_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exit_time", "net_return", "sized_return"])
        writer.writeheader()
        writer.writerow({"exit_time": "2026-01-01T00:00:00Z", "net_return": "0.02", "sized_return": "0.002"})
        writer.writerow({"exit_time": "2026-01-01T01:00:00Z", "net_return": "bad", "sized_return": "also_bad"})
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                "products": [
                    {
                        "ok": True,
                        "product": {"name": "active_income"},
                        "equity": 1001.0,
                        "open_positions": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, trade_log=trade_log)],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)
    trades = report["products"][0]["trade_summary"]

    assert trades["trades"] == 2
    assert trades["wins"] == 1
    assert trades["invalid_rows"] == 1
    assert trades["numeric_errors"] == [
        {"line": 3, "field": "net_return", "value": "bad"},
        {"line": 3, "field": "sized_return", "value": "also_bad"},
    ]
    assert trades["issue"] == "trade log has 1 row(s) with invalid numeric fields"
    assert "trade log has 1 row(s) with invalid numeric fields" in markdown


def test_operator_report_surfaces_missing_trade_log_return_fields(tmp_path):
    trade_log = tmp_path / "trades.csv"
    trade_log.parent.mkdir(parents=True, exist_ok=True)
    with trade_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exit_time", "exit_reason"])
        writer.writeheader()
        writer.writerow({"exit_time": "2026-01-01T00:00:00Z", "exit_reason": "time"})
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                "products": [
                    {
                        "ok": True,
                        "product": {"name": "active_income"},
                        "equity": 1001.0,
                        "open_positions": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, trade_log=trade_log)],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    trades = report["products"][0]["trade_summary"]

    assert trades["trades"] == 1
    assert trades["wins"] == 0
    assert trades["invalid_rows"] == 1
    assert trades["numeric_errors"] == [
        {"line": 2, "field": "net_return", "value": None},
        {"line": 2, "field": "sized_return", "value": None},
    ]
    assert trades["issue"] == "trade log has 1 row(s) with invalid numeric fields"


def test_operator_report_surfaces_invalid_broker_exit_trade_fields(tmp_path):
    trade_log = tmp_path / "trades.csv"
    trade_log.parent.mkdir(parents=True, exist_ok=True)
    with trade_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exit_time",
                "net_return",
                "sized_return",
                "broker_exit_qty",
                "broker_exit_price",
                "broker_exit_fee",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "exit_time": "2026-01-01T00:00:00Z",
                "net_return": "0.02",
                "sized_return": "0.002",
                "broker_exit_qty": "0",
                "broker_exit_price": "nan",
                "broker_exit_fee": "-0.01",
            }
        )
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                "products": [
                    {
                        "ok": True,
                        "product": {"name": "active_income"},
                        "equity": 1001.0,
                        "open_positions": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        approval_ledger=tmp_path / "approvals.json",
        products=[product(tmp_path, trade_log=trade_log)],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    trades = report["products"][0]["trade_summary"]

    assert trades["invalid_rows"] == 1
    assert trades["numeric_errors"] == [
        {"line": 2, "field": "broker_exit_qty", "value": "0"},
        {"line": 2, "field": "broker_exit_price", "value": "nan"},
        {"line": 2, "field": "broker_exit_fee", "value": "-0.01"},
    ]
    assert trades["issue"] == "trade log has 1 row(s) with invalid numeric fields"


def test_operator_report_surfaces_malformed_runtime_json(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text('{"ok": true,', encoding="utf-8")
    research_cycle_file = tmp_path / "research_cycle.json"
    research_cycle_file.write_text('{"ok": true}', encoding="utf-8")
    cfg = AutopilotConfig(
        status_file=status_file,
        approval_ledger=tmp_path / "approvals.json",
        research_cycle_file=research_cycle_file,
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["runtime_load_errors"][0]["name"] == "status"
    assert report["runtime_load_errors"][0]["path"] == str(status_file)
    assert "JSONDecodeError" in report["runtime_load_errors"][0]["error"]
    assert "Runtime file issues: `status: JSONDecodeError" in markdown


def test_operator_report_surfaces_malformed_runtime_status_sections(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "products": {"active_income": {"ok": True}},
                "jobs": ["bad-job-entry"],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, products=[])

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["runtime_load_errors"] == []
    assert report["runtime_shape_errors"] == [
        {
            "name": "status",
            "path": str(status_file),
            "field": "products",
            "error": "expected list, got dict",
        },
        {
            "name": "status",
            "path": str(status_file),
            "field": "jobs",
            "error": "expected list entries to be JSON objects",
            "invalid_entries": [{"index": 0, "type": "str"}],
        },
    ]
    assert "Runtime file issues: `status: expected list, got dict" in markdown


def test_operator_report_surfaces_malformed_job_state_json(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state_file = tmp_path / "job_state.json"
    job_state_file.write_text("{", encoding="utf-8")
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state_file,
        jobs=[
            JobConfig(
                name="research_synthetic_smoke",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_smoke"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["runtime_load_errors"][0]["name"] == "job_state"
    assert report["runtime_load_errors"][0]["path"] == str(job_state_file)
    assert "JSONDecodeError" in report["runtime_load_errors"][0]["error"]
    assert "Runtime file issues: `job_state: JSONDecodeError" in markdown


def test_operator_markdown_surfaces_testnet_rehearsal_status():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "alert": None,
            "readiness_alert": None,
            "promotion_alert": None,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "markets": {}},
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "testnet_rehearsal": {
                "ok": True,
                "status": "ok",
                "product": "active_income",
                "notional_usd": 5.0,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "final_position_flat": True,
            },
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Testnet rehearsal: `ok` (ok, active_income, notional $5" in markdown


def test_operator_markdown_surfaces_missing_testnet_rehearsal_next_action():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "alert": None,
            "readiness_alert": None,
            "promotion_alert": None,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "markets": {}},
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "testnet_rehearsal": {
                "ok": False,
                "status": "missing",
                "next_action": {"status_command": "make testnet-status"},
            },
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Testnet rehearsal: `missing` (missing, next make testnet-status)" in markdown


def test_operator_markdown_surfaces_invalid_testnet_rehearsal_reasons():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "alert": None,
            "readiness_alert": None,
            "promotion_alert": None,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "markets": {}},
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "testnet_rehearsal": {
                "ok": False,
                "status": "failed",
                "product": "active_income",
                "final_position_flat": False,
                "invalid_reasons": ["not_testnet", "final_position_not_flat"],
                "next_action": {"status_command": "make testnet-status"},
            },
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert (
        "Testnet rehearsal: `failed` "
        "(failed, active_income, final position not flat, "
        "invalid: not_testnet, final_position_not_flat, next make testnet-status)"
    ) in markdown


def test_operator_report_uses_required_product_testnet_rehearsal_path(tmp_path):
    rehearsal = tmp_path / "custom_testnet_rehearsal.json"
    rehearsal.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "1970-01-01T00:03:10+00:00",
                "generated_ts": 190.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=tmp_path / "missing_status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[
            product(
                tmp_path,
                require_testnet_rehearsal=True,
                testnet_rehearsal_report=rehearsal,
                testnet_rehearsal_max_age_seconds=60,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["testnet_rehearsal"]["ok"] is True
    assert report["testnet_rehearsal"]["path"] == str(rehearsal)
    assert report["testnet_rehearsal"]["required"] is True
    assert report["testnet_rehearsal"]["required_by"] == ["active_income"]
    assert report["products"][0]["require_testnet_rehearsal"] is True
    assert report["products"][0]["testnet_rehearsal_report"] == str(rehearsal)


def test_operator_report_rejects_wrong_product_testnet_rehearsal(tmp_path):
    rehearsal = tmp_path / "custom_testnet_rehearsal.json"
    rehearsal.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "1970-01-01T00:03:10+00:00",
                "generated_ts": 190.0,
                "product": {
                    "name": "active_income",
                    "objective": "active_income",
                    "base_asset": "USDT",
                    "market": "futures",
                    "symbol": "ETHUSDT",
                },
                "exchange": "binanceusdm",
                "testnet": True,
                "risk_controls": {
                    "max_futures_leverage": 1,
                    "futures_margin_mode": "isolated",
                    "max_notional_usd": 100.0,
                    "max_fill_slippage_bps": 100.0,
                },
                "notional_usd": 5.0,
                "order_qty": 0.05,
                "entry_fill": {"symbol": "BTCUSDT", "side": "buy", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1000.0},
                "close_fill": {"symbol": "BTCUSDT", "side": "sell", "qty": 0.05, "price": 100.0, "fee": 0.01, "timestamp": 1001.0},
                "final_position_qty": 0.0,
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=tmp_path / "missing_status.json",
        approval_ledger=tmp_path / "approvals.json",
        products=[
            product(
                tmp_path,
                require_testnet_rehearsal=True,
                testnet_rehearsal_report=rehearsal,
                testnet_rehearsal_max_age_seconds=60,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["testnet_rehearsal"]["ok"] is False
    assert report["testnet_rehearsal"]["status"] == "failed"
    assert report["testnet_rehearsal"]["invalid_reasons"] == ["product_symbol_mismatch"]
    assert report["testnet_rehearsal"]["report_product"]["symbol"] == "ETHUSDT"
    assert report["testnet_rehearsal"]["expected_product"]["symbol"] == "BTCUSDT"


def test_operator_markdown_surfaces_alert_outcomes():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "alert": {"sent": False, "reason": "cooldown"},
            "readiness_alert": {
                "sent": True,
                "state_error": "OSError: cannot write alert_state.json",
            },
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "markets": {}},
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Error alert: `not sent (cooldown)`" in markdown
    assert "Readiness alert: `sent, state error: OSError: cannot write alert_state.json`" in markdown


def test_operator_report_summarizes_approval_status_and_latest_event(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "fingerprint": "sha256:abc123456789000",
                        "status": "approved",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": "henrique",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                        "history": [
                            {
                                "event": "revoked",
                                "event_at": "2026-01-01T02:00:00+00:00",
                                "actor": "henrique",
                                "status": "revoked",
                                "strategy_id": "s1",
                                "product": {
                                    "name": "active_income",
                                    "objective": "active_income",
                                    "market": "futures",
                                    "symbol": "BTCUSDT",
                                    "base_asset": "USDT",
                                },
                                "artifact_path": "outputs/active_strategies_flow.json",
                                "revocation_reason": "temporary halt",
                            }
                        ],
                    },
                    "sha256:def456789abc000": {
                        "status": "revoked",
                        "approved_at": "2026-01-01T01:00:00+00:00",
                        "approved_by": "henrique",
                        "revoked_at": "2026-01-02T00:00:00+00:00",
                        "revoked_by": "henrique",
                        "revocation_reason": "paper drawdown breached",
                        "strategy_id": "s2",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_count"] == 2
    assert report["approval_summary"]["counts"] == {"approved": 1, "revoked": 1}
    assert report["approval_summary"]["by_product"] == {
        "active_income/BTCUSDT": {"approved": 1, "revoked": 1}
    }
    assert report["approval_summary"]["latest_event"]["event"] == "revoked"
    assert report["approval_summary"]["latest_event"]["actor"] == "henrique"
    assert [event["event"] for event in report["approval_summary"]["recent_events"][:3]] == [
        "revoked",
        "revoked",
        "approved",
    ]
    assert report["approval_summary"]["recent_events"][1]["strategy_id"] == "s1"
    assert report["approval_summary"]["recent_events"][1]["revocation_reason"] == "temporary halt"
    assert "Strategy approvals: `2` (approved 1, revoked 1); latest revoked s2 def456789abc" in markdown
    assert "for active_income/BTCUSDT by henrique at 2026-01-02T00:00:00+00:00" in markdown
    assert "reason paper drawdown breached" in markdown


def test_operator_report_surfaces_research_cycle_recovery_and_mutation_read_error(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    research_cycle_file = tmp_path / "research_cycle.json"
    research_cycle_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "2026-01-01T01:05:00+00:00",
                "state_recovered": True,
                "state_error": "JSONDecodeError: bad state",
                "mutation_batch": {
                    "status": "read_error",
                    "path": "runtime/mutation_hypotheses.json",
                    "error": "JSONDecodeError: bad mutation batch",
                },
                "summary": {
                    "scenarios": 1,
                    "keepers": 0,
                    "incubation_candidates": 0,
                    "exported": 0,
                    "top_reasons": {"no_train_edge": 1},
                    "next_actions": ["continue bounded search"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        research_cycle_file=research_cycle_file,
        approval_ledger=tmp_path / "approvals.json",
        job_state_file=tmp_path / "job_state.json",
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["research_cycle"]["state_recovered"] is True
    assert report["research_cycle"]["state_error"] == "JSONDecodeError: bad state"
    assert report["research_cycle"]["mutation_batch"] == {
        "status": "read_error",
        "path": "runtime/mutation_hypotheses.json",
        "error": "JSONDecodeError: bad mutation batch",
    }
    assert "state recovered" in markdown
    assert "mutation batch read_error" in markdown


def test_operator_report_marks_blank_actor_approval_invalid(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "status": "approved",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": " ",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_summary"]["counts"] == {"invalid_actor": 1}
    assert report["approval_summary"]["by_product"] == {
        "active_income/BTCUSDT": {"invalid_actor": 1}
    }
    assert "Strategy approvals: `1` (invalid_actor 1)" in markdown


def test_operator_report_marks_approval_fingerprint_mismatch_invalid(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "fingerprint": "sha256:wrong",
                        "status": "approved",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": "henrique",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_summary"]["counts"] == {"fingerprint_mismatch": 1}
    assert report["approval_summary"]["by_product"] == {
        "active_income/BTCUSDT": {"fingerprint_mismatch": 1}
    }
    assert "Strategy approvals: `1` (fingerprint_mismatch 1)" in markdown


def test_operator_report_marks_missing_approval_fingerprint_invalid(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "status": "approved",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": "henrique",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_summary"]["counts"] == {"fingerprint_mismatch": 1}
    assert report["approval_summary"]["by_product"] == {
        "active_income/BTCUSDT": {"fingerprint_mismatch": 1}
    }
    assert "Strategy approvals: `1` (fingerprint_mismatch 1)" in markdown


def test_operator_report_marks_invalid_revocation_audit(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "status": "revoked",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": "henrique",
                        "revoked_at": "2026-01-02T00:00:00+00:00",
                        "revoked_by": " ",
                        "revocation_reason": "",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_summary"]["counts"] == {"invalid_revocation_audit": 1}
    assert report["approval_summary"]["by_product"] == {
        "active_income/BTCUSDT": {"invalid_revocation_audit": 1}
    }
    assert report["approval_summary"]["latest_event"]["audit_reasons"] == [
        "invalid_revoked_by",
        "missing_revocation_reason",
    ]
    assert "Strategy approvals: `1` (invalid_revocation_audit 1)" in markdown
    assert "audit invalid_revoked_by/missing_revocation_reason" in markdown


def test_operator_report_allows_revocation_reason_to_mention_automation(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(
        json.dumps(
            {
                "approvals": {
                    "sha256:abc123456789000": {
                        "status": "revoked",
                        "approved_at": "2026-01-01T00:00:00+00:00",
                        "approved_by": "henrique",
                        "revoked_at": "2026-01-02T00:00:00+00:00",
                        "revoked_by": "reviewer",
                        "revocation_reason": "system outage",
                        "strategy_id": "s1",
                        "artifact_path": "outputs/active_strategies_flow.json",
                        "product": {
                            "name": "active_income",
                            "objective": "active_income",
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "base_asset": "USDT",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, approval_ledger=approval_ledger)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["approval_summary"]["counts"] == {"revoked": 1}
    assert report["approval_summary"]["latest_event"]["revocation_reason"] == "system outage"
    assert "Strategy approvals: `1` (revoked 1)" in markdown
    assert "audit invalid" not in markdown


def test_operator_report_loads_configured_backup_report(tmp_path):
    backup_report = tmp_path / "backup_report.json"
    backup_report.write_text(
        json.dumps(
            {
                "ok": True,
                "output": "runtime/backups/autopilot_state.zip",
                "archive_size_bytes": 1234,
                "verification": {"ok": True, "checked_files": 3, "issues": []},
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=tmp_path / "missing_status.json",
        approval_ledger=tmp_path / "missing_approvals.json",
        backup_report_file=backup_report,
        products=[],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["backup_report"]["output"] == "runtime/backups/autopilot_state.zip"
    assert report["backup_report"]["verification"]["ok"] is True


def test_operator_report_surfaces_non_object_runtime_json_load_error(tmp_path):
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text("[]", encoding="utf-8")
    cfg = AutopilotConfig(
        status_file=tmp_path / "missing_status.json",
        approval_ledger=approval_ledger,
        products=[],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["approval_count"] == 0
    assert report["runtime_load_errors"] == [
        {
            "name": "approval_ledger",
            "path": str(approval_ledger),
            "error": "TypeError: expected JSON object, got list",
        }
    ]


def test_operator_report_surfaces_malformed_runtime_nested_maps(tmp_path):
    job_state = tmp_path / "job_state.json"
    job_state.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    approval_ledger = tmp_path / "approvals.json"
    approval_ledger.write_text(json.dumps({"version": 1, "approvals": []}), encoding="utf-8")
    cfg = AutopilotConfig(
        status_file=tmp_path / "missing_status.json",
        job_state_file=job_state,
        approval_ledger=approval_ledger,
        products=[],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["runtime_shape_errors"] == [
        {
            "name": "job_state",
            "path": str(job_state),
            "field": "jobs",
            "error": "expected dict, got list",
        },
        {
            "name": "approval_ledger",
            "path": str(approval_ledger),
            "field": "approvals",
            "error": "expected dict, got list",
        },
    ]


def test_operator_markdown_surfaces_market_data_bootstrap_command():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": "2026-01-01T00:00:00+00:00",
            "ok": True,
            "approval_count": 0,
            "alert": None,
            "readiness_alert": None,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {
                "ok": False,
                "markets": {
                    "spot": {
                        "ok": False,
                        "reason": "missing_seed_dataset",
                        "remediation": {
                            "command": [
                                ".venv/bin/python",
                                "-m",
                                "src.update_candles",
                                "--market",
                                "spot",
                                "--bootstrap-days",
                                "365",
                                "--timeframes",
                                "1h",
                                "4h",
                                "1d",
                                "1w",
                            ]
                        },
                    }
                },
            },
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Market data: `fail` (spot: missing_seed_dataset)" in markdown
    assert (
        "Market data remediation: `spot: .venv/bin/python -m src.update_candles "
        "--market spot --bootstrap-days 365 --timeframes 1h 4h 1d 1w`"
    ) in markdown


def test_operator_report_marks_missing_seed_bootstrap_job_due(tmp_path, monkeypatch):
    seed_path = tmp_path / "missing" / "BTCUSDT_1m.parquet"
    monkeypatch.setattr("src.autopilot.jobs.default_1m_candle_path", lambda market: seed_path)
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state_file = tmp_path / "jobs.json"
    job_state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "market_data_update_spot": {
                        "last_started_ts": 100.0,
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_ok": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state_file,
        jobs=[
            JobConfig(
                name="market_data_update_spot",
                enabled=True,
                command=[
                    "python",
                    "-m",
                    "src.update_candles",
                    "--market",
                    "spot",
                    "--bootstrap-days",
                    "365",
                ],
                cadence_seconds=6 * 60 * 60,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=120.0)

    assert report["scheduled_jobs"][0]["due"] is True


def test_operator_report_marks_failed_market_data_job_recovered_when_data_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.autopilot.reporting.build_market_data_statuses",
        lambda markets: {"spot": {"ok": True, "reason": "fresh"}},
    )
    monkeypatch.setattr(
        "src.autopilot.reporting.build_indicator_feature_statuses",
        lambda markets, **kwargs: {"spot": {"ok": True, "timeframes": {}}},
    )
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state_file = tmp_path / "jobs.json"
    job_state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "market_data_update_spot": {
                        "last_started_ts": 100.0,
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_ok": False,
                        "last_error": "dns failure",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state_file,
        jobs=[
            JobConfig(
                name="market_data_update_spot",
                enabled=True,
                command=[
                    "python",
                    "-m",
                    "src.update_candles",
                    "--market",
                    "spot",
                    "--bootstrap-days",
                    "365",
                ],
                cadence_seconds=6 * 60 * 60,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
        products=[
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=120.0)

    assert report["scheduled_jobs"][0]["status"] == "recovered"
    assert report["scheduled_jobs"][0]["last_error"] is None
    assert report["scheduled_jobs"][0]["last_reason"] == "last failure resolved; current market data is ready"


def test_operator_report_summarizes_promotion_reviews_from_configured_jobs(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    review_path = tmp_path / "active_review.json"
    review_path.write_text(
        json.dumps(
            {
                "generated_at": "1970-01-01T00:00:00+00:00",
                "status": "ready",
                "product": {"name": "active_income"},
                "strategies": [
                    {
                        "id": "s1",
                        "recommendation": "needs_approval",
                        "approval_command": "python -m src.autopilot.approvals approve --strategy-id s1",
                    },
                    {"id": "s2", "recommendation": "not_ready"},
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        jobs=[
            JobConfig(
                name="active_income_promotion_review",
                enabled=True,
                command=[
                    "python",
                    "-m",
                    "src.autopilot.promotion",
                    "--product",
                    "active_income",
                    "--output-json",
                    str(review_path),
                ],
                cadence_seconds=86400,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["promotion_reviews"] == [
        {
            "job": "active_income_promotion_review",
            "product": "active_income",
            "enabled": True,
            "path": str(review_path),
            "exists": True,
            "status": "ready",
            "generated_at": "1970-01-01T00:00:00+00:00",
            "age_seconds": 200.0,
            "max_age_seconds": 172800.0,
            "fresh": True,
            "reason": None,
            "strategies": 2,
            "recommendations": {"needs_approval": 1, "not_ready": 1},
            "needs_approval": 1,
            "approval_commands": ["python -m src.autopilot.approvals approve --strategy-id s1"],
        }
    ]
    assert "Promotion reviews: `active_income: ready (needs_approval 1, not_ready 1 approval command available)`" in markdown


def test_operator_report_summarizes_promotion_reviews_with_inline_flags(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    review_path = tmp_path / "active_review.json"
    review_path.write_text(
        json.dumps(
            {
                "generated_at": "1970-01-01T00:00:00+00:00",
                "status": "ready",
                "strategies": [{"id": "s1", "recommendation": "not_ready"}],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        jobs=[
            JobConfig(
                name="active_income_promotion_review",
                enabled=True,
                command=[
                    "python",
                    "-m",
                    "src.autopilot.promotion",
                    "--product=active_income",
                    f"--output-json={review_path}",
                ],
                cadence_seconds=86400,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["promotion_reviews"][0]["product"] == "active_income"
    assert report["promotion_reviews"][0]["path"] == str(review_path)
    assert report["promotion_reviews"][0]["exists"] is True


def test_operator_report_marks_future_promotion_review_stale(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    review_path = tmp_path / "active_review.json"
    review_path.write_text(
        json.dumps(
            {
                "generated_at": "1970-01-01T00:03:21+00:00",
                "status": "ready",
                "product": {"name": "active_income"},
                "strategies": [{"id": "s1", "recommendation": "not_ready"}],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        jobs=[
            JobConfig(
                name="active_income_promotion_review",
                enabled=True,
                command=[
                    "python",
                    "-m",
                    "src.autopilot.promotion",
                    "--product",
                    "active_income",
                    "--output-json",
                    str(review_path),
                ],
                cadence_seconds=86400,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["promotion_reviews"][0]["fresh"] is False
    assert report["promotion_reviews"][0]["age_seconds"] is None
    assert report["promotion_reviews"][0]["reason"] == "future_generated_at"


def test_operator_report_summarizes_waiting_promotion_review(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    review_path = tmp_path / "btc_review.json"
    review_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "status": "waiting_for_strategy_artifact",
                "reason": "Strategy artifact not found: outputs/active_strategies_position.json",
                "strategies": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        jobs=[
            JobConfig(
                name="btc_accumulation_promotion_review",
                enabled=True,
                command=["python", "--output-json", str(review_path), "--product", "btc_accumulation"],
                cadence_seconds=86400,
                timeout_seconds=120,
                working_dir=tmp_path,
            )
        ],
    )

    markdown = render_operator_markdown(build_operator_report(cfg, now_ts=200.0))

    assert "btc_accumulation: waiting_for_strategy_artifact" in markdown


def test_operator_report_marks_enabled_jobs_that_never_ran(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=tmp_path / "missing_job_state.json",
        jobs=[
            JobConfig(
                name="research_synthetic_smoke",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_smoke"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["scheduled_jobs"][0]["status"] == "never_run"
    assert report["scheduled_jobs"][0]["due"] is True


def test_operator_report_surfaces_cycle_limited_scheduled_job(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_deferred_at": "2026-01-01T00:00:00+00:00",
                        "last_deferred_ts": 100.0,
                        "last_deferred_reason": "cycle_job_limit",
                        "consecutive_deferrals": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_cycle"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    job = report["scheduled_jobs"][0]
    assert job["status"] == "deferred"
    assert job["due"] is True
    assert job["last_deferred_reason"] == "cycle_job_limit"
    assert job["consecutive_deferrals"] == 3
    assert "| research_cycle | `True` | `deferred` | `True` | never | cycle_job_limit |" in markdown


def test_operator_report_marks_changed_job_definition_due(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    original = JobConfig(
        name="research_cycle",
        enabled=True,
        command=["python", "-m", "src.autopilot.research_cycle", "--old"],
        cadence_seconds=86400,
        timeout_seconds=300,
        working_dir=tmp_path,
    )
    changed = JobConfig(
        name="research_cycle",
        enabled=True,
        command=["python", "-m", "src.autopilot.research_cycle", "--new"],
        cadence_seconds=86400,
        timeout_seconds=300,
        working_dir=tmp_path,
    )
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": True,
                        "last_returncode": 0,
                        "last_duration_seconds": 1.0,
                        "definition_fingerprint": job_definition_fingerprint(original),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, job_state_file=job_state, jobs=[changed])

    report = build_operator_report(cfg, now_ts=120.0)

    assert report["scheduled_jobs"][0]["status"] == "ok"
    assert report["scheduled_jobs"][0]["due"] is True


def test_operator_report_surfaces_malformed_job_timestamp(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": "bad",
                        "last_ok": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_cycle"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=120.0)
    markdown = render_operator_markdown(report)

    assert report["scheduled_jobs"][0]["status"] == "ok"
    assert report["scheduled_jobs"][0]["due"] is True
    assert report["scheduled_jobs"][0]["age_seconds"] is None
    assert report["scheduled_jobs"][0]["last_reason"] == "invalid job state last_started_ts: 'bad'"
    assert "invalid job state last_started_ts: 'bad'" in markdown


def test_operator_report_surfaces_future_job_timestamp(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 9999.0,
                        "last_ok": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_cycle"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=120.0)
    markdown = render_operator_markdown(report)

    assert report["scheduled_jobs"][0]["status"] == "ok"
    assert report["scheduled_jobs"][0]["due"] is True
    assert report["scheduled_jobs"][0]["age_seconds"] is None
    assert report["scheduled_jobs"][0]["last_reason"] == "invalid job state future last_started_ts: 9999.0"
    assert "invalid job state future last_started_ts: 9999.0" in markdown


def test_operator_report_sanitizes_malformed_job_numeric_fields(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": False,
                        "last_duration_seconds": "slow",
                        "consecutive_failures": "many",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_cycle"],
                cadence_seconds=86400,
                timeout_seconds=300,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=120.0)
    markdown = render_operator_markdown(report)

    job = report["scheduled_jobs"][0]
    assert job["status"] == "fail"
    assert job["last_duration_seconds"] is None
    assert job["consecutive_failures"] == 0
    assert job["last_reason"] == (
        "invalid job state last_duration_seconds: 'slow'; "
        "invalid job state consecutive_failures: 'many'"
    )
    assert "invalid job state last_duration_seconds: 'slow'" in markdown


def test_operator_report_surfaces_failed_scheduled_job_reason(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "market_data_update_spot": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": False,
                        "last_returncode": 0,
                        "last_duration_seconds": 1.5,
                        "last_reason": "empty_seed_dataset",
                        "last_error": "1m candle dataset is empty",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="market_data_update_spot",
                enabled=True,
                command=["python", "-m", "src.update_candles"],
                cadence_seconds=6 * 60 * 60,
                timeout_seconds=10,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    assert report["scheduled_jobs"][0]["status"] == "fail"
    assert report["scheduled_jobs"][0]["due"] is True
    assert report["scheduled_jobs"][0]["effective_cadence_seconds"] == 900
    assert report["scheduled_jobs"][0]["last_reason"] == "empty_seed_dataset"
    assert "| market_data_update_spot | `True` | `fail` | `True` | 2026-01-01T00:00:00+00:00 | 1m candle dataset is empty |" in markdown


def test_operator_report_surfaces_scheduled_job_structured_errors(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    errors = [
        {"task": "alert_state", "error": "ValueError: alert state path must not be a symlink"},
        {"task": "control_audit", "error": "OSError: disk full"},
        {"task": "experiment_log", "error": "OSError: permission denied"},
    ]
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "maintenance": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": False,
                        "last_returncode": 0,
                        "last_duration_seconds": 1.5,
                        "last_error": "alert_state: ValueError: alert state path must not be a symlink",
                        "last_structured_errors_count": 4,
                        "last_structured_errors": errors,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="maintenance",
                enabled=True,
                command=["python", "-m", "src.autopilot.maintenance"],
                cadence_seconds=86400,
                timeout_seconds=900,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    job = report["scheduled_jobs"][0]
    assert job["last_structured_errors_count"] == 4
    assert job["last_structured_errors"] == errors
    assert (
        "structured errors (4 total, 2 shown): control_audit: OSError: disk full; "
        "experiment_log: OSError: permission denied"
    ) in markdown


def test_operator_report_surfaces_truncated_scheduled_job_output(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"ok": True, "products": []}), encoding="utf-8")
    job_state = tmp_path / "job_state.json"
    job_state.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "research_cycle": {
                        "last_started_at": "2026-01-01T00:00:00+00:00",
                        "last_started_ts": 100.0,
                        "last_ok": True,
                        "last_returncode": 0,
                        "last_duration_seconds": 12.5,
                        "last_stdout_truncated": True,
                        "last_stdout_bytes": 1234567,
                        "last_stderr_truncated": True,
                        "last_stderr_bytes": 345678,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(
        status_file=status_file,
        job_state_file=job_state,
        jobs=[
            JobConfig(
                name="research_cycle",
                enabled=True,
                command=["python", "-m", "src.autopilot.research_cycle"],
                cadence_seconds=86400,
                timeout_seconds=900,
                working_dir=tmp_path,
            )
        ],
    )

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    job = report["scheduled_jobs"][0]
    assert job["last_stdout_truncated"] is True
    assert job["last_stdout_bytes"] == 1234567
    assert job["last_stderr_truncated"] is True
    assert job["last_stderr_bytes"] == 345678
    assert "stdout truncated (1234567 bytes); stderr truncated (345678 bytes)" in markdown


def test_operator_report_flags_stale_status_heartbeat(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps({"ok": True, "generated_at": "1970-01-01T00:00:00Z", "products": []}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, loop_sleep_seconds=60)

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    assert report["status_heartbeat"]["fresh"] is False
    assert report["status_heartbeat"]["age_seconds"] == 1000.0
    assert report["status_heartbeat"]["limit_seconds"] == 300.0
    assert report["status_heartbeat"]["reason"] == "stale"
    assert "Status heartbeat: `fail` (age 1000s, limit 300s)" in markdown


def test_operator_report_flags_future_status_heartbeat(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps({"ok": True, "generated_at": "1970-01-01T00:20:00Z", "products": []}),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file, loop_sleep_seconds=60)

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    assert report["status_heartbeat"]["fresh"] is False
    assert report["status_heartbeat"]["age_seconds"] is None
    assert report["status_heartbeat"]["limit_seconds"] == 300.0
    assert report["status_heartbeat"]["reason"] == "future_generated_at"
    assert "Status heartbeat: `fail` (age unknown, limit 300s)" in markdown


def test_operator_report_marks_missing_status_heartbeat_unknown(tmp_path):
    cfg = AutopilotConfig(status_file=tmp_path / "missing_status.json")

    report = build_operator_report(cfg, now_ts=1000.0)
    markdown = render_operator_markdown(report)

    assert report["status_heartbeat"]["fresh"] is None
    assert report["status_heartbeat"]["age_seconds"] is None
    assert "Last status: `missing`" in markdown
    assert "Status heartbeat: `unknown` (age unknown, limit 300s)" in markdown


def test_operator_report_surfaces_control_error(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": False,
                "generated_at": "1970-01-01T00:02:00+00:00",
                "control": {
                    "paused": True,
                    "pause_jobs": True,
                    "paused_products": [],
                    "paused_jobs": [],
                    "flatten_products": [],
                    "control_error": "runtime/control.json: JSONDecodeError",
                },
                "control_error": "runtime/control.json: JSONDecodeError",
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["control_error"] == "runtime/control.json: JSONDecodeError"
    assert "Control issue: `runtime/control.json: JSONDecodeError`" in markdown


def test_operator_report_surfaces_control_clear_outcomes(tmp_path):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps(
            {
                "ok": True,
                "generated_at": "1970-01-01T00:02:00+00:00",
                "control": {
                    "paused": False,
                    "pause_jobs": False,
                    "paused_products": [],
                    "paused_jobs": [],
                    "flatten_products": [],
                },
                "control_clear": [
                    {
                        "command": "clear-flatten",
                        "name": "active_income",
                        "ok": True,
                    },
                    {
                        "command": "clear-flatten",
                        "name": None,
                        "ok": True,
                        "skipped": True,
                        "reason": "flatten_all_has_failures",
                        "targets": [
                            {"product_name": "active_income", "ok": False, "error": "exchange timeout"},
                            {
                                "product_name": "btc_accumulation",
                                "ok": True,
                                "skipped": True,
                                "reason": "spot_flatten_not_supported",
                            },
                        ],
                    },
                ],
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file)

    report = build_operator_report(cfg, now_ts=200.0)
    markdown = render_operator_markdown(report)

    assert report["control_clear"][0]["name"] == "active_income"
    assert (
        "Control clear: `active_income: cleared; all: skipped (flatten_all_has_failures) "
        "[active_income failed exchange timeout, btc_accumulation skipped spot_flatten_not_supported]`"
        in markdown
    )


def test_operator_report_preserves_unknown_control_selectors(tmp_path):
    status_file = tmp_path / "status.json"
    unknown = {
        "paused_products": ["active-incme"],
        "paused_jobs": ["network-jb"],
    }
    status_file.write_text(
        json.dumps(
            {
                "ok": False,
                "generated_at": "1970-01-01T00:02:00+00:00",
                "control": {
                    "paused": True,
                    "pause_jobs": True,
                    "paused_products": ["active-incme"],
                    "paused_jobs": ["network-jb"],
                    "flatten_products": [],
                    "control_error": "unknown control selectors",
                },
                "control_error": "unknown control selectors",
                "unknown_control_selectors": unknown,
                "products": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = AutopilotConfig(status_file=status_file)

    report = build_operator_report(cfg, now_ts=200.0)

    assert report["control_error"] == "unknown control selectors"
    assert report["unknown_control_selectors"] == unknown


def test_operator_markdown_surfaces_missing_indicator_features():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "reason": "fresh"},
            "indicator_features": {
                "ok": False,
                "timeframes": {
                    "1m": {"missing_features": ["volume_z_20"]},
                    "5m": {"missing_features": []},
                },
            },
            "research_smoke": {},
            "research_cycle": {},
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Indicator features: `fail` (missing 1m: volume_z_20)" in markdown


def test_operator_markdown_surfaces_regime_data_status():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "reason": "fresh"},
            "indicator_features": {"ok": True, "timeframes": {}},
            "regime_data": {
                "ok": True,
                "datasets": [
                    {
                        "name": "regime_tag_futures_15m",
                        "available": True,
                        "rows": 100,
                        "regime_counts": {"0": 60, "1": 40},
                    }
                ],
            },
            "research_smoke": {},
            "research_cycle": {},
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Regime data: `ok`" in markdown
    assert "regime_tag_futures_15m: rows 100, regimes 0:60, 1:40" in markdown


def test_operator_markdown_surfaces_strategy_smoke_status():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "reason": "fresh"},
            "indicator_features": {"ok": True, "timeframes": {}},
            "regime_data": {"ok": True, "datasets": []},
            "research_smoke": {},
            "strategy_smoke": {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "scenarios": [
                    {
                        "name": "synthetic_strategy_sweep",
                        "ok": True,
                        "rows": 4,
                        "best_strategy": "sma_cross",
                        "best_dsr": 0.75,
                    },
                    {
                        "name": "regime_filter_sweep",
                        "ok": True,
                        "skipped": True,
                        "reason": "missing_regime_input",
                    },
                ],
            },
            "research_cycle": {},
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Strategy smoke: `ok`" in markdown
    assert "synthetic_strategy_sweep: rows 4, best sma_cross, dsr 0.750" in markdown
    assert "regime_filter_sweep: skipped missing_regime_input" in markdown


def test_operator_markdown_surfaces_backup_status():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "reason": "fresh"},
            "indicator_features": {"ok": True, "timeframes": {}},
            "regime_data": {"ok": True, "datasets": []},
            "research_smoke": {},
            "strategy_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {},
            "backup_report": {
                "ok": True,
                "output": "runtime/backups/autopilot_state.zip",
                "archive_size_bytes": 1234,
                "manifest": {"included_files": 16, "missing_files": 2, "skipped_files": 1},
                "verification": {"ok": True, "checked_files": 16, "issues": []},
                "retention": {"keep": 30, "archives": 7, "deleted_archives": 0},
            },
            "testnet_rehearsal": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Backup: `ok`" in markdown
    assert "runtime/backups/autopilot_state.zip" in markdown
    assert "included 16, missing 2, skipped 1" in markdown
    assert "verified ok, checked 16" in markdown
    assert "retention keep 30, archives 7, deleted 0" in markdown


def test_operator_markdown_surfaces_artifact_hygiene_errors():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "markets": {}},
            "indicator_features": {"ok": True, "timeframes": {}},
            "regime_data": {"ok": True, "datasets": []},
            "research_smoke": {},
            "strategy_smoke": {},
            "research_cycle": {},
            "mutation_plan": {},
            "mutation_batch": {},
            "promotion_reviews": [],
            "artifact_hygiene": {
                "ok": False,
                "dry_run": False,
                "summary": {
                    "quarantine_candidates": 1,
                    "unreferenced_active_artifacts": 2,
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
            "backup_report": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "Artifact hygiene: `fail`" in markdown
    assert "1 quarantine candidates, 2 unreferenced active artifacts" in markdown
    assert "errors 1, first unreferenced_active_artifact: ValueError: refusing to quarantine symlink source" in markdown


def test_operator_markdown_infers_research_coverage_from_older_cycle_reports():
    markdown = render_operator_markdown(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "status_heartbeat": {"fresh": True, "age_seconds": 10, "limit_seconds": 300},
            "market_data": {"ok": True, "reason": "fresh"},
            "indicator_features": {"ok": True, "timeframes": {}},
            "research_smoke": {},
            "research_cycle": {
                "ok": True,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "scenarios": [
                    {"product": "active_income", "base_tf": "1m"},
                    {"product": "active_income", "base_tf": "5m"},
                    {"product": "active_income", "base_tf": "15m"},
                    {"product": "btc_accumulation", "base_tf": "4h"},
                ],
                "exports": [],
                "summary": {
                    "scenarios": 4,
                    "keepers": 0,
                    "exported": 0,
                    "top_reasons": {"no_train_edge": 4},
                    "next_actions": ["continue rotating curated candidates"],
                },
            },
            "artifact_hygiene": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        }
    )

    assert "coverage active_income: scalping, day_trading, swing_trading" in markdown
    assert "btc_accumulation: btc_accumulation" in markdown


def test_operator_report_cli_writes_failure_report_when_config_load_fails(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "bad_config.json"
    markdown = tmp_path / "operator.md"
    json_output = tmp_path / "operator.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "operator-report",
            "--config",
            str(config_path),
            "--output",
            str(markdown),
            "--json-output",
            str(json_output),
        ],
    )

    def fail_load(path):
        raise ValueError("bad config")

    monkeypatch.setattr("src.autopilot.reporting.load_config", fail_load)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert capsys.readouterr().out.strip() == str(markdown)
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["report_errors"] == [
        {
            "code": "operator_report_build_failed",
            "message": "operator report could not load config or build its payload",
            "detail": {"config": str(config_path), "error": "ValueError: bad config"},
        }
    ]
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "Autopilot Operator Report" in markdown_text
    assert "Report issues: `operator_report_build_failed: ValueError: bad config`" in markdown_text


def test_operator_report_cli_prints_json_when_json_output_write_fails(monkeypatch, tmp_path, capsys):
    markdown = tmp_path / "operator.md"
    json_output = tmp_path / "operator.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "operator-report",
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(markdown),
            "--json-output",
            str(json_output),
        ],
    )
    monkeypatch.setattr("src.autopilot.reporting.load_config", lambda path: AutopilotConfig())
    monkeypatch.setattr(
        "src.autopilot.reporting.build_operator_report",
        lambda config: {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "status_generated_at": None,
            "ok": True,
            "approval_count": 0,
            "approval_summary": {},
            "runtime_load_errors": [],
            "market_data": {},
            "indicator_features": {},
            "regime_data": {},
            "research_smoke": {},
            "strategy_smoke": {},
            "research_cycle": {},
            "artifact_hygiene": {},
            "testnet_rehearsal": {},
            "control": {},
            "products": [],
            "scheduled_jobs": [],
            "jobs": [],
        },
    )

    def fail_json(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr("src.autopilot.reporting.write_json_atomic", fail_json)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["report_errors"] == [
        {
            "code": "operator_report_json_write_failed",
            "message": "operator report could not write JSON output",
            "detail": {"path": str(json_output), "error": "OSError: disk full"},
        }
    ]
