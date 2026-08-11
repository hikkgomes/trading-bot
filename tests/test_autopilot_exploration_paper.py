import json
from pathlib import Path

import pytest

from research_exploration.experiment_log import ExperimentRecord, log_result
from research_exploration.hypothesis_schema import ExitRule, Hypothesis, Predicate, RiskRule
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.exploration_paper import (
    SCHEMA,
    build_exploration_manifest,
    run_exploration_paper,
)


def _product(tmp_path):
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="paper",
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active.json",
        state_file=tmp_path / "active_state.json",
        trade_log=tmp_path / "active_trades.csv",
        starting_equity=1000.0,
    )


def _hypothesis():
    return Hypothesis(
        id="EXPLORE_LONG_5M",
        family="momentum_continuation",
        idea="exploration candidate",
        market_logic="observe forward signal behavior",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[Predicate("5m", "rsi_14", "ge", reference=45.0)],
        setup=[Predicate("5m", "rsi_14", "le", reference=80.0)],
        trigger=[Predicate("5m", "rsi_14", "ge", reference=55.0)],
        exit=ExitRule(take_profit=0.02, stop_loss=0.01, horizon_bars=12),
        risk=RiskRule(risk_per_trade=0.003, max_position_fraction=0.1),
    )


def _write_sources(tmp_path):
    hypothesis = _hypothesis()
    log_path = tmp_path / "experiment_log.jsonl"
    log_result(
        ExperimentRecord(
            hypothesis_id=hypothesis.id,
            family=hypothesis.family,
            direction=hypothesis.direction,
            fingerprint="experiment-fingerprint",
            verdict="reject",
            metrics={
                "train": {"trades": 40, "total_return": 0.03, "win_rate": 0.55},
                "validation": {"trades": 8, "total_return": -0.01, "win_rate": 0.4},
            },
            config={
                "eval": {
                    "market": "futures",
                    "symbol": "BTCUSDT",
                    "pnl_unit": "usdt",
                    "fee_bps": 5.0,
                    "slippage_bps": 2.0,
                }
            },
            hypothesis=hypothesis.to_dict(),
        ),
        log_path,
    )
    incubation = tmp_path / "incubation.json"
    incubation.write_text(
        json.dumps(
            {
                "schema": "autopilot.incubation_candidates/v1",
                "research_only": True,
                "executable": False,
                "paper_trade_allowed": False,
                "live_allowed": False,
                "promotion_eligible": False,
                "products": {
                    "active_income": [
                        {
                            "id": hypothesis.id,
                            "market": "futures",
                            "symbol": "BTCUSDT",
                            "pnl_unit": "usdt",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return incubation, log_path


def test_build_exploration_manifest_compiles_rejected_candidate_as_non_promotable(tmp_path):
    incubation, log_path = _write_sources(tmp_path)
    root = tmp_path / "exploration"

    manifest = build_exploration_manifest(
        AutopilotConfig(products=[_product(tmp_path)]),
        incubation_path=incubation,
        log_path=log_path,
        root=root,
    )

    assert manifest["ok"] is True
    assert manifest["summary"] == {"candidates": 1, "missing_experiment_records": 0}
    assert manifest["adaptive_evidence"] is True
    assert manifest["promotion_eligible"] is False
    artifact = json.loads(Path(manifest["candidates"][0]["artifact"]).read_text(encoding="utf-8"))
    assert artifact["paper_trade_allowed"] is True
    assert artifact["live_allowed"] is False
    assert artifact["promotion_eligible"] is False
    assert artifact["exploration_only"] is True
    assert artifact["strategies"][0]["id"] == "EXPLORE_LONG_5M"


def test_run_exploration_paper_records_decision_trace_without_promotion_evidence(
    monkeypatch,
    tmp_path,
):
    incubation, log_path = _write_sources(tmp_path)
    config = AutopilotConfig(products=[_product(tmp_path)])
    root = tmp_path / "exploration"
    build_exploration_manifest(
        config,
        incubation_path=incubation,
        log_path=log_path,
        root=root,
    )

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {"equity": 1002.0, "open_positions": {}, "drawdown_halted": False}
            self.decision_trace = {}

        def run_cycle(self):
            self.decision_trace = {
                "summary": {
                    "data_ready": 1,
                    "market_bars_processed": 1,
                    "signals": 0,
                    "entries_opened": 0,
                    "positions_managed": 0,
                    "outcomes": {"signal_not_triggered": 1},
                },
                "strategies": {
                    "EXPLORE_LONG_5M": {
                        "outcome": "signal_not_triggered",
                        "failed_stage": "trigger",
                        "failed_predicate": "5m:rsi_14 >= 55.0",
                        "latest_bar": "2026-08-10T00:00:00+00:00",
                    }
                },
            }

    monkeypatch.setattr("src.autopilot.exploration_paper.PaperTradingBot", FakeBot)

    report = run_exploration_paper(
        config,
        manifest_path=root / "manifest.json",
        previous_status_path=None,
    )

    assert report["schema"] == SCHEMA
    assert report["ok"] is True
    assert report["promotion_eligible"] is False
    assert report["summary"] == {
        "candidates": 1,
        "healthy": 1,
        "failed": 0,
        "diagnoses": {"collecting_forward_observations": 1},
    }
    assert report["aggregate"]["data_ready"] == 1
    assert report["aggregate"]["market_bars_processed"] == 1
    assert report["aggregate"]["outcomes"] == {"signal_not_triggered": 1}
    assert report["aggregate"]["failed_predicates"] == {"5m:rsi_14 >= 55.0": 1}
    feedback = next(iter(report["candidate_feedback"].values()))
    assert feedback["diagnosis"] == "collecting_forward_observations"
    assert feedback["failed_stages"] == {"trigger": 1}
    assert feedback["signal_frequency"]["regime_coverage"] == 1.0


def test_exploration_feedback_diagnoses_forward_trigger_starvation(monkeypatch, tmp_path):
    incubation, log_path = _write_sources(tmp_path)
    config = AutopilotConfig(products=[_product(tmp_path)])
    root = tmp_path / "exploration"
    manifest = build_exploration_manifest(
        config,
        incubation_path=incubation,
        log_path=log_path,
        root=root,
    )
    digest = manifest["candidates"][0]["artifact_digest"]
    status = root / "status.json"
    status.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "adaptive_evidence": True,
                "promotion_eligible": False,
                "aggregate": {"cycles": 11},
                "candidate_feedback": {
                    digest: {
                        "cycles": 11,
                        "data_ready": 11,
                        "market_bars_processed": 11,
                        "signals": 0,
                        "entries_opened": 0,
                        "positions_managed": 0,
                        "outcomes": {"signal_not_triggered": 11},
                        "failed_stages": {"trigger": 11},
                        "failed_predicates": {"5m:rsi_14 >= 55.0": 11},
                        "first_observed_at": "2026-08-09T23:00:00+00:00",
                        "last_observed_at": "2026-08-09T23:55:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeBot:
        def __init__(self, **kwargs):
            self.state = {"equity": 1000.0, "open_positions": {}, "drawdown_halted": False}
            self.decision_trace = {}

        def run_cycle(self):
            self.decision_trace = {
                "summary": {
                    "data_ready": 1,
                    "market_bars_processed": 1,
                    "signals": 0,
                    "entries_opened": 0,
                    "positions_managed": 0,
                    "outcomes": {"signal_not_triggered": 1},
                },
                "strategies": {
                    "EXPLORE_LONG_5M": {
                        "outcome": "signal_not_triggered",
                        "failed_stage": "trigger",
                        "failed_predicate": "5m:rsi_14 >= 55.0",
                        "latest_bar": "2026-08-10T00:00:00+00:00",
                    }
                },
            }

    monkeypatch.setattr("src.autopilot.exploration_paper.PaperTradingBot", FakeBot)
    report = run_exploration_paper(
        config,
        manifest_path=root / "manifest.json",
        previous_status_path=status,
    )

    feedback = report["candidate_feedback"][digest]
    assert feedback["cycles"] == 12
    assert feedback["data_ready"] == 12
    assert feedback["diagnosis"] == "trigger_never_fires"
    assert feedback["mutation_focus_stage"] == "trigger"


def test_build_exploration_manifest_rejects_weakened_incubation_contract(tmp_path):
    incubation, log_path = _write_sources(tmp_path)
    payload = json.loads(incubation.read_text(encoding="utf-8"))
    payload["promotion_eligible"] = True
    incubation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="safety contract"):
        build_exploration_manifest(
            AutopilotConfig(products=[_product(tmp_path)]),
            incubation_path=incubation,
            log_path=log_path,
            root=tmp_path / "exploration",
        )


def test_build_exploration_manifest_normalizes_legacy_nan_metrics(tmp_path):
    incubation, log_path = _write_sources(tmp_path)
    record = json.loads(log_path.read_text(encoding="utf-8"))
    record["metrics"]["train"]["win_rate"] = float("nan")
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    manifest = build_exploration_manifest(
        AutopilotConfig(products=[_product(tmp_path)]),
        incubation_path=incubation,
        log_path=log_path,
        root=tmp_path / "exploration",
    )

    artifact = json.loads(Path(manifest["candidates"][0]["artifact"]).read_text())
    assert artifact["strategies"][0]["metrics"]["train_win_rate"] is None
