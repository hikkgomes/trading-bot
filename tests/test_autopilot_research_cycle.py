import json
import math

import pytest

from research_exploration.dsr import DSR_METHOD
from research_exploration.hypothesis_generator import generate_batch
from src.autopilot import research_cycle as rc
from src.autopilot.approvals import artifact_digest
from src.autopilot.candidate_activation import product_identity
from src.autopilot.candidate_paper import candidate_paper_paths
from src.autopilot.config import ProductConfig


def _market_status(timestamp="2026-07-08T11:22:00+00:00"):
    return {
        "ok": True,
        "last_timestamp": timestamp,
        "rows": 1000,
        "path": "data/candles/futures/BTCUSDT/BTCUSDT_1m.parquet",
    }


def _market_statuses(timestamp="2026-07-08T11:22:00+00:00"):
    return {
        "futures": _market_status(timestamp),
        "spot": {"ok": False, "reason": "missing_seed_dataset", "exists": False},
    }


def _live_candidate_payload():
    return {
        "version": 2,
        "market": "futures",
        "symbol": "BTCUSDT",
        "pnl_unit": "usdt",
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "strategies": [
            {
                "id": "KEEP_THIS",
                "market": "futures",
                "symbol": "BTCUSDT",
                "base_timeframe": "5m",
                "direction": "long",
                "horizon_bars": 12,
                "take_profit": 0.02,
                "stop_loss": 0.01,
                "pnl_unit": "usdt",
                "conditions": [
                    {
                        "feature": "tf_5m_rsi_14",
                        "kind": "value_ge",
                        "threshold": 50,
                        "description": "rsi >= 50",
                    }
                ],
                "risk": {
                    "risk_per_trade": 0.003,
                    "max_position_fraction": 0.25,
                    "daily_stop_loss": -0.02,
                    "max_consecutive_losses": 3,
                    "cooldown_bars": 24,
                    "max_trades_per_day": 4,
                },
                "fees": {"fee_bps": 5, "slippage_bps": 2},
                "metrics": {
                    "holdout_total_return": 0.03,
                    "dsr_deflated": 0.72,
                    "dsr_method": DSR_METHOD,
                    "n_trials": 20,
                    "sr_std_trials": 0.18,
                    "trial_sharpe_count": 12,
                    "trial_sharpe_observed_std": 0.16,
                    "trial_sharpe_conservative_floor": 0.10,
                },
            }
        ],
    }


def test_default_research_cycle_includes_guarded_btc_accumulation_scenarios():
    by_name = {scenario.name: scenario for scenario in rc.DEFAULT_SCENARIOS}

    assert by_name["btc_accumulation_4h"].with_guards is False
    assert by_name["btc_accumulation_1h"].with_guards is False
    assert by_name["btc_accumulation_4h_guarded"].with_guards is True
    assert by_name["btc_accumulation_1h_guarded"].with_guards is True
    assert by_name["btc_accumulation_4h_guarded"].pnl_unit == "btc"
    assert by_name["btc_accumulation_4h_guarded"].market == "spot"
    assert by_name["btc_accumulation_1h_guarded"].product == "btc_accumulation"


def test_default_research_cycle_covers_active_income_horizons():
    active = {
        scenario.opportunity_type: scenario
        for scenario in rc.DEFAULT_SCENARIOS
        if scenario.product == "active_income"
    }

    assert set(active) == {"swing_trading", "day_trading", "scalping"}
    assert active["swing_trading"].base_tf == "1h"
    assert active["swing_trading"].candidate_set == "swing"
    assert active["swing_trading"].with_guards is True
    assert active["day_trading"].base_tf == "5m"
    assert active["scalping"].base_tf == "1m"
    assert all(scenario.pnl_unit == "usdt" for scenario in active.values())
    assert all(scenario.market == "futures" for scenario in active.values())


def test_protected_epoch_order_prioritizes_short_history_fast_scenarios():
    ordered = sorted(rc.DEFAULT_SCENARIOS, key=rc._protected_epoch_scenario_order)

    assert [
        scenario.opportunity_type for scenario in ordered if scenario.product == "active_income"
    ] == ["scalping", "day_trading", "swing_trading"]
    btc_timeframes = [
        scenario.base_tf for scenario in ordered if scenario.product == "btc_accumulation"
    ]
    assert btc_timeframes[:2] == ["1h", "1h"]


def test_active_income_swing_scenario_uses_multi_day_hypotheses():
    scenarios = {
        scenario.opportunity_type: scenario
        for scenario in rc.DEFAULT_SCENARIOS
        if scenario.product == "active_income"
    }

    hypotheses_by_opportunity = {
        opportunity: rc._hypotheses_for(scenario) for opportunity, scenario in scenarios.items()
    }

    assert all(hyp.base_timeframe == "1m" for hyp in hypotheses_by_opportunity["scalping"])
    assert all(hyp.base_timeframe == "5m" for hyp in hypotheses_by_opportunity["day_trading"])

    swing_hypotheses = hypotheses_by_opportunity["swing_trading"]
    assert swing_hypotheses
    assert {hyp.family for hyp in swing_hypotheses} == {
        "trend_continuation",
        "volatility_breakout",
        "mean_reversion",
        "momentum_continuation",
        "liquidity_sweep",
    }
    assert all(hyp.base_timeframe == "1h" for hyp in swing_hypotheses)
    assert all(hyp.exit.horizon_bars >= 48 for hyp in swing_hypotheses)
    assert all("multi_day" in hyp.tags for hyp in swing_hypotheses)
    assert all(hyp.risk.risk_per_trade <= 0.005 for hyp in swing_hypotheses)
    assert all(hyp.risk.max_position_fraction <= 0.15 for hyp in swing_hypotheses)

    scalp_max_seconds = max(
        hyp.exit.horizon_bars * 60 for hyp in hypotheses_by_opportunity["scalping"]
    )
    day_max_seconds = max(
        hyp.exit.horizon_bars * 5 * 60 for hyp in hypotheses_by_opportunity["day_trading"]
    )
    swing_min_seconds = min(hyp.exit.horizon_bars * 60 * 60 for hyp in swing_hypotheses)
    assert scalp_max_seconds <= 2 * 60 * 60
    assert day_max_seconds <= 8 * 60 * 60
    assert swing_min_seconds >= 2 * 24 * 60 * 60


def test_hypothesis_selection_skips_consumed_holdout_ids_across_wrap():
    scenario = rc.ResearchScenario(
        name="holdout_rotation",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        candidate_set="full",
        max_hypotheses=3,
    )
    hypotheses = [hyp for hyp in generate_batch() if hyp.base_timeframe == "5m"][:4]
    state = {
        "scenario_offsets": {scenario.name: 0},
        "consumed_holdout_ids": {
            scenario.name: [hypotheses[0].id, hypotheses[2].id],
        },
    }

    selected, selection = rc._select_from_hypotheses(
        scenario,
        hypotheses,
        state,
    )

    assert [hyp.id for hyp in selected] == [hypotheses[1].id, hypotheses[3].id]
    assert selection["available"] == 4
    assert selection["eligible"] == 2
    assert selection["consumed_holdout"] == 2
    assert selection["selected"] == 2
    assert selection["exhausted"] is False

    state["consumed_holdout_ids"][scenario.name] = [hyp.id for hyp in hypotheses]
    selected, selection = rc._select_from_hypotheses(scenario, hypotheses, state)

    assert selected == []
    assert selection["eligible"] == 0
    assert selection["exhausted"] is True


def test_research_cycle_skips_when_market_data_unchanged(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    coverage = {
        "ok": True,
        "actual": {
            "earliest": "2020-01-01T00:00:00Z",
            "latest": "2026-07-08T11:22:00Z",
            "span_days": 2000,
            "rows": 500_000,
        },
        "failed_checks": [],
        "path": "indicators.parquet",
    }
    history_marker = rc._history_coverage_skip_marker(
        {scenario.name: coverage for scenario in rc.DEFAULT_SCENARIOS}
    )
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "last_market_timestamp": json.dumps(
                    {"futures": "2026-07-08T11:22:00+00:00"},
                    sort_keys=True,
                ),
                "last_market_marker": rc._market_data_skip_marker(_market_statuses()),
                "last_history_coverage_marker": history_marker,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())
    monkeypatch.setattr(rc, "_scenario_indicator_coverage_status", lambda *args, **kwargs: coverage)

    def fail_validation(*args, **kwargs):
        raise AssertionError("validation should be skipped")

    monkeypatch.setattr(rc, "run_validation_scenario", fail_validation)

    report = rc.run_research_cycle(state_path=state_path, output_path=output_path)

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["reason"] == "market_data_unchanged"
    assert json.loads(output_path.read_text(encoding="utf-8"))["skipped"] is True


def test_research_cycle_reruns_when_market_readiness_changes(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    previous_statuses = _market_statuses()
    current_statuses = {
        "futures": _market_status(),
        "spot": {
            "ok": True,
            "last_timestamp": "2026-07-08T11:22:00+00:00",
            "rows": 500,
            "path": "data/candles/spot/BTCUSDT/BTCUSDT_1h.parquet",
        },
    }
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "last_market_timestamp": json.dumps(
                    {"futures": "2026-07-08T11:22:00+00:00"},
                    sort_keys=True,
                ),
                "last_market_marker": rc._market_data_skip_marker(previous_statuses),
            }
        ),
        encoding="utf-8",
    )
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: current_statuses)

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path=None
    ):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
    )

    assert report["ok"] is True
    assert report["skipped"] is False
    assert report["last_market_timestamp"] == json.dumps(
        {
            "futures": "2026-07-08T11:22:00+00:00",
            "spot": "2026-07-08T11:22:00+00:00",
        },
        sort_keys=True,
    )
    assert json.loads(state_path.read_text(encoding="utf-8"))[
        "last_market_marker"
    ] == rc._market_data_skip_marker(current_statuses)


def test_research_cycle_defers_unprotected_epoch_without_advancing_selection(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    scenario = rc.ResearchScenario(
        name="active_income_15m",
        product="active_income",
        base_tf="15m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2022-01-01",
    )
    state_path.write_text(
        json.dumps({"version": 1, "scenario_offsets": {scenario.name: 1}}),
        encoding="utf-8",
    )
    market_timestamp = {"value": "2026-07-08T11:22:00+00:00"}
    monkeypatch.setattr(
        rc,
        "build_market_data_statuses",
        lambda markets: _market_statuses(market_timestamp["value"]),
    )
    validation_offsets = []

    def defer_validation(selected, *, selection, **_kwargs):
        validation_offsets.append(selection["offset"])
        return rc._unprotected_epoch_deferral_report(
            selected,
            selection=selection,
            unsupported_hypotheses=[],
            retired_unsupported_ids=[],
            detail="no unprotected chronological research epoch remains",
        )

    monkeypatch.setattr(rc, "run_validation_scenario", defer_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(scenario,),
        force=True,
    )

    deferred = report["scenarios"][0]
    assert report["ok"] is True
    assert deferred["ok"] is True
    assert deferred["skipped"] is True
    assert deferred["deferred"] is True
    assert deferred["reason"] == "unprotected_epoch_unavailable"
    assert deferred["selection"]["offset"] == 1
    assert report["summary"]["scenario_errors"] == 0
    assert report["summary"]["unprotected_epoch_deferrals"] == 1
    assert report["summary"]["unprotected_epoch_deferred_scenarios"] == [scenario.name]
    assert report["summary"]["next_actions"][0] == (
        "wait for additional market history to create an unprotected research epoch "
        f"for {scenario.name}"
    )
    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["scenario_offsets"][scenario.name] == 1

    unchanged = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(scenario,),
    )
    assert unchanged["ok"] is True
    assert unchanged["skipped"] is True
    assert unchanged["reason"] == "market_data_unchanged"
    assert validation_offsets == [1]

    market_timestamp["value"] = "2026-07-08T11:23:00+00:00"
    retried = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(scenario,),
    )
    assert retried["scenarios"][0]["deferred"] is True
    assert validation_offsets == [1, 1]


def test_research_cycle_keeps_other_evaluation_conflicts_failing(tmp_path, monkeypatch):
    scenario = rc.ResearchScenario(
        name="active_income_15m",
        product="active_income",
        base_tf="15m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2022-01-01",
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())
    monkeypatch.setattr(
        rc,
        "run_validation_scenario",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            rc.EvaluationConflictError("immutable evidence changed")
        ),
    )

    report = rc.run_research_cycle(
        state_path=tmp_path / "state.json",
        output_path=tmp_path / "research_cycle.json",
        scenarios=(scenario,),
        force=True,
    )

    assert report["ok"] is False
    assert report["summary"]["scenario_errors"] == 1
    assert report["exports"] == []
    assert report["scenarios"][0]["error"] == (
        "EvaluationConflictError: immutable evidence changed"
    )


def test_research_cycle_recovers_corrupt_state_and_runs(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    state_path.write_text("{not-json", encoding="utf-8")
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path=None
    ):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
    )

    assert report["ok"] is True
    assert report["skipped"] is False
    assert report["state_recovered"] is True
    assert "JSONDecodeError" in report["state_error"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["state_recovered"] is True
    recovered_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert recovered_state["version"] == 1
    assert "_state_error" not in recovered_state
    assert "active_income_15m" in recovered_state["scenario_offsets"]


def test_research_cycle_runs_validation_and_handles_no_exportable_strategies(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    incubation_output_path = tmp_path / "incubation_candidates.json"
    log_path = tmp_path / "experiment_log.jsonl"
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path
    ):
        hypothesis_count = len(hypotheses or [])
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": hypothesis_count,
            "keepers": 0,
            "incubation_candidates": [
                {
                    "id": "REJECTED_CANDIDATE",
                    "score": 1.5,
                    "verdict": "reject",
                    "next_step": "discard_or_mutate_entry_logic",
                }
            ],
            "selection": selection,
            "verdicts": {"reject": hypothesis_count},
            "top_reasons": {"no_train_edge": hypothesis_count},
        }

    def fake_export(product, *, pnl_unit, market, out, top_k, ids, min_dsr, log_path):
        return {
            "ok": True,
            "product": product,
            "pnl_unit": pnl_unit,
            "market": market,
            "exported": False,
            "reason": "no_exportable_strategies",
            "artifact": str(out),
            "min_dsr": min_dsr,
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)
    monkeypatch.setattr(rc, "export_product", fake_export)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        incubation_output_path=incubation_output_path,
        log_path=log_path,
        scenarios=scenarios,
        force=True,
    )

    assert report["ok"] is True
    assert report["scenarios"][0]["keepers"] == 0
    assert report["summary"]["scenarios"] == 1
    assert report["summary"]["opportunity_types"] == {"research": 1}
    assert report["summary"]["opportunity_types_by_product"] == {"active_income": {"research": 1}}
    assert report["summary"]["hypotheses"] == report["scenarios"][0]["hypotheses"]
    assert report["summary"]["keepers"] == 0
    assert report["summary"]["incubation_candidates"] == 1
    assert report["summary"]["top_reasons"] == {
        "no_train_edge": report["scenarios"][0]["hypotheses"]
    }
    assert report["summary"]["next_actions"] == [
        "continue rotating curated candidates; no positive train edge found yet"
    ]
    assert {item["reason"] for item in report["exports"]} == {"no_current_cycle_keepers"}
    assert {item["market"] for item in report["exports"]} == {"futures", "spot"}
    assert {item["product"]: item["min_dsr"] for item in report["exports"]} == {
        "active_income": 0.60,
        "btc_accumulation": 0.60,
    }
    assert report["incubation_review"]["path"] == str(incubation_output_path)
    assert report["incubation_review"]["research_only"] is True
    assert report["incubation_review"]["executable"] is False
    assert report["incubation_review"]["paper_trade_allowed"] is False
    incubation_review = json.loads(incubation_output_path.read_text(encoding="utf-8"))
    assert incubation_review["summary"] == {"candidates": 1, "by_product": {"active_income": 1}}
    assert incubation_review["products"]["active_income"][0]["id"] == "REJECTED_CANDIDATE"
    assert json.loads(state_path.read_text(encoding="utf-8"))[
        "last_market_timestamp"
    ] == json.dumps(
        {"futures": "2026-07-08T11:22:00+00:00"},
        sort_keys=True,
    )
    assert json.loads(state_path.read_text(encoding="utf-8"))[
        "last_market_marker"
    ] == rc._market_data_skip_marker(_market_statuses())


def test_incubation_candidates_rank_non_keeper_research_attention_only():
    candidates = rc._incubation_candidates_from_results(
        [
            {
                "hypothesis_id": "NO_EDGE",
                "family": "momentum",
                "direction": "long",
                "verdict": "reject",
                "reasons": ["no_train_edge"],
                "train": {"trades": 40, "total_return": -0.02, "win_rate": 0.4, "sharpe": -0.5},
            },
            {
                "hypothesis_id": "FAILED_VAL",
                "family": "breakout",
                "direction": "short",
                "verdict": "reject",
                "reasons": ["failed_validation"],
                "train": {"trades": 60, "total_return": 0.08, "win_rate": 0.58, "sharpe": 1.1},
                "validation": {
                    "trades": 12,
                    "total_return": -0.01,
                    "win_rate": 0.42,
                    "sharpe": -0.2,
                },
            },
            {
                "hypothesis_id": "KEEPER",
                "family": "trend",
                "direction": "long",
                "verdict": "keep",
                "reasons": [],
                "train": {"trades": 70, "total_return": 0.1},
            },
        ],
        limit=2,
    )

    assert [item["id"] for item in candidates] == ["FAILED_VAL", "NO_EDGE"]
    assert candidates[0]["stage_reached"] == "validation"
    assert candidates[0]["next_step"] == "mutate_thresholds_before_retest"
    assert candidates[1]["next_step"] == "discard_or_mutate_entry_logic"


def test_incubation_candidates_sanitize_non_finite_metrics():
    candidates = rc._incubation_candidates_from_results(
        [
            {
                "hypothesis_id": "ZERO_TRADE",
                "family": "trend",
                "direction": "short",
                "verdict": "inconclusive",
                "reasons": ["insufficient_train_trades"],
                "train": {
                    "trades": 0,
                    "total_return": 0.0,
                    "win_rate": math.nan,
                    "sharpe": math.nan,
                },
            }
        ]
    )

    assert candidates[0]["train"]["win_rate"] is None
    assert candidates[0]["train"]["sharpe"] is None


def test_incubation_candidates_include_mutation_lineage_for_research_feedback():
    candidates = rc._incubation_candidates_from_results(
        [
            {
                "hypothesis_id": "MUT_SOURCE_NO_TRAIN_EDGE_001",
                "family": "trend",
                "direction": "long",
                "verdict": "reject",
                "reasons": ["failed_validation"],
                "train": {"trades": 40, "total_return": 0.05},
                "validation": {"trades": 12, "total_return": -0.02},
            }
        ],
        hypothesis_metadata={
            "MUT_SOURCE_NO_TRAIN_EDGE_001": {
                "source_candidate_id": "SOURCE",
                "source_scenario": "active_income_5m_guarded",
                "reason": "no_train_edge",
                "validation_scope": {"candidate_set": "full", "pnl_unit": "usdt"},
            }
        },
    )

    assert candidates[0]["mutation_lineage"] == {
        "source_candidate_id": "SOURCE",
        "source_scenario": "active_income_5m_guarded",
        "validation_scope": {"candidate_set": "full", "pnl_unit": "usdt"},
        "mutation_reason": "no_train_edge",
    }


def test_incubation_review_is_non_executable_research_queue():
    review = rc.build_incubation_review(
        [
            {
                "ok": True,
                "name": "active_income_5m_guarded",
                "product": "active_income",
                "market": "futures",
                "pnl_unit": "usdt",
                "opportunity_type": "day_trading",
                "base_tf": "5m",
                "incubation_candidates": [
                    {"id": "LOW", "score": 1.0, "verdict": "reject"},
                    {"id": "HIGH", "score": 3.0, "verdict": "reject"},
                ],
            },
            {
                "ok": True,
                "skipped": True,
                "name": "skipped",
                "product": "active_income",
                "incubation_candidates": [{"id": "SKIP", "score": 9.0}],
            },
        ],
        generated_at="2026-07-08T12:00:00+00:00",
        limit_per_product=1,
    )

    assert review["research_only"] is True
    assert review["executable"] is False
    assert review["paper_trade_allowed"] is False
    assert review["live_allowed"] is False
    assert review["promotion_eligible"] is False
    assert review["summary"] == {"candidates": 1, "by_product": {"active_income": 1}}
    assert review["products"]["active_income"][0]["id"] == "HIGH"
    assert review["products"]["active_income"][0]["scenario"] == "active_income_5m_guarded"


def test_research_cycle_exports_only_current_cycle_keeper_ids(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path
    ):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": 2,
            "keepers": 1,
            "keeper_ids": ["KEEP_THIS"],
            "selection": selection,
            "verdicts": {"keep": 1, "reject": 1},
        }

    export_calls = []

    def fake_export(
        product,
        *,
        pnl_unit,
        market,
        out,
        top_k,
        ids,
        min_dsr,
        log_path,
        state_file,
    ):
        export_calls.append({"product": product, "market": market, "ids": ids, "min_dsr": min_dsr})
        return {
            "ok": True,
            "product": product,
            "pnl_unit": pnl_unit,
            "market": market,
            "exported": True,
            "artifact": str(out),
            "strategies": len(ids),
            "ids": ids,
            "min_dsr": min_dsr,
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)
    monkeypatch.setattr(rc, "export_product", fake_export)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
        force=True,
    )

    assert report["ok"] is True
    assert export_calls == [
        {"product": "active_income", "market": "futures", "ids": ["KEEP_THIS"], "min_dsr": 0.60}
    ]
    assert next(item for item in report["exports"] if item["product"] == "btc_accumulation")[
        "reason"
    ] == ("no_current_cycle_keepers")


def test_export_product_does_not_replace_active_artifact_while_positions_are_open(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "active_strategies_flow.json"
    artifact.write_text('{"version": 1, "strategies": [{"id": "old"}]}', encoding="utf-8")
    state_file = tmp_path / "active_income_state.json"
    state_file.write_text(
        json.dumps({"open_positions": {"old": {"direction": "long"}}}),
        encoding="utf-8",
    )

    def fail_export(**kwargs):
        raise AssertionError("export should not run while open positions exist")

    monkeypatch.setattr(rc, "export_strategies", fail_export)

    report = rc.export_product(
        "active_income",
        pnl_unit="usdt",
        market="futures",
        out=artifact,
        top_k=3,
        ids=["KEEP_THIS"],
        min_dsr=0.60,
        log_path=tmp_path / "experiment_log.jsonl",
        state_file=state_file,
    )

    assert report == {
        "ok": True,
        "product": "active_income",
        "pnl_unit": "usdt",
        "market": "futures",
        "exported": False,
        "reason": "open_positions_block_export",
        "detail": "active strategy artifact is left unchanged while positions are open",
        "artifact": str(artifact),
        "ids": ["KEEP_THIS"],
        "open_positions": ["old"],
        "min_dsr": 0.60,
    }
    assert json.loads(artifact.read_text(encoding="utf-8"))["strategies"][0]["id"] == "old"


def test_live_candidate_is_policy_checked_and_staged_without_touching_active(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text('{"sentinel": "approved-live-artifact"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    staged = tmp_path / "candidates" / "active_income.json"
    product = ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=active,
        state_file=state,
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )

    def fake_export(*, output_path, **kwargs):
        output_path.write_text(json.dumps(_live_candidate_payload()), encoding="utf-8")
        return output_path

    monkeypatch.setattr(rc, "export_strategies", fake_export)

    report = rc.stage_live_product_candidate(
        product,
        pnl_unit="usdt",
        market="futures",
        out=staged,
        top_k=3,
        ids=["KEEP_THIS"],
        min_dsr=0.60,
        log_path=tmp_path / "experiments.jsonl",
    )

    assert json.loads(active.read_text(encoding="utf-8")) == {"sentinel": "approved-live-artifact"}
    candidate = json.loads(staged.read_text(encoding="utf-8"))
    assert candidate["product"] == product_identity(product)
    assert candidate["candidate_staging"]["activation_required"] is True
    assert candidate["candidate_staging"]["approval_granted"] is False
    assert report["destination"] == "staging"
    assert report["staged"] is True
    assert report["active_artifact"] == str(active)
    assert report["artifact_digest"] == artifact_digest(candidate)


def test_live_candidate_restage_preserves_exact_candidate_paper_state(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text('{"sentinel": "approved-live-artifact"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    staged = tmp_path / "candidates" / "active_income.json"
    product = ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=active,
        state_file=state,
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )

    def fake_export(*, output_path, **kwargs):
        output_path.write_text(json.dumps(_live_candidate_payload()), encoding="utf-8")
        return output_path

    monkeypatch.setattr(rc, "export_strategies", fake_export)
    first = rc.stage_live_product_candidate(
        product,
        pnl_unit="usdt",
        market="futures",
        out=staged,
        top_k=3,
    )
    original = staged.read_bytes()
    second = rc.stage_live_product_candidate(
        product,
        pnl_unit="usdt",
        market="futures",
        out=staged,
        top_k=3,
    )

    assert first["staged"] is True
    assert second["reason"] == "candidate_already_staged"
    assert second["artifact_digest"] == first["artifact_digest"]
    assert staged.read_bytes() == original


def test_live_candidate_replacement_waits_for_prior_paper_position(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text('{"sentinel": "approved-live-artifact"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    staged = tmp_path / "candidates" / "active_income.json"
    product = ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=active,
        state_file=state,
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )
    existing = _live_candidate_payload()
    existing["product"] = product_identity(product)
    existing["candidate_staging"] = {
        "staged_at": "2026-01-01T00:00:00+00:00",
        "activation_required": True,
        "approval_granted": False,
        "active_artifact": str(active),
    }
    staged.parent.mkdir(parents=True)
    staged.write_text(json.dumps(existing), encoding="utf-8")
    existing_digest = artifact_digest(existing)
    old_state = candidate_paper_paths(
        product.name,
        existing_digest,
        candidate_dir=staged.parent,
    )["state"]
    old_state.write_text(
        json.dumps({"open_positions": {"KEEP_THIS": {"entry_price": 100.0}}}),
        encoding="utf-8",
    )

    replacement = _live_candidate_payload()
    replacement["strategies"][0]["take_profit"] = 0.03

    def fake_export(*, output_path, **kwargs):
        output_path.write_text(json.dumps(replacement), encoding="utf-8")
        return output_path

    monkeypatch.setattr(rc, "export_strategies", fake_export)
    report = rc.stage_live_product_candidate(
        product,
        pnl_unit="usdt",
        market="futures",
        out=staged,
        top_k=3,
    )

    assert report["reason"] == "prior_candidate_open_positions"
    assert report["open_positions"] == ["KEEP_THIS"]
    assert artifact_digest(json.loads(staged.read_text(encoding="utf-8"))) == existing_digest


def test_live_candidate_refuses_staging_path_that_aliases_active_artifact(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text('{"sentinel": "approved"}', encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    product = ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode="live",
        symbol="BTCUSDT",
        strategies_path=active,
        state_file=state,
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )
    monkeypatch.setattr(
        rc,
        "export_strategies",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not export")),
    )

    with pytest.raises(ValueError, match="distinct from the active artifact"):
        rc.stage_live_product_candidate(
            product,
            pnl_unit="usdt",
            market="futures",
            out=active,
            top_k=3,
            ids=["KEEP_THIS"],
        )

    assert json.loads(active.read_text(encoding="utf-8")) == {"sentinel": "approved"}


def test_research_cycle_routes_live_product_to_deterministic_staging_path(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    active.write_text('{"sentinel": "unchanged"}', encoding="utf-8")
    active_state = tmp_path / "active_state.json"
    active_state.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "name": "active_income",
                        "enabled": True,
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "execution_mode": "live",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(active),
                        "state_file": str(active_state),
                        "trade_log": str(tmp_path / "active_trades.csv"),
                        "starting_equity": 1000.0,
                    },
                    {
                        "name": "btc_accumulation",
                        "enabled": True,
                        "objective": "btc_accumulation",
                        "base_asset": "BTC",
                        "market": "spot",
                        "execution_mode": "paper",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(tmp_path / "btc.json"),
                        "state_file": str(tmp_path / "btc_state.json"),
                        "trade_log": str(tmp_path / "btc_trades.csv"),
                        "starting_equity": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scenario = rc.ResearchScenario(
        name="active_income_15m",
        product="active_income",
        base_tf="15m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2022-01-01",
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())
    monkeypatch.setattr(
        rc,
        "run_validation_scenario",
        lambda selected, **kwargs: {
            "ok": True,
            "name": selected.name,
            "product": selected.product,
            "market": selected.market,
            "opportunity_type": selected.opportunity_type,
            "hypotheses": 1,
            "keepers": 1,
            "keeper_ids": ["KEEP_THIS"],
            "selection": kwargs["selection"],
            "verdicts": {"keep": 1},
        },
    )
    staging_calls = []

    def fake_stage(product, *, out, **kwargs):
        staging_calls.append((product.name, out))
        return {
            "ok": True,
            "product": product.name,
            "market": product.market,
            "exported": True,
            "staged": True,
            "destination": "staging",
            "activation_required": True,
            "artifact_digest": "sha256:" + "a" * 64,
            "artifact": str(out),
            "active_artifact": str(product.strategies_path),
        }

    monkeypatch.setattr(rc, "stage_live_product_candidate", fake_stage)
    candidate_dir = tmp_path / "candidates"
    report = rc.run_research_cycle(
        config_path=config_path,
        candidate_dir=candidate_dir,
        state_path=tmp_path / "research_state.json",
        output_path=tmp_path / "report.json",
        scenarios=(scenario,),
        force=True,
    )

    assert report["ok"] is True
    assert staging_calls == [("active_income", candidate_dir / "active_income.json")]
    assert json.loads(active.read_text(encoding="utf-8")) == {"sentinel": "unchanged"}
    assert report["summary"]["staged"] == 1
    assert report["summary"]["active_exports"] == 0
    assert report["summary"]["staged_candidates"][0]["artifact_digest"] == ("sha256:" + "a" * 64)
    assert "activate it explicitly" in report["summary"]["next_actions"][0]


def test_research_cycle_blocks_keeper_export_when_product_has_open_positions(tmp_path, monkeypatch):
    state_path = tmp_path / "research_state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    product_state = tmp_path / "active_income_state.json"
    product_state.write_text(
        json.dumps({"open_positions": {"running_strategy": {"direction": "short"}}}),
        encoding="utf-8",
    )
    artifact = tmp_path / "active_strategies_flow.json"
    config_path = tmp_path / "autopilot.json"
    config_path.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "name": "active_income",
                        "enabled": True,
                        "objective": "active_income",
                        "base_asset": "USDT",
                        "market": "futures",
                        "execution_mode": "paper",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(artifact),
                        "state_file": str(product_state),
                        "trade_log": str(tmp_path / "active_trades.csv"),
                        "starting_equity": 1000.0,
                    },
                    {
                        "name": "btc_accumulation",
                        "enabled": True,
                        "objective": "btc_accumulation",
                        "base_asset": "BTC",
                        "market": "spot",
                        "execution_mode": "paper",
                        "symbol": "BTCUSDT",
                        "strategies_path": str(tmp_path / "btc.json"),
                        "state_file": str(tmp_path / "btc_state.json"),
                        "trade_log": str(tmp_path / "btc_trades.csv"),
                        "starting_equity": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path
    ):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": 1,
            "keepers": 1,
            "keeper_ids": ["KEEP_THIS"],
            "selection": selection,
            "verdicts": {"keep": 1},
        }

    def fail_export(**kwargs):
        raise AssertionError("export should not run while open positions exist")

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)
    monkeypatch.setattr(rc, "export_strategies", fail_export)

    report = rc.run_research_cycle(
        config_path=config_path,
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
        force=True,
    )

    active_export = next(item for item in report["exports"] if item["product"] == "active_income")
    assert report["ok"] is True
    assert active_export["exported"] is False
    assert active_export["reason"] == "open_positions_block_export"
    assert active_export["open_positions"] == ["running_strategy"]
    assert report["summary"]["keepers"] == 1
    assert report["summary"]["exported"] == 0
    assert report["summary"]["export_reasons"] == {
        "open_positions_block_export": 1,
        "no_current_cycle_keepers": 1,
    }
    assert report["summary"]["next_actions"] == [
        "wait for open positions to close before replacing the active paper artifact"
    ]


def _mutation_batch_payload(
    hypothesis, *, executable=False, generated_at="2026-01-01T00:00:00+00:00"
):
    return {
        "ok": True,
        "schema": "research_exploration.hypothesis_schema/v1",
        "generated_at": generated_at,
        "research_only": True,
        "executable": executable,
        "count": 1,
        "summary": {
            "hypotheses": 1,
            "skipped": 0,
            "by_product": {"active_income": 1},
            "executable": executable,
        },
        "mutation_metadata": [
            {
                "id": hypothesis.id,
                "source_candidate_id": "SOURCE",
                "product": "active_income",
                "market": "futures",
                "opportunity_type": "day_trading",
                "reason": "no_train_edge",
                "validation_scope": {
                    "candidate_set": "full",
                    "pnl_unit": "usdt",
                    "with_guards": True,
                },
            }
        ],
        "hypotheses": [hypothesis.to_dict()],
    }


def test_research_cycle_validates_research_only_mutation_batch(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    mutation_batch_path = tmp_path / "mutation_hypotheses.json"
    hypothesis = next(hyp for hyp in generate_batch(with_guards=True) if hyp.base_timeframe == "5m")
    mutation_batch_path.write_text(
        json.dumps(_mutation_batch_payload(hypothesis)), encoding="utf-8"
    )
    monkeypatch.setattr(
        rc, "build_market_data_statuses", lambda markets: {"futures": _market_status()}
    )
    validated = []

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path
    ):
        validated.append((scenario, hypotheses, selection, hypothesis_metadata))
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "candidate_set": scenario.candidate_set,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
            "top_reasons": {"no_train_edge": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=(),
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
        force=True,
    )

    assert report["ok"] is True
    assert report["mutation_batch"]["status"] == "loaded"
    assert report["mutation_batch"]["scenarios"] == 1
    assert validated[0][0].candidate_set == "mutation"
    assert validated[0][0].product == "active_income"
    assert [hyp.id for hyp in validated[0][1]] == [hypothesis.id]
    assert validated[0][2]["candidate_set"] == "mutation"
    assert validated[0][3][hypothesis.id]["source_candidate_id"] == "SOURCE"
    assert validated[0][3][hypothesis.id]["reason"] == "no_train_edge"
    assert report["summary"]["selected_hypotheses"] == 1
    assert report["summary"]["mutation_effectiveness"] == {
        "status": "loaded",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "batch_hypotheses": 1,
        "batch_scenarios": 1,
        "evaluated_scenarios": 1,
        "evaluated_hypotheses": 1,
        "keepers": 0,
        "incubation_candidates": 0,
        "skipped_scenarios": 0,
        "scenario_errors": 0,
        "by_product": {"active_income": 1},
        "verdicts": {"reject": 1},
        "top_reasons": {"no_train_edge": 1},
        "outcome": "no_keeper",
    }
    assert (
        report["summary"]["next_actions"][0]
        == "mutation batch found no keepers; top mutation reason no_train_edge"
    )
    assert {item["reason"] for item in report["exports"]} == {"no_current_cycle_keepers"}


def test_research_cycle_ignores_executable_mutation_batch(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    mutation_batch_path = tmp_path / "mutation_hypotheses.json"
    hypothesis = next(hyp for hyp in generate_batch(with_guards=True) if hyp.base_timeframe == "5m")
    mutation_batch_path.write_text(
        json.dumps(_mutation_batch_payload(hypothesis, executable=True)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rc, "build_market_data_statuses", lambda markets: {"futures": _market_status()}
    )

    def fail_validation(*args, **kwargs):
        raise AssertionError("unsafe mutation batch should not create validation scenarios")

    monkeypatch.setattr(rc, "run_validation_scenario", fail_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(),
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
        force=True,
    )

    assert report["ok"] is True
    assert report["mutation_batch"]["status"] == "ignored"
    assert report["summary"]["scenarios"] == 0
    assert report["summary"]["hypotheses"] == 0
    assert report["summary"]["mutation_effectiveness"]["status"] == "ignored"
    assert report["summary"]["mutation_effectiveness"]["outcome"] == "ignored"
    assert report["summary"]["mutation_effectiveness"]["evaluated_hypotheses"] == 0


def test_research_cycle_ignores_corrupt_mutation_batch_and_runs_curated_scenarios(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    mutation_batch_path = tmp_path / "mutation_hypotheses.json"
    mutation_batch_path.write_text("{not-json", encoding="utf-8")
    scenarios = (
        rc.ResearchScenario(
            name="active_income_15m",
            product="active_income",
            base_tf="15m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2022-01-01",
        ),
    )
    monkeypatch.setattr(
        rc, "build_market_data_statuses", lambda markets: {"futures": _market_status()}
    )
    validated = []

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path=None
    ):
        validated.append(scenario.name)
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)

    report = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=scenarios,
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
    )

    assert report["ok"] is True
    assert validated == ["active_income_15m"]
    assert report["mutation_batch"]["status"] == "read_error"
    assert "JSONDecodeError" in report["mutation_batch"]["error"]
    assert report["summary"]["scenarios"] == 1
    assert report["summary"]["mutation_effectiveness"]["status"] == "read_error"
    assert report["summary"]["mutation_effectiveness"]["outcome"] == "read_error"
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["mutation_batch"]["status"]
        == "read_error"
    )


def test_research_cycle_skip_marker_tracks_mutation_batch(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    mutation_batch_path = tmp_path / "mutation_hypotheses.json"
    hypothesis = next(hyp for hyp in generate_batch(with_guards=True) if hyp.base_timeframe == "5m")
    mutation_batch_path.write_text(
        json.dumps(_mutation_batch_payload(hypothesis, generated_at="2026-01-01T00:00:00+00:00")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rc, "build_market_data_statuses", lambda markets: {"futures": _market_status()}
    )

    def fake_validation(
        scenario, *, hypotheses=None, selection=None, hypothesis_metadata=None, log_path
    ):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "market": scenario.market,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)
    first = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(),
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
    )
    second = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(),
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
    )
    mutation_batch_path.write_text(
        json.dumps(_mutation_batch_payload(hypothesis, generated_at="2026-01-02T00:00:00+00:00")),
        encoding="utf-8",
    )
    third = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        scenarios=(),
        include_mutations=True,
        mutation_batch_path=mutation_batch_path,
    )

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert third["skipped"] is False


def test_research_cycle_advances_scenario_offsets_after_success(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    scenarios = (
        rc.ResearchScenario(
            name="active_income_5m_guarded",
            product="active_income",
            base_tf="5m",
            pnl_unit="usdt",
            market="futures",
            position=False,
            start="2023-01-01",
            with_guards=True,
            candidate_set="full",
            max_hypotheses=3,
        ),
    )
    monkeypatch.setattr(
        rc,
        "build_market_data_statuses",
        lambda markets: _market_statuses("2026-07-08T11:23:00+00:00"),
    )

    def fake_validation(scenario, *, hypotheses=None, selection=None, log_path):
        return {
            "ok": True,
            "name": scenario.name,
            "product": scenario.product,
            "opportunity_type": scenario.opportunity_type,
            "hypotheses": len(hypotheses or []),
            "keepers": 0,
            "selection": selection,
            "verdicts": {"reject": len(hypotheses or [])},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)
    monkeypatch.setattr(
        rc,
        "export_product",
        lambda product, *, pnl_unit, market, out, top_k, ids, log_path: {
            "ok": True,
            "product": product,
            "market": market,
            "exported": False,
            "reason": "no_exportable_strategies",
        },
    )

    first = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
        force=True,
    )
    second = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=scenarios,
        force=True,
    )

    assert first["scenarios"][0]["selection"]["offset"] == 0
    assert first["scenarios"][0]["selection"]["selected"] == 3
    assert second["scenarios"][0]["selection"]["offset"] == 3
    assert (
        json.loads(state_path.read_text(encoding="utf-8"))["scenario_offsets"][
            "active_income_5m_guarded"
        ]
        == second["scenarios"][0]["selection"]["next_offset"]
    )


def test_research_cycle_never_reuses_consumed_holdout_candidate_and_skips_exhausted(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "research_cycle.json"
    log_path = tmp_path / "experiment_log.jsonl"
    scenario = rc.ResearchScenario(
        name="holdout_once",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        candidate_set="full",
        max_hypotheses=2,
    )
    hypotheses = [hyp for hyp in generate_batch() if hyp.base_timeframe == "5m"][:2]
    validation_calls = []
    monkeypatch.setattr(
        rc,
        "build_market_data_statuses",
        lambda markets: _market_statuses("2026-07-08T11:24:00+00:00"),
    )
    monkeypatch.setattr(rc, "_hypotheses_for", lambda selected: hypotheses)

    def fake_validation(selected, *, hypotheses=None, selection=None, log_path):
        ids = [hyp.id for hyp in hypotheses or []]
        validation_calls.append(ids)
        return {
            "ok": True,
            "name": selected.name,
            "product": selected.product,
            "market": selected.market,
            "opportunity_type": selected.opportunity_type,
            "hypotheses": len(ids),
            "keepers": 0,
            "keeper_ids": [],
            "holdout_exposed_ids": ids,
            "selection": selection,
            "verdicts": {"reject": len(ids)},
        }

    monkeypatch.setattr(rc, "run_validation_scenario", fake_validation)

    first = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=(scenario,),
        force=True,
    )
    second = rc.run_research_cycle(
        state_path=state_path,
        output_path=output_path,
        log_path=log_path,
        scenarios=(scenario,),
        force=True,
    )

    assert first["ok"] is True
    assert validation_calls == [[hyp.id for hyp in hypotheses]]
    assert second["ok"] is True
    assert second["scenarios"][0]["skipped"] is True
    assert second["scenarios"][0]["reason"] == "holdout_registry_exhausted"
    assert second["scenarios"][0]["selection"]["exhausted"] is True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["consumed_holdout_ids"][scenario.name] == sorted(hyp.id for hyp in hypotheses)


def test_research_cycle_persists_holdout_consumption_even_if_export_fails(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "state.json"
    scenario = rc.ResearchScenario(
        name="holdout_before_export_failure",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        candidate_set="full",
        max_hypotheses=1,
    )
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    monkeypatch.setattr(rc, "build_market_data_statuses", lambda markets: _market_statuses())
    monkeypatch.setattr(rc, "_hypotheses_for", lambda selected: [hypothesis])
    monkeypatch.setattr(
        rc,
        "run_validation_scenario",
        lambda selected, *, hypotheses=None, selection=None, log_path: {
            "ok": True,
            "name": selected.name,
            "product": selected.product,
            "market": selected.market,
            "opportunity_type": selected.opportunity_type,
            "hypotheses": 1,
            "keepers": 1,
            "keeper_ids": [hypothesis.id],
            "holdout_exposed_ids": [hypothesis.id],
            "selection": selection,
            "verdicts": {"keep": 1},
        },
    )
    monkeypatch.setattr(
        rc,
        "export_product",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("export failed")),
    )

    report = rc.run_research_cycle(
        state_path=state_path,
        scenarios=(scenario,),
        force=True,
    )

    assert report["ok"] is False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["consumed_holdout_ids"] == {scenario.name: [hypothesis.id]}
    assert "last_market_marker" not in state


def test_validation_scenario_deflates_by_available_rotation_universe(monkeypatch):
    scenario = rc.ResearchScenario(
        name="active_income_5m_guarded",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        with_guards=True,
        candidate_set="full",
        max_hypotheses=2,
    )
    hypotheses, selection = rc._select_hypotheses(scenario, {"version": 1})
    captured = {}
    monkeypatch.setattr(rc, "_missing_columns_for_hypothesis", lambda hypothesis, indicator_dir: {})
    monkeypatch.setattr(
        rc,
        "with_trial_sharpe_dispersion",
        lambda _frame, _hypotheses, cfg, _eval_cfg: cfg,
    )

    def fake_build_frame(hyps, *, base_tf, start, end, indicator_dir):
        import pandas as pd

        assert [hyp.id for hyp in hyps] == [hyp.id for hyp in hypotheses]
        return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC")})

    def fake_validate_batch(frame, hyps, cfg, *, eval_cfg, log_path):
        captured["n_trials"] = cfg.n_trials
        return [
            {
                "hypothesis_id": hyp.id,
                "family": hyp.family,
                "direction": hyp.direction,
                "verdict": "reject",
                "reasons": ["no_train_edge"],
                "train": {"trades": 30},
            }
            for hyp in hyps
        ]

    monkeypatch.setattr(rc, "build_aligned_frame", fake_build_frame)
    monkeypatch.setattr(rc, "validate_batch", fake_validate_batch)

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=hypotheses,
        selection=selection,
        log_path=None,
    )

    assert selection["available"] > selection["selected"]
    assert captured["n_trials"] == selection["available"]
    assert report["trial_count"] == selection["available"]


def test_validation_scenario_records_only_hypotheses_that_touch_holdout(monkeypatch):
    scenario = rc.ResearchScenario(
        name="active_income_5m_guarded",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        candidate_set="full",
        max_hypotheses=2,
    )
    hypotheses, selection = rc._select_hypotheses(scenario, {"version": 1})
    monkeypatch.setattr(
        rc,
        "_missing_columns_for_hypothesis",
        lambda hypothesis, indicator_dir: {},
    )
    monkeypatch.setattr(
        rc,
        "with_trial_sharpe_dispersion",
        lambda _frame, _hypotheses, cfg, _eval_cfg: cfg,
    )

    def fake_build_frame(hyps, *, base_tf, start, end, indicator_dir):
        import pandas as pd

        return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC")})

    monkeypatch.setattr(rc, "build_aligned_frame", fake_build_frame)
    monkeypatch.setattr(
        rc,
        "validate_batch",
        lambda frame, hyps, cfg, *, eval_cfg, log_path: [
            {
                "hypothesis_id": hyps[0].id,
                "family": hyps[0].family,
                "direction": hyps[0].direction,
                "verdict": "reject",
                "reasons": ["failed_holdout"],
                "holdout": {"trades": 5, "total_return": -0.01},
            },
            {
                "hypothesis_id": hyps[1].id,
                "family": hyps[1].family,
                "direction": hyps[1].direction,
                "verdict": "reject",
                "reasons": ["failed_validation"],
                "holdout": None,
            },
        ],
    )

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=hypotheses,
        selection=selection,
        log_path=None,
    )

    assert report["holdout_exposed_ids"] == [hypotheses[0].id]


def test_validation_scenario_skips_only_unsupported_hypotheses(monkeypatch):
    scenario = rc.ResearchScenario(
        name="active_income_5m_guarded",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2023-01-01",
        with_guards=True,
        candidate_set="full",
        max_hypotheses=2,
    )
    hypotheses, selection = rc._select_hypotheses(scenario, {"version": 1})
    unsupported_id = hypotheses[0].id
    monkeypatch.setattr(
        rc,
        "_missing_columns_for_hypothesis",
        lambda hypothesis, indicator_dir: {"5m": ["volume_z_20"]}
        if hypothesis.id == unsupported_id
        else {},
    )
    monkeypatch.setattr(
        rc,
        "with_trial_sharpe_dispersion",
        lambda _frame, _hypotheses, cfg, _eval_cfg: cfg,
    )

    def fake_build_frame(hyps, *, base_tf, start, end, indicator_dir):
        import pandas as pd

        assert [hyp.id for hyp in hyps] == [hypotheses[1].id]
        return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC")})

    monkeypatch.setattr(rc, "build_aligned_frame", fake_build_frame)
    monkeypatch.setattr(
        rc,
        "validate_batch",
        lambda frame, hyps, cfg, *, eval_cfg, log_path: [
            {"verdict": "reject", "reasons": ["no_train_edge"]}
        ],
    )

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=hypotheses,
        selection=selection,
        log_path=None,
    )

    assert report["ok"] is True
    assert report["hypotheses"] == 1
    assert report["unsupported_hypotheses"] == [
        {"id": unsupported_id, "missing_columns": {"5m": ["volume_z_20"]}}
    ]
    assert report["verdicts"] == {"reject": 1}


def test_research_cycle_fails_closed_when_market_data_is_not_ready(tmp_path, monkeypatch):
    output_path = tmp_path / "research_cycle.json"
    monkeypatch.setattr(
        rc,
        "build_market_data_statuses",
        lambda markets: {
            market: {"ok": False, "reason": "missing_seed_dataset", "exists": False}
            for market in markets
        },
    )

    report = rc.run_research_cycle(output_path=output_path)

    assert report["ok"] is False
    assert report["error"] == "market_data_not_ready"
    assert json.loads(output_path.read_text(encoding="utf-8"))["ok"] is False


def test_research_epoch_selection_uses_largest_contiguous_unprotected_run():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=125,
                freq="5min",
            ),
            "value": range(125),
        }
    )
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": str(frame["timestamp"].iloc[80]),
            "end": str(frame["timestamp"].iloc[99]),
        },
    )

    selected, detail = rc._select_unprotected_epoch(frame, protected)

    assert selected["value"].tolist() == list(range(80))
    assert detail == {
        "policy": "largest_contiguous_unprotected_epoch",
        "input_rows": 125,
        "selected_rows": 80,
        "excluded_rows": 20,
        "start": str(frame["timestamp"].iloc[0]),
        "end": str(frame["timestamp"].iloc[79]),
        "protected_intervals_considered": 1,
    }


def test_research_epoch_selection_prefers_newest_run_when_sizes_tie():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=21,
                freq="5min",
            ),
            "value": range(21),
        }
    )
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": str(frame["timestamp"].iloc[10]),
            "end": str(frame["timestamp"].iloc[10]),
        },
    )

    selected, _ = rc._select_unprotected_epoch(frame, protected)

    assert selected["value"].tolist() == list(range(11, 21))


def test_research_epoch_selection_prefers_capacity_qualified_run_over_denser_run():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T12:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-07T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                    "2026-01-10T00:00:00Z",
                ],
                utc=True,
            ),
            "value": range(10),
        }
    )
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": "2026-01-05T00:00:00Z",
            "end": "2026-01-05T00:00:00Z",
        },
    )

    selected, detail = rc._select_unprotected_epoch(
        frame,
        protected,
        capacity_requirements={"minimum_rows": 4, "minimum_span_days": 4.0},
    )

    assert selected["value"].tolist() == [6, 7, 8, 9]
    assert detail["capacity_selection"] == {
        "requirements": {"minimum_rows": 4, "minimum_span_days": 4.0},
        "available_runs": 2,
        "qualified_runs": 1,
    }


def test_research_epoch_selection_raises_specific_error_when_every_row_is_protected():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=12,
                freq="5min",
            )
        }
    )
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": str(frame["timestamp"].iloc[0]),
            "end": str(frame["timestamp"].iloc[-1]),
        },
    )

    with pytest.raises(
        rc.UnprotectedResearchEpochUnavailableError,
        match="no unprotected chronological research epoch remains",
    ):
        rc._select_unprotected_epoch(frame, protected)
    assert issubclass(rc.UnprotectedResearchEpochUnavailableError, rc.EvaluationConflictError)


def test_validation_scenario_defers_when_protected_epochs_cover_all_rows(monkeypatch):
    import pandas as pd

    scenario = rc.ResearchScenario(
        name="protected_epoch_deferred",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2026-01-01",
    )
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=24,
                freq="5min",
            )
        }
    )

    class FullyProtectedMemory:
        def protected_intervals(self, **_kwargs):
            return (
                {
                    "interval_key": "sealed",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                    "start": str(frame["timestamp"].iloc[0]),
                    "end": str(frame["timestamp"].iloc[-1]),
                },
            )

    monkeypatch.setattr(rc, "_scenario_indicator_coverage_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rc,
        "_partition_supported_hypotheses",
        lambda hypotheses, **_kwargs: (hypotheses, []),
    )
    monkeypatch.setattr(rc, "build_aligned_frame", lambda *args, **kwargs: frame.copy())

    report = rc.run_validation_scenario(
        scenario,
        hypotheses=[hypothesis],
        selection={"offset": 2, "next_offset": 3, "selected": 1},
        experiment_memory=FullyProtectedMemory(),
        log_path=None,
    )

    assert report["ok"] is True
    assert report["skipped"] is True
    assert report["deferred"] is True
    assert report["reason"] == "unprotected_epoch_unavailable"
    assert report["selection"]["offset"] == 2
    assert report["holdout_exposed_ids"] == []
    assert report["remediation"]["action"] == "wait_for_unprotected_history"


def _memory_scenario_setup(monkeypatch, *, rows=40):
    import pandas as pd

    scenario = rc.ResearchScenario(
        name="active_income_5m_guarded",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2026-01-01",
    )
    hypothesis = next(hyp for hyp in generate_batch() if hyp.base_timeframe == "5m")
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="5min")}
    )
    monkeypatch.setattr(rc, "_scenario_indicator_coverage_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rc,
        "_partition_supported_hypotheses",
        lambda hypotheses, **_kwargs: (hypotheses, []),
    )
    monkeypatch.setattr(rc, "build_aligned_frame", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        rc,
        "with_trial_sharpe_dispersion",
        lambda _frame, _hypotheses, cfg, _eval_cfg: cfg,
    )
    monkeypatch.setattr(
        rc,
        "_dataset_snapshot",
        lambda scenario_arg, hyps, *, frame, indicator_dir: {
            "snapshot_id": "scenario-snapshot-v1",
            "symbol": "BTCUSDT",
            "market": scenario_arg.market,
            "timeframe": scenario_arg.base_tf,
        },
    )
    return scenario, hypothesis, frame


def test_validation_scenario_does_not_seal_holdout_without_a_gated_candidate(monkeypatch, tmp_path):
    """Regression: eager per-run sealing burned the whole history as protected.

    A scenario pass where no candidate earns a holdout read must not seal the
    frame's holdout window into ``protected_intervals``.
    """
    from src.autopilot.experiment_memory import ExperimentMemory

    scenario, hypothesis, _frame = _memory_scenario_setup(monkeypatch)

    def fake_validate_batch(
        frame_arg, hyps, cfg, *, eval_cfg, log_path, before_holdout, after_candidate
    ):
        results = []
        for hyp in hyps:
            result = {
                "hypothesis_id": hyp.id,
                "family": hyp.family,
                "direction": hyp.direction,
                "verdict": "reject",
                "reasons": ["no_train_edge"],
                "holdout": None,
            }
            after_candidate(hyp, result)
            results.append(result)
        return results

    monkeypatch.setattr(rc, "validate_batch", fake_validate_batch)
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        report = rc.run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            selection={"offset": 0, "next_offset": 1, "selected": 1},
            experiment_memory=memory,
            log_path=None,
        )
        assert report["ok"] is True
        assert "holdout_cohort_scope" not in report
        assert report["holdout_exposed_ids"] == []
        assert memory.protected_intervals(market="futures", symbol="BTCUSDT") == ()


def test_validation_scenario_seals_and_claims_lazily_at_first_holdout_need(monkeypatch, tmp_path):
    from src.autopilot.experiment_memory import ExperimentMemory

    scenario, hypothesis, _frame = _memory_scenario_setup(monkeypatch)
    gate_results = []

    def fake_validate_batch(
        frame_arg, hyps, cfg, *, eval_cfg, log_path, before_holdout, after_candidate
    ):
        results = []
        for hyp in hyps:
            segs = rc.split_frame(frame_arg, cfg)
            partial = {
                "hypothesis_id": hyp.id,
                "family": hyp.family,
                "direction": hyp.direction,
                "splits": {name: rc._segment_bounds(seg) for name, seg in segs.items()},
            }
            gate = before_holdout(hyp, partial)
            gate_results.append(gate)
            result = {
                **partial,
                "verdict": "reject",
                "reasons": ["failed_holdout"],
                "holdout": {"trades": 5, "total_return": -0.01},
            }
            after_candidate(hyp, result)
            results.append(result)
        return results

    monkeypatch.setattr(rc, "validate_batch", fake_validate_batch)
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        report = rc.run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            selection={"offset": 0, "next_offset": 1, "selected": 1},
            experiment_memory=memory,
            log_path=None,
        )
        assert gate_results == [True]
        assert report["holdout_cohort_scope"] is not None
        assert report["holdout_cohort_created"] is True
        assert report["holdout_cohort_members"] == 1
        assert report["holdout_exposed_ids"] == [hypothesis.id]
        assert len(memory.protected_intervals(market="futures", symbol="BTCUSDT")) == 1


def test_validation_scenario_defers_holdout_when_seal_budget_is_exhausted(monkeypatch, tmp_path):
    from src.autopilot.experiment_memory import ExperimentMemory

    scenario, hypothesis, _frame = _memory_scenario_setup(monkeypatch)
    gate_results = []

    def fake_validate_batch(
        frame_arg, hyps, cfg, *, eval_cfg, log_path, before_holdout, after_candidate
    ):
        results = []
        for hyp in hyps:
            segs = rc.split_frame(frame_arg, cfg)
            partial = {
                "hypothesis_id": hyp.id,
                "family": hyp.family,
                "direction": hyp.direction,
                "splits": {name: rc._segment_bounds(seg) for name, seg in segs.items()},
            }
            gate = before_holdout(hyp, partial)
            gate_results.append(gate)
            result = {
                **partial,
                "verdict": "inconclusive",
                "reasons": [gate],
                "holdout": None,
            }
            after_candidate(hyp, result)
            results.append(result)
        return results

    monkeypatch.setattr(rc, "validate_batch", fake_validate_batch)
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        prior = memory.register_strategy(
            {"id": "prior", "kind": "budget-anchor"},
            strategy_id="prior",
            generation_method="grammar_sample",
            metadata={"family": "anchor", "product": "active_income"},
        )
        # A far-past window keeps epoch selection and its feature embargo away
        # from the 2026 scenario frame while still consuming today's budget.
        memory.register_holdout_cohort(
            [prior.behavior_hash],
            dataset={
                "snapshot_id": "prior-snapshot",
                "symbol": "BTCUSDT",
                "market": "futures",
                "timeframe": "5m",
            },
            window={"start": "2024-01-01", "end": "2024-06-01"},
            protocol={"fees_bps": 10},
        )
        report = rc.run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            selection={"offset": 0, "next_offset": 1, "selected": 1},
            experiment_memory=memory,
            log_path=None,
        )
        assert gate_results == ["holdout_seal_budget_exhausted"]
        assert "holdout_cohort_scope" not in report
        assert report["holdout_exposed_ids"] == []
        assert len(memory.protected_intervals(market="futures", symbol="BTCUSDT")) == 1


def test_research_epoch_selection_embargoes_cross_timeframe_feature_dependencies():
    import pandas as pd

    # A 5m strategy that consumes a 1d feature can retain protected-price
    # influence for the grammar's full 240-day rolling dependency. Immediate
    # post-holdout rows must therefore remain unavailable to adaptive research.
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01T00:00:00Z",
                periods=24 * 250 + 1,
                freq="1h",
            ),
            "value": range(24 * 250 + 1),
        }
    )
    protected_end = frame["timestamp"].iloc[0]
    protected = (
        {
            "interval_key": "sealed",
            "market": "futures",
            "symbol": "BTCUSDT",
            "start": str(protected_end),
            "end": str(protected_end),
        },
    )

    selected, detail = rc._select_unprotected_epoch(
        frame,
        protected,
        feature_timeframes=("5m", "1d"),
    )

    assert selected["timestamp"].iloc[0] > protected_end + pd.Timedelta(days=240)
    assert detail["protected_rows_excluded"] == 1
    assert detail["feature_dependency_embargo_rows_excluded"] == 24 * 240
    assert detail["feature_dependency_embargo"] == {
        "policy": "maximum_supported_native_rolling_dependency",
        "max_native_bars": 240,
        "feature_timeframes": ["1d", "5m"],
        "max_timeframe": "1d",
        "duration_seconds": 240 * 86_400,
    }
