import json

from src.autopilot.mutation_plan import build_mutation_plan, render_markdown, run


def research_cycle_payload():
    return {
        "ok": True,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"keepers": 0, "exported": 0},
        "scenarios": [
            {
                "name": "active_income_5m_guarded",
                "product": "active_income",
                "market": "futures",
                "opportunity_type": "day_trading",
                "base_tf": "5m",
                "candidate_set": "full",
                "pnl_unit": "usdt",
                "with_guards": True,
                "incubation_candidates": [
                    {
                        "id": "VOLBREAK_LONG",
                        "family": "volatility_breakout",
                        "direction": "long",
                        "reasons": ["no_train_edge"],
                        "verdict": "reject",
                        "stage_reached": "train",
                        "score": 1.2,
                    },
                    {
                        "id": "MOM_SHORT",
                        "family": "momentum_continuation",
                        "direction": "short",
                        "reasons": ["insufficient_train_trades"],
                        "verdict": "inconclusive",
                        "stage_reached": "train",
                        "score": 1.1,
                    },
                ],
            },
            {
                "name": "btc_accumulation_1h",
                "product": "btc_accumulation",
                "market": "spot",
                "opportunity_type": "btc_accumulation",
                "base_tf": "1h",
                "candidate_set": "position",
                "pnl_unit": "btc",
                "with_guards": False,
                "incubation_candidates": [
                    {
                        "id": "POS_VOLBREAK",
                        "family": "volatility_breakout",
                        "direction": "short",
                        "reasons": ["insufficient_train_trades"],
                        "verdict": "inconclusive",
                        "stage_reached": "train",
                        "score": 1.05,
                    }
                ],
            },
            {
                "name": "mutation_active_income_futures_5m_day_trading",
                "product": "active_income",
                "market": "futures",
                "opportunity_type": "day_trading",
                "base_tf": "5m",
                "candidate_set": "mutation",
                "pnl_unit": "usdt",
                "with_guards": False,
                "incubation_candidates": [
                    {
                        "id": "MUT_VOLBREAK_LONG",
                        "family": "volatility_breakout",
                        "direction": "long",
                        "reasons": ["failed_validation"],
                        "verdict": "reject",
                        "stage_reached": "validation",
                        "score": 9.9,
                    }
                ],
            },
        ],
    }


def test_mutation_plan_builds_research_only_proposals():
    plan = build_mutation_plan(research_cycle_payload(), top_per_scenario=2, max_total=3)

    assert plan["ok"] is True
    assert plan["summary"]["proposals"] == 3
    assert plan["summary"]["by_product"] == {"active_income": 2, "btc_accumulation": 1}
    assert plan["summary"]["skipped_scenarios"] == 1
    assert plan["summary"]["suppressed_repeated_sources"] == 0
    assert plan["summary"]["suppressed_by_product"] == {}
    assert plan["summary"]["suppressed_by_reason"] == {}
    assert plan["skipped_scenarios"] == [
        {
            "name": "mutation_active_income_futures_5m_day_trading",
            "candidate_set": "mutation",
            "reason": "mutation_depth_limit",
            "incubation_candidates": 1,
        }
    ]
    assert plan["summary"]["executable"] is False
    first = plan["proposals"][0]
    assert first["source_candidate_id"] == "VOLBREAK_LONG"
    assert all(
        not proposal["source_candidate_id"].startswith("MUT_") for proposal in plan["proposals"]
    )
    assert first["safety"] == {
        "executable": False,
        "paper_trade_allowed": False,
        "promotion_allowed": False,
        "live_allowed": False,
        "requires_full_validation_before_export": True,
    }
    assert "discard this exact breakout trigger" in first["actions"][0]


def test_mutation_plan_run_writes_json_and_markdown(tmp_path):
    input_path = tmp_path / "research_cycle.json"
    output_path = tmp_path / "mutation_plan.json"
    markdown_path = tmp_path / "mutation_plan.md"
    input_path.write_text(json.dumps(research_cycle_payload()), encoding="utf-8")

    plan = run(input_path=input_path, output_path=output_path, markdown_path=markdown_path)

    assert plan["summary"]["proposals"] == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["proposals"] == 3
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Executable: `False`" in markdown
    assert "Skipped scenarios: `1`" in markdown
    assert "Suppressed repeat sources: `0`" in markdown
    assert "`VOLBREAK_LONG`" in markdown


def test_mutation_plan_waits_when_research_cycle_missing(tmp_path):
    output_path = tmp_path / "mutation_plan.json"

    plan = run(input_path=tmp_path / "missing.json", output_path=output_path, markdown_path=None)

    assert plan["ok"] is True
    assert plan["status"] == "waiting_for_research_cycle"
    assert plan["summary"]["proposals"] == 0
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["status"]
        == "waiting_for_research_cycle"
    )


def test_mutation_plan_markdown_handles_no_proposals():
    markdown = render_markdown({"generated_at": "x", "summary": {"proposals": 0}, "proposals": []})

    assert "no watchlist candidates available" in markdown


def test_mutation_plan_suppresses_recent_failed_mutation_sources():
    payload = research_cycle_payload()
    payload["scenarios"][2]["incubation_candidates"][0]["mutation_lineage"] = {
        "source_candidate_id": "VOLBREAK_LONG",
        "source_scenario": "active_income_5m_guarded",
        "mutation_reason": "no_train_edge",
        "validation_scope": {"candidate_set": "full", "pnl_unit": "usdt"},
    }
    payload["scenarios"][0]["incubation_candidates"].append(
        {
            "id": "SWEEP_LONG",
            "family": "liquidity_sweep",
            "direction": "long",
            "reasons": ["no_train_edge"],
            "verdict": "reject",
            "stage_reached": "train",
            "score": 0.8,
        }
    )

    plan = build_mutation_plan(payload, top_per_scenario=2, max_total=4)

    sources = [proposal["source_candidate_id"] for proposal in plan["proposals"]]
    assert "VOLBREAK_LONG" not in sources
    assert "MOM_SHORT" in sources
    assert "SWEEP_LONG" in sources
    assert plan["summary"]["suppressed_repeated_sources"] == 1
    assert plan["summary"]["suppressed_by_product"] == {"active_income": 1}
    assert plan["summary"]["suppressed_by_reason"] == {"no_train_edge": 1}
    assert plan["suppressed_repeated_sources"] == [
        {
            "product": "active_income",
            "source_scenario": "active_income_5m_guarded",
            "source_candidate_id": "VOLBREAK_LONG",
            "reason": "no_train_edge",
            "mutation_candidate_id": "MUT_VOLBREAK_LONG",
            "mutation_result_reasons": ["failed_validation"],
            "stage_reached": "validation",
            "verdict": "reject",
        }
    ]


def test_mutation_plan_retires_failed_holdout_without_generating_feedback():
    payload = research_cycle_payload()
    payload["scenarios"][0]["incubation_candidates"].append(
        {
            "id": "HOLDOUT_FAILURE",
            "family": "volatility_breakout",
            "direction": "long",
            "reasons": ["failed_holdout"],
            "verdict": "reject",
            "stage_reached": "holdout",
            "score": 99.0,
        }
    )

    plan = build_mutation_plan(payload, top_per_scenario=3, max_total=10)

    assert "HOLDOUT_FAILURE" not in {
        proposal["source_candidate_id"] for proposal in plan["proposals"]
    }
    assert plan["summary"]["retired_candidates"] == 1
    assert plan["summary"]["retired_by_product"] == {"active_income": 1}
    assert plan["summary"]["retired_by_reason"] == {"failed_holdout": 1}
    assert plan["retired_candidates"] == [
        {
            "source_scenario": "active_income_5m_guarded",
            "product": "active_income",
            "source_candidate_id": "HOLDOUT_FAILURE",
            "reason": "failed_holdout",
            "source_verdict": "reject",
            "stage_reached": "holdout",
            "score": 99.0,
            "disposition": "retired_from_autonomous_mutation",
        }
    ]
