import json

from research_exploration.hypothesis_generator import generate_batch, position_trading_set
from research_exploration.hypothesis_schema import Hypothesis
from src.autopilot.mutation_batch import build_mutation_batch, run


def mutation_plan_payload():
    active_source = generate_batch(with_guards=True)[0]
    btc_source = position_trading_set(with_guards=False)[0]
    return {
        "ok": True,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"proposals": 2, "executable": False},
        "proposals": [
            {
                "id": f"mutate:{active_source.id}:no_train_edge",
                "source_candidate_id": active_source.id,
                "source_scenario": "active_income_5m_guarded",
                "product": "active_income",
                "market": "futures",
                "opportunity_type": "day_trading",
                "family": active_source.family,
                "direction": active_source.direction,
                "reason": "no_train_edge",
                "validation_scope": {
                    "candidate_set": "full",
                    "pnl_unit": "usdt",
                    "with_guards": True,
                },
            },
            {
                "id": f"mutate:{btc_source.id}:insufficient_train_trades",
                "source_candidate_id": btc_source.id,
                "source_scenario": "btc_accumulation_1h",
                "product": "btc_accumulation",
                "market": "spot",
                "opportunity_type": "btc_accumulation",
                "family": btc_source.family,
                "direction": btc_source.direction,
                "reason": "insufficient_train_trades",
                "validation_scope": {
                    "candidate_set": "position",
                    "pnl_unit": "btc",
                    "with_guards": False,
                },
            },
        ],
    }


def test_mutation_batch_builds_research_only_schema_valid_hypotheses():
    batch = build_mutation_batch(mutation_plan_payload())

    assert batch["ok"] is True
    assert batch["schema"] == "research_exploration.hypothesis_schema/v1"
    assert batch["research_only"] is True
    assert batch["executable"] is False
    assert batch["paper_trade_allowed"] is False
    assert batch["promotion_allowed"] is False
    assert batch["live_allowed"] is False
    assert batch["summary"]["hypotheses"] == 2
    assert batch["summary"]["skipped"] == 0
    assert batch["summary"]["by_product"] == {"active_income": 1, "btc_accumulation": 1}
    for item in batch["hypotheses"]:
        hyp = Hypothesis.from_dict(item)
        assert hyp.id.startswith("MUT_")
        assert "mutation_plan" in hyp.tags
        assert "research_only" in hyp.tags


def test_mutation_batch_run_writes_only_runtime_output(tmp_path):
    input_path = tmp_path / "mutation_plan.json"
    output_path = tmp_path / "runtime" / "mutation_hypotheses.json"
    input_path.write_text(json.dumps(mutation_plan_payload()), encoding="utf-8")

    batch = run(input_path=input_path, output_path=output_path)

    assert batch["count"] == 2
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["executable"] is False
    assert not (tmp_path / "outputs" / "active_strategies_flow.json").exists()
    assert not (tmp_path / "outputs" / "active_strategies_position.json").exists()


def test_mutation_batch_waits_when_plan_missing(tmp_path):
    output_path = tmp_path / "mutation_hypotheses.json"

    batch = run(input_path=tmp_path / "missing.json", output_path=output_path)

    assert batch["ok"] is True
    assert batch["status"] == "waiting_for_mutation_plan"
    assert batch["count"] == 0
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["status"] == "waiting_for_mutation_plan"
    )


def test_mutation_batch_skips_unknown_source_candidate():
    plan = mutation_plan_payload()
    plan["proposals"][0]["source_candidate_id"] = "DOES_NOT_EXIST"

    batch = build_mutation_batch(plan)

    assert batch["summary"]["hypotheses"] == 1
    assert batch["summary"]["skipped"] == 1
    assert batch["skipped"] == [
        {"source_candidate_id": "DOES_NOT_EXIST", "reason": "source_candidate_not_found"}
    ]


def test_mutation_batch_fails_closed_for_executable_plan():
    plan = mutation_plan_payload()
    plan["summary"]["executable"] = True

    batch = build_mutation_batch(plan)

    assert batch["ok"] is False
    assert batch["status"] == "unsafe_mutation_plan"
    assert batch["count"] == 0
    assert batch["hypotheses"] == []
    assert batch["summary"]["unsafe_flags"] == ["summary.executable"]
    assert batch["research_only"] is True
    assert batch["executable"] is False
    assert batch["live_allowed"] is False


def test_mutation_batch_skips_proposal_with_explicit_live_safety_flag():
    plan = mutation_plan_payload()
    plan["proposals"][0]["safety"] = {"live_allowed": True}

    batch = build_mutation_batch(plan)

    assert batch["ok"] is True
    assert batch["count"] == 1
    assert batch["summary"]["hypotheses"] == 1
    assert batch["summary"]["skipped"] == 1
    assert batch["skipped"] == [
        {
            "source_candidate_id": plan["proposals"][0]["source_candidate_id"],
            "reason": "unsafe_proposal",
            "unsafe_flags": ["safety.live_allowed"],
        }
    ]
    assert (
        batch["mutation_metadata"][0]["source_candidate_id"]
        == plan["proposals"][1]["source_candidate_id"]
    )


def test_mutation_batch_does_not_remutate_mutation_candidate_set():
    plan = mutation_plan_payload()
    source_id = plan["proposals"][0]["source_candidate_id"]
    plan["proposals"] = [
        {
            **plan["proposals"][0],
            "source_candidate_id": f"MUT_{source_id}_NO_TRAIN_EDGE_001",
            "validation_scope": {
                "candidate_set": "mutation",
                "pnl_unit": "usdt",
                "with_guards": False,
            },
        }
    ]

    batch = build_mutation_batch(plan)

    assert batch["count"] == 0
    assert batch["summary"]["skipped"] == 1
    assert batch["skipped"] == [
        {
            "source_candidate_id": f"MUT_{source_id}_NO_TRAIN_EDGE_001",
            "reason": "source_candidate_not_found",
        }
    ]
