from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest

from research_exploration.strategy_grammar import SearchSpace, build_fresh_hypothesis
from src.autopilot.experiment_memory import ExperimentMemory
from src.autopilot.openclaw_bridge import build_accepted_proposal
from src.autopilot.research_factory import (
    DEFAULT_CONFIG,
    FactoryBudgets,
    ResearchFactoryConfig,
    _candidate_is_near_duplicate,
    _load_accepted_proposals,
    _method_schedule,
    _search_spaces_for_cycle,
    build_generation,
    load_factory_config,
    strategy_behavior_spec,
)


def test_default_factory_uses_dynamic_symbol_scoped_active_income_templates():
    config = load_factory_config(DEFAULT_CONFIG)
    active = [space for space in config.search_spaces if space.product == "active_income"]

    assert config.dynamic_active_income_universe is True
    assert {space.symbol for space in active} == {"BTCUSDT"}
    assert len(active) == 3
    assert all(
        space.symbol == "BTCUSDT"
        for space in config.search_spaces
        if space.product == "btc_accumulation"
    )


def test_factory_expands_fresh_market_universe_across_all_active_horizons(tmp_path):
    config = load_factory_config(DEFAULT_CONFIG)
    report = tmp_path / "market_universe.json"
    report.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "autopilot.market_universe/v2",
                "generated_at": "2026-07-27T00:00:00+00:00",
                "eligible_research_symbols": ["BTCUSDT", "ETHUSDT", "DOGEUSDT"],
                "symbols": [
                    {"symbol": symbol, "metrics": {"listing_days": 2000}}
                    for symbol in ("BTCUSDT", "ETHUSDT", "DOGEUSDT")
                ],
                "snapshot": {"id": "sha256:" + "3" * 64},
            }
        ),
        encoding="utf-8",
    )

    spaces = _search_spaces_for_cycle(
        config,
        generated_at="2026-07-27T12:00:00+00:00",
        report_path=report,
    )
    active = [space for space in spaces if space.product == "active_income"]

    assert {space.symbol for space in active} == {
        "BTCUSDT",
        "ETHUSDT",
        "DOGEUSDT",
    }
    assert len(active) == 9
    assert {space.opportunity_type for space in active if space.symbol == "DOGEUSDT"} == {
        "scalping",
        "day_trading",
        "swing_trading",
    }


def test_factory_excludes_horizons_that_predate_symbol_listing(tmp_path):
    config = load_factory_config(DEFAULT_CONFIG)
    report = tmp_path / "market_universe.json"
    report.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "autopilot.market_universe/v2",
                "generated_at": "2026-07-27T00:00:00+00:00",
                "eligible_research_symbols": ["BTCUSDT", "SUIUSDT"],
                "symbols": [
                    {"symbol": "BTCUSDT", "metrics": {"listing_days": 2500}},
                    {"symbol": "SUIUSDT", "metrics": {"listing_days": 1181.5}},
                ],
                "snapshot": {"id": "sha256:" + "5" * 64},
            }
        ),
        encoding="utf-8",
    )

    spaces = _search_spaces_for_cycle(
        config,
        generated_at="2026-07-27T12:00:00+00:00",
        report_path=report,
    )

    sui = [space for space in spaces if space.symbol == "SUIUSDT"]
    assert [space.opportunity_type for space in sui] == ["scalping"]


def _spaces() -> tuple[SearchSpace, ...]:
    return (
        SearchSpace(
            name="active_day",
            product="active_income",
            market="futures",
            pnl_unit="usdt",
            opportunity_type="day_trading",
            base_timeframe="5m",
            regime_timeframe="1h",
            setup_timeframe="15m",
            trigger_timeframe="5m",
            directions=("long", "short"),
            take_profit_range=(0.004, 0.025),
            stop_loss_range=(0.003, 0.015),
            horizon_range=(6, 144),
            risk_per_trade_range=(0.001, 0.005),
            max_position_fraction=0.12,
            max_trades_per_day=6,
        ),
        SearchSpace(
            name="btc_position",
            product="btc_accumulation",
            market="spot",
            pnl_unit="btc",
            opportunity_type="btc_accumulation",
            base_timeframe="1h",
            regime_timeframe="1d",
            setup_timeframe="4h",
            trigger_timeframe="1h",
            directions=("short",),
            take_profit_range=(0.01, 0.08),
            stop_loss_range=(0.008, 0.04),
            horizon_range=(12, 200),
            risk_per_trade_range=(0.001, 0.003),
            max_position_fraction=0.25,
            max_trades_per_day=2,
        ),
    )


def _config(tmp_path: Path, *, candidates: int = 6) -> ResearchFactoryConfig:
    return ResearchFactoryConfig(
        path=tmp_path / "research_factory.json",
        memory_path=tmp_path / "memory.sqlite3",
        generated_batch_path=tmp_path / "batch.json",
        openclaw_accepted_dir=tmp_path / "openclaw" / "accepted",
        proposal_state_path=tmp_path / "proposal_state.json",
        budgets=FactoryBudgets(
            max_candidates_per_cycle=candidates,
            max_candidates_per_space=max(1, candidates // 2),
            max_generation_attempts=500,
            max_generation_seconds=10,
            max_parent_pool=100,
            max_lineage_depth=5,
            max_total_predicates=7,
            max_memory_bytes=50 * 1024 * 1024,
            near_duplicate_threshold=0.88,
            exploration_fraction=0.5,
            mutation_fraction=0.3,
            crossover_fraction=0.2,
        ),
        search_spaces=_spaces(),
    )


def test_native_factory_generates_unique_bounded_specs_and_resumes_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    config = _config(tmp_path)

    first = build_generation(config, seed=11, now="2026-07-10T00:00:00+00:00")
    second = build_generation(config, seed=11, now="2026-07-10T01:00:00+00:00")

    first_hashes = {item["strategy_hash"] for item in first["generation_metadata"]}
    second_hashes = {item["strategy_hash"] for item in second["generation_metadata"]}
    assert first["ok"] is True
    assert len(first["hypotheses"]) == 6
    assert len(first_hashes) == 6
    assert first["summary"]["by_product"] == {"active_income": 3, "btc_accumulation": 3}
    assert first["research_only"] is True and first["live_allowed"] is False
    assert second_hashes == first_hashes
    assert second["summary"]["resumed_pending"] == 6
    assert second["summary"]["new_hypotheses"] == 0


def test_completed_development_results_create_a_new_generation(monkeypatch, tmp_path):
    engine_digest = "sha256:" + "4" * 64
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    monkeypatch.setattr(
        "src.autopilot.research_factory.execution_engine_digest", lambda: engine_digest
    )
    config = _config(tmp_path)
    first = build_generation(config, seed=3, now="2026-07-10T00:00:00+00:00")
    first_hashes = {item["strategy_hash"] for item in first["generation_metadata"]}

    with ExperimentMemory(config.memory_path) as memory:
        for item in first["generation_metadata"]:
            memory.record_outcome(
                item["strategy_hash"],
                dataset={"snapshot_id": "train-snapshot-v1"},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={"version": 1, "research_engine_digest": engine_digest},
                phase="development",
                outcome="reject",
                rejection_reasons=("no_train_edge",),
                metrics={"train_trades": 50},
            )

    second = build_generation(config, seed=4, now="2026-07-11T00:00:00+00:00")
    second_hashes = {item["strategy_hash"] for item in second["generation_metadata"]}

    assert len(second_hashes) == 6
    assert first_hashes.isdisjoint(second_hashes)
    assert second["summary"]["resumed_pending"] == 0
    assert second["summary"]["new_hypotheses"] == 6
    assert set(second["summary"]["by_method"]) & {
        "recursive_mutation",
        "crossover",
        "grammar_sample",
    }


def test_openclaw_is_optional_untrusted_and_never_bypasses_native_grammar(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    config = _config(tmp_path, candidates=4)
    config.openclaw_accepted_dir.mkdir(parents=True)
    accepted = build_accepted_proposal(
        {
            "schema": "research_proposal/v1",
            "source": "openclaw",
            "created_at": "2026-07-10T12:00:00+00:00",
            "objective": "active_income",
            "opportunity_type": "day",
            "base_timeframe": "5m",
            "thesis": "A volatility compression followed by range expansion may continue intraday.",
            "suggested_primitives": ["volatility", "range expansion"],
            "constraints": ["research only"],
            "suggested_spec": {"arbitrary_python": "import os"},
        }
    )
    config.openclaw_accepted_dir.joinpath("proposal.json").write_text(
        json.dumps(accepted), encoding="utf-8"
    )

    report = build_generation(config, seed=8, now="2026-07-10T13:00:00+00:00")

    assert report["ok"] is True
    assert len(report["hypotheses"]) == 4
    # The unknown suggested structure is rejected by the trusted compiler;
    # native generation still fills the bounded batch.
    assert any(item.get("reason") == "openclaw_compile_rejected" for item in report["rejected"])
    assert list(config.openclaw_accepted_dir.glob("*.json")) == []
    assert report["summary"]["by_product"]["btc_accumulation"] >= 1
    assert all(item["live_allowed"] is False for item in [report])


def test_alfred_revision_is_lineage_linked_and_queued_for_trusted_testing(monkeypatch, tmp_path):
    engine = "sha256:" + "9" * 64
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    monkeypatch.setattr("src.autopilot.research_factory.execution_engine_digest", lambda: engine)
    config = _config(tmp_path, candidates=4)
    first = build_generation(config, seed=50, now="2026-07-10T00:00:00+00:00")
    with ExperimentMemory(config.memory_path) as memory:
        for item in first["generation_metadata"]:
            memory.record_outcome(
                item["strategy_hash"],
                dataset={"snapshot_id": "development-v1"},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={"research_engine_digest": engine},
                phase="development",
                outcome="reject",
                rejection_reasons=("no_train_edge",),
            )
    parent = next(
        item for item in first["generation_metadata"] if item["product"] == "active_income"
    )
    config.openclaw_accepted_dir.mkdir(parents=True)
    accepted = build_accepted_proposal(
        {
            "schema": "research_action/v1",
            "source": "openclaw",
            "created_at": "2026-07-10T12:00:00+00:00",
            "action": "revise",
            "parent_hypothesis_id": parent["strategy_hash"],
            "objective": "active_income",
            "opportunity_type": "day",
            "base_timeframe": "5m",
            "thesis": "Simplifying the weak parent may recover a measurable development edge.",
            "reasoning": "The parent completed development with no train edge, so alter one bounded grammar element.",
            "changes": ["simplify or replace one entry condition"],
            "expected_outcome": "Development edge becomes positive with sufficient trades.",
            "falsification_criteria": "Reject if edge remains non-positive or trade count is insufficient.",
            "source_proposal_id": "review-1:revision-1",
        }
    )
    config.openclaw_accepted_dir.joinpath("revision.json").write_text(
        json.dumps(accepted), encoding="utf-8"
    )

    report = build_generation(config, seed=51, now="2026-07-10T13:00:00+00:00")
    revised = next(
        item
        for item in report["generation_metadata"]
        if item.get("proposal_id") == accepted["proposal_id"]
    )
    state = json.loads(config.proposal_state_path.read_text())
    disposition = state["processed"][accepted["proposal_id"]]

    assert revised["research_action"] == "revise"
    assert revised["parent_hashes"] == [parent["strategy_hash"]]
    assert revised["strategy_hash"] != parent["strategy_hash"]
    assert disposition["status"] == "queued_for_test"
    assert disposition["parent_hypothesis_id"] == parent["strategy_hash"]
    assert disposition["strategy_hash"] == revised["strategy_hash"]


def test_alfred_retry_queues_existing_identity_and_retire_closes_research_branch(
    monkeypatch, tmp_path
):
    engine = "sha256:" + "8" * 64
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    monkeypatch.setattr("src.autopilot.research_factory.execution_engine_digest", lambda: engine)
    config = _config(tmp_path, candidates=4)
    first = build_generation(config, seed=60, now="2026-07-10T00:00:00+00:00")
    with ExperimentMemory(config.memory_path) as memory:
        for item in first["generation_metadata"]:
            memory.record_outcome(
                item["strategy_hash"],
                dataset={"snapshot_id": "development-v1"},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={"research_engine_digest": engine},
                phase="development",
                outcome="inconclusive",
                rejection_reasons=(),
            )
    active = next(
        item for item in first["generation_metadata"] if item["product"] == "active_income"
    )
    btc = next(
        item for item in first["generation_metadata"] if item["product"] == "btc_accumulation"
    )
    config.openclaw_accepted_dir.mkdir(parents=True)
    retry = build_accepted_proposal(
        {
            "schema": "research_action/v1",
            "source": "openclaw",
            "created_at": "2026-07-10T12:00:00+00:00",
            "action": "retry",
            "parent_hypothesis_id": active["strategy_hash"],
            "objective": "active_income",
            "opportunity_type": "day",
            "base_timeframe": "5m",
            "thesis": "The inconclusive parent merits one test on the latest eligible snapshot.",
            "reasoning": "The prior result was inconclusive rather than rejected and new data may resolve it.",
            "expected_outcome": "The new snapshot produces a decisive development result.",
            "falsification_criteria": "Retire or revise if the result remains inconclusive.",
            "source_proposal_id": "review-2:retry-1",
        }
    )
    retire = build_accepted_proposal(
        {
            "schema": "research_action/v1",
            "source": "openclaw",
            "created_at": "2026-07-10T12:01:00+00:00",
            "action": "retire",
            "parent_hypothesis_id": btc["strategy_hash"],
            "objective": "btc_accumulation",
            "opportunity_type": "position",
            "base_timeframe": "1h",
            "thesis": "This exhausted BTC research branch should stop consuming experiment budget.",
            "reasoning": "The branch is no longer the best use of bounded research capacity.",
            "expected_outcome": "Research budget shifts to more informative hypotheses.",
            "falsification_criteria": "Reconsider only if materially new development evidence appears.",
            "source_proposal_id": "review-2:retire-1",
        }
    )
    config.openclaw_accepted_dir.joinpath("retry.json").write_text(
        json.dumps(retry), encoding="utf-8"
    )
    config.openclaw_accepted_dir.joinpath("retire.json").write_text(
        json.dumps(retire), encoding="utf-8"
    )

    report = build_generation(config, seed=61, now="2026-07-10T13:00:00+00:00")
    retried = next(
        item
        for item in report["generation_metadata"]
        if item.get("proposal_id") == retry["proposal_id"]
    )
    state = json.loads(config.proposal_state_path.read_text())["processed"]

    assert retried["strategy_hash"] == active["strategy_hash"]
    assert retried["requested_test"] is True
    assert state[retry["proposal_id"]]["status"] == "queued_for_test"
    assert state[retire["proposal_id"]]["status"] == "retired"
    with ExperimentMemory(config.memory_path) as memory:
        assert memory.get_strategy(btc["strategy_hash"])["retired_at"] is not None


def test_load_config_requires_both_products_horizons_and_fail_closed_holdout(monkeypatch, tmp_path):
    monkeypatch.setattr("src.autopilot.research_factory.PROJECT_ROOT", tmp_path)
    payload = json.loads(Path("config/research_factory.json").read_text(encoding="utf-8"))
    payload["memory_path"] = "runtime/memory.sqlite3"
    payload["generated_batch_path"] = "runtime/batch.json"
    payload["openclaw_accepted_dir"] = "runtime/openclaw/accepted"
    payload["openclaw_proposal_state_path"] = "runtime/proposals.json"
    path = tmp_path / "research_factory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_factory_config(path)

    assert {space.product for space in config.search_spaces} == {
        "active_income",
        "btc_accumulation",
    }
    assert {space.opportunity_type for space in config.search_spaces} >= {
        "scalping",
        "day_trading",
        "swing_trading",
        "btc_accumulation",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["search_spaces"][0].update(
                {"take_profit_range": ["NaN", 0.012]}
            ),
            "take_profit_range values must be numeric",
        ),
        (
            lambda payload: payload["search_spaces"][0].update({"max_trades_per_day": True}),
            "max_trades_per_day must be an integer or null",
        ),
        (
            lambda payload: payload["search_spaces"][0].update(
                {"base_timeframe": "5m", "trigger_timeframe": "5m"}
            ),
            "scalping search spaces must use base_timeframe 1m",
        ),
        (
            lambda payload: payload["search_spaces"][0].update(
                {"market": "spot", "pnl_unit": "btc"}
            ),
            "active_income must use futures market and usdt PnL",
        ),
    ],
)
def test_load_config_strictly_rejects_unsafe_search_space_coercions(
    monkeypatch,
    tmp_path,
    mutate,
    message,
):
    monkeypatch.setattr("src.autopilot.research_factory.PROJECT_ROOT", tmp_path)
    payload = json.loads(Path("config/research_factory.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "memory_path": "runtime/memory.sqlite3",
            "generated_batch_path": "runtime/batch.json",
            "openclaw_accepted_dir": "runtime/openclaw/accepted",
            "openclaw_proposal_state_path": "runtime/proposals.json",
        }
    )
    mutate(payload)
    path = tmp_path / "research_factory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_factory_config(path)


def test_operator_schedule_adapts_but_preserves_fresh_exploration(tmp_path):
    budgets = _config(tmp_path, candidates=100).budgets
    feedback = {
        "generation_methods": {
            "recursive_mutation": {
                "experiments": 20,
                "outcomes": {"reject": 20},
                "proposals": 20,
                "duplicates": 12,
                "mean_novelty": 0.1,
            },
            "crossover": {
                "experiments": 20,
                "outcomes": {"pre_holdout_pass": 15, "reject": 5},
                "proposals": 20,
                "duplicates": 0,
                "mean_novelty": 0.8,
            },
        }
    }

    schedule = _method_schedule(100, budgets, random.Random(1), feedback=feedback)

    assert schedule.count("grammar_sample") >= 25
    assert schedule.count("crossover") > schedule.count("recursive_mutation")


def test_factory_fails_closed_when_experiment_memory_exceeds_storage_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    config = _config(tmp_path, candidates=2)
    config = dataclasses.replace(
        config,
        budgets=dataclasses.replace(config.budgets, max_memory_bytes=1024),
    )
    with ExperimentMemory(config.memory_path):
        pass

    with pytest.raises(ValueError, match="remains above max_memory_bytes"):
        build_generation(config, seed=1, now="2026-07-10T00:00:00+00:00")


def test_factory_compacts_valid_memory_before_the_storage_ceiling(monkeypatch, tmp_path):
    engine = "sha256:" + "8" * 64
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    monkeypatch.setattr("src.autopilot.research_factory.execution_engine_digest", lambda: engine)
    config = _config(tmp_path, candidates=2)
    first = build_generation(config, seed=40, now="2026-07-10T00:00:00+00:00")
    large = "compressible-development-evidence-" * 30_000
    with ExperimentMemory(config.memory_path) as memory:
        for index, item in enumerate(first["generation_metadata"]):
            memory.record_outcome(
                item["strategy_hash"],
                dataset={"snapshot_id": f"development-{index}", "manifest": large},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={"research_engine_digest": engine},
                phase="development",
                outcome="reject",
                rejection_reasons=("no_train_edge",),
                metrics={"trace": large},
                details={"diagnostics": large},
            )
    before = config.memory_path.stat().st_size
    bounded = dataclasses.replace(
        config,
        budgets=dataclasses.replace(config.budgets, max_memory_bytes=512 * 1024),
    )

    report = build_generation(bounded, seed=41, now="2026-07-11T00:00:00+00:00")

    maintenance = report["memory"]["maintenance"]
    assert before > bounded.budgets.max_memory_bytes
    assert report["ok"] is True
    assert maintenance["triggered"] is True
    assert maintenance["rows_compacted"] >= 6
    assert maintenance["after_bytes"] < maintenance["before_bytes"]
    assert report["budget"]["memory_bytes"] <= bounded.budgets.max_memory_bytes


def test_factory_revalidates_existing_behavior_after_engine_change_without_reregistering(
    monkeypatch, tmp_path
):
    old_engine = "sha256:" + "1" * 64
    current_engine = "sha256:" + "2" * 64
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    monkeypatch.setattr(
        "src.autopilot.research_factory.execution_engine_digest", lambda: old_engine
    )
    config = _config(tmp_path, candidates=4)
    first = build_generation(config, seed=17, now="2026-07-10T00:00:00+00:00")
    first_hashes = {item["strategy_hash"] for item in first["generation_metadata"]}
    with ExperimentMemory(config.memory_path) as memory:
        for item in first["generation_metadata"]:
            memory.record_outcome(
                item["strategy_hash"],
                dataset={"snapshot_id": "old-engine-train"},
                window={"start": "2024-01-01", "end": "2025-01-01"},
                protocol={"research_engine_digest": old_engine},
                phase="development",
                outcome="reject",
                rejection_reasons=("no_train_edge",),
            )
        strategy_count = memory.generator_feedback()["totals"]["strategies"]

    monkeypatch.setattr(
        "src.autopilot.research_factory.execution_engine_digest", lambda: current_engine
    )
    second = build_generation(config, seed=18, now="2026-07-11T00:00:00+00:00")

    second_hashes = {item["strategy_hash"] for item in second["generation_metadata"]}
    assert len(first_hashes & second_hashes) == 2
    assert len(second_hashes - first_hashes) == 2
    assert second["summary"]["revalidation_pending"] == 2
    assert second["summary"]["new_hypotheses"] == 2
    with ExperimentMemory(config.memory_path) as memory:
        assert memory.generator_feedback()["totals"]["strategies"] == strategy_count + 2


def test_taxonomy_only_search_space_rename_resumes_canonical_pending_work(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    config = _config(tmp_path, candidates=4)
    first = build_generation(config, seed=31, now="2026-07-10T00:00:00+00:00")
    renamed = dataclasses.replace(
        config,
        search_spaces=tuple(
            dataclasses.replace(space, name=f"renamed_{space.name}")
            for space in config.search_spaces
        ),
    )

    second = build_generation(renamed, seed=32, now="2026-07-10T01:00:00+00:00")

    assert {item["strategy_hash"] for item in second["generation_metadata"]} == {
        item["strategy_hash"] for item in first["generation_metadata"]
    }
    assert second["summary"]["resumed_pending"] == 4
    assert {item["search_space"] for item in second["generation_metadata"]} == {
        "renamed_active_day",
        "renamed_btc_position",
    }


def test_openclaw_scan_skips_processed_prefix_instead_of_stalling(tmp_path):
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    processed: set[str] = set()
    for index in range(101):
        proposal = build_accepted_proposal(
            {
                "schema": "research_proposal/v1",
                "source": "openclaw",
                "created_at": "2026-07-10T12:00:00+00:00",
                "objective": "active_income",
                "opportunity_type": "day",
                "base_timeframe": "5m",
                "thesis": f"Distinct bounded research thesis number {index}",
                "source_proposal_id": f"proposal-{index:03d}",
            }
        )
        accepted_dir.joinpath(f"{index:03d}.json").write_text(
            json.dumps(proposal), encoding="utf-8"
        )
        if index < 100:
            processed.add(proposal["proposal_id"])

    loaded = _load_accepted_proposals(accepted_dir, processed)

    assert len(loaded) == 1
    assert loaded[0]["proposal_id"] not in processed


def test_factory_restart_purges_durably_processed_openclaw_file_without_reprocessing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "src.autopilot.research_factory._feature_inventory_for_space", lambda _space: None
    )
    config = _config(tmp_path, candidates=2)
    config.openclaw_accepted_dir.mkdir(parents=True)
    accepted = build_accepted_proposal(
        {
            "schema": "research_proposal/v1",
            "source": "openclaw",
            "created_at": "2026-07-10T12:00:00+00:00",
            "objective": "active_income",
            "opportunity_type": "day",
            "base_timeframe": "5m",
            "thesis": "A distinct restart-safety proposal that must not be processed twice.",
        }
    )
    accepted_path = config.openclaw_accepted_dir / "proposal.json"
    accepted_path.write_text(json.dumps(accepted), encoding="utf-8")
    config.proposal_state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "processed": {
                    accepted["proposal_id"]: {
                        "processed_at": "2026-07-10T12:30:00+00:00",
                        "status": "rejected",
                        "reason": "already_durable",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_generation(config, seed=22, now="2026-07-10T13:00:00+00:00")

    assert report["summary"]["openclaw_proposals_seen"] == 0
    assert report["summary"]["openclaw_accepted_files_purged"] == 1
    assert not accepted_path.exists()


def test_global_near_dedup_blocks_renamed_roots_but_allows_parent_parameter_adaptation():
    space = _spaces()[0]
    parent = build_fresh_hypothesis(space, rng=random.Random(7)).hypothesis
    parent_hash = "sha256:" + "a" * 64
    population = [
        {
            "behavior_hash": parent_hash,
            "submitted_spec": strategy_behavior_spec(parent, space),
            "metadata": {"product": space.product},
        }
    ]
    adjusted = dataclasses.replace(
        parent,
        exit=dataclasses.replace(
            parent.exit,
            take_profit=min(
                space.take_profit_range[1],
                max(space.take_profit_range[0], parent.exit.take_profit * 1.1),
            ),
        ),
    )

    root_near, _, _ = _candidate_is_near_duplicate(
        adjusted,
        population,
        0.88,
        include_values=False,
    )
    descendant_near, _, _ = _candidate_is_near_duplicate(
        adjusted,
        population,
        0.88,
        include_values=True,
        excluded_hashes=frozenset({parent_hash}),
    )

    assert root_near is True
    assert descendant_near is False
