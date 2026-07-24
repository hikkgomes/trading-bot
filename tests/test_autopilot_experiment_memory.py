import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.autopilot import experiment_memory as experiment_memory_module
from src.autopilot.experiment_memory import (
    EvaluationConflictError,
    ExperimentMemory,
    ExperimentMemoryClosedError,
    ExperimentMemoryCorruptionError,
    HoldoutSealBudgetError,
    StrategyIdentityConflictError,
    canonical_strategy_hash,
    canonical_test_hash,
)


def strategy_spec(
    strategy_id: str,
    *,
    reference: float = 50,
    family: str = "momentum",
    predicates: list[dict] | None = None,
) -> dict:
    if predicates is None:
        predicates = [
            {"timeframe": "1h", "feature": "ema_20", "op": "gt_feature", "feature_b": "ema_50"},
            {"timeframe": "1h", "feature": "rsi_14", "op": "gt", "reference": reference},
        ]
    return {
        "id": strategy_id,
        "family": family,
        "idea": f"presentation for {strategy_id}",
        "market_logic": "human-facing explanation",
        "direction": "long",
        "product": "active_income",
        "opportunity_type": "swing",
        "regime": predicates,
        "setup": [],
        "trigger": [
            {"timeframe": "5m", "feature": "close", "op": "gt_feature", "feature_b": "ema_20"}
        ],
        "exit": {"take_profit": 0.02, "stop_loss": 0.01, "horizon_bars": 24},
        "risk": {"risk_per_trade": 0.005, "max_position_fraction": 0.1},
        "expected_holding": "hours",
        "expected_frequency": "weekly",
        "invalidation": "prose only",
        "feature_columns": ["derived", "presentation", "cache"],
        "lineage": {"parent": "presentation-only"},
    }


def metadata(*, family: str = "momentum") -> dict:
    return {
        "family": family,
        "product": "active_income",
        "opportunity_type": "swing",
    }


def dataset(snapshot_id: str = "candles:BTCUSDT:2026-01-01:sha256-abc") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "symbol": "BTCUSDT",
        "market": "futures",
        "timeframe": "5m",
    }


def window(start: str = "2025-01-01", end: str = "2025-06-01") -> dict:
    return {"start": start, "end": end, "purge_bars": 24, "embargo_bars": 24}


def test_behavior_hash_ignores_identity_prose_and_commutative_predicate_order():
    first = strategy_spec("first")
    second = strategy_spec("second", family="called-something-else")
    second["idea"] = "entirely different prose"
    second["market_logic"] = "different explanation"
    second["regime"] = list(reversed(second["regime"]))
    second["lineage"] = {"parents": ["anything"]}

    assert canonical_strategy_hash(first) == canonical_strategy_hash(second)

    changed = strategy_spec("changed", reference=55)
    assert canonical_strategy_hash(first) != canonical_strategy_hash(changed)


def test_behavior_hash_ignores_search_taxonomy_but_keeps_execution_context():
    first = strategy_spec("first")
    first.update(
        _search_space="active_day_v1",
        _opportunity_type="day_trading",
        _market="futures",
        _pnl_unit="usdt",
    )
    renamed = {**first, "_search_space": "renamed_campaign", "_opportunity_type": "swing"}
    changed_market = {**first, "_market": "spot"}

    assert canonical_strategy_hash(first) == canonical_strategy_hash(renamed)
    assert canonical_strategy_hash(first) != canonical_strategy_hash(changed_market)


def test_registers_full_spec_identity_dedup_novelty_and_lineage(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with ExperimentMemory(path) as memory:
        parent = memory.register_strategy(
            strategy_spec("parent"),
            strategy_id="parent",
            generation_method="seed",
            metadata=metadata(),
        )
        child = memory.register_strategy(
            strategy_spec("child", reference=55),
            strategy_id="child",
            generation_method="threshold_mutation",
            parent_hashes=[parent.behavior_hash],
            metadata=metadata(),
        )
        duplicate = memory.register_strategy(
            strategy_spec("duplicate", reference=55, family="renamed"),
            strategy_id="duplicate",
            generation_method="crossover",
            parent_hashes=[parent.behavior_hash],
            metadata=metadata(family="renamed"),
        )

        assert parent.created is True
        assert parent.novelty_score == 1.0
        assert child.created is True
        assert 0 < child.novelty_score < 1
        assert child.nearest_behavior_hash == parent.behavior_hash
        assert duplicate.created is False
        assert duplicate.duplicate is True
        assert duplicate.behavior_hash == child.behavior_hash
        assert memory.find_duplicate(strategy_spec("another", reference=55)) == child.behavior_hash

        stored = memory.get_strategy(child.behavior_hash)
        assert stored["spec"].get("id") is None
        assert stored["submitted_spec"]["id"] == "child"
        assert stored["submitted_spec"]["family"] == "momentum"
        assert stored["parent_hashes"] == [parent.behavior_hash]
        assert "op:gt" in stored["primitives"]
        assert memory.ancestry(child.behavior_hash)["edges"] == [
            {
                "child_hash": child.behavior_hash,
                "parent_hash": parent.behavior_hash,
                "depth": 1,
            }
        ]

        idempotent = memory.register_strategy(
            strategy_spec("child", reference=55),
            strategy_id="child",
            generation_method="threshold_mutation",
            parent_hashes=[parent.behavior_hash],
            metadata=metadata(),
        )
        assert idempotent.identity_created is False

        with pytest.raises(StrategyIdentityConflictError, match="different behavior"):
            memory.register_strategy(
                strategy_spec("child", reference=60),
                strategy_id="child",
                generation_method="threshold_mutation",
                metadata=metadata(),
            )

        with pytest.raises(ValueError, match="unknown parent"):
            memory.register_strategy(
                strategy_spec("orphan", reference=65),
                strategy_id="orphan",
                generation_method="mutation",
                parent_hashes=["sha256:" + "0" * 64],
                metadata=metadata(),
            )


def test_evaluation_key_uses_snapshot_window_and_protocol_and_is_immutable(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("candidate"),
            strategy_id="candidate",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        key = canonical_test_hash(
            registered.behavior_hash,
            dataset=dataset(),
            window=window(),
            protocol={"fees_bps": 6, "walk_forward": 4},
        )
        assert key != canonical_test_hash(
            registered.behavior_hash,
            dataset=dataset("different-snapshot"),
            window=window(),
            protocol={"fees_bps": 6, "walk_forward": 4},
        )
        assert key != canonical_test_hash(
            registered.behavior_hash,
            dataset=dataset(),
            window=window(end="2025-07-01"),
            protocol={"fees_bps": 6, "walk_forward": 4},
        )

        with pytest.raises(ValueError, match="snapshot_id"):
            memory.record_outcome(
                registered.behavior_hash,
                dataset={"path": "mutable.parquet"},
                window=window(),
                outcome="keep",
            )
        with pytest.raises(ValueError, match="rejection reason"):
            memory.record_outcome(
                registered.behavior_hash,
                dataset=dataset(),
                window=window(),
                outcome="reject",
            )

        result = memory.record_outcome(
            registered.behavior_hash,
            dataset=dataset(),
            window=window(),
            protocol={"fees_bps": 6, "walk_forward": 4},
            outcome="reject",
            rejection_reasons=["unstable_across_windows"],
            metrics={"sharpe": -0.2},
        )
        assert result.evaluation_key == key
        assert result.created is True
        evidence = memory.list_evaluations(
            behavior_hash=registered.behavior_hash,
            status="completed",
            phase="validation",
            limit=1,
        )
        assert evidence[0]["data_snapshot_id"] == dataset()["snapshot_id"]
        assert evidence[0]["window"] == window()
        assert evidence[0]["protocol"] == {"fees_bps": 6, "walk_forward": 4}
        assert evidence[0]["rejection_reasons"] == ["unstable_across_windows"]
        assert evidence[0]["metrics"] == {"sharpe": -0.2}
        assert memory.is_tested(
            registered.behavior_hash,
            dataset=dataset(),
            window=window(),
            protocol={"fees_bps": 6, "walk_forward": 4},
        )

        repeated = memory.record_outcome(
            registered.behavior_hash,
            dataset=dataset(),
            window=window(),
            protocol={"fees_bps": 6, "walk_forward": 4},
            outcome="reject",
            rejection_reasons=["unstable_across_windows"],
            metrics={"sharpe": -0.2},
        )
        assert repeated.created is False
        with pytest.raises(EvaluationConflictError, match="different evidence"):
            memory.record_outcome(
                registered.behavior_hash,
                dataset=dataset(),
                window=window(),
                protocol={"fees_bps": 6, "walk_forward": 4},
                outcome="keep",
                metrics={"sharpe": 3.0},
            )


def test_holdout_claim_survives_restart_and_blocks_entire_lineage(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with ExperimentMemory(path) as memory:
        root = memory.register_strategy(
            strategy_spec("root"),
            strategy_id="root",
            generation_method="seed",
            metadata=metadata(),
        )
        selected = memory.register_strategy(
            strategy_spec("selected", reference=55),
            strategy_id="selected",
            generation_method="mutation",
            parent_hashes=[root.behavior_hash],
            metadata=metadata(),
        )
        sibling = memory.register_strategy(
            strategy_spec("sibling", reference=60),
            strategy_id="sibling",
            generation_method="mutation",
            parent_hashes=[root.behavior_hash],
            metadata=metadata(),
        )
        final_window = window("2025-06-01", "2026-01-01")
        final_protocol = {"fees_bps": 10}
        memory.register_holdout_cohort(
            [selected.behavior_hash, sibling.behavior_hash],
            dataset={"snapshot_id": "final-snapshot-v1", "symbol": "BTCUSDT"},
            window=final_window,
            protocol=final_protocol,
        )
        claim = memory.claim_holdout(
            selected.behavior_hash,
            snapshot_id="final-snapshot-v1",
            dataset={"symbol": "BTCUSDT"},
            window=final_window,
            protocol=final_protocol,
        )
        assert claim.created is True
        assert claim.status == "claimed"
        assert memory.is_tested(
            selected.behavior_hash,
            dataset={"snapshot_id": "final-snapshot-v1", "symbol": "BTCUSDT"},
            window=window("2025-06-01", "2026-01-01"),
            protocol={"fees_bps": 10},
            phase="holdout",
        )

    # A crash/restart does not release the claim or feed its result back into
    # generation. Siblings sharing the lineage root cannot inspect it either.
    with ExperimentMemory(path) as memory:
        assert memory.holdout_claimed(sibling.behavior_hash, snapshot_id="final-snapshot-v1")
        with pytest.raises(EvaluationConflictError, match="already consumed"):
            memory.claim_holdout(
                sibling.behavior_hash,
                snapshot_id="final-snapshot-v1",
                window=window("2025-06-01", "2026-01-01"),
                protocol={"fees_bps": 10},
            )
        memory.complete_evaluation(
            claim.evaluation_key,
            outcome="reject",
            rejection_reasons=["failed_holdout"],
            metrics={"secret_holdout_sharpe": -9.0},
        )
        feedback = memory.generator_feedback()
        assert "reject" not in feedback["outcomes"]
        assert "failed_holdout" not in feedback["rejection_reasons"]
        assert all(
            "failed_holdout" not in group["rejection_reasons"]
            for section in ("generation_methods", "families", "primitives")
            for group in feedback[section].values()
        )
        assert feedback["parent_performance"][0]["child_outcomes"] == {}
        assert memory.candidate_parents() == []
        assert {
            item["behavior_hash"]
            for item in memory.candidate_parents(exclude_holdout_exposed=False)
        } == {
            root.behavior_hash,
            selected.behavior_hash,
            sibling.behavior_hash,
        }


def test_holdout_cohort_seals_snapshot_against_later_independent_roots(tmp_path):
    final_dataset = dataset("shared-final-snapshot")
    final_window = window("2025-06-01", "2026-01-01")
    final_protocol = {"fees_bps": 10, "cohort": "pre-registered"}
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        first = memory.register_strategy(
            strategy_spec("first-root"),
            strategy_id="first-root",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        later = memory.register_strategy(
            strategy_spec("later-root", reference=65),
            strategy_id="later-root",
            generation_method="grammar_sample",
            metadata=metadata(),
        )

        created = memory.register_holdout_cohort(
            [first.behavior_hash],
            dataset=final_dataset,
            window=final_window,
            protocol=final_protocol,
        )
        assert created.created is True
        with pytest.raises(EvaluationConflictError, match="sealed against new candidates"):
            memory.register_holdout_cohort(
                [later.behavior_hash],
                dataset=final_dataset,
                window=final_window,
                protocol=final_protocol,
            )
        memory.claim_holdout(
            first.behavior_hash,
            snapshot_id=final_dataset["snapshot_id"],
            dataset=final_dataset,
            window=final_window,
            protocol=final_protocol,
        )
        with pytest.raises(EvaluationConflictError, match="sealed against new candidates"):
            memory.claim_holdout(
                later.behavior_hash,
                snapshot_id=final_dataset["snapshot_id"],
                dataset=final_dataset,
                window=final_window,
                protocol=final_protocol,
            )


def test_holdout_seal_budget_defers_rapid_new_interval_sealing(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        first = memory.register_strategy(
            strategy_spec("first"),
            strategy_id="first",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        second = memory.register_strategy(
            strategy_spec("second", reference=60),
            strategy_id="second",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        sealed = memory.register_holdout_cohort(
            [first.behavior_hash],
            dataset=dataset("snapshot-one"),
            window=window("2025-06-01", "2025-09-01"),
            protocol={"fees_bps": 10},
            min_seconds_since_last_seal=3600.0,
        )
        assert sealed.created is True

        # Resuming the already sealed cohort never consumes budget.
        resumed = memory.register_holdout_cohort(
            [first.behavior_hash],
            dataset=dataset("snapshot-one"),
            window=window("2025-06-01", "2025-09-01"),
            protocol={"fees_bps": 10},
            min_seconds_since_last_seal=3600.0,
        )
        assert resumed.created is False

        # A second disjoint interval inside the spacing window is deferred,
        # and the refused attempt leaves no cohort or interval behind.
        with pytest.raises(HoldoutSealBudgetError, match="seal budget"):
            memory.register_holdout_cohort(
                [second.behavior_hash],
                dataset=dataset("snapshot-two"),
                window=window("2024-01-01", "2024-06-01"),
                protocol={"fees_bps": 10},
                min_seconds_since_last_seal=3600.0,
            )
        assert len(memory.protected_intervals(market="futures", symbol="BTCUSDT")) == 1

        # With the budget disabled the same seal is accepted.
        allowed = memory.register_holdout_cohort(
            [second.behavior_hash],
            dataset=dataset("snapshot-two"),
            window=window("2024-01-01", "2024-06-01"),
            protocol={"fees_bps": 10},
            min_seconds_since_last_seal=0,
        )
        assert allowed.created is True
        assert len(memory.protected_intervals(market="futures", symbol="BTCUSDT")) == 2


def test_holdout_seal_budget_rejects_invalid_values(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("first"),
            strategy_id="first",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        for invalid in (True, float("nan"), "3600"):
            with pytest.raises(ValueError, match="min_seconds_since_last_seal"):
                memory.register_holdout_cohort(
                    [registered.behavior_hash],
                    dataset=dataset(),
                    window=window(),
                    protocol={"fees_bps": 10},
                    min_seconds_since_last_seal=invalid,
                )


def test_protected_interval_blocks_grown_snapshot_adaptive_overlap(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("interval-owner"),
            strategy_id="interval-owner",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        first_hundred = {
            "snapshot_id": "rows-100",
            "market": "futures",
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "rows": 100,
            "content_digest": "sha256:" + "1" * 64,
        }
        protected_window = window("2026-01-01T06:40:00Z", "2026-01-01T08:15:00Z")
        first = memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset=first_hundred,
            window=protected_window,
            protocol={"market": "futures", "base_timeframe": "5m"},
        )
        resumed = memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset=first_hundred,
            window=protected_window,
            protocol={"market": "futures", "base_timeframe": "5m"},
        )

        assert first.created is True
        assert resumed.created is False
        assert resumed.member_hashes == (registered.behavior_hash,)
        intervals = memory.protected_intervals(
            market="futures",
            symbol="BTCUSDT",
        )
        assert len(intervals) == 1
        assert set(intervals[0]) == {"interval_key", "market", "symbol", "start", "end"}
        assert intervals[0]["market"] == "futures"
        assert intervals[0]["symbol"] == "BTCUSDT"
        assert intervals[0]["start"] == "2026-01-01T06:40:00.000000+00:00"
        assert intervals[0]["end"] == "2026-01-01T08:15:00.000000+00:00"

        grown_to_125 = {
            **first_hundred,
            "snapshot_id": "rows-125",
            "rows": 125,
            "content_digest": "sha256:" + "2" * 64,
        }
        with pytest.raises(EvaluationConflictError, match="permanently protected timestamps"):
            memory.assert_adaptive_window_allowed(
                dataset=grown_to_125,
                window={
                    "train": {
                        "start": "2026-01-01T00:00:00Z",
                        "end": "2026-01-01T06:10:00Z",
                        "rows": 75,
                    },
                    "validation": {
                        "start": "2026-01-01T06:15:00Z",
                        "end": "2026-01-01T08:15:00Z",
                        "rows": 25,
                    },
                },
                protocol={"market": "futures", "base_timeframe": "5m"},
            )


def test_protected_interval_blocks_cross_timeframe_adaptive_overlap(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("cross-timeframe-owner"),
            strategy_id="cross-timeframe-owner",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset={
                "snapshot_id": "five-minute-final",
                "market": "futures",
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "content_digest": "sha256:" + "3" * 64,
            },
            window=window("2026-02-01T00:00:00Z", "2026-02-02T00:00:00Z"),
            protocol={"market": "futures", "base_timeframe": "5m"},
        )

        with pytest.raises(EvaluationConflictError, match="phase=validation"):
            memory.assert_adaptive_window_allowed(
                dataset={
                    "snapshot_id": "hourly-later",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                },
                window=window("2026-01-15T00:00:00Z", "2026-02-01T12:00:00Z"),
                protocol={"market": "futures", "base_timeframe": "1h"},
                phase="validation",
            )
        with pytest.raises(EvaluationConflictError, match="overlaps an already sealed interval"):
            memory.register_holdout_cohort(
                [registered.behavior_hash],
                dataset={
                    "snapshot_id": "hourly-final-overlap",
                    "market": "futures",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                },
                window=window("2026-02-01T12:00:00Z", "2026-02-03T00:00:00Z"),
                protocol={"market": "futures", "base_timeframe": "1h"},
            )


def test_changed_evidence_for_sealed_interval_fails_closed(tmp_path):
    sealed_dataset = {
        "snapshot_id": "sealed-evidence",
        "market": "futures",
        "symbol": "BTCUSDT",
        "files": [{"file": "BTCUSDT_5m.parquet", "content_digest": "sha256:" + "4" * 64}],
    }
    protected_window = window("2026-03-01T00:00:00Z", "2026-03-02T00:00:00Z")
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("evidence-owner"),
            strategy_id="evidence-owner",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset=sealed_dataset,
            window=protected_window,
            protocol={"market": "futures"},
        )

        changed = json.loads(json.dumps(sealed_dataset))
        changed["files"][0]["content_digest"] = "sha256:" + "5" * 64
        with pytest.raises(EvaluationConflictError, match="immutable evidence changed"):
            memory.register_holdout_cohort(
                [registered.behavior_hash],
                dataset=changed,
                window=protected_window,
                protocol={"market": "futures"},
            )


def test_protected_interval_registry_migrates_and_detects_tampering(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with ExperimentMemory(path) as memory:
        registered = memory.register_strategy(
            strategy_spec("migration-owner"),
            strategy_id="migration-owner",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        final_dataset = dataset("migration-final")
        final_window = window("2026-04-01", "2026-05-01")
        final_protocol = {"fees_bps": 10}
        memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset=final_dataset,
            window=final_window,
            protocol=final_protocol,
        )
        memory.claim_holdout(
            registered.behavior_hash,
            snapshot_id=final_dataset["snapshot_id"],
            dataset=final_dataset,
            window=final_window,
            protocol=final_protocol,
        )

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE protected_intervals")
    connection.execute(
        "DELETE FROM memory_meta WHERE key = ?",
        (experiment_memory_module.PROTECTED_INTERVAL_BACKFILL_META,),
    )
    connection.commit()
    connection.close()

    with ExperimentMemory(path) as memory:
        assert memory.integrity_check(deep=True)["ok"] is True

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE protected_intervals SET evidence_hash = ?",
        ("sha256:" + "0" * 64,),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ExperimentMemoryCorruptionError, match="identity mismatch"):
        ExperimentMemory(path)


def test_pending_candidates_and_adaptive_feedback_are_bounded_and_actionable(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        parent = memory.register_strategy(
            strategy_spec("parent"),
            strategy_id="parent",
            generation_method="seed",
            metadata=metadata(),
        )
        child = memory.register_strategy(
            strategy_spec("child", reference=55),
            strategy_id="child",
            generation_method="mutation",
            parent_hashes=[parent.behavior_hash],
            metadata=metadata(),
        )
        memory.register_strategy(
            strategy_spec("duplicate", reference=55),
            strategy_id="duplicate",
            generation_method="crossover",
            parent_hashes=[parent.behavior_hash],
            metadata=metadata(),
        )

        pending = memory.pending_strategies(product="active_income", opportunity_type="swing")
        assert {item["strategy_id"] for item in pending} == {"parent", "child"}
        assert all("submitted_spec" in item for item in pending)

        memory.record_outcome(
            parent.behavior_hash,
            dataset=dataset("train-parent"),
            window=window(),
            outcome="keep",
            metrics={"sharpe": 1.2},
            phase="validation",
        )
        memory.record_outcome(
            child.behavior_hash,
            dataset=dataset("train-child"),
            window=window(),
            outcome="reject",
            rejection_reasons=["parameter_fragile"],
            metrics={"sharpe": -0.1},
            phase="validation",
        )
        assert memory.pending_strategies() == []

        feedback = memory.generator_feedback(category_limit=20)
        assert feedback["totals"] == {
            "strategies": 2,
            "identities": 3,
            "duplicate_identities": 1,
            "evaluations": 2,
            "claimed": 0,
            "completed": 2,
            "holdout_exposed": 0,
            "retired": 0,
        }
        assert feedback["outcomes"] == {"keep": 1, "reject": 1}
        assert feedback["rejection_reasons"] == {"parameter_fragile": 1}
        assert feedback["generation_methods"]["crossover"]["duplicate_rate"] == 1.0
        assert feedback["families"]["momentum"]["experiments"] == 2
        assert feedback["families"]["momentum"]["proposals"] == 3
        assert feedback["families"]["momentum"]["duplicates"] == 1
        assert feedback["primitives"]["op:gt"]["outcomes"] == {
            "keep": 1,
            "reject": 1,
        }
        assert feedback["parent_performance"] == [
            {
                "parent_hash": parent.behavior_hash,
                "children": 1,
                "child_outcomes": {"reject": 1},
                "child_rejection_reasons": {"parameter_fragile": 1},
            }
        ]
        assert [
            item["behavior_hash"] for item in memory.candidate_parents(latest_outcomes=["keep"])
        ] == [parent.behavior_hash]

        memory.retire_strategy(parent.behavior_hash, reason="search branch exhausted")
        assert memory.candidate_parents() == []
        assert {
            item["behavior_hash"] for item in memory.dedup_population(product="active_income")
        } == {parent.behavior_hash, child.behavior_hash}

        with pytest.raises(ValueError, match="cannot exceed"):
            memory.candidate_parents(limit=501)


def test_only_false_unsupported_retirement_can_be_reinstated(tmp_path):
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        registered = memory.register_strategy(
            strategy_spec("false-unsupported"),
            strategy_id="false-unsupported",
            generation_method="sample",
            metadata=metadata(),
        )
        memory.retire_strategy(
            registered.behavior_hash,
            reason='unsupported_feature_contract:{"5m":["close"]}',
        )

        assert memory.pending_strategies(product="active_income") == []
        assert memory.reinstate_false_unsupported_strategy(registered.behavior_hash) is True
        assert memory.reinstate_false_unsupported_strategy(registered.behavior_hash) is False
        assert [
            item["behavior_hash"] for item in memory.pending_strategies(product="active_income")
        ] == [registered.behavior_hash]

        memory.retire_strategy(
            registered.behavior_hash,
            reason="search branch exhausted",
        )
        with pytest.raises(
            EvaluationConflictError,
            match="only false unsupported-feature retirements",
        ):
            memory.reinstate_false_unsupported_strategy(registered.behavior_hash)


def test_concurrent_writers_preserve_every_strategy(tmp_path):
    path = tmp_path / "memory.sqlite3"

    def register(index: int) -> str:
        with ExperimentMemory(path, timeout_seconds=30) as memory:
            return memory.register_strategy(
                strategy_spec(f"candidate-{index}", reference=50 + index),
                strategy_id=f"candidate-{index}",
                generation_method="parallel_generation",
                metadata=metadata(),
            ).behavior_hash

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(register, range(20)))

    assert len(set(hashes)) == 20
    with ExperimentMemory(path) as memory:
        assert memory.generator_feedback()["totals"]["strategies"] == 20
        assert memory.integrity_check(deep=True)["ok"] is True


def test_adaptive_feedback_and_parents_are_scoped_to_current_research_engine(tmp_path):
    old_engine = "sha256:" + "1" * 64
    current_engine = "sha256:" + "2" * 64
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        old = memory.register_strategy(
            strategy_spec("old-engine"),
            strategy_id="old-engine",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        current = memory.register_strategy(
            strategy_spec("current-engine", reference=60),
            strategy_id="current-engine",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        protected = memory.register_strategy(
            strategy_spec("protected-old-engine", reference=65),
            strategy_id="protected-old-engine",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        memory.record_outcome(
            old.behavior_hash,
            dataset=dataset("old"),
            window=window(),
            protocol={"research_engine_digest": old_engine},
            phase="development",
            outcome="pre_holdout_pass",
        )
        memory.record_outcome(
            current.behavior_hash,
            dataset=dataset("current"),
            window=window(),
            protocol={"research_engine_digest": current_engine},
            phase="development",
            outcome="reject",
            rejection_reasons=("no_train_edge",),
        )
        memory.record_outcome(
            protected.behavior_hash,
            dataset=dataset("protected-old"),
            window=window(),
            protocol={"research_engine_digest": old_engine},
            phase="development",
            outcome="reject",
            rejection_reasons=("no_train_edge",),
        )
        protected_window = window()
        protected_protocol = {"research_engine_digest": old_engine}
        memory.register_holdout_cohort(
            [protected.behavior_hash],
            dataset={"snapshot_id": "protected-final"},
            window=protected_window,
            protocol=protected_protocol,
        )
        memory.claim_holdout(
            protected.behavior_hash,
            snapshot_id="protected-final",
            window=protected_window,
            protocol=protected_protocol,
        )

        feedback = memory.generator_feedback(research_engine_digest=current_engine)
        parents = memory.candidate_parents(
            latest_outcomes=("reject", "pre_holdout_pass"),
            research_engine_digest=current_engine,
        )
        pending = memory.pending_strategies(
            research_engine_digest=current_engine,
            product="active_income",
            opportunity_type="swing",
        )

    assert feedback["research_engine_digest"] == current_engine
    assert feedback["adaptive_evaluations"] == 1
    assert feedback["outcomes"] == {"reject": 1}
    assert [item["behavior_hash"] for item in parents] == [current.behavior_hash]
    assert [item["behavior_hash"] for item in pending] == [old.behavior_hash]
    assert pending[0]["revalidation_required"] is True


def test_compaction_preserves_exact_evidence_engine_scope_and_holdout_claims(tmp_path):
    path = tmp_path / "memory.sqlite3"
    engine = "sha256:" + "7" * 64
    large = "repeated-evidence-" * 80_000
    development_dataset = {"snapshot_id": "development-v1", "manifest": large}
    development_window = {"start": "2024-01-01", "end": "2025-01-01"}
    development_protocol = {"research_engine_digest": engine, "version": 3}

    with ExperimentMemory(path) as memory:
        registered = memory.register_strategy(
            strategy_spec("compact-me"),
            strategy_id="compact-me",
            generation_method="grammar_sample",
            metadata=metadata(),
        )
        memory.record_outcome(
            registered.behavior_hash,
            dataset=development_dataset,
            window=development_window,
            protocol=development_protocol,
            phase="development",
            outcome="reject",
            rejection_reasons=("parameter_fragile",),
            metrics={"diagnostic": large},
            details={"trace": large},
        )
        holdout_window = window("2025-01-01", "2026-01-01")
        holdout_protocol = {"research_engine_digest": engine}
        memory.register_holdout_cohort(
            [registered.behavior_hash],
            dataset={"snapshot_id": "holdout-v1"},
            window=holdout_window,
            protocol=holdout_protocol,
        )
        holdout = memory.claim_holdout(
            registered.behavior_hash,
            snapshot_id="holdout-v1",
            window=holdout_window,
            protocol=holdout_protocol,
        )
        memory.complete_evaluation(
            holdout.evaluation_key,
            outcome="reject",
            rejection_reasons=("failed_holdout",),
            metrics={"protected": large},
        )
        before = path.stat().st_size

        report = memory.compact_storage(maximum_rows=100)

        assert report["ok"] is True
        assert report["rows_compacted"] == 4
        assert report["after_bytes"] < before
        assert memory.integrity_check(deep=True)["ok"] is True
        assert memory.is_tested(
            registered.behavior_hash,
            dataset=development_dataset,
            window=development_window,
            protocol=development_protocol,
            phase="development",
        )
        evidence = memory.list_evaluations(
            behavior_hash=registered.behavior_hash,
            phase="development",
        )[0]
        assert evidence["metrics"] == {"diagnostic": large}
        assert evidence["details"] == {"trace": large}
        assert memory.holdout_claimed(
            registered.behavior_hash,
            snapshot_id="holdout-v1",
        )
        assert memory.pending_strategies(research_engine_digest=engine) == []
        assert (
            "failed_holdout"
            not in memory.generator_feedback(research_engine_digest=engine)["rejection_reasons"]
        )

    with ExperimentMemory(path) as reopened:
        assert reopened.integrity_check(deep=True)["ok"] is True
        assert reopened.get_strategy(registered.behavior_hash)["strategy_id"] == "compact-me"


def test_legacy_engine_scope_backfill_runs_only_once(monkeypatch, tmp_path):
    path = tmp_path / "memory.sqlite3"
    with ExperimentMemory(path) as memory:
        registered = memory.register_strategy(
            strategy_spec("legacy-no-engine"),
            strategy_id="legacy-no-engine",
            generation_method="seed",
            metadata=metadata(),
        )
        memory.record_outcome(
            registered.behavior_hash,
            dataset=dataset("legacy"),
            window=window(),
            protocol={"version": 1},
            outcome="keep",
        )
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM memory_meta WHERE key = 'evaluation_engine_scopes_backfill_v1'")
    connection.commit()
    connection.close()

    original = experiment_memory_module._stored_json
    decoded_protocols = 0

    def count_protocol_decodes(value, *, label):
        nonlocal decoded_protocols
        if label == "evaluation protocol JSON":
            decoded_protocols += 1
        return original(value, label=label)

    monkeypatch.setattr(experiment_memory_module, "_stored_json", count_protocol_decodes)
    with ExperimentMemory(path, deep_on_open=False):
        pass
    assert decoded_protocols == 1

    decoded_protocols = 0
    with ExperimentMemory(path, deep_on_open=False):
        pass
    assert decoded_protocols == 0


def test_corruption_fails_closed_and_backup_is_reopenable(tmp_path):
    path = tmp_path / "memory.sqlite3"
    backup = tmp_path / "backups" / "memory.sqlite3"
    memory = ExperimentMemory(path)
    registered = memory.register_strategy(
        strategy_spec("candidate"),
        strategy_id="candidate",
        generation_method="seed",
        metadata=metadata(),
    )
    assert memory.backup_to(backup) == backup
    memory.close()
    assert memory.closed is True
    with pytest.raises(ExperimentMemoryClosedError):
        memory.get_strategy(registered.behavior_hash)

    with ExperimentMemory(backup) as restored:
        assert restored.get_strategy(registered.behavior_hash)["strategy_id"] == "candidate"

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE strategies SET canonical_spec_json = ? WHERE behavior_hash = ?",
        (json.dumps({"entry": "tampered"}), registered.behavior_hash),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ExperimentMemoryCorruptionError, match="hash mismatch"):
        ExperimentMemory(path)


def test_rejects_non_database_and_symlink_files(tmp_path):
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"this is not sqlite")
    with pytest.raises(ExperimentMemoryCorruptionError, match="cannot open"):
        ExperimentMemory(corrupt)

    unrelated = tmp_path / "unrelated.sqlite3"
    connection = sqlite3.connect(unrelated)
    connection.execute("CREATE TABLE unrelated(value TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(ExperimentMemoryCorruptionError, match="unrelated SQLite"):
        ExperimentMemory(unrelated)

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"external")
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(target)
    with pytest.raises(ExperimentMemoryCorruptionError, match="symlink"):
        ExperimentMemory(link)
