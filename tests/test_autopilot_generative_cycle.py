from __future__ import annotations

import dataclasses
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_exploration.hypothesis_schema import ExitRule, Hypothesis, Predicate, RiskRule
from research_exploration.strategy_grammar import SearchSpace, build_fresh_hypothesis
from src.autopilot.experiment_memory import ExperimentMemory, canonical_strategy_hash
from src.autopilot.research_cycle import (
    ResearchScenario,
    _dataset_snapshot,
    _load_generated_scenarios,
    _retire_unsupported_generated_hypotheses,
    run_validation_scenario,
)
from src.autopilot.research_factory import (
    BATCH_SCHEMA,
    strategy_behavior_spec,
)


def _active_space() -> SearchSpace:
    return SearchSpace(
        name="active_income_day",
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
    )


def _generated_batch(hypothesis: Hypothesis, space: SearchSpace) -> dict:
    behavior_hash = canonical_strategy_hash(strategy_behavior_spec(hypothesis, space))
    hypothesis = dataclasses.replace(
        hypothesis,
        id=f"GEN_{space.name.upper()}_{behavior_hash[7:23]}",
    )
    return {
        "ok": True,
        "schema": BATCH_SCHEMA,
        "generated_at": "2026-07-10T00:00:00+00:00",
        "research_only": True,
        "executable": False,
        "paper_trade_allowed": False,
        "promotion_allowed": False,
        "live_allowed": False,
        "requires_full_validation_before_export": True,
        "summary": {"cumulative_trials": 123},
        "generation_metadata": [
            {
                "id": hypothesis.id,
                "strategy_hash": canonical_strategy_hash(
                    strategy_behavior_spec(hypothesis, space)
                ),
                "search_space": space.name,
                "product": space.product,
                "market": space.market,
                "pnl_unit": space.pnl_unit,
                "opportunity_type": space.opportunity_type,
                "base_timeframe": space.base_timeframe,
                "generation_method": "grammar_sample",
                "lineage_depth": 0,
            }
        ],
        "hypotheses": [hypothesis.to_dict()],
    }


def test_generated_batch_loader_requires_canonical_digest_and_safety_flags(tmp_path):
    space = _active_space()
    idea = build_fresh_hypothesis(space, rng=random.Random(4)).hypothesis
    payload = _generated_batch(idea, space)
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scenarios, hypotheses, metadata, summary = _load_generated_scenarios(
        path,
        factory_config_path=Path("config/research_factory.json"),
    )

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.candidate_set == "generated"
    assert scenario.product == "active_income"
    assert len(hypotheses[scenario.name]) == 1
    assert next(iter(metadata[scenario.name].values()))["cumulative_trials"] == 123
    assert summary["skipped"] == 0

    payload["generation_metadata"][0]["strategy_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    scenarios, _, _, summary = _load_generated_scenarios(
        path,
        factory_config_path=Path("config/research_factory.json"),
    )
    assert scenarios == ()
    assert summary["skipped"] == 1
    assert "canonical behavior" in summary["skipped_errors"][0]


def _rising_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="MEMORY_TEST",
        family="generated_test",
        idea="test",
        market_logic="test",
        direction="long",
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[Predicate("5m", "close", "gt", reference=0.0)],
        setup=[Predicate("5m", "close", "gt", reference=0.0)],
        trigger=[Predicate("5m", "close", "rising", lookback=1)],
        exit=ExitRule(0.02, 0.02, 12),
        risk=RiskRule(max_trades_per_day=6),
    )


def _sawtooth_frame(rows: int = 3000) -> pd.DataFrame:
    close = [100.0]
    for index in range(rows - 1):
        close.append(close[-1] * (1 + (0.015 if index % 2 == 0 else -0.005)))
    values = np.asarray(close)
    opened = np.concatenate([[values[0]], values[:-1]])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC"),
            "tf_5m_open": opened,
            "tf_5m_high": np.maximum(opened, values) * 1.001,
            "tf_5m_low": np.minimum(opened, values) * 0.999,
            "tf_5m_close": values,
        }
    )


def _memory_scenario() -> ResearchScenario:
    return ResearchScenario(
        name="generated_memory_test",
        product="active_income",
        base_tf="5m",
        pnl_unit="usdt",
        market="futures",
        position=False,
        start="2024-01-01",
        opportunity_type="day_trading",
        candidate_set="generated",
    )


def _register(memory: ExperimentMemory, hypothesis: Hypothesis) -> str:
    spec = {
        **hypothesis.to_dict(),
        "_product": "active_income",
        "_market": "futures",
        "_pnl_unit": "usdt",
        "_opportunity_type": "day_trading",
        "_search_space": "generated_memory_test",
    }
    registration = memory.register_strategy(
        spec,
        strategy_id=hypothesis.id,
        generation_method="grammar_sample",
        metadata={
            "product": "active_income",
            "opportunity_type": "day_trading",
            "search_space": "generated_memory_test",
        },
    )
    return registration.behavior_hash


def test_validation_persists_development_and_hides_holdout_from_feedback(monkeypatch, tmp_path):
    hypothesis = _rising_hypothesis()
    scenario = _memory_scenario()
    frame = _sawtooth_frame()
    monkeypatch.setattr(
        "src.autopilot.research_cycle._missing_columns_for_hypothesis",
        lambda _hypothesis, _directory: {},
    )
    monkeypatch.setattr(
        "src.autopilot.research_cycle.build_aligned_frame",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        "src.autopilot.research_cycle._dataset_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_id": "snapshot-v1",
            "market": "futures",
            "rows": len(frame),
        },
    )

    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        behavior_hash = _register(memory, hypothesis)
        metadata = {hypothesis.id: {"strategy_hash": behavior_hash}}
        first = run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            selection={"available": 50, "selected": 1, "cumulative_trials": 50},
            hypothesis_metadata=metadata,
            log_path=tmp_path / "experiments.jsonl",
            experiment_memory=memory,
        )
        second = run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            selection={"available": 50, "selected": 1, "cumulative_trials": 50},
            hypothesis_metadata=metadata,
            log_path=tmp_path / "experiments.jsonl",
            experiment_memory=memory,
        )
        feedback = memory.generator_feedback()

    assert first["keepers"] == 1
    assert first["trial_count"] == 50
    assert first["dataset_snapshot_id"] == "snapshot-v1"
    assert second["reason"] == "already_evaluated_on_snapshot"
    assert feedback["outcomes"] == {"pre_holdout_pass": 1}
    assert "keep" not in feedback["outcomes"]
    assert "failed_holdout" not in json.dumps(feedback).lower()


def test_crash_after_holdout_claim_cannot_reuse_protected_snapshot(monkeypatch, tmp_path):
    hypothesis = _rising_hypothesis()
    scenario = _memory_scenario()
    frame = _sawtooth_frame(rows=200)
    monkeypatch.setattr(
        "src.autopilot.research_cycle._missing_columns_for_hypothesis",
        lambda _hypothesis, _directory: {},
    )
    monkeypatch.setattr(
        "src.autopilot.research_cycle.build_aligned_frame",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        "src.autopilot.research_cycle._dataset_snapshot",
        lambda *_args, **_kwargs: {"snapshot_id": "crash-snapshot", "rows": len(frame)},
    )

    def crash_after_claim(_frame, hypotheses, _config, *, before_holdout, **_kwargs):
        partial = {
            "hypothesis_id": hypotheses[0].id,
            "family": hypotheses[0].family,
            "direction": hypotheses[0].direction,
            "verdict": None,
            "reasons": [],
            "train": {"trades": 40, "total_return": 0.1, "win_rate": 0.6, "sharpe": 1.0},
            "validation": {"trades": 20, "total_return": 0.04, "win_rate": 0.55, "sharpe": 0.8},
            "holdout": None,
            "oos": {"pass_rate": 0.75},
            "sensitivity": {"pass_fraction": 0.75},
            "dsr_deflated": 0.7,
            "splits": {"holdout": {"start": "2024-01-02", "end": "2024-01-03", "rows": 40}},
        }
        assert before_holdout(hypotheses[0], partial) is True
        raise RuntimeError("simulated process death")

    monkeypatch.setattr("src.autopilot.research_cycle.validate_batch", crash_after_claim)
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        behavior_hash = _register(memory, hypothesis)
        metadata = {hypothesis.id: {"strategy_hash": behavior_hash}}
        with pytest.raises(RuntimeError, match="simulated process death"):
            run_validation_scenario(
                scenario,
                hypotheses=[hypothesis],
                hypothesis_metadata=metadata,
                log_path=tmp_path / "experiments.jsonl",
                experiment_memory=memory,
            )
        assert memory.holdout_claimed(behavior_hash, snapshot_id="crash-snapshot") is True

        resumed = run_validation_scenario(
            scenario,
            hypotheses=[hypothesis],
            hypothesis_metadata=metadata,
            log_path=tmp_path / "experiments.jsonl",
            experiment_memory=memory,
        )

    assert resumed["reason"] == "already_evaluated_on_snapshot"


def test_unsupported_generated_candidate_is_retired_from_restart_queue(tmp_path):
    hypothesis = _rising_hypothesis()
    with ExperimentMemory(tmp_path / "memory.sqlite3") as memory:
        behavior_hash = _register(memory, hypothesis)
        retired = _retire_unsupported_generated_hypotheses(
            [{"id": hypothesis.id, "missing_columns": {"5m": ["rsi_14"]}}],
            hypothesis_metadata={hypothesis.id: {"strategy_hash": behavior_hash}},
            experiment_memory=memory,
        )

        assert retired == [hypothesis.id]
        assert memory.get_strategy(behavior_hash)["retirement_reason"].startswith(
            "unsupported_feature_contract:"
        )
        assert memory.pending_strategies(product="active_income") == []


def test_dataset_snapshot_uses_content_not_path_or_mtime(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "moved"
    first_dir.mkdir()
    second_dir.mkdir()
    source = first_dir / "BTCUSDT_5m_all_indicators.parquet"
    moved = second_dir / source.name
    indicator = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
        }
    )
    indicator.to_parquet(source, index=False)
    moved.write_bytes(source.read_bytes())
    frame = indicator.rename(
        columns={name: f"tf_5m_{name}" for name in ("open", "high", "low", "close")}
    )
    hypothesis = _rising_hypothesis()
    scenario = _memory_scenario()

    first = _dataset_snapshot(scenario, [hypothesis], frame=frame, indicator_dir=first_dir)
    os.utime(source, None)
    touched = _dataset_snapshot(scenario, [hypothesis], frame=frame, indicator_dir=first_dir)
    relocated = _dataset_snapshot(scenario, [hypothesis], frame=frame, indicator_dir=second_dir)
    changed_frame = indicator.copy()
    changed_frame.loc[1, "close"] = 9.5
    changed_frame.to_parquet(source, index=False)
    changed = _dataset_snapshot(scenario, [hypothesis], frame=frame, indicator_dir=first_dir)

    assert first["snapshot_id"] == touched["snapshot_id"] == relocated["snapshot_id"]
    assert first["snapshot_id"] != changed["snapshot_id"]
