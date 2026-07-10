from __future__ import annotations

import dataclasses
import random

import pytest

from research_exploration.strategy_grammar import (
    GrammarLimits,
    SearchSpace,
    build_fresh_hypothesis,
    crossover_hypotheses,
    mutate_hypothesis,
    structural_similarity,
    structural_tokens,
    validate_hypothesis_against_space,
)


def _space(*, product: str = "active_income") -> SearchSpace:
    if product == "btc_accumulation":
        return SearchSpace(
            name="btc_position",
            product=product,
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
        )
    return SearchSpace(
        name="active_day",
        product=product,
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


def test_fresh_generation_is_typed_bounded_and_diverse() -> None:
    space = _space()
    ideas = [build_fresh_hypothesis(space, rng=random.Random(seed)) for seed in range(30)]

    assert all(not validate_hypothesis_against_space(idea.hypothesis, space) for idea in ideas)
    assert all(idea.hypothesis.regime for idea in ideas)
    assert all(idea.hypothesis.setup for idea in ideas)
    assert all(idea.hypothesis.trigger for idea in ideas)
    assert all(
        len(idea.hypothesis.all_predicates()) <= GrammarLimits().max_total_predicates
        for idea in ideas
    )
    assert len({frozenset(structural_tokens(idea.hypothesis, include_values=True)) for idea in ideas}) >= 25
    assert {idea.hypothesis.direction for idea in ideas} == {"long", "short"}


def test_btc_generation_only_emits_conservative_spot_dodge_signals() -> None:
    space = _space(product="btc_accumulation")
    ideas = [build_fresh_hypothesis(space, rng=random.Random(seed)) for seed in range(10)]

    assert {idea.hypothesis.direction for idea in ideas} == {"short"}
    assert all(idea.hypothesis.risk.max_position_fraction <= 0.25 for idea in ideas)
    assert all(idea.hypothesis.risk.risk_per_trade <= 0.003 for idea in ideas)


def test_recursive_mutation_changes_behavior_and_retains_parent() -> None:
    space = _space()
    parent = build_fresh_hypothesis(space, rng=random.Random(10)).hypothesis
    parent_tokens = structural_tokens(parent, include_values=True)

    descendants = [
        mutate_hypothesis(
            parent,
            space,
            parent_hash="sha256:parent",
            rng=random.Random(seed),
        )
        for seed in range(20, 40)
    ]

    assert any(structural_tokens(item.hypothesis, include_values=True) != parent_tokens for item in descendants)
    assert all(item.parent_hashes == ("sha256:parent",) for item in descendants)
    assert all(item.generation_method == "recursive_mutation" for item in descendants)


def test_recursive_mutation_records_the_development_failure_that_drove_it() -> None:
    space = _space()
    parent = build_fresh_hypothesis(space, rng=random.Random(10)).hypothesis

    child = mutate_hypothesis(
        parent,
        space,
        parent_hash="sha256:parent",
        rng=random.Random(22),
        failure_reasons=("insufficient_train_trades", "insufficient_train_trades"),
    )

    assert child.adaptation_reasons == ("insufficient_train_trades",)
    assert not validate_hypothesis_against_space(child.hypothesis, space)


def test_crossover_recombines_compatible_parents() -> None:
    space = _space()
    first = build_fresh_hypothesis(space, rng=random.Random(2)).hypothesis
    # Crossover is intentionally same-direction so select a deterministic mate.
    second = next(
        build_fresh_hypothesis(space, rng=random.Random(seed)).hypothesis
        for seed in range(3, 100)
        if build_fresh_hypothesis(space, rng=random.Random(seed)).hypothesis.direction
        == first.direction
    )

    child = crossover_hypotheses(
        first,
        second,
        space,
        parent_hashes=("sha256:first", "sha256:second"),
        rng=random.Random(99),
    )

    assert child.generation_method == "crossover"
    assert child.parent_hashes == ("sha256:first", "sha256:second")
    assert not validate_hypothesis_against_space(child.hypothesis, space)
    assert child.hypothesis.direction == first.direction


def test_dynamic_inventory_adds_unseen_feature_atoms() -> None:
    space = _space()
    available = {
        "1h": {"open", "high", "low", "close", "natr_14", "ema_50", "ema_200", "mfi_14"},
        "15m": {
            "open",
            "high",
            "low",
            "close",
            "natr_14",
            "rsi_14",
            "custom_placeholder",
            "mfi_14",
        },
        "5m": {
            "open",
            "high",
            "low",
            "close",
            "natr_14",
            "mom_10",
            "volume_z_20",
            "mfi_14",
        },
    }
    observed = set()
    for seed in range(100):
        idea = build_fresh_hypothesis(
            space,
            rng=random.Random(seed),
            available_features=available,
            motif="hybrid",
        )
        observed.update(predicate.feature for predicate in idea.hypothesis.all_predicates())
    assert "mfi_14" in observed
    assert "custom_placeholder" not in observed


def test_strict_validation_rejects_unsafe_or_incompatible_specs() -> None:
    space = _space()
    hypothesis = build_fresh_hypothesis(space, rng=random.Random(1)).hypothesis

    with pytest.raises(ValueError, match="direction"):
        dataclasses.replace(hypothesis, direction="both")
    assert "trigger_predicate_count" in validate_hypothesis_against_space(
        dataclasses.replace(hypothesis, trigger=[]),
        space,
    )
    assert "risk_per_trade_out_of_bounds" in validate_hypothesis_against_space(
        dataclasses.replace(
            hypothesis,
            risk=dataclasses.replace(hypothesis.risk, risk_per_trade=0.5),
        ),
        space,
    )
    with pytest.raises(ValueError, match="take_profit"):
        dataclasses.replace(hypothesis.exit, take_profit=-1)


def test_search_space_rejects_invalid_budget_bounds() -> None:
    with pytest.raises(ValueError, match="take_profit_range"):
        dataclasses.replace(_space(), take_profit_range=(0.02, 0.01))


def test_structural_similarity_ignores_prose_but_not_entry_logic() -> None:
    space = _space()
    first = build_fresh_hypothesis(space, rng=random.Random(42)).hypothesis
    renamed = dataclasses.replace(first, id="renamed", idea="different prose", tags=["other"])
    changed = dataclasses.replace(first, trigger=first.trigger[:-1])

    assert structural_similarity(first, renamed) == 1.0
    assert structural_similarity(first, changed) < 1.0
