from collections import Counter

import pytest

from src.data.database import PlatformDatabase
from src.domain._codec import canonical_hash
from src.domain.strategies import StrategySourceType
from src.research.executors import ProviderExecutorRegistry, UnsupportedExecutorError
from src.research.generation import (
    CAMPAIGNS,
    GenerationAllocator,
    GenerationError,
    GenerationFeedback,
    HypothesisGenerator,
    SqlGenerationFeedbackStore,
    build_hypothesis,
)
from src.research.providers import provider_candidate


@pytest.mark.parametrize(
    "family,direction",
    [
        (family, "signed")
        for family in (
            "cross_sectional",
            "relative_value",
            "microstructure",
            "machine_learning",
            "meta_strategy",
            "execution",
            "market_making",
        )
    ]
    + [("time_series", "market_neutral"), ("time_series", "hedged")],
)
def test_scalar_dsl_cannot_bypass_specialised_execution_contracts(family, direction):
    identity = canonical_hash({"family": family, "direction": direction})
    candidate = provider_candidate(
        identity="scalar-capability-check",
        version="1",
        family=family,
        product="active_income",
        thesis_id=identity,
        lineage_id=identity,
        provider="fixture",
        source_type=StrategySourceType.GENERATED_DSL,
        source_payload={
            "kind": "typed_rule",
            "rule": {
                "feature": "trend",
                "operator": "gt",
                "threshold": 0,
                "direction": direction,
            },
        },
        dataset_snapshot_hashes=(identity,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    with pytest.raises(UnsupportedExecutorError, match="directional DSL") as error:
        ProviderExecutorRegistry.default().execute(candidate, {})
    assert error.value.outcome_category.value == "unsupported"


def test_direct_relative_value_submission_cannot_bypass_capability_gate():
    identity = canonical_hash("fixture")
    candidate = provider_candidate(
        identity="beta_neutral_spreads",
        version="1",
        family="relative_value",
        product="active_income",
        thesis_id=identity,
        lineage_id=identity,
        provider="agent",
        source_type=StrategySourceType.RELATIVE_VALUE,
        source_payload={"semantic_strategy": "beta_neutral_spreads"},
        dataset_snapshot_hashes=(identity,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    with pytest.raises(UnsupportedExecutorError, match="frozen hedge estimates") as error:
        ProviderExecutorRegistry.default().execute(candidate, {})
    assert error.value.outcome_category.value == "unsupported"


@pytest.mark.parametrize(
    "name", ["basis_convergence", "pairs_mean_reversion", "funding_carry", "event_microstructure"]
)
def test_unsupported_campaign_cannot_be_enabled_by_explicit_selection(name):
    campaign = next(item for item in CAMPAIGNS if item.name == name)
    arguments = {
        "dataset_snapshot_hashes": (canonical_hash("fixture"),),
        "submitted_at": "2026-09-07T00:00:00+00:00",
    }
    generator = HypothesisGenerator(
        product="active_income", instrument_universe=("BTCUSDT", "ETHUSDT")
    )
    assert generator.generate(**arguments, total_budget=1, campaigns=(campaign,)) == ()
    with pytest.raises(GenerationError, match="unsupported campaign"):
        build_hypothesis(
            campaign, variant=0, instrument_universe=("BTCUSDT", "ETHUSDT"), **arguments
        )


def test_single_slot_budget_rotates_after_allocator_restart(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'rotation.sqlite3'}")
    database.create_schema()
    store = SqlGenerationFeedbackStore(database.engine)
    selected = []
    for iteration in range(9):
        allocations = GenerationAllocator().allocate(
            CAMPAIGNS[:3], total_budget=1, feedback=store.load()
        )
        winner = next(item.campaign for item in allocations if item.trials)
        selected.append(winner)
        store.append(
            GenerationFeedback(
                winner,
                "generated",
                "2026-09-07T00:00:00+00:00",
                candidate_id=f"candidate-{iteration}",
            )
        )
    assert Counter(selected) == {campaign.name: 3 for campaign in CAMPAIGNS[:3]}
    database.dispose()


def test_registration_and_missing_data_do_not_change_alpha_score():
    observed = "2026-09-07T00:00:00+00:00"
    neutral = [
        GenerationFeedback("campaign", outcome, observed, candidate_id="candidate")
        for outcome in ("generated", "accepted", "data_unavailable", "duplicate_exact")
    ]
    assert GenerationAllocator._score(neutral) == 1
    economic = GenerationFeedback(
        "campaign",
        "accepted",
        observed,
        candidate_id="candidate",
        metadata={
            "stage": "development",
            "economic_engine": "economic-engine/v4",
            "net_return": 0.01,
        },
    )
    assert GenerationAllocator._score([*neutral, economic, economic]) == 2
