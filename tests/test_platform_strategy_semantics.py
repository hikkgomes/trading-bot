import pytest

from src.domain._codec import canonical_hash, to_primitive
from src.domain.strategies import StrategySourceType
from src.research.accounting import FuturesResearchAccounting
from src.research.executors import execute_machine_learning
from src.research.generation import CAMPAIGNS, build_hypothesis
from src.research.position_execution import managed_futures_events
from src.research.providers import provider_candidate
from src.services.artefact_dispatcher import ArtefactDispatcher
from src.strategies.behaviour import BtcTargetAllocationBehaviour, TypedRuleBehaviour
from src.strategies.frozen_model import FrozenLinearModel
from src.strategies.position_state import PositionRules, position_path


@pytest.mark.parametrize(
    "operator,expected",
    [
        ("gt", (-1, 0, 0, 0, 1)),
        ("ge", (-1, -1, 0, 1, 1)),
        ("lt", (1, 0, 0, 0, -1)),
        ("le", (1, 1, 0, -1, -1)),
    ],
)
def test_signed_rule_comparison_preserves_trend_or_reversion_in_research_and_paper(
    operator, expected
):
    rule = {"feature": "score", "operator": operator, "threshold": 1.0, "direction": "signed"}
    rows = [{"score": value} for value in (-2, -1, 0, 1, 2)]
    behaviour = TypedRuleBehaviour(rule)
    assert behaviour.generate_signals(rows) == expected
    assert behaviour.behaviour_hash != canonical_hash(
        {"contract_version": "typed_rule/v1", "rule": rule}
    )
    with pytest.raises(ValueError, match="unsupported typed-rule"):
        TypedRuleBehaviour(rule, contract_version="typed_rule/v1")
    candidate = provider_candidate(
        identity="signed-comparison",
        version="1",
        family="mean_reversion",
        product="active_income",
        thesis_id=canonical_hash("thesis"),
        lineage_id=canonical_hash("lineage"),
        provider="fixture",
        source_type=StrategySourceType.GENERATED_DSL,
        source_payload={"kind": "typed_rule", "rule": rule},
        dataset_snapshot_hashes=(canonical_hash("data"),),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    artefact = {
        "definition": to_primitive(candidate.definition),
        "definition_hash": candidate.definition.definition_hash,
        "position_limits": {"maximum_position": 0.1},
    }
    artefact["artefact_hash"] = canonical_hash(artefact)
    directions = {"long": 1, "short": -1, "flat": 0}
    assert (
        tuple(
            directions[ArtefactDispatcher.default().evaluate(row, artefact)["direction"]]
            for row in rows
        )
        == expected
    )


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("btc_trend_breakout", 0.01, 1),
        ("btc_pullback_reversion", -3, 1),
        ("btc_risk_off_reentry", 0.1, -1),
        ("futures_mean_reversion", -3, 1),
        ("futures_mean_reversion", 3, -1),
        ("futures_trend_following", 0.01, 1),
        ("futures_trend_following", -0.01, -1),
    ],
)
def test_generated_direction_matches_declared_economic_condition(name, value, expected):
    campaign = next(item for item in CAMPAIGNS if item.name == name)
    hypothesis = build_hypothesis(
        campaign,
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=("sha256:" + "a" * 64,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    definition = hypothesis.candidate.definition
    behaviour = (
        BtcTargetAllocationBehaviour.from_definition(definition)
        if campaign.product == "btc_accumulation"
        else TypedRuleBehaviour.from_definition(definition)
    )
    assert behaviour.signal({campaign.feature: value}) == expected


def test_directional_rule_does_not_invent_expected_return_or_confidence():
    campaign = next(item for item in CAMPAIGNS if item.name == "futures_trend_following")
    candidate = build_hypothesis(
        campaign,
        variant=0,
        instrument_universe=("BTCUSDT",),
        dataset_snapshot_hashes=("sha256:" + "a" * 64,),
        submitted_at="2026-09-07T00:00:00+00:00",
    ).candidate
    artefact = {
        "definition": to_primitive(candidate.definition),
        "definition_hash": candidate.definition.definition_hash,
        "position_limits": {"maximum_position": 0.1},
    }
    artefact["artefact_hash"] = canonical_hash(artefact)
    result = ArtefactDispatcher.default().evaluate({"trend": 0.01}, artefact)
    assert result["direction"] == "long"
    assert result["expected_return"] is None
    assert result["confidence"] is None


def test_frozen_model_uses_each_timestamp_vector():
    snapshot = "sha256:" + "a" * 64
    model = FrozenLinearModel(snapshot, snapshot, ("x",), (1.0,), 0.0)
    candidate = provider_candidate(
        identity="linear-test",
        version="1",
        family="ml",
        product="active_income",
        thesis_id=snapshot,
        lineage_id=snapshot,
        provider="machine_learning",
        source_type=StrategySourceType.MACHINE_LEARNING,
        source_payload={"feature_names": ["x"]},
        dataset_snapshot_hashes=(snapshot,),
        submitted_at="2026-09-07T00:00:00+00:00",
    )
    context = {
        "dataset_snapshot_ids": [snapshot],
        "frozen_model": model,
        "model_artefact_hash": snapshot,
        "feature_manifest_hash": snapshot,
        "feature_vector": {"x": 1.0},
        "feature_rows": [{"x": -1.0}, {"x": 1.0}, {"x": 1.0}],
        "returns": [0.1, 0.2],
        "fee_bps": 0,
        "slippage_bps": 0,
        "bootstrap_iterations": 10,
    }
    result = execute_machine_learning(candidate, context)
    assert result.evidence["return_ledger"]["net_returns"] == pytest.approx([-0.1, 0.2])
    changed = execute_machine_learning(
        candidate, {**context, "feature_rows": [{"x": -1.0}, {"x": 1.0}, {"x": -1.0}]}
    )
    assert (
        changed.evidence["return_ledger"]["net_returns"]
        == result.evidence["return_ledger"]["net_returns"]
    )


def test_entry_pulse_holds_until_horizon_and_ledger_uses_actual_exit():
    rows = [{"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i} for i in range(4)]
    signals = [1, 0, 0, 0]
    path = position_path(rows, signals, PositionRules(0.03, 0.06, 3))
    assert path.decisions == (1, 1, 1, 0)
    assert [(action.bar_index, action.price) for action in path.actions] == [(1, 101), (3, 103)]
    events = managed_futures_events(
        rows,
        [f"2026-09-07T00:0{i}:00+00:00" for i in range(4)],
        path.actions,
        notional=101,
        fee_rate=0,
        slippage_rate=0,
    )
    report = FuturesResearchAccounting().evaluate(initial_cash=1000, events=events)
    assert report.final_equity == pytest.approx(1002)
    assert report.fills == 2


def test_gap_stop_uses_open_and_future_bars_do_not_change_prior_decisions():
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(4)]
    signals = [1, 0, 0, 0]
    rules = PositionRules(0.03, 0.06, 10)
    before = position_path(rows, signals, rules)
    rows[2] = {"open": 90, "high": 92, "low": 89, "close": 91}
    after = position_path(rows, signals, rules)
    assert after.decisions[:2] == before.decisions[:2]
    assert after.actions[-1].reason == "stop"
    assert after.actions[-1].price == 90


def test_strategy_state_uses_fills_and_preserves_campaign_age_after_partial_reduction(tmp_path):
    from src.data.database import PlatformDatabase
    from src.domain.forecasts import AlphaForecast, ForecastDirection
    from src.domain.orders import Fill, OrderSide
    from src.execution.position_manager import PositionManager, SqlPositionStore
    from src.services.portfolio_service import SqlPortfolioRepository

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'state.sqlite3'}")
    database.create_schema()
    repository = SqlPortfolioRepository(database.engine)
    repository.save_forecast(
        AlphaForecast(
            "version",
            "active_income",
            "BTCUSDT",
            ForecastDirection.LONG,
            1,
            None,
            None,
            3600,
            "2026-09-07T00:00:00+00:00",
            "2026-09-07T01:00:00+00:00",
            0.1,
            0.1,
            metadata={
                "assignment_id": "assignment",
                "position_state": {"direction": 1, "age": 40},
                "entry_signal": 1,
            },
        )
    )
    arguments = {
        "assignment_id": "assignment",
        "version_id": "version",
        "portfolio_id": "portfolio",
        "instrument_id": "BTCUSDT",
        "before": "2026-09-07T03:00:00+00:00",
        "interval_seconds": 3600,
    }
    assert repository.prior_strategy_state(**arguments)["state"]["direction"] == 0
    manager = PositionManager(SqlPositionStore(database.engine))
    for identity, side, quantity, at in (
        ("entry", OrderSide.BUY, 2, "2026-09-07T00:00:00.001+00:00"),
        ("reduce", OrderSide.SELL, 1, "2026-09-07T02:00:00+00:00"),
        ("exit", OrderSide.SELL, 1, "2026-09-07T04:00:00+00:00"),
    ):
        manager.apply_fill(
            "portfolio",
            Fill(
                identity,
                identity,
                "BTCUSDT",
                side,
                quantity,
                102,
                0,
                at,
                metadata={"reference_price": 101},
            ),
            contributions={"version": 1},
            assignment_id="assignment",
        )
        if identity == "reduce":
            manager.reconcile_position(
                portfolio_id="portfolio",
                instrument_id="BTCUSDT",
                quantity=1,
                average_entry_price=102,
                updated_at="2026-09-07T02:00:01+00:00",
            )
    recovered = SqlPortfolioRepository(database.engine).prior_strategy_state(**arguments)
    assert recovered["state"] == {"direction": 1, "entry_price": 101, "age": 2}
    assert recovered["entry_signal"] == 0
    assert recovered["position_event_id"]
    arguments["before"] = "2026-09-07T05:00:00+00:00"
    assert repository.prior_strategy_state(**arguments)["state"]["direction"] == 0
    database.dispose()


def test_exit_is_a_flat_target_even_when_entry_signal_remains_active():
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100}] * 3
    path = position_path(rows, [1, 1, 1], PositionRules(0.03, 0.06, 1))
    assert path.decisions == (1, 0, 0)
    assert [action.kind for action in path.actions] == ["entry", "exit", "entry", "exit"]
