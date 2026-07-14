"""Research -> execution handoff: export kept hypotheses, run them in the bot.

Covers the full contract chain the repo's two products rely on:
experiment log (staged-validation verdicts) -> research_exploration.export
-> active-strategies artifact -> src.run_bot executes it with the SAME
predicate mask code that validated it.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_exploration.dsr import DSR_METHOD
from research_exploration.experiment_log import ExperimentRecord, log_result
from research_exploration.export import build_payload
from research_exploration.export import run as export_run
from research_exploration.hypothesis_generator import position_trading_set
from research_exploration.hypothesis_schema import ExitRule, Hypothesis, Predicate, RiskRule
from research_exploration.predicates import hypothesis_history_requirements
from src.autopilot.config import ProductConfig
from src.autopilot.strategy_policy import validate_strategy_artifact
from src.run_bot import PaperTradingBot

BASE_TS_MS = 1609459200000  # 2021-01-01 00:00 UTC — far in the past, all candles closed


def simple_hypothesis(hyp_id="TEST_LONG_5m_001", direction="long") -> Hypothesis:
    """Single-timeframe hypothesis the bot can evaluate on mocked 5m candles."""
    return Hypothesis(
        id=hyp_id,
        family="momentum_continuation",
        idea="test idea",
        market_logic="test logic",
        direction=direction,
        base_timeframe="5m",
        regime_timeframe="5m",
        setup_timeframe="5m",
        trigger_timeframe="5m",
        regime=[Predicate("5m", "rsi_14", "ge", reference=50.0)],
        setup=[Predicate("5m", "rsi_14", "le", reference=90.0)],
        trigger=[Predicate("5m", "rsi_14", "ge", reference=55.0)],
        exit=ExitRule(take_profit=0.04, stop_loss=0.02, horizon_bars=4),
        risk=RiskRule(risk_per_trade=0.02, max_daily_loss_r=2.5, cooldown_bars=10),
    )


def keep_record(
    hyp: Hypothesis,
    dsr=0.91,
    holdout_return=0.03,
    holdout_trades=12,
    pnl_unit="usdt",
    verdict="keep",
    market=None,
    symbol="BTCUSDT",
) -> ExperimentRecord:
    """Record shaped exactly like validation.validate_batch logs it."""
    metrics = {
        "dsr_deflated": dsr,
        "dsr_method": DSR_METHOD,
        "n_trials": 20,
        "sr_std_trials": 0.18,
        "trial_sharpe_count": 12,
        "trial_sharpe_observed_std": 0.16,
        "trial_sharpe_conservative_floor": 0.10,
        "reasons": [],
        "train": {"trades": 80, "total_return": 0.10, "win_rate": 0.58, "sharpe": 1.1},
        "validation": {"trades": 25, "total_return": 0.04, "win_rate": 0.56, "sharpe": 0.9},
        "holdout": {
            "trades": holdout_trades,
            "total_return": holdout_return,
            "win_rate": 0.55,
            "sharpe": 0.8,
        },
        "oos_pass_rate": 0.83,
        "sensitivity_pass_fraction": 0.75,
    }
    if market is None:
        market = "spot" if pnl_unit == "btc" else "futures"
    config = {
        "validation": {"train_frac": 0.6},
        "eval": {
            "fee_bps": 5.0,
            "slippage_bps": 2.0,
            "pnl_unit": pnl_unit,
            "market": market,
            "symbol": symbol,
        },
    }
    return ExperimentRecord(
        hypothesis_id=hyp.id,
        family=hyp.family,
        direction=hyp.direction,
        fingerprint=f"fp_{hyp.id}",
        verdict=verdict,
        metrics=metrics,
        config=config,
        notes="validation.py staged pipeline",
        hypothesis=hyp.to_dict(),
    )


def write_log(tmp_path: Path, records) -> Path:
    log = tmp_path / "experiment_log.jsonl"
    for rec in records:
        log_result(rec, log)
    return log


def product(
    tmp_path,
    *,
    name="active_income",
    objective="active_income",
    base_asset="USDT",
    market="futures",
    symbol="BTCUSDT",
    strategies_path=None,
) -> ProductConfig:
    return ProductConfig(
        name=name,
        enabled=True,
        objective=objective,
        base_asset=base_asset,
        market=market,
        execution_mode="paper",
        symbol=symbol,
        strategies_path=strategies_path or tmp_path / "active.json",
        state_file=tmp_path / f"{name}_state.json",
        trade_log=tmp_path / f"{name}_trades.csv",
        starting_equity=1000.0,
    )


# --------------------------------------------------------------------------- #
# Exporter gates and mapping
# --------------------------------------------------------------------------- #
def test_export_only_keeps_with_positive_holdout(tmp_path):
    kept = simple_hypothesis("KEEP_001")
    rejected = simple_hypothesis("REJECT_001")
    lost_holdout = simple_hypothesis("KEEP_BAD_HOLDOUT")
    log = write_log(
        tmp_path,
        [
            keep_record(kept),
            keep_record(rejected, verdict="reject"),
            keep_record(lost_holdout, holdout_return=-0.02),
        ],
    )
    payload = build_payload(log)
    ids = [s["id"] for s in payload["strategies"]]
    assert ids == ["KEEP_001"]
    assert payload["version"] == 2
    assert payload["pnl_unit"] == "usdt"
    assert payload["market"] == "futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["strategies"][0]["market"] == "futures"
    assert payload["strategies"][0]["symbol"] == "BTCUSDT"


def test_export_ranks_by_deflated_dsr_and_respects_top_k(tmp_path):
    log = write_log(
        tmp_path,
        [
            keep_record(simple_hypothesis("LOW_DSR"), dsr=0.62),
            keep_record(simple_hypothesis("HIGH_DSR"), dsr=0.95),
        ],
    )
    payload = build_payload(log)
    assert [s["id"] for s in payload["strategies"]] == ["HIGH_DSR", "LOW_DSR"]
    payload = build_payload(log, top_k=1)
    assert [s["id"] for s in payload["strategies"]] == ["HIGH_DSR"]
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(log, min_dsr=0.99)


def test_export_rejects_legacy_plain_psr_mislabeled_as_deflated(tmp_path):
    record = keep_record(simple_hypothesis("STALE_DSR"), dsr=0.99)
    for field in (
        "dsr_method",
        "sr_std_trials",
        "trial_sharpe_count",
        "trial_sharpe_observed_std",
        "trial_sharpe_conservative_floor",
    ):
        record.metrics.pop(field)
    log = write_log(tmp_path, [record])

    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(log)


def test_export_rejects_mixed_symbols(tmp_path):
    log = write_log(
        tmp_path,
        [
            keep_record(simple_hypothesis("BTC_SYMBOL"), symbol="BTCUSDT"),
            keep_record(simple_hypothesis("ETH_SYMBOL"), symbol="ETHUSDT"),
        ],
    )

    with pytest.raises(ValueError, match="mix symbols"):
        build_payload(log)


def test_export_pnl_unit_filter_and_mixed_unit_guard(tmp_path):
    log = write_log(
        tmp_path,
        [
            keep_record(simple_hypothesis("USDT_ONE"), pnl_unit="usdt"),
            keep_record(simple_hypothesis("BTC_ONE", direction="short"), pnl_unit="btc"),
        ],
    )
    btc = build_payload(log, pnl_unit="btc")
    assert [s["id"] for s in btc["strategies"]] == ["BTC_ONE"]
    assert btc["pnl_unit"] == "btc"
    assert btc["market"] == "spot"
    assert btc["strategies"][0]["market"] == "spot"
    assert btc["strategies"][0]["metrics"]["holdout_excess_return_vs_buy_hold"] == 0.03
    assert btc["strategies"][0]["metrics"]["holdout_buy_hold_return"] == 0.0
    with pytest.raises(ValueError, match="mix pnl units"):
        build_payload(log)


def test_export_maps_exit_risk_fees_and_baseline(tmp_path):
    hyp = simple_hypothesis()
    log = write_log(tmp_path, [keep_record(hyp)])
    entry = build_payload(log)["strategies"][0]
    assert entry["entry_type"] == "hypothesis"
    assert entry["take_profit"] == 0.04
    assert entry["stop_loss"] == 0.02
    assert entry["horizon_bars"] == 4
    assert entry["risk"]["risk_per_trade"] == 0.005
    assert entry["risk"]["max_position_fraction"] == 0.25
    assert entry["risk"]["daily_stop_loss"] == pytest.approx(-0.03)
    assert entry["risk"]["cooldown_bars"] == 12
    assert entry["risk"]["max_trades_per_day"] == 4
    assert entry["fees"] == {"fee_bps": 5.0, "slippage_bps": 2.0}
    assert entry["baseline_win_rate"] == 0.55  # holdout win rate preferred
    assert entry["metrics"]["holdout_total_return"] == 0.03
    # The hypothesis payload round-trips through the schema.
    assert Hypothesis.from_dict(entry["hypothesis"]).id == hyp.id


def test_export_active_income_artifact_passes_product_policy(tmp_path):
    hyp = simple_hypothesis()
    log = write_log(tmp_path, [keep_record(hyp, pnl_unit="usdt", market="futures")])

    artifact = build_payload(log, pnl_unit="usdt", market="futures", min_dsr=0.60)

    assert artifact["paper_trade_allowed"] is True
    assert artifact["live_allowed"] is True
    assert artifact["promotion_eligible"] is True
    assert (
        validate_strategy_artifact(
            product(tmp_path, strategies_path=tmp_path / "active_income.json"),
            artifact,
        )
        == []
    )


def test_export_btc_accumulation_artifact_passes_product_policy(tmp_path):
    hyp = simple_hypothesis("BTC_STEP_ASIDE", direction="short")
    log = write_log(tmp_path, [keep_record(hyp, pnl_unit="btc", market="spot")])

    artifact = build_payload(log, pnl_unit="btc", market="spot")
    entry = artifact["strategies"][0]

    assert entry["risk"]["risk_per_trade"] == 0.003
    assert entry["risk"]["daily_stop_loss"] == pytest.approx(-0.01)
    assert entry["risk"]["cooldown_bars"] == 24
    assert entry["risk"]["max_trades_per_day"] == 1
    assert (
        validate_strategy_artifact(
            product(
                tmp_path,
                name="btc_accumulation",
                objective="btc_accumulation",
                base_asset="BTC",
                market="spot",
                strategies_path=tmp_path / "btc_accumulation.json",
            ),
            artifact,
        )
        == []
    )


def test_export_allows_explicit_matching_market_stamp(tmp_path):
    hyp = simple_hypothesis()
    log = write_log(tmp_path, [keep_record(hyp, pnl_unit="usdt")])

    payload = build_payload(log, market="futures")

    assert payload["market"] == "futures"
    assert payload["strategies"][0]["market"] == "futures"


def test_export_rejects_market_override_mismatch(tmp_path):
    hyp = simple_hypothesis()
    log = write_log(tmp_path, [keep_record(hyp, pnl_unit="usdt", market="futures")])

    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(log, market="spot")


def test_export_rejects_legacy_records_without_market_provenance(tmp_path):
    rec = keep_record(simple_hypothesis())
    rec.config["eval"].pop("market")
    log = write_log(tmp_path, [rec])

    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(log)


def test_export_latest_record_wins_per_hypothesis(tmp_path):
    hyp = simple_hypothesis("DUPL_001")
    older = keep_record(hyp, dsr=0.70)
    older.timestamp = "2026-06-01T00:00:00+00:00"
    newer = keep_record(hyp, dsr=0.88)
    newer.timestamp = "2026-07-01T00:00:00+00:00"
    log = write_log(tmp_path, [older, newer])
    payload = build_payload(log)
    assert len(payload["strategies"]) == 1
    assert payload["strategies"][0]["metrics"]["dsr_deflated"] == 0.88


def test_export_empty_log_raises(tmp_path):
    log = write_log(tmp_path, [keep_record(simple_hypothesis(), verdict="reject")])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(log)


# --------------------------------------------------------------------------- #
# Bot executes an exported hypothesis strategy (golden path)
# --------------------------------------------------------------------------- #
def get_mock_binance_klines(
    n=10, start_time_ms=BASE_TS_MS, close_price=100.0, high=100.0, low=100.0, open_p=100.0
):
    klines = []
    current_time = start_time_ms
    for _ in range(n):
        klines.append(
            [
                current_time,
                str(open_p),
                str(high),
                str(low),
                str(close_price),
                "1000.0",
                current_time + 299999,
                "100000.0",
                100,
                "500.0",
                "50000.0",
                "0",
            ]
        )
        current_time += 300000
    return klines


def mock_indicator_features(df, timeframe, rsi_value=60.0):
    df_out = df.copy()
    df_out["rsi_14"] = rsi_value
    return df_out


def export_and_load_bot(tmp_path, rsi_value):
    log = write_log(tmp_path, [keep_record(simple_hypothesis())])
    artifact = export_run(log, tmp_path / "active_strategies_research.json")
    bot = PaperTradingBot(
        strategies_path=artifact,
        state_file=tmp_path / "bot_state.json",
        trade_log=tmp_path / "paper_trades.csv",
    )
    assert bot.strategies[0]["_hypothesis"].id == "TEST_LONG_5m_001"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5)
    with (
        patch("src.run_bot.requests.get", return_value=mock_resp),
        patch(
            "build_binance_indicator_dataset.build_indicator_features",
            side_effect=lambda df, tf: mock_indicator_features(df, tf, rsi_value),
        ),
    ):
        bot.run_cycle()
    return bot


def test_bot_runs_exported_hypothesis_and_opens_position(tmp_path):
    bot = export_and_load_bot(tmp_path, rsi_value=60.0)  # all predicates true
    pos = bot.state["open_positions"]["TEST_LONG_5m_001"]
    assert pos["direction"] == "long"
    assert pos["entry_price"] == 100.0
    assert pos["tp_pct"] == 0.04
    assert pos["sl_pct"] == 0.02
    assert pos["position_size"] == 0.25  # capped by max_position_fraction


def test_bot_hypothesis_no_entry_when_trigger_fails(tmp_path):
    # rsi 52 satisfies regime (>=50) and setup (<=90) but not trigger (>=55).
    bot = export_and_load_bot(tmp_path, rsi_value=52.0)
    assert bot.state["open_positions"] == {}


def test_bot_rejects_hypothesis_entry_missing_payload(tmp_path):
    artifact = {
        "version": 2,
        "strategies": [
            {
                "id": "x",
                "entry_type": "hypothesis",
                "base_timeframe": "5m",
                "direction": "long",
                "horizon_bars": 4,
                "take_profit": 0.04,
                "stop_loss": 0.02,
                "risk": {},
                "fees": {},
            }
        ],
    }
    path = tmp_path / "active_strategies.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key 'hypothesis'"):
        PaperTradingBot(
            strategies_path=path, state_file=tmp_path / "s.json", trade_log=tmp_path / "t.csv"
        )


def test_fetch_limits_cover_rolling_windows(tmp_path):
    """A 180-bar 4h rolling quantile on a 15m base needs 180*16 base rows —
    the executor must size its fetches so the mask is defined on the last bar."""
    hyp = Hypothesis(
        id="DEEP_WINDOW",
        family="volatility_breakout",
        idea="x",
        market_logic="x",
        direction="long",
        base_timeframe="15m",
        regime_timeframe="4h",
        setup_timeframe="1h",
        trigger_timeframe="15m",
        regime=[Predicate("4h", "natr_14", "q_le", quantile=0.35, window=180)],
        setup=[Predicate("1h", "adx_14", "le", reference=20.0)],
        trigger=[Predicate("15m", "close", "gt_feature", feature_b="max_20", shift_b=1)],
        exit=ExitRule(take_profit=0.018, stop_loss=0.007, horizon_bars=36),
    )
    needed = hypothesis_history_requirements(hyp)
    assert needed["15m"] >= 180 * 16  # scaled quantile window on the base frame
    assert needed["4h"] >= 180  # native bars for the HTF fetch

    log = write_log(tmp_path, [keep_record(hyp)])
    artifact = export_run(log, tmp_path / "art.json")
    bot = PaperTradingBot(
        strategies_path=artifact, state_file=tmp_path / "s.json", trade_log=tmp_path / "t.csv"
    )
    base_limit, htf_limits = bot._fetch_limits(bot.strategies[0])
    assert base_limit >= 180 * 16
    assert htf_limits["4h"] >= 180
    assert "1h" in htf_limits


@patch("src.run_bot.requests.get")
def test_fetch_live_candles_paginates_past_request_cap(mock_get, tmp_path):
    log = write_log(tmp_path, [keep_record(simple_hypothesis())])
    artifact = export_run(log, tmp_path / "art.json")
    bot = PaperTradingBot(
        strategies_path=artifact, state_file=tmp_path / "s.json", trade_log=tmp_path / "t.csv"
    )
    bot.KLINES_PER_REQUEST = 3

    newest = get_mock_binance_klines(3, start_time_ms=BASE_TS_MS + 2 * 300000)
    older = get_mock_binance_klines(2, start_time_ms=BASE_TS_MS)
    responses = []
    for page in (newest, older):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = page
        responses.append(resp)
    mock_get.side_effect = responses

    df = bot.fetch_live_candles("BTCUSDT", "futures", "5m", limit=5)
    assert len(df) == 5
    assert df["timestamp"].is_monotonic_increasing
    # Second request pages backwards from just before the first batch.
    second_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_params["endTime"] == newest[0][0] - 1
    assert second_params["limit"] == 2


# --------------------------------------------------------------------------- #
# Scenario-1 batch (position / BTC accumulation)
# --------------------------------------------------------------------------- #
def test_position_trading_set_is_short_coarse_and_wider():
    hyps = position_trading_set()
    assert hyps, "position batch must not be empty"
    for hyp in hyps:
        assert hyp.direction == "short"  # pnl_unit=btc: longs == holding
        assert "position_btc" in hyp.tags
        assert hyp.regime_timeframe in ("1d", "1w")
        assert hyp.base_timeframe in ("1h", "4h")
    # Exits widened vs the day-trade counterpart of the same family/stack.
    from research_exploration.strategy_families import build_family

    pos = next(h for h in hyps if h.family == "trend_continuation")
    day = build_family(
        "trend_continuation",
        "short",
        (pos.regime_timeframe, pos.setup_timeframe, pos.trigger_timeframe),
        1,
    )
    assert pos.exit.take_profit > day.exit.take_profit
    assert pos.exit.stop_loss > day.exit.stop_loss
    assert pos.exit.horizon_bars > day.exit.horizon_bars


def test_position_trading_set_guarded_variant():
    plain = position_trading_set()
    guarded = position_trading_set(with_guards=True)
    assert len(guarded) == len(plain)
    assert all(h.id.endswith("_G") for h in guarded)
    assert all(len(g.regime) > len(p.regime) for g, p in zip(guarded, plain, strict=False))
