import json

import pandas as pd
import pytest

from research_exploration.dsr import DSR_METHOD
from src.autopilot.approvals import artifact_digest
from src.autopilot.candidate_activation import product_identity
from src.autopilot.candidate_evidence import (
    CANDIDATE_PAPER_BACKFILL_FILL_SOURCE,
    CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON,
    CANDIDATE_PAPER_FORWARD_FILL_SOURCE,
    CANDIDATE_PAPER_FORWARD_REASON,
)
from src.autopilot.candidate_paper import candidate_paper_paths, main, run_candidate_paper
from src.autopilot.config import AutopilotConfig, ProductConfig
from src.autopilot.locking import acquire_runtime_lock
from src.run_bot import PaperTradingBot


def _product(tmp_path, *, mode="live"):
    return ProductConfig(
        name="active_income",
        enabled=True,
        objective="active_income",
        base_asset="USDT",
        market="futures",
        execution_mode=mode,
        symbol="BTCUSDT",
        strategies_path=tmp_path / "active.json",
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        starting_equity=1000.0,
    )


def _candidate(product):
    return {
        "version": 2,
        "market": "futures",
        "symbol": "BTCUSDT",
        "pnl_unit": "usdt",
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "product": product_identity(product),
        "strategies": [
            {
                "id": "candidate",
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


def test_candidate_paper_uses_digest_isolated_state_and_exact_artifact(
    monkeypatch,
    tmp_path,
):
    product = _product(tmp_path)
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate = _candidate(product)
    candidate_path = candidate_dir / "active_income.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    captured = {}

    class FakeBot:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.state = {
                "equity": 1001.0,
                "open_positions": {},
                "drawdown_halted": False,
            }

        def run_candidate_replay_cycle(
            self,
            *,
            max_unseen_bars,
            max_observation_delay_seconds,
        ):
            captured["ran"] = True
            captured["max_unseen_bars"] = max_unseen_bars
            captured["max_observation_delay_seconds"] = max_observation_delay_seconds
            return {"processed_events": 0}

    monkeypatch.setattr("src.autopilot.candidate_paper.PaperTradingBot", FakeBot)

    def fake_review(**kwargs):
        captured["require_candidate_paper_binding"] = kwargs["require_candidate_paper_binding"]
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "artifact_path": str(kwargs["artifact_path"]),
            "artifact_digest": artifact_digest(candidate),
            "trade_log": str(kwargs["trade_log"]),
            "strategies": [
                {
                    "recommendation": "needs_approval",
                    "reasons": ["passes"],
                    "approval_command": "unsafe-before-activation",
                }
            ],
        }

    monkeypatch.setattr(
        "src.autopilot.candidate_paper.build_promotion_review",
        fake_review,
    )
    monkeypatch.setattr(
        "src.autopilot.candidate_paper.write_review",
        lambda review, output_json, output_md: captured.update(review=review),
    )

    report = run_candidate_paper(
        AutopilotConfig(products=[product]),
        candidate_dir=candidate_dir,
    )

    item = report["products"][0]
    digest = artifact_digest(candidate)
    assert report["ok"] is True
    assert item["candidate_digest"] == digest
    assert item["candidate_activation_ready"] is True
    assert captured["artifact_payload"] == candidate
    assert captured["ran"] is True
    assert captured["max_unseen_bars"] == 240
    assert captured["max_observation_delay_seconds"] == 90
    assert item["execution_path"] == "paper_only_forward_observation_with_quarantined_replay"
    assert captured["require_candidate_paper_binding"] is True
    assert digest.removeprefix("sha256:")[:16] in str(captured["state_file"])
    assert captured["review"]["strategies"][0]["recommendation"] == "ready_for_activation"
    assert captured["review"]["strategies"][0]["approval_command"] is None


def test_candidate_paper_skips_non_live_product_without_candidate(tmp_path):
    report = run_candidate_paper(
        AutopilotConfig(products=[_product(tmp_path, mode="paper")]),
        candidate_dir=tmp_path / "candidates",
    )

    assert report["ok"] is True
    assert report["products"][0]["reason"] == "product_not_live"


def test_candidate_paper_fails_closed_on_wrong_product_identity(tmp_path):
    product = _product(tmp_path)
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate = _candidate(product)
    candidate["product"]["symbol"] = "ETHUSDT"
    (candidate_dir / "active_income.json").write_text(json.dumps(candidate), encoding="utf-8")

    report = run_candidate_paper(
        AutopilotConfig(products=[product]),
        candidate_dir=candidate_dir,
    )

    assert report["ok"] is False
    assert "product identity mismatch" in report["products"][0]["error"]


def test_candidate_paper_paths_reject_bad_digest(tmp_path):
    with pytest.raises(ValueError, match="sha256"):
        candidate_paper_paths(
            "active_income",
            "not-a-digest",
            candidate_dir=tmp_path,
        )


def test_candidate_replay_backlog_limit_is_strict_config():
    configured = AutopilotConfig.from_dict({"candidate_paper_max_unseen_bars": 17})
    assert configured.candidate_paper_max_unseen_bars == 17

    with pytest.raises(ValueError, match="candidate_paper_max_unseen_bars"):
        AutopilotConfig.from_dict({"candidate_paper_max_unseen_bars": 0})


def test_candidate_paper_main_skips_overlap_without_overwriting_status(tmp_path, capsys):
    lock = tmp_path / "candidate-paper.lock"
    output = tmp_path / "candidate-paper.json"
    output.write_text('{"sentinel": true}', encoding="utf-8")

    with acquire_runtime_lock(lock):
        main(
            [
                "--config",
                str(tmp_path / "unused.json"),
                "--output",
                str(output),
                "--lock",
                str(lock),
            ]
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}
    printed = json.loads(capsys.readouterr().out)
    assert printed["skipped"] is True
    assert printed["reason"] == "candidate_paper_cycle_already_running"


def _replay_artifact(tmp_path):
    artifact = _candidate(_product(tmp_path))
    strategy = artifact["strategies"][0]
    strategy.update(
        base_timeframe="1m",
        horizon_bars=10,
        take_profit=0.10,
        stop_loss=0.20,
        conditions=[
            {
                "feature": "tf_1m_close",
                "kind": "value_ge",
                "threshold": 105,
                "description": "close >= 105",
            }
        ],
        fees={"fee_bps": 0, "slippage_bps": 0},
    )
    return artifact


def _six_replay_bars():
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1min")
    rows = [
        (100, 101, 99, 100),
        (100, 106, 99, 105),
        (110, 115, 109, 112),
        (112, 122, 111, 120),
        (120, 121, 99, 100),
        (100, 101, 99, 100),
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "tf_1m_open": [row[0] for row in rows],
            "tf_1m_high": [row[1] for row in rows],
            "tf_1m_low": [row[2] for row in rows],
            "tf_1m_close": [row[3] for row in rows],
        }
    )


def _replay_bot(tmp_path):
    artifact = _replay_artifact(tmp_path)
    return PaperTradingBot(
        strategies_path=tmp_path / "candidate.json",
        state_file=tmp_path / "candidate-state.json",
        trade_log=tmp_path / "candidate-trades.csv",
        starting_equity=1000,
        symbol="BTCUSDT",
        market="futures",
        objective="active_income",
        base_asset="USDT",
        artifact_payload=artifact,
    )


def test_candidate_replay_catches_up_all_bars_at_next_open_and_is_restart_safe(
    monkeypatch,
    tmp_path,
):
    frame = _six_replay_bars()
    visible = frame.iloc[:1].copy()
    bot = _replay_bot(tmp_path)
    monkeypatch.setattr(
        bot,
        "_build_feature_frame",
        lambda strategy: (visible.copy(), float(visible["tf_1m_close"].iloc[-1])),
    )

    initialized = bot.run_candidate_replay_cycle(max_unseen_bars=10)
    assert initialized["initialized_strategies"] == ["candidate"]

    visible = frame.copy()
    replayed = bot.run_candidate_replay_cycle(max_unseen_bars=10)

    assert replayed["processed_events"] == 5
    assert replayed["pending_entries"] == []
    assert bot.state["open_positions"] == {}
    trades = pd.read_csv(tmp_path / "candidate-trades.csv")
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["candidate_paper_execution_schema"] == replayed["execution_schema"]
    assert trade["candidate_paper_engine_digest"] == replayed["execution_engine_digest"]
    assert bot.state["candidate_paper_engine_digest"] == replayed["execution_engine_digest"]
    assert trade["entry_time"] == "2026-01-01T00:02:00+00:00"
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["exit_time"].startswith("2026-01-01 00:04:00+00:00")
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(121.0)
    assert bool(trade["candidate_paper_evidence_eligible"]) is False
    assert trade["candidate_paper_evidence_reason"] == CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON
    assert trade["candidate_paper_entry_fill_source"] == CANDIDATE_PAPER_BACKFILL_FILL_SOURCE
    assert replayed["backfilled_events"] == 5
    assert replayed["forward_observed_events"] == 0

    # Simulate a process death after the exit accounting WAL committed bar 3
    # but before its replay cursor advanced. The durable decision marker must
    # suppress a same-bar re-entry when the bar is retried.
    bot.state["candidate_replay_cursor_by_strategy"]["candidate"] = frame.iloc[2][
        "timestamp"
    ].isoformat()
    bot.state["last_entry_decision_bar_by_strategy"]["candidate"] = frame.iloc[3][
        "timestamp"
    ].isoformat()
    bot._save_state()
    recovered = _replay_bot(tmp_path)
    monkeypatch.setattr(
        recovered,
        "_build_feature_frame",
        lambda strategy: (frame.copy(), float(frame["tf_1m_close"].iloc[-1])),
    )
    recovery = recovered.run_candidate_replay_cycle(max_unseen_bars=10)
    assert recovery["processed_events"] == 3
    assert recovered.state["open_positions"] == {}
    assert len(pd.read_csv(tmp_path / "candidate-trades.csv")) == 1

    restarted = _replay_bot(tmp_path)
    monkeypatch.setattr(
        restarted,
        "_build_feature_frame",
        lambda strategy: (frame.copy(), float(frame["tf_1m_close"].iloc[-1])),
    )
    repeat = restarted.run_candidate_replay_cycle(max_unseen_bars=10)
    assert repeat["processed_events"] == 0
    assert len(pd.read_csv(tmp_path / "candidate-trades.csv")) == 1


def test_candidate_replay_backlog_overflow_is_fail_closed_without_cursor_advance(
    monkeypatch,
    tmp_path,
):
    frame = _six_replay_bars()
    visible = frame.iloc[:1].copy()
    bot = _replay_bot(tmp_path)
    monkeypatch.setattr(
        bot,
        "_build_feature_frame",
        lambda strategy: (visible.copy(), float(visible["tf_1m_close"].iloc[-1])),
    )
    bot.run_candidate_replay_cycle(max_unseen_bars=10)
    cursor_before = bot.state["candidate_replay_cursor_by_strategy"]["candidate"]

    visible = frame.copy()
    with pytest.raises(RuntimeError, match="backlog overflow"):
        bot.run_candidate_replay_cycle(max_unseen_bars=4)

    assert bot.state["candidate_replay_cursor_by_strategy"]["candidate"] == cursor_before
    assert not (tmp_path / "candidate-trades.csv").exists()


def test_candidate_replay_engine_change_resets_flat_state_and_evidence_cursor(
    monkeypatch,
    tmp_path,
):
    frame = _six_replay_bars()
    visible = frame.iloc[:1].copy()
    bot = _replay_bot(tmp_path)
    monkeypatch.setattr(
        bot,
        "_build_feature_frame",
        lambda strategy: (visible.copy(), float(visible["tf_1m_close"].iloc[-1])),
    )
    bot.run_candidate_replay_cycle(max_unseen_bars=10)
    bot.state["candidate_paper_engine_digest"] = "sha256:" + "0" * 64
    bot.state["equity"] = 900.0
    bot.state["daily_pnl"] = -0.05
    bot._save_state()

    restarted = _replay_bot(tmp_path)
    monkeypatch.setattr(
        restarted,
        "_build_feature_frame",
        lambda strategy: (frame.copy(), float(frame["tf_1m_close"].iloc[-1])),
    )
    result = restarted.run_candidate_replay_cycle(max_unseen_bars=10)

    assert result["initialized_strategies"] == ["candidate"]
    assert result["processed_events"] == 0
    assert restarted.state["equity"] == 1000.0
    assert restarted.state["daily_pnl"] == 0.0
    assert restarted.state["candidate_paper_engine_digest"] == result["execution_engine_digest"]
    assert (
        restarted.state["candidate_paper_execution_history"][-1]["engine_digest"]
        == "sha256:" + "0" * 64
    )


def _forward_replay_bars():
    timestamps = pd.date_range("2026-07-10T10:00:00Z", periods=4, freq="1min")
    rows = [
        (100, 101, 99, 100),
        (100, 106, 99, 105),
        # The signal is observed at 10:02:35, inside this 10:02 bar. Its
        # extreme prices must never be applied to the later paper entry.
        (111, 200, 1, 110),
        (110, 122, 100, 121),
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "tf_1m_open": [row[0] for row in rows],
            "tf_1m_high": [row[1] for row in rows],
            "tf_1m_low": [row[2] for row in rows],
            "tf_1m_close": [row[3] for row in rows],
        }
    )


def _set_visible_frame(monkeypatch, bot, frame):
    monkeypatch.setattr(
        bot,
        "_build_feature_frame",
        lambda strategy: (frame.copy(), float(frame["tf_1m_close"].iloc[-1])),
    )


def test_candidate_fresh_signal_uses_quote_response_time_and_skips_partial_bar(
    monkeypatch,
    tmp_path,
):
    frame = _forward_replay_bars()
    bot = _replay_bot(tmp_path)
    _set_visible_frame(monkeypatch, bot, frame.iloc[:1])
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:01:10Z",
    )

    _set_visible_frame(monkeypatch, bot, frame.iloc[:2])
    quote_observed_at = pd.Timestamp("2026-07-10T10:02:35Z")
    monkeypatch.setattr(
        bot,
        "fetch_public_observation_quote",
        lambda: (110.0, quote_observed_at),
    )
    entered = bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:02:30Z",
    )

    position = bot.state["open_positions"]["candidate"]
    assert entered["forward_observed_events"] == 1
    assert position["entry_time"] == quote_observed_at.isoformat()
    assert position["candidate_paper_observed_at"] == quote_observed_at.isoformat()
    assert position["candidate_paper_evidence_eligible"] is True
    assert position["candidate_paper_evidence_reason"] == CANDIDATE_PAPER_FORWARD_REASON
    assert position["candidate_paper_entry_fill_source"] == CANDIDATE_PAPER_FORWARD_FILL_SOURCE

    _set_visible_frame(monkeypatch, bot, frame.iloc[:3])
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:03:10Z",
    )
    assert "candidate" in bot.state["open_positions"]
    assert bot.state["open_positions"]["candidate"]["candidate_paper_evidence_eligible"] is True
    assert not (tmp_path / "candidate-trades.csv").exists()

    _set_visible_frame(monkeypatch, bot, frame)
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:04:10Z",
    )
    trade = pd.read_csv(tmp_path / "candidate-trades.csv").iloc[0]
    assert bool(trade["candidate_paper_evidence_eligible"]) is True
    assert trade["entry_time"] == quote_observed_at.isoformat()
    assert trade["candidate_paper_observed_at"] == quote_observed_at.isoformat()
    assert trade["exit_time"].startswith("2026-07-10 10:04:10+00:00")


def test_candidate_downtime_management_quarantines_forward_entry(
    monkeypatch,
    tmp_path,
):
    frame = _forward_replay_bars()
    bot = _replay_bot(tmp_path)
    _set_visible_frame(monkeypatch, bot, frame.iloc[:1])
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:01:10Z",
    )
    _set_visible_frame(monkeypatch, bot, frame.iloc[:2])
    monkeypatch.setattr(
        bot,
        "fetch_public_observation_quote",
        lambda: (110.0, pd.Timestamp("2026-07-10T10:02:35Z")),
    )
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:02:30Z",
    )

    # Both 10:02 and 10:03 arrive together after downtime. The first catch-up
    # event permanently quarantines the position before either bar is managed.
    _set_visible_frame(monkeypatch, bot, frame)
    replayed = bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:04:10Z",
    )

    assert replayed["backfilled_events"] == 1
    assert replayed["forward_observed_events"] == 1
    trade = pd.read_csv(tmp_path / "candidate-trades.csv").iloc[0]
    assert bool(trade["candidate_paper_evidence_eligible"]) is False
    assert trade["candidate_paper_evidence_reason"] == CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON


def test_candidate_rejects_quote_observed_outside_forward_window(
    monkeypatch,
    tmp_path,
):
    frame = _forward_replay_bars()
    bot = _replay_bot(tmp_path)
    _set_visible_frame(monkeypatch, bot, frame.iloc[:1])
    bot.run_candidate_replay_cycle(
        max_unseen_bars=10,
        observation_time="2026-07-10T10:01:10Z",
    )
    cursor_before = bot.state["candidate_replay_cursor_by_strategy"]["candidate"]
    _set_visible_frame(monkeypatch, bot, frame.iloc[:2])
    monkeypatch.setattr(
        bot,
        "fetch_public_observation_quote",
        lambda: (110.0, pd.Timestamp("2026-07-10T10:04:00Z")),
    )

    with pytest.raises(RuntimeError, match="outside the promotable signal-delay window"):
        bot.run_candidate_replay_cycle(
            max_unseen_bars=10,
            max_observation_delay_seconds=90,
            observation_time="2026-07-10T10:02:30Z",
        )

    assert bot.state["candidate_replay_cursor_by_strategy"]["candidate"] == cursor_before
    assert bot.state["open_positions"] == {}


def test_candidate_mixed_timeframes_order_by_information_availability(
    monkeypatch,
    tmp_path,
):
    artifact = _candidate(_product(tmp_path))
    one_hour = artifact["strategies"][0]
    one_hour.update(
        id="one_hour",
        base_timeframe="1h",
        conditions=[
            {
                "feature": "tf_1h_close",
                "kind": "value_ge",
                "threshold": 999,
                "description": "disabled",
            }
        ],
    )
    five_minute = json.loads(json.dumps(one_hour))
    five_minute.update(
        id="five_minute",
        base_timeframe="5m",
        conditions=[
            {
                "feature": "tf_5m_close",
                "kind": "value_ge",
                "threshold": 999,
                "description": "disabled",
            }
        ],
    )
    artifact["strategies"] = [one_hour, five_minute]
    bot = PaperTradingBot(
        strategies_path=tmp_path / "candidate.json",
        state_file=tmp_path / "candidate-state.json",
        trade_log=tmp_path / "candidate-trades.csv",
        starting_equity=1000,
        symbol="BTCUSDT",
        market="futures",
        objective="active_income",
        base_asset="USDT",
        artifact_payload=artifact,
    )

    def frame(timeframe, timestamps):
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, utc=True),
                f"tf_{timeframe}_open": 100.0,
                f"tf_{timeframe}_high": 101.0,
                f"tf_{timeframe}_low": 99.0,
                f"tf_{timeframe}_close": 100.0,
            }
        )

    initial = {
        "one_hour": frame("1h", ["2026-07-10T09:00:00Z"]),
        "five_minute": frame("5m", ["2026-07-10T10:00:00Z"]),
    }
    visible = initial
    monkeypatch.setattr(
        bot,
        "_build_feature_frame",
        lambda strategy: (
            visible[strategy["id"]].copy(),
            float(visible[strategy["id"]][f"tf_{strategy['base_timeframe']}_close"].iloc[-1]),
        ),
    )
    bot.run_candidate_replay_cycle(
        max_unseen_bars=20,
        observation_time="2026-07-10T10:05:30Z",
    )

    visible = {
        "one_hour": frame(
            "1h",
            ["2026-07-10T09:00:00Z", "2026-07-10T10:00:00Z"],
        ),
        "five_minute": frame(
            "5m",
            pd.date_range("2026-07-10T10:00:00Z", periods=12, freq="5min"),
        ),
    }
    result = bot.run_candidate_replay_cycle(
        max_unseen_bars=20,
        observation_time="2026-07-10T11:00:30Z",
    )

    ordered = [(event["strategy_id"], event["bar_open"]) for event in result["event_order_tail"]]
    assert ordered[-2][0] == "five_minute"
    assert ordered[-2][1].startswith("2026-07-10T10:55:00")
    assert ordered[-1][0] == "one_hour"
    assert ordered[-1][1].startswith("2026-07-10T10:00:00")
    assert {event["information_available_at"] for event in result["event_order_tail"][-2:]} == {
        "2026-07-10T11:00:00+00:00"
    }
