import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.run_bot import PaperTradingBot

BASE_TS_MS = 1609459200000  # 2021-01-01 00:00 UTC — far in the past, all candles closed


def create_artifact(path: Path, baseline_win_rate=0.6, direction="long"):
    condition = {
        "feature": "tf_5m_rsi_14",
        "kind": "value_ge",
        "threshold": 50.0,
        "description": "tf_5m_rsi_14 >= 50.0",
    }
    artifact = {
        "version": 1,
        "generated_at": "2026-06-10T00:00:00+00:00",
        "export_git_sha": "test",
        "source_dir": "test",
        "search_git_sha": "test",
        "search_timestamp": None,
        "strategies": [
            {
                "id": "5m_long_r1",
                "rank": 1,
                "base_timeframe": "5m",
                "direction": direction,
                "horizon_bars": 4,
                "take_profit": 2.0,
                "stop_loss": 1.0,
                "use_atr_tp_sl": True,
                "pnl_unit": "usdt",
                "conditions": [condition],
                "rule": "tf_5m_rsi_14 >= 50.0",
                "risk": {
                    "risk_per_trade": 0.02,
                    "daily_stop_loss": -0.05,
                    "max_consecutive_losses": 2,
                    "cooldown_bars": 10,
                },
                "fees": {"fee_bps": 2.0, "slippage_bps": 1.0},
                "metrics": {},
                "baseline_win_rate": baseline_win_rate,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")


def get_mock_binance_klines(n=10, start_time_ms=BASE_TS_MS, close_price=100.0, high=100.0, low=100.0, open_p=100.0):
    klines = []
    current_time = start_time_ms
    for _ in range(n):
        klines.append([
            current_time, str(open_p), str(high), str(low), str(close_price),
            "1000.0", current_time + 299999, "100000.0", 100, "500.0", "50000.0", "0",
        ])
        current_time += 300000
    return klines


def last_kline_time_iso(n=5, start_time_ms=BASE_TS_MS) -> str:
    return str(pd.to_datetime(start_time_ms + (n - 1) * 300000, unit="ms", utc=True))


def mock_indicator_features(df, timeframe, rsi_value=60.0, atr_value=1.0):
    df_out = df.copy()
    df_out["rsi_14"] = rsi_value
    df_out["atr"] = atr_value
    df_out["atr_14"] = atr_value
    return df_out


@pytest.fixture
def bot_env(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    return strategies_path, state_file, trade_log


def make_bot(env, starting_equity=10_000.0):
    strategies_path, state_file, trade_log = env
    return PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=starting_equity,
    )


def test_bot_init_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_strategies"):
        PaperTradingBot(
            strategies_path=tmp_path / "missing.json",
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_empty_artifact(tmp_path):
    path = tmp_path / "active_strategies.json"
    path.write_text(json.dumps({"version": 1, "strategies": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no strategies"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_incomplete_strategy(tmp_path):
    path = tmp_path / "active_strategies.json"
    path.write_text(
        json.dumps({"version": 1, "strategies": [{"id": "x", "direction": "long"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required key"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_success(bot_env):
    bot = make_bot(bot_env)
    assert len(bot.strategies) == 1
    assert bot.strategies[0]["base_timeframe"] == "5m"
    assert bot.state["equity"] == 10_000.0
    assert bot.state["open_positions"] == {}
    assert bot.state["inactive_strategies"] == []


def test_bot_migrates_legacy_state(bot_env):
    strategies_path, state_file, trade_log = bot_env
    state_file.write_text(json.dumps({
        "equity": 5000.0,
        "open_position": {"direction": "long"},
        "strategy_active": True,
        "consecutive_losses": 1,
        "cooldown_until_ts": 0.0,
        "daily_pnl": 0.0,
        "last_pnl_reset_date": "2026-06-01",
    }), encoding="utf-8")
    bot = make_bot(bot_env)
    assert bot.state["equity"] == 5000.0
    assert bot.state["open_positions"] == {}
    assert "open_position" not in bot.state
    assert "strategy_active" not in bot.state


@patch("src.run_bot.requests.get")
def test_fetch_live_candles_drops_forming_candle(mock_get, bot_env):
    bot = make_bot(bot_env)
    now = pd.Timestamp.now(tz="UTC")
    current_bar_start = int(now.floor("5min").value // 10**6)
    klines = get_mock_binance_klines(3, start_time_ms=current_bar_start - 2 * 300000)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = klines
    mock_get.return_value = mock_resp

    df = bot.fetch_live_candles("BTCUSDT", "futures", "5m", limit=3)
    # The last kline is the still-forming current bar and must be dropped.
    assert len(df) == 2
    # Nanosecond resolution required: ms/us units break merge_asof on newer pandas.
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"


@patch("src.run_bot.requests.get")
def test_fetch_live_candles_failure(mock_get, bot_env):
    bot = make_bot(bot_env)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp
    with pytest.raises(RuntimeError, match="Binance API error: Internal Server Error"):
        bot.fetch_live_candles("BTCUSDT", "futures", "5m")


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_signal_not_triggered(mock_build_ind, mock_get, bot_env):
    bot = make_bot(bot_env)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=40.0)

    bot.run_cycle()
    assert bot.state["open_positions"] == {}


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_signal_triggered_opens_position(mock_build_ind, mock_get, bot_env):
    bot = make_bot(bot_env)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    pos = bot.state["open_positions"]["5m_long_r1"]
    assert pos["direction"] == "long"
    assert pos["entry_price"] == 100.0
    assert pos["sl_pct"] == 2.0 / 100.0  # (1 x ATR) / close
    assert pos["tp_pct"] == 4.0 / 100.0  # (2 x ATR) / close
    assert pos["sl_price"] == 98.0
    assert pos["tp_price"] == 104.0
    assert pos["position_size"] == 1.0  # 0.02 risk / 0.02 SL


def open_position_state(bot, entry_time, bars_ago_n=5):
    bot.state["open_positions"]["5m_long_r1"] = {
        "entry_time": entry_time,
        "direction": "long",
        "entry_price": 100.0,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "sl_price": 98.0,
        "tp_price": 104.0,
        "position_size": 1.0,
    }
    bot.state["equity"] = 10_000.0
    bot.state["daily_pnl"] = 0.0
    bot._save_state()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_exit_take_profit(mock_build_ind, mock_get, bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    # Entry on the latest bar: holding time 0 bars, so no time exit interferes.
    open_position_state(bot, last_kline_time_iso(5))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    # gross 4%, costs 6 bps round trip -> net 3.94%
    assert abs(bot.state["equity"] - 10_394.0) < 1e-6
    assert abs(bot.state["daily_pnl"] - 0.0394) < 1e-6

    df_trades = pd.read_csv(trade_log)
    assert len(df_trades) == 1
    assert df_trades["strategy_id"].iloc[0] == "5m_long_r1"
    assert df_trades["exit_reason"].iloc[0] == "take_profit"
    assert df_trades["exit_price"].iloc[0] == 104.0


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_exit_stop_loss(mock_build_ind, mock_get, bot_env):
    bot = make_bot(bot_env)
    open_position_state(bot, last_kline_time_iso(5))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=99.0, high=101.0, low=97.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert abs(bot.state["equity"] - 9_794.0) < 1e-6
    assert bot.state["consecutive_losses"] == 1


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_exit_time_horizon_from_timestamps(mock_build_ind, mock_get, bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    # Entry at the FIRST bar of the mock series: latest closed bar is 4 bars
    # later, which reaches the horizon (4) without touching TP/SL.
    open_position_state(bot, str(pd.to_datetime(BASE_TS_MS, unit="ms", utc=True)))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.5, high=102.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert abs(bot.state["equity"] - 10_144.0) < 1e-6
    df_trades = pd.read_csv(trade_log)
    assert df_trades["exit_reason"].iloc[0] == "time"
    assert df_trades["exit_price"].iloc[0] == 101.5


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_cooldown_trigger(mock_build_ind, mock_get, bot_env):
    bot = make_bot(bot_env)
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["consecutive_losses"] = 1
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=97.0, high=99.0, low=96.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    now = time.time()
    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert bot.state["consecutive_losses"] == 0
    assert bot.state["cooldown_until_ts"] > now
    # Cooldown is sized in base-TF bars: 10 bars x 300s
    assert bot.state["cooldown_until_ts"] <= now + 10 * 300 + 5

    # During cooldown no entry happens even though the signal fires.
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0)
    bot.run_cycle()
    assert bot.state["open_positions"] == {}


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_daily_stop_loss_limit(mock_build_ind, mock_get, bot_env):
    bot = make_bot(bot_env)
    bot.state["daily_pnl"] = -0.06
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0)

    bot.run_cycle()
    assert bot.state["open_positions"] == {}


def write_losing_trades(trade_log: Path, n=20, strategy_id="5m_long_r1", net_return=-0.01):
    trades = [
        {
            "strategy_id": strategy_id,
            "entry_time": "2026-06-05 20:00:00",
            "exit_time": "2026-06-05 20:20:00",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 99.0,
            "exit_reason": "stop",
            "gross_return": net_return,
            "net_return": net_return,
            "sized_return": net_return,
            "position_size": 1.0,
            "equity_after": 10000.0,
        }
        for _ in range(n)
    ]
    pd.DataFrame(trades).to_csv(trade_log, index=False)


def test_check_drift_no_kill_when_winrate_matches(bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    trades = []
    for i in range(15):
        net = 0.02 if i < 9 else -0.01
        trades.append({
            "strategy_id": "5m_long_r1", "entry_time": "x", "exit_time": "x",
            "direction": "long", "entry_price": 100.0, "exit_price": 100.0,
            "exit_reason": "time", "gross_return": net, "net_return": net,
            "sized_return": net, "position_size": 1.0, "equity_after": 10000.0,
        })
    pd.DataFrame(trades).to_csv(trade_log, index=False)
    bot.check_drift_and_ood(bot.strategies[0])
    assert bot.state["inactive_strategies"] == []


def test_check_drift_kill_trigger_deactivates_strategy(bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    write_losing_trades(trade_log)
    bot.check_drift_and_ood(bot.strategies[0])
    assert "5m_long_r1" in bot.state["inactive_strategies"]

    with patch("src.run_bot.requests.get") as mock_get:
        bot.run_cycle()
        mock_get.assert_not_called()


def test_check_drift_disabled_without_baseline(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path, baseline_win_rate=None)
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=tmp_path / "state.json",
        trade_log=trade_log,
    )
    write_losing_trades(trade_log)
    bot.check_drift_and_ood(bot.strategies[0])
    # No baseline -> drift detection must not fire (and must not crash).
    assert bot.state["inactive_strategies"] == []


def test_drift_only_counts_own_strategy_trades(bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    write_losing_trades(trade_log, strategy_id="other_strategy")
    bot.check_drift_and_ood(bot.strategies[0])
    assert bot.state["inactive_strategies"] == []
