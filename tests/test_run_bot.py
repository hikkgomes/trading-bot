import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.execution import Fill, Order, OrderSide, PaperBroker, Position
from src.run_bot import PaperTradingBot, parse_args

BASE_TS_MS = 1609459200000  # 2021-01-01 00:00 UTC — far in the past, all candles closed


def create_artifact(
    path: Path,
    baseline_win_rate=0.6,
    direction="long",
    pnl_unit="usdt",
    max_trades_per_day=4,
):
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
                "pnl_unit": pnl_unit,
                "conditions": [condition],
                "rule": "tf_5m_rsi_14 >= 50.0",
                "risk": {
                "risk_per_trade": 0.02,
                "daily_stop_loss": -0.05,
                "max_consecutive_losses": 2,
                "cooldown_bars": 10,
                "max_position_fraction": 1.0,
                "max_trades_per_day": max_trades_per_day,
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


class PriceSource:
    def __init__(self, price):
        self.price = float(price)

    def __call__(self, symbol):
        return self.price


class SpotPaperBroker(PaperBroker):
    class Config:
        market_type = "spot"

    config = Config()


class LiveBroker(PaperBroker):
    class Config:
        live = True
        market_type = "futures"

    config = Config()


def test_parse_args_accepts_product_execution_guards(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_bot",
            "--market",
            "spot",
            "--objective",
            "btc_accumulation",
            "--base-asset",
            "BTC",
        ],
    )

    args = parse_args()

    assert args.market == "spot"
    assert args.objective == "btc_accumulation"
    assert args.base_asset == "BTC"


def test_bot_rejects_live_broker_without_autopilot_gate(bot_env):
    strategies_path, state_file, trade_log = bot_env

    with pytest.raises(RuntimeError, match="Live broker injection requires"):
        PaperTradingBot(
            strategies_path=strategies_path,
            state_file=state_file,
            trade_log=trade_log,
            broker=LiveBroker(PriceSource(100.0), starting_balance=1000.0),
        )


def test_bot_allows_live_broker_after_autopilot_gate(bot_env):
    strategies_path, state_file, trade_log = bot_env

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        broker=LiveBroker(PriceSource(100.0), starting_balance=1000.0),
        live_gate_approved=True,
    )

    assert bot.broker is not None


def test_bot_rejects_symlink_strategy_artifact_without_trusting_target(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    target = tmp_path / "external_strategies.json"
    state_file = tmp_path / "state.json"
    trade_log = tmp_path / "trades.csv"
    create_artifact(target)
    strategies_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="Strategy artifact must not be a symlink"):
        PaperTradingBot(
            strategies_path=strategies_path,
            state_file=state_file,
            trade_log=trade_log,
        )

    assert strategies_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["strategies"][0]["id"] == "5m_long_r1"
    assert not state_file.exists()


def test_bot_rejects_symlink_state_file_without_trusting_target(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "state.json"
    target = tmp_path / "external_state.json"
    trade_log = tmp_path / "trades.csv"
    create_artifact(strategies_path)
    target.write_text(json.dumps({"equity": 1234.0, "open_positions": {}}), encoding="utf-8")
    state_file.symlink_to(target)

    with pytest.raises(RuntimeError, match="State file must not be a symlink"):
        PaperTradingBot(
            strategies_path=strategies_path,
            state_file=state_file,
            trade_log=trade_log,
        )

    assert state_file.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"equity": 1234.0, "open_positions": {}}
    assert not trade_log.exists()


def test_bot_rejects_symlink_trade_log_without_touching_target(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "state.json"
    trade_log = tmp_path / "trades.csv"
    target = tmp_path / "external_trades.csv"
    create_artifact(strategies_path)
    target.write_text("existing\n", encoding="utf-8")
    trade_log.symlink_to(target)

    with pytest.raises(RuntimeError, match="Trade log must not be a symlink"):
        PaperTradingBot(
            strategies_path=strategies_path,
            state_file=state_file,
            trade_log=trade_log,
        )

    assert trade_log.is_symlink()
    assert target.read_text(encoding="utf-8") == "existing\n"
    assert not state_file.exists()


def test_bot_with_broker_rejects_persisted_open_position_without_broker_metadata(bot_env):
    strategies_path, state_file, trade_log = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": 10_000.0,
                "open_positions": {
                    "5m_long_r1": {
                        "entry_time": last_kline_time_iso(5),
                        "direction": "long",
                        "entry_price": 100.0,
                        "sl_pct": 0.02,
                        "tp_pct": 0.04,
                        "sl_price": 98.0,
                        "tp_price": 104.0,
                        "position_size": 1.0,
                    }
                },
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": {},
                "last_pnl_reset_date": "2026-01-01",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="broker metadata is required"):
        PaperTradingBot(
            strategies_path=strategies_path,
            state_file=state_file,
            trade_log=trade_log,
            broker=PaperBroker(price_source=PriceSource(100.0), starting_balance=10_000.0),
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


def test_bot_init_rejects_invalid_strategy_id(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["id"] = " "
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Strategy id must be a non-empty string"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_duplicate_strategy_ids(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"].append(dict(payload["strategies"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate strategy id '5m_long_r1'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_incomplete_risk_block(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["strategies"][0]["risk"]["daily_stop_loss"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="risk is missing required key"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_unsafe_risk_values(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"]["daily_stop_loss"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="daily_stop_loss must be negative"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_missing_daily_trade_cap(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, max_trades_per_day=None)

    with pytest.raises(ValueError, match="max_trades_per_day must be a positive integer"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_btc_accumulation_long_strategy(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, direction="long", pnl_unit="btc")

    with pytest.raises(ValueError, match="spot step-aside short"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="spot",
            objective="btc_accumulation",
            base_asset="BTC",
        )


def test_bot_init_rejects_btc_accumulation_without_buy_hold_excess(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, direction="short", pnl_unit="btc")

    with pytest.raises(ValueError, match="holdout_excess_return_vs_buy_hold"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="spot",
            objective="btc_accumulation",
            base_asset="BTC",
        )


def test_bot_init_accepts_active_income_product_guards(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, pnl_unit="usdt")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["strategies"][0]["market"] = "futures"
    payload["strategies"][0]["metrics"] = {"holdout_total_return": 0.03, "dsr_deflated": 0.72}
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        market="futures",
        objective="active_income",
        base_asset="USDT",
    )

    assert bot.strategies[0]["metrics"]["dsr_deflated"] == pytest.approx(0.72)


def test_bot_init_accepts_active_income_paper_only_without_research_metrics(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, pnl_unit="usdt")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["paper_trade_allowed"] = True
    payload["live_allowed"] = False
    payload["promotion_eligible"] = False
    payload["strategies"][0]["market"] = "futures"
    payload["strategies"][0]["metrics"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        market="futures",
        objective="active_income",
        base_asset="USDT",
    )

    assert bot.artifact["promotion_eligible"] is False
    assert bot.strategies[0]["metrics"] == {}


def test_bot_init_rejects_active_income_without_dsr(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, pnl_unit="usdt")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["strategies"][0]["market"] = "futures"
    payload["strategies"][0]["metrics"] = {"holdout_total_return": 0.03}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics is missing required key 'dsr'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="futures",
            objective="active_income",
            base_asset="USDT",
        )


def test_bot_init_rejects_active_income_low_dsr(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, pnl_unit="usdt")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["strategies"][0]["market"] = "futures"
    payload["strategies"][0]["metrics"] = {"holdout_total_return": 0.03, "dsr_deflated": 0.12}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="active income DSR 0.120000 below 0.600000"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="futures",
            objective="active_income",
            base_asset="USDT",
        )


@pytest.mark.parametrize(
    ("risk_key", "value", "message"),
    [
        ("risk_per_trade", float("nan"), "risk_per_trade must be finite"),
        ("daily_stop_loss", float("inf"), "daily_stop_loss must be finite"),
        ("max_consecutive_losses", float("nan"), "max_consecutive_losses must be finite"),
        ("cooldown_bars", float("inf"), "cooldown_bars must be finite"),
        ("max_position_fraction", float("nan"), "max_position_fraction must be finite"),
        ("max_trades_per_day", float("nan"), "max_trades_per_day must be finite"),
    ],
)
def test_bot_init_rejects_nonfinite_risk_values(tmp_path, risk_key, value, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, max_trades_per_day=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"][risk_key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0.0, "max_position_fraction must be > 0 and <= 1"),
        (-0.1, "max_position_fraction must be > 0 and <= 1"),
        (1.1, "max_position_fraction must be > 0 and <= 1"),
    ],
)
def test_bot_init_rejects_invalid_max_position_fraction(tmp_path, value, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"]["max_position_fraction"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_normalizes_numeric_risk_strings(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, max_trades_per_day=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"] = {
        "risk_per_trade": "0.02",
        "daily_stop_loss": "-0.05",
        "max_consecutive_losses": "2",
        "cooldown_bars": "10",
        "max_position_fraction": "0.25",
        "max_trades_per_day": "2",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
    )

    assert bot.strategies[0]["risk"] == {
        "risk_per_trade": 0.02,
        "daily_stop_loss": -0.05,
        "max_consecutive_losses": 2,
        "cooldown_bars": 10,
        "max_position_fraction": 0.25,
        "max_trades_per_day": 2,
    }


def test_bot_init_normalizes_baseline_win_rate_string(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, baseline_win_rate="0.61")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
    )

    assert bot.strategies[0]["baseline_win_rate"] == pytest.approx(0.61)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "baseline_win_rate must be between 0 and 1"),
        (1, "baseline_win_rate must be between 0 and 1"),
        (-0.1, "baseline_win_rate must be between 0 and 1"),
        (1.1, "baseline_win_rate must be between 0 and 1"),
        (float("nan"), "baseline_win_rate must be finite"),
        (float("inf"), "baseline_win_rate must be finite"),
        ("not-a-number", "baseline_win_rate must be numeric"),
    ],
)
def test_bot_init_rejects_invalid_baseline_win_rate(tmp_path, value, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path, baseline_win_rate=value)

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_incomplete_fees_block(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["strategies"][0]["fees"]["fee_bps"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fees is missing required key"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_rejects_negative_fee_values(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["fees"]["slippage_bps"] = -1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


@pytest.mark.parametrize(
    ("fee_key", "value", "message"),
    [
        ("fee_bps", float("nan"), "fee_bps must be finite"),
        ("slippage_bps", float("inf"), "slippage_bps must be finite"),
    ],
)
def test_bot_init_rejects_nonfinite_fee_values(tmp_path, fee_key, value, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["fees"][fee_key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_normalizes_numeric_fee_strings(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["fees"] = {"fee_bps": "2.5", "slippage_bps": "1.25"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
    )

    assert bot.strategies[0]["fees"] == {"fee_bps": 2.5, "slippage_bps": 1.25}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("direction", "sideways", "direction must be long or short"),
        ("base_timeframe", "", "base_timeframe must be a non-empty string"),
        ("horizon_bars", 0, "horizon_bars must be positive"),
        ("horizon_bars", 1.5, "horizon_bars must be an integer"),
        ("take_profit", float("nan"), "take_profit must be finite"),
        ("stop_loss", -1.0, "stop_loss must be positive"),
        ("entry_type", "external", "entry_type must be 'conditions' or 'hypothesis'"),
    ],
)
def test_bot_init_rejects_invalid_strategy_execution_fields(tmp_path, field, value, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
        )


def test_bot_init_accepts_matching_declared_markets(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["strategies"][0]["market"] = "futures"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        market="futures",
    )

    assert bot.artifact["market"] == "futures"
    assert bot.strategies[0]["market"] == "futures"


def test_bot_init_rejects_artifact_market_mismatch(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "spot"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Strategy artifact market 'spot' does not match bot market 'futures'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="futures",
        )


def test_bot_init_rejects_strategy_market_mismatch(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market"] = "futures"
    payload["strategies"][0]["market"] = "spot"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Strategy 5m_long_r1 market 'spot' does not match bot market 'futures'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="futures",
        )


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("artifact", "Strategy artifact market must be 'futures' or 'spot'"),
        ("strategy", "Strategy 5m_long_r1 market must be 'futures' or 'spot'"),
    ],
)
def test_bot_init_rejects_invalid_declared_market(tmp_path, location, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if location == "artifact":
        payload["market"] = "margin"
    else:
        payload["market"] = "futures"
        payload["strategies"][0]["market"] = "margin"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            market="futures",
        )


def test_bot_init_accepts_matching_declared_symbols(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["symbol"] = "BTCUSDT"
    payload["strategies"][0]["symbol"] = "BTC/USDT"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bot = PaperTradingBot(
        strategies_path=path,
        state_file=tmp_path / "state.json",
        trade_log=tmp_path / "trades.csv",
        symbol="BTC/USDT:USDT",
        market="futures",
    )

    assert bot.artifact["symbol"] == "BTCUSDT"
    assert bot.strategies[0]["symbol"] == "BTC/USDT"


def test_bot_init_rejects_artifact_symbol_mismatch(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["symbol"] = "ETHUSDT"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Strategy artifact symbol 'ETHUSDT' does not match bot symbol 'BTCUSDT'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            symbol="BTCUSDT",
            market="futures",
        )


def test_bot_init_rejects_strategy_symbol_mismatch(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["symbol"] = "BTCUSDT"
    payload["strategies"][0]["symbol"] = "ETHUSDT"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Strategy 5m_long_r1 symbol 'ETHUSDT' does not match bot symbol 'BTCUSDT'"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            symbol="BTCUSDT",
            market="futures",
        )


def test_bot_init_rejects_invalid_declared_symbol(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["symbol"] = "BTC"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must include base and quote assets"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            symbol="BTCUSDT",
            market="futures",
        )


def test_bot_init_rejects_spot_symbol_with_settlement_asset(tmp_path):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)

    with pytest.raises(ValueError, match="spot market symbol must not include a settlement asset"):
        PaperTradingBot(
            strategies_path=path,
            state_file=tmp_path / "state.json",
            trade_log=tmp_path / "trades.csv",
            symbol="BTC/USDT:USDT",
            market="spot",
        )


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ({}, "feature must be a non-empty string"),
        ({"feature": "tf_5m_rsi_14", "kind": "unknown", "threshold": 50.0}, "kind is unsupported"),
        ({"feature": "tf_5m_rsi_14", "kind": "value_ge", "threshold": float("nan")}, "threshold must be finite"),
        ({"feature": "tf_5m_rsi_14", "kind": "ratio_ge", "threshold": 1.0}, "feature_b is required"),
    ],
)
def test_bot_init_rejects_invalid_condition_payloads(tmp_path, condition, message):
    path = tmp_path / "active_strategies.json"
    create_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategies"][0]["conditions"] = [condition]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
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


def test_bot_rejects_corrupt_state_file(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="State file is unreadable or invalid"):
        make_bot(bot_env)


def test_bot_rejects_non_object_state_file(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="State file must contain a JSON object"):
        make_bot(bot_env)


def test_bot_rejects_invalid_position_state_shape(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": 5000.0,
                "open_positions": [],
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "last_pnl_reset_date": "2026-06-01",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="open_positions must be an object"):
        make_bot(bot_env)


def test_bot_migrates_missing_state_safety_fields(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps({"equity": 5000.0, "open_positions": {}, "inactive_strategies": []}),
        encoding="utf-8",
    )

    bot = make_bot(bot_env)

    assert bot.state["equity"] == 5000.0
    assert bot.state["consecutive_losses"] == 0
    assert bot.state["cooldown_until_ts"] == 0.0
    assert bot.state["daily_pnl"] == 0.0
    assert bot.state["daily_trades_by_strategy"] == {}
    assert "last_pnl_reset_date" in bot.state


def test_bot_normalizes_numeric_state_strings(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": "5000.5",
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": "1",
                "cooldown_until_ts": "123.5",
                "daily_pnl": "-0.01",
                "daily_trades_by_strategy": {"5m_long_r1": "2"},
                "last_pnl_reset_date": "2026-01-01",
            }
        ),
        encoding="utf-8",
    )

    bot = make_bot(bot_env)

    assert bot.state["equity"] == 5000.5
    assert bot.state["consecutive_losses"] == 1
    assert bot.state["cooldown_until_ts"] == 123.5
    assert bot.state["daily_pnl"] == -0.01
    assert bot.state["daily_trades_by_strategy"] == {"5m_long_r1": 2}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-date", "State last_pnl_reset_date must be an ISO date string"),
        ("2026-1-1", "State last_pnl_reset_date must be an ISO date string"),
        ("", "State last_pnl_reset_date must be an ISO date string"),
        (123, "State last_pnl_reset_date must be an ISO date string"),
        ("2999-01-01", "State last_pnl_reset_date must not be in the future"),
    ],
)
def test_bot_rejects_invalid_last_pnl_reset_date(bot_env, value, message):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": 5000.0,
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": {},
                "last_pnl_reset_date": value,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        make_bot(bot_env)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("equity", 0.0, "State equity must be positive"),
        ("equity", float("nan"), "State equity must be finite"),
        ("equity", float("inf"), "State equity must be finite"),
        ("consecutive_losses", -1, "State consecutive_losses must be non-negative"),
        ("consecutive_losses", 1.5, "State consecutive_losses must be an integer"),
        ("cooldown_until_ts", -1.0, "State cooldown_until_ts must be non-negative"),
        ("daily_pnl", float("nan"), "State daily_pnl must be finite"),
    ],
)
def test_bot_rejects_invalid_numeric_state_fields(bot_env, key, value, message):
    _, state_file, _ = bot_env
    state = {
        "equity": 5000.0,
        "open_positions": {},
        "inactive_strategies": [],
        "consecutive_losses": 0,
        "cooldown_until_ts": 0.0,
        "daily_pnl": 0.0,
        "daily_trades_by_strategy": {},
        "last_pnl_reset_date": "2026-01-01",
    }
    state[key] = value
    state_file.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        make_bot(bot_env)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "must be non-negative"),
        (1.5, "must be an integer"),
        (float("nan"), "must be finite"),
        ("bad", "must be numeric"),
    ],
)
def test_bot_rejects_invalid_daily_trade_count_values(bot_env, value, message):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": 5000.0,
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": {"5m_long_r1": value},
                "last_pnl_reset_date": "2026-01-01",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        make_bot(bot_env)


def test_bot_rejects_invalid_daily_trade_count_state_shape(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            {
                "equity": 5000.0,
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": [],
                "last_pnl_reset_date": "2026-06-01",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="daily_trades_by_strategy must be an object"):
        make_bot(bot_env)


def _valid_persisted_state_with_open_position(**position_overrides):
    position = {
        "entry_time": last_kline_time_iso(5),
        "direction": "long",
        "entry_price": 100.0,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "sl_price": 98.0,
        "tp_price": 104.0,
        "position_size": 1.0,
    }
    position.update(position_overrides)
    return {
        "equity": 5000.0,
        "open_positions": {"5m_long_r1": position},
        "inactive_strategies": [],
        "consecutive_losses": 0,
        "cooldown_until_ts": 0.0,
        "daily_pnl": 0.0,
        "daily_trades_by_strategy": {},
        "last_pnl_reset_date": "2026-01-01",
    }


def test_bot_normalizes_open_position_numeric_strings(bot_env):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(
            _valid_persisted_state_with_open_position(
                entry_price="100.0",
                sl_pct="0.02",
                tp_pct="0.04",
                sl_price="98.0",
                tp_price="104.0",
                position_size="0.5",
                broker_qty="2.5",
                broker_requested_qty="2.5",
                broker_fill_ratio="1.0",
                broker_entry_fee="0.01",
                broker_entry_price="100.0",
                broker_side="buy",
                broker_symbol="BTCUSDT",
            )
        ),
        encoding="utf-8",
    )

    bot = make_bot(bot_env)
    position = bot.state["open_positions"]["5m_long_r1"]

    assert position["entry_price"] == 100.0
    assert position["position_size"] == 0.5
    assert position["broker_qty"] == 2.5
    assert position["broker_requested_qty"] == 2.5
    assert position["broker_fill_ratio"] == 1.0
    assert position["broker_entry_fee"] == 0.01
    assert position["broker_entry_price"] == 100.0


@pytest.mark.parametrize(
    ("position_update", "message"),
    [
        ({"entry_time": "not-a-time"}, "entry_time must be a valid timestamp"),
        ({"direction": "sideways"}, "direction must be long or short"),
        ({"entry_price": float("nan")}, "entry_price must be finite"),
        ({"sl_pct": 0.0}, "sl_pct must be positive"),
        ({"tp_price": -1.0}, "tp_price must be positive"),
        ({"position_size": 1.5}, "position_size must be <= 1"),
        ({"broker_qty": 0.0}, "broker_qty must be positive"),
        ({"broker_fill_ratio": 1.1}, "broker_fill_ratio must be 1"),
        (
            {"broker_requested_qty": 100.0, "broker_qty": 101.0},
            "broker_qty must match broker_requested_qty",
        ),
        (
            {"broker_requested_qty": 100.0, "broker_qty": 50.0, "broker_fill_ratio": 0.6},
            "broker_fill_ratio must be 1",
        ),
        (
            {"broker_requested_qty": 100.0, "broker_qty": 100.0, "broker_fill_ratio": 0.6},
            "broker_fill_ratio must be 1",
        ),
        ({"broker_entry_fee": -0.01}, "broker_entry_fee must be non-negative"),
        ({"broker_side": "hold"}, "broker_side must be buy or sell"),
        ({"broker_symbol": ""}, "broker_symbol must be non-empty"),
    ],
)
def test_bot_rejects_invalid_open_position_values(bot_env, position_update, message):
    _, state_file, _ = bot_env
    state_file.write_text(
        json.dumps(_valid_persisted_state_with_open_position(**position_update)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        make_bot(bot_env)


def test_bot_rejects_unknown_open_position_strategy(bot_env):
    _, state_file, _ = bot_env
    state = _valid_persisted_state_with_open_position()
    state["open_positions"]["unknown_strategy"] = state["open_positions"].pop("5m_long_r1")
    state_file.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unknown strategy"):
        make_bot(bot_env)


def test_bot_rejects_open_position_above_strategy_position_cap(bot_env):
    strategies_path, state_file, _ = bot_env
    payload = json.loads(strategies_path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"]["max_position_fraction"] = 0.5
    strategies_path.write_text(json.dumps(payload), encoding="utf-8")
    state_file.write_text(
        json.dumps(_valid_persisted_state_with_open_position(position_size=0.75)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="position_size exceeds max_position_fraction"):
        make_bot(bot_env)


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


@pytest.mark.parametrize(
    ("column_index", "value", "message"),
    [
        (4, "not-a-number", "non-finite close"),
        (4, "0", "non-positive close"),
        (5, "-1", "negative volume"),
        (7, "not-a-number", "non-finite quote_asset_volume"),
    ],
)
@patch("src.run_bot.requests.get")
def test_fetch_live_candles_rejects_invalid_closed_numeric_values(
    mock_get,
    bot_env,
    column_index,
    value,
    message,
):
    bot = make_bot(bot_env)
    klines = get_mock_binance_klines(3, close_price=100.0)
    klines[0][column_index] = value
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = klines
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match=message):
        bot.fetch_live_candles("BTCUSDT", "futures", "5m", limit=3)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({2: "98", 3: "99"}, "high below low"),
        ({2: "99", 4: "100"}, "high below open/close"),
        ({3: "101", 4: "100"}, "low above open/close"),
    ],
)
@patch("src.run_bot.requests.get")
def test_fetch_live_candles_rejects_inconsistent_closed_ohlc(mock_get, bot_env, updates, message):
    bot = make_bot(bot_env)
    klines = get_mock_binance_klines(3, open_p=100.0, high=101.0, low=99.0, close_price=100.0)
    for column_index, value in updates.items():
        klines[0][column_index] = value
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = klines
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match=message):
        bot.fetch_live_candles("BTCUSDT", "futures", "5m", limit=3)


@pytest.mark.parametrize(
    "timestamps",
    [
        (BASE_TS_MS, BASE_TS_MS, BASE_TS_MS + 300000),
        (BASE_TS_MS, BASE_TS_MS + 600000, BASE_TS_MS + 300000),
    ],
)
@patch("src.run_bot.requests.get")
def test_fetch_live_candles_rejects_duplicate_or_unsorted_closed_timestamps(mock_get, bot_env, timestamps):
    bot = make_bot(bot_env)
    klines = get_mock_binance_klines(3, open_p=100.0, high=101.0, low=99.0, close_price=100.0)
    for kline, timestamp in zip(klines, timestamps):
        kline[0] = timestamp
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = klines
    mock_get.return_value = mock_resp

    with pytest.raises(RuntimeError, match="strictly increasing timestamps"):
        bot.fetch_live_candles("BTCUSDT", "futures", "5m", limit=3)


@patch("src.run_bot.requests.get")
def test_fetch_live_candles_compacts_ccxt_symbol_for_binance_rest(mock_get, bot_env):
    bot = make_bot(bot_env)
    now = pd.Timestamp.now(tz="UTC")
    current_bar_start = int(now.floor("5min").value // 10**6)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(2, start_time_ms=current_bar_start - 2 * 300000)
    mock_get.return_value = mock_resp

    bot.fetch_live_candles("BTC/USDT:USDT", "futures", "5m", limit=2)

    assert mock_get.call_args.kwargs["params"]["symbol"] == "BTCUSDT"


def test_build_feature_frame_uses_configured_symbol_and_market(bot_env, monkeypatch):
    strategies_path, state_file, trade_log = bot_env
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        symbol="ETH/USDT",
        market="spot",
    )
    calls = []

    def fake_fetch(symbol, market, timeframe, limit=500):
        calls.append((symbol, market, timeframe, limit))
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [10.0] * 5,
                "quote_asset_volume": [1000.0] * 5,
                "number_of_trades": [100] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            }
        )

    monkeypatch.setattr(bot, "fetch_live_candles", fake_fetch)
    monkeypatch.setattr("build_binance_indicator_dataset.build_indicator_features", mock_indicator_features)

    frame, base_close = bot._build_feature_frame(bot.strategies[0])

    assert base_close == pytest.approx(100.0)
    assert calls == [("ETHUSDT", "spot", "5m", 500)]
    assert "tf_5m_close" in frame.columns


def test_build_feature_frame_requests_only_strategy_required_features(bot_env, monkeypatch):
    strategies_path, state_file, trade_log = bot_env
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
    )
    requested = []

    def fake_fetch(symbol, market, timeframe, limit=500):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC"),
                "open": [100.0] * 20,
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": [100.0] * 20,
                "volume": [10.0] * 20,
                "quote_asset_volume": [1000.0] * 20,
                "number_of_trades": [100] * 20,
                "taker_buy_base_volume": [5.0] * 20,
                "taker_buy_quote_volume": [500.0] * 20,
            }
        )

    def fake_build_features(df, timeframe, required_features=None):
        requested.append((timeframe, set(required_features or [])))
        return mock_indicator_features(df, timeframe, rsi_value=60.0, atr_value=1.0)

    monkeypatch.setattr(bot, "fetch_live_candles", fake_fetch)
    monkeypatch.setattr("build_binance_indicator_dataset.build_indicator_features", fake_build_features)

    bot._build_feature_frame(bot.strategies[0])

    assert requested == [("5m", {"open", "high", "low", "close", "rsi_14", "atr"})]


def test_macro_regime_failure_marks_guard_fail_closed(bot_env, monkeypatch):
    bot = PaperTradingBot(
        strategies_path=bot_env[0],
        state_file=bot_env[1],
        trade_log=bot_env[2],
        regime_guard=True,
    )

    def fail_fetch(symbol, market, timeframe, limit=500):
        raise RuntimeError("daily feed unavailable")

    monkeypatch.setattr(bot, "fetch_live_candles", fail_fetch)

    bot._evaluate_macro_regime()

    assert bot._macro_aside is True
    assert bot._macro_detail == {"error": "daily feed unavailable", "fail_closed": True}


@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_blocks_new_entries_when_macro_regime_unavailable(mock_build_ind, bot_env, monkeypatch):
    bot = PaperTradingBot(
        strategies_path=bot_env[0],
        state_file=bot_env[1],
        trade_log=bot_env[2],
        regime_guard=True,
    )
    calls = []

    def fake_fetch(symbol, market, timeframe, limit=500):
        calls.append((symbol, market, timeframe, limit))
        if timeframe == "1d":
            raise RuntimeError("daily feed unavailable")
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [10.0] * 5,
                "quote_asset_volume": [1000.0] * 5,
                "number_of_trades": [100] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            }
        )

    monkeypatch.setattr(bot, "fetch_live_candles", fake_fetch)
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert calls == [("BTCUSDT", "futures", "1d", 500), ("BTCUSDT", "futures", "5m", 500)]


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
    assert bot.state["daily_trades_by_strategy"]["5m_long_r1"] == 1


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_caps_position_size_by_strategy_risk_limit(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    payload = json.loads(strategies_path.read_text(encoding="utf-8"))
    payload["strategies"][0]["risk"]["max_position_fraction"] = 0.25
    strategies_path.write_text(json.dumps(payload), encoding="utf-8")
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    assert bot.state["open_positions"]["5m_long_r1"]["position_size"] == pytest.approx(0.25)


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_blocks_entry_after_strategy_daily_trade_limit(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path, max_trades_per_day=1)
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    bot.state["daily_trades_by_strategy"] = {"5m_long_r1": 1}
    bot._save_state()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {"5m_long_r1": 1}


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_allows_only_one_open_position_per_product(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
    second = dict(artifact["strategies"][0])
    second["id"] = "5m_long_r2"
    artifact["strategies"].append(second)
    strategies_path.write_text(json.dumps(artifact), encoding="utf-8")
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    assert set(bot.state["open_positions"]) == {"5m_long_r1"}
    assert bot.state["daily_trades_by_strategy"] == {"5m_long_r1": 1}
    assert mock_build_ind.call_count == 1


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_reuses_feature_frame_for_shared_strategy_requirements(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
    second = dict(artifact["strategies"][0])
    second["id"] = "5m_long_r2"
    second["conditions"] = [dict(second["conditions"][0])]
    second["conditions"][0]["threshold"] = 30.0
    artifact["strategies"].append(second)
    strategies_path.write_text(json.dumps(artifact), encoding="utf-8")
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=40.0, atr_value=2.0)

    bot.run_cycle()

    assert set(bot.state["open_positions"]) == {"5m_long_r2"}
    assert mock_get.call_count == 1
    assert mock_build_ind.call_count == 1


def test_run_cycle_skips_feature_build_for_new_entries_when_position_open(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
    second = dict(artifact["strategies"][0])
    second["id"] = "5m_long_r2"
    artifact["strategies"].append(second)
    strategies_path.write_text(json.dumps(artifact), encoding="utf-8")
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    bot.state["open_positions"]["5m_long_r1"] = {
        "entry_time": pd.Timestamp.now(tz="UTC").isoformat(),
        "direction": "long",
        "entry_price": 100.0,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "sl_price": 98.0,
        "tp_price": 104.0,
        "position_size": 1.0,
    }
    bot._save_state()

    feature_frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "tf_5m_high": 101.0,
                "tf_5m_low": 99.0,
                "tf_5m_close": 100.0,
            }
        ]
    )

    def build_feature_frame(strategy):
        assert strategy["id"] == "5m_long_r1"
        return feature_frame, 100.0

    with patch.object(bot, "_build_feature_frame", side_effect=build_feature_frame):
        # Inactive blocks new entries only; existing exposure is still managed.
        bot.state["inactive_strategies"] = ["5m_long_r1"]
        bot.run_cycle()

    assert set(bot.state["open_positions"]) == {"5m_long_r1"}


def test_run_cycle_manages_open_position_for_inactive_strategy(tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    bot.state["open_positions"]["5m_long_r1"] = {
        "entry_time": last_kline_time_iso(5),
        "direction": "long",
        "entry_price": 100.0,
        "sl_pct": 0.02,
        "tp_pct": 0.04,
        "sl_price": 98.0,
        "tp_price": 104.0,
        "position_size": 1.0,
    }
    bot.state["inactive_strategies"] = ["5m_long_r1"]
    bot._save_state()
    feature_frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "tf_5m_high": 105.0,
                "tf_5m_low": 99.0,
                "tf_5m_close": 104.0,
            }
        ]
    )

    with patch.object(bot, "_build_feature_frame", return_value=(feature_frame, 104.0)):
        bot.run_cycle()

    assert bot.state["open_positions"] == {}
    df_trades = pd.read_csv(trade_log)
    assert df_trades["strategy_id"].iloc[0] == "5m_long_r1"
    assert df_trades["exit_reason"].iloc[0] == "take_profit"


def test_daily_reset_clears_pnl_and_trade_counts(bot_env):
    bot = make_bot(bot_env)
    bot.state["daily_pnl"] = -0.01
    bot.state["daily_trades_by_strategy"] = {"5m_long_r1": 3}
    bot.state["last_pnl_reset_date"] = "2000-01-01"
    bot._save_state()

    bot.process_daily_reset()

    assert bot.state["daily_pnl"] == 0.0
    assert bot.state["daily_trades_by_strategy"] == {}
    assert bot.state["last_pnl_reset_date"] != "2000-01-01"


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_with_broker_places_entry_order(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env
    px = PriceSource(100.0)
    broker = PaperBroker(price_source=px, starting_balance=10_000, fee_bps=0, slippage_bps=0)
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    bot.run_cycle()

    assert len(broker.fills) == 1
    pos = bot.state["open_positions"]["5m_long_r1"]
    assert pos["broker_qty"] == pytest.approx(100.0)
    assert broker.get_position("BTCUSDT").qty == pytest.approx(100.0)


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_unmanaged_broker_position_before_new_entry(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class UnmanagedPositionBroker:
        name = "unmanaged-position"

        def get_price(self, symbol):
            raise AssertionError("broker price should not be read with unmanaged exposure")

        def get_balance(self):
            raise AssertionError("broker balance should not be read with unmanaged exposure")

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.25, avg_price=100.0)

        def place_order(self, order):
            raise AssertionError("entry order should not be placed with unmanaged exposure")

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=UnmanagedPositionBroker(),
    )

    with pytest.raises(RuntimeError, match="Unexpected broker position"):
        bot.run_cycle()

    mock_get.assert_not_called()
    mock_build_ind.assert_not_called()
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@pytest.mark.parametrize("balance", [0.0, float("nan"), float("inf")])
@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_invalid_broker_balance_before_entry_order(mock_build_ind, mock_get, bot_env, balance):
    strategies_path, state_file, trade_log = bot_env

    class InvalidBalanceBroker:
        name = "invalid-balance"

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return balance

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

        def place_order(self, order):
            raise AssertionError("entry order should not be placed with invalid broker balance")

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=InvalidBalanceBroker(),
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(ValueError, match="Broker quote balance"):
        bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_partial_broker_entry_without_opening_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class PartialEntryBroker:
        name = "partial-entry"

        def __init__(self):
            self.position_qty = 0.0
            self.orders = []

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            self.position_qty = 50.0
            return Fill(order.symbol, order.side, qty=50.0, price=100.0, fee=0.0)

    broker = PartialEntryBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(RuntimeError, match="Broker entry partial fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert broker.orders[0].qty == pytest.approx(100.0)
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_overfilled_broker_entry_without_opening_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class OverfilledEntryBroker:
        name = "overfilled-entry"

        def __init__(self):
            self.orders = []

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

        def place_order(self, order):
            self.orders.append(order)
            return Fill(order.symbol, order.side, qty=101.0, price=100.0, fee=0.0)

    broker = OverfilledEntryBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(RuntimeError, match="Broker entry overfill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_invalid_broker_entry_fill_without_opening_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class InvalidEntryBroker:
        name = "invalid-entry"

        def __init__(self):
            self.orders = []

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

        def place_order(self, order):
            self.orders.append(order)
            return Fill(order.symbol, order.side, qty=0.0, price=100.0, fee=0.0)

    broker = InvalidEntryBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(RuntimeError, match="Broker entry invalid fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qty", float("nan")),
        ("qty", float("inf")),
        ("price", float("nan")),
        ("price", float("inf")),
        ("fee", -0.1),
        ("fee", float("nan")),
        ("fee", float("inf")),
    ],
)
@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_nonfinite_broker_entry_fill_without_opening_state(
    mock_build_ind,
    mock_get,
    bot_env,
    field,
    value,
):
    strategies_path, state_file, trade_log = bot_env

    class NonfiniteEntryBroker:
        name = "nonfinite-entry"

        def __init__(self):
            self.orders = []

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

        def place_order(self, order):
            self.orders.append(order)
            fill = {"qty": 100.0, "price": 100.0, "fee": 0.0}
            fill[field] = value
            return Fill(order.symbol, order.side, **fill)

    broker = NonfiniteEntryBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(RuntimeError, match="Broker entry invalid fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_mismatched_broker_entry_fill_without_opening_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class MismatchedEntryBroker:
        name = "mismatched-entry"

        def __init__(self):
            self.orders = []

        def get_price(self, symbol):
            return 100.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=0.0, avg_price=0.0)

        def place_order(self, order):
            self.orders.append(order)
            return Fill("ETHUSDT", order.side, qty=50.0, price=100.0, fee=0.0)

    broker = MismatchedEntryBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=2.0)

    with pytest.raises(RuntimeError, match="Broker entry fill mismatch"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}
    assert not trade_log.exists()


def test_spot_sell_order_qty_uses_existing_base_position(bot_env):
    strategies_path, state_file, trade_log = bot_env
    broker = SpotPaperBroker(price_source=PriceSource(100.0), starting_balance=10_000, fee_bps=0, slippage_bps=0)
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=2.0))
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )

    qty = bot._broker_order_qty(price=100.0, position_size=0.25, side=OrderSide.SELL)

    assert qty == pytest.approx(0.5)


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_spot_step_aside_entry_records_base_and_quote_budget(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path, direction="short", pnl_unit="btc")
    broker = SpotPaperBroker(price_source=PriceSource(100.0), starting_balance=10_000, fee_bps=0, slippage_bps=0)
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=2.0))
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=1.0,
        broker=broker,
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0, atr_value=4.0)

    bot.run_cycle()

    pos = bot.state["open_positions"]["5m_long_r1"]
    assert pos["direction"] == "short"
    assert pos["broker_qty"] == pytest.approx(1.0)
    assert pos["broker_entry_base_qty_before"] == pytest.approx(2.0)
    assert pos["broker_entry_base_qty_after"] == pytest.approx(1.0)
    assert pos["broker_entry_quote_value"] == pytest.approx(100.0)
    assert pos["broker_exit_sizing"] == "quote_reinvest"


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_spot_step_aside_exit_reinvests_quote_budget_to_accumulate_btc(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path, direction="short", pnl_unit="btc")
    px = PriceSource(100.0)
    broker = SpotPaperBroker(price_source=px, starting_balance=10_000, fee_bps=0, slippage_bps=0)
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=2.0))
    broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.5))
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=1.0,
        broker=broker,
    )
    bot.state["open_positions"]["5m_long_r1"] = {
        "entry_time": last_kline_time_iso(5),
        "direction": "short",
        "entry_price": 100.0,
        "sl_pct": 0.10,
        "tp_pct": 0.10,
        "sl_price": 110.0,
        "tp_price": 90.0,
        "position_size": 0.25,
        "broker_symbol": "BTCUSDT",
        "broker_qty": 0.5,
        "broker_requested_qty": 0.5,
        "broker_fill_ratio": 1.0,
        "broker_side": "sell",
        "broker_entry_price": 100.0,
        "broker_entry_fee": 0.0,
        "broker_entry_base_qty_before": 2.0,
        "broker_entry_base_qty_after": 1.5,
        "broker_entry_quote_value": 50.0,
        "broker_exit_sizing": "quote_reinvest",
    }
    bot._save_state()
    px.price = 90.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=90.0, high=100.0, low=89.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    bot.run_cycle()

    assert bot.state["open_positions"] == {}
    assert broker.fills[-1].side == OrderSide.BUY
    assert broker.fills[-1].qty == pytest.approx(50.0 / 90.0)
    assert broker.get_position("BTCUSDT").qty == pytest.approx(1.5 + (50.0 / 90.0))
    assert broker.get_position("BTCUSDT").qty > 2.0
    df_trades = pd.read_csv(trade_log)
    assert df_trades["broker_exit_qty"].iloc[0] == pytest.approx(50.0 / 90.0)
    assert df_trades["net_return"].iloc[0] > 0


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_spot_step_aside_exit_rejects_missing_quote_budget(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path, direction="short", pnl_unit="btc")
    px = PriceSource(100.0)
    broker = SpotPaperBroker(price_source=px, starting_balance=10_000, fee_bps=0, slippage_bps=0)
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=2.0))
    broker.place_order(Order("BTCUSDT", OrderSide.SELL, qty=0.5))
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=1.0,
        broker=broker,
    )
    bot.state["open_positions"]["5m_long_r1"] = {
        "entry_time": last_kline_time_iso(5),
        "direction": "short",
        "entry_price": 100.0,
        "sl_pct": 0.10,
        "tp_pct": 0.10,
        "sl_price": 110.0,
        "tp_price": 90.0,
        "position_size": 0.25,
        "broker_symbol": "BTCUSDT",
        "broker_qty": 0.5,
        "broker_requested_qty": 0.5,
        "broker_fill_ratio": 1.0,
        "broker_side": "sell",
        "broker_entry_price": 100.0,
        "broker_entry_fee": 0.0,
        "broker_entry_base_qty_before": 2.0,
        "broker_entry_base_qty_after": 1.5,
        "broker_exit_sizing": "quote_reinvest",
    }
    bot._save_state()
    setup_fill_count = len(broker.fills)
    px.price = 90.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=90.0, high=100.0, low=89.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="missing broker_entry_quote_value"):
        bot.run_cycle()

    assert len(broker.fills) == setup_fill_count
    assert "5m_long_r1" in bot.state["open_positions"]
    assert not trade_log.exists()


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
def test_run_cycle_with_broker_reduces_strategy_qty_on_exit(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env
    px = PriceSource(100.0)
    broker = PaperBroker(price_source=px, starting_balance=10_000, fee_bps=0, slippage_bps=0)
    broker.place_order(Order("BTCUSDT", OrderSide.BUY, qty=100.0))
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()
    px.price = 104.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    bot.run_cycle()

    assert len(broker.fills) == 2
    assert broker.get_position("BTCUSDT").is_flat
    df_trades = pd.read_csv(trade_log)
    assert df_trades["broker_exit_qty"].iloc[0] == pytest.approx(100.0)
    assert df_trades["broker_exit_price"].iloc[0] == pytest.approx(104.0)


@pytest.mark.parametrize("broker_qty", [0.0, -1.0, "not-a-number"])
@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_invalid_broker_qty_state_before_broker_use(
    mock_build_ind,
    mock_get,
    bot_env,
    broker_qty,
):
    strategies_path, state_file, trade_log = bot_env

    class UnusedBroker:
        name = "unused"

        def get_price(self, symbol):
            raise AssertionError("broker price should not be read with invalid local broker_qty")

        def get_balance(self):
            raise AssertionError("broker balance should not be read with invalid local broker_qty")

        def get_position(self, symbol):
            raise AssertionError("broker position should not be read with invalid local broker_qty")

        def place_order(self, order):
            raise AssertionError("broker order should not be placed with invalid local broker_qty")

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=UnusedBroker(),
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=broker_qty,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker state invalid"):
        bot.run_cycle()

    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == broker_qty
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_with_broker_rejects_open_position_without_broker_metadata(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class UnusedBroker:
        name = "unused"

        def get_price(self, symbol):
            raise AssertionError("broker price should not be read without broker metadata")

        def get_balance(self):
            raise AssertionError("broker balance should not be read without broker metadata")

        def get_position(self, symbol):
            raise AssertionError("broker position should not be read without broker metadata")

        def place_order(self, order):
            raise AssertionError("broker order should not be placed without broker metadata")

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=UnusedBroker(),
    )
    open_position_state(bot, last_kline_time_iso(5))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="broker metadata is required"):
        bot.run_cycle()

    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_incomplete_broker_metadata_before_broker_use(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class UnusedBroker:
        name = "unused"

        def get_price(self, symbol):
            raise AssertionError("broker price should not be read with incomplete broker metadata")

        def get_balance(self):
            raise AssertionError("broker balance should not be read with incomplete broker metadata")

        def get_position(self, symbol):
            raise AssertionError("broker position should not be read with incomplete broker metadata")

        def place_order(self, order):
            raise AssertionError("broker order should not be placed with incomplete broker metadata")

    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=UnusedBroker(),
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="broker metadata missing required key"):
        bot.run_cycle()

    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_partial_broker_exit_without_closing_local_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class PartialExitBroker:
        name = "partial-exit"

        def __init__(self):
            self.position_qty = 100.0
            self.orders = []

        def get_price(self, symbol):
            return 104.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            self.position_qty = 50.0
            return Fill(order.symbol, order.side, qty=50.0, price=104.0, fee=0.0)

    broker = PartialExitBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker exit partial fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == pytest.approx(100.0)
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_overfilled_broker_exit_without_closing_local_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class OverfilledExitBroker:
        name = "overfilled-exit"

        def __init__(self):
            self.position_qty = 100.0
            self.orders = []

        def get_price(self, symbol):
            return 104.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            self.position_qty = 0.0
            return Fill(order.symbol, order.side, qty=101.0, price=104.0, fee=0.0)

    broker = OverfilledExitBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker exit overfill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == pytest.approx(100.0)
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_invalid_broker_exit_fill_without_closing_local_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class InvalidExitBroker:
        name = "invalid-exit"

        def __init__(self):
            self.position_qty = 100.0
            self.orders = []

        def get_price(self, symbol):
            return 104.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            self.position_qty = 0.0
            return Fill(order.symbol, order.side, qty=100.0, price=0.0, fee=0.0)

    broker = InvalidExitBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker exit invalid fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == pytest.approx(100.0)
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qty", float("nan")),
        ("qty", float("inf")),
        ("price", float("nan")),
        ("price", float("inf")),
        ("fee", -0.1),
        ("fee", float("nan")),
        ("fee", float("inf")),
    ],
)
@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_nonfinite_broker_exit_fill_without_closing_local_state(
    mock_build_ind,
    mock_get,
    bot_env,
    field,
    value,
):
    strategies_path, state_file, trade_log = bot_env

    class NonfiniteExitBroker:
        name = "nonfinite-exit"

        def __init__(self):
            self.position_qty = 100.0
            self.orders = []

        def get_price(self, symbol):
            return 104.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            fill = {"qty": 100.0, "price": 104.0, "fee": 0.0}
            fill[field] = value
            return Fill(order.symbol, order.side, **fill)

    broker = NonfiniteExitBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker exit invalid fill"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == pytest.approx(100.0)
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_rejects_mismatched_broker_exit_fill_without_closing_local_state(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env

    class MismatchedExitBroker:
        name = "mismatched-exit"

        def __init__(self):
            self.position_qty = 100.0
            self.orders = []

        def get_price(self, symbol):
            return 104.0

        def get_balance(self):
            return 10_000.0

        def get_position(self, symbol):
            return Position(symbol=symbol, qty=self.position_qty, avg_price=100.0)

        def place_order(self, order):
            self.orders.append(order)
            self.position_qty = 0.0
            wrong_side = OrderSide.BUY if order.side == OrderSide.SELL else OrderSide.SELL
            return Fill(order.symbol, wrong_side, qty=100.0, price=104.0, fee=0.0)

    broker = MismatchedExitBroker()
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=100.0,
        broker_requested_qty=100.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=105.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker exit fill mismatch"):
        bot.run_cycle()

    assert len(broker.orders) == 1
    assert bot.state["open_positions"]["5m_long_r1"]["broker_qty"] == pytest.approx(100.0)
    assert bot.state["equity"] == pytest.approx(10_000.0)
    assert not trade_log.exists()


def test_trade_log_schema_upgrades_when_broker_rows_follow_paper_rows(bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    bot.state["equity"] = 10_010.0
    bot._log_trade(
        "paper_strategy",
        "2026-07-08T00:00:00+00:00",
        "2026-07-08T00:05:00+00:00",
        "long",
        100.0,
        101.0,
        "time",
        0.01,
        0.0094,
        0.0094,
        1.0,
    )

    bot.state["equity"] = 10_050.0
    bot._log_trade(
        "broker_strategy",
        "2026-07-08T00:10:00+00:00",
        "2026-07-08T00:15:00+00:00",
        "long",
        100.0,
        104.0,
        "take_profit",
        0.04,
        0.0394,
        0.0394,
        1.0,
        broker_exit_fill=Fill("BTCUSDT", OrderSide.SELL, qty=0.5, price=104.0, fee=0.01),
    )

    df_trades = pd.read_csv(trade_log)
    assert len(df_trades) == 2
    assert "broker_exit_qty" in df_trades.columns
    assert pd.isna(df_trades.loc[0, "broker_exit_qty"])
    assert df_trades.loc[1, "broker_symbol"] == "BTCUSDT"
    assert df_trades.loc[1, "broker_exit_qty"] == pytest.approx(0.5)
    assert df_trades.loc[1, "broker_exit_price"] == pytest.approx(104.0)


def test_trade_log_schema_keeps_broker_columns_when_paper_rows_follow_broker_rows(bot_env):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    bot.state["equity"] = 10_040.0
    bot._log_trade(
        "broker_strategy",
        "2026-07-08T00:00:00+00:00",
        "2026-07-08T00:05:00+00:00",
        "short",
        100.0,
        96.0,
        "take_profit",
        0.041666,
        0.041066,
        0.020533,
        0.5,
        broker_exit_fill=Fill("BTCUSDT", OrderSide.BUY, qty=0.25, price=96.0, fee=0.01),
    )

    bot.state["equity"] = 10_050.0
    bot._log_trade(
        "paper_strategy",
        "2026-07-08T00:10:00+00:00",
        "2026-07-08T00:15:00+00:00",
        "long",
        100.0,
        101.0,
        "time",
        0.01,
        0.0094,
        0.0094,
        1.0,
    )

    df_trades = pd.read_csv(trade_log)
    assert len(df_trades) == 2
    assert df_trades.loc[0, "broker_exit_qty"] == pytest.approx(0.25)
    assert pd.isna(df_trades.loc[1, "broker_exit_qty"])
    assert pd.isna(df_trades.loc[1, "broker_symbol"])


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_with_broker_reconciliation_failure(mock_build_ind, mock_get, bot_env):
    strategies_path, state_file, trade_log = bot_env
    broker = PaperBroker(price_source=PriceSource(100.0), starting_balance=10_000, fee_bps=0, slippage_bps=0)
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
        broker=broker,
    )
    open_position_state(bot, last_kline_time_iso(5))
    bot.state["open_positions"]["5m_long_r1"].update(
        broker_symbol="BTCUSDT",
        broker_qty=1.0,
        broker_requested_qty=1.0,
        broker_fill_ratio=1.0,
        broker_side="buy",
        broker_entry_price=100.0,
        broker_entry_fee=0.0,
    )
    bot._save_state()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=101.0, high=101.0, low=99.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf)

    with pytest.raises(RuntimeError, match="Broker position mismatch"):
        bot.run_cycle()


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
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=97.0, high=99.0, low=96.0, open_p=97.0)
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


@patch("src.run_bot.requests.get")
@patch("build_binance_indicator_dataset.build_indicator_features")
def test_run_cycle_uses_strictest_daily_stop_across_strategies(mock_build_ind, mock_get, tmp_path):
    strategies_path = tmp_path / "active_strategies.json"
    state_file = tmp_path / "bot_state.json"
    trade_log = tmp_path / "paper_trades.csv"
    create_artifact(strategies_path)
    artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
    second = dict(artifact["strategies"][0])
    second["id"] = "5m_long_r2"
    second["risk"] = dict(second["risk"])
    second["risk"]["daily_stop_loss"] = -0.02
    artifact["strategies"].append(second)
    strategies_path.write_text(json.dumps(artifact), encoding="utf-8")
    bot = PaperTradingBot(
        strategies_path=strategies_path,
        state_file=state_file,
        trade_log=trade_log,
        starting_equity=10_000,
    )
    bot.state["daily_pnl"] = -0.03
    bot._save_state()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = get_mock_binance_klines(5, close_price=100.0)
    mock_get.return_value = mock_resp
    mock_build_ind.side_effect = lambda df, tf: mock_indicator_features(df, tf, rsi_value=60.0)

    bot.run_cycle()

    assert bot._account_risk()["daily_stop_loss"] == -0.02
    assert bot.state["open_positions"] == {}
    assert bot.state["daily_trades_by_strategy"] == {}


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


def test_check_drift_rejects_symlink_trade_log_swapped_after_start(bot_env, tmp_path):
    _, _, trade_log = bot_env
    bot = make_bot(bot_env)
    target = tmp_path / "external_trades.csv"
    target.write_text("strategy_id,net_return\n5m_long_r1,-0.01\n", encoding="utf-8")
    trade_log.unlink(missing_ok=True)
    trade_log.symlink_to(target)

    with pytest.raises(RuntimeError, match="Trade log must not be a symlink"):
        bot.check_drift_and_ood(bot.strategies[0])

    assert trade_log.is_symlink()
    assert target.read_text(encoding="utf-8") == "strategy_id,net_return\n5m_long_r1,-0.01\n"
    assert bot.state["inactive_strategies"] == []


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
