import json

import pandas as pd
import pytest

from src.export_strategies import build_payload, run


def write_search_dir(tmp_path, rows, config_extra=None):
    search_dir = tmp_path / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "git_sha": "searchsha",
        "search_timestamp": "2026-06-10T00:00:00+00:00",
        "base_timeframe": "5m",
        "use_atr_tp_sl": False,
        "fee_bps": 5.0,
        "slippage_bps": 2.0,
        "risk_per_trade": 0.003,
        "daily_stop_loss": -0.02,
        "max_consecutive_losses": 3,
        "cooldown_bars": 24,
        "max_position_fraction": 0.25,
    }
    config.update(config_extra or {})
    (search_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(rows).to_csv(search_dir / "ranked_strategies.csv", index=False)
    return search_dir


def make_row(passes=True, expectancy=0.002, win_rate=0.6, direction="long",
             holdout_return=0.05):
    condition = {
        "feature": "tf_5m_rsi_14",
        "kind": "value_le",
        "threshold": 30.0,
        "description": "rsi low",
    }
    return {
        "direction": direction,
        "horizon_bars": 8,
        "take_profit": 0.008,
        "stop_loss": 0.004,
        "rule": "rsi low",
        "conditions_json": json.dumps([condition]),
        "passes_filters": passes,
        "wf_expectancy": expectancy,
        "wf_pass_rate": 0.8,
        "dsr": 0.95,
        "train_win_rate": win_rate,
        "holdout_total_return": holdout_return,
    }


def test_build_payload_exports_passing_strategy(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row()])
    payload = build_payload(search_dir, top_k=3)
    assert payload["version"] == 1
    assert payload["search_git_sha"] == "searchsha"
    assert payload["market"] == "futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["pnl_unit"] == "usdt"
    assert payload["paper_trade_allowed"] is True
    assert payload["live_allowed"] is True
    assert payload["promotion_eligible"] is True
    assert len(payload["strategies"]) == 1
    strategy = payload["strategies"][0]
    assert strategy["market"] == "futures"
    assert strategy["symbol"] == "BTCUSDT"
    assert strategy["base_timeframe"] == "5m"
    assert strategy["direction"] == "long"
    assert strategy["horizon_bars"] == 8
    assert strategy["risk"]["daily_stop_loss"] == -0.02
    assert strategy["risk"]["max_position_fraction"] == 0.25
    assert strategy["risk"]["max_trades_per_day"] == 4
    assert strategy["fees"]["fee_bps"] == 5.0
    assert strategy["baseline_win_rate"] == 0.6
    assert strategy["conditions"][0]["feature"] == "tf_5m_rsi_14"


def test_build_payload_prefers_clustered_ranked_file(tmp_path):
    raw_best = make_row(direction="long")
    clustered_representative = make_row(direction="short")
    search_dir = write_search_dir(tmp_path, [raw_best])
    pd.DataFrame([clustered_representative]).to_csv(
        search_dir / "ranked_strategies_clustered.csv", index=False
    )

    payload = build_payload(search_dir, top_k=3)

    assert payload["source_ranked_file"] == "ranked_strategies_clustered.csv"
    assert payload["strategies"][0]["direction"] == "short"


def test_build_payload_can_use_raw_ranked_file(tmp_path):
    raw_best = make_row(direction="long")
    clustered_representative = make_row(direction="short")
    search_dir = write_search_dir(tmp_path, [raw_best])
    pd.DataFrame([clustered_representative]).to_csv(
        search_dir / "ranked_strategies_clustered.csv", index=False
    )

    payload = build_payload(search_dir, top_k=3, prefer_clustered=False)

    assert payload["source_ranked_file"] == "ranked_strategies.csv"
    assert payload["strategies"][0]["direction"] == "long"


def test_build_payload_rejects_failing_rows(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(passes=False)])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(search_dir)


def test_build_payload_rejects_negative_expectancy(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(expectancy=-0.001)])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(search_dir)


def test_build_payload_min_dsr_gate(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row()])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(search_dir, min_dsr=0.99)


def test_build_payload_rejects_invalid_top_k(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row()])
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        build_payload(search_dir, top_k=0)


@pytest.mark.parametrize(
    ("config_extra", "message"),
    [
        ({"base_timeframe": ""}, "base_timeframe must be a non-empty string"),
        ({"symbol": ""}, "symbol must be a non-empty string"),
        ({"pnl_unit": "eth"}, "pnl_unit must be 'btc' or 'usdt'"),
        ({"market": "margin"}, "market must be 'spot' or 'futures'"),
        ({"pnl_unit": "usdt", "market": "spot"}, "spot strategy exports must use pnl_unit 'btc'"),
        ({"pnl_unit": "btc", "market": "futures"}, "futures strategy exports must use pnl_unit 'usdt'"),
        ({"risk_per_trade": 0.0}, "risk_per_trade must be positive"),
        ({"daily_stop_loss": 0.0}, "daily_stop_loss must be negative"),
        ({"max_consecutive_losses": 1.5}, "max_consecutive_losses must be an integer"),
        ({"cooldown_bars": -1}, "cooldown_bars must be non-negative"),
        ({"max_position_fraction": 1.5}, "max_position_fraction must be > 0 and <= 1"),
        ({"max_trades_per_day": 0}, "max_trades_per_day must be positive"),
        ({"fee_bps": -0.1}, "fee_bps must be non-negative"),
        ({"slippage_bps": float("nan")}, "slippage_bps must be finite"),
    ],
)
def test_build_payload_rejects_invalid_config_metadata(tmp_path, config_extra, message):
    search_dir = write_search_dir(tmp_path, [make_row()], config_extra=config_extra)
    with pytest.raises(ValueError, match=message):
        build_payload(search_dir)


@pytest.mark.parametrize(
    ("row_extra", "message"),
    [
        ({"direction": "flat"}, "direction must be 'long' or 'short'"),
        ({"rule": ""}, "rule must be a non-empty string"),
        ({"horizon_bars": 1.5}, "horizon_bars must be an integer"),
        ({"take_profit": float("nan")}, "take_profit must be finite"),
        ({"stop_loss": 0.0}, "stop_loss must be positive"),
        ({"conditions_json": "[]"}, "conditions_json must decode to a non-empty list"),
    ],
)
def test_build_payload_rejects_invalid_strategy_rows(tmp_path, row_extra, message):
    row = make_row()
    row.update(row_extra)
    search_dir = write_search_dir(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        build_payload(search_dir)


def test_build_payload_gates_on_negative_holdout(tmp_path):
    # The documented flaw of the old pipeline: strategies that lost on their own
    # untouched holdout still shipped. The gate is now on by default.
    search_dir = write_search_dir(tmp_path, [make_row(holdout_return=-0.02)])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(search_dir)


def test_build_payload_gates_on_missing_holdout_values(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(holdout_return=None)])
    with pytest.raises(ValueError, match="No exportable strategies"):
        build_payload(search_dir)


def test_build_payload_missing_holdout_column_raises(tmp_path):
    row = make_row()
    del row["holdout_total_return"]
    search_dir = write_search_dir(tmp_path, [row])
    with pytest.raises(ValueError, match="no holdout_total_return column"):
        build_payload(search_dir)


def test_build_payload_holdout_gate_can_be_disabled(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(holdout_return=-0.02)])
    payload = build_payload(search_dir, min_holdout_return=None)
    assert len(payload["strategies"]) == 1


def test_build_payload_never_exports_zero_baseline(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(win_rate=0.0)])
    payload = build_payload(search_dir)
    assert payload["strategies"][0]["baseline_win_rate"] is None


def test_build_payload_defaults_to_15m_for_position_search(tmp_path):
    row = make_row()
    search_dir = write_search_dir(tmp_path, [row])
    config = json.loads((search_dir / "config.json").read_text())
    del config["base_timeframe"]
    (search_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    payload = build_payload(search_dir)
    assert payload["strategies"][0]["base_timeframe"] == "15m"


def test_build_payload_stamps_symbol_from_search_config(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row()], config_extra={"symbol": "ETHUSDT"})

    payload = build_payload(search_dir)

    assert payload["symbol"] == "ETHUSDT"
    assert payload["strategies"][0]["symbol"] == "ETHUSDT"


def test_build_payload_marks_btc_holdout_as_excess_vs_buy_hold(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(holdout_return=0.04)], config_extra={"pnl_unit": "btc"})

    payload = build_payload(search_dir)
    metrics = payload["strategies"][0]["metrics"]

    assert payload["strategies"][0]["pnl_unit"] == "btc"
    assert payload["market"] == "spot"
    assert payload["strategies"][0]["market"] == "spot"
    assert metrics["holdout_total_return"] == 0.04
    assert metrics["holdout_buy_hold_return"] == 0.0
    assert metrics["holdout_excess_return_vs_buy_hold"] == 0.04


def test_run_writes_artifact(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(), make_row(direction="short")])
    output = tmp_path / "active_strategies.json"
    path = run(search_dir, output, top_k=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["strategies"]) == 1
    assert payload["strategies"][0]["id"].endswith("_r1")
