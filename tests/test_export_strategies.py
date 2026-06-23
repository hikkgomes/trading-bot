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
    }
    config.update(config_extra or {})
    (search_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(rows).to_csv(search_dir / "ranked_strategies.csv", index=False)
    return search_dir


def make_row(passes=True, expectancy=0.002, win_rate=0.6, direction="long"):
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
    }


def test_build_payload_exports_passing_strategy(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row()])
    payload = build_payload(search_dir, top_k=3)
    assert payload["version"] == 1
    assert payload["search_git_sha"] == "searchsha"
    assert len(payload["strategies"]) == 1
    strategy = payload["strategies"][0]
    assert strategy["base_timeframe"] == "5m"
    assert strategy["direction"] == "long"
    assert strategy["horizon_bars"] == 8
    assert strategy["risk"]["daily_stop_loss"] == -0.02
    assert strategy["fees"]["fee_bps"] == 5.0
    assert strategy["baseline_win_rate"] == 0.6
    assert strategy["conditions"][0]["feature"] == "tf_5m_rsi_14"


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


def test_run_writes_artifact(tmp_path):
    search_dir = write_search_dir(tmp_path, [make_row(), make_row(direction="short")])
    output = tmp_path / "active_strategies.json"
    path = run(search_dir, output, top_k=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["strategies"]) == 1
    assert payload["strategies"][0]["id"].endswith("_r1")
