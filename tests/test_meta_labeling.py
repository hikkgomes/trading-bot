import pandas as pd

import src.meta_labeling as meta_labeling
from src.discover_patterns import Condition
from src.meta_labeling import _label_column
from src.strategy_search import StrategyCandidate, _conditions_payload


def test_label_column_matches_trade_label_schema():
    assert _label_column("long", 8, 0.003, 0.002) == "label_long_tp30_sl20_h8"


def test_meta_labeling_passes_fee_cost(tmp_path, monkeypatch):
    captured = {}

    def fake_walk_forward_model_signals(data, features, label_column, wf, tp, sl, fee_cost, max_features=80):
        captured["fee_cost"] = fee_cost
        return pd.DataFrame({"timestamp": data["timestamp"], "signal": [False] * len(data), "prob": [0.5] * len(data), "ev": [0.0] * len(data)})

    monkeypatch.setattr(meta_labeling, "walk_forward_model_signals", fake_walk_forward_model_signals)
    rules_path = tmp_path / "rules.csv"
    labels_path = tmp_path / "labels.parquet"
    output_path = tmp_path / "signals.parquet"
    candidate = StrategyCandidate("long", 4, (Condition("signal", "value_ge", 1, "signal >= 1"),))
    pd.DataFrame([
        {
            "direction": "long",
            "horizon_bars": 4,
            "take_profit": 0.003,
            "stop_loss": 0.002,
            "conditions_json": _conditions_payload(candidate),
        }
    ]).to_csv(rules_path, index=False)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
            "tf_15m_close": range(20),
            "tf_15m_rsi": range(20),
            "signal": [1] * 20,
            "label_long_tp30_sl20_h4": [1] * 20,
        }
    )
    data.to_parquet(labels_path, index=False)
    meta_labeling.run(rules_path, labels_path, output_path, top_k=1, min_rule_rows=1, fee_bps=5.0, slippage_bps=2.0)
    assert captured["fee_cost"] == 0.0014
