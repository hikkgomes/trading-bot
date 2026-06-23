import pandas as pd
import pytest

from src.discover_patterns import Condition
from src.optimize_topk import run
from src.strategy_search import StrategyCandidate, _conditions_payload


def test_optimize_topk_writes_output(tmp_path):
    pytest.importorskip("optuna")
    rules = tmp_path / "rules.csv"
    out = tmp_path / "out.csv"
    data_path = tmp_path / "data.parquet"
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC"),
            "tf_15m_open": [100.0] * 40,
            "tf_15m_high": [101.0] * 40,
            "tf_15m_low": [99.0] * 40,
            "tf_15m_close": [100.0] * 40,
            "signal": [1] * 40,
        }
    )
    data.to_parquet(data_path, index=False)
    candidate = StrategyCandidate("long", 4, (Condition("signal", "value_ge", 1, "signal >= 1"),))
    pd.DataFrame([
        {
            "direction": "long",
            "take_profit": 0.003,
            "stop_loss": 0.002,
            "horizon_bars": 4,
            "conditions_json": _conditions_payload(candidate),
        }
    ]).to_csv(rules, index=False)
    run(rules, out, top_k=1, trials=2, input_path=data_path)
    assert out.exists()
    result = pd.read_csv(out)
    assert "optuna_score" in result.columns
