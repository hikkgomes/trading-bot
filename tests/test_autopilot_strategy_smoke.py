import pandas as pd

from src.autopilot.strategy_smoke import run_strategy_smoke


def _regime_frame(rows=900):
    index = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series(range(rows), dtype=float) + 30_000.0
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "high": (close + 10).to_numpy(),
            "low": (close - 10).to_numpy(),
            "close": close.to_numpy(),
            "volume": 1.0,
            "tf_1d_regime_id": [0, 1, 2, 3] * (rows // 4) + [0] * (rows % 4),
        }
    )


def test_strategy_smoke_runs_synthetic_and_regime_scenarios(tmp_path):
    regime_path = tmp_path / "regime.parquet"
    _regime_frame().to_parquet(regime_path, index=False)

    report = run_strategy_smoke(
        synthetic_rows=700,
        regime_input=regime_path,
        max_regime_rows=800,
    )

    assert report["ok"] is True
    scenarios = {scenario["name"]: scenario for scenario in report["scenarios"]}
    assert scenarios["synthetic_strategy_sweep"]["rows"] > 0
    assert scenarios["regime_filter_sweep"]["scored_rows"] == 800
    assert scenarios["regime_filter_sweep"]["rows"] == 4


def test_strategy_smoke_skips_missing_regime_input(tmp_path):
    report = run_strategy_smoke(
        synthetic_rows=700,
        regime_input=tmp_path / "missing.parquet",
    )

    assert report["ok"] is True
    regime = next(
        scenario for scenario in report["scenarios"] if scenario["name"] == "regime_filter_sweep"
    )
    assert regime["skipped"] is True
    assert regime["reason"] == "missing_regime_input"
