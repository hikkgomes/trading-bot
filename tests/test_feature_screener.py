import builtins

import numpy as np
import pandas as pd

from src.feature_screener import screen_features, screen_features_per_scenario


def test_screen_features_respects_max_features():
    n = 1000
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    train["label"] = (train["x1"] > 0).astype(int)
    ranked = screen_features(
        train, "label", ["x1", "x2", "x3"], max_features=2, method="importance"
    )
    assert len(ranked) == 2


def test_screen_features_changes_with_train_frame_deterministic():
    n = 1200
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    a["label"] = (a["x1"] > 0).astype(int)
    b = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    b["label"] = (b["x2"] > 0).astype(int)
    ranked_a = screen_features(a, "label", ["x1", "x2", "x3"], max_features=1, method="importance")
    ranked_b = screen_features(b, "label", ["x1", "x2", "x3"], max_features=1, method="importance")
    assert ranked_a[0] == "x1"
    assert ranked_b[0] == "x2"


def test_screen_features_per_scenario():
    n = 1000
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["label_long_tp50_sl30_h8"] = (df["x1"] > 0).astype(int)
    out = screen_features_per_scenario(
        df, [(0.005, 0.003)], [8], ["long"], ["x1", "x2"], max_features=1
    )
    assert "long|h8|tp50|sl30" in out
    assert len(out["long|h8|tp50|sl30"]) == 1


def test_shap_fallback_logs_warning(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no shap")
        return real_import(name, *args, **kwargs)

    n = 1000
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["label"] = (df["x1"] > 0).astype(int)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    caplog.set_level("WARNING")
    _ = screen_features(df, "label", ["x1", "x2"], max_features=1, method="shap")
    assert "SHAP not available, falling back to gain-based importance" in caplog.text
