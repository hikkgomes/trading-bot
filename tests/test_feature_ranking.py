import numpy as np
import pandas as pd

from src.feature_ranking import (
    rank_features_blended,
    rank_features_by_importance,
)


def _make_dataset(n_rows=2000, n_features=20):
    np.random.seed(42)
    data = {}
    for i in range(n_features):
        data[f"feat_{i}"] = np.random.randn(n_rows)
    target = sum(data[f"feat_{i}"] * (0.5 if i < 3 else 0.01) for i in range(n_features))
    target += np.random.randn(n_rows) * 0.1
    data["target"] = target
    return pd.DataFrame(data)


def test_rank_features_by_importance_returns_top_n():
    df = _make_dataset()
    features = [c for c in df.columns if c != "target"]

    ranked = rank_features_by_importance(df, features, "target", max_features=5)

    assert len(ranked) == 5
    assert all(f in features for f in ranked)


def test_rank_features_by_importance_surfaces_important_features():
    df = _make_dataset()
    features = [c for c in df.columns if c != "target"]

    ranked = rank_features_by_importance(df, features, "target", max_features=5)

    top_set = set(ranked)
    assert "feat_0" in top_set or "feat_1" in top_set or "feat_2" in top_set


def test_rank_features_blended_returns_top_n():
    df = _make_dataset()
    features = [c for c in df.columns if c != "target"]

    ranked = rank_features_blended(df, features, "target", max_features=5)

    assert len(ranked) == 5
    assert all(f in features for f in ranked)


def test_rank_features_blended_with_direction():
    df = _make_dataset()
    features = [c for c in df.columns if c != "target"]

    ranked = rank_features_blended(
        df,
        features,
        "target",
        max_features=5,
        direction="short",
    )

    assert len(ranked) == 5
