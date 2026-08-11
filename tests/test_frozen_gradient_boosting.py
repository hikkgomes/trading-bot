import copy

import numpy as np
import pandas as pd
import pytest

from src.alpha.frozen_gradient_boosting import (
    FrozenGradientBoostingModel,
    export_sklearn_gradient_boosting,
)
from src.strategies.library.ml_classifier import MlClassifierStrategy
from src.strategies.library.ml_regressor import MlRegressorStrategy


def _frame():
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1, 10, 300),
            "feature_x": rng.normal(size=300),
        }
    )


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_frozen_gradient_boosting_matches_sklearn_predictions(kind):
    frame = _frame()
    common = {
        "model": "sklearn",
        "horizon": 2,
        "n_estimators": 20,
        "max_depth": 2,
        "feature_cols": ["volume", "feature_x"],
        "max_features": 2,
    }
    strategy = (
        MlClassifierStrategy(**common) if kind == "classifier" else MlRegressorStrategy(**common)
    ).fit(frame.iloc[:250])
    frozen = FrozenGradientBoostingModel.from_dict(export_sklearn_gradient_boosting(strategy))
    rows = frame.iloc[250:260]

    expected = (
        strategy._model.predict_proba(rows[strategy._features])[:, 1]
        if kind == "classifier"
        else strategy._model.predict(rows[strategy._features])
    )
    actual = [frozen.prediction(row.to_dict()) for _, row in rows.iterrows()]

    assert actual == pytest.approx(expected, abs=1e-12)


def test_frozen_gradient_boosting_rejects_non_array_tree_payload():
    frame = _frame()
    strategy = MlRegressorStrategy(
        model="sklearn",
        horizon=2,
        n_estimators=20,
        max_depth=2,
        feature_cols=["volume", "feature_x"],
        max_features=2,
    ).fit(frame.iloc[:250])
    payload = copy.deepcopy(export_sklearn_gradient_boosting(strategy))
    payload["trees"][0]["value"] = 1.0

    with pytest.raises(ValueError, match="tree arrays are invalid"):
        FrozenGradientBoostingModel.from_dict(payload)
