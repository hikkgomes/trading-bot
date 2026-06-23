"""Machine-learning regression signal: predict the *magnitude* of forward return.

Where ``ml_classifier`` predicts direction probability, this predicts the
expected forward return over ``horizon`` bars and only trades when the predicted
edge clears ``min_edge`` (a return threshold), so weak-conviction bars are
skipped. Uses LightGBM when available, else scikit-learn GradientBoosting.

Discipline note: ``fit`` on a training slice, ``generate_signals`` on a later
slice — never fit on the data you score (the framework CLI does this split).
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register

LOGGER = logging.getLogger(__name__)

_EXCLUDE_SUBSTRINGS = ("timestamp", "future_return", "label_", "bars_to_exit", "target")
_OHLCV_SUFFIXES = ("_open", "_high", "_low", "_close", "_volume")


def _make_model(kind: str):
    if kind in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
            )
        except ImportError:
            if kind == "lightgbm":
                raise
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(random_state=42)


@register
class MlRegressorStrategy(Strategy):
    name = "ml_regressor"
    description = "Gradient-boosted regressor on feature columns; trade when predicted edge > min_edge."

    @classmethod
    def default_params(cls):
        return {
            "horizon": 96,
            "min_edge": 0.004,        # predicted forward return needed to trade
            "allow_short": True,
            "model": "auto",          # "auto" | "lightgbm" | "sklearn"
            "feature_cols": None,      # None = auto-select numeric feature columns
            "max_features": 80,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=96)

    def __init__(self, **params):
        super().__init__(**params)
        self._model = None
        self._features: Optional[List[str]] = None

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        if self.params["feature_cols"]:
            return [c for c in self.params["feature_cols"] if c in df.columns]
        numeric = df.select_dtypes(include=[np.number]).columns
        cols = [
            c for c in numeric
            if not any(sub in c for sub in _EXCLUDE_SUBSTRINGS)
            and not c.endswith(_OHLCV_SUFFIXES)
            and c not in ("open", "high", "low", "close", "volume")
        ]
        return cols[: int(self.params["max_features"])]

    def _target(self, df: pd.DataFrame) -> pd.Series:
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        return close.shift(-int(self.params["horizon"])) / close - 1.0

    def fit(self, df: pd.DataFrame) -> "MlRegressorStrategy":
        self._features = self._select_features(df)
        if not self._features:
            raise ValueError("No usable feature columns found to train the ML regressor.")
        y = self._target(df)
        X = df[self._features]
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() < 50:
            raise ValueError(f"Too few training rows ({int(valid.sum())}) for the ML regressor.")
        self._model = _make_model(self.params["model"])
        self._model.fit(X[valid], y[valid].to_numpy())
        LOGGER.info("Fitted %s on %d rows, %d features", self.name, int(valid.sum()), len(self._features))
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            warnings.warn("ml_regressor.generate_signals called before fit(); fitting in-sample "
                          "(smoke-test only — results are leaked).", stacklevel=2)
            self.fit(df)
        X = df[self._features]
        pred = pd.Series(np.nan, index=df.index)
        valid = X.notna().all(axis=1)
        if valid.any():
            pred.loc[valid] = self._model.predict(X[valid])
        edge = float(self.params["min_edge"])
        sig = self._empty_signals(df)
        sig[pred > edge] = 1
        if self.params["allow_short"]:
            sig[pred < -edge] = -1
        return sig
