"""Machine-learning entry signal: a classifier predicting next-horizon direction.

Uses LightGBM when available (it is in requirements.txt), otherwise falls back
to scikit-learn's GradientBoostingClassifier so the strategy still runs in a
minimal environment. The model predicts P(forward return over ``horizon`` bars
> 0); longs fire above ``long_threshold`` and shorts below ``short_threshold``.

Discipline note: call ``fit`` on a *training* slice, then ``generate_signals``
on a later slice — never fit on the data you score (the framework CLI does this
split for you). If ``generate_signals`` is called before ``fit`` it will fit
in-sample and emit a warning; that path is for smoke-testing only.
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
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1,
            )
        except ImportError:
            if kind == "lightgbm":
                raise
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=42)


@register
class MlClassifierStrategy(Strategy):
    name = "ml_classifier"
    description = "Gradient-boosted classifier on feature columns predicting next-horizon direction."

    @classmethod
    def default_params(cls):
        return {
            "horizon": 96,
            "long_threshold": 0.55,
            "short_threshold": 0.45,
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

    # -- feature / target plumbing -----------------------------------------
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
        fwd = close.shift(-int(self.params["horizon"])) / close - 1.0
        return (fwd > 0).astype(int).where(fwd.notna())

    def fit(self, df: pd.DataFrame) -> "MlClassifierStrategy":
        self._features = self._select_features(df)
        if not self._features:
            raise ValueError("No usable feature columns found to train the ML strategy.")
        y = self._target(df)
        X = df[self._features]
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() < 50:
            raise ValueError(f"Too few training rows ({int(valid.sum())}) for the ML strategy.")
        self._model = _make_model(self.params["model"])
        self._model.fit(X[valid], y[valid].to_numpy())
        LOGGER.info("Fitted %s on %d rows, %d features", self.name, int(valid.sum()), len(self._features))
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            warnings.warn("ml_classifier.generate_signals called before fit(); fitting in-sample "
                          "(smoke-test only — results are leaked).", stacklevel=2)
            self.fit(df)
        X = df[self._features]
        proba = pd.Series(np.nan, index=df.index)
        valid = X.notna().all(axis=1)
        if valid.any():
            proba.loc[valid] = self._model.predict_proba(X[valid])[:, 1]
        sig = self._empty_signals(df)
        sig[proba > self.params["long_threshold"]] = 1
        if self.params["allow_short"]:
            sig[proba < self.params["short_threshold"]] = -1
        return sig
