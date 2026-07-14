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

import numpy as np
import pandas as pd

from src.strategies.base import BacktestConfig, Strategy
from src.strategies.library.ml_common import screen_feature_columns, triple_barrier_direction_target
from src.strategies.registry import register

LOGGER = logging.getLogger(__name__)


def _make_model(kind: str):
    if kind in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            if kind == "lightgbm":
                raise
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=42)


@register
class MlClassifierStrategy(Strategy):
    name = "ml_classifier"
    description = (
        "Gradient-boosted classifier on feature columns predicting next-horizon direction."
    )

    @classmethod
    def default_params(cls):
        return {
            "horizon": 96,
            "long_threshold": 0.55,
            "short_threshold": 0.45,
            "allow_short": True,
            "model": "auto",  # "auto" | "lightgbm" | "sklearn"
            "feature_cols": None,  # None = auto-select numeric feature columns
            "max_features": 80,
            "feature_screen": "spearman",
            "min_feature_corr": 0.0,
            "label_mode": "direction",  # "direction" | "triple_barrier"
            "label_tp": None,
            "label_sl": None,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig(take_profit=0.05, stop_loss=0.03, horizon_bars=96)

    def __init__(self, **params):
        super().__init__(**params)
        self._model = None
        self._features: list[str] | None = None

    # -- feature / target plumbing -----------------------------------------
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> list[str]:
        return screen_feature_columns(
            df,
            target,
            feature_cols=self.params["feature_cols"],
            max_features=int(self.params["max_features"]),
            method=str(self.params["feature_screen"]),
            min_abs_corr=float(self.params["min_feature_corr"]),
        )

    def _target(self, df: pd.DataFrame) -> pd.Series:
        horizon = int(self.params["horizon"])
        if self.params["label_mode"] == "triple_barrier":
            cfg = self.default_config()
            return triple_barrier_direction_target(
                self.ohlcv(df),
                df.index,
                horizon=horizon,
                take_profit=float(self.params["label_tp"] or cfg.take_profit),
                stop_loss=float(self.params["label_sl"] or cfg.stop_loss),
            )
        if self.params["label_mode"] != "direction":
            raise ValueError(f"Unsupported ml_classifier label_mode: {self.params['label_mode']!r}")
        close = pd.Series(self.ohlcv(df).close, index=df.index)
        fwd = close.shift(-horizon) / close - 1.0
        return (fwd > 0).astype(int).where(fwd.notna())

    def fit(self, df: pd.DataFrame) -> MlClassifierStrategy:
        y = self._target(df)
        self._features = self._select_features(df, y)
        if not self._features:
            raise ValueError("No usable feature columns found to train the ML strategy.")
        X = df[self._features].replace([np.inf, -np.inf], np.nan)
        valid = y.notna() & np.isfinite(y) & X.notna().all(axis=1)
        if valid.sum() < 50:
            raise ValueError(f"Too few training rows ({int(valid.sum())}) for the ML strategy.")
        self._model = _make_model(self.params["model"])
        self._model.fit(X[valid], y[valid].to_numpy())
        LOGGER.info(
            "Fitted %s on %d rows, %d features", self.name, int(valid.sum()), len(self._features)
        )
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            warnings.warn(
                "ml_classifier.generate_signals called before fit(); fitting in-sample "
                "(smoke-test only — results are leaked).",
                stacklevel=2,
            )
            self.fit(df)
        X = df[self._features].replace([np.inf, -np.inf], np.nan)
        proba = pd.Series(np.nan, index=df.index)
        valid = X.notna().all(axis=1)
        if valid.any():
            proba.loc[valid] = self._model.predict_proba(X[valid])[:, 1]
        sig = self._empty_signals(df)
        sig[proba > self.params["long_threshold"]] = 1
        if self.params["allow_short"]:
            sig[proba < self.params["short_threshold"]] = -1
        return sig
