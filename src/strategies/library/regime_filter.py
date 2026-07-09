"""Regime-conditional wrapper for any registered strategy.

Use this to test whether an otherwise generic rule only works in specific
market states already tagged by ``src.regime.add_regime_column``.
"""

from __future__ import annotations

import json

import pandas as pd

from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import get, register


def _parse_regime_ids(value) -> set[int]:
    if value is None or value == "":
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, float) and value.is_integer():
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    text = str(value).strip()
    if not text:
        return set()
    if text.startswith("["):
        return {int(item) for item in json.loads(text)}
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _parse_child_params(value) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(json.loads(str(value)))


@register
class RegimeFilterStrategy(Strategy):
    name = "regime_filter"
    description = "Run another strategy only when tf_1d_regime_id is in a selected set."

    @classmethod
    def default_params(cls):
        return {
            "strategy": "sma_cross",
            "regime_ids": "",
            "regime_column": "tf_1d_regime_id",
            "child_params": None,
        }

    @classmethod
    def default_config(cls) -> BacktestConfig:
        return BacktestConfig()

    def __init__(self, **params):
        super().__init__(**params)
        child_name = str(self.params["strategy"])
        if child_name == self.name:
            raise ValueError("regime_filter cannot wrap itself.")
        child_cls = get(child_name)
        self._child = child_cls(**_parse_child_params(self.params["child_params"]))
        self._regime_ids = _parse_regime_ids(self.params["regime_ids"])

    def resolved_default_config(self) -> BacktestConfig:
        return self._child.default_config()

    def fit(self, df: pd.DataFrame) -> RegimeFilterStrategy:
        self._child.base_tf = self.base_tf
        self._child.fit(df)
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        regime_column = str(self.params["regime_column"])
        if regime_column not in df.columns:
            raise ValueError(f"Missing regime column {regime_column!r}. Run src.regime first.")
        if not self._regime_ids:
            raise ValueError("regime_filter requires at least one regime id.")
        self._child.base_tf = self.base_tf
        signals = self._child.generate_signals(df).copy()
        allowed = df[regime_column].isin(self._regime_ids)
        signals.loc[~allowed] = 0
        return signals.astype(int)
