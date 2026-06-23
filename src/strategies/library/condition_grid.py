"""Adapter that runs a condition-grid rule (from the search / active_strategies)
through the unified backtester.

This is the bridge between the existing search→export→bot path and the new
strategy framework: any exported strategy in ``active_strategies*.json`` can be
re-backtested, stress-tested or compared against framework strategies with the
*same* engine. The entry rule is the AND of all conditions; a trade is opened on
the bar where the combined mask flips from False to True.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from src.discover_patterns import Condition, condition_mask
from src.strategies.base import BacktestConfig, Strategy
from src.strategies.registry import register


@register
class ConditionGridStrategy(Strategy):
    name = "condition_grid"
    description = "Runs a discover_patterns Condition rule (e.g. from active_strategies.json)."

    @classmethod
    def default_params(cls):
        return {"conditions": [], "direction": "long"}

    def __init__(self, conditions: Sequence | None = None, direction: str = "long", **params):
        super().__init__(**params)
        conds = conditions if conditions is not None else self.params.get("conditions", [])
        self.conditions: List[Condition] = [
            c if isinstance(c, Condition) else Condition(**c) for c in conds
        ]
        self.direction = direction or self.params.get("direction", "long")
        self.params["direction"] = self.direction

    @classmethod
    def from_active_strategies(cls, path: str | Path, strategy_id: str | None = None):
        """Build adapters from an exported active_strategies*.json file.

        Returns a list of (config, strategy) so each rule keeps its own TP/SL.
        """
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        out = []
        for entry in artifact.get("strategies", []):
            if strategy_id and entry.get("id") != strategy_id:
                continue
            strat = cls(conditions=entry["conditions"], direction=entry["direction"])
            strat.name = f"condition_grid:{entry.get('id', '?')}"
            fees = entry.get("fees", {})
            cfg = BacktestConfig(
                fee_bps=fees.get("fee_bps", 10.0),
                slippage_bps=fees.get("slippage_bps", 2.0),
                take_profit=entry.get("take_profit", 0.05),
                stop_loss=entry.get("stop_loss", 0.03),
                horizon_bars=entry.get("horizon_bars", 96),
                pnl_unit=artifact.get("pnl_unit", "usdt"),
            )
            out.append((cfg, strat))
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = self._empty_signals(df)
        if not self.conditions:
            return sig
        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            mask &= condition_mask(df, cond).reindex(df.index).fillna(False)
        # Entry only on the False -> True transition (avoid re-firing every bar).
        entry = mask & ~mask.shift(1, fill_value=False)
        sig[entry] = 1 if self.direction == "long" else -1
        return sig
