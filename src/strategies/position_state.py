"""Causal next-open entry pulses with shared stop, profit and horizon exits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PositionRules:
    stop_loss: float
    take_profit: float | None
    horizon_bars: int
    execution_policy: str = "resting_bracket/v1"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0 < value < 1
            for value in (self.stop_loss, self.take_profit)
            if value is not None
        ):
            raise ValueError("position stop and profit fractions must be in (0, 1)")
        if self.stop_loss is None or self.execution_policy != "resting_bracket/v1":
            raise ValueError("position requires a declared stop and supported execution policy")
        if (
            isinstance(self.horizon_bars, bool)
            or not isinstance(self.horizon_bars, int)
            or self.horizon_bars < 1
        ):
            raise ValueError("position horizon must be positive")

    @classmethod
    def from_model(cls, model: Mapping[str, Any]) -> PositionRules:
        return cls(
            stop_loss=model["stop_loss"],
            take_profit=model.get("take_profit"),
            horizon_bars=model["horizon_bars"],
            execution_policy=model["execution_policy"],
        )


def directional_position_model(horizon_bars: int) -> dict[str, Any]:
    return {
        **PositionRules(0.02, None, horizon_bars).__dict__,
        "kind": "entry_pulse",
        "flat_action": "hold",
        "decision_timing": "next_open",
        "maximum_position": 0.1,
        "reentry": "next_bar_after_exit",
    }


@dataclass(frozen=True)
class PositionState:
    direction: int = 0
    entry_price: float = 0.0
    age: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.direction, bool) or self.direction not in {-1, 0, 1}:
            raise ValueError("position direction must be -1, 0 or 1")
        if isinstance(self.age, bool) or not isinstance(self.age, int) or self.age < 0:
            raise ValueError("position age must be a non-negative integer")
        if not math.isfinite(self.entry_price) or (self.direction and self.entry_price <= 0):
            raise ValueError("open position requires a positive finite entry price")


@dataclass(frozen=True)
class PositionAction:
    bar_index: int
    kind: str
    direction: int
    price: float
    reason: str
    ambiguous_bar: bool = False


@dataclass(frozen=True)
class PositionPath:
    decisions: tuple[int, ...]
    actions: tuple[PositionAction, ...]
    state: PositionState


def position_step(
    state: PositionState,
    *,
    entry_signal: int,
    bar: Mapping[str, Any],
    index: int,
    rules: PositionRules,
) -> tuple[PositionState, tuple[PositionAction, ...]]:
    prices = {key: float(bar[key]) for key in ("open", "high", "low", "close")}
    if not all(math.isfinite(value) and value > 0 for value in prices.values()):
        raise ValueError("position state requires positive finite OHLC prices")
    actions = []
    if state.direction and state.age >= rules.horizon_bars:
        return PositionState(), (
            PositionAction(index, "exit", state.direction, prices["open"], "horizon"),
        )
    if state.direction == 0 and entry_signal:
        state = PositionState(entry_signal, prices["open"], 0)
        actions.append(PositionAction(index, "entry", entry_signal, prices["open"], "entry_pulse"))
    if state.direction == 0:
        return state, tuple(actions)
    state = PositionState(state.direction, state.entry_price, state.age + 1)
    exit = _exit_price(state, prices, rules)
    if exit is not None:
        price, reason = exit
        ambiguous = reason == "stop" and _profit_crossed(state, prices, rules)
        actions.append(PositionAction(index, "exit", state.direction, price, reason, ambiguous))
        state = PositionState()
    return state, tuple(actions)


def _exit_price(
    state: PositionState, prices: Mapping[str, float], rules: PositionRules
) -> tuple[float, str] | None:
    sign = state.direction
    stop = state.entry_price * (1 - sign * rules.stop_loss)
    adverse = prices["low"] if sign > 0 else prices["high"]
    if sign * (adverse - stop) <= 0:
        return (min(prices["open"], stop) if sign > 0 else max(prices["open"], stop)), "stop"
    if rules.take_profit is not None and _profit_crossed(state, prices, rules):
        profit = state.entry_price * (1 + sign * rules.take_profit)
        return (
            max(prices["open"], profit) if sign > 0 else min(prices["open"], profit)
        ), "take_profit"
    return None


def _profit_crossed(
    state: PositionState, prices: Mapping[str, float], rules: PositionRules
) -> bool:
    if rules.take_profit is None:
        return False
    profit = state.entry_price * (1 + state.direction * rules.take_profit)
    favourable = prices["high"] if state.direction > 0 else prices["low"]
    return state.direction * (favourable - profit) >= 0


def position_decision(
    state: PositionState, actions: Sequence[PositionAction], entry_signal: int, rules: PositionRules
) -> int:
    if any(action.kind == "exit" for action in actions) or state.age >= rules.horizon_bars:
        return 0
    return state.direction or entry_signal


def resume_position_path(
    rows: Sequence[Mapping[str, Any]], signal: int, rules: PositionRules, prior: Mapping[str, Any]
) -> PositionPath:
    if prior.get("exited_current_interval"):
        return PositionPath((0,), (), PositionState())
    state, actions = position_step(
        PositionState(**prior["state"]),
        entry_signal=int(prior["entry_signal"]),
        bar=rows[-1],
        index=len(rows) - 1,
        rules=rules,
    )
    return PositionPath((position_decision(state, actions, signal, rules),), actions, state)


def position_path(
    rows: Sequence[Mapping[str, Any]], signals: Sequence[int], rules: PositionRules
) -> PositionPath:
    if len(rows) != len(signals):
        raise ValueError("entry pulses must be aligned with market bars")
    state = PositionState()
    actions: list[PositionAction] = []
    decisions: list[int] = []
    for index, bar in enumerate(rows):
        state, current = position_step(
            state,
            entry_signal=decisions[-1] if decisions else 0,
            bar=bar,
            index=index,
            rules=rules,
        )
        actions.extend(current)
        decisions.append(position_decision(state, current, signals[index], rules))
    return PositionPath(tuple(decisions), tuple(actions), state)
