"""Causal next-open entry pulses with shared stop, profit and horizon exits."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PositionRules:
    stop_loss: float
    take_profit: float
    horizon_bars: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0 < value < 1 for value in (self.stop_loss, self.take_profit)
        ):
            raise ValueError("position stop and profit fractions must be in (0, 1)")
        if isinstance(self.horizon_bars, bool) or self.horizon_bars < 1:
            raise ValueError("position horizon must be positive")


@dataclass(frozen=True)
class PositionState:
    direction: int = 0
    entry_price: float = 0.0
    age: int = 0


@dataclass(frozen=True)
class PositionAction:
    bar_index: int
    kind: str
    direction: int
    price: float
    reason: str


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
    if state.direction == 0 and entry_signal:
        state = PositionState(entry_signal, prices["open"], 0)
        actions.append(PositionAction(index, "entry", entry_signal, prices["open"], "entry_pulse"))
    if state.direction == 0:
        return state, tuple(actions)
    state = PositionState(state.direction, state.entry_price, state.age + 1)
    exit = _exit_price(state, prices, rules)
    if exit is not None:
        price, reason = exit
        actions.append(PositionAction(index, "exit", state.direction, price, reason))
        state = PositionState()
    return state, tuple(actions)


def _exit_price(
    state: PositionState, prices: Mapping[str, float], rules: PositionRules
) -> tuple[float, str] | None:
    sign = state.direction
    stop = state.entry_price * (1 - sign * rules.stop_loss)
    profit = state.entry_price * (1 + sign * rules.take_profit)
    adverse = prices["low"] if sign > 0 else prices["high"]
    favourable = prices["high"] if sign > 0 else prices["low"]
    if sign * (adverse - stop) <= 0:
        return (min(prices["open"], stop) if sign > 0 else max(prices["open"], stop)), "stop"
    if sign * (favourable - profit) >= 0:
        return profit, "take_profit"
    if state.age >= rules.horizon_bars:
        return prices["close"], "horizon"
    return None


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
            entry_signal=signals[index - 1] if index else 0,
            bar=bar,
            index=index,
            rules=rules,
        )
        actions.extend(current)
        decisions.append(
            0
            if any(action.kind == "exit" for action in current)
            else state.direction or signals[index]
        )
    return PositionPath(tuple(decisions), tuple(actions), state)
