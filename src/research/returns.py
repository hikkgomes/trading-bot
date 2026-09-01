"""Signed position returns with explicit turnover and financing costs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


class ReturnLedgerError(ValueError):
    """A return-ledger input is missing or not finite."""


def _series(value: Iterable[Any], *, field: str) -> tuple[float, ...]:
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ReturnLedgerError(f"{field}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ReturnLedgerError(f"{field}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ReturnLedgerError(f"{field}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _rate(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ReturnLedgerError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReturnLedgerError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ReturnLedgerError(f"{field} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class PositionReturnReport:
    """One signed, cost-adjusted return series for a strategy position."""

    positions: tuple[float, ...]
    market_returns: tuple[float, ...]
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    turnover: float
    fees: float
    slippage: float
    funding_pnl: float
    gross_pnl: float
    net_pnl: float
    maximum_drawdown: float
    period_turnover: tuple[float, ...] = ()
    period_fees: tuple[float, ...] = ()
    period_slippage: tuple[float, ...] = ()
    period_funding_pnl: tuple[float, ...] = ()

    @property
    def effective_observations(self) -> int:
        return len(self.net_returns)

    @property
    def funding_cost(self) -> float:
        return max(0.0, -self.funding_pnl)

    @property
    def cost_adjusted_return(self) -> float:
        return self.net_pnl


@dataclass(frozen=True)
class PositionReturnLedger:
    """Calculate signed position PnL once for all research execution paths."""

    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    funding_rate: float = 0.0

    def __post_init__(self) -> None:
        _rate(self.fee_rate, field="fee_rate")
        _rate(self.slippage_rate, field="slippage_rate")
        if not math.isfinite(float(self.funding_rate)):
            raise ReturnLedgerError("funding_rate must be finite")

    def measure(
        self,
        positions: Sequence[Any] | Iterable[Any],
        market_returns: Sequence[Any] | Iterable[Any],
        *,
        funding_rates: float | Sequence[Any] | Iterable[Any] | None = None,
        initial_position: Any | None = None,
    ) -> PositionReturnReport:
        position_values = _series(positions, field="positions")
        return_values = _series(market_returns, field="market_returns")
        aligned = min(len(return_values), max(0, len(position_values) - 1))
        held_positions = position_values[:aligned]
        held_returns = return_values[:aligned]
        gross_returns = tuple(
            position * market_return
            for position, market_return in zip(held_positions, held_returns, strict=False)
        )
        period_turnover_values = [
            abs(position_values[index + 1] - position_values[index]) for index in range(aligned)
        ]
        if initial_position is not None and aligned:
            initial_values = _series((initial_position,), field="initial_position")
            period_turnover_values[0] += abs(position_values[0] - initial_values[0])
        period_turnover = tuple(period_turnover_values)
        turnover = sum(period_turnover)
        fee_rate = _rate(self.fee_rate, field="fee_rate")
        slippage_rate = _rate(self.slippage_rate, field="slippage_rate")
        period_fees = tuple(value * fee_rate for value in period_turnover)
        period_slippage = tuple(value * slippage_rate for value in period_turnover)
        fees = sum(period_fees)
        slippage = sum(period_slippage)
        funding_values = self._funding_values(funding_rates, aligned)
        period_funding_pnl = tuple(
            -position * rate for position, rate in zip(held_positions, funding_values, strict=False)
        )
        funding_pnl = sum(period_funding_pnl)
        net_returns = tuple(
            gross - fee - slip + funding
            for gross, fee, slip, funding in zip(
                gross_returns,
                period_fees,
                period_slippage,
                period_funding_pnl,
                strict=True,
            )
        )
        net_pnl = sum(net_returns)
        maximum_drawdown = self._maximum_drawdown(net_returns)
        return PositionReturnReport(
            positions=held_positions,
            market_returns=held_returns,
            gross_returns=gross_returns,
            net_returns=tuple(net_returns),
            turnover=turnover,
            fees=fees,
            slippage=slippage,
            funding_pnl=funding_pnl,
            gross_pnl=sum(gross_returns),
            net_pnl=net_pnl,
            maximum_drawdown=maximum_drawdown,
            period_turnover=period_turnover,
            period_fees=period_fees,
            period_slippage=period_slippage,
            period_funding_pnl=period_funding_pnl,
        )

    def _funding_values(
        self,
        funding_rates: float | Sequence[Any] | Iterable[Any] | None,
        count: int,
    ) -> tuple[float, ...]:
        if funding_rates is None:
            rate = float(self.funding_rate)
            return (rate,) * count
        if isinstance(funding_rates, int | float) and not isinstance(funding_rates, bool):
            rate = float(funding_rates)
            if not math.isfinite(rate):
                raise ReturnLedgerError("funding_rate must be finite")
            return (rate,) * count
        if isinstance(funding_rates, bool):
            raise ReturnLedgerError("funding_rates must be numeric or iterable")
        values = _series(funding_rates, field="funding_rates")
        if len(values) < count:
            raise ReturnLedgerError("funding_rates must cover every return observation")
        return values[:count]

    @staticmethod
    def _maximum_drawdown(values: Sequence[float]) -> float:
        equity = 1.0
        peak = equity
        maximum = 0.0
        for value in values:
            equity += value
            peak = max(peak, equity)
            maximum = max(maximum, peak - equity)
        return maximum
