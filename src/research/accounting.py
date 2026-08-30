"""Product-specific accounting for research and forward evidence.

The research executor must report the same economic objective that the product
trades.  BTC accumulation is measured in BTC against passive BTC holding;
active income is measured in USDT with signed futures mark-to-market PnL.
The inputs are immutable event and mark records, not strategy-produced return
series.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.domain._codec import canonical_hash, timestamp


class ProductAccountingError(ValueError):
    """A product accounting input is missing, invalid, or inconsistent."""


def _number(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ProductAccountingError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProductAccountingError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        bound = f" at least {minimum:g}" if minimum is not None else ""
        raise ProductAccountingError(f"{field_name} must be finite{bound}")
    return result


def _event_time(event: Mapping[str, Any], *, index: int) -> str:
    value = event.get("occurred_at", event.get("timestamp", event.get("time")))
    if value is None:
        raise ProductAccountingError(f"event {index} has no timestamp")
    return timestamp(str(value), field=f"events[{index}].timestamp")


def _event_kind(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type", event.get("type", "fill"))).strip().casefold()


def _events(value: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    materialised: list[tuple[str, Mapping[str, Any]]] = []
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise ProductAccountingError(f"event {index} must be an object")
        materialised.append((_event_time(event, index=index), dict(event)))
    return tuple(sorted(materialised, key=lambda item: item[0]))


def _price(event: Mapping[str, Any], *, field_name: str = "price") -> float:
    for key in (field_name, "execution_price", "mark_price", "close", "stablecoin_per_btc"):
        if event.get(key) is not None:
            return _number(event[key], field_name=key, minimum=0.0)
    raise ProductAccountingError(f"event has no {field_name}")


def _quantity(event: Mapping[str, Any]) -> float:
    for key in ("quantity_btc", "base_quantity", "quantity", "filled_quantity"):
        if event.get(key) is not None:
            return _number(event[key], field_name=key, minimum=0.0)
    raise ProductAccountingError("trade event has no base quantity")


def _fee(event: Mapping[str, Any]) -> float:
    for key in ("fee", "fee_amount", "commission"):
        if event.get(key) is not None:
            return _number(event[key], field_name=key, minimum=0.0)
    return 0.0


@dataclass(frozen=True)
class BtcAccountingSnapshot:
    observed_at: str
    btc_balance: float
    stablecoin_balance: float
    stablecoin_per_btc: float
    btc_nav: float
    passive_btc_nav: float
    regime: str = "unclassified"


@dataclass(frozen=True)
class BtcAccountingReport:
    initial_btc_nav: float
    final_btc_nav: float
    passive_btc_nav: float
    excess_btc: float
    return_fraction: float
    fees_btc: float
    time_outside_btc_fraction: float
    stablecoin_exposure_fraction: float
    missed_btc_appreciation: float
    cycles: int
    regime_pnl: Mapping[str, float]
    nav_series: tuple[BtcAccountingSnapshot, ...]
    event_receipts: tuple[Mapping[str, Any], ...] = ()

    @property
    def btc_vs_passive_hold(self) -> float:
        return self.excess_btc

    @property
    def final_nav(self) -> float:
        return self.final_btc_nav

    @property
    def objective_unit(self) -> str:
        return "BTC"


class BtcAccumulationAccounting:
    """Account spot BTC trades in BTC units against passive holding."""

    def evaluate(
        self,
        *,
        trade_events: Iterable[Mapping[str, Any]] = (),
        marks: Iterable[Mapping[str, Any]] | Mapping[str, Any] = (),
        initial_btc: float = 0.0,
        initial_stablecoin: float = 0.0,
        initial_price: float | None = None,
        reserve_fraction: float | None = None,
    ) -> BtcAccountingReport:
        btc = _number(initial_btc, field_name="initial_btc", minimum=0.0)
        stable = _number(initial_stablecoin, field_name="initial_stablecoin", minimum=0.0)
        trades = _events(trade_events)
        mark_events = self._normalise_marks(marks)
        if not mark_events and not trades:
            raise ProductAccountingError("BTC accounting requires trade events or marks")
        first_price = initial_price
        if first_price is None:
            first_price = mark_events[0][1] if mark_events else _price(trades[0][1])
        first_price = _number(first_price, field_name="initial_price", minimum=0.0)
        if first_price <= 0:
            raise ProductAccountingError("initial_price must be positive")
        if reserve_fraction is not None:
            if _number(reserve_fraction, field_name="reserve_fraction", minimum=0.0) > 1:
                raise ProductAccountingError("reserve_fraction must be at most 1")
        initial_nav = btc + stable / first_price
        passive = initial_nav
        all_times = sorted({time for time, _ in trades} | {time for time, _, _ in mark_events})
        trade_index = 0
        mark_index = 0
        current_price = first_price
        snapshots: list[BtcAccountingSnapshot] = []
        fee_btc = 0.0
        missed = 0.0
        seconds_total = 0.0
        seconds_outside = 0.0
        cycles = 0
        saw_sell = False
        previous_snapshot: BtcAccountingSnapshot | None = None
        current_regime = "unclassified"
        receipts: list[Mapping[str, Any]] = []

        for observed_at in all_times:
            while trade_index < len(trades) and trades[trade_index][0] == observed_at:
                _, event = trades[trade_index]
                side = str(event.get("side", "")).casefold()
                if side not in {"buy", "sell"}:
                    raise ProductAccountingError("BTC trade side must be buy or sell")
                price = _price(event)
                quantity = _quantity(event)
                fee = _fee(event)
                fee_asset = str(event.get("fee_asset", event.get("commission_asset", "USDT"))).upper()
                quote = quantity * price
                if side == "buy":
                    if fee_asset == "BTC":
                        btc += quantity - fee
                        stable -= quote
                    else:
                        btc += quantity
                        stable -= quote + fee
                    if btc < -1e-12 or stable < -1e-12:
                        raise ProductAccountingError("BTC buy exceeds available balance")
                    if saw_sell:
                        cycles += 1
                        saw_sell = False
                else:
                    if fee_asset == "BTC":
                        btc -= quantity + fee
                        stable += quote
                    else:
                        btc -= quantity
                        stable += quote - fee
                    if btc < -1e-12 or stable < -1e-12:
                        raise ProductAccountingError("BTC sell exceeds available balance")
                    saw_sell = True
                fee_btc += fee if fee_asset == "BTC" else fee / price
                receipts.append(
                    {
                        "event_hash": canonical_hash(dict(event)),
                        "occurred_at": observed_at,
                        "side": side,
                        "quantity_btc": quantity,
                        "price": price,
                        "fee_btc": fee if fee_asset == "BTC" else fee / price,
                        "fee_asset": fee_asset,
                    }
                )
                trade_index += 1
            while mark_index < len(mark_events) and mark_events[mark_index][0] == observed_at:
                _, mark_price, mark_event = mark_events[mark_index]
                current_price = mark_price
                current_regime = str(mark_event.get("regime", "unclassified"))
                mark_index += 1
            nav = btc + stable / current_price
            snapshot = BtcAccountingSnapshot(
                observed_at=observed_at,
                btc_balance=max(0.0, btc),
                stablecoin_balance=max(0.0, stable),
                stablecoin_per_btc=current_price,
                btc_nav=nav,
                passive_btc_nav=passive,
                regime=current_regime,
            )
            if previous_snapshot is not None:
                start = dt.datetime.fromisoformat(previous_snapshot.observed_at)
                end = dt.datetime.fromisoformat(observed_at)
                seconds = max(0.0, (end - start).total_seconds())
                seconds_total += seconds
                outside = (
                    previous_snapshot.stablecoin_balance / previous_snapshot.stablecoin_per_btc
                )
                if previous_snapshot.btc_nav > 0 and outside > 1e-12:
                    seconds_outside += seconds
                if current_price > previous_snapshot.stablecoin_per_btc and outside > 0:
                    missed += outside - previous_snapshot.stablecoin_balance / current_price
            snapshots.append(snapshot)
            previous_snapshot = snapshot

        if not snapshots:
            snapshots.append(
                BtcAccountingSnapshot(
                    observed_at=trades[0][0] if trades else mark_events[0][0],
                    btc_balance=btc,
                    stablecoin_balance=stable,
                    stablecoin_per_btc=current_price,
                    btc_nav=btc + stable / current_price,
                    passive_btc_nav=passive,
                )
            )
        latest = snapshots[-1]
        regime_pnl: dict[str, float] = {}
        for previous, current in zip(snapshots, snapshots[1:], strict=False):
            regime_pnl[previous.regime] = regime_pnl.get(previous.regime, 0.0) + (
                current.btc_nav - previous.btc_nav
            )
        return BtcAccountingReport(
            initial_btc_nav=initial_nav,
            final_btc_nav=latest.btc_nav,
            passive_btc_nav=passive,
            excess_btc=latest.btc_nav - passive,
            return_fraction=(latest.btc_nav / initial_nav - 1.0 if initial_nav > 0 else 0.0),
            fees_btc=fee_btc,
            time_outside_btc_fraction=(
                seconds_outside / seconds_total if seconds_total > 0 else 0.0
            ),
            stablecoin_exposure_fraction=(
                latest.stablecoin_balance / latest.stablecoin_per_btc / latest.btc_nav
                if latest.btc_nav > 0
                else 0.0
            ),
            missed_btc_appreciation=max(0.0, missed),
            cycles=cycles,
            regime_pnl=regime_pnl,
            nav_series=tuple(snapshots),
            event_receipts=tuple(receipts),
        )

    @staticmethod
    def _normalise_marks(
        marks: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    ) -> tuple[tuple[str, float, Mapping[str, Any]], ...]:
        if isinstance(marks, Mapping):
            source = tuple(
                {"timestamp": key, "price": value} for key, value in marks.items()
            )
        else:
            source = tuple(marks)
        result: list[tuple[str, float, Mapping[str, Any]]] = []
        for index, mark in enumerate(source):
            if not isinstance(mark, Mapping):
                raise ProductAccountingError(f"marks[{index}] must be an object")
            observed_at = _event_time(mark, index=index)
            price = _price(mark, field_name="price")
            if price <= 0:
                raise ProductAccountingError("BTC mark price must be positive")
            result.append((observed_at, price, dict(mark)))
        return tuple(sorted(result, key=lambda item: item[0]))


@dataclass(frozen=True)
class FuturesAccountingReport:
    initial_equity: float
    final_equity: float
    net_pnl: float
    return_fraction: float
    realised_pnl: float
    unrealised_pnl: float
    fees: float
    funding_pnl: float
    spread_cost: float
    slippage_cost: float
    fills: int
    partial_fills: int
    capacity_violations: int
    max_leverage: float
    max_margin_fraction: float
    liquidation: bool
    effective_observations: int
    event_receipts: tuple[Mapping[str, Any], ...] = ()

    @property
    def objective_unit(self) -> str:
        return "USDT"

    @property
    def cost_adjusted_return(self) -> float:
        return self.net_pnl


class FuturesIncomeAccounting:
    """Account signed linear futures fills, funding, margin, and liquidation."""

    def evaluate(
        self,
        *,
        events: Iterable[Mapping[str, Any]],
        initial_cash: float,
        leverage: float = 1.0,
        maintenance_margin_fraction: float = 0.0,
        max_participation_fraction: float = 1.0,
    ) -> FuturesAccountingReport:
        starting_cash = _number(initial_cash, field_name="initial_cash", minimum=0.0)
        cash = starting_cash
        leverage = _number(leverage, field_name="leverage", minimum=1e-12)
        maintenance = _number(
            maintenance_margin_fraction,
            field_name="maintenance_margin_fraction",
            minimum=0.0,
        )
        participation = _number(
            max_participation_fraction,
            field_name="max_participation_fraction",
            minimum=0.0,
        )
        ordered = _events(events)
        if not ordered:
            raise ProductAccountingError("futures accounting requires events")
        positions: dict[str, tuple[float, float]] = {}
        marks: dict[str, float] = {}
        realised = 0.0
        funding_pnl = 0.0
        fees = 0.0
        spread = 0.0
        slippage = 0.0
        fills = 0
        partials = 0
        capacity_violations = 0
        max_leverage = 0.0
        max_margin_fraction = 0.0
        liquidation = False
        observations = 0
        receipts: list[Mapping[str, Any]] = []

        for observed_at, event in ordered:
            kind = _event_kind(event)
            symbol = str(event.get("instrument_id", event.get("symbol", "BTCUSDT")))
            if kind in {"fill", "trade", "execution", "order_fill"}:
                side = str(event.get("side", "")).casefold()
                if side not in {"buy", "sell"}:
                    raise ProductAccountingError("futures fill side must be buy or sell")
                quantity = _quantity(event)
                price = _price(event)
                signed = quantity if side == "buy" else -quantity
                old_quantity, old_entry = positions.get(symbol, (0.0, 0.0))
                realised_delta, new_quantity, new_entry = self._apply_fill(
                    old_quantity, old_entry, signed, price
                )
                realised += realised_delta
                positions[symbol] = (new_quantity, new_entry)
                marks[symbol] = price
                cash += realised_delta
                fill_fee = _fee(event)
                fees += fill_fee
                cash -= fill_fee
                spread_delta = self._cost(event, signed, price, "expected_price", "spread_cost")
                slippage_delta = self._cost(
                    event, signed, price, "reference_price", "slippage_cost"
                )
                spread += spread_delta
                slippage += slippage_delta
                cash -= spread_delta + slippage_delta
                fills += 1
                requested = event.get("requested_quantity", event.get("order_quantity"))
                if requested is not None:
                    requested_value = _number(requested, field_name="requested_quantity", minimum=0.0)
                    if quantity + 1e-12 < requested_value:
                        partials += 1
                visible_depth = event.get("visible_depth", event.get("available_depth"))
                if visible_depth is not None and participation > 0:
                    depth = _number(visible_depth, field_name="visible_depth", minimum=0.0)
                    if quantity > depth * participation + 1e-12:
                        capacity_violations += 1
            elif kind in {"mark", "mark_price", "price"}:
                marks[symbol] = _price(event, field_name="mark_price")
                observations += 1
            elif kind in {"funding", "funding_rate"}:
                mark = _price(event, field_name="mark_price")
                marks[symbol] = mark
                quantity = positions.get(symbol, (0.0, 0.0))[0]
                rate = _number(event.get("funding_rate", event.get("rate", 0.0)), field_name="funding_rate")
                amount = -quantity * mark * rate
                if event.get("funding_pnl") is not None:
                    amount = _number(event["funding_pnl"], field_name="funding_pnl")
                funding_pnl += amount
                cash += amount
                observations += 1
            elif kind in {"fee", "commission"}:
                amount = _fee(event)
                fees += amount
                cash -= amount
            elif kind in {"liquidation", "liquidated"}:
                liquidation = True
                observations += 1
            else:
                raise ProductAccountingError(f"unsupported futures accounting event type: {kind}")
            equity, unrealised, notional = self._equity(cash, positions, marks)
            margin = notional / leverage
            if equity > 0:
                max_leverage = max(max_leverage, notional / equity)
            max_margin_fraction = max(
                max_margin_fraction,
                margin / equity if equity > 0 else 0.0,
            )
            if equity <= maintenance * margin and notional > 0:
                liquidation = True
            receipts.append(
                {
                    "event_hash": canonical_hash(dict(event)),
                    "occurred_at": observed_at,
                    "event_type": kind,
                    "equity": equity,
                    "unrealised_pnl": unrealised,
                }
            )

        final_equity, unrealised, _ = self._equity(cash, positions, marks)
        return FuturesAccountingReport(
            initial_equity=starting_cash,
            final_equity=final_equity,
            net_pnl=final_equity - starting_cash,
            return_fraction=(final_equity / starting_cash - 1.0 if starting_cash else 0.0),
            realised_pnl=realised,
            unrealised_pnl=unrealised,
            fees=fees,
            funding_pnl=funding_pnl,
            spread_cost=spread,
            slippage_cost=slippage,
            fills=fills,
            partial_fills=partials,
            capacity_violations=capacity_violations,
            max_leverage=max_leverage,
            max_margin_fraction=max_margin_fraction,
            liquidation=liquidation,
            effective_observations=observations + fills,
            event_receipts=tuple(receipts),
        )

    @staticmethod
    def _apply_fill(
        old_quantity: float, old_entry: float, signed_quantity: float, price: float
    ) -> tuple[float, float, float]:
        if abs(old_quantity) <= 1e-12:
            return 0.0, signed_quantity, price
        if old_quantity * signed_quantity >= 0:
            total = abs(old_quantity) + abs(signed_quantity)
            entry = (abs(old_quantity) * old_entry + abs(signed_quantity) * price) / total
            return 0.0, old_quantity + signed_quantity, entry
        close = min(abs(old_quantity), abs(signed_quantity))
        realised = close * (price - old_entry) * (1.0 if old_quantity > 0 else -1.0)
        remaining = old_quantity + signed_quantity
        return realised, remaining, (0.0 if abs(remaining) <= 1e-12 else price)

    @staticmethod
    def _cost(
        event: Mapping[str, Any],
        signed_quantity: float,
        price: float,
        reference_key: str,
        explicit_key: str,
    ) -> float:
        if event.get(explicit_key) is not None:
            return _number(event[explicit_key], field_name=explicit_key, minimum=0.0)
        reference = event.get(reference_key)
        if reference is None:
            return 0.0
        expected = _number(reference, field_name=reference_key, minimum=0.0)
        return max(0.0, (price - expected) * signed_quantity)

    @staticmethod
    def _equity(
        cash: float,
        positions: Mapping[str, tuple[float, float]],
        marks: Mapping[str, float],
    ) -> tuple[float, float, float]:
        unrealised = 0.0
        notional = 0.0
        for symbol, (quantity, entry) in positions.items():
            mark = marks.get(symbol, entry)
            unrealised += quantity * (mark - entry)
            notional += abs(quantity * mark)
        return cash + unrealised, unrealised, notional


FuturesAccounting = FuturesIncomeAccounting
BtcAccounting = BtcAccumulationAccounting
