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
from collections.abc import Callable, Iterable, Mapping
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


def _fee_values(
    event: Mapping[str, Any], *, fee_asset: str, fee: float, price: float
) -> tuple[float, float, str]:
    """Return fee in BTC, quote currency, and the deterministic conversion source."""

    if fee <= 0.0:
        return 0.0, 0.0, "zero_fee"
    if price <= 0.0:
        raise ProductAccountingError("trade price must be positive for fee conversion")
    if fee_asset == "BTC":
        return fee, fee * price, "base_asset"
    if fee_asset in {"USDT", "USDC", "BUSD"}:
        return fee / price, fee, "quote_asset"
    explicit = _explicit_fee_values(event, fee=fee, price=price)
    if explicit is not None:
        return explicit
    raise ProductAccountingError(
        f"fee asset {fee_asset} needs a deterministic BTC or quote conversion"
    )


def _explicit_fee_values(
    event: Mapping[str, Any], *, fee: float, price: float
) -> tuple[float, float, str] | None:
    for key in ("fee_btc", "fee_in_btc", "commission_btc"):
        if event.get(key) is not None:
            fee_btc = _number(event[key], field_name=key, minimum=0.0)
            return fee_btc, fee_btc * price, f"explicit_{key}"
    for key in (
        "fee_quote",
        "fee_quote_value",
        "fee_in_quote",
        "commission_quote",
        "commission_quote_value",
    ):
        if event.get(key) is not None:
            quote_value = _number(event[key], field_name=key, minimum=0.0)
            return quote_value / price, quote_value, f"explicit_{key}"
    for key in (
        "fee_conversion_price",
        "fee_asset_price",
        "fee_asset_to_quote_price",
        "fee_quote_price",
    ):
        if event.get(key) is not None:
            conversion_price = _number(event[key], field_name=key, minimum=0.0)
            if conversion_price <= 0.0:
                raise ProductAccountingError(f"{key} must be positive")
            quote_value = fee * conversion_price
            return quote_value / price, quote_value, f"explicit_{key}"
    return None


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
    maximum_btc_drawdown: float = 0.0
    btc_saved_in_drawdown_periods: float = 0.0
    round_trip_btc_gain: float = 0.0
    maximum_tactical_allocation: float = 0.0
    average_stablecoin_exposure_fraction: float = 0.0
    worst_reentry_slippage: float = 0.0
    failed_reentries: int = 0
    external_deposits_btc: float = 0.0
    external_withdrawals_btc: float = 0.0

    @property
    def btc_vs_passive_hold(self) -> float:
        return self.excess_btc

    @property
    def final_nav(self) -> float:
        return self.final_btc_nav

    @property
    def objective_unit(self) -> str:
        return "BTC"


@dataclass
class _BtcEvaluationState:
    btc: float
    stable: float
    passive: float
    current_price: float
    current_regime: str
    core_btc: float
    max_tactical_fraction: float | None
    snapshots: list[BtcAccountingSnapshot]
    receipts: list[Mapping[str, Any]]
    pending_sells: list[tuple[float, float, float, float]]
    reentry_slippages: list[float]
    trade_index: int = 0
    flow_index: int = 0
    mark_index: int = 0
    fee_btc_total: float = 0.0
    missed: float = 0.0
    seconds_total: float = 0.0
    seconds_outside: float = 0.0
    cycles: int = 0
    saw_sell: bool = False
    previous_snapshot: BtcAccountingSnapshot | None = None
    maximum_drawdown: float = 0.0
    peak_nav: float = 0.0
    btc_saved_in_drawdown: float = 0.0
    round_trip_gain: float = 0.0
    maximum_tactical_allocation: float = 0.0
    exposure_time: float = 0.0
    failed_reentries: int = 0
    external_deposits_btc: float = 0.0
    external_withdrawals_btc: float = 0.0


@dataclass(frozen=True)
class _PreparedBtcEvaluation:
    initial_nav: float
    trades: tuple[tuple[str, Mapping[str, Any]], ...]
    flows: tuple[tuple[str, Mapping[str, Any]], ...]
    mark_events: tuple[tuple[str, float, Mapping[str, Any]], ...]
    all_times: tuple[str, ...]
    state: _BtcEvaluationState


def _prepare_btc_evaluation(
    *,
    trade_events: Iterable[Mapping[str, Any]],
    marks: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    initial_btc: float,
    initial_stablecoin: float,
    initial_price: float | None,
    reserve_fraction: float | None,
    max_tactical_fraction: float | None,
    external_events: Iterable[Mapping[str, Any]],
    normalise_marks: Callable[
        [Iterable[Mapping[str, Any]] | Mapping[str, Any]],
        tuple[tuple[str, float, Mapping[str, Any]], ...],
    ],
) -> _PreparedBtcEvaluation:
    btc = _number(initial_btc, field_name="initial_btc", minimum=0.0)
    stable = _number(initial_stablecoin, field_name="initial_stablecoin", minimum=0.0)
    trades = _events(trade_events)
    flows = _events(external_events)
    mark_events = normalise_marks(marks)
    if not mark_events and not trades and not flows:
        raise ProductAccountingError("BTC accounting requires trade events or marks")
    first_price = initial_price
    if first_price is None:
        if mark_events:
            first_price = mark_events[0][1]
        elif trades:
            first_price = _price(trades[0][1])
        else:
            raise ProductAccountingError("initial_price is required when no trade or mark exists")
    first_price = _number(first_price, field_name="initial_price", minimum=0.0)
    if first_price <= 0:
        raise ProductAccountingError("initial_price must be positive")
    reserve = _bounded_fraction(reserve_fraction, field_name="reserve_fraction")
    tactical = _bounded_fraction(max_tactical_fraction, field_name="max_tactical_fraction")
    initial_nav = btc + stable / first_price
    all_times = tuple(
        sorted(
            {time for time, _ in trades}
            | {time for time, _ in flows}
            | {time for time, _, _ in mark_events}
        )
    )
    state = _BtcEvaluationState(
        btc=btc,
        stable=stable,
        passive=initial_nav,
        current_price=first_price,
        current_regime="unclassified",
        core_btc=initial_nav * (reserve or 0.0),
        max_tactical_fraction=tactical,
        snapshots=[],
        receipts=[],
        pending_sells=[],
        reentry_slippages=[],
        peak_nav=initial_nav,
    )
    return _PreparedBtcEvaluation(initial_nav, trades, flows, mark_events, all_times, state)


def _bounded_fraction(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    result = _number(value, field_name=field_name, minimum=0.0)
    if result > 1:
        raise ProductAccountingError(f"{field_name} must be at most 1")
    return result


def _apply_btc_marks(
    state: _BtcEvaluationState,
    mark_events: tuple[tuple[str, float, Mapping[str, Any]], ...],
    observed_at: str,
) -> None:
    while state.mark_index < len(mark_events) and mark_events[state.mark_index][0] == observed_at:
        _, mark_price, mark_event = mark_events[state.mark_index]
        state.current_price = mark_price
        state.current_regime = str(mark_event.get("regime", "unclassified"))
        state.mark_index += 1


def _apply_btc_external_events(
    state: _BtcEvaluationState,
    flows: tuple[tuple[str, Mapping[str, Any]], ...],
    observed_at: str,
) -> None:
    while state.flow_index < len(flows) and flows[state.flow_index][0] == observed_at:
        _, event = flows[state.flow_index]
        kind = _event_kind(event)
        if kind not in {"deposit", "withdrawal", "transfer"}:
            raise ProductAccountingError(f"unsupported BTC external event type: {kind}")
        amount = _number(
            event.get("amount", event.get("quantity", event.get("value", 0.0))),
            field_name="external amount",
            minimum=0.0,
        )
        asset = (
            str(event.get("asset", event.get("currency", event.get("fee_asset", "BTC"))))
            .strip()
            .upper()
        )
        if asset == "BTC":
            value_btc, balance_name = amount, "btc"
        elif asset in {"USDT", "USDC", "BUSD"}:
            value_btc, balance_name = amount / state.current_price, "stable"
        else:
            raise ProductAccountingError(f"unsupported BTC external asset: {asset}")
        direction = 1.0 if kind == "deposit" else -1.0
        if balance_name == "btc":
            state.btc += direction * amount
        else:
            state.stable += direction * amount
        if state.btc < -1e-12 or state.stable < -1e-12:
            raise ProductAccountingError("external BTC balance flow exceeds available balance")
        state.passive += direction * value_btc
        if direction > 0:
            state.external_deposits_btc += value_btc
        else:
            state.external_withdrawals_btc += value_btc
        state.receipts.append(
            {
                "event_hash": canonical_hash(dict(event)),
                "occurred_at": observed_at,
                "event_type": kind,
                "asset": asset,
                "amount": amount,
                "amount_btc": value_btc,
            }
        )
        state.flow_index += 1


def _apply_btc_buy(
    state: _BtcEvaluationState,
    *,
    quantity: float,
    price: float,
    fee_asset: str,
    converted_fee_btc: float,
    fee_quote: float,
) -> None:
    if fee_asset == "BTC":
        state.btc += quantity - converted_fee_btc
        state.stable -= quantity * price
    else:
        state.btc += quantity
        state.stable -= quantity * price + fee_quote
    if state.btc < -1e-12 or state.stable < -1e-12:
        raise ProductAccountingError("BTC buy exceeds available balance")
    if state.saw_sell:
        state.cycles += 1
        state.saw_sell = False
    remaining_buy = quantity
    buy_fee_per_btc = converted_fee_btc / quantity if quantity > 0 else 0.0
    while remaining_buy > 1e-12 and state.pending_sells:
        sold_quantity, sold_price, sell_fee_btc, sell_fee_quote = state.pending_sells[0]
        paired_quantity = min(remaining_buy, sold_quantity)
        sell_fraction = paired_quantity / sold_quantity
        state.round_trip_gain += (
            (paired_quantity * sold_price - sell_fee_quote * sell_fraction) / price
            - sell_fee_btc * sell_fraction
            - paired_quantity
            - buy_fee_per_btc * paired_quantity
        )
        state.reentry_slippages.append(price / sold_price - 1.0)
        remaining_buy -= paired_quantity
        remaining_sell = sold_quantity - paired_quantity
        if remaining_sell <= 1e-12:
            state.pending_sells.pop(0)
        else:
            state.pending_sells[0] = (
                remaining_sell,
                sold_price,
                sell_fee_btc * (remaining_sell / sold_quantity),
                sell_fee_quote * (remaining_sell / sold_quantity),
            )


def _apply_btc_sell(
    state: _BtcEvaluationState,
    *,
    quantity: float,
    price: float,
    fee_asset: str,
    converted_fee_btc: float,
    fee_quote: float,
) -> None:
    if fee_asset == "BTC":
        state.btc -= quantity + converted_fee_btc
        state.stable += quantity * price
    else:
        state.btc -= quantity
        state.stable += quantity * price - fee_quote
    if state.core_btc and state.btc < state.core_btc - 1e-12:
        raise ProductAccountingError("BTC sell path breached core BTC reserve")
    if state.btc < -1e-12 or state.stable < -1e-12:
        raise ProductAccountingError("BTC sell exceeds available balance")
    state.saw_sell = True
    state.pending_sells.append(
        (
            quantity,
            price,
            converted_fee_btc if fee_asset == "BTC" else 0.0,
            fee_quote if fee_asset != "BTC" else 0.0,
        )
    )


def _apply_btc_trade(
    state: _BtcEvaluationState,
    event: Mapping[str, Any],
    observed_at: str,
) -> None:
    side = str(event.get("side", "")).casefold()
    if side not in {"buy", "sell"}:
        raise ProductAccountingError("BTC trade side must be buy or sell")
    price = _price(event)
    quantity = _quantity(event)
    fee = _fee(event)
    if price <= 0:
        raise ProductAccountingError("BTC trade price must be positive")
    fee_asset = (
        str(event.get("fee_asset", event.get("commission_asset", "USDT"))).strip().upper() or "USDT"
    )
    converted_fee_btc, fee_quote, fee_conversion = _fee_values(
        event, fee_asset=fee_asset, fee=fee, price=price
    )
    if side == "buy":
        _apply_btc_buy(
            state,
            quantity=quantity,
            price=price,
            fee_asset=fee_asset,
            converted_fee_btc=converted_fee_btc,
            fee_quote=fee_quote,
        )
    else:
        _apply_btc_sell(
            state,
            quantity=quantity,
            price=price,
            fee_asset=fee_asset,
            converted_fee_btc=converted_fee_btc,
            fee_quote=fee_quote,
        )
    if side == "buy" and str(event.get("reentry_status") or "").casefold() == "failed":
        state.failed_reentries += 1
    state.fee_btc_total += converted_fee_btc
    state.receipts.append(
        {
            "event_hash": canonical_hash(dict(event)),
            "occurred_at": observed_at,
            "side": side,
            "quantity_btc": quantity,
            "price": price,
            "fee_btc": converted_fee_btc,
            "fee_quote": fee_quote,
            "fee_conversion": fee_conversion,
            "fee_asset": fee_asset,
        }
    )


def _apply_btc_trades(
    state: _BtcEvaluationState,
    trades: tuple[tuple[str, Mapping[str, Any]], ...],
    observed_at: str,
) -> None:
    while state.trade_index < len(trades) and trades[state.trade_index][0] == observed_at:
        _, event = trades[state.trade_index]
        _apply_btc_trade(state, event, observed_at)
        state.trade_index += 1


def _record_btc_snapshot(state: _BtcEvaluationState, observed_at: str) -> None:
    nav = state.btc + state.stable / state.current_price
    snapshot = BtcAccountingSnapshot(
        observed_at=observed_at,
        btc_balance=max(0.0, state.btc),
        stablecoin_balance=max(0.0, state.stable),
        stablecoin_per_btc=state.current_price,
        btc_nav=nav,
        passive_btc_nav=state.passive,
        regime=state.current_regime,
    )
    previous = state.previous_snapshot
    if previous is not None:
        start = dt.datetime.fromisoformat(previous.observed_at)
        end = dt.datetime.fromisoformat(observed_at)
        seconds = max(0.0, (end - start).total_seconds())
        state.seconds_total += seconds
        outside = previous.stablecoin_balance / previous.stablecoin_per_btc
        if previous.btc_nav > 0 and outside > 1e-12:
            state.seconds_outside += seconds
        if state.current_price > previous.stablecoin_per_btc and outside > 0:
            state.missed += outside - previous.stablecoin_balance / state.current_price
        if state.current_price < previous.stablecoin_per_btc and outside > 0:
            state.btc_saved_in_drawdown += (
                previous.stablecoin_balance / state.current_price - outside
            )
        exposure = (
            snapshot.stablecoin_balance / snapshot.stablecoin_per_btc / snapshot.btc_nav
            if snapshot.btc_nav > 0
            else 0.0
        )
        state.exposure_time += seconds * exposure
    state.snapshots.append(snapshot)
    state.previous_snapshot = snapshot
    state.peak_nav = max(state.peak_nav, nav)
    if state.peak_nav > 0:
        state.maximum_drawdown = max(
            state.maximum_drawdown, (state.peak_nav - nav) / state.peak_nav
        )
    tactical = state.stable / state.current_price / nav if nav > 0 else 0.0
    if state.max_tactical_fraction is not None and tactical > state.max_tactical_fraction + 1e-12:
        raise ProductAccountingError("BTC tactical allocation exceeded configured limit")
    state.maximum_tactical_allocation = max(state.maximum_tactical_allocation, tactical)


def _build_btc_report(
    state: _BtcEvaluationState,
    *,
    initial_nav: float,
    first_observed_at: str,
) -> BtcAccountingReport:
    if not state.snapshots:
        state.snapshots.append(
            BtcAccountingSnapshot(
                observed_at=first_observed_at,
                btc_balance=state.btc,
                stablecoin_balance=state.stable,
                stablecoin_per_btc=state.current_price,
                btc_nav=state.btc + state.stable / state.current_price,
                passive_btc_nav=state.passive,
            )
        )
    latest = state.snapshots[-1]
    regime_pnl: dict[str, float] = {}
    for previous, current in zip(state.snapshots, state.snapshots[1:], strict=False):
        regime_pnl[previous.regime] = regime_pnl.get(previous.regime, 0.0) + (
            current.btc_nav - previous.btc_nav
        )
    return BtcAccountingReport(
        initial_btc_nav=initial_nav,
        final_btc_nav=latest.btc_nav,
        passive_btc_nav=state.passive,
        excess_btc=latest.btc_nav - state.passive,
        return_fraction=(latest.btc_nav / initial_nav - 1.0 if initial_nav > 0 else 0.0),
        fees_btc=state.fee_btc_total,
        time_outside_btc_fraction=(
            state.seconds_outside / state.seconds_total if state.seconds_total > 0 else 0.0
        ),
        stablecoin_exposure_fraction=(
            latest.stablecoin_balance / latest.stablecoin_per_btc / latest.btc_nav
            if latest.btc_nav > 0
            else 0.0
        ),
        missed_btc_appreciation=max(0.0, state.missed),
        cycles=state.cycles,
        regime_pnl=regime_pnl,
        nav_series=tuple(state.snapshots),
        event_receipts=tuple(state.receipts),
        maximum_btc_drawdown=state.maximum_drawdown,
        btc_saved_in_drawdown_periods=max(0.0, state.btc_saved_in_drawdown),
        round_trip_btc_gain=state.round_trip_gain,
        maximum_tactical_allocation=state.maximum_tactical_allocation,
        average_stablecoin_exposure_fraction=(
            state.exposure_time / state.seconds_total if state.seconds_total > 0 else 0.0
        ),
        worst_reentry_slippage=max(state.reentry_slippages, default=0.0),
        failed_reentries=state.failed_reentries,
        external_deposits_btc=state.external_deposits_btc,
        external_withdrawals_btc=state.external_withdrawals_btc,
    )


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
        max_tactical_fraction: float | None = None,
        external_events: Iterable[Mapping[str, Any]] = (),
    ) -> BtcAccountingReport:
        prepared = _prepare_btc_evaluation(
            trade_events=trade_events,
            marks=marks,
            initial_btc=initial_btc,
            initial_stablecoin=initial_stablecoin,
            initial_price=initial_price,
            reserve_fraction=reserve_fraction,
            max_tactical_fraction=max_tactical_fraction,
            external_events=external_events,
            normalise_marks=self._normalise_marks,
        )
        for observed_at in prepared.all_times:
            _apply_btc_marks(prepared.state, prepared.mark_events, observed_at)
            _apply_btc_external_events(prepared.state, prepared.flows, observed_at)
            _apply_btc_trades(prepared.state, prepared.trades, observed_at)
            _record_btc_snapshot(prepared.state, observed_at)
        return _build_btc_report(
            prepared.state,
            initial_nav=prepared.initial_nav,
            first_observed_at=prepared.all_times[0],
        )

    @staticmethod
    def _normalise_marks(
        marks: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    ) -> tuple[tuple[str, float, Mapping[str, Any]], ...]:
        if isinstance(marks, Mapping):
            source = tuple({"timestamp": key, "price": value} for key, value in marks.items())
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
    turnover_notional: float = 0.0
    implementation_shortfall: float = 0.0
    capital_efficiency: float = 0.0
    funding_adjusted_expectancy: float = 0.0
    margin_mode: str = "isolated"
    target_notional: float | None = None
    liquidation_buffer_fraction: float = 0.0

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
        funding_timestamps: Iterable[str] | None = None,
        max_margin_fraction: float = 1.0,
        target_notional: float | Mapping[str, float] | None = None,
        margin_mode: str = "isolated",
        liquidation_buffer_fraction: float = 0.0,
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
        if isinstance(funding_timestamps, str):
            raise ProductAccountingError("funding_timestamps must be an iterable of timestamps")
        funding_schedule = (
            None
            if funding_timestamps is None
            else frozenset(
                timestamp(str(value), field="funding_timestamps[]") for value in funding_timestamps
            )
        )
        margin_limit = _number(
            max_margin_fraction,
            field_name="max_margin_fraction",
            minimum=0.0,
        )
        if margin_limit > 1.0:
            raise ProductAccountingError("max_margin_fraction must be at most 1")
        margin_mode = str(margin_mode).strip().casefold()
        if margin_mode not in {"isolated", "cross"}:
            raise ProductAccountingError("margin_mode must be isolated or cross")
        liquidation_buffer = _number(
            liquidation_buffer_fraction,
            field_name="liquidation_buffer_fraction",
            minimum=0.0,
        )
        if liquidation_buffer > 1.0:
            raise ProductAccountingError("liquidation_buffer_fraction must be at most 1")
        target_value: float | None = None
        target_by_symbol: dict[str, float] = {}
        if isinstance(target_notional, Mapping):
            for symbol, value in target_notional.items():
                target_by_symbol[str(symbol)] = _number(
                    value, field_name=f"target_notional[{symbol}]", minimum=0.0
                )
        elif target_notional is not None:
            target_value = _number(target_notional, field_name="target_notional", minimum=0.0)
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
        observed_max_margin_fraction = 0.0
        liquidation = False
        observations = 0
        turnover_notional = 0.0
        receipts: list[Mapping[str, Any]] = []

        for observed_at, event in ordered:
            kind = _event_kind(event)
            symbol = str(event.get("instrument_id", event.get("symbol", "BTCUSDT")))
            funding_applied: bool | None = None
            funding_amount = 0.0
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
                turnover_notional += quantity * price
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
                    requested_value = _number(
                        requested, field_name="requested_quantity", minimum=0.0
                    )
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
                rate = _number(
                    event.get("funding_rate", event.get("rate", 0.0)), field_name="funding_rate"
                )
                funding_applied = funding_schedule is None or observed_at in funding_schedule
                if funding_applied:
                    amount = -quantity * mark * rate
                    if event.get("funding_pnl") is not None:
                        amount = _number(event["funding_pnl"], field_name="funding_pnl")
                    funding_amount = amount
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
            symbol_target = target_by_symbol.get(symbol, target_value)
            event_target = event.get("target_notional")
            if event_target is not None:
                symbol_target = _number(event_target, field_name="target_notional", minimum=0.0)
            if symbol_target is not None and notional > symbol_target + 1e-12:
                capacity_violations += 1
            if equity > 0:
                max_leverage = max(max_leverage, notional / equity)
                if notional > equity * leverage + 1e-12:
                    capacity_violations += 1
            observed_max_margin_fraction = max(
                observed_max_margin_fraction,
                margin / equity if equity > 0 else 0.0,
            )
            if equity > 0 and margin / equity > margin_limit + 1e-12:
                capacity_violations += 1
            if equity <= (maintenance + liquidation_buffer) * margin and notional > 0:
                liquidation = True
            receipt = {
                "event_hash": canonical_hash(dict(event)),
                "occurred_at": observed_at,
                "event_type": kind,
                "equity": equity,
                "unrealised_pnl": unrealised,
                "notional": notional,
                "margin": margin,
                "target_notional": symbol_target,
                "liquidation_buffer_fraction": liquidation_buffer,
            }
            if funding_applied is not None:
                receipt.update({"funding_applied": funding_applied, "funding_pnl": funding_amount})
            receipts.append(receipt)

        final_equity, unrealised, _ = self._equity(cash, positions, marks)
        target_report = target_value if target_value is not None else None
        implementation_shortfall = fees + spread + slippage
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
            max_margin_fraction=observed_max_margin_fraction,
            liquidation=liquidation,
            effective_observations=observations + fills,
            event_receipts=tuple(receipts),
            turnover_notional=turnover_notional,
            implementation_shortfall=implementation_shortfall,
            capital_efficiency=(
                (final_equity - starting_cash) / turnover_notional
                if turnover_notional > 0.0
                else 0.0
            ),
            funding_adjusted_expectancy=(
                (final_equity - starting_cash) / fills if fills > 0 else 0.0
            ),
            margin_mode=margin_mode,
            target_notional=target_report,
            liquidation_buffer_fraction=liquidation_buffer,
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


class FuturesResearchAccounting(FuturesIncomeAccounting):
    """Research-facing futures ledger with the product accounting contract."""


class BtcResearchAccounting(BtcAccumulationAccounting):
    """Research-facing BTC ledger with the accumulation objective contract."""


FuturesAccounting = FuturesIncomeAccounting
BtcAccounting = BtcAccumulationAccounting
