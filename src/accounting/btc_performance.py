"""BTC-denominated performance report against passive BTC holding."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from src.accounting.ledger import Ledger
from src.accounting.nav import NavSnapshot


@dataclass(frozen=True)
class BtcPerformanceReport:
    btc_balance: float
    btc_nav: float
    btc_vs_passive_hold: float
    time_outside_btc_fraction: float
    stablecoin_exposure_fraction: float
    missed_btc_appreciation: float
    btc_drawdown_fraction: float
    fees_paid_btc: Decimal
    performance_by_regime: dict[str, float]


def build_btc_performance_report(
    snapshots: tuple[NavSnapshot, ...], *, ledger: Ledger
) -> BtcPerformanceReport:
    if not snapshots:
        raise ValueError("BTC performance requires NAV snapshots")
    ordered = tuple(sorted(snapshots, key=lambda item: item.observed_at))
    if any(
        item.product_id != "btc_accumulation" or item.accounting_asset.upper() != "BTC"
        for item in ordered
    ):
        raise ValueError("BTC performance received another product or accounting asset")
    if ledger.product_id != "btc_accumulation" or ledger.accounting_asset != "BTC":
        raise ValueError("BTC performance requires the BTC accumulation ledger")
    values = [float(item.nav) for item in ordered]
    if any(value < 0 for value in values):
        raise ValueError("BTC NAV cannot be negative")
    peak = max(values)
    drawdown = 0.0 if peak <= 0 else max(0.0, (peak - values[-1]) / peak)
    seconds_total = 0.0
    seconds_outside = 0.0
    missed = 0.0
    regime: dict[str, float] = {}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        start = dt.datetime.fromisoformat(previous.observed_at)
        end = dt.datetime.fromisoformat(current.observed_at)
        seconds = max(0.0, (end - start).total_seconds())
        seconds_total += seconds
        previous_btc = float(previous.components.get("btc_balance", 0.0))
        previous_fraction = 0.0 if previous.nav <= 0 else previous_btc / float(previous.nav)
        if previous_fraction < 1.0 - 1e-12:
            seconds_outside += seconds
        prior_price = float(previous.components.get("stablecoin_per_btc", 0.0))
        current_price = float(current.components.get("stablecoin_per_btc", 0.0))
        stablecoin = float(previous.components.get("stablecoin_balance", 0.0))
        if prior_price > 0 and current_price > prior_price and stablecoin > 0:
            missed += stablecoin / prior_price - stablecoin / current_price
        regime_name = str(previous.components.get("regime") or "unclassified")
        regime[regime_name] = regime.get(regime_name, 0.0) + float(current.nav - previous.nav)
    latest = ordered[-1]
    btc_balance = float(latest.components.get("btc_balance", 0.0))
    price = float(latest.components.get("stablecoin_per_btc", 0.0))
    stablecoin = float(latest.components.get("stablecoin_balance", 0.0))
    stablecoin_btc = stablecoin / price if price > 0 else 0.0
    passive = latest.passive_benchmark_nav
    fees = sum(
        (entry.postings.get("expense:fees", Decimal("0")) for entry in ledger.entries),
        Decimal("0"),
    )
    return BtcPerformanceReport(
        btc_balance=btc_balance,
        btc_nav=float(latest.nav),
        btc_vs_passive_hold=0.0 if passive is None else float(latest.nav - passive),
        time_outside_btc_fraction=(seconds_outside / seconds_total if seconds_total > 0 else 0.0),
        stablecoin_exposure_fraction=(
            stablecoin_btc / float(latest.nav) if latest.nav > 0 else 0.0
        ),
        missed_btc_appreciation=missed,
        btc_drawdown_fraction=drawdown,
        fees_paid_btc=fees,
        performance_by_regime=regime,
    )
