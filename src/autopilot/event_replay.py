"""Deterministic replay and fill simulation over captured market events."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from src.alpha.microstructure import MicrostructureAlphaPolicy, forecast_from_microstructure
from src.autopilot.event_capture import MAX_EVENT_BYTES, SCHEMA
from src.autopilot.io import write_json_atomic
from src.microstructure.features import MicrostructureState, RestingLimitOrder

REPORT_SCHEMA = "autopilot.event_replay_report/v1"


def iter_events(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    previous_received_ns = -1
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"event replay input must be a regular non-symlink file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(line.encode()) > MAX_EVENT_BYTES + 1:
                    raise ValueError(f"event line exceeds maximum size: {path}:{line_number}")
                event = json.loads(line)
                if not isinstance(event, dict) or event.get("schema") != SCHEMA:
                    raise ValueError(f"invalid event schema: {path}:{line_number}")
                received_ns = event.get("received_ns")
                if isinstance(received_ns, bool) or not isinstance(received_ns, int):
                    raise ValueError(f"invalid event receive time: {path}:{line_number}")
                if received_ns < previous_received_ns:
                    raise ValueError(f"event receive order regressed: {path}:{line_number}")
                previous_received_ns = received_ns
                yield event


def replay(
    paths: Iterable[Path],
    *,
    symbol: str,
    sample_every: int = 1_000,
    market_order: tuple[str, float] | None = None,
    limit_order: dict[str, Any] | None = None,
    microstructure_policy: MicrostructureAlphaPolicy | None = None,
    strategy_quantity: float = 0.001,
    max_events: int | None = None,
) -> dict[str, Any]:
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    if strategy_quantity <= 0:
        raise ValueError("strategy_quantity must be positive")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive when supplied")
    state = MicrostructureState(symbol.upper())
    events = 0
    snapshots: list[dict[str, Any]] = []
    first_received_ns = None
    last_received_ns = None
    passive_order: RestingLimitOrder | None = None
    strategy_position: dict[str, Any] | None = None
    strategy_signals = 0
    strategy_trades: list[dict[str, Any]] = []
    latest_forecast: dict[str, Any] | None = None
    for event in iter_events(paths):
        if max_events is not None and events >= max_events:
            break
        events += 1
        first_received_ns = first_received_ns or event["received_ns"]
        last_received_ns = event["received_ns"]
        state.apply(event)
        if limit_order is not None and passive_order is None:
            passive_order = RestingLimitOrder(
                submitted_ns=event["received_ns"],
                **limit_order,
            )
        if passive_order is not None:
            passive_order.observe(event, state)
        if events % sample_every == 0:
            snapshot = state.snapshot()
            if snapshot.get("ok"):
                snapshot["received_ns"] = last_received_ns
                snapshots.append(snapshot)
                if len(snapshots) > 1_000:
                    snapshots.pop(0)
                if microstructure_policy is not None:
                    generated_at = event.get("received_at") or str(last_received_ns)
                    forecast, signal_detail = forecast_from_microstructure(
                        snapshot,
                        market=str(event.get("market") or "futures"),
                        policy=microstructure_policy,
                        generated_at=str(generated_at),
                    )
                    if forecast is not None:
                        strategy_signals += 1
                        latest_forecast = forecast.to_dict()
                    should_exit = False
                    exit_reason = None
                    if strategy_position is not None:
                        elapsed_ns = int(last_received_ns) - int(
                            strategy_position["entry_received_ns"]
                        )
                        if elapsed_ns >= int(strategy_position["horizon_seconds"]) * 1_000_000_000:
                            should_exit, exit_reason = True, "horizon"
                        elif (
                            forecast is not None
                            and forecast.direction != strategy_position["direction"]
                        ):
                            should_exit, exit_reason = True, "opposite_signal"
                    if strategy_position is not None and should_exit:
                        exit_side = "sell" if strategy_position["direction"] == "long" else "buy"
                        fill = state.market_fill(
                            side=exit_side,
                            quantity=float(strategy_position["filled_quantity"]),
                        )
                        if fill["filled_quantity"] > 0:
                            signed = 1.0 if strategy_position["direction"] == "long" else -1.0
                            pnl = (
                                signed
                                * (
                                    float(fill["average_price"])
                                    - float(strategy_position["average_price"])
                                )
                                * float(fill["filled_quantity"])
                            )
                            pnl -= float(strategy_position["fee"]) + float(fill["fee"])
                            strategy_trades.append(
                                {
                                    "direction": strategy_position["direction"],
                                    "entry_received_ns": strategy_position["entry_received_ns"],
                                    "exit_received_ns": last_received_ns,
                                    "entry_price": strategy_position["average_price"],
                                    "exit_price": fill["average_price"],
                                    "filled_quantity": fill["filled_quantity"],
                                    "net_pnl_quote": pnl,
                                    "reason": exit_reason,
                                }
                            )
                            strategy_trades = strategy_trades[-100:]
                            strategy_position = None
                    if strategy_position is None and forecast is not None:
                        entry_side = "buy" if forecast.direction == "long" else "sell"
                        fill = state.market_fill(side=entry_side, quantity=strategy_quantity)
                        if fill["filled_quantity"] > 0:
                            strategy_position = {
                                "direction": forecast.direction,
                                "entry_received_ns": last_received_ns,
                                "average_price": fill["average_price"],
                                "filled_quantity": fill["filled_quantity"],
                                "fee": fill["fee"],
                                "horizon_seconds": forecast.horizon_seconds,
                                "signal": forecast.to_dict(),
                                "signal_detail": signal_detail,
                            }
    final_features = state.snapshot()
    simulated_fill = None
    if market_order is not None:
        simulated_fill = state.market_fill(side=market_order[0], quantity=market_order[1])
    simulated_limit_fill = (
        passive_order.result(
            state,
            finished_ns=last_received_ns,
            funding_rate_per_8h=state.funding_rate,
        )
        if passive_order is not None
        else None
    )
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "symbol": symbol.upper(),
        "events": events,
        "first_received_ns": first_received_ns,
        "last_received_ns": last_received_ns,
        "sample_every": sample_every,
        "sampled_features": snapshots,
        "final_features": final_features,
        "simulated_market_fill": simulated_fill,
        "simulated_limit_fill": simulated_limit_fill,
        "microstructure_strategy": (
            {
                "enabled": True,
                "signals": strategy_signals,
                "completed_trades": strategy_trades,
                "open_position": strategy_position,
                "latest_forecast": latest_forecast,
                "research_only": True,
                "promotion_eligible": False,
                "live_allowed": False,
            }
            if microstructure_policy is not None
            else None
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay captured Binance events deterministically."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--sample-every", type=int, default=1_000)
    parser.add_argument("--market-side", choices=("buy", "sell"))
    parser.add_argument("--market-quantity", type=float)
    parser.add_argument("--limit-side", choices=("buy", "sell"))
    parser.add_argument("--limit-price", type=float)
    parser.add_argument("--limit-quantity", type=float)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--cancel-after-ms", type=float)
    parser.add_argument("--maker-fee-bps", type=float, default=1.0)
    parser.add_argument("--market", choices=("spot", "futures"), default="futures")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--microstructure-strategy", action="store_true")
    parser.add_argument("--strategy-quantity", type=float, default=0.001)
    parser.add_argument("--max-events", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    order = None
    if args.market_side is not None or args.market_quantity is not None:
        if args.market_side is None or args.market_quantity is None:
            raise SystemExit("--market-side and --market-quantity must be supplied together")
        order = (args.market_side, args.market_quantity)
    limit_values = (args.limit_side, args.limit_price, args.limit_quantity)
    if any(value is not None for value in limit_values) and not all(
        value is not None for value in limit_values
    ):
        raise SystemExit(
            "--limit-side, --limit-price and --limit-quantity must be supplied together"
        )
    limit_order = None
    if all(value is not None for value in limit_values):
        limit_order = {
            "side": args.limit_side,
            "price": args.limit_price,
            "quantity": args.limit_quantity,
            "latency_ns": int(args.latency_ms * 1_000_000),
            "cancel_after_ns": (
                int(args.cancel_after_ms * 1_000_000) if args.cancel_after_ms is not None else None
            ),
            "maker_fee_bps": args.maker_fee_bps,
            "market": args.market,
        }
    report = replay(
        args.paths,
        symbol=args.symbol,
        sample_every=args.sample_every,
        market_order=order,
        limit_order=limit_order,
        microstructure_policy=(
            MicrostructureAlphaPolicy() if args.microstructure_strategy else None
        ),
        strategy_quantity=args.strategy_quantity,
        max_events=args.max_events,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
