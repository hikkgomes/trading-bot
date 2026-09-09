"""Perturb economic inputs and preserve local dependence in return stress paths."""

import math
import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.accounting.fees import instrument_asset

RETURN_BLOCK_METHOD = "nonoverlapping_return_block_permutation/v1"


def return_block_permutations(values, *, block_length: int, seed: int, iterations: int = 250):
    if block_length < 2 or len(values) < block_length * 2 or iterations < 1:
        raise ValueError("return block stress requires at least two multi-observation blocks")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("return block stress requires finite observations")
    blocks = [
        tuple(values[start : start + block_length]) for start in range(0, len(values), block_length)
    ]
    randomiser = random.Random(seed)
    for _ in range(iterations):
        order = list(range(len(blocks)))
        randomiser.shuffle(order)
        yield [value for index in order for value in blocks[index]]


def cost_stress_context(
    context: Mapping[str, Any], *, fee: float = 1, slippage: float = 1, funding: float = 1
) -> dict[str, Any]:
    result = deepcopy(dict(context))
    result["fee_bps"] = float(context.get("fee_bps", 1)) * fee
    result["slippage_bps"] = float(context.get("slippage_bps", 1)) * slippage
    result["funding_rate"] = float(context.get("funding_rate", 0)) * funding
    for key in ("funding_rates", "funding_period_rates"):
        if context.get(key) is not None:
            result[key] = [float(value) * funding for value in context[key]]
    for key in ("market_frame", "bars"):
        if isinstance(result.get(key), list | tuple):
            result[key] = deepcopy(context[key])
            for row in result[key]:
                for event in row.get("funding_events", ()):
                    event["funding_rate"] *= funding
    for key in ("futures_events", "trade_events", "btc_trade_events"):
        if result.get(key) is not None:
            result[key] = [_event_costs(event, fee, slippage, funding) for event in context[key]]
    return result


def _event_costs(
    event: Mapping[str, Any], fee: float, slippage: float, funding: float
) -> dict[str, Any]:
    result = dict(event)
    kind = str(event.get("event_type", event.get("type", "fill"))).lower()
    if kind in {"funding", "funding_rate"}:
        result["funding_rate"] = float(event.get("funding_rate", event.get("rate", 0))) * funding
        return result
    if kind in {"fee", "commission"}:
        _stress_fees(event, result, fee, 1.0)
        return result
    if kind not in {"fill", "trade", "execution", "order_fill"}:
        return result
    price = float(event["price"])
    reference = float(event.get("reference_price", price))
    side = str(event.get("side", "")).lower()
    if side not in {"buy", "sell"}:
        raise ValueError("cost stress requires the execution side")
    result["price"] = price + (1 if side == "buy" else -1) * abs(price - reference) * (slippage - 1)
    _stress_fees(event, result, fee, result["price"] / price)
    return result


def _stress_fees(event, result, multiplier, price_ratio):
    symbol = str(event.get("instrument_id", event.get("symbol", "BTCUSDT")))
    asset = str(event.get("fee_asset", event.get("commission_asset", "USDT"))).upper()
    native_multiplier = (
        multiplier if asset == instrument_asset(symbol, "base") else multiplier * price_ratio
    )
    for names, scale in (
        (("fee", "fee_amount", "commission"), native_multiplier),
        (("fee_btc", "fee_in_btc", "commission_btc"), multiplier),
        (
            (
                "fee_quote",
                "fee_quote_value",
                "fee_in_quote",
                "commission_quote",
                "commission_quote_value",
            ),
            multiplier * price_ratio,
        ),
    ):
        for name in names:
            if event.get(name) is not None:
                result[name] = float(event[name]) * scale
