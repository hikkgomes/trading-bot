import dataclasses
import json
from pathlib import Path

import pytest

from src.alpha.microstructure import MicrostructureAlphaPolicy
from src.autopilot.event_capture import (
    DEFAULT_CONFIG,
    EventWriter,
    load_event_capture_config,
    normalize_event,
    stream_names,
)
from src.autopilot.event_replay import iter_events, replay
from src.microstructure.features import MicrostructureState, RestingLimitOrder


def _event(stream, payload, received_ns, *, market="futures", symbol="BTCUSDT"):
    event = normalize_event(
        market=market,
        stream=stream,
        payload=payload,
        received_ns=received_ns,
    )
    event["symbol"] = symbol
    return event


def test_default_event_capture_config_is_bounded_and_uses_approved_streams():
    config = load_event_capture_config(DEFAULT_CONFIG)

    assert config.max_total_bytes == 5 * 1024**3
    assert config.retention_seconds == 7 * 86400
    futures = next(source for source in config.sources if "depth20@100ms" in source.streams)
    names = stream_names(futures, ("ETHUSDT",))
    assert "btcusdt@depth20@100ms" in names
    assert "ethusdt@aggTrade" in names
    assert "!forceOrder@arr" in names
    bounded = stream_names(
        futures,
        tuple(f"SYMBOL{index}USDT" for index in range(100)),
        data_tier_budgets=config.data_tier_budgets,
    )
    assert len(bounded) == 5 * futures.max_dynamic_symbols + 1
    candle_source = next(source for source in config.sources if source.streams == ("kline_1m",))
    assert candle_source.max_dynamic_symbols == 1_000
    assert "ethusdt@kline_1m" in stream_names(candle_source, ("ETHUSDT",))


def test_event_writer_rotates_and_never_exceeds_bounded_line_size(tmp_path):
    config = dataclasses.replace(
        load_event_capture_config(DEFAULT_CONFIG),
        root=tmp_path,
        max_file_bytes=350,
        max_total_bytes=2_000,
    )
    writer = EventWriter(config)
    for index in range(8):
        writer.write(
            _event(
                "btcusdt@aggTrade",
                {"s": "BTCUSDT", "E": index, "p": "100", "q": "1", "m": False},
                1_700_000_000_000_000_000 + index,
            )
        )
    writer.close()

    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) > 1
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files) == 8


def test_microstructure_features_and_market_fill_are_causal():
    state = MicrostructureState("BTCUSDT")
    state.apply(
        normalize_event(
            market="futures",
            stream="!forceOrder@arr",
            payload={
                "E": 3,
                "o": {"s": "BTCUSDT", "S": "SELL", "ap": "100", "z": "3"},
            },
            received_ns=3,
        )
    )
    state.apply(
        _event(
            "btcusdt@depth20@100ms",
            {
                "s": "BTCUSDT",
                "E": 1,
                "bids": [["99", "2"], ["98", "3"]],
                "asks": [["101", "1"], ["102", "4"]],
            },
            1,
        )
    )
    state.apply(
        _event(
            "btcusdt@aggTrade",
            {"s": "BTCUSDT", "E": 2, "p": "101", "q": "2", "m": False},
            2,
        )
    )
    state.apply(
        _event(
            "btcusdt@markPrice@1s",
            {"s": "BTCUSDT", "E": 3, "p": "101", "i": "100", "r": "0.0001"},
            3,
        )
    )

    features = state.snapshot(depth=2)
    fill = state.market_fill(side="buy", quantity=2)

    assert features["ok"] is True
    assert features["depth_imbalance"] == pytest.approx(0.0)
    assert features["aggressor_imbalance"] == pytest.approx(1.0)
    assert features["bid_depth_slope_quantity_per_bps"] > 0
    assert features["ask_depth_slope_quantity_per_bps"] > 0
    assert -1 <= features["cancel_add_pressure"] <= 1
    assert features["basis_bps"] == pytest.approx(100.0)
    assert features["liquidation_notional"] == pytest.approx(300.0)
    assert features["liquidation_imbalance"] == pytest.approx(-1.0)
    assert fill["partial_fill"] is False
    assert fill["average_price"] == pytest.approx(101.5)
    assert fill["impact_bps"] == pytest.approx(150.0)


def test_event_replay_is_deterministic_and_rejects_receive_time_regression(tmp_path):
    path = tmp_path / "futures_20260810T00_0000.jsonl"
    events = [
        _event(
            "btcusdt@depth20@100ms",
            {
                "s": "BTCUSDT",
                "E": 1,
                "bids": [["99", "2"]],
                "asks": [["101", "2"]],
            },
            1,
        ),
        _event(
            "btcusdt@aggTrade",
            {"s": "BTCUSDT", "E": 2, "p": "101", "q": "1", "m": False},
            2,
        ),
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    first = replay([path], symbol="BTCUSDT", sample_every=1, market_order=("sell", 1.0))
    second = replay([path], symbol="BTCUSDT", sample_every=1, market_order=("sell", 1.0))

    assert first == second
    assert first["events"] == 2
    assert first["simulated_market_fill"]["average_price"] == pytest.approx(99.0)

    events.reverse()
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    with pytest.raises(ValueError, match="receive order regressed"):
        list(iter_events([path]))


def test_event_replay_runs_short_horizon_microstructure_alpha(tmp_path):
    path = tmp_path / "futures_20260810T00_0000.jsonl"
    events = [
        _event(
            "btcusdt@depth20@100ms",
            {
                "s": "BTCUSDT",
                "E": 1,
                "bids": [["99.99", "10"], ["99.98", "8"]],
                "asks": [["100.01", "1"], ["100.02", "1"]],
            },
            1_000_000_000,
        ),
        _event(
            "btcusdt@aggTrade",
            {"s": "BTCUSDT", "E": 2, "p": "100.01", "q": "5", "m": False},
            2_000_000_000,
        ),
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = replay(
        [path],
        symbol="BTCUSDT",
        sample_every=1,
        microstructure_policy=MicrostructureAlphaPolicy(minimum_abs_score=0.2),
        strategy_quantity=0.5,
    )

    assert report["microstructure_strategy"]["signals"] >= 1
    assert report["microstructure_strategy"]["open_position"]["direction"] == "long"
    assert report["microstructure_strategy"]["live_allowed"] is False


def test_passive_limit_replay_models_queue_partial_fills_and_adverse_selection():
    state = MicrostructureState("BTCUSDT")
    order = RestingLimitOrder(
        side="buy",
        price=99,
        quantity=2,
        submitted_ns=1,
        maker_fee_bps=1,
    )
    depth = _event(
        "btcusdt@depth20@100ms",
        {"s": "BTCUSDT", "E": 1, "bids": [["99", "2"]], "asks": [["101", "2"]]},
        1,
    )
    state.apply(depth)
    order.observe(depth, state)
    for received_ns, quantity in ((2, "1"), (3, "2"), (4, "2")):
        trade = _event(
            "btcusdt@aggTrade",
            {"s": "BTCUSDT", "E": received_ns, "p": "99", "q": quantity, "m": True},
            received_ns,
        )
        state.apply(trade)
        order.observe(trade, state)

    result = order.result(state, finished_ns=5, funding_rate_per_8h=0.0001)

    assert result["filled_quantity"] == pytest.approx(2)
    assert result["fill_ratio"] == pytest.approx(1)
    assert result["average_price"] == pytest.approx(99)
    assert result["queue_ahead_remaining"] == pytest.approx(0)
    assert result["adverse_selection_bps"] < 0


def test_passive_limit_replay_applies_latency_and_cancellation():
    state = MicrostructureState("BTCUSDT")
    depth = _event(
        "btcusdt@depth20@100ms",
        {"s": "BTCUSDT", "E": 1, "bids": [["99", "1"]], "asks": [["101", "1"]]},
        1,
    )
    state.apply(depth)
    order = RestingLimitOrder(
        side="sell",
        price=101,
        quantity=1,
        submitted_ns=1,
        latency_ns=10,
        cancel_after_ns=5,
    )
    order.observe(depth, state)
    early_trade = _event(
        "btcusdt@aggTrade",
        {"s": "BTCUSDT", "E": 2, "p": "101", "q": "10", "m": False},
        2,
    )
    state.apply(early_trade)
    order.observe(early_trade, state)
    heartbeat = _event(
        "btcusdt@markPrice@1s",
        {"s": "BTCUSDT", "E": 16, "p": "100", "i": "100", "r": "0"},
        16,
    )
    state.apply(heartbeat)
    order.observe(heartbeat, state)

    result = order.result(state, finished_ns=16)
    assert result["filled_quantity"] == 0
    assert result["canceled_ns"] == 16


def test_event_capture_config_rejects_non_binance_websocket(tmp_path):
    payload = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    payload["sources"][0]["url"] = "wss://example.com/stream"
    path = tmp_path / "event_capture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="approved Binance"):
        load_event_capture_config(path)
