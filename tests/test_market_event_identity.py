from __future__ import annotations

from src.domain.market_events import ExchangeSequenceTracker, MarketEvent, MarketEventType


def _event(received: str, sequence: int = 7) -> MarketEvent:
    return MarketEvent(
        instrument_id="binance:futures:BTCUSDT:USDT",
        event_type=MarketEventType.TRADE,
        exchange_timestamp="2026-08-23T00:00:00+00:00",
        receive_timestamp=received,
        sequence=sequence,
        payload={"data": {"a": 99, "p": "100"}},
    )


def test_receive_time_is_observation_metadata_not_event_identity() -> None:
    assert (
        _event("2026-08-23T00:00:01+00:00").event_id == _event("2026-08-23T00:00:02+00:00").event_id
    )


def test_exchange_sequence_tracker_reports_duplicates_and_gaps() -> None:
    tracker = ExchangeSequenceTracker()
    assert tracker.observe(_event("2026-08-23T00:00:01+00:00", 1)) == "ok"
    assert tracker.observe(_event("2026-08-23T00:00:02+00:00", 1)) == "duplicate"
    assert tracker.observe(_event("2026-08-23T00:00:03+00:00", 3)) == "gap"
