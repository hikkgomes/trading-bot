import dataclasses
import json
from pathlib import Path

from src.autopilot import microstructure_research
from src.autopilot.event_capture import normalize_event


def test_microstructure_research_replays_recent_files_with_hard_event_budget(tmp_path, monkeypatch):
    config = dataclasses.replace(
        microstructure_research.load_config(),
        maximum_symbols=1,
        maximum_files=1,
        maximum_events_per_symbol=100,
        sample_every=1,
    )
    event_root = tmp_path / "events"
    event_root.mkdir()
    capture = dataclasses.replace(
        microstructure_research.load_event_capture_config(config.event_capture_config),
        root=event_root,
    )
    monkeypatch.setattr(microstructure_research, "load_event_capture_config", lambda path: capture)
    path = event_root / "futures_20260810T00_0000.jsonl"
    events = [
        normalize_event(
            market="futures",
            stream="btcusdt@depth20@100ms",
            payload={
                "s": "BTCUSDT",
                "E": 1,
                "bids": [["99.99", "10"]],
                "asks": [["100.01", "1"]],
            },
            received_ns=1_000_000_000,
        ),
        normalize_event(
            market="futures",
            stream="btcusdt@aggTrade",
            payload={"s": "BTCUSDT", "E": 2, "p": "100.01", "q": "5", "m": False},
            received_ns=2_000_000_000,
        ),
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")

    report = microstructure_research.build_report(config)

    assert report["ok"] is True
    assert report["summary"]["events"] == 2
    assert report["summary"]["signals"] >= 1
    assert report["safety"]["order_api_available"] is False


def test_microstructure_research_waits_without_event_files(tmp_path, monkeypatch):
    config = microstructure_research.load_config(Path("config/microstructure_research.json"))
    capture = dataclasses.replace(
        microstructure_research.load_event_capture_config(config.event_capture_config),
        root=tmp_path,
    )
    monkeypatch.setattr(microstructure_research, "load_event_capture_config", lambda path: capture)

    report = microstructure_research.build_report(config)

    assert report["ok"] is True
    assert report["status"] == "waiting_for_events"
