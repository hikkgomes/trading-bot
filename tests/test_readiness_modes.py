from __future__ import annotations

from src.services.readiness import build_readiness


def test_readiness_reports_paper_and_live_modes(tmp_path) -> None:
    paper = build_readiness(live=False)
    live = build_readiness(live=True)
    assert paper["mode"] == "paper"
    assert live["mode"] == "live"
