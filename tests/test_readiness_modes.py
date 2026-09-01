from __future__ import annotations

from types import SimpleNamespace

from src.services.readiness import _heartbeat_readiness, _readiness_paths, build_readiness


def test_readiness_path_check_does_not_walk_data_tree(tmp_path, monkeypatch) -> None:
    def unexpected_recursive_walk(_path):
        raise AssertionError("readiness must not recursively inventory data files")

    monkeypatch.setattr(type(tmp_path), "rglob", unexpected_recursive_walk)
    config = SimpleNamespace(
        paths={
            "parquet": str(tmp_path / "data"),
            "artefacts": str(tmp_path / "artefacts"),
            "backups": str(tmp_path / "backups"),
        }
    )
    checks: list[dict[str, object]] = []

    paths = _readiness_paths(config, checks)

    assert all(detail["ok"] is True for detail in paths.values())
    assert all("parquet_files" not in detail for detail in paths.values())


def test_readiness_ignores_heartbeats_from_the_future() -> None:
    rows = [
        {
            "service_name": "platform-scheduler",
            "observed_at": "2026-09-01T10:00:01+00:00",
            "healthy": True,
        },
        {
            "service_name": "account-reconciliation",
            "observed_at": "2026-09-01T10:00:01+00:00",
            "healthy": True,
        },
        {
            "service_name": "platform-scheduler",
            "observed_at": "2026-09-01T10:00:02+00:00",
            "healthy": False,
        },
    ]

    report = _heartbeat_readiness(
        rows,
        current="2026-09-01T10:00:01+00:00",
        maximum_age=60.0,
    )

    assert report["ok"] is True
    assert report["details"]["platform-scheduler"]["age_seconds"] == 0.0


def test_readiness_reports_paper_and_live_modes(tmp_path) -> None:
    paper = build_readiness(live=False)
    live = build_readiness(live=True)
    assert paper["mode"] == "paper"
    assert live["mode"] == "live"
