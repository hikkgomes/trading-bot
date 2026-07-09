import json

from src.autopilot.config import JobConfig
from src.autopilot.regime_data import build_regime_data_statuses


def _job(tmp_path, enabled=True):
    return JobConfig(
        name="regime_tag_futures_15m",
        enabled=enabled,
        command=[
            ".venv/bin/python",
            "-m",
            "src.regime",
            "--output",
            str(tmp_path / "regime.parquet"),
            "--report",
            str(tmp_path / "regime.json"),
        ],
        cadence_seconds=3600,
    )


def _inline_job(tmp_path, enabled=True):
    return JobConfig(
        name="regime_tag_futures_15m",
        enabled=enabled,
        command=[
            ".venv/bin/python",
            "-m",
            "src.regime",
            f"--output={tmp_path / 'regime.parquet'}",
            f"--report={tmp_path / 'regime.json'}",
        ],
        cadence_seconds=3600,
    )


def test_regime_data_status_ready(tmp_path):
    output = tmp_path / "regime.parquet"
    output.write_text("placeholder", encoding="utf-8")
    (tmp_path / "regime.json").write_text(
        json.dumps(
            {
                "ok": True,
                "skipped": False,
                "output": str(output),
                "rows": 12,
                "regime_counts": {"0": 8, "1": 4},
            }
        ),
        encoding="utf-8",
    )

    status = build_regime_data_statuses([_job(tmp_path)])[0]

    assert status["available"] is True
    assert status["ok"] is True
    assert status["rows"] == 12
    assert status["regime_counts"] == {"0": 8, "1": 4}


def test_regime_data_status_ready_with_inline_flags(tmp_path):
    output = tmp_path / "regime.parquet"
    output.write_text("placeholder", encoding="utf-8")
    (tmp_path / "regime.json").write_text(
        json.dumps(
            {
                "ok": True,
                "skipped": False,
                "output": str(output),
                "rows": 12,
                "regime_counts": {"0": 8, "1": 4},
            }
        ),
        encoding="utf-8",
    )

    status = build_regime_data_statuses([_inline_job(tmp_path)])[0]

    assert status["available"] is True
    assert status["ok"] is True
    assert status["output_path"] == str(output)


def test_regime_data_status_missing_report(tmp_path):
    status = build_regime_data_statuses([_job(tmp_path)])[0]

    assert status["available"] is False
    assert status["ok"] is False
    assert status["reason"] == "missing_report"


def test_regime_data_status_disabled_job_is_neutral(tmp_path):
    status = build_regime_data_statuses([_job(tmp_path, enabled=False)])[0]

    assert status["available"] is None
    assert status["ok"] is True
    assert status["reason"] == "disabled"
