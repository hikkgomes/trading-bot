import pytest

from src.autopilot.research_smoke import run_research_smoke


def test_research_smoke_runs_both_products_on_synthetic_data():
    report = run_research_smoke(synthetic_rows=700)

    assert report["ok"] is True
    assert report["synthetic_only"] is True
    assert {scenario["name"] for scenario in report["scenarios"]} == {
        "active_income",
        "btc_accumulation",
    }
    assert {scenario["pnl_unit"] for scenario in report["scenarios"]} == {"usdt", "btc"}
    assert all(scenario["hypotheses"] > 0 for scenario in report["scenarios"])
    assert all(scenario["verdicts"] for scenario in report["scenarios"])


def test_research_smoke_rejects_tiny_synthetic_frame():
    with pytest.raises(ValueError, match="synthetic_rows must be at least 500"):
        run_research_smoke(synthetic_rows=100)
