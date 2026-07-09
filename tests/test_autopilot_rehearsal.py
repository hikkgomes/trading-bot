import json
from pathlib import Path

from src.autopilot.rehearsal import run_rehearsal


def test_run_rehearsal_writes_end_to_end_artifacts(tmp_path):
    report = run_rehearsal(tmp_path)

    assert report["ok"] is True
    assert report["before_recommendation"] == "needs_approval"
    assert report["after_recommendation"] == "already_approved"
    assert report["preflight_ok"] is True
    assert set(report["products"]) == {"active_income", "btc_accumulation"}
    assert set(report["preflight_products"]) == {"active_income", "btc_accumulation"}

    for key in (
        "artifact",
        "trade_log",
        "promotion_review_json",
        "promotion_review_md",
        "preflight_report",
    ):
        assert Path(report[key]).exists()

    preflight = json.loads((tmp_path / "preflight_report.json").read_text(encoding="utf-8"))
    assert preflight["ok"] is True
    assert len(preflight["products"]) == 2
    checks = {item["name"]: item for item in preflight["products"][0]["checks"]}
    assert checks["approval_gate"]["ok"] is True
    assert checks["exchange_read_connectivity"]["detail"]["price"] == 100.0
    for product_name, product_report in report["products"].items():
        assert product_report["before_recommendation"] == "needs_approval"
        assert product_report["after_recommendation"] == "already_approved"
        for key in ("artifact", "trade_log", "promotion_review_json", "promotion_review_md"):
            assert Path(product_report[key]).exists(), product_name

    btc_artifact = json.loads(
        Path(report["products"]["btc_accumulation"]["artifact"]).read_text(encoding="utf-8")
    )
    btc_strategy = btc_artifact["strategies"][0]
    assert btc_artifact["market"] == "spot"
    assert btc_strategy["pnl_unit"] == "btc"
    assert btc_strategy["metrics"]["holdout_excess_return_vs_buy_hold"] > 0


def test_run_rehearsal_is_repeatable(tmp_path):
    first = run_rehearsal(tmp_path)
    second = run_rehearsal(tmp_path)

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["before_recommendation"] == "needs_approval"
    assert second["after_recommendation"] == "already_approved"
    assert set(second["products"]) == {"active_income", "btc_accumulation"}
