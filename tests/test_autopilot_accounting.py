import csv
import json

import pytest

from src.autopilot.accounting import (
    build_accounting_report,
    load_journal,
    load_trade_rows,
    update_journal,
)


def _write_trades(path):
    fields = [
        "exit_event_id",
        "strategy_id",
        "strategy_fingerprint",
        "artifact_digest",
        "exit_time",
        "direction",
        "gross_return",
        "transaction_cost_fraction",
        "accounting_return_source",
        "net_return",
        "sized_return",
        "position_size",
        "equity_after",
        "alpha_source_id",
        "alpha_product",
        "alpha_market",
        "alpha_symbol",
        "alpha_expected_return",
    ]
    rows = [
        {
            "exit_event_id": "a" * 64,
            "strategy_id": "trend",
            "strategy_fingerprint": "sha256:" + "1" * 64,
            "artifact_digest": "sha256:" + "2" * 64,
            "exit_time": "2026-08-10T01:00:00+00:00",
            "direction": "long",
            "gross_return": "0.012",
            "transaction_cost_fraction": "0.002",
            "accounting_return_source": "modeled_trade",
            "net_return": "0.01",
            "sized_return": "0.01",
            "position_size": "1",
            "equity_after": "1010",
            "alpha_source_id": "trend",
            "alpha_product": "active_income",
            "alpha_market": "futures",
            "alpha_symbol": "BTCUSDT",
            "alpha_expected_return": "0.006",
        },
        {
            "exit_event_id": "b" * 64,
            "strategy_id": "reversion",
            "strategy_fingerprint": "sha256:" + "3" * 64,
            "artifact_digest": "sha256:" + "4" * 64,
            "exit_time": "2026-08-10T02:00:00+00:00",
            "direction": "short",
            "gross_return": "-0.003",
            "transaction_cost_fraction": "0.002",
            "accounting_return_source": "broker_balance",
            "net_return": "-0.005",
            "sized_return": "-0.005",
            "position_size": "1",
            "equity_after": "1004.95",
            "alpha_source_id": "reversion",
            "alpha_product": "active_income",
            "alpha_market": "futures",
            "alpha_symbol": "ETHUSDT",
            "alpha_expected_return": "0.004",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_accounting_journal_is_balanced_hash_chained_and_idempotent(tmp_path):
    trade_log = tmp_path / "trades.csv"
    journal_path = tmp_path / "journal.jsonl"
    _write_trades(trade_log)
    rows = load_trade_rows([trade_log])

    first = update_journal(rows, journal_path)
    second = update_journal(rows, journal_path)

    assert first == second
    assert len(second) == 2
    assert second[0]["previous_hash"] == "0" * 64
    assert second[1]["previous_hash"] == second[0]["event_hash"]
    assert sum(float(entry["amount"]) for entry in second[0]["entries"]) == pytest.approx(0)


def test_accounting_journal_detects_tampering(tmp_path):
    trade_log = tmp_path / "trades.csv"
    journal_path = tmp_path / "journal.jsonl"
    _write_trades(trade_log)
    update_journal(load_trade_rows([trade_log]), journal_path)
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    events[0]["measurement"]["net_pnl"] = "999"
    journal_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    with pytest.raises(ValueError, match="hash chain"):
        load_journal(journal_path)


def test_accounting_report_reconciles_equity_and_attributes_alpha(tmp_path):
    trade_log = tmp_path / "trades.csv"
    journal_path = tmp_path / "journal.jsonl"
    _write_trades(trade_log)
    rows = load_trade_rows([trade_log])
    journal = update_journal(rows, journal_path)

    report = build_accounting_report(rows, journal)

    assert report["ok"] is True
    assert report["summary"]["trades"] == 2
    assert report["summary"]["reconciliation_errors"] == 0
    btc = report["attribution"]["alpha_symbol"]["BTCUSDT"]
    assert btc["trades"] == 1
    assert btc["mean_realized_return"] == pytest.approx(0.01)
    assert btc["forecast_bias"] == pytest.approx(0.004)


def test_accounting_migrates_pre_identity_rows_deterministically(tmp_path):
    trade_log = tmp_path / "legacy.csv"
    trade_log.write_text(
        "exit_event_id,strategy_id,strategy_fingerprint,artifact_digest,entry_time,exit_time,"
        "direction,entry_price,exit_price,gross_return,net_return,sized_return,position_size,"
        "equity_after\n"
        ",legacy,,,2026-01-01T00:00:00Z,2026-01-01T01:00:00Z,long,100,101,0.01,"
        "0.009,0.009,1,1009\n",
        encoding="utf-8",
    )

    first = load_trade_rows([trade_log])
    second = load_trade_rows([trade_log])

    assert first[0]["exit_event_id"] == second[0]["exit_event_id"]
    assert first[0]["_legacy_exit_event_id"] == "true"
    assert len(first[0]["exit_event_id"]) == 64


def test_accounting_rejects_modern_row_without_exit_event_id(tmp_path):
    trade_log = tmp_path / "invalid.csv"
    trade_log.write_text(
        "exit_event_id,strategy_id,strategy_fingerprint,artifact_digest,entry_time,exit_time,"
        "direction,equity_after\n"
        ",modern,sha256:abc,sha256:def,2026-01-01T00:00:00Z,"
        "2026-01-01T01:00:00Z,long,1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="modern trade row"):
        load_trade_rows([trade_log])


def test_accounting_reconciles_each_product_ledger_independently(tmp_path):
    first_log = tmp_path / "btc.csv"
    second_log = tmp_path / "usdt.csv"
    _write_trades(first_log)
    _write_trades(second_log)
    second_text = second_log.read_text(encoding="utf-8")
    second_log.write_text(
        second_text.replace("a" * 64, "c" * 64).replace("b" * 64, "d" * 64),
        encoding="utf-8",
    )
    rows = load_trade_rows([first_log, second_log])
    journal = update_journal(rows, tmp_path / "journal.jsonl")

    report = build_accounting_report(rows, journal)

    assert report["ok"] is True
    assert report["summary"]["trades"] == 4
