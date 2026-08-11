"""Hash-chained trade journal, reconciliation, and alpha attribution."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.autopilot.io import write_json_atomic, write_text_atomic
from src.config import PROJECT_ROOT

JOURNAL_SCHEMA = "autopilot.accounting_journal/v1"
REPORT_SCHEMA = "autopilot.accounting_report/v1"
GENESIS_HASH = "0" * 64
DEFAULT_JOURNAL = PROJECT_ROOT / "runtime" / "accounting" / "journal.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "runtime" / "accounting" / "report.json"
MAX_TRADE_LOG_BYTES = 100 * 1024 * 1024
MAX_JOURNAL_EVENTS = 1_000_000


def _finite(row: dict[str, str], field: str, *, default: float | None = None) -> float:
    raw = row.get(field)
    if raw in {None, ""} and default is not None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trade field {field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"trade field {field} must be finite")
    return value


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _legacy_event_id(row: dict[str, str], path: Path) -> str:
    """Derive a stable identity only for rows predating execution fingerprints."""
    if row.get("strategy_fingerprint") or row.get("artifact_digest"):
        raise ValueError(
            "modern trade row is missing exit_event_id despite carrying execution identity"
        )
    identity = {
        "schema": "autopilot.legacy_trade_identity/v1",
        "trade_log_name": path.name,
        "strategy_id": row.get("strategy_id"),
        "entry_time": row.get("entry_time"),
        "exit_time": row.get("exit_time"),
        "direction": row.get("direction"),
        "entry_price": row.get("entry_price"),
        "exit_price": row.get("exit_price"),
        "sized_return": row.get("sized_return"),
        "equity_after": row.get("equity_after"),
    }
    required = ("strategy_id", "entry_time", "exit_time", "direction", "equity_after")
    if any(not identity.get(field) for field in required):
        raise ValueError("legacy trade row lacks fields required for deterministic identity")
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_trade_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trade log must be a regular non-symlink file: {path}")
        if path.stat().st_size > MAX_TRADE_LOG_BYTES:
            raise ValueError(f"trade log exceeds the accounting read budget: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                event_id = str(row.get("exit_event_id") or "")
                legacy_identity = False
                if not event_id:
                    event_id = _legacy_event_id(row, path)
                    row["exit_event_id"] = event_id
                    legacy_identity = True
                if len(event_id) != 64 or any(char not in "0123456789abcdef" for char in event_id):
                    raise ValueError(f"trade log row has invalid exit_event_id: {path}")
                if event_id in seen_ids:
                    raise ValueError(f"duplicate exit_event_id across trade logs: {event_id}")
                seen_ids.add(event_id)
                row["_trade_log"] = str(path)
                row["_legacy_exit_event_id"] = "true" if legacy_identity else "false"
                rows.append(row)
    return sorted(rows, key=lambda row: (row.get("exit_time") or "", row["exit_event_id"]))


def _money(value: float) -> str:
    return format(Decimal(str(value)).quantize(Decimal("0.000000000001")), "f")


def journal_event(row: dict[str, str], *, previous_hash: str) -> dict[str, Any]:
    sized_return = _finite(row, "sized_return")
    equity_after = _finite(row, "equity_after")
    if sized_return <= -1:
        raise ValueError("sized_return must be greater than -1")
    equity_before = equity_after / (1 + sized_return)
    position_fraction = _finite(row, "position_size")
    gross = equity_before * _finite(row, "gross_return") * position_fraction
    costs = (
        equity_before * _finite(row, "transaction_cost_fraction", default=0.0) * position_fraction
    )
    net = equity_after - equity_before
    adjustment = net - (gross - costs)
    entries = [
        {"account": "assets:trading_equity", "amount": _money(net)},
        {"account": "income:trading", "amount": _money(-gross)},
        {"account": "expense:transaction_costs", "amount": _money(costs)},
    ]
    entries.append(
        {
            "account": "equity:accounting_adjustment",
            "amount": format(-sum(Decimal(entry["amount"]) for entry in entries), "f"),
        }
    )
    if sum(Decimal(entry["amount"]) for entry in entries) != 0:
        raise RuntimeError("accounting journal event does not balance")
    body = {
        "schema": JOURNAL_SCHEMA,
        "event_id": row["exit_event_id"],
        "recorded_at": row.get("exit_time"),
        "previous_hash": previous_hash,
        "source": {
            "trade_log": row["_trade_log"],
            "strategy_id": row.get("strategy_id"),
            "strategy_fingerprint": row.get("strategy_fingerprint"),
            "artifact_digest": row.get("artifact_digest"),
            "alpha_source_id": row.get("alpha_source_id"),
            "product": row.get("alpha_product"),
            "market": row.get("alpha_market"),
            "symbol": row.get("alpha_symbol") or row.get("broker_symbol"),
            "legacy_exit_event_id": row.get("_legacy_exit_event_id") == "true",
        },
        "measurement": {
            "equity_before": _money(equity_before),
            "equity_after": _money(equity_after),
            "gross_pnl": _money(gross),
            "transaction_cost": _money(costs),
            "accounting_adjustment": _money(adjustment),
            "net_pnl": _money(net),
            "accounting_return_source": row.get("accounting_return_source"),
        },
        "entries": entries,
    }
    return {**body, "event_hash": _canonical_hash(body)}


def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError("accounting journal must be a regular non-symlink file")
    events: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("schema") != JOURNAL_SCHEMA:
            raise ValueError(f"accounting journal schema is invalid at line {line_number}")
        event_hash = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_hash") != previous_hash or event_hash != _canonical_hash(body):
            raise ValueError(f"accounting journal hash chain is invalid at line {line_number}")
        event_id = str(event.get("event_id") or "")
        if event_id in seen:
            raise ValueError(f"accounting journal has duplicate event_id {event_id}")
        if sum(Decimal(entry["amount"]) for entry in event.get("entries") or []) != 0:
            raise ValueError(f"accounting journal event is unbalanced at line {line_number}")
        seen.add(event_id)
        previous_hash = str(event_hash)
        events.append(event)
        if len(events) > MAX_JOURNAL_EVENTS:
            raise ValueError("accounting journal exceeds maximum event count")
    return events


def update_journal(
    rows: list[dict[str, str]], path: Path = DEFAULT_JOURNAL
) -> list[dict[str, Any]]:
    existing = load_journal(path)
    seen = {str(event["event_id"]) for event in existing}
    previous_hash = str(existing[-1]["event_hash"]) if existing else GENESIS_HASH
    events = list(existing)
    for row in rows:
        if row["exit_event_id"] in seen:
            continue
        event = journal_event(row, previous_hash=previous_hash)
        events.append(event)
        seen.add(row["exit_event_id"])
        previous_hash = str(event["event_hash"])
    text = "".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n" for event in events)
    write_text_atomic(path, text)
    return events


def _attribution(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    result: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        net = [_finite(item, "net_return") for item in items]
        sized = [_finite(item, "sized_return") for item in items]
        expected = [_finite(item, "alpha_expected_return", default=0.0) for item in items]
        result[key] = {
            "trades": len(items),
            "wins": sum(value > 0 for value in net),
            "win_rate": sum(value > 0 for value in net) / len(items),
            "net_return_sum": sum(net),
            "sized_return_sum": sum(sized),
            "transaction_cost_fraction_sum": sum(
                _finite(item, "transaction_cost_fraction", default=0.0) for item in items
            ),
            "mean_expected_return": sum(expected) / len(items),
            "mean_realized_return": sum(net) / len(items),
            "forecast_bias": sum(
                realized - forecast for realized, forecast in zip(net, expected, strict=True)
            )
            / len(items),
        }
    return result


def build_accounting_report(
    rows: list[dict[str, str]],
    journal: list[dict[str, Any]],
) -> dict[str, Any]:
    reconciliation_errors: list[dict[str, Any]] = []
    previous_equity_by_account: dict[str, float] = {}
    for row in rows:
        sized = _finite(row, "sized_return")
        after = _finite(row, "equity_after")
        before = after / (1 + sized)
        account = row["_trade_log"]
        previous_equity = previous_equity_by_account.get(account)
        if previous_equity is not None and not math.isclose(
            before, previous_equity, rel_tol=1e-9, abs_tol=1e-8
        ):
            reconciliation_errors.append(
                {
                    "exit_event_id": row["exit_event_id"],
                    "account": account,
                    "expected_equity_before": previous_equity,
                    "observed_equity_before": before,
                }
            )
        previous_equity_by_account[account] = after
    dimensions = (
        "strategy_id",
        "alpha_source_id",
        "alpha_product",
        "alpha_market",
        "alpha_symbol",
        "direction",
        "accounting_return_source",
    )
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "ok": not reconciliation_errors,
        "summary": {
            "trades": len(rows),
            "legacy_trade_identities": sum(
                row.get("_legacy_exit_event_id") == "true" for row in rows
            ),
            "journal_events": len(journal),
            "latest_journal_hash": journal[-1]["event_hash"] if journal else GENESIS_HASH,
            "reconciliation_errors": len(reconciliation_errors),
        },
        "reconciliation_errors": reconciliation_errors,
        "attribution": {dimension: _attribution(rows, dimension) for dimension in dimensions},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reconciled, hash-chained trade accounting.")
    parser.add_argument("trade_logs", nargs="*", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    trade_logs = list(args.trade_logs)
    if args.config is not None:
        from src.autopilot.config import load_config

        trade_logs.extend(
            product.trade_log
            for product in load_config(args.config).products
            if product.trade_log.exists()
        )
    trade_logs = list(dict.fromkeys(trade_logs))
    if not trade_logs and args.config is None:
        raise SystemExit("pass at least one trade log or --config")
    rows = load_trade_rows(trade_logs)
    journal = update_journal(rows, args.journal)
    report = build_accounting_report(rows, journal)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
