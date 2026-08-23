"""Verify the PostgreSQL platform event-to-trade chain for both products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.data.database import (
    PlatformDatabase,
    accounting_entry,
    alpha_forecast,
    decision_trace,
    fill,
    order_intent,
    position,
    risk_decision,
    strategy_approval,
    target_position,
)
from src.services.config import load_split_configuration

_STAGES = (
    ("strategy_assignment", strategy_approval),
    ("alpha_forecast", alpha_forecast),
    ("target_position", target_position),
    ("risk_decision", risk_decision),
    ("order_intent", order_intent),
    ("fill", fill),
    ("position", position),
    ("accounting_entry", accounting_entry),
    ("decision_trace", decision_trace),
)


def run_smoke(
    database_url: str, *, config_path: Path = Path("config/platform.json")
) -> dict[str, Any]:
    database = PlatformDatabase(database_url)
    if not database.is_postgresql:
        raise ValueError("platform smoke requires PostgreSQL")
    database.assert_migrated()
    products = load_split_configuration(config_path.parent)["products"]["products"]
    results: list[dict[str, Any]] = []
    try:
        with database.engine.connect() as connection:
            for product in products:
                product_id = str(product["product_id"])
                blocked = None
                counts: dict[str, int] = {}
                for stage, table in _STAGES:
                    if stage == "strategy_assignment":
                        statement = select(strategy_approval.c.id).where(
                            strategy_approval.c.product_id == product_id
                        )
                    elif "product_id" in table.c:
                        statement = select(table.c.id).where(table.c.product_id == product_id)
                    else:
                        statement = select(table.c.id)
                    count = len(connection.execute(statement).all())
                    counts[stage] = count
                    if blocked is None and count == 0:
                        blocked = stage
                results.append(
                    {
                        "product_id": product_id,
                        "ok": blocked is None,
                        "first_blocked_stage": blocked,
                        "counts": counts,
                    }
                )
    finally:
        database.dispose()
    return {
        "schema": "platform.smoke/v1",
        "ok": all(item["ok"] for item in results),
        "empty_database": all(
            not any(counts.values()) for counts in (item["counts"] for item in results)
        ),
        "products": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="pass a clean migrated database while retaining blocked-stage output",
    )
    args = parser.parse_args(argv)
    report = run_smoke(args.database_url, config_path=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    empty_allowed = args.allow_empty and report["empty_database"]
    raise SystemExit(0 if report["ok"] or empty_allowed else 1)


if __name__ == "__main__":
    main()
