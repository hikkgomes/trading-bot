"""Apply or verify the platform database migrations."""

from __future__ import annotations

import argparse
import json
import os

from src.data.database import PlatformDatabase
from src.services.platform_upgrade import quarantine_legacy_research_jobs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply or verify platform PostgreSQL migrations.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("TRADING_PLATFORM_DATABASE_URL", ""),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("--database-url or TRADING_PLATFORM_DATABASE_URL is required")
    database = PlatformDatabase(args.database_url)
    if not database.is_postgresql:
        raise SystemExit("platform migrations require PostgreSQL")
    if args.check:
        database.assert_migrated()
        report = {"ok": True, "mode": "check"}
    else:
        applied = list(database.migrate())
        legacy_queue = quarantine_legacy_research_jobs(database.engine)
        legacy_queue.pop("quarantined_jobs", None)
        report = {
            "ok": True,
            "mode": "migrate",
            "applied": applied,
            "legacy_queue": legacy_queue,
        }
    print(json.dumps(report, sort_keys=True))
    database.dispose()


if __name__ == "__main__":
    main()
