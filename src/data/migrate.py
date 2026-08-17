"""Apply or verify the platform database migrations."""

from __future__ import annotations

import argparse
import json
import os

from src.data.database import PlatformDatabase


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
        report = {"ok": True, "mode": "migrate", "applied": list(database.migrate())}
    print(json.dumps(report, sort_keys=True))
    database.dispose()


if __name__ == "__main__":
    main()
