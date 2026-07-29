"""Shared development-history policy for generated research search spaces.

This module intentionally has no imports from the factory, research cycle, or
history downloader.  Both sides can depend on the same policy without creating
an orchestration import cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class SearchSpaceHistoryIdentity(Protocol):
    product: str
    base_timeframe: str


def generated_history_contract(space: SearchSpaceHistoryIdentity) -> dict[str, object]:
    """Return the fail-closed development-history contract for a search space."""

    if space.product == "btc_accumulation":
        return {
            "start": "2020-06-01",
            "coverage_earliest": "2020-06-01",
            "coverage_max_start_delay_days": 2,
            "coverage_max_latest_age_hours": 24,
            "coverage_min_span_days": 1_000,
            "coverage_min_rows": 8_000 if space.base_timeframe == "4h" else 30_000,
        }
    if space.base_timeframe == "1m":
        return {
            "start": "2026-01-01",
            "coverage_earliest": "2026-01-01",
            "coverage_max_start_delay_days": 1,
            "coverage_max_latest_age_hours": 24,
            "coverage_min_span_days": 90,
            "coverage_min_rows": 100_000,
        }
    if space.base_timeframe == "5m":
        return {
            "start": "2023-01-01",
            "coverage_earliest": "2023-01-01",
            "coverage_max_start_delay_days": 1,
            "coverage_max_latest_age_hours": 24,
            "coverage_min_span_days": 730,
            "coverage_min_rows": 200_000,
        }
    return {
        "start": "2022-01-01",
        "coverage_earliest": "2022-01-01",
        "coverage_max_start_delay_days": 2,
        "coverage_max_latest_age_hours": 24,
        "coverage_min_span_days": 900,
        "coverage_min_rows": 20_000,
    }


def listing_history_compatibility(
    space: SearchSpaceHistoryIdentity,
    *,
    listing_days: float,
    as_of: str | datetime,
) -> dict[str, object]:
    """Check whether a symbol can physically satisfy a search-space contract."""

    reference = (
        as_of
        if isinstance(as_of, datetime)
        else datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
    )
    reference = (
        reference.replace(tzinfo=UTC) if reference.tzinfo is None else reference.astimezone(UTC)
    )
    contract = generated_history_contract(space)
    earliest = datetime.fromisoformat(str(contract["coverage_earliest"])).replace(tzinfo=UTC)
    maximum_delay_days = float(contract["coverage_max_start_delay_days"])
    latest_compatible_listing = earliest + timedelta(days=maximum_delay_days)
    listed_at = reference - timedelta(days=float(listing_days))
    compatible = listed_at <= latest_compatible_listing
    return {
        "ok": compatible,
        "listing_days": round(float(listing_days), 3),
        "listed_at": listed_at.isoformat(),
        "coverage_earliest": earliest.isoformat(),
        "latest_compatible_listing": latest_compatible_listing.isoformat(),
    }
