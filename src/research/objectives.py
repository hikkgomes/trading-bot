"""Product objective contracts used by canonical research acceptance."""

from __future__ import annotations

import math
from collections.abc import Mapping

OBJECTIVE_UNITS = {
    "btc_accumulation": "BTC",
    "active_income": "USDT",
}


def objective_unit(product_id: str) -> str | None:
    """Return the accounting unit for a supported product."""

    return OBJECTIVE_UNITS.get(str(product_id))


def objective_is_available(evidence: Mapping[str, object], *, product_id: str) -> bool:
    """Check that measured objective values are complete and correctly bound."""

    unit = objective_unit(product_id)
    if unit is None or evidence.get("objective_status") != "measured":
        return False
    if evidence.get("objective_unit") != unit:
        return False
    return all(
        _finite(evidence.get(field))
        for field in (
            "objective_value",
            "benchmark_value",
            "objective_excess",
            "objective_excess_fraction",
        )
    )


def objective_passes(
    evidence: Mapping[str, object], *, product_id: str, minimum_excess_fraction: float
) -> bool:
    """Require positive excess in the product's native accounting unit."""

    if not objective_is_available(evidence, product_id=product_id):
        return False
    excess_fraction = float(evidence["objective_excess_fraction"])
    return excess_fraction >= minimum_excess_fraction


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))
