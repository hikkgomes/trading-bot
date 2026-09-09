"""Typed exposure limits; ratios and currency amounts are separate concepts."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class ExposureScope(StrEnum):
    PRODUCT_GROSS = "product_gross"
    PRODUCT_NET = "product_net"
    INSTRUMENT = "instrument"
    ORDER = "order"
    ASSIGNMENT = "assignment"
    SLEEVE = "sleeve"
    ACCOUNT = "account"


@dataclass(frozen=True)
class ExposureLimit:
    name: str
    scope: ExposureScope
    maximum: float
    currency: str = "USDT"

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum) or self.maximum <= 0:
            raise ValueError("exposure limit must be a positive currency amount")


def assert_exposure_limits(
    values: Mapping[ExposureScope, float], limits: Iterable[ExposureLimit]
) -> None:
    for limit in limits:
        value = values[limit.scope]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {limit.scope} exposure")
        if value > limit.maximum + max(1e-9, abs(limit.maximum) * 1e-12):
            raise ValueError(f"live order exceeds {limit.name}: {value:g} > {limit.maximum:g}")
