"""Aggregate forecasts by instrument without giving strategies order priority."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from src.domain.forecasts import AlphaForecast, ForecastDirection


def aggregate_forecasts(forecasts: Iterable[AlphaForecast]) -> tuple[AlphaForecast, ...]:
    """Return one confidence-weighted forecast per product and instrument.

    A directional conflict decreases score and confidence. It never silently
    chooses whichever strategy happened to run first.
    """
    grouped: dict[tuple[str, str], list[AlphaForecast]] = defaultdict(list)
    for forecast in forecasts:
        grouped[(forecast.product_id, forecast.instrument_id)].append(forecast)
    result: list[AlphaForecast] = []
    for _, items in sorted(grouped.items()):
        reference = items[0]
        if len(items) == 1:
            result.append(
                replace(
                    reference,
                    metadata={
                        **dict(reference.metadata),
                        "contributors": [reference.strategy_version_id],
                        "contributor_count": 1,
                        "agreement": 1.0,
                    },
                )
            )
            continue
        weights = [item.confidence * item.score for item in items]
        signed = [
            weight * item.signed_strength for item, weight in zip(items, weights, strict=True)
        ]
        total_weight = sum(weights)
        net = sum(signed)
        direction = (
            ForecastDirection.LONG
            if net > 0
            else ForecastDirection.SHORT
            if net < 0
            else ForecastDirection.FLAT
        )
        agreement = abs(net) / total_weight if total_weight else 0.0
        score = min(1.0, agreement)
        confidence = min(1.0, sum(item.confidence for item in items) / len(items) * agreement)
        weighted_return = (
            sum(item.expected_return * weight for item, weight in zip(items, weights, strict=True))
            / total_weight
            if total_weight
            else 0.0
        )
        confidence_total = sum(item.confidence for item in items)
        maximum_position = min(
            1.0,
            sum(item.maximum_position * item.confidence for item in items) / confidence_total
            if confidence_total
            else 0.0,
        )
        if direction is ForecastDirection.FLAT:
            maximum_position = 0.0
        metadata = {
            "contributors": [item.strategy_version_id for item in items],
            "contributor_count": len(items),
            "agreement": agreement,
            "assignment_ids": sorted(
                {
                    str(item.metadata["assignment_id"])
                    for item in items
                    if item.metadata.get("assignment_id")
                }
            ),
        }
        group_keys = tuple(
            sorted(
                {
                    str(item.metadata["order_group_key"])
                    for item in items
                    if item.metadata.get("order_group_key")
                }
            )
        )
        if len(group_keys) == 1:
            metadata["order_group_key"] = group_keys[0]
        semantic_groups = tuple(
            sorted(
                {
                    str(item.metadata["semantic_group_id"])
                    for item in items
                    if item.metadata.get("semantic_group_id")
                }
            )
        )
        if len(semantic_groups) == 1:
            metadata["semantic_group_id"] = semantic_groups[0]
        assignment_ids = metadata["assignment_ids"]
        if len(assignment_ids) == 1:
            metadata["assignment_id"] = assignment_ids[0]
        result.append(
            replace(
                reference,
                strategy_version_id="ensemble:"
                + "+".join(sorted(item.strategy_version_id for item in items)),
                direction=direction,
                score=score,
                expected_return=weighted_return,
                confidence=confidence,
                maximum_position=maximum_position,
                metadata=metadata,
            )
        )
    return tuple(result)
