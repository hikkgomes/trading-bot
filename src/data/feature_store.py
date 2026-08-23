"""Availability-safe deterministic feature values for live and historical use."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import feature_manifest
from src.domain._codec import canonical_hash, finite, non_empty, timestamp, to_primitive


@dataclass(frozen=True)
class FeatureValue:
    feature_set_version: str
    feature_name: str
    instrument_id: str
    source_event_time: str
    source_close_time: str
    availability_time: str
    value: float

    def __post_init__(self) -> None:
        for attribute in ("feature_set_version", "feature_name", "instrument_id"):
            object.__setattr__(
                self, attribute, non_empty(getattr(self, attribute), field=attribute)
            )
        for attribute in ("source_event_time", "source_close_time", "availability_time"):
            object.__setattr__(
                self, attribute, timestamp(getattr(self, attribute), field=attribute)
            )
        if self.availability_time < self.source_close_time:
            raise ValueError("features cannot be available before the source candle closes")
        object.__setattr__(self, "value", finite(self.value, field="value"))

    @property
    def feature_id(self) -> str:
        return canonical_hash(self)


FeatureFunction = Callable[[Mapping[str, float]], Mapping[str, float]]


class DeterministicFeatureCalculator:
    def __init__(self, *, version: str, function: FeatureFunction):
        self.version = non_empty(version, field="version")
        self.function = function

    def calculate(
        self,
        *,
        instrument_id: str,
        source_event_time: str,
        source_close_time: str,
        availability_time: str,
        inputs: Mapping[str, float],
    ) -> tuple[FeatureValue, ...]:
        first = dict(self.function(dict(inputs)))
        second = dict(self.function(dict(inputs)))
        if first != second:
            raise ValueError("feature calculation is not deterministic")
        return tuple(
            FeatureValue(
                feature_set_version=self.version,
                feature_name=name,
                instrument_id=instrument_id,
                source_event_time=source_event_time,
                source_close_time=source_close_time,
                availability_time=availability_time,
                value=value,
            )
            for name, value in sorted(first.items())
        )

    @staticmethod
    def assert_live_historical_match(
        historical: Iterable[FeatureValue], live: Iterable[FeatureValue]
    ) -> None:
        historical_values = tuple(historical)
        live_values = tuple(live)
        if historical_values != live_values:
            raise ValueError("historical and live feature values differ")


class SqlFeatureStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, values: Iterable[FeatureValue]) -> tuple[str, ...]:
        identities: list[str] = []
        with self.engine.begin() as connection:
            for value in values:
                payload = to_primitive(value)
                identity = value.feature_id
                existing = connection.execute(
                    select(feature_manifest.c.payload).where(feature_manifest.c.id == identity)
                ).scalar_one_or_none()
                if existing is None:
                    connection.execute(
                        insert(feature_manifest).values(
                            id=identity,
                            created_at=value.availability_time,
                            payload=payload,
                        )
                    )
                elif dict(existing) != payload:
                    raise ValueError("feature content-hash collision")
                identities.append(identity)
        return tuple(identities)

    def available(
        self,
        *,
        instrument_id: str,
        at: str,
        feature_set_version: str,
    ) -> tuple[FeatureValue, ...]:
        at = timestamp(at, field="at")
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(feature_manifest.c.payload).order_by(feature_manifest.c.created_at)
            ).scalars()
            values = tuple(FeatureValue(**dict(payload)) for payload in payloads)
        return tuple(
            value
            for value in values
            if value.instrument_id == instrument_id
            and value.feature_set_version == feature_set_version
            and value.availability_time <= at
        )

    def by_ids(self, feature_ids: Iterable[str]) -> tuple[FeatureValue, ...]:
        expected = tuple(feature_ids)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("feature_ids must be a non-empty unique sequence")
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(feature_manifest.c.payload).where(feature_manifest.c.id.in_(expected))
            ).scalars()
            values = {value.feature_id: value for value in (FeatureValue(**dict(item)) for item in payloads)}
        if set(values) != set(expected):
            raise KeyError("one or more canonical feature IDs do not exist")
        return tuple(values[item] for item in expected)
