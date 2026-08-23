"""Canonical immutable dataset resolution for every research executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


class DatasetResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedDataset:
    snapshot_id: str
    content_hash: str
    interval: Mapping[str, str]
    universe_snapshot_id: str
    availability_timestamp: str
    feature_manifest_hash: str
    cost_model_hash: str
    parameter_set_hash: str
    product_id: str
    instrument_scope: tuple[str, ...]
    engine_version: str
    payload: Any

    def __post_init__(self) -> None:
        for attribute in (
            "snapshot_id",
            "content_hash",
            "universe_snapshot_id",
            "feature_manifest_hash",
            "cost_model_hash",
            "parameter_set_hash",
            "product_id",
            "engine_version",
        ):
            value = non_empty(getattr(self, attribute), field=attribute)
            object.__setattr__(self, attribute, value)
        for attribute in (
            "snapshot_id",
            "content_hash",
            "feature_manifest_hash",
            "cost_model_hash",
            "parameter_set_hash",
        ):
            value = getattr(self, attribute)
            if not value.startswith("sha256:") or len(value) != 71:
                raise DatasetResolutionError(f"{attribute} must be a SHA-256 identity")
        interval = json_value(dict(self.interval), field="dataset interval")
        if set(interval) != {"start", "end"}:
            raise DatasetResolutionError("dataset interval needs start and end")
        start = timestamp(interval["start"], field="interval.start")
        end = timestamp(interval["end"], field="interval.end")
        if start >= end:
            raise DatasetResolutionError("dataset interval must be chronological")
        object.__setattr__(self, "interval", {"start": start, "end": end})
        object.__setattr__(
            self,
            "availability_timestamp",
            timestamp(self.availability_timestamp, field="availability_timestamp"),
        )
        scope = tuple(non_empty(item, field="instrument_scope") for item in self.instrument_scope)
        if not scope:
            raise DatasetResolutionError("instrument_scope cannot be empty")
        object.__setattr__(self, "instrument_scope", scope)

    @property
    def receipt(self) -> Mapping[str, Any]:
        identities = {
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "interval": self.interval,
            "universe_snapshot_id": self.universe_snapshot_id,
            "availability_timestamp": self.availability_timestamp,
            "feature_manifest_hash": self.feature_manifest_hash,
            "cost_model_hash": self.cost_model_hash,
            "parameter_set_hash": self.parameter_set_hash,
            "product_id": self.product_id,
            "instrument_scope": self.instrument_scope,
            "engine_version": self.engine_version,
        }
        return {**identities, "identity_hash": canonical_hash(identities)}


class DatasetRepository(Protocol):
    def resolve(self, snapshot_id: str) -> ResolvedDataset: ...


class CanonicalDatasetResolver:
    def __init__(self, repository: DatasetRepository):
        self.repository = repository

    def resolve(self, snapshot_id: str, *, expected: Mapping[str, Any]) -> ResolvedDataset:
        dataset = self.repository.resolve(snapshot_id)
        if dataset.snapshot_id != snapshot_id:
            raise DatasetResolutionError("repository returned the wrong snapshot")
        actual = dataset.receipt
        for field, value in expected.items():
            if field not in actual or actual[field] != value:
                raise DatasetResolutionError(f"dataset {field} does not match the request")
        if canonical_hash(dataset.payload) != dataset.content_hash:
            raise DatasetResolutionError("dataset content hash verification failed")
        return dataset
