"""Canonical immutable dataset resolution for every research executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.data.database import dataset_snapshot, universe_snapshot
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


class DatasetResolutionError(RuntimeError):
    pass


DATASET_ROLES = frozenset(
    {
        "screening",
        "development",
        "robustness",
        "protected_holdout",
        "forward_observation",
        "unspecified",
    }
)


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
    model_artefact_id: str | None = None
    event_data_segment_ids: tuple[str, ...] = ()
    role: str = "unspecified"

    def __post_init__(self) -> None:
        for attribute in (
            "snapshot_id",
            "content_hash",
            "universe_snapshot_id",
            "feature_manifest_hash",
            "cost_model_hash",
            "parameter_set_hash",
            "universe_snapshot_id",
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
        if self.model_artefact_id is not None and (
            not self.model_artefact_id.startswith("sha256:") or len(self.model_artefact_id) != 71
        ):
            raise DatasetResolutionError("model_artefact_id must be a SHA-256 identity")
        event_ids = tuple(self.event_data_segment_ids)
        if len(event_ids) != len(set(event_ids)) or any(
            not value.startswith("sha256:") or len(value) != 71 for value in event_ids
        ):
            raise DatasetResolutionError(
                "event_data_segment_ids must contain unique SHA-256 identities"
            )
        object.__setattr__(self, "event_data_segment_ids", event_ids)
        role = non_empty(self.role, field="role")
        if role not in DATASET_ROLES:
            raise DatasetResolutionError(f"unsupported dataset role: {role}")
        object.__setattr__(self, "role", role)
        interval = json_value(dict(self.interval), field="dataset interval")
        if set(interval) != {"start", "end"}:
            raise DatasetResolutionError("dataset interval needs start and end")
        start = timestamp(interval["start"], field="interval.start")
        end = timestamp(interval["end"], field="interval.end")
        if start >= end:
            raise DatasetResolutionError("dataset interval must be chronological")
        object.__setattr__(self, "interval", {"start": start, "end": end})
        availability = timestamp(self.availability_timestamp, field="availability_timestamp")
        if availability < end:
            raise DatasetResolutionError("dataset became available before its information interval")
        object.__setattr__(self, "availability_timestamp", availability)
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
            "model_artefact_id": self.model_artefact_id,
            "event_data_segment_ids": self.event_data_segment_ids,
            "role": self.role,
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

    def resolve_context(
        self,
        *,
        snapshot_ids: tuple[str, ...],
        feature_manifest_id: str,
        cost_model_id: str,
        parameter_set_id: str,
        allowed_roles: frozenset[str] | None = None,
        forbidden_roles: frozenset[str] = frozenset(),
        minimum_availability_timestamp: str | None = None,
        maximum_availability_timestamp: str | None = None,
    ) -> Mapping[str, Any]:
        allowed_roles = None if allowed_roles is None else frozenset(allowed_roles)
        unknown_roles = (allowed_roles or frozenset()) | forbidden_roles
        if not unknown_roles.issubset(DATASET_ROLES):
            raise DatasetResolutionError("dataset role filter contains an unsupported role")
        resolved = tuple(
            self.resolve(
                snapshot_id,
                expected={
                    "feature_manifest_hash": feature_manifest_id,
                    "cost_model_hash": cost_model_id,
                    "parameter_set_hash": parameter_set_id,
                },
            )
            for snapshot_id in snapshot_ids
        )
        if allowed_roles is None and any(item.role == "protected_holdout" for item in resolved):
            raise DatasetResolutionError(
                "protected_holdout datasets require an explicit protected boundary"
            )
        if allowed_roles is not None and any(item.role not in allowed_roles for item in resolved):
            raise DatasetResolutionError("dataset role is not permitted for this evaluation stage")
        if any(item.role in forbidden_roles for item in resolved):
            raise DatasetResolutionError("forbidden dataset role was resolved outside its boundary")
        if minimum_availability_timestamp is not None:
            minimum = timestamp(
                minimum_availability_timestamp, field="minimum_availability_timestamp"
            )
            if any(item.availability_timestamp <= minimum for item in resolved):
                raise DatasetResolutionError(
                    "forward observation must become available after artefact creation"
                )
        if maximum_availability_timestamp is not None:
            maximum = timestamp(
                maximum_availability_timestamp, field="maximum_availability_timestamp"
            )
            if any(item.availability_timestamp > maximum for item in resolved):
                raise DatasetResolutionError(
                    "forward observation is not available by evaluation time"
                )
        intervals = sorted((item.interval["start"], item.interval["end"]) for item in resolved)
        if any(intervals[index][0] < intervals[index - 1][1] for index in range(1, len(intervals))):
            raise DatasetResolutionError("immutable dataset information intervals overlap")
        context: dict[str, Any] = {
            "dataset_snapshot_ids": list(snapshot_ids),
            "feature_manifest_id": feature_manifest_id,
            "cost_model_id": cost_model_id,
            "parameter_set_id": parameter_set_id,
            "dataset_receipts": [dict(item.receipt) for item in resolved],
            "instrument_scope": tuple(
                dict.fromkeys(symbol for item in resolved for symbol in item.instrument_scope)
            ),
            "universe_snapshot_ids": [item.universe_snapshot_id for item in resolved],
            "model_artefact_ids": [
                item.model_artefact_id for item in resolved if item.model_artefact_id is not None
            ],
            "event_data_segment_ids": list(
                dict.fromkeys(
                    identity for item in resolved for identity in item.event_data_segment_ids
                )
            ),
        }
        for item in resolved:
            if isinstance(item.payload, Mapping):
                for key, value in item.payload.items():
                    if key not in context:
                        context[str(key)] = value
        raw_segments = context.get("event_data_segments", {})
        if context["event_data_segment_ids"]:
            if not isinstance(raw_segments, Mapping) or any(
                identity not in raw_segments or canonical_hash(raw_segments[identity]) != identity
                for identity in context["event_data_segment_ids"]
            ):
                raise DatasetResolutionError("event-data segment content hash verification failed")
        return context


class SqlCanonicalDatasetRepository:
    """Load content-addressed research snapshots from the platform database."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def resolve(self, snapshot_id: str) -> ResolvedDataset:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(dataset_snapshot.c.payload).where(dataset_snapshot.c.id == snapshot_id)
            ).scalar_one_or_none()
            universe_row = None
            if isinstance(payload, Mapping):
                universe_id = str(payload.get("universe_snapshot_id") or "")
                universe_row = (
                    connection.execute(
                        select(
                            universe_snapshot.c.content_hash,
                            universe_snapshot.c.payload,
                        ).where(universe_snapshot.c.id == universe_id)
                    )
                    .mappings()
                    .first()
                )
        if not isinstance(payload, Mapping):
            raise DatasetResolutionError(f"dataset snapshot does not exist: {snapshot_id}")
        if (
            universe_row is None
            or canonical_hash(universe_row["payload"]) != universe_row["content_hash"]
        ):
            raise DatasetResolutionError("universe snapshot content hash verification failed")
        record = dict(payload)
        data = record.get("payload", record.get("data"))
        if data is None:
            data = {
                key: value for key, value in record.items() if key not in _DATASET_METADATA_FIELDS
            }
        content_hash = str(record.get("content_hash") or canonical_hash(data))
        if snapshot_id != str(record.get("snapshot_id", snapshot_id)):
            raise DatasetResolutionError("dataset snapshot identity does not match its row")
        return ResolvedDataset(
            snapshot_id=snapshot_id,
            content_hash=content_hash,
            interval=dict(record["interval"]),
            universe_snapshot_id=str(record["universe_snapshot_id"]),
            availability_timestamp=str(record["availability_timestamp"]),
            feature_manifest_hash=str(record["feature_manifest_id"]),
            cost_model_hash=str(record["cost_model_id"]),
            parameter_set_hash=str(record["parameter_set_id"]),
            product_id=str(record["product_id"]),
            instrument_scope=tuple(record["instrument_scope"]),
            engine_version=str(record["engine_version"]),
            payload=data,
            model_artefact_id=(
                str(record["model_artefact_id"])
                if record.get("model_artefact_id") is not None
                else None
            ),
            event_data_segment_ids=tuple(record.get("event_data_segment_ids", ())),
            role=str(record.get("role", "unspecified")),
        )


_DATASET_METADATA_FIELDS = frozenset(
    {
        "snapshot_id",
        "content_hash",
        "interval",
        "universe_snapshot_id",
        "availability_timestamp",
        "feature_manifest_id",
        "cost_model_id",
        "parameter_set_id",
        "product_id",
        "instrument_scope",
        "engine_version",
        "model_artefact_id",
        "event_data_segment_ids",
        "role",
    }
)
