"""Canonical immutable dataset resolution for every research executor."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import dataset_bundle, dataset_snapshot, universe_snapshot
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.products.btc_accumulation import BTC_SPOT_INSTRUMENT_ID


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


def dataset_payload_is_non_promotable(payload: Any) -> bool:
    """Identify diagnostic or synthetic data that cannot support authority."""

    if not isinstance(payload, Mapping):
        return True
    data = payload.get("payload")
    return any(
        isinstance(source, Mapping)
        and (
            source.get("synthetic") is True
            or source.get("diagnostic") is True
            or source.get("promotable") is False
        )
        for source in (payload, data)
    )


CORE_RESEARCH_BUNDLE_ROLES = (
    "screening",
    "development",
    "robustness",
    "protected_holdout",
)

# Forward data is created after an artefact is sealed.  Keep the inclusive
# export for existing diagnostic callers, while readiness only requires the
# four pre-artefact research roles.
RESEARCH_BUNDLE_ROLES = (
    *CORE_RESEARCH_BUNDLE_ROLES,
    "forward_observation",
)


class DatasetLifecycleState(StrEnum):
    READY = "ready"
    DATA_PENDING = "data_pending"
    INVALID = "invalid"


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise DatasetResolutionError(f"{field} must be a SHA-256 identity")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise DatasetResolutionError(f"{field} must be a SHA-256 identity") from exc
    return value


def _normalise_dataset_identities(dataset: ResolvedDataset) -> None:
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
        value = non_empty(getattr(dataset, attribute), field=attribute)
        object.__setattr__(dataset, attribute, value)
    for attribute in (
        "snapshot_id",
        "content_hash",
        "feature_manifest_hash",
        "cost_model_hash",
        "parameter_set_hash",
    ):
        _identity(getattr(dataset, attribute), field=attribute)


def _normalise_event_segments(dataset: ResolvedDataset) -> None:
    if dataset.model_artefact_id is not None:
        _identity(dataset.model_artefact_id, field="model_artefact_id")
    event_ids = tuple(dataset.event_data_segment_ids)
    if len(event_ids) != len(set(event_ids)) or any(
        not value.startswith("sha256:") or len(value) != 71 for value in event_ids
    ):
        raise DatasetResolutionError(
            "event_data_segment_ids must contain unique SHA-256 identities"
        )
    object.__setattr__(dataset, "event_data_segment_ids", event_ids)


def _normalise_dataset_interval(dataset: ResolvedDataset) -> None:
    interval = json_value(dict(dataset.interval), field="dataset interval")
    if set(interval) != {"start", "end"}:
        raise DatasetResolutionError("dataset interval needs start and end")
    start = timestamp(interval["start"], field="interval.start")
    end = timestamp(interval["end"], field="interval.end")
    if start >= end:
        raise DatasetResolutionError("dataset interval must be chronological")
    object.__setattr__(dataset, "interval", {"start": start, "end": end})
    availability = timestamp(dataset.availability_timestamp, field="availability_timestamp")
    if availability < end:
        raise DatasetResolutionError("dataset became available before its information interval")
    object.__setattr__(dataset, "availability_timestamp", availability)


def _normalise_dataset_scope(dataset: ResolvedDataset) -> None:
    role = non_empty(dataset.role, field="role")
    if role not in DATASET_ROLES:
        raise DatasetResolutionError(f"unsupported dataset role: {role}")
    object.__setattr__(dataset, "role", role)
    scope = tuple(non_empty(item, field="instrument_scope") for item in dataset.instrument_scope)
    if not scope:
        raise DatasetResolutionError("instrument_scope cannot be empty")
    object.__setattr__(dataset, "instrument_scope", scope)


@dataclass(frozen=True)
class CandidateDatasetPlan:
    """Typed stage plan bound to one product and one point-in-time universe."""

    screening_snapshot_ids: tuple[str, ...]
    development_snapshot_ids: tuple[str, ...]
    robustness_snapshot_ids: tuple[str, ...]
    protected_holdout_snapshot_id: str
    product_id: str
    universe_snapshot_id: str
    feature_manifest_id: str
    cost_model_id: str
    parameter_set_id: str
    forward_snapshot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_id", non_empty(self.product_id, field="product_id"))
        for field_name in (
            "universe_snapshot_id",
            "feature_manifest_id",
            "cost_model_id",
            "parameter_set_id",
            "protected_holdout_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _identity(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "screening_snapshot_ids",
            "development_snapshot_ids",
            "robustness_snapshot_ids",
            "forward_snapshot_ids",
        ):
            values = tuple(
                _identity(value, field=f"{field_name}[]") for value in getattr(self, field_name)
            )
            if not values and field_name != "forward_snapshot_ids":
                raise DatasetResolutionError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, values)
        all_ids = self.all_snapshot_ids
        if len(all_ids) != len(set(all_ids)):
            raise DatasetResolutionError("candidate dataset stages must not overlap")

    @property
    def all_snapshot_ids(self) -> tuple[str, ...]:
        return (
            *self.screening_snapshot_ids,
            *self.development_snapshot_ids,
            *self.robustness_snapshot_ids,
            self.protected_holdout_snapshot_id,
            *self.forward_snapshot_ids,
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_payload())

    def snapshot_ids_for_stage(self, stage: str) -> tuple[str, ...]:
        values = {
            "screening": self.screening_snapshot_ids,
            "development": self.development_snapshot_ids,
            "robustness": self.robustness_snapshot_ids,
            "protected": (self.protected_holdout_snapshot_id,),
            "forward": self.forward_snapshot_ids,
        }
        try:
            return values[stage]
        except KeyError as exc:
            raise DatasetResolutionError(f"unsupported candidate dataset stage: {stage}") from exc

    def to_payload(self) -> dict[str, Any]:
        return {
            "screening_snapshot_ids": list(self.screening_snapshot_ids),
            "development_snapshot_ids": list(self.development_snapshot_ids),
            "robustness_snapshot_ids": list(self.robustness_snapshot_ids),
            "protected_holdout_snapshot_id": self.protected_holdout_snapshot_id,
            "forward_snapshot_ids": list(self.forward_snapshot_ids),
            "product_id": self.product_id,
            "universe_snapshot_id": self.universe_snapshot_id,
            "feature_manifest_id": self.feature_manifest_id,
            "cost_model_id": self.cost_model_id,
            "parameter_set_id": self.parameter_set_id,
        }

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle) -> CandidateDatasetPlan:
        if bundle.lifecycle_state is not DatasetLifecycleState.READY:
            raise DatasetResolutionError("only ready dataset bundles can create candidate plans")
        try:
            protected = bundle.stage_snapshot_ids["protected_holdout"]
        except KeyError as exc:
            raise DatasetResolutionError("dataset bundle has no protected holdout") from exc
        return cls(
            screening_snapshot_ids=(bundle.stage_snapshot_ids["screening"],),
            development_snapshot_ids=(bundle.stage_snapshot_ids["development"],),
            robustness_snapshot_ids=(bundle.stage_snapshot_ids["robustness"],),
            protected_holdout_snapshot_id=protected,
            forward_snapshot_ids=(
                (bundle.stage_snapshot_ids["forward_observation"],)
                if "forward_observation" in bundle.stage_snapshot_ids
                else ()
            ),
            product_id=bundle.product_id,
            universe_snapshot_id=bundle.universe_snapshot_id,
            feature_manifest_id=bundle.feature_manifest_id,
            cost_model_id=bundle.cost_model_id,
            parameter_set_id=bundle.parameter_set_id,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CandidateDatasetPlan:
        if not isinstance(payload, Mapping):
            raise DatasetResolutionError("candidate dataset plan must be an object")
        return cls(
            screening_snapshot_ids=tuple(payload["screening_snapshot_ids"]),
            development_snapshot_ids=tuple(payload["development_snapshot_ids"]),
            robustness_snapshot_ids=tuple(payload["robustness_snapshot_ids"]),
            protected_holdout_snapshot_id=str(payload["protected_holdout_snapshot_id"]),
            forward_snapshot_ids=tuple(payload.get("forward_snapshot_ids", ())),
            product_id=str(payload["product_id"]),
            universe_snapshot_id=str(payload["universe_snapshot_id"]),
            feature_manifest_id=str(payload["feature_manifest_id"]),
            cost_model_id=str(payload["cost_model_id"]),
            parameter_set_id=str(payload["parameter_set_id"]),
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Immutable stage-to-snapshot mapping used by one research candidate."""

    bundle_id: str
    product_id: str
    stage_snapshot_ids: Mapping[str, str]
    created_at: str
    universe_snapshot_id: str
    feature_manifest_id: str
    cost_model_id: str
    parameter_set_id: str
    engine_version: str
    instrument_scope: tuple[str, ...]
    lifecycle_state: DatasetLifecycleState = DatasetLifecycleState.READY
    source_partition_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _identity(self.bundle_id, field="bundle_id"))
        object.__setattr__(self, "product_id", non_empty(self.product_id, field="product_id"))
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        object.__setattr__(
            self,
            "universe_snapshot_id",
            _identity(self.universe_snapshot_id, field="universe_snapshot_id"),
        )
        for field_name in ("feature_manifest_id", "cost_model_id", "parameter_set_id"):
            object.__setattr__(
                self,
                field_name,
                _identity(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self, "engine_version", non_empty(self.engine_version, field="engine_version")
        )
        scope = tuple(non_empty(item, field="instrument_scope") for item in self.instrument_scope)
        if not scope:
            raise DatasetResolutionError("dataset bundle instrument_scope cannot be empty")
        object.__setattr__(self, "instrument_scope", scope)
        state = DatasetLifecycleState(self.lifecycle_state)
        if not isinstance(self.stage_snapshot_ids, Mapping):
            raise DatasetResolutionError("dataset bundle stage_snapshot_ids must be an object")
        stages = {
            str(key): _identity(value, field=f"stage_snapshot_ids[{key}]")
            for key, value in self.stage_snapshot_ids.items()
        }
        if not set(stages).issubset(set(RESEARCH_BUNDLE_ROLES)):
            raise DatasetResolutionError("dataset bundle contains an unsupported research role")
        object.__setattr__(self, "stage_snapshot_ids", stages)
        if state is DatasetLifecycleState.READY and not set(CORE_RESEARCH_BUNDLE_ROLES).issubset(
            stages
        ):
            raise DatasetResolutionError("ready dataset bundles must be complete")
        object.__setattr__(self, "lifecycle_state", state)
        partitions = tuple(
            _identity(item, field="source_partition_hashes")
            for item in self.source_partition_hashes
        )
        if len(partitions) != len(set(partitions)):
            raise DatasetResolutionError("source_partition_hashes must be unique")
        object.__setattr__(self, "source_partition_hashes", partitions)

    @property
    def content(self) -> dict[str, Any]:
        return {
            "schema": "platform.dataset_bundle/v1",
            "product_id": self.product_id,
            "stage_snapshot_ids": dict(self.stage_snapshot_ids),
            "created_at": self.created_at,
            "universe_snapshot_id": self.universe_snapshot_id,
            "feature_manifest_id": self.feature_manifest_id,
            "cost_model_id": self.cost_model_id,
            "parameter_set_id": self.parameter_set_id,
            "engine_version": self.engine_version,
            "instrument_scope": list(self.instrument_scope),
            "lifecycle_state": self.lifecycle_state.value,
            "source_partition_hashes": list(self.source_partition_hashes),
        }

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.content)

    def to_payload(self) -> dict[str, Any]:
        return {"bundle_id": self.bundle_id, "content_hash": self.content_hash, **self.content}


def _bar_timestamp(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise DatasetResolutionError(f"{field} must be a timestamp")
    if isinstance(value, int | float):
        try:
            return (
                dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC)
                .replace(microsecond=0)
                .isoformat()
            )
        except (OverflowError, OSError, ValueError) as exc:
            raise DatasetResolutionError(f"{field} is invalid") from exc
    return timestamp(str(value), field=field)


class CanonicalResearchDatasetBuilder:
    """Build deterministic, non-overlapping role snapshots and one bundle."""

    REQUIRED_ROLES = frozenset(CORE_RESEARCH_BUNDLE_ROLES)

    def __init__(self, engine: Engine, *, default_engine_version: str = "research-engine/v1"):
        self.engine = engine
        self.default_engine_version = non_empty(default_engine_version, field="engine_version")

    def build(
        self,
        product_id: str,
        *,
        intervals: Mapping[str, Mapping[str, str]],
        payload_by_role: Mapping[str, Any],
        universe_snapshot_id: str,
        feature_manifest_id: str,
        cost_model_id: str,
        parameter_set_id: str,
        instrument_scope: tuple[str, ...] | list[str],
        availability_timestamp: str | Mapping[str, str],
        created_at: str,
        engine_version: str | None = None,
        source_partition_hashes: tuple[str, ...] = (),
        lifecycle_state: DatasetLifecycleState = DatasetLifecycleState.READY,
    ) -> DatasetBundle:
        product_id = non_empty(product_id, field="product_id")
        state = DatasetLifecycleState(lifecycle_state)
        roles = set(payload_by_role)
        unsupported = sorted(roles - set(RESEARCH_BUNDLE_ROLES))
        if unsupported:
            raise DatasetResolutionError(
                "dataset bundle contains unsupported roles: " + ", ".join(unsupported)
            )
        missing = sorted(self.REQUIRED_ROLES - roles)
        if state is DatasetLifecycleState.READY and missing:
            raise DatasetResolutionError(
                "dataset bundle cannot be built; missing roles: " + ", ".join(missing)
            )
        if set(intervals) != roles:
            raise DatasetResolutionError(
                "dataset bundle intervals must match the supplied research roles"
            )
        normalised_intervals = {
            role: self._interval(intervals[role], field=f"intervals.{role}")
            for role in sorted(roles)
        }
        ordered = sorted(normalised_intervals.items(), key=lambda item: item[1]["start"])
        for (_, left), (_, right) in zip(ordered, ordered[1:], strict=False):
            if left["end"] > right["start"]:
                raise DatasetResolutionError("dataset bundle information intervals overlap")
        created = timestamp(created_at, field="created_at")
        availability = self._availability(availability_timestamp, created, roles=roles)
        universe_id = _identity(universe_snapshot_id, field="universe_snapshot_id")
        manifest_id = _identity(feature_manifest_id, field="feature_manifest_id")
        costs_id = _identity(cost_model_id, field="cost_model_id")
        parameters_id = _identity(parameter_set_id, field="parameter_set_id")
        engine = non_empty(engine_version or self.default_engine_version, field="engine_version")
        scope = tuple(non_empty(item, field="instrument_scope") for item in instrument_scope)
        if not scope:
            raise DatasetResolutionError("instrument_scope cannot be empty")
        _validate_product_scope(product_id, scope)
        snapshots: dict[str, str] = {}
        snapshot_payloads: dict[str, dict[str, Any]] = {}
        for role in sorted(roles):
            data = json_value(payload_by_role[role], field=f"{role} dataset payload")
            if availability[role] < normalised_intervals[role]["end"]:
                raise DatasetResolutionError(
                    f"{role} dataset became available before its information interval ended"
                )
            content_hash = canonical_hash(data)
            metadata = {
                "product_id": product_id,
                "role": role,
                "content_hash": content_hash,
                "interval": normalised_intervals[role],
                "universe_snapshot_id": universe_id,
                "availability_timestamp": availability[role],
                "feature_manifest_id": manifest_id,
                "cost_model_id": costs_id,
                "parameter_set_id": parameters_id,
                "engine_version": engine,
                "instrument_scope": list(scope),
            }
            snapshot_id = canonical_hash(metadata)
            payload = {"snapshot_id": snapshot_id, **metadata, "payload": data}
            snapshots[role] = snapshot_id
            snapshot_payloads[role] = payload
        bundle_payload = {
            "schema": "platform.dataset_bundle/v1",
            "product_id": product_id,
            "stage_snapshot_ids": snapshots,
            "created_at": created,
            "universe_snapshot_id": universe_id,
            "feature_manifest_id": manifest_id,
            "cost_model_id": costs_id,
            "parameter_set_id": parameters_id,
            "engine_version": engine,
            "instrument_scope": list(scope),
            "lifecycle_state": state.value,
            "source_partition_hashes": list(source_partition_hashes),
        }
        bundle_id = canonical_hash(bundle_payload)
        bundle = DatasetBundle(
            bundle_id=bundle_id,
            product_id=product_id,
            stage_snapshot_ids=snapshots,
            created_at=created,
            universe_snapshot_id=universe_id,
            feature_manifest_id=manifest_id,
            cost_model_id=costs_id,
            parameter_set_id=parameters_id,
            engine_version=engine,
            instrument_scope=scope,
            lifecycle_state=state,
            source_partition_hashes=tuple(source_partition_hashes),
        )
        with self.engine.begin() as connection:
            for role in sorted(roles):
                self._insert_immutable(
                    connection,
                    dataset_snapshot,
                    {
                        "id": snapshots[role],
                        "created_at": created,
                        "payload": snapshot_payloads[role],
                    },
                )
            self._insert_immutable(
                connection,
                dataset_bundle,
                {
                    "id": bundle.bundle_id,
                    "product_id": bundle.product_id,
                    "created_at": bundle.created_at,
                    "content_hash": bundle.content_hash,
                    "payload": bundle.to_payload(),
                },
            )
        return bundle

    def build_bundle(self, product_id: str, **kwargs: Any) -> DatasetBundle:
        return self.build(product_id, **kwargs)

    def build_from_bars(
        self,
        product_id: str,
        *,
        bars: Any,
        intervals: Mapping[str, Mapping[str, str]],
        universe_snapshot_id: str,
        feature_manifest_id: str,
        cost_model_id: str,
        parameter_set_id: str,
        instrument_scope: tuple[str, ...] | list[str],
        created_at: str,
        engine_version: str | None = None,
        source_partition_hashes: tuple[str, ...] = (),
    ) -> DatasetBundle:
        """Build stage snapshots from immutable, availability-stamped bar rows.

        Rows unavailable by ``created_at`` are excluded. A missing role raises
        an explicit data-pending error so callers cannot advance a candidate
        with a partial bundle.
        """

        materialised = self._materialise_bars(bars)
        created = timestamp(created_at, field="created_at")
        scope = {str(item) for item in instrument_scope}
        _validate_product_scope(product_id, tuple(scope))
        payload_by_role: dict[str, Any] = {}
        availability: dict[str, str] = {}
        for role, raw_interval in intervals.items():
            interval = self._interval(raw_interval, field=f"intervals.{role}")
            selected = self._select_bars(
                materialised,
                role=role,
                interval=interval,
                scope=scope,
                created=created,
            )
            if not selected:
                raise DatasetResolutionError(
                    f"dataset data_pending: no available bars for role {role}"
                )
            selected_scope = {
                str(row.get("instrument_id") or row.get("symbol") or "") for row in selected
            }
            missing = sorted(scope - selected_scope)
            if missing:
                raise DatasetResolutionError(
                    f"dataset data_pending: missing instruments for role {role}: {', '.join(missing)}"
                )
            payload_by_role[role] = self._bar_payload(selected)
            availability[role] = created
        return self.build(
            product_id,
            intervals=intervals,
            payload_by_role=payload_by_role,
            universe_snapshot_id=universe_snapshot_id,
            feature_manifest_id=feature_manifest_id,
            cost_model_id=cost_model_id,
            parameter_set_id=parameter_set_id,
            instrument_scope=instrument_scope,
            availability_timestamp=availability,
            created_at=created,
            engine_version=engine_version,
            source_partition_hashes=source_partition_hashes,
        )

    @staticmethod
    def _bar_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
        market_frame = [dict(row) for row in rows]
        aligned_returns: list[float] = []
        funding_rates: list[float] = []
        funding_period_rates: list[float] = []
        funding_timestamps: list[str] = []
        previous_by_instrument: dict[str, float] = {}
        history_by_instrument: dict[str, list[float]] = {}
        returns_by_instrument: dict[str, list[float]] = {}
        feature_rows: list[dict[str, Any]] = []
        for row in market_frame:
            close_value = _numeric_close(row.get("close"))
            instrument = str(row.get("instrument_id") or row.get("symbol") or "")
            history = history_by_instrument.setdefault(instrument, [])
            previous = previous_by_instrument.get(instrument)
            row_return = close_value / previous - 1.0 if previous and close_value else 0.0
            aligned_returns.append(row_return)
            if previous is not None:
                returns_by_instrument.setdefault(instrument, []).append(row_return)
            funding_rate = _signed_numeric(row.get("funding_rate", row.get("funding")), default=0.0)
            funding_rates.append(funding_rate)
            funding_period_rates.append(funding_rate if row.get("funding_event") is True else 0.0)
            if row.get("funding_event") is True:
                funding_timestamps.append(str(row.get("close_timestamp", row.get("timestamp"))))
            feature_rows.append(_bar_feature_row(row, close_value, row_return, history))
            if close_value > 0.0:
                history.append(close_value)
                previous_by_instrument[instrument] = close_value
        return {
            "bars": market_frame,
            "market_frame": market_frame,
            "returns": aligned_returns[1:],
            "funding_rates": funding_rates,
            "funding_period_rates": funding_period_rates,
            "funding_timestamps": funding_timestamps,
            "symbol_returns": returns_by_instrument,
            "feature_rows": feature_rows,
            "independent_units": len(market_frame),
            "data_quality": {
                "rows": len(market_frame),
                "instruments": sorted(
                    {str(row.get("instrument_id") or row.get("symbol") or "") for row in rows}
                ),
                "missing_fields": [],
                "duplicate_timestamps": 0,
            },
        }

    @staticmethod
    def _materialise_bars(bars: Any) -> tuple[dict[str, Any], ...]:
        if hasattr(bars, "to_pylist"):
            bars = bars.to_pylist()
        if isinstance(bars, Mapping) or isinstance(bars, str):
            raise DatasetResolutionError("bars must be an iterable of row objects")
        rows = tuple(bars)
        if any(not isinstance(row, Mapping) for row in rows):
            raise DatasetResolutionError("bars must contain only row objects")
        materialised = tuple(dict(row) for row in rows)
        if not materialised:
            raise DatasetResolutionError("dataset data_pending: no immutable bars are available")
        return materialised

    @classmethod
    def _select_bars(
        cls,
        rows: tuple[dict[str, Any], ...],
        *,
        role: str,
        interval: Mapping[str, str],
        scope: set[str],
        created: str,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in rows:
            if cls._bar_is_available(
                row, role=role, interval=interval, scope=scope, created=created
            ):
                selected.append(row)
        selected.sort(key=cls._bar_sort_key)
        return selected

    @staticmethod
    def _bar_is_available(
        row: Mapping[str, Any],
        *,
        role: str,
        interval: Mapping[str, str],
        scope: set[str],
        created: str,
    ) -> bool:
        instrument_id = str(row.get("instrument_id") or row.get("symbol") or "")
        if instrument_id not in scope:
            return False
        observed = row.get("close_timestamp", row.get("timestamp"))
        available = row.get("availability_time", row.get("available_at", observed))
        if observed is None or available is None:
            return False
        observed_at = _bar_timestamp(observed, field=f"{role}.bar_timestamp")
        available_at = _bar_timestamp(available, field=f"{role}.availability_time")
        return interval["start"] <= observed_at < interval["end"] and available_at <= created

    @staticmethod
    def _bar_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
        observed = row.get("close_timestamp", row.get("timestamp"))
        return (
            _bar_timestamp(observed, field="bar_timestamp") if observed is not None else "",
            str(row.get("instrument_id", row.get("symbol", ""))),
        )

    @staticmethod
    def _interval(value: Mapping[str, str], *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
            raise DatasetResolutionError(f"{field} must contain start and end")
        start = timestamp(str(value["start"]), field=f"{field}.start")
        end = timestamp(str(value["end"]), field=f"{field}.end")
        if start >= end:
            raise DatasetResolutionError(f"{field} must be chronological")
        return {"start": start, "end": end}

    @staticmethod
    def _availability(
        value: str | Mapping[str, str], created: str, *, roles: set[str]
    ) -> dict[str, str]:
        if isinstance(value, Mapping):
            if set(value) != roles:
                raise DatasetResolutionError(
                    "availability timestamps must match the supplied research roles"
                )
            result = {
                role: timestamp(str(value[role]), field=f"availability.{role}")
                for role in sorted(roles)
            }
        else:
            shared = timestamp(value, field="availability_timestamp")
            result = {role: shared for role in sorted(roles)}
        if any(item < created for item in result.values()):
            raise DatasetResolutionError("dataset availability cannot precede bundle creation")
        return result

    @staticmethod
    def _insert_immutable(connection, table, values: Mapping[str, Any]) -> None:
        existing = (
            connection.execute(select(table).where(table.c.id == values["id"])).mappings().first()
        )
        if existing is None:
            connection.execute(insert(table).values(**dict(values)))
            return
        if any(existing[key] != value for key, value in values.items()):
            raise DatasetResolutionError(f"immutable dataset identity collision: {values['id']}")


def _validate_product_scope(product_id: str, scope: tuple[str, ...]) -> None:
    if product_id == "btc_accumulation" and scope != (BTC_SPOT_INSTRUMENT_ID,):
        raise DatasetResolutionError("BTC accumulation datasets require BTCUSDT spot only")


def _numeric_close(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _bar_feature_row(
    row: Mapping[str, Any], close: float, row_return: float, history: list[float]
) -> dict[str, Any]:
    recent = history[-20:]
    mean = sum(recent) / len(recent) if recent else close
    variance = sum((value - mean) ** 2 for value in recent) / len(recent) if recent else 0.0
    deviation = (close - mean) / math.sqrt(variance) if variance > 0.0 else 0.0
    prior_high = max(recent) if recent else close
    range_fraction = _range_fraction(row, close)
    return {
        "timestamp": row.get("close_timestamp", row.get("timestamp")),
        "bar_return": row_return,
        "trend": row_return,
        "breakout": close / prior_high - 1.0 if prior_high > 0.0 else 0.0,
        "normalised_price_deviation": deviation,
        "oscillator": deviation,
        "range_state": range_fraction,
        "realised_volatility": math.sqrt(variance) / mean if mean > 0.0 else 0.0,
        "relative_return": row_return,
        "funding_rank": _signed_numeric(
            row.get("funding", row.get("funding_rate", 0.0)), default=0.0
        ),
    }


def _range_fraction(row: Mapping[str, Any], close: float) -> float:
    high = _numeric_close(row.get("high"))
    low = _numeric_close(row.get("low"))
    return (high - low) / close if close > 0.0 and high >= low else 0.0


def _signed_numeric(value: Any, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


class SqlDatasetBundleRepository:
    """Read and verify immutable research bundles."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def latest_ready(self, product_id: str, *, at: str) -> DatasetBundle | None:
        at = timestamp(at, field="at")
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(dataset_bundle.c.id)
                .where(
                    dataset_bundle.c.product_id == product_id,
                    dataset_bundle.c.created_at <= at,
                )
                .order_by(dataset_bundle.c.created_at.desc(), dataset_bundle.c.id.desc())
            ).scalars()
            for bundle_id in rows:
                try:
                    bundle = self.get(str(bundle_id))
                except (DatasetResolutionError, KeyError):
                    continue
                if (
                    bundle.lifecycle_state is DatasetLifecycleState.READY
                    and self._bundle_is_promotable(bundle)
                ):
                    return bundle
        return None

    def _bundle_is_promotable(self, bundle: DatasetBundle) -> bool:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(dataset_snapshot.c.payload).where(
                    dataset_snapshot.c.id.in_(tuple(bundle.stage_snapshot_ids.values()))
                )
            ).scalars()
            payloads = tuple(rows)
        return (
            bool(payloads)
            and len(payloads) == len(bundle.stage_snapshot_ids)
            and all(not dataset_payload_is_non_promotable(payload) for payload in payloads)
        )

    def get(self, bundle_id: str) -> DatasetBundle:
        bundle_id = _identity(bundle_id, field="bundle_id")
        with self.engine.connect() as connection:
            payload = (
                connection.execute(
                    select(dataset_bundle.c.payload, dataset_bundle.c.content_hash).where(
                        dataset_bundle.c.id == bundle_id
                    )
                )
                .mappings()
                .first()
            )
            if payload is None or not isinstance(payload["payload"], Mapping):
                raise KeyError(f"dataset bundle does not exist: {bundle_id}")
            record = dict(payload["payload"])
            stage_ids = record.get("stage_snapshot_ids")
            if not isinstance(stage_ids, Mapping):
                raise DatasetResolutionError("dataset bundle stage mapping is invalid")
            for role, snapshot_id in stage_ids.items():
                snapshot_payload = connection.execute(
                    select(dataset_snapshot.c.payload).where(dataset_snapshot.c.id == snapshot_id)
                ).scalar_one_or_none()
                if not isinstance(snapshot_payload, Mapping):
                    raise DatasetResolutionError(
                        f"dataset bundle stage snapshot is missing: {role}:{snapshot_id}"
                    )
                if (
                    snapshot_payload.get("snapshot_id") != snapshot_id
                    or snapshot_payload.get("role") != role
                    or snapshot_payload.get("product_id") != record.get("product_id")
                    or canonical_hash(snapshot_payload.get("payload"))
                    != snapshot_payload.get("content_hash")
                ):
                    raise DatasetResolutionError(
                        f"dataset bundle stage snapshot is not canonical: {role}:{snapshot_id}"
                    )
        bundle = DatasetBundle(
            bundle_id=str(record.get("bundle_id") or bundle_id),
            product_id=str(record["product_id"]),
            stage_snapshot_ids=dict(record["stage_snapshot_ids"]),
            created_at=str(record["created_at"]),
            universe_snapshot_id=str(record["universe_snapshot_id"]),
            feature_manifest_id=str(record["feature_manifest_id"]),
            cost_model_id=str(record["cost_model_id"]),
            parameter_set_id=str(record["parameter_set_id"]),
            engine_version=str(record["engine_version"]),
            instrument_scope=tuple(record["instrument_scope"]),
            lifecycle_state=DatasetLifecycleState(str(record.get("lifecycle_state", "ready"))),
            source_partition_hashes=tuple(record.get("source_partition_hashes", ())),
        )
        if bundle.bundle_id != bundle_id or bundle.content_hash != str(payload["content_hash"]):
            raise DatasetResolutionError("dataset bundle content hash is invalid")
        return bundle


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
        _normalise_dataset_identities(self)
        _normalise_event_segments(self)
        _normalise_dataset_interval(self)
        _normalise_dataset_scope(self)

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

    @staticmethod
    def _validate_context_filters(
        resolved: tuple[ResolvedDataset, ...],
        *,
        allowed_roles: frozenset[str] | None,
        forbidden_roles: frozenset[str],
        minimum_availability_timestamp: str | None,
        maximum_availability_timestamp: str | None,
    ) -> None:
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

    @staticmethod
    def _build_context(
        snapshot_ids: tuple[str, ...],
        resolved: tuple[ResolvedDataset, ...],
        *,
        feature_manifest_id: str,
        cost_model_id: str,
        parameter_set_id: str,
    ) -> dict[str, Any]:
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
        return context

    @staticmethod
    def _validate_event_segments(context: Mapping[str, Any]) -> None:
        identities = context["event_data_segment_ids"]
        raw_segments = context.get("event_data_segments", {})
        if identities and (
            not isinstance(raw_segments, Mapping)
            or any(
                identity not in raw_segments or canonical_hash(raw_segments[identity]) != identity
                for identity in identities
            )
        ):
            raise DatasetResolutionError("event-data segment content hash verification failed")

    def resolve_context(
        self,
        *,
        snapshot_ids: tuple[str, ...],
        product_id: str | None = None,
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
                    **({"product_id": product_id} if product_id is not None else {}),
                    "feature_manifest_hash": feature_manifest_id,
                    "cost_model_hash": cost_model_id,
                    "parameter_set_hash": parameter_set_id,
                },
            )
            for snapshot_id in snapshot_ids
        )
        self._validate_context_filters(
            resolved,
            allowed_roles=allowed_roles,
            forbidden_roles=forbidden_roles,
            minimum_availability_timestamp=minimum_availability_timestamp,
            maximum_availability_timestamp=maximum_availability_timestamp,
        )
        context = self._build_context(
            snapshot_ids,
            resolved,
            feature_manifest_id=feature_manifest_id,
            cost_model_id=cost_model_id,
            parameter_set_id=parameter_set_id,
        )
        self._validate_event_segments(context)
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
