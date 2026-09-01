"""Shared live and historical deterministic feature worker."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.data.feature_graph import AvailableValue, FeatureGraphRegistry, default_feature_engine
from src.data.feature_store import DeterministicFeatureCalculator, FeatureValue, SqlFeatureStore
from src.data.parquet_store import PartitionedFeatureStore
from src.domain._codec import canonical_hash
from src.risk.engine import SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash
from src.services.scheduler import DatabaseJobQueue


def core_bar_features(inputs: Mapping[str, float]) -> dict[str, float]:
    required = {name: float(inputs[name]) for name in ("open", "high", "low", "close", "volume")}
    if required["open"] <= 0 or required["close"] <= 0 or required["volume"] < 0:
        raise ValueError("bar prices must be positive and volume non-negative")
    return {
        "bar_return": required["close"] / required["open"] - 1.0,
        "log_volume": math.log1p(required["volume"]),
        "range_fraction": (required["high"] - required["low"]) / required["close"],
    }


class DatabaseFeatureWorker:
    def __init__(
        self,
        *,
        queue: DatabaseJobQueue,
        worker_id: str,
        store: SqlFeatureStore,
        job_names: tuple[str, ...],
        parquet_root: Path | None = None,
        active_assignments: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
        snapshot_store: SqlRiskSnapshotStore | None = None,
        feature_graph_for_assignment: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        lease_seconds: int = 60,
    ) -> None:
        if not job_names:
            raise ValueError("feature worker requires at least one job name")
        self.queue = queue
        self.worker_id = worker_id
        self.store = store
        self.parquet_store = (
            PartitionedFeatureStore(parquet_root) if parquet_root is not None else None
        )
        self.active_assignments = active_assignments or (lambda _instrument_id: ())
        self.snapshot_store = snapshot_store
        self.feature_graph_for_assignment = feature_graph_for_assignment
        self.graph_registry = FeatureGraphRegistry.default()
        self.graph_engine = default_feature_engine()
        self.job_names = job_names
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        claimed = self.queue.claim(
            worker_id=self.worker_id,
            now=now,
            lease_seconds=self.lease_seconds,
            names=self.job_names,
        )
        if claimed is None:
            return {"reason_code": "feature_queue_empty"}
        try:
            payload = claimed.payload
            raw_inputs, scalar_inputs = self._input_values(payload, job_name=claimed.name)
            values = self._calculate_core_values(payload, scalar_inputs)
            assignments = tuple(self.active_assignments(str(payload["instrument_id"])))
            required_by_assignment, graph_values, graph_output_features = self._graph_features(
                payload=payload,
                raw_inputs=raw_inputs,
                assignments=assignments,
            )
            combined = {value.feature_name: value.value for value in values}
            combined.update(graph_values)
            values = tuple(
                FeatureValue(
                    feature_set_version=str(payload["feature_set_version"]),
                    feature_name=name,
                    instrument_id=str(payload["instrument_id"]),
                    source_event_time=str(payload["source_event_time"]),
                    source_close_time=str(payload["source_close_time"]),
                    availability_time=str(payload["availability_time"]),
                    value=value,
                )
                for name, value in sorted(combined.items())
            )
            identities = self.store.save(values)
            identity_by_name = dict(
                zip((item.feature_name for item in values), identities, strict=True)
            )
            parquet_path = self._persist_parquet(values, payload)
            evaluation_jobs = self._enqueue_evaluation_jobs(
                payload=payload,
                raw_inputs=raw_inputs,
                scalar_inputs=scalar_inputs,
                assignments=assignments,
                required_by_assignment=required_by_assignment,
                graph_output_features=graph_output_features,
                identity_by_name=identity_by_name,
                identities=identities,
            )
        except Exception as exc:
            self.queue.fail(
                claimed,
                completed_at=now,
                error=f"{type(exc).__name__}: {exc}",
                retry_at=_retry_at(now, self.lease_seconds),
            )
            return {
                "reason_code": "feature_calculation_failed",
                "job_id": claimed.job_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.queue.complete(claimed, completed_at=now)
        return {
            "reason_code": "features_persisted",
            "job_id": claimed.job_id,
            "features": len(identities),
            "feature_ids": list(identities),
            "parquet_path": str(parquet_path) if parquet_path is not None else None,
            "strategy_evaluation_jobs": evaluation_jobs,
        }

    def _input_values(
        self, payload: Mapping[str, Any], *, job_name: str
    ) -> tuple[Mapping[str, Any], dict[str, float]]:
        if job_name == "live_feature_calculation" and not isinstance(
            payload.get("input_references"), Mapping
        ):
            raise ValueError("live feature jobs require immutable input references")
        raw_inputs = payload.get("inputs")
        if payload.get("input_references") is not None:
            raw_inputs = self._resolve_input_references(payload)
        if not isinstance(raw_inputs, Mapping):
            raise ValueError("feature job requires immutable input references or an input object")
        scalar_inputs = {
            str(key): float(value)
            for key, value in raw_inputs.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        return raw_inputs, scalar_inputs

    @staticmethod
    def _calculate_core_values(
        payload: Mapping[str, Any], scalar_inputs: Mapping[str, float]
    ) -> tuple[FeatureValue, ...]:
        calculator = DeterministicFeatureCalculator(
            version=str(payload["feature_set_version"]),
            function=core_bar_features,
        )
        return calculator.calculate(
            instrument_id=str(payload["instrument_id"]),
            source_event_time=str(payload["source_event_time"]),
            source_close_time=str(payload["source_close_time"]),
            availability_time=str(payload["availability_time"]),
            inputs=scalar_inputs,
        )

    def _graph_features(
        self,
        *,
        payload: Mapping[str, Any],
        raw_inputs: Mapping[str, Any],
        assignments: tuple[Mapping[str, Any], ...],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, float], dict[str, tuple[str, ...]]]:
        required_by_assignment = self._assignment_feature_requirements(assignments)
        if self.feature_graph_for_assignment is None:
            return required_by_assignment, {}, {}
        union = tuple(
            dict.fromkeys(name for names in required_by_assignment.values() for name in names)
        )
        if not union:
            return required_by_assignment, {}, {}
        graph = self.graph_registry.graph(union, version=str(payload["feature_set_version"]))
        available = {
            str(name): AvailableValue(
                value,
                information_time=str(payload["source_close_time"]),
                availability_time=str(payload["availability_time"]),
            )
            for name, value in raw_inputs.items()
        }
        calculated = self.graph_engine.evaluate(
            graph,
            information_timestamp=str(payload["availability_time"]),
            inputs=available,
        )
        graph_values, output_features = self._normalise_graph_outputs(calculated)
        return required_by_assignment, graph_values, output_features

    def _assignment_feature_requirements(
        self, assignments: tuple[Mapping[str, Any], ...]
    ) -> dict[str, tuple[str, ...]]:
        if self.feature_graph_for_assignment is None:
            return {}
        requirements: dict[str, tuple[str, ...]] = {}
        for assignment in assignments:
            declaration = self.feature_graph_for_assignment(assignment)
            raw_required = declaration.get("required_nodes", declaration.get("nodes", ()))
            requirements[str(assignment["id"])] = tuple(
                str(item["name"] if isinstance(item, Mapping) else item) for item in raw_required
            )
        return requirements

    def _normalise_graph_outputs(
        self, calculated: Mapping[str, Any]
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
        values: dict[str, float] = {}
        output_features: dict[str, tuple[str, ...]] = {}
        for name, value in calculated.items():
            expanded = self._graph_output(name, value)
            values.update(expanded)
            if not isinstance(value, int | float) or isinstance(value, bool):
                output_features[name] = tuple(sorted(expanded))
        return values, output_features

    @staticmethod
    def _graph_output(name: str, value: Any) -> dict[str, float]:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return {name: float(value)}
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"feature node {name} returned no scalar values")
        expanded: dict[str, float] = {}
        for feature_name, feature_value in value.items():
            if (
                isinstance(feature_value, bool)
                or not isinstance(feature_value, int | float)
                or not math.isfinite(float(feature_value))
            ):
                raise ValueError(f"feature node {name} returned an invalid component")
            expanded[str(feature_name)] = float(feature_value)
        if not expanded:
            raise ValueError(f"feature node {name} returned no scalar components")
        return expanded

    def _persist_parquet(
        self, values: tuple[FeatureValue, ...], payload: Mapping[str, Any]
    ) -> Path | None:
        if self.parquet_store is None:
            return None
        return self.parquet_store.put(
            values,
            venue=str(payload["venue"]),
            market=str(payload["market"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
        )

    def _enqueue_evaluation_jobs(
        self,
        *,
        payload: Mapping[str, Any],
        raw_inputs: Mapping[str, Any],
        scalar_inputs: Mapping[str, float],
        assignments: tuple[Mapping[str, Any], ...],
        required_by_assignment: Mapping[str, tuple[str, ...]],
        graph_output_features: Mapping[str, tuple[str, ...]],
        identity_by_name: Mapping[str, str],
        identities: tuple[str, ...],
    ) -> list[str]:
        jobs: list[str] = []
        for assignment in assignments:
            evaluation_payload = self._evaluation_payload(
                payload=payload,
                raw_inputs=raw_inputs,
                scalar_inputs=scalar_inputs,
                assignment=assignment,
                required=required_by_assignment.get(str(assignment["id"]), ()),
                graph_output_features=graph_output_features,
                identity_by_name=identity_by_name,
                identities=identities,
            )
            evaluation_payload["content_hash"] = build_content_hash(evaluation_payload)
            job_id = (
                f"strategy-evaluation:{canonical_hash(evaluation_payload).removeprefix('sha256:')}"
            )
            self.queue.enqueue_if_absent(
                job_id=job_id,
                name="strategy_evaluation",
                payload=evaluation_payload,
                available_at=str(payload["availability_time"]),
                priority=12,
                producer_identity=self.worker_id,
            )
            jobs.append(job_id)
        return jobs

    def _evaluation_payload(
        self,
        *,
        payload: Mapping[str, Any],
        raw_inputs: Mapping[str, Any],
        scalar_inputs: Mapping[str, float],
        assignment: Mapping[str, Any],
        required: tuple[str, ...],
        graph_output_features: Mapping[str, tuple[str, ...]],
        identity_by_name: Mapping[str, str],
        identities: tuple[str, ...],
    ) -> dict[str, Any]:
        snapshot_id = self._market_snapshot_id(payload, raw_inputs, scalar_inputs, assignment)
        required_names = self._required_feature_names(
            required, graph_output_features=graph_output_features, identity_by_name=identity_by_name
        )
        feature_ids = (
            [identity_by_name[name] for name in required_names] if required else list(identities)
        )
        result = {
            "event_id": str(payload["source_market_event_id"]),
            "product_id": str(assignment["product_id"]),
            "instrument_id": str(payload["instrument_id"]),
            "assignment_id": str(assignment["id"]),
            "feature_ids": feature_ids,
            "feature_set_version": str(payload["feature_set_version"]),
            "evaluated_at": str(payload["availability_time"]),
            "horizon_seconds": int(payload.get("horizon_seconds", 60)),
            "producer_identity": self.worker_id,
        }
        if snapshot_id is not None:
            result["market_data_snapshot_id"] = snapshot_id
        input_references = payload.get("input_references")
        if isinstance(input_references, Mapping):
            reference_hashes = {
                str(name): str(reference["content_hash"])
                for name, reference in input_references.items()
                if isinstance(reference, Mapping) and reference.get("content_hash")
            }
            if reference_hashes:
                result["input_reference_id"] = canonical_hash(reference_hashes)
        return result

    @staticmethod
    def _required_feature_names(
        required: tuple[str, ...],
        *,
        graph_output_features: Mapping[str, tuple[str, ...]],
        identity_by_name: Mapping[str, str],
    ) -> list[str]:
        names: list[str] = []
        for name in required:
            if name in identity_by_name:
                names.append(name)
            elif name in graph_output_features:
                names.extend(graph_output_features[name])
            else:
                raise ValueError("artefact feature graph did not produce every required node")
        return names

    def _market_snapshot_id(
        self,
        payload: Mapping[str, Any],
        raw_inputs: Mapping[str, Any],
        scalar_inputs: Mapping[str, float],
        assignment: Mapping[str, Any],
    ) -> str | None:
        if self.snapshot_store is None:
            return None
        market_frame = raw_inputs.get("market_frame")
        if not isinstance(market_frame, list | tuple):
            market_frame = [
                {
                    name: float(scalar_inputs[name])
                    for name in ("open", "high", "low", "close", "volume")
                    if name in scalar_inputs
                }
            ]
        return self.snapshot_store.save(
            {
                "kind": "market_data_input",
                "product_id": str(assignment["product_id"]),
                "event_id": str(payload["source_market_event_id"]),
                "instrument_id": str(payload["instrument_id"]),
                "source_event_time": str(payload["source_event_time"]),
                "availability_time": str(payload["availability_time"]),
                "values": {
                    **{str(key): value for key, value in raw_inputs.items()},
                    **scalar_inputs,
                    "market_frame": list(market_frame),
                },
            },
            created_at=str(payload["availability_time"]),
        )

    def _resolve_input_references(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        references = self._validated_references(payload)
        bar = references.get("bar_window")
        if not isinstance(bar, Mapping):
            raise ValueError("feature job requires a bar_window input reference")
        resolved, available_at = self._resolve_bar_window(payload, bar)
        self._merge_reference_inputs(
            resolved,
            references=references,
            bar=bar,
            payload=payload,
            available_at=available_at,
        )
        return resolved

    def _validated_references(self, payload: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        references = payload.get("input_references")
        if not isinstance(references, Mapping):
            raise ValueError("feature input_references must be an object")
        for name, reference in references.items():
            if not isinstance(reference, Mapping):
                raise ValueError(f"feature input reference {name} must be an object")
            self._verify_input_reference(name=str(name), reference=reference)
        return references

    def _resolve_bar_window(
        self, payload: Mapping[str, Any], bar: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dt.datetime]:
        pattern = str(bar.get("relative_pattern") or "")
        if not pattern.startswith("bars/") or ".." in Path(pattern).parts:
            raise ValueError("feature bar reference has an unsafe partition")
        if self.parquet_store is None:
            raise ValueError("feature bar references require a parquet root")
        through = dt.datetime.fromisoformat(str(bar["through_close_time"]))
        available_at = dt.datetime.fromisoformat(str(payload["availability_time"]))
        source_ids = bar.get("source_event_ids", ())
        if not isinstance(source_ids, list | tuple):
            raise ValueError("feature bar reference source_event_ids must be a list")
        rows = self._bar_rows(pattern, payload, through=through, available_at=available_at)
        ordered = sorted(
            rows.values(), key=lambda row: (int(row["close_time_ms"]), str(row["event_id"]))
        )
        if len(ordered) < int(bar.get("minimum_history", 1)):
            raise ValueError("feature input reference does not contain the required bar history")
        target_rows = (
            [
                row
                for row in ordered
                if str(row.get("event_id")) in {str(item) for item in source_ids}
            ]
            if source_ids
            else ordered
        )
        latest = self._latest_bar(payload, target_rows, through=through)
        return self._bar_values(latest, ordered), available_at

    def _bar_rows(
        self,
        pattern: str,
        payload: Mapping[str, Any],
        *,
        through: dt.datetime,
        available_at: dt.datetime,
    ) -> dict[str, dict[str, Any]]:
        assert self.parquet_store is not None
        rows: dict[str, dict[str, Any]] = {}
        for path in self.parquet_store.root.glob(pattern):
            if path.is_symlink() or not path.is_file():
                continue
            for row in pq.read_table(path).to_pylist():
                if str(row.get("instrument_id")) != str(payload["instrument_id"]):
                    continue
                close_time = dt.datetime.fromtimestamp(float(row["close_time_ms"]) / 1_000, dt.UTC)
                availability = dt.datetime.fromisoformat(str(row["availability_time"]))
                if close_time <= through and availability <= available_at:
                    rows[str(row["event_id"])] = row
        return rows

    @staticmethod
    def _latest_bar(
        payload: Mapping[str, Any],
        target_rows: list[dict[str, Any]],
        *,
        through: dt.datetime,
    ) -> dict[str, Any]:
        if not target_rows:
            raise ValueError("feature bar reference does not contain its source event")
        latest = target_rows[-1]
        declared_event_id = str(payload.get("source_market_event_id") or "")
        if declared_event_id and str(latest.get("event_id")) != declared_event_id:
            raise ValueError("feature bar reference does not match the source event identity")
        expected_open = dt.datetime.fromisoformat(str(payload["source_event_time"]))
        expected_close = dt.datetime.fromisoformat(str(payload["source_close_time"]))
        actual_open = dt.datetime.fromtimestamp(float(latest["open_time_ms"]) / 1_000, dt.UTC)
        actual_close = dt.datetime.fromtimestamp(float(latest["close_time_ms"]) / 1_000, dt.UTC)
        if actual_open != expected_open or actual_close != expected_close:
            raise ValueError("feature bar reference does not match the source event timestamps")
        if actual_close > through:
            raise ValueError("feature bar reference source event is outside its time boundary")
        return latest

    @staticmethod
    def _bar_values(
        latest: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            name: float(latest[name]) for name in ("open", "high", "low", "close", "volume")
        }
        for name in ("open", "high", "low", "close", "volume"):
            result[f"{name}_history"] = [float(row[name]) for row in ordered]
        result["market_frame"] = [
            {name: float(row[name]) for name in ("open", "high", "low", "close", "volume")}
            for row in ordered
        ]
        for name in ("spread_bps", "visible_depth", "volatility", "funding"):
            if latest.get(name) is not None:
                result[name] = float(latest[name])
        if latest.get("funding") is not None:
            result["funding_rate"] = float(latest["funding"])
        return result

    def _merge_reference_inputs(
        self,
        resolved: dict[str, Any],
        *,
        references: Mapping[str, Mapping[str, Any]],
        bar: Mapping[str, Any],
        payload: Mapping[str, Any],
        available_at: dt.datetime,
    ) -> None:
        for name, reference in references.items():
            if name == "bar_window":
                continue
            reference_available = dt.datetime.fromisoformat(
                str(reference.get("availability_time", payload["availability_time"]))
            )
            if reference_available > available_at:
                raise ValueError(f"feature input reference {name} is not available")
            reference_through = dt.datetime.fromisoformat(
                str(reference.get("through_close_time", bar["through_close_time"]))
            )
            auxiliary_rows = self._resolve_auxiliary_rows(
                reference,
                instrument_id=str(payload["instrument_id"]),
                available_at=available_at,
                through=reference_through,
                same_instrument=name
                not in {"cross_sectional", "correlation_beta", "spot_perpetual"},
            )
            _merge_auxiliary_inputs(
                resolved,
                name=str(name),
                rows=auxiliary_rows,
                instrument_id=str(payload["instrument_id"]),
            )

    @staticmethod
    def _verify_input_reference(*, name: str, reference: Mapping[str, Any]) -> None:
        content_hash = reference.get("content_hash")
        if (
            not isinstance(content_hash, str)
            or canonical_hash(
                {str(key): value for key, value in reference.items() if key != "content_hash"}
            )
            != content_hash
        ):
            raise ValueError(f"feature input reference {name} has an invalid content hash")

    def _resolve_auxiliary_rows(
        self,
        reference: Mapping[str, Any],
        *,
        instrument_id: str,
        available_at: dt.datetime,
        through: dt.datetime,
        same_instrument: bool,
    ) -> list[dict[str, Any]]:
        pattern = str(reference.get("relative_pattern") or "")
        if not pattern.startswith("raw/") or ".." in Path(pattern).parts:
            raise ValueError("feature auxiliary reference has an unsafe partition")
        event_types, source_ids = _auxiliary_filters(reference)
        rows: list[dict[str, Any]] = []
        assert self.parquet_store is not None
        for path in self.parquet_store.root.glob(pattern):
            if path.is_symlink() or not path.is_file():
                continue
            for row in pq.read_table(path).to_pylist():
                decoded = self._auxiliary_row(
                    row,
                    source_ids=source_ids,
                    event_types=event_types,
                    instrument_id=instrument_id,
                    same_instrument=same_instrument,
                    available_at=available_at,
                    through=through,
                )
                if decoded is not None:
                    rows.append(decoded)
        return sorted(
            rows,
            key=lambda row: (
                row["exchange_timestamp"],
                row["availability_time"],
                row["event_id"],
            ),
        )

    @staticmethod
    def _auxiliary_row(
        row: Mapping[str, Any],
        *,
        source_ids: set[str],
        event_types: set[str],
        instrument_id: str,
        same_instrument: bool,
        available_at: dt.datetime,
        through: dt.datetime,
    ) -> dict[str, Any] | None:
        if source_ids and str(row.get("event_id")) not in source_ids:
            return None
        if same_instrument and str(row.get("instrument_id")) != instrument_id:
            return None
        if event_types and str(row.get("event_type")) not in event_types:
            return None
        try:
            row_availability = dt.datetime.fromisoformat(str(row["availability_time"]))
            row_exchange = dt.datetime.fromisoformat(str(row["exchange_timestamp"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feature auxiliary row has invalid timestamps") from exc
        if row_availability > available_at or row_exchange > through:
            return None
        payload_json = row.get("payload_json")
        if not isinstance(payload_json, str):
            raise ValueError("feature auxiliary row has no immutable payload")
        try:
            decoded = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("feature auxiliary row payload is not valid JSON") from exc
        return {
            "event_id": str(row.get("event_id")),
            "instrument_id": str(row.get("instrument_id")),
            "event_type": str(row.get("event_type")),
            "exchange_timestamp": row_exchange,
            "availability_time": row_availability,
            "payload": decoded,
        }


def _auxiliary_filters(reference: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    raw_event_types = reference.get("event_types", ())
    if not isinstance(raw_event_types, list | tuple):
        raise ValueError("feature auxiliary reference event_types must be a list")
    raw_ids = reference.get("source_event_ids", ())
    if not isinstance(raw_ids, list | tuple):
        raise ValueError("feature auxiliary reference source_event_ids must be a list")
    return {str(item) for item in raw_event_types}, {str(item) for item in raw_ids}


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def _payload_views(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, Mapping):
        return ()
    views: list[Mapping[str, Any]] = [payload]
    data = payload.get("data")
    if isinstance(data, Mapping):
        views.append(data)
        candle = data.get("k")
        if isinstance(candle, Mapping):
            views.append(candle)
    return tuple(views)


def _number(views: tuple[Mapping[str, Any], ...], *names: str) -> float | None:
    for view in views:
        for name in names:
            value = view.get(name)
            if value is None or isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _levels(views: tuple[Mapping[str, Any], ...], *names: str) -> tuple[float, float] | None:
    for view in views:
        raw_levels = next((view.get(name) for name in names if view.get(name) is not None), None)
        if not isinstance(raw_levels, list):
            continue
        levels: list[tuple[float, float]] = []
        for raw_level in raw_levels:
            if not isinstance(raw_level, list | tuple) or len(raw_level) < 2:
                continue
            try:
                levels.append((float(raw_level[0]), float(raw_level[1])))
            except (TypeError, ValueError):
                continue
        if levels:
            total_depth = sum(quantity for _, quantity in levels)
            weighted_price = sum(price * quantity for price, quantity in levels)
            return weighted_price / total_depth if total_depth else levels[0][0], total_depth
    return None


def _merge_auxiliary_inputs(
    resolved: dict[str, Any],
    *,
    name: str,
    rows: list[dict[str, Any]],
    instrument_id: str,
) -> None:
    if not rows:
        return
    views = tuple(view for row in reversed(rows) for view in _payload_views(row["payload"]))
    handlers = {
        "higher_timeframe": _merge_higher_timeframe,
        "order_book": _merge_order_book,
        "trade_flow": _merge_trade_flow,
        "funding_open_interest": _merge_funding_open_interest,
        "spot_perpetual": _merge_spot_perpetual,
        "cross_sectional": _merge_cross_sectional,
        "correlation_beta": _merge_correlation_beta,
        "sentiment": _merge_sentiment,
        "ml_manifest": _merge_ml_manifest,
        "liquidation": _merge_liquidation,
    }
    handler = handlers.get(name)
    if handler is not None:
        handler(resolved, rows, views, instrument_id)


def _merge_higher_timeframe(
    resolved: dict[str, Any], rows: list[dict[str, Any]], _views: Any, _instrument_id: str
) -> None:
    closes = [
        close
        for row in rows
        if (close := _number(_payload_views(row["payload"]), "c", "close")) is not None
    ]
    if len(closes) >= 2 and closes[-2] > 0:
        resolved["higher_timeframe_return"] = closes[-1] / closes[-2] - 1.0


def _merge_order_book(
    resolved: dict[str, Any], _rows: list[dict[str, Any]], views: Any, _instrument_id: str
) -> None:
    bid = _number(views, "bid_price", "b")
    ask = _number(views, "ask_price", "a")
    bid_depth = _number(views, "bid_depth", "B")
    ask_depth = _number(views, "ask_depth", "A")
    bid_levels = _levels(views, "b", "bids")
    ask_levels = _levels(views, "a", "asks")
    if bid_levels is not None:
        bid, bid_depth = bid_levels
    if ask_levels is not None:
        ask, ask_depth = ask_levels
    for key, value in (
        ("bid_price", bid),
        ("ask_price", ask),
        ("bid_depth", bid_depth),
        ("ask_depth", ask_depth),
    ):
        if value is not None:
            resolved[key] = value


def _merge_trade_flow(
    resolved: dict[str, Any], rows: list[dict[str, Any]], _views: Any, _instrument_id: str
) -> None:
    buy = sell = 0.0
    for row in rows:
        row_views = _payload_views(row["payload"])
        direct_buy = _number(row_views, "taker_buy_volume", "aggressor_buy_volume")
        direct_sell = _number(row_views, "taker_sell_volume", "aggressor_sell_volume")
        quantity = _number(row_views, "q", "quantity", "qty", "volume")
        maker = next(
            (bool(view["m"]) for view in row_views if isinstance(view.get("m"), bool)),
            None,
        )
        if direct_buy is not None:
            buy += direct_buy
        if direct_sell is not None:
            sell += direct_sell
        elif quantity is not None and maker is not None:
            sell += quantity if maker else 0.0
            buy += quantity if not maker else 0.0
    if buy or sell:
        resolved.update(
            {
                "taker_buy_volume": buy,
                "aggressor_buy_volume": buy,
                "aggressor_sell_volume": sell,
            }
        )


def _merge_funding_open_interest(
    resolved: dict[str, Any], _rows: list[dict[str, Any]], views: Any, _instrument_id: str
) -> None:
    funding = _number(views, "funding_rate", "fundingRate", "r", "funding")
    open_interest = _number(views, "open_interest", "openInterest", "oi")
    if funding is not None:
        resolved["funding_rate"] = funding
    if open_interest is not None:
        resolved["open_interest"] = open_interest


def _merge_spot_perpetual(
    resolved: dict[str, Any], rows: list[dict[str, Any]], _views: Any, instrument_id: str
) -> None:
    prices = {
        str(row.get("instrument_id")): price
        for row in rows
        if (price := _number(_payload_views(row["payload"]), "close", "c", "price")) is not None
        and price > 0
    }
    current_market = "futures" if ":futures:" in instrument_id else "spot"
    other_market = "spot" if current_market == "futures" else "futures"
    current_price = prices.get(instrument_id)
    other_price = next(
        (price for identity, price in prices.items() if f":{other_market}:" in identity),
        None,
    )
    if current_price is not None:
        resolved["perpetual_price" if current_market == "futures" else "spot_price"] = current_price
    if other_price is not None:
        resolved["spot_price" if current_market == "futures" else "perpetual_price"] = other_price


def _merge_cross_sectional(
    resolved: dict[str, Any], rows: list[dict[str, Any]], views: Any, instrument_id: str
) -> None:
    explicit = next(
        (
            view.get("cross_sectional_values")
            for view in views
            if isinstance(view.get("cross_sectional_values"), Mapping)
        ),
        None,
    )
    if isinstance(explicit, Mapping) and explicit:
        panel = {str(key): float(value) for key, value in explicit.items()}
    else:
        closes: dict[str, list[float]] = {}
        for row in rows:
            close = _number(_payload_views(row["payload"]), "c", "close")
            if close is not None:
                closes.setdefault(str(row.get("instrument_id")), []).append(close)
        panel = {
            identity: values[-1] / values[-2] - 1.0
            for identity, values in closes.items()
            if len(values) >= 2 and values[-2] > 0
        }
    if instrument_id in panel:
        panel["__current__"] = panel[instrument_id]
    if panel:
        resolved["cross_sectional_values"] = panel


def _merge_correlation_beta(
    resolved: dict[str, Any], rows: list[dict[str, Any]], _views: Any, instrument_id: str
) -> None:
    closes: dict[str, list[float]] = {}
    for row in rows:
        close = _number(_payload_views(row["payload"]), "c", "close")
        if close is not None and close > 0:
            closes.setdefault(str(row.get("instrument_id")), []).append(close)
    returns = {
        identity: [
            values[index] / values[index - 1] - 1.0
            for index in range(1, len(values))
            if values[index - 1] > 0
        ]
        for identity, values in closes.items()
    }
    asset = returns.get(instrument_id, [])
    benchmark = next(
        (values for identity, values in returns.items() if identity != instrument_id and values),
        [],
    )
    if asset and benchmark:
        resolved.update({"asset_returns": asset, "benchmark_returns": benchmark})


def _merge_sentiment(
    resolved: dict[str, Any], _rows: list[dict[str, Any]], views: Any, _instrument_id: str
) -> None:
    value = _number(views, "sentiment_score", "score")
    if value is not None:
        resolved["sentiment_score"] = value


def _merge_ml_manifest(
    resolved: dict[str, Any], _rows: list[dict[str, Any]], views: Any, _instrument_id: str
) -> None:
    vector = next(
        (
            view.get("feature_vector", view.get("features"))
            for view in views
            if isinstance(view.get("feature_vector", view.get("features")), Mapping)
        ),
        None,
    )
    if isinstance(vector, Mapping) and vector:
        resolved["feature_vector"] = {str(key): float(value) for key, value in vector.items()}


def _merge_liquidation(
    resolved: dict[str, Any], _rows: list[dict[str, Any]], views: Any, _instrument_id: str
) -> None:
    buy = _number(views, "liquidation_buy_volume")
    sell = _number(views, "liquidation_sell_volume")
    if buy is not None:
        resolved["liquidation_buy_volume"] = buy
    if sell is not None:
        resolved["liquidation_sell_volume"] = sell
