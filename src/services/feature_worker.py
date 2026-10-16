"""Shared live and historical deterministic feature worker."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.data.feature_graph import (
    AvailableValue,
    FeatureGraphRegistry,
    FeatureNodeType,
    default_feature_engine,
)
from src.data.feature_store import DeterministicFeatureCalculator, FeatureValue, SqlFeatureStore
from src.data.parquet_store import (
    DurableFeatureBatchSpool,
    FeatureBatchSegment,
    PartitionedFeatureStore,
    partition_files,
)
from src.domain._codec import canonical_hash, timeframe_seconds
from src.domain.market_events import MarketEvent, MarketEventType
from src.risk.engine import SqlRiskSnapshotStore
from src.services.job_schemas import build_content_hash
from src.services.scheduler import DatabaseJobQueue

_EMA_WARMUP_BARS = 200
_AUXILIARY_FAMILIES_BY_NODE_TYPE = {
    FeatureNodeType.MULTI_TIMEFRAME: {"higher_timeframe"},
    FeatureNodeType.CROSS_SECTIONAL_RANK: {"cross_sectional"},
    FeatureNodeType.FUNDING_OPEN_INTEREST: {"funding_open_interest"},
    FeatureNodeType.SPOT_PERPETUAL_BASIS: {"spot_perpetual"},
    FeatureNodeType.CORRELATION_BETA: {"correlation_beta", "linked_instruments"},
    FeatureNodeType.ORDER_BOOK: {"order_book"},
    FeatureNodeType.TRADE_FLOW: {"trade_flow"},
    FeatureNodeType.LIQUIDATION: {"liquidation"},
    FeatureNodeType.SENTIMENT: {"sentiment"},
    FeatureNodeType.FROZEN_ML_FEATURE_SET: {"ml_manifest"},
}


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
        self.feature_spool = (
            DurableFeatureBatchSpool(parquet_root.parent / "runtime" / "events" / "feature-spool")
            if parquet_root is not None
            else None
        )
        self.active_assignments = active_assignments or (lambda _instrument_id: ())
        self.snapshot_store = snapshot_store
        self.feature_graph_for_assignment = feature_graph_for_assignment
        self.graph_registry = FeatureGraphRegistry.default()
        self.graph_engine = default_feature_engine()
        self.job_names = job_names
        self.lease_seconds = lease_seconds

    def run_once(self, *, now: str) -> dict[str, Any]:
        try:
            self._flush_recoverable_features()
        except Exception as exc:
            return {
                "reason_code": "feature_spool_recovery_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
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
            assignments = tuple(self.active_assignments(str(payload["instrument_id"])))
            assignments = self._matching_timeframes(assignments, payload)
            required_families, history_limit = self._feature_input_requirements(
                assignments, payload
            )
            raw_inputs, scalar_inputs = self._input_values(
                payload,
                job_name=claimed.name,
                required_families=required_families,
                history_limit=history_limit,
            )
            values = self._calculate_core_values(payload, scalar_inputs)
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
        self,
        payload: Mapping[str, Any],
        *,
        job_name: str,
        required_families: set[str] | None = None,
        history_limit: int = 1,
    ) -> tuple[Mapping[str, Any], dict[str, float]]:
        if job_name == "live_feature_calculation" and not isinstance(
            payload.get("input_references"), Mapping
        ):
            raise ValueError("live feature jobs require immutable input references")
        raw_inputs = payload.get("inputs")
        if payload.get("input_references") is not None:
            raw_inputs = self._resolve_input_references(
                payload,
                required_families=required_families,
                history_limit=history_limit,
            )
        if not isinstance(raw_inputs, Mapping):
            raise ValueError("feature job requires immutable input references or an input object")
        scalar_inputs = {
            str(key): float(value)
            for key, value in raw_inputs.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        return raw_inputs, scalar_inputs

    def _feature_input_requirements(
        self, assignments: tuple[Mapping[str, Any], ...], payload: Mapping[str, Any]
    ) -> tuple[set[str] | None, int]:
        if self.feature_graph_for_assignment is None:
            return None, 1
        requirements = self._assignment_feature_requirements(assignments)
        required_nodes = tuple(
            dict.fromkeys(name for names in requirements.values() for name in names)
        )
        if not required_nodes:
            return set(), 1
        graph = self.graph_registry.graph(
            required_nodes, version=str(payload["feature_set_version"])
        )
        families: set[str] = set()
        bar_history = 1
        for node in graph.nodes:
            families.update(_AUXILIARY_FAMILIES_BY_NODE_TYPE.get(node.node_type, ()))
            if node.node_type is FeatureNodeType.BAR_INDICATOR:
                bar_history = max(bar_history, node.minimum_history)
                if node.name in {"ema", "macd"}:
                    bar_history = max(bar_history, node.minimum_history + _EMA_WARMUP_BARS)
        return families, bar_history

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

    def _matching_timeframes(self, assignments, payload):
        if self.feature_graph_for_assignment is None:
            return assignments
        result = []
        for assignment in assignments:
            declared = self.feature_graph_for_assignment(assignment).get("timeframe")
            if declared is None or timeframe_seconds(str(declared)) == timeframe_seconds(
                str(payload.get("timeframe", "1m"))
            ):
                result.append(assignment)
        return tuple(result)

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
        if self.parquet_store is None or self.feature_spool is None:
            return None
        segment = self.feature_spool.append(
            values,
            venue=str(payload["venue"]),
            market=str(payload["market"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
        )
        if segment is None:
            return None
        paths = self._write_feature_segment(segment)
        return paths[0] if paths else None

    def flush(self) -> tuple[Path, ...]:
        """Flush in-memory and recoverable feature batches before shutdown."""

        if self.feature_spool is None or self.parquet_store is None:
            return ()
        paths: list[Path] = []
        for path in self.feature_spool.recoverable():
            paths.extend(
                self._write_feature_segment(FeatureBatchSegment(path, _segment_hash(path), 0))
            )
        segment = self.feature_spool.flush()
        if segment is not None:
            paths.extend(self._write_feature_segment(segment))
        return tuple(paths)

    def _flush_recoverable_features(self) -> None:
        if self.feature_spool is None:
            return
        for path in self.feature_spool.recoverable():
            self._write_feature_segment(FeatureBatchSegment(path, _segment_hash(path), 0))

    def _write_feature_segment(self, segment: FeatureBatchSegment) -> tuple[Path, ...]:
        assert self.feature_spool is not None
        assert self.parquet_store is not None
        claimed = self.feature_spool.claim(segment.path)
        rows = self.feature_spool.read(claimed)
        body = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        content_hash = canonical_hash(body)
        if content_hash != segment.content_hash:
            raise ValueError("feature batch segment content hash is invalid")
        grouped: dict[tuple[str, str, str, str, str, str, str], list[FeatureValue]] = {}
        for row in rows:
            raw = row.get("value")
            if not isinstance(raw, Mapping):
                raise ValueError("feature batch row has no feature value")
            value = FeatureValue(**dict(raw))
            key = (
                str(row["venue"]),
                str(row["market"]),
                str(row["symbol"]),
                str(row["timeframe"]),
                value.feature_set_version,
                value.instrument_id,
                value.availability_time[:10],
            )
            grouped.setdefault(key, []).append(value)
        paths = tuple(
            path
            for (
                venue,
                market,
                symbol,
                timeframe,
                _feature_set,
                _instrument,
                _date,
            ), values in sorted(grouped.items())
            for path in self.parquet_store.put_batch(
                values,
                venue=venue,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
            )
        )
        committed = self.feature_spool.commit(claimed)
        self.feature_spool.delete(committed)
        return paths

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
            "horizon_seconds": int(
                payload.get(
                    "horizon_seconds", timeframe_seconds(str(payload.get("timeframe", "1m")))
                )
            ),
            "observation_interval_seconds": timeframe_seconds(str(payload.get("timeframe", "1m"))),
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

    def _resolve_input_references(
        self,
        payload: Mapping[str, Any],
        *,
        required_families: set[str] | None = None,
        history_limit: int = 1,
    ) -> dict[str, Any]:
        references = self._validated_references(payload)
        bar = references.get("bar_window")
        if not isinstance(bar, Mapping):
            raise ValueError("feature job requires a bar_window input reference")
        resolved, available_at = self._resolve_bar_window(payload, bar, history_limit=history_limit)
        self._merge_reference_inputs(
            resolved,
            references=references,
            bar=bar,
            payload=payload,
            available_at=available_at,
            required_families=required_families,
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
        self,
        payload: Mapping[str, Any],
        bar: Mapping[str, Any],
        *,
        history_limit: int = 1,
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
        minimum_history = max(int(bar.get("minimum_history", 1)), history_limit)
        rows = self._bar_rows(
            pattern,
            payload,
            through=through,
            available_at=available_at,
            source_ids={str(item) for item in source_ids},
            history_limit=minimum_history,
        )
        ordered = self._dedupe_bar_rows(rows.values(), limit=minimum_history)
        if len(ordered) < minimum_history:
            raise ValueError("feature input reference does not contain the required bar history")
        if source_ids and not {str(item) for item in source_ids}.issubset(rows):
            raise ValueError("feature input reference does not contain all target event IDs")
        target_rows = [rows[str(item)] for item in source_ids] if source_ids else ordered
        target_rows.sort(key=lambda row: (int(row["close_time_ms"]), str(row["event_id"])))
        latest = self._latest_bar(payload, target_rows, through=through)
        return self._bar_values(latest, ordered), available_at

    def _bar_rows(
        self,
        pattern: str,
        payload: Mapping[str, Any],
        *,
        through: dt.datetime,
        available_at: dt.datetime,
        source_ids: set[str],
        history_limit: int,
    ) -> dict[str, dict[str, Any]]:
        assert self.parquet_store is not None
        rows: dict[str, dict[str, Any]] = {}
        paths_by_partition: dict[Path, list[Path]] = {}
        for path in partition_files(self.parquet_store.root, pattern):
            paths_by_partition.setdefault(path.parent, []).append(path)
        for directory in sorted(paths_by_partition, reverse=True):
            for path in paths_by_partition[directory]:
                if path.is_symlink() or not path.is_file():
                    continue
                for row in pq.ParquetFile(path).read().to_pylist():
                    if str(row.get("instrument_id")) != str(payload["instrument_id"]):
                        continue
                    close_time = dt.datetime.fromtimestamp(
                        float(row["close_time_ms"]) / 1_000, dt.UTC
                    )
                    availability = dt.datetime.fromisoformat(str(row["availability_time"]))
                    if close_time <= through and availability <= available_at:
                        identity = str(row["event_id"])
                        current = rows.get(identity)
                        if current is None or availability < dt.datetime.fromisoformat(
                            str(current["availability_time"])
                        ):
                            rows[identity] = row
            ordered = self._dedupe_bar_rows(rows.values(), limit=history_limit)
            if len(ordered) >= history_limit and source_ids.issubset(rows):
                break
        return rows

    @staticmethod
    def _dedupe_bar_rows(rows: Iterable[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        logical: dict[tuple[str, int, int], dict[str, Any]] = {}
        for raw_row in rows:
            row = dict(raw_row)
            key = (
                str(row.get("instrument_id") or ""),
                int(row["open_time_ms"]),
                int(row["close_time_ms"]),
            )
            current = logical.get(key)
            availability = dt.datetime.fromisoformat(str(row["availability_time"]))
            if (
                current is None
                or availability < dt.datetime.fromisoformat(str(current["availability_time"]))
                or availability == dt.datetime.fromisoformat(str(current["availability_time"]))
                and str(row["event_id"]) < str(current["event_id"])
            ):
                logical[key] = row
        ordered = sorted(
            logical.values(),
            key=lambda row: (int(row["close_time_ms"]), str(row["event_id"])),
        )
        return ordered[-limit:]

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
        for name in ("spread_bps", "visible_depth_quote", "volatility", "funding"):
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
        required_families: set[str] | None,
    ) -> None:
        for name, reference in references.items():
            if name == "bar_window":
                continue
            if required_families is not None and name not in required_families:
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
                not in {
                    "cross_sectional",
                    "correlation_beta",
                    "spot_perpetual",
                    "linked_instruments",
                },
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
        snapshot_source = reference.get("snapshot_source")
        if snapshot_source == "disabled":
            return []
        if snapshot_source == "market_input":
            return self._resolve_market_input_rows(
                reference,
                instrument_id=instrument_id,
                available_at=available_at,
                through=through,
                same_instrument=same_instrument,
            )
        if snapshot_source is not None:
            raise ValueError("feature auxiliary snapshot source is unsupported")
        pattern = str(reference.get("relative_pattern") or "")
        if not pattern.startswith("raw/") or ".." in Path(pattern).parts:
            raise ValueError("feature auxiliary reference has an unsafe partition")
        event_types, source_ids = _auxiliary_filters(reference)
        rows: list[dict[str, Any]] = []
        assert self.parquet_store is not None
        for path in partition_files(self.parquet_store.root, pattern):
            if path.is_symlink() or not path.is_file():
                continue
            for row in pq.ParquetFile(path).read().to_pylist():
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
        rows = _dedupe_auxiliary_candles(rows)
        return sorted(
            rows,
            key=lambda row: (
                row["source_time"],
                row["availability_time"],
                row["event_id"],
            ),
        )

    def _resolve_market_input_rows(
        self,
        reference: Mapping[str, Any],
        *,
        instrument_id: str,
        available_at: dt.datetime,
        through: dt.datetime,
        same_instrument: bool,
    ) -> list[dict[str, Any]]:
        if self.snapshot_store is None:
            raise ValueError("market input snapshot references require a snapshot store")
        event_types, _source_ids = _auxiliary_filters(reference)
        raw_ids = reference.get("source_snapshot_ids")
        if not isinstance(raw_ids, list | tuple):
            raise ValueError("feature market input snapshot IDs must be a list")
        rows: dict[str, dict[str, Any]] = {}
        for snapshot_id in raw_ids:
            if not isinstance(snapshot_id, str):
                raise ValueError("feature market input snapshot ID must be a string")
            event = self._decode_market_input_snapshot(snapshot_id)
            if event_types and event.event_type.value not in event_types:
                raise ValueError("feature market input event type is outside its reference")
            if same_instrument and event.instrument_id != instrument_id:
                raise ValueError("feature market input instrument is outside its reference")
            event_available_at = dt.datetime.fromisoformat(event.availability_timestamp)
            event_exchange_at = dt.datetime.fromisoformat(event.exchange_timestamp)
            if event_available_at > available_at or event_exchange_at > through:
                continue
            row = {
                "event_id": event.event_id,
                "instrument_id": event.instrument_id,
                "event_type": event.event_type.value,
                "sequence": event.sequence,
                "exchange_timestamp": event_exchange_at,
                "receive_timestamp": dt.datetime.fromisoformat(event.receive_timestamp),
                "close_timestamp": dt.datetime.fromisoformat(
                    str(event.close_timestamp or event.exchange_timestamp)
                ),
                "availability_time": event_available_at,
                "payload": dict(event.payload),
            }
            existing = rows.get(event.event_id)
            if existing is not None and existing != row:
                raise ValueError("duplicate market input IDs disagree on their event")
            rows[event.event_id] = row
        return sorted(
            rows.values(),
            key=lambda row: (
                row["exchange_timestamp"],
                row["availability_time"],
                row["event_id"],
            ),
        )

    def _decode_market_input_snapshot(self, snapshot_id: str) -> MarketEvent:
        assert self.snapshot_store is not None
        payload = self.snapshot_store.get(snapshot_id)
        if payload.get("kind") != "market_data_input":
            raise ValueError("feature market input snapshot has the wrong kind")
        sequence = payload.get("source_sequence")
        source_payload = payload.get("source_payload")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(source_payload, Mapping)
        ):
            raise ValueError("feature market input snapshot lacks a canonical event")
        fields = (
            "instrument_id",
            "source_event_type",
            "source_event_time",
            "source_receive_time",
            "source_close_time",
            "availability_time",
            "source_event_id",
        )
        if any(not isinstance(payload.get(field), str) for field in fields):
            raise ValueError("feature market input snapshot has invalid event fields")
        try:
            event = MarketEvent(
                instrument_id=str(payload["instrument_id"]),
                event_type=MarketEventType(str(payload["source_event_type"])),
                exchange_timestamp=str(payload["source_event_time"]),
                receive_timestamp=str(payload["source_receive_time"]),
                sequence=sequence,
                payload=source_payload,
                close_timestamp=str(payload["source_close_time"]),
                availability_time=str(payload["availability_time"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("feature market input snapshot has an invalid event") from exc
        if event.event_id != payload["source_event_id"]:
            raise ValueError("feature market input source event identity is invalid")
        return event

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
        if row_availability > available_at:
            return None
        payload_json = row.get("payload_json")
        if not isinstance(payload_json, str):
            raise ValueError("feature auxiliary row has no immutable payload")
        try:
            decoded = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("feature auxiliary row payload is not valid JSON") from exc
        event_type = str(row.get("event_type"))
        source_time = _auxiliary_source_time(event_type, decoded, row_exchange)
        if source_time is None or source_time > through:
            return None
        return {
            "event_id": str(row.get("event_id")),
            "instrument_id": str(row.get("instrument_id")),
            "event_type": event_type,
            "sequence": row.get("sequence"),
            "exchange_timestamp": row_exchange,
            "source_time": source_time,
            "availability_time": row_availability,
            "payload": decoded,
        }


def _auxiliary_source_time(
    event_type: str, payload: Any, exchange_time: dt.datetime
) -> dt.datetime | None:
    if event_type != "candle":
        return exchange_time
    data = payload.get("data") if isinstance(payload, Mapping) else None
    candle = data.get("k") if isinstance(data, Mapping) else None
    if not isinstance(candle, Mapping):
        raise ValueError("feature candle auxiliary row has no candle payload")
    close_ms = candle.get("T")
    if not isinstance(close_ms, int) or isinstance(close_ms, bool):
        raise ValueError("feature candle auxiliary row has an invalid close time")
    if candle.get("x") is not True:
        return None
    return dt.datetime.fromtimestamp(close_ms / 1_000, dt.UTC)


def _auxiliary_filters(reference: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    raw_event_types = reference.get("event_types", ())
    if not isinstance(raw_event_types, list | tuple):
        raise ValueError("feature auxiliary reference event_types must be a list")
    raw_ids = reference.get("source_event_ids", ())
    if not isinstance(raw_ids, list | tuple):
        raise ValueError("feature auxiliary reference source_event_ids must be a list")
    return {str(item) for item in raw_event_types}, {str(item) for item in raw_ids}


def _dedupe_auxiliary_candles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    other: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        data = payload.get("data") if isinstance(payload, Mapping) else None
        candle = data.get("k") if isinstance(data, Mapping) else None
        if row.get("event_type") != "candle" or not isinstance(candle, Mapping):
            other.append(row)
            continue
        try:
            key = (str(row["instrument_id"]), str(candle["i"]), int(candle["t"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feature candle auxiliary row has no logical identity") from exc
        current = selected.get(key)
        if current is None or (row["availability_time"], row["event_id"]) < (
            current["availability_time"],
            current["event_id"],
        ):
            selected[key] = row
    return [*other, *selected.values()]


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()


def _segment_hash(path: Path) -> str:
    return canonical_hash(path.read_text(encoding="utf-8"))


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
        "linked_instruments": _merge_linked_instruments,
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
    resolved: dict[str, Any], rows: list[dict[str, Any]], views: Any, _instrument_id: str
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
    sequences = [
        int(row["sequence"])
        for row in rows
        if isinstance(row.get("sequence"), int)
        and not isinstance(row.get("sequence"), bool)
        and int(row["sequence"]) >= 0
    ]
    if sequences:
        resolved["sequence"] = max(sequences)


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
    latest_time, latest_rows = _latest_synchronised_rows(rows)
    del latest_time
    prices = {
        str(row.get("instrument_id")): price
        for row in latest_rows
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


def _merge_linked_instruments(
    resolved: dict[str, Any], rows: list[dict[str, Any]], _views: Any, _instrument_id: str
) -> None:
    _latest_time, rows = _latest_synchronised_rows(rows)
    legs: dict[str, dict[str, float]] = {}
    for row in rows:
        price = _number(_payload_views(row["payload"]), "close", "c", "price")
        identity = str(row.get("instrument_id") or "").strip()
        if identity and price is not None and price > 0.0:
            legs[identity] = {"price": price, "beta": 1.0}
    if len(legs) >= 2:
        resolved["linked_instruments"] = legs


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
        panel = {
            str(key): float(
                value.get("momentum", value.get("relative_return", value.get("value", 0.0)))
                if isinstance(value, Mapping)
                else value
            )
            for key, value in explicit.items()
            if not str(key).startswith("__")
        }
    else:
        latest_time, latest_rows = _latest_synchronised_rows(rows)
        if latest_time is None:
            panel = {}
        else:
            current = {
                str(row.get("instrument_id")): close
                for row in latest_rows
                if (close := _number(_payload_views(row["payload"]), "c", "close")) is not None
            }
            previous: dict[str, tuple[dt.datetime, float]] = {}
            for row in rows:
                identity = str(row.get("instrument_id") or "")
                observed = row.get("exchange_timestamp")
                close = _number(_payload_views(row["payload"]), "c", "close")
                if not identity or close is None or observed is None:
                    continue
                if not isinstance(observed, dt.datetime):
                    try:
                        observed = dt.datetime.fromisoformat(str(observed))
                    except (TypeError, ValueError):
                        continue
                if observed < latest_time and (
                    identity not in previous or observed > previous[identity][0]
                ):
                    previous[identity] = (observed, close)
            panel = {
                identity: close / previous[identity][1] - 1.0
                for identity, close in current.items()
                if identity in previous and previous[identity][1] > 0.0
            }
    if instrument_id in panel:
        panel["__current__"] = panel[instrument_id]
    if panel:
        resolved["cross_sectional_values"] = panel


def _latest_synchronised_rows(
    rows: list[dict[str, Any]],
) -> tuple[dt.datetime | None, list[dict[str, Any]]]:
    """Keep only the latest exchange-time cross-section for linked inputs."""

    by_time: dict[dt.datetime, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get("exchange_timestamp")
        if not isinstance(value, dt.datetime):
            try:
                value = dt.datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                continue
        by_time.setdefault(value, []).append(row)
    candidates = [
        (observed_at, values)
        for observed_at, values in by_time.items()
        if len({str(row.get("instrument_id") or "") for row in values}) >= 2
    ]
    if not candidates:
        return None, []
    observed_at, values = max(candidates, key=lambda item: item[0])
    return observed_at, values


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
