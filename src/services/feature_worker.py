"""Shared live and historical deterministic feature worker."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

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
            raw_inputs = payload.get("inputs")
            if not isinstance(raw_inputs, Mapping):
                raise ValueError("feature job inputs must be an object")
            calculator = DeterministicFeatureCalculator(
                version=str(payload["feature_set_version"]),
                function=core_bar_features,
            )
            values = calculator.calculate(
                instrument_id=str(payload["instrument_id"]),
                source_event_time=str(payload["source_event_time"]),
                source_close_time=str(payload["source_close_time"]),
                availability_time=str(payload["availability_time"]),
                inputs={str(key): float(value) for key, value in raw_inputs.items()},
            )
            assignments = tuple(self.active_assignments(str(payload["instrument_id"])))
            required_by_assignment: dict[str, tuple[str, ...]] = {}
            graph_values: dict[str, float] = {}
            if self.feature_graph_for_assignment is not None:
                for assignment in assignments:
                    declaration = self.feature_graph_for_assignment(assignment)
                    raw_required = declaration.get("required_nodes", declaration.get("nodes", ()))
                    required = tuple(
                        str(item["name"] if isinstance(item, Mapping) else item)
                        for item in raw_required
                    )
                    required_by_assignment[str(assignment["id"])] = required
                union = tuple(
                    dict.fromkeys(
                        name for names in required_by_assignment.values() for name in names
                    )
                )
                if union:
                    graph = self.graph_registry.graph(
                        union, version=str(payload["feature_set_version"])
                    )
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
                    graph_values = {
                        name: float(value)
                        for name, value in calculated.items()
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    }
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
            market_data_snapshot_id = None
            if self.snapshot_store is not None:
                market_data_snapshot_id = self.snapshot_store.save(
                    {
                        "kind": "market_data_input",
                        "event_id": str(payload["source_market_event_id"]),
                        "instrument_id": str(payload["instrument_id"]),
                        "source_event_time": str(payload["source_event_time"]),
                        "availability_time": str(payload["availability_time"]),
                        "values": {str(key): float(value) for key, value in raw_inputs.items()},
                    },
                    created_at=str(payload["availability_time"]),
                )
            parquet_path = None
            if self.parquet_store is not None:
                parquet_path = self.parquet_store.put(
                    values,
                    venue=str(payload["venue"]),
                    market=str(payload["market"]),
                    symbol=str(payload["symbol"]),
                    timeframe=str(payload["timeframe"]),
                )
            evaluation_jobs = []
            for assignment in assignments:
                product_id = str(assignment["product_id"])
                required = required_by_assignment.get(str(assignment["id"]), ())
                assignment_feature_ids = (
                    [identity_by_name[name] for name in required if name in identity_by_name]
                    if required
                    else list(identities)
                )
                if required and len(assignment_feature_ids) != len(required):
                    raise ValueError("artefact feature graph did not produce every required node")
                evaluation_payload = {
                    "event_id": str(payload["source_market_event_id"]),
                    "product_id": product_id,
                    "instrument_id": str(payload["instrument_id"]),
                    "assignment_id": str(assignment["id"]),
                    "feature_ids": assignment_feature_ids,
                    "feature_set_version": str(payload["feature_set_version"]),
                    **(
                        {"market_data_snapshot_id": market_data_snapshot_id}
                        if market_data_snapshot_id is not None
                        else {}
                    ),
                    "evaluated_at": str(payload["availability_time"]),
                    "horizon_seconds": int(payload.get("horizon_seconds", 60)),
                    "producer_identity": self.worker_id,
                }
                evaluation_payload["content_hash"] = build_content_hash(evaluation_payload)
                evaluation_job_id = f"strategy-evaluation:{canonical_hash(evaluation_payload).removeprefix('sha256:')}"
                self.queue.enqueue_if_absent(
                    job_id=evaluation_job_id,
                    name="strategy_evaluation",
                    payload=evaluation_payload,
                    available_at=str(payload["availability_time"]),
                    priority=12,
                    producer_identity=self.worker_id,
                )
                evaluation_jobs.append(evaluation_job_id)
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


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
