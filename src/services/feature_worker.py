"""Shared live and historical deterministic feature worker."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.data.feature_store import DeterministicFeatureCalculator, SqlFeatureStore
from src.data.parquet_store import PartitionedFeatureStore
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
            identities = self.store.save(values)
            parquet_path = None
            if self.parquet_store is not None:
                parquet_path = self.parquet_store.put(
                    values,
                    venue=str(payload["venue"]),
                    market=str(payload["market"]),
                    symbol=str(payload["symbol"]),
                    timeframe=str(payload["timeframe"]),
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
        }


def _retry_at(value: str, seconds: int) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()
