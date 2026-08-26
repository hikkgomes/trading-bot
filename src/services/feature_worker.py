"""Shared live and historical deterministic feature worker."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable, Mapping
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
            if claimed.name == "live_feature_calculation" and not isinstance(
                payload.get("input_references"), Mapping
            ):
                raise ValueError("live feature jobs require immutable input references")
            raw_inputs = payload.get("inputs")
            if payload.get("input_references") is not None:
                raw_inputs = self._resolve_input_references(payload)
            if not isinstance(raw_inputs, Mapping):
                raise ValueError(
                    "feature job requires immutable input references or an input object"
                )
            scalar_inputs = {
                str(key): float(value)
                for key, value in raw_inputs.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            calculator = DeterministicFeatureCalculator(
                version=str(payload["feature_set_version"]),
                function=core_bar_features,
            )
            values = calculator.calculate(
                instrument_id=str(payload["instrument_id"]),
                source_event_time=str(payload["source_event_time"]),
                source_close_time=str(payload["source_close_time"]),
                availability_time=str(payload["availability_time"]),
                inputs=scalar_inputs,
            )
            assignments = tuple(self.active_assignments(str(payload["instrument_id"])))
            required_by_assignment: dict[str, tuple[str, ...]] = {}
            graph_values: dict[str, float] = {}
            graph_output_features: dict[str, tuple[str, ...]] = {}
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
                    for name, value in calculated.items():
                        if isinstance(value, int | float) and not isinstance(value, bool):
                            graph_values[name] = float(value)
                            continue
                        if not isinstance(value, Mapping) or not value:
                            raise ValueError(f"feature node {name} returned no scalar values")
                        expanded: dict[str, float] = {}
                        for feature_name, feature_value in value.items():
                            if (
                                isinstance(feature_value, bool)
                                or not isinstance(feature_value, int | float)
                                or not math.isfinite(float(feature_value))
                            ):
                                raise ValueError(
                                    f"feature node {name} returned an invalid component"
                                )
                            expanded[str(feature_name)] = float(feature_value)
                        if not expanded:
                            raise ValueError(f"feature node {name} returned no scalar components")
                        graph_values.update(expanded)
                        graph_output_features[name] = tuple(sorted(expanded))
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
                market_data_snapshot_id = None
                if self.snapshot_store is not None:
                    market_data_snapshot_id = self.snapshot_store.save(
                        {
                            "kind": "market_data_input",
                            "product_id": product_id,
                            "event_id": str(payload["source_market_event_id"]),
                            "instrument_id": str(payload["instrument_id"]),
                            "source_event_time": str(payload["source_event_time"]),
                            "availability_time": str(payload["availability_time"]),
                            "values": scalar_inputs,
                        },
                        created_at=str(payload["availability_time"]),
                    )
                required = required_by_assignment.get(str(assignment["id"]), ())
                if required:
                    required_feature_names: list[str] = []
                    for name in required:
                        if name in identity_by_name:
                            required_feature_names.append(name)
                        elif name in graph_output_features:
                            required_feature_names.extend(graph_output_features[name])
                        else:
                            raise ValueError(
                                "artefact feature graph did not produce every required node"
                            )
                    assignment_feature_ids = [
                        identity_by_name[name] for name in required_feature_names
                    ]
                else:
                    assignment_feature_ids = list(identities)
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
                input_references = payload.get("input_references")
                if isinstance(input_references, Mapping):
                    reference_hashes = {
                        str(name): str(reference["content_hash"])
                        for name, reference in input_references.items()
                        if isinstance(reference, Mapping) and reference.get("content_hash")
                    }
                    if reference_hashes:
                        evaluation_payload["input_reference_id"] = canonical_hash(reference_hashes)
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

    def _resolve_input_references(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        references = payload.get("input_references")
        if not isinstance(references, Mapping):
            raise ValueError("feature input_references must be an object")
        for name, reference in references.items():
            if not isinstance(reference, Mapping):
                raise ValueError(f"feature input reference {name} must be an object")
            self._verify_input_reference(name=str(name), reference=reference)
        bar = references.get("bar_window")
        if not isinstance(bar, Mapping):
            raise ValueError("feature job requires a bar_window input reference")
        pattern = str(bar.get("relative_pattern") or "")
        if not pattern.startswith("bars/") or ".." in Path(pattern).parts:
            raise ValueError("feature bar reference has an unsafe partition")
        if self.parquet_store is None:
            raise ValueError("feature bar references require a parquet root")
        through = dt.datetime.fromisoformat(str(bar["through_close_time"]))
        available_at = dt.datetime.fromisoformat(str(payload["availability_time"]))
        raw_source_ids = bar.get("source_event_ids", ())
        if not isinstance(raw_source_ids, list | tuple):
            raise ValueError("feature bar reference source_event_ids must be a list")
        source_event_ids = {str(value) for value in raw_source_ids}
        bar_rows: dict[str, dict[str, Any]] = {}
        for path in self.parquet_store.root.glob(pattern):
            if path.is_symlink() or not path.is_file():
                continue
            for row in pq.read_table(path).to_pylist():
                if str(row.get("instrument_id")) != str(payload["instrument_id"]):
                    continue
                close_time = dt.datetime.fromtimestamp(float(row["close_time_ms"]) / 1_000, dt.UTC)
                availability = dt.datetime.fromisoformat(str(row["availability_time"]))
                if close_time <= through and availability <= available_at:
                    bar_rows[str(row["event_id"])] = row
        ordered = sorted(
            bar_rows.values(), key=lambda row: (int(row["close_time_ms"]), str(row["event_id"]))
        )
        minimum_history = int(bar.get("minimum_history", 1))
        if len(ordered) < minimum_history:
            raise ValueError("feature input reference does not contain the required bar history")
        target_rows = (
            [row for row in ordered if str(row.get("event_id")) in source_event_ids]
            if source_event_ids
            else ordered
        )
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
        resolved = {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
            "open_history": [float(row["open"]) for row in ordered],
            "high_history": [float(row["high"]) for row in ordered],
            "low_history": [float(row["low"]) for row in ordered],
            "close_history": [float(row["close"]) for row in ordered],
            "volume_history": [float(row["volume"]) for row in ordered],
            **{
                name: float(latest[name])
                for name in ("spread_bps", "visible_depth", "volatility", "funding")
                if latest.get(name) is not None
            },
            **(
                {"funding_rate": float(latest["funding"])}
                if latest.get("funding") is not None
                else {}
            ),
        }
        for name, reference in references.items():
            if name == "bar_window" or not isinstance(reference, Mapping):
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
        return resolved

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
        raw_event_types = reference.get("event_types", ())
        if not isinstance(raw_event_types, list | tuple):
            raise ValueError("feature auxiliary reference event_types must be a list")
        event_types = {str(item) for item in raw_event_types}
        raw_ids = reference.get("source_event_ids", ())
        if not isinstance(raw_ids, list | tuple):
            raise ValueError("feature auxiliary reference source_event_ids must be a list")
        source_ids = {str(item) for item in raw_ids}
        rows: list[dict[str, Any]] = []
        assert self.parquet_store is not None
        for path in self.parquet_store.root.glob(pattern):
            if path.is_symlink() or not path.is_file():
                continue
            for row in pq.read_table(path).to_pylist():
                if source_ids and str(row.get("event_id")) not in source_ids:
                    continue
                if same_instrument and str(row.get("instrument_id")) != instrument_id:
                    continue
                if event_types and str(row.get("event_type")) not in event_types:
                    continue
                try:
                    row_availability = dt.datetime.fromisoformat(str(row["availability_time"]))
                    row_exchange = dt.datetime.fromisoformat(str(row["exchange_timestamp"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("feature auxiliary row has invalid timestamps") from exc
                if row_availability > available_at or row_exchange > through:
                    continue
                payload_json = row.get("payload_json")
                if not isinstance(payload_json, str):
                    raise ValueError("feature auxiliary row has no immutable payload")
                try:
                    decoded = json.loads(payload_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("feature auxiliary row payload is not valid JSON") from exc
                rows.append(
                    {
                        "event_id": str(row.get("event_id")),
                        "instrument_id": str(row.get("instrument_id")),
                        "event_type": str(row.get("event_type")),
                        "exchange_timestamp": row_exchange,
                        "availability_time": row_availability,
                        "payload": decoded,
                    }
                )
        return sorted(
            rows,
            key=lambda row: (
                row["exchange_timestamp"],
                row["availability_time"],
                row["event_id"],
            ),
        )


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
    if name == "higher_timeframe":
        closes = [
            close
            for row in rows
            if (close := _number(_payload_views(row["payload"]), "c", "close")) is not None
        ]
        if len(closes) >= 2 and closes[-2] > 0:
            resolved["higher_timeframe_return"] = closes[-1] / closes[-2] - 1.0
        return
    if name == "order_book":
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
        return
    if name == "trade_flow":
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
                if maker:
                    sell += quantity
                else:
                    buy += quantity
        if buy or sell:
            resolved.update(
                {
                    "taker_buy_volume": buy,
                    "aggressor_buy_volume": buy,
                    "aggressor_sell_volume": sell,
                }
            )
        return
    if name == "funding_open_interest":
        funding = _number(views, "funding_rate", "fundingRate", "r", "funding")
        open_interest = _number(views, "open_interest", "openInterest", "oi")
        if funding is not None:
            resolved["funding_rate"] = funding
        if open_interest is not None:
            resolved["open_interest"] = open_interest
        return
    if name == "spot_perpetual":
        prices: dict[str, float] = {}
        for row in rows:
            price = _number(_payload_views(row["payload"]), "close", "c", "price")
            if price is not None and price > 0:
                prices[str(row.get("instrument_id"))] = price
        current_market = "futures" if ":futures:" in instrument_id else "spot"
        other_market = "spot" if current_market == "futures" else "futures"
        current_price = prices.get(instrument_id)
        other_price = next(
            (price for identity, price in prices.items() if f":{other_market}:" in identity),
            None,
        )
        if current_price is not None:
            resolved["perpetual_price" if current_market == "futures" else "spot_price"] = (
                current_price
            )
        if other_price is not None:
            resolved["spot_price" if current_market == "futures" else "perpetual_price"] = (
                other_price
            )
        return
    if name == "cross_sectional":
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
            if instrument_id in panel:
                panel["__current__"] = panel[instrument_id]
            resolved["cross_sectional_values"] = panel
            return
        closes_by_instrument: dict[str, list[float]] = {}
        for row in rows:
            close = _number(_payload_views(row["payload"]), "c", "close")
            if close is not None:
                closes_by_instrument.setdefault(str(row.get("instrument_id")), []).append(close)
        panel = {
            instrument: closes[-1] / closes[-2] - 1.0
            for instrument, closes in closes_by_instrument.items()
            if len(closes) >= 2 and closes[-2] > 0
        }
        if panel:
            if instrument_id in panel:
                panel["__current__"] = panel[instrument_id]
            resolved["cross_sectional_values"] = panel
        return
    if name == "correlation_beta":
        correlation_closes_by_instrument: dict[str, list[float]] = {}
        for row in rows:
            close = _number(_payload_views(row["payload"]), "c", "close")
            if close is not None and close > 0:
                correlation_closes_by_instrument.setdefault(
                    str(row.get("instrument_id")), []
                ).append(close)
        returns_by_instrument = {
            identity: [
                closes[index] / closes[index - 1] - 1.0
                for index in range(1, len(closes))
                if closes[index - 1] > 0
            ]
            for identity, closes in correlation_closes_by_instrument.items()
        }
        asset_returns = returns_by_instrument.get(instrument_id, [])
        benchmark_returns = next(
            (
                values
                for identity, values in returns_by_instrument.items()
                if identity != instrument_id and values
            ),
            [],
        )
        if asset_returns and benchmark_returns:
            resolved["asset_returns"] = asset_returns
            resolved["benchmark_returns"] = benchmark_returns
        return
    if name == "sentiment":
        value = _number(views, "sentiment_score", "score")
        if value is not None:
            resolved["sentiment_score"] = value
        return
    if name == "ml_manifest":
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
        return
    if name == "liquidation":
        liquid_buy: float | None = _number(views, "liquidation_buy_volume")
        liquid_sell: float | None = _number(views, "liquidation_sell_volume")
        if liquid_buy is not None:
            resolved["liquidation_buy_volume"] = liquid_buy
        if liquid_sell is not None:
            resolved["liquidation_sell_volume"] = liquid_sell
