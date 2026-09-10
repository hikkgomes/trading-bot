"""Build canonical research bundles from immutable Parquet bars."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import or_, select

from src.data.bar_window import RecentBarWindow
from src.data.database import (
    cost_model_manifest,
    feature_manifest,
    funding_entry,
    historical_bootstrap_progress,
    universe_member,
    universe_snapshot,
)
from src.data.universe import SqlUniverseStore
from src.domain._codec import canonical_hash, timeframe_seconds, timestamp
from src.products.btc_accumulation import BTC_SPOT_INSTRUMENT_ID
from src.research.data_plan import ResearchDataPlan
from src.research.datasets import (
    CORE_RESEARCH_BUNDLE_ROLES,
    CanonicalResearchDatasetBuilder,
    DatasetLifecycleState,
    DatasetResolutionError,
    _prior_bars,
)


class DatasetBundleBuildError(RuntimeError):
    """A source partition cannot be used for a canonical research bundle."""


class FundingCoverageUnavailable(DatasetBundleBuildError):
    """A complete native settlement query is still required."""


@dataclass(frozen=True)
class DatasetBundleBuildResult:
    state: str
    reason_code: str
    product_id: str
    bundle_id: str | None = None
    snapshot_ids: tuple[str, ...] = ()
    instrument_scope: tuple[str, ...] = ()
    source_partition_hashes: tuple[str, ...] = ()
    detail: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "product_id": self.product_id,
            "bundle_id": self.bundle_id,
            "snapshot_ids": list(self.snapshot_ids),
            "instrument_scope": list(self.instrument_scope),
            "source_partition_hashes": list(self.source_partition_hashes),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class _ResolvedUniverse:
    latest_snapshot_id: str
    instruments: tuple[tuple[str, str, str], ...]
    memberships: tuple[tuple[str, frozenset[str]], ...]
    snapshot_ids: tuple[str, ...]
    selection_mode: str = "point_in_time"


class DatabaseDatasetBundleService:
    """Resolve point-in-time inputs and publish one immutable stage bundle."""

    def __init__(
        self,
        engine,
        parquet_root: Path,
        *,
        maximum_rows: int = 1_000_000,
        timeframe: str = "1m",
        minimum_history_days: int = 0,
        warmup_bars: int = 0,
        maximum_bytes: int = 2_147_483_648,
        estimated_bytes_per_row: int = 2048,
        batch_rows: int = 4096,
    ) -> None:
        if maximum_rows <= 0:
            raise ValueError("maximum_rows must be positive")
        if minimum_history_days < 0:
            raise ValueError("minimum_history_days must be non-negative")
        self.engine = engine
        self.parquet_root = parquet_root.resolve()
        self.maximum_rows = maximum_rows
        self.timeframe = _text(timeframe, field="timeframe")
        self.minimum_history_days = minimum_history_days
        self.plan_limits = {
            "warmup_bars": warmup_bars,
            "maximum_bytes": maximum_bytes,
            "estimated_bytes_per_row": estimated_bytes_per_row,
            "batch_rows": batch_rows,
        }

    def run(
        self,
        *,
        product_id: str,
        universe_id: str,
        market_type: str,
        created_at: str,
        timeframe: str | None = None,
        intervals: Mapping[str, Mapping[str, str]] | None = None,
        data_plan: ResearchDataPlan | None = None,
    ) -> DatasetBundleBuildResult:
        created = timestamp(created_at, field="created_at")
        resolved_universe = self._universe(
            universe_id, created, product_id=product_id, include_history=True
        )
        if resolved_universe is None:
            return self._waiting(product_id, "point_in_time_universe_unavailable")
        universe_snapshot_id = resolved_universe.latest_snapshot_id
        instruments = resolved_universe.instruments
        plan = self._validated_plan(data_plan, product_id, instruments, timeframe or self.timeframe)
        feature_id = self._latest_manifest(
            feature_manifest,
            created=created,
            matches=lambda payload: _matches_feature(payload, market_type),
        )
        cost_id = self._latest_manifest(
            cost_model_manifest,
            created=created,
            matches=lambda payload: str(payload.get("product_id") or "") == product_id,
        )
        if feature_id is None or cost_id is None:
            return self._waiting(product_id, "dataset_manifests_unavailable")
        instrument_scope = tuple(item[0] for item in instruments)
        prepared = self._prepare_historical_bars(
            instruments,
            market_type=market_type,
            timeframe=timeframe or self.timeframe,
            created=created,
            product_id=product_id,
            data_plan=plan,
        )
        if isinstance(prepared, DatasetBundleBuildResult):
            return prepared
        bars, source_hashes = prepared
        try:
            if market_type.lower() == "futures":
                bars = self._attach_funding_rates(
                    product_id=product_id,
                    bars=bars,
                    instrument_scope=instrument_scope,
                    created=created,
                )
            _validate_market_bars(
                bars,
                market_type=market_type,
                timeframe=timeframe or self.timeframe,
            )
        except FundingCoverageUnavailable as exc:
            return self._waiting(product_id, "funding_history_coverage_incomplete", detail=str(exc))
        except DatasetBundleBuildError as exc:
            if str(exc).startswith("historical bars are missing"):
                return self._waiting(product_id, "historical_bars_incomplete")
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="bar_quality_invalid",
                product_id=product_id,
                detail=str(exc),
            )
        try:
            intervals = (
                _stage_intervals(bars, warmup_bars=plan.warmup_bars)
                if intervals is None
                else intervals
            )
            payload_by_role = self._role_payloads(
                bars,
                intervals=intervals,
                memberships=resolved_universe.memberships,
                snapshot_ids=resolved_universe.snapshot_ids,
                timeframe=timeframe or self.timeframe,
                selection_mode=resolved_universe.selection_mode,
                warmup_bars=plan.warmup_bars,
                require_complete_warmup=True,
            )
            for payload in payload_by_role.values():
                payload["data_plan"] = {**plan.to_payload(), "intervals": intervals}
                payload["data_plan_id"] = plan.plan_id
            bundle = CanonicalResearchDatasetBuilder(self.engine).build(
                product_id,
                intervals=intervals,
                payload_by_role=payload_by_role,
                universe_snapshot_id=universe_snapshot_id,
                feature_manifest_id=feature_id,
                cost_model_id=cost_id,
                parameter_set_id=_parameter_set_id(product_id),
                instrument_scope=instrument_scope,
                created_at=created,
                availability_timestamp=created,
                source_partition_hashes=source_hashes,
                engine_version="dataset-service/v2",
            )
        except DatasetResolutionError as exc:
            return self._waiting(product_id, _reason_code(str(exc)), detail=str(exc))
        return DatasetBundleBuildResult(
            state="ready",
            reason_code="canonical_dataset_bundle_ready",
            product_id=product_id,
            bundle_id=bundle.bundle_id,
            snapshot_ids=tuple(bundle.stage_snapshot_ids.values()),
            instrument_scope=instrument_scope,
            source_partition_hashes=source_hashes,
        )

    def run_forward(
        self,
        *,
        product_id: str,
        universe_id: str,
        market_type: str,
        artefact_created_at: str,
        created_at: str,
        timeframe: str | None = None,
    ) -> DatasetBundleBuildResult:
        """Publish one immutable forward-observation snapshot after an artefact."""

        created = timestamp(created_at, field="created_at")
        artefact_created = timestamp(artefact_created_at, field="artefact_created_at")
        if artefact_created >= created:
            return self._waiting(product_id, "forward_data_not_available_after_artefact")
        resolved_universe = self._universe(
            universe_id, created, product_id=product_id, include_history=False
        )
        if resolved_universe is None:
            return self._waiting(product_id, "point_in_time_universe_unavailable")
        universe_snapshot_id = resolved_universe.latest_snapshot_id
        instruments = resolved_universe.instruments
        feature_id = self._latest_manifest(
            feature_manifest,
            created=created,
            matches=lambda payload: _matches_feature(payload, market_type),
        )
        cost_id = self._latest_manifest(
            cost_model_manifest,
            created=created,
            matches=lambda payload: str(payload.get("product_id") or "") == product_id,
        )
        if feature_id is None or cost_id is None:
            return self._waiting(product_id, "dataset_manifests_unavailable")
        scope = tuple(item[0] for item in instruments)
        try:
            bars, source_hashes = self._load_bars(
                instruments,
                market_type=market_type,
                timeframe=timeframe or self.timeframe,
                created=created,
            )
            bars = tuple(
                row
                for row in bars
                if str(row["close_timestamp"]) > artefact_created
                and str(row["availability_time"]) > artefact_created
            )
            if not bars:
                return self._waiting(product_id, "forward_bars_unavailable")
            if market_type.lower() == "futures":
                bars = self._attach_funding_rates(
                    product_id=product_id,
                    bars=bars,
                    instrument_scope=scope,
                    created=created,
                )
            _validate_market_bars(
                bars,
                market_type=market_type,
                timeframe=timeframe or self.timeframe,
            )
            interval = _single_interval(bars)
            payload = CanonicalResearchDatasetBuilder._bar_payload(
                list(bars), timeframe=timeframe or self.timeframe
            )
            bundle = CanonicalResearchDatasetBuilder(self.engine).build(
                product_id,
                intervals={"forward_observation": interval},
                payload_by_role={"forward_observation": payload},
                universe_snapshot_id=universe_snapshot_id,
                feature_manifest_id=feature_id,
                cost_model_id=cost_id,
                parameter_set_id=_parameter_set_id(product_id),
                instrument_scope=scope,
                availability_timestamp={"forward_observation": max(created, interval["end"])},
                created_at=created,
                engine_version="dataset-service/forward-v1",
                source_partition_hashes=source_hashes,
                lifecycle_state=DatasetLifecycleState.DATA_PENDING,
            )
        except FundingCoverageUnavailable as exc:
            return self._waiting(product_id, "funding_history_coverage_incomplete", detail=str(exc))
        except DatasetBundleBuildError as exc:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="bar_quality_invalid",
                product_id=product_id,
                detail=str(exc),
            )
        except DatasetResolutionError as exc:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code=_reason_code(str(exc)),
                product_id=product_id,
                detail=str(exc),
            )
        return DatasetBundleBuildResult(
            state="ready",
            reason_code="forward_dataset_ready",
            product_id=product_id,
            bundle_id=bundle.bundle_id,
            snapshot_ids=tuple(bundle.stage_snapshot_ids.values()),
            instrument_scope=scope,
            source_partition_hashes=source_hashes,
        )

    @staticmethod
    def _waiting(
        product_id: str, reason_code: str, *, detail: str | None = None
    ) -> DatasetBundleBuildResult:
        return DatasetBundleBuildResult(
            state="waiting_for_dataset",
            reason_code=reason_code,
            product_id=product_id,
            detail=detail,
        )

    def _prepare_historical_bars(
        self,
        instruments: tuple[tuple[str, str, str], ...],
        *,
        market_type: str,
        timeframe: str,
        created: str,
        product_id: str,
        data_plan: ResearchDataPlan | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]] | DatasetBundleBuildResult:
        try:
            interval_seconds = timeframe_seconds(timeframe)
        except ValueError as exc:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="unsupported_timeframe",
                product_id=product_id,
                detail=str(exc),
            )
        minimum_days = data_plan.minimum_history_days if data_plan else self.minimum_history_days
        minimum_rows = ((minimum_days * 86_400) // interval_seconds) + 1 if minimum_days else 0
        limit = data_plan.maximum_rows if data_plan else self.maximum_rows
        if minimum_rows * len(instruments) > limit:
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="history_exceeds_resource_budget",
                product_id=product_id,
                detail="Select an explicit family timeframe or increase the approved row budget.",
            )
        try:
            bars, source_hashes = self._load_bars(
                instruments,
                market_type=market_type,
                timeframe=timeframe,
                created=created,
                maximum_rows=data_plan.required_rows if data_plan and minimum_days else limit,
                batch_rows=data_plan.batch_rows if data_plan else 4096,
                range_start=(
                    data_plan.acquisition_interval(created)[0]
                    if data_plan and minimum_days
                    else None
                ),
            )
        except DatasetBundleBuildError as exc:
            if str(exc).startswith("historical bars are missing"):
                return self._waiting(product_id, "historical_bars_incomplete")
            return DatasetBundleBuildResult(
                state="blocked_dataset",
                reason_code="bar_partition_invalid",
                product_id=product_id,
                detail=str(exc),
            )
        if not bars:
            return self._waiting(product_id, "historical_bars_unavailable")
        if not _history_span_satisfies(bars, minimum_days):
            return self._waiting(product_id, "historical_history_insufficient")
        return bars, source_hashes

    def _universe(
        self,
        universe_id: str,
        observed_at: str,
        *,
        product_id: str,
        include_history: bool,
    ) -> _ResolvedUniverse | None:
        memberships = SqlUniverseStore(self.engine).members_at(
            universe_id=universe_id,
            observed_at=observed_at,
            eligible_only=True,
        )
        if not memberships:
            return None
        current_instruments = {
            item.instrument.instrument_id: (
                item.instrument.instrument_id,
                item.instrument.exchange_symbol,
                item.instrument.venue,
            )
            for item in memberships
            if product_id != "btc_accumulation"
            or item.instrument.instrument_id == BTC_SPOT_INSTRUMENT_ID
        }
        if not current_instruments:
            return None
        with self.engine.connect() as connection:
            selection = connection.execute(
                select(universe_snapshot.c.payload).where(
                    universe_snapshot.c.id == memberships[0].snapshot_id
                )
            ).scalar_one()
        if selection.get("historical_liquidity_eligibility") == "not_asserted":
            if selection.get("product_id") != product_id:
                raise DatasetBundleBuildError("fixed research scope belongs to another product")
            return _ResolvedUniverse(
                latest_snapshot_id=memberships[0].snapshot_id,
                instruments=tuple(sorted(current_instruments.values())),
                memberships=(),
                snapshot_ids=(memberships[0].snapshot_id,),
                selection_mode="fixed_research_scope",
            )
        if not include_history:
            return _ResolvedUniverse(
                latest_snapshot_id=memberships[0].snapshot_id,
                instruments=tuple(sorted(current_instruments.values())),
                memberships=(
                    (
                        timestamp(observed_at, field="universe observed_at"),
                        frozenset(current_instruments),
                    ),
                ),
                snapshot_ids=(memberships[0].snapshot_id,),
            )
        history, historical_instruments, snapshot_ids = self._universe_history(
            universe_id=universe_id,
            observed_at=observed_at,
            product_id=product_id,
        )
        current_instruments.update(historical_instruments)
        return _ResolvedUniverse(
            latest_snapshot_id=memberships[0].snapshot_id,
            instruments=tuple(sorted(current_instruments.values())),
            memberships=history,
            snapshot_ids=snapshot_ids,
        )

    def _universe_history(
        self, *, universe_id: str, observed_at: str, product_id: str
    ) -> tuple[
        tuple[tuple[str, frozenset[str]], ...],
        dict[str, tuple[str, str, str]],
        tuple[str, ...],
    ]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    universe_snapshot.c.id,
                    universe_snapshot.c.observed_at,
                    universe_member.c.instrument_id,
                    universe_member.c.eligible,
                    universe_member.c.payload,
                )
                .select_from(
                    universe_snapshot.join(
                        universe_member,
                        universe_snapshot.c.id == universe_member.c.snapshot_id,
                    )
                )
                .where(
                    universe_snapshot.c.universe_id == universe_id,
                    universe_snapshot.c.observed_at <= observed_at,
                )
                .order_by(
                    universe_snapshot.c.observed_at,
                    universe_snapshot.c.id,
                    universe_member.c.instrument_id,
                )
            ).mappings()
        memberships: dict[str, set[str]] = {}
        observed_at_by_snapshot: dict[str, str] = {}
        instruments: dict[str, tuple[str, str, str]] = {}
        for row in rows:
            instrument_id = str(row["instrument_id"])
            if product_id == "btc_accumulation" and instrument_id != BTC_SPOT_INSTRUMENT_ID:
                continue
            snapshot_id = str(row["id"])
            observed = timestamp(str(row["observed_at"]), field="universe observed_at")
            observed_at_by_snapshot[snapshot_id] = observed
            raw_payload = row["payload"]
            instrument_payload = (
                raw_payload.get("instrument") if isinstance(raw_payload, Mapping) else None
            )
            if isinstance(instrument_payload, Mapping):
                symbol = str(instrument_payload.get("exchange_symbol") or "")
                venue = str(instrument_payload.get("venue") or "")
                if symbol and venue:
                    instruments[instrument_id] = (instrument_id, symbol, venue)
            if row["eligible"] is True:
                memberships.setdefault(snapshot_id, set()).add(instrument_id)
        history = tuple(
            (
                observed_at_by_snapshot[snapshot_id],
                frozenset(memberships.get(snapshot_id, set())),
            )
            for snapshot_id in sorted(
                observed_at_by_snapshot,
                key=lambda value: (observed_at_by_snapshot[value], value),
            )
        )
        return (
            history,
            instruments,
            tuple(
                snapshot_id
                for snapshot_id, _observed in sorted(
                    observed_at_by_snapshot.items(), key=lambda item: (item[1], item[0])
                )
            ),
        )

    @staticmethod
    def _role_payloads(
        bars: tuple[dict[str, Any], ...],
        *,
        intervals: Mapping[str, Mapping[str, str]],
        memberships: tuple[tuple[str, frozenset[str]], ...],
        snapshot_ids: tuple[str, ...],
        timeframe: str | None = None,
        selection_mode: str = "point_in_time",
        warmup_bars: int = 200,
        require_complete_warmup: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if selection_mode not in {"point_in_time", "fixed_research_scope"}:
            raise DatasetBundleBuildError("unknown research universe selection mode")
        payloads: dict[str, dict[str, Any]] = {}
        for role, interval in intervals.items():
            selected = [
                row
                for row in bars
                if interval["start"] <= str(row["close_timestamp"]) < interval["end"]
                and (
                    selection_mode == "fixed_research_scope"
                    or DatabaseDatasetBundleService._eligible_at(
                        str(row["instrument_id"]), str(row["close_timestamp"]), memberships
                    )
                )
            ]
            if not selected:
                raise DatasetResolutionError(
                    f"dataset data_pending: no available bars for role {role}"
                )
            warmup = _prior_bars(
                bars,
                before=interval["start"],
                scope={str(row["instrument_id"]) for row in selected},
                limit=warmup_bars,
                created=max(str(row["availability_time"]) for row in bars),
            )
            if require_complete_warmup:
                _require_warmup(
                    warmup, {str(row["instrument_id"]) for row in selected}, warmup_bars
                )
            payload = CanonicalResearchDatasetBuilder._bar_payload(
                selected, timeframe=timeframe, warmup=warmup
            )
            payload["point_in_time_universe"] = selection_mode == "point_in_time"
            payload["universe_selection_mode"] = selection_mode
            payload["universe_snapshot_ids"] = list(snapshot_ids)
            payloads[role] = payload
        return payloads

    @staticmethod
    def _eligible_at(
        instrument_id: str,
        observed_at: str,
        memberships: tuple[tuple[str, frozenset[str]], ...],
    ) -> bool:
        eligible: frozenset[str] = frozenset()
        for snapshot_at, snapshot_members in memberships:
            if snapshot_at > observed_at:
                break
            eligible = snapshot_members
        return instrument_id in eligible

    def _latest_manifest(
        self,
        table: Any,
        *,
        created: str,
        matches: Any,
    ) -> str | None:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(table.c.id, table.c.payload)
                .where(table.c.created_at <= created)
                .order_by(table.c.created_at.desc(), table.c.id.desc())
            ).mappings()
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, Mapping) and matches(payload):
                    return str(row["id"])
        return None

    def _load_bars(
        self,
        instruments: tuple[tuple[str, str, str], ...],
        *,
        market_type: str,
        timeframe: str,
        created: str,
        maximum_rows: int | None = None,
        batch_rows: int = 4096,
        range_start: int | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        limit = maximum_rows if maximum_rows is not None else self.maximum_rows
        start = _bar_time(range_start, field="range_start") if range_start is not None else None
        windows = {
            key: RecentBarWindow(max(1, limit // len(instruments))) for key, _, _ in instruments
        }
        source_hashes: set[str] = set()
        for instrument_id, symbol, venue in instruments:
            root = self.parquet_root / "bars" / venue.lower() / market_type.lower() / symbol.upper()
            root = root / _text(timeframe, field="timeframe")
            for path in sorted(root.rglob("*.parquet")) if root.is_dir() else ():
                if path.is_symlink() or not path.is_file():
                    continue
                if start and any(
                    part.startswith("date=") and part[5:] < start[:10] for part in path.parts
                ):
                    continue
                try:
                    file_rows = (
                        row
                        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows)
                        for row in batch.to_pylist()
                        if start is None
                        or _bar_time(
                            row.get("close_time_ms", row.get("close_timestamp")), field="close"
                        )
                        >= start
                    )
                    added = _ingest_bar_rows(
                        file_rows, windows[instrument_id], instrument_id, created
                    )
                except Exception as exc:
                    raise DatasetBundleBuildError(f"cannot read bar partition {path}") from exc
                if added:
                    source_hashes.add(_file_identity(path))
        rows_by_instrument = {
            key: window.rows() for key, window in windows.items() if window.rows()
        }
        missing = sorted({item[0] for item in instruments} - set(rows_by_instrument))
        if missing:
            raise DatasetBundleBuildError(
                "historical bars are missing for instrument scope: " + ", ".join(missing)
            )
        rows = _bounded_instrument_rows(
            rows_by_instrument,
            maximum_rows=limit,
        )
        return tuple(rows), tuple(sorted(source_hashes))

    def _attach_funding_rates(
        self,
        *,
        product_id: str,
        bars: tuple[dict[str, Any], ...],
        instrument_scope: tuple[str, ...],
        created: str,
    ) -> tuple[dict[str, Any], ...]:
        try:
            funding = self._funding_events(product_id, created)
        except (ValueError, KeyError, TypeError) as exc:
            raise DatasetBundleBuildError(f"native funding data invalid: {exc}") from exc
        coverage = self._funding_coverage(product_id, bars, instrument_scope, created)
        from src.research.funding import attach_settlements

        return attach_settlements(
            tuple({**row, "funding_coverage": coverage[row["instrument_id"]]} for row in bars),
            {instrument: funding.get(instrument, {}) for instrument in instrument_scope},
        )

    def _validated_plan(self, data_plan, product_id, instruments, timeframe):
        scope = tuple(sorted(item[1] for item in instruments))
        plan = data_plan or ResearchDataPlan(
            product_id=product_id,
            timeframe=timeframe,
            instrument_scope=scope,
            minimum_history_days=self.minimum_history_days,
            maximum_rows=self.maximum_rows,
            **self.plan_limits,
        )
        if (plan.product_id, plan.instrument_scope, plan.timeframe) != (
            product_id,
            scope,
            timeframe,
        ):
            raise DatasetBundleBuildError(
                "research data plan conflicts with resolved product, scope or timeframe"
            )
        return plan

    def _funding_coverage(self, product_id, bars, instruments, created):
        from src.research.funding import _interval_start

        with self.engine.connect() as connection:
            records = (
                connection.execute(
                    select(historical_bootstrap_progress)
                    .where(
                        historical_bootstrap_progress.c.product_id == product_id,
                        historical_bootstrap_progress.c.market == "futures",
                        historical_bootstrap_progress.c.timeframe == "funding",
                        historical_bootstrap_progress.c.status == "completed",
                        historical_bootstrap_progress.c.updated_at <= created,
                    )
                    .order_by(historical_bootstrap_progress.c.range_start)
                )
                .mappings()
                .all()
            )
        result = {}
        for instrument_id in instruments:
            selected = [row for row in bars if row["instrument_id"] == instrument_id]
            start = min(_interval_start(row, None) for row in selected)
            end = max(row["close_timestamp"] for row in selected)
            cursor, proof = start, []
            for row in records:
                if row["symbol"] != instrument_id.split(":")[2] or row["payload"].get("gaps"):
                    continue
                if (
                    row["range_start"] <= cursor < row["range_end"]
                    and row["next_start"] >= row["range_end"]
                ):
                    cursor = row["range_end"]
                    proof.append(
                        {
                            key: row[key]
                            for key in ("id", "content_hash", "range_start", "range_end")
                        }
                    )
                if cursor >= end:
                    break
            if cursor < end:
                raise FundingCoverageUnavailable(
                    f"futures funding coverage is incomplete for: {instrument_id}"
                )
            result[instrument_id] = {"start": start, "end": end, "source_progress": proof}
        return result

    def _funding_events(
        self, product_id: str, created: str
    ) -> dict[str, dict[str, dict[str, Any]]]:
        from src.research.funding import settlement_index

        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(funding_entry.c.id, funding_entry.c.payload, funding_entry.c.created_at)
                    .where(
                        funding_entry.c.payload["product_id"].as_string() == product_id,
                        or_(
                            funding_entry.c.created_at <= created,
                            funding_entry.c.payload["supersedes_record_id"]
                            .as_string()
                            .is_not(None),
                        ),
                    )
                    .order_by(funding_entry.c.created_at, funding_entry.c.id)
                )
                .mappings()
                .all()
            )
        return settlement_index(rows, product_id=product_id, created=created)


def _ingest_bar_rows(rows, window: RecentBarWindow, instrument_id: str, created: str) -> bool:
    added = False
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        normalised = _normalise_bar(raw, instrument_id, created)
        if normalised is not None:
            window.add(normalised)
            added = True
    return added


def _normalise_bar(
    raw: Mapping[str, Any], instrument_id: str, created: str
) -> dict[str, Any] | None:
    close_value = raw.get("close_timestamp", raw.get("close_time_ms", raw.get("timestamp")))
    if close_value is None:
        return None
    available_value = raw.get("availability_time", raw.get("available_at", close_value))
    close = _bar_time(close_value, field="close_timestamp")
    available = _bar_time(available_value, field="availability_time")
    if close > created or available > created:
        return None
    row = dict(raw)
    row["instrument_id"] = instrument_id
    row["close_timestamp"] = close
    row["availability_time"] = available
    return row


def _history_span_satisfies(rows: tuple[dict[str, Any], ...], minimum_days: int) -> bool:
    if minimum_days == 0:
        return True
    by_instrument: dict[str, list[dt.datetime]] = {}
    for row in rows:
        by_instrument.setdefault(str(row["instrument_id"]), []).append(
            dt.datetime.fromisoformat(str(row["close_timestamp"]))
        )
    return bool(by_instrument) and all(
        max(times) - min(times) >= dt.timedelta(days=minimum_days)
        for times in by_instrument.values()
    )


def _bounded_instrument_rows(
    rows_by_instrument: Mapping[str, list[dict[str, Any]]], *, maximum_rows: int
) -> list[dict[str, Any]]:
    """Keep a deterministic, time-spanning sample for every instrument."""

    total = sum(len(rows) for rows in rows_by_instrument.values())
    if total <= maximum_rows:
        rows = [row for values in rows_by_instrument.values() for row in values]
        return sorted(
            rows, key=lambda row: (str(row["close_timestamp"]), str(row["instrument_id"]))
        )
    instrument_ids = tuple(sorted(rows_by_instrument))
    if len(instrument_ids) > maximum_rows:
        raise DatasetBundleBuildError("maximum_rows cannot represent every instrument in the scope")
    base, remainder = divmod(maximum_rows, len(instrument_ids))
    quotas = {
        instrument_id: min(len(rows_by_instrument[instrument_id]), base + (index < remainder))
        for index, instrument_id in enumerate(instrument_ids)
    }
    unused = maximum_rows - sum(quotas.values())
    while unused:
        expanded = False
        for instrument_id in instrument_ids:
            if quotas[instrument_id] < len(rows_by_instrument[instrument_id]):
                quotas[instrument_id] += 1
                unused -= 1
                expanded = True
                if not unused:
                    break
        if not expanded:
            break
    # A capped dataset must remain a real contiguous time series. Evenly
    # selecting endpoints creates false multi-minute or hourly candles while
    # retaining the original timeframe label. Prefer an aligned recent panel
    # window, then use a contiguous tail per instrument when no common panel
    # exists.
    common_times = set.intersection(
        *(
            {str(row["close_timestamp"]) for row in rows_by_instrument[instrument_id]}
            for instrument_id in instrument_ids
        )
    )
    aligned_limit = maximum_rows // len(instrument_ids)
    if common_times and aligned_limit > 0:
        selected_times = set(sorted(common_times)[-aligned_limit:])
        selected = [
            row
            for instrument_id in instrument_ids
            for row in rows_by_instrument[instrument_id]
            if str(row["close_timestamp"]) in selected_times
        ]
    else:
        selected = [
            row
            for instrument_id in instrument_ids
            for row in sorted(
                rows_by_instrument[instrument_id], key=lambda item: str(item["close_timestamp"])
            )[-quotas[instrument_id] :]
        ]
    return sorted(
        selected, key=lambda row: (str(row["close_timestamp"]), str(row["instrument_id"]))
    )


def _validate_market_bars(
    rows: tuple[dict[str, Any], ...], *, market_type: str, timeframe: str | None = None
) -> None:
    interval = timeframe_seconds(timeframe) if timeframe is not None else None
    previous: dict[str, str] = {}
    for index, row in enumerate(rows):
        instrument_id = str(row.get("instrument_id") or "")
        observed = str(row.get("close_timestamp") or "")
        if previous.get(instrument_id, "") >= observed:
            raise DatasetBundleBuildError(
                f"bar timestamps are not strictly increasing at row {index}"
            )
        previous_time = previous.get(instrument_id)
        _validate_bar_timing(row, timeframe, interval)
        if interval is not None and previous_time is not None:
            delta = dt.datetime.fromisoformat(observed) - dt.datetime.fromisoformat(previous_time)
            if delta.total_seconds() != interval:
                raise DatasetBundleBuildError(
                    f"bar timestamps do not follow the declared {timeframe} cadence"
                )
        previous[instrument_id] = observed
        values = {
            name: _positive_number(row.get(name), field=f"bar[{index}].{name}")
            for name in ("open", "high", "low", "close")
        }
        volume = _nonnegative_number(row.get("volume"), field=f"bar[{index}].volume")
        if values["high"] < max(values["open"], values["close"]) or values["low"] > min(
            values["open"], values["close"]
        ):
            raise DatasetBundleBuildError(f"bar[{index}] has inconsistent OHLC values")
        if volume < 0.0:
            raise DatasetBundleBuildError(f"bar[{index}].volume is negative")
        if market_type.lower() == "futures" and "funding_rate" not in row:
            raise DatasetBundleBuildError(f"bar[{index}] has no signed funding rate")


def _validate_bar_timing(
    row: Mapping[str, Any], timeframe: str | None, interval: int | None
) -> None:
    declared = row.get("timeframe")
    if timeframe is not None and declared is not None and declared != timeframe:
        raise DatasetBundleBuildError("bar timeframe conflicts with the declared data plan")
    if interval is not None and row.get("open_time_ms") is not None:
        opened = dt.datetime.fromtimestamp(float(row["open_time_ms"]) / 1000, dt.UTC)
        closed = dt.datetime.fromisoformat(str(row["close_timestamp"]))
        duration = (closed - opened).total_seconds()
        if not interval - 1 <= duration <= interval or float(row["open_time_ms"]) % (
            interval * 1000
        ):
            raise DatasetBundleBuildError("bar open/close times conflict with the declared cadence")


def _require_warmup(rows: tuple[dict[str, Any], ...], scope: set[str], count: int) -> None:
    from collections import Counter

    counts = Counter(str(row["instrument_id"]) for row in rows)
    if any(counts[instrument] < count for instrument in scope):
        raise DatasetResolutionError("dataset data_pending: incomplete feature warm-up")


def _positive_number(value: Any, *, field: str) -> float:
    result = _signed_number(value, field=field)
    if result <= 0.0:
        raise DatasetBundleBuildError(f"{field} must be positive")
    return result


def _nonnegative_number(value: Any, *, field: str) -> float:
    result = _signed_number(value, field=field)
    if result < 0.0:
        raise DatasetBundleBuildError(f"{field} must be non-negative")
    return result


def _signed_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise DatasetBundleBuildError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetBundleBuildError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise DatasetBundleBuildError(f"{field} must be finite")
    return result


def _bar_time(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise DatasetBundleBuildError(f"{field} must be a timestamp")
    if isinstance(value, int | float):
        try:
            parsed = dt.datetime.fromtimestamp(float(value) / 1_000, dt.UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise DatasetBundleBuildError(f"{field} is invalid") from exc
        return timestamp(parsed, field=field, preserve_subseconds=True)
    try:
        return timestamp(str(value), field=field, preserve_subseconds=True)
    except ValueError as exc:
        raise DatasetBundleBuildError(f"{field} is invalid") from exc


def _stage_intervals(
    rows: tuple[dict[str, Any], ...], *, warmup_bars: int = 0
) -> dict[str, dict[str, str]]:
    times = sorted({str(row["close_timestamp"]) for row in rows})[warmup_bars:]
    if len(times) < len(CORE_RESEARCH_BUNDLE_ROLES):
        raise DatasetResolutionError("dataset data_pending: insufficient distinct bar timestamps")
    indexes = (0, len(times) // 4, len(times) // 2, (len(times) * 3) // 4)
    starts = [times[index] for index in indexes]
    final_end = (
        (dt.datetime.fromisoformat(times[-1]) + dt.timedelta(seconds=1))
        .replace(microsecond=0)
        .isoformat()
    )
    ends = [*starts[1:], final_end]
    return {
        role: {"start": start, "end": end}
        for role, start, end in zip(CORE_RESEARCH_BUNDLE_ROLES, starts, ends, strict=True)
    }


def _single_interval(rows: tuple[dict[str, Any], ...]) -> dict[str, str]:
    times = sorted(str(row["close_timestamp"]) for row in rows)
    if not times:
        raise DatasetResolutionError("dataset data_pending: no forward bars are available")
    end = (
        (dt.datetime.fromisoformat(times[-1]) + dt.timedelta(seconds=1))
        .replace(microsecond=0)
        .isoformat()
    )
    return {"start": times[0], "end": end}


def _matches_feature(payload: Mapping[str, Any], market_type: str) -> bool:
    declared = payload.get("market_type")
    return declared is None or str(declared).lower() == market_type.lower()


def _parameter_set_id(product_id: str) -> str:
    return canonical_hash(
        {"schema": "platform.parameter_set/v1", "product_id": product_id, "parameters": {}}
    )


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetBundleBuildError(f"cannot hash bar partition {path}") from exc
    return f"sha256:{digest.hexdigest()}"


def _reason_code(message: str) -> str:
    if "insufficient distinct" in message:
        return "insufficient_distinct_bar_timestamps"
    if "no available bars for role" in message:
        return "historical_bars_incomplete"
    return "canonical_dataset_bundle_invalid"


def _text(value: Any, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result
