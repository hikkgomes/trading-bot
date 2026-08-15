"""Machine-readable decision-funnel traces for every processed event."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import decision_trace as decision_trace_table
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp


class DecisionTraceStage(StrEnum):
    DATA_AVAILABLE = "data_available"
    FEATURE_AVAILABLE = "feature_available"
    STRATEGY_EVALUATED = "strategy_evaluated"
    REGIME_PASSED = "regime_passed"
    SETUP_PASSED = "setup_passed"
    TRIGGER_PASSED = "trigger_passed"
    SIGNAL_PRODUCED = "signal_produced"
    PORTFOLIO_ACCEPTED = "portfolio_accepted"
    RISK_ACCEPTED = "risk_accepted"
    ORDER_PLANNED = "order_planned"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"


_STAGES = tuple(DecisionTraceStage)


@dataclass(frozen=True)
class DecisionTrace:
    event_id: str
    instrument_id: str
    evaluated_at: str
    stages: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", non_empty(self.event_id, field="event_id"))
        object.__setattr__(
            self, "instrument_id", non_empty(self.instrument_id, field="instrument_id")
        )
        object.__setattr__(self, "evaluated_at", timestamp(self.evaluated_at, field="evaluated_at"))
        if not isinstance(self.stages, Mapping):
            raise ValueError("stages must be an object")
        normalised = {}
        for stage, detail in self.stages.items():
            if stage not in {item.value for item in _STAGES}:
                raise ValueError(f"unknown decision stage: {stage}")
            if not isinstance(detail, Mapping):
                raise ValueError("stage detail must be an object")
            outcome = detail.get("outcome")
            if outcome not in {"passed", "blocked", "not_evaluated"}:
                raise ValueError("stage outcome must be passed, blocked, or not_evaluated")
            if outcome == "blocked" and not detail.get("reason_code"):
                raise ValueError("blocked stages require reason_code")
            normalised[stage] = json_value(dict(detail), field="stage detail")
        object.__setattr__(self, "stages", normalised)

    @classmethod
    def start(
        cls, *, event_id: str, instrument_id: str, evaluated_at: str | None = None
    ) -> DecisionTrace:
        return cls(
            event_id=event_id,
            instrument_id=instrument_id,
            evaluated_at=evaluated_at or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        )

    def pass_stage(self, stage: DecisionTraceStage, **metadata: Any) -> DecisionTrace:
        self._assert_can_record(stage)
        stages = dict(self.stages)
        stages[stage.value] = {
            "outcome": "passed",
            "metadata": json_value(metadata, field="metadata"),
        }
        return DecisionTrace(self.event_id, self.instrument_id, self.evaluated_at, stages)

    def block(
        self, stage: DecisionTraceStage, *, reason_code: str, **metadata: Any
    ) -> DecisionTrace:
        self._assert_can_record(stage)
        stages = dict(self.stages)
        stages[stage.value] = {
            "outcome": "blocked",
            "reason_code": non_empty(reason_code, field="reason_code"),
            "metadata": json_value(metadata, field="metadata"),
        }
        for later in _STAGES[_STAGES.index(stage) + 1 :]:
            stages[later.value] = {"outcome": "not_evaluated"}
        return DecisionTrace(self.event_id, self.instrument_id, self.evaluated_at, stages)

    def _assert_can_record(self, stage: DecisionTraceStage) -> None:
        if self.first_blocked_stage is not None:
            raise ValueError("cannot record after a blocked decision stage")
        index = _STAGES.index(stage)
        missing = [item.value for item in _STAGES[:index] if item.value not in self.stages]
        if missing:
            raise ValueError(f"cannot record {stage.value} before {missing[-1]}")
        if stage.value in self.stages:
            raise ValueError(f"stage is already recorded: {stage.value}")

    @property
    def first_blocked_stage(self) -> str | None:
        for stage in _STAGES:
            if self.stages.get(stage.value, {}).get("outcome") == "blocked":
                return stage.value
        return None

    @property
    def is_complete(self) -> bool:
        return all(self.stages.get(stage.value, {}).get("outcome") == "passed" for stage in _STAGES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "platform.decision_trace/v1",
            "event_id": self.event_id,
            "instrument_id": self.instrument_id,
            "evaluated_at": self.evaluated_at,
            "stages": dict(self.stages),
            "first_blocked_stage": self.first_blocked_stage,
            "complete": self.is_complete,
        }


class JsonlDecisionTraceStore:
    """Append-only, hash-chained decision trace persistence."""

    def __init__(self, path: Path):
        self.path = path
        self._previous_hash = "0" * 64
        existing = self.read()
        if existing:
            self._previous_hash = existing[-1][0]

    def append(self, trace: DecisionTrace) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "schema": "platform.decision_trace_event/v1",
            "previous_hash": self._previous_hash,
            "trace": trace.to_dict(),
        }
        event_hash = canonical_hash(body)
        encoded = json.dumps(
            {**body, "event_hash": event_hash}, sort_keys=True, separators=(",", ":")
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous_hash = event_hash
        return event_hash

    def read(self) -> tuple[tuple[str, DecisionTrace], ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("decision trace journal must be a regular file")
        events: list[tuple[str, DecisionTrace]] = []
        previous_hash = "0" * 64
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                event_hash = payload.pop("event_hash")
                trace_payload = payload["trace"]
                trace = DecisionTrace(
                    event_id=trace_payload["event_id"],
                    instrument_id=trace_payload["instrument_id"],
                    evaluated_at=trace_payload["evaluated_at"],
                    stages=trace_payload["stages"],
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid decision trace at line {line_number}") from exc
            if payload.get("previous_hash") != previous_hash:
                raise ValueError(f"decision trace chain is invalid at line {line_number}")
            if canonical_hash(payload) != event_hash:
                raise ValueError(f"decision trace hash is invalid at line {line_number}")
            previous_hash = event_hash
            events.append((event_hash, trace))
        return tuple(events)


class SqlDecisionTraceStore:
    """Immutable PostgreSQL decision traces with content-hash identities."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, trace: DecisionTrace) -> str:
        payload = trace.to_dict()
        event_hash = canonical_hash(payload)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(decision_trace_table.c.payload).where(
                    decision_trace_table.c.id == event_hash
                )
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("decision trace content-hash collision")
                return event_hash
            connection.execute(
                insert(decision_trace_table).values(
                    id=event_hash,
                    event_id=trace.event_id,
                    instrument_id=trace.instrument_id,
                    evaluated_at=trace.evaluated_at,
                    first_blocked_stage=trace.first_blocked_stage,
                    payload=payload,
                )
            )
        return event_hash

    def read(self) -> tuple[tuple[str, DecisionTrace], ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(decision_trace_table).order_by(
                    decision_trace_table.c.evaluated_at,
                    decision_trace_table.c.id,
                )
            ).mappings()
            events: list[tuple[str, DecisionTrace]] = []
            for row in rows:
                payload = dict(row["payload"])
                trace = DecisionTrace(
                    event_id=payload["event_id"],
                    instrument_id=payload["instrument_id"],
                    evaluated_at=payload["evaluated_at"],
                    stages=payload["stages"],
                )
                if canonical_hash(trace.to_dict()) != row["id"]:
                    raise ValueError("SQL decision trace content hash is invalid")
                events.append((row["id"], trace))
            return tuple(events)
