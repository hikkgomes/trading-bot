"""Persistence and aggregation for the six-level risk hierarchy."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import risk_decision as risk_decision_table
from src.domain._codec import non_empty, to_primitive
from src.domain.risk import RiskDecision

REQUIRED_RISK_SCOPES = ("strategy", "instrument", "sleeve", "product", "account", "global")


class JsonlRiskDecisionStore:
    def __init__(self, path: Path):
        self.path = path

    def append(self, risk_decision: RiskDecision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(to_primitive(risk_decision), sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[RiskDecision, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("risk-decision journal must be a regular file")
        decisions: list[RiskDecision] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                decisions.append(RiskDecision(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid risk decision at line {line_number}") from exc
        return tuple(decisions)


class RiskDecisionStore(Protocol):
    def append(self, risk_decision: RiskDecision) -> None: ...

    def read(self) -> tuple[RiskDecision, ...]: ...


class SqlRiskDecisionStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def append(self, risk_decision: RiskDecision) -> None:
        payload = to_primitive(risk_decision)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(risk_decision_table.c.payload).where(
                    risk_decision_table.c.id == risk_decision.decision_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("risk-decision identity collision")
                return
            connection.execute(
                insert(risk_decision_table).values(
                    id=risk_decision.decision_id,
                    scope=risk_decision.scope,
                    evaluated_at=risk_decision.evaluated_at,
                    accepted=risk_decision.accepted,
                    reason_code=risk_decision.reason_code,
                    payload=payload,
                )
            )

    def read(self) -> tuple[RiskDecision, ...]:
        with self.engine.connect() as connection:
            payloads = connection.execute(
                select(risk_decision_table.c.payload).order_by(
                    risk_decision_table.c.evaluated_at,
                    risk_decision_table.c.id,
                )
            ).scalars()
            return tuple(RiskDecision(**dict(payload)) for payload in payloads)

    def assessment(self, assessment_id: str) -> HierarchicalRiskAssessment:
        decisions = {item.decision_id: item for item in self.read()}
        aggregate = decisions.get(assessment_id)
        if aggregate is None or aggregate.scope != "portfolio":
            raise KeyError(f"risk assessment does not exist: {assessment_id}")
        raw_ids = aggregate.input_snapshot.get("decision_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("risk assessment has no component decision IDs")
        components = tuple(decisions[str(identity)] for identity in raw_ids)
        if tuple(item.scope for item in components) != REQUIRED_RISK_SCOPES:
            raise ValueError("risk assessment components are incomplete")
        return HierarchicalRiskAssessment(components, aggregate)


@dataclass(frozen=True)
class HierarchicalRiskAssessment:
    decisions: tuple[RiskDecision, ...]
    aggregate: RiskDecision

    @property
    def accepted(self) -> bool:
        return self.aggregate.accepted


def combine_risk_decisions(
    decisions: tuple[RiskDecision, ...],
    *,
    assessment_id: str,
    product_id: str,
    store: RiskDecisionStore | None = None,
) -> HierarchicalRiskAssessment:
    product_id = non_empty(product_id, field="product_id")
    scopes = tuple(item.scope for item in decisions)
    if scopes != REQUIRED_RISK_SCOPES:
        raise ValueError(f"risk decisions must be ordered as {REQUIRED_RISK_SCOPES}")
    rejected = next((item for item in decisions if not item.accepted), None)
    aggregate = RiskDecision(
        decision_id=assessment_id,
        scope="portfolio",
        accepted=rejected is None,
        reason_code=rejected.reason_code if rejected else None,
        evaluated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        input_snapshot={
            "product_id": product_id,
            "decision_ids": [item.decision_id for item in decisions],
            "input_hashes": [item.input_hash for item in decisions],
            "first_rejected_scope": rejected.scope if rejected else None,
        },
        limits={"required_scopes": list(REQUIRED_RISK_SCOPES)},
    )
    if store is not None:
        for item in (*decisions, aggregate):
            store.append(item)
    return HierarchicalRiskAssessment(decisions=decisions, aggregate=aggregate)
