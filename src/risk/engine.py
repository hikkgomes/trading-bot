"""Persistence and aggregation for the six-level risk hierarchy."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import delete, insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src.data.database import market_input_buffer as market_input_buffer_table
from src.data.database import market_input_sources
from src.data.database import risk_decision as risk_decision_table
from src.data.database import risk_policy as risk_policy_table
from src.data.database import risk_snapshot as risk_snapshot_table
from src.domain._codec import canonical_hash, json_value, non_empty, timestamp, to_primitive
from src.domain.risk import RiskDecision

REQUIRED_RISK_SCOPES = ("strategy", "instrument", "sleeve", "product", "account", "global")
_MARKET_INPUT_RETENTION = dt.timedelta(minutes=15)
_MARKET_INPUT_CLEANUP_LIMIT = 1_000
_BUFFERED_MARKET_EVENT_TYPES = {
    "best_bid_ask",
    "depth_update",
    "mark_price",
    "funding_rate",
    "trade",
    "aggregate_trade",
}


class SqlRiskPolicyStore:
    """Immutable typed limits resolved by policy identity."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, policy_id: str, limits_by_scope: dict, *, contract: dict | None = None) -> str:
        policy_id = non_empty(policy_id, field="risk_policy_id")
        payload = json_value(limits_by_scope, field="risk policy")
        unknown = set(payload) - set(REQUIRED_RISK_SCOPES)
        if unknown or not payload:
            raise ValueError("risk policy contains unknown or no scopes")
        record = {"policy_id": policy_id, "limits": payload}
        if contract is not None or policy_id.startswith("sha256:"):
            contract = json_value(contract, field="risk policy contract")
            self._validate_contract(policy_id, contract, payload)
            record["contract"] = contract
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(risk_policy_table.c.payload).where(risk_policy_table.c.id == policy_id)
            ).scalar_one_or_none()
            if existing is None:
                connection.execute(
                    insert(risk_policy_table).values(
                        id=policy_id,
                        created_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                        payload=record,
                    )
                )
            elif dict(existing) != record:
                raise ValueError("risk policy records are immutable")
        return policy_id

    def resolve(
        self,
        policy_ids: tuple[str, ...],
        *,
        historical: bool = False,
        account_id: str | None = None,
    ) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        with self.engine.connect() as connection:
            for policy_id in policy_ids:
                payload = connection.execute(
                    select(risk_policy_table.c.payload).where(risk_policy_table.c.id == policy_id)
                ).scalar_one_or_none()
                if payload is None:
                    raise KeyError(f"risk policy does not exist: {policy_id}")
                if not historical or "contract" in payload or policy_id.startswith("sha256:"):
                    self._validate_contract(
                        policy_id, payload.get("contract"), payload["limits"], account_id=account_id
                    )
                for scope, limits in dict(payload)["limits"].items():
                    if scope in merged:
                        raise ValueError(f"multiple risk policies define scope {scope}")
                    merged[scope] = dict(limits)
        return merged

    @staticmethod
    def _validate_contract(
        policy_id: str, contract: object, limits: dict, *, account_id: str | None = None
    ) -> None:
        from src.risk.policies import compile_risk_limits
        from src.risk.policy import risk_policy_identity

        if not isinstance(contract, dict) or set(contract) != {
            "schema",
            "logical_policy_id",
            "configuration",
            "account_id",
        }:
            raise ValueError("risk policy unit contract is missing or malformed")
        if contract["schema"] != "platform.risk_policy/v2" or policy_id != risk_policy_identity(
            contract["logical_policy_id"],
            contract["configuration"],
            account_id=contract["account_id"],
        ):
            raise ValueError("risk policy identity does not match its unit contract")
        if limits != compile_risk_limits(
            contract["configuration"], contract["logical_policy_id"], contract["account_id"]
        ):
            raise ValueError("risk limits do not match their canonical configuration")
        if account_id is not None and contract["account_id"] != account_id:
            raise ValueError("risk policy account does not match the assessed account")


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


class SqlRiskSnapshotStore:
    """Immutable canonical inputs consumed by the risk worker.

    Queue commands carry only these content-addressed IDs. The values stay in
    PostgreSQL and are therefore not producer-controlled job results.
    """

    def __init__(self, engine: Engine, *, book_id: str | None = None):
        self.engine = engine
        self.book_id = book_id

    @staticmethod
    def _lock_identity(connection, snapshot_id: str) -> None:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:identity, 0))"),
                {"identity": f"risk-snapshot:{snapshot_id}"},
            )

    def save(self, payload: dict[str, object], *, created_at: str) -> str:
        if self.book_id is not None:
            payload = {**payload, "book_id": self.book_id}
        clean = json_value(payload, field="risk snapshot")
        snapshot_id = canonical_hash(clean)
        created_at = timestamp(created_at, field="created_at")
        with self.engine.begin() as connection:
            self._lock_identity(connection, snapshot_id)
            existing = connection.execute(
                select(risk_snapshot_table.c.payload).where(risk_snapshot_table.c.id == snapshot_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != clean:
                    raise ValueError("risk snapshot content-hash collision")
                return snapshot_id
            connection.execute(
                insert(risk_snapshot_table).values(
                    id=snapshot_id,
                    created_at=created_at,
                    payload=clean,
                )
            )
        return snapshot_id

    def save_market_input(
        self,
        payload: dict[str, object],
        *,
        created_at: str,
        retention_at: str | None = None,
    ) -> str:
        return self.save_market_inputs(((payload, created_at),), retention_at=retention_at)[0]

    def save_market_inputs(
        self,
        inputs: Iterable[tuple[dict[str, object], str]],
        *,
        retention_at: str | None = None,
    ) -> tuple[str, ...]:
        retained_at = timestamp(
            retention_at or dt.datetime.now(dt.UTC).isoformat(), field="retention_at"
        )
        cutoff = (dt.datetime.fromisoformat(retained_at) - _MARKET_INPUT_RETENTION).isoformat()
        identities, rows = self._market_input_rows(inputs, cutoff)
        with self.engine.begin() as connection:
            if rows:
                for identity in sorted(rows):
                    self._lock_identity(connection, identity)
                self._store_market_input_rows(connection, rows)
            self._expire_unpinned_market_inputs(connection, cutoff)
        return identities

    def _market_input_rows(self, inputs, cutoff):
        identities, rows = [], {}
        for payload, created_at in inputs:
            clean, event_type, product_id, instrument_id = self._prepare_market_input(payload)
            identity = canonical_hash(clean)
            observed = timestamp(created_at, field="created_at")
            identities.append(identity)
            # Expired quotes cannot authorise a current decision. Immutable pins
            # already survive independently, so replay need not write then delete.
            if observed < cutoff:
                continue
            row = {
                "id": identity,
                "product_id": product_id,
                "instrument_id": instrument_id,
                "event_type": event_type,
                "created_at": observed,
                "payload": clean,
            }
            if identity in rows and rows[identity] != row:
                raise ValueError("market input identity has conflicting content or timestamp")
            rows[identity] = row
        return tuple(identities), rows

    @staticmethod
    def _store_market_input_rows(connection, rows):
        pinned = dict(
            connection.execute(
                select(risk_snapshot_table.c.id, risk_snapshot_table.c.payload).where(
                    risk_snapshot_table.c.id.in_(rows)
                )
            ).all()
        )
        buffered = dict(
            connection.execute(
                select(market_input_buffer_table.c.id, market_input_buffer_table.c.payload).where(
                    market_input_buffer_table.c.id.in_(rows)
                )
            ).all()
        )
        for identity, existing in (*pinned.items(), *buffered.items()):
            if dict(existing) != rows[identity]["payload"]:
                raise ValueError("market input content-hash collision")
        if pinned:
            connection.execute(
                delete(market_input_buffer_table).where(market_input_buffer_table.c.id.in_(pinned))
            )
        missing = [
            row
            for identity, row in rows.items()
            if identity not in pinned and identity not in buffered
        ]
        if missing:
            connection.execute(insert(market_input_buffer_table), missing)

    def expire_market_inputs(self, *, at: str) -> int:
        current = timestamp(at, field="at")
        cutoff = (dt.datetime.fromisoformat(current) - _MARKET_INPUT_RETENTION).isoformat()
        with self.engine.begin() as connection:
            return self._expire_unpinned_market_inputs(connection, cutoff)

    def _prepare_market_input(
        self, payload: dict[str, object]
    ) -> tuple[dict[str, object], str, str, str]:
        if self.book_id is not None:
            payload = {**payload, "book_id": self.book_id}
        clean = json_value(payload, field="market input")
        if not isinstance(clean, dict) or clean.get("kind") != "market_data_input":
            raise ValueError("market input buffer accepts only market_data_input snapshots")
        raw_event_type = clean.get("source_event_type")
        product_id = clean.get("product_id")
        instrument_id = clean.get("instrument_id")
        if not isinstance(raw_event_type, str) or not isinstance(product_id, str):
            raise ValueError("market input event and product identities must be strings")
        if not isinstance(instrument_id, str):
            raise ValueError("market input instrument identity must be a string")
        event_type = non_empty(raw_event_type, field="source_event_type")
        if event_type == "candle" and clean.get("source_candle_closed") is True:
            raise ValueError("closed candles must remain immutable risk snapshots")
        if event_type not in _BUFFERED_MARKET_EVENT_TYPES and event_type != "candle":
            raise ValueError("market input event type is not eligible for the recent buffer")
        return (
            clean,
            event_type,
            non_empty(product_id, field="product_id"),
            non_empty(instrument_id, field="instrument_id"),
        )

    @staticmethod
    def _expire_unpinned_market_inputs(connection, cutoff: str) -> int:
        pinned = select(risk_snapshot_table.c.id).where(
            risk_snapshot_table.c.id == market_input_buffer_table.c.id
        )
        expired = (
            select(market_input_buffer_table.c.id)
            .where(
                market_input_buffer_table.c.created_at < cutoff,
                ~pinned.exists(),
            )
            .order_by(market_input_buffer_table.c.created_at, market_input_buffer_table.c.id)
            .limit(_MARKET_INPUT_CLEANUP_LIMIT)
        )
        result = connection.execute(
            delete(market_input_buffer_table).where(market_input_buffer_table.c.id.in_(expired))
        )
        return max(0, result.rowcount or 0)

    def get(self, snapshot_id: str) -> dict[str, object]:
        with self.engine.begin() as connection:
            self._lock_identity(connection, snapshot_id)
            payload = connection.execute(
                select(risk_snapshot_table.c.payload).where(risk_snapshot_table.c.id == snapshot_id)
            ).scalar_one_or_none()
            if payload is not None:
                clean = self._verified_payload(payload, snapshot_id)
                connection.execute(
                    delete(market_input_buffer_table).where(
                        market_input_buffer_table.c.id == snapshot_id
                    )
                )
                return clean
            buffered = connection.execute(
                select(
                    market_input_buffer_table.c.created_at, market_input_buffer_table.c.payload
                ).where(market_input_buffer_table.c.id == snapshot_id)
            ).one_or_none()
            if buffered is None:
                payload = connection.execute(
                    select(risk_snapshot_table.c.payload).where(
                        risk_snapshot_table.c.id == snapshot_id
                    )
                ).scalar_one_or_none()
                if payload is None:
                    raise KeyError(f"risk snapshot does not exist: {snapshot_id}")
                return self._verified_payload(payload, snapshot_id)
            created_at, payload = buffered
            clean = self._verified_payload(payload, snapshot_id)
            self._pin_market_input(connection, snapshot_id, created_at, clean)
            connection.execute(
                delete(market_input_buffer_table).where(
                    market_input_buffer_table.c.id == snapshot_id
                )
            )
            return clean

    @staticmethod
    def _verified_payload(payload: object, snapshot_id: str) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise KeyError(f"risk snapshot does not exist: {snapshot_id}")
        if canonical_hash(payload) != snapshot_id:
            raise ValueError("risk snapshot content hash does not match its identity")
        return dict(payload)

    @staticmethod
    def _pin_market_input(connection, snapshot_id: str, created_at: str, payload: dict) -> None:
        try:
            with connection.begin_nested():
                connection.execute(
                    insert(risk_snapshot_table).values(
                        id=snapshot_id,
                        created_at=created_at,
                        payload=payload,
                    )
                )
        except IntegrityError:
            pass
        existing = connection.execute(
            select(risk_snapshot_table.c.payload).where(risk_snapshot_table.c.id == snapshot_id)
        ).scalar_one_or_none()
        if not isinstance(existing, dict) or dict(existing) != payload:
            raise ValueError("pinned market input does not match its content hash")

    def latest(
        self, *, kind: str, product_id: str, at: str, require_source_envelope: bool = False
    ) -> tuple[str, dict[str, object]]:
        at = timestamp(at, field="at")
        source = market_input_sources if kind == "market_data_input" else risk_snapshot_table
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(
                        source.c.id,
                        source.c.created_at,
                        source.c.payload,
                    )
                    .where(
                        source.c.created_at <= at,
                        source.c.payload["kind"].as_string() == kind,
                        source.c.payload["product_id"].as_string() == product_id,
                        source.c.payload["book_id"].as_string() == self.book_id,
                    )
                    .order_by(source.c.created_at.desc(), source.c.id.desc())
                )
                .mappings()
                .all()
            )
            for row in rows:
                payload = row["payload"]
                if (
                    isinstance(payload, dict)
                    and payload.get("kind") == kind
                    and payload.get("product_id") == product_id
                    and (
                        not require_source_envelope
                        or isinstance(payload.get("values"), dict)
                        and isinstance(payload.get("observed_at"), str)
                    )
                ):
                    if canonical_hash(payload) != str(row["id"]):
                        raise ValueError("risk snapshot content hash does not match its identity")
                    identity = str(row["id"])
                    return identity, self.get(identity) if kind == "market_data_input" else dict(
                        payload
                    )
        raise KeyError(f"no {kind} snapshot exists for product {product_id} at {at}")


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
        evaluated_at=max(item.evaluated_at for item in decisions),
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
