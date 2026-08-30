"""Durable operational alerts with optional external delivery."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from src.data.database import alert
from src.domain._codec import canonical_hash, finite, json_value, non_empty, timestamp


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str
    event_type: str
    severity: AlertSeverity
    dedupe_key: str
    target: str
    message: str
    emitted_at: str
    payload: dict[str, Any] = field(default_factory=dict)
    parent_alert_id: str | None = None
    suppressed: bool = False
    delivery_status: str = "persisted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "platform.alert/v1",
            "alert_id": self.alert_id,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "dedupe_key": self.dedupe_key,
            "target": self.target,
            "message": self.message,
            "emitted_at": self.emitted_at,
            "payload": json_value(self.payload, field="alert payload"),
            "parent_alert_id": self.parent_alert_id,
            "suppressed": self.suppressed,
            "delivery_status": self.delivery_status,
        }


@dataclass(frozen=True)
class AlertDelivery:
    alert_id: str
    delivered: tuple[str, ...]
    failed: tuple[str, ...]


class AlertSink(Protocol):
    """External alert delivery contract."""

    name: str

    def send(self, record: AlertRecord) -> None:
        ...


class WebhookAlertSink:
    """Send an alert to a JSON webhook without persisting its URL."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 5.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        url = non_empty(url, field="webhook URL")
        if not url.startswith(("https://", "http://")):
            raise ValueError("webhook URL must use http or https")
        timeout_seconds = finite(timeout_seconds, field="webhook timeout", minimum=0.1)
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._opener = opener

    def send(self, record: AlertRecord) -> None:
        body = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        request = Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = self._opener(request, timeout=self._timeout_seconds)
            close = getattr(response, "close", None)
            if callable(close):
                close()
        except (HTTPError, URLError, OSError) as exc:
            raise RuntimeError(type(exc).__name__) from exc


class SqlAlertService:
    """Append-only alert events with database-backed cooldown state."""

    def __init__(
        self,
        engine: Engine,
        *,
        sinks: Iterable[AlertSink] = (),
        default_cooldown_seconds: int = 900,
    ) -> None:
        if isinstance(default_cooldown_seconds, bool) or default_cooldown_seconds < 0:
            raise ValueError("default_cooldown_seconds must be non-negative")
        self.engine = engine
        self.sinks = tuple(sinks)
        self.default_cooldown_seconds = default_cooldown_seconds

    def emit(
        self,
        *,
        event_type: str,
        severity: AlertSeverity | str,
        dedupe_key: str,
        message: str,
        emitted_at: str,
        target: str = "platform",
        payload: Mapping[str, Any] | None = None,
        cooldown_seconds: int | None = None,
    ) -> AlertRecord:
        event_type = non_empty(event_type, field="event_type")
        severity = AlertSeverity(severity)
        dedupe_key = non_empty(dedupe_key, field="dedupe_key")
        message = non_empty(message, field="message")
        target = non_empty(target, field="target")
        emitted_at = timestamp(emitted_at, field="emitted_at")
        details = json_value(dict(payload or {}), field="alert payload")
        cooldown = (
            self.default_cooldown_seconds
            if cooldown_seconds is None
            else _cooldown_seconds(cooldown_seconds)
        )
        previous = self._latest_emission(dedupe_key)
        if previous is not None and _within_cooldown(
            previous.emitted_at, emitted_at, cooldown
        ):
            return AlertRecord(
                alert_id=previous.alert_id,
                event_type=previous.event_type,
                severity=previous.severity,
                dedupe_key=previous.dedupe_key,
                target=previous.target,
                message=previous.message,
                emitted_at=previous.emitted_at,
                payload=previous.payload,
                parent_alert_id=previous.parent_alert_id,
                suppressed=True,
                delivery_status="suppressed",
            )

        record = AlertRecord(
            alert_id=_event_id(
                event_type=event_type,
                severity=severity,
                dedupe_key=dedupe_key,
                target=target,
                message=message,
                emitted_at=emitted_at,
                payload=details,
            ),
            event_type=event_type,
            severity=severity,
            dedupe_key=dedupe_key,
            target=target,
            message=message,
            emitted_at=emitted_at,
            payload=details,
        )
        self._append(record)
        delivered: list[str] = []
        failed: list[str] = []
        for sink in self.sinks:
            sink_name = _sink_name(sink)
            try:
                sink.send(record)
            except Exception as exc:  # external delivery must not stop trading controls
                failed.append(sink_name)
                self._append_delivery_failure(record, sink_name, exc, emitted_at)
            else:
                delivered.append(sink_name)
        if failed:
            return AlertRecord(
                **{
                    **record.__dict__,
                    "delivery_status": "failed",
                }
            )
        if delivered:
            return AlertRecord(
                **{
                    **record.__dict__,
                    "delivery_status": "delivered",
                }
            )
        return record

    def acknowledge(
        self,
        *,
        alert_id: str,
        acknowledged_by: str,
        acknowledged_at: str,
    ) -> AlertRecord:
        alert_id = non_empty(alert_id, field="alert_id")
        acknowledged_by = non_empty(acknowledged_by, field="acknowledged_by")
        acknowledged_at = timestamp(acknowledged_at, field="acknowledged_at")
        source = self._get(alert_id)
        if source is None:
            raise KeyError(f"alert does not exist: {alert_id}")
        record = AlertRecord(
            alert_id=_event_id(
                event_type="acknowledged",
                severity=source.severity,
                dedupe_key=f"ack:{alert_id}",
                target=source.target,
                message="alert acknowledged",
                emitted_at=acknowledged_at,
                payload={"acknowledged_by": acknowledged_by, "alert_id": alert_id},
            ),
            event_type="acknowledged",
            severity=source.severity,
            dedupe_key=f"ack:{alert_id}",
            target=source.target,
            message="alert acknowledged",
            emitted_at=acknowledged_at,
            payload={"acknowledged_by": acknowledged_by, "alert_id": alert_id},
            parent_alert_id=alert_id,
        )
        self._append(record)
        return record

    def events(self, *, dedupe_key: str | None = None) -> tuple[AlertRecord, ...]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(alert).order_by(alert.c.created_at.asc(), alert.c.id.asc())
            ).mappings()
            records = tuple(_record_from_payload(dict(row["payload"])) for row in rows)
        if dedupe_key is None:
            return records
        return tuple(record for record in records if record.dedupe_key == dedupe_key)

    def _latest_emission(self, dedupe_key: str) -> AlertRecord | None:
        records = [
            record
            for record in self.events(dedupe_key=dedupe_key)
            if record.event_type != "delivery_failed"
            and record.event_type != "acknowledged"
        ]
        return records[-1] if records else None

    def _get(self, alert_id: str) -> AlertRecord | None:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(alert.c.payload).where(alert.c.id == alert_id)
            ).scalar_one_or_none()
        return _record_from_payload(dict(payload)) if payload is not None else None

    def _append(self, record: AlertRecord) -> None:
        payload = record.to_dict()
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(alert.c.payload).where(alert.c.id == record.alert_id)
            ).scalar_one_or_none()
            if existing is not None:
                if dict(existing) != payload:
                    raise ValueError("alert identity collision")
                return
            connection.execute(
                insert(alert).values(
                    id=record.alert_id,
                    created_at=record.emitted_at,
                    payload=payload,
                )
            )

    def _append_delivery_failure(
        self, record: AlertRecord, sink_name: str, exc: Exception, emitted_at: str
    ) -> None:
        failure = AlertRecord(
            alert_id=_event_id(
                event_type="delivery_failed",
                severity=AlertSeverity.CRITICAL,
                dedupe_key=f"delivery-failure:{record.alert_id}:{sink_name}",
                target=record.target,
                message="external alert delivery failed",
                emitted_at=emitted_at,
                payload={
                    "parent_alert_id": record.alert_id,
                    "sink": sink_name,
                    "error_type": type(exc).__name__,
                },
            ),
            event_type="delivery_failed",
            severity=AlertSeverity.CRITICAL,
            dedupe_key=f"delivery-failure:{record.alert_id}:{sink_name}",
            target=record.target,
            message="external alert delivery failed",
            emitted_at=emitted_at,
            payload={
                "parent_alert_id": record.alert_id,
                "sink": sink_name,
                "error_type": type(exc).__name__,
            },
            parent_alert_id=record.alert_id,
            delivery_status="failed",
        )
        self._append(failure)


def configured_alert_service(
    engine: Engine,
    *,
    configuration: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> SqlAlertService:
    """Build alert delivery from non-secret config and an environment secret."""

    settings = dict(configuration or {})
    environment = environment or os.environ
    enabled = bool(settings.get("enabled", True))
    url = str(environment.get("TRADING_PLATFORM_ALERT_WEBHOOK_URL") or "").strip()
    sinks: tuple[AlertSink, ...] = (WebhookAlertSink(url),) if enabled and url else ()
    cooldown = _cooldown_seconds(settings.get("cooldown_seconds", 900))
    return SqlAlertService(engine, sinks=sinks, default_cooldown_seconds=cooldown)


def _event_id(
    *,
    event_type: str,
    severity: AlertSeverity,
    dedupe_key: str,
    target: str,
    message: str,
    emitted_at: str,
    payload: Mapping[str, Any],
) -> str:
    return "alert:" + canonical_hash(
        {
            "event_type": event_type,
            "severity": severity.value,
            "dedupe_key": dedupe_key,
            "target": target,
            "message": message,
            "emitted_at": emitted_at,
            "payload": dict(payload),
        }
    ).removeprefix("sha256:")


def _record_from_payload(payload: Mapping[str, Any]) -> AlertRecord:
    return AlertRecord(
        alert_id=non_empty(str(payload["alert_id"]), field="alert_id"),
        event_type=non_empty(str(payload["event_type"]), field="event_type"),
        severity=AlertSeverity(payload["severity"]),
        dedupe_key=non_empty(str(payload["dedupe_key"]), field="dedupe_key"),
        target=non_empty(str(payload["target"]), field="target"),
        message=non_empty(str(payload["message"]), field="message"),
        emitted_at=timestamp(payload["emitted_at"], field="emitted_at"),
        payload=json_value(dict(payload.get("payload") or {}), field="alert payload"),
        parent_alert_id=(
            str(payload["parent_alert_id"])
            if payload.get("parent_alert_id") is not None
            else None
        ),
        suppressed=bool(payload.get("suppressed", False)),
        delivery_status=str(payload.get("delivery_status") or "persisted"),
    )


def _cooldown_seconds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cooldown_seconds must be a non-negative integer")
    return value


def _within_cooldown(previous: str, current: str, seconds: int) -> bool:
    if seconds == 0:
        return False
    previous_at = dt.datetime.fromisoformat(timestamp(previous, field="previous alert"))
    current_at = dt.datetime.fromisoformat(timestamp(current, field="current alert"))
    elapsed = (current_at - previous_at).total_seconds()
    return 0 <= elapsed < seconds


def _sink_name(sink: AlertSink) -> str:
    return non_empty(str(getattr(sink, "name", type(sink).__name__)), field="sink name")
