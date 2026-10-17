from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select

from src.data.database import PlatformDatabase, alert
from src.observability.reports import DatabasePlatformReport
from src.services.alerting import AlertSeverity, SqlAlertService

FIRST = "2026-08-31T10:00:00+00:00"
WITHIN = "2026-08-31T10:00:30+00:00"
AFTER = "2026-08-31T10:02:00+00:00"


@dataclass
class FakeSink:
    name: str = "fake"
    fail: bool = False
    records: list[dict] = field(default_factory=list)

    def send(self, record) -> None:
        if self.fail:
            raise OSError("sink unavailable")
        self.records.append(record.to_dict())


def _service(tmp_path, sink=None):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'alerts.sqlite3'}")
    database.create_schema()
    service = SqlAlertService(
        database.engine, sinks=(sink,) if sink else (), default_cooldown_seconds=60
    )
    return database, service


def test_alert_cooldown_is_durable(tmp_path) -> None:
    database, service = _service(tmp_path)
    try:
        first = service.emit(
            event_type="risk_blocked",
            severity=AlertSeverity.WARNING,
            dedupe_key="risk:active_income",
            target="active_income",
            message="risk blocked a live order",
            emitted_at=FIRST,
        )
        suppressed = service.emit(
            event_type="risk_blocked",
            severity=AlertSeverity.WARNING,
            dedupe_key="risk:active_income",
            target="active_income",
            message="risk blocked a live order",
            emitted_at=WITHIN,
        )
        after = service.emit(
            event_type="risk_blocked",
            severity=AlertSeverity.WARNING,
            dedupe_key="risk:active_income",
            target="active_income",
            message="risk blocked a live order",
            emitted_at=AFTER,
        )
        assert first.suppressed is False
        assert suppressed.suppressed is True
        assert after.suppressed is False
        with database.engine.connect() as connection:
            assert connection.execute(select(alert.c.id)).fetchall().__len__() == 2
    finally:
        database.dispose()


def test_acknowledgement_is_an_append_only_event(tmp_path) -> None:
    database, service = _service(tmp_path)
    try:
        emitted = service.emit(
            event_type="control_mode_changed",
            severity="critical",
            dedupe_key="control:global",
            message="emergency flatten requested",
            emitted_at=FIRST,
        )
        acknowledgement = service.acknowledge(
            alert_id=emitted.alert_id,
            acknowledged_by="operator",
            acknowledged_at=WITHIN,
        )
        assert acknowledgement.event_type == "acknowledged"
        assert acknowledgement.parent_alert_id == emitted.alert_id
        assert [item.event_type for item in service.events()] == [
            "control_mode_changed",
            "acknowledged",
        ]
    finally:
        database.dispose()


def test_failed_external_delivery_is_visible_without_recursive_delivery(tmp_path) -> None:
    sink = FakeSink(fail=True)
    database, service = _service(tmp_path, sink)
    try:
        result = service.emit(
            event_type="protective_stop_failed",
            severity="critical",
            dedupe_key="stop:one",
            target="active_income",
            message="protective stop placement failed",
            emitted_at=FIRST,
        )
        events = service.events()
        assert result.delivery_status == "failed"
        assert len(sink.records) == 0
        assert {item.event_type for item in events} == {"protective_stop_failed", "delivery_failed"}
        failure = next(item for item in events if item.event_type == "delivery_failed")
        assert failure.payload["parent_alert_id"] == result.alert_id
        assert "url" not in str(failure.payload).lower()
        reports = DatabasePlatformReport(database.engine)
        assert reports.build(now=FIRST)["operations"]["alert_delivery_failure_count"] == 1
        sink.fail = False
        service.deliver_pending(now=AFTER)
        assert len(sink.records) == 1
        assert reports.build(now=AFTER)["operations"]["alert_delivery_failure_count"] == 0
        assert any(item.event_type == "delivery_failed" for item in service.events())
    finally:
        database.dispose()


def test_acknowledged_delivery_failure_is_not_retried_or_reported(tmp_path) -> None:
    sink = FakeSink(fail=True)
    database, service = _service(tmp_path, sink)
    try:
        result = service.emit(
            event_type="protective_stop_failed",
            severity="critical",
            dedupe_key="stop:acknowledged",
            message="protective stop placement failed",
            emitted_at=FIRST,
        )
        service.acknowledge(
            alert_id=result.alert_id, acknowledged_by="operator", acknowledged_at=WITHIN
        )
        sink.fail = False
        service.deliver_pending(now=AFTER)
        assert not sink.records
        assert len(service.delivery_attempts(alert_id=result.alert_id)) == 1
        report = DatabasePlatformReport(database.engine).build(now=AFTER)
        assert report["operations"]["alert_delivery_failure_count"] == 0
        assert any(item.event_type == "delivery_failed" for item in service.events())
    finally:
        database.dispose()


def test_acknowledgement_survives_an_in_flight_delivery_failure(tmp_path) -> None:
    sink = FakeSink(fail=True)
    database, service = _service(tmp_path, sink)
    try:
        result = service.emit(
            event_type="protective_stop_failed",
            severity="critical",
            dedupe_key="stop:in-flight",
            message="protective stop placement failed",
            emitted_at=FIRST,
        )
        attempt = service._start_delivery_attempt(result, sink.name, AFTER, expected_attempt=1)
        assert attempt == 2
        assert service._start_delivery_attempt(result, sink.name, AFTER, expected_attempt=1) is None
        service.acknowledge(
            alert_id=result.alert_id, acknowledged_by="operator", acknowledged_at=AFTER
        )
        service._finish_delivery_attempt(
            result,
            sink.name,
            attempt_no=attempt,
            status="failed",
            attempted_at=AFTER,
            error=OSError("delivery failed after acknowledgement"),
        )
        assert service.delivery_attempts(alert_id=result.alert_id)[-1]["acknowledged_at"] == AFTER
        assert service._start_delivery_attempt(result, sink.name, AFTER, expected_attempt=2) is None
        sink.fail = False
        service.deliver_pending(now="2026-08-31T11:00:00+00:00")
        assert not sink.records
        assert len(service.delivery_attempts(alert_id=result.alert_id)) == 2
        report = DatabasePlatformReport(database.engine).build(now=AFTER)
        assert report["operations"]["alert_delivery_failure_count"] == 0
    finally:
        database.dispose()
