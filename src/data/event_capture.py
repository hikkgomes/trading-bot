"""Canonical data boundary for bounded Binance public-event capture.

The implementation remains shared with the archived offline tools. Production
services import this boundary so the legacy package is not an operational
authority.
"""

from src.autopilot.event_capture import (
    EventCaptureConfig,
    capture,
    load_event_capture_config,
)

__all__ = ("EventCaptureConfig", "capture", "load_event_capture_config")
