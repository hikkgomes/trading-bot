#!/usr/bin/env python3
"""Return OpenClaw's no-reply signal when loopback platform health is good."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

DEFAULT_URL = "http://127.0.0.1:8088/healthz"
MAX_RESPONSE_BYTES = 16_384
_FAILURE_MESSAGES = {
    "heartbeat_missing": "service heartbeat missing",
    "heartbeat_stale": "service heartbeat stale",
    "heartbeat_unhealthy": "service unhealthy",
    "report_missing": "report missing",
    "report_stale": "report stale",
    "report_invalid": "report invalid",
    "backup_unhealthy": "backup unhealthy",
    "backup_status_missing": "backup status missing",
    "backup_status_invalid": "backup status invalid",
    "disk_unhealthy": "disk unhealthy",
    "disk_status_missing": "disk status missing",
    "disk_status_invalid": "disk status invalid",
    "recovery_unresolved": "recovery unresolved",
    "account_authority_stale": "account authority stale",
    "market_data_stale": "market data stale",
    "risk_data_missing": "risk data missing",
    "execution_authority_conflict": "execution authority conflict",
    "alert_delivery_failed": "alert delivery failed",
    "health_check_unavailable": "health check unavailable",
}


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _loopback_health_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme != "http"
            or host is None
            or port is not None
            and not 1 <= port <= 65_535
            or parsed.path != "/healthz"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            return False
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _health_response(url: str) -> tuple[bool, str]:
    generic_failure = "Platform health check failed."
    if not _loopback_health_url(url):
        return False, generic_failure
    opener = build_opener(ProxyHandler({}), _RejectRedirect())
    try:
        with opener.open(url, timeout=10) as response:
            status = response.status
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        status = error.code
        try:
            body = error.read(MAX_RESPONSE_BYTES + 1)
        finally:
            error.close()
    except Exception:
        return False, generic_failure
    if len(body) > MAX_RESPONSE_BYTES:
        return False, generic_failure
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False, generic_failure
    if not isinstance(payload, dict) or payload.get("schema") != "platform.healthz/v1":
        return False, generic_failure
    if status == 200 and payload.get("ok") is True:
        return True, ""
    reason_codes = payload.get("reason_codes")
    if status != 503 or payload.get("ok") is not False or not isinstance(reason_codes, list):
        return False, generic_failure
    if not reason_codes or any(
        not isinstance(reason, str) or reason not in _FAILURE_MESSAGES for reason in reason_codes
    ):
        return False, generic_failure
    messages = sorted({_FAILURE_MESSAGES[reason] for reason in reason_codes})
    return False, f"Platform health check failed: {'; '.join(messages)}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local platform health for OpenClaw.")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    try:
        healthy, failure = _health_response(args.url)
    except Exception:
        healthy = False
        failure = "Platform health check failed."
    if healthy:
        print("NO_REPLY")
        return 0
    print(failure, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
