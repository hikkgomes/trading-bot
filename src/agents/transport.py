"""Authenticated review envelopes; signing keys belong to the transport, never the model."""

from __future__ import annotations

import hmac
from typing import Any

from src.domain._codec import canonical_hash, timestamp


def sign_review_response(
    request: dict[str, Any], proposal: dict[str, Any], *, key: bytes
) -> dict[str, Any]:
    envelope = {
        "schema": "openclaw.review_response/v1",
        "review_id": request["review_id"],
        "request_hash": canonical_hash(request),
        "expires_at": request["expires_at"],
        "proposal": proposal,
    }
    return {**envelope, "signature": _signature(envelope, key)}


def verify_review_response(
    envelope: dict[str, Any], request: dict[str, Any], *, key: bytes, now: str
) -> dict[str, Any]:
    expected = {"schema", "review_id", "request_hash", "expires_at", "proposal", "signature"}
    if set(envelope) != expected or envelope.get("schema") != "openclaw.review_response/v1":
        raise ValueError("unsupported OpenClaw review envelope")
    unsigned = {name: value for name, value in envelope.items() if name != "signature"}
    signature = envelope["signature"]
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, _signature(unsigned, key)
    ):
        raise PermissionError("invalid OpenClaw response signature")
    if (
        envelope["review_id"] != request["review_id"]
        or envelope["request_hash"] != canonical_hash(request)
        or envelope["expires_at"] != request.get("expires_at")
    ):
        raise PermissionError("OpenClaw response does not match the sealed request")
    if timestamp(now, field="now") >= timestamp(envelope["expires_at"], field="expires_at"):
        raise PermissionError("OpenClaw review response expired")
    proposal = envelope["proposal"]
    if not isinstance(proposal, dict):
        raise ValueError("OpenClaw response requires a proposal object")
    return proposal


def _signature(envelope: dict[str, Any], key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("OpenClaw transport key requires at least 32 bytes")
    return hmac.digest(key, canonical_hash(envelope).encode("ascii"), "sha256").hex()
