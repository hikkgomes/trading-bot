"""Exact execution identity for staged-candidate forward-paper evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.autopilot.execution_identity import execution_engine_digest

CANDIDATE_PAPER_EXECUTION_SCHEMA = "autopilot.candidate_paper.forward_observation/v2"
CANDIDATE_PAPER_FORWARD_FILL_SOURCE = "public_observation_quote"
CANDIDATE_PAPER_BACKFILL_FILL_SOURCE = "historical_next_open_replay"
CANDIDATE_PAPER_FORWARD_REASON = "fresh_forward_observation"
CANDIDATE_PAPER_BACKFILL_ENTRY_REASON = "downtime_backfill_entry"
CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON = "downtime_backfill_position_management"
CANDIDATE_PAPER_EXECUTION_MANIFEST: dict[str, Any] = {
    "schema": CANDIDATE_PAPER_EXECUTION_SCHEMA,
    "market_data": "closed_base_timeframe_bars",
    "event_order": ("information_available_at_close_then_shorter_timeframe_then_artifact_order"),
    "signal_fill": (
        "fresh_latest_signal_uses_public_observation_quote;"
        "historical_next_open_replay_is_non_promotable"
    ),
    "exit_observation": (
        "fresh_closed_bar_at_wall_clock_observation;any_position_catch_up_is_non_promotable"
    ),
    "partial_entry_bar": "excluded_from_exit_evaluation",
    "cursor": "digest_state_per_strategy_atomic_cursor_v2",
    "exit_accounting": "keyed_write_ahead_log",
    "backlog": (
        "bounded_fail_closed_without_cursor_advance;within_bound_replay_is_explicitly_quarantined"
    ),
    "broker": "forbidden_paper_only",
}


def _sha256_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_candidate_paper_engine_digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("candidate paper engine digest must be a sha256 digest")
    return value


def candidate_paper_engine_digest(*, runtime_digest: str | None = None) -> str:
    """Bind replay semantics to the complete executable source/environment."""

    runtime_digest = runtime_digest or execution_engine_digest()
    validate_candidate_paper_engine_digest(runtime_digest)
    return _sha256_digest(
        {
            "candidate_paper_execution": CANDIDATE_PAPER_EXECUTION_MANIFEST,
            "runtime_execution_engine_digest": runtime_digest,
        }
    )
