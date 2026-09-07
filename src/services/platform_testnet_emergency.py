"""Dedicated connected-testnet emergency execution rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.domain._codec import canonical_hash
from src.services.platform_testnet_connected import run_connected_testnet_rehearsal


class EmergencyRehearsalError(RuntimeError):
    """The emergency state machine did not prove recovery and reconciliation."""


@dataclass(frozen=True)
class EmergencyState:
    state: str
    order_id: str
    client_order_id: str
    attempts: int
    reconciled_flat: bool
    history: tuple[str, ...] = ()


class EmergencyStateMachine:
    """Exercise ambiguous submission without treating it as a success."""

    STATES = (
        "requested",
        "entries_blocked",
        "exchange_state_refreshed",
        "close_planned",
        "close_submitted",
        "close_acknowledged",
        "partial_fill_observed",
        "fills_observed",
        "reconciled_flat",
        "completed",
    )

    @staticmethod
    def _is_filled(acknowledgement: Any) -> bool:
        status = str(getattr(acknowledgement, "status", "open")).casefold()
        return status in {"filled", "closed", "reconciled"}

    @staticmethod
    def _is_partial(acknowledgement: Any) -> bool:
        status = str(getattr(acknowledgement, "status", "open")).casefold()
        return status in {"partial", "partially_filled", "partially-filled"}

    @staticmethod
    def _submit_or_recover(
        *,
        submit: Callable[[], Any],
        query: Callable[[str], Any],
        client_order_id: str,
        ambiguous_submission: bool,
    ) -> tuple[Any, bool]:
        if ambiguous_submission:
            return query(client_order_id), True
        try:
            return submit(), False
        except TimeoutError:
            return query(client_order_id), True
        except Exception as exc:
            raise EmergencyRehearsalError(
                f"emergency submission failed before recovery: {type(exc).__name__}"
            ) from exc

    def _observe_fill(
        self,
        acknowledgement: Any,
        *,
        query: Callable[[str], Any],
        client_order_id: str,
    ) -> tuple[bool, bool]:
        partial = self._is_partial(acknowledgement)
        if self._is_filled(acknowledgement):
            return True, partial
        queried = query(client_order_id)
        return self._is_filled(queried), partial or self._is_partial(queried)

    def _process_attempt(
        self,
        *,
        state: str,
        acknowledged: bool,
        filled: bool,
        ambiguous_submission: bool,
        submit: Callable[[], Any],
        query: Callable[[str], Any],
        client_order_id: str,
        advance: Callable[[str], None],
        reconcile: Callable[[], bool],
    ) -> tuple[str, bool, bool, bool, bool]:
        acknowledgement = query(client_order_id) if acknowledged else None
        if not acknowledged:
            if state == "close_planned":
                advance("close_submitted")
                state = "close_submitted"
            acknowledgement, ambiguous_submission = self._submit_or_recover(
                submit=submit,
                query=query,
                client_order_id=client_order_id,
                ambiguous_submission=ambiguous_submission,
            )
            acknowledged = acknowledgement is not None
            if not acknowledged:
                return state, acknowledged, filled, ambiguous_submission, False
            advance("close_acknowledged")
            state = "close_acknowledged"
        filled, partial = self._observe_fill(
            acknowledgement,
            query=query,
            client_order_id=client_order_id,
        )
        if partial and state == "close_acknowledged":
            advance("partial_fill_observed")
            state = "partial_fill_observed"
        if filled and state != "fills_observed":
            advance("fills_observed")
            state = "fills_observed"
        return state, acknowledged, filled, ambiguous_submission, filled and reconcile()

    def run(
        self,
        *,
        order_id: str,
        client_order_id: str,
        submit: Callable[[], Any],
        query: Callable[[str], Any],
        reconcile: Callable[[], bool],
        maximum_attempts: int = 8,
    ) -> EmergencyState:
        state = "requested"
        history = [state]
        attempts = 0
        acknowledged = False
        filled = False
        ambiguous_submission = False

        def advance(next_state: str) -> None:
            nonlocal state
            state = next_state
            history.append(next_state)

        advance("entries_blocked")
        advance("exchange_state_refreshed")
        advance("close_planned")
        while attempts < maximum_attempts:
            attempts += 1
            state, acknowledged, filled, ambiguous_submission, reconciled = self._process_attempt(
                state=state,
                acknowledged=acknowledged,
                filled=filled,
                ambiguous_submission=ambiguous_submission,
                submit=submit,
                query=query,
                client_order_id=client_order_id,
                advance=advance,
                reconcile=reconcile,
            )
            if reconciled:
                advance("reconciled_flat")
                advance("completed")
                return EmergencyState(
                    state="completed",
                    order_id=order_id,
                    client_order_id=client_order_id,
                    attempts=attempts,
                    reconciled_flat=True,
                    history=tuple(history),
                )
        return EmergencyState(
            state=state,
            order_id=order_id,
            client_order_id=client_order_id,
            attempts=attempts,
            reconciled_flat=False,
            history=tuple(history),
        )


def run_emergency_testnet_rehearsal(
    *,
    database_url: str,
    config_path: Path,
    product_id: str,
    notional_usd: float,
    confirm: bool,
) -> dict[str, Any]:
    """Run the connected entry/protection/emergency/close rehearsal."""

    report = run_connected_testnet_rehearsal(
        database_url=database_url,
        config_path=config_path,
        product_id=product_id,
        notional_usd=notional_usd,
        confirm=confirm,
    )
    if (
        report.get("emergency_reduction_tested") is not True
        or report.get("flat_reconciliation") is not True
    ):
        raise EmergencyRehearsalError(
            "connected report did not prove emergency reduction and flat state"
        )
    state = _accepted_then_timeout_case()
    if not state.reconciled_flat or state.state != "completed":
        raise EmergencyRehearsalError("ambiguous emergency submission case did not reconcile flat")
    enriched = {
        **report,
        "schema": "platform.connected-emergency-testnet-report/v1",
        "emergency_rehearsal": {
            "accepted_then_timeout_case": True,
            "deterministic_client_id_recovery": True,
            "authenticated_flat_reconciliation": True,
            "state": state.state,
            "attempts": state.attempts,
            "history": list(state.history),
        },
    }
    unsigned = {
        key: value for key, value in enriched.items() if key not in {"report_hash", "signature"}
    }
    report_hash = canonical_hash(unsigned)
    enriched["report_hash"] = report_hash
    signing_key = os.environ.get("TRADING_PLATFORM_REHEARSAL_SIGNING_KEY", "")
    if signing_key:
        enriched["signature"] = hmac.new(
            signing_key.encode(), report_hash.encode(), hashlib.sha256
        ).hexdigest()
    return enriched


def _accepted_then_timeout_case() -> EmergencyState:
    exchange_status = "open"
    submissions = 0

    def submit() -> Any:
        nonlocal exchange_status, submissions
        submissions += 1
        exchange_status = "filled"
        raise TimeoutError("exchange accepted the order before the response timed out")

    def query(_client_order_id: str) -> Any:
        return SimpleNamespace(status=exchange_status)

    def reconcile() -> bool:
        return exchange_status == "filled"

    state = EmergencyStateMachine().run(
        order_id="emergency-rehearsal-order",
        client_order_id="emergency-rehearsal-client",
        submit=submit,
        query=query,
        reconcile=reconcile,
    )
    if submissions != 1:
        raise EmergencyRehearsalError("ambiguous emergency submission was retried")
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run connected Binance emergency rehearsal.")
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--product", default="active_income")
    parser.add_argument("--notional-usd", type=float, default=10.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    import json

    database_url = os.environ.get("TRADING_PLATFORM_DATABASE_URL", "")
    if not database_url:
        parser.error("TRADING_PLATFORM_DATABASE_URL is required")
    print(
        json.dumps(
            run_emergency_testnet_rehearsal(
                database_url=database_url,
                config_path=args.config,
                product_id=args.product,
                notional_usd=args.notional_usd,
                confirm=args.confirm,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
