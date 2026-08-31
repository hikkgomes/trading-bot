"""Command-line access to the PostgreSQL control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.database import PlatformDatabase
from src.services.config import load_platform_config, load_split_configuration
from src.services.control_api import ControlMode, DatabaseControlPlane
from src.services.health import DatabaseHeartbeatStore
from src.services.runtime import utc_now


def run(args: argparse.Namespace) -> int:
    config = load_platform_config(args.config)
    configuration = load_split_configuration(args.config.parent)
    database = PlatformDatabase(config.database_url())
    try:
        control = DatabaseControlPlane(
            database.engine,
            DatabaseHeartbeatStore(database.engine),
            configuration=configuration,
        )
        result = _execute(control, args)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    finally:
        database.dispose()


def _execute(control: DatabaseControlPlane, args: argparse.Namespace) -> object:
    if args.command == "status":
        return control.status()
    target = args.target
    if args.command == "suspend-strategy":
        if not args.strategy_id:
            raise ValueError("--strategy-id is required for suspend-strategy")
        target = f"strategy:{args.strategy_id}"
    operator = _required(args.operator, "--operator")
    reason = _required(args.reason, "--reason")
    changed_at = args.changed_at or utc_now()
    if args.command == "cancel-all-entry-orders":
        return control.cancel_all_entry_orders(
            target=target,
            reason_code=reason,
            requested_by=operator,
            changed_at=changed_at,
        )
    if args.command == "emergency-flatten" and not args.confirm:
        raise ValueError("--confirm is required for emergency-flatten")
    mode = {
        "pause": ControlMode.MANAGEMENT_ONLY,
        "resume": ControlMode.RUN,
        "block-new-risk": ControlMode.BLOCK_NEW_RISK,
        "management-only": ControlMode.MANAGEMENT_ONLY,
        "suspend-strategy": ControlMode.SUSPENDED,
        "emergency-flatten": ControlMode.EMERGENCY_FLATTEN,
    }[args.command]
    return control.set_mode(
        target=target,
        mode=mode,
        reason_code=reason,
        requested_by=operator,
        changed_at=changed_at,
        confirm_resume=args.command == "resume" and args.confirm,
    ).__dict__


def _required(value: str | None, flag: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{flag} is required")
    return value.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operate the PostgreSQL platform control plane.")
    parser.add_argument(
        "command",
        choices=(
            "status",
            "pause",
            "resume",
            "block-new-risk",
            "management-only",
            "suspend-strategy",
            "emergency-flatten",
            "cancel-all-entry-orders",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config/platform.json"))
    parser.add_argument("--target", default="global")
    parser.add_argument("--strategy-id")
    parser.add_argument("--operator")
    parser.add_argument("--reason")
    parser.add_argument("--changed-at")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
