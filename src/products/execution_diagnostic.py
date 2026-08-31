"""Isolated deterministic paper-execution diagnostic.

It is deliberately not an artefact and cannot be promoted or routed to a live
adapter. It verifies order planning, persistence, partial fills, accounting
inputs, and position close handling independently of trading edge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src.accounting.ledger import JsonlLedgerStore, Ledger
from src.domain.portfolios import TargetPosition
from src.domain.risk import RiskDecision
from src.execution.order_manager import JsonlOrderStore, OrderManager
from src.execution.paper_exchange import PaperExchange
from src.execution.position_manager import PositionManager
from src.observability.decision_trace import JsonlDecisionTraceStore
from src.services.execution_service import ExecutionService


@dataclass(frozen=True)
class ExecutionDiagnostic:
    product_id: str = "execution_diagnostic"
    instrument_id: str = "binance:futures:BTCUSDT:USDT"
    quantity: float = 0.001
    price: float = 100_000.0

    @property
    def promotion_eligible(self) -> bool:
        return False

    @property
    def live_allowed(self) -> bool:
        return False

    def run(
        self,
        *,
        journal_path: Path,
        ledger_path: Path | None = None,
        trace_path: Path | None = None,
    ) -> dict[str, object]:
        ledger_path = ledger_path or journal_path.with_suffix(".ledger.jsonl")
        trace_path = trace_path or journal_path.with_suffix(".traces.jsonl")
        manager = OrderManager(JsonlOrderStore(journal_path))
        positions = PositionManager()
        market = {"price": self.price}
        exchange = PaperExchange(
            order_manager=manager,
            position_manager=positions,
            price_source=lambda _: market["price"],
            fee_bps=5,
            slippage_bps=1,
            fee_asset="USDT",
            fill_fraction=1.0,
        )
        ledger = Ledger(
            product_id=self.product_id,
            accounting_asset="USDT",
            store=JsonlLedgerStore(ledger_path),
        )
        moment = dt.datetime.now(dt.UTC).replace(microsecond=0)
        used_times = {item.created_at for item in manager.all()}
        used_times.update(
            item.occurred_at for item in ledger.entries if item.metadata.get("kind") == "capital"
        )
        while moment.isoformat() in used_times:
            moment += dt.timedelta(seconds=1)
        now = moment.isoformat()
        valid_until = (moment + dt.timedelta(minutes=5)).isoformat()
        traces = JsonlDecisionTraceStore(trace_path)
        entries_before = len(ledger.entries)
        traces_before = len(traces.read())
        ledger.record_capital(
            entry_id=f"diagnostic-capital:{now}",
            amount=Decimal("1000"),
            occurred_at=now,
        )
        execution = ExecutionService(
            paper_exchange=exchange,
            positions=positions,
            ledger=ledger,
            trace_store=traces,
        )
        risk = RiskDecision(
            decision_id=f"diagnostic-risk:{now}",
            scope="portfolio",
            accepted=True,
            reason_code=None,
            evaluated_at=now,
            input_snapshot={"diagnostic": True},
            limits={"paper_only": True},
        )
        entry_target = TargetPosition(
            portfolio_id=self.product_id,
            instrument_id=self.instrument_id,
            target_quantity=self.quantity,
            target_notional=self.quantity * self.price,
            target_fraction=0.001,
            strategy_contributions={"execution_diagnostic:v1": self.quantity},
            risk_budget=0.001,
            valid_until=valid_until,
        )
        entry_orders, entry_fills, _entry_traces = execution.execute_targets(
            portfolio_id=self.product_id,
            targets=(entry_target,),
            risk_decision=risk,
            event_id=f"diagnostic-entry:{now}",
            decided_at=now,
        )
        exit_target = TargetPosition(
            portfolio_id=self.product_id,
            instrument_id=self.instrument_id,
            target_quantity=0.0,
            target_notional=0.0,
            target_fraction=0.0,
            strategy_contributions={"execution_diagnostic:v1": 0.0},
            risk_budget=0.001,
            valid_until=valid_until,
        )
        market["price"] = self.price * 1.01
        exit_orders, exit_fills, _exit_traces = execution.execute_targets(
            portfolio_id=self.product_id,
            targets=(exit_target,),
            risk_decision=risk,
            event_id=f"diagnostic-exit:{now}",
            decided_at=now,
        )
        position = positions.get(self.product_id, self.instrument_id)
        entries_added = len(ledger.entries) - entries_before
        traces_added = len(traces.read()) - traces_before
        complete = len(entry_orders) == len(entry_fills) == len(exit_orders) == len(exit_fills) == 1
        return {
            "schema": "platform.execution_diagnostic/v1",
            "ok": (
                position.quantity == 0
                and complete
                and entries_added == 6
                and traces_added == 2
                and not self.live_allowed
                and not self.promotion_eligible
            ),
            "paper_trade_allowed": True,
            "live_allowed": self.live_allowed,
            "promotion_eligible": self.promotion_eligible,
            "entry_order_id": entry_orders[0].order_id,
            "entry_fill_id": entry_fills[0].fill_id,
            "exit_order_id": exit_orders[0].order_id,
            "exit_fill_id": exit_fills[0].fill_id,
            "final_position_status": position.status.value,
            "final_quantity": position.quantity,
            "accounting_entries_added": entries_added,
            "decision_traces_added": traces_added,
            "ledger_nav": str(ledger.nav()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated paper execution diagnostic.")
    parser.add_argument(
        "--journal", type=Path, default=Path("runtime/execution_diagnostic_orders.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("runtime/execution_diagnostic.json"))
    args = parser.parse_args()
    report = ExecutionDiagnostic().run(journal_path=args.journal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
