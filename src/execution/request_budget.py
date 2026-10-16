"""Single-host Binance IP budgets shared by public data and execution processes."""

import json
import math
import time
from pathlib import Path

from src.data.atomic import write_json_atomic
from src.data.locking import acquire_file_update_lock


class SharedRequestBudget:
    def __init__(self, path: Path, *, clock=time.time):
        self.path = path
        self.clock = clock

    def reserve(self, cost: float, *, background: bool, funding: bool) -> float:
        maximum = 100 if background else 200
        if not math.isfinite(cost) or not 0 < cost <= maximum:
            raise ValueError(f"request cost must be in (0, {maximum}]")
        with acquire_file_update_lock(self.path, label="exchange IP budget"):
            now = self.clock()
            state = self._read(now)
            # Retain half the burst allowance for orders, cancellation and recovery.
            delay = self._consume(
                state, "weight", cost, now, 200, 1000 / 60, reserve=100 if background else 0
            )
            if funding:
                delay = max(delay, self._consume(state, "funding", 1, now, 100, 1.2))
            delay = max(delay, float(state.get("blocked_until", 0)) - now)
            if background:
                delay = max(delay, float(state.get("foreground_until", 0)) - now)
            elif delay > 0:
                state["foreground_until"] = now + delay + 1
            if delay <= 0:
                state["weight"]["tokens"] -= cost
                if funding:
                    state["funding"]["tokens"] -= 1
            write_json_atomic(self.path, state)
            return max(0.0, delay)

    @staticmethod
    def _consume(state, name, cost, now, capacity, rate, *, reserve=0):
        bucket = state.setdefault(name, {"tokens": capacity, "updated": now})
        tokens, updated = float(bucket["tokens"]), float(bucket["updated"])
        if not all(math.isfinite(value) for value in (tokens, updated)):
            raise ValueError("exchange budget state is invalid")
        bucket["tokens"] = min(capacity, tokens + max(0, now - updated) * rate)
        bucket["updated"] = now
        return max(0, (cost + reserve - bucket["tokens"]) / rate)

    def backoff(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("exchange retry delay must be finite and non-negative")
        with acquire_file_update_lock(self.path, label="exchange IP budget"):
            now = self.clock()
            state = self._read(now)
            state["blocked_until"] = max(float(state.get("blocked_until", 0)), now + seconds)
            write_json_atomic(self.path, state)

    def _read(self, now):
        if self.path.is_symlink():
            raise ValueError("exchange budget must not be a symlink")
        if not self.path.exists():
            return {"schema": "exchange_request_budget/v1", "created_at": now}
        state = json.loads(self.path.read_text())
        if state.get("schema") != "exchange_request_budget/v1":
            raise ValueError("exchange budget state has an unsupported schema")
        return state
