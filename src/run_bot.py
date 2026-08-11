"""Paper-trading executor.

Reads active_strategies.json (produced by src.export_strategies from a search
output directory) and runs one evaluation cycle per invocation. Designed to be
invoked directly for manual checks or by the autopilot runtime. State is
persisted to a configured JSON file so each cycle is stateless from the
process's point of view.
"""

import argparse
import copy
import datetime
import hashlib
import inspect
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import build_binance_indicator_dataset as bbid
from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.predicates import (
    entry_mask,
    entry_score_series,
    hypothesis_history_requirements,
    predicate_mask,
)
from src.alpha.frozen_gradient_boosting import FrozenGradientBoostingModel
from src.autopilot.approvals import (
    artifact_digest as approval_artifact_digest,
)
from src.autopilot.approvals import (
    strategy_fingerprint as approval_strategy_fingerprint,
)
from src.autopilot.candidate_evidence import (
    CANDIDATE_PAPER_BACKFILL_ENTRY_REASON,
    CANDIDATE_PAPER_BACKFILL_FILL_SOURCE,
    CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON,
    CANDIDATE_PAPER_EXECUTION_SCHEMA,
    CANDIDATE_PAPER_FORWARD_FILL_SOURCE,
    CANDIDATE_PAPER_FORWARD_REASON,
    candidate_paper_engine_digest,
)
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.autopilot.portfolio import AlphaForecast, aggregate_forecasts, forecast_from_strategy
from src.build_dataset import TIMEFRAME_SECONDS
from src.config import PROJECT_ROOT
from src.discover_patterns import Condition, condition_mask
from src.execution.broker import (
    Broker,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    ProtectiveOrder,
    ProtectiveOrderStatus,
)
from src.execution.config import ACCOUNT_FINGERPRINT_PREFIX
from src.trade_utils import gross_return_for_pnl_unit

LOGGER = logging.getLogger("trading_bot")

DEFAULT_STRATEGIES_PATH = PROJECT_ROOT / "outputs" / "active_strategies.json"
STATE_FILE_PATH = PROJECT_ROOT / "outputs" / "bot_state.json"
TRADE_LOG_PATH = PROJECT_ROOT / "outputs" / "paper_trades.csv"
DEFAULT_STARTING_EQUITY = 1_000.0
ACTIVE_INCOME_MIN_DSR = 0.60
DRIFT_MIN_TRADES = 10
DRIFT_WINDOW_TRADES = 20
DRIFT_Z_THRESHOLD = -2.0
REGIME_MAYER_TOP = 2.4
MAX_EQUITY_DRAWDOWN_BY_OBJECTIVE = {
    "active_income": 0.10,
    "btc_accumulation": 0.05,
}
DEFAULT_MAX_EQUITY_DRAWDOWN = 0.05
LIVE_SPOT_QUOTE_PROCEEDS_MAX_SHORTFALL_FRACTION = 0.01
LIVE_SPOT_BASE_BALANCE_MAX_FEE_FRACTION = 0.01
CANDIDATE_REPLAY_SCHEMA_VERSION = 2
CANDIDATE_REPLAY_CURSOR_KEY = "candidate_replay_cursor_by_strategy"
CANDIDATE_REPLAY_PENDING_KEY = "candidate_replay_pending_entries_by_strategy"
DECISION_TRACE_SCHEMA = "autopilot.decision_trace/v1"
TRADE_LOG_COLUMNS = (
    "exit_event_id",
    "strategy_id",
    "strategy_fingerprint",
    "artifact_digest",
    "alpha_source_id",
    "alpha_product",
    "alpha_market",
    "alpha_symbol",
    "alpha_score",
    "alpha_expected_return",
    "alpha_confidence",
    "alpha_horizon_seconds",
    "candidate_paper_execution_schema",
    "candidate_paper_engine_digest",
    "candidate_paper_evidence_eligible",
    "candidate_paper_evidence_reason",
    "candidate_paper_entry_fill_source",
    "candidate_paper_observed_at",
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "exit_reason",
    "gross_return",
    "transaction_cost_fraction",
    "transaction_cost_source",
    "accounting_return_source",
    "accounting_adjustment_fraction",
    "net_return",
    "sized_return",
    "position_size",
    "equity_after",
)
BROKER_TRADE_LOG_COLUMNS = (
    "broker_symbol",
    "broker_entry_balance",
    "broker_exit_balance",
    "broker_balance_return",
    "broker_entry_fee",
    "broker_exit_qty",
    "broker_exit_price",
    "broker_exit_fee",
)
BROKER_POSITION_REQUIRED_KEYS = (
    "broker_symbol",
    "broker_side",
    "broker_qty",
    "broker_requested_qty",
    "broker_fill_ratio",
    "broker_entry_price",
    "broker_entry_fee",
)
BROKER_PROTECTIVE_STOP_REQUIRED_KEYS = (
    "broker_stop_order_id",
    "broker_stop_client_id",
    "broker_stop_trigger_price",
)
BROKER_LIVE_FUTURES_ACCOUNTING_REQUIRED_KEYS = ("broker_entry_balance",)
BROKER_LIVE_SPOT_ACCOUNTING_REQUIRED_KEYS = (
    "broker_entry_base_qty_before",
    "broker_entry_base_qty_after",
    "broker_entry_quote_balance_before",
    "broker_entry_quote_balance_after",
    "broker_entry_quote_value",
    "broker_entry_quote_value_source",
    "broker_exit_sizing",
)
REQUIRED_RISK_KEYS = (
    "risk_per_trade",
    "daily_stop_loss",
    "max_consecutive_losses",
    "cooldown_bars",
    "max_position_fraction",
    "max_trades_per_day",
)
REQUIRED_FEE_KEYS = ("fee_bps", "slippage_bps")
QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
CONDITION_KINDS = {
    "value_le",
    "value_ge",
    "delta_le",
    "delta_ge",
    "cross_above",
    "cross_below",
    "ratio_le",
    "ratio_ge",
}
SLOPE_KIND_RE = re.compile(r"^slope_(\d+)_(le|ge)$")
DIVERGENCE_KIND_RE = re.compile(r"^divergence_(bull|bear)_(\d+)$")
CLIENT_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,36}$")
PENDING_ORDER_REQUIRED_KEYS = (
    "version",
    "strategy_id",
    "stage",
    "intent_ref",
    "symbol",
    "side",
    "qty",
    "order_type",
    "reduce_only",
    "client_id",
    "broker_account_fingerprint",
    "created_ts",
)
EXIT_ACCOUNTING_INTENT_REQUIRED_KEYS = (
    "version",
    "phase",
    "exit_event_id",
    "strategy_id",
    "created_at",
    "state_before_digest",
    "position_digest",
    "broker_flat_proven",
    "trade_data",
    "state_after",
    "payload_digest",
)


def _split_symbol(symbol: str) -> tuple[str, str, str | None]:
    raw = re.sub(r"\s+", "", str(symbol or "").upper())
    if not raw:
        raise ValueError("symbol must be non-empty.")
    settlement = None
    pair = raw
    if ":" in pair:
        pair, settlement = pair.split(":", 1)
        if not settlement:
            raise ValueError(f"symbol {symbol!r} has an empty settlement asset.")
    if "/" in pair:
        base, quote = pair.split("/", 1)
    else:
        compact = re.sub(r"[^A-Z0-9]", "", pair)
        base, quote = compact, ""
        for known_quote in QUOTE_ASSETS:
            if compact.endswith(known_quote) and len(compact) > len(known_quote):
                base, quote = compact[: -len(known_quote)], known_quote
                break
    if not base or not quote:
        raise ValueError(f"symbol {symbol!r} must include base and quote assets.")
    return base, quote, settlement


def _binance_rest_symbol(symbol: str) -> str:
    base, quote, _settlement = _split_symbol(symbol)
    return f"{base}{quote}"


def _symbols_match(left: str, right: str) -> bool:
    left_base, left_quote, left_settlement = _split_symbol(left)
    right_base, right_quote, right_settlement = _split_symbol(right)
    if (left_base, left_quote) != (right_base, right_quote):
        return False
    if left_settlement is not None and right_settlement is not None:
        return left_settlement == right_settlement
    return True


def _ordered_trade_log_columns(existing_columns, incoming_columns) -> list[str]:
    seen = set(existing_columns) | set(incoming_columns)
    preferred = list(TRADE_LOG_COLUMNS) + list(BROKER_TRADE_LOG_COLUMNS)
    columns = [column for column in preferred if column in seen]
    columns.extend(column for column in existing_columns if column not in columns)
    columns.extend(column for column in incoming_columns if column not in columns)
    return columns


def _canonical_json_digest(value: object, *, label: str) -> str:
    """Return a stable digest, rejecting state that JSON cannot reproduce."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be finite JSON data.") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_strategy_risk(strategy: dict) -> dict:
    strategy_id = strategy.get("id", "<unknown>")
    risk = strategy.get("risk")
    if not isinstance(risk, dict):
        raise ValueError(f"Strategy {strategy_id} risk must be an object.")
    missing = [key for key in REQUIRED_RISK_KEYS if key not in risk]
    if missing:
        raise ValueError(
            f"Strategy {strategy_id} risk is missing required key(s): {', '.join(missing)}."
        )
    if risk.get("max_trades_per_day") is None:
        raise ValueError(f"Strategy {strategy_id} max_trades_per_day must be a positive integer.")
    try:
        max_losses_raw = float(risk["max_consecutive_losses"])
        cooldown_raw = float(risk["cooldown_bars"])
        max_trades_raw = float(risk["max_trades_per_day"])
        normalized = {
            "risk_per_trade": float(risk["risk_per_trade"]),
            "daily_stop_loss": float(risk["daily_stop_loss"]),
            "max_consecutive_losses": max_losses_raw,
            "cooldown_bars": cooldown_raw,
            "max_trades_per_day": max_trades_raw,
            "max_position_fraction": float(risk["max_position_fraction"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} risk has non-numeric value: {exc}") from exc
    for key, value in normalized.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    for key in ("max_consecutive_losses", "cooldown_bars", "max_trades_per_day"):
        value = normalized[key]
        if value is not None and value != int(value):
            raise ValueError(f"Strategy {strategy_id} {key} must be an integer.")
        if value is not None:
            normalized[key] = int(value)
    if normalized["risk_per_trade"] <= 0:
        raise ValueError(f"Strategy {strategy_id} risk_per_trade must be positive.")
    if normalized["max_position_fraction"] <= 0 or normalized["max_position_fraction"] > 1:
        raise ValueError(f"Strategy {strategy_id} max_position_fraction must be > 0 and <= 1.")
    if normalized["daily_stop_loss"] >= 0:
        raise ValueError(f"Strategy {strategy_id} daily_stop_loss must be negative.")
    if normalized["max_consecutive_losses"] <= 0:
        raise ValueError(f"Strategy {strategy_id} max_consecutive_losses must be positive.")
    if normalized["cooldown_bars"] < 0:
        raise ValueError(f"Strategy {strategy_id} cooldown_bars must be non-negative.")
    if normalized["max_trades_per_day"] <= 0:
        raise ValueError(f"Strategy {strategy_id} max_trades_per_day must be positive.")
    return normalized


def _normalize_strategy_fees(strategy: dict) -> dict:
    strategy_id = strategy.get("id", "<unknown>")
    fees = strategy.get("fees")
    if not isinstance(fees, dict):
        raise ValueError(f"Strategy {strategy_id} fees must be an object.")
    missing = [key for key in REQUIRED_FEE_KEYS if key not in fees]
    if missing:
        raise ValueError(
            f"Strategy {strategy_id} fees is missing required key(s): {', '.join(missing)}."
        )
    try:
        normalized = {
            "fee_bps": float(fees["fee_bps"]),
            "slippage_bps": float(fees["slippage_bps"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} fees has non-numeric value: {exc}") from exc
    for key, value in normalized.items():
        if not math.isfinite(value):
            raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    if normalized["fee_bps"] < 0:
        raise ValueError(f"Strategy {strategy_id} fee_bps must be non-negative.")
    if normalized["slippage_bps"] < 0:
        raise ValueError(f"Strategy {strategy_id} slippage_bps must be non-negative.")
    return normalized


def _normalize_positive_float(strategy: dict, key: str) -> float:
    strategy_id = strategy.get("id", "<unknown>")
    try:
        value = float(strategy[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} {key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    if value <= 0:
        raise ValueError(f"Strategy {strategy_id} {key} must be positive.")
    strategy[key] = value
    return value


def _normalize_positive_int(strategy: dict, key: str) -> int:
    strategy_id = strategy.get("id", "<unknown>")
    try:
        value = float(strategy[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} {key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    if value != int(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be an integer.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"Strategy {strategy_id} {key} must be positive.")
    strategy[key] = integer
    return integer


def _normalize_optional_probability(strategy: dict, key: str) -> float | None:
    strategy_id = strategy.get("id", "<unknown>")
    if strategy.get(key) is None:
        strategy[key] = None
        return None
    try:
        value = float(strategy[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} {key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    if not 0 < value < 1:
        raise ValueError(f"Strategy {strategy_id} {key} must be between 0 and 1.")
    strategy[key] = value
    return value


def _finite_metric(metrics: dict, key: str, strategy_id: str) -> float:
    try:
        value = float(metrics[key])
    except KeyError as exc:
        raise ValueError(
            f"Strategy {strategy_id} metrics is missing required key {key!r}."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} {key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    return value


def _condition_kind_supported(kind: str) -> bool:
    return kind in CONDITION_KINDS or bool(
        SLOPE_KIND_RE.match(kind) or DIVERGENCE_KIND_RE.match(kind)
    )


def _normalize_conditions(strategy: dict) -> list[Condition]:
    strategy_id = strategy.get("id", "<unknown>")
    conditions = strategy.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"Strategy {strategy_id} conditions must be a non-empty list.")
    normalized = []
    for index, payload in enumerate(conditions):
        if not isinstance(payload, dict):
            raise ValueError(f"Strategy {strategy_id} condition[{index}] must be an object.")
        feature = payload.get("feature")
        if not isinstance(feature, str) or not feature:
            raise ValueError(
                f"Strategy {strategy_id} condition[{index}].feature must be a non-empty string."
            )
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                f"Strategy {strategy_id} condition[{index}].kind must be a non-empty string."
            )
        if not _condition_kind_supported(kind):
            raise ValueError(
                f"Strategy {strategy_id} condition[{index}].kind is unsupported: {kind!r}."
            )
        try:
            threshold = float(payload["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Strategy {strategy_id} condition[{index}].threshold must be numeric."
            ) from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Strategy {strategy_id} condition[{index}].threshold must be finite.")
        payload = dict(payload)
        payload["threshold"] = threshold
        if kind in {"cross_above", "cross_below", "ratio_le", "ratio_ge"}:
            feature_b = payload.get("feature_b")
            if not isinstance(feature_b, str) or not feature_b:
                raise ValueError(
                    f"Strategy {strategy_id} condition[{index}].feature_b is required for {kind}."
                )
        normalized.append(Condition(**payload))
    strategy["conditions"] = [condition.__dict__ for condition in normalized]
    return normalized


def compute_macro_step_aside(
    close: pd.Series,
    mayer_top: float = REGIME_MAYER_TOP,
    trend_ema: int = 200,
    mayer_window: int = 200,
    pi_fast: int = 111,
    pi_slow: int = 350,
) -> tuple[bool, dict]:
    """BTC macro "step aside" state from a *daily* close series (lookahead-safe).

    Returns ``(step_aside, detail)``. Risk-off when the macro trend breaks (close
    below the long EMA), the market is overheated (Mayer Multiple > ``mayer_top``),
    or a Pi-Cycle Top cross prints. Mirrors the ``btc_cycle_guard`` strategy; used
    by the bot to gate new long entries during the accumulation regime.
    """
    close = close.astype(float)
    detail: dict = {"bars": int(len(close))}
    if len(close) < trend_ema:
        detail["reason"] = "insufficient_daily_history"
        return False, detail

    ema = close.ewm(span=trend_ema, adjust=False, min_periods=trend_ema).mean()
    mayer = close / close.rolling(mayer_window).mean()
    sma_fast = close.rolling(pi_fast).mean()
    sma_slow_x2 = 2.0 * close.rolling(pi_slow).mean()

    last_close = close.iloc[-1]
    trend_break = bool(pd.notna(ema.iloc[-1]) and last_close < ema.iloc[-1])
    overheated = bool(pd.notna(mayer.iloc[-1]) and mayer.iloc[-1] > mayer_top)
    pi_top = bool(
        len(close) >= pi_slow + 1
        and pd.notna(sma_slow_x2.iloc[-1])
        and sma_fast.iloc[-1] > sma_slow_x2.iloc[-1]
        and sma_fast.iloc[-2] <= sma_slow_x2.iloc[-2]
    )
    detail.update(
        close=float(last_close),
        mayer=float(mayer.iloc[-1]) if pd.notna(mayer.iloc[-1]) else None,
        trend_break=trend_break,
        overheated=overheated,
        pi_cycle_top=pi_top,
    )
    return bool(trend_break or overheated or pi_top), detail


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _utc_date_today() -> datetime.date:
    return datetime.datetime.now(datetime.UTC).date()


def _utc_now_text() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def _reject_symlink_path(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")


class PaperTradingBot:
    def __init__(
        self,
        strategies_path: Path = DEFAULT_STRATEGIES_PATH,
        state_file: Path = STATE_FILE_PATH,
        trade_log: Path = TRADE_LOG_PATH,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
        regime_guard: bool = False,
        regime_mayer_top: float = REGIME_MAYER_TOP,
        broker: Broker | None = None,
        symbol: str = "BTCUSDT",
        market: str = "futures",
        objective: str | None = None,
        base_asset: str | None = None,
        live_gate_approved: bool = False,
        allow_entries: bool = True,
        artifact_payload: dict | None = None,
        pre_entry_gate=None,
        portfolio_gate=None,
    ):
        self.strategies_path = strategies_path
        self.state_file = state_file
        self.trade_log = trade_log
        _reject_symlink_path(self.state_file, "State file")
        _reject_symlink_path(self.trade_log, "Trade log")
        self.starting_equity = float(starting_equity)
        self.regime_guard = bool(regime_guard)
        self.regime_mayer_top = float(regime_mayer_top)
        self.broker = broker
        if (
            broker is not None
            and bool(getattr(getattr(broker, "config", None), "live", False))
            and not live_gate_approved
        ):
            raise RuntimeError(
                "Live broker injection requires the autopilot approval, preflight, and testnet rehearsal gates."
            )
        _, _, symbol_settlement = _split_symbol(symbol)
        if market not in {"futures", "spot"}:
            raise ValueError("market must be 'futures' or 'spot'.")
        if market == "spot" and symbol_settlement is not None:
            raise ValueError("spot market symbol must not include a settlement asset.")
        self.symbol = str(symbol)
        self.data_symbol = _binance_rest_symbol(symbol)
        self.market = market
        if objective is not None and objective not in {"btc_accumulation", "active_income"}:
            raise ValueError("objective must be 'btc_accumulation', 'active_income', or None.")
        self.objective = objective
        self.base_asset = str(base_asset).upper() if base_asset is not None else None
        if not isinstance(allow_entries, bool):
            raise ValueError("allow_entries must be a boolean.")
        self.allow_entries = allow_entries
        if pre_entry_gate is not None and not callable(pre_entry_gate):
            raise ValueError("pre_entry_gate must be callable when supplied.")
        self.pre_entry_gate = pre_entry_gate
        if portfolio_gate is not None and not callable(portfolio_gate):
            raise ValueError("portfolio_gate must be callable when supplied.")
        self.portfolio_gate = portfolio_gate
        self._artifact_payload: dict | None = None
        if artifact_payload is not None:
            if not isinstance(artifact_payload, dict):
                raise ValueError("artifact_payload must be a JSON object when supplied.")
            try:
                # Make an immutable-by-convention private snapshot. The runtime
                # passes the exact payload that cleared policy, approval,
                # preflight, and rehearsal gates; the bot must never reread a
                # path that can be replaced between those checks and execution.
                canonical_artifact = json.dumps(
                    artifact_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("artifact_payload must be JSON-safe.") from exc
            self._artifact_payload = json.loads(canonical_artifact)
        self.artifact: dict = {}
        self.strategies: list[dict] = []
        self.artifact_content_digest: str | None = None
        self.approval_fingerprints_by_strategy: dict[str, str] = {}
        self.state: dict = {}
        self.cycle_errors: list[dict] = []
        # Ephemeral notifications emitted only after the corresponding state
        # transition is durable. The autopilot consumes these after each cycle;
        # they never authorize, delay, or alter order execution.
        self.position_events: list[dict] = []
        # Read-only per-cycle observability. This never participates in a
        # trading decision and is copied into runtime/status.json by the
        # supervisor so an idle product explains exactly where it stopped.
        self.decision_trace: dict[str, Any] = {}
        # Per-cycle macro regime evaluation (held-vs-flat overlay for the BTC bot).
        self._macro_aside: bool = False
        self._macro_detail: dict = {}
        self._feature_frame_cache: dict[tuple, tuple[pd.DataFrame, float]] = {}

        self._load_strategies()
        self._load_state()

    def _start_decision_trace(self) -> None:
        self.decision_trace = {
            "schema": DECISION_TRACE_SCHEMA,
            "generated_at": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
            "product": self.objective,
            "market": self.market,
            "symbol": self.symbol,
            "entries_enabled": self.allow_entries,
            "strategies": {},
            "summary": {
                "strategies": len(self.strategies),
                "data_ready": 0,
                "market_bars_processed": 0,
                "market_bars": [],
                "signals": 0,
                "entries_opened": 0,
                "positions_managed": 0,
                "outcomes": {},
            },
        }

    def _record_market_bar(self, strategy: dict, frame: pd.DataFrame) -> None:
        """Record each unique closed product/timeframe bar observed this cycle."""
        timestamp = self._normalized_bar_timestamp(frame.iloc[-1]["timestamp"])
        bar = {
            "timeframe": str(strategy["base_timeframe"]),
            "timestamp": timestamp,
        }
        bars = self.decision_trace["summary"]["market_bars"]
        if bar not in bars:
            bars.append(bar)
            self.decision_trace["summary"]["market_bars_processed"] = len(bars)

    @staticmethod
    def _decision_trace_value(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(key): PaperTradingBot._decision_trace_value(item) for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [PaperTradingBot._decision_trace_value(item) for item in value]
        return str(value)

    def _record_decision(self, strategy: dict, outcome: str, **detail: Any) -> None:
        if not self.decision_trace:
            self._start_decision_trace()
        strategy_id = str(strategy["id"])
        item = {
            "strategy_id": strategy_id,
            "base_timeframe": strategy.get("base_timeframe"),
            "direction": strategy.get("direction"),
            "outcome": outcome,
            **{
                key: self._decision_trace_value(value)
                for key, value in detail.items()
                if value is not None
            },
        }
        self.decision_trace["strategies"][strategy_id] = item
        outcomes = self.decision_trace["summary"]["outcomes"]
        outcomes[outcome] = int(outcomes.get(outcome, 0)) + 1

    def _trace_hypothesis_signal(self, strategy: dict, frame: pd.DataFrame) -> tuple[bool, dict]:
        hypothesis = strategy["_hypothesis"]
        if hypothesis.entry_score is not None:
            predicates = hypothesis.all_predicates()
            matches = [
                bool(
                    predicate_mask(frame, predicate, base_tf=hypothesis.base_timeframe)
                    .fillna(False)
                    .iloc[-1]
                )
                for predicate in predicates
            ]
            score = float(entry_score_series(frame, hypothesis).iloc[-1])
            threshold = float(hypothesis.entry_score.threshold)
            triggered = bool(entry_mask(frame, hypothesis).iloc[-1])
            score_passed = score >= threshold
            return triggered, {
                "failed_stage": (
                    None
                    if triggered
                    else "volatility_filter"
                    if score_passed
                    else "score_threshold"
                ),
                "alpha_score": round(score, 8),
                "score_threshold": threshold,
                "matched_predicates": sum(matches),
                "total_predicates": len(predicates),
                "predicate_matches": {
                    predicate.describe(): matched
                    for predicate, matched in zip(predicates, matches, strict=True)
                },
            }
        matched = 0
        total = 0
        stage_counts: dict[str, dict[str, int]] = {}
        for stage, predicates in (
            ("regime", hypothesis.regime),
            ("setup", hypothesis.setup),
            ("trigger", hypothesis.trigger),
        ):
            stage_matched = 0
            for predicate in predicates:
                total += 1
                passed = bool(
                    predicate_mask(
                        frame,
                        predicate,
                        base_tf=hypothesis.base_timeframe,
                    )
                    .fillna(False)
                    .iloc[-1]
                )
                if passed:
                    matched += 1
                    stage_matched += 1
                    continue
                return False, {
                    "failed_stage": stage,
                    "failed_predicate": predicate.describe(),
                    "matched_predicates": matched,
                    "total_predicates": len(hypothesis.all_predicates()),
                    "stage_counts": {
                        **stage_counts,
                        stage: {"matched": stage_matched, "total": len(predicates)},
                    },
                }
            stage_counts[stage] = {"matched": stage_matched, "total": len(predicates)}
        # Keep the exact canonical entry mask as the final authority because it
        # also applies the hypothesis volatility risk filter.
        triggered = bool(entry_mask(frame, hypothesis).iloc[-1])
        return triggered, {
            "failed_stage": None if triggered else "volatility_filter",
            "matched_predicates": matched,
            "total_predicates": total,
            "stage_counts": stage_counts,
        }

    @staticmethod
    def _frozen_ml_regime_mask(
        frame: pd.DataFrame,
        regime: str,
        *,
        close_feature: str | None = None,
        base_timeframe: str | None = None,
    ) -> pd.Series:
        if regime == "all":
            return pd.Series(True, index=frame.index)
        selected_close = close_feature
        if selected_close not in frame.columns and selected_close and base_timeframe:
            prefixed = f"tf_{base_timeframe}_{selected_close}"
            selected_close = prefixed if prefixed in frame.columns else selected_close
        if not selected_close:
            close_columns = [
                name for name in frame.columns if name == "close" or name.endswith("_close")
            ]
            selected_close = close_columns[0] if close_columns else None
        if selected_close not in frame.columns:
            raise ValueError("frozen ML regime close feature is missing")
        returns = frame[selected_close].astype(float).pct_change()
        volatility = returns.rolling(48, min_periods=24).std()
        baseline = volatility.rolling(480, min_periods=96).median()
        if regime == "high_volatility":
            return (volatility >= baseline).fillna(False)
        if regime == "low_volatility":
            return (volatility < baseline).fillna(False)
        if regime == "trend":
            trend = returns.rolling(48, min_periods=24).sum().abs()
            return (trend >= volatility * math.sqrt(48)).fillna(False)
        raise ValueError(f"unsupported frozen ML regime: {regime!r}")

    @staticmethod
    def _trace_frozen_ml_signal(strategy: dict, frame: pd.DataFrame) -> tuple[bool, dict]:
        model = strategy["_frozen_ml"]
        row = frame.iloc[-1].to_dict()
        base_prefix = f"tf_{strategy['base_timeframe']}_"
        for feature in model.feature_names:
            runtime_feature = f"{base_prefix}{feature}"
            if feature not in row and runtime_feature in row:
                row[feature] = row[runtime_feature]
        prediction = model.prediction(row)
        threshold_triggered = model.triggered(row, strategy["direction"])
        regime = strategy.get("ml_regime", "all")
        regime_triggered = bool(
            PaperTradingBot._frozen_ml_regime_mask(
                frame,
                regime,
                close_feature=strategy.get("ml_regime_close_feature"),
                base_timeframe=strategy["base_timeframe"],
            ).iloc[-1]
        )
        triggered = threshold_triggered and regime_triggered
        return triggered, {
            "failed_stage": (
                None
                if triggered
                else "frozen_ml_threshold"
                if not threshold_triggered
                else "frozen_ml_regime"
            ),
            "model_kind": model.kind,
            "prediction": prediction,
            "long_threshold": model.long_threshold,
            "short_threshold": model.short_threshold,
            "min_edge": model.min_edge,
            "feature_count": len(model.feature_names),
            "ml_regime": regime,
            "regime_triggered": regime_triggered,
        }

    # ------------------------------------------------------------------
    # Configuration / state
    # ------------------------------------------------------------------
    def _validate_product_strategy(self, strategy: dict) -> None:
        strategy_id = strategy["id"]
        metrics = strategy.get("metrics") or {}
        require_performance_evidence = not (
            self.artifact.get("live_allowed") is False
            and self.artifact.get("promotion_eligible") is False
        )
        if self.objective == "btc_accumulation":
            if self.market != "spot":
                raise ValueError("BTC accumulation strategies must run on spot.")
            if self.base_asset is not None and self.base_asset != "BTC":
                raise ValueError("BTC accumulation strategies must use base_asset BTC.")
            if strategy["direction"] != "short":
                raise ValueError(
                    f"Strategy {strategy_id} must be a spot step-aside short for BTC accumulation."
                )
            pnl_unit = strategy.get("pnl_unit")
            if pnl_unit not in {None, "btc", "BTC"}:
                raise ValueError(f"Strategy {strategy_id} BTC accumulation pnl_unit must be BTC.")
            if require_performance_evidence:
                excess = _finite_metric(metrics, "holdout_excess_return_vs_buy_hold", strategy_id)
                if excess <= 0:
                    raise ValueError(
                        f"Strategy {strategy_id} holdout_excess_return_vs_buy_hold must be positive."
                    )
        elif self.objective == "active_income":
            if self.market != "futures":
                raise ValueError("Active income strategies must run on futures.")
            if self.base_asset is not None and self.base_asset != "USDT":
                raise ValueError("Active income strategies must use base_asset USDT.")
            pnl_unit = strategy.get("pnl_unit")
            if pnl_unit not in {None, "usdt", "USDT"}:
                raise ValueError(f"Strategy {strategy_id} active income pnl_unit must be USDT.")
            if require_performance_evidence:
                holdout = _finite_metric(metrics, "holdout_total_return", strategy_id)
                if holdout <= 0:
                    raise ValueError(
                        f"Strategy {strategy_id} holdout_total_return must be positive."
                    )
                dsr_key = "dsr_deflated" if "dsr_deflated" in metrics else "dsr"
                dsr = _finite_metric(metrics, dsr_key, strategy_id)
                if dsr < ACTIVE_INCOME_MIN_DSR:
                    raise ValueError(
                        f"Strategy {strategy_id} active income DSR {dsr:.6f} below "
                        f"{ACTIVE_INCOME_MIN_DSR:.6f}."
                    )

    def _load_strategies(self):
        if self._artifact_payload is None:
            _reject_symlink_path(self.strategies_path, "Strategy artifact")
            if not self.strategies_path.exists():
                raise FileNotFoundError(
                    f"{self.strategies_path} not found. Run a search and then "
                    "`python -m src.export_strategies --search-dir <output dir>` first."
                )
            self.artifact = json.loads(self.strategies_path.read_text(encoding="utf-8"))
        else:
            # A second JSON round trip prevents later caller mutation from
            # changing the strategies while this bot instance is alive.
            self.artifact = json.loads(json.dumps(self._artifact_payload))
        artifact_market = self.artifact.get("market")
        if artifact_market is not None:
            if str(artifact_market) not in {"futures", "spot"}:
                raise ValueError(
                    f"Strategy artifact market must be 'futures' or 'spot', got {artifact_market!r}."
                )
            if str(artifact_market) != self.market:
                raise ValueError(
                    f"Strategy artifact market {artifact_market!r} does not match bot market {self.market!r}."
                )
        artifact_symbol = self.artifact.get("symbol")
        if artifact_symbol is not None and not _symbols_match(str(artifact_symbol), self.symbol):
            raise ValueError(
                f"Strategy artifact symbol {artifact_symbol!r} does not match bot symbol {self.symbol!r}."
            )
        self.strategies = self.artifact.get("strategies", [])
        if not self.strategies:
            raise ValueError(f"{self.strategies_path} contains no strategies.")
        # Capture the exact approval identities before normalization mutates
        # numeric spellings in the in-memory execution copy. Paper evidence
        # must remain bound to the raw artifact behavior the operator reviews.
        self.artifact_content_digest = approval_artifact_digest(self.artifact)
        self.approval_fingerprints_by_strategy = {
            str(strategy.get("id")): approval_strategy_fingerprint(strategy)
            for strategy in self.strategies
            if isinstance(strategy, dict)
            and isinstance(strategy.get("id"), str)
            and strategy.get("id").strip()
        }
        seen_strategy_ids: set[str] = set()
        for strategy in self.strategies:
            entry_type = strategy.get("entry_type", "conditions")
            if entry_type not in {"conditions", "hypothesis", "frozen_ml"}:
                raise ValueError(
                    f"Strategy {strategy.get('id', '<unknown>')} entry_type must be conditions, hypothesis, or frozen_ml."
                )
            entry_key = (
                "hypothesis"
                if entry_type == "hypothesis"
                else "frozen_model"
                if entry_type == "frozen_ml"
                else "conditions"
            )
            for key in (
                "id",
                "base_timeframe",
                "direction",
                "horizon_bars",
                "take_profit",
                "stop_loss",
                entry_key,
                "risk",
                "fees",
            ):
                if key not in strategy:
                    raise ValueError(f"Strategy entry is missing required key {key!r}.")
            strategy_id = strategy["id"]
            if not isinstance(strategy_id, str) or not strategy_id.strip():
                raise ValueError("Strategy id must be a non-empty string.")
            if strategy_id in seen_strategy_ids:
                raise ValueError(
                    f"Duplicate strategy id {strategy_id!r} in {self.strategies_path}."
                )
            seen_strategy_ids.add(strategy_id)
            if strategy["direction"] not in {"long", "short"}:
                raise ValueError(f"Strategy {strategy['id']} direction must be long or short.")
            self._validate_product_strategy(strategy)
            if not isinstance(strategy["base_timeframe"], str) or not strategy["base_timeframe"]:
                raise ValueError(
                    f"Strategy {strategy['id']} base_timeframe must be a non-empty string."
                )
            strategy_market = strategy.get("market")
            if strategy_market is not None:
                if str(strategy_market) not in {"futures", "spot"}:
                    raise ValueError(
                        f"Strategy {strategy['id']} market must be 'futures' or 'spot', got {strategy_market!r}."
                    )
                if str(strategy_market) != self.market:
                    raise ValueError(
                        f"Strategy {strategy['id']} market {strategy_market!r} does not match bot market {self.market!r}."
                    )
            strategy_symbol = strategy.get("symbol")
            if strategy_symbol is not None and not _symbols_match(
                str(strategy_symbol), self.symbol
            ):
                raise ValueError(
                    f"Strategy {strategy['id']} symbol {strategy_symbol!r} does not match bot symbol {self.symbol!r}."
                )
            _normalize_positive_int(strategy, "horizon_bars")
            _normalize_positive_float(strategy, "take_profit")
            _normalize_positive_float(strategy, "stop_loss")
            _normalize_optional_probability(strategy, "baseline_win_rate")
            strategy["risk"] = _normalize_strategy_risk(strategy)
            strategy["fees"] = _normalize_strategy_fees(strategy)
            if entry_type == "hypothesis":
                strategy["_hypothesis"] = Hypothesis.from_dict(strategy["hypothesis"])
                strategy["_conditions"] = []
            elif entry_type == "frozen_ml":
                if strategy.get("ml_regime", "all") not in {
                    "all",
                    "trend",
                    "high_volatility",
                    "low_volatility",
                }:
                    raise ValueError(f"Strategy {strategy['id']} ml_regime is invalid.")
                close_feature = strategy.get("ml_regime_close_feature")
                if close_feature is not None and (
                    not isinstance(close_feature, str) or not close_feature
                ):
                    raise ValueError(
                        f"Strategy {strategy['id']} ml_regime_close_feature is invalid."
                    )
                strategy["_frozen_ml"] = FrozenGradientBoostingModel.from_dict(
                    strategy["frozen_model"]
                )
                strategy["_conditions"] = []
            else:
                strategy["_conditions"] = _normalize_conditions(strategy)
        LOGGER.info(
            "Loaded %s strategies from %s (search sha %s)",
            len(self.strategies),
            self.strategies_path,
            self.artifact.get("search_git_sha", "unknown"),
        )
        for strategy in self.strategies:
            LOGGER.info(
                "  %s: %s %s tf=%s horizon=%s TP=%s SL=%s",
                strategy["id"],
                strategy["direction"],
                strategy.get("rule", ""),
                strategy["base_timeframe"],
                strategy["horizon_bars"],
                strategy["take_profit"],
                strategy["stop_loss"],
            )

    def _account_risk(self) -> dict:
        risk = dict(self.strategies[0].get("risk") or {})
        daily_stops = [
            float((strategy.get("risk") or {}).get("daily_stop_loss"))
            for strategy in self.strategies
            if (strategy.get("risk") or {}).get("daily_stop_loss") is not None
        ]
        if daily_stops:
            # daily_stop_loss is negative; the closest value to zero is strictest.
            risk["daily_stop_loss"] = max(daily_stops)
        return risk

    def _max_equity_drawdown(self) -> float:
        """Return the fixed product-level peak-equity loss envelope.

        BTC accumulation is deliberately tighter because its purpose is
        conservative base-asset preservation. Unknown/direct CLI objectives
        receive that same fail-safe 5% envelope.
        """

        return MAX_EQUITY_DRAWDOWN_BY_OBJECTIVE.get(
            self.objective,
            DEFAULT_MAX_EQUITY_DRAWDOWN,
        )

    def _normalize_drawdown_state(self) -> bool:
        """Validate, migrate, and (when necessary) latch the drawdown breaker."""

        equity = float(self.state["equity"])
        changed = False
        raw_peak = self.state.get("peak_equity")
        if raw_peak is None:
            starting = float(self.starting_equity)
            if not math.isfinite(starting) or starting <= 0:
                starting = equity
            peak = max(equity, starting)
            changed = True
        else:
            if isinstance(raw_peak, bool):
                raise RuntimeError(f"State peak_equity must be numeric: {self.state_file}")
            try:
                peak = float(raw_peak)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"State peak_equity must be numeric: {self.state_file}") from exc
            if not math.isfinite(peak):
                raise RuntimeError(f"State peak_equity must be finite: {self.state_file}")
            if peak <= 0:
                raise RuntimeError(f"State peak_equity must be positive: {self.state_file}")
            if peak < equity:
                peak = equity
                changed = True
            elif raw_peak != peak:
                changed = True
        self.state["peak_equity"] = peak

        limit = self._max_equity_drawdown()
        if self.state.get("drawdown_limit_fraction") != limit:
            self.state["drawdown_limit_fraction"] = limit
            changed = True
        drawdown = max(0.0, (peak - equity) / peak)
        if self.state.get("drawdown_fraction") != drawdown:
            self.state["drawdown_fraction"] = drawdown
            changed = True

        if "drawdown_halted" not in self.state:
            self.state["drawdown_halted"] = False
            changed = True
        halted = self.state["drawdown_halted"]
        if not isinstance(halted, bool):
            raise RuntimeError(f"State drawdown_halted must be boolean: {self.state_file}")

        if halted:
            halted_at = self.state.get("drawdown_halted_at")
            if not isinstance(halted_at, str) or not halted_at:
                raise RuntimeError(
                    f"State drawdown_halted_at must be an ISO timestamp while halted: {self.state_file}"
                )
            try:
                parsed = datetime.datetime.fromisoformat(halted_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise RuntimeError(
                    f"State drawdown_halted_at must be an ISO timestamp while halted: {self.state_file}"
                ) from exc
            if parsed.tzinfo is None:
                raise RuntimeError(
                    f"State drawdown_halted_at must include a timezone while halted: {self.state_file}"
                )
            normalized_halted_at = (
                parsed.astimezone(datetime.UTC).replace(microsecond=0).isoformat()
            )
            if halted_at != normalized_halted_at:
                self.state["drawdown_halted_at"] = normalized_halted_at
                changed = True
            reason = self.state.get("drawdown_halt_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise RuntimeError(
                    f"State drawdown_halt_reason must be non-empty while halted: {self.state_file}"
                )
        else:
            for key in ("drawdown_halted_at", "drawdown_halt_reason"):
                if key not in self.state:
                    self.state[key] = None
                    changed = True
                elif self.state[key] is not None:
                    raise RuntimeError(
                        f"State {key} must be null while drawdown_halted is false: {self.state_file}"
                    )
            if drawdown + 1e-12 >= limit:
                objective = self.objective or "unspecified"
                self.state["drawdown_halted"] = True
                self.state["drawdown_halted_at"] = _utc_now_text()
                self.state["drawdown_halt_reason"] = (
                    "equity_drawdown_limit_reached "
                    f"objective={objective} drawdown={drawdown:.8f} limit={limit:.8f}"
                )
                changed = True
                LOGGER.critical(
                    "DRAWDOWN CIRCUIT BREAKER: objective=%s equity=%.8f peak=%.8f "
                    "drawdown=%.4f%% limit=%.4f%%. New entries are halted until reviewed recovery.",
                    objective,
                    equity,
                    peak,
                    drawdown * 100,
                    limit * 100,
                )
        return changed

    def _normalize_state_float(
        self,
        key: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> bool:
        raw = self.state.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"State {key} must be numeric: {self.state_file}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"State {key} must be finite: {self.state_file}")
        if positive and value <= 0:
            raise RuntimeError(f"State {key} must be positive: {self.state_file}")
        if non_negative and value < 0:
            raise RuntimeError(f"State {key} must be non-negative: {self.state_file}")
        changed = raw != value
        self.state[key] = value
        return changed

    def _normalize_state_int(self, key: str, *, non_negative: bool = False) -> bool:
        raw = self.state.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"State {key} must be numeric: {self.state_file}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"State {key} must be finite: {self.state_file}")
        if value != int(value):
            raise RuntimeError(f"State {key} must be an integer: {self.state_file}")
        integer = int(value)
        if non_negative and integer < 0:
            raise RuntimeError(f"State {key} must be non-negative: {self.state_file}")
        changed = raw != integer
        self.state[key] = integer
        return changed

    def _normalize_daily_trade_counts(self) -> bool:
        counts = self.state["daily_trades_by_strategy"]
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"State daily_trades_by_strategy must be an object: {self.state_file}"
            )
        changed = False
        for strategy_id, raw in list(counts.items()):
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"State daily_trades_by_strategy[{strategy_id!r}] must be numeric: {self.state_file}"
                ) from exc
            if not math.isfinite(value):
                raise RuntimeError(
                    f"State daily_trades_by_strategy[{strategy_id!r}] must be finite: {self.state_file}"
                )
            if value != int(value):
                raise RuntimeError(
                    f"State daily_trades_by_strategy[{strategy_id!r}] must be an integer: {self.state_file}"
                )
            integer = int(value)
            if integer < 0:
                raise RuntimeError(
                    f"State daily_trades_by_strategy[{strategy_id!r}] must be non-negative: {self.state_file}"
                )
            if raw != integer:
                counts[strategy_id] = integer
                changed = True
        return changed

    def _normalize_entry_decision_bars(self) -> bool:
        key = "last_entry_decision_bar_by_strategy"
        cursors = self.state[key]
        if not isinstance(cursors, dict):
            raise RuntimeError(f"State {key} must be an object: {self.state_file}")
        known = {strategy["id"] for strategy in self.strategies}
        known.update(self.state.get("open_positions", {}))
        changed = False
        for strategy_id, raw in list(cursors.items()):
            if strategy_id not in known:
                raise RuntimeError(
                    f"State {key} contains unknown strategy {strategy_id!r}: {self.state_file}"
                )
            try:
                timestamp = pd.Timestamp(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"State {key}[{strategy_id!r}] must be a valid timestamp: {self.state_file}"
                ) from exc
            if pd.isna(timestamp):
                raise RuntimeError(
                    f"State {key}[{strategy_id!r}] must be a valid timestamp: {self.state_file}"
                )
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            normalized = timestamp.isoformat()
            if raw != normalized:
                cursors[strategy_id] = normalized
                changed = True
        return changed

    def _normalize_last_pnl_reset_date(self) -> bool:
        key = "last_pnl_reset_date"
        raw = self.state.get(key)
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"State {key} must be an ISO date string: {self.state_file}")
        try:
            parsed = datetime.date.fromisoformat(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"State {key} must be an ISO date string: {self.state_file}"
            ) from exc
        today = _utc_date_today()
        if parsed > today:
            raise RuntimeError(f"State {key} must not be in the future: {self.state_file}")
        normalized = parsed.isoformat()
        changed = raw != normalized
        self.state[key] = normalized
        return changed

    def _normalize_position_float(
        self,
        position: dict,
        strategy_id: str,
        key: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> bool:
        try:
            raw = position[key]
        except KeyError as exc:
            raise RuntimeError(
                f"State open_positions[{strategy_id!r}].{key} is required: {self.state_file}"
            ) from exc
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State open_positions[{strategy_id!r}].{key} must be numeric: {self.state_file}"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(
                f"State open_positions[{strategy_id!r}].{key} must be finite: {self.state_file}"
            )
        if positive and value <= 0:
            raise RuntimeError(
                f"State open_positions[{strategy_id!r}].{key} must be positive: {self.state_file}"
            )
        if non_negative and value < 0:
            raise RuntimeError(
                f"State open_positions[{strategy_id!r}].{key} must be non-negative: {self.state_file}"
            )
        changed = raw != value
        position[key] = value
        return changed

    def _normalize_optional_position_float(
        self,
        position: dict,
        strategy_id: str,
        key: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> bool:
        if key not in position:
            return False
        return self._normalize_position_float(
            position,
            strategy_id,
            key,
            positive=positive,
            non_negative=non_negative,
        )

    @staticmethod
    def _strategy_snapshot(strategy: dict) -> tuple[dict, str]:
        public_strategy = {
            key: value for key, value in strategy.items() if not str(key).startswith("_")
        }
        canonical = json.dumps(
            public_strategy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        snapshot = json.loads(canonical)
        return snapshot, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _strategy_for_open_position(self, current_strategy: dict, position: dict) -> dict:
        has_snapshot = "strategy_snapshot" in position
        has_fingerprint = "strategy_fingerprint" in position
        if has_snapshot != has_fingerprint:
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}] must contain both strategy_snapshot "
                f"and strategy_fingerprint: {self.state_file}"
            )
        if not has_snapshot:
            if self._requires_native_protective_stop():
                raise RuntimeError(
                    f"Live state open_positions[{current_strategy['id']!r}] has no frozen strategy "
                    f"snapshot: {self.state_file}"
                )
            # Compatibility for paper positions created before snapshots were
            # introduced. Every newly-opened position takes the strict path.
            return current_strategy

        snapshot = position["strategy_snapshot"]
        fingerprint = position["strategy_fingerprint"]
        if not isinstance(snapshot, dict):
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_snapshot must be an object: "
                f"{self.state_file}"
            )
        if any(str(key).startswith("_") for key in snapshot):
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_snapshot contains "
                f"non-JSON runtime fields: {self.state_file}"
            )
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_fingerprint is invalid: "
                f"{self.state_file}"
            )
        try:
            canonical = json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_snapshot is not JSON-safe: "
                f"{self.state_file}"
            ) from exc
        expected_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_fingerprint does not "
                f"match its snapshot: {self.state_file}"
            )

        runtime_strategy = json.loads(canonical)
        try:
            if runtime_strategy.get("id") != current_strategy["id"]:
                raise ValueError("snapshot strategy id does not match its open-position key")
            if runtime_strategy.get("direction") not in {"long", "short"}:
                raise ValueError("snapshot direction must be long or short")
            if (
                position.get("direction") is not None
                and runtime_strategy["direction"] != position["direction"]
            ):
                raise ValueError("snapshot direction does not match the open position")
            if (
                not isinstance(runtime_strategy.get("base_timeframe"), str)
                or not runtime_strategy["base_timeframe"]
            ):
                raise ValueError("snapshot base_timeframe must be non-empty")
            _normalize_positive_int(runtime_strategy, "horizon_bars")
            _normalize_positive_float(runtime_strategy, "take_profit")
            _normalize_positive_float(runtime_strategy, "stop_loss")
            _normalize_optional_probability(runtime_strategy, "baseline_win_rate")
            runtime_strategy["risk"] = _normalize_strategy_risk(runtime_strategy)
            runtime_strategy["fees"] = _normalize_strategy_fees(runtime_strategy)
            entry_type = runtime_strategy.get("entry_type", "conditions")
            if entry_type == "hypothesis":
                runtime_strategy["_hypothesis"] = Hypothesis.from_dict(
                    runtime_strategy["hypothesis"]
                )
                runtime_strategy["_conditions"] = []
            elif entry_type == "frozen_ml":
                if runtime_strategy.get("ml_regime", "all") not in {
                    "all",
                    "trend",
                    "high_volatility",
                    "low_volatility",
                }:
                    raise ValueError("snapshot ml_regime is invalid")
                close_feature = runtime_strategy.get("ml_regime_close_feature")
                if close_feature is not None and (
                    not isinstance(close_feature, str) or not close_feature
                ):
                    raise ValueError("snapshot ml_regime_close_feature is invalid")
                runtime_strategy["_frozen_ml"] = FrozenGradientBoostingModel.from_dict(
                    runtime_strategy["frozen_model"]
                )
                runtime_strategy["_conditions"] = []
            elif entry_type == "conditions":
                runtime_strategy["_conditions"] = _normalize_conditions(runtime_strategy)
            else:
                raise ValueError("snapshot entry_type must be conditions, hypothesis, or frozen_ml")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State open_positions[{current_strategy['id']!r}].strategy_snapshot is not a valid "
                f"executable strategy: {self.state_file}: {exc}"
            ) from exc
        return runtime_strategy

    def _normalize_open_positions(self) -> bool:
        positions = self.state["open_positions"]
        if not isinstance(positions, dict):
            raise RuntimeError(f"State open_positions must be an object: {self.state_file}")
        strategies_by_id = {strategy["id"]: strategy for strategy in self.strategies}
        changed = False
        for strategy_id, position in positions.items():
            current_strategy = strategies_by_id.get(strategy_id)
            if current_strategy is None and not (
                isinstance(position, dict)
                and "strategy_snapshot" in position
                and "strategy_fingerprint" in position
            ):
                raise RuntimeError(
                    f"State open_positions contains unknown strategy {strategy_id!r}: {self.state_file}"
                )
            if not isinstance(position, dict):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}] must be an object: {self.state_file}"
                )
            frozen_strategy = self._strategy_for_open_position(
                current_strategy or {"id": strategy_id},
                position,
            )
            try:
                entry_time = pd.Timestamp(position["entry_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].entry_time must be a valid timestamp: {self.state_file}"
                ) from exc
            if pd.isna(entry_time):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].entry_time must be a valid timestamp: {self.state_file}"
                )
            if entry_time.tzinfo is None:
                entry_time = entry_time.tz_localize("UTC")
            else:
                entry_time = entry_time.tz_convert("UTC")
            signal_time_raw = position.get("signal_time")
            if signal_time_raw is None:
                # States written before the closed-bar causality fix stored the
                # signal candle's opening time as the fill time.  Migrate them
                # to the next-bar-open effective entry boundary.
                signal_time = entry_time
                tf_seconds = TIMEFRAME_SECONDS.get(
                    frozen_strategy["base_timeframe"],
                    300,
                )
                entry_time = entry_time + pd.Timedelta(seconds=tf_seconds)
                position["signal_time"] = signal_time.isoformat()
                position["entry_time"] = entry_time.isoformat()
                changed = True
            else:
                try:
                    signal_time = pd.Timestamp(signal_time_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}].signal_time must be a valid timestamp: "
                        f"{self.state_file}"
                    ) from exc
                if pd.isna(signal_time):
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}].signal_time must be a valid timestamp: "
                        f"{self.state_file}"
                    )
                if signal_time.tzinfo is None:
                    signal_time = signal_time.tz_localize("UTC")
                else:
                    signal_time = signal_time.tz_convert("UTC")
                if entry_time <= signal_time:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}].entry_time must be after signal_time: "
                        f"{self.state_file}"
                    )
                normalized_signal = signal_time.isoformat()
                normalized_entry = entry_time.isoformat()
                if signal_time_raw != normalized_signal:
                    position["signal_time"] = normalized_signal
                    changed = True
                if position["entry_time"] != normalized_entry:
                    position["entry_time"] = normalized_entry
                    changed = True
            candidate_schema = position.get("candidate_paper_execution_schema")
            candidate_engine = position.get("candidate_paper_engine_digest")
            if candidate_schema is not None or candidate_engine is not None:
                if candidate_schema != CANDIDATE_PAPER_EXECUTION_SCHEMA:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] has an unsupported candidate "
                        f"paper execution schema: {self.state_file}"
                    )
                if (
                    not isinstance(candidate_engine, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_engine) is None
                ):
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] has an invalid candidate "
                        f"paper engine digest: {self.state_file}"
                    )
                evidence_eligible = position.get("candidate_paper_evidence_eligible")
                evidence_reason = position.get("candidate_paper_evidence_reason")
                fill_source = position.get("candidate_paper_entry_fill_source")
                if not isinstance(evidence_eligible, bool):
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] has invalid candidate "
                        f"paper evidence eligibility: {self.state_file}"
                    )
                try:
                    observed_at = self._candidate_replay_observation_timestamp(
                        position["candidate_paper_observed_at"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] has an invalid candidate "
                        f"paper observation time: {self.state_file}"
                    ) from exc
                signal_close = signal_time + pd.Timedelta(
                    seconds=TIMEFRAME_SECONDS.get(
                        frozen_strategy["base_timeframe"],
                        300,
                    )
                )
                if observed_at < signal_close:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] candidate observation "
                        f"precedes signal availability: {self.state_file}"
                    )
                if evidence_eligible:
                    if (
                        evidence_reason != CANDIDATE_PAPER_FORWARD_REASON
                        or fill_source != CANDIDATE_PAPER_FORWARD_FILL_SOURCE
                        or observed_at != entry_time
                    ):
                        raise RuntimeError(
                            f"State open_positions[{strategy_id!r}] has inconsistent "
                            f"promotable candidate evidence: {self.state_file}"
                        )
                elif evidence_reason == CANDIDATE_PAPER_BACKFILL_ENTRY_REASON:
                    if fill_source != CANDIDATE_PAPER_BACKFILL_FILL_SOURCE:
                        raise RuntimeError(
                            f"State open_positions[{strategy_id!r}] has inconsistent "
                            f"backfill entry evidence: {self.state_file}"
                        )
                elif evidence_reason == CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON:
                    if fill_source not in {
                        CANDIDATE_PAPER_FORWARD_FILL_SOURCE,
                        CANDIDATE_PAPER_BACKFILL_FILL_SOURCE,
                    }:
                        raise RuntimeError(
                            f"State open_positions[{strategy_id!r}] has inconsistent "
                            f"backfill management evidence: {self.state_file}"
                        )
                else:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}] has an invalid candidate "
                        f"paper evidence reason: {self.state_file}"
                    )
                normalized_observed_at = observed_at.isoformat()
                if position["candidate_paper_observed_at"] != normalized_observed_at:
                    position["candidate_paper_observed_at"] = normalized_observed_at
                    changed = True
            direction = position.get("direction")
            if direction not in {"long", "short"}:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].direction must be long or short: {self.state_file}"
                )
            for key in ("entry_price", "sl_pct", "tp_pct", "sl_price", "tp_price"):
                changed = (
                    self._normalize_position_float(position, strategy_id, key, positive=True)
                    or changed
                )
            changed = (
                self._normalize_position_float(
                    position, strategy_id, "position_size", positive=True
                )
                or changed
            )
            if position["position_size"] > 1.0:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].position_size must be <= 1: {self.state_file}"
                )
            max_position_fraction = float(frozen_strategy["risk"]["max_position_fraction"])
            if position["position_size"] - max_position_fraction > 1e-12:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].position_size exceeds "
                    f"max_position_fraction {max_position_fraction:g}: {self.state_file}"
                )
            for key in (
                "broker_requested_qty",
                "broker_fill_ratio",
                "broker_qty",
                "broker_entry_price",
            ):
                changed = (
                    self._normalize_optional_position_float(
                        position,
                        strategy_id,
                        key,
                        positive=True,
                    )
                    or changed
                )
            if (
                "broker_fill_ratio" in position
                and abs(float(position["broker_fill_ratio"]) - 1.0) > 1e-9
            ):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_fill_ratio must be 1: {self.state_file}"
                )
            if "broker_requested_qty" in position and "broker_qty" in position:
                requested_qty = float(position["broker_requested_qty"])
                broker_qty = float(position["broker_qty"])
                tolerance = self._fill_quantity_tolerance(requested_qty)
                if abs(broker_qty - requested_qty) > tolerance:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}].broker_qty must match broker_requested_qty: "
                        f"{self.state_file}"
                    )
                if "broker_fill_ratio" in position:
                    if abs(float(position["broker_fill_ratio"]) - 1.0) > max(tolerance, 1e-9):
                        raise RuntimeError(
                            f"State open_positions[{strategy_id!r}].broker_fill_ratio must be 1: "
                            f"{self.state_file}"
                        )
            for key in (
                "broker_entry_fee",
                "broker_entry_base_qty_before",
                "broker_entry_base_qty_after",
                "broker_entry_quote_balance_before",
                "broker_entry_quote_balance_after",
                "broker_entry_quote_value",
            ):
                changed = (
                    self._normalize_optional_position_float(
                        position,
                        strategy_id,
                        key,
                        non_negative=True,
                    )
                    or changed
                )
            quote_value_source = position.get("broker_entry_quote_value_source")
            if quote_value_source is not None and quote_value_source not in {
                "observed_free_quote_delta",
                "fill_notional_less_reported_fee",
            }:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_entry_quote_value_source "
                    f"is invalid: {self.state_file}"
                )
            if quote_value_source == "observed_free_quote_delta":
                observed_keys = (
                    "broker_qty",
                    "broker_entry_price",
                    "broker_entry_quote_balance_before",
                    "broker_entry_quote_balance_after",
                    "broker_entry_quote_value",
                )
                if all(key in position for key in observed_keys):
                    observed_quote_value = self._validated_live_spot_quote_proceeds(
                        strategy_id,
                        qty=float(position["broker_qty"]),
                        price=float(position["broker_entry_price"]),
                        balance_before=float(position["broker_entry_quote_balance_before"]),
                        balance_after=float(position["broker_entry_quote_balance_after"]),
                    )
                    tolerance = max(abs(observed_quote_value) * 1e-9, 1e-9)
                    if (
                        abs(float(position["broker_entry_quote_value"]) - observed_quote_value)
                        > tolerance
                    ):
                        raise RuntimeError(
                            f"State open_positions[{strategy_id!r}].broker_entry_quote_value "
                            f"does not match its observed free-quote balance delta: {self.state_file}"
                        )
            changed = (
                self._normalize_optional_position_float(
                    position,
                    strategy_id,
                    "broker_entry_balance",
                    positive=True,
                )
                or changed
            )
            changed = (
                self._normalize_optional_position_float(
                    position,
                    strategy_id,
                    "broker_stop_trigger_price",
                    positive=True,
                )
                or changed
            )
            if "broker_side" in position and position["broker_side"] not in {
                OrderSide.BUY.value,
                OrderSide.SELL.value,
            }:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_side must be buy or sell: {self.state_file}"
                )
            if "broker_symbol" in position and not str(position["broker_symbol"]):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_symbol must be non-empty: {self.state_file}"
                )
            if (
                "broker_stop_order_id" in position
                and not str(position["broker_stop_order_id"]).strip()
            ):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_stop_order_id must be non-empty: "
                    f"{self.state_file}"
                )
            stop_client_id = position.get("broker_stop_client_id")
            if stop_client_id is not None and (
                not isinstance(stop_client_id, str)
                or not CLIENT_ORDER_ID_RE.fullmatch(stop_client_id)
            ):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_stop_client_id is unsafe: "
                    f"{self.state_file}"
                )
            if "broker_stop_trigger_price" in position:
                stop_trigger = float(position["broker_stop_trigger_price"])
                sl_price = float(position["sl_price"])
                tolerance = max(abs(sl_price) * 1e-9, 1e-12)
                if abs(stop_trigger - sl_price) > tolerance:
                    raise RuntimeError(
                        f"State open_positions[{strategy_id!r}].broker_stop_trigger_price must match "
                        f"sl_price: {self.state_file}"
                    )
            self._assert_broker_metadata_complete(
                strategy_id,
                position,
                state_detail=f"{self.state_file}",
                require_present=self.broker is not None,
            )
        return changed

    @staticmethod
    def _deterministic_client_order_id(
        *,
        strategy_id: str,
        stage: str,
        intent_ref: str,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType,
        reduce_only: bool,
    ) -> str:
        payload = {
            "strategy_id": strategy_id,
            "stage": stage,
            "intent_ref": intent_ref,
            "symbol": symbol,
            "side": side.value,
            "qty": format(float(qty), ".17g"),
            "order_type": order_type.value,
            "reduce_only": bool(reduce_only),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        stage_code = {
            "entry": "en",
            "exit": "ex",
            "stop": "sl",
            "recovery": "rc",
        }.get(stage)
        if stage_code is None:
            raise ValueError(f"Unsupported broker order id stage: {stage!r}")
        client_id = f"tb-{stage_code}-{digest[:28]}"
        if not CLIENT_ORDER_ID_RE.fullmatch(client_id):  # pragma: no cover - generated invariant
            raise RuntimeError(f"Generated unsafe broker client order id: {client_id!r}")
        return client_id

    @staticmethod
    def _expected_pending_side(strategy: dict, stage: str) -> OrderSide:
        entry_side = OrderSide.BUY if strategy["direction"] == "long" else OrderSide.SELL
        if stage == "entry":
            return entry_side
        return OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY

    def _validated_pending_order(self, pending: object) -> dict:
        if not isinstance(pending, dict):
            raise RuntimeError(f"State pending_order must be an object: {self.state_file}")
        missing = [key for key in PENDING_ORDER_REQUIRED_KEYS if key not in pending]
        if missing:
            raise RuntimeError(
                f"State pending_order is missing required key(s): {', '.join(missing)}: {self.state_file}"
            )
        unexpected = sorted(set(pending) - set(PENDING_ORDER_REQUIRED_KEYS))
        if unexpected:
            raise RuntimeError(
                f"State pending_order has unexpected key(s): {', '.join(unexpected)}: {self.state_file}"
            )
        version = pending.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version != 1:
            raise RuntimeError(f"State pending_order.version must be 1: {self.state_file}")
        strategy_id = pending.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise RuntimeError(
                f"State pending_order.strategy_id must be a non-empty string: {self.state_file}"
            )
        strategies_by_id = {strategy["id"]: strategy for strategy in self.strategies}
        strategy = strategies_by_id.get(strategy_id)
        if strategy is None:
            position = self.state.get("open_positions", {}).get(strategy_id)
            if isinstance(position, dict):
                strategy = self._strategy_for_open_position({"id": strategy_id}, position)
        if strategy is None:
            raise RuntimeError(
                f"State pending_order.strategy_id is unknown: {strategy_id!r}: {self.state_file}"
            )
        stage = pending.get("stage")
        if not isinstance(stage, str) or stage not in {"entry", "exit"}:
            raise RuntimeError(
                f"State pending_order.stage must be entry or exit: {self.state_file}"
            )
        intent_ref = pending.get("intent_ref")
        if not isinstance(intent_ref, str) or not intent_ref:
            raise RuntimeError(
                f"State pending_order.intent_ref must be non-empty: {self.state_file}"
            )
        symbol = pending.get("symbol")
        if not isinstance(symbol, str) or symbol != self.symbol:
            raise RuntimeError(
                f"State pending_order.symbol must match configured symbol {self.symbol!r}: {self.state_file}"
            )
        side_value = pending.get("side")
        if not isinstance(side_value, str):
            raise RuntimeError(f"State pending_order.side must be buy or sell: {self.state_file}")
        try:
            side = OrderSide(side_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State pending_order.side must be buy or sell: {self.state_file}"
            ) from exc
        expected_side = self._expected_pending_side(strategy, stage)
        if side != expected_side:
            raise RuntimeError(
                f"State pending_order.side {side.value!r} does not match {stage} side "
                f"{expected_side.value!r}: {self.state_file}"
            )
        qty_raw = pending.get("qty")
        if isinstance(qty_raw, bool):
            raise RuntimeError(f"State pending_order.qty must be numeric: {self.state_file}")
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State pending_order.qty must be numeric: {self.state_file}"
            ) from exc
        if not math.isfinite(qty) or qty <= 0:
            raise RuntimeError(
                f"State pending_order.qty must be finite and positive: {self.state_file}"
            )
        order_type_value = pending.get("order_type")
        if not isinstance(order_type_value, str):
            raise RuntimeError(f"State pending_order.order_type is invalid: {self.state_file}")
        try:
            order_type = OrderType(order_type_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State pending_order.order_type is invalid: {self.state_file}"
            ) from exc
        if order_type != OrderType.MARKET:
            raise RuntimeError(f"State pending_order.order_type must be market: {self.state_file}")
        reduce_only = pending.get("reduce_only")
        if not isinstance(reduce_only, bool):
            raise RuntimeError(
                f"State pending_order.reduce_only must be boolean: {self.state_file}"
            )
        if reduce_only != (stage == "exit"):
            raise RuntimeError(
                f"State pending_order.reduce_only does not match {stage} semantics: {self.state_file}"
            )
        client_id = pending.get("client_id")
        if not isinstance(client_id, str) or not CLIENT_ORDER_ID_RE.fullmatch(client_id):
            raise RuntimeError(f"State pending_order.client_id is unsafe: {self.state_file}")
        expected_client_id = self._deterministic_client_order_id(
            strategy_id=strategy_id,
            stage=stage,
            intent_ref=intent_ref,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            reduce_only=reduce_only,
        )
        if client_id != expected_client_id:
            raise RuntimeError(
                f"State pending_order.client_id does not match its order intent: {self.state_file}"
            )
        broker_account_fingerprint = pending.get("broker_account_fingerprint")
        self._assert_broker_account_fingerprint(
            broker_account_fingerprint,
            state_detail=f"pending_order in {self.state_file}",
        )
        created_ts_raw = pending.get("created_ts")
        if isinstance(created_ts_raw, bool):
            raise RuntimeError(f"State pending_order.created_ts must be numeric: {self.state_file}")
        try:
            created_ts = float(created_ts_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"State pending_order.created_ts must be numeric: {self.state_file}"
            ) from exc
        if not math.isfinite(created_ts) or created_ts < 0:
            raise RuntimeError(
                f"State pending_order.created_ts must be finite and non-negative: {self.state_file}"
            )
        return {
            "version": 1,
            "strategy_id": strategy_id,
            "stage": stage,
            "intent_ref": intent_ref,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "order_type": order_type.value,
            "reduce_only": reduce_only,
            "client_id": client_id,
            "broker_account_fingerprint": broker_account_fingerprint,
            "created_ts": created_ts,
        }

    def _normalize_pending_order(self) -> bool:
        if "pending_order" not in self.state:
            return False
        normalized = self._validated_pending_order(self.state["pending_order"])
        changed = self.state["pending_order"] != normalized
        self.state["pending_order"] = normalized
        return changed

    def _pending_entry_recovery_marker(self, pending: dict) -> dict | None:
        marker = self.state.get("pending_entry_recovery")
        if marker is None:
            return None
        if not isinstance(marker, dict):
            raise RuntimeError(f"State pending_entry_recovery must be an object: {self.state_file}")
        if marker.get("version") != 1:
            raise RuntimeError(f"State pending_entry_recovery.version must be 1: {self.state_file}")
        if marker.get("original_pending_client_id") != pending.get("client_id"):
            raise RuntimeError(
                "State pending_entry_recovery does not match the unresolved pending entry: "
                f"{self.state_file}"
            )
        if marker.get("strategy_id") != pending.get("strategy_id"):
            raise RuntimeError(
                "State pending_entry_recovery strategy does not match the unresolved pending entry: "
                f"{self.state_file}"
            )
        if marker.get("symbol") != pending.get("symbol"):
            raise RuntimeError(
                "State pending_entry_recovery symbol does not match the unresolved pending entry: "
                f"{self.state_file}"
            )
        if marker.get("broker_account_fingerprint") != pending.get("broker_account_fingerprint"):
            raise RuntimeError(
                "State pending_entry_recovery account does not match the unresolved pending "
                f"entry: {self.state_file}"
            )
        self._assert_broker_account_fingerprint(
            marker.get("broker_account_fingerprint"),
            state_detail=f"pending_entry_recovery in {self.state_file}",
        )
        return marker

    def _record_pending_entry_recovery(
        self,
        pending: dict,
        *,
        status: str,
        recovery_order: Order | None = None,
        fill: Fill | None = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        marker = self._pending_entry_recovery_marker(pending)
        now = _utc_now_text()
        if marker is None:
            marker = {
                "version": 1,
                "original_pending_client_id": pending["client_id"],
                "strategy_id": pending["strategy_id"],
                "symbol": pending["symbol"],
                "broker_account_fingerprint": pending["broker_account_fingerprint"],
                "first_detected_at": now,
                "attempt_count": 0,
            }
        else:
            marker = dict(marker)
        attempts_raw = marker.get("attempt_count", 0)
        if not isinstance(attempts_raw, int) or isinstance(attempts_raw, bool) or attempts_raw < 0:
            raise RuntimeError(
                f"State pending_entry_recovery.attempt_count is invalid: {self.state_file}"
            )
        marker["attempt_count"] = attempts_raw + int(increment_attempt)
        marker["status"] = status
        marker["last_updated_at"] = now
        if recovery_order is not None:
            marker["recovery_client_id"] = recovery_order.client_id
            marker["observed_position_qty"] = (
                float(recovery_order.qty)
                if recovery_order.side == OrderSide.SELL
                else -float(recovery_order.qty)
            )
            marker["recovery_side"] = recovery_order.side.value
            marker["recovery_qty"] = float(recovery_order.qty)
        if fill is not None:
            marker["fill"] = {
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": float(fill.qty),
                "price": float(fill.price),
                "fee": float(fill.fee),
                "timestamp": float(fill.timestamp),
            }
        if error is None:
            marker.pop("last_error", None)
        else:
            marker["last_error"] = str(error)
        self.state["pending_entry_recovery"] = marker
        self._save_state()

    def _recover_live_futures_pending_entry(self) -> None:
        """Flatten broker exposure left by an ambiguous live entry submission.

        The original entry intent is never retried or cleared. A deterministic,
        reduce-only recovery close is persisted before submission; even after a
        broker-flat proof, the recovery marker keeps the product blocked for
        explicit operator reconciliation.
        """

        pending = self.state.get("pending_order")
        if not isinstance(pending, dict) or pending.get("stage") != "entry":
            return
        if not self._requires_native_protective_stop():
            return
        if self.state.get("open_positions"):
            return

        marker = self._pending_entry_recovery_marker(pending)
        actual = self.broker.get_position(str(pending["symbol"]))
        if actual.is_flat:
            if marker is None:
                self._record_pending_entry_recovery(
                    pending,
                    status="broker_flat_observed_no_recovery_order",
                )
            elif marker.get("status") not in {
                "broker_flat_observed_no_recovery_order",
                "recovery_close_filled_and_flat",
                "broker_flat_after_ambiguous_recovery_close",
            }:
                self._record_pending_entry_recovery(
                    pending,
                    status="broker_flat_after_previous_recovery_attempt",
                    error=marker.get("last_error"),
                )
            return
        if not _symbols_match(actual.symbol, str(pending["symbol"])):
            raise RuntimeError(
                "Pending-entry recovery broker symbol mismatch; refusing recovery close. "
                "Original pending entry remains for operator reconciliation."
            )

        pending_side = OrderSide(str(pending["side"]))
        actual_is_long = float(actual.qty) > 0
        if (pending_side == OrderSide.BUY) != actual_is_long:
            self._record_pending_entry_recovery(
                pending,
                status="broker_position_direction_mismatch",
                error=(
                    f"pending entry side {pending_side.value} does not match broker position "
                    f"qty {float(actual.qty):g}"
                ),
            )
            raise RuntimeError(
                "Pending-entry recovery found broker exposure opposite the original entry intent; "
                "refusing automatic close. Operator reconciliation is required."
            )

        strategy_ref = {"id": str(pending["strategy_id"])}
        recovery_side = OrderSide.SELL if actual_is_long else OrderSide.BUY
        reference_price = float(actual.avg_price)
        if not math.isfinite(reference_price) or reference_price <= 0:
            reference_price = float(self.broker.get_price(str(pending["symbol"])))
        qty = self._normalize_broker_order_qty(
            str(pending["symbol"]),
            abs(float(actual.qty)),
            price=reference_price,
            reduce_only=True,
        )
        recovery_client_id = self._deterministic_client_order_id(
            strategy_id=str(pending["strategy_id"]),
            stage="recovery",
            intent_ref=str(pending["client_id"]),
            symbol=str(pending["symbol"]),
            side=recovery_side,
            qty=qty,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )
        recovery_order = Order(
            symbol=str(pending["symbol"]),
            side=recovery_side,
            qty=qty,
            type=OrderType.MARKET,
            reduce_only=True,
            client_id=recovery_client_id,
        )
        self._record_pending_entry_recovery(
            pending,
            status="recovery_close_prepared",
            recovery_order=recovery_order,
            increment_attempt=True,
        )
        try:
            fill = self.broker.place_order(recovery_order)
            self._assert_broker_exit_fill_valid(strategy_ref, recovery_order, qty, fill)
            self._assert_broker_flat_after_exit(strategy_ref, recovery_order.symbol)
        except Exception as exc:
            try:
                after = self.broker.get_position(recovery_order.symbol)
            except Exception as readback_exc:
                self._record_pending_entry_recovery(
                    pending,
                    status="recovery_close_unverified",
                    recovery_order=recovery_order,
                    error=(
                        f"{type(exc).__name__}: {exc}; position readback failed: "
                        f"{type(readback_exc).__name__}: {readback_exc}"
                    ),
                )
                raise RuntimeError(
                    "Pending-entry recovery close failed and broker-flat state could not be "
                    "verified. Original pending entry remains for operator reconciliation."
                ) from exc
            if after.is_flat:
                self._record_pending_entry_recovery(
                    pending,
                    status="broker_flat_after_ambiguous_recovery_close",
                    recovery_order=recovery_order,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            self._record_pending_entry_recovery(
                pending,
                status="recovery_close_failed_position_remains",
                recovery_order=recovery_order,
                error=f"{type(exc).__name__}: {exc}; remaining_qty={float(after.qty):g}",
            )
            raise RuntimeError(
                "Pending-entry recovery close failed and broker exposure remains. Original "
                "pending entry is preserved; operator reconciliation is required."
            ) from exc

        self._record_pending_entry_recovery(
            pending,
            status="recovery_close_filled_and_flat",
            recovery_order=recovery_order,
            fill=fill,
        )

    def _record_risk_recovery_incident(
        self,
        *,
        strategy_id: str,
        cause: str,
        status: str,
        recovery_order: Order | None = None,
        fill: Fill | None = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        existing = self.state.get("risk_recovery_incident")
        now = _utc_now_text()
        if existing is None:
            marker = {
                "version": 1,
                "strategy_id": strategy_id,
                "symbol": self.symbol,
                "broker_account_fingerprint": self._current_broker_account_fingerprint(),
                "cause": cause,
                "first_detected_at": now,
                "attempt_count": 0,
            }
        else:
            if not isinstance(existing, dict) or existing.get("version") != 1:
                raise RuntimeError(f"State risk_recovery_incident is invalid: {self.state_file}")
            if existing.get("strategy_id") != strategy_id or existing.get("symbol") != self.symbol:
                raise RuntimeError(
                    f"State risk_recovery_incident does not match current exposure: {self.state_file}"
                )
            self._assert_broker_account_fingerprint(
                existing.get("broker_account_fingerprint"),
                state_detail=f"risk_recovery_incident in {self.state_file}",
            )
            marker = dict(existing)
        attempts = marker.get("attempt_count", 0)
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            raise RuntimeError(
                f"State risk_recovery_incident.attempt_count is invalid: {self.state_file}"
            )
        marker["attempt_count"] = attempts + int(increment_attempt)
        marker["cause"] = cause
        marker["status"] = status
        marker["last_updated_at"] = now
        if recovery_order is not None:
            marker["recovery_client_id"] = recovery_order.client_id
            marker["recovery_side"] = recovery_order.side.value
            marker["recovery_qty"] = float(recovery_order.qty)
        if fill is not None:
            marker["fill"] = {
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": float(fill.qty),
                "price": float(fill.price),
                "fee": float(fill.fee),
                "timestamp": float(fill.timestamp),
            }
        if error is None:
            marker.pop("last_error", None)
        else:
            marker["last_error"] = str(error)
        self.state["risk_recovery_incident"] = marker
        self._save_state()

    def _recover_live_futures_exposure_incident(
        self,
        strategy: dict,
        open_position: dict | None,
        actual: Position,
        *,
        cause: str,
        cancel_native_stop: bool,
        finalize_accounting: bool = True,
    ) -> None:
        """Durably flatten all actual futures exposure and latch an incident."""

        if not self._requires_native_protective_stop() or actual.is_flat:
            raise RuntimeError("Live futures exposure recovery requires a non-flat live position.")
        if not _symbols_match(actual.symbol, self.symbol):
            raise RuntimeError("Live futures exposure recovery symbol mismatch; refusing close.")
        recovery_side = OrderSide.SELL if float(actual.qty) > 0 else OrderSide.BUY
        reference_price = float(actual.avg_price)
        if not math.isfinite(reference_price) or reference_price <= 0:
            reference_price = float(self.broker.get_price(self.symbol))
        qty = self._normalize_broker_order_qty(
            self.symbol,
            abs(float(actual.qty)),
            price=reference_price,
            reduce_only=True,
        )
        intent_ref = cause
        if open_position is not None:
            intent_ref = (
                f"{cause}|{open_position.get('entry_time')}|"
                f"{open_position.get('broker_stop_client_id')}"
            )
        client_id = self._deterministic_client_order_id(
            strategy_id=strategy["id"],
            stage="recovery",
            intent_ref=intent_ref,
            symbol=self.symbol,
            side=recovery_side,
            qty=qty,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )
        recovery_order = Order(
            symbol=self.symbol,
            side=recovery_side,
            qty=qty,
            type=OrderType.MARKET,
            reduce_only=True,
            client_id=client_id,
        )
        self._record_risk_recovery_incident(
            strategy_id=strategy["id"],
            cause=cause,
            status="recovery_close_prepared",
            recovery_order=recovery_order,
            increment_attempt=True,
        )
        try:
            fill = self.broker.place_order(recovery_order)
            self._assert_broker_exit_fill_valid(strategy, recovery_order, qty, fill)
            self._assert_broker_flat_after_exit(strategy, recovery_order.symbol)
            if cancel_native_stop and open_position is not None:
                self._cancel_native_protection(strategy, open_position)
        except Exception as exc:
            try:
                after = self.broker.get_position(recovery_order.symbol)
            except Exception as readback_exc:
                self._record_risk_recovery_incident(
                    strategy_id=strategy["id"],
                    cause=cause,
                    status="recovery_close_unverified",
                    recovery_order=recovery_order,
                    error=(
                        f"{type(exc).__name__}: {exc}; position readback failed: "
                        f"{type(readback_exc).__name__}: {readback_exc}"
                    ),
                )
                raise RuntimeError(
                    "Live futures safety recovery could not verify broker-flat state; "
                    "operator reconciliation is required."
                ) from exc
            status = (
                "broker_flat_after_ambiguous_recovery_close"
                if after.is_flat
                else "recovery_close_failed_position_remains"
            )
            self._record_risk_recovery_incident(
                strategy_id=strategy["id"],
                cause=cause,
                status=status,
                recovery_order=recovery_order,
                error=f"{type(exc).__name__}: {exc}; remaining_qty={float(after.qty):g}",
            )
            raise RuntimeError(
                "Live futures safety recovery is latched after an ambiguous or failed close; "
                "operator reconciliation is required."
            ) from exc

        self._record_risk_recovery_incident(
            strategy_id=strategy["id"],
            cause=cause,
            status=(
                "recovery_close_filled_and_flat"
                if finalize_accounting
                else "recovery_close_filled_flat_accounting_unresolved"
            ),
            recovery_order=recovery_order,
            fill=fill,
        )
        if open_position is not None and finalize_accounting:
            self._complete_position_exit(
                strategy,
                open_position,
                exit_time=pd.Timestamp.now(tz="UTC"),
                exit_price=float(fill.price),
                exit_reason=f"risk_recovery_{cause}",
                broker_exit_fill=fill,
                clear_pending=False,
            )
        raise RuntimeError(
            "Live futures exposure was emergency-flattened and a sticky risk-recovery "
            "incident now requires operator reconciliation."
        )

    def _resume_live_futures_risk_recovery(self) -> None:
        marker = self.state.get("risk_recovery_incident")
        if marker is None or not self._requires_native_protective_stop():
            return
        if not isinstance(marker, dict) or marker.get("version") != 1:
            raise RuntimeError(f"State risk_recovery_incident is invalid: {self.state_file}")
        self._assert_broker_account_fingerprint(
            marker.get("broker_account_fingerprint"),
            state_detail=f"risk_recovery_incident in {self.state_file}",
        )
        actual = self.broker.get_position(self.symbol)
        if actual.is_flat:
            return
        strategy_id = marker.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise RuntimeError(
                f"State risk_recovery_incident strategy id is invalid: {self.state_file}"
            )
        position = self.state.get("open_positions", {}).get(strategy_id)
        current = next(
            (item for item in self.strategies if item["id"] == strategy_id), {"id": strategy_id}
        )
        strategy = (
            self._strategy_for_open_position(current, position)
            if isinstance(position, dict)
            else current
        )
        self._recover_live_futures_exposure_incident(
            strategy,
            position if isinstance(position, dict) else None,
            actual,
            cause=str(marker.get("cause") or "resumed_risk_recovery"),
            cancel_native_stop=bool(
                isinstance(position, dict)
                and marker.get("cause") == "broker_position_quantity_mismatch"
            ),
        )

    @staticmethod
    def _exit_event_id(strategy_id: str, open_position: dict) -> str:
        """Identify the sole close transition for one durable position."""

        identity = {
            "version": 1,
            "strategy_id": strategy_id,
            "entry_time": open_position.get("entry_time"),
            "entry_price": open_position.get("entry_price"),
            "direction": open_position.get("direction"),
            "strategy_fingerprint": open_position.get("strategy_fingerprint"),
            "approval_strategy_fingerprint": open_position.get("approval_strategy_fingerprint"),
            "artifact_digest": open_position.get("artifact_digest"),
            "broker_symbol": open_position.get("broker_symbol"),
            "broker_entry_client_id": open_position.get("broker_entry_client_id"),
            "broker_account_fingerprint": open_position.get("broker_account_fingerprint"),
        }
        return _canonical_json_digest(identity, label="Exit event identity")

    def _validated_exit_accounting_intent(self, raw: object) -> dict:
        if not isinstance(raw, dict):
            raise RuntimeError(f"State exit_accounting_intent must be an object: {self.state_file}")
        missing = [key for key in EXIT_ACCOUNTING_INTENT_REQUIRED_KEYS if key not in raw]
        unexpected = [key for key in raw if key not in EXIT_ACCOUNTING_INTENT_REQUIRED_KEYS]
        if missing:
            raise RuntimeError(
                "State exit_accounting_intent is missing required key(s): "
                f"{', '.join(missing)}: {self.state_file}"
            )
        if unexpected:
            raise RuntimeError(
                "State exit_accounting_intent has unexpected key(s): "
                f"{', '.join(unexpected)}: {self.state_file}"
            )
        if raw.get("version") != 1:
            raise RuntimeError(f"State exit_accounting_intent.version must be 1: {self.state_file}")
        if raw.get("phase") not in {"ready_to_commit", "trade_logged"}:
            raise RuntimeError(
                "State exit_accounting_intent.phase must be ready_to_commit or trade_logged: "
                f"{self.state_file}"
            )
        event_id = raw.get("exit_event_id")
        if not isinstance(event_id, str) or re.fullmatch(r"[0-9a-f]{64}", event_id) is None:
            raise RuntimeError(
                f"State exit_accounting_intent.exit_event_id is invalid: {self.state_file}"
            )
        strategy_id = raw.get("strategy_id")
        if not isinstance(strategy_id, str) or not strategy_id:
            raise RuntimeError(
                f"State exit_accounting_intent.strategy_id is invalid: {self.state_file}"
            )
        if not isinstance(raw.get("created_at"), str) or not raw["created_at"]:
            raise RuntimeError(
                f"State exit_accounting_intent.created_at is invalid: {self.state_file}"
            )
        for key in ("state_before_digest", "position_digest", "payload_digest"):
            value = raw.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise RuntimeError(
                    f"State exit_accounting_intent.{key} is invalid: {self.state_file}"
                )
        if not isinstance(raw.get("broker_flat_proven"), bool):
            raise RuntimeError(
                "State exit_accounting_intent.broker_flat_proven must be boolean: "
                f"{self.state_file}"
            )
        trade_data = raw.get("trade_data")
        state_after = raw.get("state_after")
        if not isinstance(trade_data, dict) or not isinstance(state_after, dict):
            raise RuntimeError(
                "State exit_accounting_intent trade_data and state_after must be objects: "
                f"{self.state_file}"
            )
        if trade_data.get("exit_event_id") != event_id:
            raise RuntimeError(
                f"State exit_accounting_intent trade event id mismatch: {self.state_file}"
            )
        if trade_data.get("strategy_id") != strategy_id:
            raise RuntimeError(
                f"State exit_accounting_intent trade strategy mismatch: {self.state_file}"
            )
        if "exit_accounting_intent" in state_after:
            raise RuntimeError(
                "State exit_accounting_intent.state_after must not contain an intent: "
                f"{self.state_file}"
            )
        positions_after = state_after.get("open_positions")
        if not isinstance(positions_after, dict) or strategy_id in positions_after:
            raise RuntimeError(
                "State exit_accounting_intent.state_after does not close its position: "
                f"{self.state_file}"
            )

        payload_without_digest = {
            key: value for key, value in raw.items() if key != "payload_digest"
        }
        actual_payload_digest = _canonical_json_digest(
            payload_without_digest,
            label="Exit accounting intent payload",
        )
        if actual_payload_digest != raw["payload_digest"]:
            raise RuntimeError(
                f"State exit_accounting_intent failed its integrity check: {self.state_file}"
            )

        state_before = copy.deepcopy(self.state)
        state_before.pop("exit_accounting_intent", None)
        if (
            _canonical_json_digest(state_before, label="Exit accounting pre-state")
            != raw["state_before_digest"]
        ):
            raise RuntimeError(
                "State changed underneath exit_accounting_intent; refusing duplicate or "
                f"partial accounting: {self.state_file}"
            )
        positions_before = state_before.get("open_positions")
        position = positions_before.get(strategy_id) if isinstance(positions_before, dict) else None
        if not isinstance(position, dict):
            raise RuntimeError(
                "State exit_accounting_intent has no matching pre-close position: "
                f"{self.state_file}"
            )
        if (
            _canonical_json_digest(position, label="Exit accounting position")
            != raw["position_digest"]
        ):
            raise RuntimeError(
                "State exit_accounting_intent position failed its integrity check: "
                f"{self.state_file}"
            )
        if self._exit_event_id(strategy_id, position) != event_id:
            raise RuntimeError(
                "State exit_accounting_intent event id does not match the position: "
                f"{self.state_file}"
            )
        if self._is_live_broker():
            self._assert_broker_account_fingerprint(
                position.get("broker_account_fingerprint"),
                state_detail="Exit accounting remains uncommitted.",
            )
        return raw

    def _resume_exit_accounting_intent(self) -> bool:
        """Finish a broker-flat exit without ever submitting another order."""

        raw = self.state.get("exit_accounting_intent")
        if raw is None:
            return False
        intent = self._validated_exit_accounting_intent(raw)
        if intent["broker_flat_proven"]:
            if self.broker is None:
                raise RuntimeError(
                    "Live exit accounting cannot resume without a broker-flat readback."
                )
            symbol = intent["trade_data"].get("broker_symbol")
            if not isinstance(symbol, str) or not symbol:
                raise RuntimeError("Live exit accounting intent is missing its broker symbol.")
            actual = self.broker.get_position(symbol)
            if not actual.is_flat:
                raise RuntimeError(
                    "Exit accounting intent was prepared after a flat proof, but the broker "
                    f"now reports exposure {float(actual.qty):g}; refusing local commit."
                )

        # CSV replacement is atomic and keyed. If the process died after this
        # append, the retry verifies the existing row instead of duplicating it.
        self._append_trade_data_idempotent(intent["trade_data"])
        if intent["phase"] == "ready_to_commit":
            logged_without_digest = {
                key: copy.deepcopy(value)
                for key, value in intent.items()
                if key != "payload_digest"
            }
            logged_without_digest["phase"] = "trade_logged"
            logged_intent = {
                **logged_without_digest,
                "payload_digest": _canonical_json_digest(
                    logged_without_digest,
                    label="Exit accounting intent payload",
                ),
            }
            previous_intent = self.state["exit_accounting_intent"]
            self.state["exit_accounting_intent"] = logged_intent
            try:
                self._validated_exit_accounting_intent(logged_intent)
                self._save_state()
            except Exception:
                self.state["exit_accounting_intent"] = previous_intent
                raise
            intent = logged_intent
        target_state = copy.deepcopy(intent["state_after"])
        try:
            _reject_symlink_path(self.state_file, "State file")
            write_json_atomic(self.state_file, target_state)
        except Exception:
            # Keep the durable intent in memory as well as on disk so an
            # in-process retry follows the same idempotent recovery path.
            raise
        self.state = target_state
        trade_data = intent["trade_data"]
        self.position_events.append(
            {
                "schema": "autopilot.position_event/v1",
                "event_id": intent["exit_event_id"],
                "event_type": "closed",
                "strategy_id": intent["strategy_id"],
                "symbol": trade_data.get("broker_symbol") or self.symbol,
                "market": self.market,
                "execution": "broker" if self.broker is not None else "paper",
                **{
                    key: trade_data.get(key)
                    for key in (
                        "direction",
                        "entry_time",
                        "exit_time",
                        "entry_price",
                        "exit_price",
                        "exit_reason",
                        "net_return",
                        "sized_return",
                        "position_size",
                        "equity_after",
                    )
                },
            }
        )
        LOGGER.critical(
            "EXIT ACCOUNTING COMMITTED [%s]: event=%s equity=%.8f",
            intent["strategy_id"],
            intent["exit_event_id"],
            float(self.state["equity"]),
        )
        return True

    def _assert_no_pending_order(self) -> None:
        flatten_intent = self.state.get("flatten_intent")
        if flatten_intent is not None:
            client_id = (
                flatten_intent.get("client_id") if isinstance(flatten_intent, dict) else None
            )
            raise RuntimeError(
                "Unresolved emergency flatten intent blocks all trading cycles until "
                "operator reconciliation: "
                f"client_id={client_id}."
            )
        exit_intent = self.state.get("exit_accounting_intent")
        if exit_intent is not None:
            event_id = exit_intent.get("exit_event_id") if isinstance(exit_intent, dict) else None
            raise RuntimeError(
                "Unresolved exit accounting intent blocks trading until its idempotent "
                f"state transition completes: exit_event_id={event_id}."
            )
        pending_entry_recovery = self.state.get("pending_entry_recovery")
        if pending_entry_recovery is not None:
            if not isinstance(pending_entry_recovery, dict):
                raise RuntimeError(
                    f"Unresolved pending-entry recovery has invalid state: {self.state_file}"
                )
            raise RuntimeError(
                "Unresolved pending-entry recovery blocks all trading cycles until operator "
                "reconciliation: "
                f"status={pending_entry_recovery.get('status')} "
                f"original_client_id={pending_entry_recovery.get('original_pending_client_id')} "
                f"recovery_client_id={pending_entry_recovery.get('recovery_client_id')}."
            )
        risk_recovery_incident = self.state.get("risk_recovery_incident")
        if risk_recovery_incident is not None:
            if not isinstance(risk_recovery_incident, dict):
                raise RuntimeError(
                    f"Unresolved risk-recovery incident has invalid state: {self.state_file}"
                )
            raise RuntimeError(
                "Unresolved risk-recovery incident blocks all trading cycles until operator "
                "reconciliation: "
                f"cause={risk_recovery_incident.get('cause')} "
                f"status={risk_recovery_incident.get('status')} "
                f"recovery_client_id={risk_recovery_incident.get('recovery_client_id')}."
            )
        pending = self.state.get("pending_order")
        if pending is None:
            return
        if not isinstance(pending, dict):  # Defensive for in-memory mutation after load.
            raise RuntimeError(
                f"Unresolved pending broker order has invalid state: {self.state_file}"
            )
        raise RuntimeError(
            "Unresolved pending broker order blocks all trading cycles until operator reconciliation: "
            f"stage={pending.get('stage')} strategy={pending.get('strategy_id')} "
            f"client_id={pending.get('client_id')}."
        )

    def _persist_pending_order(
        self,
        strategy: dict,
        *,
        stage: str,
        intent_ref: str,
        order: Order,
    ) -> None:
        self._assert_no_pending_order()
        pending = self._validated_pending_order(
            {
                "version": 1,
                "strategy_id": strategy["id"],
                "stage": stage,
                "intent_ref": intent_ref,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": float(order.qty),
                "order_type": order.type.value,
                "reduce_only": bool(order.reduce_only),
                "client_id": order.client_id,
                "broker_account_fingerprint": self._current_broker_account_fingerprint(),
                "created_ts": time.time(),
            }
        )
        self.state["pending_order"] = pending
        self._save_state()

    def _save_state_clearing_pending_order(self) -> None:
        if "pending_order" not in self.state:
            raise RuntimeError(
                "Cannot complete broker state transition without a pending order intent."
            )
        pending = self.state.pop("pending_order")
        try:
            self._save_state()
        except Exception:
            self.state["pending_order"] = pending
            raise

    def _load_state(self):
        _reject_symlink_path(self.state_file, "State file")
        if self.state_file.exists():
            try:
                loaded = json.loads(self.state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"State file is unreadable or invalid: {self.state_file}: {exc}"
                ) from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"State file must contain a JSON object: {self.state_file}")
            self.state = loaded
            if "exit_accounting_intent" in self.state:
                # The intent hashes the exact pre-transition state. Do not run
                # migration normalizers before recovery or their benign writes
                # would look like an unsafe concurrent accounting mutation.
                self._validated_exit_accounting_intent(self.state["exit_accounting_intent"])
                LOGGER.warning(
                    "Loaded unresolved exit accounting intent %s; it will be resumed "
                    "before any order or market-data work.",
                    self.state["exit_accounting_intent"].get("exit_event_id"),
                )
                return
            changed = False
            if "open_positions" not in self.state:
                self.state["open_positions"] = {}
                changed = True
            elif not isinstance(self.state["open_positions"], dict):
                raise RuntimeError(f"State open_positions must be an object: {self.state_file}")
            if "inactive_strategies" not in self.state:
                self.state["inactive_strategies"] = []
                changed = True
            elif not isinstance(self.state["inactive_strategies"], list):
                raise RuntimeError(f"State inactive_strategies must be a list: {self.state_file}")
            for key, default in (
                ("equity", self.starting_equity),
                ("consecutive_losses", 0),
                ("cooldown_until_ts", 0.0),
                ("daily_pnl", 0.0),
                ("daily_trades_by_strategy", {}),
                ("last_entry_decision_bar_by_strategy", {}),
                ("last_pnl_reset_date", str(_utc_date_today())),
            ):
                if key not in self.state:
                    self.state[key] = default
                    changed = True
            if not isinstance(self.state["daily_trades_by_strategy"], dict):
                raise RuntimeError(
                    f"State daily_trades_by_strategy must be an object: {self.state_file}"
                )
            if not isinstance(self.state["last_entry_decision_bar_by_strategy"], dict):
                raise RuntimeError(
                    f"State last_entry_decision_bar_by_strategy must be an object: {self.state_file}"
                )
            changed = self._normalize_state_float("equity", positive=True) or changed
            changed = self._normalize_state_int("consecutive_losses", non_negative=True) or changed
            changed = self._normalize_state_float("cooldown_until_ts", non_negative=True) or changed
            changed = self._normalize_state_float("daily_pnl") or changed
            changed = self._normalize_drawdown_state() or changed
            changed = self._normalize_daily_trade_counts() or changed
            changed = self._normalize_entry_decision_bars() or changed
            changed = self._normalize_last_pnl_reset_date() or changed
            changed = self._normalize_open_positions() or changed
            changed = self._normalize_pending_order() or changed
            self.state.pop("open_position", None)
            self.state.pop("strategy_active", None)
            if changed:
                self._save_state()
            LOGGER.info(
                "Loaded bot state. Current Equity: %.2f USDT",
                self.state.get("equity", self.starting_equity),
            )
        else:
            self.state = {
                "equity": self.starting_equity,
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": {},
                "last_entry_decision_bar_by_strategy": {},
                "last_pnl_reset_date": str(_utc_date_today()),
                "peak_equity": self.starting_equity,
                "drawdown_fraction": 0.0,
                "drawdown_limit_fraction": self._max_equity_drawdown(),
                "drawdown_halted": False,
                "drawdown_halted_at": None,
                "drawdown_halt_reason": None,
            }
            self._normalize_state_float("equity", positive=True)
            self._normalize_drawdown_state()
            self._save_state()
            LOGGER.info("Initialized new paper trading state with %.2f USDT.", self.starting_equity)

    def _save_state(self):
        _reject_symlink_path(self.state_file, "State file")
        write_json_atomic(self.state_file, self.state)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    # Both spot and futures kline endpoints accept up to 1000 rows per request.
    KLINES_PER_REQUEST = 1000
    POSITIVE_CANDLE_COLUMNS = ("open", "high", "low", "close")
    NON_NEGATIVE_CANDLE_COLUMNS = (
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )

    @classmethod
    def _validate_closed_candles(cls, df: pd.DataFrame, *, symbol: str, timeframe: str) -> None:
        if df.empty:
            return
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        if timestamps.isna().any():
            raise RuntimeError(
                f"Closed {timeframe} candles for {symbol} contain invalid timestamps."
            )
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise RuntimeError(
                f"Closed {timeframe} candles for {symbol} must have strictly increasing timestamps."
            )
        for column in cls.POSITIVE_CANDLE_COLUMNS:
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Closed {timeframe} candles for {symbol} contain non-finite {column}."
                )
            if (values <= 0).any():
                raise RuntimeError(
                    f"Closed {timeframe} candles for {symbol} contain non-positive {column}."
                )
        for column in cls.NON_NEGATIVE_CANDLE_COLUMNS:
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Closed {timeframe} candles for {symbol} contain non-finite {column}."
                )
            if (values < 0).any():
                raise RuntimeError(
                    f"Closed {timeframe} candles for {symbol} contain negative {column}."
                )
        open_values = df["open"].to_numpy(dtype=float)
        high_values = df["high"].to_numpy(dtype=float)
        low_values = df["low"].to_numpy(dtype=float)
        close_values = df["close"].to_numpy(dtype=float)
        if (high_values < low_values).any():
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain high below low.")
        if (high_values < np.maximum(open_values, close_values)).any():
            raise RuntimeError(
                f"Closed {timeframe} candles for {symbol} contain high below open/close."
            )
        if (low_values > np.minimum(open_values, close_values)).any():
            raise RuntimeError(
                f"Closed {timeframe} candles for {symbol} contain low above open/close."
            )

    def fetch_live_candles(
        self, symbol: str, market: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Fetch recent candles and DROP the still-forming last candle.

        Binance returns the in-progress kline as the final row; evaluating
        signals on it would repaint intra-bar and diverge from the research
        simulation, which only ever sees closed candles.

        ``limit`` may exceed one request's worth (deep rolling-quantile windows
        need thousands of base bars); older pages are then fetched via endTime.
        """
        if market == "futures":
            url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            url = "https://api.binance.com/api/v3/klines"

        data: list = []
        remaining = int(limit)
        end_time_ms = None
        while remaining > 0:
            params = {
                "symbol": _binance_rest_symbol(symbol),
                "interval": timeframe,
                "limit": min(remaining, self.KLINES_PER_REQUEST),
            }
            if end_time_ms is not None:
                params["endTime"] = end_time_ms
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Binance API error: {response.text}")
            batch = response.json()
            if not batch:
                break
            data = batch + data
            remaining -= len(batch)
            if len(batch) < params["limit"]:
                break  # history exhausted
            end_time_ms = int(batch[0][0]) - 1

        df = pd.DataFrame(data, columns=bbid.BINANCE_COLUMNS)
        df["open_time"] = pd.to_numeric(df["open_time"])
        for col in bbid.CANDLE_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Pin to nanosecond resolution: newer pandas otherwise yields ms/us
        # units that later make merge_asof refuse to join timeframes.
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).astype(
            "datetime64[ns, UTC]"
        )
        df = df[bbid.CANDLE_COLUMNS]
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 300)
        now = pd.Timestamp.now(tz="UTC")
        closed = df["timestamp"] + pd.Timedelta(seconds=tf_seconds) <= now
        df_closed = df[closed].reset_index(drop=True)
        self._validate_closed_candles(df_closed, symbol=symbol, timeframe=timeframe)
        return df_closed

    def fetch_public_observation_quote(self) -> tuple[float, pd.Timestamp]:
        """Fetch a credential-free quote at the moment a fresh signal is observed."""

        if self.market == "futures":
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
        elif self.market == "spot":
            url = "https://api.binance.com/api/v3/ticker/price"
        else:
            raise RuntimeError(f"Candidate paper quote does not support market {self.market!r}.")
        response = requests.get(
            url,
            params={"symbol": _binance_rest_symbol(self.data_symbol)},
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Binance public quote API error: {response.text}")
        try:
            payload = response.json()
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Binance public quote response has no valid price.") from exc
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError("Binance public quote price must be finite and positive.")
        return price, pd.Timestamp.now(tz="UTC")

    # Indicator warmup margin: the longest bundled indicator lookback is ~200
    # native bars (ema_200 / max_200), plus slack so the last rows are non-NaN.
    INDICATOR_WARMUP_BARS = 250

    def _split_prefixed_feature(self, feature: str, default_tf: str) -> tuple[str, str]:
        if feature.startswith("tf_"):
            parts = feature.split("_", 2)
            if len(parts) == 3:
                return parts[1], parts[2]
        return default_tf, feature

    def _required_features_by_timeframe(self, strategy: dict) -> dict[str, set[str]]:
        base_tf = strategy["base_timeframe"]
        required: dict[str, set[str]] = {base_tf: {"open", "high", "low", "close"}}
        if strategy.get("use_atr_tp_sl"):
            required[base_tf].add("atr")
        if strategy.get("entry_type", "conditions") == "hypothesis":
            hypothesis = strategy["_hypothesis"]
            for predicate in hypothesis.all_predicates():
                required.setdefault(predicate.timeframe, {"open", "high", "low", "close"}).add(
                    predicate.feature
                )
                if predicate.feature_b:
                    required[predicate.timeframe].add(predicate.feature_b)
            if hypothesis.risk.min_atr_pct or hypothesis.risk.max_atr_pct:
                required.setdefault(base_tf, {"open", "high", "low", "close"}).add("natr_14")
            return required

        if strategy.get("entry_type") == "frozen_ml":
            for feature in strategy["_frozen_ml"].feature_names:
                timeframe, name = self._split_prefixed_feature(feature, base_tf)
                required.setdefault(timeframe, {"open", "high", "low", "close"}).add(name)
            return required

        for condition in strategy["_conditions"]:
            tf, feature = self._split_prefixed_feature(condition.feature, base_tf)
            required.setdefault(tf, {"open", "high", "low", "close"}).add(feature)
            if condition.feature_b:
                tf_b, feature_b = self._split_prefixed_feature(condition.feature_b, tf)
                required.setdefault(tf_b, {"open", "high", "low", "close"}).add(feature_b)
        return required

    def _build_indicator_features(
        self,
        df: pd.DataFrame,
        timeframe: str,
        required_features: set[str],
    ) -> pd.DataFrame:
        builder = bbid.build_indicator_features
        signature_target = getattr(builder, "side_effect", None) or builder
        try:
            signature = inspect.signature(signature_target)
            supports_required = "required_features" in signature.parameters or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_required = True
        if not supports_required:
            return builder(df, timeframe)
        try:
            return builder(df, timeframe, required_features=required_features)
        except TypeError as exc:
            if "required_features" not in str(exc):
                raise
            return builder(df, timeframe)

    def _fetch_limits(self, strategy: dict) -> tuple[int, dict[str, int]]:
        """(base_limit, {higher_tf: limit}) sized so every feature and rolling
        predicate window the strategy references is defined on the last bar."""
        base_tf = strategy["base_timeframe"]
        if strategy.get("entry_type", "conditions") == "hypothesis":
            needed = hypothesis_history_requirements(strategy["_hypothesis"])
            base_limit = max(500, needed.get(base_tf, 1) + self.INDICATOR_WARMUP_BARS)
            replay_bars = getattr(self, "_candidate_replay_max_unseen_bars", None)
            if replay_bars is not None:
                base_limit = max(
                    base_limit,
                    int(replay_bars) + self.INDICATOR_WARMUP_BARS + 2,
                )
            htf_limits = {
                tf: max(200, bars + self.INDICATOR_WARMUP_BARS)
                for tf, bars in needed.items()
                if tf != base_tf
            }
            return base_limit, htf_limits
        required_tfs = {
            tf for tf in self._required_features_by_timeframe(strategy) if tf != base_tf
        }
        base_limit = 500
        replay_bars = getattr(self, "_candidate_replay_max_unseen_bars", None)
        if replay_bars is not None:
            base_limit = max(
                base_limit,
                int(replay_bars) + self.INDICATOR_WARMUP_BARS + 2,
            )
        return base_limit, {tf: 200 for tf in required_tfs}

    def _feature_frame_cache_key(
        self,
        *,
        base_tf: str,
        base_limit: int,
        htf_limits: dict[str, int],
        required_features: dict[str, set[str]],
    ) -> tuple:
        return (
            self.data_symbol,
            self.market,
            base_tf,
            int(base_limit),
            tuple(sorted((tf, int(limit)) for tf, limit in htf_limits.items())),
            tuple(
                sorted((tf, tuple(sorted(features))) for tf, features in required_features.items())
            ),
        )

    def _build_feature_frame(self, strategy: dict) -> tuple[pd.DataFrame, float]:
        """Fetch candles for the strategy's base + required higher timeframes
        and assemble the aligned indicator frame (closed candles only)."""
        base_tf = strategy["base_timeframe"]
        base_limit, htf_limits = self._fetch_limits(strategy)
        required_features = self._required_features_by_timeframe(strategy)
        cache_key = self._feature_frame_cache_key(
            base_tf=base_tf,
            base_limit=base_limit,
            htf_limits=htf_limits,
            required_features=required_features,
        )
        cached = self._feature_frame_cache.get(cache_key)
        if cached is not None:
            frame, base_close = cached
            return frame.copy(), base_close

        df_base = self.fetch_live_candles(self.data_symbol, self.market, base_tf, limit=base_limit)
        if df_base.empty:
            raise RuntimeError(f"No closed {base_tf} candles returned for {self.symbol}.")
        base_close = float(df_base["close"].iloc[-1])

        df_base_ind = self._build_indicator_features(
            df_base,
            base_tf,
            required_features.get(base_tf, set()),
        )
        base_prefix = f"tf_{base_tf}_"
        df_base_ind = df_base_ind.rename(
            columns={c: f"{base_prefix}{c}" for c in df_base_ind.columns if c != "timestamp"}
        )
        df_base_ind["timestamp"] = pd.to_datetime(df_base_ind["timestamp"], utc=True).astype(
            "datetime64[ns, UTC]"
        )

        merged = df_base_ind.copy()
        for tf in sorted(htf_limits):
            df_tf = self.fetch_live_candles(self.data_symbol, self.market, tf, limit=htf_limits[tf])
            if df_tf.empty:
                raise RuntimeError(f"No closed {tf} candles returned for {self.symbol}.")
            df_tf_ind = self._build_indicator_features(
                df_tf,
                tf,
                required_features.get(tf, set()),
            )
            tf_prefix = f"tf_{tf}_"
            df_tf_ind = df_tf_ind.rename(
                columns={c: f"{tf_prefix}{c}" for c in df_tf_ind.columns if c != "timestamp"}
            )
            df_tf_ind["timestamp"] = pd.to_datetime(df_tf_ind["timestamp"], utc=True).astype(
                "datetime64[ns, UTC]"
            )
            tf_shifted = df_tf_ind.copy()
            seconds = TIMEFRAME_SECONDS.get(tf, 300)
            tf_shifted["timestamp"] = (
                tf_shifted["timestamp"] + pd.Timedelta(seconds=seconds)
            ).astype("datetime64[ns, UTC]")
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                tf_shifted.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                allow_exact_matches=True,
            )

        self._feature_frame_cache[cache_key] = (merged.copy(), base_close)
        return merged, base_close

    # ------------------------------------------------------------------
    # Safety nets
    # ------------------------------------------------------------------
    def check_drift_and_ood(self, strategy: dict):
        """Win-rate drift z-test against the exported baseline. Deactivates
        the strategy (not the whole bot) when live results are significantly
        worse than the research baseline."""
        baseline_wr = strategy.get("baseline_win_rate")
        if not baseline_wr or baseline_wr <= 0 or baseline_wr >= 1:
            LOGGER.warning(
                "Strategy %s has no usable baseline win rate; drift detection is disabled for it.",
                strategy["id"],
            )
            return
        _reject_symlink_path(self.trade_log, "Trade log")
        if not self.trade_log.exists():
            return
        df_trades = pd.read_csv(self.trade_log)
        if "strategy_id" in df_trades.columns:
            df_trades = df_trades[df_trades["strategy_id"] == strategy["id"]]
        if len(df_trades) < DRIFT_MIN_TRADES:
            return
        recent = df_trades.tail(DRIFT_WINDOW_TRADES)
        recent_win_rate = float((recent["net_return"] > 0).mean())
        std_error = np.sqrt(baseline_wr * (1 - baseline_wr) / len(recent))
        z_score = (recent_win_rate - baseline_wr) / std_error if std_error > 0 else 0.0
        LOGGER.info(
            "Drift %s: trades=%d baseline WR=%.2f recent WR=%.2f z=%.2f",
            strategy["id"],
            len(recent),
            baseline_wr,
            recent_win_rate,
            z_score,
        )
        if z_score < DRIFT_Z_THRESHOLD:
            LOGGER.critical(
                "OOD KILL SWITCH: %s win rate drifted significantly below baseline — deactivating.",
                strategy["id"],
            )
            if strategy["id"] not in self.state["inactive_strategies"]:
                self.state["inactive_strategies"].append(strategy["id"])
            self._save_state()

    def process_daily_reset(self):
        today = str(_utc_date_today())
        if self.state["last_pnl_reset_date"] != today:
            LOGGER.info("New day detected. Resetting daily PNL tracker.")
            self.state["daily_pnl"] = 0.0
            self.state["daily_trades_by_strategy"] = {}
            self.state["last_pnl_reset_date"] = today
            self._save_state()

    def _daily_trade_count(self, strategy: dict) -> int:
        counts = self.state.setdefault("daily_trades_by_strategy", {})
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"State daily_trades_by_strategy must be an object: {self.state_file}"
            )
        return int(counts.get(strategy["id"], 0) or 0)

    def _increment_daily_trade_count(self, strategy: dict) -> None:
        counts = self.state.setdefault("daily_trades_by_strategy", {})
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"State daily_trades_by_strategy must be an object: {self.state_file}"
            )
        counts[strategy["id"]] = self._daily_trade_count(strategy) + 1

    def _daily_trade_limit_reached(self, strategy: dict) -> bool:
        limit = (strategy.get("risk") or {}).get("max_trades_per_day")
        if limit is None:
            return False
        limit = int(limit)
        if limit <= 0:
            return True
        return self._daily_trade_count(strategy) >= limit

    @staticmethod
    def _normalized_bar_timestamp(value) -> str:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise RuntimeError("Latest closed bar has an invalid timestamp.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()

    def _entry_decision_already_processed(self, strategy: dict, bar_timestamp) -> bool:
        current = pd.Timestamp(self._normalized_bar_timestamp(bar_timestamp))
        raw_previous = self.state["last_entry_decision_bar_by_strategy"].get(strategy["id"])
        if raw_previous is None:
            return False
        return pd.Timestamp(raw_previous) >= current

    def _mark_entry_decision_processed(self, strategy: dict, bar_timestamp) -> None:
        self.state["last_entry_decision_bar_by_strategy"][strategy["id"]] = (
            self._normalized_bar_timestamp(bar_timestamp)
        )
        self._save_state()

    def _has_other_open_position(self, strategy: dict) -> bool:
        positions = self.state.get("open_positions", {})
        if not isinstance(positions, dict):
            raise RuntimeError(f"State open_positions must be an object: {self.state_file}")
        return any(strategy_id != strategy["id"] for strategy_id in positions)

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    def _evaluate_macro_regime(self):
        """Refresh the macro step-aside state once per cycle (BTC daily candles)."""
        if not self.regime_guard:
            return
        try:
            df_daily = self.fetch_live_candles(self.symbol, self.market, "1d", limit=500)
            self._macro_aside, self._macro_detail = compute_macro_step_aside(
                df_daily["close"], mayer_top=self.regime_mayer_top
            )
            LOGGER.info(
                "Macro regime: %s | %s",
                "STEP ASIDE (risk-off)" if self._macro_aside else "engaged (risk-on)",
                self._macro_detail,
            )
        except Exception as exc:
            LOGGER.error("Macro regime evaluation failed (%s); blocking new entries.", exc)
            self._macro_aside, self._macro_detail = (
                True,
                {
                    "error": str(exc),
                    "fail_closed": True,
                },
            )

    def _normalize_candidate_replay_state(self) -> bool:
        """Validate the paper-only event cursor and pending next-open entries."""

        changed = False
        version = self.state.get("candidate_replay_schema_version")
        if version is None:
            self.state["candidate_replay_schema_version"] = CANDIDATE_REPLAY_SCHEMA_VERSION
            changed = True
        elif version == 1 and CANDIDATE_REPLAY_SCHEMA_VERSION == 2:
            legacy_pending = self.state.get(CANDIDATE_REPLAY_PENDING_KEY, {})
            if not isinstance(legacy_pending, dict):
                raise RuntimeError(f"State {CANDIDATE_REPLAY_PENDING_KEY} must be an object.")
            if self.state.get("open_positions") or legacy_pending:
                raise RuntimeError(
                    "Candidate replay v1 cannot be upgraded while replay exposure or a "
                    "pending entry exists; preserve the old environment to close it or "
                    "explicitly abandon that isolated paper state."
                )
            self.state["candidate_replay_schema_version"] = CANDIDATE_REPLAY_SCHEMA_VERSION
            self.state[CANDIDATE_REPLAY_CURSOR_KEY] = {}
            self.state[CANDIDATE_REPLAY_PENDING_KEY] = {}
            changed = True
        elif version != CANDIDATE_REPLAY_SCHEMA_VERSION:
            raise RuntimeError(
                f"Candidate replay state has an unsupported schema version: {version!r}."
            )

        cursors = self.state.setdefault(CANDIDATE_REPLAY_CURSOR_KEY, {})
        pending = self.state.setdefault(CANDIDATE_REPLAY_PENDING_KEY, {})
        if not isinstance(cursors, dict):
            raise RuntimeError(f"State {CANDIDATE_REPLAY_CURSOR_KEY} must be an object.")
        if not isinstance(pending, dict):
            raise RuntimeError(f"State {CANDIDATE_REPLAY_PENDING_KEY} must be an object.")

        known_ids = {strategy["id"] for strategy in self.strategies}
        unknown = (set(cursors) | set(pending)) - known_ids
        if unknown:
            raise RuntimeError(
                "Candidate replay state references unknown strategies: "
                f"{', '.join(sorted(unknown))}."
            )
        if len(pending) > 1:
            raise RuntimeError(
                "Candidate replay state cannot contain more than one pending entry "
                "for the shared product account."
            )

        expected_schema = CANDIDATE_PAPER_EXECUTION_SCHEMA
        expected_engine = getattr(self, "_candidate_paper_engine_digest", None)
        if not isinstance(expected_engine, str):
            raise RuntimeError("Candidate replay execution identity was not initialized.")
        stored_schema = self.state.get("candidate_paper_execution_schema")
        stored_engine = self.state.get("candidate_paper_engine_digest")
        if stored_schema != expected_schema or stored_engine != expected_engine:
            if self.state["open_positions"] or pending:
                raise RuntimeError(
                    "Candidate paper execution identity changed while replay exposure or a "
                    "pending entry exists; preserve the old environment to close it or "
                    "explicitly abandon that isolated paper state."
                )
            history = self.state.setdefault("candidate_paper_execution_history", [])
            if not isinstance(history, list):
                raise RuntimeError("State candidate_paper_execution_history must be a list.")
            if stored_schema is not None or stored_engine is not None:
                history.append(
                    {
                        "execution_schema": stored_schema,
                        "engine_digest": stored_engine,
                        "superseded_at": _utc_now_text(),
                    }
                )
                del history[:-8]
            # Execution changes start a clean forward-paper account. Historical
            # CSV rows remain immutable and are quarantined by promotion's
            # exact schema/engine filter.
            self.state.update(
                equity=self.starting_equity,
                open_positions={},
                inactive_strategies=[],
                consecutive_losses=0,
                cooldown_until_ts=0.0,
                daily_pnl=0.0,
                daily_trades_by_strategy={},
                last_entry_decision_bar_by_strategy={},
                last_pnl_reset_date=str(_utc_date_today()),
                peak_equity=self.starting_equity,
                drawdown_fraction=0.0,
                drawdown_limit_fraction=self._max_equity_drawdown(),
                drawdown_halted=False,
                drawdown_halted_at=None,
                drawdown_halt_reason=None,
                candidate_paper_execution_schema=expected_schema,
                candidate_paper_engine_digest=expected_engine,
            )
            cursors.clear()
            pending.clear()
            changed = True
        for strategy_id, raw_cursor in list(cursors.items()):
            normalized = self._normalized_bar_timestamp(raw_cursor)
            if raw_cursor != normalized:
                cursors[strategy_id] = normalized
                changed = True
        pending_keys = {
            "signal_time",
            "signal_observed_at",
            "evidence_eligible",
            "evidence_reason",
            "fill_source",
        }
        for strategy_id, item in pending.items():
            if not isinstance(item, dict) or set(item) != pending_keys:
                raise RuntimeError(
                    f"State {CANDIDATE_REPLAY_PENDING_KEY}[{strategy_id!r}] must contain "
                    f"exactly {', '.join(sorted(pending_keys))}."
                )
            normalized = self._normalized_bar_timestamp(item["signal_time"])
            if item["signal_time"] != normalized:
                item["signal_time"] = normalized
                changed = True
            observed = self._normalized_bar_timestamp(item["signal_observed_at"])
            if item["signal_observed_at"] != observed:
                item["signal_observed_at"] = observed
                changed = True
            if item["evidence_eligible"] is not False:
                raise RuntimeError(
                    f"State {CANDIDATE_REPLAY_PENDING_KEY}[{strategy_id!r}] cannot mark "
                    "historical next-open replay as promotable."
                )
            if item["evidence_reason"] != CANDIDATE_PAPER_BACKFILL_ENTRY_REASON:
                raise RuntimeError(
                    f"State {CANDIDATE_REPLAY_PENDING_KEY}[{strategy_id!r}] has an "
                    "invalid evidence reason."
                )
            if item["fill_source"] != CANDIDATE_PAPER_BACKFILL_FILL_SOURCE:
                raise RuntimeError(
                    f"State {CANDIDATE_REPLAY_PENDING_KEY}[{strategy_id!r}] has an "
                    "invalid fill source."
                )
        return changed

    @staticmethod
    def _candidate_replay_signal(strategy: dict, frame: pd.DataFrame) -> bool:
        if strategy.get("entry_type", "conditions") == "hypothesis":
            return bool(entry_mask(frame, strategy["_hypothesis"]).iloc[-1])
        if strategy.get("entry_type") == "frozen_ml":
            return PaperTradingBot._trace_frozen_ml_signal(strategy, frame)[0]
        for condition in strategy["_conditions"]:
            if not bool(condition_mask(frame, condition).fillna(False).iloc[-1]):
                return False
        return True

    def _candidate_replay_daily_reset(self, event_close: pd.Timestamp) -> None:
        event_date = str(event_close.date())
        if self.state["last_pnl_reset_date"] == event_date:
            return
        self.state["daily_pnl"] = 0.0
        self.state["daily_trades_by_strategy"] = {}
        self.state["last_pnl_reset_date"] = event_date

    def _candidate_replay_macro_regime(
        self,
        daily_candles: pd.DataFrame | None,
        *,
        event_close: pd.Timestamp,
    ) -> None:
        if not self.regime_guard:
            self._macro_aside = False
            self._macro_detail = {}
            return
        if daily_candles is None:
            self._macro_aside, self._macro_detail = (
                True,
                {
                    "error": "daily regime candles unavailable",
                    "fail_closed": True,
                },
            )
            return
        known = daily_candles[daily_candles["timestamp"] + pd.Timedelta(days=1) <= event_close]
        if known.empty:
            self._macro_aside, self._macro_detail = (
                True,
                {
                    "error": "no daily regime candle was closed at replay event time",
                    "fail_closed": True,
                },
            )
            return
        self._macro_aside, self._macro_detail = compute_macro_step_aside(
            known["close"],
            mayer_top=self.regime_mayer_top,
        )

    def _candidate_replay_entry_allowed(
        self,
        strategy: dict,
        *,
        event_close: pd.Timestamp,
    ) -> bool:
        if not self.allow_entries:
            return False
        if self._has_other_open_position(strategy):
            return False
        pending = self.state[CANDIDATE_REPLAY_PENDING_KEY]
        if any(strategy_id != strategy["id"] for strategy_id in pending):
            return False
        if strategy["id"] in self.state["inactive_strategies"]:
            return False
        if self.state["drawdown_halted"]:
            return False
        if event_close.timestamp() < float(self.state["cooldown_until_ts"]):
            return False
        if self.regime_guard and self._macro_detail.get("fail_closed"):
            return False
        if self.regime_guard and self._macro_aside and strategy["direction"] == "long":
            return False
        if self.state["daily_pnl"] <= self._account_risk()["daily_stop_loss"]:
            return False
        return not self._daily_trade_limit_reached(strategy)

    @staticmethod
    def _candidate_replay_observation_timestamp(value=None) -> pd.Timestamp:
        timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("candidate replay observation_time must be a valid timestamp.")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp

    @staticmethod
    def _candidate_replay_event_is_forward_observed(
        *,
        event_close: pd.Timestamp,
        frame_index: int,
        frame_length: int,
        observation_time: pd.Timestamp,
        max_observation_delay_seconds: int,
    ) -> bool:
        delay_seconds = (observation_time - event_close).total_seconds()
        return frame_index == frame_length - 1 and 0.0 <= delay_seconds <= float(
            max_observation_delay_seconds
        )

    @staticmethod
    def _quarantine_candidate_position(position: dict) -> bool:
        if (
            position.get("candidate_paper_evidence_eligible") is False
            and position.get("candidate_paper_evidence_reason")
            == CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON
        ):
            return False
        position["candidate_paper_evidence_eligible"] = False
        position["candidate_paper_evidence_reason"] = CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON
        return True

    def run_candidate_replay_cycle(
        self,
        *,
        max_unseen_bars: int,
        max_observation_delay_seconds: int = 90,
        observation_time=None,
    ) -> dict:
        """Replay every unseen closed candle for an isolated paper candidate.

        The digest-specific state file is the event log cursor. A genuinely
        fresh latest signal enters only at a public quote observed after the
        signal close. Historical signals retain deterministic next-open replay
        for state recovery, but their positions and trades are permanently
        quarantined from promotion. No broker is accepted on this path, so
        catch-up can never submit a stale live order.
        """

        if self.broker is not None:
            raise RuntimeError("Candidate replay is paper-only and refuses broker injection.")
        if isinstance(max_unseen_bars, bool) or not isinstance(max_unseen_bars, int):
            raise ValueError("max_unseen_bars must be a positive integer.")
        if max_unseen_bars <= 0:
            raise ValueError("max_unseen_bars must be a positive integer.")
        if (
            isinstance(max_observation_delay_seconds, bool)
            or not isinstance(max_observation_delay_seconds, int)
            or max_observation_delay_seconds <= 0
        ):
            raise ValueError("max_observation_delay_seconds must be a positive integer.")
        fixed_observation_time = (
            None
            if observation_time is None
            else self._candidate_replay_observation_timestamp(observation_time)
        )

        self.cycle_errors = []
        self.position_events = []
        self._feature_frame_cache = {}
        self._candidate_replay_max_unseen_bars = max_unseen_bars
        self._candidate_paper_engine_digest = candidate_paper_engine_digest()
        try:
            self._resume_exit_accounting_intent()
            self._assert_no_pending_order()
            self._assert_broker_open_positions_have_metadata()
            if self._normalize_candidate_replay_state():
                self._save_state()

            frames: dict[str, pd.DataFrame] = {}
            events: list[tuple[pd.Timestamp, int, int, pd.Timestamp, int, dict]] = []
            initialized: list[str] = []
            unseen_by_strategy: dict[str, int] = {}
            cursors = self.state[CANDIDATE_REPLAY_CURSOR_KEY]

            for strategy_order, strategy in enumerate(self.strategies):
                timeframe = strategy["base_timeframe"]
                if timeframe not in TIMEFRAME_SECONDS:
                    raise RuntimeError(
                        f"Candidate replay does not recognize timeframe {timeframe!r}."
                    )
                frame, _ = self._build_feature_frame(strategy)
                if frame.empty:
                    raise RuntimeError(
                        f"Candidate replay received no closed {timeframe} bars for "
                        f"{strategy['id']}."
                    )
                frame = frame.copy().reset_index(drop=True)
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                frames[strategy["id"]] = frame

                raw_cursor = cursors.get(strategy["id"])
                if raw_cursor is None:
                    has_prior_activity = bool(
                        self.state["open_positions"].get(strategy["id"])
                        or self.state[CANDIDATE_REPLAY_PENDING_KEY].get(strategy["id"])
                        or self._daily_trade_count(strategy)
                    )
                    if has_prior_activity:
                        raise RuntimeError(
                            f"Cannot safely initialize replay cursor for active legacy candidate "
                            f"state {strategy['id']}; use a fresh digest-isolated paper state."
                        )
                    latest = self._normalized_bar_timestamp(frame.iloc[-1]["timestamp"])
                    cursors[strategy["id"]] = latest
                    self.state["last_entry_decision_bar_by_strategy"][strategy["id"]] = latest
                    initialized.append(strategy["id"])
                    unseen_by_strategy[strategy["id"]] = 0
                    continue

                cursor = pd.Timestamp(raw_cursor)
                if cursor.tzinfo is None:
                    cursor = cursor.tz_localize("UTC")
                else:
                    cursor = cursor.tz_convert("UTC")
                latest = pd.Timestamp(frame.iloc[-1]["timestamp"])
                if cursor > latest:
                    raise RuntimeError(
                        f"Candidate replay cursor for {strategy['id']} is ahead of market data."
                    )
                unseen_indices = [
                    index
                    for index, timestamp in enumerate(frame["timestamp"])
                    if pd.Timestamp(timestamp) > cursor
                ]
                unseen_count = len(unseen_indices)
                unseen_by_strategy[strategy["id"]] = unseen_count
                if unseen_count > max_unseen_bars:
                    raise RuntimeError(
                        f"Candidate replay backlog overflow for {strategy['id']}: "
                        f"{unseen_count} unseen {timeframe} bars exceeds configured maximum "
                        f"{max_unseen_bars}; cursor was not advanced."
                    )
                if not unseen_indices:
                    continue
                seconds = TIMEFRAME_SECONDS[timeframe]
                expected = cursor + pd.Timedelta(seconds=seconds)
                first_unseen = pd.Timestamp(frame.iloc[unseen_indices[0]]["timestamp"])
                if first_unseen != expected:
                    raise RuntimeError(
                        f"Candidate replay candle gap for {strategy['id']}: expected "
                        f"{expected.isoformat()}, received {first_unseen.isoformat()}; cursor "
                        "was not advanced."
                    )
                previous = cursor
                for index in unseen_indices:
                    timestamp = pd.Timestamp(frame.iloc[index]["timestamp"])
                    if timestamp != previous + pd.Timedelta(seconds=seconds):
                        raise RuntimeError(
                            f"Candidate replay candle gap for {strategy['id']} after "
                            f"{previous.isoformat()}; cursor was not advanced."
                        )
                    seconds = TIMEFRAME_SECONDS[timeframe]
                    event_close = timestamp + pd.Timedelta(seconds=seconds)
                    events.append(
                        (
                            event_close,
                            seconds,
                            strategy_order,
                            timestamp,
                            index,
                            strategy,
                        )
                    )
                    previous = timestamp

            if initialized:
                latest_initialized = max(
                    pd.Timestamp(cursors[strategy_id])
                    + pd.Timedelta(
                        seconds=TIMEFRAME_SECONDS[
                            next(
                                strategy["base_timeframe"]
                                for strategy in self.strategies
                                if strategy["id"] == strategy_id
                            )
                        ]
                    )
                    for strategy_id in initialized
                )
                self._candidate_replay_daily_reset(latest_initialized)
                self._save_state()

            cycle_observation_time = (
                fixed_observation_time
                if fixed_observation_time is not None
                else self._candidate_replay_observation_timestamp()
            )

            daily_candles = None
            if self.regime_guard and events:
                try:
                    daily_candles = self.fetch_live_candles(
                        self.data_symbol,
                        self.market,
                        "1d",
                        limit=500,
                    )
                except Exception as exc:
                    LOGGER.error(
                        "Candidate replay macro data failed; entries will fail closed "
                        "while open positions continue replaying: %s",
                        exc,
                    )
                    self.cycle_errors.append(
                        {
                            "stage": "candidate_replay_macro_data",
                            "error": str(exc),
                        }
                    )

            processed = 0
            forward_observed_events = 0
            backfilled_events = 0
            event_order_tail: list[dict[str, Any]] = []
            event_order_total = 0
            for event_close, seconds, _, event_time, frame_index, strategy in sorted(
                events,
                key=lambda event: (event[0], event[1], event[2], event[3]),
            ):
                strategy_id = strategy["id"]
                timeframe = strategy["base_timeframe"]
                frame = frames[strategy_id]
                event_frame = frame.iloc[: frame_index + 1]
                forward_observed = self._candidate_replay_event_is_forward_observed(
                    event_close=event_close,
                    frame_index=frame_index,
                    frame_length=len(frame),
                    observation_time=cycle_observation_time,
                    max_observation_delay_seconds=max_observation_delay_seconds,
                )
                if forward_observed:
                    forward_observed_events += 1
                    observation_class = "fresh_forward_observation"
                else:
                    backfilled_events += 1
                    observation_class = "downtime_backfill"
                event_order_total += 1
                event_order_tail.append(
                    {
                        "strategy_id": strategy_id,
                        "timeframe": timeframe,
                        "bar_open": self._normalized_bar_timestamp(event_time),
                        "information_available_at": self._normalized_bar_timestamp(event_close),
                        "observation_class": observation_class,
                    }
                )
                del event_order_tail[:-64]
                self._candidate_replay_daily_reset(event_close)

                pending = self.state[CANDIDATE_REPLAY_PENDING_KEY].get(strategy_id)
                open_position = self.state["open_positions"].get(strategy_id)
                if pending is not None or open_position is not None:
                    # A bar that starts with exposure (or its durable next-open
                    # intent) is never reconsidered for a same-bar re-entry.
                    self.state["last_entry_decision_bar_by_strategy"][strategy_id] = (
                        self._normalized_bar_timestamp(event_time)
                    )
                    self._save_state()

                    if pending is not None:
                        signal_time = pd.Timestamp(pending["signal_time"])
                        if signal_time.tzinfo is None:
                            signal_time = signal_time.tz_localize("UTC")
                        else:
                            signal_time = signal_time.tz_convert("UTC")
                        expected_entry = signal_time + pd.Timedelta(seconds=seconds)
                        if event_time != expected_entry:
                            raise RuntimeError(
                                f"Pending candidate entry for {strategy_id} expected next-open "
                                f"bar {expected_entry.isoformat()}, received "
                                f"{event_time.isoformat()}."
                            )
                        if open_position is None:
                            if self._has_other_open_position(strategy):
                                raise RuntimeError(
                                    f"Pending candidate entry for {strategy_id} cannot fill "
                                    "while another strategy has product exposure."
                                )
                            matching = frame.index[frame["timestamp"] == signal_time].tolist()
                            if len(matching) != 1:
                                raise RuntimeError(
                                    f"Pending candidate entry signal bar for {strategy_id} is "
                                    "outside the bounded replay frame."
                                )
                            signal_frame = frame.iloc[: matching[0] + 1]
                            entry_open = float(frame.iloc[frame_index][f"tf_{timeframe}_open"])
                            self._enter_position(
                                strategy,
                                signal_frame,
                                entry_open,
                                paper_entry_time=event_time,
                                candidate_paper_evidence={
                                    "eligible": False,
                                    "reason": pending["evidence_reason"],
                                    "fill_source": pending["fill_source"],
                                    "observed_at": pending["signal_observed_at"],
                                },
                            )
                        elif self._normalized_bar_timestamp(
                            open_position.get("signal_time")
                        ) != self._normalized_bar_timestamp(signal_time):
                            raise RuntimeError(
                                f"Pending candidate entry for {strategy_id} does not match "
                                "the durable open position."
                            )
                        self.state[CANDIDATE_REPLAY_PENDING_KEY].pop(strategy_id)
                        self._save_state()
                        open_position = self.state["open_positions"].get(strategy_id)

                    if open_position is not None:
                        if not forward_observed and self._quarantine_candidate_position(
                            open_position
                        ):
                            self._save_state()
                        self._candidate_replay_risk_clock_ts = event_close.timestamp()
                        self._candidate_replay_observation_time = (
                            cycle_observation_time if forward_observed else event_close
                        )
                        try:
                            self._manage_open_position(strategy, event_frame)
                        finally:
                            self._candidate_replay_risk_clock_ts = None
                            self._candidate_replay_observation_time = None
                else:
                    self._candidate_replay_macro_regime(
                        daily_candles,
                        event_close=event_close,
                    )
                    signal_triggered = False
                    decision_already_processed = self._entry_decision_already_processed(
                        strategy,
                        event_time,
                    )
                    if not decision_already_processed and self._candidate_replay_entry_allowed(
                        strategy,
                        event_close=event_close,
                    ):
                        signal_triggered = self._candidate_replay_signal(
                            strategy,
                            event_frame,
                        )
                    self.state["last_entry_decision_bar_by_strategy"][strategy_id] = (
                        self._normalized_bar_timestamp(event_time)
                    )
                    if signal_triggered:
                        if forward_observed:
                            (
                                observation_quote,
                                quote_observed_at,
                            ) = self.fetch_public_observation_quote()
                            quote_observed_at = self._candidate_replay_observation_timestamp(
                                quote_observed_at
                            )
                            quote_delay_seconds = (quote_observed_at - event_close).total_seconds()
                            if not (
                                quote_observed_at >= cycle_observation_time
                                and 0.0
                                <= quote_delay_seconds
                                <= float(max_observation_delay_seconds)
                            ):
                                raise RuntimeError(
                                    f"Candidate public quote for {strategy_id} was observed "
                                    "outside the promotable signal-delay window; cursor was "
                                    "not advanced."
                                )
                            self._enter_position(
                                strategy,
                                event_frame,
                                observation_quote,
                                paper_entry_time=quote_observed_at,
                                candidate_paper_evidence={
                                    "eligible": True,
                                    "reason": CANDIDATE_PAPER_FORWARD_REASON,
                                    "fill_source": CANDIDATE_PAPER_FORWARD_FILL_SOURCE,
                                    "observed_at": quote_observed_at,
                                },
                            )
                        else:
                            self.state[CANDIDATE_REPLAY_PENDING_KEY][strategy_id] = {
                                "signal_time": self._normalized_bar_timestamp(event_time),
                                "signal_observed_at": self._normalized_bar_timestamp(
                                    cycle_observation_time
                                ),
                                "evidence_eligible": False,
                                "evidence_reason": CANDIDATE_PAPER_BACKFILL_ENTRY_REASON,
                                "fill_source": CANDIDATE_PAPER_BACKFILL_FILL_SOURCE,
                            }

                self.state[CANDIDATE_REPLAY_CURSOR_KEY][strategy_id] = (
                    self._normalized_bar_timestamp(event_time)
                )
                self._save_state()
                processed += 1

            return {
                "schema_version": CANDIDATE_REPLAY_SCHEMA_VERSION,
                "execution_schema": CANDIDATE_PAPER_EXECUTION_SCHEMA,
                "execution_engine_digest": self._candidate_paper_engine_digest,
                "max_unseen_bars": max_unseen_bars,
                "max_observation_delay_seconds": max_observation_delay_seconds,
                "observation_time": self._normalized_bar_timestamp(cycle_observation_time),
                "processed_events": processed,
                "forward_observed_events": forward_observed_events,
                "backfilled_events": backfilled_events,
                "event_order_tail": event_order_tail,
                "event_order_truncated": event_order_total > len(event_order_tail),
                "unseen_bars_by_strategy": unseen_by_strategy,
                "initialized_strategies": initialized,
                "cursors": dict(self.state[CANDIDATE_REPLAY_CURSOR_KEY]),
                "pending_entries": sorted(self.state[CANDIDATE_REPLAY_PENDING_KEY]),
            }
        finally:
            self._candidate_replay_max_unseen_bars = None

    def run_cycle(self):
        self.cycle_errors = []
        self.position_events = []
        self._start_decision_trace()
        self._feature_frame_cache = {}
        self._resume_exit_accounting_intent()
        self._recover_live_futures_pending_entry()
        self._resume_live_futures_risk_recovery()
        self._assert_no_pending_order()
        self.process_daily_reset()
        self._assert_broker_open_positions_have_metadata()
        cycle_strategies = list(self.strategies)
        known_ids = {strategy["id"] for strategy in cycle_strategies}
        for strategy_id, position in self.state["open_positions"].items():
            if strategy_id not in known_ids:
                cycle_strategies.append(
                    self._strategy_for_open_position({"id": strategy_id}, position)
                )
        native_stop_exits: set[str] = set()
        # Exchange-side protection is checked before any market-data update or
        # macro research call. A triggered stop is adopted exactly once here.
        for strategy in cycle_strategies:
            open_position = self.state["open_positions"].get(strategy["id"])
            if open_position is None:
                continue
            frozen_strategy = self._strategy_for_open_position(strategy, open_position)
            if self._reconcile_native_protection(frozen_strategy, open_position):
                native_stop_exits.add(strategy["id"])
                self._record_decision(strategy, "native_stop_exit")
                continue
            self._reconcile_broker_position(frozen_strategy, open_position)
        self._evaluate_macro_regime()
        pending_entry_candidates: list[dict[str, Any]] = []
        for strategy in cycle_strategies:
            if strategy["id"] in native_stop_exits:
                continue
            open_position = self.state["open_positions"].get(strategy["id"])
            cycle_strategy = (
                self._strategy_for_open_position(strategy, open_position)
                if open_position is not None
                else strategy
            )
            if open_position is None:
                if not self.allow_entries:
                    LOGGER.info(
                        "New entries are disabled. Skipping flat strategy %s.", strategy["id"]
                    )
                    self._record_decision(strategy, "entry_disabled")
                    continue
                if self._has_other_open_position(strategy):
                    LOGGER.info(
                        "Another strategy already has an open %s position. Skipping new entry for %s.",
                        self.symbol,
                        strategy["id"],
                    )
                    self._record_decision(strategy, "position_capacity_blocked")
                    continue
                if strategy["id"] in self.state["inactive_strategies"]:
                    LOGGER.info(
                        "Strategy %s is deactivated (OOD kill switch). Skipping new entry.",
                        strategy["id"],
                    )
                    self._record_decision(strategy, "strategy_inactive")
                    continue
                self._assert_broker_flat_before_new_entry(strategy)
                if self.state["drawdown_halted"]:
                    LOGGER.critical(
                        "Drawdown circuit breaker is latched. Skipping new entry for %s; "
                        "open-position management remains enabled. Reason: %s",
                        strategy["id"],
                        self.state.get("drawdown_halt_reason"),
                    )
                    self._record_decision(strategy, "drawdown_halted")
                    continue
            try:
                df_features, base_close = self._build_feature_frame(cycle_strategy)
            except Exception as exc:
                LOGGER.error("Failed to build features for %s: %s", strategy["id"], exc)
                self.cycle_errors.append(
                    {
                        "strategy_id": strategy["id"],
                        "stage": "feature_build",
                        "error": str(exc),
                    }
                )
                self._record_decision(strategy, "feature_build_failed", error=str(exc))
                continue

            self.decision_trace["summary"]["data_ready"] += 1
            self._record_market_bar(cycle_strategy, df_features)

            if open_position is not None:
                self._manage_open_position(cycle_strategy, df_features)
                self.decision_trace["summary"]["positions_managed"] += 1
                self._record_decision(
                    strategy,
                    "position_managed",
                    latest_bar=df_features.iloc[-1]["timestamp"],
                )
                continue

            signal_bar_time = df_features.iloc[-1]["timestamp"]
            if self._entry_decision_already_processed(strategy, signal_bar_time):
                LOGGER.info(
                    "Entry decision for %s already processed on closed bar %s.",
                    strategy["id"],
                    self._normalized_bar_timestamp(signal_bar_time),
                )
                self._record_decision(
                    strategy,
                    "bar_already_processed",
                    latest_bar=signal_bar_time,
                )
                continue
            # Persist the decision cursor before any risk gate or exchange
            # side effect.  Restarts cannot evaluate the same closed signal bar
            # twice, even if a flatten or transient failure follows.
            self._mark_entry_decision_processed(strategy, signal_bar_time)

            if time.time() < self.state["cooldown_until_ts"]:
                LOGGER.info("Account in cooldown. Skipping entries.")
                self._record_decision(strategy, "cooldown")
                continue
            if self.regime_guard and self._macro_detail.get("fail_closed"):
                LOGGER.warning(
                    "Macro regime unavailable: skipping new entry for %s (%s).",
                    strategy["id"],
                    self._macro_detail,
                )
                self._record_decision(strategy, "macro_data_unavailable")
                continue
            if self.regime_guard and self._macro_aside and strategy["direction"] == "long":
                LOGGER.warning(
                    "Macro regime risk-off: skipping new LONG entry for %s (%s).",
                    strategy["id"],
                    self._macro_detail,
                )
                self._record_decision(strategy, "macro_regime_blocked")
                continue
            if self.state["daily_pnl"] <= self._account_risk()["daily_stop_loss"]:
                LOGGER.warning(
                    "Daily stop hit (%.4f <= %.4f). Skipping entries.",
                    self.state["daily_pnl"],
                    self._account_risk()["daily_stop_loss"],
                )
                self._record_decision(strategy, "daily_stop")
                continue
            if self._daily_trade_limit_reached(strategy):
                LOGGER.info(
                    "Daily trade limit hit for %s (%s/%s). Skipping entries.",
                    strategy["id"],
                    self._daily_trade_count(strategy),
                    strategy["risk"].get("max_trades_per_day"),
                )
                self._record_decision(strategy, "daily_trade_limit")
                continue

            if strategy.get("entry_type", "conditions") == "hypothesis":
                # Same mask code that scored the hypothesis in research
                # (research_exploration.predicates) — live == validated.
                signal_triggered, signal_detail = self._trace_hypothesis_signal(
                    strategy,
                    df_features,
                )
            elif strategy.get("entry_type") == "frozen_ml":
                signal_triggered, signal_detail = self._trace_frozen_ml_signal(
                    strategy, df_features
                )
            else:
                signal_triggered = True
                signal_detail = {
                    "matched_predicates": 0,
                    "total_predicates": len(strategy["_conditions"]),
                }
                for condition_index, cond in enumerate(strategy["_conditions"]):
                    mask = condition_mask(df_features, cond).fillna(False)
                    if not bool(mask.iloc[-1]):
                        signal_triggered = False
                        signal_detail.update(
                            failed_stage="conditions",
                            failed_predicate=getattr(cond, "description", None)
                            or f"{cond.feature} {cond.kind}",
                            matched_predicates=condition_index,
                        )
                        break
                    signal_detail["matched_predicates"] = condition_index + 1
            if signal_triggered:
                forecast = forecast_from_strategy(
                    strategy,
                    product=self.objective or "unclassified",
                    market=self.market,
                    symbol=self.symbol,
                    signal_detail=signal_detail,
                ).to_dict()
                signal_detail["alpha_forecast"] = forecast
                self.decision_trace["summary"]["signals"] += 1
                tp_pct, sl_pct = self._resolve_tp_sl(strategy, df_features, base_close)
                requested_fraction = (
                    min(
                        strategy["risk"]["risk_per_trade"] / sl_pct,
                        strategy["risk"]["max_position_fraction"],
                        1.0,
                    )
                    if sl_pct > 0
                    else 0.0
                )
                pending_entry_candidates.append(
                    {
                        "strategy": strategy,
                        "df_features": df_features,
                        "base_close": base_close,
                        "signal_bar_time": signal_bar_time,
                        "signal_detail": signal_detail,
                        "forecast": forecast,
                        "requested_fraction": requested_fraction,
                        "take_profit_fraction": tp_pct,
                        "stop_loss_fraction": sl_pct,
                    }
                )
            else:
                self._record_decision(
                    strategy,
                    "signal_not_triggered",
                    latest_bar=signal_bar_time,
                    **signal_detail,
                )
        self._resolve_pending_entry_candidates(pending_entry_candidates)

    def _resolve_pending_entry_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Aggregate same-symbol alpha before selecting one strategy execution envelope."""
        if not candidates:
            return
        aggregation = aggregate_forecasts(
            [AlphaForecast.from_dict(candidate["forecast"]) for candidate in candidates]
        )
        if not aggregation["allowed"]:
            for candidate in candidates:
                self._record_decision(
                    candidate["strategy"],
                    "alpha_ensemble_rejected",
                    latest_bar=candidate["signal_bar_time"],
                    alpha_aggregation=aggregation,
                    **candidate["signal_detail"],
                )
            return
        ensemble = AlphaForecast.from_dict(aggregation["forecast"])
        aligned = [
            candidate
            for candidate in candidates
            if candidate["forecast"]["direction"] == ensemble.direction
        ]
        selected = max(
            aligned,
            key=lambda candidate: AlphaForecast.from_dict(candidate["forecast"]).utility,
        )
        for candidate in candidates:
            if candidate is selected:
                continue
            outcome = (
                "alpha_ensemble_not_selected" if candidate in aligned else "alpha_ensemble_conflict"
            )
            self._record_decision(
                candidate["strategy"],
                outcome,
                latest_bar=candidate["signal_bar_time"],
                alpha_aggregation=aggregation,
                **candidate["signal_detail"],
            )
        allocated_position_fraction = None
        signal_detail = {
            **selected["signal_detail"],
            "alpha_forecast": aggregation["forecast"],
            "alpha_aggregation": aggregation,
        }
        if self.portfolio_gate is not None:
            allocation = self.portfolio_gate(
                {
                    "forecast": aggregation["forecast"],
                    "requested_fraction": selected["requested_fraction"],
                    "take_profit_fraction": selected["take_profit_fraction"],
                    "stop_loss_fraction": selected["stop_loss_fraction"],
                }
            )
            if not isinstance(allocation, dict) or not isinstance(allocation.get("allowed"), bool):
                raise RuntimeError("portfolio gate returned an invalid decision")
            signal_detail["portfolio_allocation"] = allocation
            if not allocation["allowed"]:
                self._record_decision(
                    selected["strategy"],
                    "portfolio_rejected",
                    latest_bar=selected["signal_bar_time"],
                    **signal_detail,
                )
                return
            allocated_position_fraction = float(allocation["allocated_fraction"])
        self._enter_position(
            selected["strategy"],
            selected["df_features"],
            selected["base_close"],
            alpha_forecast=aggregation["forecast"],
            allocated_position_fraction=allocated_position_fraction,
        )
        self.decision_trace["summary"]["entries_opened"] += 1
        self._record_decision(
            selected["strategy"],
            "entry_opened",
            latest_bar=selected["signal_bar_time"],
            **signal_detail,
        )

    def _resolve_tp_sl(
        self, strategy: dict, df_features: pd.DataFrame, base_close: float
    ) -> tuple[float, float]:
        base_tf = strategy["base_timeframe"]
        if not strategy.get("use_atr_tp_sl"):
            return float(strategy["take_profit"]), float(strategy["stop_loss"])
        atr_col = (
            f"tf_{base_tf}_atr"
            if f"tf_{base_tf}_atr" in df_features.columns
            else f"tf_{base_tf}_atr_14"
        )
        latest = df_features.iloc[-1]
        atr_val = float(latest[atr_col]) if atr_col in df_features.columns else (base_close * 0.005)
        tp_pct = (float(strategy["take_profit"]) * atr_val) / base_close
        sl_pct = (float(strategy["stop_loss"]) * atr_val) / base_close
        if sl_pct <= 0:
            sl_pct = 0.01
        return tp_pct, sl_pct

    def _enter_position(
        self,
        strategy: dict,
        df_features: pd.DataFrame,
        base_close: float,
        *,
        paper_entry_time=None,
        candidate_paper_evidence: dict | None = None,
        alpha_forecast: dict | None = None,
        allocated_position_fraction: float | None = None,
    ):
        strategy_snapshot, strategy_fingerprint = self._strategy_snapshot(strategy)
        signal_time = pd.Timestamp(df_features.iloc[-1]["timestamp"])
        if signal_time.tzinfo is None:
            signal_time = signal_time.tz_localize("UTC")
        else:
            signal_time = signal_time.tz_convert("UTC")
        signal_close_time = signal_time + pd.Timedelta(
            seconds=TIMEFRAME_SECONDS.get(strategy["base_timeframe"], 300)
        )
        entry_time = signal_close_time
        if paper_entry_time is not None:
            if self.broker is not None:
                raise RuntimeError("Broker entries cannot override their fill timestamp.")
            entry_time = self._candidate_replay_observation_timestamp(paper_entry_time)
            if entry_time < signal_close_time:
                raise RuntimeError(
                    "Candidate paper entry cannot precede signal information availability."
                )
        signal_time_text = signal_time.isoformat()
        entry_time_text = entry_time.isoformat()
        direction = strategy["direction"]
        tp_pct, sl_pct = self._resolve_tp_sl(strategy, df_features, base_close)
        risk_per_trade = strategy["risk"]["risk_per_trade"]
        max_position_fraction = strategy["risk"]["max_position_fraction"]
        position_size = (
            min(risk_per_trade / sl_pct, max_position_fraction, 1.0) if sl_pct > 0 else 0.0
        )
        if allocated_position_fraction is not None:
            if (
                not math.isfinite(allocated_position_fraction)
                or allocated_position_fraction <= 0
                or allocated_position_fraction - position_size > 1e-12
            ):
                raise RuntimeError("portfolio allocation exceeds the strategy risk envelope")
            position_size = allocated_position_fraction
        entry_price = base_close
        broker_fill = None
        spot_sell_base_before = None
        broker_requested_qty = None
        broker_fill_ratio = None
        broker_entry_balance = None
        spot_sell_quote_before = None
        spot_sell_quote_after = None
        spot_sell_quote_value = None

        if self.broker is not None:
            self._assert_native_protection_available()
            side = OrderSide.BUY if direction == "long" else OrderSide.SELL
            if self._is_spot_broker() and side == OrderSide.SELL:
                spot_sell_base_before = max(float(self.broker.get_position(self.symbol).qty), 0.0)
            if self._requires_native_protective_stop():
                # Capture the flat-account baseline before any exchange side
                # effect.  The corresponding post-exit read lets live risk
                # accounting include funding and broker-booked adjustments
                # that are absent from order-fill payloads.
                broker_entry_balance = self._broker_quote_balance()
            raw_qty = self._broker_order_qty(
                price=base_close,
                position_size=position_size,
                side=side,
                quote_equity=broker_entry_balance,
            )
            qty = self._normalize_broker_order_qty(
                self.symbol,
                raw_qty,
                price=base_close,
            )
            broker_requested_qty = float(qty)
            client_id = self._deterministic_client_order_id(
                strategy_id=strategy["id"],
                stage="entry",
                intent_ref=signal_time_text,
                symbol=self.symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                reduce_only=False,
            )
            entry_order = Order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                type=OrderType.MARKET,
                client_id=client_id,
            )
            if self._requires_observed_spot_quote_proceeds() and side == OrderSide.SELL:
                # The Broker contract returns free quote currency. Capture the
                # baseline as close as possible to submission; the observed
                # post-fill delta is authoritative even when the fill fee was
                # paid in BNB, base, or quote currency.
                spot_sell_quote_before = self._broker_quote_balance(allow_zero=True)
            # Control, approval, and environment evidence can change while the
            # cycle is fetching candles and building features. Re-sample every
            # risk-increasing gate at the last application boundary before the
            # durable order intent and broker submission.
            if self.pre_entry_gate is not None:
                self.pre_entry_gate()
            self._persist_pending_order(
                strategy,
                stage="entry",
                intent_ref=signal_time_text,
                order=entry_order,
            )
            try:
                broker_fill = self.broker.place_order(entry_order)
                self._assert_broker_entry_fill_valid(strategy, entry_order, broker_fill)
                if spot_sell_quote_before is not None:
                    spot_sell_quote_after = self._broker_quote_balance(allow_zero=True)
                    spot_sell_quote_value = self._validated_live_spot_quote_proceeds(
                        strategy["id"],
                        qty=float(broker_fill.qty),
                        price=float(broker_fill.price),
                        balance_before=spot_sell_quote_before,
                        balance_after=spot_sell_quote_after,
                    )
            except Exception as entry_exc:
                if self._requires_native_protective_stop():
                    try:
                        self._recover_live_futures_pending_entry()
                    except Exception as recovery_exc:
                        raise RuntimeError(
                            f"Live futures entry failed for {strategy['id']} and immediate "
                            "pending-entry recovery could not prove broker-flat state. The "
                            "original pending entry remains for operator reconciliation."
                        ) from recovery_exc
                raise entry_exc
            entry_price = float(broker_fill.price)
            broker_fill_ratio = 1.0

        if direction == "long":
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)
        else:
            sl_price = entry_price * (1.0 + sl_pct)
            tp_price = entry_price * (1.0 - tp_pct)

        position = {
            "signal_time": signal_time_text,
            "entry_time": entry_time_text,
            "direction": direction,
            "entry_price": entry_price,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "position_size": position_size,
            "strategy_snapshot": strategy_snapshot,
            "strategy_fingerprint": strategy_fingerprint,
            "approval_strategy_fingerprint": self.approval_fingerprints_by_strategy[strategy["id"]],
            "artifact_digest": self.artifact_content_digest,
        }
        if alpha_forecast is None:
            signal_detail: dict[str, Any] = {}
            hypothesis = strategy.get("_hypothesis")
            if hypothesis is not None and hypothesis.entry_score is not None:
                signal_detail["alpha_score"] = float(
                    entry_score_series(df_features, hypothesis).iloc[-1]
                )
            alpha_forecast = forecast_from_strategy(
                strategy,
                product=self.objective or "unclassified",
                market=self.market,
                symbol=self.symbol,
                signal_detail=signal_detail,
                generated_at=entry_time_text,
            ).to_dict()
        position["alpha_forecast"] = self._decision_trace_value(alpha_forecast)
        if allocated_position_fraction is not None:
            position["portfolio_allocated_fraction"] = allocated_position_fraction
        candidate_engine_digest = getattr(
            self,
            "_candidate_paper_engine_digest",
            None,
        )
        if candidate_engine_digest is not None:
            if not isinstance(candidate_paper_evidence, dict) or set(candidate_paper_evidence) != {
                "eligible",
                "reason",
                "fill_source",
                "observed_at",
            }:
                raise RuntimeError(
                    "Candidate paper entry requires complete forward-evidence metadata."
                )
            evidence_eligible = candidate_paper_evidence["eligible"]
            evidence_reason = candidate_paper_evidence["reason"]
            evidence_fill_source = candidate_paper_evidence["fill_source"]
            if not isinstance(evidence_eligible, bool):
                raise RuntimeError("Candidate paper evidence eligibility must be boolean.")
            if candidate_paper_evidence["observed_at"] is None:
                raise RuntimeError("Candidate paper evidence requires an observation time.")
            observed_at = self._candidate_replay_observation_timestamp(
                candidate_paper_evidence["observed_at"]
            )
            if observed_at < signal_close_time:
                raise RuntimeError(
                    "Candidate paper observation cannot precede signal information availability."
                )
            if evidence_eligible:
                if (
                    evidence_reason != CANDIDATE_PAPER_FORWARD_REASON
                    or evidence_fill_source != CANDIDATE_PAPER_FORWARD_FILL_SOURCE
                    or observed_at != entry_time
                ):
                    raise RuntimeError(
                        "Promotable candidate entry must use its public observation quote "
                        "at the recorded observation time."
                    )
            elif (
                evidence_reason != CANDIDATE_PAPER_BACKFILL_ENTRY_REASON
                or evidence_fill_source != CANDIDATE_PAPER_BACKFILL_FILL_SOURCE
            ):
                raise RuntimeError(
                    "Non-promotable candidate entry must be identified as downtime backfill."
                )
            position.update(
                candidate_paper_execution_schema=CANDIDATE_PAPER_EXECUTION_SCHEMA,
                candidate_paper_engine_digest=str(candidate_engine_digest),
                candidate_paper_evidence_eligible=evidence_eligible,
                candidate_paper_evidence_reason=evidence_reason,
                candidate_paper_entry_fill_source=evidence_fill_source,
                candidate_paper_observed_at=observed_at.isoformat(),
            )
        elif candidate_paper_evidence is not None or paper_entry_time is not None:
            raise RuntimeError("Candidate paper evidence was supplied outside candidate replay.")
        if broker_fill is not None:
            position.update(
                broker_symbol=broker_fill.symbol,
                broker_requested_qty=float(broker_requested_qty),
                broker_fill_ratio=float(broker_fill_ratio),
                broker_qty=float(broker_fill.qty),
                broker_side=broker_fill.side.value,
                broker_entry_fee=float(broker_fill.fee),
                broker_entry_price=float(broker_fill.price),
            )
            broker_account_fingerprint = self._current_broker_account_fingerprint()
            if broker_account_fingerprint is not None:
                position["broker_account_fingerprint"] = broker_account_fingerprint
            if broker_entry_balance is not None:
                position["broker_entry_balance"] = float(broker_entry_balance)
            if self._is_spot_broker() and broker_fill.side == OrderSide.SELL:
                base_after = self._broker_spot_base_balance()
                if self._requires_observed_spot_quote_proceeds():
                    self._validated_live_spot_base_change(
                        strategy["id"],
                        side=OrderSide.SELL,
                        fill_qty=float(broker_fill.qty),
                        balance_before=float(spot_sell_base_before),
                        balance_after=base_after,
                    )
                if spot_sell_quote_value is not None:
                    position.update(
                        broker_entry_base_qty_before=spot_sell_base_before,
                        broker_entry_base_qty_after=base_after,
                        broker_entry_quote_balance_before=float(spot_sell_quote_before),
                        broker_entry_quote_balance_after=float(spot_sell_quote_after),
                        broker_entry_quote_value=float(spot_sell_quote_value),
                        broker_entry_quote_value_source="observed_free_quote_delta",
                        broker_exit_sizing="quote_reinvest",
                    )
                else:
                    position.update(
                        broker_entry_base_qty_before=spot_sell_base_before,
                        broker_entry_base_qty_after=base_after,
                        broker_entry_quote_value=max(
                            float(broker_fill.qty) * float(broker_fill.price)
                            - float(broker_fill.fee),
                            0.0,
                        ),
                        broker_entry_quote_value_source=("fill_notional_less_reported_fee"),
                        broker_exit_sizing="quote_reinvest",
                    )
            if self._requires_native_protective_stop():
                stop_side = self._protective_stop_side(direction)
                stop_client_id = self._deterministic_client_order_id(
                    strategy_id=strategy["id"],
                    stage="stop",
                    intent_ref=str(entry_order.client_id),
                    symbol=broker_fill.symbol,
                    side=stop_side,
                    qty=float(broker_fill.qty),
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                )
                placed_stop = None
                stop_submission_attempted = False
                try:
                    normalized_stop_trigger = self._normalize_broker_order_price(
                        broker_fill.symbol,
                        float(sl_price),
                    )
                    if direction == "long" and normalized_stop_trigger >= entry_price:
                        raise ValueError(
                            "Normalized long stop trigger must remain below the entry price."
                        )
                    if direction == "short" and normalized_stop_trigger <= entry_price:
                        raise ValueError(
                            "Normalized short stop trigger must remain above the entry price."
                        )
                    sl_price = normalized_stop_trigger
                    position["sl_price"] = normalized_stop_trigger
                    stop_submission_attempted = True
                    placed_stop = self.broker.place_protective_stop(
                        symbol=broker_fill.symbol,
                        side=stop_side,
                        qty=float(broker_fill.qty),
                        trigger_price=normalized_stop_trigger,
                        client_id=stop_client_id,
                    )
                    self._assert_protective_order_valid(
                        strategy,
                        placed_stop,
                        symbol=broker_fill.symbol,
                        side=stop_side,
                        qty=float(broker_fill.qty),
                        trigger_price=float(sl_price),
                        client_id=stop_client_id,
                        allowed_statuses={
                            ProtectiveOrderStatus.OPEN,
                            ProtectiveOrderStatus.TRIGGERED,
                        },
                    )
                    verified_stop = self.broker.get_protective_stop(
                        symbol=broker_fill.symbol,
                        order_id=placed_stop.order_id,
                        client_id=stop_client_id,
                    )
                    self._assert_protective_order_valid(
                        strategy,
                        verified_stop,
                        symbol=broker_fill.symbol,
                        side=stop_side,
                        qty=float(broker_fill.qty),
                        trigger_price=float(sl_price),
                        client_id=stop_client_id,
                        order_id=placed_stop.order_id,
                        allowed_statuses={
                            ProtectiveOrderStatus.OPEN,
                            ProtectiveOrderStatus.TRIGGERED,
                        },
                    )
                except Exception as exc:
                    self._recover_failed_native_protection(
                        strategy,
                        position,
                        broker_fill,
                        stop_client_id=stop_client_id,
                        stop_order_id=(placed_stop.order_id if placed_stop is not None else None),
                        stop_submission_attempted=stop_submission_attempted,
                        failure=exc,
                    )
                position.update(
                    broker_stop_order_id=verified_stop.order_id,
                    broker_stop_client_id=verified_stop.client_id,
                    broker_stop_trigger_price=float(verified_stop.trigger_price),
                )
        self.state["open_positions"][strategy["id"]] = position
        self._increment_daily_trade_count(strategy)
        if (
            broker_fill is not None
            and self._requires_native_protective_stop()
            and verified_stop.status == ProtectiveOrderStatus.TRIGGERED
        ):
            self._assert_broker_flat_after_exit(strategy, broker_fill.symbol)
            stop_fill = Fill(
                symbol=verified_stop.symbol,
                side=verified_stop.side,
                qty=float(verified_stop.filled_qty),
                price=float(verified_stop.average_price),
                fee=float(verified_stop.fee),
            )
            self._complete_position_exit(
                strategy,
                position,
                exit_time=pd.Timestamp.now(tz="UTC"),
                exit_price=float(stop_fill.price),
                exit_reason="native_stop",
                broker_exit_fill=stop_fill,
                clear_pending=True,
            )
            return
        if broker_fill is not None:
            self._save_state_clearing_pending_order()
        else:
            self._save_state()
        entry_event_id = _canonical_json_digest(
            {
                "event_type": "opened",
                "strategy_id": strategy["id"],
                "symbol": self.symbol,
                "signal_time": signal_time_text,
                "entry_time": entry_time_text,
                "direction": direction,
            },
            label="Position entry event",
        )
        self.position_events.append(
            {
                "schema": "autopilot.position_event/v1",
                "event_id": entry_event_id,
                "event_type": "opened",
                "strategy_id": strategy["id"],
                "symbol": self.symbol,
                "market": self.market,
                "execution": "broker" if broker_fill is not None else "paper",
                "direction": direction,
                "entry_time": entry_time_text,
                "entry_price": entry_price,
                "position_size": position_size,
                "sl_price": sl_price,
                "tp_price": tp_price,
            }
        )
        LOGGER.critical(
            "%s ORDER OPENED [%s]: %s %s @ %.2f | SL: %.2f | TP: %.2f | Size: %.4f",
            "BROKER" if self.broker is not None else "PAPER",
            strategy["id"],
            direction.upper(),
            self.symbol,
            entry_price,
            sl_price,
            tp_price,
            position_size,
        )

    def _is_spot_broker(self) -> bool:
        return getattr(getattr(self.broker, "config", None), "market_type", None) == "spot"

    def _is_live_broker(self) -> bool:
        return bool(getattr(getattr(self.broker, "config", None), "live", False))

    def _current_broker_account_fingerprint(self) -> str | None:
        if not self._is_live_broker():
            return None
        fingerprint = getattr(self.broker, "account_fingerprint", None)
        if callable(fingerprint):
            fingerprint = fingerprint()
        if not isinstance(fingerprint, str) or not fingerprint.startswith(
            ACCOUNT_FINGERPRINT_PREFIX
        ):
            raise RuntimeError("Live broker is missing a valid non-secret account fingerprint.")
        digest = fingerprint.removeprefix(ACCOUNT_FINGERPRINT_PREFIX)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("Live broker is missing a valid non-secret account fingerprint.")
        return fingerprint

    def _assert_broker_account_fingerprint(
        self,
        expected: object,
        *,
        state_detail: str,
    ) -> None:
        current = self._current_broker_account_fingerprint()
        if current is None:
            if expected is not None:
                raise RuntimeError(
                    f"Paper broker state unexpectedly contains a live account fingerprint. "
                    f"{state_detail}"
                )
            return
        if expected != current:
            raise RuntimeError(
                "Live broker account fingerprint does not match durable order/position state; "
                f"refusing reconciliation or order submission. {state_detail}"
            )

    def _requires_observed_spot_quote_proceeds(self) -> bool:
        config = getattr(self.broker, "config", None)
        return bool(
            self.broker is not None
            and getattr(config, "live", False)
            and getattr(config, "market_type", None) == "spot"
        )

    def _requires_native_protective_stop(self) -> bool:
        config = getattr(self.broker, "config", None)
        return bool(
            self.broker is not None
            and getattr(config, "live", False)
            and getattr(config, "market_type", None) == "futures"
        )

    def _assert_native_protection_available(self) -> None:
        if not self._requires_native_protective_stop():
            return
        capability = getattr(self.broker, "supports_native_protective_stops", None)
        if not callable(capability) or not capability():
            raise RuntimeError(
                "Refusing live futures entry: broker does not provide verified exchange-native "
                "reduce-only stop protection."
            )

    @staticmethod
    def _protective_stop_side(direction: str) -> OrderSide:
        return OrderSide.SELL if direction == "long" else OrderSide.BUY

    def _assert_protective_order_valid(
        self,
        strategy: dict,
        protective: ProtectiveOrder,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        trigger_price: float,
        client_id: str,
        order_id: str | None = None,
        allowed_statuses: set[ProtectiveOrderStatus],
    ) -> None:
        detail = "Local position left pending for operator reconciliation."
        if not isinstance(protective, ProtectiveOrder):
            raise RuntimeError(
                f"Protective stop response for {strategy['id']} is not a validated ProtectiveOrder. {detail}"
            )
        if not protective.order_id or (order_id is not None and protective.order_id != order_id):
            raise RuntimeError(f"Protective stop order id mismatch for {strategy['id']}. {detail}")
        if protective.client_id != client_id:
            raise RuntimeError(f"Protective stop client id mismatch for {strategy['id']}. {detail}")
        if not _symbols_match(protective.symbol, symbol):
            raise RuntimeError(f"Protective stop symbol mismatch for {strategy['id']}. {detail}")
        if protective.side != side:
            raise RuntimeError(f"Protective stop side mismatch for {strategy['id']}. {detail}")
        if not isinstance(protective.status, ProtectiveOrderStatus):
            raise RuntimeError(f"Protective stop status is invalid for {strategy['id']}. {detail}")
        qty_tolerance = self._fill_quantity_tolerance(qty)
        if (
            not math.isfinite(float(protective.qty))
            or abs(float(protective.qty) - qty) > qty_tolerance
        ):
            raise RuntimeError(f"Protective stop quantity mismatch for {strategy['id']}. {detail}")
        trigger_tolerance = max(abs(trigger_price) * 1e-9, 1e-12)
        if (
            not math.isfinite(float(protective.trigger_price))
            or abs(float(protective.trigger_price) - trigger_price) > trigger_tolerance
        ):
            raise RuntimeError(f"Protective stop trigger mismatch for {strategy['id']}. {detail}")
        if protective.status not in allowed_statuses:
            raise RuntimeError(
                f"Protective stop for {strategy['id']} has unsafe status "
                f"{protective.status.value!r}. {detail}"
            )
        filled_qty = float(protective.filled_qty)
        fee = float(protective.fee)
        if not math.isfinite(filled_qty) or filled_qty < 0:
            raise RuntimeError(
                f"Protective stop fill quantity is invalid for {strategy['id']}. {detail}"
            )
        if not math.isfinite(fee) or fee < 0:
            raise RuntimeError(f"Protective stop fee is invalid for {strategy['id']}. {detail}")
        if protective.status == ProtectiveOrderStatus.OPEN and filled_qty != 0:
            raise RuntimeError(
                f"Open protective stop reports a fill for {strategy['id']}. {detail}"
            )
        if protective.status != ProtectiveOrderStatus.TRIGGERED and filled_qty != 0:
            raise RuntimeError(
                f"Terminal untriggered protective stop reports a fill for {strategy['id']}. {detail}"
            )
        if protective.status == ProtectiveOrderStatus.TRIGGERED:
            if abs(filled_qty - qty) > qty_tolerance:
                raise RuntimeError(
                    f"Triggered protective stop is not fully filled for {strategy['id']}. {detail}"
                )
            average_price = protective.average_price
            if (
                average_price is None
                or not math.isfinite(float(average_price))
                or float(average_price) <= 0
            ):
                raise RuntimeError(
                    f"Triggered protective stop has no valid average fill price for {strategy['id']}. {detail}"
                )

    def _assert_broker_flat_after_exit(self, strategy: dict, symbol: str) -> None:
        actual = self.broker.get_position(symbol)
        if not actual.is_flat:
            raise RuntimeError(
                f"Broker exit for {strategy['id']} did not flatten {symbol}; broker reports "
                f"qty {actual.qty:g}. Local position left open for reconciliation."
            )

    def _cancel_native_protection(self, strategy: dict, open_position: dict) -> ProtectiveOrder:
        symbol = str(open_position["broker_symbol"])
        client_id = str(open_position["broker_stop_client_id"])
        order_id = str(open_position["broker_stop_order_id"])
        canceled = self.broker.cancel_protective_stop(
            symbol=symbol,
            order_id=order_id,
            client_id=client_id,
        )
        self._assert_protective_order_valid(
            strategy,
            canceled,
            symbol=symbol,
            side=self._protective_stop_side(open_position["direction"]),
            qty=float(open_position["broker_qty"]),
            trigger_price=float(open_position["broker_stop_trigger_price"]),
            client_id=client_id,
            order_id=order_id,
            allowed_statuses={
                ProtectiveOrderStatus.CANCELED,
                ProtectiveOrderStatus.EXPIRED,
                ProtectiveOrderStatus.REJECTED,
            },
        )
        return canceled

    def _recover_failed_native_protection(
        self,
        strategy: dict,
        position: dict,
        entry_fill: Fill,
        *,
        stop_client_id: str,
        stop_order_id: str | None,
        stop_submission_attempted: bool,
        failure: Exception,
    ) -> None:
        """Flatten an entry whose exchange-native stop cannot be proven open.

        The original entry write-ahead intent remains durable throughout this
        recovery.  It is cleared only after both the broker position is flat
        and any possibly-created conditional order is proven terminal.
        """

        side = self._protective_stop_side(position["direction"])
        stop_qty = float(entry_fill.qty)
        terminal_stop = None
        try:
            qty = self._normalize_broker_order_qty(
                entry_fill.symbol,
                stop_qty,
                price=float(entry_fill.price),
                reduce_only=True,
            )
            recovery_client_id = self._deterministic_client_order_id(
                strategy_id=strategy["id"],
                stage="recovery",
                intent_ref=stop_client_id,
                symbol=entry_fill.symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                reduce_only=True,
            )
            recovery_order = Order(
                symbol=entry_fill.symbol,
                side=side,
                qty=qty,
                type=OrderType.MARKET,
                reduce_only=True,
                client_id=recovery_client_id,
            )
            recovery_fill = self.broker.place_order(recovery_order)
            self._assert_broker_exit_fill_valid(strategy, recovery_order, qty, recovery_fill)
            self._assert_broker_flat_after_exit(strategy, entry_fill.symbol)
            if stop_submission_attempted:
                terminal_stop = self.broker.cancel_protective_stop(
                    symbol=entry_fill.symbol,
                    order_id=stop_order_id,
                    client_id=stop_client_id,
                )
                self._assert_protective_order_valid(
                    strategy,
                    terminal_stop,
                    symbol=entry_fill.symbol,
                    side=side,
                    qty=stop_qty,
                    trigger_price=float(position["sl_price"]),
                    client_id=stop_client_id,
                    order_id=stop_order_id,
                    allowed_statuses={
                        ProtectiveOrderStatus.CANCELED,
                        ProtectiveOrderStatus.EXPIRED,
                        ProtectiveOrderStatus.REJECTED,
                    },
                )
        except Exception as recovery_exc:
            raise RuntimeError(
                f"Native protective stop failed for {strategy['id']} and automatic recovery "
                "could not prove both a flat broker position and a terminal stop order; "
                "the pending entry intent remains for operator reconciliation."
            ) from recovery_exc

        if terminal_stop is not None:
            position.update(
                broker_stop_order_id=terminal_stop.order_id,
                broker_stop_client_id=terminal_stop.client_id,
                broker_stop_trigger_price=float(terminal_stop.trigger_price),
            )
        self.state["open_positions"][strategy["id"]] = position
        self._increment_daily_trade_count(strategy)
        self._complete_position_exit(
            strategy,
            position,
            exit_time=pd.Timestamp.now(tz="UTC"),
            exit_price=float(recovery_fill.price),
            exit_reason="protection_failed_flatten",
            broker_exit_fill=recovery_fill,
            clear_pending=True,
        )
        raise RuntimeError(
            f"Native protective stop failed for {strategy['id']}; the new position was immediately "
            "flattened and the failed trade was recorded."
        ) from failure

    @staticmethod
    def _validated_live_spot_quote_proceeds(
        strategy_id: str,
        *,
        qty: float,
        price: float,
        balance_before: float,
        balance_after: float,
    ) -> float:
        gross_proceeds = float(qty) * float(price)
        observed_proceeds = float(balance_after) - float(balance_before)
        if not math.isfinite(gross_proceeds) or gross_proceeds <= 0:
            raise RuntimeError(
                f"Live spot quote reconciliation has invalid fill notional for {strategy_id}."
            )
        tolerance = max(gross_proceeds * 1e-6, 1e-8)
        minimum_proceeds = gross_proceeds * (1.0 - LIVE_SPOT_QUOTE_PROCEEDS_MAX_SHORTFALL_FRACTION)
        maximum_proceeds = gross_proceeds + tolerance
        if (
            not math.isfinite(observed_proceeds)
            or observed_proceeds <= 0
            or observed_proceeds < minimum_proceeds - tolerance
            or observed_proceeds > maximum_proceeds
        ):
            raise RuntimeError(
                "Live spot quote reconciliation failed for "
                f"{strategy_id}: observed free-quote delta {observed_proceeds:g} is outside "
                f"the bounded fill-proceeds range [{minimum_proceeds:g}, "
                f"{maximum_proceeds:g}]. The pending entry intent remains for operator "
                "reconciliation."
            )
        return observed_proceeds

    @staticmethod
    def _validated_live_spot_base_change(
        strategy_id: str,
        *,
        side: OrderSide,
        fill_qty: float,
        balance_before: float,
        balance_after: float,
    ) -> float:
        """Prove a spot fill through the actual free-BTC balance movement."""

        expected = float(fill_qty)
        before = float(balance_before)
        after = float(balance_after)
        if not all(math.isfinite(value) for value in (expected, before, after)):
            raise RuntimeError(
                f"Live spot base-balance reconciliation is non-finite for {strategy_id}."
            )
        if expected <= 0 or before < 0 or after < 0:
            raise RuntimeError(
                f"Live spot base-balance reconciliation is invalid for {strategy_id}."
            )
        observed = before - after if side == OrderSide.SELL else after - before
        tolerance = max(expected * 1e-6, 1e-12)
        minimum = expected * (1.0 - LIVE_SPOT_BASE_BALANCE_MAX_FEE_FRACTION)
        maximum = expected * (1.0 + LIVE_SPOT_BASE_BALANCE_MAX_FEE_FRACTION)
        if observed < minimum - tolerance or observed > maximum + tolerance:
            raise RuntimeError(
                "Live spot base-balance reconciliation failed for "
                f"{strategy_id}: observed {side.value} delta {observed:g} is outside "
                f"[{minimum:g}, {maximum:g}]. The durable order intent remains for "
                "operator reconciliation."
            )
        return observed

    def _broker_spot_base_balance(self) -> float:
        if self.broker is None or not self._is_spot_broker():
            raise RuntimeError("spot base balance requested without a spot broker")
        balance = float(self.broker.get_position(self.symbol).qty)
        if not math.isfinite(balance) or balance < 0:
            raise ValueError(
                f"Broker spot base balance must be finite and non-negative, got {balance:g}."
            )
        return balance

    def _broker_quote_balance(self, *, allow_zero: bool = False) -> float:
        if self.broker is None:
            raise RuntimeError("broker balance requested without a broker")
        quote_equity = float(self.broker.get_balance())
        invalid = quote_equity < 0 if allow_zero else quote_equity <= 0
        if not math.isfinite(quote_equity) or invalid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"Broker quote balance must be finite and {qualifier}, got {quote_equity:g}."
            )
        return quote_equity

    def _broker_order_qty(
        self,
        price: float,
        position_size: float,
        side: OrderSide,
        *,
        quote_equity: float | None = None,
    ) -> float:
        if self.broker is None:
            raise RuntimeError("broker order sizing requested without a broker")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Cannot size a broker order with a non-positive price.")
        if not math.isfinite(position_size) or position_size <= 0:
            raise ValueError(f"Cannot size a broker order with position_size={position_size:g}.")
        if self._is_spot_broker() and side == OrderSide.SELL:
            base_qty = float(self.broker.get_position(self.symbol).qty)
            if not math.isfinite(base_qty):
                raise ValueError(f"Spot base quantity is not finite: {base_qty:g}.")
            base_qty = max(base_qty, 0.0)
            qty = base_qty * position_size
            if not math.isfinite(qty) or qty <= 0:
                raise ValueError(
                    f"Spot sell quantity is non-positive (base_qty={base_qty}, "
                    f"position_size={position_size})."
                )
            return qty
        if quote_equity is None:
            quote_equity = self._broker_quote_balance()
        elif not math.isfinite(float(quote_equity)) or float(quote_equity) <= 0:
            raise ValueError(
                f"Broker quote balance must be finite and positive, got {float(quote_equity):g}."
            )
        quote_equity = float(quote_equity)
        qty = (quote_equity * position_size) / price
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError(
                f"Broker order quantity is non-positive (balance={quote_equity}, "
                f"position_size={position_size}, price={price})."
            )
        return qty

    def _normalize_broker_order_qty(
        self,
        symbol: str,
        qty: float,
        *,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> float:
        raw_qty = float(qty)
        if not math.isfinite(raw_qty) or raw_qty <= 0:
            raise ValueError(f"Broker order quantity must be finite and positive, got {raw_qty:g}.")
        hook = getattr(self.broker, "normalize_order_qty", None)
        normalized_raw = (
            hook(symbol, raw_qty, price=price, reduce_only=reduce_only)
            if callable(hook)
            else raw_qty
        )
        try:
            normalized = float(normalized_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Broker normalized order quantity must be numeric, got {normalized_raw!r}."
            ) from exc
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                f"Broker normalized order quantity must be finite and positive, got {normalized:g}."
            )
        tolerance = max(abs(raw_qty) * 1e-12, 1e-12)
        if normalized - raw_qty > tolerance:
            raise ValueError(
                f"Broker quantity normalization increased intended exposure from {raw_qty:g} "
                f"to {normalized:g}. Refusing."
            )
        return normalized

    def _normalize_broker_order_price(self, symbol: str, price: float) -> float:
        raw_price = float(price)
        if not math.isfinite(raw_price) or raw_price <= 0:
            raise ValueError(f"Broker order price must be finite and positive, got {raw_price:g}.")
        hook = getattr(self.broker, "normalize_order_price", None)
        normalized_raw = hook(symbol, raw_price) if callable(hook) else raw_price
        try:
            normalized = float(normalized_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Broker normalized order price must be numeric, got {normalized_raw!r}."
            ) from exc
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                f"Broker normalized order price must be finite and positive, got {normalized:g}."
            )
        return normalized

    def _broker_state_qty(self, strategy: dict, open_position: dict) -> float:
        self._assert_broker_metadata_complete(
            strategy["id"],
            open_position,
            state_detail="Local position left open for reconciliation.",
        )
        try:
            qty = float(open_position["broker_qty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Broker state invalid for {strategy['id']}: broker_qty is missing or non-numeric. "
                "Local position left open for reconciliation."
            ) from exc
        if not math.isfinite(qty) or qty <= 0:
            raise RuntimeError(
                f"Broker state invalid for {strategy['id']}: broker_qty {qty:g} must be positive. "
                "Local position left open for reconciliation."
            )
        return qty

    def _assert_broker_metadata_complete(
        self,
        strategy_id: str,
        position: dict,
        *,
        state_detail: str,
        require_present: bool = False,
    ) -> None:
        if not any(str(key).startswith("broker_") for key in position):
            if require_present:
                raise RuntimeError(
                    f"Broker state invalid for {strategy_id}: broker metadata is required. {state_detail}"
                )
            return
        missing = [key for key in BROKER_POSITION_REQUIRED_KEYS if key not in position]
        if self._is_live_broker():
            if "broker_account_fingerprint" not in position:
                missing.append("broker_account_fingerprint")
            else:
                self._assert_broker_account_fingerprint(
                    position.get("broker_account_fingerprint"),
                    state_detail=f"open position {strategy_id} in {self.state_file}",
                )
        if self._requires_native_protective_stop():
            missing.extend(
                key for key in BROKER_PROTECTIVE_STOP_REQUIRED_KEYS if key not in position
            )
            missing.extend(
                key for key in BROKER_LIVE_FUTURES_ACCOUNTING_REQUIRED_KEYS if key not in position
            )
        if self._requires_observed_spot_quote_proceeds() and position.get("direction") == "short":
            missing.extend(
                key for key in BROKER_LIVE_SPOT_ACCOUNTING_REQUIRED_KEYS if key not in position
            )
            source = position.get("broker_entry_quote_value_source")
            if source is not None and source != "observed_free_quote_delta":
                raise RuntimeError(
                    f"Broker state invalid for {strategy_id}: live spot quote proceeds "
                    "must come from an observed free-quote balance delta. "
                    f"{state_detail}"
                )
            exit_sizing = position.get("broker_exit_sizing")
            if exit_sizing is not None and exit_sizing != "quote_reinvest":
                raise RuntimeError(
                    f"Broker state invalid for {strategy_id}: live spot exit sizing must be "
                    f"quote_reinvest. {state_detail}"
                )
        if missing:
            raise RuntimeError(
                f"Broker state invalid for {strategy_id}: broker metadata missing required key(s): "
                f"{', '.join(missing)}. {state_detail}"
            )

    def _assert_broker_open_positions_have_metadata(self) -> None:
        if self.broker is None:
            return
        positions = self.state.get("open_positions", {})
        if not isinstance(positions, dict):
            raise RuntimeError(f"State open_positions must be an object: {self.state_file}")
        for strategy_id, position in positions.items():
            if not isinstance(position, dict):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}] must be an object: {self.state_file}"
                )
            self._assert_broker_metadata_complete(
                str(strategy_id),
                position,
                state_detail="Local position left open for reconciliation.",
                require_present=True,
            )

    def _spot_step_aside_quote_value(self, strategy: dict, open_position: dict) -> float:
        if open_position.get("broker_exit_sizing") != "quote_reinvest":
            raise RuntimeError(
                f"Broker state invalid for {strategy['id']}: spot step-aside state must use "
                "quote_reinvest exit sizing. Local position left open for reconciliation."
            )
        try:
            quote_value = float(open_position["broker_entry_quote_value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Broker state invalid for {strategy['id']}: spot step-aside state is missing "
                "broker_entry_quote_value. Local position left open for reconciliation."
            ) from exc
        if not math.isfinite(quote_value) or quote_value <= 0:
            raise RuntimeError(
                f"Broker state invalid for {strategy['id']}: spot step-aside broker_entry_quote_value "
                f"{quote_value:g} must be positive. Local position left open for reconciliation."
            )
        return quote_value

    def _broker_exit_order_qty(
        self,
        strategy: dict,
        open_position: dict,
        side: OrderSide,
        fallback_price: float,
    ) -> float:
        if self.broker is None:
            raise RuntimeError("broker exit sizing requested without a broker")
        if (
            self._is_spot_broker()
            and open_position.get("direction") == "short"
            and side == OrderSide.BUY
        ):
            quote_value = self._spot_step_aside_quote_value(strategy, open_position)
            price = float(
                self.broker.get_price(open_position.get("broker_symbol", self.symbol))
                or fallback_price
            )
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Cannot size a spot buyback with a non-positive price.")
            qty = quote_value / price
            if not math.isfinite(qty) or qty <= 0:
                raise ValueError(
                    f"Spot buyback quantity is non-positive (quote_value={quote_value}, price={price})."
                )
            return qty
        return self._broker_state_qty(strategy, open_position)

    @staticmethod
    def _side_value(side: OrderSide | str) -> str:
        return side.value if isinstance(side, OrderSide) else str(side)

    def _assert_broker_fill_matches_order(
        self,
        strategy: dict,
        *,
        stage: str,
        order: Order,
        fill: Fill,
        state_detail: str,
    ) -> None:
        if fill.symbol != order.symbol:
            raise RuntimeError(
                f"Broker {stage} fill mismatch for {strategy['id']}: expected symbol "
                f"{order.symbol}, got {fill.symbol}. {state_detail}"
            )
        expected_side = self._side_value(order.side)
        actual_side = self._side_value(fill.side)
        if actual_side != expected_side:
            raise RuntimeError(
                f"Broker {stage} fill mismatch for {strategy['id']}: expected side "
                f"{expected_side}, got {actual_side}. {state_detail}"
            )

    @staticmethod
    def _fill_quantity_tolerance(requested_qty: float) -> float:
        return max(float(requested_qty) * 1e-6, 1e-9)

    def _assert_broker_fill_not_over_requested(
        self,
        strategy: dict,
        *,
        stage: str,
        requested_qty: float,
        fill_qty: float,
        state_detail: str,
    ) -> None:
        tolerance = self._fill_quantity_tolerance(requested_qty)
        if fill_qty - float(requested_qty) > tolerance:
            raise RuntimeError(
                f"Broker {stage} overfill for {strategy['id']}: requested {requested_qty:g}, "
                f"filled {fill_qty:g}. {state_detail}"
            )

    def _assert_broker_entry_fill_valid(self, strategy: dict, order: Order, fill: Fill) -> None:
        self._assert_broker_fill_matches_order(
            strategy,
            stage="entry",
            order=order,
            fill=fill,
            state_detail="Local position not opened.",
        )
        fill_qty = float(fill.qty)
        fill_price = float(fill.price)
        fill_fee = float(fill.fee)
        if not math.isfinite(fill_qty) or fill_qty <= 0:
            raise RuntimeError(
                f"Broker entry invalid fill for {strategy['id']}: filled quantity "
                f"{fill_qty:g}. Local position not opened."
            )
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise RuntimeError(
                f"Broker entry invalid fill for {strategy['id']}: fill price "
                f"{fill_price:g}. Local position not opened."
            )
        if not math.isfinite(fill_fee) or fill_fee < 0:
            raise RuntimeError(
                f"Broker entry invalid fill for {strategy['id']}: fill fee "
                f"{fill_fee:g}. Local position not opened."
            )
        self._assert_broker_fill_not_over_requested(
            strategy,
            stage="entry",
            requested_qty=float(order.qty),
            fill_qty=fill_qty,
            state_detail="Local position not opened.",
        )
        tolerance = self._fill_quantity_tolerance(float(order.qty))
        if fill_qty + tolerance < float(order.qty):
            raise RuntimeError(
                f"Broker entry partial fill for {strategy['id']}: requested {float(order.qty):g}, "
                f"filled {fill_qty:g}. Local position not opened."
            )

    def _assert_broker_exit_fill_valid(
        self, strategy: dict, order: Order, requested_qty: float, fill: Fill
    ) -> None:
        self._assert_broker_fill_matches_order(
            strategy,
            stage="exit",
            order=order,
            fill=fill,
            state_detail="Local position left open for reconciliation.",
        )
        fill_qty = float(fill.qty)
        fill_price = float(fill.price)
        fill_fee = float(fill.fee)
        if not math.isfinite(fill_qty) or fill_qty <= 0:
            raise RuntimeError(
                f"Broker exit invalid fill for {strategy['id']}: filled quantity "
                f"{fill_qty:g}. Local position left open for reconciliation."
            )
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise RuntimeError(
                f"Broker exit invalid fill for {strategy['id']}: fill price "
                f"{fill_price:g}. Local position left open for reconciliation."
            )
        if not math.isfinite(fill_fee) or fill_fee < 0:
            raise RuntimeError(
                f"Broker exit invalid fill for {strategy['id']}: fill fee "
                f"{fill_fee:g}. Local position left open for reconciliation."
            )
        self._assert_broker_fill_not_over_requested(
            strategy,
            stage="exit",
            requested_qty=float(requested_qty),
            fill_qty=fill_qty,
            state_detail="Local position left open for reconciliation.",
        )
        tolerance = self._fill_quantity_tolerance(requested_qty)
        if fill_qty + tolerance < float(requested_qty):
            raise RuntimeError(
                f"Broker exit partial fill for {strategy['id']}: requested {requested_qty:g}, "
                f"filled {fill_qty:g}. Local position left open for reconciliation."
            )

    def _reconcile_broker_position(self, strategy: dict, open_position: dict):
        if self.broker is None or "broker_qty" not in open_position:
            return
        expected_qty = self._broker_state_qty(strategy, open_position)
        actual = self.broker.get_position(open_position.get("broker_symbol", self.symbol))
        direction = open_position["direction"]
        tolerance = self._fill_quantity_tolerance(expected_qty)
        if self._is_spot_broker() and direction == "short":
            expected_base_after = open_position.get("broker_entry_base_qty_after")
            if expected_base_after is None:
                raise RuntimeError(
                    f"Broker position mismatch for {strategy['id']}: spot step-aside state has no "
                    "broker_entry_base_qty_after."
                )
            self._spot_step_aside_quote_value(strategy, open_position)
            expected_base_after = float(expected_base_after)
            if actual.qty + tolerance < expected_base_after:
                raise RuntimeError(
                    f"Broker position mismatch for {strategy['id']}: expected spot BTC at least "
                    f"{expected_base_after}, got {actual.qty}."
                )
            return
        expected_signed_qty = expected_qty if direction == "long" else -expected_qty
        if abs(float(actual.qty) - expected_signed_qty) > tolerance:
            if self._requires_native_protective_stop() and not actual.is_flat:
                self._recover_live_futures_exposure_incident(
                    strategy,
                    open_position,
                    actual,
                    cause="broker_position_quantity_mismatch",
                    cancel_native_stop=True,
                )
            raise RuntimeError(
                f"Broker position mismatch for {strategy['id']}: expected signed quantity "
                f"{expected_signed_qty:g}, got {float(actual.qty):g}."
            )

    def _reconcile_native_protection(
        self,
        strategy: dict,
        open_position: dict,
        *,
        clear_pending: bool = False,
    ) -> bool:
        """Verify protection, adopting a fully-triggered native stop as exit."""

        if not self._requires_native_protective_stop() or "broker_qty" not in open_position:
            return False
        symbol = str(open_position["broker_symbol"])
        order_id = str(open_position["broker_stop_order_id"])
        client_id = str(open_position["broker_stop_client_id"])
        qty = float(open_position["broker_qty"])
        trigger_price = float(open_position["broker_stop_trigger_price"])
        try:
            protective = self.broker.get_protective_stop(
                symbol=symbol,
                order_id=order_id,
                client_id=client_id,
            )
            self._assert_protective_order_valid(
                strategy,
                protective,
                symbol=symbol,
                side=self._protective_stop_side(open_position["direction"]),
                qty=qty,
                trigger_price=trigger_price,
                client_id=client_id,
                order_id=order_id,
                allowed_statuses=set(ProtectiveOrderStatus),
            )
        except Exception as protection_exc:
            cause = "native_stop_lookup_or_validation_error"
            self._record_risk_recovery_incident(
                strategy_id=strategy["id"],
                cause=cause,
                status="native_stop_lookup_or_validation_failed",
                error=f"{type(protection_exc).__name__}: {protection_exc}",
            )
            try:
                actual = self.broker.get_position(symbol)
            except Exception as position_exc:
                self._record_risk_recovery_incident(
                    strategy_id=strategy["id"],
                    cause=cause,
                    status="native_stop_error_position_unverified",
                    error=(
                        f"{type(protection_exc).__name__}: {protection_exc}; position "
                        f"readback failed: {type(position_exc).__name__}: {position_exc}"
                    ),
                )
                raise RuntimeError(
                    f"Native protective stop for {strategy['id']} could not be validated and "
                    "the live position could not be read; operator reconciliation is required."
                ) from protection_exc
            if not actual.is_flat:
                self._recover_live_futures_exposure_incident(
                    strategy,
                    open_position,
                    actual,
                    cause=cause,
                    cancel_native_stop=True,
                    finalize_accounting=False,
                )
            self._record_risk_recovery_incident(
                strategy_id=strategy["id"],
                cause=cause,
                status="broker_flat_accounting_unresolved_after_native_stop_error",
                error=f"{type(protection_exc).__name__}: {protection_exc}",
            )
            raise RuntimeError(
                f"Native protective stop for {strategy['id']} could not be validated; the "
                "broker is flat but exact stop-fill accounting remains unresolved."
            ) from protection_exc
        if protective.status == ProtectiveOrderStatus.OPEN:
            return False
        if protective.status != ProtectiveOrderStatus.TRIGGERED:
            actual = self.broker.get_position(symbol)
            if not actual.is_flat:
                self._recover_live_futures_exposure_incident(
                    strategy,
                    open_position,
                    actual,
                    cause=f"native_stop_terminal_{protective.status.value}",
                    cancel_native_stop=False,
                )
            raise RuntimeError(
                f"Native protective stop for {strategy['id']} is {protective.status.value}; "
                "the live futures position may be unprotected. Refusing further automation."
            )

        self._assert_broker_flat_after_exit(strategy, symbol)
        fill = Fill(
            symbol=symbol,
            side=protective.side,
            qty=float(protective.filled_qty),
            price=float(protective.average_price),
            fee=float(protective.fee),
        )
        self._complete_position_exit(
            strategy,
            open_position,
            exit_time=pd.Timestamp.now(tz="UTC"),
            exit_price=float(fill.price),
            exit_reason="native_stop",
            broker_exit_fill=fill,
            clear_pending=clear_pending,
        )
        return True

    def _assert_broker_flat_before_new_entry(self, strategy: dict) -> None:
        if self.broker is None or self._is_spot_broker():
            return
        actual = self.broker.get_position(self.symbol)
        if not actual.is_flat:
            raise RuntimeError(
                f"Unexpected broker position for {strategy['id']}: local state is flat for "
                f"{self.symbol}, but broker reports qty {actual.qty:g}. Refusing new entry."
            )

    def _bars_held(self, strategy: dict, open_position: dict, latest_time: pd.Timestamp) -> int:
        """Holding duration in closed base-TF bars, derived from timestamps so
        it stays correct even if scheduler cadence differs from the bar size."""
        tf_seconds = TIMEFRAME_SECONDS.get(strategy["base_timeframe"], 300)
        entry_time = pd.Timestamp(open_position["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        latest_open = pd.Timestamp(latest_time)
        if latest_open.tzinfo is None:
            latest_open = latest_open.tz_localize("UTC")
        latest_close = latest_open + pd.Timedelta(seconds=tf_seconds)
        elapsed = (latest_close - entry_time).total_seconds()
        return max(0, int(elapsed // tf_seconds))

    def _manage_open_position(self, strategy: dict, df_features: pd.DataFrame):
        open_position = self.state["open_positions"][strategy["id"]]
        latest_bar = df_features.iloc[-1]
        latest_time = latest_bar["timestamp"]
        base_tf = strategy["base_timeframe"]
        tf_seconds = TIMEFRAME_SECONDS.get(base_tf, 300)
        latest_open = pd.Timestamp(latest_time)
        if latest_open.tzinfo is None:
            latest_open = latest_open.tz_localize("UTC")
        latest_close = latest_open + pd.Timedelta(seconds=tf_seconds)
        entry_effective = pd.Timestamp(open_position["entry_time"])
        if entry_effective.tzinfo is None:
            entry_effective = entry_effective.tz_localize("UTC")
        if latest_close <= entry_effective:
            LOGGER.info(
                "No post-entry %s candle has closed for %s; deferring exit evaluation.",
                base_tf,
                strategy["id"],
            )
            return
        candidate_observation_time = getattr(
            self,
            "_candidate_replay_observation_time",
            None,
        )
        if candidate_observation_time is not None and latest_open < entry_effective:
            LOGGER.info(
                "Candidate %s entered after the %s bar opened; excluding that partial "
                "bar from exit evidence.",
                strategy["id"],
                base_tf,
            )
            return
        high = float(latest_bar[f"tf_{base_tf}_high"])
        low = float(latest_bar[f"tf_{base_tf}_low"])
        close = float(latest_bar[f"tf_{base_tf}_close"])

        direction = open_position["direction"]
        sl_price = open_position["sl_price"]
        tp_price = open_position["tp_price"]
        horizon = int(strategy["horizon_bars"])

        exit_triggered = False
        exit_price = 0.0
        exit_reason = ""

        if direction == "long":
            if low <= sl_price:
                exit_triggered, exit_price, exit_reason = True, sl_price, "stop"
            elif high >= tp_price:
                exit_triggered, exit_price, exit_reason = True, tp_price, "take_profit"
        else:
            if high >= sl_price:
                exit_triggered, exit_price, exit_reason = True, sl_price, "stop"
            elif low <= tp_price:
                exit_triggered, exit_price, exit_reason = True, tp_price, "take_profit"

        if not exit_triggered and self._bars_held(strategy, open_position, latest_time) >= horizon:
            exit_triggered, exit_price, exit_reason = True, close, "time"

        if not exit_triggered:
            return

        broker_exit_fill = None
        spot_base_reconciliation = None
        if self.broker is not None and "broker_qty" in open_position:
            side = OrderSide.SELL if direction == "long" else OrderSide.BUY
            broker_symbol = str(open_position.get("broker_symbol", self.symbol))
            spot_base_before_exit = None
            if (
                self._requires_observed_spot_quote_proceeds()
                and direction == "short"
                and side == OrderSide.BUY
            ):
                spot_base_before_exit = self._broker_spot_base_balance()
                expected_flat_balance = float(open_position["broker_entry_base_qty_after"])
                tolerance = max(abs(expected_flat_balance) * 1e-9, 1e-12)
                if abs(spot_base_before_exit - expected_flat_balance) > tolerance:
                    raise RuntimeError(
                        "Live spot BTC balance changed while the step-aside position was open "
                        f"for {strategy['id']}: {spot_base_before_exit:g} != "
                        f"{expected_flat_balance:g}. Refusing buyback until the account is "
                        "reconciled."
                    )
            raw_qty = self._broker_exit_order_qty(
                strategy,
                open_position,
                side=side,
                fallback_price=exit_price,
            )
            qty = self._normalize_broker_order_qty(
                broker_symbol,
                raw_qty,
                price=exit_price,
                reduce_only=True,
            )
            intent_ref = f"{open_position['entry_time']}|{latest_time}|{exit_reason}"
            client_id = self._deterministic_client_order_id(
                strategy_id=strategy["id"],
                stage="exit",
                intent_ref=intent_ref,
                symbol=broker_symbol,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                reduce_only=True,
            )
            exit_order = Order(
                symbol=broker_symbol,
                side=side,
                qty=qty,
                type=OrderType.MARKET,
                reduce_only=True,
                client_id=client_id,
            )
            self._persist_pending_order(
                strategy,
                stage="exit",
                intent_ref=intent_ref,
                order=exit_order,
            )
            try:
                broker_exit_fill = self.broker.place_order(exit_order)
            except Exception:
                if self._requires_native_protective_stop():
                    try:
                        adopted = self._reconcile_native_protection(
                            strategy,
                            open_position,
                            clear_pending=True,
                        )
                    except Exception as reconciliation_exc:
                        raise RuntimeError(
                            f"Broker exit failed for {strategy['id']} and the native stop could not "
                            "be proven fully triggered with a flat broker position; pending exit "
                            "intent remains for operator reconciliation."
                        ) from reconciliation_exc
                    if adopted:
                        return
                raise
            exit_price = float(broker_exit_fill.price)
            self._assert_broker_exit_fill_valid(strategy, exit_order, qty, broker_exit_fill)
            if spot_base_before_exit is not None:
                spot_base_after_exit = self._broker_spot_base_balance()
                observed_bought = self._validated_live_spot_base_change(
                    strategy["id"],
                    side=OrderSide.BUY,
                    fill_qty=float(broker_exit_fill.qty),
                    balance_before=spot_base_before_exit,
                    balance_after=spot_base_after_exit,
                )
                original_base_balance = float(open_position["broker_entry_base_qty_before"])
                if not math.isfinite(original_base_balance) or original_base_balance <= 0:
                    raise RuntimeError(f"Live spot BTC baseline is invalid for {strategy['id']}.")
                observed_account_return = (
                    spot_base_after_exit - original_base_balance
                ) / original_base_balance
                if not math.isfinite(observed_account_return) or observed_account_return <= -1:
                    raise RuntimeError(
                        f"Live spot BTC account return is invalid for {strategy['id']}."
                    )
                spot_base_reconciliation = {
                    "entry_base_balance_before": original_base_balance,
                    "entry_base_balance_after": float(open_position["broker_entry_base_qty_after"]),
                    "exit_base_balance_before": spot_base_before_exit,
                    "exit_base_balance_after": spot_base_after_exit,
                    "observed_buy_qty": observed_bought,
                    "account_return": observed_account_return,
                }
            if not self._is_spot_broker():
                self._assert_broker_flat_after_exit(strategy, exit_order.symbol)
            if self._requires_native_protective_stop():
                self._cancel_native_protection(strategy, open_position)

        self._complete_position_exit(
            strategy,
            open_position,
            exit_time=(
                candidate_observation_time
                if candidate_observation_time is not None
                else latest_time
            ),
            exit_price=exit_price,
            exit_reason=exit_reason,
            broker_exit_fill=broker_exit_fill,
            clear_pending=broker_exit_fill is not None,
            spot_base_reconciliation=spot_base_reconciliation,
        )

    def _complete_position_exit(
        self,
        strategy: dict,
        open_position: dict,
        *,
        exit_time,
        exit_price: float,
        exit_reason: str,
        broker_exit_fill: Fill | None,
        clear_pending: bool,
        spot_base_reconciliation: dict[str, float] | None = None,
    ) -> None:
        """Apply one validated close to accounting, log, and durable state."""

        direction = open_position["direction"]
        entry_price = float(open_position["entry_price"])
        position_size = float(open_position["position_size"])
        base_tf = strategy["base_timeframe"]
        fees = strategy["fees"]
        total_cost = 2 * ((fees["fee_bps"] + fees["slippage_bps"]) / 10_000)
        transaction_cost_source = "modeled_round_trip"
        if broker_exit_fill is not None and self._requires_native_protective_stop():
            entry_qty = float(open_position["broker_qty"])
            entry_notional = entry_price * entry_qty
            if not math.isfinite(entry_notional) or entry_notional <= 0:
                raise RuntimeError(
                    f"Live transaction-cost accounting has invalid entry notional for {strategy['id']}."
                )
            actual_fee_fraction = (
                float(open_position["broker_entry_fee"]) + float(broker_exit_fill.fee)
            ) / entry_notional
            if not math.isfinite(actual_fee_fraction) or actual_fee_fraction < 0:
                raise RuntimeError(
                    f"Live transaction-cost accounting has invalid fill fees for {strategy['id']}."
                )
            # Fill prices already include realized slippage. Keep the modeled
            # fee component as a conservative floor because an exchange order
            # lookup (especially a triggered conditional) can omit commission.
            modeled_fee_floor = 2 * (float(fees["fee_bps"]) / 10_000)
            total_cost = max(actual_fee_fraction, modeled_fee_floor)
            transaction_cost_source = (
                "broker_fill_fees"
                if actual_fee_fraction >= modeled_fee_floor
                else "modeled_fee_floor"
            )
        pnl_unit = strategy.get("pnl_unit") or self.artifact.get("pnl_unit")
        if pnl_unit is None:
            pnl_unit = "btc" if self.objective == "btc_accumulation" else "usdt"
        gross_return = gross_return_for_pnl_unit(
            entry_price,
            exit_price,
            is_long=direction == "long",
            pnl_unit=str(pnl_unit),
        )
        net_return = gross_return - total_cost
        sized_return = net_return * position_size
        accounting_return_source = "modeled_trade"
        accounting_adjustment_fraction = 0.0
        broker_entry_balance = None
        broker_exit_balance = None
        broker_balance_return = None
        broker_entry_base_balance = None
        broker_exit_base_balance = None
        broker_base_balance_return = None

        if broker_exit_fill is not None and self._requires_native_protective_stop():
            try:
                broker_entry_balance = float(open_position["broker_entry_balance"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Live account reconciliation is missing broker_entry_balance for "
                    f"{strategy['id']}; local accounting was not finalized."
                ) from exc
            if not math.isfinite(broker_entry_balance) or broker_entry_balance <= 0:
                raise RuntimeError(
                    f"Live account reconciliation has invalid broker_entry_balance for "
                    f"{strategy['id']}; local accounting was not finalized."
                )
            # Callers have already proved the futures position flat and the
            # native stop terminal. At that boundary the quote balance includes
            # realized PnL, commissions, and funding paid while the trade was
            # open. Use it only as a downside reconciliation: deposits or other
            # positive credits can never improve the modeled result.
            broker_exit_balance = self._broker_quote_balance()
            broker_balance_return = (
                broker_exit_balance - broker_entry_balance
            ) / broker_entry_balance
            if not math.isfinite(broker_balance_return):
                raise RuntimeError(
                    f"Live account reconciliation produced a non-finite balance return for "
                    f"{strategy['id']}; local accounting was not finalized."
                )
            reconciliation_tolerance = max(abs(sized_return) * 1e-9, 1e-12)
            if broker_balance_return < sized_return - reconciliation_tolerance:
                accounting_adjustment_fraction = broker_balance_return - sized_return
                sized_return = broker_balance_return
                net_return = sized_return / position_size
                accounting_return_source = "conservative_broker_balance"

        if spot_base_reconciliation is not None:
            if broker_exit_fill is None or not self._requires_observed_spot_quote_proceeds():
                raise RuntimeError(
                    f"Spot BTC reconciliation was supplied outside a live spot exit for "
                    f"{strategy['id']}."
                )
            broker_entry_base_balance = float(spot_base_reconciliation["entry_base_balance_before"])
            broker_exit_base_balance = float(spot_base_reconciliation["exit_base_balance_after"])
            broker_base_balance_return = float(spot_base_reconciliation["account_return"])
            modeled_sized_return = sized_return
            sized_return = broker_base_balance_return
            if position_size <= 0:
                raise RuntimeError(f"Live spot position size is invalid for {strategy['id']}.")
            net_return = sized_return / position_size
            accounting_adjustment_fraction = sized_return - modeled_sized_return
            accounting_return_source = "observed_btc_balance"

        if "exit_accounting_intent" in self.state:
            raise RuntimeError("Cannot replace an unresolved exit accounting intent.")
        persisted_position = self.state.get("open_positions", {}).get(strategy["id"])
        if persisted_position != open_position:
            raise RuntimeError(
                f"Open position changed before exit accounting for {strategy['id']}."
            )
        if clear_pending and "pending_order" not in self.state:
            raise RuntimeError(
                "Cannot clear an exit broker intent that is absent from durable state."
            )

        # Calculate the complete target state on a detached copy. No local
        # accounting mutation becomes visible or durable until the keyed trade
        # row is safely present and the final state replacement succeeds.
        state_before = copy.deepcopy(self.state)
        state_after = copy.deepcopy(state_before)
        self.state = state_after
        try:
            self.state["equity"] *= 1.0 + sized_return
            self._accumulate_daily_return(sized_return)
            self._normalize_drawdown_state()

            risk = strategy["risk"]
            if sized_return < 0:
                self.state["consecutive_losses"] += 1
                if self.state["consecutive_losses"] >= risk["max_consecutive_losses"]:
                    tf_seconds = TIMEFRAME_SECONDS.get(base_tf, 300)
                    cooldown_duration = risk["cooldown_bars"] * tf_seconds
                    replay_clock = getattr(
                        self,
                        "_candidate_replay_risk_clock_ts",
                        None,
                    )
                    cooldown_anchor = (
                        float(replay_clock) if replay_clock is not None else time.time()
                    )
                    self.state["cooldown_until_ts"] = cooldown_anchor + cooldown_duration
                    self.state["consecutive_losses"] = 0
                    LOGGER.warning(
                        "Consecutive losses hit limit. Cooling down for %d %s bars.",
                        risk["cooldown_bars"],
                        base_tf,
                    )
            else:
                self.state["consecutive_losses"] = 0

            del self.state["open_positions"][strategy["id"]]
            active_ids = {active_strategy["id"] for active_strategy in self.strategies}
            if strategy["id"] not in active_ids:
                self.state["last_entry_decision_bar_by_strategy"].pop(strategy["id"], None)
            if clear_pending:
                self.state.pop("pending_order")
        finally:
            self.state = state_before

        exit_event_id = self._exit_event_id(strategy["id"], open_position)
        trade_data = self._build_trade_data(
            strategy["id"],
            open_position["entry_time"],
            str(exit_time),
            direction,
            entry_price,
            exit_price,
            exit_reason,
            gross_return,
            net_return,
            sized_return,
            position_size,
            equity_after=float(state_after["equity"]),
            exit_event_id=exit_event_id,
            transaction_cost_fraction=total_cost,
            transaction_cost_source=transaction_cost_source,
            accounting_return_source=accounting_return_source,
            accounting_adjustment_fraction=accounting_adjustment_fraction,
            broker_entry_balance=broker_entry_balance,
            broker_exit_balance=broker_exit_balance,
            broker_balance_return=broker_balance_return,
            broker_entry_base_balance=broker_entry_base_balance,
            broker_exit_base_balance=broker_exit_base_balance,
            broker_base_balance_return=broker_base_balance_return,
            broker_entry_fee=open_position.get("broker_entry_fee"),
            broker_exit_fill=broker_exit_fill,
            strategy_fingerprint_value=open_position.get("approval_strategy_fingerprint"),
            artifact_digest_value=open_position.get("artifact_digest"),
            alpha_forecast=open_position.get("alpha_forecast"),
            candidate_paper_execution_schema=open_position.get("candidate_paper_execution_schema"),
            candidate_paper_engine_digest_value=open_position.get("candidate_paper_engine_digest"),
            candidate_paper_evidence_eligible=open_position.get(
                "candidate_paper_evidence_eligible"
            ),
            candidate_paper_evidence_reason=open_position.get("candidate_paper_evidence_reason"),
            candidate_paper_entry_fill_source=open_position.get(
                "candidate_paper_entry_fill_source"
            ),
            candidate_paper_observed_at=open_position.get("candidate_paper_observed_at"),
        )
        intent_without_digest = {
            "version": 1,
            "phase": "ready_to_commit",
            "exit_event_id": exit_event_id,
            "strategy_id": strategy["id"],
            "created_at": _utc_now_text(),
            "state_before_digest": _canonical_json_digest(
                state_before,
                label="Exit accounting pre-state",
            ),
            "position_digest": _canonical_json_digest(
                open_position,
                label="Exit accounting position",
            ),
            "broker_flat_proven": bool(
                broker_exit_fill is not None and self._requires_native_protective_stop()
            ),
            "trade_data": trade_data,
            "state_after": state_after,
        }
        intent = {
            **intent_without_digest,
            "payload_digest": _canonical_json_digest(
                intent_without_digest,
                label="Exit accounting intent payload",
            ),
        }
        self.state["exit_accounting_intent"] = intent
        try:
            self._validated_exit_accounting_intent(intent)
            self._save_state()
        except Exception:
            self.state.pop("exit_accounting_intent", None)
            raise

        # From this point every retry is a pure, keyed accounting recovery; no
        # exit order path is entered again.
        self._resume_exit_accounting_intent()
        LOGGER.critical(
            "%s ORDER CLOSED [%s]: %s @ %.2f | Reason: %s | Net: %.4f%% | Sized: %.4f%% | Equity: %.2f",
            "BROKER" if broker_exit_fill is not None else "PAPER",
            strategy["id"],
            direction.upper(),
            exit_price,
            exit_reason,
            net_return * 100,
            sized_return * 100,
            self.state["equity"],
        )
        self.check_drift_and_ood(strategy)

    def _accumulate_daily_return(self, sized_return: float) -> None:
        """Update the daily stop tracker without letting arithmetic hide loss.

        Simple addition understates a gain-then-equal-loss sequence because the
        second percentage applies to a larger balance. Pure compounding, on the
        other hand, is less conservative for consecutive losses. Keep the worse
        of both measures so the daily gate never receives the more favorable
        interpretation.
        """

        current = float(self.state["daily_pnl"])
        additive = current + sized_return
        compounded = ((1.0 + current) * (1.0 + sized_return)) - 1.0
        updated = min(additive, compounded)
        if not math.isfinite(updated):
            raise RuntimeError("Daily PnL accounting produced a non-finite result.")
        self.state["daily_pnl"] = updated

    def _build_trade_data(
        self,
        strategy_id: str,
        entry_time: str,
        exit_time: str,
        direction: str,
        entry: float,
        exit: float,
        exit_reason: str,
        gross_return: float,
        net_return: float,
        sized_return: float,
        position_size: float,
        *,
        equity_after: float,
        exit_event_id: str,
        transaction_cost_fraction: float = 0.0,
        transaction_cost_source: str = "unknown",
        accounting_return_source: str = "modeled_trade",
        accounting_adjustment_fraction: float = 0.0,
        broker_entry_balance=None,
        broker_exit_balance=None,
        broker_balance_return=None,
        broker_entry_base_balance=None,
        broker_exit_base_balance=None,
        broker_base_balance_return=None,
        broker_entry_fee=None,
        broker_exit_fill=None,
        strategy_fingerprint_value: str | None = None,
        artifact_digest_value: str | None = None,
        alpha_forecast: dict | None = None,
        candidate_paper_execution_schema: str | None = None,
        candidate_paper_engine_digest_value: str | None = None,
        candidate_paper_evidence_eligible: bool | None = None,
        candidate_paper_evidence_reason: str | None = None,
        candidate_paper_entry_fill_source: str | None = None,
        candidate_paper_observed_at: str | None = None,
    ) -> dict:
        if strategy_fingerprint_value is None:
            strategy_fingerprint_value = self.approval_fingerprints_by_strategy.get(strategy_id)
        if artifact_digest_value is None:
            artifact_digest_value = self.artifact_content_digest
        trade_data = {
            "exit_event_id": exit_event_id,
            "strategy_id": strategy_id,
            "strategy_fingerprint": strategy_fingerprint_value,
            "artifact_digest": artifact_digest_value,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit,
            "exit_reason": exit_reason,
            "gross_return": gross_return,
            "transaction_cost_fraction": transaction_cost_fraction,
            "transaction_cost_source": transaction_cost_source,
            "accounting_return_source": accounting_return_source,
            "accounting_adjustment_fraction": accounting_adjustment_fraction,
            "net_return": net_return,
            "sized_return": sized_return,
            "position_size": position_size,
            "equity_after": equity_after,
        }
        if alpha_forecast is not None:
            forecast = AlphaForecast.from_dict(alpha_forecast)
            trade_data.update(
                alpha_source_id=forecast.source_id,
                alpha_product=forecast.product,
                alpha_market=forecast.market,
                alpha_symbol=forecast.symbol,
                alpha_score=forecast.score,
                alpha_expected_return=forecast.expected_return,
                alpha_confidence=forecast.confidence,
                alpha_horizon_seconds=forecast.horizon_seconds,
            )
        if (
            candidate_paper_execution_schema is not None
            or candidate_paper_engine_digest_value is not None
        ):
            if candidate_paper_execution_schema != CANDIDATE_PAPER_EXECUTION_SCHEMA:
                raise RuntimeError("Candidate paper trade has an invalid execution schema.")
            if not isinstance(candidate_paper_engine_digest_value, str):
                raise RuntimeError("Candidate paper trade has no execution engine digest.")
            if not isinstance(candidate_paper_evidence_eligible, bool):
                raise RuntimeError("Candidate paper trade has invalid evidence eligibility.")
            if candidate_paper_observed_at is None:
                raise RuntimeError("Candidate paper trade has no observation time.")
            try:
                observed_at = self._candidate_replay_observation_timestamp(
                    candidate_paper_observed_at
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Candidate paper trade has an invalid observation time."
                ) from exc
            if candidate_paper_evidence_eligible:
                if (
                    candidate_paper_evidence_reason != CANDIDATE_PAPER_FORWARD_REASON
                    or candidate_paper_entry_fill_source != CANDIDATE_PAPER_FORWARD_FILL_SOURCE
                    or observed_at != self._candidate_replay_observation_timestamp(entry_time)
                ):
                    raise RuntimeError(
                        "Promotable candidate paper trade has inconsistent observation evidence."
                    )
            elif candidate_paper_evidence_reason not in {
                CANDIDATE_PAPER_BACKFILL_ENTRY_REASON,
                CANDIDATE_PAPER_BACKFILL_MANAGEMENT_REASON,
            }:
                raise RuntimeError("Non-promotable candidate paper trade has no backfill reason.")
            trade_data.update(
                candidate_paper_execution_schema=candidate_paper_execution_schema,
                candidate_paper_engine_digest=candidate_paper_engine_digest_value,
                candidate_paper_evidence_eligible=candidate_paper_evidence_eligible,
                candidate_paper_evidence_reason=candidate_paper_evidence_reason,
                candidate_paper_entry_fill_source=candidate_paper_entry_fill_source,
                candidate_paper_observed_at=observed_at.isoformat(),
            )
        if broker_exit_fill is not None:
            trade_data.update(
                broker_symbol=broker_exit_fill.symbol,
                broker_exit_qty=float(broker_exit_fill.qty),
                broker_exit_price=float(broker_exit_fill.price),
                broker_exit_fee=float(broker_exit_fill.fee),
            )
            if broker_entry_fee is not None:
                trade_data["broker_entry_fee"] = float(broker_entry_fee)
            if broker_entry_balance is not None:
                trade_data["broker_entry_balance"] = float(broker_entry_balance)
            if broker_exit_balance is not None:
                trade_data["broker_exit_balance"] = float(broker_exit_balance)
            if broker_balance_return is not None:
                trade_data["broker_balance_return"] = float(broker_balance_return)
            if broker_entry_base_balance is not None:
                trade_data["broker_entry_base_balance"] = float(broker_entry_base_balance)
            if broker_exit_base_balance is not None:
                trade_data["broker_exit_base_balance"] = float(broker_exit_base_balance)
            if broker_base_balance_return is not None:
                trade_data["broker_base_balance_return"] = float(broker_base_balance_return)
        # The WAL is JSON, so reject NaN/Infinity before it can become a
        # non-reproducible accounting record.
        _canonical_json_digest(trade_data, label="Trade log row")
        return trade_data

    @staticmethod
    def _trade_row_matches(existing: pd.Series, expected: dict) -> bool:
        for key, expected_value in expected.items():
            if key not in existing.index:
                return False
            actual_value = existing[key]
            if expected_value is None:
                if pd.isna(actual_value) or str(actual_value) == "":
                    continue
                return False
            if isinstance(expected_value, int | float) and not isinstance(expected_value, bool):
                try:
                    actual_number = float(actual_value)
                except (TypeError, ValueError):
                    return False
                if not math.isclose(
                    actual_number,
                    float(expected_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    return False
                continue
            if str(actual_value) != str(expected_value):
                return False
        return True

    def _append_trade_data_idempotent(self, trade_data: dict) -> None:
        event_id = trade_data.get("exit_event_id")
        if not isinstance(event_id, str) or re.fullmatch(r"[0-9a-f]{64}", event_id) is None:
            raise RuntimeError("Trade row requires a valid exit_event_id.")
        df_new = pd.DataFrame([trade_data])
        _reject_symlink_path(self.trade_log, "Trade log")
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        if self.trade_log.exists() and self.trade_log.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(self.trade_log)
            except Exception as exc:
                raise RuntimeError(f"Trade log is unreadable: {self.trade_log}") from exc
            if "exit_event_id" in df_existing.columns:
                matches = df_existing[df_existing["exit_event_id"].astype(str) == event_id]
                if len(matches) > 1:
                    raise RuntimeError(f"Trade log contains duplicate exit_event_id {event_id}.")
                if len(matches) == 1:
                    if not self._trade_row_matches(matches.iloc[0], trade_data):
                        raise RuntimeError(
                            "Trade log exit_event_id already exists with different "
                            f"accounting data: {event_id}."
                        )
                    return
            columns = _ordered_trade_log_columns(df_existing.columns, df_new.columns)
            df_out = pd.concat(
                [
                    df_existing.reindex(columns=columns),
                    df_new.reindex(columns=columns),
                ],
                ignore_index=True,
            )
        else:
            columns = _ordered_trade_log_columns([], df_new.columns)
            df_out = df_new.reindex(columns=columns)
        write_text_atomic(self.trade_log, df_out.to_csv(index=False))

    def _log_trade(
        self,
        strategy_id: str,
        entry_time: str,
        exit_time: str,
        direction: str,
        entry: float,
        exit: float,
        exit_reason: str,
        gross_return: float,
        net_return: float,
        sized_return: float,
        position_size: float,
        transaction_cost_fraction: float = 0.0,
        transaction_cost_source: str = "unknown",
        accounting_return_source: str = "modeled_trade",
        accounting_adjustment_fraction: float = 0.0,
        broker_entry_balance=None,
        broker_exit_balance=None,
        broker_balance_return=None,
        broker_entry_base_balance=None,
        broker_exit_base_balance=None,
        broker_base_balance_return=None,
        broker_entry_fee=None,
        broker_exit_fill=None,
        strategy_fingerprint_value: str | None = None,
        artifact_digest_value: str | None = None,
        exit_event_id: str | None = None,
    ) -> None:
        """Compatibility helper for direct audit-log writes outside exit WALs."""

        if exit_event_id is None:
            exit_event_id = _canonical_json_digest(
                {
                    "version": 1,
                    "strategy_id": strategy_id,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": direction,
                    "entry_price": entry,
                    "exit_price": exit,
                    "exit_reason": exit_reason,
                },
                label="Direct trade event identity",
            )
        trade_data = self._build_trade_data(
            strategy_id,
            entry_time,
            exit_time,
            direction,
            entry,
            exit,
            exit_reason,
            gross_return,
            net_return,
            sized_return,
            position_size,
            equity_after=float(self.state["equity"]),
            exit_event_id=exit_event_id,
            transaction_cost_fraction=transaction_cost_fraction,
            transaction_cost_source=transaction_cost_source,
            accounting_return_source=accounting_return_source,
            accounting_adjustment_fraction=accounting_adjustment_fraction,
            broker_entry_balance=broker_entry_balance,
            broker_exit_balance=broker_exit_balance,
            broker_balance_return=broker_balance_return,
            broker_entry_base_balance=broker_entry_base_balance,
            broker_exit_base_balance=broker_exit_base_balance,
            broker_base_balance_return=broker_base_balance_return,
            broker_entry_fee=broker_entry_fee,
            broker_exit_fill=broker_exit_fill,
            strategy_fingerprint_value=strategy_fingerprint_value,
            artifact_digest_value=artifact_digest_value,
        )
        self._append_trade_data_idempotent(trade_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-trading bot cycle.")
    parser.add_argument("--strategies", type=Path, default=DEFAULT_STRATEGIES_PATH)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE_PATH)
    parser.add_argument("--trade-log", type=Path, default=TRADE_LOG_PATH)
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=("futures", "spot"), default="futures")
    parser.add_argument(
        "--objective",
        choices=("btc_accumulation", "active_income"),
        help="Optional product objective for product-specific execution guards.",
    )
    parser.add_argument(
        "--base-asset", help="Optional product base asset for product-specific execution guards."
    )
    parser.add_argument(
        "--regime-guard",
        action="store_true",
        help="BTC accumulation overlay: block new LONG entries when the daily macro "
        "regime is risk-off (trend break / Mayer overheat / Pi-Cycle top).",
    )
    parser.add_argument(
        "--regime-mayer-top",
        type=float,
        default=REGIME_MAYER_TOP,
        help="Mayer Multiple threshold for the macro overheat gate (default 2.4).",
    )
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    LOGGER.info("Starting Paper Trading Bot cycle...")
    bot = PaperTradingBot(
        strategies_path=args.strategies,
        state_file=args.state_file,
        trade_log=args.trade_log,
        starting_equity=args.starting_equity,
        regime_guard=args.regime_guard,
        regime_mayer_top=args.regime_mayer_top,
        symbol=args.symbol,
        market=args.market,
        objective=args.objective,
        base_asset=args.base_asset,
    )
    bot.run_cycle()
    LOGGER.info("Bot cycle complete.")


if __name__ == "__main__":
    main()
