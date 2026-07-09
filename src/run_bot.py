"""Paper-trading executor.

Reads active_strategies.json (produced by src.export_strategies from a search
output directory) and runs one evaluation cycle per invocation. Designed to be
invoked directly for manual checks or by the autopilot runtime. State is
persisted to a configured JSON file so each cycle is stateless from the
process's point of view.
"""

import argparse
import datetime
import inspect
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import build_binance_indicator_dataset as bbid
from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.predicates import entry_mask, hypothesis_history_requirements
from src.autopilot.io import write_json_atomic, write_text_atomic
from src.build_dataset import TIMEFRAME_SECONDS
from src.config import PROJECT_ROOT
from src.discover_patterns import Condition, condition_mask
from src.execution.broker import Broker, Fill, Order, OrderSide, OrderType

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
TRADE_LOG_COLUMNS = (
    "strategy_id",
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "exit_reason",
    "gross_return",
    "net_return",
    "sized_return",
    "position_size",
    "equity_after",
)
BROKER_TRADE_LOG_COLUMNS = (
    "broker_symbol",
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


def _normalize_strategy_risk(strategy: dict) -> dict:
    strategy_id = strategy.get("id", "<unknown>")
    risk = strategy.get("risk")
    if not isinstance(risk, dict):
        raise ValueError(f"Strategy {strategy_id} risk must be an object.")
    missing = [key for key in REQUIRED_RISK_KEYS if key not in risk]
    if missing:
        raise ValueError(f"Strategy {strategy_id} risk is missing required key(s): {', '.join(missing)}.")
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
        raise ValueError(f"Strategy {strategy_id} fees is missing required key(s): {', '.join(missing)}.")
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
        raise ValueError(f"Strategy {strategy_id} metrics is missing required key {key!r}.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Strategy {strategy_id} {key} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Strategy {strategy_id} {key} must be finite.")
    return value


def _condition_kind_supported(kind: str) -> bool:
    return kind in CONDITION_KINDS or bool(SLOPE_KIND_RE.match(kind) or DIVERGENCE_KIND_RE.match(kind))


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
            raise ValueError(f"Strategy {strategy_id} condition[{index}].feature must be a non-empty string.")
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"Strategy {strategy_id} condition[{index}].kind must be a non-empty string.")
        if not _condition_kind_supported(kind):
            raise ValueError(f"Strategy {strategy_id} condition[{index}].kind is unsupported: {kind!r}.")
        try:
            threshold = float(payload["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Strategy {strategy_id} condition[{index}].threshold must be numeric.") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Strategy {strategy_id} condition[{index}].threshold must be finite.")
        payload = dict(payload)
        payload["threshold"] = threshold
        if kind in {"cross_above", "cross_below", "ratio_le", "ratio_ge"}:
            feature_b = payload.get("feature_b")
            if not isinstance(feature_b, str) or not feature_b:
                raise ValueError(f"Strategy {strategy_id} condition[{index}].feature_b is required for {kind}.")
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
        broker: Optional[Broker] = None,
        symbol: str = "BTCUSDT",
        market: str = "futures",
        objective: str | None = None,
        base_asset: str | None = None,
        live_gate_approved: bool = False,
    ):
        self.strategies_path = strategies_path
        self.state_file = state_file
        self.trade_log = trade_log
        _reject_symlink_path(self.strategies_path, "Strategy artifact")
        _reject_symlink_path(self.state_file, "State file")
        _reject_symlink_path(self.trade_log, "Trade log")
        self.starting_equity = float(starting_equity)
        self.regime_guard = bool(regime_guard)
        self.regime_mayer_top = float(regime_mayer_top)
        self.broker = broker
        if broker is not None and bool(getattr(getattr(broker, "config", None), "live", False)) and not live_gate_approved:
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
        self.artifact: dict = {}
        self.strategies: list[dict] = []
        self.state: dict = {}
        self.cycle_errors: list[dict] = []
        # Per-cycle macro regime evaluation (held-vs-flat overlay for the BTC bot).
        self._macro_aside: bool = False
        self._macro_detail: dict = {}
        self._feature_frame_cache: dict[tuple, tuple[pd.DataFrame, float]] = {}

        self._load_strategies()
        self._load_state()

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
                raise ValueError(f"Strategy {strategy_id} must be a spot step-aside short for BTC accumulation.")
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
                    raise ValueError(f"Strategy {strategy_id} holdout_total_return must be positive.")
                dsr_key = "dsr_deflated" if "dsr_deflated" in metrics else "dsr"
                dsr = _finite_metric(metrics, dsr_key, strategy_id)
                if dsr < ACTIVE_INCOME_MIN_DSR:
                    raise ValueError(
                        f"Strategy {strategy_id} active income DSR {dsr:.6f} below "
                        f"{ACTIVE_INCOME_MIN_DSR:.6f}."
                    )

    def _load_strategies(self):
        _reject_symlink_path(self.strategies_path, "Strategy artifact")
        if not self.strategies_path.exists():
            raise FileNotFoundError(
                f"{self.strategies_path} not found. Run a search and then "
                "`python -m src.export_strategies --search-dir <output dir>` first."
            )
        self.artifact = json.loads(self.strategies_path.read_text(encoding="utf-8"))
        artifact_market = self.artifact.get("market")
        if artifact_market is not None:
            if str(artifact_market) not in {"futures", "spot"}:
                raise ValueError(f"Strategy artifact market must be 'futures' or 'spot', got {artifact_market!r}.")
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
        seen_strategy_ids: set[str] = set()
        for strategy in self.strategies:
            entry_type = strategy.get("entry_type", "conditions")
            if entry_type not in {"conditions", "hypothesis"}:
                raise ValueError(
                    f"Strategy {strategy.get('id', '<unknown>')} entry_type must be 'conditions' or 'hypothesis'."
                )
            entry_key = "hypothesis" if entry_type == "hypothesis" else "conditions"
            for key in ("id", "base_timeframe", "direction", "horizon_bars",
                        "take_profit", "stop_loss", entry_key, "risk", "fees"):
                if key not in strategy:
                    raise ValueError(f"Strategy entry is missing required key {key!r}.")
            strategy_id = strategy["id"]
            if not isinstance(strategy_id, str) or not strategy_id.strip():
                raise ValueError("Strategy id must be a non-empty string.")
            if strategy_id in seen_strategy_ids:
                raise ValueError(f"Duplicate strategy id {strategy_id!r} in {self.strategies_path}.")
            seen_strategy_ids.add(strategy_id)
            if strategy["direction"] not in {"long", "short"}:
                raise ValueError(f"Strategy {strategy['id']} direction must be long or short.")
            self._validate_product_strategy(strategy)
            if not isinstance(strategy["base_timeframe"], str) or not strategy["base_timeframe"]:
                raise ValueError(f"Strategy {strategy['id']} base_timeframe must be a non-empty string.")
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
            if strategy_symbol is not None and not _symbols_match(str(strategy_symbol), self.symbol):
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
            else:
                strategy["_conditions"] = _normalize_conditions(strategy)
        LOGGER.info(
            "Loaded %s strategies from %s (search sha %s)",
            len(self.strategies), self.strategies_path,
            self.artifact.get("search_git_sha", "unknown"),
        )
        for strategy in self.strategies:
            LOGGER.info(
                "  %s: %s %s tf=%s horizon=%s TP=%s SL=%s",
                strategy["id"], strategy["direction"], strategy.get("rule", ""),
                strategy["base_timeframe"], strategy["horizon_bars"],
                strategy["take_profit"], strategy["stop_loss"],
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
            raise RuntimeError(f"State daily_trades_by_strategy must be an object: {self.state_file}")
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

    def _normalize_last_pnl_reset_date(self) -> bool:
        key = "last_pnl_reset_date"
        raw = self.state.get(key)
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"State {key} must be an ISO date string: {self.state_file}")
        try:
            parsed = datetime.date.fromisoformat(raw)
        except ValueError as exc:
            raise RuntimeError(f"State {key} must be an ISO date string: {self.state_file}") from exc
        today = datetime.date.today()
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

    def _normalize_open_positions(self) -> bool:
        positions = self.state["open_positions"]
        if not isinstance(positions, dict):
            raise RuntimeError(f"State open_positions must be an object: {self.state_file}")
        strategies_by_id = {strategy["id"]: strategy for strategy in self.strategies}
        changed = False
        for strategy_id, position in positions.items():
            if strategy_id not in strategies_by_id:
                raise RuntimeError(
                    f"State open_positions contains unknown strategy {strategy_id!r}: {self.state_file}"
                )
            if not isinstance(position, dict):
                raise RuntimeError(f"State open_positions[{strategy_id!r}] must be an object: {self.state_file}")
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
            direction = position.get("direction")
            if direction not in {"long", "short"}:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].direction must be long or short: {self.state_file}"
                )
            for key in ("entry_price", "sl_pct", "tp_pct", "sl_price", "tp_price"):
                changed = self._normalize_position_float(position, strategy_id, key, positive=True) or changed
            changed = self._normalize_position_float(position, strategy_id, "position_size", positive=True) or changed
            if position["position_size"] > 1.0:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].position_size must be <= 1: {self.state_file}"
                )
            max_position_fraction = float(strategies_by_id[strategy_id]["risk"]["max_position_fraction"])
            if position["position_size"] - max_position_fraction > 1e-12:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].position_size exceeds "
                    f"max_position_fraction {max_position_fraction:g}: {self.state_file}"
                )
            for key in ("broker_requested_qty", "broker_fill_ratio", "broker_qty", "broker_entry_price"):
                changed = self._normalize_optional_position_float(
                    position,
                    strategy_id,
                    key,
                    positive=True,
                ) or changed
            if "broker_fill_ratio" in position and abs(float(position["broker_fill_ratio"]) - 1.0) > 1e-9:
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
                "broker_entry_quote_value",
            ):
                changed = self._normalize_optional_position_float(
                    position,
                    strategy_id,
                    key,
                    non_negative=True,
                ) or changed
            if "broker_side" in position and position["broker_side"] not in {OrderSide.BUY.value, OrderSide.SELL.value}:
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_side must be buy or sell: {self.state_file}"
                )
            if "broker_symbol" in position and not str(position["broker_symbol"]):
                raise RuntimeError(
                    f"State open_positions[{strategy_id!r}].broker_symbol must be non-empty: {self.state_file}"
                )
            self._assert_broker_metadata_complete(
                strategy_id,
                position,
                state_detail=f"{self.state_file}",
                require_present=self.broker is not None,
            )
        return changed

    def _load_state(self):
        _reject_symlink_path(self.state_file, "State file")
        if self.state_file.exists():
            try:
                loaded = json.loads(self.state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"State file is unreadable or invalid: {self.state_file}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"State file must contain a JSON object: {self.state_file}")
            self.state = loaded
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
                ("last_pnl_reset_date", str(datetime.date.today())),
            ):
                if key not in self.state:
                    self.state[key] = default
                    changed = True
            if not isinstance(self.state["daily_trades_by_strategy"], dict):
                raise RuntimeError(f"State daily_trades_by_strategy must be an object: {self.state_file}")
            changed = self._normalize_state_float("equity", positive=True) or changed
            changed = self._normalize_state_int("consecutive_losses", non_negative=True) or changed
            changed = self._normalize_state_float("cooldown_until_ts", non_negative=True) or changed
            changed = self._normalize_state_float("daily_pnl") or changed
            changed = self._normalize_daily_trade_counts() or changed
            changed = self._normalize_last_pnl_reset_date() or changed
            changed = self._normalize_open_positions() or changed
            self.state.pop("open_position", None)
            self.state.pop("strategy_active", None)
            if changed:
                self._save_state()
            LOGGER.info("Loaded bot state. Current Equity: %.2f USDT", self.state.get("equity", self.starting_equity))
        else:
            self.state = {
                "equity": self.starting_equity,
                "open_positions": {},
                "inactive_strategies": [],
                "consecutive_losses": 0,
                "cooldown_until_ts": 0.0,
                "daily_pnl": 0.0,
                "daily_trades_by_strategy": {},
                "last_pnl_reset_date": str(datetime.date.today()),
            }
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
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain invalid timestamps.")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} must have strictly increasing timestamps.")
        for column in cls.POSITIVE_CANDLE_COLUMNS:
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain non-finite {column}.")
            if (values <= 0).any():
                raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain non-positive {column}.")
        for column in cls.NON_NEGATIVE_CANDLE_COLUMNS:
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain non-finite {column}.")
            if (values < 0).any():
                raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain negative {column}.")
        open_values = df["open"].to_numpy(dtype=float)
        high_values = df["high"].to_numpy(dtype=float)
        low_values = df["low"].to_numpy(dtype=float)
        close_values = df["close"].to_numpy(dtype=float)
        if (high_values < low_values).any():
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain high below low.")
        if (high_values < np.maximum(open_values, close_values)).any():
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain high below open/close.")
        if (low_values > np.minimum(open_values, close_values)).any():
            raise RuntimeError(f"Closed {timeframe} candles for {symbol} contain low above open/close.")

    def fetch_live_candles(self, symbol: str, market: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
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
            params = {"symbol": _binance_rest_symbol(symbol), "interval": timeframe,
                      "limit": min(remaining, self.KLINES_PER_REQUEST)}
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
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
        df = df[bbid.CANDLE_COLUMNS]
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 300)
        now = pd.Timestamp.now(tz="UTC")
        closed = df["timestamp"] + pd.Timedelta(seconds=tf_seconds) <= now
        df_closed = df[closed].reset_index(drop=True)
        self._validate_closed_candles(df_closed, symbol=symbol, timeframe=timeframe)
        return df_closed

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
                required.setdefault(predicate.timeframe, {"open", "high", "low", "close"}).add(predicate.feature)
                if predicate.feature_b:
                    required[predicate.timeframe].add(predicate.feature_b)
            if hypothesis.risk.min_atr_pct or hypothesis.risk.max_atr_pct:
                required.setdefault(base_tf, {"open", "high", "low", "close"}).add("natr_14")
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
            supports_required = (
                "required_features" in signature.parameters
                or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
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
            htf_limits = {
                tf: max(200, bars + self.INDICATOR_WARMUP_BARS)
                for tf, bars in needed.items() if tf != base_tf
            }
            return base_limit, htf_limits
        required_tfs = {
            tf for tf in self._required_features_by_timeframe(strategy)
            if tf != base_tf
        }
        return 500, {tf: 200 for tf in required_tfs}

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
                sorted(
                    (tf, tuple(sorted(features)))
                    for tf, features in required_features.items()
                )
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
        df_base_ind = df_base_ind.rename(columns={
            c: f"{base_prefix}{c}" for c in df_base_ind.columns if c != "timestamp"
        })
        df_base_ind["timestamp"] = pd.to_datetime(df_base_ind["timestamp"], utc=True).astype("datetime64[ns, UTC]")

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
            df_tf_ind = df_tf_ind.rename(columns={
                c: f"{tf_prefix}{c}" for c in df_tf_ind.columns if c != "timestamp"
            })
            df_tf_ind["timestamp"] = pd.to_datetime(df_tf_ind["timestamp"], utc=True).astype("datetime64[ns, UTC]")
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
            strategy["id"], len(recent), baseline_wr, recent_win_rate, z_score,
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
        today = str(datetime.date.today())
        if self.state["last_pnl_reset_date"] != today:
            LOGGER.info("New day detected. Resetting daily PNL tracker.")
            self.state["daily_pnl"] = 0.0
            self.state["daily_trades_by_strategy"] = {}
            self.state["last_pnl_reset_date"] = today
            self._save_state()

    def _daily_trade_count(self, strategy: dict) -> int:
        counts = self.state.setdefault("daily_trades_by_strategy", {})
        if not isinstance(counts, dict):
            raise RuntimeError(f"State daily_trades_by_strategy must be an object: {self.state_file}")
        return int(counts.get(strategy["id"], 0) or 0)

    def _increment_daily_trade_count(self, strategy: dict) -> None:
        counts = self.state.setdefault("daily_trades_by_strategy", {})
        if not isinstance(counts, dict):
            raise RuntimeError(f"State daily_trades_by_strategy must be an object: {self.state_file}")
        counts[strategy["id"]] = self._daily_trade_count(strategy) + 1

    def _daily_trade_limit_reached(self, strategy: dict) -> bool:
        limit = (strategy.get("risk") or {}).get("max_trades_per_day")
        if limit is None:
            return False
        limit = int(limit)
        if limit <= 0:
            return True
        return self._daily_trade_count(strategy) >= limit

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
            self._macro_aside, self._macro_detail = True, {
                "error": str(exc),
                "fail_closed": True,
            }

    def run_cycle(self):
        self.cycle_errors = []
        self._feature_frame_cache = {}
        self.process_daily_reset()
        self._assert_broker_open_positions_have_metadata()
        self._evaluate_macro_regime()
        for strategy in self.strategies:
            open_position = self.state["open_positions"].get(strategy["id"])
            if open_position is None:
                if self._has_other_open_position(strategy):
                    LOGGER.info(
                        "Another strategy already has an open %s position. Skipping new entry for %s.",
                        self.symbol,
                        strategy["id"],
                    )
                    continue
                if strategy["id"] in self.state["inactive_strategies"]:
                    LOGGER.info("Strategy %s is deactivated (OOD kill switch). Skipping new entry.", strategy["id"])
                    continue
                self._assert_broker_flat_before_new_entry(strategy)
            try:
                df_features, base_close = self._build_feature_frame(strategy)
            except Exception as exc:
                LOGGER.error("Failed to build features for %s: %s", strategy["id"], exc)
                self.cycle_errors.append(
                    {
                        "strategy_id": strategy["id"],
                        "stage": "feature_build",
                        "error": str(exc),
                    }
                )
                continue

            if open_position is not None:
                self._reconcile_broker_position(strategy, open_position)
                self._manage_open_position(strategy, df_features)
                continue

            if time.time() < self.state["cooldown_until_ts"]:
                LOGGER.info("Account in cooldown. Skipping entries.")
                continue
            if self.regime_guard and self._macro_detail.get("fail_closed"):
                LOGGER.warning(
                    "Macro regime unavailable: skipping new entry for %s (%s).",
                    strategy["id"], self._macro_detail,
                )
                continue
            if self.regime_guard and self._macro_aside and strategy["direction"] == "long":
                LOGGER.warning(
                    "Macro regime risk-off: skipping new LONG entry for %s (%s).",
                    strategy["id"], self._macro_detail,
                )
                continue
            if self.state["daily_pnl"] <= self._account_risk()["daily_stop_loss"]:
                LOGGER.warning(
                    "Daily stop hit (%.4f <= %.4f). Skipping entries.",
                    self.state["daily_pnl"], self._account_risk()["daily_stop_loss"],
                )
                continue
            if self._daily_trade_limit_reached(strategy):
                LOGGER.info(
                    "Daily trade limit hit for %s (%s/%s). Skipping entries.",
                    strategy["id"],
                    self._daily_trade_count(strategy),
                    strategy["risk"].get("max_trades_per_day"),
                )
                continue

            if strategy.get("entry_type", "conditions") == "hypothesis":
                # Same mask code that scored the hypothesis in research
                # (research_exploration.predicates) — live == validated.
                signal_triggered = bool(entry_mask(df_features, strategy["_hypothesis"]).iloc[-1])
            else:
                signal_triggered = True
                for cond in strategy["_conditions"]:
                    mask = condition_mask(df_features, cond).fillna(False)
                    if not bool(mask.iloc[-1]):
                        signal_triggered = False
                        break
            if signal_triggered:
                self._enter_position(strategy, df_features, base_close)

    def _resolve_tp_sl(self, strategy: dict, df_features: pd.DataFrame, base_close: float) -> tuple[float, float]:
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

    def _enter_position(self, strategy: dict, df_features: pd.DataFrame, base_close: float):
        latest_time = str(df_features.iloc[-1]["timestamp"])
        direction = strategy["direction"]
        tp_pct, sl_pct = self._resolve_tp_sl(strategy, df_features, base_close)
        risk_per_trade = strategy["risk"]["risk_per_trade"]
        max_position_fraction = strategy["risk"]["max_position_fraction"]
        position_size = min(risk_per_trade / sl_pct, max_position_fraction, 1.0) if sl_pct > 0 else 0.0
        entry_price = base_close
        broker_fill = None
        spot_sell_base_before = None
        broker_requested_qty = None
        broker_fill_ratio = None

        if self.broker is not None:
            side = OrderSide.BUY if direction == "long" else OrderSide.SELL
            if self._is_spot_broker() and side == OrderSide.SELL:
                spot_sell_base_before = max(float(self.broker.get_position(self.symbol).qty), 0.0)
            qty = self._broker_order_qty(price=base_close, position_size=position_size, side=side)
            broker_requested_qty = float(qty)
            entry_order = Order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                type=OrderType.MARKET,
                client_id=f"{strategy['id']}-entry-{int(time.time())}",
            )
            broker_fill = self.broker.place_order(entry_order)
            self._assert_broker_entry_fill_valid(strategy, entry_order, broker_fill)
            entry_price = float(broker_fill.price)
            broker_fill_ratio = 1.0

        if direction == "long":
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)
        else:
            sl_price = entry_price * (1.0 + sl_pct)
            tp_price = entry_price * (1.0 - tp_pct)

        position = {
            "entry_time": latest_time,
            "direction": direction,
            "entry_price": entry_price,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "position_size": position_size,
        }
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
            if self._is_spot_broker() and broker_fill.side == OrderSide.SELL:
                base_after = max(float(self.broker.get_position(broker_fill.symbol).qty), 0.0)
                position.update(
                    broker_entry_base_qty_before=spot_sell_base_before,
                    broker_entry_base_qty_after=base_after,
                    broker_entry_quote_value=max(
                        float(broker_fill.qty) * float(broker_fill.price) - float(broker_fill.fee),
                        0.0,
                    ),
                    broker_exit_sizing="quote_reinvest",
                )
        self.state["open_positions"][strategy["id"]] = position
        self._increment_daily_trade_count(strategy)
        self._save_state()
        LOGGER.critical(
            "%s ORDER OPENED [%s]: %s %s @ %.2f | SL: %.2f | TP: %.2f | Size: %.4f",
            "BROKER" if self.broker is not None else "PAPER",
            strategy["id"], direction.upper(), self.symbol, entry_price, sl_price, tp_price, position_size,
        )

    def _is_spot_broker(self) -> bool:
        return getattr(getattr(self.broker, "config", None), "market_type", None) == "spot"

    def _broker_order_qty(self, price: float, position_size: float, side: OrderSide) -> float:
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
        quote_equity = float(self.broker.get_balance())
        if not math.isfinite(quote_equity) or quote_equity <= 0:
            raise ValueError(f"Broker quote balance must be finite and positive, got {quote_equity:g}.")
        qty = (quote_equity * position_size) / price
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError(
                f"Broker order quantity is non-positive (balance={quote_equity}, "
                f"position_size={position_size}, price={price})."
            )
        return qty

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
                raise RuntimeError(f"State open_positions[{strategy_id!r}] must be an object: {self.state_file}")
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
            price = float(self.broker.get_price(open_position.get("broker_symbol", self.symbol)) or fallback_price)
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Cannot size a spot buyback with a non-positive price.")
            qty = quote_value / price
            if not math.isfinite(qty) or qty <= 0:
                raise ValueError(f"Spot buyback quantity is non-positive (quote_value={quote_value}, price={price}).")
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

    def _assert_broker_exit_fill_valid(self, strategy: dict, order: Order, requested_qty: float, fill: Fill) -> None:
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
        tolerance = max(expected_qty * 0.001, 1e-9)
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
        if direction == "long" and actual.qty + tolerance < expected_qty:
            raise RuntimeError(
                f"Broker position mismatch for {strategy['id']}: expected long at least "
                f"{expected_qty}, got {actual.qty}."
            )
        if direction == "short" and actual.qty - tolerance > -expected_qty:
            raise RuntimeError(
                f"Broker position mismatch for {strategy['id']}: expected short at least "
                f"{expected_qty}, got {actual.qty}."
            )

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
        elapsed = (pd.Timestamp(latest_time) - entry_time).total_seconds()
        return max(0, int(elapsed // tf_seconds))

    def _manage_open_position(self, strategy: dict, df_features: pd.DataFrame):
        open_position = self.state["open_positions"][strategy["id"]]
        latest_bar = df_features.iloc[-1]
        latest_time = latest_bar["timestamp"]
        base_tf = strategy["base_timeframe"]
        high = float(latest_bar[f"tf_{base_tf}_high"])
        low = float(latest_bar[f"tf_{base_tf}_low"])
        close = float(latest_bar[f"tf_{base_tf}_close"])

        direction = open_position["direction"]
        sl_price = open_position["sl_price"]
        tp_price = open_position["tp_price"]
        entry_price = open_position["entry_price"]
        position_size = open_position["position_size"]
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
        if self.broker is not None and "broker_qty" in open_position:
            side = OrderSide.SELL if direction == "long" else OrderSide.BUY
            qty = self._broker_exit_order_qty(strategy, open_position, side=side, fallback_price=exit_price)
            exit_order = Order(
                symbol=open_position.get("broker_symbol", self.symbol),
                side=side,
                qty=qty,
                type=OrderType.MARKET,
                reduce_only=True,
                client_id=f"{strategy['id']}-exit-{int(time.time())}",
            )
            broker_exit_fill = self.broker.place_order(exit_order)
            exit_price = float(broker_exit_fill.price)
            self._assert_broker_exit_fill_valid(strategy, exit_order, qty, broker_exit_fill)

        fees = strategy["fees"]
        total_cost = 2 * ((fees["fee_bps"] + fees["slippage_bps"]) / 10_000)
        if direction == "long":
            gross_return = exit_price / entry_price - 1.0
        else:
            gross_return = entry_price / exit_price - 1.0
        net_return = gross_return - total_cost
        sized_return = net_return * position_size

        self.state["equity"] *= 1.0 + sized_return
        self.state["daily_pnl"] += sized_return

        risk = strategy["risk"]
        if net_return < 0:
            self.state["consecutive_losses"] += 1
            if self.state["consecutive_losses"] >= risk["max_consecutive_losses"]:
                tf_seconds = TIMEFRAME_SECONDS.get(base_tf, 300)
                cooldown_duration = risk["cooldown_bars"] * tf_seconds
                self.state["cooldown_until_ts"] = time.time() + cooldown_duration
                self.state["consecutive_losses"] = 0
                LOGGER.warning(
                    "Consecutive losses hit limit. Cooling down for %d %s bars.",
                    risk["cooldown_bars"], base_tf,
                )
        else:
            self.state["consecutive_losses"] = 0

        self._log_trade(
            strategy["id"], open_position["entry_time"], str(latest_time), direction,
            entry_price, exit_price, exit_reason,
            gross_return, net_return, sized_return, position_size,
            broker_exit_fill=broker_exit_fill,
        )
        del self.state["open_positions"][strategy["id"]]
        self._save_state()
        LOGGER.critical(
            "%s ORDER CLOSED [%s]: %s @ %.2f | Reason: %s | Net: %.4f%% | Sized: %.4f%% | Equity: %.2f",
            "BROKER" if broker_exit_fill is not None else "PAPER",
            strategy["id"], direction.upper(), exit_price, exit_reason,
            net_return * 100, sized_return * 100, self.state["equity"],
        )
        self.check_drift_and_ood(strategy)

    def _log_trade(
        self, strategy_id: str, entry_time: str, exit_time: str, direction: str,
        entry: float, exit: float, exit_reason: str,
        gross_return: float, net_return: float, sized_return: float, position_size: float,
        broker_exit_fill=None,
    ):
        trade_data = {
            "strategy_id": strategy_id,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit,
            "exit_reason": exit_reason,
            "gross_return": gross_return,
            "net_return": net_return,
            "sized_return": sized_return,
            "position_size": position_size,
            "equity_after": self.state["equity"],
        }
        if broker_exit_fill is not None:
            trade_data.update(
                broker_symbol=broker_exit_fill.symbol,
                broker_exit_qty=float(broker_exit_fill.qty),
                broker_exit_price=float(broker_exit_fill.price),
                broker_exit_fee=float(broker_exit_fill.fee),
            )
        df_new = pd.DataFrame([trade_data])
        _reject_symlink_path(self.trade_log, "Trade log")
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        if self.trade_log.exists() and self.trade_log.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(self.trade_log)
            except Exception as exc:
                raise RuntimeError(f"Trade log is unreadable: {self.trade_log}") from exc
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-trading bot cycle.")
    parser.add_argument("--strategies", type=Path, default=DEFAULT_STRATEGIES_PATH)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE_PATH)
    parser.add_argument("--trade-log", type=Path, default=TRADE_LOG_PATH)
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=("futures", "spot"), default="futures")
    parser.add_argument("--objective", choices=("btc_accumulation", "active_income"),
                        help="Optional product objective for product-specific execution guards.")
    parser.add_argument("--base-asset", help="Optional product base asset for product-specific execution guards.")
    parser.add_argument(
        "--regime-guard", action="store_true",
        help="BTC accumulation overlay: block new LONG entries when the daily macro "
             "regime is risk-off (trend break / Mayer overheat / Pi-Cycle top).",
    )
    parser.add_argument("--regime-mayer-top", type=float, default=REGIME_MAYER_TOP,
                        help="Mayer Multiple threshold for the macro overheat gate (default 2.4).")
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
