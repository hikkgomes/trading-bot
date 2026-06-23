"""Paper-trading executor.

Reads active_strategies.json (produced by src.export_strategies from a search
output directory) and runs one evaluation cycle per invocation. Designed to be
invoked by cron/systemd at the cadence of the smallest base timeframe among
the active strategies. All state is persisted in bot_state.json so each cycle
is stateless from the process's point of view.
"""

import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from src.build_dataset import TIMEFRAME_SECONDS
from src.config import PROJECT_ROOT
from src.discover_patterns import Condition, condition_mask
import build_binance_indicator_dataset as bbid

LOGGER = logging.getLogger("trading_bot")

DEFAULT_STRATEGIES_PATH = PROJECT_ROOT / "outputs" / "active_strategies.json"
STATE_FILE_PATH = PROJECT_ROOT / "outputs" / "bot_state.json"
TRADE_LOG_PATH = PROJECT_ROOT / "outputs" / "paper_trades.csv"
DEFAULT_STARTING_EQUITY = 1_000.0
DRIFT_MIN_TRADES = 10
DRIFT_WINDOW_TRADES = 20
DRIFT_Z_THRESHOLD = -2.0
REGIME_MAYER_TOP = 2.4


def compute_macro_step_aside(
    close: pd.Series,
    mayer_top: float = REGIME_MAYER_TOP,
    trend_ema: int = 200,
    mayer_window: int = 200,
    pi_fast: int = 111,
    pi_slow: int = 350,
) -> Tuple[bool, Dict]:
    """BTC macro "step aside" state from a *daily* close series (lookahead-safe).

    Returns ``(step_aside, detail)``. Risk-off when the macro trend breaks (close
    below the long EMA), the market is overheated (Mayer Multiple > ``mayer_top``),
    or a Pi-Cycle Top cross prints. Mirrors the ``btc_cycle_guard`` strategy; used
    by the bot to gate new long entries during the accumulation regime.
    """
    close = close.astype(float)
    detail: Dict = {"bars": int(len(close))}
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


class PaperTradingBot:
    def __init__(
        self,
        strategies_path: Path = DEFAULT_STRATEGIES_PATH,
        state_file: Path = STATE_FILE_PATH,
        trade_log: Path = TRADE_LOG_PATH,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
        regime_guard: bool = False,
        regime_mayer_top: float = REGIME_MAYER_TOP,
    ):
        self.strategies_path = strategies_path
        self.state_file = state_file
        self.trade_log = trade_log
        self.starting_equity = float(starting_equity)
        self.regime_guard = bool(regime_guard)
        self.regime_mayer_top = float(regime_mayer_top)
        self.artifact: Dict = {}
        self.strategies: List[Dict] = []
        self.state: Dict = {}
        # Per-cycle macro regime evaluation (held-vs-flat overlay for the BTC bot).
        self._macro_aside: bool = False
        self._macro_detail: Dict = {}

        self._load_strategies()
        self._load_state()

    # ------------------------------------------------------------------
    # Configuration / state
    # ------------------------------------------------------------------
    def _load_strategies(self):
        if not self.strategies_path.exists():
            raise FileNotFoundError(
                f"{self.strategies_path} not found. Run a search and then "
                "`python -m src.export_strategies --search-dir <output dir>` first."
            )
        self.artifact = json.loads(self.strategies_path.read_text(encoding="utf-8"))
        self.strategies = self.artifact.get("strategies", [])
        if not self.strategies:
            raise ValueError(f"{self.strategies_path} contains no strategies.")
        for strategy in self.strategies:
            for key in ("id", "base_timeframe", "direction", "horizon_bars",
                        "take_profit", "stop_loss", "conditions", "risk", "fees"):
                if key not in strategy:
                    raise ValueError(f"Strategy entry is missing required key {key!r}.")
            strategy["_conditions"] = [Condition(**payload) for payload in strategy["conditions"]]
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

    def _account_risk(self) -> Dict:
        return self.strategies[0]["risk"]

    def _load_state(self):
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
            changed = False
            if "open_positions" not in self.state:
                self.state["open_positions"] = {}
                changed = True
            if "inactive_strategies" not in self.state:
                self.state["inactive_strategies"] = []
                changed = True
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
                "last_pnl_reset_date": str(datetime.date.today()),
            }
            self._save_state()
            LOGGER.info("Initialized new paper trading state with %.2f USDT.", self.starting_equity)

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_live_candles(self, symbol: str, market: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch recent candles and DROP the still-forming last candle.

        Binance returns the in-progress kline as the final row; evaluating
        signals on it would repaint intra-bar and diverge from the research
        simulation, which only ever sees closed candles.
        """
        if market == "futures":
            url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            url = "https://api.binance.com/api/v3/klines"

        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Binance API error: {response.text}")

        data = response.json()
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
        return df[closed].reset_index(drop=True)

    def _build_feature_frame(self, strategy: Dict) -> Tuple[pd.DataFrame, float]:
        """Fetch candles for the strategy's base + required higher timeframes
        and assemble the aligned indicator frame (closed candles only)."""
        base_tf = strategy["base_timeframe"]
        symbol = bbid.SYMBOL
        market = bbid.MARKET

        df_base = self.fetch_live_candles(symbol, market, base_tf, limit=500)
        if df_base.empty:
            raise RuntimeError(f"No closed {base_tf} candles returned for {symbol}.")
        base_close = float(df_base["close"].iloc[-1])

        df_base_ind = bbid.build_indicator_features(df_base, base_tf)
        base_prefix = f"tf_{base_tf}_"
        df_base_ind = df_base_ind.rename(columns={
            c: f"{base_prefix}{c}" for c in df_base_ind.columns if c != "timestamp"
        })
        df_base_ind["timestamp"] = pd.to_datetime(df_base_ind["timestamp"], utc=True).astype("datetime64[ns, UTC]")

        merged = df_base_ind.copy()
        required_tfs = set()
        for cond in strategy["_conditions"]:
            if cond.feature.startswith("tf_"):
                parts = cond.feature.split("_")
                if len(parts) >= 3 and parts[1] != base_tf:
                    required_tfs.add(parts[1])

        for tf in sorted(required_tfs):
            df_tf = self.fetch_live_candles(symbol, market, tf, limit=200)
            if df_tf.empty:
                raise RuntimeError(f"No closed {tf} candles returned for {symbol}.")
            df_tf_ind = bbid.build_indicator_features(df_tf, tf)
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

        return merged, base_close

    # ------------------------------------------------------------------
    # Safety nets
    # ------------------------------------------------------------------
    def check_drift_and_ood(self, strategy: Dict):
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
            self.state["last_pnl_reset_date"] = today
            self._save_state()

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    def _evaluate_macro_regime(self):
        """Refresh the macro step-aside state once per cycle (BTC daily candles)."""
        if not self.regime_guard:
            return
        try:
            df_daily = self.fetch_live_candles(bbid.SYMBOL, bbid.MARKET, "1d", limit=500)
            self._macro_aside, self._macro_detail = compute_macro_step_aside(
                df_daily["close"], mayer_top=self.regime_mayer_top
            )
            LOGGER.info(
                "Macro regime: %s | %s",
                "STEP ASIDE (risk-off)" if self._macro_aside else "engaged (risk-on)",
                self._macro_detail,
            )
        except Exception as exc:  # never let the gate crash a trading cycle
            LOGGER.error("Macro regime evaluation failed (%s); leaving entries enabled.", exc)
            self._macro_aside, self._macro_detail = False, {"error": str(exc)}

    def run_cycle(self):
        self.process_daily_reset()
        self._evaluate_macro_regime()
        for strategy in self.strategies:
            if strategy["id"] in self.state["inactive_strategies"]:
                LOGGER.info("Strategy %s is deactivated (OOD kill switch). Skipping.", strategy["id"])
                continue
            try:
                df_features, base_close = self._build_feature_frame(strategy)
            except Exception as exc:
                LOGGER.error("Failed to build features for %s: %s", strategy["id"], exc)
                continue

            open_position = self.state["open_positions"].get(strategy["id"])
            if open_position is not None:
                self._manage_open_position(strategy, df_features)
                continue

            if time.time() < self.state["cooldown_until_ts"]:
                LOGGER.info("Account in cooldown. Skipping entries.")
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

            signal_triggered = True
            for cond in strategy["_conditions"]:
                mask = condition_mask(df_features, cond).fillna(False)
                if not bool(mask.iloc[-1]):
                    signal_triggered = False
                    break
            if signal_triggered:
                self._enter_position(strategy, df_features, base_close)

    def _resolve_tp_sl(self, strategy: Dict, df_features: pd.DataFrame, base_close: float) -> Tuple[float, float]:
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

    def _enter_position(self, strategy: Dict, df_features: pd.DataFrame, base_close: float):
        latest_time = str(df_features.iloc[-1]["timestamp"])
        direction = strategy["direction"]
        tp_pct, sl_pct = self._resolve_tp_sl(strategy, df_features, base_close)
        risk_per_trade = strategy["risk"]["risk_per_trade"]
        position_size = min(risk_per_trade / sl_pct, 1.0) if sl_pct > 0 else 0.0

        if direction == "long":
            sl_price = base_close * (1.0 - sl_pct)
            tp_price = base_close * (1.0 + tp_pct)
        else:
            sl_price = base_close * (1.0 + sl_pct)
            tp_price = base_close * (1.0 - tp_pct)

        self.state["open_positions"][strategy["id"]] = {
            "entry_time": latest_time,
            "direction": direction,
            "entry_price": base_close,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "position_size": position_size,
        }
        self._save_state()
        LOGGER.critical(
            "PAPER ORDER OPENED [%s]: %s %s @ %.2f | SL: %.2f | TP: %.2f | Size: %.4f",
            strategy["id"], direction.upper(), bbid.SYMBOL, base_close, sl_price, tp_price, position_size,
        )

    def _bars_held(self, strategy: Dict, open_position: Dict, latest_time: pd.Timestamp) -> int:
        """Holding duration in closed base-TF bars, derived from timestamps so
        it stays correct even if the cron cadence differs from the bar size."""
        tf_seconds = TIMEFRAME_SECONDS.get(strategy["base_timeframe"], 300)
        entry_time = pd.Timestamp(open_position["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        elapsed = (pd.Timestamp(latest_time) - entry_time).total_seconds()
        return max(0, int(elapsed // tf_seconds))

    def _manage_open_position(self, strategy: Dict, df_features: pd.DataFrame):
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
        )
        del self.state["open_positions"][strategy["id"]]
        self._save_state()
        LOGGER.critical(
            "PAPER ORDER CLOSED [%s]: %s @ %.2f | Reason: %s | Net: %.4f%% | Sized: %.4f%% | Equity: %.2f",
            strategy["id"], direction.upper(), exit_price, exit_reason,
            net_return * 100, sized_return * 100, self.state["equity"],
        )
        self.check_drift_and_ood(strategy)

    def _log_trade(
        self, strategy_id: str, entry_time: str, exit_time: str, direction: str,
        entry: float, exit: float, exit_reason: str,
        gross_return: float, net_return: float, sized_return: float, position_size: float,
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
        df_new = pd.DataFrame([trade_data])
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        if not self.trade_log.exists():
            df_new.to_csv(self.trade_log, index=False)
        else:
            df_new.to_csv(self.trade_log, mode="a", header=False, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-trading bot cycle.")
    parser.add_argument("--strategies", type=Path, default=DEFAULT_STRATEGIES_PATH)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE_PATH)
    parser.add_argument("--trade-log", type=Path, default=TRADE_LOG_PATH)
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
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
    )
    bot.run_cycle()
    LOGGER.info("Bot cycle complete.")


if __name__ == "__main__":
    main()
