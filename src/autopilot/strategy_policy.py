"""Product-aware safety policy for executable strategy artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_exploration.dsr import DSR_METHOD, LIVE_MIN_DSR
from research_exploration.hypothesis_schema import Hypothesis
from src.alpha.frozen_gradient_boosting import FrozenGradientBoostingModel
from src.autopilot.config import ProductConfig
from src.autopilot.exchange_policy import split_symbol


@dataclass(frozen=True)
class RiskEnvelope:
    max_risk_per_trade: float
    max_position_fraction: float
    max_daily_loss: float
    max_consecutive_losses: int
    max_trades_per_day: int
    min_cooldown_bars: int
    max_stop_loss: float


ENVELOPES = {
    "btc_accumulation": RiskEnvelope(
        max_risk_per_trade=0.003,
        max_position_fraction=0.35,
        max_daily_loss=0.01,
        max_consecutive_losses=3,
        max_trades_per_day=2,
        min_cooldown_bars=24,
        max_stop_loss=0.05,
    ),
    "active_income": RiskEnvelope(
        max_risk_per_trade=0.005,
        max_position_fraction=0.25,
        max_daily_loss=0.03,
        max_consecutive_losses=4,
        max_trades_per_day=8,
        min_cooldown_bars=12,
        max_stop_loss=0.05,
    ),
}
ACTIVE_INCOME_MIN_DSR = LIVE_MIN_DSR
REQUIRED_RISK_KEYS = (
    "risk_per_trade",
    "max_position_fraction",
    "daily_stop_loss",
    "max_consecutive_losses",
    "cooldown_bars",
    "max_trades_per_day",
)
REQUIRED_FEE_KEYS = ("fee_bps", "slippage_bps")
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


class StrategyPolicyError(RuntimeError):
    """Raised when an artifact violates the executable strategy policy."""


def load_strategy_artifact(path: Path, *, require_live_eligible: bool = True) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Strategy artifact not found: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StrategyPolicyError(f"{path} must be valid JSON: {exc}") from exc
    if not isinstance(artifact, dict):
        raise StrategyPolicyError(f"{path} must be a JSON object.")
    non_executable = _non_executable_artifact_errors(
        path.name,
        artifact,
        require_live_eligible=require_live_eligible,
    )
    if non_executable:
        raise StrategyPolicyError(f"{path}: " + "; ".join(non_executable))
    strategies = artifact.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise StrategyPolicyError(f"{path} has no strategies.")
    bad_indexes = [
        index for index, strategy in enumerate(strategies) if not isinstance(strategy, dict)
    ]
    if bad_indexes:
        indexes = ", ".join(str(index) for index in bad_indexes)
        raise StrategyPolicyError(
            f"{path} strategies must be JSON objects; invalid indexes: {indexes}."
        )
    return artifact


def _non_executable_artifact_errors(
    label: str,
    artifact: dict[str, Any],
    *,
    require_live_eligible: bool = True,
) -> list[str]:
    errors: list[str] = []
    if artifact.get("research_only") is True:
        errors.append(f"{label}: artifact is research-only.")
    if artifact.get("executable") is False:
        errors.append(f"{label}: artifact is marked non-executable.")
    paper_trade_allowed = artifact.get("paper_trade_allowed")
    live_allowed = artifact.get("live_allowed")
    promotion_eligible = artifact.get("promotion_eligible")
    if paper_trade_allowed is False:
        errors.append(f"{label}: artifact is not allowed for paper trading.")
    if require_live_eligible:
        if paper_trade_allowed is not False and paper_trade_allowed is not True:
            errors.append(
                f"{label}: artifact must explicitly allow paper trading before live review."
            )
        if live_allowed is False:
            errors.append(f"{label}: artifact is not allowed for live trading.")
        elif live_allowed is not True:
            errors.append(f"{label}: artifact must explicitly allow live trading.")
        if promotion_eligible is False:
            errors.append(f"{label}: artifact is not eligible for promotion.")
        elif promotion_eligible is not True:
            errors.append(f"{label}: artifact must explicitly be eligible for promotion.")
    return errors


def _strategy_label(strategy: dict[str, Any], index: int) -> str:
    return str(strategy.get("id") or f"strategy[{index}]")


def _finite_float(value: Any, label: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: must be numeric.")
        return None
    if not math.isfinite(number):
        errors.append(f"{label}: must be finite.")
        return None
    return number


def _positive_int(value: Any, label: str, errors: list[str]) -> int | None:
    number = _finite_float(value, label, errors)
    if number is None:
        return None
    if number != int(number):
        errors.append(f"{label}: must be an integer.")
        return None
    integer = int(number)
    if integer <= 0:
        errors.append(f"{label}: must be positive.")
        return None
    return integer


def _required_finite_float(
    mapping: dict[str, Any], key: str, label: str, errors: list[str]
) -> float | None:
    if key not in mapping:
        errors.append(f"{label}: missing required key {key}.")
        return None
    return _finite_float(mapping.get(key), f"{label}: {key}", errors)


def _valid_condition_kind(kind: str) -> bool:
    return kind in CONDITION_KINDS or bool(
        SLOPE_KIND_RE.match(kind) or DIVERGENCE_KIND_RE.match(kind)
    )


def _symbols_match(left: str, right: str) -> bool:
    left_base, left_quote, left_settlement = split_symbol(left)
    right_base, right_quote, right_settlement = split_symbol(right)
    if (left_base, left_quote) != (right_base, right_quote):
        return False
    if left_settlement is not None and right_settlement is not None:
        return left_settlement == right_settlement
    return True


def _validate_condition(label: str, condition: Any, index: int, errors: list[str]) -> None:
    prefix = f"{label}: condition[{index}]"
    if not isinstance(condition, dict):
        errors.append(f"{prefix}: must be an object.")
        return
    feature = condition.get("feature")
    if not isinstance(feature, str) or not feature:
        errors.append(f"{prefix}.feature: must be a non-empty string.")
    kind = condition.get("kind")
    if not isinstance(kind, str) or not kind:
        errors.append(f"{prefix}.kind: must be a non-empty string.")
    elif not _valid_condition_kind(kind):
        errors.append(f"{prefix}.kind: unsupported condition kind {kind!r}.")
    _finite_float(condition.get("threshold"), f"{prefix}.threshold", errors)
    if kind in {"cross_above", "cross_below", "ratio_le", "ratio_ge"}:
        feature_b = condition.get("feature_b")
        if not isinstance(feature_b, str) or not feature_b:
            errors.append(f"{prefix}.feature_b: required for {kind}.")
    if condition.get("lookback") is not None:
        _positive_int(condition.get("lookback"), f"{prefix}.lookback", errors)
    if condition.get("quantile") is not None:
        quantile = _finite_float(condition.get("quantile"), f"{prefix}.quantile", errors)
        if quantile is not None and not 0 <= quantile <= 1:
            errors.append(f"{prefix}.quantile: must be between 0 and 1.")


def _validate_entry_payload(label: str, strategy: dict[str, Any], errors: list[str]) -> None:
    entry_type = strategy.get("entry_type", "conditions")
    if entry_type == "frozen_ml":
        regime = strategy.get("ml_regime", "all")
        if regime not in {"all", "trend", "high_volatility", "low_volatility"}:
            errors.append(f"{label}: ml_regime is invalid.")
        close_feature = strategy.get("ml_regime_close_feature")
        if close_feature is not None and (
            not isinstance(close_feature, str) or not close_feature
        ):
            errors.append(f"{label}: ml_regime_close_feature must be a non-empty string.")
        payload = strategy.get("frozen_model")
        try:
            FrozenGradientBoostingModel.from_dict(payload)
        except Exception as exc:
            errors.append(f"{label}: invalid frozen ML payload: {type(exc).__name__}: {exc}")
        return
    if entry_type == "hypothesis":
        payload = strategy.get("hypothesis")
        if not isinstance(payload, dict):
            errors.append(f"{label}: hypothesis entry must include a hypothesis object.")
            return
        try:
            Hypothesis.from_dict(payload)
        except Exception as exc:
            errors.append(f"{label}: invalid hypothesis payload: {type(exc).__name__}: {exc}")
        return
    if entry_type != "conditions":
        errors.append(f"{label}: entry_type must be conditions, hypothesis, or frozen_ml.")
        return
    conditions = strategy.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        errors.append(f"{label}: conditions entry must include at least one condition.")
        return
    for index, condition in enumerate(conditions):
        _validate_condition(label, condition, index, errors)


def _validate_leverage_metadata(
    product: ProductConfig,
    label: str,
    payload: dict[str, Any],
    errors: list[str],
) -> None:
    leverage = payload.get("leverage")
    if leverage is not None:
        leverage_value = _finite_float(leverage, f"{label}: leverage", errors)
        if leverage_value is None:
            pass
        elif leverage_value <= 0:
            errors.append(f"{label}: leverage must be positive.")
        elif leverage_value != 1:
            errors.append(f"{label}: leverage must be 1 for {product.objective}.")

    margin_mode = payload.get("margin_mode")
    if margin_mode is None:
        return
    if not isinstance(margin_mode, str) or not margin_mode.strip():
        errors.append(f"{label}: margin_mode must be a non-empty string.")
        return
    normalized = margin_mode.strip().lower()
    if product.market == "spot":
        errors.append(f"{label}: margin_mode is not allowed for spot strategies.")
    elif normalized != "isolated":
        errors.append(f"{label}: futures margin_mode must be isolated.")


def validate_strategy(
    product: ProductConfig,
    strategy: dict[str, Any],
    index: int = 0,
    *,
    require_performance_evidence: bool = True,
    require_live_metadata: bool = True,
) -> list[str]:
    errors: list[str] = []
    envelope = ENVELOPES.get(product.objective)
    if envelope is None:
        errors.append(f"{product.name}: unknown objective {product.objective!r}")
        return errors

    if product.objective == "btc_accumulation" and product.market != "spot":
        errors.append(f"{product.name}: BTC accumulation strategies must run on spot.")
    if product.objective == "active_income" and product.market != "futures":
        errors.append(f"{product.name}: active income strategies must run on futures.")

    label = _strategy_label(strategy, index)
    strategy_id = strategy.get("id")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        errors.append(f"strategy[{index}]: id must be a non-empty string.")
    strategy_market = strategy.get("market")
    if strategy_market is None:
        errors.append(f"{label}: missing strategy market.")
    elif str(strategy_market) != product.market:
        errors.append(
            f"{label}: strategy market {strategy_market!r} does not match product market {product.market!r}."
        )
    strategy_symbol = strategy.get("symbol")
    if strategy_symbol is None:
        if require_live_metadata:
            errors.append(f"{label}: missing strategy symbol.")
    elif not _symbols_match(str(strategy_symbol), product.symbol):
        errors.append(
            f"{label}: strategy symbol {strategy_symbol!r} does not match product symbol {product.symbol!r}."
        )
    direction = strategy.get("direction")
    if direction not in {"long", "short"}:
        errors.append(f"{label}: direction must be long or short.")
    if product.objective == "btc_accumulation":
        if direction != "short":
            errors.append(f"{label}: BTC accumulation strategies must be spot step-aside shorts.")
        if direction == "short" and product.market != "spot":
            errors.append(f"{label}: BTC accumulation short/step-aside logic must be spot-only.")
    _validate_leverage_metadata(product, label, strategy, errors)

    base_timeframe = strategy.get("base_timeframe")
    if not isinstance(base_timeframe, str) or not base_timeframe:
        errors.append(f"{label}: base_timeframe must be a non-empty string.")
    _positive_int(strategy.get("horizon_bars"), f"{label}: horizon_bars", errors)
    _validate_entry_payload(label, strategy, errors)

    stop_loss = _finite_float(strategy.get("stop_loss"), f"{label}: stop_loss", errors)
    take_profit = _finite_float(strategy.get("take_profit"), f"{label}: take_profit", errors)
    if stop_loss is None:
        pass
    elif stop_loss <= 0:
        errors.append(f"{label}: stop_loss must be positive.")
    elif stop_loss > envelope.max_stop_loss:
        errors.append(f"{label}: stop_loss {stop_loss:.6f} exceeds {envelope.max_stop_loss:.6f}.")
    if take_profit is None:
        pass
    elif take_profit <= 0:
        errors.append(f"{label}: take_profit must be positive.")

    risk = strategy.get("risk")
    if not isinstance(risk, dict):
        errors.append(f"{label}: risk must be an object.")
        risk = {}
    risk_per_trade = _required_finite_float(risk, "risk_per_trade", label, errors)
    if risk_per_trade is None:
        pass
    elif risk_per_trade <= 0:
        errors.append(f"{label}: risk_per_trade must be positive.")
    elif risk_per_trade > envelope.max_risk_per_trade:
        errors.append(
            f"{label}: risk_per_trade {risk_per_trade:.6f} exceeds "
            f"{envelope.max_risk_per_trade:.6f}."
        )

    max_position_fraction = _required_finite_float(risk, "max_position_fraction", label, errors)
    if max_position_fraction is None:
        pass
    elif max_position_fraction <= 0 or max_position_fraction > 1:
        errors.append(f"{label}: max_position_fraction must be > 0 and <= 1.")
    elif max_position_fraction > envelope.max_position_fraction:
        errors.append(
            f"{label}: max_position_fraction {max_position_fraction:.6f} exceeds "
            f"{envelope.max_position_fraction:.6f}."
        )

    daily_stop_loss = _required_finite_float(risk, "daily_stop_loss", label, errors)
    if daily_stop_loss is None:
        pass
    elif daily_stop_loss >= 0:
        errors.append(f"{label}: daily_stop_loss must be negative.")
    elif abs(daily_stop_loss) > envelope.max_daily_loss:
        errors.append(
            f"{label}: daily_stop_loss {daily_stop_loss:.6f} exceeds "
            f"-{envelope.max_daily_loss:.6f}."
        )

    max_losses = (
        _positive_int(
            risk.get("max_consecutive_losses"), f"{label}: max_consecutive_losses", errors
        )
        if "max_consecutive_losses" in risk
        else None
    )
    if "max_consecutive_losses" not in risk:
        errors.append(f"{label}: missing required key max_consecutive_losses.")
    if max_losses is not None:
        if max_losses > envelope.max_consecutive_losses:
            errors.append(
                f"{label}: max_consecutive_losses {max_losses} exceeds "
                f"{envelope.max_consecutive_losses}."
            )

    cooldown = (
        _positive_int(risk.get("cooldown_bars"), f"{label}: cooldown_bars", errors)
        if "cooldown_bars" in risk
        else None
    )
    if "cooldown_bars" not in risk:
        errors.append(f"{label}: missing required key cooldown_bars.")
    if cooldown is not None:
        if cooldown < envelope.min_cooldown_bars:
            errors.append(f"{label}: cooldown_bars {cooldown} below {envelope.min_cooldown_bars}.")

    max_trades = (
        _positive_int(risk.get("max_trades_per_day"), f"{label}: max_trades_per_day", errors)
        if "max_trades_per_day" in risk
        else None
    )
    if "max_trades_per_day" not in risk:
        errors.append(f"{label}: missing required key max_trades_per_day.")
    if max_trades is not None and max_trades > envelope.max_trades_per_day:
        errors.append(
            f"{label}: max_trades_per_day {max_trades} exceeds {envelope.max_trades_per_day}."
        )

    fees = strategy.get("fees")
    if not isinstance(fees, dict):
        errors.append(f"{label}: fees must be an object.")
        fees = {}
    for fee_key in ("fee_bps", "slippage_bps"):
        fee_value = _required_finite_float(fees, fee_key, label, errors)
        if fee_value is not None and fee_value < 0:
            errors.append(f"{label}: {fee_key} must be non-negative.")

    if strategy.get("baseline_win_rate") is not None:
        baseline_win_rate = _finite_float(
            strategy.get("baseline_win_rate"), f"{label}: baseline_win_rate", errors
        )
        if baseline_win_rate is not None and not 0 < baseline_win_rate < 1:
            errors.append(f"{label}: baseline_win_rate must be between 0 and 1.")

    metrics = strategy.get("metrics") or {}
    if require_performance_evidence:
        holdout = metrics.get("holdout_total_return")
        if holdout is None:
            errors.append(f"{label}: missing holdout_total_return metric.")
        else:
            holdout_value = _finite_float(holdout, f"{label}: holdout_total_return", errors)
            if holdout_value is not None and holdout_value <= 0:
                errors.append(
                    f"{label}: holdout_total_return {holdout_value:.6f} must be positive."
                )

    pnl_unit = strategy.get("pnl_unit")
    if pnl_unit is None:
        if require_live_metadata:
            errors.append(f"{label}: missing strategy pnl_unit.")
    elif product.objective == "active_income" and pnl_unit not in {"usdt", "USDT"}:
        errors.append(f"{label}: active income pnl_unit must be USDT.")
    elif product.objective == "btc_accumulation" and pnl_unit not in {"btc", "BTC"}:
        errors.append(f"{label}: BTC accumulation pnl_unit must be BTC.")
    if require_performance_evidence and product.objective == "btc_accumulation":
        excess = metrics.get("holdout_excess_return_vs_buy_hold")
        if excess is None:
            errors.append(f"{label}: missing holdout_excess_return_vs_buy_hold metric.")
        else:
            excess_value = _finite_float(
                excess, f"{label}: holdout_excess_return_vs_buy_hold", errors
            )
            if excess_value is not None and excess_value <= 0:
                errors.append(
                    f"{label}: holdout_excess_return_vs_buy_hold "
                    f"{excess_value:.6f} must be positive."
                )
    if require_performance_evidence and product.objective in {
        "active_income",
        "btc_accumulation",
    }:
        objective_label = (
            "active income" if product.objective == "active_income" else "BTC accumulation"
        )
        dsr = metrics.get("dsr_deflated", metrics.get("dsr"))
        dsr_value = None
        if dsr is None:
            errors.append(f"{label}: missing {objective_label} DSR metric.")
        else:
            dsr_value = _finite_float(dsr, f"{label}: {objective_label} DSR", errors)
            if dsr_value is not None and dsr_value < LIVE_MIN_DSR:
                errors.append(
                    f"{label}: {objective_label} DSR {dsr_value:.6f} below {LIVE_MIN_DSR:.6f}."
                )

        # Every numeric score used for live review must carry V2 evidence so a
        # pre-fix artifact whose value was actually plain PSR cannot pass under
        # a ``dsr_deflated`` label. Paper-only bootstrap artifacts bypass this
        # through the existing require_performance_evidence gate.
        if dsr_value is not None:
            if metrics.get("dsr_method") != DSR_METHOD:
                errors.append(
                    f"{label}: DSR method must be {DSR_METHOD!r}; "
                    "legacy plain-PSR evidence is not live-eligible."
                )
            n_trials = _positive_int(metrics.get("n_trials"), f"{label}: n_trials", errors)
            trial_sharpe_count = metrics.get("trial_sharpe_count")
            if (
                isinstance(trial_sharpe_count, bool)
                or not isinstance(trial_sharpe_count, int)
                or trial_sharpe_count < 0
            ):
                errors.append(f"{label}: trial_sharpe_count must be a non-negative integer.")
            dispersion = _finite_float(
                metrics.get("sr_std_trials"),
                f"{label}: sr_std_trials",
                errors,
            )
            observed_std = _finite_float(
                metrics.get("trial_sharpe_observed_std"),
                f"{label}: trial_sharpe_observed_std",
                errors,
            )
            floor = _finite_float(
                metrics.get("trial_sharpe_conservative_floor"),
                f"{label}: trial_sharpe_conservative_floor",
                errors,
            )
            for field, value in (
                ("sr_std_trials", dispersion),
                ("trial_sharpe_observed_std", observed_std),
                ("trial_sharpe_conservative_floor", floor),
            ):
                if value is not None and value < 0:
                    errors.append(f"{label}: {field} must be non-negative.")
            if n_trials is not None and n_trials > 1:
                if dispersion is not None and dispersion <= 0:
                    errors.append(
                        f"{label}: sr_std_trials must be positive for multiple-trial DSR."
                    )
                if floor is not None and floor <= 0:
                    errors.append(
                        f"{label}: trial_sharpe_conservative_floor must be positive "
                        "for multiple-trial DSR."
                    )

    return errors


def validate_strategy_artifact(
    product: ProductConfig,
    artifact: dict[str, Any],
    *,
    require_live_eligible: bool = True,
) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _non_executable_artifact_errors(
            product.name,
            artifact,
            require_live_eligible=require_live_eligible,
        )
    )
    if errors:
        return errors
    artifact_market = artifact.get("market")
    if artifact_market is None:
        errors.append(f"{product.name}: artifact missing market.")
    elif str(artifact_market) != product.market:
        errors.append(
            f"{product.name}: artifact market {artifact_market!r} does not match product market {product.market!r}."
        )
    artifact_symbol = artifact.get("symbol")
    if artifact_symbol is None:
        if require_live_eligible:
            errors.append(f"{product.name}: artifact missing symbol.")
    elif not _symbols_match(str(artifact_symbol), product.symbol):
        errors.append(
            f"{product.name}: artifact symbol {artifact_symbol!r} does not match product symbol {product.symbol!r}."
        )
    artifact_pnl_unit = artifact.get("pnl_unit")
    if artifact_pnl_unit is None:
        if require_live_eligible:
            errors.append(f"{product.name}: artifact missing pnl_unit.")
    elif product.objective == "active_income" and artifact_pnl_unit not in {"usdt", "USDT"}:
        errors.append(f"{product.name}: active income artifact pnl_unit must be USDT.")
    elif product.objective == "btc_accumulation" and artifact_pnl_unit not in {"btc", "BTC"}:
        errors.append(f"{product.name}: BTC accumulation artifact pnl_unit must be BTC.")
    _validate_leverage_metadata(product, product.name, artifact, errors)
    strategies = artifact.get("strategies", [])
    if not isinstance(strategies, list):
        errors.append(f"{product.name}: artifact strategies must be a list.")
        return errors
    if not strategies:
        errors.append(f"{product.name}: artifact has no strategies.")
        return errors
    bad_indexes = [
        index for index, strategy in enumerate(strategies) if not isinstance(strategy, dict)
    ]
    if bad_indexes:
        indexes = ", ".join(str(index) for index in bad_indexes)
        errors.append(
            f"{product.name}: artifact strategies must be JSON objects; invalid indexes: {indexes}."
        )
        return errors

    require_performance_evidence = require_live_eligible or not (
        artifact.get("live_allowed") is False and artifact.get("promotion_eligible") is False
    )
    seen_ids: set[str] = set()
    for index, strategy in enumerate(strategies):
        strategy_id = strategy.get("id")
        if isinstance(strategy_id, str) and strategy_id.strip():
            if strategy_id in seen_ids:
                errors.append(f"{product.name}: duplicate strategy id {strategy_id!r}.")
            seen_ids.add(strategy_id)
        errors.extend(
            validate_strategy(
                product,
                strategy,
                index,
                require_performance_evidence=require_performance_evidence,
                require_live_metadata=require_live_eligible,
            )
        )

    return errors


def assert_strategy_artifact_allowed(
    product: ProductConfig,
    artifact_path: Path | None = None,
    *,
    require_live_eligible: bool | None = None,
) -> dict[str, Any]:
    path = artifact_path or product.strategies_path
    if require_live_eligible is None:
        require_live_eligible = product.execution_mode == "live"
    artifact = load_strategy_artifact(path, require_live_eligible=require_live_eligible)
    return assert_loaded_strategy_artifact_allowed(
        product,
        artifact,
        artifact_path=path,
        require_live_eligible=require_live_eligible,
    )


def assert_loaded_strategy_artifact_allowed(
    product: ProductConfig,
    artifact: dict[str, Any],
    *,
    artifact_path: Path | None = None,
    require_live_eligible: bool | None = None,
) -> dict[str, Any]:
    """Apply product policy to an artifact payload that was loaded once."""
    path = artifact_path or product.strategies_path
    if require_live_eligible is None:
        require_live_eligible = product.execution_mode == "live"
    errors = validate_strategy_artifact(
        product, artifact, require_live_eligible=require_live_eligible
    )
    if errors:
        raise StrategyPolicyError(
            f"{product.name}: strategy artifact violates policy: " + "; ".join(errors)
        )
    return {"ok": True, "strategies": len(artifact.get("strategies", [])), "artifact": str(path)}
