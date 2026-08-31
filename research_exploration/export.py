"""Export KEPT hypotheses from the experiment log to an active-strategies file.

This is the research -> execution handoff for the exploration workflow: the
same contract shape ``src.export_strategies`` writes for the condition-grid
searches, extended with ``entry_type: "hypothesis"`` entries that carry the
full hypothesis (predicates, exits, risk) verbatim. ``src.run_bot`` evaluates
those predicates with ``research_exploration.predicates.entry_mask`` — the
exact code that validated them — so what trades is what was tested.

Only records that earned a ``keep`` verdict from the staged validation
pipeline (``validation.py``: train -> validation -> OOS windows -> sensitivity
-> GATING holdout) are exportable, and the holdout result is re-checked here:
a strategy without a positive, populated holdout never ships. This is the
explicit fix for the old pipeline's report-only holdout.

Run:  python -m research_exploration.export --pnl-unit usdt \
          --out outputs/active_strategies_research.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from research_exploration.dsr import DSR_METHOD, LIVE_MIN_DSR
from research_exploration.experiment_log import DEFAULT_LOG, load_log
from research_exploration.hypothesis_schema import Hypothesis
from research_exploration.risk_policy import effective_risk_block
from src.autopilot.io import write_json_atomic

SCHEMA_VERSION = 2  # v1 = condition-grid entries only; v2 adds entry_type "hypothesis"
DEFAULT_OUTPUT = Path("outputs/active_strategies_research.json")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _eval_config(record: dict) -> dict:
    return (record.get("config") or {}).get("eval") or {}


def _seg_metrics(record: dict, seg: str) -> dict:
    return (record.get("metrics") or {}).get(seg) or {}


def keep_records(log_path: Path = DEFAULT_LOG) -> list[dict]:
    """Latest ``keep`` record per hypothesis id, with full hypothesis payload."""
    latest: dict[str, dict] = {}
    for rec in load_log(log_path):
        if rec.get("verdict") != "keep" or not rec.get("hypothesis"):
            continue
        hid = rec["hypothesis_id"]
        if hid not in latest or rec.get("timestamp", "") > latest[hid].get("timestamp", ""):
            latest[hid] = rec
    return sorted(
        latest.values(),
        key=lambda r: (r.get("metrics") or {}).get("dsr_deflated") or 0.0,
        reverse=True,
    )


def _record_market(record: dict) -> str | None:
    market = _eval_config(record).get("market")
    return str(market).lower() if market else None


def _record_symbol(record: dict) -> str:
    return str(_eval_config(record).get("symbol") or "BTCUSDT")


def _holdout_export_gate(record: dict) -> tuple[bool, str]:
    holdout = _seg_metrics(record, "holdout")
    if not holdout:
        return False, "no holdout metrics recorded"
    if not holdout.get("trades"):
        return False, "holdout had no trades"
    if float(holdout.get("total_return") or 0.0) <= 0:
        return False, f"holdout not positive ({holdout.get('total_return')})"
    return True, ""


def _numeric_dsr_evidence(metrics: dict) -> tuple[dict[str, float] | None, str]:
    evidence: dict[str, float] = {}
    for field in (
        "sr_std_trials",
        "trial_sharpe_observed_std",
        "trial_sharpe_conservative_floor",
    ):
        try:
            value = float(metrics.get(field))
        except (TypeError, ValueError):
            return None, f"DSR evidence {field} must be numeric"
        if not math.isfinite(value) or value < 0:
            return None, f"DSR evidence {field} must be finite and non-negative"
        evidence[field] = value
    return evidence, ""


def _dsr_export_gate(record: dict, min_dsr: float | None) -> tuple[bool, str]:
    metrics = record.get("metrics") or {}
    if metrics.get("dsr_method") != DSR_METHOD:
        return False, f"DSR evidence method is not current ({metrics.get('dsr_method')!r})"
    n_trials = metrics.get("n_trials")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        return False, "DSR evidence n_trials must be a positive integer"
    numeric_evidence, reason = _numeric_dsr_evidence(metrics)
    if numeric_evidence is None:
        return False, reason
    trial_sharpe_count = metrics.get("trial_sharpe_count")
    if (
        isinstance(trial_sharpe_count, bool)
        or not isinstance(trial_sharpe_count, int)
        or trial_sharpe_count < 0
    ):
        return False, "DSR evidence trial_sharpe_count must be a non-negative integer"
    if n_trials > 1 and (
        numeric_evidence["sr_std_trials"] <= 0
        or numeric_evidence["trial_sharpe_conservative_floor"] <= 0
    ):
        return False, "multiple-trial DSR evidence must use positive dispersion and floor"
    dsr = metrics.get("dsr_deflated")
    try:
        dsr_value = float(dsr)
    except (TypeError, ValueError):
        return False, f"dsr_deflated {dsr!r} is not numeric"
    if not math.isfinite(dsr_value):
        return False, f"dsr_deflated {dsr!r} is not finite"
    if min_dsr is not None and dsr_value < min_dsr:
        return False, f"dsr_deflated {dsr} < --min-dsr {min_dsr}"
    return True, ""


def _provenance_export_gate(
    record: dict, pnl_unit: str | None, market: str | None
) -> tuple[bool, str]:
    if pnl_unit is not None:
        rec_unit = _eval_config(record).get("pnl_unit", "usdt")
        if rec_unit != pnl_unit:
            return False, f"validated with pnl_unit={rec_unit!r}, artifact wants {pnl_unit!r}"
    rec_market = _record_market(record)
    if rec_market not in {"spot", "futures"}:
        return False, "validation record has no market provenance"
    if market is not None and rec_market != market:
        return False, f"validated on market={rec_market!r}, artifact wants {market!r}"
    return True, ""


def _exportable(
    record: dict,
    min_dsr: float | None,
    pnl_unit: str | None,
    market: str | None,
) -> tuple[bool, str]:
    """Re-check the gates at export time. Returns (ok, reason_if_not)."""
    for passed, reason in (
        _holdout_export_gate(record),
        _dsr_export_gate(record, min_dsr),
        _provenance_export_gate(record, pnl_unit, market),
    ):
        if not passed:
            return False, reason
    return True, ""


def _baseline_win_rate(record: dict) -> float | None:
    # Prefer the untouched holdout; a zero/one/missing baseline would disable
    # (or immediately trip) the bot's drift kill-switch, so never export it.
    for seg in ("holdout", "validation", "train"):
        wr = _seg_metrics(record, seg).get("win_rate")
        if wr is not None and 0.0 < float(wr) < 1.0:
            return float(wr)
    return None


def _headline_metrics(record: dict) -> dict:
    eval_cfg = _eval_config(record)
    metrics = record.get("metrics") or {}
    headline = {
        key: metrics.get(key)
        for key in (
            "dsr_deflated",
            "dsr_method",
            "n_trials",
            "sr_std_trials",
            "trial_sharpe_count",
            "trial_sharpe_observed_std",
            "trial_sharpe_conservative_floor",
            "oos_pass_rate",
            "sensitivity_pass_fraction",
        )
    }
    for seg in ("train", "validation", "holdout"):
        for key, value in _seg_metrics(record, seg).items():
            headline[f"{seg}_{key}"] = value
    headline = {k: v for k, v in headline.items() if v is not None}
    if eval_cfg.get("pnl_unit", "usdt") == "btc" and "holdout_total_return" in headline:
        # BTC evaluation reports extra BTC accumulated versus simply holding.
        headline.setdefault("holdout_buy_hold_return", 0.0)
        headline.setdefault("holdout_excess_return_vs_buy_hold", headline["holdout_total_return"])
    return headline


def _risk_block(hyp: Hypothesis, *, market: str | None, pnl_unit: str | None) -> dict:
    if market is None or pnl_unit is None:
        raise ValueError("market and pnl_unit are required to resolve effective strategy risk")
    return effective_risk_block(hyp, market=market, pnl_unit=pnl_unit)


def strategy_entry(record: dict, rank: int, *, market: str | None = None) -> dict:
    """One active-strategies entry from one keep record. Round-trips the
    hypothesis through the schema so a malformed payload fails at export time,
    not in the bot."""
    hyp = Hypothesis.from_dict(record["hypothesis"])
    eval_cfg = _eval_config(record)
    headline = _headline_metrics(record)
    pnl_unit = eval_cfg.get("pnl_unit", "usdt")
    rec_market = _record_market(record)
    market = market or rec_market
    if market not in {"spot", "futures"}:
        raise ValueError(f"{hyp.id}: validation record has no market provenance")
    symbol = _record_symbol(record)
    return {
        "id": hyp.id,
        "rank": rank,
        "market": market,
        "symbol": symbol,
        "entry_type": "hypothesis",
        "base_timeframe": hyp.base_timeframe,
        "direction": hyp.direction,
        "horizon_bars": int(hyp.exit.horizon_bars),
        "take_profit": float(hyp.exit.take_profit),
        "stop_loss": float(hyp.exit.stop_loss),
        "use_atr_tp_sl": False,
        "pnl_unit": pnl_unit,
        "hypothesis": record["hypothesis"],
        "rule": " AND ".join(p.describe() for p in hyp.all_predicates()),
        "risk": _risk_block(hyp, market=market, pnl_unit=pnl_unit),
        "fees": {
            "fee_bps": float(eval_cfg.get("fee_bps", 5.0)),
            "slippage_bps": float(eval_cfg.get("slippage_bps", 2.0)),
        },
        "metrics": headline,
        "baseline_win_rate": _baseline_win_rate(record),
        "family": hyp.family,
        "validated_at": record.get("timestamp"),
        "validated_git_sha": record.get("git_sha"),
        "fingerprint": record.get("fingerprint"),
    }


def build_payload(
    log_path: Path = DEFAULT_LOG,
    top_k: int | None = None,
    min_dsr: float | None = LIVE_MIN_DSR,
    pnl_unit: str | None = None,
    ids: list[str] | None = None,
    market: str | None = None,
) -> dict:
    records = keep_records(log_path)
    if ids:
        records = [r for r in records if r["hypothesis_id"] in set(ids)]
    kept, skipped = [], []
    for rec in records:
        ok, reason = _exportable(rec, min_dsr=min_dsr, pnl_unit=pnl_unit, market=market)
        (kept if ok else skipped).append((rec, reason))
    for rec, reason in skipped:
        print(f"  skipping {rec['hypothesis_id']}: {reason}")
    if not kept:
        raise ValueError(
            f"No exportable strategies in {log_path}: need 'keep' verdicts from the "
            "staged validation (with a positive holdout) matching the requested "
            "filters. Run `python -m research_exploration.validation --real ...` first."
        )
    if top_k is not None:
        kept = kept[:top_k]
    strategies = [
        strategy_entry(rec, rank, market=market) for rank, (rec, _) in enumerate(kept, start=1)
    ]
    units = {s["pnl_unit"] for s in strategies}
    if len(units) > 1:
        raise ValueError(
            f"Selected strategies mix pnl units {sorted(units)} — one artifact serves one "
            "bot. Pass --pnl-unit btc (position bot) or --pnl-unit usdt (day-trade bot)."
        )
    markets = {s["market"] for s in strategies}
    if len(markets) > 1:
        raise ValueError(
            f"Selected strategies mix markets {sorted(markets)} — one artifact serves one "
            "execution market. Pass --market spot or --market futures."
        )
    symbols = {s["symbol"] for s in strategies}
    if len(symbols) > 1:
        raise ValueError(
            f"Selected strategies mix symbols {sorted(symbols)} — one artifact serves one "
            "execution symbol."
        )
    return {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "export_git_sha": _git_sha(),
        "source": str(log_path),
        "pnl_unit": strategies[0]["pnl_unit"],
        "market": strategies[0]["market"],
        "symbol": strategies[0]["symbol"],
        "paper_trade_allowed": True,
        "live_allowed": True,
        "promotion_eligible": True,
        "strategies": strategies,
    }


def run(
    log_path: Path = DEFAULT_LOG,
    output_path: Path = DEFAULT_OUTPUT,
    top_k: int | None = None,
    min_dsr: float | None = LIVE_MIN_DSR,
    pnl_unit: str | None = None,
    ids: list[str] | None = None,
    market: str | None = None,
) -> Path:
    payload = build_payload(
        log_path, top_k=top_k, min_dsr=min_dsr, pnl_unit=pnl_unit, ids=ids, market=market
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export kept (staged-validated) hypotheses to an active-strategies file."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Experiment log JSONL (default: outputs/research_exploration/experiment_log.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Artifact path. Point the bot at it with `python -m src.run_bot --strategies <out>`.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the K best by deflated DSR (default: all that pass).",
    )
    parser.add_argument(
        "--min-dsr",
        type=float,
        default=LIVE_MIN_DSR,
        help="Minimum batch-deflated DSR recorded at validation time.",
    )
    parser.add_argument(
        "--pnl-unit",
        choices=("usdt", "btc"),
        default=None,
        help="Only strategies validated in this unit (btc = position/accumulation bot, "
        "usdt = day-trade bot). Required when the log mixes units.",
    )
    parser.add_argument(
        "--market",
        choices=("spot", "futures"),
        default=None,
        help="Execution market stamped into the artifact. Must match the market recorded "
        "during validation.",
    )
    parser.add_argument(
        "--ids", nargs="*", default=None, help="Restrict to specific hypothesis ids."
    )
    args = parser.parse_args()

    path = run(
        args.log,
        args.out,
        top_k=args.top_k,
        min_dsr=args.min_dsr,
        pnl_unit=args.pnl_unit,
        ids=args.ids,
        market=args.market,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path} ({len(payload['strategies'])} strategies, pnl_unit={payload['pnl_unit']})")
    for s in payload["strategies"]:
        print(
            f"  #{s['rank']} {s['id']} [{s['direction']}, base {s['base_timeframe']}] "
            f"dsr={s['metrics'].get('dsr_deflated')} holdout_ret={s['metrics'].get('holdout_total_return')}"
        )


if __name__ == "__main__":
    main()
