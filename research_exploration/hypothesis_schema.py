"""Multi-timeframe hypothesis grammar.

Every candidate strategy is expressed as:

    REGIME_CONTEXT  (higher timeframe)   -- are we allowed to trade this side at all?
  + SETUP           (mid timeframe)      -- is the market in the right posture?
  + TRIGGER         (lower timeframe)    -- the precise entry event
  + EXIT            (TP / SL / time)     -- how the trade ends
  + RISK            (filters, limits)    -- when to stand aside / how much to risk

The atom is a ``Predicate``: a single causal condition on one feature column of
one timeframe. Predicates are deliberately a *small, serialisable* language so a
hypothesis round-trips to JSON and so the evaluator (``evaluate.py``) can turn it
into an entry-signal series with no eval/exec and no lookahead.

Column convention matches the rest of the repo: once timeframes are aligned onto
a base table, every field is prefixed ``tf_{timeframe}_{feature}`` — e.g. a 4h
50-EMA is ``tf_4h_ema_50``. A predicate stores the *unprefixed* feature root plus
its timeframe, and ``column()`` builds the prefixed name on demand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# Finest -> coarsest. Used to sanity-check that regime >= setup >= trigger.
TF_ORDER = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")
TF_RANK = {tf: i for i, tf in enumerate(TF_ORDER)}

DIRECTIONS = ("long", "short")

# Predicate operators understood by the evaluator. Kept explicit (no eval) so the
# grammar is auditable and safe to round-trip through JSON.
OPS = {
    # feature  vs  numeric reference
    "gt", "ge", "lt", "le",
    # feature  vs  another feature on the same timeframe (state, not event)
    "gt_feature", "lt_feature",
    # event: feature crosses another feature this bar
    "cross_above", "cross_below",
    # feature now relative to itself `lookback` bars ago
    "rising", "falling",
    # slope = (f - f.shift(lookback)) / lookback  compared to `reference` (default 0)
    "slope_up", "slope_down",
    # percentage distance to another feature: (f / f_b - 1)  vs reference
    "pct_above", "pct_below",
    # low <= feature <= high  (band / range membership)
    "between",
    # causal rolling-quantile membership over `window` bars (refit-free, no leak)
    "q_ge", "q_le",
    # discrete pattern columns (candlesticks are -100/0/+100)
    "bullish", "bearish", "nonzero",
}


@dataclass(frozen=True)
class Predicate:
    """A single causal condition on one feature of one timeframe."""

    timeframe: str
    feature: str                       # unprefixed root, e.g. 'ema_50', 'rsi_14', 'close', 'max_20'
    op: str
    reference: float | None = None  # numeric RHS for gt/ge/lt/le/slope_*/pct_*
    feature_b: str | None = None    # unprefixed root on the same timeframe (cross/pct/gt_feature)
    lookback: int | None = None     # rising/falling/slope_*
    window: int | None = None       # q_ge/q_le rolling quantile window (in NATIVE bars of this tf)
    quantile: float | None = None   # q_ge/q_le
    low: float | None = None        # between
    high: float | None = None       # between
    shift_b: int = 0                # shift feature_b back N bars before comparing (prior-range breakouts)
    note: str = ""

    def __post_init__(self) -> None:
        if self.timeframe not in TF_RANK:
            raise ValueError(f"Unknown timeframe {self.timeframe!r}")
        if self.op not in OPS:
            raise ValueError(f"Unknown op {self.op!r}. Allowed: {sorted(OPS)}")
        if not isinstance(self.feature, str) or not self.feature.strip():
            raise ValueError("predicate feature must be a non-empty string")
        if self.shift_b < 0:
            raise ValueError("predicate shift_b must be non-negative")
        if self.lookback is not None and self.lookback <= 0:
            raise ValueError("predicate lookback must be positive")
        if self.window is not None and self.window <= 1:
            raise ValueError("predicate rolling window must be greater than one")
        if self.quantile is not None and not 0 < self.quantile < 1:
            raise ValueError("predicate quantile must be in (0, 1)")
        if self.op in {"gt", "ge", "lt", "le", "pct_above", "pct_below"} and self.reference is None:
            raise ValueError(f"predicate op {self.op!r} requires reference")
        if self.op in {
            "gt_feature",
            "lt_feature",
            "cross_above",
            "cross_below",
            "pct_above",
            "pct_below",
        } and not self.feature_b:
            raise ValueError(f"predicate op {self.op!r} requires feature_b")
        if self.op in {"rising", "falling", "slope_up", "slope_down"} and self.lookback is None:
            raise ValueError(f"predicate op {self.op!r} requires lookback")
        if self.op in {"q_ge", "q_le"} and (self.window is None or self.quantile is None):
            raise ValueError(f"predicate op {self.op!r} requires window and quantile")
        if self.op == "between":
            if self.low is None or self.high is None:
                raise ValueError("predicate op 'between' requires low and high")
            if self.low > self.high:
                raise ValueError("predicate between low cannot exceed high")

    def column(self) -> str:
        return f"tf_{self.timeframe}_{self.feature}"

    def column_b(self) -> str | None:
        return f"tf_{self.timeframe}_{self.feature_b}" if self.feature_b else None

    def describe(self) -> str:
        if self.note:
            return self.note
        c = f"{self.timeframe}:{self.feature}"
        cb = f"{self.timeframe}:{self.feature_b}" if self.feature_b else None
        if self.op in ("gt", "ge", "lt", "le"):
            sym = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}[self.op]
            return f"{c} {sym} {self.reference}"
        prior = f" (prior {self.shift_b}-bar)" if self.shift_b else ""
        if self.op in ("gt_feature", "lt_feature"):
            sym = ">" if self.op == "gt_feature" else "<"
            return f"{c} {sym} {cb}{prior}"
        if self.op in ("cross_above", "cross_below"):
            return f"{c} {self.op.replace('_', ' ')} {cb}{prior}"
        if self.op in ("rising", "falling"):
            return f"{c} {self.op} over {self.lookback} bars"
        if self.op in ("slope_up", "slope_down"):
            return f"{c} slope({self.lookback}) {'>' if self.op == 'slope_up' else '<'} {self.reference or 0}"
        if self.op in ("pct_above", "pct_below"):
            return f"{c} {'>' if self.op == 'pct_above' else '<'} {cb} by {self.reference:.2%}"
        if self.op == "between":
            return f"{self.low} <= {c} <= {self.high}"
        if self.op in ("q_ge", "q_le"):
            return f"{c} {'>=' if self.op == 'q_ge' else '<='} rolling-q{self.quantile} (win {self.window})"
        if self.op in ("bullish", "bearish", "nonzero"):
            return f"{c} {self.op}"
        return f"{c} {self.op} {self.reference}"

    def columns_used(self) -> list[str]:
        cols = [self.column()]
        if self.feature_b:
            cols.append(self.column_b())  # type: ignore[arg-type]
        return cols

    def to_dict(self) -> dict:
        return {
            k: v for k, v in asdict(self).items()
            if v is not None and v != "" and not (k == "shift_b" and v == 0)
        }

    @staticmethod
    def from_dict(d: dict) -> Predicate:
        return Predicate(**d)


@dataclass(frozen=True)
class ExitRule:
    """How a trade ends. ``horizon_bars`` is the time-stop on the trigger TF.

    TP/SL are fractional (e.g. 0.012 == 1.2%) to match the canonical simulator.
    The ATR-multiple fields are *advisory*: the v1 evaluator converts them to a
    fractional TP/SL using a reference ATR%/price at backtest build time. Leaving
    them ``None`` means "use the fractional values directly".
    """

    take_profit: float
    stop_loss: float
    horizon_bars: int
    atr_take_profit: float | None = None  # ATR multiples
    atr_stop_loss: float | None = None
    trail: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.take_profit < 1:
            raise ValueError("take_profit must be in (0, 1)")
        if not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be in (0, 1)")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.atr_take_profit is not None and self.atr_take_profit <= 0:
            raise ValueError("atr_take_profit must be positive")
        if self.atr_stop_loss is not None and self.atr_stop_loss <= 0:
            raise ValueError("atr_stop_loss must be positive")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v is not False}

    @staticmethod
    def from_dict(d: dict) -> ExitRule:
        return ExitRule(**d)


@dataclass(frozen=True)
class RiskRule:
    """Position sizing + stand-aside filters. Daily limits are honoured by the
    day-trade-style evaluator path; volatility filters are applied pre-entry."""

    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.25
    max_trades_per_day: int | None = None
    max_daily_loss_r: float | None = None
    cooldown_bars: int = 0
    min_atr_pct: float | None = None  # skip entries when ATR% below this (too quiet)
    max_atr_pct: float | None = None  # skip entries when ATR% above this (too wild)
    note: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1]")
        if self.max_trades_per_day is not None and self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive")
        if self.max_daily_loss_r is not None and self.max_daily_loss_r <= 0:
            raise ValueError("max_daily_loss_r must be positive")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")
        if self.min_atr_pct is not None and self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be positive")
        if self.max_atr_pct is not None and self.max_atr_pct <= 0:
            raise ValueError("max_atr_pct must be positive")
        if (
            self.min_atr_pct is not None
            and self.max_atr_pct is not None
            and self.min_atr_pct >= self.max_atr_pct
        ):
            raise ValueError("min_atr_pct must be less than max_atr_pct")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != 0}

    @staticmethod
    def from_dict(d: dict) -> RiskRule:
        return RiskRule(**d)


@dataclass
class Hypothesis:
    """A complete, testable multi-timeframe trade idea."""

    id: str
    family: str
    idea: str
    market_logic: str
    direction: str  # 'long' | 'short'
    base_timeframe: str       # execution/trigger TF; the aligned frame is built on this
    regime_timeframe: str
    setup_timeframe: str
    trigger_timeframe: str
    regime: list[Predicate]
    setup: list[Predicate]
    trigger: list[Predicate]
    exit: ExitRule
    risk: RiskRule = field(default_factory=RiskRule)
    expected_holding: str = ""
    expected_frequency: str = ""
    invalidation: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        for label, timeframe in (
            ("base_timeframe", self.base_timeframe),
            ("regime_timeframe", self.regime_timeframe),
            ("setup_timeframe", self.setup_timeframe),
            ("trigger_timeframe", self.trigger_timeframe),
        ):
            if timeframe not in TF_RANK:
                raise ValueError(f"{label} has unknown timeframe {timeframe!r}")
        # Causality sanity: regime should be >= setup >= trigger in coarseness.
        if not (TF_RANK[self.regime_timeframe] >= TF_RANK[self.setup_timeframe] >= TF_RANK[self.trigger_timeframe]):
            raise ValueError(
                f"Timeframe ordering must satisfy regime >= setup >= trigger "
                f"(got {self.regime_timeframe} / {self.setup_timeframe} / {self.trigger_timeframe})"
            )
        if TF_RANK[self.trigger_timeframe] < TF_RANK[self.base_timeframe]:
            raise ValueError("trigger_timeframe cannot be finer than base_timeframe")

    def all_predicates(self) -> list[Predicate]:
        return [*self.regime, *self.setup, *self.trigger]

    def timeframes(self) -> set[str]:
        return {self.base_timeframe, self.regime_timeframe, self.setup_timeframe,
                self.trigger_timeframe, *(p.timeframe for p in self.all_predicates())}

    def feature_columns(self) -> list[str]:
        cols: list[str] = []
        for p in self.all_predicates():
            cols.extend(p.columns_used())
        # de-dup, stable order
        seen: set[str] = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def human_summary(self) -> str:
        lines = [
            f"[{self.id}] {self.family} / {self.direction}",
            f"  idea : {self.idea}",
            f"  logic: {self.market_logic}",
            f"  TFs  : regime={self.regime_timeframe}  setup={self.setup_timeframe}  trigger={self.trigger_timeframe}  (base {self.base_timeframe})",
            "  REGIME : " + " AND ".join(p.describe() for p in self.regime),
            "  SETUP  : " + " AND ".join(p.describe() for p in self.setup),
            "  TRIGGER: " + " AND ".join(p.describe() for p in self.trigger),
            f"  EXIT   : tp={self.exit.take_profit:.3%} sl={self.exit.stop_loss:.3%} time={self.exit.horizon_bars} bars",
            f"  HOLD   : {self.expected_holding}   FREQ: {self.expected_frequency}",
            f"  INVALID: {self.invalidation}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "idea": self.idea,
            "market_logic": self.market_logic,
            "direction": self.direction,
            "base_timeframe": self.base_timeframe,
            "regime_timeframe": self.regime_timeframe,
            "setup_timeframe": self.setup_timeframe,
            "trigger_timeframe": self.trigger_timeframe,
            "regime": [p.to_dict() for p in self.regime],
            "setup": [p.to_dict() for p in self.setup],
            "trigger": [p.to_dict() for p in self.trigger],
            "exit": self.exit.to_dict(),
            "risk": self.risk.to_dict(),
            "expected_holding": self.expected_holding,
            "expected_frequency": self.expected_frequency,
            "invalidation": self.invalidation,
            "feature_columns": self.feature_columns(),
            "tags": self.tags,
        }

    @staticmethod
    def from_dict(d: dict) -> Hypothesis:
        return Hypothesis(
            id=d["id"],
            family=d["family"],
            idea=d["idea"],
            market_logic=d.get("market_logic", ""),
            direction=d["direction"],
            base_timeframe=d["base_timeframe"],
            regime_timeframe=d["regime_timeframe"],
            setup_timeframe=d["setup_timeframe"],
            trigger_timeframe=d["trigger_timeframe"],
            regime=[Predicate.from_dict(p) for p in d.get("regime", [])],
            setup=[Predicate.from_dict(p) for p in d.get("setup", [])],
            trigger=[Predicate.from_dict(p) for p in d.get("trigger", [])],
            exit=ExitRule.from_dict(d["exit"]),
            risk=RiskRule.from_dict(d.get("risk", {})),
            expected_holding=d.get("expected_holding", ""),
            expected_frequency=d.get("expected_frequency", ""),
            invalidation=d.get("invalidation", ""),
            tags=list(d.get("tags", [])),
        )


def validate_columns_against_inventory(
    hyp: Hypothesis, inventory: dict[str, dict]
) -> list[str]:
    """Return a list of referenced columns that don't exist in the inventory.

    ``inventory`` is the JSON produced by ``feature_inventory`` *with*
    ``--include-columns`` (so per-tf column lists are present). Empty list ==
    every column the hypothesis references really exists.
    """
    problems: list[str] = []
    for p in hyp.all_predicates():
        tf_entry = inventory.get(p.timeframe, {})
        cols = set(tf_entry.get("columns", []))
        if not cols:
            continue  # can't check this tf; skip rather than false-positive
        for feat in [p.feature] + ([p.feature_b] if p.feature_b else []):
            if feat not in cols:
                problems.append(f"{p.timeframe}:{feat}")
    return sorted(set(problems))


def predicates_from_pairs(timeframe: str, specs: Sequence[dict]) -> list[Predicate]:
    """Convenience: build predicates for one timeframe from lightweight dicts."""
    return [Predicate(timeframe=timeframe, **s) for s in specs]
