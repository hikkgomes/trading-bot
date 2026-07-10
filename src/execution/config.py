"""Execution config loaded from environment / .env (no external dependency).

``load_dotenv`` is a tiny reader so we don't add python-dotenv. Values already
present in the real environment take precedence over the .env file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT
from src.envfile import parse_env_lines

ACCOUNT_FINGERPRINT_PREFIX = "account-v1:"


def load_dotenv(path: Path | None = None) -> None:
    path = path or (PROJECT_ROOT / ".env")
    if path.is_symlink():
        raise ValueError(f".env must not be a symlink: {path}")
    if not path.exists():
        return
    for key, val in parse_env_lines(path.read_text(encoding="utf-8").splitlines()).items():
        os.environ.setdefault(key, val)


@dataclass
class ExchangeConfig:
    exchange: str = "binanceusdm"
    market_type: str = "futures"
    api_key: str = ""
    api_secret: str = ""
    api_password: str = ""
    testnet: bool = True
    live: bool = False
    max_notional_usd: float = 100.0
    max_fill_slippage_bps: float = 100.0
    max_futures_leverage: int = 1
    futures_margin_mode: str = "isolated"
    quote_asset: str = "USDT"

    @property
    def account_fingerprint(self) -> str:
        """Return a non-secret identity for the configured exchange account.

        Binance API keys identify an account credential without exposing the
        credential itself.  The secret and password are intentionally excluded:
        rotating either must never leak into reports, while changing the API key,
        venue, market, or testnet routing invalidates prior preflight evidence.
        """

        identity = {
            "api_key": str(self.api_key),
            "exchange": str(self.exchange).strip().lower(),
            "market_type": str(self.market_type).strip().lower(),
            "testnet": bool(self.testnet),
            "version": 1,
        }
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return f"{ACCOUNT_FINGERPRINT_PREFIX}{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def from_env(cls, load_file: bool = True, market_type: str | None = None) -> ExchangeConfig:
        if load_file:
            load_dotenv()

        selected_market = (
            (market_type or os.environ.get("EXCHANGE_MARKET_TYPE", "futures")).strip().lower()
        )
        if selected_market not in ("futures", "spot"):
            raise ValueError("EXCHANGE_MARKET_TYPE must be 'futures' or 'spot'.")
        if selected_market == "spot":
            exchange = _env_non_empty("SPOT_EXCHANGE", "binance")
        else:
            exchange = (
                _env_non_empty("FUTURES_EXCHANGE", None)
                if "FUTURES_EXCHANGE" in os.environ
                else _env_non_empty("EXCHANGE", "binanceusdm")
            )
        max_notional_usd = _positive_float("MAX_NOTIONAL_USD", "100")
        max_fill_slippage_bps = _positive_float("MAX_FILL_SLIPPAGE_BPS", "100")
        max_futures_leverage = _bounded_int("MAX_FUTURES_LEVERAGE", "1", minimum=1, maximum=3)
        futures_margin_mode = os.environ.get("FUTURES_MARGIN_MODE", "isolated").strip().lower()
        if selected_market == "futures" and futures_margin_mode != "isolated":
            raise ValueError("FUTURES_MARGIN_MODE must be 'isolated'.")
        quote_asset = os.environ.get("QUOTE_ASSET", "USDT").strip().upper()
        if not quote_asset:
            raise ValueError("QUOTE_ASSET must be non-empty.")

        return cls(
            exchange=exchange,
            market_type=selected_market,
            api_key=_env_optional_str("EXCHANGE_API_KEY"),
            api_secret=_env_optional_str("EXCHANGE_API_SECRET"),
            api_password=_env_optional_str("EXCHANGE_API_PASSWORD"),
            testnet=_env_bool("EXCHANGE_TESTNET", True),
            live=_env_bool("TRADING_LIVE", False),
            max_notional_usd=max_notional_usd,
            max_fill_slippage_bps=max_fill_slippage_bps,
            max_futures_leverage=max_futures_leverage,
            futures_margin_mode=futures_margin_mode,
            quote_asset=quote_asset,
        )


def _positive_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value:g}.")
    return value


def _env_non_empty(name: str, default: str | None) -> str:
    raw = os.environ.get(name, default)
    value = "" if raw is None else str(raw).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _env_optional_str(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag: 1/0, true/false, yes/no, or on/off.")


def _bounded_int(name: str, default: str, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, default).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value
