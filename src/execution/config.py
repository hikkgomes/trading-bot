"""Execution config loaded from environment / .env (no external dependency).

``load_dotenv`` is a tiny reader so we don't add python-dotenv. Values already
present in the real environment take precedence over the .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT


def load_dotenv(path: Optional[Path] = None) -> None:
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass
class ExchangeConfig:
    exchange: str = "binanceusdm"
    api_key: str = ""
    api_secret: str = ""
    api_password: str = ""
    testnet: bool = True
    live: bool = False
    max_notional_usd: float = 100.0

    @classmethod
    def from_env(cls, load_file: bool = True) -> "ExchangeConfig":
        if load_file:
            load_dotenv()

        def _bool(name: str, default: bool) -> bool:
            return os.environ.get(name, "1" if default else "0").strip() in ("1", "true", "True")

        return cls(
            exchange=os.environ.get("EXCHANGE", "binanceusdm"),
            api_key=os.environ.get("EXCHANGE_API_KEY", ""),
            api_secret=os.environ.get("EXCHANGE_API_SECRET", ""),
            api_password=os.environ.get("EXCHANGE_API_PASSWORD", ""),
            testnet=_bool("EXCHANGE_TESTNET", True),
            live=_bool("TRADING_LIVE", False),
            max_notional_usd=float(os.environ.get("MAX_NOTIONAL_USD", "100")),
        )
