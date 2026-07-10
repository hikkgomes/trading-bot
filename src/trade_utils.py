def _gross_return_core(
    entry_price: float,
    exit_price: float,
    is_long: bool,
    pnl_btc: bool,
) -> float:
    """Return one trade's gross return in the configured account asset.

    USDT returns model Binance's linear USDT-margined futures contract. BTC
    returns model the spot step-aside operation: sell BTC for quote, then use
    that quote to repurchase BTC. A BTC-denominated long is just buy-and-hold,
    so it has no excess return before costs.
    """
    if pnl_btc:
        return 0.0 if is_long else entry_price / exit_price - 1.0
    if is_long:
        return exit_price / entry_price - 1.0
    return (entry_price - exit_price) / entry_price


def gross_return_for_pnl_unit(
    entry_price: float,
    exit_price: float,
    *,
    is_long: bool,
    pnl_unit: str,
) -> float:
    """Product-aware gross return for ``usdt`` futures or ``btc`` spot."""
    normalized_unit = str(pnl_unit).lower()
    if normalized_unit not in {"usdt", "btc"}:
        raise ValueError(f"Unsupported pnl_unit {pnl_unit!r}; expected 'usdt' or 'btc'")
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("entry_price and exit_price must be positive")
    return float(_gross_return_core(entry_price, exit_price, is_long, normalized_unit == "btc"))


def linear_usdt_futures_return(entry_price: float, exit_price: float, *, is_long: bool) -> float:
    """Gross return on entry notional for a linear USDT-margined future."""
    return gross_return_for_pnl_unit(
        entry_price,
        exit_price,
        is_long=is_long,
        pnl_unit="usdt",
    )


try:
    from numba import njit

    gross_return_numba = njit(cache=True)(_gross_return_core)

    @njit(cache=True)
    def scan_tp_sl_numba(
        high, low, entry_price: float, is_long: bool, tp: float, sl: float, start_idx: int, end_idx: int
    ) -> tuple[int, int]:
        for k in range(start_idx, end_idx + 1):
            if is_long:
                if low[k] <= entry_price * (1.0 - sl):
                    return k, 1
                if high[k] >= entry_price * (1.0 + tp):
                    return k, 2
            else:
                if high[k] >= entry_price * (1.0 + sl):
                    return k, 1
                if low[k] <= entry_price * (1.0 - tp):
                    return k, 2
        return end_idx, 0

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    gross_return_numba = _gross_return_core
    scan_tp_sl_numba = None


def scan_tp_sl_python(
    high, low, entry_price: float, is_long: bool, tp: float, sl: float, start_idx: int, end_idx: int
) -> tuple[int, int]:
    for k in range(start_idx, end_idx + 1):
        if is_long:
            if low[k] <= entry_price * (1.0 - sl):
                return k, 1
            if high[k] >= entry_price * (1.0 + tp):
                return k, 2
        else:
            if high[k] >= entry_price * (1.0 + sl):
                return k, 1
            if low[k] <= entry_price * (1.0 - tp):
                return k, 2
    return end_idx, 0


def scan_tp_sl(
    high, low, entry_price: float, is_long: bool, tp: float, sl: float, start_idx: int, end_idx: int
) -> tuple[int, int]:
    if HAS_NUMBA and scan_tp_sl_numba is not None:
        return scan_tp_sl_numba(high, low, entry_price, is_long, tp, sl, start_idx, end_idx)
    return scan_tp_sl_python(high, low, entry_price, is_long, tp, sl, start_idx, end_idx)
