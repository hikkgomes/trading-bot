from typing import Tuple

try:
    from numba import njit

    @njit(cache=True)
    def scan_tp_sl_numba(
        high, low, entry_price: float, is_long: bool, tp: float, sl: float, start_idx: int, end_idx: int
    ) -> Tuple[int, int]:
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
    scan_tp_sl_numba = None


def scan_tp_sl_python(
    high, low, entry_price: float, is_long: bool, tp: float, sl: float, start_idx: int, end_idx: int
) -> Tuple[int, int]:
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
) -> Tuple[int, int]:
    if HAS_NUMBA and scan_tp_sl_numba is not None:
        return scan_tp_sl_numba(high, low, entry_price, is_long, tp, sl, start_idx, end_idx)
    return scan_tp_sl_python(high, low, entry_price, is_long, tp, sl, start_idx, end_idx)
