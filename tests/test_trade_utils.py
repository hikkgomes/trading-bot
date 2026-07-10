import numpy as np
import pytest

from src.trade_utils import gross_return_for_pnl_unit, scan_tp_sl


def test_product_aware_short_returns_distinguish_linear_usdt_from_btc_step_aside():
    assert gross_return_for_pnl_unit(100.0, 90.0, is_long=False, pnl_unit="usdt") == pytest.approx(0.10)
    assert gross_return_for_pnl_unit(100.0, 90.0, is_long=False, pnl_unit="btc") == pytest.approx(1 / 9)


def test_product_aware_return_rejects_unknown_pnl_unit():
    with pytest.raises(ValueError, match="Unsupported pnl_unit"):
        gross_return_for_pnl_unit(100.0, 90.0, is_long=False, pnl_unit="usd")


def test_scan_tp_sl_long_tp():
    high = np.array([100.0, 101.5, 101.0])
    low = np.array([100.0, 99.9, 99.8])
    idx, reason = scan_tp_sl(high, low, 100.0, True, 0.01, 0.02, 0, 2)
    assert idx == 1
    assert reason == 2


def test_scan_tp_sl_short_stop():
    high = np.array([100.0, 102.0, 101.0])
    low = np.array([100.0, 99.0, 98.0])
    idx, reason = scan_tp_sl(high, low, 100.0, False, 0.01, 0.01, 0, 2)
    assert idx == 1
    assert reason == 1


def test_scan_tp_sl_time_exit():
    high = np.array([100.0, 100.2, 100.1])
    low = np.array([100.0, 99.8, 99.9])
    idx, reason = scan_tp_sl(high, low, 100.0, True, 0.01, 0.01, 0, 2)
    assert idx == 2
    assert reason == 0


def test_scan_tp_sl_same_bar_dual_hit_prefers_stop():
    high = np.array([100.0, 102.0])
    low = np.array([100.0, 98.0])
    idx, reason = scan_tp_sl(high, low, 100.0, True, 0.01, 0.01, 0, 1)
    assert idx == 1
    assert reason == 1
