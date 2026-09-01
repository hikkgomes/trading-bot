"""Product-specific allocation policies built on shared domain contracts."""

from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import (
    BTC_SPOT_INSTRUMENT_ID,
    BtcAllocationPolicy,
    BtcAllocationTarget,
    assert_btc_spot_instrument,
)

__all__ = [
    "ActiveIncomePortfolio",
    "BTC_SPOT_INSTRUMENT_ID",
    "BtcAllocationPolicy",
    "BtcAllocationTarget",
    "assert_btc_spot_instrument",
]
