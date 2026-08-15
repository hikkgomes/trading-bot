"""Product-specific allocation policies built on shared domain contracts."""

from src.products.active_income import ActiveIncomePortfolio
from src.products.btc_accumulation import BtcAllocationPolicy, BtcAllocationTarget

__all__ = ["ActiveIncomePortfolio", "BtcAllocationPolicy", "BtcAllocationTarget"]
