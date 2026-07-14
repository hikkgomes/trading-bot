"""Versioned Deflated Sharpe Ratio evidence shared by research and live policy."""

from __future__ import annotations

# Version 1 records called a zero-benchmark Probabilistic Sharpe Ratio
# ``dsr_deflated`` because no cross-trial Sharpe dispersion was supplied.  V2
# is intentionally a new identity: it subtracts the expected maximum Sharpe
# across the searched trial universe using an observed, conservatively floored
# trial-Sharpe dispersion.
DSR_METHOD = "autopilot.dsr.expected_max_trial_dispersion/v2"
LIVE_MIN_DSR = 0.60

# Per-trade Sharpe is mean/std rather than annualised.  A 0.10 floor prevents a
# homogeneous or one-candidate batch from claiming that the searched trial
# distribution has zero width.  Sampling uncertainty can raise this floor.
MIN_TRIAL_SHARPE_STD = 0.10
