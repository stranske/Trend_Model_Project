"""
Central configuration for the Trend-Analysis package.
Extend or override values as the project evolves.
"""

DEFAULT_CONFIG = {
    # ── I/O ─────────────────────────────────────────────────────────────
    "output_dir": "outputs",          # where run_analysis writes results

    # ── Portfolio construction ─────────────────────────────────────────
    "rebalance_freq": "M",            # monthly
    "vol_target": 0.10,               # 10 % annualised volatility

    # ── Data requirements ──────────────────────────────────────────────
    "min_history_months": 36,         # managers need 3 yrs of returns
}
