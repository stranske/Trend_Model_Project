#!/usr/bin/env python
"""
Generate extended demo data from July 2005 to June 2025 (20 years).
"""

import datetime as dt
import os

import numpy as np
import pandas as pd

OUT_DIR = "demo"
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    # Generate data from July 2005 to June 2025 (20 years = 240 months)
    start = dt.date(2005, 7, 1)
    periods = 240  # 20 years * 12 months
    dates = pd.date_range(start, periods=periods, freq="ME")

    rng = np.random.default_rng(42)
    data = {}

    # Create 20 synthetic fund managers with different characteristics
    for i in range(1, 21):
        # Base monthly returns with varying skill levels
        skill_factor = rng.uniform(0.003, 0.012)  # Some managers are better
        base_vol = rng.uniform(0.03, 0.06)  # Varying volatility
        base = rng.normal(loc=skill_factor, scale=base_vol, size=periods)

        # Add long-term trend (some managers improve/decline over time)
        trend_slope = rng.normal(scale=0.0002)
        trend = trend_slope * np.arange(periods)

        # Add regime shifts and market cycles
        regime_shifts = rng.normal(scale=0.001, size=periods).cumsum()

        # Add market beta exposure (some correlation with market)
        market_beta = rng.uniform(0.3, 1.2)

        # Create final return series
        data[f"Mgr_{i:02d}"] = base + trend + regime_shifts

    # Generate a realistic market index (SPX) with volatility clustering
    # and realistic crisis periods (2008, 2020, etc.)
    spx_base = rng.normal(loc=0.008, scale=0.04, size=periods)

    # Add crisis periods
    crisis_2008 = np.zeros(periods)
    crisis_2008[36:48] = rng.normal(loc=-0.03, scale=0.08, size=12)  # 2008-2009 crisis

    crisis_2020 = np.zeros(periods)
    crisis_2020[168:180] = rng.normal(
        loc=-0.02, scale=0.06, size=12
    )  # 2020 COVID crisis

    # Combine SPX components
    spx = spx_base + crisis_2008 + crisis_2020

    # Add market correlation to some managers
    for i, mgr_key in enumerate(data.keys()):
        if i % 3 == 0:  # Every third manager has higher market correlation
            market_beta = rng.uniform(0.6, 1.0)
            data[mgr_key] = data[mgr_key] + market_beta * spx * 0.3

    data["SPX"] = spx

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"

    # Save the extended dataset
    csv_path = f"{OUT_DIR}/extended_returns.csv"
    xlsx_path = f"{OUT_DIR}/extended_returns.xlsx"

    df.to_csv(csv_path)
    df.to_excel(xlsx_path)

    print(f"Generated extended demo data: {csv_path}")
    start_str = df.index[0].strftime("%Y-%m")
    end_str = df.index[-1].strftime("%Y-%m")
    print(f"Date range: {start_str} to {end_str}")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Managers: {len([c for c in df.columns if c.startswith('Mgr_')])}")


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    main()
