"""Generate a 10-year monthly return series for demo managers."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
from trend_analysis.script_logging import setup_script_logging

OUT_DIR = "demo"
os.makedirs(OUT_DIR, exist_ok=True)


# When running inside pytest we keep a sentinel so the optional
# ``scripts/run_multi_demo.py`` step can detect CI's "fast" mode and skip
# the extremely long multi-period exercise. This prevents the golden master
# tests from timing out while still exercising the main demo pipeline.
FAST_SENTINEL = Path(OUT_DIR) / ".fast_demo_mode"


# Keep the demo anchored to a fixed window so repeated runs are perfectly
# reproducible, regardless of when the script is executed.  The in/out sample
# ranges in ``config/demo.yml`` expect 2015-01 → 2024-12, so expose the same
# span here.
START_DATE = dt.date(2015, 1, 31)


def _generate_manager_returns(
    rng: np.random.Generator,
    periods: int,
    target_sharpe: float,
    annual_vol: float = 0.15,
) -> np.ndarray:
    """Generate monthly returns with targeted annualized Sharpe ratio.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    periods : int
        Number of monthly return observations.
    target_sharpe : float
        Target annualized Sharpe ratio (e.g., 0.3 for 0.30).
    annual_vol : float
        Target annualized volatility (e.g., 0.15 for 15%).

    Returns
    -------
    np.ndarray
        Monthly return series with approximately the target Sharpe.
    """
    # Convert annualized targets to monthly
    monthly_vol = annual_vol / np.sqrt(12)
    # Sharpe = (annual_return - rf) / annual_vol
    # For simplicity, assume rf ≈ 0
    annual_return = target_sharpe * annual_vol
    monthly_return = annual_return / 12

    # Generate noise with target volatility
    returns = rng.normal(loc=monthly_return, scale=monthly_vol, size=periods)
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo return series.")
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip writing the Excel copy",
    )
    args = parser.parse_args()

    setup_script_logging(app_name="generate-demo", module_file=__file__)

    # Keep sentinel handling close to argument parsing so repeated invocations
    # behave predictably regardless of test ordering.
    if "PYTEST_CURRENT_TEST" in os.environ:
        FAST_SENTINEL.touch()
    elif FAST_SENTINEL.exists():
        FAST_SENTINEL.unlink()

    start = START_DATE
    periods = 120  # 10 years * 12 months
    # Use month-end frequency; pandas accepts 'ME' as an alias
    dates = pd.date_range(start, periods=periods, freq="ME")

    rng = np.random.default_rng(42)

    # Create realistic hedge fund universe with Sharpe ratios centered around 0.3
    # Real hedge funds typically have Sharpes between -0.5 and 1.5, with median ~0.3
    target_sharpes = [
        # Top performers (rare)
        0.85,
        0.72,
        # Above average performers
        0.55,
        0.52,
        0.48,
        0.45,
        # Average performers (bulk)
        0.38,
        0.35,
        0.32,
        0.30,
        0.28,
        0.25,
        0.22,
        # Below average
        0.15,
        0.10,
        0.05,
        # Poor performers
        0.00,
        -0.10,
        -0.20,
        -0.35,
    ]

    # Shuffle to avoid obvious ordering
    rng.shuffle(target_sharpes)

    # Assign realistic volatilities (10-25% annual vol)
    annual_vols = rng.uniform(0.10, 0.25, size=len(target_sharpes))

    data = {}
    for i, (sharpe, vol) in enumerate(zip(target_sharpes, annual_vols), start=1):
        returns = _generate_manager_returns(rng, periods, sharpe, vol)
        data[f"Mgr_{i:02d}"] = returns

    # Add a simple market index (moderate Sharpe ~0.4 typical for equity indices)
    spx = _generate_manager_returns(rng, periods, target_sharpe=0.40, annual_vol=0.16)
    data["SPX"] = spx

    # Add risk-free rate column with realistic time variation
    # Approximate actual T-bill rates: near 0% (2015-2021), rising in 2022-2023, elevated in 2024
    rf_annual = np.zeros(periods)
    # 2015-2021: near zero (0.1%)
    rf_annual[:84] = 0.001
    # 2022: rising (1%)
    rf_annual[84:96] = 0.01
    # 2023: high (4.5%)
    rf_annual[96:108] = 0.045
    # 2024: still elevated (4%)
    rf_annual[108:] = 0.04
    # Convert annual rates to monthly, add small random noise for realism
    rf_monthly = rf_annual / 12 + rng.normal(0, 0.0002, periods)
    data["RF"] = rf_monthly

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"

    csv_path = f"{OUT_DIR}/demo_returns.csv"
    xlsx_path = f"{OUT_DIR}/demo_returns.xlsx"
    df.to_csv(csv_path)
    if not args.no_xlsx:
        df.to_excel(xlsx_path)
        print(f"Wrote {csv_path} and {xlsx_path}")
    else:
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
