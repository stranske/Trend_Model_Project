"""Generate a 10-year monthly return series for demo managers."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trend_analysis.logging_setup import setup_logging

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo return series.")
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip writing the Excel copy",
    )
    args = parser.parse_args()

    log_path = setup_logging(app_name="generate_demo")
    logging.getLogger(__name__).info("Log file initialised at %s", log_path)

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
    data = {}
    for i in range(1, 21):
        base = rng.normal(loc=0.006, scale=0.04, size=periods)
        slope = rng.normal(scale=0.0005)
        trend = slope * np.arange(periods)
        drift = rng.normal(scale=0.002, size=periods).cumsum()
        data[f"Mgr_{i:02d}"] = base + trend + drift

    # Add a simple market index so benchmark logic can run
    spx = rng.normal(loc=0.005, scale=0.03, size=periods)
    data["SPX"] = spx

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
