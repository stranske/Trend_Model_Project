"""
Generate a 10‑year monthly return series for 20
fake managers and dump to CSV + XLSX.
"""

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

OUT_DIR = "demo"
os.makedirs(OUT_DIR, exist_ok=True)
# Ensure the directory is writable
os.chmod(OUT_DIR, 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo return series.")
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip writing the Excel copy",
    )
    args = parser.parse_args()

    start = dt.date.today().replace(year=dt.date.today().year - 10, day=1)
    periods = 120  # 10 years * 12 months
    # Use month-end frequency; 'M' is deprecated in pandas in favor of 'ME'
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
    
    try:
        # Ensure files are writable if they exist
        if os.path.exists(csv_path):
            os.chmod(csv_path, 0o644)
        if os.path.exists(xlsx_path):
            os.chmod(xlsx_path, 0o644)
            
        df.to_csv(csv_path)
        if not args.no_xlsx:
            df.to_excel(xlsx_path)
            print(f"Wrote {csv_path} and {xlsx_path}")
        else:
            print(f"Wrote {csv_path}")
    except PermissionError as e:
        print(f"Permission error writing to {csv_path}: {e}")
        print("Make sure the demo/ directory and files are writable")
        raise
    except Exception as e:
        print(f"Error writing demo files: {e}")
        raise


if __name__ == "__main__":
    main()
