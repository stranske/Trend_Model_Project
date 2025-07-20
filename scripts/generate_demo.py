"""
Generate a 10‑year monthly return series for 20
fake managers and dump to CSV + XLSX.
"""

import numpy as np
import pandas as pd
import os
import datetime as dt

OUT_DIR = "demo"
os.makedirs(OUT_DIR, exist_ok=True)

start = dt.date.today().replace(year=dt.date.today().year - 10, day=1)
periods = 120  # 10 years * 12 months
dates = pd.date_range(start, periods=periods, freq="M")

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
df.to_excel(xlsx_path)
print(f"Wrote {csv_path} and {xlsx_path}")
