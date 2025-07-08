#!/usr/bin/env python
"""Run the multi-period demo using the generated data."""
from trend_analysis.config import load
from trend_analysis.multi_period import run as run_mp

cfg = load("config/demo.yml")
results = run_mp(cfg)
num_periods = len(results)
print(f"Generated {num_periods} period results")
if num_periods <= 1:
    raise SystemExit("Multi-period demo produced insufficient results")
