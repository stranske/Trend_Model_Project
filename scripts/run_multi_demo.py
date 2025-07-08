#!/usr/bin/env python
"""Run the multi-period demo using the generated data."""
from trend_analysis.config import load
from trend_analysis.multi_period import run as run_mp

cfg = load("config/demo.yml")
results = run_mp(cfg)
print(f"Generated {len(results)} period results")
if not results:
    raise SystemExit("Multi-period demo produced no results")
