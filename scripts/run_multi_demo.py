#!/usr/bin/env python
"""Run the multi-period demo using the generated data.

This script exercises the Phase‑2 multi-period engine by running
``multi_period.run`` to obtain a collection of ``score_frame`` objects and then
feeding them through ``run_schedule`` with a selector and weighting scheme.
"""

from trend_analysis.config import load
from trend_analysis.multi_period import run as run_mp, run_schedule
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import AdaptiveBayesWeighting

cfg = load("config/demo.yml")
results = run_mp(cfg)
num_periods = len(results)
print(f"Generated {num_periods} period results")
if num_periods <= 1:
    raise SystemExit("Multi-period demo produced insufficient results")

score_frames = {r["period"][3]: r["score_frame"] for r in results}
selector = RankSelector(top_n=3, rank_column="Sharpe")
weighting = AdaptiveBayesWeighting(max_w=None)
portfolio = run_schedule(score_frames, selector, weighting, rank_column="Sharpe")
print(f"Weight history generated for {len(portfolio.history)} periods")
if len(portfolio.history) != num_periods:
    raise SystemExit("Weight schedule did not cover all periods")

