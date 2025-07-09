#!/usr/bin/env python
"""Run the multi-period demo using the generated data.

This script exercises the Phase‑2 multi-period engine by running
``multi_period.run`` to obtain a collection of ``score_frame`` objects and then
feeding them through ``run_schedule`` with a selector and weighting scheme.
"""

from trend_analysis.config import load
from trend_analysis.multi_period import (
    run as run_mp,
    run_schedule,
    scheduler,
)
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import AdaptiveBayesWeighting

cfg = load("config/demo.yml")
results = run_mp(cfg)
num_periods = len(results)
periods = scheduler.generate_periods(cfg.model_dump())
expected_periods = len(periods)
print(f"Generated {num_periods} period results (expected {expected_periods})")
if num_periods != expected_periods:
    raise SystemExit("Multi-period demo produced an unexpected number of periods")
if num_periods <= 1:
    raise SystemExit("Multi-period demo produced insufficient results")

# check that the generated periods line up with the scheduler output
result_periods = [r["period"] for r in results]
sched_tuples = [(p.in_start, p.in_end, p.out_start, p.out_end) for p in periods]
if result_periods != sched_tuples:
    raise SystemExit("Period sequence mismatch")

score_frames = {r["period"][3]: r["score_frame"] for r in results}
selector = RankSelector(top_n=3, rank_column="Sharpe")
weighting = AdaptiveBayesWeighting(max_w=None)
rebalancer = Rebalancer(cfg.model_dump())
portfolio = run_schedule(
    score_frames,
    selector,
    weighting,
    rank_column="Sharpe",
    rebalancer=rebalancer,
)
print(f"Weight history generated for {len(portfolio.history)} periods")
if len(portfolio.history) != num_periods:
    raise SystemExit("Weight schedule did not cover all periods")

# ensure metadata lines up with the generated periods
for r in results:
    sf = r["score_frame"]
    p = r["period"]
    if sf.attrs.get("period") != (p[0][:7], p[1][:7]):
        raise SystemExit("Score frame period metadata mismatch")
    if sf.attrs.get("insample_len", 0) <= 0:
        raise SystemExit("Score frame contains no data")

weights = list(portfolio.history.values())
if len(weights) > 1 and all(w.equals(weights[0]) for w in weights[1:]):
    raise SystemExit("Weights did not change across periods")

fund_sets = [set(w.index) for w in portfolio.history.values()]
if len(fund_sets) > 1 and all(fund_sets[0] == fs for fs in fund_sets[1:]):
    raise SystemExit("Fund selection identical across periods")

print("Multi-period demo checks passed")
