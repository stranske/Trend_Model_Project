#!/usr/bin/env python
"""Run the multi-period demo using the generated data.

This script exercises the Phase‑2 multi-period engine by running
``multi_period.run`` to obtain a collection of ``score_frame`` objects and then
feeding them through ``run_schedule`` with a selector and weighting scheme.
"""

from trend_analysis.config import load
import subprocess
import sys
from pathlib import Path
from trend_analysis.multi_period import (
    run as run_mp,
    run_schedule,
    scheduler,
)
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.weighting import (
    AdaptiveBayesWeighting,
    EqualWeight,
    ScorePropSimple,
    ScorePropBayesian,
)


def _check_schedule(
    score_frames,
    selector,
    weighting,
    cfg,
    *,
    rank_column=None,
):
    rebalancer = Rebalancer(cfg.model_dump())
    pf = run_schedule(
        score_frames,
        selector,
        weighting,
        rank_column=rank_column,
        rebalancer=rebalancer,
    )
    if len(pf.history) != len(score_frames):
        raise SystemExit("Weight schedule did not cover all periods")
    weights = list(pf.history.values())
    if len(weights) > 1 and all(w.equals(weights[0]) for w in weights[1:]):
        raise SystemExit("Weights did not change across periods")
    return pf


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

# ensure metadata lines up with the generated periods
for r in results:
    sf = r["score_frame"]
    p = r["period"]
    if sf.attrs.get("period") != (p[0][:7], p[1][:7]):
        raise SystemExit("Score frame period metadata mismatch")
    if sf.attrs.get("insample_len", 0) <= 0:
        raise SystemExit("Score frame contains no data")

_check_schedule(
    score_frames,
    RankSelector(top_n=3, rank_column="Sharpe"),
    AdaptiveBayesWeighting(max_w=None),
    cfg,
    rank_column="Sharpe",
)

_check_schedule(
    score_frames,
    ZScoreSelector(threshold=0.0, column="Sharpe"),
    EqualWeight(),
    cfg,
    rank_column="Sharpe",
)

_check_schedule(
    score_frames,
    RankSelector(top_n=3, rank_column="Sharpe"),
    ScorePropSimple("Sharpe"),
    cfg,
    rank_column="Sharpe",
)

_check_schedule(
    score_frames,
    RankSelector(top_n=2, rank_column="MaxDrawdown"),
    ScorePropBayesian("Sharpe"),
    cfg,
    rank_column="MaxDrawdown",
)

print("Multi-period demo checks passed")

# Run the CLI entry point to verify it behaves correctly
subprocess.run(
    [
        sys.executable,
        "-m",
        "trend_analysis.run_analysis",
        "-c",
        "config/demo.yml",
        "--detailed",
    ],
    check=True,
)

# Execute the full test suite to cover the entire code base
run_tests = Path(__file__).resolve().with_name("run_tests.sh")
subprocess.run([str(run_tests)], check=True)
