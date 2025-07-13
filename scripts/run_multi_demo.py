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
import pandas as pd
from trend_analysis import pipeline, export, gui, cli
from trend_analysis.multi_period import (
    run as run_mp,
    run_schedule,
    scheduler,
)
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.data import load_csv
from trend_analysis.core.rank_selection import rank_select_funds, RiskStatsConfig
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
        print("Warning: weights did not change across periods")
    return pf


def _check_gui(cfg_path: str) -> None:
    """Exercise basic GUI helpers."""
    store = gui.ParamStore.from_yaml(Path(cfg_path))
    gui.save_state(store)
    loaded = gui.load_state()
    if loaded.cfg != store.cfg:
        raise SystemExit("GUI state roundtrip failed")
    gui.reset_weight_state(loaded)
    gui.discover_plugins()
    if not gui.list_builtin_cfgs():
        raise SystemExit("list_builtin_cfgs returned no configs")


def _check_cli(cfg_path: str) -> None:
    """Exercise the simple CLI wrapper."""
    rc = cli.main(["--version", "-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI --version failed")
    rc = cli.main(["-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI default run failed")


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

# export all score frames in one go so Excel, CSV, JSON and TXT versions are
# produced for CI. Each period becomes a separate sheet/file.
mp_prefix = Path("demo/exports/multi_period_scores")
export.export_data(score_frames, str(mp_prefix), formats=["xlsx", "csv", "json", "txt"])
if not mp_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Multi-period score frame export failed")

# ensure metadata lines up with the generated periods
for r in results:
    sf = r["score_frame"]
    p = r["period"]
    if sf.attrs.get("period") != (p[0][:7], p[1][:7]):
        raise SystemExit("Score frame period metadata mismatch")
    if sf.attrs.get("insample_len", 0) <= 0:
        raise SystemExit("Score frame contains no data")

# Exercise rank_select_funds via the additional inclusion approaches
df_full = load_csv(cfg.data["csv_path"])
if df_full is None:
    raise SystemExit("Failed to load demo CSV")
mask = df_full["Date"].between(cfg.sample_split["in_start"], cfg.sample_split["in_end"])
window = df_full.loc[mask].drop(columns=["Date"])
rs_cfg = RiskStatsConfig()
top_pct_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_pct",
    pct=0.2,
    score_by="Sharpe",
)
if not top_pct_ids:
    raise SystemExit("top_pct selection produced no funds")
threshold_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="threshold",
    threshold=0.0,
    score_by="Sharpe",
)
if not threshold_ids:
    raise SystemExit("threshold selection produced no funds")

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

# Exercise the single-period pipeline and export helpers
metrics_df = pipeline.run(cfg)
if metrics_df.empty:
    raise SystemExit("pipeline.run produced empty metrics")
out_prefix = Path("demo/exports/pipeline_demo")
export.export_data(
    {"metrics": metrics_df},
    str(out_prefix),
    formats=["xlsx", "csv", "json", "txt"],
)
if not out_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Excel export failed")
if not out_prefix.with_name(f"{out_prefix.stem}_metrics.csv").exists():
    raise SystemExit("CSV export failed")
if not out_prefix.with_name(f"{out_prefix.stem}_metrics.json").exists():
    raise SystemExit("JSON export failed")
if not out_prefix.with_name(f"{out_prefix.stem}_metrics.txt").exists():
    raise SystemExit("TXT export failed")

full_res = pipeline.run_full(cfg)
sf = full_res.get("score_frame") if isinstance(full_res, dict) else None
if sf is None or sf.empty:
    raise SystemExit("pipeline.run_full missing score_frame")

# Export a formatted summary workbook and text summary
split = cfg.sample_split
sheet_fmt = export.make_summary_formatter(
    full_res,
    str(split.get("in_start")),
    str(split.get("in_end")),
    str(split.get("out_start")),
    str(split.get("out_end")),
)
summary_prefix = Path("demo/exports/summary_demo")
export.export_to_excel(
    {"metrics": metrics_df, "summary": pd.DataFrame()},
    str(summary_prefix.with_suffix(".xlsx")),
    default_sheet_formatter=sheet_fmt,
)
text = export.format_summary_text(
    full_res,
    str(split.get("in_start")),
    str(split.get("in_end")),
    str(split.get("out_start")),
    str(split.get("out_end")),
)
if "Vol-Adj Trend Analysis" not in text:
    raise SystemExit("Text summary missing header")
if not summary_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Summary Excel not created")

_check_gui("config/demo.yml")
_check_cli("config/demo.yml")

print("Multi-period demo checks passed")

# Run the CLI entry point in both modes to verify it behaves correctly
subprocess.run(
    [
        sys.executable,
        "-m",
        "trend_analysis.run_analysis",
        "-c",
        "config/demo.yml",
    ],
    check=True,
)
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
