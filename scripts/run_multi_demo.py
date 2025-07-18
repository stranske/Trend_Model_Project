#!/usr/bin/env python
"""Run the multi-period demo using the generated data.

This script exercises the Phase‑2 multi-period engine by running
``multi_period.run`` to obtain a collection of ``score_frame`` objects and then
feeding them through ``run_schedule`` with a selector and weighting scheme.
"""

from trend_analysis.config import load, Config
import subprocess
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
from trend_analysis import (
    pipeline,
    export,
    gui,
    cli,
    metrics,
    run_analysis,
    run_multi_analysis,
)
from trend_analysis.multi_period import (
    run as run_mp,
    run_schedule,
    scheduler,
)
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.selector import RankSelector, ZScoreSelector
from trend_analysis.data import load_csv, identify_risk_free_fund, ensure_datetime
from trend_analysis.core.rank_selection import rank_select_funds, RiskStatsConfig
from trend_analysis.core import rank_selection as rs
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
    if hasattr(weighting, "mean") and hasattr(weighting, "tau"):
        if getattr(weighting, "mean") is None or getattr(weighting, "tau") is None:
            raise SystemExit("Weighting state not updated")
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

    class _DummyPlugin:
        pass

    gui.register_plugin(_DummyPlugin)
    if _DummyPlugin not in list(gui.iter_plugins()):
        raise SystemExit("Plugin registration failed")
    if "demo" not in gui.list_builtin_cfgs():
        raise SystemExit("list_builtin_cfgs missing demo.yml")
    from trend_analysis.core.rank_selection import build_ui
    import ipywidgets as widgets

    ui = build_ui()
    if not isinstance(ui, widgets.Widget):
        raise SystemExit("build_ui did not return a Widget")


def _check_selection_modes(cfg: Config) -> None:
    """Verify legacy selection modes still operate."""
    base = cfg.model_dump()
    for mode in ("all", "random", "manual"):
        cfg_copy = Config(**base)
        cfg_copy.portfolio["selection_mode"] = mode
        if mode == "random":
            cfg_copy.portfolio["random_n"] = 4
        if mode == "manual":
            cfg_copy.portfolio["manual_list"] = ["Mgr_01", "Mgr_02"]
        res = pipeline.run_full(cfg_copy)
        sel = res.get("selected_funds")
        if not sel:
            raise SystemExit(f"{mode} mode produced no funds")


def _check_cli_env(cfg_path: str) -> None:
    """Invoke the CLI using the TREND_CFG environment variable."""
    env = os.environ.copy()
    env["TREND_CFG"] = cfg_path
    subprocess.run(
        [sys.executable, "-m", "trend_analysis.run_analysis", "--detailed"],
        check=True,
        env=env,
    )


def _check_cli(cfg_path: str) -> None:
    """Exercise the simple CLI wrapper."""
    rc = cli.main(["--version", "-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI --version failed")
    rc = cli.main(["-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI default run failed")


def _check_misc(cfg_path: str, cfg: Config, results) -> None:
    """Exercise smaller utility modules."""
    from trend_analysis import metrics
    import asyncio

    if "annual_return" not in metrics.available_metrics():
        raise SystemExit("Metrics registry incomplete")

    # scheduler.generate_periods should agree with the results length
    periods = scheduler.generate_periods(cfg.model_dump())
    if len(periods) != len(results):
        raise SystemExit("Scheduler period count mismatch")

    store = gui.ParamStore.from_yaml(Path(cfg_path))
    if gui.build_config_from_store(store).version != cfg.version:
        raise SystemExit("Config build roundtrip failed")

    called: list[int] = []

    @gui.debounce(50)
    async def ping(val: int) -> None:
        called.append(val)

    async def drive() -> None:
        await ping(1)
        await ping(2)
        await asyncio.sleep(0.1)

    asyncio.run(drive())
    if called != [2]:
        raise SystemExit("debounce failed")

    os.environ["TREND_CFG"] = cfg_path
    cfg_env = load(None)
    os.environ.pop("TREND_CFG", None)
    if cfg_env.version != cfg.version:
        raise SystemExit("TREND_CFG not honoured")

    df_demo = load_csv(cfg.data["csv_path"])
    df_demo = ensure_datetime(df_demo)
    sf_single = pipeline.single_period_run(
        df_demo[["Date", "Mgr_01", "Mgr_02"]],
        str(cfg.sample_split["in_start"]),
        str(cfg.sample_split["in_end"]),
    )
    if sf_single.attrs.get("period") != (
        str(cfg.sample_split["in_start"]),
        str(cfg.sample_split["in_end"]),
    ):
        raise SystemExit("single_period_run metadata mismatch")

    pr = pipeline.calc_portfolio_returns(
        np.array([0.5, 0.5]),
        df_demo[["Mgr_01", "Mgr_02"]].iloc[:4],
    )
    if len(pr) != 4:
        raise SystemExit("calc_portfolio_returns length mismatch")

    # dynamic metric registration
    @metrics.register_metric("dummy_metric")
    def _dummy(series):
        return float(series.mean())

    if "dummy_metric" not in metrics.available_metrics():
        raise SystemExit("Metric registration failed")
    metrics._METRIC_REGISTRY.pop("dummy_metric", None)

    clist = rs.canonical_metric_list(["sharpe_ratio", "max_drawdown"])
    if clist != ["Sharpe", "MaxDrawdown"]:
        raise SystemExit("canonical_metric_list failed")

    # direct calls to helper functions for coverage
    series = pd.Series([1.0, 2.0, 3.0])
    ranked = rs._apply_transform(series, mode="rank")
    if ranked.iloc[0] != 3:
        raise SystemExit("_apply_transform failed")

    df_tmp = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})
    scores = rs._compute_metric_series(df_tmp, "AnnualReturn", RiskStatsConfig())
    if len(scores) != 2:
        raise SystemExit("_compute_metric_series failed")


def _check_rebalancer_logic() -> None:
    """Verify Rebalancer triggers drop and add events."""
    reb = Rebalancer({})
    prev = pd.Series({"A": 0.5, "B": 0.5})
    sf1 = pd.DataFrame({"zscore": [0.0, -1.1]}, index=["A", "B"])
    w1 = reb.apply_triggers(prev, sf1)
    sf2 = pd.DataFrame({"zscore": [0.0, -1.2]}, index=["A", "B"])
    w2 = reb.apply_triggers(w1, sf2)
    sf3 = pd.DataFrame({"zscore": [0.0, 1.2]}, index=["A", "C"])
    w3 = reb.apply_triggers(w2, sf3)
    if "B" in w3.index or "C" not in w3.index or not np.isclose(w3.sum(), 1.0):
        raise SystemExit("Rebalancer logic failed")


def _check_load_csv_error() -> None:
    """Ensure ``load_csv`` returns ``None`` for missing files."""
    if load_csv("_no_such_file_.csv") is not None:
        raise SystemExit("load_csv error handling failed")


def _check_metrics_basic() -> None:
    """Run a couple of metrics on a short series."""
    s = pd.Series([0.0, 0.01, -0.02])
    if not isinstance(metrics.sharpe_ratio(s), float):
        raise SystemExit("metrics.sharpe_ratio failed")


cfg = load("config/demo.yml")
if cfg.export.get("filename") != "alias_demo.csv":
    raise SystemExit("Output alias not parsed")
results = run_mp(cfg)
num_periods = len(results)
periods = scheduler.generate_periods(cfg.model_dump())
expected_periods = len(periods)
print(f"Generated {num_periods} period results (expected {expected_periods})")
if num_periods != expected_periods:
    raise SystemExit("Multi-period demo produced an unexpected number of periods")
if num_periods <= 1:
    raise SystemExit("Multi-period demo produced insufficient results")

# Ensure run_mp works with a pre-loaded DataFrame
df_pre = load_csv(cfg.data["csv_path"])
results_pre = run_mp(cfg, df_pre)
if len(results_pre) != num_periods:
    raise SystemExit("Preloaded DataFrame run mismatch")

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

# Exercise multi-period export helpers
frames = export.workbook_frames_from_results(results)
if "summary" not in frames:
    raise SystemExit("workbook_frames_from_results missing summary")
phase1_prefix = Path("demo/exports/phase1_multi")
export.export_phase1_multi_metrics(
    results,
    str(phase1_prefix),
    formats=["xlsx", "csv", "json"],
    include_metrics=True,
)
if not phase1_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Phase1 multi metrics export failed")
mpm_prefix = Path("demo/exports/multi_period_metrics")
export.export_multi_period_metrics(
    results,
    str(mpm_prefix),
    formats=["xlsx", "csv", "json", "txt"],
    include_metrics=True,
)
if not mpm_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Multi-period metrics export failed")
summary = export.combined_summary_result(results)
summary_frame = export.summary_frame_from_result(summary)
metrics_frame = export.metrics_from_result(summary)
if summary_frame.empty or metrics_frame.empty:
    raise SystemExit("Summary export helpers failed")

# Exercise rank_select_funds via the additional inclusion approaches
df_full = load_csv(cfg.data["csv_path"])
if df_full is None:
    raise SystemExit("Failed to load demo CSV")
df_full = ensure_datetime(df_full)
rf_col = identify_risk_free_fund(df_full)
if rf_col is None:
    raise SystemExit("identify_risk_free_fund failed")
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
blended_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_n",
    n=3,
    score_by="blended",
    blended_weights={"Sharpe": 0.6, "AnnualReturn": 0.3, "MaxDrawdown": 0.1},
)
if not blended_ids:
    raise SystemExit("blended selection produced no funds")
zscore_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_n",
    n=2,
    score_by="Sharpe",
    transform="zscore",
)
if not zscore_ids:
    raise SystemExit("zscore selection produced no funds")

percentile_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_pct",
    pct=0.3,
    score_by="Sharpe",
    transform="percentile",
    rank_pct=0.5,
)
if not percentile_ids:
    raise SystemExit("percentile transform produced no funds")

rank_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_n",
    n=3,
    score_by="Sharpe",
    transform="rank",
)
if not rank_ids:
    raise SystemExit("rank transform produced no funds")

# quality_filter and select_funds interfaces
qcfg = rs.FundSelectionConfig(max_missing_ratio=0.5)
eligible = rs.quality_filter(df_full, qcfg)
if not eligible or not set(eligible).issubset(df_full.columns):
    raise SystemExit("quality_filter failed")

simple_sel = rs.select_funds(df_full, rf_col, mode="random", n=2)
if len(simple_sel) != 2:
    raise SystemExit("select_funds simple mode failed")

cols = [c for c in df_full.columns if c not in {"Date", rf_col}]
ext_sel = rs.select_funds(
    df_full,
    rf_col,
    cols,
    str(cfg.sample_split["in_start"]),
    str(cfg.sample_split["in_end"]),
    str(cfg.sample_split["out_start"]),
    str(cfg.sample_split["out_end"]),
    qcfg,
    "rank",
    2,
    {"inclusion_approach": "top_n", "n": 2, "score_by": "Sharpe"},
)
if len(ext_sel) != 2:
    raise SystemExit("select_funds extended mode failed")

abw = AdaptiveBayesWeighting(max_w=None)
pf_abw = _check_schedule(
    score_frames,
    RankSelector(top_n=3, rank_column="Sharpe"),
    abw,
    cfg,
    rank_column="Sharpe",
)
abw_prefix = Path("demo/exports/abw_weights")
export.export_data(
    {"weights": pd.DataFrame(pf_abw.history).T},
    str(abw_prefix),
    formats=["xlsx", "csv", "json", "txt"],
)
if not abw_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("ABW weight export failed")
state = abw.get_state()
abw2 = AdaptiveBayesWeighting(max_w=None)
abw2.set_state(state)
if abw2.get_state() != state:
    raise SystemExit("AdaptiveBayesWeighting state mismatch")

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

# Reuse the sample split for the convenience wrapper
split = cfg.sample_split
# Exercise the convenience wrapper around ``_run_analysis``
analysis_res = pipeline.run_analysis(
    df_full,
    str(split.get("in_start")),
    str(split.get("in_end")),
    str(split.get("out_start")),
    str(split.get("out_end")),
    cfg.vol_adjust.get("target_vol", 1.0),
    getattr(cfg, "run", {}).get("monthly_cost", 0.0),
    selection_mode="rank",
    rank_kwargs={"n": 5, "score_by": "Sharpe", "inclusion_approach": "top_n"},
)
if analysis_res is None or analysis_res.get("score_frame") is None:
    raise SystemExit("pipeline.run_analysis failed")

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

# Exercise formatter registry helpers
export.reset_formatters_excel()


@export.register_formatter_excel("dummy")
def _demo_fmt(ws, wb):
    ws.write(0, 0, "demo")


dummy_prefix = Path("demo/exports/dummy")
export.export_to_excel(
    {"dummy": pd.DataFrame({"A": [1, 2]})}, str(dummy_prefix.with_suffix(".xlsx"))
)
if not dummy_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Custom Excel export failed")

_check_gui("config/demo.yml")
_check_selection_modes(cfg)
_check_cli_env("config/demo.yml")
_check_cli("config/demo.yml")
_check_misc("config/demo.yml", cfg, results)
_check_rebalancer_logic()
_check_load_csv_error()
_check_metrics_basic()

# run_analysis.main directly
if run_analysis.main(["-c", "config/demo.yml"]) != 0:
    raise SystemExit("run_analysis.main failed")
if run_analysis.main(["-c", "config/demo.yml", "--detailed"]) != 0:
    raise SystemExit("run_analysis.main detailed failed")

# Run the multi-period CLI using a temporary config file
cli_cfg = Path("demo/exports/mp_cli_cfg.yml")
cli_out = Path("demo/exports/mp_cli")
data = cfg.model_dump()
data.setdefault("export", {})["directory"] = str(cli_out)
data["export"]["formats"] = ["csv"]
with cli_cfg.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh)
rc = run_multi_analysis.main(["-c", str(cli_cfg)])
if rc != 0:
    raise SystemExit("run_multi_analysis CLI failed")
if not list(cli_out.glob("*.csv")):
    raise SystemExit("run_multi_analysis CLI produced no output")
cli_cfg.unlink()


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
