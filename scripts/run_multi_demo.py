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
import shutil
import pandas as pd
import numpy as np
import yaml  # type: ignore[import-untyped]
import openpyxl
from trend_analysis import (
    pipeline,
    export,
    gui,
    cli,
    metrics,
    run_analysis,
    run_multi_analysis,
)
import trend_analysis as ta
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
    cfg_dict = gui.build_config_dict(store)
    cfg_obj = gui.build_config_from_store(store)
    dumped = cfg_obj.model_dump()
    for k, v in cfg_dict.items():
        if dumped.get(k) != v:
            raise SystemExit("build_config_from_store mismatch")

    class _DummyPlugin:
        pass

    gui.register_plugin(_DummyPlugin)
    if _DummyPlugin not in list(gui.iter_plugins()):
        raise SystemExit("Plugin registration failed")
    if "demo" not in gui.list_builtin_cfgs():
        raise SystemExit("list_builtin_cfgs missing demo.yml")
    from trend_analysis.core.rank_selection import build_ui
    from trend_analysis.gui import app as gui_app
    import ipywidgets as widgets

    ui = build_ui()
    if not isinstance(ui, widgets.Widget):
        raise SystemExit("build_ui did not return a Widget")

    # exercise more GUI construction helpers
    man = gui_app._build_manual_override(store)  # type: ignore[attr-defined]
    weight = gui_app._build_weighting_options(store)  # type: ignore[attr-defined]
    step0 = gui_app._build_step0(store)  # type: ignore[attr-defined]
    rank = gui_app._build_rank_options(store)  # type: ignore[attr-defined]
    for widget in (man, weight, step0, rank):
        if not isinstance(widget, widgets.Widget):
            raise SystemExit("GUI builder did not return a Widget")

    # exercise weight state persistence
    store.weight_state = {"dummy": [1, 2, 3]}
    gui.save_state(store)
    loaded_ws = gui.load_state()
    if loaded_ws.weight_state != store.weight_state:
        raise SystemExit("weight_state roundtrip failed")

    # verify launch() returns a Widget
    app = gui.launch()
    if not isinstance(app, widgets.Widget):
        raise SystemExit("launch() did not return a Widget")


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
    os.environ["TREND_CFG"] = cfg_path
    rc = run_analysis.main([])
    os.environ.pop("TREND_CFG", None)
    if rc != 0:
        raise SystemExit("run_analysis.main env failed")


def _check_cli_env_multi(cfg_path: str) -> None:
    """Invoke the multi-period CLI using the TREND_CFG variable."""
    env = os.environ.copy()
    env["TREND_CFG"] = cfg_path
    subprocess.run(
        [sys.executable, "-m", "trend_analysis.run_multi_analysis", "--detailed"],
        check=True,
        env=env,
    )
    os.environ["TREND_CFG"] = cfg_path
    rc = run_multi_analysis.main([])
    os.environ.pop("TREND_CFG", None)
    if rc != 0:
        raise SystemExit("run_multi_analysis.main env failed")


def _check_cli(cfg_path: str) -> None:
    """Exercise the simple CLI wrapper."""
    rc = cli.main(["--version", "-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI --version failed")
    rc = cli.main(["-c", cfg_path])
    if rc != 0:
        raise SystemExit("CLI default run failed")
    rc = cli.main([])
    if rc != 0:
        raise SystemExit("CLI default config failed")


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
    if cfg_env.jobs != 1 or cfg_env.checkpoint_dir != "demo/checkpoints":
        raise SystemExit("Config optional fields not parsed")

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

    # verify pipeline alias handling
    stats_cls = getattr(pipeline, "Stats")
    if stats_cls is not pipeline._Stats:
        raise SystemExit("pipeline.Stats alias failed")


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
    """Run basic metrics on a short series."""
    s = pd.Series([0.0, 0.01, -0.02])
    bench = pd.Series([0.0, 0.005, -0.01])

    if not isinstance(metrics.annual_return(s), float):
        raise SystemExit("metrics.annual_return failed")
    if not isinstance(metrics.volatility(s), float):
        raise SystemExit("metrics.volatility failed")
    if not isinstance(metrics.sharpe_ratio(s), float):
        raise SystemExit("metrics.sharpe_ratio failed")
    if not isinstance(metrics.sortino_ratio(s), float):
        raise SystemExit("metrics.sortino_ratio failed")
    if not isinstance(metrics.information_ratio(s, bench), float):
        raise SystemExit("metrics.information_ratio failed")
    if not isinstance(metrics.max_drawdown(s), float):
        raise SystemExit("metrics.max_drawdown failed")
    # legacy aliases
    if metrics.annualize_return(s) != metrics.annual_return(s):
        raise SystemExit("annualize_return mismatch")
    if metrics.annualize_volatility(s) != metrics.volatility(s):
        raise SystemExit("annualize_volatility mismatch")
    if metrics.info_ratio(s, bench) != metrics.information_ratio(s, bench):
        raise SystemExit("info_ratio mismatch")


def _check_builtin_metric_aliases() -> None:
    """Ensure legacy metrics are accessible via builtins."""
    import builtins
    import importlib

    legacy = importlib.import_module("tests.legacy_metrics")
    s = pd.Series([0.0, 0.02, -0.01])

    if not hasattr(builtins, "annualize_return"):
        raise SystemExit("builtins missing annualize_return")
    if not hasattr(builtins, "annualize_volatility"):
        raise SystemExit("builtins missing annualize_volatility")

    if builtins.annualize_return(s) != legacy.annualize_return(s):
        raise SystemExit("builtins annualize_return mismatch")
    if builtins.annualize_volatility(s) != legacy.annualize_volatility(s):
        raise SystemExit("builtins annualize_volatility mismatch")


def _check_selector_errors() -> None:
    """Ensure selectors raise ``KeyError`` for missing columns."""
    df = pd.DataFrame({"A": [1, 2, 3]})

    try:
        RankSelector(1, "B").select(df)
    except KeyError:
        pass
    else:
        raise SystemExit("RankSelector missing-column check failed")

    try:
        ZScoreSelector(0.0).select(df)
    except KeyError:
        pass
    else:
        raise SystemExit("ZScoreSelector missing-column check failed")


def _check_weighting_errors() -> None:
    """Ensure weighting classes validate input columns."""
    df = pd.DataFrame({"metric": [0.1, 0.2]}, index=["A", "B"])

    for cls in (ScorePropSimple, ScorePropBayesian):
        try:
            cls("other").weight(df)
        except KeyError:
            continue
        raise SystemExit(f"{cls.__name__} missing-column check failed")


def _check_notebook_utils() -> None:
    """Exercise notebook helper scripts."""
    src = Path("Vol_Adj_Trend_Analysis1.5.TrEx.ipynb")
    if not src.exists():
        return
    tmp = Path("demo/exports/strip_tmp.ipynb")
    shutil.copy(src, tmp)
    subprocess.run([sys.executable, "tools/strip_output.py", str(tmp)], check=True)  # type: ignore
    data = tmp.read_text(encoding="utf-8")
    if '"outputs": []' not in data:
        raise SystemExit("strip_output failed")
    tmp.unlink()
    subprocess.run(["sh", "tools/pre-commit"], check=True)  # type: ignore


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
results_prefix = Path("demo/exports/workbook_frames")
export.export_data(frames, str(results_prefix), formats=["xlsx", "csv", "json", "txt"])
if not results_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Workbook frame export failed")
csv_files = list(results_prefix.parent.glob(f"{results_prefix.stem}_*.csv"))
json_files = list(results_prefix.parent.glob(f"{results_prefix.stem}_*.json"))
txt_files = list(results_prefix.parent.glob(f"{results_prefix.stem}_*.txt"))
if not csv_files or not json_files or not txt_files:
    raise SystemExit("Workbook frame CSV/JSON/TXT missing")
phase1_prefix = Path("demo/exports/phase1_multi")
export.export_phase1_multi_metrics(
    results,
    str(phase1_prefix),
    formats=["xlsx", "csv", "json"],
    include_metrics=True,
)
if not phase1_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Phase1 multi metrics export failed")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_periods.csv").exists():
    raise SystemExit("Phase1 multi metrics CSV missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_periods.json").exists():
    raise SystemExit("Phase1 multi metrics JSON missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_summary.csv").exists():
    raise SystemExit("Phase1 multi metrics summary CSV missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_summary.json").exists():
    raise SystemExit("Phase1 multi metrics summary JSON missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_metrics.csv").exists():
    raise SystemExit("Phase1 multi metrics metrics CSV missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_metrics.json").exists():
    raise SystemExit("Phase1 multi metrics metrics JSON missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_metrics_summary.csv").exists():
    raise SystemExit("Phase1 multi metrics metrics summary CSV missing")
if not phase1_prefix.with_name(f"{phase1_prefix.stem}_metrics_summary.json").exists():
    raise SystemExit("Phase1 multi metrics metrics summary JSON missing")
mpm_prefix = Path("demo/exports/multi_period_metrics")
export.export_multi_period_metrics(
    results,
    str(mpm_prefix),
    formats=["xlsx", "csv", "json", "txt"],
    include_metrics=True,
)
if not mpm_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("Multi-period metrics export failed")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_periods.csv").exists():
    raise SystemExit("Multi-period metrics CSV missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_periods.json").exists():
    raise SystemExit("Multi-period metrics JSON missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_periods.txt").exists():
    raise SystemExit("Multi-period metrics TXT missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_summary.csv").exists():
    raise SystemExit("Multi-period metrics summary CSV missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_summary.json").exists():
    raise SystemExit("Multi-period metrics summary JSON missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_summary.txt").exists():
    raise SystemExit("Multi-period metrics summary TXT missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics.csv").exists():
    raise SystemExit("Multi-period metrics metrics CSV missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics.json").exists():
    raise SystemExit("Multi-period metrics metrics JSON missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics.txt").exists():
    raise SystemExit("Multi-period metrics metrics TXT missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics_summary.csv").exists():
    raise SystemExit("Multi-period metrics metrics summary CSV missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics_summary.json").exists():
    raise SystemExit("Multi-period metrics metrics summary JSON missing")
if not mpm_prefix.with_name(f"{mpm_prefix.stem}_metrics_summary.txt").exists():
    raise SystemExit("Multi-period metrics metrics summary TXT missing")
wb_direct = Path("demo/exports/phase1_direct.xlsx")
export.export_phase1_workbook(results, str(wb_direct))
if not wb_direct.exists():
    raise SystemExit("export_phase1_workbook failed")
wb = openpyxl.load_workbook(wb_direct)
expected_sheets = {str(r["period"][3]) for r in results}
expected_sheets.add("summary")
if set(wb.sheetnames) != expected_sheets:
    raise SystemExit("phase1 workbook sheets mismatch")
pf_frames = export.period_frames_from_results(results)
if len(pf_frames) != len(results):
    raise SystemExit("period_frames_from_results count mismatch")
if "summary" in pf_frames:
    raise SystemExit("period_frames_from_results should not include summary")
pf_prefix = Path("demo/exports/period_frames")
export.export_data(pf_frames, str(pf_prefix), formats=["xlsx", "csv", "json", "txt"])
if not pf_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("period frame Excel missing")
created = list(pf_prefix.parent.glob(f"{pf_prefix.stem}_*.csv"))
if not created:
    raise SystemExit("period frame CSV/JSON/TXT missing")
summary = export.combined_summary_result(results)
summary_frame = export.summary_frame_from_result(summary)
metrics_frame = export.metrics_from_result(summary)
if summary_frame.empty or metrics_frame.empty:
    raise SystemExit("Summary export helpers failed")
summ_prefix = Path("demo/exports/summary_frames")
export.export_data(
    {"summary": summary_frame, "metrics": metrics_frame},
    str(summ_prefix),
    formats=["xlsx", "csv", "json", "txt"],
)
if not summ_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("summary frame Excel missing")

# Export per-period metrics using all exporters to cover the helper
metrics_prefix = Path("demo/exports/period_metrics")
period_metrics = {
    str(res["period"][3]): export.metrics_from_result(res) for res in results
}
export.export_data(
    period_metrics, str(metrics_prefix), formats=["xlsx", "csv", "json", "txt"]
)
if not metrics_prefix.with_suffix(".xlsx").exists():
    raise SystemExit("period metrics Excel missing")
created = list(metrics_prefix.parent.glob(f"{metrics_prefix.stem}_*.csv"))
if not created:
    raise SystemExit("period metrics CSV/JSON/TXT missing")

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

alias_ids = rank_select_funds(
    window,
    rs_cfg,
    inclusion_approach="top_n",
    n=2,
    score_by="Sharpe",
    transform_mode="zscore",
)
if not alias_ids:
    raise SystemExit("transform_mode alias failed")

df_tmp = pd.DataFrame({"Time": ["2020-01-01", "2020-02-01"], "Val": [0.0, 0.1]})
dt_tmp = ensure_datetime(df_tmp, "Time")
if not pd.api.types.is_datetime64_any_dtype(dt_tmp["Time"]):
    raise SystemExit("ensure_datetime custom column failed")
ew_df = EqualWeight().weight(
    pd.DataFrame({"metric": [1.0, 2.0, 3.0]}, index=["A", "B", "C"])
)
if not np.isclose(ew_df["weight"].sum(), 1.0, atol=1e-4):
    raise SystemExit("EqualWeight weight sum mismatch")

direct_res = pipeline._run_analysis(
    df_full,
    str(cfg.sample_split["in_start"]),
    str(cfg.sample_split["in_end"]),
    str(cfg.sample_split["out_start"]),
    str(cfg.sample_split["out_end"]),
    cfg.vol_adjust.get("target_vol", 1.0),
    getattr(cfg, "run", {}).get("monthly_cost", 0.0),
    selection_mode="rank",
    rank_kwargs={"inclusion_approach": "top_n", "n": 2, "score_by": "Sharpe"},
)
if direct_res is None or "score_frame" not in direct_res:
    raise SystemExit("_run_analysis direct call failed")

# cover helper with missing type annotations
scores = rs._compute_metric_series(window, "Sharpe", rs_cfg)
extra_ids = rs.some_function_missing_annotation(
    scores,
    "top_n",
    n=2,
    ascending=False,
)
if len(extra_ids) != 2:
    raise SystemExit("some_function_missing_annotation failed")

# quality_filter and select_funds interfaces
qcfg = rs.FundSelectionConfig(max_missing_ratio=0.5)
eligible = rs.quality_filter(df_full, qcfg)
if not eligible or not set(eligible).issubset(df_full.columns):
    raise SystemExit("quality_filter failed")

filtered = rs._quality_filter(
    df_full,
    [c for c in df_full.columns if c not in {"Date", rf_col}],
    str(cfg.sample_split["in_start"]),
    str(cfg.sample_split["out_end"]),
    qcfg,
)
if not filtered:
    raise SystemExit("_quality_filter returned no funds")

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
    rank_kwargs={"inclusion_approach": "top_n", "n": 2, "score_by": "Sharpe"},
)
if len(ext_sel) != 2:
    raise SystemExit("select_funds extended mode failed")

ext_sel_direct = rs.select_funds_extended(
    df_full,
    rf_col,
    cols,
    str(cfg.sample_split["in_start"]),
    str(cfg.sample_split["in_end"]),
    str(cfg.sample_split["out_start"]),
    str(cfg.sample_split["out_end"]),
    qcfg,
    selection_mode="rank",
    rank_kwargs={"inclusion_approach": "top_n", "n": 2, "score_by": "Sharpe"},
)
if len(ext_sel_direct) != 2:
    raise SystemExit("select_funds_extended direct call failed")

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
for ext in ("csv", "json", "txt"):
    if not abw_prefix.with_name(f"{abw_prefix.stem}_weights.{ext}").exists():
        raise SystemExit(f"ABW weight {ext} missing")
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
analysis_idx = pipeline.run_analysis(
    df_full,
    str(split.get("in_start")),
    str(split.get("in_end")),
    str(split.get("out_start")),
    str(split.get("out_end")),
    cfg.vol_adjust.get("target_vol", 1.0),
    getattr(cfg, "run", {}).get("monthly_cost", 0.0),
    indices_list=["Mgr_01", "Mgr_02"],
)
if analysis_idx is None or not analysis_idx.get("benchmark_stats"):
    raise SystemExit("pipeline.run_analysis with indices_list failed")

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

# Exercise export_data with a formatter to cover formatting hooks
fmt_prefix = Path("demo/exports/formatted_demo")
df_fmt = pd.DataFrame({"A": [1, 2], "B": [3, 4]})


def _double(df: pd.DataFrame) -> pd.DataFrame:
    return df * 2


export.export_data(
    {"tbl": df_fmt},
    str(fmt_prefix),
    formats=["xlsx", "csv", "json", "txt"],
    formatter=_double,
)
csv_file = fmt_prefix.with_name(f"{fmt_prefix.stem}_tbl.csv")
if not csv_file.exists():
    raise SystemExit("Formatted CSV not created")
chk = pd.read_csv(csv_file, index_col=0)
if chk["A"].iloc[0] != 2:
    raise SystemExit("Formatter did not apply")

# Exercise individual exporters and the period formatter helper
indiv_prefix = Path("demo/exports/indiv")
export.export_to_csv({"metrics": metrics_df}, str(indiv_prefix))
export.export_to_json({"metrics": metrics_df}, str(indiv_prefix))
export.export_to_txt({"metrics": metrics_df}, str(indiv_prefix))
for ext in ("csv", "json", "txt"):
    if not indiv_prefix.with_name(f"{indiv_prefix.stem}_metrics.{ext}").exists():
        raise SystemExit(f"export_to_{ext} failed")

extra_prefix = Path("demo/exports/extra_period.xlsx")
export.reset_formatters_excel()
export.make_period_formatter(
    "extra",
    results[0],
    str(results[0]["period"][0]),
    str(results[0]["period"][1]),
    str(results[0]["period"][2]),
    str(results[0]["period"][3]),
)
export.export_to_excel(
    {"extra": export.summary_frame_from_result(results[0])},
    str(extra_prefix),
)
wb_extra = openpyxl.load_workbook(extra_prefix)
if wb_extra["extra"]["A1"].value != "Vol-Adj Trend Analysis":
    raise SystemExit("make_period_formatter formatting failed")

_check_gui("config/demo.yml")
_check_selection_modes(cfg)
_check_cli_env("config/demo.yml")
_check_cli_env_multi("config/demo.yml")
_check_cli("config/demo.yml")
_check_misc("config/demo.yml", cfg, results)
_check_rebalancer_logic()
_check_load_csv_error()
_check_metrics_basic()
_check_builtin_metric_aliases()
_check_selector_errors()
_check_weighting_errors()
_check_notebook_utils()

# ------------------------------------------------------------
# Additional error handling checks


def _check_export_errors() -> None:
    """Ensure exporter functions validate input."""
    df = pd.DataFrame({"A": [1]})
    try:
        export.export_data({"A": df}, "demo/exports/bad", formats=["foo"])
    except ValueError:
        pass
    else:  # pragma: no cover - should not happen
        raise SystemExit("export_data did not reject unknown format")


def _check_config_errors() -> None:
    """Verify config loader rejects invalid YAML."""
    tmp = Path("demo/exports/invalid.yml")
    tmp.write_text("- a\n- b\n", encoding="utf-8")
    try:
        load(str(tmp))
    except TypeError:
        pass
    else:  # pragma: no cover - should not happen
        raise SystemExit("config.load accepted non-mapping YAML")
    finally:
        tmp.unlink(missing_ok=True)


# Execute additional error handling checks
_check_export_errors()
_check_config_errors()

# Verify top-level package exports
pkg_cfg = ta.config.load("config/demo.yml")
pkg_df = ta.load_csv(pkg_cfg.data["csv_path"])
if ta.identify_risk_free_fund(pkg_df) is None:
    raise SystemExit("Package export check failed")
ta.reset_formatters_excel()


@ta.register_formatter_excel("pkg")
def _pkg_fmt(ws, _wb):
    ws.write(0, 0, "pkg")


pkg_path = Path("demo/exports/pkg.xlsx")
ta.export_to_excel({"pkg": pd.DataFrame({"A": [1]})}, str(pkg_path))
if not pkg_path.exists():
    raise SystemExit("Package export_to_excel failed")

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
rc = run_multi_analysis.main(["-c", str(cli_cfg), "--detailed"])
if rc != 0:
    raise SystemExit("run_multi_analysis CLI detailed failed")
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
subprocess.run(
    [
        sys.executable,
        "-m",
        "trend_analysis.run_multi_analysis",
        "-c",
        "config/demo.yml",
        "--detailed",
    ],
    check=True,
)

# Execute the full test suite to cover the entire code base
run_tests = Path(__file__).resolve().with_name("run_tests.sh")
subprocess.run([str(run_tests)], check=True)
