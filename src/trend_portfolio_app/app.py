from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml  # type: ignore[import-untyped]

from trend_analysis import api
from trend_analysis.multi_period import run as run_multi
from trend_analysis.config import DEFAULTS as DEFAULT_CFG_PATH, Config
from trend_analysis.data import load_csv as ta_load_csv, identify_risk_free_fund
from trend_analysis.core.rank_selection import METRIC_REGISTRY

# Optional drag-and-drop support (falls back gracefully if not installed)
try:  # streamlit-sortables by okld
    from streamlit_sortables import sort_items as _st_sort_items  # type: ignore
except Exception:  # pragma: no cover - optional UI nicety
    _st_sort_items = None  # type: ignore


# ---- small helpers -----------------------------------------------------


def _read_defaults() -> Dict[str, Any]:
    data = yaml.safe_load(Path(DEFAULT_CFG_PATH).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    # Provide a sensible default CSV for local demos if present
    demo_csv = Path("demo/demo_returns.csv")
    data.setdefault("data", {})
    data["data"].setdefault("csv_path", str(demo_csv) if demo_csv.exists() else "")
    # Add portfolio.policy for UI convenience
    data.setdefault("portfolio", {})
    data["portfolio"].setdefault("policy", "")  # "threshold_hold" or empty
    return data


def _to_yaml(d: Dict[str, Any]) -> str:
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)


def _merge_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _build_cfg(d: Dict[str, Any]) -> Config:
    # Construct Config object (works with or without pydantic installed)
    return Config(**d)


def _summarise_run_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Round for display
    disp = df.copy()
    for c in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].astype(float).round(4)
    return disp


def _summarise_multi(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        per = r.get("period")
        out_ew = r.get("out_ew_stats", {})
        out_user = r.get("out_user_stats", {})

        def _get(d: Any, key: str) -> float:
            try:
                return float(getattr(d, key, d.get(key)))
            except Exception:
                return float("nan")

        rows.append(
            {
                "in_start": per[0] if isinstance(per, (list, tuple)) else "",
                "in_end": per[1] if isinstance(per, (list, tuple)) else "",
                "out_start": per[2] if isinstance(per, (list, tuple)) else "",
                "out_end": per[3] if isinstance(per, (list, tuple)) else "",
                "ew_sharpe": _get(out_ew, "sharpe"),
                "user_sharpe": _get(out_user, "sharpe"),
                "ew_cagr": _get(out_ew, "cagr"),
                "user_cagr": _get(out_user, "cagr"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ["ew_sharpe", "user_sharpe", "ew_cagr", "user_cagr"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
    return df


# ---- UI ----------------------------------------------------------------

st.set_page_config(page_title="Trend Portfolio App", layout="wide")

st.title("Trend Portfolio App")

# Session state
if "config_dict" not in st.session_state:
    st.session_state.config_dict = _read_defaults()

with st.sidebar:
    st.header("Configuration")
    # Load/Reset
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset to defaults", use_container_width=True):
            st.session_state.config_dict = _read_defaults()
    with col_b:
        uploaded = st.file_uploader(
            "Import YAML", type=["yml", "yaml"], label_visibility="collapsed"
        )
        if uploaded is not None:
            try:
                data = yaml.safe_load(uploaded.getvalue())
                if isinstance(data, dict):
                    st.session_state.config_dict = _merge_update(_read_defaults(), data)
                    st.success("Config imported")
                else:
                    st.error("Uploaded YAML must be a mapping")
            except Exception as exc:
                st.error(f"Failed to parse YAML: {exc}")

    cfg = st.session_state.config_dict

    # Minimal convenience inputs
    st.subheader("Required inputs")
    cfg.setdefault("data", {})
    # Use a keyed input so other UI (like uploaders) can update it live
    st.text_input(
        "CSV path",
        key="data.csv_path",
        value=str(cfg["data"].get("csv_path", "")),
        help="Path to a CSV with a Date column and return columns",
    )
    # Keep config in sync with session state
    cfg["data"]["csv_path"] = st.session_state.get(
        "data.csv_path", cfg["data"].get("csv_path", "")
    )

    # Choose run mode
    run_mode = st.radio("Run mode", ["Single period", "Multi-period"], horizontal=True)

    # Policy for multi-period
    if run_mode == "Multi-period":
        cfg.setdefault("portfolio", {})
        policy_options = [
            "",  # none
            "threshold_hold",  # existing
            "periodic_rebalance",  # every N periods
            "drift_band",  # rebalance when weights drift beyond bands
            "turnover_cap",  # rebalance constrained by turnover budget
            "drawdown_guard",  # reduce risk on drawdown triggers
            "vol_target_rebalance",  # rebalance to maintain target vol
        ]
        current = str(cfg["portfolio"].get("policy", ""))
        idx = policy_options.index(current) if current in policy_options else 0
        cfg["portfolio"]["policy"] = st.selectbox(
            "Policy",
            options=policy_options,
            index=idx,
            help=(
                "Portfolio update rule: threshold_hold (implemented). Others are previews: "
                "periodic_rebalance, drift_band, turnover_cap, drawdown_guard, vol_target_rebalance."
            ),
        )

    # Download current YAML
    yaml_bytes = _to_yaml(cfg).encode("utf-8")
    st.download_button(
        "Download YAML", data=yaml_bytes, file_name="config.yml", mime="text/yaml"
    )

# Main layout

tabs = st.tabs(
    [
        "YAML Editor",
        "Data",
        "Preprocessing",
        "Volatility",
        "Single Period",
        "Portfolio",
        "Metrics",
        "Multi-Period",
        "Run",
    ]
)

# YAML editor (one place to edit everything)
with tabs[0]:
    st.caption("Full configuration (editable). Changes here update the live config.")
    yaml_text = st.text_area(
        "YAML",
        value=_to_yaml(st.session_state.config_dict),
        height=380,
        label_visibility="collapsed",
    )
    if st.button("Apply YAML changes"):
        try:
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict):
                st.session_state.config_dict = _merge_update(
                    st.session_state.config_dict, data
                )
                st.success("Applied")
            else:
                st.error("YAML must be a mapping")
        except Exception as exc:
            st.error(f"Invalid YAML: {exc}")

# Data tab
with tabs[1]:
    d = st.session_state.config_dict.setdefault("data", {})
    st.text_input(
        "Managers glob",
        value=str(d.get("managers_glob", "")),
        key="data.managers_glob",
        help="Pattern to find manager CSVs if not using a single CSV",
    )
    st.text_input(
        "Indices glob",
        value=str(d.get("indices_glob", "")),
        key="data.indices_glob",
        help="Pattern to find index/benchmark CSVs",
    )
    # Optional: upload a CSV and sync the sidebar required input
    up = st.file_uploader(
        "Upload CSV", type=["csv"], help="Upload a CSV to quickly set the data source"
    )
    if up is not None:
        try:
            import os
            import time

            os.makedirs(".uploads", exist_ok=True)
            fname = f".uploads/{int(time.time())}_{up.name}"
            with open(fname, "wb") as f:
                f.write(up.getbuffer())
            st.session_state["data.csv_path"] = fname
            st.session_state.config_dict.setdefault("data", {})["csv_path"] = fname
            st.success(f"Saved upload to {fname}")
        except Exception as exc:
            st.error(f"Failed to save upload: {exc}")
    st.text_input(
        "Date column",
        value=str(d.get("date_column", "Date")),
        key="data.date_column",
        help="Name of the date column in your CSV",
    )
    st.text_input(
        "Price column",
        value=str(d.get("price_column", "Adj_Close")),
        key="data.price_column",
        help="If you import prices rather than returns",
    )
    st.selectbox(
        "Frequency",
        ["D", "W", "M"],
        index=["D", "W", "M"].index(str(d.get("frequency", "M"))),
        key="data.frequency",
        help="Expected data frequency: Daily/Weekly/Monthly (default Monthly)",
    )
    st.text_input(
        "Timezone",
        value=str(d.get("timezone", "UTC")),
        key="data.timezone",
        help="Timezone for parsing dates if needed",
    )
    st.text_input(
        "Currency",
        value=str(d.get("currency", "USD")),
        key="data.currency",
        help="Base currency label",
    )
    st.selectbox(
        "NaN policy",
        ["drop", "ffill", "bfill", "both"],
        index=max(
            0,
            (
                ["drop", "ffill", "bfill", "both"].index(
                    str(d.get("nan_policy", "ffill"))
                )
                if str(d.get("nan_policy", "ffill"))
                in ["drop", "ffill", "bfill", "both"]
                else 0
            ),
        ),
        key="data.nan_policy",
        help="How to treat missing values during ingestion.",
    )
    st.caption(
        "NaN policies: drop = remove rows with any NaN; ffill = forward‑fill last valid value; bfill = back‑fill next valid value; both = ffill then bfill to fill gaps."
    )

    st.markdown("---")
    st.subheader("Risk‑free rate source")
    rf_source = st.radio(
        "Choose how the risk‑free is provided",
        ["Auto (from data)", "Choose column", "Constant (annual)"],
        index=(
            2
            if bool(d.get("rf_use_constant", False))
            else (1 if d.get("rf_column") else 0)
        ),
        horizontal=True,
        key="data._rf_source_choice",
        help="Prefer a series from your dataset. We'll auto‑detect using typical RF names (RF, Risk Free, T‑Bill, Cash) and, if none match, the very low‑volatility set (bottom decile) before falling back to the absolute lowest vol. Use a constant only if no series is available.",
    )

    # Load CSV once to provide choices
    csv_for_rf = st.session_state.config_dict.get("data", {}).get("csv_path")
    numeric_cols: list[str] = []
    auto_rf: str | None = None
    if csv_for_rf:
        try:
            df_preview = ta_load_csv(str(csv_for_rf))
            if isinstance(df_preview, pd.DataFrame):
                numeric_cols = [
                    c for c in df_preview.select_dtypes("number").columns if c != "Date"
                ]
                auto_rf = identify_risk_free_fund(df_preview)
                # Persist for other tabs
                st.session_state["data._numeric_columns"] = numeric_cols
        except Exception:
            pass

    if rf_source == "Auto (from data)":
        if auto_rf:
            st.info(f"Auto‑detected RF column: {auto_rf}")
            st.session_state.config_dict["data"]["rf_column"] = auto_rf
        else:
            st.warning(
                "Could not detect a numeric RF series from the CSV—falling back to constant rate."
            )
            st.session_state.config_dict["data"].pop("rf_column", None)
            st.session_state.config_dict["data"]["rf_use_constant"] = True
        if numeric_cols:
            st.session_state.config_dict["data"]["rf_use_constant"] = False
    elif rf_source == "Choose column":
        if not numeric_cols:
            st.error(
                "No numeric columns found. Provide a valid CSV to choose an RF column."
            )
        choice = st.selectbox(
            "Risk‑free column",
            options=numeric_cols or [""],
            index=max(
                0,
                (
                    (numeric_cols or [""]).index(
                        str(d.get("rf_column", numeric_cols[0] if numeric_cols else ""))
                    )
                    if d.get("rf_column") in numeric_cols
                    else 0
                ),
            ),
            key="data.rf_column",
            help="Pick the column to use as risk‑free series",
        )
        st.session_state.config_dict["data"]["rf_use_constant"] = False
    else:  # Constant
        st.session_state.config_dict["data"].pop("rf_column", None)
        st.session_state.config_dict["data"]["rf_use_constant"] = True
        # Also surface the constant on Metrics tab via a quick control here
        st.session_state.config_dict.setdefault("metrics", {})
        st.number_input(
            "Constant RF (annualised, decimal)",
            min_value=-0.10,
            max_value=0.20,
            step=0.005,
            value=float(
                st.session_state.config_dict["metrics"].get("rf_rate_annual", 0.02)
            ),
            help="Used only when RF source is Constant",
            key="metrics.rf_rate_annual",
        )

    st.caption(
        "Data tab: point to your CSV(s) and set parsing rules. For RF, we auto‑detect by name (RF/Risk‑free/T‑Bill/Cash) and only then by very low volatility to avoid illiquid private marks looking ‘low‑vol’."
    )

    # Manager/Index column selection from the loaded CSV (optional)
    if numeric_cols:
        base_cols = [
            c
            for c in numeric_cols
            if c
            != (
                st.session_state.config_dict.get("data", {}).get("rf_column")
                or "__none__"
            )
        ]
        sel_man = st.multiselect(
            "Manager columns",
            options=base_cols,
            default=st.session_state.config_dict.get("data", {}).get(
                "manager_columns", []
            ),
            help="Select one or more manager return columns. Leave empty to include all.",
        )
        sel_idx = st.multiselect(
            "Index columns",
            options=base_cols,
            default=st.session_state.config_dict.get("data", {}).get(
                "indices_columns", []
            ),
            help="Select benchmark/index series for metrics and alpha. Leave empty to skip.",
        )
        st.session_state.config_dict["data"]["manager_columns"] = sel_man
        st.session_state.config_dict["data"]["indices_columns"] = sel_idx
        st.session_state["data._indices_candidates"] = sel_idx or base_cols

# Preprocessing tab
with tabs[2]:
    pre = st.session_state.config_dict.setdefault("preprocessing", {})
    st.checkbox(
        "De-duplicate",
        value=bool(pre.get("de_duplicate", True)),
        key="preprocessing.de_duplicate",
    )
    win = pre.setdefault("winsorise", {"enabled": True, "limits": [0.01, 0.99]})
    st.checkbox(
        "Winsorise enabled",
        value=bool(win.get("enabled", True)),
        key="preprocessing.winsorise.enabled",
    )
    limits = win.get("limits", [0.01, 0.99]) or [0.01, 0.99]
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Lower limit",
            min_value=0.0,
            max_value=0.49,
            value=float(limits[0]),
            step=0.005,
            key="preprocessing.winsorise.lower",
        )
    with c2:
        st.number_input(
            "Upper limit",
            min_value=0.51,
            max_value=1.0,
            value=float(limits[1]),
            step=0.005,
            key="preprocessing.winsorise.upper",
        )
    st.checkbox(
        "Log prices",
        value=bool(pre.get("log_prices", False)),
        key="preprocessing.log_prices",
    )
    st.text_input(
        "Holiday calendar",
        value=str(pre.get("holiday_calendar", "NYSE")),
        key="preprocessing.holiday_calendar",
    )
    rs = pre.setdefault(
        "resample", {"target": "D", "method": "last", "business_only": True}
    )
    st.selectbox(
        "Resample target",
        ["D", "W", "M"],
        index=["D", "W", "M"].index(str(rs.get("target", "D"))),
        key="preprocessing.resample.target",
    )
    st.selectbox(
        "Resample method",
        ["last", "mean", "sum", "pad"],
        index=["last", "mean", "sum", "pad"].index(str(rs.get("method", "last"))),
        key="preprocessing.resample.method",
    )
    st.checkbox(
        "Business days only",
        value=bool(rs.get("business_only", True)),
        key="preprocessing.resample.business_only",
    )

# Volatility tab
with tabs[3]:
    va = st.session_state.config_dict.setdefault("vol_adjust", {})
    st.checkbox(
        "Enable vol adjust",
        value=bool(va.get("enabled", True)),
        key="vol_adjust.enabled",
        help="Turn on to normalise each fund to the same annual volatility (fairer comparisons).",
    )
    st.number_input(
        "Target vol (annual)",
        min_value=0.0,
        max_value=1.0,
        value=float(va.get("target_vol", 0.10)),
        step=0.005,
        key="vol_adjust.target_vol",
        help="Your desired annualised volatility level. Example: 0.10 = 10%.",
    )
    win = va.setdefault("window", {"length": 63, "decay": "ewma", "lambda": 0.94})
    # Show months for UI; convert to trading days on run
    months_default = max(1, int(round(int(win.get("length", 63)) / 21)))
    st.number_input(
        "Window length (months)",
        min_value=1,
        max_value=36,
        value=months_default,
        step=1,
        key="vol_adjust.window._months",
        help="How far back to measure volatility (calendar months, ~21 trading days each). Common: 3 months ≈ 63 days.",
    )
    st.selectbox(
        "Decay",
        ["ewma", "simple"],
        index=["ewma", "simple"].index(str(win.get("decay", "ewma"))),
        key="vol_adjust.window.decay",
        help="EWMA weights recent moves more; simple gives equal weight to the whole window.",
    )
    st.number_input(
        "EWMA lambda",
        min_value=0.50,
        max_value=0.999,
        value=float(win.get("lambda", 0.94)),
        step=0.001,
        key="vol_adjust.window.lambda",
        help="Only used for EWMA. Higher values react more slowly (longer memory).",
    )
    st.number_input(
        "Floor vol",
        min_value=0.0,
        max_value=1.0,
        value=float(va.get("floor_vol", 0.04)),
        step=0.005,
        key="vol_adjust.floor_vol",
        help="Minimum volatility used to avoid extreme scaling when σ is very small.",
    )
    st.info(
        "Set a target, pick how to estimate volatility, and apply a floor so leverage doesn’t explode when volatility is tiny."
    )

# Single Period tab (harmonized labels)
with tabs[4]:
    s = st.session_state.config_dict.setdefault("sample_split", {})
    method = st.selectbox(
        "Split method",
        ["date", "ratio"],
        index=(0 if str(s.get("method", "date")) == "date" else 1),
        help="Choose a specific split date or a proportion for in/out samples",
    )
    st.text_input(
        "Split date (YYYY-MM-DD)",
        value=str(s.get("date", "2017-12-31")),
        key="sample_split.date",
        help="Used when method = date",
    )
    st.slider(
        "Split ratio (in-sample)",
        min_value=0.1,
        max_value=0.9,
        value=float(s.get("ratio", 0.7)),
        step=0.05,
        key="sample_split.ratio",
        help="Used when method = ratio; the rest becomes out-of-sample",
    )
    st.checkbox(
        "Rolling walk (CV)",
        value=bool(s.get("rolling_walk", False)),
        key="sample_split.rolling_walk",
        help="If enabled, trains on a rolling window and tests on the next period repeatedly (like cross‑validation over time).",
    )
    st.number_input(
        "Folds (if rolling)",
        min_value=1,
        max_value=24,
        value=int(s.get("folds", 5)),
        step=1,
        key="sample_split.folds",
    )
    st.markdown("---")
    st.caption("Explicit single‑period windows (YYYY‑MM)")
    st.text_input(
        "In-sample start (YYYY-MM)",
        value=str(s.get("in_start", "2015-01")),
        key="sample_split.in_start",
    )
    st.text_input(
        "In-sample end (YYYY-MM)",
        value=str(s.get("in_end", "2017-12")),
        key="sample_split.in_end",
    )
    st.text_input(
        "Out-of-sample start (YYYY-MM)",
        value=str(s.get("out_start", "2018-01")),
        key="sample_split.out_start",
    )
    st.text_input(
        "Out-of-sample end (YYYY-MM)",
        value=str(s.get("out_end", "2020-12")),
        key="sample_split.out_end",
    )
    st.info(
        "Single Period defines one training window and one test window. Multi‑Period runs a schedule of such windows over time."
    )

# Portfolio tab (subset of common controls)
with tabs[5]:
    p = st.session_state.config_dict.setdefault("portfolio", {})
    st.selectbox(
        "Selection mode",
        ["all", "random", "manual", "rank"],
        index=["all", "random", "manual", "rank"].index(
            str(p.get("selection_mode", "all"))
        ),
        key="portfolio.selection_mode",
        help="How to choose funds: all = include all; random = pick N at random; manual = pick by names; rank = pick by metric ranking.",
    )
    st.number_input(
        "Random N",
        min_value=1,
        max_value=100,
        value=int(p.get("random_n", 10)),
        step=1,
        key="portfolio.random_n",
        help="When using 'random', how many funds to sample.",
    )
    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000,
        value=int(p.get("random_seed", 42)),
        step=1,
        key="portfolio.random_seed",
        help="Seed for reproducible random selection.",
    )
    # Rank params
    rank = p.setdefault("rank", {})
    st.selectbox(
        "Rank inclusion approach",
        ["top_n", "top_pct", "threshold"],
        index=["top_n", "top_pct", "threshold"].index(
            str(rank.get("inclusion_approach", "top_n"))
        ),
        key="portfolio.rank.inclusion_approach",
        help="How to include funds based on metric scores: top N, top percentage, or threshold value.",
    )
    st.number_input(
        "Rank n",
        min_value=1,
        max_value=100,
        value=int(rank.get("n", 10)),
        step=1,
        key="portfolio.rank.n",
        help="Number of funds to keep when using top_n.",
    )
    st.slider(
        "Rank pct",
        0.01,
        1.0,
        float(rank.get("pct", 0.1)),
        0.01,
        key="portfolio.rank.pct",
        help="Fraction of funds to keep when using top_pct.",
    )
    st.text_input(
        "Rank threshold metric value",
        value=str(rank.get("threshold", 1.5)),
        key="portfolio.rank.threshold",
        help="Minimum metric value to include when using threshold.",
    )
    metric_options = sorted(list(METRIC_REGISTRY.keys()))
    st.selectbox(
        "Rank score by",
        options=metric_options,
        index=max(
            0,
            (
                metric_options.index(str(rank.get("score_by", "Sharpe")))
                if str(rank.get("score_by", "Sharpe")) in metric_options
                else metric_options.index("Sharpe") if "Sharpe" in metric_options else 0
            ),
        ),
        key="portfolio.rank.score_by",
        help="Which metric to rank funds by.",
    )
    # Weighting
    w = p.setdefault("weighting", {"name": "equal", "params": {}})
    st.selectbox(
        "Weighting",
        ["equal", "score_prop_bayes", "adaptive_bayes"],
        index=["equal", "score_prop_bayes", "adaptive_bayes"].index(
            str(w.get("name", "equal"))
        ),
        key="portfolio.weighting.name",
        help="How to set portfolio weights: equal = simple average; score_prop_bayes = proportional to scores with shrinkage; adaptive_bayes = stateful adaptive scheme.",
    )
    w_params = w.setdefault("params", {})
    # Default to existing value or a common metric
    default_metric = str(w_params.get("column", "Sharpe"))
    st.selectbox(
        "Weighting metric (for score‑prop)",
        options=metric_options,
        index=max(
            0,
            (
                metric_options.index(default_metric)
                if default_metric in metric_options
                else metric_options.index("Sharpe") if "Sharpe" in metric_options else 0
            ),
        ),
        key="portfolio.weighting.params.column",
        help="Which score_frame column to weight by (e.g., Sharpe).",
    )
    st.number_input(
        "Bayesian shrink_tau",
        min_value=0.0,
        max_value=5.0,
        value=float(w_params.get("shrink_tau", 0.25)),
        step=0.05,
        key="portfolio.weighting.params.shrink_tau",
        help="Strength of shrinkage towards the prior (higher = more shrink).",
    )
    st.number_input(
        "Adaptive half_life (days)",
        min_value=1,
        max_value=3650,
        value=int(w_params.get("half_life", 90)),
        step=1,
        key="portfolio.weighting.params.half_life",
        help="Speed of adaptation in days; shorter = reacts faster.",
    )
    st.number_input(
        "Adaptive obs_sigma",
        min_value=0.0,
        max_value=2.0,
        value=float(w_params.get("obs_sigma", 0.25)),
        step=0.01,
        key="portfolio.weighting.params.obs_sigma",
        help="Assumed observation noise of scores (standard deviation).",
    )
    st.text_input(
        "Adaptive prior_mean (equal|index or list)",
        value=str(w_params.get("prior_mean", "equal")),
        key="portfolio.weighting.params.prior_mean",
        help="Prior mean weights: 'equal', an index name, or explicit list.",
    )
    st.number_input(
        "Adaptive prior_tau",
        min_value=0.0,
        max_value=10.0,
        value=float(w_params.get("prior_tau", 1.0)),
        step=0.1,
        key="portfolio.weighting.params.prior_tau",
        help="Confidence in the prior (higher = stickier prior).",
    )
    # Constraints
    cons = p.setdefault(
        "constraints",
        {
            "max_funds": 10,
            "min_weight": 0.05,
            "max_weight": 0.18,
            "min_weight_strikes": 2,
        },
    )
    st.number_input(
        "Max funds",
        min_value=1,
        max_value=100,
        value=int(cons.get("max_funds", 10)),
        step=1,
        key="portfolio.constraints.max_funds",
        help="Upper bound on number of holdings.",
    )
    st.number_input(
        "Min weight",
        min_value=0.0,
        max_value=1.0,
        value=float(cons.get("min_weight", 0.05)),
        step=0.005,
        key="portfolio.constraints.min_weight",
        help="Smallest allowed position size.",
    )
    st.number_input(
        "Max weight",
        min_value=0.0,
        max_value=1.0,
        value=float(cons.get("max_weight", 0.18)),
        step=0.005,
        key="portfolio.constraints.max_weight",
        help="Largest allowed position size.",
    )
    st.number_input(
        "Min-weight strikes before replace",
        min_value=1,
        max_value=12,
        value=int(cons.get("min_weight_strikes", 2)),
        step=1,
        key="portfolio.constraints.min_weight_strikes",
        help="If a fund hits min weight this many times, allow replacement.",
    )

    st.markdown("---")
    st.subheader("Selection policy")
    sel_cfg = p.setdefault("selection_policy", {})
    # Elements (composable guards/caps)
    element_options = [
        "min_tenure",
        "turnover_budget_selection",
        "diversification_guard",  # scaffold
        "drawdown_kickout",
    ]
    chosen_elements = st.multiselect(
        "Elements (optional)",
        options=element_options,
        default=sel_cfg.get("elements", []),
        key="portfolio.selection_policy.elements",
        help="Composable guards that apply before add/drop decisions.",
    )
    # Per-element params
    sp_params = sel_cfg.setdefault("params", {})
    if "min_tenure" in chosen_elements:
        st.number_input(
            "Min tenure (periods)",
            min_value=1,
            max_value=36,
            value=int(sp_params.get("min_tenure", {}).get("n", 3)),
            step=1,
            key="portfolio.selection_policy.params.min_tenure.n",
            help="Once added, must be held at least this many periods before eligible for removal.",
        )
    if "turnover_budget_selection" in chosen_elements:
        st.number_input(
            "Max changes per period",
            min_value=0,
            max_value=20,
            value=int(
                sp_params.get("turnover_budget_selection", {}).get("max_changes", 2)
            ),
            step=1,
            key="portfolio.selection_policy.params.turnover_budget_selection.max_changes",
            help="Limit the number of adds/removes each period.",
        )
    if "diversification_guard" in chosen_elements:
        st.text_area(
            "Bucket mapping (scaffold)",
            value=str(
                sp_params.get("diversification_guard", {}).get(
                    "buckets_text", "# name: bucket\n# FUND_A: Macro\n# FUND_B: Trend"
                )
            ),
            height=120,
            key="portfolio.selection_policy.params.diversification_guard.buckets_text",
            help="Scaffold: define name→bucket lines to enforce per-bucket caps in a later phase.",
        )
        st.number_input(
            "Max per bucket",
            min_value=1,
            max_value=20,
            value=int(
                sp_params.get("diversification_guard", {}).get("max_per_bucket", 2)
            ),
            step=1,
            key="portfolio.selection_policy.params.diversification_guard.max_per_bucket",
            help="Placeholder cap; enforcement to be implemented later.",
        )
    if "drawdown_kickout" in chosen_elements:
        st.number_input(
            "Kickout DD threshold (decimal)",
            min_value=0.0,
            max_value=1.0,
            value=float(
                sp_params.get("drawdown_kickout", {}).get("dd_threshold", 0.25)
            ),
            step=0.01,
            key="portfolio.selection_policy.params.drawdown_kickout.dd_threshold",
            help="Remove a fund if its trailing drawdown exceeds this level (to be implemented).",
        )
        st.number_input(
            "Cooldown (periods)",
            min_value=0,
            max_value=36,
            value=int(sp_params.get("drawdown_kickout", {}).get("cooldown", 3)),
            step=1,
            key="portfolio.selection_policy.params.drawdown_kickout.cooldown",
            help="Number of periods before re-entry is allowed.",
        )

    # Competing add/drop rules with priority (DnD with fallback)
    st.markdown("**Competing rules priority**")
    add_defaults = ["threshold_hold", "sticky_rank_window", "confidence_interval"]
    drop_defaults = ["sticky_rank_window", "threshold_hold"]
    add_rules = sel_cfg.get("add_rules", add_defaults)
    drop_rules = sel_cfg.get("drop_rules", drop_defaults)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Add rules (priority order)")
        if _st_sort_items is not None and add_rules:
            new_add = _st_sort_items(items=list(add_rules), direction="vertical", key="sel.add.sort")  # type: ignore
            if isinstance(new_add, list) and new_add != add_rules:
                add_rules = new_add
                st.session_state.config_dict["portfolio"].setdefault(
                    "selection_policy", {}
                )["add_rules"] = add_rules
                st.rerun()
        else:
            # Fallback up/down
            if add_rules:
                cols = st.columns(len(add_rules))
                names = list(add_rules)
                for idx, name in enumerate(names):
                    with cols[idx]:
                        up_k = f"sel.add.up.{name}"
                        dn_k = f"sel.add.down.{name}"
                        st.button("⬆️", key=up_k)
                        st.button("⬇️", key=dn_k)
                        if st.session_state.get(up_k) and idx > 0:
                            names[idx - 1], names[idx] = names[idx], names[idx - 1]
                            st.session_state.config_dict["portfolio"].setdefault(
                                "selection_policy", {}
                            )["add_rules"] = names
                            st.rerun()
                        if st.session_state.get(dn_k) and idx < len(names) - 1:
                            names[idx + 1], names[idx] = names[idx], names[idx + 1]
                            st.session_state.config_dict["portfolio"].setdefault(
                                "selection_policy", {}
                            )["add_rules"] = names
                            st.rerun()
        # Params for add-rule variants
        st.number_input(
            "Sticky add: X consecutive periods",
            min_value=1,
            max_value=24,
            value=int(sp_params.get("sticky_rank_window", {}).get("add_x", 2)),
            step=1,
            key="portfolio.selection_policy.params.sticky_rank_window.add_x",
            help="Require X periods above threshold/top‑k before adding.",
        )
        st.number_input(
            "CI add: confidence level",
            min_value=0.5,
            max_value=0.999,
            value=float(sp_params.get("confidence_interval", {}).get("ci", 0.90)),
            step=0.01,
            key="portfolio.selection_policy.params.confidence_interval.ci",
            help="Only add if metric lower CI bound exceeds threshold.",
        )
    with c2:
        st.caption("Drop rules (priority order)")
        if _st_sort_items is not None and drop_rules:
            new_drop = _st_sort_items(items=list(drop_rules), direction="vertical", key="sel.drop.sort")  # type: ignore
            if isinstance(new_drop, list) and new_drop != drop_rules:
                drop_rules = new_drop
                st.session_state.config_dict["portfolio"].setdefault(
                    "selection_policy", {}
                )["drop_rules"] = drop_rules
                st.rerun()
        else:
            if drop_rules:
                cols = st.columns(len(drop_rules))
                names = list(drop_rules)
                for idx, name in enumerate(names):
                    with cols[idx]:
                        up_k = f"sel.drop.up.{name}"
                        dn_k = f"sel.drop.down.{name}"
                        st.button("⬆️", key=up_k)
                        st.button("⬇️", key=dn_k)
                        if st.session_state.get(up_k) and idx > 0:
                            names[idx - 1], names[idx] = names[idx], names[idx - 1]
                            st.session_state.config_dict["portfolio"].setdefault(
                                "selection_policy", {}
                            )["drop_rules"] = names
                            st.rerun()
                        if st.session_state.get(dn_k) and idx < len(names) - 1:
                            names[idx + 1], names[idx] = names[idx], names[idx + 1]
                            st.session_state.config_dict["portfolio"].setdefault(
                                "selection_policy", {}
                            )["drop_rules"] = names
                            st.rerun()
        st.number_input(
            "Sticky drop: Y consecutive misses",
            min_value=1,
            max_value=24,
            value=int(sp_params.get("sticky_rank_window", {}).get("drop_y", 2)),
            step=1,
            key="portfolio.selection_policy.params.sticky_rank_window.drop_y",
            help="Require Y periods below threshold/top‑k before dropping.",
        )

    st.markdown("---")
    st.subheader("Rebalancing strategy")
    rb_cfg = p.setdefault("rebalance", {})
    bayes_only = st.checkbox(
        "Use Bayesian‑only rebalancing",
        value=bool(rb_cfg.get("bayesian_only", True)),
        key="portfolio.rebalance.bayesian_only",
        help="If on, rebalancing is driven by the selected Bayesian weighting method only.",
    )
    if bayes_only:
        # Nudge weighting to a Bayesian method by default
        if p.get("weighting", {}).get("name") not in (
            "score_prop_bayes",
            "adaptive_bayes",
        ):
            p.setdefault("weighting", {})["name"] = "adaptive_bayes"
        st.caption(
            "Bayesian-only: pick between score_prop_bayes and adaptive_bayes in the Weighting section above."
        )
    else:
        # Choose and order non-Bayesian strategies
        strategy_options = [
            "periodic_rebalance",
            "drift_band",
            "turnover_cap",
            "vol_target_rebalance",
            "drawdown_guard",
        ]
        chosen = rb_cfg.get("strategies", ["drift_band"]) or ["drift_band"]
        st.caption(
            "Select and order how target weights are realized into trades and positions."
        )
        # Order editor
        if _st_sort_items is not None and chosen:
            new_order = _st_sort_items(items=list(dict.fromkeys([c for c in chosen if c in strategy_options])), direction="vertical", key="rb.strats.sort")  # type: ignore
            if isinstance(new_order, list) and new_order and new_order != chosen:
                st.session_state.config_dict["portfolio"].setdefault("rebalance", {})[
                    "strategies"
                ] = new_order
                st.rerun()
        else:
            # Fallback order with arrows
            names = list(
                dict.fromkeys([c for c in chosen if c in strategy_options])
            ) or ["drift_band"]
            cols = st.columns(len(names))
            for idx, name in enumerate(names):
                with cols[idx]:
                    up_k = f"rb.up.{name}"
                    dn_k = f"rb.down.{name}"
                    st.button("⬆️", key=up_k)
                    st.button("⬇️", key=dn_k)
                    if st.session_state.get(up_k) and idx > 0:
                        names[idx - 1], names[idx] = names[idx], names[idx - 1]
                        st.session_state.config_dict["portfolio"].setdefault(
                            "rebalance", {}
                        )["strategies"] = names
                        st.rerun()
                    if st.session_state.get(dn_k) and idx < len(names) - 1:
                        names[idx + 1], names[idx] = names[idx], names[idx + 1]
                        st.session_state.config_dict["portfolio"].setdefault(
                            "rebalance", {}
                        )["strategies"] = names
                        st.rerun()
        # Params per strategy
        rb_params = rb_cfg.setdefault("params", {})
        with st.expander("periodic_rebalance params"):
            st.number_input(
                "Interval (periods)",
                min_value=1,
                max_value=24,
                value=int(rb_params.get("periodic_rebalance", {}).get("interval", 1)),
                step=1,
                key="portfolio.rebalance.params.periodic_rebalance.interval",
            )
        with st.expander("drift_band params"):
            st.number_input(
                "Band width (abs)",
                min_value=0.0,
                max_value=0.5,
                value=float(rb_params.get("drift_band", {}).get("band_pct", 0.03)),
                step=0.005,
                key="portfolio.rebalance.params.drift_band.band_pct",
            )
            st.number_input(
                "Min trade weight",
                min_value=0.0,
                max_value=0.1,
                value=float(rb_params.get("drift_band", {}).get("min_trade", 0.005)),
                step=0.001,
                key="portfolio.rebalance.params.drift_band.min_trade",
            )
            st.selectbox(
                "Mode",
                ["partial", "full"],
                index=["partial", "full"].index(
                    str(rb_params.get("drift_band", {}).get("mode", "partial"))
                ),
                key="portfolio.rebalance.params.drift_band.mode",
            )
        with st.expander("turnover_cap params"):
            st.number_input(
                "Max turnover (sum |trades|)",
                min_value=0.0,
                max_value=1.0,
                value=float(rb_params.get("turnover_cap", {}).get("max_turnover", 0.2)),
                step=0.01,
                key="portfolio.rebalance.params.turnover_cap.max_turnover",
            )
            st.number_input(
                "Cost (bps)",
                min_value=0,
                max_value=100,
                value=int(rb_params.get("turnover_cap", {}).get("cost_bps", 10)),
                step=1,
                key="portfolio.rebalance.params.turnover_cap.cost_bps",
            )
            st.selectbox(
                "Priority",
                ["largest_gap", "best_score_delta"],
                index=["largest_gap", "best_score_delta"].index(
                    str(
                        rb_params.get("turnover_cap", {}).get("priority", "largest_gap")
                    )
                ),
                key="portfolio.rebalance.params.turnover_cap.priority",
            )
        with st.expander("vol_target_rebalance params"):
            st.number_input(
                "Target vol (annual)",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    rb_params.get("vol_target_rebalance", {}).get("target", 0.10)
                ),
                step=0.005,
                key="portfolio.rebalance.params.vol_target_rebalance.target",
            )
            st.number_input(
                "Window (months)",
                min_value=1,
                max_value=36,
                value=int(rb_params.get("vol_target_rebalance", {}).get("window", 6)),
                step=1,
                key="portfolio.rebalance.params.vol_target_rebalance.window",
            )
            st.selectbox(
                "Estimator",
                ["ewma", "simple"],
                index=["ewma", "simple"].index(
                    str(
                        rb_params.get("vol_target_rebalance", {}).get(
                            "estimator", "ewma"
                        )
                    )
                ),
                key="portfolio.rebalance.params.vol_target_rebalance.estimator",
            )
            st.number_input(
                "Leverage min",
                min_value=0.0,
                max_value=5.0,
                value=float(
                    rb_params.get("vol_target_rebalance", {}).get("lev_min", 0.5)
                ),
                step=0.05,
                key="portfolio.rebalance.params.vol_target_rebalance.lev_min",
            )
            st.number_input(
                "Leverage max",
                min_value=0.0,
                max_value=5.0,
                value=float(
                    rb_params.get("vol_target_rebalance", {}).get("lev_max", 1.5)
                ),
                step=0.05,
                key="portfolio.rebalance.params.vol_target_rebalance.lev_max",
            )
        with st.expander("drawdown_guard params"):
            st.number_input(
                "Window (periods)",
                min_value=1,
                max_value=60,
                value=int(rb_params.get("drawdown_guard", {}).get("dd_window", 12)),
                step=1,
                key="portfolio.rebalance.params.drawdown_guard.dd_window",
            )
            st.number_input(
                "DD threshold (decimal)",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    rb_params.get("drawdown_guard", {}).get("dd_threshold", 0.10)
                ),
                step=0.01,
                key="portfolio.rebalance.params.drawdown_guard.dd_threshold",
            )
            st.number_input(
                "Guard multiplier",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    rb_params.get("drawdown_guard", {}).get("guard_multiplier", 0.5)
                ),
                step=0.05,
                key="portfolio.rebalance.params.drawdown_guard.guard_multiplier",
            )
            st.number_input(
                "Recover threshold (decimal)",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    rb_params.get("drawdown_guard", {}).get("recover_threshold", 0.05)
                ),
                step=0.01,
                key="portfolio.rebalance.params.drawdown_guard.recover_threshold",
            )

# Metrics tab
with tabs[6]:
    m = st.session_state.config_dict.setdefault("metrics", {})
    st.checkbox(
        "Use continuous returns",
        value=bool(m.get("use_continuous", False)),
        key="metrics.use_continuous",
        help="If checked, log‑returns are used for certain metrics.",
    )
    # Alpha reference from indices in dataset
    idx_candidates = (
        st.session_state.get("data._indices_candidates")
        or st.session_state.get("data._numeric_columns")
        or []
    )
    idx_options = [""] + list(dict.fromkeys(idx_candidates))  # keep order, add empty
    st.selectbox(
        "Alpha reference index",
        options=idx_options,
        index=max(
            0,
            (
                idx_options.index(str(m.get("alpha_reference", "")))
                if str(m.get("alpha_reference", "")) in idx_options
                else 0
            ),
        ),
        key="metrics.alpha_reference",
        help="Optional index used as alpha/benchmark reference in metrics.",
    )
    # RF only needed if constant RF mode is selected globally
    d_glob = st.session_state.config_dict.get("data", {})
    if bool(d_glob.get("rf_use_constant", False)):
        st.number_input(
            "Constant RF (annual, decimal)",
            min_value=-0.10,
            max_value=0.20,
            step=0.005,
            value=float(m.get("rf_rate_annual", 0.02)),
            key="metrics.rf_rate_annual",
            help="Used only when RF source is 'Constant'.",
        )
    else:
        st.caption(
            "Risk‑free comes from a series (Data tab). The constant RF input is hidden."
        )
    reg = m.setdefault("registry", ["annual_return", "volatility", "sharpe_ratio"])
    st.write("Registry:", ", ".join(str(x) for x in reg))
    st.caption(
        "Metrics registry defines which statistics are computed per fund during scoring."
    )

# Multi-Period tab
with tabs[7]:
    mp_cfg = st.session_state.config_dict.setdefault("multi_period", {})
    st.selectbox(
        "Frequency",
        ["M", "Q", "A"],
        index=["M", "Q", "A"].index(str(mp_cfg.get("frequency", "A"))),
        key="multi_period.frequency",
        help="How often to roll the windows: Monthly, Quarterly, or Annually.",
    )
    st.number_input(
        "In-sample length (periods)",
        min_value=1,
        max_value=120,
        value=int(mp_cfg.get("in_sample_len", 3)),
        step=1,
        key="multi_period.in_sample_len",
        help="Training window length, in the chosen frequency units.",
    )
    st.number_input(
        "Out-of-sample length (periods)",
        min_value=1,
        max_value=120,
        value=int(mp_cfg.get("out_sample_len", 1)),
        step=1,
        key="multi_period.out_sample_len",
        help="Testing window length, in the chosen frequency units.",
    )
    st.text_input(
        "Start (YYYY-MM)",
        value=str(mp_cfg.get("start", "1990-01")),
        key="multi_period.start",
        help="First period to start the schedule.",
    )
    st.text_input(
        "End (YYYY-MM)",
        value=str(mp_cfg.get("end", "2024-12")),
        key="multi_period.end",
        help="Last period to end the schedule.",
    )
    # Triggers editor (simple key/value for sigmaX)
    trig = mp_cfg.setdefault("triggers", {"sigma1": {"sigma": 1, "periods": 2}})
    st.caption("Rebalancer triggers (key-> {sigma, periods})")
    # Drag-and-drop reorder (optional)
    if _st_sort_items is not None and trig:
        current_order = list(trig.keys())
        new_order = _st_sort_items(items=current_order, direction="vertical", key="mp.triggers.sort")  # type: ignore
        if isinstance(new_order, list) and new_order and new_order != current_order:
            trig = {k: trig[k] for k in new_order if k in trig}
            st.session_state.config_dict["multi_period"]["triggers"] = trig
            st.rerun()
        st.caption("Tip: Drag to reorder triggers.")
    elif trig:
        # Fallback: up/down buttons
        st.caption("Reorder (fallback): use arrows to move items.")
        names = list(trig.keys())
        cols = st.columns(len(names)) if names else []
        for idx, name in enumerate(names):
            with cols[idx]:
                up_key = f"mp.triggers.up.{name}"
                down_key = f"mp.triggers.down.{name}"
                st.button("⬆️", key=up_key)
                st.button("⬇️", key=down_key)
                if st.session_state.get(up_key):
                    if idx > 0:
                        names[idx - 1], names[idx] = names[idx], names[idx - 1]
                        trig = {k: trig[k] for k in names}
                        st.session_state.config_dict["multi_period"]["triggers"] = trig
                        st.rerun()
                if st.session_state.get(down_key):
                    if idx < len(names) - 1:
                        names[idx + 1], names[idx] = names[idx], names[idx + 1]
                        trig = {k: trig[k] for k in names}
                        st.session_state.config_dict["multi_period"]["triggers"] = trig
                        st.rerun()
    cta1, cta2 = st.columns([1, 1])
    with cta1:
        if st.button("Add trigger"):
            # Create a new unique key like sigma<N>
            existing = [k for k in trig.keys() if str(k).startswith("sigma")]
            next_idx = 1
            if existing:
                try:
                    next_idx = 1 + max(
                        int(str(k).replace("sigma", ""))
                        for k in existing
                        if str(k).replace("sigma", " ").strip().isdigit()
                    )
                except Exception:
                    next_idx = len(existing) + 1
            new_key = f"sigma{next_idx}"
            trig[new_key] = {"sigma": 1.0, "periods": 2}
            st.session_state.config_dict["multi_period"]["triggers"] = trig
            st.rerun()
    with cta2:
        if st.button("Remove last trigger"):
            if trig:
                last_key = list(trig.keys())[-1]
                trig.pop(last_key, None)
                st.session_state.config_dict["multi_period"]["triggers"] = trig
                st.rerun()
    for name, cfgv in list(trig.items()):
        with st.expander(f"Trigger {name}"):
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.6, 0.6])
            with col1:
                st.number_input(
                    f"{name} sigma",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(cfgv.get("sigma", 1)),
                    step=0.1,
                    key=f"multi_period.triggers.{name}.sigma",
                    help="Signal threshold in standard deviations",
                )
            with col2:
                st.number_input(
                    f"{name} periods",
                    min_value=1,
                    max_value=12,
                    value=int(cfgv.get("periods", 1)),
                    step=1,
                    key=f"multi_period.triggers.{name}.periods",
                    help="How many consecutive periods the signal must persist",
                )
            with col3:
                st.text_input(
                    f"Rename {name}",
                    value=name,
                    key=f"multi_period.triggers.{name}._name",
                    help="Update the trigger key name",
                )
            with col4:
                if st.button("Delete", key=f"btn.del.trigger.{name}"):
                    trig.pop(name, None)
                    st.session_state.config_dict["multi_period"]["triggers"] = trig
                    st.rerun()
            with col5:
                if st.button("Duplicate", key=f"btn.dup.trigger.{name}"):
                    # Find unique copy key
                    base = f"{name}_copy"
                    new_name = base
                    idx = 2
                    while new_name in trig:
                        new_name = f"{base}{idx}"
                        idx += 1
                    trig[new_name] = dict(cfgv)
                    st.session_state.config_dict["multi_period"]["triggers"] = trig
                    st.rerun()

            # Persist inline edits back to config on each rerun
            sigma_key = f"multi_period.triggers.{name}.sigma"
            periods_key = f"multi_period.triggers.{name}.periods"
            rename_key = f"multi_period.triggers.{name}._name"
            sigma_val = st.session_state.get(sigma_key, cfgv.get("sigma", 1))
            periods_val = st.session_state.get(periods_key, cfgv.get("periods", 1))
            new_name = st.session_state.get(rename_key, name)

            # Apply updates
            try:
                trig[name]["sigma"] = float(sigma_val)
            except Exception:
                pass
            try:
                trig[name]["periods"] = int(periods_val)
            except Exception:
                pass

            # Handle rename (avoid collision or empty)
            if (
                isinstance(new_name, str)
                and new_name
                and new_name != name
                and new_name not in trig
            ):
                trig[new_name] = trig.pop(name)
                st.session_state.config_dict["multi_period"]["triggers"] = trig
                st.rerun()

    st.info(
        "Inline edit triggers, drag to reorder (if enabled), duplicate, rename, or delete."
    )
    # Weight curve anchors
    wc = mp_cfg.setdefault(
        "weight_curve", {"anchors": [[0, 1.2], [50, 1.0], [100, 0.8]]}
    )
    anchors = wc.get("anchors", []) or []
    st.caption("Weight curve anchors [percentile, multiplier]")
    st.markdown(
        "Anchors map performance percentiles to weight multipliers. Example: [0,1.2] gives a 1.2× boost to the worst performers; [100,0.8] caps the best at 0.8×. The curve between anchors is interpolated."
    )
    # Drag-and-drop reorder (optional)
    if _st_sort_items is not None and anchors:
        labels = [
            f"{i}: {int(a[0]) if isinstance(a, (list, tuple)) and len(a)>0 else 0} | {float(a[1]) if isinstance(a, (list, tuple)) and len(a)>1 else 1.0}"
            for i, a in enumerate(anchors)
        ]
        new_labels = _st_sort_items(items=labels, direction="vertical", key="mp.anchors.sort")  # type: ignore
        if isinstance(new_labels, list) and new_labels and new_labels != labels:
            try:
                new_indices = []
                for lab in new_labels:
                    idx_str = str(lab).split(":", 1)[0].strip()
                    new_indices.append(int(idx_str))
                anchors = [anchors[i] for i in new_indices if 0 <= i < len(anchors)]
                wc["anchors"] = anchors
                st.session_state.config_dict["multi_period"]["weight_curve"] = wc
                st.rerun()
            except Exception:
                pass
        st.caption("Tip: Drag to reorder anchors.")
    elif anchors:
        # Fallback: up/down buttons per row
        st.caption("Reorder (fallback): use arrows to move rows.")
        cols = st.columns(len(anchors)) if anchors else []
        for idx in range(len(anchors)):
            with cols[idx]:
                up_key = f"mp.anchors.up.{idx}"
                down_key = f"mp.anchors.down.{idx}"
                st.button("⬆️", key=up_key)
                st.button("⬇️", key=down_key)
                if st.session_state.get(up_key):
                    if idx > 0:
                        anchors[idx - 1], anchors[idx] = anchors[idx], anchors[idx - 1]
                        wc["anchors"] = anchors
                        st.session_state.config_dict["multi_period"][
                            "weight_curve"
                        ] = wc
                        st.rerun()
                if st.session_state.get(down_key):
                    if idx < len(anchors) - 1:
                        anchors[idx + 1], anchors[idx] = anchors[idx], anchors[idx + 1]
                        wc["anchors"] = anchors
                        st.session_state.config_dict["multi_period"][
                            "weight_curve"
                        ] = wc
                        st.rerun()
    ac1, ac2 = st.columns([1, 1])
    with ac1:
        if st.button("Add anchor"):
            anchors.append([50, 1.0])
            wc["anchors"] = anchors
            st.session_state.config_dict["multi_period"]["weight_curve"] = wc
            st.rerun()
    with ac2:
        if st.button("Remove last anchor"):
            if anchors:
                anchors.pop()
                wc["anchors"] = anchors
                st.session_state.config_dict["multi_period"]["weight_curve"] = wc
                st.rerun()
    for i, pair in enumerate(list(anchors)):
        with st.expander(f"Anchor {i+1}"):
            c1, c2, c3, c4 = st.columns([1, 1, 0.6, 0.8])
            with c1:
                pct_val = 0
                try:
                    if isinstance(pair, (list, tuple)) and len(pair) > 0:
                        pct_val = int(pair[0])
                except Exception:
                    pct_val = 0
                st.number_input(
                    f"Percentile {i+1}",
                    min_value=0,
                    max_value=100,
                    value=pct_val,
                    step=1,
                    key=f"multi_period.weight_curve.anchors.{i}.pct",
                    help="Position along the performance percentile axis",
                )
            with c2:
                mul_val = 1.0
                try:
                    if isinstance(pair, (list, tuple)) and len(pair) > 1:
                        mul_val = float(pair[1])
                except Exception:
                    mul_val = 1.0
                st.number_input(
                    f"Multiplier {i+1}",
                    min_value=0.0,
                    max_value=5.0,
                    value=mul_val,
                    step=0.05,
                    key=f"multi_period.weight_curve.anchors.{i}.mul",
                    help="Weight multiplier applied at this percentile",
                )
            with c3:
                if st.button("Delete", key=f"btn.del.anchor.{i}"):
                    if 0 <= i < len(anchors):
                        anchors.pop(i)
                        wc["anchors"] = anchors
                        st.session_state.config_dict["multi_period"][
                            "weight_curve"
                        ] = wc
                        st.rerun()
            with c4:
                if st.button("Duplicate", key=f"btn.dup.anchor.{i}"):
                    if 0 <= i < len(anchors):
                        ins_pos = min(i + 1, len(anchors))
                        anchors.insert(ins_pos, list(anchors[i]))
                        wc["anchors"] = anchors
                        st.session_state.config_dict["multi_period"][
                            "weight_curve"
                        ] = wc
                        st.rerun()

            # Persist inline edits back to config
            pct_key = f"multi_period.weight_curve.anchors.{i}.pct"
            mul_key = f"multi_period.weight_curve.anchors.{i}.mul"
            pct_val = st.session_state.get(pct_key, pct_val)
            mul_val = st.session_state.get(mul_key, mul_val)
            try:
                anchors[i][0] = int(pct_val)
            except Exception:
                pass
            try:
                anchors[i][1] = float(mul_val)
            except Exception:
                pass
            wc["anchors"] = anchors
            st.session_state.config_dict["multi_period"]["weight_curve"] = wc

    st.info(
        "Inline edit anchors, drag to reorder (if enabled), duplicate, or delete. Anchors map percentiles to multipliers."
    )

# Run tab
with tabs[8]:
    st.subheader("Execute")
    col1, col2 = st.columns(2)
    with col1:
        go_single = st.button("Run Single Period", type="primary")
    with col2:
        go_multi = st.button("Run Multi-Period", type="primary")

    cfg_obj: Config | None = None
    if go_single or go_multi:
        # Rebuild config dict from session state flat keys we used
        cfg_dict = st.session_state.config_dict

        # Apply widget changes with dotted keys into the nested dict
        def _set_nested(d: Dict[str, Any], dotted: str, value: Any) -> None:
            parts = dotted.split(".")
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]  # type: ignore[assignment]
            cur[parts[-1]] = value

        prefixes = ("data.", "sample_split.", "portfolio.", "metrics.", "multi_period.")
        for k, v in list(st.session_state.items()):
            if isinstance(k, str) and k.startswith(prefixes):
                _set_nested(cfg_dict, k, v)
        # Convert volatility window months -> days (~21 trading days per month)
        try:
            months = int(st.session_state.get("vol_adjust.window._months", 0))
            if months > 0:
                days = int(months * 21)
                cfg_dict.setdefault("vol_adjust", {}).setdefault("window", {})[
                    "length"
                ] = days
        except Exception:
            pass
        # Ensure CSV exists (we won't validate path here, pipeline will raise)
        try:
            cfg_obj = _build_cfg(cfg_dict)
        except ValueError as ve:
            st.error(
                f"Configuration error: {ve}\n\n"
                "Hint: Check for missing or invalid values in your configuration. "
                "Refer to the documentation for required fields."
            )
            st.stop()
        except Exception as exc:
            st.error(
                f"Unexpected error during configuration validation: {type(exc).__name__}: {exc}\n\n"
                "Hint: Please review your configuration for errors. If the problem persists, "
                "check the YAML format and required fields."
            )
            st.stop()
        else:
            cfg_obj = cfg_obj  # type: ignore[assignment]

    if go_single and cfg_obj is not None:
        with st.spinner("Running single-period analysis..."):
            try:
                # Load CSV data from config
                csv_path = cfg_obj.data.get("csv_path")
                if csv_path is None:
                    st.error("CSV path must be provided in configuration")
                    st.stop()

                df = ta_load_csv(csv_path)
                if df is None:
                    st.error(f"Failed to load CSV file: {csv_path}")
                    st.stop()

                # Use unified API instead of direct pipeline call
                result = api.run_simulation(cfg_obj, df)
                out_df = result.metrics

                disp = _summarise_run_df(out_df)
                st.success(f"Completed. {len(disp)} rows.")
                st.dataframe(disp, use_container_width=True)
                # Download
                csv_bytes = disp.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="single_period_summary.csv",
                    mime="text/csv",
                )
            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as exc:
                st.exception(exc)

    if go_multi and cfg_obj is not None:
        with st.spinner("Running multi-period back-test..."):
            try:
                results = run_multi(cfg_obj)
                st.success(f"Completed. Periods: {len(results)}")
                summary = _summarise_multi(results)
                if not summary.empty:
                    st.dataframe(summary, use_container_width=True)
                    csv_bytes = summary.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download periods CSV",
                        data=csv_bytes,
                        file_name="multi_period_summary.csv",
                        mime="text/csv",
                    )
                # Raw results download (JSON)
                raw = json.dumps(results, default=str)
                st.download_button(
                    "Download raw JSON",
                    data=raw.encode("utf-8"),
                    file_name="multi_period_raw.json",
                    mime="application/json",
                )
            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as exc:
                st.exception(exc)

st.caption(
    "Tip: Edit YAML for full control. Use the tabs for quick tweaks. Save your config from the sidebar."
)
