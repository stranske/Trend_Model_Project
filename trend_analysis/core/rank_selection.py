"""Rank-based fund selection utilities.

This module implements the `rank` selection mode described in Agents.md. Funds can be kept using `top_n`, `top_pct` or `threshold` rules and scored by metrics registered in `METRIC_REGISTRY`. Metrics listed in `ASCENDING_METRICS` are treated as smaller-is-better.
"""

# =============================================================================
#  Runtime imports and dataclasses
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, cast
from ..export import Formatter
import io
import numpy as np
import pandas as pd
import ipywidgets as widgets
from .. import metrics as _metrics
from ..data import load_csv, identify_risk_free_fund, ensure_datetime

DEFAULT_METRIC = "annual_return"

# ──────────────────────────────────────────────────────────────────
# Metric transformer: raw | rank | percentile | zscore
# ──────────────────────────────────────────────────────────────────
def _apply_transform(
    series: pd.Series,
    *,
    mode: str = "raw",
    window: int | None = None,
    rank_pct: float | None = None,
    ddof: int = 0,
) -> pd.Series:
    """
    Return a transformed copy of *series* without mutating the original.

    Parameters
    ----------
    mode      : 'raw' | 'rank' | 'percentile' | 'zscore'
    window    : trailing periods for z‑score (ignored otherwise)
    rank_pct  : top‑X% mask when mode == 'percentile'
    ddof      : degrees of freedom for std in z‑score
    """
    if mode == "raw":
        return series

    if mode == "rank":
        return series.rank(ascending=False, pct=False)

    if mode == "percentile":
        if rank_pct is None:
            raise ValueError("rank_pct must be set for percentile transform")
        k = max(int(round(len(series) * rank_pct)), 1)
        mask = series.rank(ascending=False, pct=False) <= k
        return series.where(mask, np.nan)

    if mode == "zscore":
        if window is None or window > len(series):
            window = len(series)
        recent = series.iloc[-window:]
        mu = recent.mean()
        sigma = recent.std(ddof=ddof)
        return (series - mu) / sigma

    raise ValueError(f"unknown transform mode '{mode}'")


def rank_select_funds(
    in_sample_df: pd.DataFrame,
    stats_cfg: RiskStatsConfig,
    *,
    inclusion_approach: str = "top_n",
    transform: str = "raw",  # NEW
    zscore_window: int | None = None,
    rank_pct: float | None = None,
    n: int | None = None,
    pct: float | None = None,
    threshold: float | None = None,
    score_by: str = DEFAULT_METRIC,
    blended_weights: dict[str, float] | None = None,
) -> list[str]:
    """
    Central routine – returns the **ordered** list of selected funds.
    """
    if score_by == "blended":
        scores = blended_score(in_sample_df, blended_weights or {}, stats_cfg)
    else:
        scores = _compute_metric_series(in_sample_df, score_by, stats_cfg)

    scores = _apply_transform(
        scores,
        mode=transform,
        window=zscore_window,
        rank_pct=rank_pct,
    )

    ascending = score_by in ASCENDING_METRICS
    scores = scores.sort_values(ascending=ascending)

    if inclusion_approach == "top_n":
        if n is None:
            raise ValueError("top_n requires parameter n")
        return cast(list[str], scores.head(n).index.tolist())

    if inclusion_approach == "top_pct":
        if pct is None or not 0 < pct <= 1:
            raise ValueError("top_pct requires 0 < pct ≤ 1")
        k = max(1, int(round(len(scores) * pct)))
        return cast(list[str], scores.head(k).index.tolist())

    if inclusion_approach == "threshold":
        if threshold is None:
            raise ValueError("threshold approach requires a threshold value")
        mask = scores <= threshold if ascending else scores >= threshold
        return cast(list[str], scores[mask].index.tolist())

    raise ValueError(f"Unknown inclusion_approach '{inclusion_approach}'")


@dataclass
class FundSelectionConfig:
    """Simple quality-gate configuration."""

    max_missing_months: int = 3
    max_consecutive_month_gap: int = 6
    implausible_value_limit: float = 1.0
    outlier_threshold: float = 0.5
    zero_return_threshold: float = 0.2
    enforce_monotonic_index: bool = True
    allow_duplicate_dates: bool = False
    max_missing_ratio: float = 0.05
    max_drawdown: float = 0.3
    min_volatility: float = 0.05
    max_volatility: float = 1.0
    min_avg_return: float = 0.0
    max_skewness: float = 3.0
    max_kurtosis: float = 10.0
    expected_freq: str = "B"
    max_gap_days: int = 3
    min_aum_usd: float = 1e7


@dataclass
class RiskStatsConfig:
    """Metrics and risk free configuration."""

    metrics_to_run: List[str] = field(
        default_factory=lambda: [
            "AnnualReturn",
            "Volatility",
            "Sharpe",
            "Sortino",
            "MaxDrawdown",
            "InformationRatio"
        ]
    )
    risk_free: float = 0.0
    periods_per_year: int = 12


METRIC_REGISTRY: Dict[str, Callable[..., float]] = {}


def register_metric(
    name: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Register ``fn`` under ``name`` in :data:`METRIC_REGISTRY`."""

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


def _quality_filter(
    df: pd.DataFrame,
    fund_columns: List[str],
    in_sdate: str,
    out_edate: str,
    cfg: FundSelectionConfig,
) -> List[str]:
    """Return funds passing very basic data-quality gates."""

    mask = df["Date"].between(
        pd.Period(in_sdate, "M").to_timestamp("M"),
        pd.Period(out_edate, "M").to_timestamp("M"),
    )
    sub = df.loc[mask, fund_columns]
    eligible: List[str] = []
    for col in fund_columns:
        series = sub[col]
        missing = series.isna().sum()
        if missing > cfg.max_missing_months:
            continue
        if len(series) > 0 and missing / len(series) > cfg.max_missing_ratio:
            continue
        if series.abs().max() > cfg.implausible_value_limit:
            continue
        eligible.append(col)
    return eligible


# Register basic metrics from the public ``metrics`` module
register_metric("AnnualReturn")(
    lambda s, *, periods_per_year=12, **k:
        _metrics.annual_return(s, periods_per_year=periods_per_year)
)

register_metric("Volatility")(
    lambda s, *, periods_per_year=12, **k:
        _metrics.volatility(s, periods_per_year=periods_per_year)
)

register_metric("Sharpe")(
    lambda s, *, periods_per_year=12, risk_free=0.0:
        _metrics.sharpe_ratio(
            s,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )
)

register_metric("Sortino")(
    lambda s, *, periods_per_year=12, target=0.0, **k:
        _metrics.sortino_ratio(
            s,
            periods_per_year=periods_per_year,
            target=target,
        )
)

register_metric("MaxDrawdown")(
    lambda s, **k: _metrics.max_drawdown(s)
)

register_metric("InformationRatio")(
    lambda s, *, periods_per_year=12, benchmark=None, **k:
        _metrics.information_ratio(
            s,
            benchmark=benchmark if benchmark is not None else pd.Series(0, index=s.index),
            periods_per_year=periods_per_year,
        )
)

# ===============================================================
#  NEW: RANK‑BASED FUND SELECTION
# ===============================================================

ASCENDING_METRICS = {"MaxDrawdown"}  # smaller is better
DEFAULT_METRIC = "annual_return"


def _compute_metric_series(
    in_sample_df: pd.DataFrame, metric_name: str, stats_cfg: RiskStatsConfig
) -> pd.Series:
    """
    Return a pd.Series (index = fund code, value = metric score).
    Vectorised: uses the registered metric on each column.
    """
    fn = METRIC_REGISTRY.get(metric_name)
    if fn is None:
        raise ValueError(f"Metric '{metric_name}' not registered")
    # map across columns without Python loops
    return in_sample_df.apply(
        fn,
        periods_per_year=stats_cfg.periods_per_year,
        risk_free=stats_cfg.risk_free,
        axis=0,
    )


def _zscore(series: pd.Series) -> pd.Series:
    """Return z‑scores (mean 0, stdev 1).  Gracefully handles zero σ."""
    μ, σ = series.mean(), series.std(ddof=0)
    if σ == 0:
        return pd.Series(0.0, index=series.index)
    return (series - μ) / σ


def blended_score(
    in_sample_df: pd.DataFrame, weights: dict[str, float], stats_cfg: RiskStatsConfig
) -> pd.Series:
    """
    Z‑score each contributing metric, then weighted linear combo.
    """
    if not weights:
        raise ValueError("blended_score requires non‑empty weights dict")
    w_norm = {k: v / sum(weights.values()) for k, v in weights.items()}

    combo = pd.Series(0.0, index=in_sample_df.columns)
    for metric, w in w_norm.items():
        raw = _compute_metric_series(in_sample_df, metric, stats_cfg)
        z = _zscore(raw)
        # If metric is "smaller‑is‑better", *invert* before z‑score
        if metric in ASCENDING_METRICS:
            z *= -1
        combo += w * z
    return combo


# ===============================================================
#  WIRES INTO EXISTING PIPELINE
# ===============================================================


def select_funds(
    df: pd.DataFrame,
    rf_col: str,
    fund_columns: list[str],
    in_sdate: str,
    in_edate: str,
    out_sdate: str,
    out_edate: str,
    cfg: FundSelectionConfig,
    selection_mode: str = "all",
    random_n: int | None = None,
    rank_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Extended to honour 'rank' mode.  Existing modes unchanged.
    """
    # -- existing quality gate logic (unchanged) ------------------
    eligible = _quality_filter(  # pseudo‑factorised
        df, fund_columns, in_sdate, out_edate, cfg
    )

    # Fast‑exit for legacy modes
    if selection_mode == "all":
        return eligible

    if selection_mode == "random":
        if random_n is None:
            raise ValueError("random_n must be provided for random mode")
        return list(np.random.choice(eligible, random_n, replace=False))

    # >>> NEW  rank‑based mode
    if selection_mode == "rank":
        if rank_kwargs is None:
            raise ValueError("rank mode requires rank_kwargs")
        # carve out the in‑sample sub‑frame
        mask = df["Date"].between(
            pd.Period(in_sdate, "M").to_timestamp("M"),
            pd.Period(in_edate, "M").to_timestamp("M"),
        )
        in_df = df.loc[mask, eligible]
        stats_cfg = RiskStatsConfig(risk_free=0.0)  # N.B. rf handled upstream
        return rank_select_funds(in_df, stats_cfg, **rank_kwargs)

    raise ValueError(f"Unsupported selection_mode '{selection_mode}'")


# ===============================================================
#  UI SCAFFOLD (very condensed – Codex expands)
# ===============================================================


def build_ui() -> widgets.VBox:
    # -------------------- Step 1: data source & periods --------------------
    source_tb = widgets.ToggleButtons(
        options=["Path/URL", "Browse"],
        description="Source:",
    )
    csv_path = widgets.Text(description="CSV or URL:")
    file_up = widgets.FileUpload(accept=".csv", multiple=False)
    file_up.layout.display = "none"
    load_btn = widgets.Button(description="Load CSV", button_style="success")
    load_out = widgets.Output()

    in_start = widgets.Text(description="In Start:")
    in_end = widgets.Text(description="In End:")
    out_start = widgets.Text(description="Out Start:")
    out_end = widgets.Text(description="Out End:")

    session: dict[str, Any] = {"df": None, "rf": None}
    idx_select = widgets.SelectMultiple(options=[], description="Indices:")
    idx_select.layout.display = "none"
    step1_box = widgets.VBox(
        [
            source_tb,
            csv_path,
            file_up,
            load_btn,
            load_out,
            idx_select,
            in_start,
            in_end,
            out_start,
            out_end,
        ]
    )

    def _load_action(_btn: widgets.Button) -> None:
        with load_out:
            load_out.clear_output()
            try:
                df: pd.DataFrame | None = None
                if source_tb.value == "Browse":
                    if not file_up.value:
                        print("Upload a CSV")
                        return
                    # ipywidgets 7.x returns a dict; 8.x returns a tuple
                    if isinstance(file_up.value, dict):
                        item = next(iter(file_up.value.values()))
                    else:
                        item = file_up.value[0]
                    df = pd.read_csv(io.BytesIO(item["content"]))
                else:
                    path = csv_path.value.strip()
                    if not path:
                        print("Enter CSV path or URL")
                        return
                    if path.startswith("http://") or path.startswith("https://"):
                        df = pd.read_csv(path)
                    else:
                        df = load_csv(path)
                if df is None:
                    print("Failed to load")
                    return
                df = ensure_datetime(df)
                session["df"] = df
                rf = identify_risk_free_fund(df) or "RF"
                session["rf"] = rf
                dates = df["Date"].dt.to_period("M")
                in_start.value = str(dates.min())
                in_end.value = str(dates.min() + 2)
                out_start.value = str(dates.min() + 3)
                out_end.value = str(dates.min() + 5)
                idx_select.options = [c for c in df.columns if c not in {"Date", rf}]
                idx_select.layout.display = "flex"
                print(f"Loaded {len(df):,} rows")
            except Exception as exc:
                session["df"] = None
                print("Error:", exc)

    load_btn.on_click(_load_action)

    def _source_toggle(*_: Any) -> None:
        if source_tb.value == "Browse":
            file_up.layout.display = "flex"
            csv_path.layout.display = "none"
        else:
            file_up.layout.display = "none"
            csv_path.layout.display = "flex"

    source_tb.observe(_source_toggle, "value")
    _source_toggle()

    # -------------------- Step 2: selection & ranking ----------------------
    mode_dd = widgets.Dropdown(
        options=["all", "random", "manual", "rank"], description="Mode:"
    )
    random_n_int = widgets.BoundedIntText(value=8, min=1, description="Random N:")
    random_n_int.layout.display = "none"
    vol_ck = widgets.Checkbox(value=True, description="Vol‑adjust?")
    target_vol = widgets.BoundedFloatText(
        value=0.10, min=0.0, max=10.0, step=0.01, description="Target Vol:"
    )
    target_vol.layout.display = "none"
    use_rank_ck = widgets.Checkbox(value=False, description="Apply ranking?")
    next_btn_1 = widgets.Button(description="Next")

    # step‑2 widgets
    incl_dd = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"], description="Approach:"
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"], description="Score by:"
    )
    topn_int = widgets.BoundedIntText(value=10, min=1, description="N:")
    pct_flt = widgets.BoundedFloatText(
        value=0.10, min=0.01, max=1.0, step=0.01, description="Pct:"
    )
    thresh_f = widgets.FloatText(value=1.0, description="Threshold:")

    # blended area
    m1_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M1")
    w1_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m2_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M2")
    w2_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m3_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M3")
    w3_sl = widgets.FloatSlider(value=0.34, min=0, max=1.0, step=0.01)

    out_fmt = widgets.Dropdown(options=["excel", "csv", "json"], description="Output:")

    # --------------------------------------------------------------
    #  Callbacks and execution wiring
    # --------------------------------------------------------------

    blended_box = widgets.VBox([m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl])
    blended_box.layout.display = "none"

    rank_box = widgets.VBox(
        [incl_dd, metric_dd, topn_int, pct_flt, thresh_f, blended_box]
    )
    rank_box.layout.display = "none"

    def _update_random_vis(*_: Any) -> None:
        show = rank_unlocked and mode_dd.value == "random"
        random_n_int.layout.display = "flex" if show else "none"

    def _update_target_vol(*_: Any) -> None:
        target_vol.layout.display = "flex" if vol_ck.value else "none"

    manual_box = widgets.VBox()
    manual_box.layout.display = "none"
    manual_checks: list[widgets.Checkbox] = []
    manual_weights: list[widgets.FloatText] = []
    manual_total_lbl = widgets.Label("Total weight: 0 %")

    # track whether the user progressed past the first step
    rank_unlocked = False

    run_btn = widgets.Button(description="Run")
    output = widgets.Output()

    def _next_action(_: Any) -> None:
        nonlocal rank_unlocked
        if session["df"] is None:
            with load_out:
                load_out.clear_output()
                print("Load data first")
            return
        rank_unlocked = not rank_unlocked
        next_btn_1.layout.display = "none"
        _update_rank_vis()
        _update_random_vis()
        _update_manual()

    def _update_rank_vis(*_: Any) -> None:
        show = (
            rank_unlocked
            and session["df"] is not None
            and (mode_dd.value == "rank" or use_rank_ck.value)
        )
        rank_box.layout.display = "flex" if show else "none"
        _update_blended_vis()
        _update_manual()

    def _update_blended_vis(*_: Any) -> None:
        show = (
            rank_unlocked
            and metric_dd.value == "blended"
            and (mode_dd.value == "rank" or use_rank_ck.value)
        )
        blended_box.layout.display = "flex" if show else "none"

    def _update_manual(*_: Any) -> None:
        if mode_dd.value != "manual" or not rank_unlocked:
            manual_box.layout.display = "none"
            return
        df = session.get("df")
        if df is None:
            manual_box.children = [widgets.Label("Load data first")]
            manual_box.layout.display = "flex"
            return

        rf = session.get("rf", "RF")
        date_col = "Date"
        funds_all = [c for c in df.columns if c not in {date_col, rf}]
        try:

            def _to_dt(s: str) -> pd.Timestamp:
                return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

            in_start_dt = _to_dt(in_start.value)
            in_end_dt = _to_dt(in_end.value)
            out_start_dt = _to_dt(out_start.value)
            out_end_dt = _to_dt(out_end.value)

            in_df = df[(df[date_col] >= in_start_dt) & (df[date_col] <= in_end_dt)]
            out_df = df[(df[date_col] >= out_start_dt) & (df[date_col] <= out_end_dt)]
            in_ok = ~in_df[funds_all].isna().any()
            out_ok = ~out_df[funds_all].isna().any()
            funds = [c for c in funds_all if in_ok[c] and out_ok[c]]
        except Exception:
            funds = funds_all
        manual_checks.clear()
        manual_weights.clear()

        def _update_total(*_: Any) -> None:
            tot = sum(
                wt.value for chk, wt in zip(manual_checks, manual_weights) if chk.value
            )
            manual_total_lbl.value = f"Total weight: {tot:.0f} %"

        rows = []
        for f in funds:
            chk = widgets.Checkbox(value=False, description=f)
            wt = widgets.FloatText(value=0.0, layout=widgets.Layout(width="80px"))
            chk.observe(_update_total, "value")
            wt.observe(_update_total, "value")
            manual_checks.append(chk)
            manual_weights.append(wt)
            rows.append(widgets.HBox([chk, wt]))
        manual_box.children = rows + [manual_total_lbl]
        manual_box.layout.display = "flex"
        _update_total()

    def _update_inclusion_fields(*_: Any) -> None:
        topn_int.layout.display = "flex" if incl_dd.value == "top_n" else "none"
        pct_flt.layout.display = "flex" if incl_dd.value == "top_pct" else "none"
        thresh_f.layout.display = "flex" if incl_dd.value == "threshold" else "none"

    next_btn_1.on_click(_next_action)
    mode_dd.observe(_update_rank_vis, "value")
    mode_dd.observe(_update_random_vis, "value")
    use_rank_ck.observe(_update_rank_vis, "value")
    metric_dd.observe(_update_blended_vis, "value")
    incl_dd.observe(_update_inclusion_fields, "value")
    mode_dd.observe(_update_manual, "value")
    vol_ck.observe(_update_target_vol, "value")

    def _run_action(_btn: widgets.Button) -> None:
        rank_kwargs: dict[str, Any] | None = None
        if mode_dd.value == "rank" or use_rank_ck.value:
            rank_kwargs = {
                "inclusion_approach": incl_dd.value,
                "score_by": metric_dd.value,
            }
            if incl_dd.value == "top_n":
                rank_kwargs["n"] = int(topn_int.value)
            elif incl_dd.value == "top_pct":
                rank_kwargs["pct"] = float(pct_flt.value)
            elif incl_dd.value == "threshold":
                rank_kwargs["threshold"] = float(thresh_f.value)
            if metric_dd.value == "blended":
                rank_kwargs["blended_weights"] = {
                    m1_dd.value: w1_sl.value,
                    m2_dd.value: w2_sl.value,
                    m3_dd.value: w3_sl.value,
                }

        manual_funds: list[str] | None = None
        custom_weights: dict[str, float] | None = None
        if mode_dd.value == "manual":
            manual_funds = []
            custom_weights = {}
            for chk, wt in zip(manual_checks, manual_weights):
                if chk.value:
                    manual_funds.append(chk.description)
                    custom_weights[chk.description] = float(wt.value)

        with output:
            output.clear_output()
            try:
                from .. import pipeline, export

                df = session.get("df")
                if df is None:
                    print("Load data first")
                    return

                mode = mode_dd.value
                if mode_dd.value == "manual" and not custom_weights:
                    print("No funds selected")
                    return

                res = pipeline.run_analysis(
                    df,
                    in_start.value,
                    in_end.value,
                    out_start.value,
                    out_end.value,
                    target_vol.value if vol_ck.value else 1.0,
                    0.0,
                    selection_mode=mode,
                    random_n=int(random_n_int.value),
                    custom_weights=custom_weights,
                    rank_kwargs=rank_kwargs,
                    manual_funds=manual_funds,
                    indices_list=list(idx_select.value),
                )
                if res is None:
                    print("No results")
                else:
                    sheet_formatter = export.make_summary_formatter(
                        res,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    text = export.format_summary_text(
                        res,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    print(text)
                    data = {"summary": pd.DataFrame()}
                    prefix = f"IS_{in_start.value}_OS_{out_start.value}"
                    export.export_data(
                        data,
                        prefix,
                        formats=[out_fmt.value],
                        formatter=cast(Formatter, sheet_formatter),
                    )
            except Exception as exc:
                print("Error:", exc)

    run_btn.on_click(_run_action)

    ui = widgets.VBox(
        [
            step1_box,
            mode_dd,
            random_n_int,
            vol_ck,
            target_vol,
            use_rank_ck,
            next_btn_1,
            rank_box,
            manual_box,
            out_fmt,
            run_btn,
            output,
        ]
    )
    _update_rank_vis()
    _update_inclusion_fields()

    _update_random_vis()

    _update_manual()
    _update_target_vol()

    return ui


#  Once build_ui() is defined, the notebook can do:
#       ui_inputs = build_ui()
#       display(ui_inputs)


__all__ = [
    "FundSelectionConfig",
    "RiskStatsConfig",
    "register_metric",
    "METRIC_REGISTRY",
    "blended_score",
    "rank_select_funds",
    "select_funds",
    "build_ui",
]
