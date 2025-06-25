"""
RANK-BASED FUND SELECTION MODULE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds a fourth selection mode, *rank*, to the Trend Model project.

Key points
----------
* Inclusion approaches: ``top_n`` · ``top_pct`` · ``threshold``.
* Metrics: any in ``METRIC_REGISTRY`` **or** a z-scored *blended* metric
  (up to three components, user-weighted).
* Direction of merit: metrics in ``ASCENDING_METRICS`` are “smaller-is-better”.
* Pure functions only – no widget imports outside the minimal UI scaffold.
* Designed so the core functions can be reused by a future Streamlit/Dash
  front-end without modification.

See Agents.md §Rank Mode for the full spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ipywidgets as widgets

# These come from existing code-base -----------------------------
from trend_analysis.metrics import METRIC_REGISTRY, RiskStatsConfig
from trend_analysis.quality import _quality_filter, FundSelectionConfig

# ===============================================================
#  RANK-BASED FUND SELECTION CORE
# ===============================================================

ASCENDING_METRICS: set[str] = {"MaxDrawdown"}      # smaller is better
DEFAULT_METRIC: str = "Sharpe"


def _compute_metric_series(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig,
) -> pd.Series:
    """
    Returns a Series with the chosen *metric* evaluated column-wise.

    Parameters
    ----------
    in_sample_df : DataFrame
        Columns = funds, rows = return series (already filtered IS window)
    metric_name : str
        Key in ``METRIC_REGISTRY``.
    stats_cfg : RiskStatsConfig
        Contains annualisation and risk-free assumptions.

    Raises
    ------
    ValueError if the metric is not registered.
    """
    fn = METRIC_REGISTRY.get(metric_name)
    if fn is None:
        raise ValueError(f"Metric '{metric_name}' not registered")
    return in_sample_df.apply(
        fn,
        periods_per_year=stats_cfg.periods_per_year,
        risk_free=stats_cfg.risk_free,
        axis=0,
    )


# ---------------- blended score helpers ------------------------


def _zscore(series: pd.Series) -> pd.Series:
    """Return z-scores (μ = 0, σ = 1).  Handles zero-σ gracefully."""
    μ, σ = series.mean(), series.std(ddof=0)
    if σ == 0:
        return pd.Series(0.0, index=series.index)
    return (series - μ) / σ


def blended_score(
    in_sample_df: pd.DataFrame,
    weights: dict[str, float],
    stats_cfg: RiskStatsConfig,
) -> pd.Series:
    """
    Z-score each contributing metric, then form a weighted sum.

    ``weights`` may contain **up to three** metric names; they are internally
    normalised so the sum is 1.0.
    """
    if not weights:
        raise ValueError("blended_score requires non-empty weights dict")

    norm = {k: v / sum(weights.values()) for k, v in weights.items()}
    combo = pd.Series(0.0, index=in_sample_df.columns)

    for metric, w in norm.items():
        raw = _compute_metric_series(in_sample_df, metric, stats_cfg)
        z   = _zscore(raw)
        if metric in ASCENDING_METRICS:      # invert “smaller-is-better”
            z *= -1
        combo += w * z

    return combo


# ---------------- main selector --------------------------------


def rank_select_funds(
    in_sample_df: pd.DataFrame,
    stats_cfg: RiskStatsConfig,
    *,
    inclusion_approach: str = "top_n",
    n: int | None = None,
    pct: float | None = None,
    threshold: float | None = None,
    score_by: str = DEFAULT_METRIC,
    blended_weights: dict[str, float] | None = None,
) -> list[str]:
    """
    Returns an **ordered** list of funds selected by the ranking rules.

    Notes
    -----
    * ``ascending`` flag flips when the metric is “smaller-is-better”.
    * For ``top_pct`` the function guarantees at least one fund.
    * ``threshold`` keeps rows meeting ≥ (or ≤) the cut-off depending on
      metric direction.
    """
    if score_by == "blended":
        scores = blended_score(in_sample_df, blended_weights or {}, stats_cfg)
    else:
        scores = _compute_metric_series(in_sample_df, score_by, stats_cfg)

    ascending = score_by in ASCENDING_METRICS
    scores = scores.sort_values(ascending=ascending)

    if inclusion_approach == "top_n":
        if n is None:
            raise ValueError("top_n requires parameter n")
        return scores.head(n).index.tolist()

    if inclusion_approach == "top_pct":
        if pct is None or not 0 < pct <= 1:
            raise ValueError("top_pct requires 0 < pct ≤ 1")
        k = max(1, int(round(len(scores) * pct)))
        return scores.head(k).index.tolist()

    if inclusion_approach == "threshold":
        if threshold is None:
            raise ValueError("threshold approach requires a threshold value")
        mask = scores <= threshold if ascending else scores >= threshold
        return scores[mask].index.tolist()

    raise ValueError(f"Unknown inclusion_approach '{inclusion_approach}'")


# ===============================================================
#  WIRED INTO EXISTING PIPELINE
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
    *,
    selection_mode: str = "all",
    random_n: int | None = None,
    rank_kwargs: dict | None = None,
) -> list[str]:
    """
    Extends existing selector with 'rank' mode.  Other modes unchanged.
    """
    # -- quality gate -------------------------------------------
    eligible = _quality_filter(df, fund_columns, in_sdate, out_edate, cfg)

    if selection_mode == "all":
        return eligible

    if selection_mode == "random":
        if random_n is None:
            raise ValueError("random_n must be provided for random mode")
        return list(np.random.choice(eligible, random_n, replace=False))

    if selection_mode == "rank":
        if rank_kwargs is None:
            raise ValueError("rank mode requires rank_kwargs")
        mask = df["Date"].between(
            pd.Period(in_sdate, "M").to_timestamp("M"),
            pd.Period(in_edate, "M").to_timestamp("M"),
        )
        in_df = df.loc[mask, eligible]
        stats_cfg = RiskStatsConfig(risk_free=0.0)  # rf handled upstream
        return rank_select_funds(in_df, stats_cfg, **rank_kwargs)

    raise ValueError(f"Unsupported selection_mode '{selection_mode}'")


# ===============================================================
#  ULTRA-LITE WIDGET SCAFFOLD (for notebook only)
# ===============================================================


def build_ui() -> widgets.VBox:
    """
    Minimal ipywidgets flow to prototype the new mode.

    The callbacks are left TODO so Codex (or you) can wire them as needed.
    """
    mode_dd     = widgets.Dropdown(
        options=["all", "random", "manual", "rank"], description="Mode:"
    )
    vol_ck      = widgets.Checkbox(value=True, description="Vol-adjust?")
    use_rank_ck = widgets.Checkbox(
        value=False, description="Apply ranking within mode?"
    )
    next_btn_1  = widgets.Button(description="Next")

    # -- step-2 widgets ----------------------------------------
    incl_dd   = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"], description="Approach:"
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"], description="Score by:"
    )
    topn_int  = widgets.BoundedIntText(value=10, min=1, description="N:")
    pct_flt   = widgets.BoundedFloatText(
        value=0.10, min=0.01, max=1.0, step=0.01, description="Pct:"
    )
    thresh_f  = widgets.FloatText(value=1.0, description="Threshold:")

    # blended controls
    m1_dd, m2_dd, m3_dd = (widgets.Dropdown(options=list(METRIC_REGISTRY)) for _ in range(3))
    w1_sl, w2_sl, w3_sl = (widgets.FloatSlider(value=v, min=0, max=1.0, step=0.01)
                           for v in (0.33, 0.33, 0.34))

    out_fmt = widgets.Dropdown(
        options=["excel", "csv", "json"], description="Output:"
    )

    # TODO: implement visibility toggles & run callback ------------
    # >>> IMPLEMENT wiring logic here

    return widgets.VBox(
        [mode_dd, vol_ck, use_rank_ck, next_btn_1]  # + later stage boxes
    )
