"""
YOU ARE CODEX.  EXTEND THE VOL_ADJ_TREND_ANALYSIS PROJECT AS FOLLOWS
--------------------------------------------------------------------

High‑level goal
~~~~~~~~~~~~~~~
Add a **performance‑based manager‑selection mode** that works alongside the
existing 'all', 'random', and 'manual' modes.  Make the pipeline fully
config‑driven and keep everything vectorised.

Functional spec
~~~~~~~~~~~~~~~
1.  New selection mode keyword:  `rank`.
    • Works on the *in‑sample* window **after** the usual data‑quality filters.  
    • Supported inclusion approaches:
         - `'top_n'`       – keep the N best funds.
         - `'top_pct'`     – keep the top P percent.  
         - `'threshold'`   – keep funds whose score ≥ user threshold
           (this is the “useful extra” beyond N and percentile).

2.  Rank criteria (`score_by`):
    • Any single metric registered in `METRIC_REGISTRY`
      (e.g. 'Sharpe', 'AnnualReturn', 'MaxDrawdown', …).  
    • Special value `'blended'` that combines up to three metrics with
      user‑supplied *positive* weights (weights will be normalised to 1.0).

3.  Direction‑of‑merit:
    • Metrics where “larger is better”  → rank descending.
    • Metrics where “smaller is better” (currently **only** MaxDrawdown)  
      → rank ascending.  Future metrics can extend `ASCENDING_METRICS`.

4.  Config file (YAML) drives everything – sample below.

5.  UI flow (ipywidgets, no external deps):
    Step 1  – Mode (‘all’, ‘random’, ‘manual’, **‘rank’**),
               checkboxes for “vol‑adj” and “use ranking”.  
    Step 2  – If mode == 'rank' **or** user ticked “use ranking”
               → reveal controls for `inclusion_approach`,
               `score_by`, `N / Pct / Threshold`, and (if blended)
               three sliders for weights + metric pickers.  
    Step 3  – If mode == 'manual'  
               → display an interactive DataFrame of the IS scores so the
               user can override selection and set weights.
    Step 4  – Output format picker (csv / xlsx / json) then fire
               `run_analysis()` and `export_to_*`.

6.  No broken changes:
    • Default behaviour (config absent) must be identical to current build.
    • All heavy computation stays in NumPy / pandas vector land.

7.  Unit‑test hooks:
    • New pure functions must be import‑safe and testable without widgets.
      (e.g. `rank_select_funds()`).

Sample YAML
~~~~~~~~~~~
selection:
  mode: rank               # all | random | manual | rank
  random_n: 12             # only if mode == random
  use_vol_adjust: true
rank:
  inclusion_approach: top_n     # top_n | top_pct | threshold
  n: 8                          # for top_n
  pct: 0.10                     # for top_pct  (decimal, not %)
  threshold: 1.5                # ignored unless approach == threshold
  score_by: blended             # Sharpe | AnnualReturn | … | blended
  blended_weights:
    Sharpe: 0.5
    AnnualReturn: 0.3
    MaxDrawdown: 0.2
output:
  format: excel                 # csv | excel | json
"""

"""
🔄 2025‑06‑15 UPDATE — PHASE‑1 ENHANCEMENTS
------------------------------------------
• Blended ranking **must** use *z‑scores* (mean‑0, stdev‑1) before the
  weighted sum so metrics on different scales are commensurable.
• MaxDrawdown is currently the only “smaller‑is‑better” metric; the
  ASCENDING_METRICS set remains {"MaxDrawdown"} until further notice.
• Config format stays YAML.



# ===============================================================
#  NEW: RANK‑BASED FUND SELECTION
# ===============================================================

ASCENDING_METRICS = {"MaxDrawdown"}   # smaller is better
DEFAULT_METRIC    = "Sharpe"

def _compute_metric_series(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig
) -> pd.Series:
    """
    Return a pd.Series (index = fund code, value = metric score).
    Vectorised: uses the registered metric on each column.
    """
    fn = METRIC_REGISTRY.get(metric_name)
    if fn is None:
        raise ValueError(f"Metric '{metric_name}' not registered")
    # map across columns without Python loops
    return in_sample_df.apply(fn, periods_per_year=stats_cfg.periods_per_year,
                              risk_free=stats_cfg.risk_free, axis=0)

# ---------------------------------------------------------------
#  Replace previous blended_score with z‑score version
# ---------------------------------------------------------------
def _zscore(series: pd.Series) -> pd.Series:
    """Return z‑scores (mean 0, stdev 1).  Gracefully handles zero σ."""
    μ, σ = series.mean(), series.std(ddof=0)
    if σ == 0:
        return pd.Series(0.0, index=series.index)
    return (series - μ) / σ

def blended_score(
    in_sample_df: pd.DataFrame,
    weights: dict[str, float],
    stats_cfg: RiskStatsConfig
) -> pd.Series:
    """
    Z‑score each contributing metric, then weighted linear combo.
    """
    if not weights:
        raise ValueError("blended_score requires non‑empty weights dict")
    w_norm = {k: v / sum(weights.values()) for k, v in weights.items()}

    combo = pd.Series(0.0, index=in_sample_df.columns)
    for metric, w in w_norm.items():
        raw   = _compute_metric_series(in_sample_df, metric, stats_cfg)
        z     = _zscore(raw)
        # If metric is "smaller‑is‑better", *invert* before z‑score
        if metric in ASCENDING_METRICS:
            z *= -1
        combo += w * z
    return combo

def rank_select_funds(
    in_sample_df: pd.DataFrame,
    stats_cfg: RiskStatsConfig,
    inclusion_approach: str = "top_n",
    n: int | None = None,
    pct: float | None = None,
    threshold: float | None = None,
    score_by: str = DEFAULT_METRIC,
    blended_weights: dict[str, float] | None = None
) -> list[str]:
    """
    Central routine – returns the **ordered** list of selected funds.
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
    rank_kwargs: dict | None = None
) -> list[str]:
    """
    Extended to honour 'rank' mode.  Existing modes unchanged.
    """
    # -- existing quality gate logic (unchanged) ------------------
    eligible = _quality_filter(                   # pseudo‑factorised
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
            pd.Period(in_edate, "M").to_timestamp("M")
        )
        in_df = df.loc[mask, eligible]
        stats_cfg = RiskStatsConfig(risk_free=0.0)  # N.B. rf handled upstream
        return rank_select_funds(in_df, stats_cfg, **rank_kwargs)

    raise ValueError(f"Unsupported selection_mode '{selection_mode}'")

# ===============================================================
#  UI SCAFFOLD (very condensed – Codex expands)
# ===============================================================

def build_ui():
    mode_dd     = widgets.Dropdown(options=["all", "random", "manual", "rank"],
                                   description="Mode:")
    vol_ck      = widgets.Checkbox(value=True, description="Vol‑adjust?")
    use_rank_ck = widgets.Checkbox(value=False,
                                   description="Apply ranking within mode?")
    next_btn_1  = widgets.Button(description="Next")

    # step‑2 widgets
    incl_dd   = widgets.Dropdown(options=["top_n", "top_pct", "threshold"],
                                 description="Approach:")
    metric_dd = widgets.Dropdown(options=list(METRIC_REGISTRY)+["blended"],
                                 description="Score by:")
    topn_int  = widgets.BoundedIntText(value=10, min=1, description="N:")
    pct_flt   = widgets.BoundedFloatText(value=0.10, min=0.01, max=1.0,
                                         step=0.01, description="Pct:")
    thresh_f  = widgets.FloatText(value=1.0, description="Threshold:")

    # blended area
    m1_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M1")
    w1_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m2_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M2")
    w2_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m3_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M3")
    w3_sl = widgets.FloatSlider(value=0.34, min=0, max=1.0, step=0.01)

    out_fmt = widgets.Dropdown(options=["excel", "csv", "json"],
                               description="Output:")

    # … callbacks wire visibility & final run_action that reads widgets
    # and assembles rank_kwargs / config dict, then calls run_analysis()
    # >>> IMPLEMENT wiring logic here

    ui = widgets.VBox([mode_dd, vol_ck, use_rank_ck, next_btn_1])
    return ui

#  Once build_ui() is defined, the notebook can do:
#       ui_inputs = build_ui()
#       display(ui_inputs)

