"""Core analysis functions for the trend model project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from trend_analysis.metrics import (
    annualize_return,
    annualize_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)


@dataclass
class Stats:
    """Container for performance metrics."""

    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float

def calc_portfolio_returns(weights: np.ndarray, returns_df: pd.DataFrame) -> pd.Series:
    """Calculate weighted portfolio returns."""
    return returns_df.mul(weights, axis=1).sum(axis=1)


def _compute_stats(df: pd.DataFrame, rf: pd.Series) -> Dict[str, Stats]:
    stats = {}
    for col in df:
        stats[col] = Stats(
            cagr=annualize_return(df[col]),
            vol=annualize_volatility(df[col]),
            sharpe=sharpe_ratio(df[col], rf),
            sortino=sortino_ratio(df[col], rf),
            max_drawdown=max_drawdown(df[col]),
        )
    return stats


def run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float,
    monthly_cost: float,
    selection_mode: str = "all",
    random_n: int = 8,
    custom_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Optional[Dict[str, object]]:
    """Run the trend-following analysis on ``df``.

    Parameters
    ----------
    df:
        DataFrame with a ``Date`` column and return columns.
    in_start, in_end, out_start, out_end:
        Analysis date range in ``YYYY-MM`` format.
    target_vol:
        Target annualised volatility for scaling returns.
    monthly_cost:
        Cost deducted from scaled returns each period.
    selection_mode:
        ``"all"`` to use every fund or ``"random"`` for a random subset.
    random_n:
        Number of funds if ``selection_mode='random'``.
    custom_weights:
        Optional mapping of fund → weight percentage for the user portfolio.
    seed:
        Random seed when sampling funds.

    Returns
    -------
    dict or None
        Analysis results dictionary, or ``None`` if no funds are selected.

    Example
    -------
    >>> df = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=3, freq="M"),
    ...                    "RF": 0.0, "A": 0.01, "B": 0.02})
    >>> run_analysis(df, "2020-01", "2020-02", "2020-03", "2020-03", 0.1, 0.0)
    {'selected_funds': ['A', 'B'], ...}
    """
    if df is None:
        return None

    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    in_sdate, in_edate = _parse_month(in_start), _parse_month(in_end)
    out_sdate, out_edate = _parse_month(out_start), _parse_month(out_end)

    in_df = df[(df[date_col] >= in_sdate) & (df[date_col] <= in_edate)]
    out_df = df[(df[date_col] >= out_sdate) & (df[date_col] <= out_edate)]

    if in_df.empty or out_df.empty:
        return None

    ret_cols = [c for c in df.columns if c != date_col]
    rf_col = min(ret_cols, key=lambda c: df[c].std())
    fund_cols = [c for c in ret_cols if c != rf_col]

    if selection_mode == "random" and len(fund_cols) > random_n:
        rng = np.random.default_rng(seed)
        fund_cols = rng.choice(fund_cols, size=random_n, replace=False).tolist()

    if not fund_cols:
        return None

    vols = in_df[fund_cols].std() * np.sqrt(12)
    scale_factors = (
        pd.Series(target_vol / vols, index=fund_cols)
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    in_scaled = in_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    out_scaled = out_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    in_scaled = in_scaled.clip(lower=-1.0)
    out_scaled = out_scaled.clip(lower=-1.0)

    rf_in = in_df[rf_col]
    rf_out = out_df[rf_col]

    in_stats = _compute_stats(in_scaled, rf_in)
    out_stats = _compute_stats(out_scaled, rf_out)
    out_stats_raw = _compute_stats(out_df[fund_cols], rf_out)

    ew_weights = np.repeat(1.0 / len(fund_cols), len(fund_cols))
    ew_w_dict = {c: w for c, w in zip(fund_cols, ew_weights)}
    in_ew = calc_portfolio_returns(ew_weights, in_scaled)
    out_ew = calc_portfolio_returns(ew_weights, out_scaled)
    out_ew_raw = calc_portfolio_returns(ew_weights, out_df[fund_cols])

    in_ew_stats = _compute_stats(pd.DataFrame({"ew": in_ew}), rf_in)["ew"]
    out_ew_stats = _compute_stats(pd.DataFrame({"ew": out_ew}), rf_out)["ew"]
    out_ew_stats_raw = _compute_stats(pd.DataFrame({"ew": out_ew_raw}), rf_out)["ew"]

    if custom_weights is None:
        custom_weights = {c: 100 / len(fund_cols) for c in fund_cols}
    user_w = np.array([custom_weights.get(c, 0) / 100 for c in fund_cols])
    user_w_dict = {c: w for c, w in zip(fund_cols, user_w)}

    in_user = calc_portfolio_returns(user_w, in_scaled)
    out_user = calc_portfolio_returns(user_w, out_scaled)
    out_user_raw = calc_portfolio_returns(user_w, out_df[fund_cols])

    in_user_stats = _compute_stats(pd.DataFrame({"user": in_user}), rf_in)["user"]
    out_user_stats = _compute_stats(pd.DataFrame({"user": out_user}), rf_out)["user"]
    out_user_stats_raw = _compute_stats(pd.DataFrame({"user": out_user_raw}), rf_out)[
        "user"
    ]

    return {
        "selected_funds": fund_cols,
        "in_sample_scaled": in_scaled,
        "out_sample_scaled": out_scaled,
        "in_sample_stats": in_stats,
        "out_sample_stats": out_stats,
        "out_sample_stats_raw": out_stats_raw,
        "in_ew_stats": in_ew_stats,
        "out_ew_stats": out_ew_stats,
        "out_ew_stats_raw": out_ew_stats_raw,
        "in_user_stats": in_user_stats,
        "out_user_stats": out_user_stats,
        "out_user_stats_raw": out_user_stats_raw,
        "ew_weights": ew_w_dict,
        "fund_weights": user_w_dict,
    }
