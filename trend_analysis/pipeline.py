from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from .config import Config
from .data import load_csv
from .metrics import (
    annualize_return,
    annualize_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from .core.rank_selection import rank_select_funds, RiskStatsConfig


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
    # Metrics expect 1D Series; iterating keeps the logic simple for a handful
    # of columns and avoids reshaping into higher-dimensional arrays.
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


def _run_analysis(
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
    rank_kwargs: Optional[Dict[str, Any]] = None,
    manual_funds: Optional[list[str]] = None,
    seed: int = 42,
) -> Optional[Dict[str, object]]:
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
    elif selection_mode == "rank":
        mask = (df[date_col] >= in_sdate) & (df[date_col] <= in_edate)
        sub = df.loc[mask, fund_cols]
        stats_cfg = RiskStatsConfig(risk_free=0.0)
        fund_cols = rank_select_funds(sub, stats_cfg, **(rank_kwargs or {}))
    elif selection_mode == "manual" and manual_funds:
        fund_cols = [c for c in fund_cols if c in manual_funds]

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
    rank_kwargs: Optional[Dict[str, Any]] = None,
    manual_funds: Optional[list[str]] = None,
    seed: int = 42,
) -> Optional[Dict[str, object]]:
    """Backward-compatible wrapper around ``_run_analysis``."""
    return _run_analysis(
        df,
        in_start,
        in_end,
        out_start,
        out_end,
        target_vol,
        monthly_cost,
        selection_mode,
        random_n,
        custom_weights,
        rank_kwargs,
        manual_funds,
        seed,
    )


def run(cfg: Config) -> pd.DataFrame:
    """Execute the analysis pipeline based on ``cfg``."""
    csv_path = cfg.data.get("csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    df = load_csv(csv_path)
    if df is None:
        raise FileNotFoundError(csv_path)

    split = cfg.sample_split
    res = _run_analysis(
        df,
        cast(str, split.get("in_start")),
        cast(str, split.get("in_end")),
        cast(str, split.get("out_start")),
        cast(str, split.get("out_end")),
        cfg.vol_adjust.get("target_vol", 1.0),
        cfg.run.get("monthly_cost", 0.0),
        selection_mode=cfg.portfolio.get("selection_mode", "all"),
        random_n=cfg.portfolio.get("random_n", 8),
        custom_weights=cfg.portfolio.get("custom_weights"),
        rank_kwargs=cfg.portfolio.get("rank"),
        manual_funds=cfg.portfolio.get("manual_list"),
        seed=cfg.portfolio.get("random_seed", 42),
    )
    if res is None:
        return pd.DataFrame()
    stats = cast(dict[str, Stats], res["out_sample_stats"])
    return pd.DataFrame({k: vars(v) for k, v in stats.items()}).T


__all__ = ["Stats", "calc_portfolio_returns", "run_analysis", "run"]
