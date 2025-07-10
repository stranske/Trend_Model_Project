"""Rolling multi-period execution engine."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING, cast

import pandas as pd
import numpy as np

from ..data import load_csv
from ..pipeline import (
    _run_analysis_period,
    _compute_stats,
    calc_portfolio_returns,
)
from .scheduler import generate_periods

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from ..config import Config


def run(cfg: "Config") -> Dict[str, object]:  # noqa: D401
    """Run multiple periods and aggregate the results."""

    csv_path = cfg.data.get("csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    df = load_csv(csv_path)
    if df is None:
        raise FileNotFoundError(csv_path)

    periods = generate_periods(cfg.model_dump())
    results: List[Dict[str, object]] = []

    def _month(s: str) -> str:
        return str(pd.Period(s).strftime("%Y-%m"))

    for p in periods:
        res = _run_analysis_period(
            df,
            _month(p.in_start),
            _month(p.in_end),
            _month(p.out_start),
            _month(p.out_end),
            cfg.vol_adjust.get("target_vol", 1.0),
            cfg.run.get("monthly_cost", 0.0),
            selection_mode=cfg.portfolio.get("selection_mode", "all"),
            random_n=cfg.portfolio.get("random_n", 8),
            custom_weights=cfg.portfolio.get("custom_weights"),
            rank_kwargs=cfg.portfolio.get("rank"),
            manual_funds=cfg.portfolio.get("manual_list"),
            indices_list=cfg.portfolio.get("indices_list"),
            benchmarks=cfg.benchmarks,
            seed=cfg.portfolio.get("random_seed", 42),
        )
        if res:
            res["period"] = p
            results.append(res)

    if not results:
        return {}

    # Concatenate returns for overall summary metrics
    out_frames = [r["out_sample_scaled"] for r in results]
    combined_out = pd.concat(out_frames)

    ret_cols = [c for c in df.columns if c != "Date"]
    rf_col = min(ret_cols, key=lambda c: df[c].std())
    rf_segments = []
    for p in periods:
        mask = (df["Date"] >= p.out_start) & (df["Date"] <= p.out_end)
        rf_segments.append(df.loc[mask, rf_col])
    rf_series = pd.concat(rf_segments)

    stats = _compute_stats(combined_out, rf_series)

    ew_returns = []
    user_returns = []
    for r in results:
        out_df = r["out_sample_scaled"]
        ew_w = np.array(
            list(cast(dict[str, float], r["ew_weights"]).values()), dtype=float
        )
        user_w = np.array(
            list(cast(dict[str, float], r["fund_weights"]).values()), dtype=float
        )
        ew_returns.append(calc_portfolio_returns(ew_w, out_df))
        user_returns.append(calc_portfolio_returns(user_w, out_df))
    ew_series = pd.concat(ew_returns)
    user_series = pd.concat(user_returns)

    summary = {
        "stats": stats,
        "ew": _compute_stats(pd.DataFrame({"ew": ew_series}), rf_series)["ew"],
        "user": _compute_stats(pd.DataFrame({"user": user_series}), rf_series)["user"],
    }

    return {"periods": results, "summary": summary}
