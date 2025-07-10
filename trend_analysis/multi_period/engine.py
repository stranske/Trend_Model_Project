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
    in_frames = [r["in_sample_scaled"] for r in results]
    combined_in = pd.concat(in_frames)

    ret_cols = [c for c in df.columns if c != "Date"]
    rf_col = min(ret_cols, key=lambda c: df[c].std())

    rf_out_segments = []
    rf_in_segments = []
    for p in periods:
        mask_out = (df["Date"] >= p.out_start) & (df["Date"] <= p.out_end)
        seg_out = df.loc[mask_out, ["Date", rf_col]].set_index("Date")[rf_col]
        rf_out_segments.append(seg_out)
        mask_in = (df["Date"] >= p.in_start) & (df["Date"] <= p.in_end)
        seg_in = df.loc[mask_in, ["Date", rf_col]].set_index("Date")[rf_col]
        rf_in_segments.append(seg_in)

    rf_out_series = pd.concat(rf_out_segments)
    rf_in_series = pd.concat(rf_in_segments)

    out_stats = _compute_stats(combined_out, rf_out_series)
    in_stats = _compute_stats(combined_in, rf_in_series)

    ew_returns_out = []
    ew_returns_in = []
    user_returns_out = []
    user_returns_in = []
    for r in results:
        out_df = r["out_sample_scaled"]
        in_df = r["in_sample_scaled"]
        ew_w = np.array(
            list(cast(dict[str, float], r["ew_weights"]).values()), dtype=float
        )
        user_w = np.array(
            list(cast(dict[str, float], r["fund_weights"]).values()), dtype=float
        )
        ew_returns_out.append(calc_portfolio_returns(ew_w, out_df))
        user_returns_out.append(calc_portfolio_returns(user_w, out_df))
        ew_returns_in.append(calc_portfolio_returns(ew_w, in_df))
        user_returns_in.append(calc_portfolio_returns(user_w, in_df))

    ew_series_out = pd.concat(ew_returns_out)
    user_series_out = pd.concat(user_returns_out)
    ew_series_in = pd.concat(ew_returns_in)
    user_series_in = pd.concat(user_returns_in)

    summary = {
        "stats": out_stats,
        "in_sample_stats": in_stats,
        "out_sample_stats": out_stats,
        "in_ew_stats": _compute_stats(pd.DataFrame({"ew": ew_series_in}), rf_in_series)[
            "ew"
        ],
        "out_ew_stats": _compute_stats(
            pd.DataFrame({"ew": ew_series_out}), rf_out_series
        )["ew"],
        "in_user_stats": _compute_stats(
            pd.DataFrame({"user": user_series_in}), rf_in_series
        )["user"],
        "out_user_stats": _compute_stats(
            pd.DataFrame({"user": user_series_out}), rf_out_series
        )["user"],
        "fund_weights": results[-1]["fund_weights"],
        "benchmark_ir": {},
    }

    return {"periods": results, "summary": summary}
