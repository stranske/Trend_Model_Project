#!/usr/bin/env python3
"""Run the real multi-period model on the real CSV.

Hires/fires each period via RankSelector and re-weights by performance
using ScorePropBayesian. Exports the per-period weight schedule and a
stitched out-of-sample portfolio return series.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import pandas as pd

from trend_analysis.config import load
from trend_analysis.config.models import ConfigProtocol
from trend_analysis.data import load_csv
from trend_analysis.logging_setup import setup_logging
from trend_analysis.multi_period import run as run_mp
from trend_analysis.multi_period import run_schedule
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import ScorePropBayesian


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(cfg_path: str = "config/long_backtest.yml") -> int:
    log_path = setup_logging(app_name="run_real_model")
    logging.getLogger(__name__).info("Log file initialised at %s", log_path)

    cfg: ConfigProtocol = load(cfg_path)
    csv_path_obj = cfg.data.get("csv_path")
    if not isinstance(csv_path_obj, str):
        raise KeyError("cfg.data['csv_path'] must be provided")
    csv_path = csv_path_obj

    # Load data once for RF stitching later
    df_all = load_csv(csv_path)
    if df_all is None:
        raise FileNotFoundError(csv_path)
    df_all = df_all.copy()
    df_all["Date"] = pd.to_datetime(df_all["Date"])  # ensure dtype
    df_all.set_index("Date", inplace=True)

    # Run multi-period to get score_frames and out-of-sample returns per period
    results_raw = run_mp(cfg)
    results: List[Dict[str, Any]] = [cast(Dict[str, Any], res) for res in results_raw]
    if not results:
        print("No periods generated")
        return 0

    # Build score_frame map keyed by OOS start (hire/fire at rebalance date)
    score_frames: Dict[str, pd.DataFrame] = {}
    for res in results:
        period = cast(Sequence[str], res.get("period"))
        if len(period) < 3:
            raise ValueError("Result period tuple is incomplete")
        out_start = str(period[2])
        sf_obj = res.get("score_frame")
        if not isinstance(sf_obj, pd.DataFrame):
            raise ValueError("score_frame must be a DataFrame")
        score_frames[out_start] = sf_obj.astype(float)

    # Selector and performance-based weighting
    rank_cfg = cast(Dict[str, Any], cfg.portfolio.get("rank", {}))
    top_n = int(rank_cfg.get("n", 8))
    rank_col = str(rank_cfg.get("score_by", "Sharpe"))
    selector = RankSelector(top_n=top_n, rank_column=rank_col)
    weighting_cfg = cast(Dict[str, Any], cfg.portfolio.get("weighting", {}))
    params_cfg = cast(Dict[str, Any], weighting_cfg.get("params", {}))
    shrink_tau = float(params_cfg.get("shrink_tau", 0.25))
    weighting = ScorePropBayesian(column=rank_col, shrink_tau=shrink_tau)

    # Generate the hire/fire schedule and weights
    pf = run_schedule(score_frames, selector, weighting)

    # Export long weight table: date, fund, weight
    out_dir = Path(cfg.export.get("directory", "results/real_model"))
    _ensure_dir(out_dir)
    long_rows: list[dict[str, object]] = []
    for date, w in sorted(pf.history.items()):
        for fund, wt in w.items():
            long_rows.append({"date": date, "fund": fund, "weight": float(wt)})
    weights_df = pd.DataFrame(long_rows).sort_values(["date", "fund"])  # stable
    weights_path = out_dir / "weights_schedule.csv"
    weights_df.to_csv(weights_path, index=False)

    # Compute stitched OOS portfolio returns using per-period out_sample_scaled
    port_series: list[pd.Series] = []
    for res in results:
        in_s, in_e, out_s, out_e = cast(Sequence[str], res.get("period"))
        out_df_obj = res.get("out_sample_scaled")
        if not isinstance(out_df_obj, pd.DataFrame):
            raise ValueError("out_sample_scaled must be a DataFrame")
        out_df = out_df_obj
        # Weight vector at the rebalance date (OOS start)
        rebalance_key = str(pd.to_datetime(out_s).date())
        weight_series = pf.history.get(rebalance_key)
        if weight_series is None:
            continue
        w = weight_series
        # Align and compute weighted return
        cols = [c for c in out_df.columns if c in w.index]
        if not cols:
            continue
        returns = out_df[cols]
        wv = w.loc[cols].astype(float)
        port = returns.mul(wv, axis=1).sum(axis=1)
        port_series.append(port)

    if port_series:
        portfolio = pd.concat(port_series).sort_index()
        portfolio.name = "portfolio_return"
        port_path = out_dir / "portfolio_oos_returns.csv"
        portfolio.to_csv(port_path, index_label="Date")
        # Basic stats
        from trend_analysis.metrics import annual_return, sharpe_ratio, volatility

        rf_series = df_all.get("Risk-Free Rate")
        rf_aligned = rf_series.loc[portfolio.index] if rf_series is not None else 0.0
        cagr = annual_return(portfolio)
        vol = volatility(portfolio)
        sr = sharpe_ratio(portfolio, rf_aligned)
        msg = (
            f"OOS CAGR: {cagr*100:.2f}%  " f"Vol: {vol*100:.2f}%  " f"Sharpe: {sr:.2f}"
        )
        print(msg)
        print(f"Weights: {weights_path}")
        print(f"OOS returns: {port_path}")
    else:
        print("No portfolio returns generated (no overlapping selections)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
