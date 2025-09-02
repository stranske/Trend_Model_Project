#!/usr/bin/env python3
"""Run the real multi-period model on the real CSV.

Hires/fires each period via RankSelector and re-weights by performance
using ScorePropBayesian. Exports the per-period weight schedule and a
stitched out-of-sample portfolio return series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from trend_analysis.config import load, Config
from trend_analysis.data import load_csv
from trend_analysis.multi_period import run as run_mp, run_schedule
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import ScorePropBayesian


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main(cfg_path: str = "config/long_backtest.yml") -> int:
    cfg: Config = load(cfg_path)
    csv_path = cfg.data.get("csv_path")
    if not csv_path:
        raise KeyError("cfg.data['csv_path'] must be provided")

    # Load data once for RF stitching later
    df_all = load_csv(csv_path)
    if df_all is None:
        raise FileNotFoundError(csv_path)
    df_all = df_all.copy()
    df_all["Date"] = pd.to_datetime(df_all["Date"])  # ensure dtype
    df_all.set_index("Date", inplace=True)

    # Run multi-period to get score_frames and out-of-sample returns per period
    results: List[Dict[str, object]] = run_mp(cfg)
    if not results:
        print("No periods generated")
        return 0

    # Build score_frame map keyed by OOS start (hire/fire at rebalance date)
    score_frames: Dict[str, pd.DataFrame] = {}
    for res in results:
        period = res["period"]  # (in_start, in_end, out_start, out_end)
        out_start = str(period[2])  # type: ignore[index]
        sf = res["score_frame"]  # type: ignore[index]
        assert isinstance(sf, pd.DataFrame)
        score_frames[out_start] = sf.astype(float)

    # Selector and performance-based weighting
    rank_cfg = cfg.portfolio.get("rank", {})
    top_n = int(rank_cfg.get("n", 8))
    rank_col = str(rank_cfg.get("score_by", "Sharpe"))
    selector = RankSelector(top_n=top_n, rank_column=rank_col)
    shrink_tau = float(
        cfg.portfolio.get("weighting", {}).get("params", {}).get("shrink_tau", 0.25)
    )
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
        in_s, in_e, out_s, out_e = res["period"]  # type: ignore[index]
        out_df = res["out_sample_scaled"]  # type: ignore[index]
        assert isinstance(out_df, pd.DataFrame)
        # Weight vector at the rebalance date (OOS start)
        w = pf.history[str(pd.to_datetime(out_s).date())]
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
        from trend_analysis.metrics import annual_return, volatility, sharpe_ratio

        rf = df_all.get("Risk-Free Rate")
        rf = rf.loc[portfolio.index] if rf is not None else 0.0
        cagr = annual_return(portfolio)
        vol = volatility(portfolio)
        sr = sharpe_ratio(portfolio, rf)
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
