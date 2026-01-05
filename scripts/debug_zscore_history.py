#!/usr/bin/env python3
"""Debug script to show raw risk stats and z-scores for portfolio and top candidates.

Displays for each year from 2009-2019:
- Funds in portfolio with their risk stats and z-scores
- Top 5 candidates under consideration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_analysis.config import load
from trend_analysis.core.rank_selection import RiskStatsConfig, _compute_metric_series
from trend_analysis.data import load_csv


def compute_score_frame(returns_df: pd.DataFrame, funds: list[str]) -> pd.DataFrame:
    """Compute metrics for a list of funds from returns data."""
    available = [f for f in funds if f in returns_df.columns]
    if not available:
        return pd.DataFrame()

    stats_cfg = RiskStatsConfig(risk_free=0.0)
    metrics = [
        "AnnualReturn",
        "Volatility",
        "Sharpe",
        "Sortino",
        "InformationRatio",
        "MaxDrawdown",
    ]

    parts = []
    for m in metrics:
        try:
            part = _compute_metric_series(returns_df[available], m, stats_cfg)
            parts.append(part)
        except Exception as e:
            print(f"  Warning: Could not compute {m}: {e}")
            parts.append(pd.Series(np.nan, index=available))

    sf = pd.concat(parts, axis=1)
    sf.columns = ["CAGR", "Volatility", "Sharpe", "Sortino", "IR", "MaxDD"]
    return sf.astype(float)


def add_zscore(sf: pd.DataFrame, metric: str = "Sharpe") -> pd.DataFrame:
    """Add z-score column based on metric."""
    if sf.empty or metric not in sf.columns:
        return sf

    col = sf[metric].astype(float)
    mu = col.mean()
    sigma = col.std(ddof=0)

    if sigma == 0 or not np.isfinite(sigma):
        sf["zscore"] = 0.0
    else:
        sf["zscore"] = (col - mu) / sigma

    return sf


def parse_month(s: str) -> pd.Timestamp:
    """Convert YYYY-MM string to month-end timestamp."""
    return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)


def main():
    print("=" * 80)
    print("Z-SCORE AND RISK STATISTICS DEBUG REPORT")
    print("=" * 80)

    # Load config and data
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "demo.yml"
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "defaults.yml"

    print(f"\nLoading config from: {cfg_path}")
    cfg = load(str(cfg_path))

    # Load returns data
    csv_path = cfg.data.get("csv_path", "demo/demo_returns.csv")
    print(f"Loading data from: {csv_path}")
    df = load_csv(csv_path)

    # Ensure date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # Get fund columns (exclude common non-fund columns)
    exclude = {"Date", "RF", "Benchmark", "Index", "date", "rf", "benchmark"}
    fund_cols = [c for c in df.columns if c not in exclude and not c.startswith("_")]

    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"Total funds: {len(fund_cols)}")

    # Generate periods for 2009-2019
    multi_cfg = getattr(cfg, "multi_period", None) or {}
    frequency = multi_cfg.get("frequency", "A")
    in_sample_len = multi_cfg.get("in_sample_len", 3)
    out_sample_len = multi_cfg.get("out_sample_len", 1)

    print(
        f"\nPeriod settings: frequency={frequency}, in_sample={in_sample_len}, out_sample={out_sample_len}"
    )

    # We'll manually iterate years 2009-2019
    # For each year, compute in-sample stats using prior 3 years
    years_to_analyze = list(range(2009, 2020))

    # Track portfolio state
    portfolio: set[str] = set()
    z_exit_soft = -1.0
    z_entry_soft = 1.0
    max_funds = 10

    for year in years_to_analyze:
        print("\n" + "=" * 80)
        print(f"YEAR: {year}")
        print("=" * 80)

        # In-sample period: 3 years ending Dec of prior year
        in_end = f"{year - 1}-12"
        in_start = f"{year - in_sample_len}-01"

        in_start_ts = parse_month(in_start)
        in_end_ts = parse_month(in_end)

        print(f"\nIn-sample period: {in_start} to {in_end}")

        # Filter returns for in-sample period
        mask = (df.index >= in_start_ts) & (df.index <= in_end_ts)
        in_df = df.loc[mask, fund_cols].copy()

        if in_df.empty:
            print("  No data for this period")
            continue

        # Filter to funds with complete data
        min_obs = int(0.7 * len(in_df))  # Require 70% of observations
        valid_funds = [c for c in fund_cols if in_df[c].notna().sum() >= min_obs]

        print(f"  Observations: {len(in_df)}, Valid funds: {len(valid_funds)}")

        if not valid_funds:
            print("  No funds with sufficient data")
            continue

        # Compute score frame
        sf = compute_score_frame(in_df, valid_funds)
        sf = add_zscore(sf, "Sharpe")

        # Universe stats
        print("\n  Universe z-score stats:")
        print(f"    Mean Sharpe: {sf['Sharpe'].mean():.3f}")
        print(f"    Std Sharpe:  {sf['Sharpe'].std():.3f}")
        print(f"    Z-score range: [{sf['zscore'].min():.2f}, {sf['zscore'].max():.2f}]")

        # Seed portfolio if empty
        if not portfolio:
            top_n = sf.nlargest(max_funds, "zscore")
            portfolio = set(top_n.index)
            print(f"\n  SEEDING PORTFOLIO with top {len(portfolio)} funds")

        # Show portfolio funds
        portfolio_in_universe = [f for f in portfolio if f in sf.index]
        print(
            f"\n  PORTFOLIO FUNDS ({len(portfolio_in_universe)} of {len(portfolio)} in universe):"
        )
        print("-" * 75)
        print(
            f"  {'Fund':<20} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Z':>8}"
        )
        print("-" * 75)

        for fund in sorted(portfolio_in_universe, key=lambda f: sf.loc[f, "zscore"], reverse=True):
            row = sf.loc[fund]
            cagr = row["CAGR"] * 100
            vol = row["Volatility"] * 100
            sharpe = row["Sharpe"]
            sortino = row["Sortino"]
            maxdd = row["MaxDD"] * 100
            z = row["zscore"]

            # Flag if below exit threshold
            flag = " ⚠️" if z < z_exit_soft else ""
            print(
                f"  {fund:<20} {cagr:>7.1f}% {vol:>7.1f}% {sharpe:>8.2f} {sortino:>8.2f} {maxdd:>7.1f}% {z:>8.2f}{flag}"
            )

        # Show candidates not in portfolio
        candidates = [f for f in sf.index if f not in portfolio]
        if candidates:
            candidate_sf = sf.loc[candidates].nlargest(5, "zscore")

            print("\n  TOP 5 CANDIDATES (not in portfolio):")
            print("-" * 75)
            print(
                f"  {'Fund':<20} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Z':>8}"
            )
            print("-" * 75)

            for fund in candidate_sf.index:
                row = sf.loc[fund]
                cagr = row["CAGR"] * 100
                vol = row["Volatility"] * 100
                sharpe = row["Sharpe"]
                sortino = row["Sortino"]
                maxdd = row["MaxDD"] * 100
                z = row["zscore"]

                # Flag if above entry threshold
                flag = " ✅" if z >= z_entry_soft else ""
                print(
                    f"  {fund:<20} {cagr:>7.1f}% {vol:>7.1f}% {sharpe:>8.2f} {sortino:>8.2f} {maxdd:>7.1f}% {z:>8.2f}{flag}"
                )

        # Simulate rebalancing
        # Check for exits (z < -1.0)
        exits = [f for f in portfolio_in_universe if sf.loc[f, "zscore"] < z_exit_soft]
        if exits:
            print(f"\n  ⚠️ POTENTIAL EXITS (z < {z_exit_soft}): {exits}")
            # Note: actual exits require 2 consecutive periods below threshold

        # Check for entries (z > 1.0 and we have capacity)
        potential_entries = [
            f for f in candidates if f in sf.index and sf.loc[f, "zscore"] >= z_entry_soft
        ]
        if potential_entries and len(portfolio) < max_funds:
            print(f"\n  ✅ POTENTIAL ENTRIES (z >= {z_entry_soft}): {potential_entries[:3]}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
