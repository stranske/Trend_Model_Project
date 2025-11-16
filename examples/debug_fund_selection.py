#!/usr/bin/env python3
"""Debug the fund selection process - why are only Mgr_01-08 considered?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_analysis.config import load
from trend_analysis.core.rank_selection import RiskStatsConfig, rank_select_funds
from trend_analysis.data import identify_risk_free_fund, load_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "config" / "portfolio_test.yml"


def debug_fund_selection():
    """Debug why only first 8 managers are being selected."""

    print("=" * 70)
    print("DEBUGGING FUND SELECTION PROCESS")
    print("=" * 70)

    # Load data
    cfg = load(str(DEFAULT_CFG))
    df = load_csv(cfg.data["csv_path"])

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Simulate the first period selection process
    # This is what _run_analysis does internally

    # Define first period (same as our test showed)
    in_start = "2008-01"
    in_end = "2010-12"

    # Convert to timestamps like _run_analysis does
    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    in_sdate = _parse_month(in_start)
    in_edate = _parse_month(in_end)
    print(f"\nAnalyzing period: {in_sdate} to {in_edate}")

    # Filter data for in-sample period
    date_col = "Date"
    df[date_col] = pd.to_datetime(df[date_col])

    in_df = df[(df[date_col] >= in_sdate) & (df[date_col] <= in_edate)].set_index(
        date_col
    )
    print(f"In-sample data shape: {in_df.shape}")

    # Identify return columns (this is the key part!)
    ret_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != date_col
    ]
    print(f"Return columns found: {len(ret_cols)}")
    print(f"Return columns: {ret_cols}")

    # Identify risk-free fund
    rf_col = identify_risk_free_fund(df)
    print(f"Risk-free column identified: {rf_col}")

    # Get fund columns (excluding risk-free)
    fund_cols = [c for c in ret_cols if c != rf_col]
    print(f"Fund columns (excluding RF): {len(fund_cols)}")
    print(f"Fund columns: {fund_cols}")

    # Check for complete data (this might be the issue!)
    print("\nChecking for complete data in in-sample period:")
    in_ok = ~in_df[fund_cols].isna().any()
    print("Missing data check:")
    for col in fund_cols:
        missing = in_df[col].isna().sum()
        total = len(in_df)
        pct = (missing / total) * 100 if total > 0 else 0
        status = "OK" if in_ok[col] else "MISSING DATA"
        print(f"  {col}: {missing}/{total} missing ({pct:.1f}%) - {status}")
    # This is the critical filter that might be removing funds
    valid_fund_cols = [c for c in fund_cols if in_ok[c]]
    print(f"\nValid fund columns after missing data filter: {len(valid_fund_cols)}")
    print(f"Valid funds: {valid_fund_cols}")

    if len(valid_fund_cols) < len(fund_cols):
        removed = set(fund_cols) - set(valid_fund_cols)
        print(f"REMOVED due to missing data: {removed}")

    # Check out-sample data too
    out_start = "2011-01"
    out_end = "2011-12"
    out_sdate = _parse_month(out_start)
    out_edate = _parse_month(out_end)

    out_df = df[(df[date_col] >= out_sdate) & (df[date_col] <= out_edate)].set_index(
        date_col
    )
    print(f"\nOut-sample data shape: {out_df.shape}")
    out_ok = ~out_df[fund_cols].isna().any()
    print("Out-sample missing data check:")
    for col in fund_cols:
        missing = out_df[col].isna().sum()
        total = len(out_df)
        pct = (missing / total) * 100 if total > 0 else 0
        status = "OK" if out_ok[col] else "MISSING DATA"
        print(f"  {col}: {missing}/{total} missing ({pct:.1f}%) - {status}")
    # Final filter: both in-sample AND out-sample must be complete
    final_fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]
    print(f"\nFinal fund columns after both filters: {len(final_fund_cols)}")
    print(f"Final funds: {final_fund_cols}")

    if len(final_fund_cols) < len(fund_cols):
        removed = set(fund_cols) - set(final_fund_cols)
        print(f"TOTAL REMOVED: {removed}")
        print("Removal reasons:")
        for col in removed:
            reasons = []
            if not in_ok[col]:
                in_missing = in_df[col].isna().sum()
                reasons.append(f"in-sample missing: {in_missing}")
            if not out_ok[col]:
                out_missing = out_df[col].isna().sum()
                reasons.append(f"out-sample missing: {out_missing}")
            print(f"  {col}: {', '.join(reasons)}")

    # Now test the ranking selection
    print("\n" + "=" * 50)
    print("TESTING RANKING SELECTION")
    print("=" * 50)

    if len(final_fund_cols) > 0:
        # Create the in-sample window for ranking
        sub = df.loc[
            (df[date_col] >= in_sdate) & (df[date_col] <= in_edate),
            [date_col, *final_fund_cols],
        ].set_index(date_col)

        print(f"Ranking data shape: {sub.shape}")
        print(f"Ranking period: {in_sdate} to {in_edate}")

        # Test rank selection with our config
        rank_kwargs = cfg.portfolio.get("rank", {})
        print(f"Rank kwargs: {rank_kwargs}")

        stats_cfg = RiskStatsConfig(risk_free=0.0)
        # This is the actual selection call
        selected = rank_select_funds(sub, stats_cfg, **rank_kwargs)
        print(f"Selected funds: {selected}")
        print(f"Selected count: {len(selected)}")

        # Show what got ranked
        print("\nShowing all available funds and their Sharpe ratios:")
        for col in final_fund_cols:
            returns = sub[col].dropna()
            if len(returns) > 1:
                excess_returns = returns - 0.0  # risk-free rate is 0
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
                selected_status = "SELECTED" if col in selected else "not selected"
                print(f"  {col}: {sharpe:6.2f} Sharpe - {selected_status}")
    else:
        print("No valid funds available for ranking.")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if len(final_fund_cols) == 8:
        print("❌ PROBLEM IDENTIFIED: Only 8 funds have complete data!")
        print("   The other 12 managers have missing data in some periods.")
        print(
            "   This explains why selection doesn't change - there are no alternatives!"
        )
    elif len(final_fund_cols) > 8:
        print("✅ Data issue ruled out - multiple funds available")
        print("   The problem is likely in the ranking/selection logic")
    else:
        print(f"❌ CRITICAL: Only {len(final_fund_cols)} funds available!")


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    debug_fund_selection()
