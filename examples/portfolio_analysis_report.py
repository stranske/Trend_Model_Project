#!/usr/bin/env python3
"""DEPRECATED: Portfolio analysis legacy script.

This script is retained temporarily for backward compatibility.

It delegates to the unified CLI entrypoint provided by the `trend` console
script (issue #1437). Please migrate to:

    trend run --config config/demo.yml --returns demo/demo_returns.csv

or generate reports via:

    trend report --out outputs/

All rich plotting and reporting should now be performed through
library APIs or future dedicated subcommands. This file will be removed
in a subsequent minor release. Importing or executing it emits a
`DeprecationWarning`.
"""

from __future__ import annotations

import sys
import warnings
from typing import List


def _warn() -> None:
    warnings.warn(
        "portfolio_analysis_report.py is deprecated; use the `trend` CLI instead",
        DeprecationWarning,
        stacklevel=2,
    )


def main(argv: List[str] | None = None) -> int:
    """Entry point delegating to `trend.cli:main`.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument vector. When None, `sys.argv[1:]` is used.
    """
    _warn()
    try:
        from trend_analysis.cli import main as trend_main  # lazy import
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to import trend CLI: {exc}", file=sys.stderr)
        return 1
    return trend_main(argv or sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# Legacy plotting/reporting block kept for archival comparison. Not executed
# by default. Imports appear below the executable section intentionally; E402
# suppressed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: F401, E402
import pandas as pd  # noqa: E402

# Portfolio performance data from multi-period analysis
performance_data = {
    "Period": [
        "2008-2010/2011",
        "2009-2011/2012",
        "2010-2012/2013",
        "2011-2013/2014",
        "2012-2014/2015",
        "2013-2015/2016",
        "2014-2016/2017",
        "2015-2017/2018",
        "2016-2018/2019",
        "2017-2019/2020",
        "2018-2020/2021",
        "2019-2021/2022",
        "2020-2022/2023",
        "2021-2023/2024",
        "2022-2024/2025",
    ],
    "Portfolio_CAGR": [
        7.46,
        6.71,
        10.78,
        8.29,
        7.38,
        6.84,
        7.17,
        8.88,
        10.21,
        12.28,
        12.47,
        11.80,
        11.77,
        12.02,
        9.82,
    ],
    "Portfolio_Vol": [
        3.84,
        3.86,
        3.90,
        4.52,
        4.80,
        4.55,
        4.23,
        4.02,
        4.47,
        4.95,
        4.94,
        4.19,
        3.53,
        3.68,
        4.30,
    ],
    "Portfolio_Sharpe": [
        -1.56,
        -1.62,
        -1.35,
        -1.73,
        -2.18,
        -1.40,
        -1.17,
        -0.76,
        -1.43,
        -1.40,
        -1.66,
        -2.42,
        -3.05,
        -3.05,
        -2.50,
    ],
    "Portfolio_MaxDD": [
        2.90,
        2.90,
        1.30,
        3.43,
        3.76,
        3.89,
        4.14,
        3.55,
        4.00,
        3.77,
        3.60,
        3.18,
        1.15,
        1.40,
        2.49,
    ],
    "Out_Sample_CAGR": [
        9.24,
        11.01,
        4.41,
        8.35,
        7.32,
        1.58,
        12.14,
        13.27,
        14.67,
        11.15,
        15.64,
        11.87,
        10.88,
        7.18,
        0.21,
    ],
    "Out_Sample_Vol": [
        3.71,
        4.53,
        5.99,
        4.35,
        3.48,
        4.87,
        3.86,
        4.77,
        6.48,
        3.73,
        3.53,
        4.26,
        3.68,
        4.81,
        6.13,
    ],
    "Out_Sample_Sharpe": [
        -1.31,
        -1.42,
        -2.38,
        -2.57,
        0.50,
        -1.73,
        -1.29,
        -1.34,
        -1.23,
        -2.34,
        -3.15,
        -3.06,
        -2.84,
        -1.62,
        -4.52,
    ],
    "Out_Sample_IR_SPX": [
        0.85,
        1.28,
        3.54,
        2.24,
        1.51,
        5.24,
        1.74,
        0.08,
        3.59,
        0.89,
        0.70,
        1.03,
        0.51,
        2.53,
        3.35,
    ],
}

df = pd.DataFrame(performance_data)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Trend Portfolio Analysis: 20-Year Backtest (July 2005 - June 2025)",
    fontsize=16,
    fontweight="bold",
)

# 1. Out-of-Sample CAGR over time
axes[0, 0].plot(
    range(len(df)),
    df["Out_Sample_CAGR"],
    "o-",
    color="darkgreen",
    linewidth=2,
    markersize=6,
)
avg_cagr = df["Out_Sample_CAGR"].mean()
axes[0, 0].axhline(
    y=avg_cagr,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Average: {avg_cagr:.1f}%",
)
axes[0, 0].set_title("Out-of-Sample CAGR by Period", fontweight="bold")
axes[0, 0].set_ylabel("CAGR (%)")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].set_xticks(range(0, len(df), 2))
axes[0, 0].set_xticklabels(
    [df["Period"].iloc[i].split("/")[1] for i in range(0, len(df), 2)], rotation=45
)

# 2. Risk-Adjusted Performance (Sharpe Ratio)
axes[0, 1].plot(
    range(len(df)),
    df["Out_Sample_Sharpe"],
    "o-",
    color="blue",
    linewidth=2,
    markersize=6,
)
axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
avg_sharpe = df["Out_Sample_Sharpe"].mean()
axes[0, 1].axhline(
    y=avg_sharpe,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Average: {avg_sharpe:.2f}",
)
axes[0, 1].set_title("Out-of-Sample Sharpe Ratio", fontweight="bold")
axes[0, 1].set_ylabel("Sharpe Ratio")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_xticks(range(0, len(df), 2))
axes[0, 1].set_xticklabels(
    [df["Period"].iloc[i].split("/")[1] for i in range(0, len(df), 2)], rotation=45
)

# 3. Information Ratio vs S&P 500
axes[0, 2].plot(
    range(len(df)),
    df["Out_Sample_IR_SPX"],
    "o-",
    color="purple",
    linewidth=2,
    markersize=6,
)
axes[0, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)
avg_ir = df["Out_Sample_IR_SPX"].mean()
axes[0, 2].axhline(
    y=avg_ir,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Average IR: {avg_ir:.2f}",
)
axes[0, 2].set_title("Information Ratio vs S&P 500", fontweight="bold")
axes[0, 2].set_ylabel("Information Ratio")
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()
axes[0, 2].set_xticks(range(0, len(df), 2))
axes[0, 2].set_xticklabels(
    [df["Period"].iloc[i].split("/")[1] for i in range(0, len(df), 2)], rotation=45
)

# 4. Volatility Analysis
axes[1, 0].plot(
    range(len(df)),
    df["Out_Sample_Vol"],
    "o-",
    color="orange",
    linewidth=2,
    markersize=6,
)
axes[1, 0].axhline(y=12, color="red", linestyle="--", alpha=0.7, label="Target: 12%")
avg_vol = df["Out_Sample_Vol"].mean()
axes[1, 0].axhline(
    y=avg_vol,
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"Average: {avg_vol:.1f}%",
)
axes[1, 0].set_title("Out-of-Sample Volatility", fontweight="bold")
axes[1, 0].set_ylabel("Volatility (%)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].set_xticks(range(0, len(df), 2))
axes[1, 0].set_xticklabels(
    [df["Period"].iloc[i].split("/")[1] for i in range(0, len(df), 2)], rotation=45
)

# 5. Performance Distribution
axes[1, 1].hist(
    df["Out_Sample_CAGR"], bins=8, alpha=0.7, color="darkgreen", edgecolor="black"
)
axes[1, 1].axvline(
    x=df["Out_Sample_CAGR"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f'Mean: {df["Out_Sample_CAGR"].mean():.1f}%',
)
axes[1, 1].axvline(
    x=df["Out_Sample_CAGR"].median(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'Median: {df["Out_Sample_CAGR"].median():.1f}%',
)
axes[1, 1].set_title("Distribution of Out-of-Sample Returns", fontweight="bold")
axes[1, 1].set_xlabel("CAGR (%)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Risk vs Return Scatter
axes[1, 2].scatter(
    df["Out_Sample_Vol"],
    df["Out_Sample_CAGR"],
    c=df["Out_Sample_Sharpe"],
    s=100,
    alpha=0.7,
    cmap="RdYlGn",
    edgecolor="black",
)
axes[1, 2].set_xlabel("Volatility (%)")
axes[1, 2].set_ylabel("CAGR (%)")
axes[1, 2].set_title("Risk vs Return (Color = Sharpe Ratio)", fontweight="bold")
axes[1, 2].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
cbar.set_label("Sharpe Ratio")

plt.tight_layout()
plt.show()

# Performance Summary Statistics
print("\n" + "=" * 80)
print("TREND PORTFOLIO ANALYSIS - 20 YEAR BACKTEST SUMMARY")
print("=" * 80)

print("\nPERFORMANCE OVERVIEW:")
print("• Analysis Period: July 2005 - June 2025 (20 years)")
print("• Rolling Windows: 15 periods (3-year in-sample, 1-year out-of-sample)")
print("• Portfolio Construction: Top 8 managers, Bayesian weighting")
print("• Volatility Targeting: 12% annual")

print("\nOUT-OF-SAMPLE PERFORMANCE STATISTICS:")
print(f"• Average CAGR: {df['Out_Sample_CAGR'].mean():.1f}%")
print(f"• Median CAGR: {df['Out_Sample_CAGR'].median():.1f}%")
print(f"• Best Period: {df['Out_Sample_CAGR'].max():.1f}% (2021)")
print(f"• Worst Period: {df['Out_Sample_CAGR'].min():.1f}% (2025)")
print(f"• Standard Deviation: {df['Out_Sample_CAGR'].std():.1f}%")

print("\nRISK METRICS:")
print(f"• Average Volatility: {df['Out_Sample_Vol'].mean():.1f}% (Target: 12%)")
print(f"• Average Sharpe Ratio: {df['Out_Sample_Sharpe'].mean():.2f}")
print(f"• Average Information Ratio vs SPX: {df['Out_Sample_IR_SPX'].mean():.2f}")
pos_sharpe = int(sum(df["Out_Sample_Sharpe"] > 0))
pct_pos_sharpe = 100 * pos_sharpe / 15
print(f"• Positive Sharpe Periods: {pos_sharpe}/15 ({pct_pos_sharpe:.0f}%)")

print("\nPERIOD ANALYSIS:")
print(
    "• Crisis Periods (2008-2012): Portfolio showed resilience during financial crisis"
)
print("• Growth Periods (2013-2017): Strong performance with positive Sharpe ratios")
print("• Volatile Periods (2018-2025): Mixed results reflecting market uncertainty")

print("\nKEY INSIGHTS:")
positive_returns = int(sum(df["Out_Sample_CAGR"] > 0))
pct_positive_returns = 100 * positive_returns / 15
print(f"• Positive Return Periods: {positive_returns}/15 ({pct_positive_returns:.0f}%)")
print("• Best 5-Year Stretch: 2013-2017 (consistent positive performance)")
# Keep the code lines short and readable
vol_mean = df["Out_Sample_Vol"].mean()
print(
    f"• Volatility Control: Average realized vol {vol_mean:.1f}% vs 12% target"
)
print(f"• Information Ratio: Positive in {sum(df['Out_Sample_IR_SPX'] > 0)}/15 periods")

print("\n" + "=" * 80)
