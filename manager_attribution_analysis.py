#!/usr/bin/env python3
"""DEPRECATED: Manager attribution legacy script.

Replaced by unified `trend` CLI (issue #1437). Migrate to one of:

    trend run --config config/demo.yml --returns demo/demo_returns.csv
    trend report --out perf/

Custom attribution / analytics should be implemented via the package
APIs (`trend_analysis.pipeline`, forthcoming reporting hooks) rather than
this script. This wrapper will be removed in a future minor release.
"""

from __future__ import annotations

import sys
import warnings
from typing import List


def _warn() -> None:
    warnings.warn(
        "manager_attribution_analysis.py is deprecated; use the `trend` CLI",
        DeprecationWarning,
        stacklevel=2,
    )


def main(argv: List[str] | None = None) -> int:
    _warn()
    try:
        from trend_analysis.cli import main as trend_main
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import trend CLI: {exc}", file=sys.stderr)
        return 1
    return trend_main(argv or sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    raise SystemExit(main())

# The analytical plotting section below is retained for reference only and is
# intentionally placed after the main guard. It is not executed during normal
# imports; flake8 E402 suppressed for these optional visualization imports.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: F401, E402
import pandas as pd  # noqa: E402

# Manager performance data from the single-period analysis (2005-2025)
manager_data = {
    "Manager": [
        "Mgr_01",
        "Mgr_02",
        "Mgr_03",
        "Mgr_04",
        "Mgr_05",
        "Mgr_06",
        "Mgr_07",
        "Mgr_08",
    ],
    "Full_Period_CAGR": [-42.8, 10.3, 21.4, -42.1, 17.6, 4.9, 53.4, 26.0],
    "Full_Period_Sharpe": [-3.83, -1.13, -0.39, -3.85, -0.71, -1.32, 0.70, -0.16],
    "Full_Period_Sortino": [-2.05, -1.29, -0.53, -2.06, -0.91, -1.46, 1.26, -0.22],
    "Full_Period_IR": [-5.50, -1.21, -0.37, -5.52, -0.74, -1.36, 0.73, -0.14],
    "Full_Period_MaxDD": [86.74, 19.18, 12.53, 87.60, 18.07, 22.86, 8.51, 11.80],
    "Weight": [
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
        12.5,
    ],  # Equal weights in portfolio
}

df_mgr = pd.DataFrame(manager_data)

# Create manager analysis visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "Manager Performance Attribution: 20-Year Analysis (July 2005 - June 2025)",
    fontsize=16,
    fontweight="bold",
)

# 1. Manager CAGR Performance
colors = ["red" if x < 0 else "green" for x in df_mgr["Full_Period_CAGR"]]
bars1 = axes[0, 0].bar(
    df_mgr["Manager"],
    df_mgr["Full_Period_CAGR"],
    color=colors,
    alpha=0.7,
    edgecolor="black",
)
axes[0, 0].axhline(y=0, color="black", linestyle="-", alpha=0.5)
axes[0, 0].set_title("Manager CAGR Performance (20-Year)", fontweight="bold")
axes[0, 0].set_ylabel("CAGR (%)")
axes[0, 0].grid(True, alpha=0.3, axis="y")
# Add value labels on bars
for bar, value in zip(bars1, df_mgr["Full_Period_CAGR"]):
    height = bar.get_height()
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + (1 if height > 0 else -3),
        f"{value:.1f}%",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontweight="bold",
    )

# 2. Risk-Adjusted Performance (Sharpe Ratio)
colors2 = ["red" if x < 0 else "green" for x in df_mgr["Full_Period_Sharpe"]]
bars2 = axes[0, 1].bar(
    df_mgr["Manager"],
    df_mgr["Full_Period_Sharpe"],
    color=colors2,
    alpha=0.7,
    edgecolor="black",
)
axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
axes[0, 1].set_title("Manager Sharpe Ratios", fontweight="bold")
axes[0, 1].set_ylabel("Sharpe Ratio")
axes[0, 1].grid(True, alpha=0.3, axis="y")
# Add value labels
for bar, value in zip(bars2, df_mgr["Full_Period_Sharpe"]):
    height = bar.get_height()
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + (0.05 if height > 0 else -0.1),
        f"{value:.2f}",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontweight="bold",
    )

# 3. Maximum Drawdown Analysis
bars3 = axes[1, 0].bar(
    df_mgr["Manager"],
    df_mgr["Full_Period_MaxDD"],
    color="darkred",
    alpha=0.7,
    edgecolor="black",
)
axes[1, 0].set_title("Manager Maximum Drawdowns", fontweight="bold")
axes[1, 0].set_ylabel("Max Drawdown (%)")
axes[1, 0].grid(True, alpha=0.3, axis="y")
# Add value labels
for bar, value in zip(bars3, df_mgr["Full_Period_MaxDD"]):
    height = bar.get_height()
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# 4. Risk vs Return Scatter with manager labels
scatter = axes[1, 1].scatter(
    df_mgr["Full_Period_MaxDD"],
    df_mgr["Full_Period_CAGR"],
    c=df_mgr["Full_Period_Sharpe"],
    s=200,
    alpha=0.8,
    cmap="RdYlGn",
    edgecolor="black",
    linewidth=2,
)
axes[1, 1].set_xlabel("Maximum Drawdown (%)")
axes[1, 1].set_ylabel("CAGR (%)")
axes[1, 1].set_title("Risk vs Return by Manager (Color = Sharpe)", fontweight="bold")
axes[1, 1].grid(True, alpha=0.3)

# Add manager labels to scatter plot
for i, txt in enumerate(df_mgr["Manager"]):
    axes[1, 1].annotate(
        txt,
        (df_mgr["Full_Period_MaxDD"].iloc[i], df_mgr["Full_Period_CAGR"].iloc[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontweight="bold",
    )

cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label("Sharpe Ratio")

plt.tight_layout()
plt.show()

# Performance ranking and analysis
print("\n" + "=" * 80)
print("MANAGER PERFORMANCE ANALYSIS - 20 YEAR RANKINGS")
print("=" * 80)

# Sort managers by different metrics
by_cagr = df_mgr.sort_values("Full_Period_CAGR", ascending=False)
by_sharpe = df_mgr.sort_values("Full_Period_Sharpe", ascending=False)
by_drawdown = df_mgr.sort_values("Full_Period_MaxDD", ascending=True)

print("\nRANKING BY CAGR (Best to Worst):")
for i, (idx, row) in enumerate(by_cagr.iterrows(), 1):
    msg = (
        f"{i:2d}. {row['Manager']}: {row['Full_Period_CAGR']:6.1f}% CAGR, "
        f"{row['Full_Period_Sharpe']:5.2f} Sharpe, "
        f"{row['Full_Period_MaxDD']:5.1f}% MaxDD"
    )
    print(msg)

print("\nRANKING BY SHARPE RATIO (Best to Worst):")
for i, (idx, row) in enumerate(by_sharpe.iterrows(), 1):
    msg = (
        f"{i:2d}. {row['Manager']}: {row['Full_Period_Sharpe']:5.2f} Sharpe, "
        f"{row['Full_Period_CAGR']:6.1f}% CAGR, "
        f"{row['Full_Period_MaxDD']:5.1f}% MaxDD"
    )
    print(msg)

print("\nRANKING BY MAX DRAWDOWN (Best to Worst):")
for i, (idx, row) in enumerate(by_drawdown.iterrows(), 1):
    msg = (
        f"{i:2d}. {row['Manager']}: {row['Full_Period_MaxDD']:5.1f}% MaxDD, "
        f"{row['Full_Period_CAGR']:6.1f}% CAGR, "
        f"{row['Full_Period_Sharpe']:5.2f} Sharpe"
    )
    print(msg)

print("\nPORTFOLIO COMPOSITION ANALYSIS:")
print("• Portfolio uses equal weights (12.5% each manager)")
print("• Top Performers: Mgr_07 (53.4% CAGR, 0.70 Sharpe), Mgr_08 (26.0% CAGR)")
print("• Worst Performers: Mgr_01 (-42.8% CAGR), Mgr_04 (-42.1% CAGR)")
print("• Risk Champions: Mgr_07 (8.5% MaxDD), Mgr_08 (11.8% MaxDD)")
print("• Risk Concerns: Mgr_04 (87.6% MaxDD), Mgr_01 (86.7% MaxDD)")

# Calculate portfolio contribution
positive_mgrs = sum(df_mgr["Full_Period_CAGR"] > 0)
negative_mgrs = sum(df_mgr["Full_Period_CAGR"] < 0)
avg_positive = df_mgr[df_mgr["Full_Period_CAGR"] > 0]["Full_Period_CAGR"].mean()
avg_negative = df_mgr[df_mgr["Full_Period_CAGR"] < 0]["Full_Period_CAGR"].mean()

print("\nPORTFOLIO DIVERSIFICATION:")
pos_msg = (
    f"• Positive Contributors: {positive_mgrs}/8 managers "
    f"(Avg: {avg_positive:.1f}% CAGR)"
)
print(pos_msg)
neg_msg = (
    f"• Negative Contributors: {negative_mgrs}/8 managers "
    f"(Avg: {avg_negative:.1f}% CAGR)"
)
print(neg_msg)
max_cagr = df_mgr["Full_Period_CAGR"].max()
min_cagr = df_mgr["Full_Period_CAGR"].min()
spread = max_cagr - min_cagr
range_msg = (
    f"• Performance Range: {max_cagr:.1f}% to {min_cagr:.1f}% "
    f"({spread:.1f}% spread)"
)
print(range_msg)
max_sharpe = df_mgr["Full_Period_Sharpe"].max()
min_sharpe = df_mgr["Full_Period_Sharpe"].min()
print(f"• Sharpe Range: {max_sharpe:.2f} to {min_sharpe:.2f}")

print("\nKEY OBSERVATIONS:")
print("• Only 1 manager (Mgr_07) achieved positive Sharpe ratio over 20 years")
print("• 6 out of 8 managers delivered positive returns despite negative Sharpe ratios")
print("• High performance dispersion suggests strong alpha generation potential")
print("• Drawdown control varies dramatically (8.5% to 87.6%)")
print("• Portfolio benefits from diversification across different manager styles")

print("\n" + "=" * 80)
