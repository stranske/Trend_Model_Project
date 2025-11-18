#!/usr/bin/env python3
"""
Generate per-period out-of-sample (OS) portfolio stats, an all-period OS
summary, and a churn report (entries/exits) of selected funds between periods
for the demo configuration.

Outputs are written under demo/exports/:
- period_os_stats.csv           (per-period EW OS stats)
- combined_os_stats.csv         (single-row EW OS stats across all periods)
- portfolio_churn.csv           (entries/exits lists per period)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trend_analysis import export
from trend_analysis.config import load
from trend_analysis.multi_period import run_from_config as run_mp


def _to_row(label: str, stats: Any) -> dict[str, Any]:
    # stats is pipeline._Stats
    return {
        "period": label,
        "cagr": getattr(stats, "cagr", float("nan")),
        "vol": getattr(stats, "vol", float("nan")),
        "sharpe": getattr(stats, "sharpe", float("nan")),
        "sortino": getattr(stats, "sortino", float("nan")),
        "information_ratio": getattr(stats, "information_ratio", float("nan")),
        "max_drawdown": getattr(stats, "max_drawdown", float("nan")),
    }


def main() -> None:
    cfg = load("config/demo.yml")
    results = list(run_mp(cfg))
    if not results:
        raise SystemExit("No results from multi-period run")

    # Per-period OS EW stats and churn
    period_rows: list[dict[str, Any]] = []
    churn_rows: list[dict[str, Any]] = []

    prev_set: set[str] | None = None
    for res in results:
        period = res.get("period")
        # Pick a compact label; default to out_end (index 3)
        if isinstance(period, (list, tuple)) and len(period) >= 4:
            label = str(period[3])
        else:
            label = str(period)
        out_ew_stats = res.get("out_ew_stats")
        if out_ew_stats is not None:
            period_rows.append(_to_row(label, out_ew_stats))
        # churn
        sel = set(res.get("selected_funds", []))
        entries: list[str] = []
        exits: list[str] = []
        if prev_set is not None:
            entries = sorted(sel - prev_set)
            exits = sorted(prev_set - sel)
        churn_rows.append(
            {
                "period": label,
                "selected_funds": ",".join(sorted(sel)),
                "entries": ",".join(entries),
                "exits": ",".join(exits),
            }
        )
        prev_set = sel

    out_dir = Path("demo/exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(period_rows).to_csv(out_dir / "period_os_stats.csv", index=False)

    # All-period combined EW OS stats using the export aggregator
    combined = export.combined_summary_result(results)
    comb_stats = combined.get("out_ew_stats")
    if comb_stats is not None:
        pd.DataFrame([_to_row("all_periods", comb_stats)]).to_csv(
            out_dir / "combined_os_stats.csv", index=False
        )

    pd.DataFrame(churn_rows).to_csv(out_dir / "portfolio_churn.csv", index=False)


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    main()
