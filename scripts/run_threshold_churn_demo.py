#!/usr/bin/env python3
"""Threshold-based churn demo

This demo simulates a portfolio with the following rules:

- Initial selection: top N managers by in-sample Sharpe.
- Evaluation cadence: rolling 36-month in-sample window (like the demo),
  stepping yearly.
- Exit rules (on each period using the current 36m window, z = Sharpe z-score):
  - Immediate exit if z <= -1.5 in the current window, OR
  - Exit if the manager has accumulated 2 windows with z <= -1.0 ("two strikes").
- Entry rules (candidates not currently held):
  - Immediate entry if z >= +1.5 in the current window, OR
  - Eligible if the manager has accumulated 2 windows with z >= +1.0.
- Weights: equal-weight across current holdings each period.

Assumptions:
- "Two rolling 3-year periods" is interpreted as two occurrences cumulatively
  across time (not necessarily consecutive). Adjust thresholds below if needed.
- We select N=5 managers to mirror the main demo.
- We extend the multi-period start to 2017-01 to cover ~8 out-of-sample years
  on the demo dataset (2015-01..2024-12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from trend_analysis.config import Config, load
from trend_analysis.multi_period import run_from_config as run_mp
from trend_analysis.script_logging import setup_script_logging


@dataclass
class RuleConfig:
    target_n: int = 5
    z_exit_hard: float = -1.5
    z_exit_soft: float = -1.0
    z_entry_hard: float = +1.5
    z_entry_soft: float = +1.0
    soft_strikes: int = 2


def _z_scores(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * math.nan
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return (series - mu) * math.nan  # undefined if no dispersion
    return (series - mu) / sd


def main() -> int:
    setup_script_logging(app_name="threshold-demo", module_file=__file__)
    # Load base demo config and tweak multi-period start for ~8 years OS
    cfg = load("config/demo.yml")
    cfg_dict = cfg.model_dump()
    # Ensure we have a long-enough run: start 2015-01 through 2024-12 (~8 periods)
    cfg_dict.setdefault("multi_period", {})
    cfg_dict["multi_period"]["start"] = "2015-01"
    # Use semiannual OOS windows (6) with 36-month rolling in-sample windows
    cfg_dict["multi_period"]["out_sample_len"] = 6
    cfg = Config(**cfg_dict)

    # Drive the multi-period engine to get per-period in-sample score frames
    results = run_mp(cfg)
    if not results:
        raise SystemExit("No results produced by multi-period engine")

    # Build mapping of period label -> score_frame and keep chronological order
    periods = [r["period"] for r in results]
    labels = [p[3] for p in periods]  # out_end labels (e.g., 2021-12-31)
    frames = {r["period"][3]: r["score_frame"] for r in results}

    # Sanity: ensure Sharpe exists in all score frames
    for lbl in labels:
        sf = frames[lbl]
        if "Sharpe" not in sf.columns:
            raise SystemExit(f"Score frame for {lbl} missing 'Sharpe'")

    rule = RuleConfig()
    holdings: Set[str] = set()
    neg_strikes: Dict[str, int] = {}
    pos_strikes: Dict[str, int] = {}

    churn_rows: List[dict] = []
    zsum_rows: List[dict] = []

    # Initialize from the first period by top Sharpe
    first_lbl = labels[0]
    first_sf = frames[first_lbl]
    init_sorted = first_sf["Sharpe"].sort_values(ascending=False)
    holdings = set(init_sorted.head(rule.target_n).index.tolist())

    # Iterate periods, apply exit/entry rules
    for lbl in labels:
        sf = frames[lbl]
        sharpe = sf["Sharpe"]
        z = _z_scores(sharpe)

        exits: List[str] = []
        entries: List[str] = []
        exit_reasons: Dict[str, str] = {}
        entry_reasons: Dict[str, str] = {}

        # Evaluate exits on current holdings
        for name in list(holdings):
            key = str(name)
            z_i = float(z.get(key, math.nan))
            # Count negative strikes
            if not math.isnan(z_i) and z_i <= rule.z_exit_soft:
                neg_strikes[key] = neg_strikes.get(key, 0) + 1
            # Hard exit condition
            if not math.isnan(z_i) and z_i <= rule.z_exit_hard:
                exits.append(key)
                exit_reasons[key] = f"hard_exit_z<={rule.z_exit_hard:.1f} (z={z_i:.2f})"
                continue
            # Soft strike accumulation
            if neg_strikes.get(key, 0) >= rule.soft_strikes:
                exits.append(key)
                exit_reasons[key] = (
                    f"soft_exit_{rule.soft_strikes}x z<={rule.z_exit_soft:.1f}"
                )

        for name in exits:
            holdings.discard(name)

        # Update positive strikes for non-holdings and identify candidates
        candidates: List[tuple[str, float, bool]] = []
        for name, z_i in z.items():
            key = str(name)
            if key in holdings:
                continue
            zf = float(z_i)
            if math.isnan(zf):
                continue
            # Count positive strikes
            if zf >= rule.z_entry_soft:
                pos_strikes[key] = pos_strikes.get(key, 0) + 1
            # Determine immediate eligibility category and score
            immediate = zf >= rule.z_entry_hard
            eligible = immediate or (pos_strikes.get(key, 0) >= rule.soft_strikes)
            if eligible:
                candidates.append((key, zf, immediate))

        # Prefer immediate candidates first, sorted by z, then other eligibles by z
        candidates.sort(key=lambda t: (not t[2], -t[1]))

        # Fill up to target_n
        for name, zf, immediate in candidates:
            if len(holdings) >= rule.target_n:
                break
            if name not in holdings:
                holdings.add(name)
                entries.append(name)
                if immediate:
                    entry_reasons[name] = (
                        f"hard_entry_z>={rule.z_entry_hard:.1f} (z={zf:.2f})"
                    )
                else:
                    entry_reasons[name] = (
                        f"soft_entry_{rule.soft_strikes}x z>={rule.z_entry_soft:.1f}"
                    )

        # Record z-summary for this period for interpretability
        z_series = pd.Series(z)
        zsum_rows.append(
            {
                "period": lbl,
                "mean_z": float(z_series.mean(skipna=True)),
                "std_z": float(z_series.std(skipna=True, ddof=0)),
                "min_z": float(z_series.min(skipna=True)),
                "max_z": float(z_series.max(skipna=True)),
                "count_z<=-1.0": int((z_series <= -1.0).sum()),
                "count_z>=+1.0": int((z_series >= +1.0).sum()),
                "bottom_names": ",".join(
                    z_series.sort_values().head(3).index.astype(str).tolist()
                ),
                "top_names": ",".join(
                    z_series.sort_values(ascending=False)
                    .head(3)
                    .index.astype(str)
                    .tolist()
                ),
            }
        )

        # Equal weights for current holdings (implicit; not exported here)

        churn_rows.append(
            {
                "period": lbl,
                "holdings": ",".join(sorted(holdings)),
                "entries": ",".join(sorted(entries)),
                "exits": ",".join(sorted(exits)),
                "exit_reasons": ";".join(f"{k}:{v}" for k, v in exit_reasons.items()),
                "entry_reasons": ";".join(f"{k}:{v}" for k, v in entry_reasons.items()),
            }
        )

    out_dir = Path("demo/exports/threshold_churn")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(churn_rows).to_csv(out_dir / "churn_schedule.csv", index=False)
    pd.DataFrame(zsum_rows).to_csv(out_dir / "z_summary.csv", index=False)

    msg = (
        f"Wrote {len(churn_rows)} periods to {out_dir}/churn_schedule.csv "
        "and z_summary.csv"
    )
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
