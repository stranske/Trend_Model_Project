#!/usr/bin/env python3
"""Simple CLI to run walk-forward aggregation on a CSV of returns.

Usage:
  python scripts/walkforward_cli.py --csv demo/demo_returns.csv \
      --train 12 --test 3 --step 3 --column Portfolio
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from trend_analysis.engine.walkforward import walk_forward


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward aggregation")
    p.add_argument(
        "--csv", required=True, help="Input CSV with Date column and metrics"
    )
    p.add_argument(
        "--column",
        default=None,
        help="Metric column to aggregate (default: first data column)",
    )
    p.add_argument("--train", type=int, default=12, help="Train window size (rows)")
    p.add_argument("--test", type=int, default=3, help="Test window size (rows)")
    p.add_argument("--step", type=int, default=3, help="Step size (rows)")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if "Date" not in df.columns:
        # Try to infer a date column
        for c in df.columns:
            if c.lower().startswith("date"):
                df = df.rename(columns={c: "Date"})
                break
    cols = [c for c in df.columns if c != "Date"]
    if not cols:
        print("No metric columns found.", file=sys.stderr)
        return 2
    metric = args.column or cols[0]

    res = walk_forward(
        df[["Date", metric]],
        train_size=args.train,
        test_size=args.test,
        step_size=args.step,
        metric_cols=[metric],
    )

    print("Full-period aggregate:")
    print(res.full.to_string())
    print()
    print("OOS aggregate:")
    print(res.oos.to_string())
    if res.by_regime is not None and not res.by_regime.empty:
        print()
        print("Per-regime (OOS):")
        print(res.by_regime.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
