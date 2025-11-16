#!/usr/bin/env python3
"""Simple CLI to run walk-forward aggregation on a CSV of returns.

Usage:
  python scripts/walkforward_cli.py --csv demo/demo_returns.csv \
      --train 12 --test 3 --step 3 --column Portfolio
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from trend.input_validation import (
    InputSchema,
    InputValidationError,
    validate_input,
)
from trend_analysis.engine.walkforward import walk_forward
from trend_analysis.script_logging import setup_script_logging


INPUT_SCHEMA = InputSchema(
    date_column="Date",
    required_columns=("Date",),
    non_nullable=("Date",),
)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [
            " / ".join(str(part) for part in col if part not in (None, ""))
            for col in out.columns
        ]
        return out
    return df


def _load_regimes(path: str, column: str | None) -> pd.Series:
    reg_df = pd.read_csv(path)
    date_col = None
    if "Date" in reg_df.columns:
        date_col = "Date"
    else:
        for c in reg_df.columns:
            if c.lower().startswith("date"):
                date_col = c
                break
    if date_col is None:
        raise SystemExit("Regime CSV must contain a Date column")

    reg_df[date_col] = pd.to_datetime(reg_df[date_col])
    label_cols = [c for c in reg_df.columns if c != date_col]
    if not label_cols:
        raise SystemExit("Regime CSV must contain at least one label column")

    target_col = column or label_cols[0]
    if target_col not in reg_df.columns:
        raise SystemExit(f"Column '{target_col}' not found in regime CSV")

    return pd.Series(reg_df[target_col].values, index=reg_df[date_col])


def main(argv: list[str] | None = None) -> int:
    setup_script_logging(app_name="walkforward", module_file=__file__)
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
    p.add_argument(
        "--regime-csv",
        default=None,
        help="Optional CSV with Date column and regime labels",
    )
    p.add_argument(
        "--regime-column",
        default=None,
        help="Column name in regime CSV (default: first non-date column)",
    )

    args = p.parse_args(argv)

    log_path = setup_logging(app_name="walkforward_cli")
    logging.getLogger(__name__).info("Log file initialised at %s", log_path)

    df = pd.read_csv(args.csv)
    if "Date" not in df.columns:
        # Try to infer a date column
        for c in df.columns:
            if c.lower().startswith("date"):
                df = df.rename(columns={c: "Date"})
                break
    try:
        df = validate_input(
            df,
            INPUT_SCHEMA,
            set_index=False,
            drop_date_column=False,
        )
    except InputValidationError as exc:
        print(f"Input validation failed: {exc.user_message}", file=sys.stderr)
        return 2
    cols = [c for c in df.columns if c != "Date"]
    if not cols:
        print("No metric columns found.", file=sys.stderr)
        return 2
    metric = args.column or cols[0]

    regimes = None
    if args.regime_csv:
        regimes = _load_regimes(args.regime_csv, args.regime_column)

    res = walk_forward(
        df[["Date", metric]],
        train_size=args.train,
        test_size=args.test,
        step_size=args.step,
        metric_cols=[metric],
        regimes=regimes,
    )

    print(f"Detected periods/year: {res.periods_per_year}")
    print("Full-period aggregate:")
    print(_flatten_columns(res.full).to_string())
    print()
    print("OOS aggregate:")
    print(_flatten_columns(res.oos).to_string())
    if not res.oos_windows.empty:
        print()
        print("Per-window (OOS) summary:")
        print(_flatten_columns(res.oos_windows).to_string())
    if res.by_regime is not None and not res.by_regime.empty:
        print()
        print("Per-regime (OOS):")
        print(_flatten_columns(res.by_regime).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
