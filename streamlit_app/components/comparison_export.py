"""Excel and CSV export utilities for A/B comparison results.

This module generates comprehensive comparison workbooks with:
- Comparison Summary sheet (key metrics with winner indicators)
- Selection Differences sheet (periods/funds where selections differ)
- Side-by-side Period Comparison sheet
- Raw Data as Excel Table (pivot-ready)
- Pre-aggregated summary views
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

# Excel formatting constants
HEADER_FORMAT = {
    "bold": True,
    "bg_color": "#4472C4",
    "font_color": "white",
    "border": 1,
    "align": "center",
    "valign": "vcenter",
}
PERCENT_FORMAT = "0.00%"
RATIO_FORMAT = "0.00"
NUMBER_FORMAT = "#,##0.00"
DATE_FORMAT = "yyyy-mm-dd"

# Conditional formatting colors
WINNER_GREEN = "#C6EFCE"
LOSER_RED = "#FFC7CE"
NEUTRAL_YELLOW = "#FFEB9C"


def _coerce_numeric(value: Any) -> float | None:
    """Safely convert value to float."""
    try:
        as_float = float(value)
        return as_float if np.isfinite(as_float) else None
    except (TypeError, ValueError):
        return None


def _extract_metrics(result: Any) -> dict[str, float | None]:
    """Extract numeric metrics from a result object.

    Priority:
    1. details.out_user_stats (portfolio-level out-of-sample metrics)
    2. result.metrics DataFrame (if available)
    3. details.out_sample_stats (per-fund metrics, fallback)
    """
    metrics_dict: dict[str, float | None] = {}
    details = getattr(result, "details", {}) or {}

    # First priority: out_user_stats - these are the PORTFOLIO-level OOS metrics
    # This matches what Performance Overview calculates from returns
    user_stats = details.get("out_user_stats")
    if user_stats is not None:
        if hasattr(user_stats, "__dict__"):
            # It's a stats object with attributes
            for k, v in vars(user_stats).items():
                if not k.startswith("_"):
                    metrics_dict[str(k)] = _coerce_numeric(v)
        elif hasattr(user_stats, "items"):
            # It's a dict-like
            for k, v in user_stats.items():
                metrics_dict[str(k)] = _coerce_numeric(v)
        if metrics_dict:
            return metrics_dict

    # Second priority: metrics DataFrame (may contain portfolio stats)
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        # Check if it has portfolio-level data
        series = metrics.iloc[0]
        for k, v in series.items():
            metrics_dict[str(k)] = _coerce_numeric(v)
        if metrics_dict:
            return metrics_dict

    # Fallback: out_sample_stats (per-fund stats, less useful for comparison)
    stats_obj = details.get("out_sample_stats") or {}
    if hasattr(stats_obj, "items"):
        for k, v in stats_obj.items():
            metrics_dict[str(k)] = _coerce_numeric(v)
    elif hasattr(stats_obj, "__dict__"):
        for k, v in vars(stats_obj).items():
            metrics_dict[str(k)] = _coerce_numeric(v)

    return metrics_dict


def _extract_period_results(result: Any) -> list[dict[str, Any]]:
    """Extract period results from a result object."""
    details = getattr(result, "details", {}) or {}
    return details.get("period_results", []) or []


def _determine_winner(
    a_val: float | None, b_val: float | None, higher_is_better: bool = True
) -> str:
    """Determine winner between two values."""
    if a_val is None or b_val is None:
        return ""
    if abs(a_val - b_val) < 1e-10:
        return "Tie"
    if higher_is_better:
        return "A" if a_val > b_val else "B"
    else:
        return "A" if a_val < b_val else "B"


def build_comparison_summary_df(
    result_a: Any,
    result_b: Any,
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> pd.DataFrame:
    """Build summary comparison DataFrame with winner indicators.

    Returns DataFrame with columns: Metric, Config A, Config B, Delta, % Change, Winner
    """
    metrics_a = _extract_metrics(result_a)
    metrics_b = _extract_metrics(result_b)
    all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    # Metrics where lower is better
    lower_is_better = {"max_drawdown", "maxdd", "volatility", "vol", "var", "cvar"}

    rows = []
    for metric in all_metrics:
        a_val = metrics_a.get(metric)
        b_val = metrics_b.get(metric)

        delta = None
        pct_change = None
        if a_val is not None and b_val is not None:
            delta = b_val - a_val
            if abs(a_val) > 1e-10:
                pct_change = (b_val - a_val) / abs(a_val)

        higher_better = metric.lower() not in lower_is_better
        winner = _determine_winner(a_val, b_val, higher_is_better=higher_better)

        rows.append(
            {
                "Metric": metric,
                label_a: a_val,
                label_b: b_val,
                "Delta (B-A)": delta,
                "% Change": pct_change,
                "Winner": winner,
            }
        )

    return pd.DataFrame(rows)


def build_selection_differences_df(
    result_a: Any,
    result_b: Any,
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> pd.DataFrame:
    """Build DataFrame showing only periods/funds where selections differ."""
    periods_a = _extract_period_results(result_a)
    periods_b = _extract_period_results(result_b)

    rows = []
    max_periods = max(len(periods_a), len(periods_b))

    for idx in range(max_periods):
        res_a = periods_a[idx] if idx < len(periods_a) else {}
        res_b = periods_b[idx] if idx < len(periods_b) else {}

        period_a = res_a.get("period", ())
        period_b = res_b.get("period", ())

        # Extract period label
        out_start = ""
        out_end = ""
        for p in [period_a, period_b]:
            if len(p) > 2 and p[2]:
                out_start = str(p[2])
            if len(p) > 3 and p[3]:
                out_end = str(p[3])
        period_label = f"{out_start} → {out_end}".strip() or f"Period {idx + 1}"

        selected_a = set(res_a.get("selected_funds", []) or [])
        selected_b = set(res_b.get("selected_funds", []) or [])
        weights_a = res_a.get("fund_weights", {}) or {}
        weights_b = res_b.get("fund_weights", {}) or {}

        all_funds = selected_a | selected_b

        for fund in sorted(all_funds):
            in_a = fund in selected_a
            in_b = fund in selected_b

            # Only include if different
            if in_a != in_b:
                weight_a = _coerce_numeric(weights_a.get(fund, 0))
                weight_b = _coerce_numeric(weights_b.get(fund, 0))

                rows.append(
                    {
                        "Period": period_label,
                        "Fund": fund,
                        f"In {label_a}": "Yes" if in_a else "No",
                        f"In {label_b}": "Yes" if in_b else "No",
                        f"{label_a} Weight": weight_a if in_a else None,
                        f"{label_b} Weight": weight_b if in_b else None,
                        "Difference": "Only in A" if in_a else "Only in B",
                    }
                )

    return pd.DataFrame(rows)


def build_period_comparison_df(
    result_a: Any,
    result_b: Any,
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> pd.DataFrame:
    """Build side-by-side period comparison DataFrame."""
    periods_a = _extract_period_results(result_a)
    periods_b = _extract_period_results(result_b)

    rows = []
    max_periods = max(len(periods_a), len(periods_b))

    for idx in range(max_periods):
        res_a = periods_a[idx] if idx < len(periods_a) else {}
        res_b = periods_b[idx] if idx < len(periods_b) else {}

        period_a = res_a.get("period", ())
        period_b = res_b.get("period", ())

        # Extract period dates
        out_start = ""
        out_end = ""
        for p in [period_a, period_b]:
            if len(p) > 2 and p[2]:
                out_start = str(p[2])
            if len(p) > 3 and p[3]:
                out_end = str(p[3])
        period_label = f"{out_start} → {out_end}".strip() or f"Period {idx + 1}"

        selected_a = res_a.get("selected_funds", []) or []
        selected_b = res_b.get("selected_funds", []) or []

        # Get turnover and costs
        def _get_turnover(res):
            return _coerce_numeric(
                res.get("turnover") or (res.get("risk_diagnostics") or {}).get("turnover")
            )

        rows.append(
            {
                "Period": period_label,
                f"{label_a} Funds Selected": len(selected_a),
                f"{label_b} Funds Selected": len(selected_b),
                "Δ Funds": len(selected_b) - len(selected_a),
                f"{label_a} Turnover": _get_turnover(res_a),
                f"{label_b} Turnover": _get_turnover(res_b),
                f"{label_a} Txn Cost": _coerce_numeric(res_a.get("transaction_cost")),
                f"{label_b} Txn Cost": _coerce_numeric(res_b.get("transaction_cost")),
            }
        )

    return pd.DataFrame(rows)


def build_raw_comparison_df(
    result_a: Any,
    result_b: Any,
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> pd.DataFrame:
    """Build pivot-ready raw data with Config column."""
    rows = []

    for config_label, result in [(label_a, result_a), (label_b, result_b)]:
        periods = _extract_period_results(result)

        for idx, res in enumerate(periods):
            period = res.get("period", ())
            out_start = str(period[2]) if len(period) > 2 else ""
            out_end = str(period[3]) if len(period) > 3 else ""
            period_label = f"{out_start} → {out_end}".strip() or f"Period {idx + 1}"

            selected = set(res.get("selected_funds", []) or [])
            weights = res.get("fund_weights", {}) or {}
            score_frame = res.get("score_frame")

            # Get all funds from score frame or selected
            if isinstance(score_frame, pd.DataFrame) and not score_frame.empty:
                all_funds = list(score_frame.index)
            else:
                all_funds = list(selected)

            for fund in all_funds:
                row = {
                    "Config": config_label,
                    "Period": period_label,
                    "Period_Num": idx + 1,
                    "Fund": fund,
                    "Selected": "Yes" if fund in selected else "No",
                    "Weight": (_coerce_numeric(weights.get(fund, 0)) if fund in selected else None),
                }

                # Add score metrics if available
                if isinstance(score_frame, pd.DataFrame) and fund in score_frame.index:
                    for col in score_frame.columns:
                        row[f"InSample_{col}"] = _coerce_numeric(score_frame.loc[fund, col])

                rows.append(row)

    return pd.DataFrame(rows)


def build_by_period_summary_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregated view: summary by period."""
    if raw_df.empty:
        return pd.DataFrame()

    summary = (
        raw_df.groupby(["Period", "Period_Num", "Config"])
        .agg(
            {
                "Selected": lambda x: (x == "Yes").sum(),
                "Weight": lambda x: x.dropna().mean() if x.notna().any() else None,
            }
        )
        .reset_index()
    )

    summary.columns = ["Period", "Period_Num", "Config", "Funds Selected", "Avg Weight"]
    return summary.sort_values(["Period_Num", "Config"])


def build_by_fund_summary_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregated view: summary by fund."""
    if raw_df.empty:
        return pd.DataFrame()

    summary = (
        raw_df.groupby(["Fund", "Config"])
        .agg(
            {
                "Selected": lambda x: (x == "Yes").sum(),
                "Weight": lambda x: x.dropna().mean() if x.notna().any() else None,
            }
        )
        .reset_index()
    )

    summary.columns = ["Fund", "Config", "Periods Selected", "Avg Weight"]
    return summary.sort_values(["Fund", "Config"])


def build_comparison_excel_workbook(
    result_a: Any,
    result_b: Any,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> bytes:
    """Generate comprehensive comparison Excel workbook.

    Sheets:
    1. Comparison Summary - Key metrics with winner indicators
    2. Selection Differences - Only periods/funds where selections differ
    3. Period Comparison - Side-by-side period data
    4. Raw Data - Pivot-ready format as Excel Table
    5. By Period - Pre-aggregated by period
    6. By Fund - Pre-aggregated by fund
    7. Config A - Full config JSON
    8. Config B - Full config JSON
    """
    import xlsxwriter

    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})

    # Define formats
    header_fmt = workbook.add_format(HEADER_FORMAT)
    percent_fmt = workbook.add_format({"num_format": PERCENT_FORMAT})
    winner_a_fmt = workbook.add_format({"bg_color": WINNER_GREEN, "bold": True})
    winner_b_fmt = workbook.add_format({"bg_color": "#BDD7EE", "bold": True})
    text_wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

    # Sheet 1: Comparison Summary
    summary_df = build_comparison_summary_df(result_a, result_b, label_a, label_b)
    if not summary_df.empty:
        ws = workbook.add_worksheet("Comparison Summary")
        ws.set_column(0, 0, 25)  # Metric
        ws.set_column(1, 2, 15)  # Config values
        ws.set_column(3, 4, 12)  # Delta, % Change
        ws.set_column(5, 5, 10)  # Winner

        # Write headers
        for col, header in enumerate(summary_df.columns):
            ws.write(0, col, header, header_fmt)

        # Write data with conditional formatting
        for row_idx, (_, row) in enumerate(summary_df.iterrows()):
            for col_idx, (col_name, value) in enumerate(row.items()):
                cell_format = None
                if col_name == "Winner":
                    if value == "A":
                        cell_format = winner_a_fmt
                    elif value == "B":
                        cell_format = winner_b_fmt
                elif col_name == "% Change" and value is not None:
                    cell_format = percent_fmt

                if pd.isna(value) or value is None:
                    ws.write(row_idx + 1, col_idx, "")
                elif cell_format:
                    ws.write(row_idx + 1, col_idx, value, cell_format)
                else:
                    ws.write(row_idx + 1, col_idx, value)

        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(summary_df), len(summary_df.columns) - 1)

    # Sheet 2: Selection Differences
    diff_df = build_selection_differences_df(result_a, result_b, label_a, label_b)
    if not diff_df.empty:
        ws = workbook.add_worksheet("Selection Differences")
        _write_dataframe_to_sheet(ws, diff_df, header_fmt, workbook)
    else:
        ws = workbook.add_worksheet("Selection Differences")
        ws.write(0, 0, "No selection differences found between configurations.")

    # Sheet 3: Period Comparison
    period_df = build_period_comparison_df(result_a, result_b, label_a, label_b)
    if not period_df.empty:
        ws = workbook.add_worksheet("Period Comparison")
        _write_dataframe_to_sheet(ws, period_df, header_fmt, workbook)

    # Sheet 4: Raw Data (Excel Table for pivots)
    raw_df = build_raw_comparison_df(result_a, result_b, label_a, label_b)
    if not raw_df.empty:
        ws = workbook.add_worksheet("Raw Data (Pivot-Ready)")

        # Write as Excel Table
        for col, header in enumerate(raw_df.columns):
            ws.write(0, col, header, header_fmt)

        for row_idx, (_, row) in enumerate(raw_df.iterrows()):
            for col_idx, value in enumerate(row):
                if pd.isna(value) or value is None:
                    ws.write(row_idx + 1, col_idx, "")
                else:
                    ws.write(row_idx + 1, col_idx, value)

        # Add Excel Table formatting
        ws.add_table(
            0,
            0,
            len(raw_df),
            len(raw_df.columns) - 1,
            {
                "name": "ComparisonData",
                "style": "Table Style Medium 2",
                "columns": [{"header": col} for col in raw_df.columns],
            },
        )

        ws.freeze_panes(1, 0)

        # Add instruction note
        ws.write(
            len(raw_df) + 3,
            0,
            "TIP: Select any cell in the table, then Insert > PivotTable to create custom views.",
            text_wrap_fmt,
        )

    # Sheet 5: By Period Summary
    by_period_df = build_by_period_summary_df(raw_df)
    if not by_period_df.empty:
        ws = workbook.add_worksheet("By Period")
        _write_dataframe_to_sheet(ws, by_period_df, header_fmt, workbook)

    # Sheet 6: By Fund Summary
    by_fund_df = build_by_fund_summary_df(raw_df)
    if not by_fund_df.empty:
        ws = workbook.add_worksheet("By Fund")
        _write_dataframe_to_sheet(ws, by_fund_df, header_fmt, workbook)

    # Sheet 7 & 8: Config JSONs
    for config, name in [
        (config_a, f"{label_a} Config"),
        (config_b, f"{label_b} Config"),
    ]:
        ws = workbook.add_worksheet(name[:31])  # Excel sheet name limit
        config_json = json.dumps(config, indent=2, sort_keys=True, default=str)
        ws.set_column(0, 0, 100)
        for row_idx, line in enumerate(config_json.split("\n")):
            ws.write(row_idx, 0, line)

    workbook.close()
    buffer.seek(0)
    return buffer.getvalue()


def _write_dataframe_to_sheet(
    ws,
    df: pd.DataFrame,
    header_fmt,
    workbook,
) -> None:
    """Helper to write a DataFrame to an Excel worksheet."""
    if df.empty:
        return

    # Set column widths
    for col_idx, col in enumerate(df.columns):
        max_len = max(len(str(col)), df[col].astype(str).str.len().max())
        ws.set_column(col_idx, col_idx, min(max_len + 2, 50))

    # Write headers
    for col, header in enumerate(df.columns):
        ws.write(0, col, header, header_fmt)

    # Write data
    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col_idx, value in enumerate(row):
            if pd.isna(value) or value is None:
                ws.write(row_idx + 1, col_idx, "")
            else:
                ws.write(row_idx + 1, col_idx, value)

    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)


def build_comparison_csv_bundle(
    result_a: Any,
    result_b: Any,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> bytes:
    """Generate ZIP bundle with comparison CSVs."""
    buffer = BytesIO()

    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        # Summary
        summary_df = build_comparison_summary_df(result_a, result_b, label_a, label_b)
        if not summary_df.empty:
            zf.writestr("comparison_summary.csv", summary_df.to_csv(index=False))

        # Selection differences
        diff_df = build_selection_differences_df(result_a, result_b, label_a, label_b)
        if not diff_df.empty:
            zf.writestr("selection_differences.csv", diff_df.to_csv(index=False))

        # Period comparison
        period_df = build_period_comparison_df(result_a, result_b, label_a, label_b)
        if not period_df.empty:
            zf.writestr("period_comparison.csv", period_df.to_csv(index=False))

        # Raw data
        raw_df = build_raw_comparison_df(result_a, result_b, label_a, label_b)
        if not raw_df.empty:
            zf.writestr("raw_data_pivot_ready.csv", raw_df.to_csv(index=False))

        # By period
        by_period_df = build_by_period_summary_df(raw_df)
        if not by_period_df.empty:
            zf.writestr("by_period_summary.csv", by_period_df.to_csv(index=False))

        # By fund
        by_fund_df = build_by_fund_summary_df(raw_df)
        if not by_fund_df.empty:
            zf.writestr("by_fund_summary.csv", by_fund_df.to_csv(index=False))

        # Configs
        zf.writestr(
            f"config_{label_a.replace(' ', '_')}.json",
            json.dumps(config_a, indent=2, sort_keys=True, default=str),
        )
        zf.writestr(
            f"config_{label_b.replace(' ', '_')}.json",
            json.dumps(config_b, indent=2, sort_keys=True, default=str),
        )

    buffer.seek(0)
    return buffer.getvalue()
