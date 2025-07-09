"""Export helpers for trend analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast
import inspect

import pandas as pd
import numpy as np

Formatter = Callable[[pd.DataFrame], pd.DataFrame]


FORMATTERS_EXCEL: dict[str, Callable[[Any, Any], None]] = {}


def register_formatter_excel(
    category: str,
) -> Callable[[Callable[[Any, Any], None]], Callable[[Any, Any], None]]:
    """Register an Excel formatter under ``category``."""

    def decorator(fn: Callable[[Any, Any], None]) -> Callable[[Any, Any], None]:
        FORMATTERS_EXCEL[category] = fn
        return fn

    return decorator


def reset_formatters_excel() -> None:
    """Clear all registered Excel formatters."""
    FORMATTERS_EXCEL.clear()


def _build_summary_formatter(
    res: Mapping[str, Any],
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> Callable[[Any, Any], None]:
    """Return a formatter function for a summary sheet."""

    def fmt_summary(ws: Any, wb: Any) -> None:
        bold = wb.add_format({"bold": True})
        int0 = wb.add_format({"num_format": "0"})
        num2 = wb.add_format({"num_format": "0.00"})
        pct2 = wb.add_format({"num_format": "0.00%"})
        red = wb.add_format({"num_format": "0.00", "font_color": "red"})

        def safe(v: float | str | None) -> str | float:
            if pd.isna(v) or not pd.notna(v):
                return ""
            return cast(str | float, v)

        def to_tuple(obj: Any) -> tuple[float, float, float, float, float, float]:
            if isinstance(obj, tuple):
                return cast(tuple[float, float, float, float, float, float], obj)
            return (
                cast(float, obj.cagr),
                cast(float, obj.vol),
                cast(float, obj.sharpe),
                cast(float, obj.sortino),
                cast(float, obj.information_ratio),
                cast(float, obj.max_drawdown),
            )

        def pct(t: Any) -> list[float]:
            tup = to_tuple(t)
            return [
                tup[0] * 100,
                tup[1] * 100,
                tup[2],
                tup[3],
                tup[4],
                tup[5] * 100,
            ]

        ws.write_row(0, 0, ["Vol-Adj Trend Analysis"], bold)
        ws.write_row(1, 0, [f"In:  {in_start} → {in_end}"], bold)
        ws.write_row(2, 0, [f"Out: {out_start} → {out_end}"], bold)
        bench_labels = list(res.get("benchmark_ir", {}))
        headers = [
            "Name",
            "Weight",
            "IS CAGR",
            "IS Vol",
            "IS Sharpe",
            "IS Sortino",
            "IS IR",
            "IS MaxDD",
            "OS CAGR",
            "OS Vol",
            "OS Sharpe",
            "OS Sortino",
            "OS IR",
        ]
        headers.extend([f"OS IR {b}" for b in bench_labels])
        headers.append("OS MaxDD")
        ws.write_row(4, 0, headers, bold)
        for idx, h in enumerate(headers):
            ws.set_column(idx, idx, len(h) + 2)
        ws.freeze_panes(5, 0)
        numeric_fmts: list[Any] = []
        for h in headers[2:]:
            if "MaxDD" in h:
                numeric_fmts.append(red)
            elif "CAGR" in h or "Vol" in h:
                numeric_fmts.append(pct2)
            else:
                numeric_fmts.append(num2)

        row = 5
        for label, ins, outs in [
            ("Equal Weight", res["in_ew_stats"], res["out_ew_stats"]),
            ("User Weight", res["in_user_stats"], res["out_user_stats"]),
        ]:
            ws.write(row, 0, label, bold)
            ws.write(row, 1, safe(""))
            vals = pct(ins) + pct(outs)
            extra = [
                res.get("benchmark_ir", {})
                .get(b, {})
                .get("equal_weight" if label == "Equal Weight" else "user_weight", "")
                for b in bench_labels
            ]
            fmts = numeric_fmts
            vals.extend(extra)
            for col, (v, fmt) in enumerate(zip(vals, fmts), start=2):
                ws.write(row, col, safe(v), fmt)
            row += 1

        row += 1
        for fund, stat_in in res["in_sample_stats"].items():
            stat_out = res["out_sample_stats"][fund]
            ws.write(row, 0, fund, bold)
            wt = res["fund_weights"][fund]
            ws.write(row, 1, safe(wt * 100), int0)
            vals = pct(stat_in) + pct(stat_out)
            extra = [
                res.get("benchmark_ir", {}).get(b, {}).get(fund, "")
                for b in bench_labels
            ]
            fmts = numeric_fmts
            vals.extend(extra)
            for col, (v, fmt) in enumerate(zip(vals, fmts), start=2):
                ws.write(row, col, safe(v), fmt)
            row += 1

        row += 1
        ws.autofilter(4, 0, row - 1, len(headers) - 1)

    return fmt_summary


def make_summary_formatter(
    res: Mapping[str, Any],
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> Callable[[Any, Any], None]:
    """Return and register a formatter for the ``summary`` sheet."""

    fmt = _build_summary_formatter(res, in_start, in_end, out_start, out_end)
    return register_formatter_excel("summary")(fmt)


def make_period_formatter(
    sheet: str,
    res: Mapping[str, Any],
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> Callable[[Any, Any], None]:
    """Return and register a formatter for a per-period sheet."""

    fmt = _build_summary_formatter(res, in_start, in_end, out_start, out_end)
    return register_formatter_excel(sheet)(fmt)


def format_summary_text(
    res: Mapping[str, Any],
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> str:
    """Return a plain-text summary table similar to the Excel output."""

    def safe(val: float | str | None) -> str:
        if pd.isna(val) or not pd.notna(val):
            return ""
        if isinstance(val, (int, float)):
            return f"{val:.2f}"
        return cast(str, val)

    def to_tuple(obj: Any) -> tuple[float, float, float, float, float, float]:
        if isinstance(obj, tuple):
            return cast(tuple[float, float, float, float, float, float], obj)
        return (
            cast(float, obj.cagr),
            cast(float, obj.vol),
            cast(float, obj.sharpe),
            cast(float, obj.sortino),
            cast(float, obj.information_ratio),
            cast(float, obj.max_drawdown),
        )

    def pct(t: Any) -> list[float]:
        a = to_tuple(t)
        return [a[0] * 100, a[1] * 100, a[2], a[3], a[4], a[5] * 100]

    bench_map = cast(Mapping[str, Mapping[str, float]], res.get("benchmark_ir", {}))
    bench_labels = list(bench_map)
    columns = [
        "Name",
        "Weight",
        "IS CAGR",
        "IS Vol",
        "IS Sharpe",
        "IS Sortino",
        "IS IR",
        "IS MaxDD",
        "OS CAGR",
        "OS Vol",
        "OS Sharpe",
        "OS Sortino",
        "OS IR",
    ]
    columns.extend([f"OS IR {b}" for b in bench_labels])
    columns.append("OS MaxDD")

    rows: list[list[str | float | None]] = []

    for label, ins, outs in [
        ("Equal Weight", res["in_ew_stats"], res["out_ew_stats"]),
        ("User Weight", res["in_user_stats"], res["out_user_stats"]),
    ]:
        vals = pct(ins) + pct(outs)
        extra = [
            res.get("benchmark_ir", {})
            .get(b, {})
            .get(
                "equal_weight" if label == "Equal Weight" else "user_weight",
                float("nan"),
            )
            for b in bench_labels
        ]
        vals.extend(extra)
        rows.append([label, None, *vals])

    rows.append([None] * len(columns))

    for fund, stat_in in res["in_sample_stats"].items():
        stat_out = res["out_sample_stats"][fund]
        weight = res["fund_weights"][fund] * 100
        vals = pct(stat_in) + pct(stat_out)
        extra = [
            res.get("benchmark_ir", {}).get(b, {}).get(fund, float("nan"))
            for b in bench_labels
        ]
        vals.extend(extra)
        rows.append([fund, weight, *vals])

    df = pd.DataFrame(rows, columns=columns)
    df_formatted = df.map(safe)
    header = [
        "Vol-Adj Trend Analysis",
        f"In:  {in_start} → {in_end}",
        f"Out: {out_start} → {out_end}",
        "",
        df_formatted.to_string(index=False),
    ]
    return "\n".join(header)


def _ensure_dir(path: Path) -> None:
    """Create parent directories for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_format(df: pd.DataFrame, formatter: Formatter | None) -> pd.DataFrame:
    """Return ``df`` after applying the optional ``formatter``."""
    return formatter(df) if formatter else df


def export_to_excel(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
    default_sheet_formatter: Callable[[Any, Any], None] | None = None,
) -> None:
    """Export dataframes to an Excel workbook.

    Each key in ``data`` becomes a sheet.  After writing a sheet, a
    formatter function is looked up in :data:`FORMATTERS_EXCEL` by sheet name
    and applied.  If absent, ``default_sheet_formatter`` is used if provided.
    The ``formatter`` argument still allows per-frame transformations before
    writing.
    """
    path = Path(output_path)
    _ensure_dir(path)

    df_formatter = formatter
    if formatter and default_sheet_formatter is None:
        params = list(inspect.signature(formatter).parameters)
        if len(params) != 1:
            default_sheet_formatter = cast(Callable[[Any, Any], None], formatter)
            # backward compat
            df_formatter = None

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # We must iterate over the mapping of DataFrames so each becomes its own
        # sheet. A vectorised approach would obscure the intent here.
        for sheet, df in data.items():
            formatted = _apply_format(df, df_formatter)
            formatted.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            fmt = FORMATTERS_EXCEL.get(sheet, default_sheet_formatter)
            if fmt:
                fmt(ws, writer.book)


def export_to_csv(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
) -> None:
    """Export each dataframe to an individual CSV file using ``output_path`` as prefix."""
    prefix = Path(output_path)
    _ensure_dir(prefix)
    # Looping over the ``data`` dictionary ensures each frame gets its own file.
    for name, df in data.items():
        formatted = _apply_format(df, formatter)
        formatted.to_csv(
            prefix.with_name(f"{prefix.stem}_{name}.csv"),
            index=True,
            header=True,
        )


def export_to_json(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formatter: Formatter | None = None,
) -> None:
    """Export each dataframe to an individual JSON file using ``output_path`` as prefix."""
    prefix = Path(output_path)
    _ensure_dir(prefix)
    # Iterate over the mapping so each DataFrame is written to its own JSON file.
    for name, df in data.items():
        formatted = _apply_format(df, formatter)
        formatted.to_json(
            prefix.with_name(f"{prefix.stem}_{name}.json"), orient="records", indent=2
        )


EXPORTERS: dict[
    str, Callable[[Mapping[str, pd.DataFrame], str, Formatter | None], None]
] = {
    "xlsx": export_to_excel,
    # ``excel`` is kept for backward compatibility with older configs/UI
    "excel": export_to_excel,
    "csv": export_to_csv,
    "json": export_to_json,
}


def metrics_from_result(res: Mapping[str, object]) -> pd.DataFrame:
    """Return a metrics DataFrame identical to :func:`pipeline.run` output."""
    from .pipeline import _Stats  # lazy import to avoid cycle

    stats = cast(Mapping[str, _Stats], res.get("out_sample_stats", {}))
    df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in cast(
        Mapping[str, Mapping[str, float]], res.get("benchmark_ir", {})
    ).items():
        col = f"ir_{label}"
        df[col] = pd.Series(
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )
    return df


def summary_frame_from_result(res: Mapping[str, object]) -> pd.DataFrame:
    """Return a DataFrame mirroring the Phase-1 summary table."""

    from .pipeline import _Stats  # lazy import to avoid cycle

    def to_tuple(obj: Any) -> tuple[float, float, float, float, float, float]:
        if isinstance(obj, tuple):
            return cast(tuple[float, float, float, float, float, float], obj)
        s = cast(_Stats, obj)
        return (
            float(s.cagr),
            float(s.vol),
            float(s.sharpe),
            float(s.sortino),
            float(s.information_ratio),
            float(s.max_drawdown),
        )

    def pct(t: Any) -> list[float]:
        a = to_tuple(t)
        return [a[0] * 100, a[1] * 100, a[2], a[3], a[4], a[5] * 100]

    bench_map = cast(Mapping[str, Mapping[str, float]], res.get("benchmark_ir", {}))
    bench_labels = list(bench_map)
    columns = [
        "Name",
        "Weight",
        "IS CAGR",
        "IS Vol",
        "IS Sharpe",
        "IS Sortino",
        "IS IR",
        "IS MaxDD",
        "OS CAGR",
        "OS Vol",
        "OS Sharpe",
        "OS Sortino",
        "OS IR",
    ]
    columns.extend([f"OS IR {b}" for b in bench_labels])
    columns.append("OS MaxDD")

    rows: list[list[Any]] = []

    for label, ins, outs in [
        ("Equal Weight", res["in_ew_stats"], res["out_ew_stats"]),
        ("User Weight", res["in_user_stats"], res["out_user_stats"]),
    ]:
        vals = pct(ins) + pct(outs)
        extra = [
            bench_map.get(b, {}).get(
                "equal_weight" if label == "Equal Weight" else "user_weight",
                pd.NA,
            )
            for b in bench_labels
        ]
        rows.append([label, pd.NA, *vals, *extra])

    rows.append([pd.NA] * len(columns))

    for fund, stat_in in cast(Mapping[str, _Stats], res["in_sample_stats"]).items():
        stat_out = cast(Mapping[str, _Stats], res["out_sample_stats"])[fund]
        weight = cast(Mapping[str, float], res["fund_weights"])[fund] * 100
        vals = pct(stat_in) + pct(stat_out)
        extra = [bench_map.get(b, {}).get(fund, pd.NA) for b in bench_labels]
        rows.append([fund, weight, *vals, *extra])

    return pd.DataFrame(rows, columns=columns)


def combined_summary_result(
    results: Iterable[Mapping[str, object]],
) -> Mapping[str, object]:
    """Return an aggregated result dict across all periods."""

    from collections import defaultdict

    from .pipeline import _compute_stats, calc_portfolio_returns

    fund_in: dict[str, list[pd.Series]] = defaultdict(list)
    fund_out: dict[str, list[pd.Series]] = defaultdict(list)
    ew_in_series: list[pd.Series] = []
    ew_out_series: list[pd.Series] = []
    user_in_series: list[pd.Series] = []
    user_out_series: list[pd.Series] = []
    weight_sum: dict[str, float] = defaultdict(float)
    periods = 0

    for res in results:
        in_df = cast(pd.DataFrame, res.get("in_sample_scaled"))
        out_df = cast(pd.DataFrame, res.get("out_sample_scaled"))
        ew_map = cast(Mapping[str, float], res.get("ew_weights", {}))
        fund_map = cast(Mapping[str, float], res.get("fund_weights", {}))
        ew_w = [ew_map.get(c, 0.0) for c in in_df.columns]
        user_w = [fund_map.get(c, 0.0) for c in in_df.columns]
        ew_in_series.append(calc_portfolio_returns(np.array(ew_w), in_df))
        ew_out_series.append(calc_portfolio_returns(np.array(ew_w), out_df))
        user_in_series.append(calc_portfolio_returns(np.array(user_w), in_df))
        user_out_series.append(calc_portfolio_returns(np.array(user_w), out_df))
        for c in in_df.columns:
            fund_in[c].append(in_df[c])
            weight_sum[c] += fund_map.get(c, 0.0)
        for c in out_df.columns:
            fund_out[c].append(out_df[c])
        periods += 1

    rf_in = pd.Series(0.0, index=pd.concat(ew_in_series).index)
    rf_out = pd.Series(0.0, index=pd.concat(ew_out_series).index)
    in_ew_stats = _compute_stats(pd.DataFrame({"ew": pd.concat(ew_in_series)}), rf_in)[
        "ew"
    ]
    out_ew_stats = _compute_stats(
        pd.DataFrame({"ew": pd.concat(ew_out_series)}), rf_out
    )["ew"]
    in_user_stats = _compute_stats(
        pd.DataFrame({"user": pd.concat(user_in_series)}), rf_in
    )["user"]
    out_user_stats = _compute_stats(
        pd.DataFrame({"user": pd.concat(user_out_series)}), rf_out
    )["user"]

    in_stats = {
        f: _compute_stats(pd.DataFrame({f: pd.concat(s)}), rf_in)[f]
        for f, s in fund_in.items()
    }
    out_stats = {
        f: _compute_stats(pd.DataFrame({f: pd.concat(s)}), rf_out)[f]
        for f, s in fund_out.items()
    }

    fund_weights = {f: weight_sum[f] / periods for f in weight_sum}

    return {
        "in_ew_stats": in_ew_stats,
        "out_ew_stats": out_ew_stats,
        "in_user_stats": in_user_stats,
        "out_user_stats": out_user_stats,
        "in_sample_stats": in_stats,
        "out_sample_stats": out_stats,
        "fund_weights": fund_weights,
        "benchmark_ir": {},
    }


def period_frames_from_results(
    results: Iterable[Mapping[str, object]],
) -> Mapping[str, pd.DataFrame]:
    """Return a mapping of sheet names to summary frames for each period."""

    frames: dict[str, pd.DataFrame] = {}
    for idx, res in enumerate(results, start=1):
        period = res.get("period")
        if isinstance(period, (list, tuple)) and len(period) >= 4:
            sheet = str(period[3])
        else:
            sheet = f"period_{idx}"
        frames[sheet] = summary_frame_from_result(res)
    return frames


def workbook_frames_from_results(
    results: Iterable[Mapping[str, object]],
) -> Mapping[str, pd.DataFrame]:
    """Return per-period frames plus a combined summary frame."""

    results_list = list(results)
    frames = period_frames_from_results(results_list)
    if results_list:
        summary = combined_summary_result(results_list)
        frames["summary"] = summary_frame_from_result(summary)
    return frames


def export_multi_period_metrics(
    results: Iterable[Mapping[str, object]],
    output_path: str,
    *,
    formats: Iterable[str] = ("xlsx",),
    include_metrics: bool = False,
) -> None:
    """Export per-period metrics using the canonical exporters.

    Parameters
    ----------
    results:
        Sequence of result dictionaries as produced by
        :func:`multi_period.engine.run`.
    output_path:
        File path prefix for the exported artefacts.
    formats:
        Output formats understood by :func:`export_data`.
    include_metrics:
        If ``True`` also emit the raw metrics frame for each period,
        mirroring the single-period "metrics" sheet.
    """

    excel_formats = [f for f in formats if f.lower() in {"excel", "xlsx"}]
    other_formats = [f for f in formats if f.lower() not in {"excel", "xlsx"}]
    excel_data: dict[str, pd.DataFrame] = {}
    other_data: dict[str, pd.DataFrame] = {}
    reset_formatters_excel()

    results_list = list(results)

    if other_formats:
        frames = workbook_frames_from_results(results_list)
        period_frames = [(k, v) for k, v in frames.items() if k != "summary"]
        combined = pd.concat(
            [df.assign(Period=name) for name, df in period_frames],
            ignore_index=True,
        ) if period_frames else pd.DataFrame()
        other_data["periods"] = combined
        if "summary" in frames:
            other_data["summary"] = frames["summary"]
        if include_metrics:
            metrics_frames: list[pd.DataFrame] = []
            for idx, res in enumerate(results_list, start=1):
                period = res.get("period")
                sheet = (
                    str(period[3])
                    if isinstance(period, (list, tuple)) and len(period) >= 4
                    else f"period_{idx}"
                )
                metrics = metrics_from_result(res)
                metrics.insert(0, "Period", sheet)
                metrics_frames.append(metrics)
            if metrics_frames:
                other_data["metrics"] = pd.concat(metrics_frames, ignore_index=True)
            if results_list:
                other_data["metrics_summary"] = metrics_from_result(
                    combined_summary_result(results_list)
                )

    for idx, res in enumerate(results_list, start=1):
        period = res.get("period")
        if isinstance(period, (list, tuple)) and len(period) >= 4:
            in_s, in_e, out_s, out_e = map(str, period[:4])
            sheet = str(out_e)
        else:
            in_s = in_e = out_s = out_e = ""
            sheet = f"period_{idx}"

        if excel_formats:
            excel_data[sheet] = pd.DataFrame()
            make_period_formatter(sheet, res, in_s, in_e, out_s, out_e)
            if include_metrics:
                excel_data[f"metrics_{sheet}"] = metrics_from_result(res)

    if results_list:
        summary = combined_summary_result(results_list)
        if excel_formats:
            excel_data["summary"] = pd.DataFrame()
            first = results_list[0].get("period")
            last = results_list[-1].get("period")
            if isinstance(first, (list, tuple)) and isinstance(last, (list, tuple)):
                make_period_formatter(
                    "summary",
                    summary,
                    str(first[0]),
                    str(first[1]),
                    str(last[2]),
                    str(last[3]),
                )
            else:
                make_period_formatter("summary", summary, "", "", "", "")
            if include_metrics:
                excel_data["metrics_summary"] = metrics_from_result(summary)


    if excel_formats:
        export_data(excel_data, output_path, formats=excel_formats)
    if other_formats:
        export_data(other_data, output_path, formats=other_formats)


def export_data(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formats: Iterable[str],
    formatter: Formatter | None = None,
) -> None:
    """Export ``data`` to the specified ``formats``."""
    for fmt in formats:
        fmt_norm = fmt.lower()
        fmt_norm = "xlsx" if fmt_norm == "excel" else fmt_norm
        exporter = EXPORTERS.get(fmt_norm)
        if exporter is None:
            raise ValueError(f"Unsupported format: {fmt}")
        path = str(Path(output_path).with_suffix(f".{fmt_norm}"))
        exporter(data, path, formatter)


__all__ = [
    "FORMATTERS_EXCEL",
    "register_formatter_excel",
    "reset_formatters_excel",
    "make_summary_formatter",
    "make_period_formatter",
    "format_summary_text",
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_data",
    "metrics_from_result",
    "combined_summary_result",
    "summary_frame_from_result",
    "period_frames_from_results",
    "workbook_frames_from_results",
    "export_multi_period_metrics",
]
