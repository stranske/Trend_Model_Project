"""Export helpers for trend analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast
import inspect

import pandas as pd

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


def make_summary_formatter(
    res: Mapping[str, Any],
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> Callable[[Any, Any], None]:
    """Return a formatter function for the 'summary' Excel sheet."""

    @register_formatter_excel("summary")
    def fmt_summary(ws: Any, wb: Any) -> None:
        bold = wb.add_format({"bold": True})
        int0 = wb.add_format({"num_format": "0"})
        num2 = wb.add_format({"num_format": "0.00"})
        red = wb.add_format({"num_format": "0.00", "font_color": "red"})

        def safe(v: float | str | None) -> str | float:
            if pd.isna(v) or not pd.notna(v):
                return ""
            return cast(str | float, v)

        def pct(t: tuple[float, float, float, float, float]) -> list[float]:
            return [t[0] * 100, t[1] * 100, t[2], t[3], t[4] * 100]

        ws.write_row(0, 0, ["Vol-Adj Trend Analysis"], bold)
        ws.write_row(1, 0, [f"In:  {in_start} → {in_end}"], bold)
        ws.write_row(2, 0, [f"Out: {out_start} → {out_end}"], bold)

        row = 5
        for label, ins, outs in [
            ("Equal Weight", res["in_ew_stats"], res["out_ew_stats"]),
            ("User Weight", res["in_user_stats"], res["out_user_stats"]),
        ]:
            ws.write(row, 0, label, bold)
            ws.write(row, 1, safe(""))
            vals = pct(tuple(ins)) + pct(tuple(outs))
            fmts = ([num2] * 4 + [red]) * 2
            for col, (v, fmt) in enumerate(zip(vals, fmts), start=2):
                ws.write(row, col, safe(v), fmt)
            row += 1

        row += 1
        for fund, stat_in in res["in_sample_stats"].items():
            stat_out = res["out_sample_stats"][fund]
            ws.write(row, 0, fund, bold)
            wt = res["fund_weights"][fund]
            ws.write(row, 1, safe(wt * 100), int0)
            vals = pct(tuple(stat_in)) + pct(tuple(stat_out))
            fmts = ([num2] * 4 + [red]) * 2
            for col, (v, fmt) in enumerate(zip(vals, fmts), start=2):
                ws.write(row, col, safe(v), fmt)
            row += 1

        row += 1
        for idx, pair in res.get("index_stats", {}).items():
            in_idx = pair["in_sample"]
            out_idx = pair["out_sample"]
            ws.write(row, 0, idx, bold)
            ws.write(row, 1, safe(""))
            vals = pct(tuple(in_idx)) + pct(tuple(out_idx))
            fmts = ([num2] * 4 + [red]) * 2
            for col, (v, fmt) in enumerate(zip(vals, fmts), start=2):
                ws.write(row, col, safe(v), fmt)
            row += 1

    return fmt_summary


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
            prefix.with_name(f"{prefix.stem}_{name}.csv"), index=False, header=True
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
    "csv": export_to_csv,
    "json": export_to_json,
}


def export_data(
    data: Mapping[str, pd.DataFrame],
    output_path: str,
    formats: Iterable[str],
    formatter: Formatter | None = None,
) -> None:
    """Export ``data`` to the specified ``formats``."""
    for fmt in formats:
        exporter = EXPORTERS.get(fmt)
        if exporter is None:
            raise ValueError(f"Unsupported format: {fmt}")
        path = str(Path(output_path).with_suffix(f".{fmt}"))
        exporter(data, path, formatter)


__all__ = [
    "FORMATTERS_EXCEL",
    "register_formatter_excel",
    "reset_formatters_excel",
    "make_summary_formatter",
    "export_to_excel",
    "export_to_csv",
    "export_to_json",
    "export_data",
]
