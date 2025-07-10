"""Helpers for multi-period export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Dict, Iterable, List

import pandas as pd

from ..export import export_to_excel, export_data, make_summary_formatter


def _result_to_frame(res: Mapping[str, Any]) -> pd.DataFrame:
    """Return a formatted metrics table matching phase‑1 output."""

    def to_tuple(obj: Any) -> List[float]:
        if isinstance(obj, tuple):
            return list(obj)
        return [
            float(obj.cagr),
            float(obj.vol),
            float(obj.sharpe),
            float(obj.sortino),
            float(obj.information_ratio),
            float(obj.max_drawdown),
        ]

    def pct(t: Any) -> List[float]:
        vals = to_tuple(t)
        return [vals[0] * 100, vals[1] * 100, vals[2], vals[3], vals[4], vals[5] * 100]

    bench_labels = list(res.get("benchmark_ir", {}))

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

    rows: List[List[Any]] = []

    for label, ins, outs in [
        ("Equal Weight", res.get("in_ew_stats"), res.get("out_ew_stats")),
        ("User Weight", res.get("in_user_stats"), res.get("out_user_stats")),
    ]:
        if ins is None or outs is None:
            continue
        vals = pct(ins) + pct(outs)
        extra = [
            res.get("benchmark_ir", {})
            .get(b, {})
            .get("equal_weight" if label == "Equal Weight" else "user_weight", float("nan"))
            for b in bench_labels
        ]
        vals.extend(extra)
        rows.append([label, None, *vals])

    rows.append([None] * len(columns))

    in_stats = res.get("in_sample_stats", {})
    out_stats = res.get("out_sample_stats", {})
    weights = res.get("fund_weights", {})
    for fund, stat_in in in_stats.items():
        stat_out = out_stats.get(fund)
        if stat_out is None:
            continue
        weight = weights.get(fund, 0.0) * 100
        vals = pct(stat_in) + pct(stat_out)
        extra = [
            res.get("benchmark_ir", {}).get(b, {}).get(fund, float("nan"))
            for b in bench_labels
        ]
        vals.extend(extra)
        rows.append([fund, weight, *vals])

    return pd.DataFrame(rows, columns=columns)


def build_frames(res: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Return per-period DataFrames plus a summary frame."""
    frames: Dict[str, pd.DataFrame] = {}
    for idx, period_res in enumerate(res.get("periods", []), start=1):
        frames[f"period_{idx}"] = _result_to_frame(period_res)
    summary_res = res.get("summary")
    if summary_res:
        frames["summary"] = _result_to_frame(summary_res)
    return frames


def export_multi_period(
    res: Mapping[str, Any], out_dir: str, formats: Iterable[str]
) -> None:
    """Export multi-period results using the canonical exporters."""
    frames = build_frames(res)
    path = Path(out_dir)

    periods = res.get("periods", [])
    sheet_fmt = None
    if periods:
        first = periods[0]["period"]
        last = periods[-1]["period"]
        sheet_fmt = make_summary_formatter(
            res.get("summary", {}),
            first.in_start,
            first.in_end,
            last.out_start,
            last.out_end,
        )

    if any(f.lower() in {"excel", "xlsx"} for f in formats):
        export_to_excel(
            frames,
            str(path / "analysis.xlsx"),
            default_sheet_formatter=sheet_fmt,
        )
        other = [f for f in formats if f.lower() not in {"excel", "xlsx"}]
        if other:
            export_data(frames, str(path / "analysis"), formats=other)
    else:
        export_data(frames, str(path / "analysis"), formats=formats)


__all__ = ["build_frames", "export_multi_period"]
