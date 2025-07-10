"""Helpers for multi-period export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Dict, Iterable

import pandas as pd

from ..export import export_to_excel, export_data


def _result_to_frame(res: Mapping[str, Any]) -> pd.DataFrame:
    """Return a metrics table for a single period."""
    stats = res.get("out_sample_stats", {})
    df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in res.get("benchmark_ir", {}).items():
        col = f"ir_{label}"
        df[col] = pd.Series(
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )
    return df


def build_frames(res: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Return per-period DataFrames plus a summary frame."""
    frames: Dict[str, pd.DataFrame] = {}
    for idx, period_res in enumerate(res.get("periods", []), start=1):
        frames[f"period_{idx}"] = _result_to_frame(period_res)
    summary_stats = res.get("summary", {}).get("stats", {})
    if summary_stats:
        frames["summary"] = pd.DataFrame(
            {k: vars(v) for k, v in summary_stats.items()}
        ).T
    return frames


def export_multi_period(
    res: Mapping[str, Any], out_dir: str, formats: Iterable[str]
) -> None:
    """Export multi-period results using the canonical exporters."""
    frames = build_frames(res)
    path = Path(out_dir)
    if any(f.lower() in {"excel", "xlsx"} for f in formats):
        export_to_excel(frames, str(path / "analysis.xlsx"))
        other = [f for f in formats if f.lower() not in {"excel", "xlsx"}]
        if other:
            export_data(frames, str(path / "analysis"), formats=other)
    else:
        export_data(frames, str(path / "analysis"), formats=formats)


__all__ = ["build_frames", "export_multi_period"]
