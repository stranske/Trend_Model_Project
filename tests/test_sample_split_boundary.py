"""A21 (#5402): validator and window slicer must agree on boundary instants.

The ordering validator (`config/validation.py`) used to parse every
``sample_split`` label with ``pd.to_datetime`` -> month **start**, while the
window slicer (`stages/preprocessing.py`) resolves ``*_end`` labels via
``pd.Period.end_time`` -> month **end**. A month-granularity ``in_end`` therefore
denoted a different instant in validation than in slicing. Both paths now route
through :func:`trend_analysis.time_utils.resolve_period_bound`, so a given label
means one instant everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trend_analysis.config.validation import validate_config
from trend_analysis.time_utils import resolve_period_bound


def _base_config(tmp_path: Path) -> dict[str, Any]:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,A,B\n2020-01-31,0.0,0.0\n", encoding="utf-8")
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0.0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


def _has_path(result, path: str) -> bool:
    return any(issue.path == path for issue in result.errors)


def test_resolve_period_bound_is_bound_aware() -> None:
    # A bare month label resolves to the month start as a `start` bound and to
    # the month end as an `end` bound; finer labels parse directly.
    assert resolve_period_bound("2020-06", bound="start") == pd.Timestamp("2020-06-01")
    assert resolve_period_bound("2020-06", bound="end") == pd.Timestamp("2020-06-30")
    assert resolve_period_bound("2020-06-15", bound="end") == pd.Timestamp("2020-06-15")


def test_validator_matches_slicer_in_end_boundary(tmp_path: Path) -> None:
    # The slicer resolves a month `in_end` to month END and a mid-month
    # `out_start` day directly, so these windows genuinely overlap.
    in_end, out_start = "2020-06", "2020-06-20"
    slicer_in_end = resolve_period_bound(in_end, bound="end")
    slicer_out_start = resolve_period_bound(out_start, bound="start")
    assert slicer_in_end >= slicer_out_start  # genuine overlap under slicing

    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {
        "in_start": "2020-01",
        "in_end": in_end,
        "out_start": out_start,
        "out_end": "2020-12",
    }
    result = validate_config(cfg, base_path=tmp_path)

    # The validator must see the same overlap the slicer would. Under the old
    # month-start parse it resolved `in_end` to 2020-06-01 and wrongly accepted
    # this config -- the deliberate-break gate for #5402.
    assert not result.valid
    assert _has_path(result, "sample_split.out_start")


def test_validator_accepts_clean_month_split(tmp_path: Path) -> None:
    # A valid, non-overlapping month-granularity split must remain valid:
    # the boundary alignment must not change in/out semantics for good configs.
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {
        "in_start": "2020-01",
        "in_end": "2020-06",
        "out_start": "2020-07",
        "out_end": "2020-12",
    }
    result = validate_config(cfg, base_path=tmp_path)

    assert not any(issue.path.startswith("sample_split.") for issue in result.errors)
