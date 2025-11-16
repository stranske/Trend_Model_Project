"""Guardrail helpers for the Streamlit interface.

The utilities defined here are intentionally independent from Streamlit so
they can be unit tested in isolation.  They provide three responsibilities:

* Infer runtime characteristics (data frequency, resource requirements).
* Produce a minimal configuration payload validated via the Pydantic models
  used by the backend CLI.  This keeps the UI aligned with the behaviour
  expected by ``trend_analysis.config``.
* Prepare a light‑weight "dry run" sample to quickly sanity‑check the
  pipeline without running the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from streamlit_app.config_bridge import build_config_payload, validate_payload

MAX_DRY_RUN_LOOKBACK_MONTHS = 12
MAX_DRY_RUN_OUT_MONTHS = 3


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Back-of-the-envelope resource estimate for a dataset."""

    rows: int
    columns: int
    approx_memory_mb: float
    estimated_runtime_s: float
    warnings: Tuple[str, ...]


@dataclass
class DryRunPlan:
    """Container describing the sample extracted for a dry run."""

    frame: pd.DataFrame
    lookback_months: int
    in_start: pd.Timestamp
    in_end: pd.Timestamp
    out_start: pd.Timestamp
    out_end: pd.Timestamp

    def sample_split(self) -> dict[str, str]:
        """Return the sample split mapping expected by ``Config``."""

        return {
            "in_start": self.in_start.strftime("%Y-%m"),
            "in_end": self.in_end.strftime("%Y-%m"),
            "out_start": self.out_start.strftime("%Y-%m"),
            "out_end": self.out_end.strftime("%Y-%m"),
        }

    def summary(self) -> dict[str, object]:
        """Provide a Streamlit-friendly summary of the plan."""

        return {
            "rows": int(self.frame.shape[0]),
            "columns": int(self.frame.shape[1]),
            "lookback_months": int(self.lookback_months),
            "window": {
                "in_start": self.in_start.strftime("%Y-%m-%d"),
                "in_end": self.in_end.strftime("%Y-%m-%d"),
                "out_start": self.out_start.strftime("%Y-%m-%d"),
                "out_end": self.out_end.strftime("%Y-%m-%d"),
            },
        }


def infer_frequency(index_like: Iterable[pd.Timestamp]) -> str:
    """Infer data frequency from a sequence of timestamps.

    Returns one of ``"D"``, ``"W"`` or ``"M"`` to align with the Pydantic
    configuration model.  The heuristic uses the median spacing between sorted
    timestamps and intentionally keeps the mapping coarse – Streamlit enforces
    monthly rebalancing and this is sufficient for guardrails.
    """

    try:
        idx = pd.to_datetime(list(index_like))
    except Exception:  # pragma: no cover - defensive guard for bogus input
        return "M"
    if len(idx) <= 1:
        return "M"
    ordered = pd.Series(idx).sort_values()
    deltas = ordered.diff().dropna()
    if deltas.empty:
        return "M"
    median_days = float(deltas.dt.days.median())
    if median_days <= 2:
        return "D"
    if median_days <= 8:
        return "W"
    return "M"


def estimate_resource_usage(rows: int, columns: int) -> ResourceEstimate:
    """Estimate runtime cost and produce guardrail warnings if necessary."""

    safe_rows = max(int(rows), 0)
    safe_cols = max(int(columns), 0)
    cells = safe_rows * max(safe_cols, 1)
    # Assume float64 values (~8 bytes) and a safety multiplier for pandas
    approx_memory_mb = cells * 8 * 1.5 / (1024**2)
    # Very coarse runtime heuristic: 75k cell operations per second
    estimated_runtime_s = cells / 75_000 if cells else 0.0
    warnings: List[str] = []
    if approx_memory_mb > 512:
        warnings.append(
            "Dataset likely exceeds 512 MB in-memory. Consider trimming columns "
            "before running the full simulation."
        )
    if estimated_runtime_s > 300:
        warnings.append(
            "Estimated runtime is over five minutes. Dry-run first or reduce the "
            "analysis horizon."
        )
    if safe_cols > 150:
        warnings.append(
            "More than 150 return series detected. Ranking that many funds may "
            "introduce look-ahead pressure from sparse histories."
        )
    return ResourceEstimate(
        rows=safe_rows,
        columns=safe_cols,
        approx_memory_mb=approx_memory_mb,
        estimated_runtime_s=estimated_runtime_s,
        warnings=tuple(warnings),
    )


def validate_startup_payload(
    *,
    csv_path: str | None,
    date_column: str,
    risk_target: float,
    timestamps: Iterable[pd.Timestamp],
) -> tuple[dict[str, object] | None, List[str]]:
    """Validate a minimal payload without importing heavyweight validators."""

    errors: List[str] = []
    csv_real: Path | None
    if not csv_path:
        errors.append("Upload must be saved to disk before validation.")
        csv_real = None
    else:
        csv_real = Path(csv_path)
        if not csv_real.exists():
            errors.append(f"CSV path '{csv_real}' does not exist.")

    if not isinstance(date_column, str) or not date_column.strip():
        errors.append("Date column must be a non-empty string.")

    try:
        risk_value = float(risk_target)
    except (TypeError, ValueError):
        errors.append("Risk target must be numeric.")
        risk_value = 0.0
    else:
        if risk_value <= 0:
            errors.append("Risk target must be greater than zero.")

    frequency = infer_frequency(timestamps)
    if errors:
        return None, errors

    payload = build_config_payload(
        csv_path=str(csv_real) if csv_real is not None else None,
        universe_membership_path=None,
        managers_glob=None,
        date_column=date_column,
        frequency=frequency,
        rebalance_calendar="NYSE",
        max_turnover=0.5,
        transaction_cost_bps=10.0,
        target_vol=risk_value,
    )
    base_dir = csv_real.parent if csv_real is not None else Path.cwd()
    validated, validation_error = validate_payload(payload, base_path=base_dir)
    if validation_error:
        error_lines = [
            line.strip() for line in validation_error.splitlines() if line.strip()
        ]
        if not error_lines:
            error_lines = [validation_error]
        return None, errors + error_lines
    return validated, []


def prepare_dry_run_plan(
    df: pd.DataFrame,
    lookback_months: int,
    *,
    horizon_months: int = 6,
) -> DryRunPlan:
    """Prepare a sample window for a dry run.

    The window intentionally pulls from the *start* of the dataset to avoid the
    temptation of peeking ahead.  When insufficient history exists the
    lookback is shortened automatically and a :class:`ValueError` is raised if
    the dataset is too small to support even a minimal out-of-sample test.
    """

    if df.empty:
        raise ValueError("No data available for dry run.")
    ordered = df.sort_index()
    periods = ordered.index.to_period("M")
    unique_periods = periods.unique().sort_values()
    if len(unique_periods) < 6:
        raise ValueError("Upload at least six months of returns to enable a dry run.")
    horizon = max(
        1,
        min(
            int(horizon_months),
            MAX_DRY_RUN_OUT_MONTHS,
            max(len(unique_periods) // 3, 1),
        ),
    )
    adjusted_lookback = max(
        3,
        min(
            int(lookback_months),
            len(unique_periods) - horizon,
            MAX_DRY_RUN_LOOKBACK_MONTHS,
        ),
    )
    total_needed = adjusted_lookback + horizon
    if total_needed > len(unique_periods):
        total_needed = len(unique_periods)
        adjusted_lookback = max(3, total_needed - horizon)
    selected_periods = unique_periods[:total_needed]
    mask = periods.isin(selected_periods)
    sample = ordered.loc[mask].copy()
    if sample.empty:
        raise ValueError("Unable to assemble a dry-run sample from the dataset.")
    out_end_period = selected_periods[-1]
    out_start_period = selected_periods[-horizon]
    in_end_period = out_start_period - 1
    in_start_period = in_end_period - (adjusted_lookback - 1)
    if in_start_period < selected_periods[0]:
        in_start_period = selected_periods[0]
    in_start = in_start_period.to_timestamp("M", "end")
    in_end = in_end_period.to_timestamp("M", "end")
    out_start = out_start_period.to_timestamp("M", "end")
    out_end = out_end_period.to_timestamp("M", "end")
    if in_start > in_end:
        raise ValueError("Insufficient history to support the requested lookback.")
    return DryRunPlan(
        frame=sample,
        lookback_months=int(adjusted_lookback),
        in_start=in_start,
        in_end=in_end,
        out_start=out_start,
        out_end=out_end,
    )


__all__ = [
    "DryRunPlan",
    "ResourceEstimate",
    "estimate_resource_usage",
    "infer_frequency",
    "prepare_dry_run_plan",
    "validate_startup_payload",
]
