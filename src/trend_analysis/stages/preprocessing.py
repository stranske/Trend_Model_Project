from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..diagnostics import PipelineReasonCode, PipelineResult, pipeline_failure
from ..risk import periods_per_year_from_code
from ..time_utils import align_calendar
from ..timefreq import MONTHLY_DATE_FREQ
from ..util.frequency import FrequencySummary, detect_frequency
from ..util.missing import MissingPolicyResult, apply_missing_policy

__all__ = [
    "_PreprocessStage",
    "_WindowStage",
    "_build_sample_windows",
    "_frequency_label",
    "_prepare_input_data",
    "_prepare_preprocess_stage",
    "_preprocessing_summary",
]


def _frequency_label(code: str) -> str:
    return {"D": "Daily", "W": "Weekly", "M": "Monthly"}.get(code, code)


def _preprocessing_summary(freq_code: str, *, normalised: bool, missing_summary: str | None) -> str:
    cadence = _frequency_label(freq_code)
    cadence_text = f"Cadence: {cadence}"
    if normalised and freq_code != "M":
        cadence_text += " → monthly"
    elif freq_code == "M":
        cadence_text += " (month-end)"
    parts = [cadence_text]
    if missing_summary:
        parts.append(f"Missing data: {missing_summary}")
    return "; ".join(parts)


@dataclass(slots=True)
class _PreprocessStage:
    df: pd.DataFrame
    date_col: str
    freq_summary: FrequencySummary
    missing_result: MissingPolicyResult
    preprocess_info: dict[str, object]
    frequency_payload: dict[str, object]
    missing_payload: Mapping[str, object]
    alignment_info: Mapping[str, Any]
    min_floor: float
    warmup: int
    value_cols_all: list[str]
    df_original: pd.DataFrame
    periods_per_year: float


@dataclass(slots=True)
class _WindowStage:
    in_df: pd.DataFrame
    out_df: pd.DataFrame
    in_start: pd.Timestamp
    in_end: pd.Timestamp
    out_start: pd.Timestamp
    out_end: pd.Timestamp
    periods_per_year: float
    date_col: str


def _prepare_preprocess_stage(
    df: pd.DataFrame | None,
    *,
    floor_vol: float | None,
    warmup_periods: int,
    missing_policy: str | Mapping[str, str] | None,
    missing_limit: int | Mapping[str, int | None] | None,
    stats_cfg: Any | None,
    periods_per_year_override: float | None,
    allow_risk_free_fallback: bool | None,
) -> _PreprocessStage | PipelineResult:
    if df is None:
        return pipeline_failure(PipelineReasonCode.INPUT_NONE)

    attrs_copy = dict(getattr(df, "attrs", {}))
    if type(df) is not pd.DataFrame:  # noqa: E721 - intentional exact type check
        df = pd.DataFrame(df)
    if attrs_copy:
        df.attrs = attrs_copy

    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    calendar_settings = attrs_copy.get("calendar_settings", {})

    date_probe = pd.to_datetime(df[date_col], errors="coerce")
    if not date_probe.notna().any():
        return pipeline_failure(
            PipelineReasonCode.NO_VALID_DATES,
            context={"date_column": date_col},
        )

    try:
        df = align_calendar(
            df,
            date_col=date_col,
            frequency=calendar_settings.get("frequency"),
            timezone=calendar_settings.get("timezone", "UTC"),
            holiday_calendar=calendar_settings.get("holiday_calendar"),
        )
    except ValueError as exc:
        message = str(exc)
        if "contains no valid timestamps" in message or "All rows were removed" in message:
            return pipeline_failure(
                PipelineReasonCode.CALENDAR_ALIGNMENT_WIPE,
                context={"error": message},
            )
        raise
    alignment_info = df.attrs.get("calendar_alignment", {})

    try:
        min_floor = float(floor_vol) if floor_vol is not None else 0.0
    except (TypeError, ValueError):  # pragma: no cover - defensive
        min_floor = 0.0
    if min_floor < 0:
        min_floor = 0.0
    try:
        warmup = int(warmup_periods)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        warmup = 0
    if warmup < 0:
        warmup = 0

    na_cfg = getattr(stats_cfg, "na_as_zero_cfg", None) if stats_cfg else None
    enforce_complete = not (na_cfg and bool(na_cfg.get("enabled", False)))
    if allow_risk_free_fallback is True:
        enforce_complete = False

    df_prepared, freq_summary, missing_result, normalised = _prepare_input_data(
        df,
        date_col=date_col,
        missing_policy=missing_policy,
        missing_limit=missing_limit,
        enforce_completeness=enforce_complete,
    )
    if df_prepared.empty:
        return pipeline_failure(
            PipelineReasonCode.PREPARED_FRAME_EMPTY,
            context={
                "missing_summary": getattr(missing_result, "summary", None),
                "dropped_assets": list(getattr(missing_result, "dropped_assets", ())),
            },
        )
    initial_value_cols = [c for c in df_prepared.columns if c != date_col]
    df_original = df_prepared.copy()
    prepared_attrs = dict(getattr(df_prepared, "attrs", {}))
    value_cols_all = [c for c in df_prepared.columns if c != date_col]

    if not initial_value_cols or not value_cols_all:
        reason = (
            PipelineReasonCode.NO_VALUE_COLUMNS
            if not initial_value_cols
            else PipelineReasonCode.INSUFFICIENT_COLUMNS
        )
        return pipeline_failure(
            reason,
            context={
                "stage": "post-prep-column-check",
                "initial_value_cols": len(initial_value_cols),
                "post_probe_value_cols": len(value_cols_all),
            },
        )

    if not isinstance(df_original, pd.DataFrame):
        df = pd.DataFrame(df_original)
    else:
        df = df_original
    if prepared_attrs:
        df.attrs = prepared_attrs

    if df.empty or df.shape[1] <= 1:
        return pipeline_failure(
            PipelineReasonCode.INSUFFICIENT_COLUMNS,
            context={"columns": df.shape[1], "rows": df.shape[0]},
        )

    frequency_payload = {
        "code": freq_summary.code,
        "label": freq_summary.label,
        "target": freq_summary.target,
        "target_label": freq_summary.target_label,
        "resampled": freq_summary.resampled,
    }
    periods_per_year = periods_per_year_override or periods_per_year_from_code(freq_summary.target)
    missing_payload = {
        "policy": missing_result.default_policy,
        "policy_map": missing_result.policy,
        "limit": missing_result.default_limit,
        "limit_map": missing_result.limit,
        "dropped_assets": list(missing_result.dropped_assets),
        "filled_assets": {asset: count for asset, count in missing_result.filled_cells},
        "total_filled": missing_result.total_filled,
    }

    preprocess_info = {
        "input_frequency": frequency_payload["code"],
        "input_frequency_details": frequency_payload,
        "resampled_to_monthly": normalised,
        "missing": missing_result,
        "missing_data_policy": missing_payload,
    }
    preprocess_info["summary"] = _preprocessing_summary(
        freq_summary.code,
        normalised=normalised,
        missing_summary=missing_result.summary,
    )
    preprocess_info["calendar_alignment"] = alignment_info

    return _PreprocessStage(
        df=df,
        date_col=date_col,
        freq_summary=freq_summary,
        missing_result=missing_result,
        preprocess_info=preprocess_info,
        frequency_payload=frequency_payload,
        missing_payload=missing_payload,
        alignment_info=alignment_info,
        min_floor=min_floor,
        warmup=warmup,
        value_cols_all=value_cols_all,
        df_original=df_original,
        periods_per_year=periods_per_year,
    )


def _build_sample_windows(
    preprocess: _PreprocessStage,
    *,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
) -> _WindowStage | PipelineResult:
    def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
        # Align arbitrary timestamps to month-end while preserving timezone.
        return ts + pd.offsets.MonthEnd(0)

    def _window_month_ends(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        # Build an inclusive month-end index for the window.
        start_me = _month_end(pd.Timestamp(start))
        end_me = _month_end(pd.Timestamp(end))
        if start_me > end_me:
            return pd.DatetimeIndex([], name=preprocess.date_col)
        return pd.date_range(start=start_me, end=end_me, freq=MONTHLY_DATE_FREQ)

    def _fill_missing_months(
        frame: pd.DataFrame,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        allow_empty_reindex: bool,
    ) -> pd.DataFrame:
        """Ensure the frame contains every month-end between start and end.

        Even after monthly resampling, certain upstream operations (e.g., strict
        completeness enforcement) can remove entire month rows. When the
        configured missing-data policy is not "drop", we want windows to retain
        a contiguous monthly index so both in-sample and out-of-sample blocks
        line up with the requested calendar.
        """

        expected = _window_month_ends(start, end)
        if expected.empty:
            return frame
        if frame.empty:
            # Only reindex an empty slice when the caller explicitly allows it.
            # This is used when one side of the sample split has data and we
            # want to preserve calendar alignment for downstream diagnostics.
            return frame.reindex(expected) if allow_empty_reindex else frame
        if (
            isinstance(frame.index, pd.DatetimeIndex)
            and frame.index.tz is not None
            and expected.tz is None
        ):
            # If the data is tz-aware, keep the expected range consistent.
            expected = expected.tz_localize(frame.index.tz)

        reindexed = frame.reindex(expected)

        policy_default = str(
            getattr(preprocess.missing_result, "default_policy", "drop") or "drop"
        ).lower()
        if policy_default == "drop":
            return reindexed

        # Re-apply missing policy to fill any newly inserted rows.
        # Use a reconstructed mapping so per-column overrides remain intact.
        try:
            policy_map = dict(getattr(preprocess.missing_result, "policy", {}) or {})
            default_policy = str(
                getattr(preprocess.missing_result, "default_policy", "drop") or "drop"
            )
            overrides = {k: v for k, v in policy_map.items() if v != default_policy}
            policy_spec: dict[str, str] | str = (
                {"default": default_policy, **overrides} if overrides else default_policy
            )

            limit_map = dict(getattr(preprocess.missing_result, "limit", {}) or {})
            default_limit = getattr(preprocess.missing_result, "default_limit", None)
            limit_overrides = {
                k: v
                for k, v in limit_map.items()
                if v is not None and default_limit is not None and v != default_limit
            }
            if default_limit is not None or limit_overrides:
                limit_spec: dict[str, int | None] | int | None
                limit_spec = {"default": default_limit, **limit_overrides}
            else:
                limit_spec = None

            reindexed, _ = apply_missing_policy(
                reindexed,
                policy=policy_spec,
                limit=limit_spec,
                enforce_completeness=False,
            )
        except Exception:  # pragma: no cover - defensive
            return reindexed
        return reindexed

    def _is_month_label(label: str) -> bool:
        text = str(label).strip()
        return len(text) == 7 and text.count("-") == 1

    def _resolve_bound(label: str, *, bound: str) -> pd.Timestamp:
        text = str(label).strip()
        if not text:
            raise ValueError("Period label must be non-empty")
        try:
            if _is_month_label(text):
                period = pd.Period(text, freq="M")
                ts = period.start_time if bound == "start" else period.end_time
            else:
                ts = pd.to_datetime(text)
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to parse period label '{label}': {exc}"
            raise ValueError(msg) from exc
        return pd.Timestamp(ts).normalize()

    in_sdate = _resolve_bound(in_start, bound="start")
    in_edate = _resolve_bound(in_end, bound="end")
    out_sdate = _resolve_bound(out_start, bound="start")
    out_edate = _resolve_bound(out_end, bound="end")

    date_series = preprocess.df[preprocess.date_col]
    if pd.DatetimeTZDtype.is_dtype(date_series.dtype):
        tz = date_series.dt.tz
        in_sdate = in_sdate.tz_localize(tz)
        in_edate = in_edate.tz_localize(tz)
        out_sdate = out_sdate.tz_localize(tz)
        out_edate = out_edate.tz_localize(tz)

    in_mask = (preprocess.df[preprocess.date_col] >= in_sdate) & (
        preprocess.df[preprocess.date_col] <= in_edate
    )
    out_mask = (preprocess.df[preprocess.date_col] >= out_sdate) & (
        preprocess.df[preprocess.date_col] <= out_edate
    )
    in_df_raw = preprocess.df.loc[in_mask].set_index(preprocess.date_col)
    out_df_raw = preprocess.df.loc[out_mask].set_index(preprocess.date_col)

    # Ensure monthly continuity inside each window.
    # If one side of the split is empty but the other has data, allow reindexing
    # to preserve the expected calendar. If both sides are empty (or the in
    # window is empty), keep empties so callers receive SAMPLE_WINDOW_EMPTY.
    allow_in_empty = not out_df_raw.empty
    allow_out_empty = not in_df_raw.empty
    in_df = _fill_missing_months(
        in_df_raw,
        start=in_sdate,
        end=in_edate,
        allow_empty_reindex=allow_in_empty,
    )
    out_df = _fill_missing_months(
        out_df_raw,
        start=out_sdate,
        end=out_edate,
        allow_empty_reindex=allow_out_empty,
    )

    def _is_effectively_empty(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return True
        # A reindexed-but-empty slice can have rows but no data (all NaNs).
        # Treat that as an empty window for diagnostics.
        try:
            return bool(frame.dropna(how="all").empty)
        except Exception:  # pragma: no cover - defensive
            return bool(frame.empty)

    if _is_effectively_empty(in_df) or _is_effectively_empty(out_df):
        return pipeline_failure(
            PipelineReasonCode.SAMPLE_WINDOW_EMPTY,
            context={
                "in_rows": int(in_df.shape[0]),
                "out_rows": int(out_df.shape[0]),
            },
        )

    return _WindowStage(
        in_df=in_df,
        out_df=out_df,
        in_start=in_sdate,
        in_end=in_edate,
        out_start=out_sdate,
        out_end=out_edate,
        periods_per_year=preprocess.periods_per_year,
        date_col=preprocess.date_col,
    )


def _prepare_input_data(
    df: pd.DataFrame,
    *,
    date_col: str,
    missing_policy: str | Mapping[str, str] | None,
    missing_limit: int | Mapping[str, int | None] | None,
    enforce_completeness: bool = True,
) -> tuple[pd.DataFrame, FrequencySummary, MissingPolicyResult, bool]:
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{date_col}' column")

    work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work[date_col].dtype):
        work[date_col] = pd.to_datetime(work[date_col])
    work.sort_values(date_col, inplace=True)

    freq_summary = detect_frequency(work[date_col])

    value_cols = [c for c in work.columns if c != date_col]
    if value_cols:
        numeric = work[value_cols].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = work[value_cols]
    numeric.index = pd.DatetimeIndex(work[date_col])

    if freq_summary.resampled:
        resampled = (1 + numeric).resample(MONTHLY_DATE_FREQ).prod(min_count=1) - 1
        normalised = True
    else:
        resampled = numeric.resample(MONTHLY_DATE_FREQ).last()
        normalised = False

    policy_spec: str | Mapping[str, str] | None = missing_policy or "drop"
    preserve_empty_periods = True
    if isinstance(policy_spec, str):
        preserve_empty_periods = policy_spec.lower() != "drop"
    elif isinstance(policy_spec, Mapping):
        values = [str(v or "").lower() for v in policy_spec.values()]
        preserve_empty_periods = any(v and v != "drop" for v in values)

    if not preserve_empty_periods:
        resampled = resampled.dropna(how="all")
    resampled.index.name = date_col

    filled, missing_result = apply_missing_policy(
        resampled,
        policy=policy_spec,
        limit=missing_limit,
        enforce_completeness=enforce_completeness,
    )
    # Only drop rows that remain completely empty after the missing-data policy.
    filled = filled.dropna(how="all")

    if filled.empty:
        monthly_df = pd.DataFrame(columns=[date_col])
    else:
        monthly_df = filled.reset_index().rename(columns={"index": date_col})
        monthly_df[date_col] = pd.to_datetime(monthly_df[date_col])
        monthly_df.sort_values(date_col, inplace=True)

    return monthly_df, freq_summary, missing_result, normalised
