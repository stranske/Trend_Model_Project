"""Fold generation helpers for Monte Carlo scenario vintages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    cast,
)

import pandas as pd

__all__ = ["Fold", "FoldGenerator"]


_MODE_ALIASES = {
    "explicit": "explicit",
    "explicit_dates": "explicit",
    "dates": "explicit",
    "date": "explicit",
    "rolling": "rolling",
    "count_spaced": "count_spaced",
    "spaced": "count_spaced",
}

# Calculating returns for bootstrap calibration requires at least two distinct
# price observations strictly before the forecast start date.
_MIN_CALIBRATION_OBSERVATIONS = 2


@dataclass(frozen=True)
class Fold:
    """Represents one fold (vintage) calibration/forecast window."""

    fold_id: int
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp | None = None
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable mapping of fold metadata."""

        return {
            "fold_id": int(self.fold_id),
            "calibration_start": self.calibration_start.isoformat(),
            "calibration_end": self.calibration_end.isoformat(),
            "forecast_start": self.forecast_start.isoformat(),
            "forecast_end": (self.forecast_end.isoformat() if self.forecast_end else None),
            "label": self.label,
        }


class FoldGenerator:
    """Generate folds from configuration and available history."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str | None = None,
        fold_starts: Sequence[object] | None = None,
        n_folds: int | None = None,
        calibration_lookback_years: float | None = None,
        start: object | None = None,
        end: object | None = None,
        step_years: float | None = None,
        step_months: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = _normalize_mode(mode)
        self.fold_starts = list(fold_starts) if fold_starts is not None else None
        self.n_folds = _coerce_optional_int(n_folds, "n_folds", minimum=1)
        self.calibration_lookback_years = _coerce_optional_float(
            calibration_lookback_years, "calibration_lookback_years", minimum=0.0
        )
        self.start = _coerce_optional_timestamp(start, "start")
        self.end = _coerce_optional_timestamp(end, "end")
        self.step_years = _coerce_optional_float(step_years, "step_years", minimum=0.0)
        self.step_months = _coerce_optional_int(step_months, "step_months", minimum=1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "FoldGenerator | None":
        """Build a fold generator from a raw config mapping."""

        if not config:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("folds config must be a mapping")

        enabled = bool(config.get("enabled", True))
        if not enabled:
            return None

        return cls(
            enabled=True,
            mode=_coerce_optional_str(config.get("mode")),
            fold_starts=_coerce_optional_sequence(config.get("fold_starts")),
            n_folds=config.get("n_folds"),
            calibration_lookback_years=config.get("calibration_lookback_years"),
            start=config.get("start"),
            end=config.get("end"),
            step_years=config.get("step_years"),
            step_months=config.get("step_months"),
        )

    def generate(self, history_index: Iterable[object]) -> list[Fold]:
        """Generate folds aligned to the available history index."""

        if not self.enabled:
            return []

        index = _coerce_index(history_index)
        if index.empty:
            raise ValueError("history_index must contain at least one date")

        mode = self.mode
        if mode == "explicit":
            starts = self._explicit_starts(index)
        elif mode == "rolling":
            starts = self._rolling_starts(index)
        elif mode == "count_spaced":
            starts = self._count_spaced_starts(index)
        else:
            raise ValueError(f"Unsupported fold mode '{mode}'")

        folds: list[Fold] = []
        for idx, start in enumerate(starts, start=1):
            fold = self._build_fold(idx, start, index)
            folds.append(fold)
        return folds

    def _explicit_starts(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        if not self.fold_starts:
            raise ValueError("fold_starts is required for explicit fold mode")
        starts: list[pd.Timestamp] = []
        for raw in self.fold_starts:
            start = _coerce_timestamp(raw, "fold_start")
            starts.append(start)
        return _dedupe_sorted(starts)

    def _default_start(self, index: pd.DatetimeIndex) -> pd.Timestamp:
        unique_dates = index.unique()
        required_dates = _MIN_CALIBRATION_OBSERVATIONS + 1
        if len(unique_dates) < required_dates:
            raise ValueError(
                "Insufficient calibration history to derive default fold start: "
                f"requires at least {required_dates} distinct history dates "
                f"({_MIN_CALIBRATION_OBSERVATIONS} calibration observations before "
                f"forecast start), got {len(unique_dates)} "
                f"({unique_dates[0].date()} to {unique_dates[-1].date()})"
            )

        earliest_feasible = pd.Timestamp(unique_dates[_MIN_CALIBRATION_OBSERVATIONS])
        if self.calibration_lookback_years is None:
            return earliest_feasible

        lookback_months = _years_to_months(self.calibration_lookback_years)
        lookback_start = pd.Timestamp(unique_dates[0]) + pd.DateOffset(months=lookback_months)
        if lookback_start > unique_dates[-1]:
            raise ValueError(
                "Insufficient calibration history for "
                f"calibration_lookback_years={self.calibration_lookback_years}: "
                f"earliest feasible fold start is {lookback_start.date()}, "
                f"after available history ({unique_dates[0].date()} to {unique_dates[-1].date()})"
            )
        candidate = max(earliest_feasible, _align_to_index(lookback_start, unique_dates))
        start_pos = int(unique_dates.searchsorted(candidate, side="left"))
        for pos in range(start_pos, len(unique_dates)):
            forecast_start = pd.Timestamp(unique_dates[pos])
            calibration_end = pd.Timestamp(unique_dates[pos - 1])
            calibration_start = _resolve_calibration_start(
                calibration_end, unique_dates, self.calibration_lookback_years
            )
            window_start_pos = int(unique_dates.searchsorted(calibration_start, side="left"))
            if pos - window_start_pos >= _MIN_CALIBRATION_OBSERVATIONS:
                return forecast_start

        raise ValueError(
            "Insufficient calibration history for "
            f"calibration_lookback_years={self.calibration_lookback_years}: "
            f"no fold start has at least {_MIN_CALIBRATION_OBSERVATIONS} distinct "
            f"calibration observations within the {lookback_months}-month lookback window "
            f"(available history: {unique_dates[0].date()} to {unique_dates[-1].date()})"
        )

    def _resolve_implicit_bounds(
        self,
        index: pd.DatetimeIndex,
        *,
        apply_lookback_to_explicit_start: bool,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("fold start must be before fold end")

        if self.end is not None and self.end <= index[0]:
            raise ValueError(
                f"Insufficient calibration history before fold end {self.end.date()}: "
                f"available history starts at {index[0].date()} "
                f"({index[0].date()} to {index[-1].date()})"
            )

        end = self.end if self.end is not None else index[-1]

        if self.start is None:
            start = self._default_start(index)
            if start > end:
                raise ValueError(
                    f"Insufficient calibration history before fold end {end.date()}: "
                    f"earliest feasible fold start is {start.date()} "
                    f"(available history: {index[0].date()} to {index[-1].date()})"
                )
            return start, end

        start = self.start
        if start > index[-1]:
            raise ValueError(
                f"Insufficient calibration history for fold start {start.date()}: "
                f"available history ends at {index[-1].date()} "
                f"({index[0].date()} to {index[-1].date()})"
            )

        if apply_lookback_to_explicit_start and self.calibration_lookback_years is not None:
            min_allowed = index[0] + pd.DateOffset(
                months=_years_to_months(self.calibration_lookback_years)
            )
            if min_allowed > start:
                start = min_allowed
            if start > end:
                raise ValueError(
                    "Insufficient calibration history for "
                    f"calibration_lookback_years={self.calibration_lookback_years}: "
                    f"earliest feasible fold start is {start.date()}, "
                    f"after fold end {end.date()} "
                    f"(available history: {index[0].date()} to {index[-1].date()})"
                )

        return start, end

    def _rolling_starts(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        start, end = self._resolve_implicit_bounds(index, apply_lookback_to_explicit_start=False)
        step_months = self._rolling_step_months(index)
        starts: list[pd.Timestamp] = []
        current = start
        offset = pd.DateOffset(months=step_months)

        while current <= end:
            starts.append(current)
            if self.n_folds is not None and len(starts) >= self.n_folds:
                break
            current = current + offset
        return _dedupe_sorted(starts)

    def _rolling_step_months(self, index: pd.DatetimeIndex) -> int:
        if self.step_months is not None:
            return self.step_months
        if self.step_years is not None:
            return _years_to_months(self.step_years)
        # Default to one year in months.
        return 12

    def _count_spaced_starts(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        if self.n_folds is None:
            raise ValueError("n_folds is required for count_spaced mode")

        start, end = self._resolve_implicit_bounds(index, apply_lookback_to_explicit_start=True)

        if self.n_folds == 1:
            return [start]

        spaced = pd.date_range(start=start, end=end, periods=self.n_folds)
        return _dedupe_sorted(list(spaced))

    def _build_fold(self, fold_id: int, raw_start: pd.Timestamp, index: pd.DatetimeIndex) -> Fold:
        if raw_start <= index[0]:
            raise ValueError(
                f"Insufficient calibration history for fold start {raw_start.date()}: "
                f"fold_start must be after the earliest history date ({index[0].date()})"
            )
        if raw_start > index[-1]:
            raise ValueError(
                f"Insufficient calibration history for fold start {raw_start.date()}: "
                f"available history ends at {index[-1].date()} "
                f"({index[0].date()} to {index[-1].date()})"
            )
        forecast_start = _align_to_index(raw_start, index)
        calibration_end = _previous_in_index(forecast_start, index)
        calibration_start = _resolve_calibration_start(
            calibration_end, index, self.calibration_lookback_years
        )

        label = forecast_start.strftime("%Y-%m")
        return Fold(
            fold_id=fold_id,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            forecast_start=forecast_start,
            label=label,
        )


def _normalize_mode(mode: str | None) -> str:
    if not mode:
        return "count_spaced"
    key = str(mode).strip().lower()
    return _MODE_ALIASES.get(key, key)


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_sequence(value: object) -> Sequence[object] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _coerce_optional_int(value: object, field: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(cast(SupportsInt | SupportsIndex | str | bytes | bytearray, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _coerce_optional_float(
    value: object, field: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(cast(SupportsFloat | SupportsIndex | str | bytes | bytearray, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _coerce_optional_timestamp(value: object, field: str) -> pd.Timestamp | None:
    if value is None:
        return None
    return _coerce_timestamp(value, field)


def _coerce_timestamp(value: object, field: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{field} must be a valid date")
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid date") from exc
    if pd.isna(ts):
        raise ValueError(f"{field} must be a valid date")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _coerce_index(history_index: Iterable[object]) -> pd.DatetimeIndex:
    if isinstance(history_index, pd.DatetimeIndex):
        index = history_index
    else:
        index = pd.DatetimeIndex(history_index)
    if index.tz is not None:
        index = index.tz_convert(None)
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    return index


def _align_to_index(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    if index.empty:
        raise ValueError("history_index must contain at least one date")
    if target in index:
        return pd.Timestamp(target)
    pos = index.searchsorted(target, side="left")
    if pos >= len(index):
        return index[-1]
    return index[pos]


def _previous_in_index(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    if index.empty:
        raise ValueError("history_index must contain at least one date")
    pos = index.searchsorted(target, side="left")
    if pos == 0:
        raise ValueError(
            f"Insufficient calibration history before {target.date()}: "
            f"fold_start must be after the earliest history date ({index[0].date()})"
        )
    return index[pos - 1]


def _resolve_calibration_start(
    calibration_end: pd.Timestamp,
    index: pd.DatetimeIndex,
    lookback_years: float | None,
) -> pd.Timestamp:
    if lookback_years is None:
        return index.min()
    months = _years_to_months(lookback_years)
    target = calibration_end - pd.DateOffset(months=months)
    pos = index.searchsorted(target, side="left")
    if pos >= len(index):
        return calibration_end
    return index[pos]


def _years_to_months(years: float) -> int:
    return max(1, int(round(float(years) * 12.0)))


def _dedupe_sorted(values: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    seen: set[pd.Timestamp] = set()
    cleaned: list[pd.Timestamp] = []
    for value in sorted(values):
        if value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned
