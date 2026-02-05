"""Distribution aggregation helpers for Monte Carlo results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict, cast

import numpy as np
import pandas as pd

__all__ = [
    "AGGREGATION_PATH_COLUMNS",
    "BREACH_COLUMNS",
    "EXPECTED_SHORTFALL_COLUMNS",
    "QUANTILE_COLUMNS",
    "BreachAggregationRow",
    "MonteCarloAggregationResults",
    "ExpectedShortfallAggregationRow",
    "QuantilesAggregationRow",
    "aggregate_monte_carlo_results",
    "build_breach_frame",
    "build_expected_shortfall_frame",
    "build_path_frame",
    "build_quantiles_frame",
    "breach_frame_schema",
    "expected_shortfall_frame_schema",
    "path_frame_schema",
    "quantiles_frame_schema",
]

AGGREGATION_PATH_COLUMNS = (
    "strategy",
    "path",
    "fold",
)

QUANTILE_COLUMNS = (
    "strategy",
    "fold",
    "metric",
    "quantile",
    "value",
    "paths",
)

BREACH_COLUMNS = (
    "strategy",
    "fold",
    "metric",
    "threshold",
    "direction",
    "breach_probability",
    "paths",
)

EXPECTED_SHORTFALL_COLUMNS = (
    "strategy",
    "fold",
    "metric",
    "tail",
    "alpha",
    "threshold",
    "expected_shortfall",
    "paths",
)

_DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
_Direction = Literal["lower", "upper"]
_Tail = Literal["lower", "upper"]


class QuantilesAggregationRow(TypedDict):
    """Schema for a single quantiles aggregation row."""

    strategy: Any
    fold: Any
    metric: str
    quantile: float
    value: float
    paths: int


class BreachAggregationRow(TypedDict):
    """Schema for a single breach probability aggregation row."""

    strategy: Any
    fold: Any
    metric: str
    threshold: float
    direction: Literal["lower", "upper"]
    breach_probability: float
    paths: int


class ExpectedShortfallAggregationRow(TypedDict):
    """Schema for a single expected shortfall aggregation row."""

    strategy: Any
    fold: Any
    metric: str
    tail: Literal["lower", "upper"]
    alpha: float
    threshold: float
    expected_shortfall: float
    paths: int


@dataclass(frozen=True)
class MonteCarloAggregationResults:
    """Container for aggregated Monte Carlo distributions."""

    path_frame: pd.DataFrame
    quantiles_frame: pd.DataFrame
    breach_frame: pd.DataFrame
    expected_shortfall_frame: pd.DataFrame


def aggregate_monte_carlo_results(
    results_frame: pd.DataFrame,
    *,
    quantiles: Sequence[float] | None = None,
    breach_spec: Mapping[str, Any] | Sequence[float] | None = None,
    expected_shortfall_spec: Mapping[str, Any] | None = None,
) -> MonteCarloAggregationResults:
    """Compute distribution summaries for Monte Carlo results."""

    path_frame = build_path_frame(results_frame)
    quantiles_frame = build_quantiles_frame(path_frame, quantiles)
    breach_frame = build_breach_frame(path_frame, breach_spec)
    expected_shortfall_frame = build_expected_shortfall_frame(path_frame, expected_shortfall_spec)
    return MonteCarloAggregationResults(
        path_frame=path_frame,
        quantiles_frame=quantiles_frame,
        breach_frame=breach_frame,
        expected_shortfall_frame=expected_shortfall_frame,
    )


def build_path_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-path metrics with strategy/path/fold identifiers."""

    metric_cols = _metric_columns(results_frame)
    if results_frame.empty:
        return pd.DataFrame(columns=list(path_frame_schema(results_frame)))

    data: dict[str, Any] = {
        "strategy": _coerce_column(results_frame, ("strategy",), default=None),
        "path": _coerce_column(results_frame, ("path", "path_id")),
        "fold": _coerce_column(results_frame, ("fold", "fold_id"), default=None),
    }
    frame = pd.DataFrame(data).reset_index(drop=True)
    if metric_cols:
        frame = pd.concat(
            [frame, results_frame[metric_cols].reset_index(drop=True)],
            axis=1,
        )
    schema = path_frame_schema(results_frame)
    if schema:
        return frame[list(schema)]
    return frame


def path_frame_schema(results_frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the schema (column order) for the per-path aggregation frame."""

    metric_cols = _metric_columns(results_frame)
    return tuple(AGGREGATION_PATH_COLUMNS) + tuple(metric_cols)


def quantiles_frame_schema() -> tuple[str, ...]:
    """Return the schema (column order) for the quantiles aggregation frame."""

    return tuple(QUANTILE_COLUMNS)


def breach_frame_schema() -> tuple[str, ...]:
    """Return the schema (column order) for the breach probability frame."""

    return tuple(BREACH_COLUMNS)


def expected_shortfall_frame_schema() -> tuple[str, ...]:
    """Return the schema (column order) for the expected shortfall frame."""

    return tuple(EXPECTED_SHORTFALL_COLUMNS)


def build_quantiles_frame(
    path_frame: pd.DataFrame,
    quantiles: Sequence[float] | None,
) -> pd.DataFrame:
    """Compute quantile summaries per strategy and fold."""

    quantile_list = _coerce_quantiles(quantiles)
    metric_cols = _path_metric_columns(path_frame)
    schema = quantiles_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    grouped = path_frame.groupby(["strategy", "fold"], dropna=False)
    rows: list[QuantilesAggregationRow] = []
    for (strategy, fold), group in grouped:
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                for q in quantile_list:
                    rows.append(
                        {
                            "strategy": strategy,
                            "fold": fold,
                            "metric": metric,
                            "quantile": q,
                            "value": np.nan,
                            "paths": 0,
                        }
                    )
                continue
            for q in quantile_list:
                rows.append(
                    {
                        "strategy": strategy,
                        "fold": fold,
                        "metric": metric,
                        "quantile": q,
                        "value": float(np.nanquantile(values, q)),
                        "paths": int(values.size),
                    }
                )
    return pd.DataFrame(rows, columns=list(schema))


def build_breach_frame(
    path_frame: pd.DataFrame,
    breach_spec: Mapping[str, Any] | Sequence[float] | None,
) -> pd.DataFrame:
    """Compute breach probabilities for configured thresholds."""

    metric_cols = _path_metric_columns(path_frame)
    schema = breach_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    specs = _coerce_breach_specs(breach_spec, metric_cols)
    if not specs:
        return pd.DataFrame(columns=list(schema))

    grouped = path_frame.groupby(["strategy", "fold"], dropna=False)
    rows: list[BreachAggregationRow] = []
    for (strategy, fold), group in grouped:
        for metric, thresholds, direction in specs:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            total = int(values.size)
            for threshold in thresholds:
                if total == 0:
                    prob = np.nan
                elif direction == "upper":
                    prob = float(np.mean(values >= threshold))
                else:
                    prob = float(np.mean(values <= threshold))
                rows.append(
                    {
                        "strategy": strategy,
                        "fold": fold,
                        "metric": metric,
                        "threshold": float(threshold),
                        "direction": direction,
                        "breach_probability": prob,
                        "paths": total,
                    }
                )
    return pd.DataFrame(rows, columns=list(schema))


def build_expected_shortfall_frame(
    path_frame: pd.DataFrame,
    expected_shortfall_spec: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Compute expected shortfall (tail mean) for configured metrics."""

    metric_cols = _path_metric_columns(path_frame)
    schema = expected_shortfall_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    specs = _coerce_shortfall_specs(expected_shortfall_spec, metric_cols)
    if not specs:
        return pd.DataFrame(columns=list(schema))

    grouped = path_frame.groupby(["strategy", "fold"], dropna=False)
    rows: list[ExpectedShortfallAggregationRow] = []
    for (strategy, fold), group in grouped:
        for metric, alpha, tail in specs:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            total = int(values.size)
            if total == 0:
                rows.append(
                    {
                        "strategy": strategy,
                        "fold": fold,
                        "metric": metric,
                        "tail": tail,
                        "alpha": float(alpha),
                        "threshold": np.nan,
                        "expected_shortfall": np.nan,
                        "paths": 0,
                    }
                )
                continue
            if tail == "upper":
                threshold = float(np.nanquantile(values, 1.0 - alpha))
                tail_values = values[values >= threshold]
            else:
                threshold = float(np.nanquantile(values, alpha))
                tail_values = values[values <= threshold]
            expected_shortfall = float(np.mean(tail_values)) if tail_values.size else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "fold": fold,
                    "metric": metric,
                    "tail": tail,
                    "alpha": float(alpha),
                    "threshold": threshold,
                    "expected_shortfall": expected_shortfall,
                    "paths": total,
                }
            )
    return pd.DataFrame(rows, columns=list(schema))


def _metric_columns(results_frame: pd.DataFrame) -> list[str]:
    numeric_cols = [
        str(col) for col in results_frame.select_dtypes(include="number").columns.tolist()
    ]
    for col in ("fold_id", "path_id", "seed", "fold", "path", "strategy"):
        if col in numeric_cols:
            numeric_cols.remove(col)
    return numeric_cols


def _path_metric_columns(path_frame: pd.DataFrame) -> list[str]:
    numeric_cols = [str(col) for col in path_frame.select_dtypes(include="number").columns.tolist()]
    for col in ("path", "fold", "strategy"):
        if col in numeric_cols:
            numeric_cols.remove(col)
    return numeric_cols


def _coerce_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    default: Any | None = None,
) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series([default] * len(frame), index=frame.index)


def _coerce_quantiles(quantiles: Sequence[float] | None) -> list[float]:
    if quantiles is None:
        values = list(_DEFAULT_QUANTILES)
    else:
        values = list(quantiles)
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        q = float(value)
        if not np.isfinite(q):
            continue
        if q < 0.0 or q > 1.0:
            raise ValueError("Quantiles must be between 0 and 1")
        cleaned.append(q)
    if not cleaned:
        cleaned = list(_DEFAULT_QUANTILES)
    return cleaned


def _coerce_breach_specs(
    breach_spec: Mapping[str, Any] | Sequence[float] | None,
    metrics: Sequence[str],
) -> list[tuple[str, list[float], _Direction]]:
    if breach_spec is None:
        return []
    if isinstance(breach_spec, (list, tuple)):
        default_thresholds = [float(value) for value in breach_spec]
        default_thresholds = [value for value in default_thresholds if np.isfinite(value)]
        if not default_thresholds:
            return []
        return [(metric, default_thresholds, "lower") for metric in metrics]
    if not isinstance(breach_spec, Mapping):
        return []

    specs: list[tuple[str, list[float], _Direction]] = []
    for metric, raw in breach_spec.items():
        metric_name = str(metric)
        thresholds: list[float] = []
        direction: _Direction = "lower"
        if isinstance(raw, Mapping):
            raw_thresholds = raw.get("thresholds", raw.get("threshold"))
            if raw_thresholds is None:
                raw_thresholds = []
            if isinstance(raw_thresholds, (list, tuple)):
                thresholds = [float(value) for value in raw_thresholds]
            else:
                thresholds = [float(raw_thresholds)]
            direction_value = str(raw.get("direction", "lower")).lower()
            if direction_value not in {"lower", "upper"}:
                raise ValueError(f"Unsupported breach direction '{direction_value}'")
            direction = cast(_Direction, direction_value)
        elif isinstance(raw, (list, tuple)):
            thresholds = [float(value) for value in raw]
        else:
            thresholds = [float(raw)]
        thresholds = [value for value in thresholds if np.isfinite(value)]
        if not thresholds:
            continue
        specs.append((metric_name, thresholds, direction))
    return specs


def _coerce_shortfall_specs(
    shortfall_spec: Mapping[str, Any] | None,
    metrics: Sequence[str],
) -> list[tuple[str, float, _Tail]]:
    if shortfall_spec is None:
        return [(metric, 0.05, "lower") for metric in metrics]
    if not isinstance(shortfall_spec, Mapping):
        return []

    specs: list[tuple[str, float, _Tail]] = []
    for metric, raw in shortfall_spec.items():
        metric_name = str(metric)
        alpha = 0.05
        tail: _Tail = "lower"
        if isinstance(raw, Mapping):
            raw_alpha = raw.get("alpha")
            if raw_alpha is not None:
                alpha = float(raw_alpha)
            tail_value = str(raw.get("tail", raw.get("direction", tail))).lower()
            if tail_value not in {"lower", "upper"}:
                raise ValueError(f"Unsupported shortfall tail '{tail_value}'")
            tail = cast(_Tail, tail_value)
        else:
            alpha = float(raw)
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Expected shortfall alpha must be between 0 and 1")
        specs.append((metric_name, alpha, tail))

    if not specs and metrics:
        specs = [(metric, 0.05, "lower") for metric in metrics]
    return specs
