"""Distribution aggregation helpers for Monte Carlo results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict, cast

import numpy as np
import pandas as pd

from .results import RESULT_BASE_COLUMNS

__all__ = [
    "AGGREGATION_PATH_COLUMNS",
    "AggregationFrameSchemas",
    "BREACH_COLUMNS",
    "EXPECTED_SHORTFALL_COLUMNS",
    "PATH_COLUMNS",
    "QUANTILE_COLUMNS",
    "aggregation_frame_schemas",
    "BreachAggregationRow",
    "MonteCarloAggregationResults",
    "ExpectedShortfallAggregationRow",
    "PathAggregationRow",
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

_DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
_Direction = Literal["lower", "upper"]
_Tail = Literal["lower", "upper"]
PathFrameSchema = tuple[str, ...]
QuantilesFrameSchema = tuple[str, ...]
BreachFrameSchema = tuple[str, ...]
ExpectedShortfallFrameSchema = tuple[str, ...]

PATH_COLUMNS = (
    "strategy",
    "path",
    "fold",
)
AGGREGATION_PATH_COLUMNS = PATH_COLUMNS


class QuantilesAggregationRow(TypedDict):
    """Schema for a single quantiles aggregation row."""

    strategy: Any
    fold: Any
    metric: str
    quantile: float
    value: float
    paths: int


class PathAggregationRow(TypedDict, total=False):
    """Schema for a single per-path aggregation row."""

    strategy: Any
    path: Any
    fold: Any


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


class AggregationFrameSchemas(TypedDict):
    """Schema definitions for all aggregation outputs."""

    path: PathFrameSchema
    quantiles: QuantilesFrameSchema
    breach: BreachFrameSchema
    expected_shortfall: ExpectedShortfallFrameSchema


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


def aggregation_frame_schemas(results_frame: pd.DataFrame) -> AggregationFrameSchemas:
    """Return column schemas for each aggregation output frame."""

    return {
        "path": path_frame_schema(results_frame),
        "quantiles": quantiles_frame_schema(),
        "breach": breach_frame_schema(),
        "expected_shortfall": expected_shortfall_frame_schema(),
    }


def build_path_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-path metrics with strategy/path/fold identifiers."""

    metric_cols = _metric_columns(results_frame)
    if results_frame.empty:
        return pd.DataFrame(columns=list(path_frame_schema(results_frame)))

    data: dict[str, Any] = {
        "strategy": _select_strategy_column(results_frame),
        "path": _coerce_column(results_frame, ("path", "path_id", "path_hash")),
        "fold": _coerce_column(results_frame, ("fold", "fold_id", "fold_label"), default=None),
    }
    frame = pd.DataFrame(data).reset_index(drop=True)
    if metric_cols:
        frame = pd.concat(
            [frame, results_frame[metric_cols].reset_index(drop=True)],
            axis=1,
        )
    schema = path_frame_schema(results_frame)
    if schema:
        frame = frame[list(schema)]
    return _sort_frame(frame, ("strategy", "fold", "path"))


def path_frame_schema(results_frame: pd.DataFrame) -> PathFrameSchema:
    """Return the schema (column order) for the per-path aggregation frame."""

    metric_cols = _metric_columns(results_frame)
    return tuple(AGGREGATION_PATH_COLUMNS) + tuple(metric_cols)


def quantiles_frame_schema() -> QuantilesFrameSchema:
    """Return the schema (column order) for the quantiles aggregation frame."""

    return tuple(QUANTILE_COLUMNS)


def breach_frame_schema() -> BreachFrameSchema:
    """Return the schema (column order) for the breach probability frame."""

    return tuple(BREACH_COLUMNS)


def expected_shortfall_frame_schema() -> ExpectedShortfallFrameSchema:
    """Return the schema (column order) for the expected shortfall frame."""

    return tuple(EXPECTED_SHORTFALL_COLUMNS)


def build_quantiles_frame(
    path_frame: pd.DataFrame,
    quantiles: Sequence[float] | None,
) -> pd.DataFrame:
    """Compute quantile summaries per strategy and fold."""

    path_frame = _ensure_path_columns(path_frame)
    quantile_list = _coerce_quantiles(quantiles)
    metric_cols = _path_metric_columns(path_frame)
    schema = quantiles_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    numeric = path_frame[metric_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    finite_frame = pd.DataFrame(finite_mask, columns=metric_cols, index=numeric.index)
    numeric = numeric.where(finite_frame)
    group_keys = [path_frame["strategy"], path_frame["fold"]]
    counts = finite_frame.groupby(group_keys, dropna=False).sum().astype(int)
    quantiles_frame = numeric.groupby(group_keys, dropna=False).quantile(quantile_list)
    quantiles_frame = quantiles_frame.reset_index()
    if "quantile" not in quantiles_frame.columns:
        candidate_cols = [
            col for col in quantiles_frame.columns if col not in {"strategy", "fold", *metric_cols}
        ]
        if len(candidate_cols) == 1:
            quantiles_frame = quantiles_frame.rename(columns={candidate_cols[0]: "quantile"})
    quantiles_long = quantiles_frame.melt(
        id_vars=["strategy", "fold", "quantile"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    counts_long = counts.reset_index().melt(
        id_vars=["strategy", "fold"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="paths",
    )
    frame = quantiles_long.merge(counts_long, on=["strategy", "fold", "metric"], how="left")
    frame = frame[list(schema)]
    return _sort_frame(frame, ("strategy", "fold", "metric", "quantile"))


def build_breach_frame(
    path_frame: pd.DataFrame,
    breach_spec: Mapping[str, Any] | Sequence[float] | None,
) -> pd.DataFrame:
    """Compute breach probabilities for configured thresholds."""

    path_frame = _ensure_path_columns(path_frame)
    metric_cols = _path_metric_columns(path_frame)
    schema = breach_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    specs = _coerce_breach_specs(breach_spec, metric_cols)
    if not specs:
        return pd.DataFrame(columns=list(schema))

    grouped = path_frame.groupby(["strategy", "fold"], dropna=False)
    rows: list[BreachAggregationRow] = []
    # Loop per metric/threshold spec to avoid large intermediate frames for mixed directions.
    for (strategy, fold), group in grouped:
        for metric, thresholds, direction in specs:
            if metric not in group.columns:
                continue
            values = _numeric_values(group[metric])
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
    frame = pd.DataFrame(rows, columns=list(schema))
    return _sort_frame(frame, ("strategy", "fold", "metric", "threshold", "direction"))


def build_expected_shortfall_frame(
    path_frame: pd.DataFrame,
    expected_shortfall_spec: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Compute expected shortfall (tail mean) for configured metrics."""

    path_frame = _ensure_path_columns(path_frame)
    metric_cols = _path_metric_columns(path_frame)
    schema = expected_shortfall_frame_schema()
    if path_frame.empty or not metric_cols:
        return pd.DataFrame(columns=list(schema))

    specs = _coerce_shortfall_specs(expected_shortfall_spec, metric_cols)
    if not specs:
        return pd.DataFrame(columns=list(schema))

    grouped = path_frame.groupby(["strategy", "fold"], dropna=False)
    rows: list[ExpectedShortfallAggregationRow] = []
    # Loop per metric/tail spec to keep per-tail thresholds explicit and readable.
    for (strategy, fold), group in grouped:
        for metric, alpha, tail in specs:
            if metric not in group.columns:
                continue
            values = _numeric_values(group[metric])
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
    frame = pd.DataFrame(rows, columns=list(schema))
    return _sort_frame(frame, ("strategy", "fold", "metric", "tail", "alpha"))


def _metric_columns(results_frame: pd.DataFrame) -> list[str]:
    excluded = set(RESULT_BASE_COLUMNS)
    excluded.update({"fold", "path", "strategy_name"})
    metric_cols: list[str] = []
    for col in results_frame.columns:
        name = str(col)
        if name in excluded:
            continue
        series = results_frame[col]
        if _is_numeric_like(series):
            metric_cols.append(name)
    return metric_cols


def _path_metric_columns(path_frame: pd.DataFrame) -> list[str]:
    excluded = set(RESULT_BASE_COLUMNS)
    excluded.update({"path", "fold", "strategy_name"})
    metric_cols: list[str] = []
    for col in path_frame.columns:
        name = str(col)
        if name in excluded:
            continue
        series = path_frame[col]
        if _is_numeric_like(series):
            metric_cols.append(name)
    return metric_cols


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


def _select_strategy_column(frame: pd.DataFrame) -> pd.Series:
    if "strategy" in frame.columns and "strategy_name" in frame.columns:
        strategy = frame["strategy"]
        strategy_name = frame["strategy_name"]
        if _is_numeric_like(strategy) and not _is_numeric_like(strategy_name):
            return strategy_name
        return strategy
    return _coerce_column(frame, ("strategy", "strategy_name"), default=None)


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        coerced = pd.to_numeric(series, errors="coerce")
        values = cast(np.ndarray, coerced.to_numpy(dtype=float))
        return bool(np.isfinite(values).any())
    return False


def _numeric_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return cast(np.ndarray, numeric.to_numpy(dtype=float))


def _ensure_path_columns(path_frame: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in AGGREGATION_PATH_COLUMNS if col not in path_frame.columns]
    if not missing_cols:
        return path_frame
    frame = path_frame.copy()
    for col in missing_cols:
        frame[col] = pd.NA
    return frame


def _sort_frame(frame: pd.DataFrame, sort_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    try:
        return frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    except TypeError:
        return frame.sort_values(
            list(sort_columns),
            kind="mergesort",
            key=lambda series: series.astype(str),
        ).reset_index(drop=True)


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
    deduped: list[float] = []
    seen: set[float] = set()
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_breach_specs(
    breach_spec: Mapping[str, Any] | Sequence[float] | None,
    metrics: Sequence[str],
) -> list[tuple[str, list[float], _Direction]]:
    def _dedupe(values: list[float]) -> list[float]:
        deduped: list[float] = []
        seen: set[float] = set()
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _coerce_threshold(value: Any) -> float | None:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(threshold):
            return None
        return threshold

    def _parse_breach_mapping(raw: Mapping[str, Any]) -> tuple[list[float], _Direction]:
        thresholds: list[float] = []
        direction: _Direction = "lower"
        raw_thresholds = raw.get("thresholds", raw.get("threshold"))
        if raw_thresholds is None:
            raw_thresholds = []
        if isinstance(raw_thresholds, (list, tuple)):
            thresholds = [
                threshold
                for value in raw_thresholds
                if (threshold := _coerce_threshold(value)) is not None
            ]
        else:
            threshold = _coerce_threshold(raw_thresholds)
            if threshold is not None:
                thresholds = [threshold]
        raw_direction = raw.get("direction", "lower")
        if raw_direction is None:
            raw_direction = "lower"
        direction_value = str(raw_direction).lower()
        if direction_value not in {"lower", "upper"}:
            raise ValueError(f"Unsupported breach direction '{direction_value}'")
        direction = cast(_Direction, direction_value)
        return _dedupe(thresholds), direction

    def _parse_breach_spec(raw: Any) -> tuple[list[float], _Direction]:
        if isinstance(raw, Mapping):
            return _parse_breach_mapping(raw)
        if isinstance(raw, (list, tuple)):
            thresholds = [
                threshold for value in raw if (threshold := _coerce_threshold(value)) is not None
            ]
            return _dedupe(thresholds), "lower"
        threshold = _coerce_threshold(raw)
        if threshold is not None:
            return [threshold], "lower"
        return [], "lower"

    if breach_spec is None:
        return []
    if isinstance(breach_spec, (list, tuple)):
        default_thresholds = [
            threshold
            for value in breach_spec
            if (threshold := _coerce_threshold(value)) is not None
        ]
        default_thresholds = _dedupe(default_thresholds)
        if not default_thresholds:
            return []
        return [(metric, default_thresholds, "lower") for metric in metrics]
    if not isinstance(breach_spec, Mapping):
        return []

    specs: list[tuple[str, list[float], _Direction]] = []
    default_raw: Any | None = None
    default_keys = {"thresholds", "threshold", "direction"}
    metrics_set = set(metrics)
    if "default" in breach_spec and "default" not in metrics_set:
        default_raw = breach_spec.get("default")
    elif default_keys.intersection(breach_spec.keys()) and not default_keys.intersection(
        metrics_set
    ):
        default_raw = {key: breach_spec[key] for key in default_keys if key in breach_spec}

    for metric, raw in breach_spec.items():
        if metric == "default" and default_raw is not None and "default" not in metrics_set:
            continue
        if metric in default_keys and default_raw is not None and metric not in metrics_set:
            continue
        metric_name = str(metric)
        thresholds, direction = _parse_breach_spec(raw)
        if thresholds:
            specs.append((metric_name, thresholds, direction))

    if default_raw is not None:
        default_thresholds, default_direction = _parse_breach_spec(default_raw)
        if default_thresholds:
            covered = {metric for metric, _, _ in specs}
            for metric in metrics:
                if metric in covered:
                    continue
                specs.append((metric, default_thresholds, default_direction))
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
        if raw is None:
            continue
        metric_name = str(metric)
        alpha = 0.05
        tail: _Tail = "lower"
        if isinstance(raw, Mapping):
            raw_alpha = raw.get("alpha")
            if raw_alpha is not None:
                alpha = float(raw_alpha)
                if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
                    raise ValueError("Expected shortfall alpha must be between 0 and 1")
            raw_tail = raw.get("tail", raw.get("direction", tail))
            if raw_tail is None:
                raw_tail = tail
            tail_value = str(raw_tail).lower()
            if tail_value not in {"lower", "upper"}:
                raise ValueError(f"Unsupported shortfall tail '{tail_value}'")
            tail = cast(_Tail, tail_value)
        else:
            alpha = float(raw)
        if not np.isfinite(alpha):
            raise ValueError("Expected shortfall alpha must be between 0 and 1")
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Expected shortfall alpha must be between 0 and 1")
        specs.append((metric_name, alpha, tail))

    if not specs and metrics:
        specs = [(metric, 0.05, "lower") for metric in metrics]
    return specs
