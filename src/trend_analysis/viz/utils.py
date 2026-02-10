"""Shared visualization utilities for Plotly figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

DEFAULT_COLORS: tuple[str, ...] = (
    "#2E86AB",
    "#F6C85F",
    "#6F4E7C",
    "#9FD356",
    "#CA6B4B",
    "#5E5E5E",
)


@dataclass(frozen=True)
class QuantileBand:
    """Represents a quantile band for fan charts."""

    lower: float
    upper: float

    def label(self) -> str:
        pct_low = int(round(self.lower * 100))
        pct_high = int(round(self.upper * 100))
        return f"{pct_low}-{pct_high}%"


def ensure_non_empty(name: str, data: object) -> None:
    """Raise when a required object is empty."""

    if data is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(data, (pd.Series, pd.DataFrame)) and data.empty:
        raise ValueError(f"{name} cannot be empty")
    if isinstance(data, (Sequence, Mapping)) and not data:
        raise ValueError(f"{name} cannot be empty")


def coerce_series(data: Sequence[float] | pd.Series, name: str) -> pd.Series:
    """Return a non-empty Series with a stable name."""

    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        series = pd.Series(list(data), name=name)

    ensure_non_empty(name, series)
    if series.name is None:
        series.name = name
    return series


def coerce_frame(
    data: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    index: Sequence[object] | None = None,
    name: str = "data",
) -> pd.DataFrame:
    """Return a non-empty DataFrame from a mapping or DataFrame."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data, index=index)

    ensure_non_empty(name, frame)
    return frame


def validate_quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    """Normalize quantiles and ensure they are within ``[0, 1]``."""

    q_values = tuple(float(q) for q in quantiles)
    if not q_values:
        raise ValueError("quantiles cannot be empty")
    if any(q < 0.0 or q > 1.0 for q in q_values):
        raise ValueError("quantiles must be between 0 and 1")
    return q_values


def quantiles_over_columns(
    data: pd.DataFrame,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    """Compute quantiles for each row across columns."""

    ensure_non_empty("data", data)
    q_values = validate_quantiles(quantiles)
    quantile_series = {q: data.quantile(q, axis=1) for q in q_values}
    frame = pd.DataFrame(quantile_series)
    frame.columns = [float(q) for q in frame.columns]
    return frame


def quantile_bands(
    quantiles: Iterable[float],
) -> tuple[QuantileBand, ...]:
    """Return symmetric quantile bands sorted from widest to narrowest."""

    q_values = sorted(validate_quantiles(quantiles))
    bands: list[QuantileBand] = []
    for idx, lower in enumerate(q_values):
        upper = q_values[-(idx + 1)]
        if lower >= upper:
            break
        bands.append(QuantileBand(lower=lower, upper=upper))
    return tuple(bands)


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color to an rgba color string."""

    if not color.startswith("#") or len(color) not in {4, 7}:
        raise ValueError("color must be a hex string like #abc or #aabbcc")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    hex_value = color.lstrip("#")
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    r, g, b = (int(hex_value[i : i + 2], 16) for i in range(0, 6, 2))
    return f"rgba({r}, {g}, {b}, {alpha})"
