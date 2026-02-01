"""Configuration helpers for Monte Carlo risk-free handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from trend_analysis.config.models import ConfigType
from trend_analysis.risk import periods_per_year_from_code
from trend_analysis.stages.selection import _resolve_risk_free_column
from trend_analysis.util.risk_free import resolve_risk_free_settings

__all__ = ["RiskFreeResolution", "resolve_risk_free_source"]


@dataclass(frozen=True)
class RiskFreeResolution:
    """Resolved risk-free source for Monte Carlo runs."""

    source: str
    risk_free: pd.Series | float
    column: str | None = None


def resolve_risk_free_source(
    returns: pd.DataFrame,
    config: ConfigType | Mapping[str, Any],
    *,
    indices_list: Sequence[str] | None = None,
    allow_risk_free_fallback: bool | None = None,
) -> RiskFreeResolution:
    """Resolve the risk-free source for Monte Carlo returns.

    Supported modes
    ---------------
    - override: ``metrics.rf_override_enabled`` uses ``metrics.rf_rate_annual`` as a constant rate.
    - configured: ``data.risk_free_column`` explicitly selects a series.
    - fallback: ``data.allow_risk_free_fallback`` selects the lowest-volatility series.
    """

    data_settings = _section(config, "data")
    metrics_settings = _section(config, "metrics")
    benchmarks = _section(config, "benchmarks")

    rf_override_enabled = bool(metrics_settings.get("rf_override_enabled", False))
    rf_rate_annual = float(metrics_settings.get("rf_rate_annual", 0.0) or 0.0)

    if rf_override_enabled:
        frequency = str(data_settings.get("frequency") or "M")
        periods_per_year = float(periods_per_year_from_code(frequency))
        rf_rate_periodic = (1.0 + rf_rate_annual) ** (1.0 / periods_per_year) - 1.0
        return RiskFreeResolution(
            source="override",
            risk_free=float(rf_rate_periodic),
            column=None,
        )

    risk_free_column, fallback_allowed = resolve_risk_free_settings(data_settings)
    if allow_risk_free_fallback is not None:
        fallback_allowed = bool(allow_risk_free_fallback)
        if risk_free_column:
            fallback_allowed = False

    idx_list = [str(x) for x in indices_list] if indices_list else []
    if not idx_list and isinstance(benchmarks, Mapping):
        idx_list = [str(val) for val in benchmarks.values() if val is not None]

    date_col = str(data_settings.get("date_column") or "Date")
    rf_col, _fund_cols, source = _resolve_risk_free_column(
        returns,
        date_col=date_col,
        indices_list=idx_list,
        risk_free_column=risk_free_column,
        allow_risk_free_fallback=fallback_allowed,
    )
    return RiskFreeResolution(source=source, risk_free=returns[rf_col], column=rf_col)


def _section(config: ConfigType | Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        section = config.get(key, {})
    else:
        section = getattr(config, key, {})  # type: ignore[assignment]
    if not isinstance(section, Mapping):
        return {}
    return section
