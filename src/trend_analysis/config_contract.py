"""Shared configuration-resolution contract for analysis entry points."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from trend.config_schema import CoreConfigError

SectionGet = Callable[[Any, str, Any], Any]

_WEIGHTING_NAME_ALIASES = {"ew": "equal", "robust": "robust_mv"}
_WEIGHTING_SCHEME_PLACEHOLDERS = {"equal", "custom"}


def mapping_get(section: Any, key: str, default: Any = None) -> Any:
    """Read either a mapping or a config-model section."""

    if isinstance(section, Mapping):
        return section.get(key, default)
    getter = getattr(section, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except KeyError:
                return default
        except KeyError:
            return default
    return getattr(section, key, default)


def normalise_weighting_name(value: Any) -> str:
    name = str(value or "equal").strip().lower()
    return _WEIGHTING_NAME_ALIASES.get(name, name)


def resolve_portfolio_weighting_name(
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> str:
    """Resolve legacy and nested weighting keys with one documented precedence."""

    weighting_cfg = section_get(portfolio_cfg, "weighting", {})
    nested_name = section_get(weighting_cfg, "name", None)
    legacy_name = section_get(portfolio_cfg, "weighting_scheme", None)
    if legacy_name not in (None, ""):
        resolved_legacy = normalise_weighting_name(legacy_name)
        if resolved_legacy not in _WEIGHTING_SCHEME_PLACEHOLDERS or not nested_name:
            return resolved_legacy
    return normalise_weighting_name(nested_name)


def optional_cost_bps(value: Any, *, field: str) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CoreConfigError(f"portfolio.{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CoreConfigError(f"portfolio.{field} must be finite")
    if parsed < 0:
        raise CoreConfigError(f"portfolio.{field} cannot be negative")
    return parsed


def first_configured_cost(
    section: Any,
    primary_key: str,
    legacy_key: str,
    *,
    section_get: SectionGet,
) -> tuple[Any, str]:
    """Return the preferred configured cost alias and its diagnostic field name."""

    primary_value = section_get(section, primary_key, None)
    if primary_value is not None:
        return primary_value, primary_key
    return section_get(section, legacy_key, None), legacy_key


def resolve_portfolio_cost_bps(
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> tuple[float, float]:
    """Return canonical transaction-cost and slippage inputs in basis points."""

    cost_model = section_get(portfolio_cfg, "cost_model", None)
    bps_per_trade_value, bps_per_trade_key = first_configured_cost(
        cost_model,
        "per_trade_bps",
        "bps_per_trade",
        section_get=section_get,
    )
    bps_per_trade = optional_cost_bps(
        bps_per_trade_value,
        field=f"cost_model.{bps_per_trade_key}",
    )
    slippage_bps_value, slippage_bps_key = first_configured_cost(
        cost_model,
        "half_spread_bps",
        "slippage_bps",
        section_get=section_get,
    )
    slippage_bps = optional_cost_bps(
        slippage_bps_value,
        field=f"cost_model.{slippage_bps_key}",
    )
    if bps_per_trade is None:
        bps_per_trade = optional_cost_bps(
            section_get(portfolio_cfg, "transaction_cost_bps", 0.0),
            field="transaction_cost_bps",
        )
    if slippage_bps is None:
        slippage_bps = optional_cost_bps(
            section_get(portfolio_cfg, "slippage_bps", 0.0), field="slippage_bps"
        )
    return float(bps_per_trade or 0.0), float(slippage_bps or 0.0)


def resolve_pipeline_monthly_cost(
    run_cfg: Any,
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> float:
    """Resolve the decimal per-period cost sent to the shared analysis pipeline."""

    tc_bps, slippage_bps = resolve_portfolio_cost_bps(
        portfolio_cfg, section_get=section_get
    )
    cost_model = section_get(portfolio_cfg, "cost_model", None)
    cost_values = (
        *(
            section_get(cost_model, key, None)
            for key in (
                "per_trade_bps",
                "bps_per_trade",
                "half_spread_bps",
                "slippage_bps",
            )
        ),
        section_get(portfolio_cfg, "transaction_cost_bps", None),
        section_get(portfolio_cfg, "slippage_bps", None),
    )
    if any(value is not None for value in cost_values):
        return (tc_bps + slippage_bps) / 10000.0
    try:
        monthly_cost = float(section_get(run_cfg, "monthly_cost", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise CoreConfigError("run.monthly_cost must be numeric") from exc
    if not math.isfinite(monthly_cost) or monthly_cost < 0:
        raise CoreConfigError("run.monthly_cost must be finite and non-negative")
    return monthly_cost
