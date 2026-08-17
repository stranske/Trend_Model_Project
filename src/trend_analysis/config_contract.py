"""Shared configuration-resolution contract for analysis entry points."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from trend.config_schema import CoreConfigError

SectionGet = Callable[[Any, str, Any], Any]

PORTFOLIO_WEIGHTING_NAME_ALIASES = {"ew": "equal", "robust": "robust_mv"}
SCORE_BASED_PORTFOLIO_WEIGHTING_NAMES = frozenset(
    {
        "adaptive",
        "adaptive_bayes",
        "bayes",
        "score",
        "score_bayes",
        "score_prop",
        "score_prop_bayes",
        "score_prop_simple",
    }
)
SUPPORTED_PORTFOLIO_WEIGHTING_NAMES = frozenset(
    {
        "adaptive",
        "adaptive_bayes",
        "bayes",
        "convex_constrained",
        "custom",
        "equal",
        "erc",
        "ew",
        "hrp",
        "risk_parity",
        "robust",
        "robust_mean_variance",
        "robust_mv",
        "robust_risk_parity",
        "score",
        "score_bayes",
        "score_prop",
        "score_prop_bayes",
        "score_prop_simple",
        "vol_inverse",
    }
)


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
    return PORTFOLIO_WEIGHTING_NAME_ALIASES.get(name, name)


def resolve_portfolio_weighting_name(
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> str:
    """Resolve the canonical nested portfolio weighting name."""

    if section_get(portfolio_cfg, "weighting_scheme", None) is not None:
        raise CoreConfigError(
            "portfolio.weighting_scheme was removed; use portfolio.weighting.name"
        )
    weighting_cfg = section_get(portfolio_cfg, "weighting", {})
    if not isinstance(weighting_cfg, Mapping):
        raise CoreConfigError("portfolio.weighting must be a mapping")
    nested_name = section_get(weighting_cfg, "name", None)
    return normalise_weighting_name(nested_name)


def resolve_portfolio_weighting_params(
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> dict[str, Any]:
    """Return the constructor parameters from canonical nested weighting config."""

    weighting_cfg = section_get(portfolio_cfg, "weighting", {})
    if not isinstance(weighting_cfg, Mapping):
        raise CoreConfigError("portfolio.weighting must be a mapping")
    params = section_get(weighting_cfg, "params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise CoreConfigError("portfolio.weighting.params must be a mapping")
    return dict(params)


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


def resolve_portfolio_cost_bps(
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> tuple[float, float]:
    """Return canonical transaction-cost and slippage inputs in basis points."""

    for removed in ("transaction_cost_bps", "slippage_bps"):
        if section_get(portfolio_cfg, removed, None) is not None:
            raise CoreConfigError(f"portfolio.{removed} was removed; use portfolio.cost_model")
    cost_model = section_get(portfolio_cfg, "cost_model", {})
    if not isinstance(cost_model, Mapping):
        raise CoreConfigError("portfolio.cost_model must be a mapping")
    for removed in ("bps_per_trade", "slippage_bps"):
        if section_get(cost_model, removed, None) is not None:
            raise CoreConfigError(
                f"portfolio.cost_model.{removed} was removed; use per_trade_bps and half_spread_bps"
            )
    per_trade_bps = optional_cost_bps(
        section_get(cost_model, "per_trade_bps", 0.0),
        field="cost_model.per_trade_bps",
    )
    half_spread_bps = optional_cost_bps(
        section_get(cost_model, "half_spread_bps", 0.0),
        field="cost_model.half_spread_bps",
    )
    return float(per_trade_bps or 0.0), float(half_spread_bps or 0.0)


def resolve_pipeline_monthly_cost(
    run_cfg: Any,
    portfolio_cfg: Any,
    *,
    section_get: SectionGet = mapping_get,
) -> float:
    """Resolve the decimal per-period cost sent to the shared analysis pipeline."""

    tc_bps, slippage_bps = resolve_portfolio_cost_bps(portfolio_cfg, section_get=section_get)
    cost_model = section_get(portfolio_cfg, "cost_model", None)
    cost_values = (
        section_get(cost_model, "per_trade_bps", None),
        section_get(cost_model, "half_spread_bps", None),
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
