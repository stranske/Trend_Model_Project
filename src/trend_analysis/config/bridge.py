"""Keep Streamlit payload validation in sync with CLI startup checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from trend.config_schema import CoreConfigError, validate_core_config
from trend_analysis.config.model import validate_trend_config
from trend_analysis.config.validation import format_validation_messages, validate_config

__all__ = ["build_config_payload", "validate_payload"]

_REQUIRED_SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "preprocessing": {},
    "sample_split": {},
    "metrics": {},
    "export": {},
    "run": {},
}


def build_config_payload(
    *,
    csv_path: str | None,
    universe_membership_path: str | None,
    managers_glob: str | None,
    date_column: str,
    frequency: str,
    rebalance_calendar: str,
    max_turnover: float,
    transaction_cost_bps: float,
    slippage_bps: float = 0.0,
    target_vol: float,
) -> Dict[str, Any]:
    """Build a raw configuration mapping for minimal validation.

    Parameters mirror the minimal startup contract.  No validation is
    performed here – callers should pass the result to ``validate_payload``.
    """

    data: Dict[str, Any] = {
        "date_column": date_column,
        "frequency": frequency,
    }
    if csv_path:
        data["csv_path"] = csv_path
    if universe_membership_path:
        data["universe_membership_path"] = universe_membership_path
    if managers_glob:
        data["managers_glob"] = managers_glob

    payload: Dict[str, Any] = {
        "version": "1",
        "data": data,
        "portfolio": {
            "rebalance_calendar": rebalance_calendar,
            "max_turnover": max_turnover,
            "transaction_cost_bps": transaction_cost_bps,
            "cost_model": {
                "bps_per_trade": transaction_cost_bps,
                "slippage_bps": slippage_bps,
                "per_trade_bps": transaction_cost_bps,
                "half_spread_bps": slippage_bps,
            },
        },
        "vol_adjust": {"target_vol": target_vol},
    }
    return payload


def validate_payload(
    payload: Dict[str, Any], *, base_path: Path
) -> Tuple[Dict[str, Any] | None, str | None]:
    """Validate a raw payload returning (validated_dict, error_message)."""

    try:
        core = validate_core_config(payload, base_path=base_path)
        trend_config = validate_trend_config(payload, base_path=base_path)
    except (CoreConfigError, ValueError) as exc:
        return None, str(exc)

    semantic_portfolio = {
        key: value
        for key, value in dict(payload.get("portfolio") or {}).items()
        if key != "cost_model"
    }
    semantic_portfolio["transaction_cost_bps"] = core.costs.transaction_cost_bps
    semantic_portfolio["max_turnover"] = trend_config.portfolio.max_turnover
    semantic_data = dict(payload.get("data") or {})
    semantic_data["csv_path"] = str(core.data.csv_path) if core.data.csv_path is not None else None
    semantic_data["universe_membership_path"] = (
        str(core.data.universe_membership_path)
        if core.data.universe_membership_path is not None
        else None
    )
    semantic_data["managers_glob"] = core.data.managers_glob
    semantic_data["date_column"] = core.data.date_column
    semantic_data["frequency"] = core.data.frequency
    semantic_payload: Dict[str, Any] = {
        **{section: dict(defaults) for section, defaults in _REQUIRED_SECTION_DEFAULTS.items()},
        **payload,
        "data": semantic_data,
        "portfolio": semantic_portfolio,
        "vol_adjust": {"target_vol": trend_config.vol_adjust.target_vol},
    }
    semantic_result = validate_config(
        semantic_payload,
        base_path=base_path,
        skip_required_fields=True,
    )
    if semantic_result.errors:
        return None, "; ".join(format_validation_messages(semantic_result, include_warnings=False))

    validated: Dict[str, Any] = dict(payload)
    data_section = dict(validated.get("data") or {})
    data_section["csv_path"] = str(core.data.csv_path) if core.data.csv_path is not None else None
    data_section["universe_membership_path"] = (
        str(core.data.universe_membership_path)
        if core.data.universe_membership_path is not None
        else None
    )
    data_section["managers_glob"] = core.data.managers_glob
    data_section["date_column"] = core.data.date_column
    data_section["frequency"] = core.data.frequency
    validated["data"] = data_section

    portfolio = dict(validated.get("portfolio") or {})
    portfolio["transaction_cost_bps"] = core.costs.transaction_cost_bps
    portfolio["max_turnover"] = trend_config.portfolio.max_turnover
    cost_model = dict(portfolio.get("cost_model") or {})
    cost_model["bps_per_trade"] = core.costs.bps_per_trade
    cost_model["slippage_bps"] = core.costs.slippage_bps
    cost_model["per_trade_bps"] = core.costs.per_trade_bps
    cost_model["half_spread_bps"] = core.costs.half_spread_bps
    portfolio["cost_model"] = cost_model
    validated["portfolio"] = portfolio
    vol_adjust = dict(validated.get("vol_adjust") or {})
    vol_adjust["target_vol"] = trend_config.vol_adjust.target_vol
    validated["vol_adjust"] = vol_adjust
    return validated, None
