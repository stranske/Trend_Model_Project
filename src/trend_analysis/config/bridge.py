"""Keep Streamlit payload validation in sync with CLI startup checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from trend.config_schema import CoreConfigError, validate_core_config

__all__ = ["build_config_payload", "validate_payload"]


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
        "data": data,
        "portfolio": {
            "rebalance_calendar": rebalance_calendar,
            "max_turnover": max_turnover,
            "transaction_cost_bps": transaction_cost_bps,
            "cost_model": {
                "bps_per_trade": transaction_cost_bps,
                "slippage_bps": slippage_bps,
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
    except CoreConfigError as exc:
        return None, str(exc)

    validated: Dict[str, Any] = dict(payload)
    data_section = dict(validated.get("data") or {})
    data_section["csv_path"] = (
        str(core.data.csv_path) if core.data.csv_path is not None else None
    )
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
    cost_model = dict(portfolio.get("cost_model") or {})
    cost_model["bps_per_trade"] = core.costs.bps_per_trade
    cost_model["slippage_bps"] = core.costs.slippage_bps
    portfolio["cost_model"] = cost_model
    validated["portfolio"] = portfolio
    return validated, None
