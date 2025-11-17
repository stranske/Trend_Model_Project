"""Bridge helpers aligning the Streamlit app with CLI configuration checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from trend.config_schema import CoreConfigError, validate_core_config


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
    data: Dict[str, Any] = {
        "version": "1",
        "data": {
            "csv_path": csv_path,
            "universe_membership_path": universe_membership_path,
            "managers_glob": managers_glob,
            "date_column": date_column,
            "frequency": frequency,
        },
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
    return data


def validate_payload(
    payload: Dict[str, Any], *, base_path: Path
) -> Tuple[dict, None] | Tuple[None, str]:
    try:
        core = validate_core_config(payload, base_path=base_path)
    except CoreConfigError as exc:
        return None, str(exc)

    validated: Dict[str, Any] = dict(payload)
    data_section = dict(validated.get("data") or {})
    data_section["csv_path"] = str(core.data.csv_path)
    data_section["universe_membership_path"] = (
        str(core.data.universe_membership_path)
        if core.data.universe_membership_path is not None
        else None
    )
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


__all__ = ["build_config_payload", "validate_payload"]
