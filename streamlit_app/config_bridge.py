"""Bridge helpers to validate Streamlit form inputs with the minimal TrendConfig model.

This keeps the UI aligned with startup validation semantics (Issue #1436).
The functions here are pure so they can be unit tested without Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from trend_analysis.config import validate_trend_config


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
        },
        "vol_adjust": {"target_vol": target_vol},
    }
    return data


def validate_payload(
    payload: Dict[str, Any], *, base_path: Path
) -> Tuple[dict, None] | Tuple[None, str]:
    try:
        cfg = validate_trend_config(payload, base_path=base_path)
        return cfg.model_dump(), None
    except ValueError as exc:  # surface first-line message
        return None, str(exc)


__all__ = ["build_config_payload", "validate_payload"]
