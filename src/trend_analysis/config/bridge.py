"""Streamlit / UI bridge helpers for the minimal startup configuration.

This module provides lightweight helpers that the Streamlit application can
import without pulling in heavier pipeline code.  It builds payloads matching
the minimal :class:`TrendConfig` schema and validates them via the existing
``validate_trend_config`` function, returning a serialisable dictionary on
success (so the UI can stash it in ``st.session_state``) or a readable error.

The functions are intentionally tiny and pure so they are easy to unit test
outside of a Streamlit runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .model import validate_trend_config

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
        },
        "vol_adjust": {"target_vol": target_vol},
    }
    return payload


def validate_payload(
    payload: Dict[str, Any], *, base_path: Path
) -> Tuple[Dict[str, Any] | None, str | None]:
    """Validate a raw payload returning (validated_dict, error_message).

    On success ``error_message`` is ``None``; on failure ``validated_dict`` is
    ``None``.  Only the first validation error is reported (consistent with
    ``validate_trend_config`` behaviour).
    """

    try:
        cfg = validate_trend_config(payload, base_path=base_path)
    except ValueError as exc:  # surface single readable message
        return None, str(exc)
    model = cfg.model_dump()
    # Normalise Path objects to strings for JSON serialisation in the UI.
    data = model.get("data", {})
    if isinstance(data.get("csv_path"), Path):
        data["csv_path"] = str(data["csv_path"])  # normalise for JSON
    if isinstance(data.get("universe_membership_path"), Path):
        data["universe_membership_path"] = str(data["universe_membership_path"])
    model["data"] = data
    return model, None
