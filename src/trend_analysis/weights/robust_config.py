"""Helpers for mapping robustness configuration into weight engine params."""

from __future__ import annotations

from typing import Any, Mapping


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def weight_engine_params_from_robustness(
    weighting_scheme: str | None,
    robustness_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Translate robustness config into weight engine constructor parameters."""
    if not weighting_scheme:
        return {}

    scheme = str(weighting_scheme).lower()
    if scheme not in {"robust_mv", "robust_mean_variance", "robust_risk_parity"}:
        return {}

    robustness_cfg = _mapping(robustness_cfg)
    shrinkage_cfg = _mapping(robustness_cfg.get("shrinkage"))
    condition_cfg = _mapping(robustness_cfg.get("condition_check"))
    logging_cfg = _mapping(robustness_cfg.get("logging"))

    shrinkage_enabled = bool(shrinkage_cfg.get("enabled", True))
    shrinkage_method = str(shrinkage_cfg.get("method", "ledoit_wolf") or "ledoit_wolf")
    if not shrinkage_enabled:
        shrinkage_method = "none"

    condition_enabled = bool(condition_cfg.get("enabled", True))
    condition_threshold = _coerce_float(condition_cfg.get("threshold"), 1.0e12)
    if not condition_enabled:
        condition_threshold = float("inf")
    safe_mode = str(condition_cfg.get("safe_mode", "hrp") or "hrp")
    diagonal_loading_factor = _coerce_float(
        condition_cfg.get("diagonal_loading_factor"), 1.0e-6
    )

    params: dict[str, Any] = {
        "condition_threshold": condition_threshold,
        "diagonal_loading_factor": diagonal_loading_factor,
    }

    if scheme in {"robust_mv", "robust_mean_variance"}:
        params.update(
            {
                "shrinkage_method": shrinkage_method,
                "safe_mode": safe_mode,
                "log_condition_numbers": bool(
                    logging_cfg.get("log_condition_numbers", True)
                ),
                "log_method_switches": bool(
                    logging_cfg.get("log_method_switches", True)
                ),
                "log_shrinkage_intensity": bool(
                    logging_cfg.get("log_shrinkage_intensity", True)
                ),
            }
        )

    return params
