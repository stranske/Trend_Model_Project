"""Compatibility helpers for legacy TrendSpec configuration tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import streamlit as st

from trend_analysis.signal_presets import (
    TrendSpecPreset,
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from trend_analysis.signals import TrendSpec


def _normalise_identifier(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


_ERROR_FIELD_ALIASES: Dict[str, str] = {
    "target_vol": "risk_target",
    "target_volatility": "risk_target",
    "risk_target": "risk_target",
    "csv_path": "column_mapping",
    "file_path": "column_mapping",
    "column_mapping": "column_mapping",
    "date_column": "date_column",
    "date": "date_column",
}

_ERROR_SECTION_DEFAULTS: Dict[str, str] = {
    "vol_adjust": "risk_target",
    "volatility": "risk_target",
    "vol": "risk_target",
    "data": "column_mapping",
}


def _resolve_error_target(path_tokens: Iterable[str]) -> str:
    tokens: List[str] = [_normalise_identifier(tok) for tok in path_tokens]
    for token in reversed(tokens):
        if token in _ERROR_FIELD_ALIASES:
            return _ERROR_FIELD_ALIASES[token]
    for token in tokens:
        if token in _ERROR_SECTION_DEFAULTS:
            return _ERROR_SECTION_DEFAULTS[token]
    return "general"


def _map_payload_errors(messages: Iterable[str]) -> Dict[str, List[str]]:
    """Map backend validation errors to Configure page input fields."""

    mapped: Dict[str, List[str]] = defaultdict(list)
    for raw in messages:
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        path_line = lines[0]
        if "->" in path_line:
            path_tokens = [seg.strip() for seg in path_line.split("->") if seg.strip()]
        else:
            path_tokens = [path_line]
        field = _resolve_error_target(path_tokens)
        message_text = text
        mapped[field].append(message_text)
    return dict(mapped)


def _trend_spec_defaults_from_spec(spec: Optional[TrendSpec] = None) -> Dict[str, Any]:
    spec = spec or TrendSpec()
    return {
        "window": int(spec.window),
        "min_periods": int(spec.min_periods) if spec.min_periods is not None else 0,
        "lag": int(spec.lag),
        "vol_adjust": bool(spec.vol_adjust),
        "vol_target": float(spec.vol_target) if spec.vol_target is not None else 0.10,
        "zscore": bool(spec.zscore),
    }


def _trend_spec_defaults_from_preset(preset_name: Optional[str]) -> Dict[str, Any]:
    if not preset_name:
        return _trend_spec_defaults_from_spec()
    try:
        preset: TrendSpecPreset = get_trend_spec_preset(preset_name)
    except KeyError:
        return _trend_spec_defaults_from_spec()
    return dict(preset.form_defaults())


def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(coerced, minimum)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _normalise_trend_spec_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    defaults = _trend_spec_defaults_from_spec()
    window = _coerce_positive_int(values.get("window"), defaults["window"], minimum=1)
    lag = _coerce_positive_int(values.get("lag"), defaults["lag"], minimum=1)

    min_periods_raw = values.get("min_periods")
    if min_periods_raw is None:
        min_periods = 0
    else:
        try:
            min_periods = int(min_periods_raw)
        except (TypeError, ValueError):
            min_periods = 0
    if min_periods < 0:
        min_periods = 0
    if min_periods > window:
        min_periods = window

    vol_adjust = _coerce_bool(values.get("vol_adjust", False))
    vol_target_raw = values.get("vol_target")
    if vol_target_raw is None:
        vol_candidate = 0.0
    else:
        try:
            vol_candidate = float(vol_target_raw)
        except (TypeError, ValueError):
            vol_candidate = 0.0
    if not vol_adjust or vol_candidate <= 0.0:
        vol_target = None
    else:
        vol_target = vol_candidate

    zscore = _coerce_bool(values.get("zscore", False))
    return {
        "window": window,
        "min_periods": min_periods if min_periods > 0 else None,
        "lag": lag,
        "vol_adjust": vol_adjust,
        "vol_target": vol_target,
        "zscore": zscore,
    }


def _trend_spec_values_to_config(values: Mapping[str, Any]) -> Dict[str, Any]:
    normalised = _normalise_trend_spec_values(values)
    config: Dict[str, Any] = {
        "kind": "tsmom",
        "window": normalised["window"],
        "lag": normalised["lag"],
        "vol_adjust": normalised["vol_adjust"],
        "zscore": normalised["zscore"],
    }
    if normalised["min_periods"] is not None:
        config["min_periods"] = normalised["min_periods"]
    if normalised["vol_target"] is not None:
        config["vol_target"] = normalised["vol_target"]
    return config


def _apply_trend_spec_preset_to_state(
    state: MutableMapping[str, Any], preset_name: Optional[str]
) -> Dict[str, Any]:
    state.setdefault("trend_spec_defaults", {})
    state.setdefault("trend_spec_values", {})
    state.setdefault("trend_spec_preset", None)
    state.setdefault("trend_spec_config", {})

    defaults = _trend_spec_defaults_from_preset(preset_name)
    state["trend_spec_defaults"] = dict(defaults)
    state["trend_spec_values"] = dict(defaults)
    state["trend_spec_preset"] = preset_name
    state["trend_spec_config"] = _trend_spec_values_to_config(defaults)
    return defaults


def render_trend_spec_settings(selected_preset_label: Optional[str]) -> None:
    config_state = st.session_state.get("config_state")
    if not isinstance(config_state, dict):
        config_state = {}
        st.session_state.config_state = config_state
    config_state.setdefault("trend_spec_values", {})
    config_state.setdefault("trend_spec_defaults", {})
    config_state.setdefault("trend_spec_preset", None)
    config_state.setdefault("trend_spec_config", {})

    available = list_trend_spec_presets()
    if selected_preset_label and selected_preset_label in available:
        _apply_trend_spec_preset_to_state(config_state, selected_preset_label)
    elif not config_state["trend_spec_values"]:
        _apply_trend_spec_preset_to_state(config_state, None)


__all__ = [
    "render_trend_spec_settings",
    "_apply_trend_spec_preset_to_state",
    "_trend_spec_values_to_config",
    "_map_payload_errors",
    "TrendSpec",
]
