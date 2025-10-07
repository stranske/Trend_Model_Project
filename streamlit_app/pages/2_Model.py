"""Model configuration page for the Streamlit application."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from app.streamlit import state as app_state
from streamlit_app.components import analysis_runner
from trend_analysis.signal_presets import (
    TrendSpecPreset,
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from trend_analysis.signals import TrendSpec

METRIC_FIELDS = [
    ("Sharpe", "sharpe"),
    ("Annual return", "return_ann"),
    ("Max drawdown", "drawdown"),
]


def _baseline_trend_spec() -> TrendSpec:
    return TrendSpec()


def _preset_defaults(name: str | None) -> dict[str, Any]:
    if not name or name.lower() in {"baseline", "custom"}:
        spec = _baseline_trend_spec()
        return {
            "window": spec.window,
            "lag": spec.lag,
            "min_periods": spec.min_periods or spec.window,
            "vol_adjust": spec.vol_adjust,
            "vol_target": spec.vol_target or 0.0,
            "zscore": spec.zscore,
        }
    try:
        preset = get_trend_spec_preset(name)
    except KeyError:
        return _preset_defaults(None)
    spec = preset.spec if isinstance(preset, TrendSpecPreset) else TrendSpec()
    defaults = {
        "window": spec.window,
        "lag": spec.lag,
        "min_periods": spec.min_periods or spec.window,
        "vol_adjust": spec.vol_adjust,
        "vol_target": spec.vol_target or 0.0,
        "zscore": spec.zscore,
    }
    return defaults


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(coerced, minimum)


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return max(coerced, 0.0)


def _trend_spec_from_form(values: Mapping[str, Any]) -> dict[str, Any]:
    window = _coerce_positive_int(values.get("window"), default=63, minimum=1)
    lag = _coerce_positive_int(values.get("lag"), default=1, minimum=1)
    if lag > window:
        lag = window
    min_periods = _coerce_positive_int(
        values.get("min_periods"), default=window, minimum=1
    )
    if min_periods > window:
        min_periods = window
    vol_adjust = bool(values.get("vol_adjust", False))
    vol_target = _coerce_non_negative_float(values.get("vol_target"), default=0.0)
    if not vol_adjust or vol_target <= 0:
        vol_target = 0.0
    zscore = bool(values.get("zscore", False))
    return {
        "window": window,
        "lag": lag,
        "min_periods": min_periods,
        "vol_adjust": vol_adjust,
        "vol_target": vol_target,
        "zscore": zscore,
    }


def _validate_model(values: Mapping[str, Any], column_count: int) -> list[str]:
    errors: list[str] = []
    trend_spec = values.get("trend_spec", {})
    window = trend_spec.get("window", 63)
    lag = trend_spec.get("lag", 1)
    if lag > window:
        errors.append("Lag must be less than or equal to the lookback window.")
    min_periods = trend_spec.get("min_periods", window)
    if min_periods > window:
        errors.append("Minimum periods cannot exceed the window length.")
    selection = values.get("selection_count", 10)
    if column_count and selection > column_count:
        errors.append(
            f"Selection count ({selection}) cannot exceed available assets ({column_count})."
        )
    weights = values.get("metric_weights", {})
    if not any(float(w or 0) > 0 for w in weights.values()):
        errors.append("Provide at least one positive metric weight.")
    if trend_spec.get("vol_adjust") and trend_spec.get("vol_target", 0.0) <= 0:
        errors.append(
            "Set a positive volatility target when volatility adjustment is on."
        )
    return errors


def _initial_model_state() -> dict[str, Any]:
    defaults = _preset_defaults("Baseline")
    return {
        "trend_spec_preset": "Baseline",
        "trend_spec": defaults,
        "lookback_months": 36,
        "evaluation_months": 12,
        "selection_count": 10,
        "weighting_scheme": "equal",
        "metric_weights": {code: 1.0 for _, code in METRIC_FIELDS},
        "risk_target": 0.1,
        "warmup_periods": 0,
    }


def render_model_page() -> None:
    app_state.initialize_session_state()
    st.title("Model")

    df, _ = app_state.get_uploaded_data()
    if df is None:
        st.error("Load data on the Data page before configuring the model.")
        return

    model_state = st.session_state.setdefault("model_state", _initial_model_state())

    preset_options = ["Baseline"] + sorted(list_trend_spec_presets()) + ["Custom"]
    current_preset = model_state.get("trend_spec_preset")
    if not current_preset:
        current_preset = "Custom" if model_state.get("trend_spec") else "Baseline"
    try:
        preset_index = preset_options.index(current_preset)
    except ValueError:
        preset_index = 0

    with st.form("model_settings", clear_on_submit=False):
        st.subheader("Trend signal")
        preset = st.selectbox("Preset", preset_options, index=preset_index)
        if preset != model_state.get("trend_spec_preset") and preset != "Custom":
            model_state["trend_spec"] = _preset_defaults(preset)
            model_state["trend_spec_preset"] = preset

        defaults = model_state.get("trend_spec", _preset_defaults(preset))
        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.number_input(
                "Window (months)", min_value=1, value=int(defaults.get("window", 63))
            )
            min_periods = st.number_input(
                "Minimum periods",
                min_value=1,
                value=int(defaults.get("min_periods", defaults.get("window", 63))),
            )
        with c2:
            lag = st.number_input("Lag", min_value=1, value=int(defaults.get("lag", 1)))
            vol_adjust = st.checkbox(
                "Volatility adjust",
                value=bool(defaults.get("vol_adjust", False)),
            )
        with c3:
            vol_target = st.number_input(
                "Volatility target",
                min_value=0.0,
                value=float(defaults.get("vol_target", 0.0)),
                step=0.01,
                format="%.2f",
            )
            zscore = st.checkbox(
                "Row z-score", value=bool(defaults.get("zscore", False))
            )

        st.divider()
        st.subheader("Portfolio")
        c4, c5, c6 = st.columns(3)
        with c4:
            lookback = st.number_input(
                "Lookback months",
                min_value=12,
                value=int(model_state.get("lookback_months", 36)),
            )
        with c5:
            evaluation = st.number_input(
                "Evaluation window (months)",
                min_value=3,
                value=int(model_state.get("evaluation_months", 12)),
            )
        with c6:
            selection = st.number_input(
                "Selection count",
                min_value=1,
                value=int(model_state.get("selection_count", 10)),
            )

        weighting = st.selectbox(
            "Weighting scheme",
            ["equal", "vol_target"],
            index=0 if model_state.get("weighting_scheme") == "equal" else 1,
            help="Equal weights or volatility targeting in the portfolio stage.",
        )

        st.divider()
        st.subheader("Metric weights")
        metric_weights: dict[str, float] = {}
        cols = st.columns(len(METRIC_FIELDS))
        for (label, code), col in zip(METRIC_FIELDS, cols):
            with col:
                metric_weights[code] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(model_state.get("metric_weights", {}).get(code, 1.0)),
                    step=0.1,
                )

        st.divider()
        risk_target = st.number_input(
            "Target volatility",
            min_value=0.0,
            value=float(model_state.get("risk_target", 0.1)),
            step=0.01,
        )

        submitted = st.form_submit_button("Save model", type="primary")

        if submitted:
            trend_spec = _trend_spec_from_form(
                {
                    "window": window,
                    "lag": lag,
                    "min_periods": min_periods,
                    "vol_adjust": vol_adjust,
                    "vol_target": vol_target,
                    "zscore": zscore,
                }
            )
            candidate_state = {
                "trend_spec_preset": None if preset == "Custom" else preset,
                "trend_spec": trend_spec,
                "lookback_months": lookback,
                "evaluation_months": evaluation,
                "selection_count": selection,
                "weighting_scheme": weighting,
                "metric_weights": metric_weights,
                "risk_target": risk_target,
                "warmup_periods": model_state.get("warmup_periods", 0),
            }
            errors = _validate_model(candidate_state, len(df.columns))
            if errors:
                st.error("\n".join(f"• {err}" for err in errors))
            else:
                st.session_state["model_state"] = candidate_state
                analysis_runner.clear_cached_analysis()
                app_state.clear_analysis_results()
                st.success("Model configuration saved.")


render_model_page()
