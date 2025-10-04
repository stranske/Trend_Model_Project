"""Model configuration page with TrendSpec presets and validation."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import streamlit as st

from app.streamlit import state as app_state
from streamlit_app.components.analysis_runner import (
    ModelSettings,
    clear_preprocessing_caches,
)
from trend_analysis.signal_presets import get_trend_spec_preset, list_trend_spec_presets
from trend_analysis.signals import TrendSpec
from trend_portfolio_app.metrics_extra import AVAILABLE_METRICS


DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    "sharpe": 0.5,
    "return_ann": 0.3,
    "drawdown": 0.2,
}


def _trend_defaults(preset_name: str | None) -> Dict[str, Any]:
    if preset_name:
        try:
            preset = get_trend_spec_preset(preset_name)
        except KeyError:
            preset = None
        if preset is not None:
            return dict(preset.form_defaults())
    spec = TrendSpec()
    return {
        "window": int(spec.window),
        "min_periods": int(spec.min_periods) if spec.min_periods else spec.window,
        "lag": int(spec.lag),
        "vol_adjust": bool(spec.vol_adjust),
        "vol_target": float(spec.vol_target) if spec.vol_target else 0.1,
        "zscore": bool(spec.zscore),
    }


def _build_trend_payload(values: Mapping[str, Any]) -> Dict[str, Any]:
    spec = TrendSpec(
        window=int(values.get("window", 126)),
        min_periods=int(values.get("min_periods", 0)) or None,
        lag=max(int(values.get("lag", 1)), 1),
        vol_adjust=bool(values.get("vol_adjust", False)),
        vol_target=(
            float(values.get("vol_target", 0.0))
            if bool(values.get("vol_adjust", False))
            else None
        ),
        zscore=bool(values.get("zscore", False)),
    )
    payload: Dict[str, Any] = {
        "kind": spec.kind,
        "window": spec.window,
        "lag": spec.lag,
        "vol_adjust": spec.vol_adjust,
        "zscore": spec.zscore,
    }
    if spec.min_periods is not None:
        payload["min_periods"] = spec.min_periods
    if spec.vol_target is not None:
        payload["vol_target"] = spec.vol_target
    return payload


def _normalise_weights(raw: Mapping[str, float]) -> Dict[str, float]:
    weights = {str(k).lower(): float(v) for k, v in raw.items() if float(v) > 0}
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_METRIC_WEIGHTS)
    return {metric: value / total for metric, value in weights.items()}


def _existing_settings() -> ModelSettings | None:
    value = st.session_state.get("model_settings")
    return value if isinstance(value, ModelSettings) else None


def _existing_mapping(df: pd.DataFrame) -> Dict[str, Any]:
    mapping = st.session_state.get("model_column_mapping")
    if isinstance(mapping, dict) and mapping.get("return_columns"):
        return dict(mapping)
    benchmark = st.session_state.get("selected_benchmark")
    return {
        "date_column": df.index.name or "Date",
        "return_columns": [col for col in df.columns if col != benchmark],
        "benchmark_column": benchmark,
    }


def main() -> None:
    app_state.initialize_session_state()
    st.title("🧠 Model")
    df = st.session_state.get("returns_df")
    if df is None:
        st.error("Load a dataset on the Data page before configuring the model.")
        st.stop()

    mapping_defaults = _existing_mapping(df)
    settings_defaults = _existing_settings()

    st.caption(
        "Choose how the Trend model should behave. Presets provide sensible TrendSpec values, "
        "and validation guards against impossible combinations."
    )

    preset_options = ["Custom"] + list(list_trend_spec_presets())
    preset_default = settings_defaults.trend_spec.get("preset") if settings_defaults else None
    preset_index = (
        preset_options.index(preset_default)
        if preset_default in preset_options
        else 0
    )

    with st.form("model_form"):
        st.subheader("Column mapping")
        cols = df.columns.tolist()
        return_columns = mapping_defaults.get("return_columns", cols)
        benchmark_column = mapping_defaults.get("benchmark_column")

        selected_returns = st.multiselect(
            "Return columns",
            options=cols,
            default=return_columns,
            help="Assets or managers to include in the simulation.",
        )
        candidate_benchmarks = st.session_state.get("benchmark_candidates", [])
        benchmark_choice = None
        if candidate_benchmarks:
            default_idx = (
                candidate_benchmarks.index(benchmark_column)
                if benchmark_column in candidate_benchmarks
                else 0
            )
            benchmark_choice = st.selectbox(
                "Benchmark (optional)",
                options=["<none>"] + candidate_benchmarks,
                index=default_idx + 1,
            )
            if benchmark_choice == "<none>":
                benchmark_choice = None
        else:
            benchmark_choice = benchmark_column if benchmark_column in cols else None

        st.divider()

        st.subheader("TrendSpec")
        preset_choice = st.selectbox("Preset", options=preset_options, index=preset_index)
        trend_defaults = _trend_defaults(preset_choice if preset_choice != "Custom" else None)
        window = st.number_input(
            "Window (days)", min_value=20, max_value=520, value=int(trend_defaults["window"])
        )
        min_periods = st.number_input(
            "Minimum periods", min_value=0, max_value=int(window), value=int(trend_defaults["min_periods"])
        )
        lag = st.number_input("Lag", min_value=1, max_value=12, value=int(trend_defaults["lag"]))
        vol_adjust = st.checkbox(
            "Apply volatility targeting",
            value=bool(trend_defaults.get("vol_adjust", False)),
        )
        vol_target = 0.10
        if vol_adjust:
            vol_target = st.number_input(
                "Target volatility",
                min_value=0.01,
                max_value=1.0,
                value=float(trend_defaults.get("vol_target", 0.10)),
                step=0.01,
                format="%.2f",
            )
        zscore = st.checkbox(
            "Normalise with z-scores",
            value=bool(trend_defaults.get("zscore", False)),
        )

        st.divider()
        st.subheader("Portfolio policy")
        max_return_cols = max(len(selected_returns), 1)
        lookback_months = st.number_input(
            "Lookback window (months)",
            min_value=12,
            max_value=120,
            value=settings_defaults.lookback_months if settings_defaults else 36,
            step=3,
        )
        selection_count = st.number_input(
            "Selection count",
            min_value=1,
            max_value=max_return_cols,
            value=min(
                settings_defaults.selection_count if settings_defaults else 10,
                max_return_cols,
            ),
            step=1,
        )
        risk_target = st.number_input(
            "Portfolio risk target",
            min_value=0.01,
            max_value=0.50,
            value=settings_defaults.risk_target if settings_defaults else 0.10,
            step=0.01,
            format="%.2f",
        )
        cooldown_months = st.number_input(
            "Cooldown (months)",
            min_value=0,
            max_value=12,
            value=settings_defaults.cooldown_months if settings_defaults else 3,
            step=1,
        )
        min_track_months = st.number_input(
            "Minimum track record (months)",
            min_value=6,
            max_value=60,
            value=settings_defaults.min_track_months if settings_defaults else 24,
            step=6,
        )
        weighting_scheme = st.selectbox(
            "Weighting scheme",
            options=["equal", "risk_budget"],
            index=(0 if not settings_defaults else ["equal", "risk_budget"].index(settings_defaults.weighting_scheme) if settings_defaults.weighting_scheme in ["equal", "risk_budget"] else 0),
        )

        st.divider()
        st.subheader("Selection metrics")
        metric_names = list(AVAILABLE_METRICS.keys())
        metric_default = (
            list(settings_defaults.metric_weights.keys())
            if settings_defaults
            else list(DEFAULT_METRIC_WEIGHTS.keys())
        )
        chosen_metrics = st.multiselect(
            "Metrics",
            options=metric_names,
            default=[m for m in metric_default if m in metric_names],
        )
        weights_input: Dict[str, float] = {}
        if chosen_metrics:
            cols_metrics = st.columns(min(len(chosen_metrics), 3))
            for idx, metric in enumerate(chosen_metrics):
                with cols_metrics[idx % len(cols_metrics)]:
                    default_weight = (
                        settings_defaults.metric_weights.get(metric)
                        if settings_defaults and metric in settings_defaults.metric_weights
                        else DEFAULT_METRIC_WEIGHTS.get(metric, 1.0 / len(chosen_metrics))
                    )
                    weights_input[metric] = st.number_input(
                        f"{metric} weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(default_weight),
                        step=0.05,
                        format="%.2f",
                    )

        submitted = st.form_submit_button("Save model settings", type="primary")

    if not submitted:
        if settings_defaults:
            st.success("Existing model settings loaded. Adjust values and save to update.")
        return

    errors: list[str] = []
    if not selected_returns:
        errors.append("Select at least one return column.")
    if selection_count > len(selected_returns):
        errors.append("Selection count cannot exceed the number of return columns.")
    if risk_target <= 0 or risk_target > 0.5:
        errors.append("Risk target must sit between 0.01 and 0.50.")
    if lookback_months <= min_track_months:
        errors.append("Lookback window should exceed the minimum track record.")
    if chosen_metrics and sum(weights_input.values()) <= 0:
        errors.append("Metric weights must contain at least one positive value.")
    if lookback_months >= len(df):
        errors.append("Lookback window consumes all available history. Reduce the window.")

    if errors:
        st.error("Please address the following:")
        for msg in errors:
            st.write("•", msg)
        return

    trend_payload = _build_trend_payload(
        {
            "window": window,
            "min_periods": min_periods,
            "lag": lag,
            "vol_adjust": vol_adjust,
            "vol_target": vol_target,
            "zscore": zscore,
        }
    )
    if preset_choice != "Custom":
        trend_payload["preset"] = preset_choice

    metric_weights = _normalise_weights(weights_input or DEFAULT_METRIC_WEIGHTS)
    mapping = {
        "date_column": df.index.name or "Date",
        "return_columns": selected_returns,
        "benchmark_column": benchmark_choice,
    }

    settings = ModelSettings(
        lookback_months=int(lookback_months),
        rebalance_frequency="monthly",
        selection_count=int(selection_count),
        risk_target=float(risk_target),
        weighting_scheme=weighting_scheme,
        cooldown_months=int(cooldown_months),
        min_track_months=int(min_track_months),
        metric_weights=metric_weights,
        trend_spec=trend_payload,
        benchmark=benchmark_choice,
    )

    st.session_state["model_settings"] = settings
    st.session_state["model_column_mapping"] = mapping
    clear_preprocessing_caches()
    st.session_state.pop("sim_results", None)
    st.success("Model settings saved. Proceed to Results to run the analysis.")


main()
