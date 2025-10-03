"""Enhanced Configure page with presets, guardrails, and inline validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd
import streamlit as st



def _ensure_src_path() -> None:
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_path()

from streamlit_app.components.guardrails import (  # noqa: E402
    estimate_resource_usage,
    validate_startup_payload,
)
from trend_analysis.presets import TrendPreset, list_trend_presets  # noqa: E402
from trend_analysis.signal_presets import (  # noqa: E402
    TrendSpecPreset,
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from trend_analysis.signals import TrendSpec  # noqa: E402
from trend_portfolio_app.metrics_extra import AVAILABLE_METRICS  # noqa: E402
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig  # noqa: E402


def _map_payload_errors(payload_errors: Iterable[str]) -> Dict[str, List[str]]:
    """Translate validator error messages into inline field errors."""

    field_mapping: Mapping[str, tuple[str, ...]] = {
        "risk_target": ("target_vol", "vol_adjust", "risk target"),
        "date_column": ("date_column", "date column"),
        "return_columns": ("return_columns", "return columns"),
        "column_mapping": ("csv_path", "managers_glob", "upload"),
    }
    mapped: Dict[str, List[str]] = {}
    for raw_message in payload_errors:
        message = raw_message.strip()
        lowered = message.lower()
        for field, keywords in field_mapping.items():
            if any(keyword in lowered for keyword in keywords):
                mapped.setdefault(field, []).append(message)
                break
    return mapped


def initialize_session_state():
    """Initialize session state variables."""
    if "config_state" not in st.session_state:
        st.session_state.config_state = {
            "preset_name": None,
            "preset_config": None,
            "column_mapping": None,
            "custom_overrides": {},
            "trend_spec_values": {},
            "trend_spec_defaults": {},
            "trend_spec_preset": None,
            "trend_spec_config": {},
            "validation_errors": [],
            "is_valid": False,
            "resource_estimate": None,
        }
    if "validation_messages" not in st.session_state:
        st.session_state.validation_messages = []
    if "validated_min_config" not in st.session_state:
        st.session_state.validated_min_config = None
    if "field_errors" not in st.session_state:
        st.session_state.field_errors = {}


def display_inline_errors(field: str) -> None:
    """Render inline validation errors for a specific field."""

    messages = (st.session_state.get("field_errors") or {}).get(field, [])
    for message in messages:
        st.markdown(f":red[⚠️ {message}]")


def render_preset_selection() -> TrendPreset | None:
    """Render preset selection UI."""

    st.subheader("📋 Configuration Preset")

    presets = list_trend_presets()
    if not presets:
        st.warning("No presets found in config/presets/")
        return None

    preset_options = ["Custom"] + [preset.label for preset in presets]
    current_label = st.session_state.config_state.get("preset_name") or "Custom"
    try:
        current_index = preset_options.index(current_label)
    except ValueError:
        current_index = 0

    selected_label = st.selectbox(
        "Choose a configuration preset:",
        options=preset_options,
        index=current_index,
        help="Presets provide sensible defaults for different risk profiles",
    )

    if selected_label == "Custom":
        st.session_state.config_state["preset_name"] = None
        st.session_state.config_state["preset_config"] = None
        return None

    selected = next((p for p in presets if p.label == selected_label), None)
    if selected is None:
        st.error(f"Unknown preset '{selected_label}'")
        return None

    st.session_state.config_state["preset_name"] = selected.label
    st.session_state.config_state["preset_config"] = selected
    if selected.description:
        st.info(f"**{selected.label}**: {selected.description}")
    return selected


def render_column_mapping(df: pd.DataFrame):
    """Render column mapping UI."""
    st.subheader("🔗 Column Mapping")
    display_inline_errors("column_mapping")

    # Initialize display names and tickers at function scope
    display_names: Dict[str, str] = {}
    tickers: Dict[str, str] = {}

    cols = df.columns.tolist()

    with st.expander("Map your data columns", expanded=True):
        # Date column selection
        date_col = st.selectbox(
            "Date column",
            options=cols,
            index=0 if cols else None,
            help="Column containing date information",
        )
        display_inline_errors("date_column")

        # Return columns selection
        return_cols = st.multiselect(
            "Return data columns",
            options=[c for c in cols if c != date_col],
            default=[c for c in cols if c != date_col][
                :10
            ],  # Default to first 10 non-date columns
            help="Columns containing return data for funds/assets",
        )
        display_inline_errors("return_columns")

        # Benchmark and risk-free rate
        benchmark_col = st.selectbox(
            "Benchmark column (optional)",
            options=["<none>"]
            + [c for c in cols if c not in return_cols and c != date_col],
            help="Column for benchmark comparison",
        )

        risk_free_col = st.selectbox(
            "Risk-free rate column (optional)",
            options=["<none>"]
            + [
                c
                for c in cols
                if c not in return_cols and c != date_col and c != benchmark_col
            ],
            help="Column containing risk-free rate data",
        )

        # Column display names and tickers
        if return_cols:
            st.subheader("Column Labels (Optional)")
            col1, col2 = st.columns(2)

            for i, col in enumerate(return_cols):
                with col1:
                    display_names[col] = st.text_input(
                        f"Display name for {col}", value=col, key=f"display_{i}"
                    )
                with col2:
                    tickers[col] = st.text_input(
                        f"Ticker for {col}",
                        value="",
                        key=f"ticker_{i}",
                        help="Optional ticker symbol",
                    )

    mapping = {
        "date_column": date_col,
        "return_columns": return_cols,
        "benchmark_column": None if benchmark_col == "<none>" else benchmark_col,
        "risk_free_column": None if risk_free_col == "<none>" else risk_free_col,
        "column_display_names": display_names,
        "column_tickers": tickers,
    }

    st.session_state.config_state["column_mapping"] = mapping
    meta = st.session_state.get("schema_meta") or {}
    if df is not None:
        n_rows = int(meta.get("n_rows", df.shape[0]))
    else:
        n_rows = int(meta.get("n_rows", 0))
    estimate = estimate_resource_usage(n_rows, len(mapping.get("return_columns", [])))
    st.session_state.config_state["resource_estimate"] = estimate
    if n_rows and mapping.get("return_columns"):
        st.caption(
            f"Resource estimate: ~{estimate.approx_memory_mb:.1f} MB in memory, "
            f"~{estimate.estimated_runtime_s/60:.1f} min for full run."
        )
        for warn in estimate.warnings:
            st.warning(f"Guardrail: {warn}")
    return mapping


def render_trend_spec_settings(selected_preset_label: Optional[str]) -> None:
    """Render the TrendSpec preset selector and synchronise session state."""

    st.subheader("📈 Trend Signal Settings")

    config_state = st.session_state.config_state
    config_state.setdefault("trend_spec_values", {})
    config_state.setdefault("trend_spec_defaults", {})
    config_state.setdefault("trend_spec_config", {})

    available_presets = list_trend_spec_presets()
    if not available_presets:
        st.warning("No trend signal presets available.")
        if not config_state.get("trend_spec_values"):
            _apply_trend_spec_preset_to_state(config_state, None)
        return

    current_values = config_state.get("trend_spec_values") or {}
    current_defaults = config_state.get("trend_spec_defaults") or {}
    current_preset = config_state.get("trend_spec_preset")

    if (
        selected_preset_label
        and selected_preset_label in available_presets
        and (
            not current_values
            or (current_preset and current_values == current_defaults)
        )
    ):
        _apply_trend_spec_preset_to_state(config_state, selected_preset_label)
        current_preset = config_state.get("trend_spec_preset")
        current_defaults = config_state.get("trend_spec_defaults") or {}
        current_values = config_state.get("trend_spec_values") or {}

    options = ["Custom"] + available_presets
    try:
        current_index = options.index(current_preset or "Custom")
    except ValueError:
        current_index = 0

    choice = st.selectbox(
        "Trend signal preset",
        options=options,
        index=current_index,
        help="Apply a preset TrendSpec profile to synchronise with the CLI.",
    )

    if choice == "Custom":
        config_state["trend_spec_preset"] = None
        if not config_state.get("trend_spec_values"):
            current_values = _apply_trend_spec_preset_to_state(config_state, None)
        st.info("Adjust the trend signal parameters below to customise the profile.")
    else:
        if choice != current_preset:
            current_values = _apply_trend_spec_preset_to_state(config_state, choice)
            current_defaults = config_state.get("trend_spec_defaults") or {}
        preset = get_trend_spec_preset(choice)
        if preset.description:
            st.caption(preset.description)
        summary_lines = [
            f"• Window: **{current_values.get('window')}** trading days",
            f"• Minimum periods: **{current_values.get('min_periods', 0)}**",
            f"• Lag: **{current_values.get('lag')}**",
            "• Volatility adjustment: **on**"
            if current_values.get("vol_adjust")
            else "• Volatility adjustment: **off**",
            (
                f"• Volatility target: **{current_values.get('vol_target', 0.0):.2f}**"
                if current_values.get("vol_adjust")
                else "• Volatility target: _not used_"
            ),
            "• Z-score normalisation: **enabled**"
            if current_values.get("zscore")
            else "• Z-score normalisation: **disabled**",
        ]
        st.markdown("\n".join(summary_lines))


def render_parameter_forms(preset: TrendPreset | None):
    """Render parameter configuration forms."""
    st.subheader("⚙️ Analysis Parameters")

    defaults = preset.form_defaults() if preset else {}
    default_lookback = int(defaults.get("lookback_months", 36))
    default_rebalance = str(defaults.get("rebalance_frequency", "monthly"))
    default_min_track = int(defaults.get("min_track_months", 24))
    default_selection = int(defaults.get("selection_count", 10))
    default_risk_target = float(defaults.get("risk_target", 0.10))

    signal_spec = preset.trend_spec if preset else None
    default_signal_window = int(signal_spec.window if signal_spec else 63)
    default_signal_min = (
        int(signal_spec.min_periods)
        if signal_spec and signal_spec.min_periods is not None
        else default_signal_window
    )
    default_signal_lag = int(signal_spec.lag if signal_spec else 1)
    default_signal_vol_adjust = bool(signal_spec.vol_adjust if signal_spec else False)
    default_signal_vol_target = (
        float(signal_spec.vol_target)
        if signal_spec and signal_spec.vol_target is not None
        else 0.10
    )
    default_signal_zscore = bool(signal_spec.zscore if signal_spec else False)

    df = st.session_state.get("returns_df")
    total_months = 0
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Ensure index is datetime before calling to_period
        if isinstance(df.index, pd.DatetimeIndex):
            total_months = len(df.index.to_period("M").unique())
        else:
            try:
                dt_index = pd.to_datetime(df.index)
                total_months = len(dt_index.to_period("M").unique())
            except Exception:
                total_months = 0
                st.warning(
                    "The data index could not be interpreted as dates. "
                    "Please ensure the date column is properly configured."
                )
    min_lookback_allowed = 6 if 0 < total_months < 24 else 12
    if total_months:
        max_lookback_allowed = max(min_lookback_allowed, total_months - 3)
    else:
        max_lookback_allowed = min_lookback_allowed
    lookback_default = min(default_lookback, max_lookback_allowed)
    lookback_step = 6 if (max_lookback_allowed - min_lookback_allowed) >= 6 else 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Time Windows**")
        lookback_months = st.number_input(
            "Lookback window (months)",
            min_value=min_lookback_allowed,
            max_value=max_lookback_allowed,
            value=lookback_default,
            step=lookback_step,
            help="Historical data period for analysis",
        )
        display_inline_errors("lookback_months")

        min_track_months = st.number_input(
            "Minimum track record (months)",
            min_value=6,
            max_value=120,
            value=default_min_track,
            step=6,
            help="Minimum history required for fund inclusion",
        )

        st.markdown("**Portfolio Construction**")
        selection_count = st.number_input(
            "Number of funds to select",
            min_value=1,
            max_value=50,
            value=default_selection,
            step=1,
            help="Maximum number of funds in portfolio",
        )
        display_inline_errors("selection_count")

    with col2:
        st.markdown("**Rebalancing**")
        rebalance_freq = st.selectbox(
            "Rebalance frequency",
            options=["monthly", "quarterly", "annually"],
            index=["monthly", "quarterly", "annually"].index(default_rebalance),
            help="How often to rebalance the portfolio",
        )

        risk_target = st.number_input(
            "Risk target (volatility)",
            min_value=0.02,
            max_value=0.30,
            value=default_risk_target,
            step=0.01,
            format="%.2f",
            help="Target portfolio volatility (e.g., 0.10 = 10%)",
        )
        display_inline_errors("risk_target")

        cooldown_months = st.number_input(
            "Cooldown period (months)",
            min_value=0,
            max_value=36,
            value=3,
            step=1,
            help="Minimum time before reconsidering fired funds",
        )

        st.markdown("**Weighting**")
        weighting_scheme = st.selectbox(
            "Weighting scheme",
            options=["equal", "risk_parity", "hrp", "erc"],
            index=0,
            help="How to allocate weights across selected funds",
        )

    # Trend signal configuration
    st.markdown("**Trend Signal Parameters**")
    sig_col1, sig_col2 = st.columns(2)
    with sig_col1:
        signal_window = st.number_input(
            "Signal lookback (trading days)",
            min_value=10,
            max_value=504,
            value=default_signal_window,
            step=5,
            help="Rolling window length used to compute the trend signal.",
        )
        display_inline_errors("signal_window")

        signal_min_periods = st.number_input(
            "Minimum periods for signal",
            min_value=0,
            max_value=int(signal_window),
            value=default_signal_min,
            step=1,
            help="Required history before the trend signal becomes active.",
        )
        display_inline_errors("signal_min_periods")

        signal_lag = st.number_input(
            "Signal lag (periods)",
            min_value=1,
            max_value=12,
            value=default_signal_lag,
            step=1,
            help="Execution delay applied to the computed signal.",
        )

    with sig_col2:
        signal_vol_adjust = st.checkbox(
            "Volatility adjust signals",
            value=default_signal_vol_adjust,
            help="Scale signals by recent volatility to stabilise exposures.",
        )
        signal_vol_target = None
        if signal_vol_adjust:
            signal_vol_target = st.number_input(
                "Signal volatility target",
                min_value=0.0,
                max_value=1.0,
                value=default_signal_vol_target,
                step=0.01,
                format="%.2f",
                help="Target volatility used when rescaling the signal.",
            )
        display_inline_errors("signal_vol_target")

        signal_zscore = st.checkbox(
            "Apply row-wise z-score normalisation",
            value=default_signal_zscore,
            help="Normalise signals across assets for each period.",
        )

    # Metrics selection
    st.markdown("**Selection Metrics**")
    metric_names = list(AVAILABLE_METRICS.keys())

    preset_metrics = defaults.get("metrics") if defaults else {}
    if isinstance(preset_metrics, Mapping) and preset_metrics:
        filtered_preset_metrics = {metric: weight for metric, weight in preset_metrics.items() if metric in metric_names}
        default_metrics = list(filtered_preset_metrics.keys())
        preset_weights = {metric: float(weight) for metric, weight in filtered_preset_metrics.items()}
    else:
        default_metrics = ["sharpe", "return_ann", "drawdown"]
        preset_weights = {}

    selected_metrics = st.multiselect(
        "Selection metrics",
        options=metric_names,
        default=[m for m in default_metrics if m in metric_names],
        help="Metrics used for fund selection and ranking",
    )

    # Metric weights
    weights = {}
    if selected_metrics:
        st.markdown("**Metric Weights**")
        metric_cols = st.columns(min(len(selected_metrics), 3))

        for i, metric in enumerate(selected_metrics):
            with metric_cols[i % len(metric_cols)]:
                default_weight = preset_weights.get(metric, 1.0 / len(selected_metrics))
                weights[metric] = st.number_input(
                    f"Weight: {metric}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.05,
                    format="%.2f",
                    key=f"weight_{metric}",
                )
        display_inline_errors("metric_weights")

    # Store custom overrides
    signals_override = {
        "window": int(signal_window),
        "lag": int(signal_lag),
        "vol_adjust": bool(signal_vol_adjust),
        "zscore": bool(signal_zscore),
    }
    min_periods_val = int(signal_min_periods)
    if min_periods_val > 0:
        signals_override["min_periods"] = min_periods_val
    vol_target_val = (
        float(signal_vol_target)
        if signal_vol_target is not None and signal_vol_adjust
        else None
    )
    if vol_target_val is not None:
        signals_override["vol_target"] = vol_target_val

    overrides = {
        "lookback_months": lookback_months,
        "rebalance_frequency": rebalance_freq,
        "min_track_months": min_track_months,
        "selection_count": selection_count,
        "risk_target": risk_target,
        "cooldown_months": cooldown_months,
        "selected_metrics": selected_metrics,
        "metric_weights": weights,
        "weighting_scheme": weighting_scheme,
        "signals": signals_override,
    }

    trend_spec_values = config_state.get("trend_spec_values") or {}
    if not trend_spec_values:
        trend_spec_values = _trend_spec_defaults_from_preset(
            config_state.get("trend_spec_preset")
        )
    trend_spec_normalised = _normalise_trend_spec_values(trend_spec_values)
    config_state["trend_spec_values"] = dict(trend_spec_normalised)
    config_state["trend_spec_config"] = _trend_spec_values_to_config(trend_spec_normalised)
    overrides["trend_spec"] = dict(trend_spec_normalised)

    st.session_state.config_state["custom_overrides"] = overrides
    return overrides


def validate_configuration() -> List[str]:
    """Validate the current configuration and return error messages."""
    errors: List[str] = []
    field_errors: Dict[str, List[str]] = {}
    config_state = st.session_state.config_state

    # Check for required data
    if "returns_df" not in st.session_state:
        errors.append("No data uploaded. Please upload data first.")
        st.session_state.field_errors = field_errors
        return errors

    # Check column mapping
    mapping = config_state.get("column_mapping")
    if not mapping:
        errors.append("Column mapping not configured.")
        field_errors.setdefault("column_mapping", []).append(
            "Map the date and return columns before continuing."
        )
    else:
        if not mapping.get("date_column"):
            errors.append("Date column not selected.")
            field_errors.setdefault("date_column", []).append(
                "Select the column that contains your dates."
            )
        if not mapping.get("return_columns"):
            errors.append("No return columns selected.")
            field_errors.setdefault("return_columns", []).append(
                "Choose at least one column with fund returns."
            )

    # Check custom overrides
    if config_state.get("custom_overrides"):
        overrides = config_state["custom_overrides"]

        return_cols = (mapping or {}).get("return_columns") or []
        df = st.session_state.get("returns_df")

        # Check metric weights sum to approximately 1
        if overrides.get("metric_weights"):
            total_weight = sum(overrides["metric_weights"].values())
            if abs(total_weight - 1.0) > 0.05:  # Allow some tolerance
                errors.append(
                    f"Metric weights should sum to 1.0, currently {total_weight:.2f}"
                )
                field_errors.setdefault("metric_weights", []).append(
                    "Adjust weights so they sum to 1.0."
                )

        # Check reasonable parameter ranges
        if overrides.get("lookback_months", 0) < overrides.get("min_track_months", 0):
            errors.append("Lookback window should be >= minimum track record.")
            field_errors.setdefault("lookback_months", []).append(
                "Increase the lookback so it is at least the minimum track record."
            )

        if overrides.get("selection_count", 0) <= 0:
            errors.append("Selection count must be positive.")
            field_errors.setdefault("selection_count", []).append(
                "Set a positive number of funds to select."
            )

        if return_cols and overrides.get("selection_count", 0) > len(return_cols):
            errors.append(
                "Selection count exceeds the number of mapped return columns."
            )
            field_errors.setdefault("selection_count", []).append(
                "Reduce the selection count or map more return columns."
            )

        risk_target = overrides.get("risk_target", 0.10)
        if not 0.01 <= risk_target <= 0.50:
            errors.append("Risk target should be between 1% and 50%.")
            field_errors.setdefault("risk_target", []).append(
                "Enter a risk target between 0.01 and 0.50."
            )

        signals = overrides.get("signals", {}) or {}
        if signals:
            window_val = signals.get("window", 0)
            try:
                window_int = int(window_val)
            except (TypeError, ValueError):
                window_int = 0
            if window_int <= 0:
                errors.append("Signal window must be a positive integer.")
                field_errors.setdefault("signal_window", []).append(
                    "Set a positive number of periods for the signal window."
                )

            min_periods_raw = signals.get("min_periods")
            if min_periods_raw is not None:
                try:
                    min_periods_int = int(min_periods_raw)
                except (TypeError, ValueError):
                    min_periods_int = None
                if min_periods_int is None or min_periods_int <= 0:
                    errors.append(
                        "Minimum periods must be a positive integer when provided."
                    )
                    field_errors.setdefault("signal_min_periods", []).append(
                        "Enter a positive number of periods or leave blank."
                    )
                elif window_int and min_periods_int > window_int:
                    errors.append(
                        "Signal minimum periods cannot exceed the signal window."
                    )
                    field_errors.setdefault("signal_min_periods", []).append(
                        "Reduce minimum periods so it is not greater than the signal window."
                    )

            if bool(signals.get("vol_adjust")):
                vol_target_raw = signals.get("vol_target")
                try:
                    vol_target_val = float(vol_target_raw) if vol_target_raw is not None else None
                except (TypeError, ValueError):
                    vol_target_val = None
                if vol_target_val is None or vol_target_val <= 0:
                    errors.append(
                        "Provide a positive signal volatility target when volatility adjustment is enabled."
                    )
                    field_errors.setdefault("signal_vol_target", []).append(
                        "Set a positive target or disable volatility adjustment."
                    )

        if df is not None and not df.empty and mapping and mapping.get("date_column"):
            unique_months = df.index.to_period("M").unique()
            lookback = int(overrides.get("lookback_months", 0) or 0)
            if lookback >= len(unique_months):
                errors.append(
                    "Lookback window consumes all available history. Reduce it to avoid look-ahead."
                )
                field_errors.setdefault("lookback_months", []).append(
                    "Shorten the lookback to leave some out-of-sample data."
                )
            elif len(unique_months) - lookback < 3:
                errors.append(
                    "Leave at least three months of out-of-sample data to avoid look-ahead bias."
                )
                field_errors.setdefault("lookback_months", []).append(
                    "Leave at least three months for the out-of-sample window."
                )

    st.session_state.field_errors = field_errors
    return errors


def save_configuration():
    """Save validated configuration to session state."""
    config_state = st.session_state.config_state

    # Validate first
    errors = validate_configuration()
    st.session_state.validation_messages = errors

    if errors:
        config_state["is_valid"] = False
        return False

    # Build policy config for compatibility with existing system
    overrides = config_state.get("custom_overrides", {})
    selected_metrics = overrides.get("selected_metrics", [])
    metric_weights = overrides.get("metric_weights", {})

    policy = PolicyConfig(
        top_k=overrides.get("selection_count", 10),
        bottom_k=0,  # Not exposed in UI for now
        cooldown_months=overrides.get("cooldown_months", 3),
        min_track_months=overrides.get("min_track_months", 24),
        max_active=100,  # Fixed for now
        max_weight=0.10,  # Fixed for now
        metrics=[
            MetricSpec(name=m, weight=metric_weights.get(m, 1.0))
            for m in selected_metrics
        ],
    )

    # Get date range from data
    df = st.session_state["returns_df"]
    mapping = config_state.get("column_mapping", {})
    validated_payload, payload_errors = validate_startup_payload(
        csv_path=st.session_state.get("uploaded_file_path"),
        date_column=mapping.get("date_column") or "Date",
        risk_target=overrides.get("risk_target", 0.10),
        timestamps=df.index,
    )
    if payload_errors:
        st.session_state.validation_messages = payload_errors
        payload_field_errors = _map_payload_errors(payload_errors)
        if payload_field_errors:
            field_errors = st.session_state.get("field_errors", {})
            for field, messages in payload_field_errors.items():
                field_errors.setdefault(field, []).extend(messages)
            st.session_state.field_errors = field_errors
        config_state["is_valid"] = False
        return False
    st.session_state.validated_min_config = validated_payload
    config_state["validated_min_config"] = validated_payload

    # Save to session state in expected format
    preset_config = config_state.get("preset_config")
    preset_slug = preset_config.slug if isinstance(preset_config, TrendPreset) else None

    st.session_state["sim_config"] = {
        "start": pd.Timestamp(df.index.min()),
        "end": pd.Timestamp(df.index.max()),
        "freq": overrides.get("rebalance_frequency", "monthly"),
        "lookback_months": overrides.get("lookback_months", 36),
        "benchmark": config_state.get("column_mapping", {}).get("benchmark_column"),
        "cash_rate": 0.0,  # Default for now
        "policy": policy.dict(),
        "rebalance": {
            "bayesian_only": True,
            "strategies": ["drift_band"],
            "params": {},
        },
        "risk_target": overrides.get("risk_target", 0.10),
        "column_mapping": config_state.get("column_mapping"),
        "preset_name": config_state.get("preset_name"),
        "trend_preset": preset_slug,
        "signals": overrides.get("signals", {}),
        "portfolio": {
            "weighting_scheme": overrides.get("weighting_scheme", "equal"),
        },
    }

    config_state["is_valid"] = True
    return True


# Main UI
def main():
    st.title("📊 Configure Analysis")

    # Initialize session state
    initialize_session_state()

    # Check for uploaded data
    if "returns_df" not in st.session_state:
        st.error("⚠️ Upload data first on the Upload page.")
        st.stop()

    df = st.session_state["returns_df"]
    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Render UI sections
    preset_config = render_preset_selection()
    st.divider()

    render_column_mapping(df)
    st.divider()

    render_trend_spec_settings(st.session_state.config_state.get("preset_name"))
    st.divider()

    render_parameter_forms(preset_config)
    st.divider()

    # Configuration summary and validation
    st.subheader("📋 Configuration Summary")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            if save_configuration():
                st.success("✅ Configuration saved successfully!")
                st.balloons()
            else:
                st.error("❌ Configuration has errors. Please fix them first.")

    with col2:
        if st.button("🔍 Validate Configuration"):
            errors = validate_configuration()
            if errors:
                st.error("❌ Configuration errors found:")
                for error in errors:
                    st.error(f"• {error}")
            else:
                st.success("✅ Configuration is valid!")
        if st.button("🧪 Validate Minimal Startup Config"):
            mapping = st.session_state.config_state.get("column_mapping")
            overrides = st.session_state.config_state.get("custom_overrides", {})
            df = st.session_state.get("returns_df")
            if mapping and df is not None:
                validated, payload_errors = validate_startup_payload(
                    csv_path=st.session_state.get("uploaded_file_path"),
                    date_column=mapping.get("date_column") or "Date",
                    risk_target=overrides.get("risk_target", 0.10),
                    timestamps=df.index,
                )
                if payload_errors:
                    for err in payload_errors:
                        st.error(f"Minimal config validation failed: {err}")
                else:
                    st.success("Minimal config validated (TrendConfig).")
                    st.session_state.validated_min_config = validated
                    st.session_state.config_state["validated_min_config"] = validated
                    with st.expander("Validated Minimal Config", expanded=False):
                        st.json(validated)
            else:
                st.warning("Upload data and map columns first.")

    # Show validation messages if any
    if st.session_state.validation_messages:
        st.subheader("⚠️ Validation Issues")
        for msg in st.session_state.validation_messages:
            st.warning(f"• {msg}")

    # Show current configuration status
    config_state = st.session_state.config_state
    if config_state.get("is_valid"):
        st.success("✅ Ready to proceed to Run analysis")

        # Show summary
        with st.expander("Configuration Details", expanded=False):
            st.json(
                {
                    "preset": config_state.get("preset_name") or "Custom",
                    "columns_mapped": len(
                        config_state.get("column_mapping", {}).get("return_columns", [])
                    ),
                    "parameters": {
                        k: v
                        for k, v in config_state.get("custom_overrides", {}).items()
                        if k not in ["selected_metrics", "metric_weights"]
                    },
                    "trend_spec": config_state.get("trend_spec_values", {}),
                    "metrics": list(
                        config_state.get("custom_overrides", {})
                        .get("metric_weights", {})
                        .keys()
                    ),
                }
            )
    else:
        st.info("Configure the settings above and save to proceed.")


if __name__ == "__main__":
    main()
def _trend_spec_defaults_from_spec(spec: TrendSpec) -> Dict[str, Any]:
    """Return form defaults derived from a ``TrendSpec`` instance."""

    return {
        "window": int(spec.window),
        "min_periods": int(spec.min_periods) if spec.min_periods is not None else 0,
        "lag": int(spec.lag),
        "vol_adjust": bool(spec.vol_adjust),
        "vol_target": float(spec.vol_target) if spec.vol_target is not None else 0.0,
        "zscore": bool(spec.zscore),
    }


def _trend_spec_defaults_from_preset(preset_name: Optional[str]) -> Dict[str, Any]:
    """Return trend signal defaults for the given preset (or baseline)."""

    if not preset_name:
        return _trend_spec_defaults_from_spec(TrendSpec())

    try:
        preset = get_trend_spec_preset(preset_name)
    except KeyError:
        return _trend_spec_defaults_from_spec(TrendSpec())
    return dict(preset.form_defaults())


def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    """Coerce ``value`` into a positive integer with bounds checking."""

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(coerced, minimum)


def _coerce_bool(value: Any) -> bool:
    """Coerce common truthy/falsey representations into a boolean."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _normalise_trend_spec_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise raw TrendSpec form values for storage and config output."""

    defaults = _trend_spec_defaults_from_spec(TrendSpec())

    window = _coerce_positive_int(values.get("window"), defaults["window"], minimum=1)
    lag = _coerce_positive_int(values.get("lag"), defaults["lag"], minimum=1)

    min_periods_raw = values.get("min_periods")
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
    vol_target: Optional[float]
    try:
        vol_target_candidate = float(vol_target_raw)
    except (TypeError, ValueError):
        vol_target_candidate = 0.0
    if not vol_adjust or vol_target_candidate <= 0.0:
        vol_target = None
    else:
        vol_target = vol_target_candidate

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
    """Convert normalised TrendSpec form values into config mapping."""

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
    """Apply a trend spec preset to session ``state`` and return defaults."""

    state.setdefault("trend_spec_defaults", {})
    state.setdefault("trend_spec_values", {})
    state.setdefault("trend_spec_config", {})

    if preset_name:
        try:
            preset: TrendSpecPreset = get_trend_spec_preset(preset_name)
        except KeyError:
            preset = None
        if preset is not None:
            defaults = dict(preset.form_defaults())
            state["trend_spec_defaults"] = dict(defaults)
            state["trend_spec_values"] = dict(defaults)
            state["trend_spec_preset"] = preset.name
            state["trend_spec_config"] = preset.as_signal_config()
            return defaults

    defaults = _trend_spec_defaults_from_spec(TrendSpec())
    state["trend_spec_defaults"] = dict(defaults)
    state["trend_spec_values"] = dict(defaults)
    state["trend_spec_preset"] = None
    state["trend_spec_config"] = _trend_spec_values_to_config(defaults)
    return defaults
