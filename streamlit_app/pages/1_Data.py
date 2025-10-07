"""Data page for the Streamlit application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.streamlit import state as app_state
from streamlit_app.components import analysis_runner, data_cache
from trend_analysis.io.market_data import MarketDataValidationError
from trend_portfolio_app.data_schema import SchemaMeta, infer_benchmarks


def _dataset_summary(df: pd.DataFrame, meta: SchemaMeta | dict[str, Any]) -> str:
    start = pd.to_datetime(df.index.min()).date()
    end = pd.to_datetime(df.index.max()).date()
    rows, cols = df.shape
    freq = getattr(meta, "frequency_label", None)
    if isinstance(meta, dict):
        freq = meta.get("frequency_label") or meta.get("frequency")
    freq_part = f" • {freq} frequency" if freq else ""
    return f"{rows} rows × {cols} columns • {start} → {end}{freq_part}"


def _render_validation(meta: SchemaMeta | dict[str, Any]) -> None:
    validation = getattr(meta, "validation", None)
    if validation is None and isinstance(meta, dict):
        validation = meta.get("validation")
    if not validation:
        return
    issues = validation.get("issues") if isinstance(validation, dict) else None
    warnings = validation.get("warnings") if isinstance(validation, dict) else None
    if issues:
        st.error("Data quality checks flagged issues:")
        for issue in issues:
            st.write(f"• {issue}")
    if warnings:
        st.warning("Warnings:")
        for warning in warnings:
            st.write(f"• {warning}")


def _store_dataset(
    df: pd.DataFrame, meta: SchemaMeta | dict[str, Any], key: str
) -> None:
    app_state.store_validated_data(df, meta)
    st.session_state["data_loaded_key"] = key
    st.session_state["data_fingerprint"] = data_cache.cache_key_for_frame(df)
    st.session_state["data_summary"] = _dataset_summary(df, meta)

    analysis_runner.clear_cached_analysis()
    app_state.clear_analysis_results()

    candidates = infer_benchmarks(list(df.columns))
    st.session_state["benchmark_candidates"] = candidates
    if candidates:
        current = st.session_state.get("selected_benchmark")
        if current not in candidates:
            st.session_state["selected_benchmark"] = candidates[0]
    else:
        st.session_state["selected_benchmark"] = None


def _handle_failure(error: Exception) -> None:
    issues: list[str] | None = None
    detail: str | None = None
    message = "We couldn't process the file. Please confirm the format and try again."
    if isinstance(error, MarketDataValidationError):
        message = error.user_message
        issues = list(error.issues)
    else:
        detail = str(error).strip() or None

    app_state.record_upload_error(message, issues, detail=detail)
    st.error(message)
    if issues:
        for issue in issues:
            st.write(f"• {issue}")
    if detail and not issues:
        st.caption(detail)


def _load_sample_dataset(label: str, path: Path) -> None:
    try:
        df, meta = data_cache.load_dataset_from_path(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        _handle_failure(exc)
        return

    key = f"sample::{path.resolve()}"
    _store_dataset(df, meta, key)
    st.success(f"Loaded sample dataset “{label}”.")
    st.caption(_dataset_summary(df, meta))
    _render_validation(meta)
    st.dataframe(df.head(20))


def _load_uploaded_dataset(uploaded) -> None:
    data = uploaded.getvalue()
    try:
        df, meta = data_cache.load_dataset_from_bytes(data, uploaded.name)
    except Exception as exc:
        _handle_failure(exc)
        return

    key = f"upload::{uploaded.name}::{data_cache.cache_key_for_frame(df)}"
    _store_dataset(df, meta, key)
    st.success(f"Loaded {uploaded.name}.")
    st.caption(_dataset_summary(df, meta))
    _render_validation(meta)
    st.dataframe(df.head(20))


def _maybe_autoload_sample() -> None:
    if app_state.has_valid_upload():
        return
    default = data_cache.default_sample_dataset()
    if not default:
        return
    _load_sample_dataset(default.label, default.path)


def render_data_page() -> None:
    app_state.initialize_session_state()
    _maybe_autoload_sample()
    st.title("Data")
    st.write("Upload a CSV or Excel file, or start from the bundled sample dataset.")

    samples = data_cache.dataset_choices()
    options: list[str] = []
    if samples:
        options.append("Sample dataset")
    options.append("Upload your own")

    default_index = (
        0
        if st.session_state.get("data_source", "Sample dataset") == "Sample dataset"
        else 1
    )
    source = st.radio("Data source", options, index=default_index)
    st.session_state["data_source"] = source

    if source == "Sample dataset" and samples:
        labels = list(samples.keys())
        default_label = st.session_state.get("selected_sample_label") or labels[0]
        try:
            default_idx = labels.index(default_label)
        except ValueError:
            default_idx = 0
        label = st.selectbox("Choose a sample", labels, index=default_idx)
        st.session_state["selected_sample_label"] = label
        dataset = samples[label]
        key = f"sample::{dataset.path.resolve()}"
        if st.session_state.get("data_loaded_key") != key:
            _load_sample_dataset(label, dataset.path)
        else:
            df, meta = app_state.get_uploaded_data()
            if df is not None and meta is not None:
                st.info(st.session_state.get("data_summary", "Sample dataset loaded."))
                _render_validation(meta)
                st.dataframe(df.head(20))
    else:
        uploaded = st.file_uploader(
            "Upload returns (CSV or Excel with a Date column)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            _load_uploaded_dataset(uploaded)
        elif not app_state.has_valid_upload():
            st.info(
                "No dataset loaded yet. Switch to the sample tab for a quick start."
            )

    if app_state.has_valid_upload():
        df, meta = app_state.get_uploaded_data()
        if df is not None:
            candidates = st.session_state.get("benchmark_candidates", [])
            if candidates:
                default_bench = st.session_state.get(
                    "selected_benchmark", candidates[0]
                )
                bench = st.selectbox(
                    "Benchmark column (optional)",
                    ["None"] + candidates,
                    index=(
                        candidates.index(default_bench) + 1
                        if default_bench in candidates
                        else 0
                    ),
                )
                st.session_state["selected_benchmark"] = (
                    None if bench == "None" else bench
                )
            else:
                st.caption("No benchmark columns detected.")


render_data_page()
