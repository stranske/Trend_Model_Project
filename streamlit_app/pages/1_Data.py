"""Data page: upload or load sample datasets with validation summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from app.streamlit import state as app_state
from streamlit_app.components import data_cache
from streamlit_app.components.analysis_runner import clear_preprocessing_caches
from streamlit_app.components.demo_runner import DEMO_DATA_CANDIDATES
from trend_analysis.io.market_data import MarketDataValidationError
from trend_portfolio_app.data_schema import infer_benchmarks


def _dataset_signature(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _describe_meta(meta: dict[str, object] | None) -> tuple[str, list[str]]:
    if not isinstance(meta, dict):
        return "", []
    metadata = meta.get("metadata")
    if metadata is None:
        return "", []
    summary = []
    try:
        rows = int(getattr(metadata, "rows", meta.get("n_rows", 0)))
        cols = int(len(getattr(metadata, "columns", [])) or meta.get("n_cols", 0))
        start = getattr(metadata, "start", None)
        end = getattr(metadata, "end", None)
        freq = getattr(metadata, "frequency_label", meta.get("frequency"))
        summary.append(f"Rows: {rows:,}")
        summary.append(f"Columns: {cols:,}")
        if start and end:
            summary.append(f"Range: {pd.to_datetime(start).date()} → {pd.to_datetime(end).date()}")
        if freq:
            summary.append(f"Frequency: {freq}")
    except Exception:  # pragma: no cover - defensive formatting
        summary = []
    issues = []
    validation = meta.get("validation")
    if isinstance(validation, dict):
        for category in ("issues", "warnings"):
            vals = validation.get(category, [])
            if isinstance(vals, Iterable):
                issues.extend(str(item) for item in vals)
    return " | ".join(summary), issues


def _reset_downstream_state() -> None:
    for key in [
        "model_settings",
        "model_column_mapping",
        "sim_results",
        "analysis_window",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    clear_preprocessing_caches()
    data_cache.clear_dataset_cache()


def _handle_failure(error: Exception) -> None:
    message = "Unable to load dataset."
    details: list[str] = []
    if isinstance(error, MarketDataValidationError):
        message = error.user_message
        details = list(error.issues)
    else:
        cause = getattr(error, "__cause__", None)
        if isinstance(cause, MarketDataValidationError):
            message = cause.user_message
            details = list(cause.issues)
        else:
            message = str(error)
    app_state.record_upload_error(message, details)
    st.error(message)
    if details:
        with st.expander("Validation details", expanded=False):
            for issue in details:
                st.write("•", issue)


def _finalise_dataset(df: pd.DataFrame, meta: dict[str, object], *, label: str) -> None:
    _reset_downstream_state()
    app_state.store_validated_data(df, meta)
    st.session_state["data_signature"] = _dataset_signature(df)
    st.session_state["data_label"] = label
    app_state.initialize_session_state()
    benchmark_candidates = infer_benchmarks(list(df.columns))
    st.session_state["benchmark_candidates"] = benchmark_candidates
    if benchmark_candidates:
        st.session_state["selected_benchmark"] = benchmark_candidates[0]
    else:
        st.session_state.pop("selected_benchmark", None)
    st.session_state["model_column_mapping"] = {
        "date_column": df.index.name or "Date",
        "return_columns": [
            col for col in df.columns if col != st.session_state.get("selected_benchmark")
        ],
        "benchmark_column": st.session_state.get("selected_benchmark"),
    }
    st.success("Dataset ready. Continue to the Model step to configure the analysis.")


def _load_sample_dataset(path: Path) -> None:
    try:
        df, meta = data_cache.load_dataset_from_path(str(path))
    except Exception as exc:
        _handle_failure(exc)
        return
    _finalise_dataset(df, meta, label=path.name)


def _load_uploaded_file(uploaded) -> None:
    try:
        content = uploaded.getvalue()
        df, meta = data_cache.load_dataset_from_bytes(content)
    except Exception as exc:
        _handle_failure(exc)
        return
    _finalise_dataset(df, meta, label=uploaded.name)


def _render_validation_summary(meta: dict[str, object] | None) -> None:
    summary, issues = _describe_meta(meta or {})
    if summary:
        st.info(summary)
    validation = (meta or {}).get("validation")
    if isinstance(validation, dict):
        warnings = validation.get("warnings") or []
        issues = validation.get("issues") or []
        if issues:
            st.error("Data quality issues detected:")
            for msg in issues:
                st.write("•", msg)
        elif warnings:
            st.warning("Warnings:")
            for msg in warnings:
                st.write("•", msg)


def render_page(st_module: Any | None = None) -> None:
    if st_module is None:
        st_module = st
    app_state.initialize_session_state()
    st_module.title("📂 Data")
    st_module.caption(
        "Upload your own dataset or explore the bundled sample file. "
        "Validated data is cached so re-runs are instantaneous, and caches are automatically reset when a new file is chosen."
    )

    sample_files = [path for path in DEMO_DATA_CANDIDATES if Path(path).exists()]
    if sample_files:
        with st_module.expander("Sample dataset", expanded=True):
            labels = {path.name: path for path in sample_files}
            choice = st_module.selectbox("Select sample", options=list(labels.keys()))
            if st_module.button("Load sample dataset", type="secondary"):
                _load_sample_dataset(labels[choice])

    uploaded = st_module.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], help="Requires a Date column."
    )
    if uploaded is not None:
        _load_uploaded_file(uploaded)

    df = st_module.session_state.get("returns_df")
    meta = st_module.session_state.get("schema_meta")
    if df is None or meta is None:
        st_module.info("No dataset loaded yet.")
        return

    _render_validation_summary(meta)

    st_module.subheader("Preview")
    st_module.dataframe(df.head(20))

    candidates = st_module.session_state.get("benchmark_candidates", [])
    if candidates:
        current = st_module.session_state.get("selected_benchmark") or candidates[0]
        selection = st_module.selectbox(
            "Benchmark column", options=candidates, index=candidates.index(current) if current in candidates else 0
        )
        if selection != st_module.session_state.get("selected_benchmark"):
            st_module.session_state["selected_benchmark"] = selection
            mapping = st_module.session_state.get("model_column_mapping", {})
            if mapping:
                mapping["benchmark_column"] = selection
                mapping["return_columns"] = [
                    col for col in df.columns if col != selection
                ]
                st_module.session_state["model_column_mapping"] = mapping
            clear_preprocessing_caches()
            st_module.session_state.pop("sim_results", None)
            st_module.info("Benchmark updated. Model inputs will refresh on the next step.")

    st_module.caption("Data cached using `st.cache_data`; refresh the browser to clear session state entirely if needed.")


render_page()
