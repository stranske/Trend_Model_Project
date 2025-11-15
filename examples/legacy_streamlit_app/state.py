"""Session state management for Streamlit app."""

from typing import Optional, Sequence

import pandas as pd
import streamlit as st


def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        "returns_df": None,
        "schema_meta": None,
        "benchmark_candidates": [],
        "validation_report": None,
        "upload_status": "pending",  # pending, success, error
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_upload_data():
    """Clear uploaded data from session state."""
    keys_to_clear = [
        "returns_df",
        "schema_meta",
        "benchmark_candidates",
        "validation_report",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["upload_status"] = "pending"
    clear_analysis_results()


def clear_analysis_results() -> None:
    """Remove any cached analysis outputs from session state."""

    for key in ("analysis_result", "analysis_result_key", "analysis_error"):
        st.session_state.pop(key, None)


def store_validated_data(df: pd.DataFrame, meta: dict):
    """Store validated data in session state."""
    st.session_state["returns_df"] = df
    st.session_state["schema_meta"] = meta
    st.session_state["validation_report"] = meta.get("validation")
    st.session_state["upload_status"] = "success"
    clear_analysis_results()


def record_upload_error(
    message: str,
    issues: Sequence[str] | None = None,
    *,
    detail: str | None = None,
) -> None:
    """Persist an upload failure and clear any stale data."""

    st.session_state["returns_df"] = None
    st.session_state["schema_meta"] = None
    st.session_state["benchmark_candidates"] = []
    report = {
        "message": message,
        "issues": list(issues or []),
    }
    if detail:
        report["detail"] = detail
    st.session_state["validation_report"] = report
    st.session_state["upload_status"] = "error"
    clear_analysis_results()


def get_uploaded_data() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Retrieve uploaded data from session state."""
    return (st.session_state.get("returns_df"), st.session_state.get("schema_meta"))


def has_valid_upload() -> bool:
    """Check if there's valid uploaded data in session state."""
    df, meta = get_uploaded_data()
    return (
        df is not None
        and meta is not None
        and st.session_state.get("upload_status") == "success"
    )


def get_upload_summary() -> str:
    """Get a summary of the uploaded data."""
    df, meta = get_uploaded_data()
    if df is None or meta is None:
        return "No data uploaded"

    summary_parts = [
        f"{df.shape[0]} rows × {df.shape[1]} columns",
        f"Range: {df.index.min().date()} to {df.index.max().date()}",
    ]

    if "frequency" in meta:
        summary_parts.append(f"Frequency: {meta['frequency']}")

    return " | ".join(summary_parts)
