"""Session state management for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import streamlit as st

_DEFAULT_STATE: dict[str, Any] = {
    "returns_df": None,
    "schema_meta": None,
    "benchmark_candidates": [],
    "validation_report": None,
    "upload_status": "pending",  # pending, success, error
    "data_hash": None,
    "data_saved_path": None,
}


def initialize_session_state() -> None:
    """Ensure expected session state keys exist."""

    for key, default_value in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_analysis_results() -> None:
    """Remove any cached analysis outputs from session state."""

    for key in ("analysis_result", "analysis_result_key", "analysis_error"):
        st.session_state.pop(key, None)


def clear_upload_data() -> None:
    """Clear uploaded data from session state."""

    for key in (
        "returns_df",
        "schema_meta",
        "benchmark_candidates",
        "validation_report",
        "data_hash",
        "data_saved_path",
        "data_loaded_key",
        "data_fingerprint",
        "data_summary",
        "uploaded_file_path",
    ):
        st.session_state.pop(key, None)
    st.session_state["upload_status"] = "pending"
    clear_analysis_results()


def store_validated_data(
    df: pd.DataFrame,
    meta: dict[str, Any] | Any,
    *,
    data_hash: str | None = None,
    saved_path: Path | None = None,
) -> None:
    """Store validated data in session state."""

    st.session_state["returns_df"] = df
    st.session_state["schema_meta"] = meta
    report = meta.get("validation") if isinstance(meta, dict) else None
    st.session_state["validation_report"] = report
    st.session_state["upload_status"] = "success"
    st.session_state["data_hash"] = data_hash
    st.session_state["data_saved_path"] = str(saved_path) if saved_path else None
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
    st.session_state["data_hash"] = None
    st.session_state["data_saved_path"] = None
    st.session_state.pop("data_loaded_key", None)
    st.session_state.pop("data_fingerprint", None)
    st.session_state.pop("data_summary", None)
    st.session_state.pop("uploaded_file_path", None)
    report = {
        "message": message,
        "issues": list(issues or []),
    }
    if detail:
        report["detail"] = detail
    st.session_state["validation_report"] = report
    st.session_state["upload_status"] = "error"
    clear_analysis_results()


def get_uploaded_data() -> tuple[Optional[pd.DataFrame], Optional[dict[str, Any]]]:
    """Retrieve uploaded data from session state."""

    df = st.session_state.get("returns_df")
    meta = st.session_state.get("schema_meta")
    return df, meta


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

    if isinstance(meta, dict) and "frequency" in meta:
        summary_parts.append(f"Frequency: {meta['frequency']}")

    return " | ".join(summary_parts)
