"""Session state management for the Streamlit app."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    "saved_model_states": {},
}


def initialize_session_state() -> None:
    """Ensure expected session state keys exist."""

    for key, default_value in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(default_value)


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


def get_saved_model_states() -> dict[str, dict[str, Any]]:
    """Return the mapping of saved model states stored in session state."""

    saved = st.session_state.get("saved_model_states")
    if not isinstance(saved, dict):
        saved = {}
        st.session_state["saved_model_states"] = saved
    return saved


def save_model_state(name: str, model_state: Mapping[str, Any]) -> None:
    """Persist a model configuration under the provided name.

    Raises:
        ValueError: If the provided name is empty or whitespace-only.
    """

    if not name or not name.strip():
        raise ValueError("A non-empty name is required to save a model configuration.")

    saved = get_saved_model_states()
    saved[name.strip()] = deepcopy(dict(model_state))


def load_saved_model_state(name: str) -> dict[str, Any]:
    """Load a saved model configuration by name.

    Raises:
        KeyError: If the requested configuration name does not exist.
    """

    saved = get_saved_model_states()
    if name not in saved:
        raise KeyError(f"No saved model configuration named '{name}'.")
    return deepcopy(saved[name])


def rename_saved_model_state(current_name: str, new_name: str) -> None:
    """Rename a saved model configuration while preserving its payload.

    Raises:
        KeyError: If ``current_name`` does not exist.
        ValueError: If ``new_name`` is empty/whitespace-only or already exists.
    """

    saved = get_saved_model_states()
    if current_name not in saved:
        raise KeyError(f"No saved model configuration named '{current_name}'.")

    stripped_new_name = new_name.strip() if new_name else ""
    if not stripped_new_name:
        raise ValueError("Provide a new name to rename the configuration.")
    if stripped_new_name in saved and stripped_new_name != current_name:
        raise ValueError(f"A configuration named '{stripped_new_name}' already exists.")

    saved[stripped_new_name] = saved.pop(current_name)


def delete_saved_model_state(name: str) -> None:
    """Remove a saved model configuration if it exists."""

    get_saved_model_states().pop(name, None)


def export_model_state(name: str) -> str:
    """Serialize the saved configuration to JSON.

    Raises:
        KeyError: If the configuration name does not exist.
    """

    payload = load_saved_model_state(name)
    return json.dumps(payload, sort_keys=True)


def import_model_state(name: str, payload: str) -> dict[str, Any]:
    """Load a configuration from JSON and store it under the provided name.

    Raises:
        ValueError: If the name is empty/whitespace-only, the payload is invalid JSON,
            or the parsed payload is not a JSON object.
    """

    if not name or not name.strip():
        raise ValueError("Provide a name for the imported configuration.")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid JSON payload for configuration import.") from exc

    if not isinstance(parsed, Mapping):
        raise ValueError("Imported configuration must be a JSON object.")

    save_model_state(name.strip(), parsed)
    return load_saved_model_state(name.strip())


def diff_model_states(
    state_a: Mapping[str, Any], state_b: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Compare two model state dictionaries and return differences.

    Returns a list of (key, value_a, value_b) tuples for keys that differ.
    Keys present in only one state are included with None for the missing value.
    """
    all_keys = set(state_a.keys()) | set(state_b.keys())
    diffs: list[tuple[str, Any, Any]] = []

    for key in sorted(all_keys):
        val_a = state_a.get(key)
        val_b = state_b.get(key)
        # Compare values - handle nested dicts specially
        if val_a != val_b:
            diffs.append((key, val_a, val_b))

    return diffs


def format_model_state_diff(
    diffs: list[tuple[str, Any, Any]],
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> str:
    """Format a list of differences into a human-readable string.

    Args:
        diffs: List of (key, value_a, value_b) tuples from diff_model_states.
        label_a: Label for the first configuration.
        label_b: Label for the second configuration.

    Returns:
        A formatted string showing the differences.
    """
    if not diffs:
        return "No differences found between configurations."

    lines = [f"Differences between {label_a} and {label_b}:", ""]
    for key, val_a, val_b in diffs:
        lines.append(f"  {key}:")
        lines.append(f"    {label_a}: {val_a}")
        lines.append(f"    {label_b}: {val_b}")
        lines.append("")

    return "\n".join(lines)
