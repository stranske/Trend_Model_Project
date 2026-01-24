"""Session state management for the Streamlit app."""

from __future__ import annotations

import json
import math
import numbers
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    from types import SimpleNamespace

    st = SimpleNamespace(session_state={})

_DEFAULT_STATE: dict[str, Any] = {
    "returns_df": None,
    "schema_meta": None,
    "benchmark_candidates": [],
    "validation_report": None,
    "upload_status": "pending",  # pending, success, error
    "data_hash": None,
    "data_saved_path": None,
    "saved_model_states": {},
    "saved_config_wrappers": {},
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


def get_saved_config_wrappers() -> dict[str, dict[str, Any]]:
    """Return the mapping of saved config wrappers stored in session state."""

    saved = st.session_state.get("saved_config_wrappers")
    if not isinstance(saved, dict):
        saved = {}
        st.session_state["saved_config_wrappers"] = saved
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


def save_config_wrapper(name: str, wrapper: Mapping[str, Any]) -> None:
    """Persist a full config wrapper under the provided name."""

    if not name or not name.strip():
        raise ValueError("A non-empty name is required to save a model configuration.")

    saved = get_saved_config_wrappers()
    saved[name.strip()] = deepcopy(dict(wrapper))


def load_saved_model_state(name: str) -> dict[str, Any]:
    """Load a saved model configuration by name.

    Raises:
        KeyError: If the requested configuration name does not exist.
    """

    saved = get_saved_model_states()
    if name not in saved:
        raise KeyError(f"No saved model configuration named '{name}'.")
    return deepcopy(saved[name])


def load_saved_config_wrapper(name: str) -> dict[str, Any] | None:
    """Load a saved config wrapper by name if available."""

    saved = get_saved_config_wrappers()
    if name not in saved:
        return None
    return deepcopy(saved[name])


def rename_saved_model_state(current_name: str, new_name: str) -> None:
    """Rename a saved model configuration while preserving its payload.

    Raises:
        KeyError: If ``current_name`` does not exist.
        ValueError: If ``new_name`` is empty/whitespace-only or already exists.
    """

    saved = get_saved_model_states()
    wrappers = get_saved_config_wrappers()
    if current_name not in saved:
        raise KeyError(f"No saved model configuration named '{current_name}'.")

    stripped_new_name = new_name.strip() if new_name else ""
    if not stripped_new_name:
        raise ValueError("Provide a new name to rename the configuration.")
    if stripped_new_name in saved and stripped_new_name != current_name:
        raise ValueError(f"A configuration named '{stripped_new_name}' already exists.")

    saved[stripped_new_name] = saved.pop(current_name)
    if current_name in wrappers:
        wrappers[stripped_new_name] = wrappers.pop(current_name)


def delete_saved_model_state(name: str) -> None:
    """Remove a saved model configuration if it exists."""

    get_saved_model_states().pop(name, None)
    get_saved_config_wrappers().pop(name, None)


def export_model_state(name: str) -> str:
    """Serialize the saved configuration to JSON.

    Raises:
        KeyError: If the configuration name does not exist.
    """

    wrapper = load_saved_config_wrapper(name)
    if isinstance(wrapper, Mapping):
        return json.dumps(wrapper, sort_keys=True)
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
    payload = payload.lstrip("\ufeff").strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid JSON payload for configuration import.") from exc

    if not isinstance(parsed, Mapping):
        raise ValueError("Imported configuration must be a JSON object.")

    model_state = parsed
    wrapper: dict[str, Any] | None = None
    if isinstance(parsed.get("model_state"), Mapping):
        model_state = parsed.get("model_state", {})
        wrapper = dict(parsed)

    save_model_state(name.strip(), model_state)
    if wrapper is not None:
        save_config_wrapper(name.strip(), wrapper)
    return load_saved_model_state(name.strip())


@dataclass(frozen=True)
class ModelStateDiff:
    """Represents a single difference between two model states."""

    path: str
    left: Any
    right: Any
    change_type: Literal["added", "removed", "changed"]
    type_changed: bool = False


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _stringify_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _values_equal(left: Any, right: Any, float_tol: float) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, numbers.Number) and isinstance(right, numbers.Number):
        return math.isclose(left, right, rel_tol=float_tol, abs_tol=float_tol)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(
            _values_equal(left[key], right[key], float_tol) for key in left.keys()
        )
    if _is_sequence(left) and _is_sequence(right):
        if len(left) != len(right):
            return False
        return all(
            _values_equal(item_left, item_right, float_tol)
            for item_left, item_right in zip(left, right)
        )
    return left == right


def diff_model_states(
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    *,
    float_tol: float = 1e-9,
) -> list[ModelStateDiff]:
    """Compute a deterministic, recursive diff between two model states."""

    diffs: list[ModelStateDiff] = []

    def _walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = sorted(set(left.keys()) | set(right.keys()))
            for key in keys:
                next_path = f"{path}.{key}" if path else str(key)
                in_left = key in left
                in_right = key in right
                if not in_left:
                    diffs.append(
                        ModelStateDiff(
                            path=next_path,
                            left=None,
                            right=deepcopy(right[key]),
                            change_type="added",
                        )
                    )
                elif not in_right:
                    diffs.append(
                        ModelStateDiff(
                            path=next_path,
                            left=deepcopy(left[key]),
                            right=None,
                            change_type="removed",
                        )
                    )
                else:
                    _walk(left[key], right[key], next_path)
            return

        if _is_sequence(left) and _is_sequence(right):
            if type(left) is not type(right):
                diffs.append(
                    ModelStateDiff(
                        path=path or "<root>",
                        left=deepcopy(left),
                        right=deepcopy(right),
                        change_type="changed",
                        type_changed=True,
                    )
                )
                return
            for idx, (item_left, item_right) in enumerate(
                zip_longest(left, right, fillvalue=_missing_sentinel)
            ):
                next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                if item_left is _missing_sentinel:
                    diffs.append(
                        ModelStateDiff(
                            path=next_path,
                            left=None,
                            right=deepcopy(item_right),
                            change_type="added",
                        )
                    )
                elif item_right is _missing_sentinel:
                    diffs.append(
                        ModelStateDiff(
                            path=next_path,
                            left=deepcopy(item_left),
                            right=None,
                            change_type="removed",
                        )
                    )
                else:
                    _walk(item_left, item_right, next_path)
            return

        if _values_equal(left, right, float_tol):
            return

        diffs.append(
            ModelStateDiff(
                path=path or "<root>",
                left=deepcopy(left),
                right=deepcopy(right),
                change_type="changed",
                type_changed=type(left) is not type(right),
            )
        )

    _missing_sentinel = object()
    from itertools import zip_longest

    _walk(dict(config_a), dict(config_b), "")
    return diffs


def format_model_state_diff(
    diffs: Sequence[ModelStateDiff],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """Create a copy-friendly text representation of model state differences."""

    if not diffs:
        return "No differences found."

    lines: list[str] = []
    for entry in diffs:
        left = _stringify_value(entry.left)
        right = _stringify_value(entry.right)
        if entry.change_type == "added":
            lines.append(f"+ {entry.path}: ({label_b}) {right}")
        elif entry.change_type == "removed":
            lines.append(f"- {entry.path}: ({label_a}) {left}")
        else:
            type_note = " [type changed]" if entry.type_changed else ""
            lines.append(
                f"~ {entry.path}: ({label_a}) {left} -> ({label_b}) {right}{type_note}"
            )
    return "\n".join(lines)
