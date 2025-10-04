import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

"""Data entry page for the Streamlit front door experience."""

from __future__ import annotations

import io
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from app.streamlit import state as app_state
from trend_analysis.io.market_data import MarketDataValidationError
from trend_portfolio_app.data_schema import infer_benchmarks, load_and_validate_file


SAMPLE_PATHS: tuple[Path, ...] = (
    Path("demo/demo_returns.parquet"),
    Path("demo/demo_returns.csv"),
    Path("demo/demo_returns.xlsx"),
)


@st.cache_data(show_spinner=False)
def _parse_buffer(data: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Parse uploaded bytes into a validated frame."""

    buffer = io.BytesIO(data)
    buffer.name = filename  # Helps pandas infer file type for Excel readers
    return load_and_validate_file(buffer)


@st.cache_data(show_spinner=False)
def _read_sample(path: str) -> tuple[pd.DataFrame, dict]:
    """Read and validate the built-in sample dataset."""

    with open(path, "rb") as handle:
        data = handle.read()
    return _parse_buffer(data, Path(path).name)


def _persist_uploaded_copy(df: pd.DataFrame) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "trend_app_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}.csv"
    reset = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    reset.to_csv(tmp_path, index=False)
    return tmp_path


def _extract_validation_details(error: Exception) -> tuple[str, list[str]]:
    if isinstance(error, MarketDataValidationError):
        return error.user_message, list(error.issues)
    cause = getattr(error, "__cause__", None)
    if isinstance(cause, MarketDataValidationError):
        return cause.user_message, list(cause.issues)
    return str(error), []


def _render_validation_feedback(message: str, issues: Iterable[str]) -> None:
    with st.container(border=True):
        st.error(message)
        for issue in issues:
            st.markdown(f"- {issue}")


def _handle_failure(error: Exception) -> None:
    message, issues = _extract_validation_details(error)
    app_state.record_upload_error(message, issues)
    _render_validation_feedback(message, issues)


def _handle_success(df: pd.DataFrame, meta: dict, *, source_path: Path | str, source: str) -> None:
    app_state.store_validated_data(df, meta, source=source)
    st.session_state["uploaded_file_path"] = str(source_path)

    start = df.index.min()
    end = df.index.max()
    start_label = start.date() if hasattr(start, "date") else start
    end_label = end.date() if hasattr(end, "date") else end

    meta_info = meta.get("metadata") if isinstance(meta, dict) else None
    cadence = None
    if meta_info:
        cadence = f"{meta_info.mode.value.title()} · {meta_info.frequency_label}"

    with st.status("Data ready", state="complete"):
        st.markdown(
            f"**Source:** {'Built-in sample' if source == 'sample' else 'Upload'}"
        )
        st.markdown(
            f"**Rows × Columns:** {df.shape[0]} × {df.shape[1]}  \
**Range:** {start_label} → {end_label}"
        )
        if cadence:
            st.markdown(f"**Detected cadence:** {cadence}")

    preview = df.head(12)
    st.dataframe(preview, use_container_width=True)

    candidates = infer_benchmarks(list(df.columns))
    st.session_state["benchmark_candidates"] = candidates
    if candidates:
        st.info("Suggested benchmark columns: " + ", ".join(candidates))


@lru_cache(maxsize=1)
def _default_sample_path() -> Path | None:
    for path in SAMPLE_PATHS:
        if path.exists():
            return path
    return None


def _auto_bootstrap_sample() -> None:
    if app_state.has_valid_upload():
        return
    if st.session_state.get("sample_autoload_complete"):
        return
    sample_path = _default_sample_path()
    if not sample_path:
        st.warning("Provide a CSV or Excel file to begin.")
        st.session_state["sample_autoload_complete"] = True
        return
    try:
        df, meta = _read_sample(str(sample_path))
    except Exception as exc:  # pragma: no cover - defensive
        _handle_failure(exc)
    else:
        _handle_success(df, meta, source_path=sample_path, source="sample")
        st.toast("Loaded sample dataset. Upload your own file to replace it.")
    finally:
        st.session_state["sample_autoload_complete"] = True


def _render_sample_controls(sample_path: Path | None) -> None:
    st.markdown("#### Built-in sample dataset")
    if not sample_path:
        st.info("No bundled dataset available in this build.")
        return
    st.caption(
        "We include a multi-decade manager returns sample so you can explore the "
        "workflow instantly."
    )
    if st.button("Reload sample dataset", key="btn_reload_sample"):
        try:
            df, meta = _read_sample(str(sample_path))
        except Exception as exc:  # pragma: no cover - defensive
            _handle_failure(exc)
        else:
            _handle_success(df, meta, source_path=sample_path, source="sample")


def _render_upload_controls() -> None:
    st.markdown("#### Upload your own data")
    uploaded = st.file_uploader(
        "CSV or Excel with a 'Date' column",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is None:
        return

    try:
        payload = uploaded.read()
        df, meta = _parse_buffer(payload, uploaded.name)
    except (MarketDataValidationError, ValueError) as exc:
        _handle_failure(exc)
        return
    except Exception as exc:  # pragma: no cover - unexpected parsing error
        _handle_failure(exc)
        return

    tmp_path = _persist_uploaded_copy(df)
    _handle_success(df, meta, source_path=tmp_path, source="upload")


def render_data_page() -> None:
    app_state.initialize_session_state()
    st.title("Data")
    st.write(
        "Load returns data to drive the simulation. Start with the built-in sample "
        "or upload a spreadsheet – either way, validation feedback keeps you on track."
    )

    _auto_bootstrap_sample()

    sample_path = _default_sample_path()

    col_sample, col_upload = st.columns(2)
    with col_sample:
        _render_sample_controls(sample_path)
    with col_upload:
        _render_upload_controls()

    if app_state.has_valid_upload():
        st.success("Data checks passed – continue to the Model step.")
    else:
        st.info("Need help? Start with the sample dataset, then revisit with your data.")


render_data_page()
