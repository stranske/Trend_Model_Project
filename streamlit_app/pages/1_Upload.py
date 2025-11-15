"""Compatibility shim preserving the legacy Upload page behaviour for tests."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import streamlit as st

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner
from streamlit_app.components.upload_guard import (
    UploadViolation,
    guard_and_buffer_upload,
    hash_path,
)
from trend_analysis.io.market_data import MarketDataValidationError
from trend_portfolio_app.data_schema import infer_benchmarks, load_and_validate_file


def _dataset_summary(df) -> str:
    start = df.index.min().date() if hasattr(df.index.min(), "date") else df.index.min()
    end = df.index.max().date() if hasattr(df.index.max(), "date") else df.index.max()
    return f"{df.shape[0]} rows × {df.shape[1]} columns. Range: {start} to {end}."


def _handle_success(
    st_module,
    df,
    meta,
    *,
    source_path: Path,
    data_hash: str,
    data_key: str,
) -> None:
    app_state.store_validated_data(
        df, meta, data_hash=data_hash, saved_path=source_path
    )
    st_module.session_state["uploaded_file_path"] = str(source_path)
    st_module.session_state["data_fingerprint"] = data_hash
    st_module.session_state["data_loaded_key"] = data_key
    st_module.session_state["data_summary"] = _dataset_summary(df)

    analysis_runner.clear_cached_analysis()

    st_module.success(_dataset_summary(df))
    meta_info = meta.get("metadata") if isinstance(meta, dict) else None
    if meta_info:
        st_module.info(
            f"Detected {meta_info.mode.value} data at {meta_info.frequency_label} cadence."
        )
    st_module.dataframe(df.head(12))
    candidates = infer_benchmarks(list(df.columns))
    st_module.session_state["benchmark_candidates"] = candidates
    if candidates:
        st_module.info("Possible benchmark columns: " + ", ".join(candidates))


_GENERIC_FAILURE_MESSAGE = (
    "We couldn't process the file. Please confirm the format and try again."
)


def _extract_validation_details(error: Exception) -> tuple[str, list[str], str | None]:
    if isinstance(error, UploadViolation):
        return str(error), [], None
    if isinstance(error, MarketDataValidationError):
        return error.user_message, list(error.issues), None
    cause = getattr(error, "__cause__", None)
    if isinstance(cause, MarketDataValidationError):
        return cause.user_message, list(cause.issues), str(error).strip() or None
    return _GENERIC_FAILURE_MESSAGE, [], str(error).strip() or None


def _handle_failure(st_module, error: Exception) -> None:
    message, issues, detail = _extract_validation_details(error)
    app_state.record_upload_error(message, issues, detail=detail)
    st_module.error(message)
    for issue in issues:
        st_module.write(f"• {issue}")
    if detail and not issues:
        st_module.caption(detail)


@lru_cache(maxsize=2)
def _load_demo(path: str):
    with open(path, "rb") as handle:
        return load_and_validate_file(handle)


def render_upload_page(st_module=st) -> None:
    app_state.initialize_session_state()
    st_module.title("Upload")

    uploaded = st_module.file_uploader(
        "Upload returns (CSV or Excel). Requires a 'Date' column.",
        type=["csv", "xlsx", "xls"],
    )

    demo_path_csv = Path("demo/demo_returns.csv")
    demo_path_xlsx = Path("demo/demo_returns.xlsx")
    if demo_path_csv.exists() or demo_path_xlsx.exists():
        if st_module.button("Load demo data"):
            path = demo_path_csv if demo_path_csv.exists() else demo_path_xlsx
            try:
                df, meta = _load_demo(str(path))
                data_hash = hash_path(path.resolve())
            except Exception as exc:  # pragma: no cover - defensive guard
                _handle_failure(st_module, exc)
            else:
                resolved = path.resolve()
                data_key = f"sample::{resolved}"
                _handle_success(
                    st_module,
                    df,
                    meta,
                    source_path=resolved,
                    data_hash=data_hash,
                    data_key=data_key,
                )

    if uploaded is not None:
        try:
            guarded = guard_and_buffer_upload(uploaded)
            buffer = io.BytesIO(guarded.data)
            buffer.name = guarded.original_name
            df, meta = load_and_validate_file(buffer)
        except (UploadViolation, MarketDataValidationError, ValueError) as exc:
            _handle_failure(st_module, exc)
        except Exception as exc:  # pragma: no cover - unexpected parsing error
            _handle_failure(st_module, exc)
        else:
            data_key = f"upload::{guarded.original_name}::{guarded.content_hash}"
            _handle_success(
                st_module,
                df,
                meta,
                source_path=guarded.stored_path,
                data_hash=guarded.content_hash,
                data_key=data_key,
            )
    elif not app_state.has_valid_upload():
        st_module.warning("No file uploaded yet.")


render_upload_page()
