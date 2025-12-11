"""Data page for the Streamlit application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner, data_cache
from streamlit_app.components.csv_validation import (
    CSVValidationError,
    DateCorrectionNeeded,
    validate_uploaded_csv,
)
from streamlit_app.components.date_correction import (
    apply_date_corrections,
    format_corrections_for_display,
)
from streamlit_app.components.upload_guard import (
    UploadViolation,
    guard_and_buffer_upload,
    hash_path,
)
from streamlit_app.components.data_schema import (
    SchemaMeta,
    infer_benchmarks,
    infer_risk_free_candidates,
)
from trend.input_validation import InputValidationError
from trend_analysis.io.market_data import MarketDataValidationError

DATE_COLUMN = "Date"
REQUIRED_UPLOAD_COLUMNS = (DATE_COLUMN,)
MAX_UPLOAD_ROWS = 50_000


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
    df: pd.DataFrame,
    meta: SchemaMeta | dict[str, Any],
    key: str,
    *,
    data_hash: str,
    saved_path: Path | None = None,
) -> None:
    # Check if this is the SAME dataset we already have loaded
    existing_hash = st.session_state.get("data_fingerprint")
    is_same_dataset = existing_hash == data_hash

    app_state.store_validated_data(df, meta, data_hash=data_hash, saved_path=saved_path)
    st.session_state["data_loaded_key"] = key
    st.session_state["data_fingerprint"] = data_hash
    st.session_state["data_summary"] = _dataset_summary(df, meta)

    analysis_runner.clear_cached_analysis()

    all_columns = list(df.columns)

    # Detect benchmark candidates
    bench_candidates = infer_benchmarks(all_columns)
    st.session_state["benchmark_candidates"] = bench_candidates

    # Set default benchmark - only if not set or invalid for this dataset
    current_bench = st.session_state.get("selected_benchmark")
    if current_bench not in all_columns:
        if bench_candidates:
            st.session_state["selected_benchmark"] = bench_candidates[0]
        else:
            st.session_state["selected_benchmark"] = None

    # Detect risk-free rate candidates
    rf_candidates = infer_risk_free_candidates(all_columns)
    st.session_state["risk_free_candidates"] = rf_candidates

    # Set default risk-free - only if not set or invalid for this dataset
    current_rf = st.session_state.get("selected_risk_free")
    if current_rf not in all_columns:
        if rf_candidates:
            st.session_state["selected_risk_free"] = rf_candidates[0]
        else:
            st.session_state["selected_risk_free"] = None

    # Store all fund columns
    st.session_state["all_fund_columns"] = all_columns

    # Only initialize fund selection for NEW datasets
    # Preserve existing selection if same dataset
    if not is_same_dataset:
        selected_rf = st.session_state.get("selected_risk_free")
        selected_bench = st.session_state.get("selected_benchmark")
        system_cols = {selected_rf, selected_bench, "Date"} - {None}
        fund_cols = [c for c in all_columns if c not in system_cols]
        st.session_state["selected_fund_columns"] = set(fund_cols)
        st.session_state["_editor_version"] = 0


def _handle_failure(error: Exception) -> None:
    issues: list[str] | None = None
    detail: str | None = None
    sample_preview: str | None = None
    date_correction: DateCorrectionNeeded | None = None
    message = "We couldn't process the file. Please confirm the format and try again."
    if isinstance(error, UploadViolation):
        message = str(error)
    elif isinstance(error, CSVValidationError):
        message = error.user_message
        issues = list(error.issues)
        sample_preview = error.sample_preview
        date_correction = error.date_correction
    elif isinstance(error, MarketDataValidationError):
        message = error.user_message
        issues = list(error.issues)
    elif isinstance(error, InputValidationError):
        message = error.user_message
        issues = list(error.issues)
    else:
        detail = str(error).strip() or None

    app_state.record_upload_error(message, issues, detail=detail)

    # Handle date corrections with user approval UI
    if date_correction is not None:
        _render_date_correction_ui(message, date_correction)
        return

    st.error(message)
    if issues:
        for issue in issues:
            st.write(f"• {issue}")
    if detail and not issues:
        st.caption(detail)
    if sample_preview:
        st.caption("Example format:")
        st.code(sample_preview, language="text")


def _render_date_correction_ui(message: str, correction: DateCorrectionNeeded) -> None:
    """Render UI for approving date corrections."""
    st.warning(message)

    # Show what corrections will be made
    has_date_fixes = len(correction.corrections) > 0
    has_trailing = len(correction.trailing_empty_rows) > 0
    has_droppable = len(correction.droppable_empty_rows) > 0

    if has_date_fixes or has_trailing or has_droppable:
        st.markdown("**The following issues can be automatically corrected:**")
        st.markdown(
            format_corrections_for_display(
                correction.corrections,
                trailing_rows=correction.trailing_empty_rows,
                droppable_rows=correction.droppable_empty_rows,
            )
        )

    if correction.unfixable:
        st.error(
            f"⚠️ {len(correction.unfixable)} date(s) cannot be automatically corrected:"
        )
        for idx, val in correction.unfixable[:5]:
            st.write(f"• Row {idx + 1}: `{val}`")
        if len(correction.unfixable) > 5:
            st.write(f"• ... and {len(correction.unfixable) - 5} more")
        st.info(
            "Please fix these dates manually in your file, or apply the available "
            "corrections and then re-upload with the remaining issues fixed."
        )

    # Store correction info in session state for the button callback
    st.session_state["pending_date_correction"] = correction

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "✅ Apply Corrections & Continue",
            type="primary",
            use_container_width=True,
            key="apply_date_corrections",
        ):
            _apply_pending_date_corrections()

    with col2:
        if st.button(
            "❌ Cancel",
            use_container_width=True,
            key="cancel_date_corrections",
        ):
            st.session_state.pop("pending_date_correction", None)
            st.info("Upload cancelled. Please fix the dates and try again.")


def _apply_pending_date_corrections() -> None:
    """Apply pending date corrections and reload the dataset."""
    import io

    correction = st.session_state.get("pending_date_correction")
    if correction is None:
        st.error("No pending corrections found.")
        return

    try:
        # Load the raw data
        is_excel = correction.original_name.lower().endswith((".xlsx", ".xls"))
        buffer = io.BytesIO(correction.raw_data)

        if is_excel:
            df = pd.read_excel(buffer)
        else:
            df = pd.read_csv(buffer)

        # Apply the corrections (including dropping all empty rows)
        df = apply_date_corrections(
            df,
            correction.date_column,
            correction.corrections,
            drop_rows=correction.all_rows_to_drop,
        )

        # Save corrected data back to bytes
        corrected_buffer = io.BytesIO()
        if is_excel:
            df.to_excel(corrected_buffer, index=False)
        else:
            df.to_csv(corrected_buffer, index=False)
        corrected_buffer.seek(0)
        corrected_data = corrected_buffer.getvalue()

        # Re-validate with corrected data (should pass now if all_fixable was True)
        try:
            validate_uploaded_csv(
                corrected_data,
                required_columns=REQUIRED_UPLOAD_COLUMNS,
                max_rows=MAX_UPLOAD_ROWS,
            )
        except CSVValidationError as exc:
            if exc.date_correction is None:
                # Different error - show it
                st.session_state.pop("pending_date_correction", None)
                _handle_failure(exc)
                return
            else:
                # Still has date issues - update the pending correction
                st.session_state["pending_date_correction"] = exc.date_correction
                st.warning("Some dates still have issues after correction.")
                st.rerun()
                return

        # Load the corrected dataset
        corrected_buffer.seek(0)
        corrected_buffer.name = correction.original_name
        df_final, meta = data_cache.load_dataset_from_bytes(
            corrected_data, correction.original_name
        )

        # Generate a hash for the corrected data
        import hashlib

        data_hash = hashlib.sha256(corrected_data).hexdigest()[:16]
        key = f"corrected::{correction.original_name}::{data_hash}"

        _store_dataset(df_final, meta, key, data_hash=data_hash, saved_path=None)
        st.session_state.pop("pending_date_correction", None)

        # Build success message and store it for display after rerun
        fixes = []
        if correction.corrections:
            fixes.append(f"{len(correction.corrections)} date correction(s)")
        total_dropped = len(correction.all_rows_to_drop)
        if total_dropped > 0:
            fixes.append(f"{total_dropped} row(s) with empty dates removed")
        fixes_str = " and ".join(fixes) if fixes else "corrections"
        st.session_state["date_correction_success"] = (
            f"✅ Applied {fixes_str}. Loaded {correction.original_name}."
        )

        # Rerun to show the loaded data properly
        st.rerun()

    except Exception as exc:
        st.session_state.pop("pending_date_correction", None)
        st.error(f"Failed to apply corrections: {exc}")


def _load_sample_dataset(label: str, path: Path) -> None:
    try:
        df, meta = data_cache.load_dataset_from_path(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        _handle_failure(exc)
        return

    resolved = path.resolve()
    try:
        data_hash = hash_path(resolved)
    except Exception as exc:  # pragma: no cover - defensive
        _handle_failure(exc)
        return

    key = f"sample::{resolved}"
    _store_dataset(df, meta, key, data_hash=data_hash, saved_path=resolved)
    st.session_state["uploaded_file_path"] = str(resolved)
    st.success(f"Loaded sample dataset “{label}”.")
    st.caption(_dataset_summary(df, meta))
    _render_validation(meta)
    st.dataframe(df.head(20))


def _load_uploaded_dataset(uploaded) -> None:
    try:
        guarded = guard_and_buffer_upload(uploaded)
    except UploadViolation as exc:
        _handle_failure(exc)
        return
    except Exception as exc:  # pragma: no cover - defensive guard
        _handle_failure(exc)
        return

    data = guarded.data
    try:
        validate_uploaded_csv(
            data,
            required_columns=REQUIRED_UPLOAD_COLUMNS,
            max_rows=MAX_UPLOAD_ROWS,
        )
    except CSVValidationError as exc:
        _handle_failure(exc)
        return
    try:
        df, meta = data_cache.load_dataset_from_bytes(data, guarded.original_name)
    except Exception as exc:
        _handle_failure(exc)
        return

    key = f"upload::{guarded.original_name}::{guarded.content_hash}"
    _store_dataset(
        df,
        meta,
        key,
        data_hash=guarded.content_hash,
        saved_path=guarded.stored_path,
    )
    st.session_state["uploaded_file_path"] = str(guarded.stored_path)
    st.success(f"Loaded {guarded.original_name}.")
    st.caption(_dataset_summary(df, meta))
    _render_validation(meta)
    st.dataframe(df.head(20))


def _maybe_autoload_sample() -> None:
    # Skip if we've already loaded data this session
    if st.session_state.get("_data_loaded_once"):
        return
    if app_state.has_valid_upload():
        st.session_state["_data_loaded_once"] = True
        return
    default = data_cache.default_sample_dataset()
    if not default:
        return
    _load_sample_dataset(default.label, default.path)
    st.session_state["_data_loaded_once"] = True


def render_data_page() -> None:
    app_state.initialize_session_state()
    _maybe_autoload_sample()
    st.title("Data")

    # Check if there's a pending date correction that needs user approval
    pending_correction = st.session_state.get("pending_date_correction")
    if pending_correction is not None:
        st.write("A file upload needs date corrections before it can be processed.")
        _render_date_correction_ui(
            "Some dates have issues that can be automatically corrected.",
            pending_correction,
        )
        return

    # Show success message if we just applied corrections
    success_msg = st.session_state.pop("date_correction_success", None)
    if success_msg:
        st.success(success_msg)

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
            st.markdown("---")
            st.subheader("Column Configuration")

            all_columns = list(df.columns)

            # =================================================================
            # Risk-Free and Benchmark Selection
            # Using on_change callbacks to capture user selection immediately
            # =================================================================
            col1, col2 = st.columns(2)

            # Risk-free rate column selection
            with col1:
                rf_candidates = st.session_state.get("risk_free_candidates", [])
                rf_auto = [c for c in rf_candidates if c in all_columns]
                rf_other = [c for c in all_columns if c not in rf_candidates]
                rf_options = ["(None)"] + rf_auto + rf_other

                current_rf = st.session_state.get("selected_risk_free")
                if current_rf and current_rf in rf_options:
                    rf_idx = rf_options.index(current_rf)
                else:
                    rf_idx = 0

                def on_rf_change():
                    val = st.session_state["_widget_rf"]
                    st.session_state["selected_risk_free"] = (
                        None if val == "(None)" else val
                    )

                st.selectbox(
                    "Risk-Free Rate Column",
                    rf_options,
                    index=rf_idx,
                    key="_widget_rf",
                    on_change=on_rf_change,
                    help="Select any column as the risk-free rate.",
                )

            # Benchmark column selection
            with col2:
                bench_candidates = st.session_state.get("benchmark_candidates", [])
                bench_auto = [c for c in bench_candidates if c in all_columns]
                bench_other = [c for c in all_columns if c not in bench_candidates]
                bench_options = ["(None)"] + bench_auto + bench_other

                current_bench = st.session_state.get("selected_benchmark")
                if current_bench and current_bench in bench_options:
                    bench_idx = bench_options.index(current_bench)
                else:
                    bench_idx = 0

                def on_bench_change():
                    val = st.session_state["_widget_bench"]
                    st.session_state["selected_benchmark"] = (
                        None if val == "(None)" else val
                    )

                st.selectbox(
                    "Benchmark Column (optional)",
                    bench_options,
                    index=bench_idx,
                    key="_widget_bench",
                    on_change=on_bench_change,
                    help="Select a benchmark for comparison.",
                )

            # Read current selections
            selected_rf = st.session_state.get("selected_risk_free")
            selected_bench = st.session_state.get("selected_benchmark")

            # =================================================================
            # Fund Column Selection with checkbox table
            # Pattern: Use returned DataFrame, no widget key (like index= pattern)
            # =================================================================
            st.markdown("---")
            st.subheader("Fund Column Selection")

            # Get all potential fund columns (excluding system columns)
            system_columns = {selected_rf, selected_bench, "Date"} - {None}
            available_funds = [c for c in all_columns if c not in system_columns]

            # Initialize fund selection if needed
            if "selected_fund_columns" not in st.session_state:
                st.session_state["selected_fund_columns"] = set(available_funds)

            # Get current valid selection
            current_selection = set(
                st.session_state.get("selected_fund_columns", [])
            ) & set(available_funds)

            # Bulk action buttons
            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("✅ Select All", key="btn_sel_all"):
                    st.session_state["selected_fund_columns"] = set(available_funds)
                    st.session_state["fund_columns"] = available_funds
                    st.rerun()
            with btn_cols[1]:
                if st.button("❌ Clear All", key="btn_clr_all"):
                    st.session_state["selected_fund_columns"] = set()
                    st.session_state["fund_columns"] = []
                    st.rerun()
            with btn_cols[2]:
                if st.button("🔄 Invert", key="btn_inv_sel"):
                    inverted = set(available_funds) - current_selection
                    st.session_state["selected_fund_columns"] = inverted
                    st.session_state["fund_columns"] = list(inverted)
                    st.rerun()

            # Stats (before editor, shows current state)
            n_selected = len(current_selection)
            n_total = len(available_funds)
            st.markdown(f"**{n_selected} of {n_total}** funds selected")

            # Build dataframe from canonical state
            fund_df = pd.DataFrame(
                {
                    "Include": [fund in current_selection for fund in available_funds],
                    "Fund Name": available_funds,
                }
            )

            st.write(f"DEBUG: fund_df has {fund_df['Include'].sum()} checked")

            # Render editor WITHOUT key - let Streamlit manage it
            # The returned DataFrame has user's edits applied
            edited_df = st.data_editor(
                fund_df,
                hide_index=True,
                use_container_width=True,
                height=400,
                column_config={
                    "Include": st.column_config.CheckboxColumn(
                        "Include",
                        help="Check to include fund in analysis",
                        width="small",
                    ),
                    "Fund Name": st.column_config.TextColumn(
                        "Fund Name",
                        disabled=True,
                        width="large",
                    ),
                },
                disabled=["Fund Name"],
            )

            st.write(f"DEBUG: edited_df has {edited_df['Include'].sum()} checked")

            # Extract selection from returned DataFrame and update canonical state
            new_selection = set(
                edited_df.loc[edited_df["Include"], "Fund Name"].tolist()
            )

            st.write(f"DEBUG: new_selection has {len(new_selection)} items")
            st.write(f"DEBUG: current_selection has {len(current_selection)} items")
            st.write(f"DEBUG: are they equal? {new_selection == current_selection}")

            # Only update if changed (avoid infinite rerun)
            if new_selection != current_selection:
                st.write("DEBUG: UPDATING selected_fund_columns!")
                st.session_state["selected_fund_columns"] = new_selection
                st.session_state["fund_columns"] = list(new_selection)

            # Show current configuration summary
            st.markdown("---")
            st.subheader("Data Summary")

            final_count = len(new_selection)
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Risk-Free", selected_rf or "Not selected")
            with summary_cols[1]:
                st.metric("Benchmark", selected_bench or "Not selected")
            with summary_cols[2]:
                st.metric("Fund Columns", final_count)


render_data_page()
