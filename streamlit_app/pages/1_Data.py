"""Data page for the Streamlit application."""

from __future__ import annotations

import time
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
from streamlit_app.components.data_schema import (
    SchemaMeta,
    infer_benchmarks,
    infer_risk_free_candidates,
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
from trend.input_validation import InputValidationError
from trend_analysis.io.market_data import MarketDataValidationError

DATE_COLUMN = "Date"
REQUIRED_UPLOAD_COLUMNS = (DATE_COLUMN,)
MAX_UPLOAD_ROWS = 50_000


def _fund_table_state_key(data_key: str) -> str:
    return f"fund_table::{data_key}"


def _ensure_fund_table_state(
    *,
    data_key: str,
    available_funds: list[str],
    current_selection: list[str],
) -> str:
    """Keep a stable, per-dataset fund Include table in session state."""

    table_key = _fund_table_state_key(data_key)
    existing = st.session_state.get(table_key)

    if isinstance(existing, pd.DataFrame) and {"Include", "Fund Name"}.issubset(
        existing.columns
    ):
        df_existing = existing[["Include", "Fund Name"]].copy()
        df_existing["Fund Name"] = df_existing["Fund Name"].astype(str)
        df_existing["Include"] = (
            df_existing["Include"].astype("boolean").fillna(False).astype(bool)
        )
        if df_existing["Fund Name"].tolist() == list(available_funds):
            st.session_state[table_key] = df_existing
            return table_key

        include_map = dict(
            zip(df_existing["Fund Name"].tolist(), df_existing["Include"].tolist())
        )
        st.session_state[table_key] = pd.DataFrame(
            {
                "Include": [
                    bool(include_map.get(fund, False)) for fund in available_funds
                ],
                "Fund Name": [str(fund) for fund in available_funds],
            }
        )
        return table_key

    st.session_state[table_key] = pd.DataFrame(
        {
            "Include": [fund in set(current_selection) for fund in available_funds],
            "Fund Name": [str(fund) for fund in available_funds],
        }
    )
    return table_key


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
        # Force Streamlit to clear the file_uploader selection on next rerun.
        # This prevents re-processing the same uploaded file (and re-triggering the
        # date-correction flow) when users interact with other widgets.
        st.session_state["upload_widget_version"] = (
            int(st.session_state.get("upload_widget_version", 0)) + 1
        )

        selected_rf = st.session_state.get("selected_risk_free")
        selected_bench = st.session_state.get("selected_benchmark")
        system_cols = {selected_rf, selected_bench, "Date"} - {None}
        fund_cols = [c for c in all_columns if c not in system_cols]
        st.session_state["selected_fund_columns"] = list(fund_cols)
        st.session_state["fund_columns"] = list(fund_cols)
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
            width="stretch",
            key="apply_date_corrections",
        ):
            _apply_pending_date_corrections()

    with col2:
        if st.button(
            "❌ Cancel",
            width="stretch",
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

        # Persist corrected bytes to disk so we can reproduce runs (especially in Codespaces).
        from streamlit_app.components.upload_guard import store_buffered_upload

        guarded = store_buffered_upload(corrected_data, correction.original_name)
        data_hash = guarded.content_hash[:16]
        key = f"corrected::{correction.original_name}::{data_hash}"

        _store_dataset(
            df_final,
            meta,
            key,
            data_hash=data_hash,
            saved_path=guarded.stored_path,
        )
        st.session_state["uploaded_file_path"] = str(guarded.stored_path)
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
        upload_widget_key = (
            f"upload_returns::{st.session_state.get('upload_widget_version', 0)}"
        )
        uploaded = st.file_uploader(
            "Upload returns (CSV or Excel with a Date column)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
            key=upload_widget_key,
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

            _t_fund_start = time.perf_counter()

            # Get all potential fund columns (excluding system columns)
            system_columns = {selected_rf, selected_bench, "Date"} - {None}
            available_funds = [c for c in all_columns if c not in system_columns]

            # Default: select all non-index columns.
            # Index-like columns are inferred from benchmark/risk-free candidates.
            index_candidates = set(
                st.session_state.get("benchmark_candidates", [])
            ) | set(st.session_state.get("risk_free_candidates", []))
            default_selected_funds = [
                c for c in available_funds if c not in index_candidates
            ]

            _t_funds_derived = time.perf_counter()

            # Derive stable key per dataset to avoid stale state bleed
            data_key = st.session_state.get("data_loaded_key", "default")
            include_prefix = f"fund_include::{data_key}::"
            init_key = f"fund_selection_initialized::{data_key}"

            # Initialize fund selection if needed (list for order stability)
            if "selected_fund_columns" not in st.session_state:
                st.session_state["selected_fund_columns"] = list(available_funds)
            if "fund_columns" not in st.session_state:
                st.session_state["fund_columns"] = list(available_funds)

            # Current valid selection (respect available funds)
            current_selection = [
                f
                for f in st.session_state.get("selected_fund_columns", [])
                if f in available_funds
            ]

            # Default: select all funds (everything except system/index columns).
            # Do this once per dataset key so we don't overwrite user choices.
            if not st.session_state.get(init_key):
                st.session_state["selected_fund_columns"] = list(default_selected_funds)
                st.session_state["fund_columns"] = list(default_selected_funds)
                current_selection = list(default_selected_funds)
                st.session_state[init_key] = True

            if not st.session_state.get("fund_columns"):
                st.session_state["fund_columns"] = list(current_selection)

            # Bulk actions
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("✅ Select All", key="btn_sel_all"):
                    for fund in available_funds:
                        st.session_state[f"{include_prefix}{fund}"] = True
                    st.session_state["selected_fund_columns"] = list(available_funds)
                    st.session_state["fund_columns"] = list(available_funds)
                    st.rerun()
            with btn_cols[1]:
                if st.button("❌ Clear All", key="btn_clr_all"):
                    for fund in available_funds:
                        st.session_state[f"{include_prefix}{fund}"] = False
                    st.session_state["selected_fund_columns"] = []
                    st.session_state["fund_columns"] = []
                    st.rerun()
            with btn_cols[2]:
                if st.button("🔄 Invert", key="btn_inv_sel"):
                    inverted: list[str] = []
                    for fund in available_funds:
                        key = f"{include_prefix}{fund}"
                        current_val = bool(
                            st.session_state.get(key, fund in set(current_selection))
                        )
                        st.session_state[key] = not current_val
                        if st.session_state[key] is True:
                            inverted.append(fund)
                    st.session_state["selected_fund_columns"] = inverted
                    st.session_state["fund_columns"] = inverted
                    st.rerun()

            # Multi-select (range) — shift-click equivalent.
            with st.expander("Bulk add/remove (range select)", expanded=True):
                st.caption(
                    "Select a start and end fund, then include/exclude the whole range."
                )

                range_cols = st.columns(4)
                with range_cols[0]:
                    start_fund = st.selectbox(
                        "From",
                        options=available_funds,
                        index=0,
                        key=f"range_from::{data_key}",
                    )
                with range_cols[1]:
                    end_fund = st.selectbox(
                        "To",
                        options=available_funds,
                        index=min(len(available_funds) - 1, 0),
                        key=f"range_to::{data_key}",
                    )

                start_idx = available_funds.index(start_fund)
                end_idx = available_funds.index(end_fund)
                lo = min(start_idx, end_idx)
                hi = max(start_idx, end_idx)
                range_funds = available_funds[lo : hi + 1]
                st.caption(f"Range size: {len(range_funds)}")

                with range_cols[2]:
                    if st.button("✅ Include range", key=f"btn_inc_range::{data_key}"):
                        for fund in range_funds:
                            st.session_state[f"{include_prefix}{fund}"] = True
                        st.rerun()
                with range_cols[3]:
                    if st.button("❌ Exclude range", key=f"btn_exc_range::{data_key}"):
                        for fund in range_funds:
                            st.session_state[f"{include_prefix}{fund}"] = False
                        st.rerun()

            # Stats (live)
            n_selected = sum(
                1
                for fund in available_funds
                if bool(st.session_state.get(f"{include_prefix}{fund}", False))
            )
            n_total = len(available_funds)
            st.markdown(f"**{n_selected} of {n_total}** funds selected")

            # Seed checkbox widget values from canonical selection (vectorized).
            _t_seed_start = time.perf_counter()
            defaults = {
                f"{include_prefix}{fund}": (fund in set(current_selection))
                for fund in available_funds
                if f"{include_prefix}{fund}" not in st.session_state
            }
            if defaults:
                st.session_state.update(defaults)

            _t_seed_done = time.perf_counter()

            # Faster rendering: one widget per row (avoid per-row columns/write).
            _t_render_start = time.perf_counter()
            with st.container(height=400):
                for fund in available_funds:
                    st.checkbox(
                        fund,
                        key=f"{include_prefix}{fund}",
                    )

            _t_render_done = time.perf_counter()

            # Always-visible measurements (so you don't need to open Debug).
            st.caption(
                " | ".join(
                    [
                        f"Perf: total {(_t_render_done - _t_fund_start) * 1000:.0f}ms",
                        f"render {(_t_render_done - _t_render_start) * 1000:.0f}ms",
                        f"seed {(_t_seed_done - _t_seed_start) * 1000:.0f}ms",
                        f"funds {len(available_funds)}",
                        f"selected {n_selected}",
                        f"default_selected {len(default_selected_funds)}",
                        f"init_applied {bool(st.session_state.get(init_key))}",
                        f"defaults_seeded {len(defaults)}",
                        f"range {len(range_funds)}",
                    ]
                )
            )

            new_selection_list = [
                fund
                for fund in available_funds
                if bool(st.session_state.get(f"{include_prefix}{fund}", False))
            ]

            # Persist canonical state from edited table
            if new_selection_list != st.session_state.get("selected_fund_columns", []):
                st.session_state["selected_fund_columns"] = new_selection_list
                st.session_state["fund_columns"] = list(new_selection_list)
            elif st.session_state.get("fund_columns") != new_selection_list:
                st.session_state["fund_columns"] = list(new_selection_list)

            # Explicitly apply/lock the selection for downstream pages.
            apply_cols = st.columns([1, 3])
            with apply_cols[0]:
                if st.button("Apply selection", key=f"btn_apply_funds::{data_key}"):
                    prohibited = system_columns
                    sanitized = [c for c in new_selection_list if c not in prohibited]
                    st.session_state["analysis_fund_columns"] = list(sanitized)
                    analysis_runner.clear_cached_analysis()
                    st.success("Fund selection applied for analysis.")
            with apply_cols[1]:
                applied = st.session_state.get("analysis_fund_columns")
                if isinstance(applied, list):
                    st.caption(f"Applied selection: {len(applied)} funds")

            # Show current configuration summary
            st.markdown("---")
            st.subheader("Data Summary")

            final_count = len(new_selection_list)
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Risk-Free", selected_rf or "Not selected")
            with summary_cols[1]:
                st.metric("Benchmark", selected_bench or "Not selected")
            with summary_cols[2]:
                st.metric("Fund Columns", final_count)

            # Debug panel to inspect selection state when diagnosing persistence
            with st.expander("Debug: Fund selection state", expanded=False):
                st.session_state["_debug_fund_run"] = (
                    int(st.session_state.get("_debug_fund_run", 0)) + 1
                )

                snapshot = {
                    "run": st.session_state.get("_debug_fund_run"),
                    "data_loaded_key": st.session_state.get("data_loaded_key"),
                    "available_funds_count": len(available_funds),
                    "index_candidates_count": len(index_candidates),
                    "default_selected_funds_count": len(default_selected_funds),
                    "checkbox_selected_count": len(new_selection_list),
                    "range_funds_count": len(range_funds),
                    "defaults_seeded_count": len(defaults),
                    "perf_ms": {
                        "derive_funds": round(
                            (_t_funds_derived - _t_fund_start) * 1000, 2
                        ),
                        "seed_defaults": round(
                            (_t_seed_done - _t_seed_start) * 1000, 2
                        ),
                        "render_checkboxes": round(
                            (_t_render_done - _t_render_start) * 1000, 2
                        ),
                        "fund_total": round((_t_render_done - _t_fund_start) * 1000, 2),
                    },
                    "selected_fund_columns_count": len(
                        st.session_state.get("selected_fund_columns") or []
                    ),
                    "fund_columns_count": len(
                        st.session_state.get("fund_columns") or []
                    ),
                }

                history = st.session_state.get("_debug_fund_history", [])
                if not isinstance(history, list):
                    history = []
                history.append(snapshot)
                st.session_state["_debug_fund_history"] = history[-8:]

                st.json(
                    {
                        "latest": snapshot,
                        "history": st.session_state.get("_debug_fund_history", []),
                        "available_funds": available_funds,
                        "selected_fund_columns": st.session_state.get(
                            "selected_fund_columns"
                        ),
                        "fund_columns": st.session_state.get("fund_columns"),
                        "checkbox_prefix": include_prefix,
                    }
                )


render_data_page()
