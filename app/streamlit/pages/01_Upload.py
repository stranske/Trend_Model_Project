"""Upload page for Streamlit trend analysis app."""

import streamlit as st
import pandas as pd
import os
from io import StringIO
from typing import Optional

# Import our custom modules
import sys

sys.path.append("/home/runner/work/Trend_Model_Project/Trend_Model_Project/src")
sys.path.append("/home/runner/work/Trend_Model_Project/Trend_Model_Project/app")

from trend_analysis.io.validators import (
    load_and_validate_upload,
    validate_returns_schema,
    create_sample_template,
)
from streamlit import state as st_state

# Import state management functions
try:
    from streamlit.state import (
        initialize_session_state,
        store_validated_data,
        get_uploaded_data,
        has_valid_upload,
        get_upload_summary,
        clear_upload_data,
    )
except ImportError:
    # Fallback to local import
    import importlib.util

    import os

    state_path = os.path.join(os.path.dirname(__file__), "..", "state.py")
    spec = importlib.util.spec_from_file_location(
        "state",
        os.path.abspath(state_path),
    )
    state_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state_module)

    initialize_session_state = state_module.initialize_session_state
    store_validated_data = state_module.store_validated_data
    get_uploaded_data = state_module.get_uploaded_data
    has_valid_upload = state_module.has_valid_upload
    get_upload_summary = state_module.get_upload_summary
    clear_upload_data = state_module.clear_upload_data


def main():
    """Main upload page functionality."""
    st.title("📤 Data Upload & Validation")

    initialize_session_state()

    # Show current upload status
    if has_valid_upload():
        st.success(f"✅ Data loaded successfully: {get_upload_summary()}")
        if st.button("🗑️ Clear uploaded data"):
            clear_upload_data()
            st.rerun()

    # File upload section
    st.header("Upload Your Returns Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain a 'Date' column and numeric return columns",
    )

    # Sample template download
    with st.expander("📋 Need a template? Download sample format"):
        st.markdown(
            """
        **Required format:**
        - First column must be named 'Date'
        - Date format: YYYY-MM-DD (e.g., 2023-01-31)
        - Additional columns should contain numeric return values
        - Returns can be in decimal format (0.05 for 5%) or percentage format
        
        **Example columns:** Date, Fund_01, Fund_02, SPX_Benchmark
        """
        )

        # Generate and provide download for template
        template_df = create_sample_template()
        csv_data = template_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Sample Template",
            data=csv_data,
            file_name="sample_returns.csv",
            mime="text/csv",
            help="Download this template to see the expected format",
        )

        st.subheader("Preview of sample template:")
        st.dataframe(template_df.head())

    # Demo data loading
    if st.button("🎯 Load Demo Data"):
        demo_path_csv = "demo/demo_returns.csv"
        if os.path.exists(demo_path_csv):
            try:
                with open(demo_path_csv, "rb") as f:
                    df, meta = load_and_validate_upload(f)
                    store_validated_data(df, meta)

                st.success("Demo data loaded successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to load demo data: {str(e)}")
        else:
            st.error("Demo data file not found. Please generate demo data first.")

    # Process uploaded file
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)


def process_uploaded_file(uploaded_file):
    """Process and validate the uploaded file."""
    st.header("📊 Schema Validation Report")

    try:
        # Show file info
        st.info(
            f"📁 Processing file: {uploaded_file.name} ({uploaded_file.size:,} bytes)"
        )

        # First, do a quick schema check without full processing
        df_preview = None
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(uploaded_file)
            else:
                df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
        except Exception as e:
            st.error(f"❌ Unable to read file: {str(e)}")
            return

        # Validate schema
        validation = validate_returns_schema(df_preview)

        # Display validation report
        st.text(validation.get_report())

        if validation.is_valid:
            # Proceed with full processing
            with st.spinner("Processing and normalizing data..."):
                try:
                    df, meta = load_and_validate_upload(uploaded_file)
                    store_validated_data(df, meta)

                    # Success display
                    st.success("✅ File uploaded and validated successfully!")

                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10))

                    # Show data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Return Columns", len(df.columns))
                    with col3:
                        st.metric("Frequency", validation.frequency or "Unknown")

                    if validation.date_range:
                        st.info(
                            f"📅 Date range: {validation.date_range[0]} to {validation.date_range[1]}"
                        )

                    # Show column info
                    with st.expander("📊 Column Statistics"):
                        stats_df = df.describe()
                        st.dataframe(stats_df)

                    # Benchmark detection
                    potential_benchmarks = detect_potential_benchmarks(df.columns)
                    if potential_benchmarks:
                        st.info(
                            f"🎯 Potential benchmark columns detected: {', '.join(potential_benchmarks)}"
                        )
                        st.session_state["benchmark_candidates"] = potential_benchmarks

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

        else:
            # Show actionable error messages
            st.error("❌ Upload failed - please fix the following issues:")

            # Provide specific guidance
            if "Missing required 'Date' column" in validation.issues:
                st.markdown(
                    """
                **🔧 How to fix:**
                1. Ensure your first column is named exactly 'Date' (case sensitive)
                2. Download the sample template above to see the correct format
                """
                )

            if any("invalid dates" in issue for issue in validation.issues):
                st.markdown(
                    """
                **🔧 Date format issues:**
                1. Use YYYY-MM-DD format (e.g., 2023-01-31)
                2. Ensure all date cells contain valid dates
                3. Remove any empty rows or invalid entries
                """
                )

            if any("numeric" in issue.lower() for issue in validation.issues):
                st.markdown(
                    """
                **🔧 Numeric data issues:**
                1. Return columns should contain only numbers
                2. Remove any text or symbols from return columns
                3. Use decimal format (0.05) or percentage format (5%)
                """
                )

            # Offer template download
            st.markdown("---")
            template_df = create_sample_template()
            csv_data = template_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Correct Template",
                data=csv_data,
                file_name="sample_returns.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")


def detect_potential_benchmarks(columns) -> list:
    """Detect potential benchmark columns."""
    benchmark_keywords = [
        "SPX",
        "S&P",
        "SP500",
        "SP-500",
        "TSX",
        "AGG",
        "BOND",
        "BENCH",
        "IDX",
        "INDEX",
        "NASDAQ",
        "DOW",
    ]

    candidates = []
    for col in columns:
        col_upper = col.upper()
        for keyword in benchmark_keywords:
            if keyword in col_upper:
                candidates.append(col)
                break

    return candidates


if __name__ == "__main__":
    main()
