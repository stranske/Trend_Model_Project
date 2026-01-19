"""Main Streamlit app for testing upload functionality."""

import os
import sys

import streamlit as st

# Add paths for our modules using relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "..", "src"))
sys.path.append(os.path.join(base_dir, "..", "app"))

st.set_page_config(page_title="Trend Analysis Upload Test", page_icon="📤", layout="wide")

st.title("🔬 Upload Page Test")

st.markdown("""
This is a test environment for the new upload page functionality.

**Features being tested:**
- File upload with validation
- Schema validation with detailed feedback
- Frequency detection
- Sample template download
- Error handling with actionable messages
""")

# Import and run the upload page
try:
    # Note: streamlit_app/pages/1_Upload.py doesn't have a main function
    # It's a streamlit page that runs directly, so we'll skip this import
    # and use the fallback implementation instead
    raise ImportError("Streamlit page doesn't have a main function")

except ImportError as e:
    st.error(f"Import error: {e}")
    st.markdown("**Fallback:** Running upload functionality directly...")

    # Direct implementation for testing
    from trend_analysis.io.validators import (
        create_sample_template,
        load_and_validate_upload,
        validate_returns_schema,
    )

    st.header("📤 File Upload Test")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain a 'Date' column and numeric return columns",
    )

    # Sample template download
    with st.expander("📋 Download Sample Template"):
        template_df = create_sample_template()
        csv_data = template_df.to_csv(index=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(template_df.head())
        with col2:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="sample_returns.csv",
                mime="text/csv",
            )

    # Process uploaded file
    if uploaded_file is not None:
        st.subheader("📊 Validation Results")

        try:
            # Basic validation first
            import pandas as pd

            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(uploaded_file)
            else:
                df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset

            validation = validate_returns_schema(df_preview)
            st.text(validation.get_report())

            if validation.is_valid:
                # Full processing
                df, meta = load_and_validate_upload(uploaded_file)

                st.success("✅ Upload successful!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Frequency", meta.get("frequency", "Unknown"))

                st.subheader("📋 Data Preview")
                st.dataframe(df.head())

            else:
                st.error("❌ Validation failed - please fix the issues above")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.sidebar.markdown("### 🔍 Debug Info")
    st.sidebar.write("Upload page loaded successfully!")
    if uploaded_file:
        st.sidebar.write(f"File: {uploaded_file.name}")
        st.sidebar.write(f"Size: {uploaded_file.size:,} bytes")


# Quick-reference fuzz cases used by automated tests and manual checks
FUZZY_UPLOAD_CASES = {
    "duplicate_columns": "Date,Return,Return\n2020-01-31,0.1,0.2\n",
    "formula_headers": "=Date,=Alpha\n2020-01-31,0.1\n",
    "too_many_rows": "Date,Value\n"
    + "\n".join([f"2020-01-{day:02d},{day}" for day in range(1, 105)]),
}
