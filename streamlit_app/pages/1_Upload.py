import streamlit as st
from trend_portfolio_app.data_schema import load_and_validate_csv, infer_benchmarks

st.title("Upload")

uploaded = st.file_uploader(
    "Upload a CSV with monthly returns. Requires a 'Date' column.", type=["csv"]
)

if uploaded is not None:
    try:
        df, meta = load_and_validate_csv(uploaded)
        st.session_state["returns_df"] = df
        st.session_state["schema_meta"] = meta
        st.success(
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns. Range: {df.index.min().date()} to {df.index.max().date()}."
        )
        st.dataframe(df.head(12))
        candidates = infer_benchmarks(df.columns)
        st.session_state["benchmark_candidates"] = candidates
        if candidates:
            st.info("Possible benchmark columns: " + ", ".join(candidates))
    except Exception as e:
        st.error(f"Validation failed: {e}")
else:
    st.warning("No file uploaded yet.")
