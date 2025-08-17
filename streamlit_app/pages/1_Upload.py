import streamlit as st
from trend_portfolio_app.data_schema import (
    load_and_validate_file,
    infer_benchmarks,
)
import os
from functools import lru_cache

st.title("Upload")

uploaded = st.file_uploader(
    "Upload returns (CSV or Excel). Requires a 'Date' column.",
    type=["csv", "xlsx", "xls"],
)

demo_path_csv = "demo/demo_returns.csv"
demo_path_xlsx = "demo/demo_returns.xlsx"
if os.path.exists(demo_path_csv) or os.path.exists(demo_path_xlsx):
    if st.button("Load demo data"):
        path = demo_path_csv if os.path.exists(demo_path_csv) else demo_path_xlsx
        try:

            @lru_cache(maxsize=2)
            def _load_demo(p: str):
                with open(p, "rb") as f:
                    return load_and_validate_file(f)

            df, meta = _load_demo(path)
            st.session_state["returns_df"] = df
            st.session_state["schema_meta"] = meta
            st.success(
                f"Loaded demo: {df.shape[0]} rows × {df.shape[1]} cols. Range: {df.index.min().date()} → {df.index.max().date()}."
            )
            st.dataframe(df.head(12))
            candidates = infer_benchmarks(list(df.columns))
            st.session_state["benchmark_candidates"] = candidates
            if candidates:
                st.info("Possible benchmark columns: " + ", ".join(candidates))
        except Exception as e:
            st.error(f"Demo load failed: {e}")

if uploaded is not None:
    try:
        df, meta = load_and_validate_file(uploaded)
        st.session_state["returns_df"] = df
        st.session_state["schema_meta"] = meta
        st.success(
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns. Range: {df.index.min().date()} to {df.index.max().date()}."
        )
        st.dataframe(df.head(12))
        candidates = infer_benchmarks(list(df.columns))
        st.session_state["benchmark_candidates"] = candidates
        if candidates:
            st.info("Possible benchmark columns: " + ", ".join(candidates))
    except Exception as e:
        st.error(f"Validation failed: {e}")
else:
    st.warning("No file uploaded yet.")
