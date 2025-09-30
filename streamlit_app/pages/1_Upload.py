import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import streamlit as st

from trend_portfolio_app.data_schema import infer_benchmarks, load_and_validate_file

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
            st.session_state["uploaded_file_path"] = str(Path(path).resolve())
            st.success(
                f"Loaded demo: {df.shape[0]} rows × {df.shape[1]} cols. Range: "
                f"{df.index.min().date()} → {df.index.max().date()}."
            )
            st.dataframe(df.head(12))
            candidates = infer_benchmarks(list(df.columns))
            st.session_state["benchmark_candidates"] = candidates
            if candidates:
                st.info("Possible benchmark columns: " + ", ".join(candidates))
        except Exception as e:
            st.error(str(e))

if uploaded is not None:
    try:
        df, meta = load_and_validate_file(uploaded)
        st.session_state["returns_df"] = df
        st.session_state["schema_meta"] = meta
        tmp_dir = Path(tempfile.gettempdir()) / "trend_app_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}.csv"
        reset = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        reset.to_csv(tmp_path, index=False)
        st.session_state["uploaded_file_path"] = str(tmp_path)
        st.success(
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns. Range: "
            f"{df.index.min().date()} to {df.index.max().date()}."
        )
        st.dataframe(df.head(12))
        candidates = infer_benchmarks(list(df.columns))
        st.session_state["benchmark_candidates"] = candidates
        if candidates:
            st.info("Possible benchmark columns: " + ", ".join(candidates))
    except Exception as e:
        st.error(str(e))
else:
    st.warning("No file uploaded yet.")
