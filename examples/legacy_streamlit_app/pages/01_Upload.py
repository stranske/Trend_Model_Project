"""Simplified, lint-compliant Upload page (reset after corruption)."""

from pathlib import Path
from typing import Any, Tuple

import streamlit as st


def _lazy_imports() -> Tuple[Any, Any, Any]:
    from trend_analysis.io.validators import (  # type: ignore
        create_sample_template,
        load_and_validate_upload,
        validate_returns_schema,
    )

    return create_sample_template, load_and_validate_upload, validate_returns_schema


def main() -> None:
    st.title("📤 Data Upload & Validation")

    try:
        (
            create_sample_template,
            load_and_validate_upload,
            validate_returns_schema,
        ) = _lazy_imports()
    except Exception as e:  # pragma: no cover - import edge
        st.error(f"Failed to import validators: {e}")
        return

    uploaded = st.file_uploader(
        "Upload returns (CSV or Excel). Requires a 'Date' column.",
        type=["csv", "xlsx", "xls"],
    )

    demo_csv = Path("demo/demo_returns.csv")
    if demo_csv.exists() and st.button("Load demo data"):
        try:
            with demo_csv.open("rb") as f:
                df, meta = load_and_validate_upload(f)
            st.session_state["returns_df"] = df
            st.session_state["schema_meta"] = meta
            st.success(
                f"Loaded demo: {df.shape[0]} rows × {df.shape[1]} cols. Range: "
                f"{df.index.min().date()} → {df.index.max().date()}."
            )
            st.dataframe(df.head(12))
        except Exception as e:
            st.error(str(e))

    if uploaded is not None:
        try:
            df, meta = load_and_validate_upload(uploaded)
            st.session_state["returns_df"] = df
            st.session_state["schema_meta"] = meta
            st.success(
                f"Loaded {df.shape[0]} rows × {df.shape[1]} columns. Range: "
                f"{df.index.min().date()} to {df.index.max().date()}."
            )
            st.dataframe(df.head(12))
        except Exception as e:
            st.error(str(e))
    else:
        st.info("No file uploaded yet.")

    with st.expander("Need a template?"):
        tmpl = create_sample_template()
        st.download_button(
            "Download Sample Template", tmpl.to_csv(index=False), "sample_returns.csv"
        )
        st.dataframe(tmpl.head())


if __name__ == "__main__":  # pragma: no cover - manual run
    main()
