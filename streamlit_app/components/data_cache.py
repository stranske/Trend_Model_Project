"""Utilities for caching dataset loads inside the Streamlit app."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from trend_portfolio_app.data_schema import SchemaMeta, load_and_validate_file


@st.cache_data(show_spinner=False)
def load_dataset_from_bytes(payload: bytes) -> Tuple[pd.DataFrame, SchemaMeta]:
    """Parse an uploaded dataset from raw bytes using the schema validator."""

    buffer = io.BytesIO(payload)
    df, meta = load_and_validate_file(buffer)
    return df, meta


@st.cache_data(show_spinner=False)
def load_dataset_from_path(path: str) -> Tuple[pd.DataFrame, SchemaMeta]:
    """Load a bundled dataset from disk."""

    handle_path = Path(path)
    with handle_path.open("rb") as handle:
        df, meta = load_and_validate_file(handle)
    return df, meta


def clear_dataset_cache() -> None:
    """Clear cached dataset entries (used when uploads change)."""

    load_dataset_from_bytes.clear()
    load_dataset_from_path.clear()
