"""Dataset loading and caching helpers for the Streamlit app."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import streamlit as st

from trend_portfolio_app.data_schema import SchemaMeta, load_and_validate_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "demo"


@dataclass(frozen=True)
class SampleDataset:
    """Description of a bundled sample dataset."""

    label: str
    path: Path
    description: str | None = None


def _available_demo_files() -> list[Path]:
    """Return a list of existing demo dataset paths."""

    candidates = [
        DEMO_DIR / "demo_returns.csv",
        DEMO_DIR / "demo_returns.xlsx",
    ]
    return [path for path in candidates if path.exists()]


def list_sample_datasets() -> list[SampleDataset]:
    """Return the bundled sample datasets that are available on disk."""

    datasets: list[SampleDataset] = []
    for path in _available_demo_files():
        label = path.name.replace("_", " ")
        description = "Default sample dataset bundled with the project"
        datasets.append(SampleDataset(label=label, path=path, description=description))
    return datasets


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Return a stable hash for a dataframe used as a cache key."""

    payload = df.to_json(date_format="iso", date_unit="ns", orient="split")
    return json.dumps({"data": payload})


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset_from_path(path: str) -> tuple[pd.DataFrame, SchemaMeta]:
    """Load and validate a dataset from a filesystem path."""

    file_path = Path(path)
    with file_path.open("rb") as handle:
        return load_and_validate_file(handle)


@st.cache_data(show_spinner="Validating upload…")
def load_dataset_from_bytes(
    data: bytes, filename: str
) -> tuple[pd.DataFrame, SchemaMeta]:
    """Load and validate a dataset from uploaded file bytes."""

    buffer = io.BytesIO(data)
    buffer.name = filename
    return load_and_validate_file(buffer)


def cache_key_for_frame(df: pd.DataFrame) -> str:
    """Return a cache key representing the dataframe contents."""

    return _hash_dataframe(df)


def clear_cache() -> None:
    """Invalidate cached datasets."""

    load_dataset_from_path.clear()
    load_dataset_from_bytes.clear()


def default_sample_dataset() -> SampleDataset | None:
    """Return the preferred sample dataset if available."""

    datasets = list_sample_datasets()
    return datasets[0] if datasets else None


def dataset_choices() -> Mapping[str, SampleDataset]:
    """Return a mapping from human readable label to dataset metadata."""

    return {dataset.label: dataset for dataset in list_sample_datasets()}
