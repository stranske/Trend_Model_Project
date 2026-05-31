"""Fixtures and catalog loading for the baseline kit."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
from baseline_kit import load_catalog as _load_catalog

HERE = Path(__file__).resolve().parent
CATALOG_PATH = HERE / "catalog.yaml"


@functools.lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return _load_catalog(CATALOG_PATH)


@pytest.fixture(scope="session")
def catalog() -> dict[str, Any]:
    return load_catalog()


@pytest.fixture(scope="session")
def baseline_config(catalog) -> str:
    return catalog["baseline"]["config"]


@pytest.fixture(scope="session")
def baseline_output(baseline_config):
    """The Tier-0 reference run, computed once per session."""
    from .harness import run_scenario

    return run_scenario(baseline_config)
