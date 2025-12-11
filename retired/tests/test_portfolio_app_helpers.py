"""Unit tests for helper utilities in the Streamlit app module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pandas as pd
import pytest
import yaml


def load_helper_namespace() -> Dict[str, Any]:
    """Import the helper portion of ``app.py`` and return its globals, mocking
    Streamlit."""
    with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(
            "trend_portfolio_app.app", "src/trend_portfolio_app/app.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert module is not None
        loader = spec.loader
        assert loader is not None
        if hasattr(loader, "exec_module") and callable(getattr(loader, "exec_module")):
            loader.exec_module(module)
        else:
            raise TypeError(
                f"Loader {type(loader)} does not have an exec_module method"
            )
        return module.__dict__


def test_merge_update_recurses_through_nested_dicts():
    ns = load_helper_namespace()
    merge = ns["_merge_update"]

    base = {"portfolio": {"policy": "all", "weights": {"A": 0.5}}}
    updates = {"portfolio": {"weights": {"B": 0.5}, "policy": "rank"}}
    merged = merge(base, updates)

    assert merged["portfolio"]["policy"] == "rank"
    assert merged["portfolio"]["weights"] == {"A": 0.5, "B": 0.5}
    # Ensure base dictionary is not mutated
    assert base["portfolio"]["policy"] == "all"


def test_summarise_multi_handles_missing_fields():
    ns = load_helper_namespace()
    summarise_multi = ns["_summarise_multi"]

    results = [
        {
            "period": ("2020-01", "2020-02", "2020-03", "2020-04"),
            "out_ew_stats": {"sharpe": 1.23456, "cagr": 0.10101},
            "out_user_stats": {"sharpe": 2.34567, "cagr": 0.20202},
        },
        {"period": None, "out_ew_stats": {}, "out_user_stats": {}},
    ]

    df = summarise_multi(results)
    assert list(df.columns) == [
        "in_start",
        "in_end",
        "out_start",
        "out_end",
        "ew_sharpe",
        "user_sharpe",
        "ew_cagr",
        "user_cagr",
    ]
    assert df.iloc[0]["ew_sharpe"] == pytest.approx(1.2346)
    assert df.iloc[0]["user_sharpe"] == pytest.approx(2.3457)


def test_to_yaml_preserves_key_order():
    ns = load_helper_namespace()
    to_yaml = ns["_to_yaml"]

    data = {"b": 1, "a": 2}
    dumped = to_yaml(data)
    loaded = yaml.safe_load(dumped)
    assert loaded == data


def test_summarise_run_df_rounds_numeric_columns():
    ns = load_helper_namespace()
    summarise_run_df = ns["_summarise_run_df"]

    df = pd.DataFrame({"metric": [0.123456, 0.654321], "label": ["x", "y"]})
    rounded = summarise_run_df(df)
    assert rounded["metric"].tolist() == [0.1235, 0.6543]


def test_read_defaults_sets_csv_path():
    ns = load_helper_namespace()
    read_defaults = ns["_read_defaults"]

    data = read_defaults()

    assert "portfolio" in data
    assert data["portfolio"].get("policy") == ""
    assert Path(data["data"]["csv_path"]).name == "demo_returns.csv"
