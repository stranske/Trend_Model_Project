"""Regression coverage for the Data-page fund-selection headline."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from tests.app.test_data_page import DummyStreamlit, data_page  # noqa: F401


def _load_sample(monkeypatch: pytest.MonkeyPatch, page: ModuleType, stub: DummyStreamlit) -> None:
    stub.session_state.clear()
    df = pd.DataFrame(
        {"FundA": [0.01, 0.02, -0.01], "SPX Index": [0.03, -0.02, 0.01]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    meta = {"validation": {"issues": [], "warnings": []}, "frequency_label": "monthly"}
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))
    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(page.data_cache, "load_dataset_from_path", lambda _path: (df, meta))
    monkeypatch.setattr(page, "infer_benchmarks", lambda _columns: ["SPX Index"])
    stub.selectbox_map["Choose a sample"] = sample.label
    stub.selectbox_map["Benchmark column (optional)"] = "SPX Index"


def test_selection_counter_matches_applied_count_on_first_render(
    monkeypatch: pytest.MonkeyPatch, data_page  # noqa: F811 - reused pytest fixture
) -> None:
    page, stub = data_page
    _load_sample(monkeypatch, page, stub)

    page.render_data_page()

    checked_count = sum(
        value for key, value in stub.session_state.items() if key.startswith("fund_include::")
    )
    assert f"**{checked_count} of {checked_count}** funds selected" in stub.markdowns
    noun = "fund" if checked_count == 1 else "funds"
    assert f"Applied automatically for analysis: {checked_count} {noun}" in stub.captions
