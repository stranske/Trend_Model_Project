from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest_plugins = ("tests.app.test_data_page",)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FundA": [0.01, 0.02, -0.01],
            "FundB": [0.02, 0.01, 0.00],
            "SPX Index": [0.03, -0.02, 0.01],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )


def test_fund_selection_commits_visible_checkbox_state(
    monkeypatch: pytest.MonkeyPatch, data_page
) -> None:
    page, stub = data_page

    stub.session_state.clear()
    stub.clear_calls = 0

    meta = {"validation": {"issues": [], "warnings": []}, "frequency_label": "monthly"}
    sample_path = Path("demo/demo_returns.csv")
    sample = page.data_cache.SampleDataset("demo.csv", sample_path)

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(page.data_cache, "load_dataset_from_path", lambda path: (_sample_frame(), meta))
    monkeypatch.setattr(page, "infer_benchmarks", lambda columns: ["SPX Index"])

    stub.selectbox_map["Choose a sample"] = sample.label
    data_key = f"sample::{sample_path.resolve()}"
    stub.session_state[f"fund_include::{data_key}::FundA"] = True
    stub.session_state[f"fund_include::{data_key}::FundB"] = False

    page.render_data_page()

    assert page.st.session_state["selected_fund_columns"] == ["FundA"]
    assert page.st.session_state["fund_columns"] == ["FundA"]
    assert page.st.session_state["analysis_fund_columns"] == ["FundA"]
    assert any("Applied automatically for analysis: 1 fund" in text for text in stub.captions)
    assert stub.clear_calls == 1


def test_fund_selection_preserves_imported_committed_subset(
    monkeypatch: pytest.MonkeyPatch, data_page
) -> None:
    page, stub = data_page

    stub.session_state.clear()
    stub.clear_calls = 0

    meta = {"validation": {"issues": [], "warnings": []}, "frequency_label": "monthly"}
    sample_path = Path("demo/demo_returns.csv")
    sample = page.data_cache.SampleDataset("demo.csv", sample_path)

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(page.data_cache, "load_dataset_from_path", lambda path: (_sample_frame(), meta))
    monkeypatch.setattr(page, "infer_benchmarks", lambda columns: ["SPX Index"])

    stub.selectbox_map["Choose a sample"] = sample.label
    stub.session_state["fund_columns"] = ["FundB"]
    stub.session_state["analysis_fund_columns"] = ["FundB"]

    page.render_data_page()

    assert page.st.session_state["selected_fund_columns"] == ["FundB"]
    assert page.st.session_state["fund_columns"] == ["FundB"]
    assert page.st.session_state["analysis_fund_columns"] == ["FundB"]
    assert any("Applied automatically for analysis: 1 fund" in text for text in stub.captions)
    assert stub.clear_calls == 1
