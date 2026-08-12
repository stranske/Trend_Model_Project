"""Regression coverage for the Results-page completion banner."""

import importlib
from types import SimpleNamespace

import pandas as pd


def test_banner_sharpe_is_portfolio_level_or_absent(monkeypatch) -> None:
    """A fund-level metrics table must never supply the banner's Sharpe value."""
    page = importlib.import_module("streamlit_app.pages.3_Results")
    monkeypatch.setattr(page.st, "session_state", {})
    result = SimpleNamespace(
        details={
            "out_sample_scaled": pd.DataFrame(
                {"Fund A": [0.02, 0.01], "Fund B": [0.01, 0.02]}
            ),
            "fund_weights": {"Fund A": 0.5, "Fund B": 0.5},
        },
        metrics=pd.DataFrame({"Sharpe": [-2.04, 1.14]}, index=["Fund A", "Fund B"]),
    )

    assert page._completed_state_sharpe(result) is None
    assert "-2.04" not in page._completed_state_line(result, fund_count=2)
