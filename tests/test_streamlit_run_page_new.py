"""Tests for the new Streamlit run page in ``streamlit_app``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest


def _ctx_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    mock.progress = MagicMock()
    return mock


def _make_streamlit(button_response: Any = False) -> MagicMock:
    st = MagicMock()
    st.session_state = {}
    st.title = MagicMock()
    st.button = MagicMock(return_value=button_response)
    st.error = MagicMock()
    st.warning = MagicMock(return_value=_ctx_mock())
    st.success = MagicMock()
    st.progress = MagicMock(return_value=_ctx_mock())
    st.write = MagicMock()
    st.rerun = MagicMock()
    return st


def _load_run_module(mock_st: MagicMock) -> Any:
    module_path = Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py"
    spec = importlib.util.spec_from_file_location("streamlit_run_page", module_path)
    module = importlib.util.module_from_spec(spec)
    disclaimer_mod = SimpleNamespace(show_disclaimer=lambda: True)

    with patch.dict(
        "sys.modules",
        {
            "streamlit": mock_st,
            "streamlit_app.components.disclaimer": disclaimer_mod,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return module


def test_main_requires_returns_and_config() -> None:
    mock_st = _make_streamlit()
    module = _load_run_module(mock_st)

    module.show_disclaimer = lambda: True
    module.run_simulation = MagicMock()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    mock_st.error.assert_called_once_with("Upload data and set configuration first.")
    module.run_simulation.assert_not_called()


def test_main_exits_when_button_not_clicked() -> None:
    mock_st = _make_streamlit(button_response=False)
    module = _load_run_module(mock_st)

    mock_returns = pd.DataFrame(
        {"A": [0.1, -0.1]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"])
    )
    mock_cfg = {
        "start": pd.Timestamp("2020-01-31"),
        "end": pd.Timestamp("2020-02-29"),
        "lookback_months": "3",
        "risk_target": 0.5,
        "portfolio": {"weighting_scheme": "equal"},
    }
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True
    module.run_simulation = MagicMock()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    module.run_simulation.assert_not_called()
    mock_st.error.assert_not_called()


def test_main_handles_invalid_dates_gracefully() -> None:
    mock_st = _make_streamlit(button_response=True)
    module = _load_run_module(mock_st)

    mock_returns = pd.DataFrame(
        {"A": [0.1, -0.1]}, index=pd.Index([1, 2], name="Date")
    )
    mock_cfg = {
        "start": "not-a-date",
        "end": object(),
        "lookback_months": "bad",
        "risk_target": 0.3,
    }
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True
    module.run_simulation = MagicMock()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    mock_st.error.assert_called_with("Missing start/end dates in configuration.")
    module.run_simulation.assert_not_called()


def test_main_runs_and_surfaces_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_st = _make_streamlit()
    # First button click starts the run, second corresponds to the dismiss button
    mock_st.button.side_effect = [True, False]

    module = _load_run_module(mock_st)

    class FaultyConfig(dict):
        def get(self, key, default=None):
            if key == "portfolio":
                raise RuntimeError("boom")
            return super().get(key, default)

    mock_cfg = FaultyConfig(
        {
            "start": pd.Timestamp("2020-01-31"),
            "end": pd.Timestamp("2020-02-29"),
            "lookback_months": "4",
            "risk_target": "0.75",
            "portfolio": {"weighting_scheme": "custom"},
        }
    )
    mock_returns = pd.DataFrame(
        {"A": [0.1, -0.1]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"])
    )
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True

    config_instance = MagicMock()
    module.Config = MagicMock(return_value=config_instance)

    result = SimpleNamespace(
        metrics={"Sharpe": 1.23},
        fallback_info={"engine": "ERC", "error_type": "Infeasible"},
    )
    module.run_simulation = MagicMock(return_value=result)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    module.Config.assert_called()
    module.run_simulation.assert_called_once_with(config_instance, ANY)
    mock_st.progress.return_value.progress.assert_called_with(100)
    assert mock_st.session_state["sim_results"] is result
    mock_st.warning.assert_called()
    mock_st.success.assert_called_with("Done.")
    mock_st.write.assert_called_with("Summary:", result.metrics)
