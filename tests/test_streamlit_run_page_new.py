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
    if isinstance(button_response, list):
        responses = list(button_response)
    else:
        responses = [button_response, button_response]
    responses.extend([False, False, False])
    st.button = MagicMock(side_effect=responses)
    st.error = MagicMock()
    st.warning = MagicMock(return_value=_ctx_mock())
    st.success = MagicMock()
    st.progress = MagicMock(return_value=_ctx_mock())
    st.write = MagicMock()
    st.rerun = MagicMock()
    st.caption = MagicMock()
    st.json = MagicMock()
    st.spinner = MagicMock(return_value=_ctx_mock())
    return st


def _load_run_module(mock_st: MagicMock) -> Any:
    module_path = Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py"
    spec = importlib.util.spec_from_file_location("streamlit_run_page", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load streamlit run page module spec")
    module = importlib.util.module_from_spec(spec)
    disclaimer_mod = SimpleNamespace(show_disclaimer=lambda: True)

    with patch.dict(
        "sys.modules",
        {
            "streamlit": mock_st,
            "streamlit_app.components.disclaimer": disclaimer_mod,
        },
    ):
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


def test_main_uses_model_state_when_sim_config_missing() -> None:
    mock_st = _make_streamlit(button_response=[False, True])
    module = _load_run_module(mock_st)

    mock_returns = pd.DataFrame(
        {"A": [0.02] * 18},
        index=pd.date_range("2022-01-31", periods=18, freq="ME"),
    )
    mock_model_state = {
        "lookback_months": 12,
        "evaluation_months": 6,
        "risk_target": 0.4,
        "weighting_scheme": "equal",
        "trend_spec": {"window": 63, "lag": 1},
    }
    mock_st.session_state.update(
        {
            "returns_df": mock_returns,
            "model_state": mock_model_state,
        }
    )

    module.show_disclaimer = lambda: True
    module.run_simulation = MagicMock()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    module.run_simulation.assert_called_once()
    mock_st.error.assert_not_called()


def test_main_supports_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_st = _make_streamlit(button_response=[True, False])
    module = _load_run_module(mock_st)

    mock_returns = pd.DataFrame(
        {"A": [0.01] * 24},
        index=pd.date_range("2020-01-31", periods=24, freq="ME"),
    )
    mock_cfg = {
        "start": pd.Timestamp("2020-06-30"),
        "end": pd.Timestamp("2021-12-31"),
        "lookback_months": 12,
        "risk_target": 0.3,
        "portfolio": {"weighting_scheme": "equal"},
    }
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True
    result = SimpleNamespace(metrics={}, fallback_info=None)
    module.run_simulation = MagicMock(return_value=result)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    module.run_simulation.assert_called_once()
    mock_st.progress.assert_not_called()
    mock_st.success.assert_called()


def test_main_handles_invalid_dates_gracefully() -> None:
    mock_st = _make_streamlit(button_response=[False, True])
    module = _load_run_module(mock_st)

    mock_returns = pd.DataFrame({"A": [0.1, -0.1]}, index=pd.Index([1, 2], name="Date"))
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
    # Simulate run button click followed by "do not dismiss" choice
    mock_st.button.side_effect = [False, True, False]

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


def test_main_handles_non_iterable_session_state() -> None:
    """The page should fall back to attribute access when session_state
    misbehaves."""

    class NonIterableState:
        def __init__(self, returns_df: pd.DataFrame, sim_config: dict[str, object]):
            self._store: dict[str, object] = {
                "returns_df": returns_df,
                "sim_config": sim_config,
            }

        def __contains__(self, key: object) -> bool:
            raise TypeError("session_state is not iterable")

        def get(self, key: str, default: object | None = None) -> object | None:
            if key in {"returns_df", "sim_config"}:
                raise TypeError("pretend streamlit broke")
            return self._store.get(key, default)

        def __getattr__(self, name: str) -> object:
            try:
                return self._store[name]
            except KeyError as exc:  # pragma: no cover - defensive
                raise AttributeError(name) from exc

        def __setitem__(self, key: str, value: object) -> None:
            self._store[key] = value

        def __getitem__(self, key: str) -> object:
            return self._store[key]

    mock_st = _make_streamlit(button_response=[False, True])
    module = _load_run_module(mock_st)

    returns = pd.DataFrame(
        {"A": [0.1, 0.2]}, index=pd.to_datetime(["2021-01-31", "2021-02-28"])
    )
    config_dict = {
        "start": pd.Timestamp("2021-01-31"),
        "end": pd.Timestamp("2021-02-28"),
        "lookback_months": "2",
        "risk_target": 0.7,
        "portfolio": {"weighting_scheme": "equal"},
    }
    mock_st.session_state = NonIterableState(returns, config_dict)

    module.show_disclaimer = lambda: True
    config_instance = MagicMock()
    module.Config = MagicMock(return_value=config_instance)
    result = SimpleNamespace(metrics={}, fallback_info=None)
    module.run_simulation = MagicMock(return_value=result)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    module.Config.assert_called_once()
    module.run_simulation.assert_called_once_with(config_instance, ANY)
    mock_st.progress.return_value.progress.assert_called_with(100)
    assert mock_st.session_state["sim_results"] is result
    mock_st.error.assert_not_called()


def test_main_uses_default_weighting_when_portfolio_missing() -> None:
    """cfg_get should fall back to defaults when both get and item access
    fail."""

    class PortfolioFault(dict):
        def get(self, key: str, default: object | None = None) -> object | None:
            if key == "portfolio":
                raise RuntimeError("broken mapping")
            return super().get(key, default)

        def __getitem__(self, key: str) -> object:
            if key == "portfolio":
                raise KeyError(key)
            return super().__getitem__(key)

    mock_st = _make_streamlit(button_response=[False, True])
    module = _load_run_module(mock_st)

    mock_cfg = PortfolioFault(
        {
            "start": pd.Timestamp("2022-01-31"),
            "end": pd.Timestamp("2022-03-31"),
            "lookback_months": 3,
            "risk_target": 1.1,
        }
    )
    mock_returns = pd.DataFrame(
        {"A": [0.05, -0.02]}, index=pd.to_datetime(["2021-12-31", "2022-01-31"])
    )
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True
    config_instance = MagicMock()
    module.Config = MagicMock(return_value=config_instance)
    result = SimpleNamespace(metrics={}, fallback_info=None)
    module.run_simulation = MagicMock(return_value=result)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    _, kwargs = module.Config.call_args
    assert kwargs["portfolio"] == {"weighting_scheme": "equal"}
    module.run_simulation.assert_called_once_with(config_instance, ANY)
    mock_st.success.assert_called_once_with("Done.")


def test_main_dismisses_weight_engine_warning() -> None:
    """Clicking the dismiss button should set state and rerun the app."""

    mock_st = _make_streamlit()
    mock_st.button.side_effect = [False, True, True]
    module = _load_run_module(mock_st)

    mock_cfg = {
        "start": pd.Timestamp("2020-01-31"),
        "end": pd.Timestamp("2020-02-29"),
        "lookback_months": 1,
        "risk_target": 0.4,
        "portfolio": {"weighting_scheme": "equal"},
    }
    mock_returns = pd.DataFrame(
        {"A": [0.1, -0.1]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"])
    )
    mock_st.session_state.update({"returns_df": mock_returns, "sim_config": mock_cfg})

    module.show_disclaimer = lambda: True
    config_instance = MagicMock()
    module.Config = MagicMock(return_value=config_instance)
    result = SimpleNamespace(
        metrics={},
        fallback_info={"engine": "ERC", "error_type": "Failure"},
    )
    module.run_simulation = MagicMock(return_value=result)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        module.main()

    assert mock_st.session_state["dismiss_weight_engine_fallback"] is True
    mock_st.rerun.assert_called_once()
