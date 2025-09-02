import importlib.util
from types import SimpleNamespace
import pathlib
from unittest.mock import MagicMock

import pandas as pd
import streamlit as st


def load_run_module():
    run_py_path = (
        pathlib.Path(__file__).parent.parent / "streamlit_app" / "pages" / "3_Run.py"
    )
    spec = importlib.util.spec_from_file_location("run_page", str(run_py_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockSessionState(dict):
    """A mock session state that behaves like both a dict and streamlit's session_state."""
    
    def clear(self):
        super().clear()
    
    def get(self, key, default=None):
        return super().get(key, default)
    
    def __contains__(self, key):
        return super().__contains__(key)


def setup_session_state(accepted: bool):
    # Create a proper mock session state
    session_dict = MockSessionState()
    
    # Replace st.session_state with our mock
    st.session_state = session_dict
    
    st.session_state.clear()
    st.session_state["returns_df"] = pd.DataFrame(
        {"A": [0.1]}, index=pd.to_datetime(["2020-01-31"])
    )
    st.session_state["sim_config"] = {
        "start": pd.Timestamp("2020-01-31"),
        "end": pd.Timestamp("2020-01-31"),
        "lookback_months": 0,
        "risk_target": 1.0,
    }
    st.session_state["disclaimer_accepted"] = accepted


def test_run_button_disabled_without_acceptance(monkeypatch):
    module = load_run_module()

    # Create an isolated session state to avoid pollution from other tests
    isolated_session_state = MockSessionState()
    
    # Bypass UI in disclaimer component
    monkeypatch.setattr(
        module,
        "show_disclaimer",
        lambda: isolated_session_state.get("disclaimer_accepted", False),
    )

    flag = SimpleNamespace(value=None)

    def fake_button(label, disabled=False):
        flag.value = bool(disabled)
        return False

    # Mock streamlit in the loaded module to prevent pollution 
    mock_st = MagicMock()
    mock_st.session_state = isolated_session_state
    mock_st.button = fake_button
    mock_st.title = MagicMock()
    mock_st.error = MagicMock()
    mock_st.stop = MagicMock()
    mock_st.modal = MagicMock()  # Context manager
    mock_st.checkbox = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.rerun = MagicMock()
    
    # Replace the module's st reference directly
    monkeypatch.setattr(module, "st", mock_st)

    # First test: disclaimer NOT accepted
    isolated_session_state.clear()
    isolated_session_state["returns_df"] = pd.DataFrame(
        {"A": [0.1]}, index=pd.to_datetime(["2020-01-31"])
    )
    isolated_session_state["sim_config"] = {
        "start": pd.Timestamp("2020-01-31"),
        "end": pd.Timestamp("2020-01-31"),
        "lookback_months": 0,
        "risk_target": 1.0,
    }
    isolated_session_state["disclaimer_accepted"] = False
    
    module.main()
    assert flag.value is True

    # Second test: disclaimer accepted
    isolated_session_state["disclaimer_accepted"] = True
    module.main()
    assert flag.value is False
