import importlib.util
from types import SimpleNamespace

import pandas as pd
import streamlit as st


def load_run_module():
    spec = importlib.util.spec_from_file_location(
        "run_page", "streamlit_app/pages/3_Run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_session_state(accepted: bool):
    st.session_state.clear()
    st.session_state["returns_df"] = pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2020-01-31"]))
    st.session_state["sim_config"] = {
        "start": pd.Timestamp("2020-01-31"),
        "end": pd.Timestamp("2020-01-31"),
        "lookback_months": 0,
        "risk_target": 1.0,
    }
    st.session_state["disclaimer_accepted"] = accepted


def test_run_button_disabled_without_acceptance(monkeypatch):
    module = load_run_module()

    # Bypass UI in disclaimer component
    monkeypatch.setattr(
        module, "show_disclaimer", lambda: st.session_state.get("disclaimer_accepted", False)
    )

    disabled = SimpleNamespace(value=None)

    def fake_button(label, disabled=False):
        disabled.value = disabled
        return False

    monkeypatch.setattr(st, "button", fake_button)

    setup_session_state(accepted=False)
    module.main()
    assert disabled.value is True

    setup_session_state(accepted=True)
    module.main()
    assert disabled.value is False
