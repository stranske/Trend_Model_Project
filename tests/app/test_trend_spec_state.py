from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Dict

import pytest


@pytest.fixture()
def configure_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    stub = ModuleType("streamlit")
    stub.session_state = {}
    noop = lambda *args, **kwargs: None
    for attr in [
        "subheader",
        "warning",
        "selectbox",
        "number_input",
        "checkbox",
        "divider",
        "info",
        "error",
        "success",
        "button",
        "markdown",
        "columns",
        "expander",
    ]:
        setattr(stub, attr, noop)
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    module = importlib.reload(importlib.import_module("streamlit_app.pages.2_Configure"))
    return module


def test_apply_trend_spec_preset_to_state_sets_defaults(
    configure_module: ModuleType,
) -> None:
    state: Dict[str, Any] = {
        "trend_spec_values": {},
        "trend_spec_defaults": {},
        "trend_spec_preset": None,
        "trend_spec_config": {},
    }
    result = configure_module._apply_trend_spec_preset_to_state(state, "Conservative")
    assert result["window"] == 126
    assert state["trend_spec_values"]["window"] == 126
    assert state["trend_spec_preset"] == "Conservative"
    assert state["trend_spec_config"]["lag"] == 1


def test_apply_trend_spec_preset_none_falls_back_to_defaults(
    configure_module: ModuleType,
) -> None:
    state: Dict[str, Any] = {
        "trend_spec_values": {},
        "trend_spec_defaults": {},
        "trend_spec_preset": None,
        "trend_spec_config": {},
    }
    result = configure_module._apply_trend_spec_preset_to_state(state, None)
    assert result["window"] == configure_module.TrendSpec().window
    assert state["trend_spec_preset"] is None
