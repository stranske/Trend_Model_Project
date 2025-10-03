from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Dict

import pytest


@pytest.fixture()
def configure_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    stub = ModuleType("streamlit")

    class SessionState(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:  # pragma: no cover - defensive
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    stub.session_state = SessionState()

    def noop(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial stub
        return None

    def selectable(
        _label: str, options: list[Any], *, index: int = 0, **kwargs: Any
    ) -> Any:  # pragma: no cover - trivial stub
        return options[index]

    setattr(stub, "selectbox", selectable)
    for attr in [
        "subheader",
        "warning",
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
        "caption",
    ]:
        setattr(stub, attr, noop)
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    module = importlib.reload(
        importlib.import_module("streamlit_app.pages.2_Configure")
    )
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


def test_render_trend_spec_settings_auto_syncs_with_selected_preset(
    configure_module: ModuleType,
) -> None:
    configure_module.st.session_state.clear()
    configure_module.st.session_state.config_state = {
        "trend_spec_values": {},
        "trend_spec_defaults": {},
        "trend_spec_preset": None,
        "trend_spec_config": {},
    }

    configure_module.render_trend_spec_settings("Conservative")

    state = configure_module.st.session_state["config_state"]
    assert state["trend_spec_preset"] == "Conservative"
    assert state["trend_spec_values"]["window"] == 126
    assert state["trend_spec_config"]["zscore"] is True


def test_trend_spec_values_to_config_normalises_optional_fields(
    configure_module: ModuleType,
) -> None:
    payload = {
        "window": "63",
        "lag": "2",
        "min_periods": 0,
        "vol_adjust": True,
        "vol_target": "0.05",
        "zscore": "0",
    }
    config = configure_module._trend_spec_values_to_config(payload)
    assert config["window"] == 63
    assert config["lag"] == 2
    assert "min_periods" not in config
    assert config["vol_target"] == 0.05
    assert config["zscore"] is False
