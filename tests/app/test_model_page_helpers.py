from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture()
def model_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def _noop(*_args, **_kwargs):
        return None

    class Context:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.title = _noop
    stub.error = _noop
    stub.subheader = _noop
    stub.divider = _noop
    stub.info = _noop
    stub.success = _noop
    stub.warning = _noop
    stub.altair_chart = _noop
    stub.form = lambda *_args, **_kwargs: Context()
    stub.form_submit_button = lambda *_args, **_kwargs: False
    stub.columns = lambda n: [Context() for _ in range(n)]
    stub.selectbox = lambda _label, options, index=0, **_kwargs: options[index]
    stub.number_input = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.checkbox = lambda _label, value=False, **_kwargs: value

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    from app.streamlit import state as app_state

    monkeypatch.setattr(app_state, "initialize_session_state", lambda: None)
    monkeypatch.setattr(
        app_state,
        "get_uploaded_data",
        lambda: (pd.DataFrame({"A": [0.01, 0.02]}), {}),
    )

    module = importlib.reload(importlib.import_module("streamlit_app.pages.2_Model"))
    return module


def test_preset_defaults_uses_preset(monkeypatch: pytest.MonkeyPatch, model_module: ModuleType) -> None:
    fake = model_module.TrendSpecPreset(
        name="Test",
        description="",
        spec=model_module.TrendSpec(
            window=120,
            lag=2,
            min_periods=60,
            vol_adjust=True,
            vol_target=0.15,
            zscore=True,
        ),
    )

    monkeypatch.setattr(model_module, "get_trend_spec_preset", lambda name: fake)
    defaults = model_module._preset_defaults("CustomPreset")
    assert defaults["window"] == 120
    assert defaults["lag"] == 2
    assert defaults["min_periods"] == 60
    assert defaults["vol_adjust"] is True
    assert defaults["vol_target"] == 0.15
    assert defaults["zscore"] is True


def test_trend_spec_from_form_normalises_inputs(model_module: ModuleType) -> None:
    values = {
        "window": 50,
        "lag": 75,
        "min_periods": 120,
        "vol_adjust": True,
        "vol_target": 0.02,
        "zscore": True,
    }
    spec = model_module._trend_spec_from_form(values)
    assert spec["window"] == 50
    assert spec["lag"] == 50  # coerced to window size
    assert spec["min_periods"] == 50
    assert spec["vol_target"] == 0.02
    assert spec["zscore"] is True


def test_validate_model_catches_errors(model_module: ModuleType) -> None:
    values = {
        "trend_spec": {
            "window": 20,
            "lag": 25,
            "min_periods": 30,
            "vol_adjust": True,
            "vol_target": 0.0,
        },
        "selection_count": 15,
        "metric_weights": {"sharpe": 0.0, "return_ann": 0.0},
    }
    errors = model_module._validate_model(values, column_count=10)
    assert any("Lag" in err for err in errors)
    assert any("Minimum periods" in err for err in errors)
    assert any("positive metric weight" in err for err in errors)
    assert any("volatility target" in err for err in errors)

