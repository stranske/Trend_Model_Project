"""Tests for the one-click demo helpers used by the Streamlit app."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd

from streamlit_app.components import demo_runner


class _DummyResult:
    def __init__(self) -> None:
        self.metrics = pd.DataFrame({"value": [1.0]})
        self.details = {"ok": True}
        self.seed = 42
        self.environment = {"python": "3"}
        self.fallback_info = None


def _make_mock_streamlit() -> ModuleType:
    class MockStreamlit(ModuleType):
        def __init__(self) -> None:
            super().__init__("streamlit")
            self.session_state = {}
            self.errors: list[str] = []
            self.successes: list[str] = []

        def error(self, message: str, *_, **__) -> None:  # noqa: D401 - helper
            self.errors.append(str(message))

        def success(self, message: str, *_, **__) -> None:  # noqa: D401 - helper
            self.successes.append(str(message))

    return MockStreamlit()


def test_run_one_click_demo_updates_state(monkeypatch):
    st_mock = _make_mock_streamlit()

    monkeypatch.setattr(
        demo_runner,
        "run_simulation",
        lambda cfg, df: _DummyResult(),
    )

    success = demo_runner.run_one_click_demo(st_module=st_mock)
    assert success
    state = st_mock.session_state
    assert "returns_df" in state
    assert "sim_config" in state
    assert "sim_results" in state
    assert state["demo_show_export_prompt"] is True


def test_app_demo_button_triggers_navigation(monkeypatch):
    class MockStreamlit(ModuleType):
        def __init__(self) -> None:  # noqa: D401 - helper
            super().__init__("streamlit")
            self.session_state = {}
            self.button_calls: list[str] = []
            self.switch_targets: list[str] = []
            self.success_messages: list[str] = []
            self._button_results = [True]

        def set_page_config(self, *_, **__):  # noqa: D401 - stub
            return None

        def title(self, *_, **__):  # noqa: D401 - stub
            return None

        def markdown(self, *_, **__):  # noqa: D401 - stub
            return None

        def info(self, *_, **__):  # noqa: D401 - stub
            return None

        def subheader(self, *_, **__):  # noqa: D401 - stub
            return None

        class _Spinner:
            def __enter__(self):  # noqa: D401 - context helper
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: D401 - context helper
                return False

        def spinner(self, *_, **__):  # noqa: D401 - stub
            return self._Spinner()

        def button(self, label: str, *_, **__):  # noqa: D401 - stub
            self.button_calls.append(label)
            if self._button_results:
                return self._button_results.pop(0)
            return False

        def success(self, message: str, *_, **__):  # noqa: D401 - stub
            self.success_messages.append(str(message))

        def switch_page(self, target: str):  # noqa: D401 - stub
            self.switch_targets.append(target)

    st_mock = MockStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", st_mock)
    monkeypatch.delitem(sys.modules, "streamlit_app.app", raising=False)

    called = {}

    def fake_run_demo(*_, **__):
        called["triggered"] = True
        return True

    monkeypatch.setattr(
        "streamlit_app.components.demo_runner.run_one_click_demo",
        fake_run_demo,
    )

    module = importlib.import_module("streamlit_app.app")
    assert module is not None
    assert called.get("triggered") is True
    assert st_mock.switch_targets == ["pages/4_Results.py"]
    assert any("Run demo" in text for text in st_mock.button_calls)

    sys.modules.pop("streamlit_app.app", None)
