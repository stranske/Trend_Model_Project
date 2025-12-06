"""Unit tests for guardrail helpers on the Configure page."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest


def _load_configure_module(fake_streamlit: Any):
    module_path = (
        Path(__file__).parent.parent / "streamlit_app" / "pages" / "2_Model.py"
    )
    if not module_path.exists():
        pytest.skip("2_Model.py page does not exist")
    spec = importlib.util.spec_from_file_location(
        "streamlit_configure_page", module_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - sanity guard
        raise AssertionError("Unable to load configure page module")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"streamlit": fake_streamlit}):
        spec.loader.exec_module(module)
    return module


def _make_streamlit_stub() -> Any:
    def _noop(*_args, **_kwargs):
        return None

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.subheader = _noop
    stub.warning = _noop
    stub.error = _noop
    stub.success = _noop
    stub.markdown = _noop
    stub.columns = lambda *_args, **_kwargs: []
    stub.number_input = _noop
    stub.selectbox = _noop
    stub.multiselect = _noop
    stub.info = _noop
    stub.caption = _noop
    stub.button = lambda *_args, **_kwargs: False
    stub.divider = _noop
    stub.json = _noop
    stub.expander = lambda *_args, **_kwargs: SimpleNamespace(
        __enter__=_noop, __exit__=_noop
    )
    return stub


@pytest.mark.skip(
    reason="Test for obsolete 2_Configure.py page - 2_Model.py has different structure"
)
def test_map_payload_errors_assigns_inline_fields() -> None:
    fake_streamlit = _make_streamlit_stub()
    module = _load_configure_module(fake_streamlit)

    messages = [
        "vol_adjust -> target_vol\n  must be greater than zero",
        "data -> csv_path\n  file does not exist",
        "data -> date_column\n  is required",
    ]

    mapped = module._map_payload_errors(messages)

    assert "risk_target" in mapped
    assert any("target_vol" in msg for msg in mapped["risk_target"])
    assert "column_mapping" in mapped
    assert any("csv_path" in msg for msg in mapped["column_mapping"])
    assert "date_column" in mapped
    assert any("date_column" in msg for msg in mapped["date_column"])
