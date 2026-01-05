import importlib
import sys
from unittest.mock import MagicMock


def test_show_disclaimer(monkeypatch):
    st = MagicMock()
    st.session_state = {}
    st.checkbox.side_effect = [False, True]
    st.modal.return_value.__enter__.return_value = None
    st.modal.return_value.__exit__.return_value = None
    st.rerun.return_value = None

    monkeypatch.setitem(sys.modules, "streamlit", st)
    disclaimer = importlib.reload(importlib.import_module("streamlit_app.components.disclaimer"))
    assert disclaimer.show_disclaimer() is False
    assert disclaimer.show_disclaimer() is True
