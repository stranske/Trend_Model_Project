from __future__ import annotations

import importlib
import sys
import warnings

from streamlit_app.components import data_cache


APP_MODULES = (
    "streamlit_app.components.guardrails",
    "streamlit_app.components.csv_validation",
    "streamlit_app.pages.1_Data",
)


def test_app_modules_do_not_import_deprecated_streamlit_shims(monkeypatch) -> None:
    monkeypatch.setattr(data_cache, "default_sample_dataset", lambda: None)
    for module_name in APP_MODULES:
        sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for module_name in APP_MODULES:
            importlib.import_module(module_name)

    shim_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "streamlit_app" in str(warning.message)
        and "deprecated" in str(warning.message)
    ]
    assert shim_warnings == []
