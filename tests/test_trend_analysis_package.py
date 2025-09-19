import importlib
import sys

import pytest


def test_trend_analysis_lazy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reload to ensure lazy attributes are not yet materialised.
    module = importlib.import_module("trend_analysis")
    module = importlib.reload(module)

    # Remove any cached attribute to force __getattr__ execution.
    module.__dict__.pop("selector", None)

    selector_module = getattr(module, "selector")
    assert selector_module is sys.modules["trend_analysis.selector"]

    with pytest.raises(AttributeError):
        getattr(module, "does_not_exist")


def test_trend_analysis_exports_include_expected_members() -> None:
    module = importlib.reload(importlib.import_module("trend_analysis"))

    # The package should expose key helpers when optional submodules are available.
    assert "export_to_csv" in module.__all__
    assert callable(module.export_to_csv)
    assert isinstance(module.__version__, str) and module.__version__
