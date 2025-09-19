import importlib
import importlib.metadata

import pytest
from typing import Optional


def test_trend_analysis_init_exposes_exports():
    import trend_analysis

    module = importlib.reload(trend_analysis)

    assert hasattr(module, "load_csv")
    assert hasattr(module, "export_to_csv")
    assert hasattr(module, "export_to_excel")


def test_trend_analysis_getattr_lazy_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trend_analysis

    module = importlib.reload(trend_analysis)

    if "selector" in module.__dict__:
        module.__dict__.pop("selector")

    selector_mod = module.selector  # lazy attribute should load module
    assert selector_mod is importlib.import_module("trend_analysis.selector")

    with pytest.raises(AttributeError):
        module.__getattr__("does_not_exist")


def test_trend_analysis_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    module = importlib.reload(trend_analysis)

    with monkeypatch.context() as ctx:

        def fake_version(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        ctx.setattr(importlib.metadata, "version", fake_version)
        module = importlib.reload(module)
        assert module.__version__ == "0.1.0-dev"

    importlib.reload(module)  # restore original metadata-driven version


def test_trend_analysis_missing_optional_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis

    module = importlib.reload(trend_analysis)

    # ensure the eager attribute is absent before simulating the import failure
    module.__dict__.pop("metrics", None)

    real_import = importlib.import_module

    def flaky_import(name: str, package: Optional[str] = None):
        if name == "trend_analysis.metrics":
            raise ImportError("optional dependency missing")
        return real_import(name, package)

    with monkeypatch.context() as ctx:
        ctx.setattr(importlib, "import_module", flaky_import)
        reloaded = importlib.reload(module)

    assert "metrics" not in reloaded.__dict__
    # restore pristine module state for subsequent tests
    importlib.reload(reloaded)
