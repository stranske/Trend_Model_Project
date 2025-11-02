import importlib
from unittest import mock

import pytest

import trend_analysis.reporting as reporting


def test_reporting_exports_real_module():
    module = importlib.reload(reporting)

    import trend.reporting as real_reporting

    assert module.generate_unified_report is real_reporting.generate_unified_report
    assert module.ReportArtifacts is real_reporting.ReportArtifacts


def test_reporting_fallback_handles_missing_dependency():
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "trend.reporting":
            raise ImportError("mocked missing trend.reporting")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        module = importlib.reload(reporting)

    try:
        artifact = module.ReportArtifacts(
            html="<h1>", pdf_bytes=None, context={"key": "value"}
        )
        assert artifact.html == "<h1>"
        assert artifact.pdf_bytes is None
        assert artifact.context == {"key": "value"}

        with pytest.raises(ImportError) as exc:
            module.generate_unified_report(object(), object())
        assert "trend.reporting.generate_unified_report" in str(exc.value)
    finally:
        importlib.reload(reporting)
