from __future__ import annotations

import importlib

import importlib.metadata
import pytest


def test_trend_analysis_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[misc]
            importlib.metadata.PackageNotFoundError()
        ),
    )

    mod = importlib.import_module("trend_analysis")
    reloaded = importlib.reload(mod)
    assert reloaded.__version__ == "0.1.0-dev"

