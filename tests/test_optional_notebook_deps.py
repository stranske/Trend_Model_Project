from __future__ import annotations

import importlib
import importlib.util
import sys

import pandas as pd
import pytest

from trend_analysis import pipeline

_OPTIONAL_PREFIXES = ("ipywidgets", "IPython.display", "ipydatagrid")


def _matches_optional(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in _OPTIONAL_PREFIXES
    )


def _force_optional_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend notebook-only dependencies are unavailable."""

    for name in list(sys.modules):
        if _matches_optional(name):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if _matches_optional(name):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if _matches_optional(name):
            raise ImportError(f"Optional dependency {name} is not installed")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def _reload(monkeypatch: pytest.MonkeyPatch, module_name: str):
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "FundA": [0.02, 0.01, 0.015, 0.02, 0.019, 0.018],
            "FundB": [0.01, 0.012, 0.011, 0.009, 0.011, 0.012],
        }
    )


def test_pipeline_runs_without_notebook_dependencies(monkeypatch: pytest.MonkeyPatch):
    _force_optional_missing(monkeypatch)
    _reload(monkeypatch, "trend_analysis.core.rank_selection")
    _reload(monkeypatch, "trend_analysis.gui.app")

    df = _sample_df()
    result = pipeline.run_analysis(
        df,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        1.0,
        0.0,
        risk_free_column="RF",
        allow_risk_free_fallback=False,
    )

    assert result.value is not None
    assert result.diagnostic is None
    score_frame = result.value.get("score_frame")
    assert score_frame is not None
    assert not score_frame.empty


def test_rank_ui_loader_requires_widgets(monkeypatch: pytest.MonkeyPatch):
    _force_optional_missing(monkeypatch)
    rank_selection = _reload(monkeypatch, "trend_analysis.core.rank_selection")

    with pytest.raises(ImportError, match="ipywidgets is required"):
        rank_selection.build_ui()
