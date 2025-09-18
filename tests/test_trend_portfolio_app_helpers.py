import importlib
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence, cast

import pandas as pd
import pytest


class _SessionState(dict):
    """Minimal mapping supporting attribute-style access."""

    def __getattr__(self, name: str):  # pragma: no cover - attribute proxy
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):  # pragma: no cover - defensive
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Context:
    """No-op context manager used for Streamlit layout primitives."""

    def __init__(self, st_module: ModuleType) -> None:
        self._st = st_module

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStreamlit(ModuleType):
    """Lightweight stand-in for the Streamlit API used during import."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state: _SessionState = _SessionState()
        self.sidebar = _Context(self)
        self._rerun_calls: int = 0

    # Basic page/layout helpers -------------------------------------------------
    def set_page_config(self, *_, **__):
        return None

    def title(self, *_, **__):
        return None

    def header(self, *_, **__):
        return None

    def subheader(self, *_, **__):
        return None

    def caption(self, *_, **__):
        return None

    def markdown(self, *_, **__):
        return None

    def write(self, *_, **__):
        return None

    def info(self, *_, **__):
        return None

    def success(self, *_, **__):
        return None

    def warning(self, *_, **__):
        return None

    def error(self, *_, **__):
        return None

    def exception(self, *_, **__):
        return None

    # Interactive widgets ------------------------------------------------------
    def button(self, *_, **__):
        return False

    def checkbox(self, *_, value: bool = False, **__):
        return value

    def text_input(self, *_, value: str | None = "", **__):
        return value or ""

    def text_area(self, *_, value: str | None = "", **__):
        return value or ""

    def number_input(self, *_, value=0, **__):
        return value

    def slider(self, *_, value=None, **__):
        return value

    def selectbox(
        self, _label: str, options: Sequence | Iterable, *_, index: int = 0, **__
    ):
        options_list = list(options)
        if not options_list:
            return None
        return options_list[min(index, len(options_list) - 1)]

    def multiselect(
        self, _label: str, options: Sequence | Iterable, *_, default=None, **__
    ):
        if default is None:
            return []
        return list(default)

    def radio(
        self, _label: str, options: Sequence | Iterable, *_, index: int = 0, **__
    ):
        options_list = list(options)
        if not options_list:
            return None
        return options_list[min(index, len(options_list) - 1)]

    def file_uploader(self, *_, **__):
        return None

    def download_button(self, *_, **__):
        return None

    def dataframe(self, *_, **__):
        return None

    def tabs(self, names: Sequence[str]):
        return [_Context(self) for _ in names]

    def columns(self, spec):
        if isinstance(spec, int):
            count = spec
        else:
            count = len(list(spec))
        return [_Context(self) for _ in range(count)]

    def expander(self, *_, **__):
        return _Context(self)

    def spinner(self, *_, **__):
        return _Context(self)

    def rerun(self):
        self._rerun_calls += 1

    def stop(self):  # pragma: no cover - sanity guard
        raise RuntimeError("st.stop invoked during test")


def _load_app(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``trend_portfolio_app.app`` with a stub Streamlit module."""

    dummy = _DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy)
    sys.modules.pop("trend_portfolio_app.app", None)
    module = importlib.import_module("trend_portfolio_app.app")
    return module


def test_read_defaults_populates_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    defaults: dict[str, Any] = cast(dict[str, Any], app_mod._read_defaults())
    assert "data" in defaults
    assert "portfolio" in defaults

    raw_portfolio = defaults["portfolio"]
    assert isinstance(raw_portfolio, dict)
    portfolio_section = cast(dict[str, Any], raw_portfolio)
    assert "policy" in portfolio_section

    raw_data = defaults["data"]
    assert isinstance(raw_data, dict)
    data_section = cast(dict[str, Any], raw_data)
    assert "csv_path" in data_section


def test_merge_update_deep_merges_nested_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    updates = {"nested": {"y": 99, "z": 5}, "b": 2}

    merged = app_mod._merge_update(base, updates)

    assert merged == {"a": 1, "nested": {"x": 10, "y": 99, "z": 5}, "b": 2}
    # Original dictionaries should remain unmodified
    nested_section = base["nested"]
    assert isinstance(nested_section, dict)
    assert nested_section["y"] == 20
    assert "z" not in nested_section


def test_build_cfg_accepts_roundtrip_from_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    app_mod = _load_app(monkeypatch)

    defaults: dict[str, Any] = cast(dict[str, Any], app_mod._read_defaults())
    data_section = defaults.setdefault("data", {})
    if not isinstance(data_section, dict):
        raise AssertionError("Expected mapping for defaults['data']")
    data_section["csv_path"] = "demo.csv"
    yaml_text = app_mod._to_yaml(defaults)
    reconstructed = app_mod._build_cfg(app_mod.yaml.safe_load(yaml_text))

    assert reconstructed.data["csv_path"] == "demo.csv"


def test_summarise_run_df_rounds_numeric_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod = _load_app(monkeypatch)

    df = pd.DataFrame({"metric": [1.234567, 2.345678], "label": ["A", "B"]})
    summary = app_mod._summarise_run_df(df)

    assert list(summary.columns) == ["metric", "label"]
    assert summary["metric"].tolist() == [1.2346, 2.3457]
    assert summary["label"].tolist() == ["A", "B"]


def test_summarise_multi_handles_missing_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_mod = _load_app(monkeypatch)

    results = [
        {
            "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
            "out_ew_stats": {"sharpe": 1.23456, "cagr": 0.05678},
            "out_user_stats": {},
        },
        {"period": None, "out_ew_stats": object(), "out_user_stats": object()},
    ]

    summary = app_mod._summarise_multi(results)

    assert list(summary.columns) == [
        "in_start",
        "in_end",
        "out_start",
        "out_end",
        "ew_sharpe",
        "user_sharpe",
        "ew_cagr",
        "user_cagr",
    ]
    assert summary.loc[0, "ew_sharpe"] == pytest.approx(1.2346)
    assert summary.loc[0, "user_sharpe"] != summary.loc[0, "user_sharpe"]  # NaN
    assert summary.loc[1, "in_start"] == ""
    assert summary.loc[1, "ew_cagr"] != summary.loc[1, "ew_cagr"]  # NaN
